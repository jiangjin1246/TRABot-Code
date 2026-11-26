"""Fail-closed canonical-neutral gate for one Gemini Live connection.

The gate owns exactly one compare-and-submit attempt.  It never replaces or
cancels simulator work: an occupied queue, an action-sequence race, or an owner
conflict leaves the shared simulator untouched and keeps the semantic session
closed.  ``completed`` means the final target was commanded; readiness is
published only after distinct physics snapshots show the neutral-controlled
joints settled in both position and velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import time
from typing import Any, Callable, Mapping

from .semantic_motion import inline_motion, motion_endpoints
from .semantic_runtime import (
    SUPPORTED_BRIDGE_POLICY,
    canonical_action_payload_sha256,
)


SESSION_NEUTRAL_Q_DRIFT_TOLERANCE_RAD = 0.001
# Velocity is a separate settling signal from inter-sample position drift.  The
# fixed-base MuJoCo controller exhibits small steady-state velocity noise up to
# roughly 0.01 rad/s even after the commanded upper-body target is unchanged.
SESSION_NEUTRAL_DQ_TOLERANCE_RAD_S = 0.01
SESSION_NEUTRAL_STABLE_SAMPLE_COUNT = 3
SESSION_NEUTRAL_STABLE_DWELL_S = 0.2
SESSION_NEUTRAL_TARGET_TOLERANCE_RAD = 1e-9
_SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_NONTERMINAL_STATUSES = frozenset(("scheduled", "started", "running"))
_TERMINAL_STATUSES = frozenset(("completed", "canceled", "replaced", "failed"))


@dataclass(frozen=True, slots=True)
class SemanticSessionNeutralResult:
    """Stable public outcome consumed by the controller and browser UI."""

    ready: bool
    reason_code: str
    action_id: str | None = None
    stable_samples: int = 0
    maximum_q_drift_rad: float | None = None
    maximum_abs_dq_rad_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason_code": self.reason_code,
            "action_id": self.action_id,
            "stable_samples": self.stable_samples,
            "maximum_q_drift_rad": self.maximum_q_drift_rad,
            "maximum_abs_dq_rad_s": self.maximum_abs_dq_rad_s,
        }


class _SessionGateFailure(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _typed_http_status(exc: BaseException) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _nonnegative_int(value: Any, reason_code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _SessionGateFailure(reason_code)
    return value


def _idle_action_seq(state: Any) -> int:
    if not isinstance(state, Mapping):
        raise _SessionGateFailure("simulator_state_invalid")
    if state.get("fault") not in (None, ""):
        raise _SessionGateFailure("simulator_fault")
    if "current" in state:
        current = state.get("current")
        if "action" in state and state.get("action") != current:
            raise _SessionGateFailure("simulator_state_invalid")
    elif "action" in state:
        current = state.get("action")
    else:
        raise _SessionGateFailure("simulator_state_invalid")
    queued = state.get("queued")
    if current is not None or not isinstance(queued, list) or queued:
        raise _SessionGateFailure("simulator_busy")
    return _nonnegative_int(state.get("action_seq"), "simulator_state_invalid")


def _finite_joint(
    values: Any,
    joint: str,
    *,
    reason_code: str,
) -> float:
    if not isinstance(values, Mapping):
        raise _SessionGateFailure(reason_code)
    aliases = (
        (joint, f"{joint}_joint")
        if not joint.endswith("_joint")
        else (joint, joint[:-6])
    )
    for alias in aliases:
        if alias not in values:
            continue
        raw = values[alias]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise _SessionGateFailure(reason_code)
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise _SessionGateFailure(reason_code)
        return numeric
    raise _SessionGateFailure(reason_code)


class SemanticSessionNeutralGate:
    """Prepare one live semantic session without taking over shared work."""

    def __init__(
        self,
        *,
        client: Any,
        neutral_motion: Mapping[str, Any],
        neutral_q: Mapping[str, Any],
        exclusive_owner: str,
        poll_interval_s: float = 0.05,
        timeout_s: float = 45.0,
        stable_sample_count: int = SESSION_NEUTRAL_STABLE_SAMPLE_COUNT,
        stable_dwell_s: float = SESSION_NEUTRAL_STABLE_DWELL_S,
        q_drift_tolerance_rad: float = SESSION_NEUTRAL_Q_DRIFT_TOLERANCE_RAD,
        dq_tolerance_rad_s: float = SESSION_NEUTRAL_DQ_TOLERANCE_RAD_S,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        if client is None or not all(
            callable(getattr(client, name, None))
            for name in ("state_sync", "post_action_sync", "action_sync")
        ):
            raise ValueError(
                "session neutral gate requires state_sync/post_action_sync/action_sync"
            )
        if (
            not isinstance(exclusive_owner, str)
            or _SAFE_TOKEN_RE.fullmatch(exclusive_owner) is None
        ):
            raise ValueError("session neutral exclusive_owner is invalid")
        if (
            isinstance(stable_sample_count, bool)
            or not isinstance(stable_sample_count, int)
            or stable_sample_count < SESSION_NEUTRAL_STABLE_SAMPLE_COUNT
        ):
            raise ValueError("stable_sample_count must be at least three")
        poll = float(poll_interval_s)
        timeout = float(timeout_s)
        stable_dwell = float(stable_dwell_s)
        q_drift_tolerance = float(q_drift_tolerance_rad)
        dq_tolerance = float(dq_tolerance_rad_s)
        if not math.isfinite(poll) or poll < 0.001 or poll > 10.0:
            raise ValueError("poll_interval_s must be between 0.001 and 10")
        if not math.isfinite(timeout) or timeout < 0.1 or timeout > 600.0:
            raise ValueError("timeout_s must be between 0.1 and 600")
        if (
            not math.isfinite(stable_dwell)
            or stable_dwell < SESSION_NEUTRAL_STABLE_DWELL_S
        ):
            raise ValueError("stable_dwell_s must be at least 0.2 seconds")
        if not math.isfinite(q_drift_tolerance) or q_drift_tolerance <= 0.0:
            raise ValueError("q_drift_tolerance_rad must be positive and finite")
        if not math.isfinite(dq_tolerance) or dq_tolerance <= 0.0:
            raise ValueError("dq_tolerance_rad_s must be positive and finite")
        if not callable(monotonic) or not callable(sleep):
            raise TypeError("monotonic and sleep must be callable")

        motion = inline_motion(neutral_motion)
        entry_q, exit_q = motion_endpoints(motion)
        if set(entry_q) != set(exit_q):
            raise ValueError("neutral motion endpoint joints are inconsistent")
        if not isinstance(neutral_q, Mapping) or not neutral_q:
            raise ValueError("neutral_q must be a non-empty mapping")
        target_q: dict[str, float] = {}
        # Project intentionally to the joints the neutral motion controls.  A
        # fixed-base simulator may report leg/ankle motion that this action
        # neither commands nor needs to settle before upper-body semantics.
        for joint in exit_q:
            target_q[joint] = _finite_joint(
                neutral_q,
                joint,
                reason_code="neutral_target_invalid",
            )
            if abs(target_q[joint] - exit_q[joint]) > 1e-12:
                raise ValueError("neutral_q does not match neutral motion exit")

        self.client = client
        self.neutral_motion = motion
        self.target_q = target_q
        self.exclusive_owner = exclusive_owner
        self.poll_interval_s = poll
        self.timeout_s = timeout
        self.stable_sample_count = stable_sample_count
        self.stable_dwell_s = stable_dwell
        self.q_drift_tolerance_rad = q_drift_tolerance
        self.dq_tolerance_rad_s = dq_tolerance
        self._monotonic = monotonic
        self._sleep = sleep

    def _result(
        self,
        ready: bool,
        reason_code: str,
        *,
        action_id: str | None = None,
        stable_samples: int = 0,
        maximum_q_drift_rad: float | None = None,
        maximum_abs_dq_rad_s: float | None = None,
    ) -> SemanticSessionNeutralResult:
        return SemanticSessionNeutralResult(
            ready=ready,
            reason_code=reason_code,
            action_id=action_id,
            stable_samples=stable_samples,
            maximum_q_drift_rad=maximum_q_drift_rad,
            maximum_abs_dq_rad_s=maximum_abs_dq_rad_s,
        )

    def _action_id(self) -> str:
        digest = hashlib.sha256(
            (self.exclusive_owner + ":safe_neutral").encode("utf-8")
        ).hexdigest()
        return f"semsession-neutral-{digest[:32]}"

    def _deadline_expired(self, deadline: float) -> bool:
        now = float(self._monotonic())
        if not math.isfinite(now):
            raise _SessionGateFailure("session_clock_invalid")
        return now >= deadline

    def _wait_for_completed(
        self,
        *,
        action_id: str,
        payload_sha256: str,
        deadline: float,
    ) -> None:
        while True:
            try:
                record = self.client.action_sync(action_id)
            except Exception:
                if self._deadline_expired(deadline):
                    raise _SessionGateFailure("neutral_action_observation_timeout")
                self._sleep(self.poll_interval_s)
                continue
            if not isinstance(record, Mapping):
                raise _SessionGateFailure("neutral_action_record_invalid")
            if (
                record.get("id") != action_id
                or record.get("payload_sha256") != payload_sha256
            ):
                raise _SessionGateFailure("neutral_action_record_invalid")
            status = record.get("status")
            if status == "completed":
                return
            if status in _TERMINAL_STATUSES:
                raise _SessionGateFailure("neutral_action_failed")
            if status not in _NONTERMINAL_STATUSES:
                raise _SessionGateFailure("neutral_action_record_invalid")
            if self._deadline_expired(deadline):
                raise _SessionGateFailure("neutral_action_timeout")
            self._sleep(self.poll_interval_s)

    def _wait_for_stable(
        self,
        *,
        action_id: str,
        expected_action_seq: int,
        deadline: float,
    ) -> SemanticSessionNeutralResult:
        stable_samples = 0
        stable_started: float | None = None
        last_state_seq: int | None = None
        previous_q: dict[str, float] | None = None
        maximum_q_drift = math.inf
        maximum_abs_dq = math.inf
        while True:
            try:
                state = self.client.state_sync()
                action_seq = _idle_action_seq(state)
                state_seq = _nonnegative_int(
                    state.get("seq") if isinstance(state, Mapping) else None,
                    "simulator_state_invalid",
                )
                if last_state_seq is not None and state_seq < last_state_seq:
                    raise _SessionGateFailure("simulator_state_invalid")
                if state_seq == last_state_seq:
                    if self._deadline_expired(deadline):
                        raise _SessionGateFailure("neutral_stability_timeout")
                    self._sleep(self.poll_interval_s)
                    continue
                last_state_seq = state_seq
                # MotionService advances action_seq exactly once for the own
                # schedule, start, and terminal transitions. Pinning every
                # post-terminal state to admission+3 closes the otherwise
                # invisible window in which a fast foreign action could finish
                # before the first settling read and become a new baseline.
                if action_seq != expected_action_seq:
                    raise _SessionGateFailure("simulator_action_sequence_changed")

                q = state.get("q")
                dq = state.get("dq")
                target = state.get("target")
                current_q = {
                    joint: _finite_joint(
                        q,
                        joint,
                        reason_code="simulator_state_invalid",
                    )
                    for joint in self.target_q
                }
                for joint, neutral_target in self.target_q.items():
                    observed_target = _finite_joint(
                        target,
                        joint,
                        reason_code="simulator_state_invalid",
                    )
                    if (
                        abs(observed_target - neutral_target)
                        > SESSION_NEUTRAL_TARGET_TOLERANCE_RAD
                    ):
                        raise _SessionGateFailure("neutral_target_changed")
                velocities = [
                    abs(
                        _finite_joint(
                            dq,
                            joint,
                            reason_code="simulator_state_invalid",
                        )
                    )
                    for joint in self.target_q
                ]
                maximum_q_drift = (
                    math.inf
                    if previous_q is None
                    else max(
                        abs(current_q[joint] - previous_q[joint])
                        for joint in self.target_q
                    )
                )
                previous_q = current_q
                maximum_abs_dq = max(velocities)
                if (
                    maximum_q_drift <= self.q_drift_tolerance_rad
                    and maximum_abs_dq <= self.dq_tolerance_rad_s
                ):
                    now = float(self._monotonic())
                    if not math.isfinite(now):
                        raise _SessionGateFailure("session_clock_invalid")
                    if stable_samples == 0:
                        stable_started = now
                    stable_samples += 1
                    if (
                        stable_samples >= self.stable_sample_count
                        and stable_started is not None
                        and now - stable_started >= self.stable_dwell_s
                    ):
                        return self._result(
                            True,
                            "session_neutral_ready",
                            action_id=action_id,
                            stable_samples=stable_samples,
                            maximum_q_drift_rad=maximum_q_drift,
                            maximum_abs_dq_rad_s=maximum_abs_dq,
                        )
                else:
                    stable_samples = 0
                    stable_started = None
            except _SessionGateFailure:
                raise
            except Exception:
                raise _SessionGateFailure("simulator_state_unavailable")
            if self._deadline_expired(deadline):
                raise _SessionGateFailure("neutral_stability_timeout")
            self._sleep(self.poll_interval_s)

    def prepare(self) -> SemanticSessionNeutralResult:
        """Run the one-shot neutral action and settling gate synchronously."""

        try:
            started = float(self._monotonic())
            if not math.isfinite(started):
                raise _SessionGateFailure("session_clock_invalid")
            deadline = started + self.timeout_s
            try:
                state = self.client.state_sync()
            except Exception:
                raise _SessionGateFailure("simulator_state_unavailable")
            action_seq = _idle_action_seq(state)
            minimum_post_action_seq = action_seq + 1
            expected_terminal_action_seq = action_seq + 3
            action_id = self._action_id()
            payload = {
                "motion": inline_motion(self.neutral_motion),
                "speed": 1.0,
                "blend_duration": 0.0,
                "bridge_policy": SUPPORTED_BRIDGE_POLICY,
                "id": action_id,
                "mode": "enqueue",
                "require_empty": True,
                "expected_action_seq": action_seq,
                "exclusive_owner": self.exclusive_owner,
            }
            payload["payload_sha256"] = canonical_action_payload_sha256(payload)
            payload_sha256 = payload["payload_sha256"]
            try:
                response = self.client.post_action_sync(payload)
            except Exception as exc:
                if _typed_http_status(exc) == 409:
                    raise _SessionGateFailure("neutral_admission_conflict")
                # The POST outcome is ambiguous. Never retry or cancel it; drain
                # only the deterministic exact ID until terminal/timeout.
                self._wait_for_completed(
                    action_id=action_id,
                    payload_sha256=payload_sha256,
                    deadline=deadline,
                )
                raise _SessionGateFailure("neutral_submission_ambiguous")

            response_valid = (
                isinstance(response, Mapping)
                and response.get("accepted") is True
                and response.get("id") == action_id
                and response.get("payload_sha256", payload_sha256)
                == payload_sha256
                and not isinstance(response.get("action_seq"), bool)
                and isinstance(response.get("action_seq"), int)
                and minimum_post_action_seq
                <= response.get("action_seq")
                <= expected_terminal_action_seq
            )
            self._wait_for_completed(
                action_id=action_id,
                payload_sha256=payload_sha256,
                deadline=deadline,
            )
            if not response_valid:
                raise _SessionGateFailure("neutral_submission_response_invalid")
            return self._wait_for_stable(
                action_id=action_id,
                expected_action_seq=expected_terminal_action_seq,
                deadline=deadline,
            )
        except _SessionGateFailure as exc:
            return self._result(False, exc.reason_code)
        except Exception:
            return self._result(False, "session_neutral_internal_error")


__all__ = [
    "SESSION_NEUTRAL_DQ_TOLERANCE_RAD_S",
    "SESSION_NEUTRAL_Q_DRIFT_TOLERANCE_RAD",
    "SESSION_NEUTRAL_STABLE_SAMPLE_COUNT",
    "SESSION_NEUTRAL_STABLE_DWELL_S",
    "SESSION_NEUTRAL_TARGET_TOLERANCE_RAD",
    "SemanticSessionNeutralGate",
    "SemanticSessionNeutralResult",
]
