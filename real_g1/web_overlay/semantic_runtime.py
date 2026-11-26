"""Fail-closed foundations for semantic gesture scheduling.

This module deliberately does not import the catalog or Planner implementation.
Integration supplies a pure ``plan_next`` callable plus adapters that turn the
first planned Atom and the final neutral recovery into strict inline motions.
Live execution is opt-in; dry-run is the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import math
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .reply_clock import (
    MAX_REPLY_DURATION_US,
    MIN_REPLY_DURATION_US,
    ReplyClock,
    ReplyClockError,
    ReplyClockSnapshot,
)


RUNTIME_FUNCTIONS = (
    "emphasize",
    "enumerate",
    "contrast_compare",
    "refer_self",
    "refer_other",
    "refer_external",
    "depict_size_shape",
    "depict_direction_motion",
    "turn_open",
    "turn_yield",
    "greet_farewell",
    "thank",
    "affirm",
    "deny",
    "certainty",
    "uncertainty",
    "enthusiasm",
)
RUNTIME_FUNCTION_SET = frozenset(RUNTIME_FUNCTIONS)
ABSOLUTE_ORDERED_FUNCTIONS_GUARD = 100_000
MIN_ESTIMATED_SPOKEN_DURATION_S = 0.1
SUPPORTED_BRIDGE_POLICY = "minimum_jerk_v1"
# MuJoCo continues to settle between the planning and admission snapshots even
# while its action queue is idle.  At 25 Hz minimum-jerk quantisation, any
# endpoint drift within 0.005 rad can increase the bridge by at most one 40 ms
# tick; that tick is reserved explicitly below instead of pretending q is
# bit-identical.
ADMISSION_Q_TOLERANCE_RAD = 0.005
ADMISSION_Q_TICK_GUARD_US = 40_000
# A materially stale bridge is never submitted.  The executor may discard and
# replan the same suffix from the admission snapshot this many times before it
# skips gestures for the turn without globally safe-stopping future turns.
MAX_ADMISSION_REPLAN_ATTEMPTS = 3
ADMISSION_Q_DRIFT_REASON = "admission_q_drift"

_TOOL_FIELDS = frozenset({"ordered_functions", "estimated_spoken_duration_s"})
_ACTION_FIELDS = frozenset(
    {"motion", "speed", "blend_duration", "bridge_policy"}
)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TERMINAL_STATUSES = frozenset({"completed", "canceled", "replaced", "failed"})
_NONTERMINAL_STATUSES = frozenset({"scheduled", "started"})


class ToolPayloadValidationError(ValueError):
    """Gemini returned a payload outside the frozen semantic tool contract."""


class SemanticRuntimeError(RuntimeError):
    """Base error for the semantic suffix runtime."""


class SemanticConflictError(SemanticRuntimeError):
    """A request/revision or global single-writer invariant conflicted."""


class SemanticExecutionError(SemanticRuntimeError):
    """Planning, admission, submission, or own-action observation failed."""


class AuditCallbackError(SemanticRuntimeError):
    """The caller could not durably record a runtime event."""


class _InterruptedBeforeSubmit(Exception):
    """Internal control-flow marker; no POST was issued."""


class _BudgetBeforeSubmit(Exception):
    """Internal control-flow marker; the reserved closed prefix no longer fits."""

    def __init__(self, snapshot: ReplyClockSnapshot, required_budget_us: int):
        super().__init__("reserved closed prefix no longer fits ReplyClock")
        self.snapshot = snapshot
        self.required_budget_us = required_budget_us


class _ReplanBeforeSubmit(Exception):
    """Admission q moved materially; no POST occurred and a fresh replan is safe."""

    def __init__(self, measured_q: Mapping[str, float], maximum_drift_rad: float):
        super().__init__("measured q changed before admission")
        self.previous_atom = {"q": dict(measured_q)}
        self.maximum_drift_rad = float(maximum_drift_rad)


@dataclass(frozen=True, slots=True)
class _SubmissionReceipt:
    action_id: str
    record: Mapping[str, Any]
    forced_reply_budget: bool
    closed_cost_us: int | None
    decision_budget_us: int | None
    planned_overrun_us: int


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticExecutionError(f"value is not finite canonical JSON: {exc}") from exc


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticExecutionError(f"{label} must be a positive integer")
    return value


def _finite_float(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticExecutionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise SemanticExecutionError(
            f"{label} must be finite and between {minimum} and {maximum}"
        )
    return result


def _typed_http_status(exc: BaseException) -> int | None:
    """Read an explicit HTTP status without brittle exception-string parsing."""

    current: BaseException | None = exc
    seen: set[int] = set()
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        for name in ("status_code", "status", "code"):
            value = getattr(current, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        current = current.__cause__ or current.__context__
    return None


@dataclass(frozen=True, slots=True)
class SemanticToolPayload:
    """Validated model output bound to a trusted upstream turn revision."""

    request_id: str
    revision: int
    ordered_functions: tuple[str, ...]
    estimated_spoken_duration_us: int

    @property
    def estimated_spoken_duration_s(self) -> float:
        return self.estimated_spoken_duration_us / 1_000_000.0

    @property
    def fingerprint_sha256(self) -> str:
        body = {
            "schema_version": "semantic_tool_payload_v1",
            "request_id": self.request_id,
            "revision": self.revision,
            "ordered_functions": list(self.ordered_functions),
            "estimated_spoken_duration_us": self.estimated_spoken_duration_us,
        }
        return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "revision": self.revision,
            "ordered_functions": list(self.ordered_functions),
            "estimated_spoken_duration_s": self.estimated_spoken_duration_s,
            "estimated_spoken_duration_us": self.estimated_spoken_duration_us,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


def gemini_tool_parameters_schema() -> dict[str, Any]:
    """Return a fresh JSON-schema-compatible Gemini tool parameter object."""

    return {
        "type": "object",
        "properties": {
            "ordered_functions": {
                "type": "array",
                "description": "Communicative functions in spoken order; [] means no gesture.",
                "items": {"type": "string", "enum": list(RUNTIME_FUNCTIONS)},
            },
            "estimated_spoken_duration_s": {
                "type": "number",
                "description": "Estimated duration of the spoken reply in seconds.",
                "minimum": MIN_ESTIMATED_SPOKEN_DURATION_S,
            },
        },
        "required": ["ordered_functions", "estimated_spoken_duration_s"],
        "additionalProperties": False,
    }


def validate_gemini_tool_payload(
    raw: Mapping[str, Any],
    *,
    request_id: str,
    revision: int,
) -> SemanticToolPayload:
    """Validate model-owned fields and bind trusted request metadata.

    ``request_id`` and ``revision`` are intentionally not accepted from Gemini.
    """

    if not isinstance(raw, Mapping) or set(raw) != _TOOL_FIELDS:
        raise ToolPayloadValidationError(
            "Gemini payload must contain exactly ordered_functions and "
            "estimated_spoken_duration_s"
        )
    if not isinstance(request_id, str) or _REQUEST_ID_RE.fullmatch(request_id) is None:
        raise ToolPayloadValidationError("request_id has an invalid shape")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ToolPayloadValidationError("revision must be a positive integer")

    functions = raw.get("ordered_functions")
    if not isinstance(functions, list):
        raise ToolPayloadValidationError("ordered_functions must be an array")
    if len(functions) > ABSOLUTE_ORDERED_FUNCTIONS_GUARD:
        raise ToolPayloadValidationError(
            "ordered_functions exceeds the absolute protocol resource guard"
        )
    for index, function in enumerate(functions):
        if not isinstance(function, str) or function not in RUNTIME_FUNCTION_SET:
            raise ToolPayloadValidationError(
                f"ordered_functions[{index}] is not a supported runtime function"
            )

    raw_duration = raw.get("estimated_spoken_duration_s")
    if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
        raise ToolPayloadValidationError(
            "estimated_spoken_duration_s duration must be a number"
        )
    try:
        numeric = float(raw_duration)
    except (OverflowError, ValueError) as exc:
        raise ToolPayloadValidationError(
            "estimated spoken duration exceeds the monotonic clock range"
        ) from exc
    if not math.isfinite(numeric):
        raise ToolPayloadValidationError(
            "estimated_spoken_duration_s must be finite"
        )
    try:
        decimal_duration = Decimal(str(raw_duration))
    except InvalidOperation as exc:
        raise ToolPayloadValidationError(
            "estimated_spoken_duration_s is invalid"
        ) from exc
    if decimal_duration < Decimal("0.1"):
        raise ToolPayloadValidationError(
            "estimated_spoken_duration_s must be at least 0.1 seconds"
        )
    duration_us = int(
        (decimal_duration * Decimal(1_000_000)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if duration_us > MAX_REPLY_DURATION_US:
        raise ToolPayloadValidationError(
            "estimated spoken duration exceeds the monotonic clock range"
        )
    return SemanticToolPayload(
        request_id=request_id,
        revision=revision,
        ordered_functions=tuple(functions),
        estimated_spoken_duration_us=duration_us,
    )


def _validate_action_payload(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping) or set(value) != _ACTION_FIELDS:
        raise SemanticExecutionError(
            "action payload must contain exactly motion, speed, blend_duration, "
            "and bridge_policy"
        )
    motion = value.get("motion")
    if not isinstance(motion, Mapping) or not motion:
        raise SemanticExecutionError("action payload motion must be a non-empty object")
    speed = _finite_float(value.get("speed"), "action payload speed", minimum=0.25, maximum=2.0)
    if speed != 1.0:
        # Planner duration/transition costs are frozen at unit speed and must
        # not drift from the submitted controller payload.
        raise SemanticExecutionError("action payload speed must be exactly 1.0")
    blend = _finite_float(
        value.get("blend_duration"),
        "action payload blend_duration",
        minimum=0.0,
        maximum=2.0,
    )
    if blend != 0.0:
        raise SemanticExecutionError(
            "action payload blend_duration must be zero for minimum_jerk_v1"
        )
    if value.get("bridge_policy") != SUPPORTED_BRIDGE_POLICY:
        raise SemanticExecutionError(
            f"action payload bridge_policy must be {SUPPORTED_BRIDGE_POLICY}"
        )
    normalized = {
        "motion": dict(motion),
        "speed": speed,
        "blend_duration": blend,
        "bridge_policy": SUPPORTED_BRIDGE_POLICY,
    }
    return _canonical_json_bytes(normalized)


@dataclass(frozen=True, slots=True, init=False)
class SemanticActionStep:
    plan_id: str
    function: str
    estimated_duration_us: int
    _atom_ref_json: bytes
    _action_payload_json: bytes

    def __init__(
        self,
        *,
        plan_id: str,
        function: str,
        atom_ref: Mapping[str, Any],
        action_payload: Mapping[str, Any],
        estimated_duration_us: int,
    ) -> None:
        if not isinstance(plan_id, str) or _SAFE_TOKEN_RE.fullmatch(plan_id) is None:
            raise SemanticExecutionError("plan_id has an invalid shape")
        if function not in RUNTIME_FUNCTION_SET:
            raise SemanticExecutionError("semantic action function is not in the runtime enum")
        if not isinstance(atom_ref, Mapping) or not atom_ref:
            raise SemanticExecutionError("atom_ref must be a non-empty object")
        duration = _positive_int(estimated_duration_us, "estimated_duration_us")
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "estimated_duration_us", duration)
        object.__setattr__(self, "_atom_ref_json", _canonical_json_bytes(dict(atom_ref)))
        object.__setattr__(self, "_action_payload_json", _validate_action_payload(action_payload))

    @property
    def atom_ref(self) -> dict[str, Any]:
        return json.loads(self._atom_ref_json.decode("utf-8"))

    @property
    def action_payload(self) -> dict[str, Any]:
        return json.loads(self._action_payload_json.decode("utf-8"))

    @property
    def action_payload_sha256(self) -> str:
        return hashlib.sha256(self._action_payload_json).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class SemanticRecoveryStep:
    recovery_id: str
    estimated_duration_us: int
    _recovery_ref_json: bytes
    _action_payload_json: bytes

    def __init__(
        self,
        *,
        recovery_id: str,
        recovery_ref: Mapping[str, Any],
        action_payload: Mapping[str, Any],
        estimated_duration_us: int,
    ) -> None:
        if not isinstance(recovery_id, str) or _SAFE_TOKEN_RE.fullmatch(recovery_id) is None:
            raise SemanticExecutionError("recovery_id has an invalid shape")
        if not isinstance(recovery_ref, Mapping) or not recovery_ref:
            raise SemanticExecutionError("recovery_ref must be a non-empty object")
        duration = _positive_int(estimated_duration_us, "estimated_duration_us")
        object.__setattr__(self, "recovery_id", recovery_id)
        object.__setattr__(self, "estimated_duration_us", duration)
        object.__setattr__(
            self, "_recovery_ref_json", _canonical_json_bytes(dict(recovery_ref))
        )
        object.__setattr__(self, "_action_payload_json", _validate_action_payload(action_payload))

    @property
    def recovery_ref(self) -> dict[str, Any]:
        return json.loads(self._recovery_ref_json.decode("utf-8"))

    @property
    def action_payload(self) -> dict[str, Any]:
        return json.loads(self._action_payload_json.decode("utf-8"))

    @property
    def action_payload_sha256(self) -> str:
        return hashlib.sha256(self._action_payload_json).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticExecutionResult:
    request_id: str
    revision: int
    request_fingerprint_sha256: str
    status: str
    planned_functions: tuple[str, ...]
    completed_functions: tuple[str, ...]
    pending_functions: tuple[str, ...]
    dropped_functions: tuple[str, ...]
    action_ids: tuple[str, ...]
    recovery_action_id: str | None
    closed_neutral: bool
    dry_run: bool
    would_force_first_atom: bool
    forced_first_atom: bool
    planned_overrun_us: int


def canonical_action_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the exact flat POST object, excluding only ``payload_sha256``."""

    if not isinstance(payload, Mapping):
        raise SemanticExecutionError("flat action payload must be an object")
    body = dict(payload)
    body.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _coerce_action_step(value: Any) -> SemanticActionStep:
    if isinstance(value, SemanticActionStep):
        return value
    if not isinstance(value, Mapping):
        raise SemanticExecutionError("prepare_action must return SemanticActionStep")
    expected = {
        "plan_id",
        "function",
        "atom_ref",
        "action_payload",
        "estimated_duration_us",
    }
    if set(value) != expected:
        raise SemanticExecutionError("prepared semantic action has an invalid shape")
    return SemanticActionStep(**dict(value))


def _coerce_recovery_step(value: Any) -> SemanticRecoveryStep:
    if isinstance(value, SemanticRecoveryStep):
        return value
    if not isinstance(value, Mapping):
        raise SemanticExecutionError("prepare_recovery must return SemanticRecoveryStep")
    expected = {
        "recovery_id",
        "recovery_ref",
        "action_payload",
        "estimated_duration_us",
    }
    if set(value) != expected:
        raise SemanticExecutionError("prepared recovery action has an invalid shape")
    return SemanticRecoveryStep(**dict(value))


class SemanticSuffixExecutor:
    """Execute at most one newly planned Atom before replanning its suffix.

    ``plan_next`` is called positionally as::

        plan_next(remaining_functions, remaining_budget_us, previous_atom, history)

    It must return an object (or mapping) with an ``atoms`` sequence.  Only
    ``atoms[0]`` is touched.  In live mode, ``previous_atom`` is the freshly
    measured opaque ``{"q": ...}`` state from an idle simulator read before
    every replan; historical completed catalog Atoms remain available through
    ``history``.  ``prepare_action`` binds the selected opaque catalog Atom to
    an inline motion.  Live mode additionally requires ``prepare_recovery``.
    Normally only a closed Atom+recovery that fits the reply budget is admitted.
    An optional explicit ``force_first_plan`` callback may select one otherwise
    unavailable first Atom; that Atom runs to terminal and immediately closes
    through canonical recovery while every suffix is dropped.  Controller,
    CAS, hash, fault, and interrupt invariants are never relaxed.
    """

    def __init__(
        self,
        plan_next: Callable[[tuple[str, ...], int, Any, tuple[Any, ...]], Any],
        prepare_action: Callable[[Any], SemanticActionStep | Mapping[str, Any]],
        *,
        prepare_recovery: Callable[
            [Any, int | None, tuple[Any, ...]],
            SemanticRecoveryStep | Mapping[str, Any],
        ]
        | None = None,
        force_first_plan: Callable[[tuple[str, ...], Any, tuple[Any, ...]], Any]
        | None = None,
        discard_plan: Callable[[Any, str], Any] | None = None,
        client: Any = None,
        dry_run: bool = True,
        exclusive_owner: str | None = None,
        audit: Callable[[Mapping[str, Any]], Any] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Any] = time.sleep,
        poll_interval_s: float = 0.05,
        action_timeout_s: float = 45.0,
    ) -> None:
        if not callable(plan_next) or not callable(prepare_action):
            raise TypeError("plan_next and prepare_action must be callable")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a boolean")
        if audit is not None and not callable(audit):
            raise TypeError("audit must be callable")
        if force_first_plan is not None and not callable(force_first_plan):
            raise TypeError("force_first_plan must be callable")
        if discard_plan is not None and not callable(discard_plan):
            raise TypeError("discard_plan must be callable")
        if not callable(monotonic_ns) or not callable(sleep):
            raise TypeError("monotonic_ns and sleep must be callable")
        poll = _finite_float(
            poll_interval_s, "poll_interval_s", minimum=0.001, maximum=10.0
        )
        timeout = _finite_float(
            action_timeout_s, "action_timeout_s", minimum=0.1, maximum=600.0
        )
        if not dry_run:
            post_action = getattr(client, "post_action_sync", None)
            post_semantic_action = getattr(
                client, "post_semantic_action_sync", None
            )
            if (
                client is None
                or not callable(getattr(client, "state_sync", None))
                or not callable(getattr(client, "action_sync", None))
                or not (
                    callable(post_action) or callable(post_semantic_action)
                )
            ):
                raise ValueError(
                    "live semantic execution requires state_sync/action_sync and "
                    "post_action_sync or post_semantic_action_sync"
                )
            if not callable(prepare_recovery):
                raise ValueError("live semantic execution requires prepare_recovery")
            if (
                not isinstance(exclusive_owner, str)
                or _SAFE_TOKEN_RE.fullmatch(exclusive_owner) is None
            ):
                raise ValueError("live semantic execution requires a safe exclusive_owner")
        if force_first_plan is not None and not callable(prepare_recovery):
            raise ValueError("force_first_plan requires prepare_recovery")
        self.plan_next = plan_next
        self.prepare_action = prepare_action
        self.prepare_recovery = prepare_recovery
        self.force_first_plan = force_first_plan
        self.discard_plan = discard_plan
        self.client = client
        self.dry_run = dry_run
        self.exclusive_owner = exclusive_owner
        self.audit = audit
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        self.poll_interval_s = poll
        self.action_timeout_s = timeout
        self._state_lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._audit_seq = 0
        self._active_key: tuple[str, int] | None = None
        self._active_interrupt: threading.Event | None = None
        self._active_accepts_interrupt = False
        self._interrupt_signals: dict[
            tuple[str, int], tuple[str, str]
        ] = {}
        self._interrupt_audited: set[tuple[str, int]] = set()
        self._request_hashes: dict[tuple[str, int], str] = {}
        self._results: dict[tuple[str, int], SemanticExecutionResult] = {}
        self._failed_keys: set[tuple[str, int]] = set()
        self._max_revision: dict[str, int] = {}
        self._safe_stopped_reason: str | None = None

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "dry_run": self.dry_run,
                "active_request": (
                    None
                    if self._active_key is None
                    else {
                        "request_id": self._active_key[0],
                        "revision": self._active_key[1],
                    }
                ),
                "safe_stopped": self._safe_stopped_reason is not None,
                "safe_stopped_reason": self._safe_stopped_reason,
                "completed_request_count": len(self._results),
            }

    def cached_result(
        self, request_id: str, revision: int
    ) -> SemanticExecutionResult | None:
        with self._state_lock:
            return self._results.get((request_id, revision))

    def _read_monotonic_ns(self) -> int:
        value = self._monotonic_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SemanticExecutionError("monotonic_ns returned an invalid integer")
        return value

    def _emit(self, event: str, request: SemanticToolPayload, **details: Any) -> None:
        if self.audit is None:
            return
        with self._audit_lock:
            self._audit_seq += 1
            payload = {
                "schema_version": "semantic_runtime_audit_v1",
                "event_seq": self._audit_seq,
                "event": event,
                "request_id": request.request_id,
                "revision": request.revision,
                "request_fingerprint_sha256": request.fingerprint_sha256,
                "observed_monotonic_ns": self._read_monotonic_ns(),
                **details,
            }
            try:
                self.audit(payload)
            except Exception as exc:
                raise AuditCallbackError(f"audit callback failed: {exc}") from exc

    @staticmethod
    def _planned_atoms(plan: Any) -> tuple[Any, ...]:
        atoms: Any
        if isinstance(plan, Mapping):
            atoms = plan.get("atoms")
        else:
            atoms = getattr(plan, "atoms", None)
        if not isinstance(atoms, (list, tuple)):
            raise SemanticExecutionError("plan_next result must expose an atoms sequence")
        return tuple(atoms)

    @classmethod
    def _first_planned_atom(cls, plan: Any) -> Any | None:
        atoms = cls._planned_atoms(plan)
        return None if not atoms else atoms[0]

    def _plan_first(
        self,
        request: SemanticToolPayload,
        remaining: Sequence[str],
        snapshot: ReplyClockSnapshot,
        previous_atom: Any,
        history: Sequence[Any],
    ) -> Any | None:
        self._emit(
            "plan_requested",
            request,
            ordered_functions=list(remaining),
            remaining_budget_us=snapshot.remaining_us,
        )
        try:
            plan = self.plan_next(
                tuple(remaining),
                snapshot.remaining_us,
                previous_atom,
                tuple(history),
            )
            return self._first_planned_atom(plan)
        except AuditCallbackError:
            raise
        except SemanticExecutionError:
            raise
        except Exception as exc:
            raise SemanticExecutionError(f"plan_next failed: {exc}") from exc

    def _force_first_atom(
        self,
        request: SemanticToolPayload,
        remaining: Sequence[str],
        snapshot: ReplyClockSnapshot,
        previous_atom: Any,
        history: Sequence[Any],
    ) -> Any | None:
        if self.force_first_plan is None:
            return None
        self._emit(
            "force_one_requested",
            request,
            function=remaining[0],
            decision_budget_us=snapshot.remaining_us,
            reply_expired=snapshot.expired,
        )
        try:
            plan = self.force_first_plan(
                tuple(remaining),
                previous_atom,
                tuple(history),
            )
            atoms = self._planned_atoms(plan)
        except AuditCallbackError:
            raise
        except SemanticExecutionError:
            raise
        except Exception as exc:
            raise SemanticExecutionError(f"force_first_plan failed: {exc}") from exc
        if not atoms:
            self._emit(
                "force_one_decision",
                request,
                function=remaining[0],
                would_force=False,
                forced_overrun=False,
                closed_cost_us=None,
                decision_budget_us=snapshot.remaining_us,
                planned_overrun_us=0,
                reason="no_candidate",
            )
            return None
        if len(atoms) != 1:
            raise SemanticExecutionError(
                "force_first_plan must return zero or exactly one Atom"
            )
        return atoms[0]

    def _prepare_atom(self, atom: Any, expected_function: str) -> SemanticActionStep:
        try:
            step = _coerce_action_step(self.prepare_action(atom))
        except SemanticExecutionError:
            raise
        except Exception as exc:
            raise SemanticExecutionError(f"prepare_action failed: {exc}") from exc
        if step.function != expected_function:
            raise SemanticExecutionError(
                f"planned function {step.function!r} does not match suffix head "
                f"{expected_function!r}"
            )
        return step

    def _prepare_neutral_recovery(
        self,
        atom: Any,
        remaining_budget_us: int | None,
        history: Sequence[Any],
    ) -> SemanticRecoveryStep:
        if self.prepare_recovery is None:
            raise SemanticExecutionError("neutral recovery adapter is unavailable")
        try:
            value = self.prepare_recovery(
                atom, remaining_budget_us, tuple(history)
            )
            return _coerce_recovery_step(value)
        except SemanticExecutionError:
            raise
        except Exception as exc:
            raise SemanticExecutionError(f"prepare_recovery failed: {exc}") from exc

    @staticmethod
    def _admission_state(value: Any) -> int:
        if not isinstance(value, Mapping):
            raise SemanticExecutionError("simulator state must be an object")
        if value.get("fault") not in (None, ""):
            raise SemanticExecutionError("simulator reports a fault")
        if "current" in value:
            current = value.get("current")
            if "action" in value and value.get("action") != current:
                raise SemanticExecutionError("simulator current/action state is inconsistent")
        elif "action" in value:
            current = value.get("action")
        else:
            raise SemanticExecutionError("simulator state has no current action field")
        queued = value.get("queued")
        if current is not None or not isinstance(queued, list) or queued:
            raise SemanticConflictError("simulator is not empty for one-at-a-time admission")
        action_seq = value.get("action_seq")
        if isinstance(action_seq, bool) or not isinstance(action_seq, int) or action_seq < 0:
            raise SemanticExecutionError("simulator state has an invalid action_seq")
        return action_seq

    @staticmethod
    def _measured_previous_atom(value: Any) -> dict[str, dict[str, float]]:
        """Return the strict opaque measured-state input understood by Planner.

        The state is copied and numerically normalised so an integration adapter
        cannot observe later mutation of a simulator response object.
        """

        if not isinstance(value, Mapping):
            raise SemanticExecutionError("simulator state must be an object")
        raw_q = value.get("q")
        if not isinstance(raw_q, Mapping) or not raw_q:
            raise SemanticExecutionError("simulator state has no measured q")
        measured_q: dict[str, float] = {}
        for joint, raw_value in raw_q.items():
            if not isinstance(joint, str) or not joint:
                raise SemanticExecutionError("simulator measured q has an invalid joint")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise SemanticExecutionError(
                    f"simulator measured q[{joint!r}] is not numeric"
                )
            numeric = float(raw_value)
            if not math.isfinite(numeric):
                raise SemanticExecutionError(
                    f"simulator measured q[{joint!r}] is not finite"
                )
            measured_q[joint] = numeric
        return {"q": measured_q}

    def _read_planning_previous(
        self,
        request: SemanticToolPayload,
        *,
        interrupt: threading.Event,
    ) -> dict[str, dict[str, float]]:
        """Read one idle measured state immediately before a live replan."""

        try:
            state = self.client.state_sync()
        except Exception as exc:
            # A real G1 barge-in tears down the bound native session.  A
            # snapshot already in flight can therefore fail with a transport
            # reset after the interrupt linearized.  No action has been
            # posted, so this is a normal interrupted turn, not a persistent
            # executor fault.
            if interrupt.is_set():
                raise _InterruptedBeforeSubmit from exc
            if isinstance(exc, (SemanticExecutionError, SemanticConflictError)):
                raise
            raise SemanticExecutionError(
                f"cannot read simulator planning state: {exc}"
            ) from exc
        try:
            action_seq = self._admission_state(state)
            previous = self._measured_previous_atom(state)
        except (SemanticExecutionError, SemanticConflictError):
            raise
        q_sha256 = hashlib.sha256(
            _canonical_json_bytes(previous["q"])
        ).hexdigest()
        self._emit(
            "planning_state_observed",
            request,
            action_seq=action_seq,
            measured_q_sha256=q_sha256,
            measured_joint_count=len(previous["q"]),
        )
        return previous

    @staticmethod
    def _verify_action_record(
        value: Any,
        *,
        action_id: str,
        payload_sha256: str,
    ) -> str:
        if not isinstance(value, Mapping):
            raise SemanticExecutionError("own action record must be an object")
        if value.get("id") != action_id:
            raise SemanticExecutionError("own action record id mismatch")
        if value.get("payload_sha256") != payload_sha256:
            raise SemanticExecutionError("own action payload_sha256 mismatch")
        status = value.get("status")
        if status not in _NONTERMINAL_STATUSES | _TERMINAL_STATUSES:
            raise SemanticExecutionError("own action record has an unknown status")
        return str(status)

    @staticmethod
    def _action_id(
        request: SemanticToolPayload,
        *,
        ordinal: int,
        kind: str,
        step_id: str,
        payload_sha256: str,
    ) -> str:
        body = {
            "schema_version": "semantic_action_id_v1",
            "request_fingerprint_sha256": request.fingerprint_sha256,
            "ordinal": ordinal,
            "kind": kind,
            "step_id": step_id,
            "prepared_payload_sha256": payload_sha256,
        }
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        return f"sem-{digest[:40]}"

    def _turn_owner(self, request: SemanticToolPayload) -> str:
        body = {
            "schema_version": "semantic_turn_owner_v1",
            "owner_namespace": self.exclusive_owner,
            "request_fingerprint_sha256": request.fingerprint_sha256,
        }
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        return f"semturn-{digest[:40]}"

    def _submit_and_wait(
        self,
        request: SemanticToolPayload,
        *,
        interrupt: threading.Event,
        ordinal: int,
        kind: str,
        step_id: str,
        action_payload: Mapping[str, Any],
        prepared_payload_sha256: str,
        atom_ref: Mapping[str, Any],
        semantic_function: str | None = None,
        planning_q: Mapping[str, Any] | None = None,
        reply_clock: ReplyClock | None = None,
        required_budget_us: int | None = None,
        force_reply_budget: bool = False,
    ) -> _SubmissionReceipt:
        if not isinstance(force_reply_budget, bool):
            raise SemanticExecutionError("force_reply_budget must be a boolean")
        if kind == "semantic":
            if semantic_function not in RUNTIME_FUNCTION_SET:
                raise SemanticExecutionError(
                    "semantic submission requires an exact runtime function"
                )
            if planning_q is None or reply_clock is None:
                raise SemanticExecutionError(
                    "semantic submission requires planning_q and an explicit ReplyClock"
                )
            required_budget_us = _positive_int(
                required_budget_us, "required_budget_us"
            )
        elif semantic_function is not None or any(
            value is not None
            for value in (planning_q, reply_clock, required_budget_us)
        ) or force_reply_budget:
            raise SemanticExecutionError(
                "only semantic submission may carry planning budget state"
            )
        try:
            state = self.client.state_sync()
        except Exception as exc:
            # See `_read_planning_previous`: interruption may close the native
            # session while this pre-POST admission snapshot is in flight.
            if interrupt.is_set():
                raise _InterruptedBeforeSubmit from exc
            if isinstance(exc, (SemanticExecutionError, SemanticConflictError)):
                raise
            raise SemanticExecutionError(
                f"cannot read simulator admission state: {exc}"
            ) from exc
        try:
            action_seq = self._admission_state(state)
            if planning_q is not None:
                admission_q = self._measured_previous_atom(state)["q"]
                normalized_planning_q = self._measured_previous_atom(
                    {"q": planning_q}
                )["q"]
                if set(admission_q) != set(normalized_planning_q):
                    raise SemanticConflictError(
                        "simulator measured q joints changed between planning and admission"
                    )
                maximum_q_drift = max(
                    abs(admission_q[joint] - normalized_planning_q[joint])
                    for joint in admission_q
                )
                if maximum_q_drift > ADMISSION_Q_TOLERANCE_RAD:
                    raise _ReplanBeforeSubmit(
                        admission_q,
                        maximum_q_drift,
                    )
                required_budget_us += ADMISSION_Q_TICK_GUARD_US
            else:
                maximum_q_drift = None
        except (
            SemanticExecutionError,
            SemanticConflictError,
            _ReplanBeforeSubmit,
        ):
            raise

        action_id = self._action_id(
            request,
            ordinal=ordinal,
            kind=kind,
            step_id=step_id,
            payload_sha256=prepared_payload_sha256,
        )
        turn_owner = self._turn_owner(request)
        flat = {
            **dict(action_payload),
            "id": action_id,
            "mode": "enqueue",
            "require_empty": True,
            "expected_action_seq": action_seq,
            "exclusive_owner": turn_owner,
        }
        flat["payload_sha256"] = canonical_action_payload_sha256(flat)
        payload_sha256 = flat["payload_sha256"]
        self._emit(
            "submit_intent",
            request,
            kind=kind,
            action_id=action_id,
            step_id=step_id,
            atom_ref=dict(atom_ref),
            expected_action_seq=action_seq,
            exclusive_owner=turn_owner,
            prepared_payload_sha256=prepared_payload_sha256,
            payload_sha256=payload_sha256,
            maximum_q_drift_rad=maximum_q_drift,
            q_tick_guard_us=(
                ADMISSION_Q_TICK_GUARD_US if planning_q is not None else 0
            ),
            required_budget_us=required_budget_us,
            force_one=force_reply_budget,
            semantic_function=semantic_function,
        )

        # This lock contains only the interrupt-vs-commit linearization point;
        # it never contains network or audit I/O.  Therefore signal_interrupt()
        # remains constant-time even while the committed POST is blocked.
        closed_cost_us: int | None = None
        decision_budget_us: int | None = None
        planned_overrun_us = 0
        if force_reply_budget:
            # Durable audit is deliberately before the final interrupt-vs-POST
            # linearization point.  A slow/fsyncing sink must never leave a
            # window in which a barge-in is accepted but the forced Atom is
            # nevertheless posted afterwards.  The receipt metadata is
            # re-snapshotted below after this audit and is therefore final.
            try:
                audit_snapshot = reply_clock.snapshot()
            except ReplyClockError as exc:
                raise SemanticExecutionError(
                    f"reply clock snapshot failed before force audit: {exc}"
                ) from exc
            audit_closed_cost_us = required_budget_us
            audit_decision_budget_us = audit_snapshot.remaining_us
            audit_planned_overrun_us = max(
                0, audit_closed_cost_us - audit_decision_budget_us
            )
            if (
                not audit_snapshot.expired
                and audit_decision_budget_us >= audit_closed_cost_us
            ):
                raise SemanticExecutionError(
                    "force-one cannot bypass a ReplyClock budget that still fits"
                )
            self._emit(
                "force_one_precommit",
                request,
                function=semantic_function,
                atom_ref=dict(atom_ref),
                would_force=True,
                forced_overrun=True,
                closed_cost_us=audit_closed_cost_us,
                decision_budget_us=audit_decision_budget_us,
                planned_overrun_us=audit_planned_overrun_us,
                reason="budget_no_fit",
            )
        with self._state_lock:
            if interrupt.is_set():
                raise _InterruptedBeforeSubmit
            if reply_clock is not None:
                try:
                    pre_submit = reply_clock.snapshot()
                except ReplyClockError as exc:
                    raise SemanticExecutionError(
                        f"reply clock snapshot failed before POST: {exc}"
                    ) from exc
                closed_cost_us = required_budget_us
                decision_budget_us = pre_submit.remaining_us
                planned_overrun_us = max(
                    0, closed_cost_us - decision_budget_us
                )
                if (
                    pre_submit.expired
                    or decision_budget_us < closed_cost_us
                ):
                    if not force_reply_budget:
                        raise _BudgetBeforeSubmit(pre_submit, closed_cost_us)
                elif force_reply_budget:
                    raise SemanticExecutionError(
                        "force-one cannot bypass a ReplyClock budget that still fits"
                    )

        submit_exc: Exception | None = None
        try:
            post_semantic_action = getattr(
                self.client, "post_semantic_action_sync", None
            )
            if callable(post_semantic_action):
                response = post_semantic_action(
                    flat,
                    kind=kind,
                    atom_ref=dict(atom_ref),
                )
            else:
                response = self.client.post_action_sync(flat)
        except Exception as exc:
            submit_exc = exc
            response = None
        if submit_exc is not None and _typed_http_status(submit_exc) == 409:
            # A typed atomic-admission rejection is definitive: the server did
            # not accept this action.  This is the only prompt no-drain branch.
            raise SemanticConflictError(
                "simulator rejected atomic admission with HTTP 409"
            ) from submit_exc

        submission_response_error: SemanticExecutionError | None = None
        if submit_exc is not None:
            response_contract_valid: bool | None = None
        elif not isinstance(response, Mapping) or response.get("accepted") is not True:
            response_contract_valid = False
            submission_response_error = SemanticExecutionError(
                "submission response did not explicitly accept own action"
            )
        elif response.get("id") != action_id:
            response_contract_valid = False
            submission_response_error = SemanticExecutionError(
                "submission response returned a different action id"
            )
        else:
            response_contract_valid = True
            response_sha = response.get("payload_sha256")
            if response_sha is not None and response_sha != payload_sha256:
                response_contract_valid = False
                submission_response_error = SemanticExecutionError(
                    "submission response payload_sha256 mismatch"
                )
        post_submit_audit_error: AuditCallbackError | None = None

        def emit_after_submit(event: str, **details: Any) -> None:
            """Remember audit failure but never abandon an accepted own action."""

            nonlocal post_submit_audit_error
            if post_submit_audit_error is not None:
                return
            try:
                self._emit(event, request, **details)
            except Exception as exc:
                if isinstance(exc, AuditCallbackError):
                    post_submit_audit_error = exc
                else:
                    post_submit_audit_error = AuditCallbackError(
                        f"post-submit audit failed: {exc}"
                    )

        emit_after_submit(
            "action_submitted",
            kind=kind,
            action_id=action_id,
            payload_sha256=payload_sha256,
            response_contract_valid=response_contract_valid,
            response_transport_error=(
                None
                if submit_exc is None
                else f"{type(submit_exc).__name__}: {submit_exc}"[:512]
            ),
            force_one=force_reply_budget,
            closed_cost_us=closed_cost_us,
            decision_budget_us=decision_budget_us,
            planned_overrun_us=planned_overrun_us,
        )

        started = self._read_monotonic_ns()
        timeout_ns = int(self.action_timeout_s * 1_000_000_000)
        record: Mapping[str, Any] | None = None
        while True:
            if record is None:
                try:
                    record = self.client.action_sync(action_id)
                except Exception as exc:
                    # POST has already been explicitly accepted.  A transient
                    # observation error cannot justify abandoning the active
                    # own action, and the POST is never retried.  Keep polling
                    # the exact ID until terminal or the bounded timeout.
                    now = self._read_monotonic_ns()
                    emit_after_submit(
                        "action_observation_retry",
                        kind=kind,
                        action_id=action_id,
                        payload_sha256=payload_sha256,
                        error=f"{type(exc).__name__}: {exc}"[:512],
                    )
                    if now - started >= timeout_ns:
                        if submit_exc is not None:
                            raise SemanticExecutionError(
                                "action submission remained ambiguous through the "
                                "bounded exact-ID drain; no POST retry or cancel was sent"
                            ) from submit_exc
                        raise SemanticExecutionError(
                            f"timed out observing own action {action_id} after "
                            f"transient read errors; no cancel was sent"
                        ) from exc
                    self._sleep(self.poll_interval_s)
                    continue
            status = self._verify_action_record(
                record,
                action_id=action_id,
                payload_sha256=payload_sha256,
            )
            emit_after_submit(
                "action_observed",
                kind=kind,
                action_id=action_id,
                payload_sha256=payload_sha256,
                status=status,
            )
            if status == "completed":
                if submission_response_error is not None:
                    raise submission_response_error
                if post_submit_audit_error is not None:
                    raise post_submit_audit_error
                return _SubmissionReceipt(
                    action_id=action_id,
                    record=dict(record),
                    forced_reply_budget=force_reply_budget,
                    closed_cost_us=closed_cost_us,
                    decision_budget_us=decision_budget_us,
                    planned_overrun_us=planned_overrun_us,
                )
            if status in _TERMINAL_STATUSES:
                raise SemanticExecutionError(
                    f"own action {action_id} ended with status {status}"
                )
            if self._read_monotonic_ns() - started >= timeout_ns:
                raise SemanticExecutionError(
                    f"timed out waiting for own action {action_id}; no cancel was sent"
                )
            self._sleep(self.poll_interval_s)
            record = None

    def _result(
        self,
        request: SemanticToolPayload,
        *,
        status: str,
        planned: Sequence[str],
        completed: Sequence[str],
        pending: Sequence[str],
        dropped: Sequence[str],
        action_ids: Sequence[str],
        recovery_action_id: str | None,
        closed_neutral: bool,
        would_force_first_atom: bool = False,
        forced_first_atom: bool = False,
        planned_overrun_us: int = 0,
    ) -> SemanticExecutionResult:
        if (
            not isinstance(would_force_first_atom, bool)
            or not isinstance(forced_first_atom, bool)
            or isinstance(planned_overrun_us, bool)
            or not isinstance(planned_overrun_us, int)
            or planned_overrun_us < 0
            or forced_first_atom and not would_force_first_atom
        ):
            raise SemanticExecutionError("force-one result metadata is inconsistent")
        with self._state_lock:
            key = (request.request_id, request.revision)
            signaled = (
                self._active_key == key
                and self._active_interrupt is not None
                and self._active_interrupt.is_set()
            )
            # `_result` is the linearization point after which a late barge-in
            # cannot truthfully change this request's committed outcome.
            if self._active_key == key:
                self._active_accepts_interrupt = False
        if signaled and status != "interrupted":
            merged_dropped = list(dropped)
            for function in pending:
                if function not in merged_dropped:
                    merged_dropped.append(function)
            status = "interrupted"
            pending = ()
            dropped = tuple(merged_dropped)
        return SemanticExecutionResult(
            request_id=request.request_id,
            revision=request.revision,
            request_fingerprint_sha256=request.fingerprint_sha256,
            status=status,
            planned_functions=tuple(planned),
            completed_functions=tuple(completed),
            pending_functions=tuple(pending),
            dropped_functions=tuple(dropped),
            action_ids=tuple(action_ids),
            recovery_action_id=recovery_action_id,
            closed_neutral=closed_neutral,
            dry_run=self.dry_run,
            would_force_first_atom=would_force_first_atom,
            forced_first_atom=forced_first_atom,
            planned_overrun_us=planned_overrun_us,
        )

    def signal_interrupt(
        self,
        request_id: str,
        revision: int,
        *,
        reason: str = "interrupt",
    ) -> bool:
        """Constant-time exact-request signal with no audit or controller I/O."""

        with self._state_lock:
            key = (request_id, revision)
            if (
                key != self._active_key
                or self._active_interrupt is None
                or not self._active_accepts_interrupt
            ):
                return False
            self._active_interrupt.set()
            self._interrupt_signals.setdefault(
                key,
                (self._request_hashes[key], str(reason)[:256]),
            )
            return True

    def audit_signaled_interrupt(self, request_id: str, revision: int) -> bool:
        """Durably audit a prior signal; callers may run this off the event loop."""

        key = (request_id, revision)
        with self._state_lock:
            signal = self._interrupt_signals.get(key)
            if signal is None or key in self._interrupt_audited:
                return False
            # Claim once before external I/O.  If a sink writes and then raises,
            # retrying could duplicate an append-only event.
            self._interrupt_audited.add(key)
        fingerprint, reason = signal
        if self.audit is not None:
            with self._audit_lock:
                self._audit_seq += 1
                event = {
                    "schema_version": "semantic_runtime_audit_v1",
                    "event_seq": self._audit_seq,
                    "event": "interrupted",
                    "request_id": request_id,
                    "revision": revision,
                    "request_fingerprint_sha256": fingerprint,
                    "observed_monotonic_ns": self._read_monotonic_ns(),
                    "reason": reason,
                }
                try:
                    self.audit(event)
                except Exception as exc:
                    raise AuditCallbackError(f"audit callback failed: {exc}") from exc
        return True

    def interrupt(self, request_id: str, revision: int, *, reason: str = "interrupt") -> bool:
        """Compatibility API: signal first, then synchronously audit the signal."""

        if not self.signal_interrupt(request_id, revision, reason=reason):
            return False
        self.audit_signaled_interrupt(request_id, revision)
        return True

    def execute(
        self,
        request: SemanticToolPayload,
        *,
        reply_clock: ReplyClock | None = None,
    ) -> SemanticExecutionResult:
        if not isinstance(request, SemanticToolPayload):
            raise TypeError("request must be a SemanticToolPayload")
        key = (request.request_id, request.revision)
        fingerprint = request.fingerprint_sha256
        with self._state_lock:
            maximum = self._max_revision.get(request.request_id, 0)
            if request.revision < maximum:
                raise SemanticConflictError("stale revision cannot be replayed")
            known = self._request_hashes.get(key)
            if known is not None and known != fingerprint:
                raise SemanticConflictError(
                    "request/revision was reused with different content"
                )
            if key in self._failed_keys:
                raise SemanticConflictError(
                    "request/revision previously failed; use a new revision"
                )
            cached = self._results.get(key)
            if cached is not None:
                return cached
            if not self.dry_run and self._safe_stopped_reason is not None:
                raise SemanticConflictError(
                    f"semantic executor is safe-stopped: {self._safe_stopped_reason}"
                )
            if self._active_key is not None:
                raise SemanticConflictError("another semantic request is active")
            self._request_hashes[key] = fingerprint
            self._max_revision[request.request_id] = max(maximum, request.revision)
            self._active_key = key
            interrupt = threading.Event()
            self._active_interrupt = interrupt
            self._active_accepts_interrupt = True

        try:
            if not self.dry_run and reply_clock is None:
                raise SemanticExecutionError(
                    "live semantic execution requires an explicit ReplyClock "
                    "bound to the frozen output start"
                )
            clock = reply_clock or ReplyClock.from_payload(
                request, monotonic_ns=self._monotonic_ns
            )
            try:
                clock.assert_binding(
                    request.request_id,
                    request.revision,
                    request.estimated_spoken_duration_us,
                )
            except ReplyClockError as exc:
                raise SemanticExecutionError(f"reply clock binding failed: {exc}") from exc
            self._emit(
                "request_started",
                request,
                dry_run=self.dry_run,
                ordered_functions=list(request.ordered_functions),
                estimated_spoken_duration_us=request.estimated_spoken_duration_us,
                effective_reply_budget_us=clock.effective_duration_us,
                reply_budget_policy=clock.budget_policy,
            )
            result = self._execute_bound(request, clock, interrupt)
            self._emit(
                "request_finished",
                request,
                status=result.status,
                completed_functions=list(result.completed_functions),
                dropped_functions=list(result.dropped_functions),
                action_ids=list(result.action_ids),
                recovery_action_id=result.recovery_action_id,
                closed_neutral=result.closed_neutral,
                would_force_first_atom=result.would_force_first_atom,
                forced_first_atom=result.forced_first_atom,
                planned_overrun_us=result.planned_overrun_us,
            )
            with self._state_lock:
                self._results[key] = result
            return result
        except Exception as exc:
            with self._state_lock:
                self._active_accepts_interrupt = False
                self._failed_keys.add(key)
                if not self.dry_run:
                    self._safe_stopped_reason = f"{type(exc).__name__}: {exc}"[:512]
            raise
        finally:
            with self._state_lock:
                if self._active_key == key:
                    self._active_accepts_interrupt = False
                    self._active_key = None
                    self._active_interrupt = None

    def _execute_bound(
        self,
        request: SemanticToolPayload,
        clock: ReplyClock,
        interrupt: threading.Event,
    ) -> SemanticExecutionResult:
        remaining = list(request.ordered_functions)
        planned: list[str] = []
        completed: list[str] = []
        action_ids: list[str] = []
        history: list[Any] = []
        pending_recovery: SemanticRecoveryStep | None = None
        recovery_action_id: str | None = None
        planning_override: dict[str, dict[str, float]] | None = None
        admission_replans = 0
        would_force_first_atom = False
        forced_first_atom = False
        planned_overrun_us = 0

        def clock_snapshot() -> ReplyClockSnapshot:
            try:
                return clock.snapshot()
            except ReplyClockError as exc:
                raise SemanticExecutionError(f"reply clock snapshot failed: {exc}") from exc

        def interrupted_result(dropped: Sequence[str]) -> SemanticExecutionResult:
            return self._result(
                request,
                status="interrupted",
                planned=planned,
                completed=completed,
                pending=(),
                dropped=dropped,
                action_ids=action_ids,
                recovery_action_id=None,
                closed_neutral=not completed,
                would_force_first_atom=would_force_first_atom,
                forced_first_atom=forced_first_atom,
                planned_overrun_us=planned_overrun_us,
            )

        def finish_closed_prefix(
            status: str,
            dropped: Sequence[str],
        ) -> SemanticExecutionResult:
            nonlocal recovery_action_id
            if not completed:
                return self._result(
                    request,
                    status=status,
                    planned=planned,
                    completed=(),
                    pending=(),
                    dropped=dropped,
                    action_ids=(),
                    recovery_action_id=None,
                    closed_neutral=True,
                    would_force_first_atom=would_force_first_atom,
                    forced_first_atom=forced_first_atom,
                    planned_overrun_us=planned_overrun_us,
                )
            if interrupt.is_set():
                return interrupted_result(dropped)
            if pending_recovery is None:
                raise SemanticExecutionError(
                    "completed semantic prefix has no reserved neutral recovery"
                )
            try:
                recovery_receipt = self._submit_and_wait(
                    request,
                    interrupt=interrupt,
                    ordinal=len(action_ids) + 1,
                    kind="recovery",
                    step_id=pending_recovery.recovery_id,
                    action_payload=pending_recovery.action_payload,
                    prepared_payload_sha256=pending_recovery.action_payload_sha256,
                    atom_ref=pending_recovery.recovery_ref,
                )
            except _InterruptedBeforeSubmit:
                return interrupted_result(dropped)
            recovery_action_id = recovery_receipt.action_id
            action_ids.append(recovery_action_id)
            if forced_first_atom:
                self._emit(
                    "force_one_closed",
                    request,
                    recovery_action_id=recovery_action_id,
                    planned_overrun_us=planned_overrun_us,
                    closed_neutral=True,
                    dropped_functions=list(dropped),
                )
            # A barge-in that linearized after recovery POST is not allowed to
            # cancel it.  `_result` observes the signal and reports interrupted
            # while preserving the now-true neutral closure.
            final_status = "interrupted" if interrupt.is_set() else status
            return self._result(
                request,
                status=final_status,
                planned=planned,
                completed=completed,
                pending=(),
                dropped=dropped,
                action_ids=action_ids,
                recovery_action_id=recovery_action_id,
                closed_neutral=True,
                would_force_first_atom=would_force_first_atom,
                forced_first_atom=forced_first_atom,
                planned_overrun_us=planned_overrun_us,
            )

        if not remaining:
            return self._result(
                request,
                status="completed",
                planned=(),
                completed=(),
                pending=(),
                dropped=(),
                action_ids=(),
                recovery_action_id=None,
                closed_neutral=True,
            )

        while remaining:
            snapshot = clock_snapshot()
            force_available = (
                self.force_first_plan is not None
                and not completed
                and not action_ids
                and len(remaining) == len(request.ordered_functions)
            )
            if interrupt.is_set():
                return interrupted_result(tuple(remaining))
            if (
                snapshot.expired or snapshot.remaining_us <= 0
            ) and not force_available:
                return finish_closed_prefix("expired", tuple(remaining))

            planning_previous: Any = None
            if not self.dry_run:
                if planning_override is None:
                    try:
                        planning_previous = self._read_planning_previous(
                            request, interrupt=interrupt
                        )
                    except _InterruptedBeforeSubmit:
                        return interrupted_result(tuple(remaining))
                else:
                    # This is the exact idle admission snapshot that rejected
                    # the stale prepared bridge.  Replan from it, then take a
                    # new admission snapshot/CAS before any POST.
                    planning_previous = planning_override
                # The state read may block while an interrupt arrives or consume
                # reply time.  Re-snapshot before exposing budget to Planner.
                snapshot = clock_snapshot()
                if interrupt.is_set():
                    return interrupted_result(tuple(remaining))
                if (
                    snapshot.expired or snapshot.remaining_us <= 0
                ) and not force_available:
                    return finish_closed_prefix("expired", tuple(remaining))

            atom = self._plan_first(
                request, remaining, snapshot, planning_previous, history
            )
            force_this_atom = False
            if atom is None:
                if not force_available:
                    return finish_closed_prefix(
                        "partial_no_plan" if completed else "no_plan",
                        tuple(remaining),
                    )
                atom = self._force_first_atom(
                    request,
                    remaining,
                    snapshot,
                    planning_previous,
                    history,
                )
                if atom is None:
                    return finish_closed_prefix("no_candidate", tuple(remaining))
                force_this_atom = True
                would_force_first_atom = True

            step = self._prepare_atom(atom, remaining[0])
            planned.append(step.function)
            self._emit(
                "plan_selected",
                request,
                function=step.function,
                plan_id=step.plan_id,
                atom_ref=step.atom_ref,
                estimated_duration_us=step.estimated_duration_us,
                prepared_payload_sha256=step.action_payload_sha256,
                force_one=force_this_atom,
            )
            if self.dry_run:
                # With force-one enabled, dry-run is a decision preview rather
                # than merely a Planner preview.  Reproduce the live closed-loop
                # fit calculation (Atom + canonical recovery + one admission-q
                # tick) without reading simulator state or issuing a POST.  This
                # keeps guard-only and prepare-time deadline crossings from being
                # reported as normal dry-runs when live would force the same Atom.
                if not force_this_atom and not force_available:
                    self._emit(
                        "dry_run",
                        request,
                        function=step.function,
                        plan_id=step.plan_id,
                        atom_ref=step.atom_ref,
                    )
                    return self._result(
                        request,
                        status="dry_run",
                        planned=planned,
                        completed=completed,
                        pending=remaining,
                        dropped=(),
                        action_ids=(),
                        recovery_action_id=None,
                        closed_neutral=True,
                    )

                prospective_history = tuple(history) + (atom,)
                if force_this_atom:
                    prospective_recovery = self._prepare_neutral_recovery(
                        atom, None, prospective_history
                    )
                else:
                    after_prepare = clock_snapshot()
                    budget_after_atom = (
                        after_prepare.remaining_us - step.estimated_duration_us
                    )
                    prospective_recovery = None
                    if budget_after_atom > 0:
                        prospective_recovery = self._prepare_neutral_recovery(
                            atom, budget_after_atom, prospective_history
                        )
                    if prospective_recovery is not None:
                        normal_decision = clock_snapshot()
                        normal_closed_cost_us = (
                            step.estimated_duration_us
                            + prospective_recovery.estimated_duration_us
                            + ADMISSION_Q_TICK_GUARD_US
                        )
                        if (
                            not normal_decision.expired
                            and normal_decision.remaining_us
                            >= normal_closed_cost_us
                        ):
                            self._emit(
                                "dry_run",
                                request,
                                function=step.function,
                                plan_id=step.plan_id,
                                atom_ref=step.atom_ref,
                            )
                            return self._result(
                                request,
                                status="dry_run",
                                planned=planned,
                                completed=completed,
                                pending=remaining,
                                dropped=(),
                                action_ids=(),
                                recovery_action_id=None,
                                closed_neutral=True,
                            )
                    # Retain the already selected first Atom.  Only its reply
                    # budget fit changes; the explicit forced recovery is
                    # prepared without a remaining-budget constraint.
                    force_this_atom = True
                    would_force_first_atom = True
                    prospective_recovery = self._prepare_neutral_recovery(
                        atom, None, prospective_history
                    )

                decision = clock_snapshot()
                closed_cost_us = (
                    step.estimated_duration_us
                    + prospective_recovery.estimated_duration_us
                    + ADMISSION_Q_TICK_GUARD_US
                )
                preview_overrun_us = max(
                    0, closed_cost_us - decision.remaining_us
                )
                if preview_overrun_us <= 0:
                    raise SemanticExecutionError(
                        "force-one selected an Atom that still fits ReplyClock"
                    )
                self._emit(
                    "force_one_decision",
                    request,
                    function=step.function,
                    plan_id=step.plan_id,
                    atom_ref=step.atom_ref,
                    would_force=True,
                    forced_overrun=True,
                    closed_cost_us=closed_cost_us,
                    decision_budget_us=decision.remaining_us,
                    planned_overrun_us=preview_overrun_us,
                    reason="budget_no_fit",
                    dry_run=True,
                )
                return self._result(
                    request,
                    status="dry_run_forced_first",
                    planned=planned,
                    completed=(),
                    pending=(remaining[0],),
                    dropped=tuple(remaining[1:]),
                    action_ids=(),
                    recovery_action_id=None,
                    closed_neutral=True,
                    would_force_first_atom=True,
                    forced_first_atom=False,
                    planned_overrun_us=preview_overrun_us,
                )

            # Planner, adapters, and audit callbacks consume real reply time.
            # Re-snapshot before reserving both the Atom and its recovery.
            after_prepare = clock_snapshot()
            if interrupt.is_set():
                return interrupted_result(tuple(remaining))
            prospective_history = tuple(history) + (atom,)
            budget_after_atom = (
                after_prepare.remaining_us - step.estimated_duration_us
            )
            if force_this_atom:
                prospective_recovery = self._prepare_neutral_recovery(
                    atom, None, prospective_history
                )
            elif budget_after_atom <= 0:
                if force_available:
                    force_this_atom = True
                    would_force_first_atom = True
                    prospective_recovery = self._prepare_neutral_recovery(
                        atom, None, prospective_history
                    )
                else:
                    prospective_recovery = None
            else:
                prospective_recovery = self._prepare_neutral_recovery(
                    atom, budget_after_atom, prospective_history
                )
                if prospective_recovery.estimated_duration_us > budget_after_atom:
                    if force_available:
                        force_this_atom = True
                        would_force_first_atom = True
                        prospective_recovery = self._prepare_neutral_recovery(
                            atom, None, prospective_history
                        )
                    else:
                        prospective_recovery = None
            if prospective_recovery is None:
                return finish_closed_prefix(
                    "partial_budget" if completed else "no_safe_prefix",
                    tuple(remaining),
                )

            required_budget_us = (
                step.estimated_duration_us
                + prospective_recovery.estimated_duration_us
            )

            def emit_force_decision(
                decision: ReplyClockSnapshot,
                *,
                reason: str,
            ) -> None:
                closed_cost_us = required_budget_us + ADMISSION_Q_TICK_GUARD_US
                overrun_us = max(0, closed_cost_us - decision.remaining_us)
                if overrun_us <= 0:
                    raise SemanticExecutionError(
                        "force-one selected a closed action that still fits ReplyClock"
                    )
                self._emit(
                    "force_one_decision",
                    request,
                    function=step.function,
                    plan_id=step.plan_id,
                    atom_ref=step.atom_ref,
                    would_force=True,
                    forced_overrun=True,
                    closed_cost_us=closed_cost_us,
                    decision_budget_us=decision.remaining_us,
                    planned_overrun_us=overrun_us,
                    reason=reason,
                    dry_run=False,
                )

            try:
                if force_this_atom:
                    emit_force_decision(
                        clock_snapshot(), reason="budget_no_fit"
                    )
                try:
                    receipt = self._submit_and_wait(
                        request,
                        interrupt=interrupt,
                        ordinal=len(action_ids) + 1,
                        kind="semantic",
                        step_id=step.plan_id,
                        action_payload=step.action_payload,
                        prepared_payload_sha256=step.action_payload_sha256,
                        atom_ref=step.atom_ref,
                        semantic_function=step.function,
                        planning_q=planning_previous["q"],
                        reply_clock=clock,
                        required_budget_us=required_budget_us,
                        force_reply_budget=force_this_atom,
                    )
                except _BudgetBeforeSubmit as budget_no_fit:
                    if not force_available or force_this_atom:
                        raise
                    if interrupt.is_set():
                        return interrupted_result(tuple(remaining))
                    force_this_atom = True
                    would_force_first_atom = True
                    prospective_recovery = self._prepare_neutral_recovery(
                        atom, None, prospective_history
                    )
                    required_budget_us = (
                        step.estimated_duration_us
                        + prospective_recovery.estimated_duration_us
                    )
                    self._emit(
                        "submit_budget_rejected",
                        request,
                        function=step.function,
                        plan_id=step.plan_id,
                        would_force=True,
                        closed_cost_us=budget_no_fit.required_budget_us,
                        decision_budget_us=budget_no_fit.snapshot.remaining_us,
                    )
                    emit_force_decision(
                        clock_snapshot(), reason="precommit_deadline"
                    )
                    receipt = self._submit_and_wait(
                        request,
                        interrupt=interrupt,
                        ordinal=len(action_ids) + 1,
                        kind="semantic",
                        step_id=step.plan_id,
                        action_payload=step.action_payload,
                        prepared_payload_sha256=step.action_payload_sha256,
                        atom_ref=step.atom_ref,
                        semantic_function=step.function,
                        planning_q=planning_previous["q"],
                        reply_clock=clock,
                        required_budget_us=required_budget_us,
                        force_reply_budget=True,
                    )
            except _InterruptedBeforeSubmit:
                return interrupted_result(tuple(remaining))
            except _BudgetBeforeSubmit:
                return finish_closed_prefix(
                    "partial_budget" if completed else "no_safe_prefix",
                    tuple(remaining),
                )
            except _ReplanBeforeSubmit as replan:
                # No POST occurred.  Undo only this attempt-local selection so
                # stateful integration adapters cannot retain a stale Catalog
                # row or duration binding across the retry.
                planned.pop()
                if self.discard_plan is not None:
                    try:
                        self.discard_plan(atom, ADMISSION_Q_DRIFT_REASON)
                    except Exception as exc:
                        raise SemanticExecutionError(
                            f"discard_plan failed: {exc}"
                        ) from exc
                admission_replans += 1
                measured_q_sha256 = hashlib.sha256(
                    _canonical_json_bytes(replan.previous_atom["q"])
                ).hexdigest()
                self._emit(
                    "plan_discarded",
                    request,
                    function=step.function,
                    plan_id=step.plan_id,
                    atom_ref=step.atom_ref,
                    reason=ADMISSION_Q_DRIFT_REASON,
                    maximum_q_drift_rad=replan.maximum_drift_rad,
                    measured_q_sha256=measured_q_sha256,
                    replan_attempt=admission_replans,
                    maximum_replan_attempts=MAX_ADMISSION_REPLAN_ATTEMPTS,
                )
                if interrupt.is_set():
                    return interrupted_result(tuple(remaining))
                if admission_replans > MAX_ADMISSION_REPLAN_ATTEMPTS:
                    return finish_closed_prefix(
                        (
                            "partial_unstable_admission"
                            if completed
                            else "no_stable_admission"
                        ),
                        tuple(remaining),
                    )
                planning_override = replan.previous_atom
                continue
            action_id = receipt.action_id
            action_ids.append(action_id)
            completed.append(step.function)
            history.append(atom)
            remaining.pop(0)
            pending_recovery = prospective_recovery
            planning_override = None
            admission_replans = 0

            if receipt.forced_reply_budget:
                forced_first_atom = True
                would_force_first_atom = True
                planned_overrun_us = receipt.planned_overrun_us
                if interrupt.is_set():
                    return interrupted_result(tuple(remaining))
                return finish_closed_prefix(
                    "forced_first_completed", tuple(remaining)
                )

            # Interrupt never triggers an extra controller command.  Natural
            # reply expiry is different: the neutral recovery was reserved and
            # prepared before this Atom was admitted, so it must still run.
            if interrupt.is_set():
                return interrupted_result(tuple(remaining))
            if clock_snapshot().expired:
                return finish_closed_prefix("expired", tuple(remaining))

        return finish_closed_prefix("completed", ())


__all__ = [
    "ADMISSION_Q_DRIFT_REASON",
    "ADMISSION_Q_TICK_GUARD_US",
    "ADMISSION_Q_TOLERANCE_RAD",
    "MAX_ADMISSION_REPLAN_ATTEMPTS",
    "AuditCallbackError",
    "ABSOLUTE_ORDERED_FUNCTIONS_GUARD",
    "MIN_ESTIMATED_SPOKEN_DURATION_S",
    "RUNTIME_FUNCTIONS",
    "SUPPORTED_BRIDGE_POLICY",
    "SemanticActionStep",
    "SemanticConflictError",
    "SemanticExecutionError",
    "SemanticExecutionResult",
    "SemanticRecoveryStep",
    "SemanticRuntimeError",
    "SemanticSuffixExecutor",
    "SemanticToolPayload",
    "ToolPayloadValidationError",
    "canonical_action_payload_sha256",
    "gemini_tool_parameters_schema",
    "validate_gemini_tool_payload",
]
