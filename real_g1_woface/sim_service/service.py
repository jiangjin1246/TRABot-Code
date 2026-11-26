"""Thread-safe persistent action queue and physics scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping
from uuid import uuid4

from .engine import PhysicsEngine, placeholder_png
from .motion import MotionClip, MotionValidationError, parse_motion


LEGACY_CUBIC_BRIDGE_POLICY = "legacy_cubic"
MINIMUM_JERK_BRIDGE_POLICY = "minimum_jerk_v1"
BRIDGE_POLICIES = frozenset(
    {LEGACY_CUBIC_BRIDGE_POLICY, MINIMUM_JERK_BRIDGE_POLICY}
)
MINIMUM_JERK_PEAK_SPEED_RAD_S = 1.0
MINIMUM_JERK_DURATION_FPS = 25.0
MINIMUM_JERK_MIN_FRAME_INTERVALS = 2
MINIMUM_JERK_MAX_DURATION_S = 120.0


def minimum_jerk_bridge_duration(maximum_delta: float) -> float:
    """Return the fixed-speed bridge duration rounded to a 25 Hz tick."""

    delta = float(maximum_delta)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("maximum_delta must be finite and non-negative")
    required_duration = 1.875 * delta / MINIMUM_JERK_PEAK_SPEED_RAD_S
    if (
        not math.isfinite(required_duration)
        or required_duration > MINIMUM_JERK_MAX_DURATION_S
    ):
        raise ValueError(
            "minimum-jerk bridge duration must be finite and no more than "
            f"{MINIMUM_JERK_MAX_DURATION_S:g} seconds"
        )
    ticks = max(
        MINIMUM_JERK_MIN_FRAME_INTERVALS,
        int(math.ceil(required_duration * MINIMUM_JERK_DURATION_FPS - 1e-12)),
    )
    return ticks / MINIMUM_JERK_DURATION_FPS


def _minimum_jerk_position(value: float) -> float:
    normalized = min(max(float(value), 0.0), 1.0)
    return (
        10.0 * normalized**3
        - 15.0 * normalized**4
        + 6.0 * normalized**5
    )


@dataclass(frozen=True)
class ServiceConfig:
    queue_capacity: int = 16
    render_fps: float = 15.0
    default_blend_duration: float = 0.25
    max_motion_duration: float = 10.0
    min_speed: float = 0.25
    max_speed: float = 2.0
    loop_stall_timeout: float = 1.0
    action_overrun_grace: float = 1.0
    metrics_window: float = 3.0
    action_history_capacity: int = 256


@dataclass
class QueuedAction:
    action_id: str
    clip: MotionClip
    speed: float
    blend_duration: float
    bridge_policy: str
    scheduled_wall: float
    scheduled_monotonic: float
    exclusive_owner: str | None = None
    payload_sha256: str | None = None
    started_sim: float | None = None
    started_wall: float | None = None
    started_monotonic: float | None = None
    blend_origin: dict[str, float] | None = None
    bridge_duration: float | None = None
    completion_pending: bool = False


TERMINAL_ACTION_STATUSES = frozenset(
    {"completed", "canceled", "replaced", "failed"}
)


class MotionConflictError(MotionValidationError):
    """Atomic queue/ownership preconditions changed before submission."""


class MotionRegistry:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root).expanduser().resolve() if root else None

    def load(self, reference: str) -> MotionClip:
        if self.root is None:
            raise MotionValidationError("named actions need --motion-root")
        relative = Path(reference)
        if relative.is_absolute():
            candidate = relative.resolve()
        else:
            candidate = (self.root / relative).resolve()
            if candidate.suffix != ".json" and not candidate.exists():
                candidate = candidate.with_suffix(".json")
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise MotionValidationError("motion path escapes the configured motion root") from exc
        if not candidate.is_file():
            matches = list(self.root.rglob(f"{reference}.json")) if "/" not in reference else []
            if len(matches) == 1:
                candidate = matches[0].resolve()
            elif len(matches) > 1:
                raise MotionValidationError(f"ambiguous action name: {reference}")
            else:
                raise MotionValidationError(f"action not found: {reference}")
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MotionValidationError(f"cannot load action {reference}: {exc}") from exc
        return parse_motion(payload, name=candidate.stem, source=str(candidate))


class MotionService:
    def __init__(
        self,
        engine: PhysicsEngine,
        *,
        config: ServiceConfig | None = None,
        motion_root: str | Path | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or ServiceConfig()
        self.registry = MotionRegistry(motion_root)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._physics_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._queue: deque[QueuedAction] = deque()
        self._current: QueuedAction | None = None
        self._desired: dict[str, float] = {}
        self._snapshot: dict[str, Any] = {
            "sim_time": 0.0,
            "q": {},
            "dq": {},
            "target": {},
        }
        self._frame = placeholder_png()
        self._frame_seq = 0
        self._state_seq = 0
        # Unlike state_seq (which advances at the physics rate), action_seq only
        # changes when the action lifecycle changes.  It is therefore suitable
        # for an HTTP compare-and-submit precondition.
        self._action_seq = 0
        self._started_wall: float | None = None
        self._last_step_wall: float | None = None
        self._last_frame_wall: float | None = None
        self._last_completed: dict[str, Any] | None = None
        # Records are inserted at scheduling time and updated only while
        # holding ``_lock``. This makes the HTTP lifecycle authoritative rather
        # than inferring it from queue membership after the fact.
        self._action_records: dict[str, dict[str, Any]] = {}
        self._fault: str | None = None
        self._render_error: str | None = None
        self._metric_samples: deque[tuple[float, float, int, int]] = deque()
        self._last_metric_sample_wall: float | None = None

    def start(self) -> None:
        with self._lock:
            if self._physics_thread and self._physics_thread.is_alive():
                return
            self._stop.clear()
            self._started_wall = time.monotonic()
            self._last_step_wall = self._started_wall
            self._metric_samples.clear()
            self._last_metric_sample_wall = None
            self._physics_thread = threading.Thread(target=self._physics_loop, name="sim-physics", daemon=True)
            self._render_thread = (
                threading.Thread(target=self._render_loop, name="sim-render", daemon=True)
                if self.config.render_fps > 0.0
                else None
            )
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="sim-watchdog", daemon=True
            )
            self._physics_thread.start()
            if self._render_thread:
                self._render_thread.start()
            self._watchdog_thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for thread in (self._physics_thread, self._render_thread, self._watchdog_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout)
        # The physics loop owns and closes MuJoCo/GL state. If that thread is
        # unexpectedly still alive after the timeout, do not race it here.
        if self._physics_thread is None or not self._physics_thread.is_alive():
            self.engine.close()

    @property
    def running(self) -> bool:
        return bool(self._physics_thread and self._physics_thread.is_alive() and not self._stop.is_set())

    def _normalize_clip(self, clip: MotionClip) -> MotionClip:
        remapped: list[str] = []
        for joint in clip.joints:
            resolved = self.engine.resolve_joint(joint)
            if resolved is None:
                raise MotionValidationError(f"model has no controllable joint named {joint!r}")
            if resolved in remapped:
                raise MotionValidationError(f"multiple motion joints resolve to {resolved!r}")
            remapped.append(resolved)
        oscillations = tuple(
            type(item)(
                self.engine.resolve_joint(item.joint) or item.joint,
                item.center,
                item.amplitude,
                item.frequency_hz,
                item.start,
                item.end,
                item.phase,
                item.ramp,
            )
            for item in clip.oscillations
        )
        return MotionClip(
            clip.name,
            tuple(remapped),
            clip.times,
            clip.values,
            clip.interpolation,
            oscillations,
            clip.source,
        )

    def _new_action_record_locked(
        self,
        action: QueuedAction,
        *,
        scheduled_sim: float,
    ) -> dict[str, Any]:
        record = {
            "id": action.action_id,
            "name": action.clip.name,
            "status": "scheduled",
            "speed": action.speed,
            "blend_duration": action.blend_duration,
            "bridge_policy": action.bridge_policy,
            "bridge_duration": None,
            "duration": action.clip.duration,
            "scheduled_wall": action.scheduled_wall,
            "started_wall": None,
            "completed_wall": None,
            "finished_wall": None,
            "scheduled_sim": scheduled_sim,
            "started_sim": None,
            "completed_sim": None,
            "finished_sim": None,
            "exclusive_owner": action.exclusive_owner,
            "payload_sha256": action.payload_sha256,
        }
        self._action_records[action.action_id] = record
        self._prune_action_records_locked()
        return record

    def _bump_action_seq_locked(self) -> None:
        self._action_seq += 1

    def _exclusive_owner_locked(self) -> str | None:
        owners = {
            action.exclusive_owner
            for action in ([self._current] if self._current is not None else [])
            + list(self._queue)
            if action.exclusive_owner is not None
        }
        if len(owners) > 1:
            raise RuntimeError("multiple exclusive motion owners are active")
        return next(iter(owners), None)

    def _prune_action_records_locked(self) -> None:
        limit = max(
            int(self.config.action_history_capacity),
            int(self.config.queue_capacity) + 1,
            1,
        )
        while len(self._action_records) > limit:
            removable_id = next(
                (
                    action_id
                    for action_id, record in self._action_records.items()
                    if record.get("status") in TERMINAL_ACTION_STATUSES
                ),
                None,
            )
            if removable_id is None:
                return
            self._action_records.pop(removable_id, None)

    def _mark_action_started_locked(
        self,
        action: QueuedAction,
        *,
        sim_time: float,
    ) -> None:
        started_wall = time.time()
        action.started_sim = sim_time
        action.started_wall = started_wall
        action.started_monotonic = time.monotonic()
        record = self._action_records[action.action_id]
        record.update(
            {
                "status": "started",
                "started_wall": started_wall,
                "started_sim": sim_time,
                "bridge_duration": action.bridge_duration,
            }
        )
        self._bump_action_seq_locked()

    def _finish_action_locked(
        self,
        action: QueuedAction,
        *,
        status: str,
        sim_time: float,
        reason: str | None = None,
        finished_wall: float | None = None,
    ) -> dict[str, Any]:
        if status not in TERMINAL_ACTION_STATUSES:
            raise ValueError(f"invalid terminal action status: {status}")
        timestamp = time.time() if finished_wall is None else finished_wall
        record = self._action_records[action.action_id]
        record.update(
            {
                "status": status,
                "finished_wall": timestamp,
                "finished_sim": sim_time,
            }
        )
        if status == "completed":
            record["completed_wall"] = timestamp
            record["completed_sim"] = sim_time
        if reason:
            record["reason"] = reason
        terminal = dict(record)
        # Retain the historical key for the untouched legacy bridge. It now
        # contains the same richer record available through the ID endpoint.
        self._last_completed = terminal
        self._prune_action_records_locked()
        self._bump_action_seq_locked()
        return terminal

    def _finish_pending_locked(
        self,
        *,
        status: str,
        reason: str,
        sim_time: float,
    ) -> None:
        timestamp = time.time()
        # Finish queued actions first so the prior physically-running action
        # remains ``last_completed``, matching the old compatibility behavior.
        queued = list(self._queue)
        current = self._current
        self._queue.clear()
        self._current = None
        for action in queued:
            self._finish_action_locked(
                action,
                status=status,
                reason=reason,
                sim_time=sim_time,
                finished_wall=timestamp,
            )
        if current is not None:
            self._finish_action_locked(
                current,
                status=status,
                reason=reason,
                sim_time=sim_time,
                finished_wall=timestamp,
            )

    def submit(
        self,
        motion: MotionClip | Mapping[str, Any] | str,
        *,
        mode: str = "enqueue",
        speed: float = 1.0,
        blend_duration: float | None = None,
        bridge_policy: str | None = None,
        action_id: str | None = None,
        require_empty: bool = False,
        expected_action_seq: int | None = None,
        exclusive_owner: str | None = None,
        payload_sha256: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"enqueue", "replace"}:
            raise MotionValidationError("mode must be enqueue or replace")
        speed = float(speed)
        if not math.isfinite(speed) or not self.config.min_speed <= speed <= self.config.max_speed:
            raise MotionValidationError(
                f"speed must be between {self.config.min_speed} and {self.config.max_speed}"
            )
        blend = self.config.default_blend_duration if blend_duration is None else float(blend_duration)
        if not math.isfinite(blend) or blend < 0.0 or blend > 2.0:
            raise MotionValidationError("blend_duration must be between 0 and 2 seconds")
        policy = (
            LEGACY_CUBIC_BRIDGE_POLICY
            if bridge_policy is None
            else bridge_policy
        )
        if not isinstance(policy, str) or policy not in BRIDGE_POLICIES:
            raise MotionValidationError(
                "bridge_policy must be legacy_cubic or minimum_jerk_v1"
            )
        if policy == MINIMUM_JERK_BRIDGE_POLICY and blend != 0.0:
            raise MotionValidationError(
                "bridge_policy=minimum_jerk_v1 requires blend_duration=0"
            )
        if isinstance(motion, MotionClip):
            clip = motion
        elif isinstance(motion, str):
            clip = self.registry.load(motion)
        else:
            clip = parse_motion(motion)
        if clip.duration <= 0.0 or clip.duration / speed > self.config.max_motion_duration:
            raise MotionValidationError(
                f"scaled motion duration must be in (0, {self.config.max_motion_duration}] seconds"
            )
        clip = self._normalize_clip(clip)
        resolved_action_id = action_id or uuid4().hex
        if not isinstance(require_empty, bool):
            raise MotionValidationError("require_empty must be a boolean")
        if expected_action_seq is not None and (
            isinstance(expected_action_seq, bool)
            or not isinstance(expected_action_seq, int)
            or expected_action_seq < 0
        ):
            raise MotionValidationError("expected_action_seq must be a non-negative integer")
        if exclusive_owner is not None and (
            not isinstance(exclusive_owner, str)
            or not exclusive_owner
            or len(exclusive_owner) > 128
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for character in exclusive_owner)
        ):
            raise MotionValidationError("exclusive_owner contains unsupported characters")
        if payload_sha256 is not None and (
            not isinstance(payload_sha256, str)
            or len(payload_sha256) != 64
            or any(character not in "0123456789abcdef" for character in payload_sha256)
        ):
            raise MotionValidationError("payload_sha256 must be a lowercase SHA-256 digest")
        with self._lock:
            if expected_action_seq is not None and expected_action_seq != self._action_seq:
                raise MotionConflictError(
                    f"expected_action_seq={expected_action_seq} does not match current action_seq={self._action_seq}"
                )
            active_owner = self._exclusive_owner_locked()
            if active_owner is not None and active_owner != exclusive_owner:
                raise MotionConflictError(
                    f"simulator is reserved by exclusive owner {active_owner!r}"
                )
            if require_empty and (self._current is not None or self._queue):
                raise MotionConflictError("require_empty precondition failed")
            if resolved_action_id in self._action_records:
                raise MotionValidationError(
                    f"action id already exists: {resolved_action_id}"
                )
            measured_q = self._snapshot.get("q", {})
            if (
                policy == MINIMUM_JERK_BRIDGE_POLICY
                and isinstance(measured_q, Mapping)
                and all(joint in measured_q for joint in clip.joints)
            ):
                # This is an early rejection only. Queued actions still derive
                # their authoritative origin and duration from the actual pose
                # when they start.
                self._minimum_jerk_duration_for_clip(clip, measured_q)
            if mode == "replace":
                self._finish_pending_locked(
                    status="replaced",
                    reason="replaced_by_new_action",
                    sim_time=float(self._snapshot.get("sim_time", 0.0)),
                )
            if len(self._queue) >= self.config.queue_capacity:
                raise MotionValidationError("action queue is full")
            scheduled_wall = time.time()
            queued = QueuedAction(
                resolved_action_id,
                clip,
                speed,
                blend,
                policy,
                scheduled_wall,
                time.monotonic(),
                exclusive_owner,
                payload_sha256,
            )
            self._queue.append(queued)
            self._bump_action_seq_locked()
            record = self._new_action_record_locked(
                queued,
                scheduled_sim=float(self._snapshot.get("sim_time", 0.0)),
            )
            depth = len(self._queue) + (1 if self._current else 0)
            scheduled = dict(record)
        return {
            "accepted": True,
            "id": queued.action_id,
            "name": clip.name,
            "queue_depth": depth,
            "status": "scheduled",
            "scheduled_wall": queued.scheduled_wall,
            "action": scheduled,
            "action_seq": self._action_seq,
        }

    def cancel(
        self,
        *,
        reason: str = "client_cancel",
        exclusive_owner: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            active_owner = self._exclusive_owner_locked()
            if active_owner is not None and active_owner != exclusive_owner:
                raise MotionConflictError(
                    f"simulator is reserved by exclusive owner {active_owner!r}"
                )
            return self._cancel_pending_locked(reason=reason)

    def _cancel_pending_locked(self, *, reason: str) -> dict[str, Any]:
        """Terminate pending work while already holding ``_lock``.

        External callers must pass through :meth:`cancel` and its owner gate.
        Internal safety paths (notably the watchdog) use this helper so an
        exclusive owner cannot prevent deadline enforcement or kill the
        watchdog thread with a conflict exception.
        """

        canceled = len(self._queue) + (1 if self._current else 0)
        current_id = self._current.action_id if self._current else None
        self._finish_pending_locked(
            status="canceled",
            reason=reason,
            sim_time=float(self._snapshot.get("sim_time", 0.0)),
        )
        # Holding the measured pose avoids jumping to an old target.
        self._desired = dict(self._snapshot.get("q", {}))
        return {"canceled": canceled, "current_id": current_id, "mode": "hold"}

    def heartbeat(self) -> dict[str, Any]:
        return {"ok": self.running and self._fault is None, "monotonic": time.monotonic()}

    def health(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            step_age = None if self._last_step_wall is None else now - self._last_step_wall
            frame_age = None if self._last_frame_wall is None else now - self._last_frame_wall
            sim_real_time_factor, step_hz, frame_fps = self._rates_locked(now)
            render_required = self.config.render_fps > 0.0
            render_ready = (
                not render_required
                or (
                    self._render_thread is not None
                    and self._render_thread.is_alive()
                    and self._render_error is None
                    and self._frame_seq > 0
                )
            )
            return {
                "ok": self.running and self._fault is None and render_ready,
                "running": self.running,
                "engine": type(self.engine).__name__,
                "fault": self._fault,
                "render_error": self._render_error,
                "step_age_s": step_age,
                "frame_age_s": frame_age,
                "uptime_s": None if self._started_wall is None else now - self._started_wall,
                "queue_depth": len(self._queue) + (1 if self._current else 0),
                "state_seq": self._state_seq,
                "action_seq": self._action_seq,
                "frame_seq": self._frame_seq,
                "sim_real_time_factor": sim_real_time_factor,
                "step_hz": step_hz,
                "expected_step_hz": 1.0 / self.engine.timestep,
                "frame_fps": frame_fps,
                "render_running": bool(self._render_thread and self._render_thread.is_alive()),
            }

    def state(self) -> dict[str, Any]:
        with self._lock:
            action = None
            if self._current:
                action = dict(self._action_records[self._current.action_id])
            return {
                "seq": self._state_seq,
                "action_seq": self._action_seq,
                "sim_time": self._snapshot.get("sim_time", 0.0),
                "q": dict(self._snapshot.get("q", {})),
                "dq": dict(self._snapshot.get("dq", {})),
                "target": dict(self._snapshot.get("target", {})),
                "action": action,
                "queued": [
                    dict(self._action_records[queued.action_id])
                    for queued in self._queue
                ],
                "last_completed": dict(self._last_completed) if self._last_completed else None,
                "fault": self._fault,
            }

    def action_state(self) -> dict[str, Any]:
        state = self.state()
        return {
            "action_seq": state["action_seq"],
            "current": state["action"],
            "queued": state["queued"],
            "last_completed": state["last_completed"],
        }

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        """Return one immutable lifecycle snapshot by its accepted ID."""

        with self._lock:
            record = self._action_records.get(str(action_id))
            return dict(record) if record is not None else None

    def frame(self) -> tuple[bytes, int]:
        with self._lock:
            return self._frame, self._frame_seq

    def _record_metrics_locked(self, now: float, sim_time: float) -> None:
        # Sampling at 10 Hz keeps the 200/500 Hz physics loop bookkeeping tiny.
        if self._last_metric_sample_wall is not None and now - self._last_metric_sample_wall < 0.1:
            return
        self._last_metric_sample_wall = now
        self._metric_samples.append((now, sim_time, self._state_seq, self._frame_seq))
        cutoff = now - max(self.config.metrics_window, 0.5)
        while len(self._metric_samples) > 2 and self._metric_samples[1][0] < cutoff:
            self._metric_samples.popleft()

    def _rates_locked(self, now: float) -> tuple[float | None, float | None, float | None]:
        if not self._metric_samples:
            return None, None, None
        first = self._metric_samples[0]
        last_recorded = self._metric_samples[-1]
        # Include current wall time so a stalled loop makes the reported rate decay.
        last = (
            now,
            float(self._snapshot.get("sim_time", 0.0)),
            self._state_seq,
            self._frame_seq,
        )
        if now - last_recorded[0] < 0.02 and len(self._metric_samples) > 1:
            last = last_recorded
        elapsed = last[0] - first[0]
        if elapsed <= 1e-6:
            return None, None, None
        rtf = (last[1] - first[1]) / elapsed
        step_hz = (last[2] - first[2]) / elapsed
        frame_fps = (last[3] - first[3]) / elapsed
        return round(rtf, 4), round(step_hz, 2), round(frame_fps, 2)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _minimum_jerk_duration_for_clip(
        self,
        clip: MotionClip,
        measured_q: Mapping[str, Any],
    ) -> float:
        first = clip.sample(0.0)
        deltas: list[float] = []
        for joint in clip.joints:
            if joint not in measured_q:
                raise MotionValidationError(
                    f"measured pose is missing controlled joint {joint!r}"
                )
            try:
                measured = float(measured_q[joint])
            except (TypeError, ValueError, OverflowError) as exc:
                raise MotionValidationError(
                    f"measured pose for controlled joint {joint!r} must be numeric"
                ) from exc
            if not math.isfinite(measured):
                raise MotionValidationError(
                    f"measured pose for controlled joint {joint!r} must be finite"
                )
            delta = abs(float(first[joint]) - measured)
            if not math.isfinite(delta):
                raise MotionValidationError(
                    f"minimum-jerk displacement for controlled joint {joint!r} "
                    "must be finite"
                )
            deltas.append(delta)
        try:
            duration = minimum_jerk_bridge_duration(max(deltas))
        except ValueError as exc:
            raise MotionValidationError(str(exc)) from exc
        if duration > self.config.max_motion_duration:
            raise MotionValidationError(
                "minimum-jerk bridge duration must be no more than "
                f"{self.config.max_motion_duration:g} seconds"
            )
        return duration

    def _start_next_locked(self, sim_time: float) -> None:
        while self._current is None and self._queue:
            current = self._queue.popleft()
            current.blend_origin = dict(self._snapshot.get("q", {}))
            try:
                if current.bridge_policy == MINIMUM_JERK_BRIDGE_POLICY:
                    current.bridge_duration = self._minimum_jerk_duration_for_clip(
                        current.clip,
                        current.blend_origin,
                    )
                else:
                    current.bridge_duration = current.blend_duration
            except MotionValidationError as exc:
                self._finish_action_locked(
                    current,
                    status="failed",
                    reason=f"MotionValidationError: {exc}",
                    sim_time=sim_time,
                )
                continue
            self._current = current
            self._mark_action_started_locked(current, sim_time=sim_time)

    def _targets_locked(self, sim_time: float) -> dict[str, float]:
        self._start_next_locked(sim_time)
        if self._current is None:
            if not self._desired:
                self._desired = dict(self._snapshot.get("q", {}))
            return dict(self._desired)

        action = self._current
        assert (
            action.started_sim is not None
            and action.blend_origin is not None
            and action.bridge_duration is not None
        )
        elapsed = max(0.0, sim_time - action.started_sim)
        first = action.clip.sample(0.0)
        if action.bridge_duration > 0.0 and elapsed < action.bridge_duration:
            if action.bridge_policy == MINIMUM_JERK_BRIDGE_POLICY:
                alpha = _minimum_jerk_position(elapsed / action.bridge_duration)
            else:
                alpha = self._smoothstep(elapsed / action.bridge_duration)
            sample = {
                joint: action.blend_origin.get(joint, first[joint])
                + (first[joint] - action.blend_origin.get(joint, first[joint])) * alpha
                for joint in action.clip.joints
            }
        else:
            motion_time = max(0.0, elapsed - action.bridge_duration) * action.speed
            sample = action.clip.sample(motion_time)
            if motion_time >= action.clip.duration:
                # The final target is applied by the physics step immediately
                # after this method returns. Publish ``completed`` only after
                # that step, never while the final command is merely pending.
                action.completion_pending = True
        self._desired.update(sample)
        return dict(self._desired)

    def _finish_completed_step_locked(self, sim_time: float) -> None:
        action = self._current
        if action is None or not action.completion_pending:
            return
        self._finish_action_locked(
            action,
            status="completed",
            sim_time=sim_time,
        )
        self._current = None

    def _action_deadline_seconds(self, action: QueuedAction) -> float:
        bridge_duration = (
            action.bridge_duration
            if action.bridge_duration is not None
            else action.blend_duration
        )
        return (
            bridge_duration
            + action.clip.duration / action.speed
            + self.config.action_overrun_grace
        )

    def _physics_loop(self) -> None:
        dt = self.engine.timestep
        if not math.isfinite(dt) or dt <= 0.0:
            with self._lock:
                self._fault = "invalid_engine_timestep"
            return
        next_tick = time.monotonic()
        try:
            initial = self.engine.snapshot()
            with self._lock:
                self._snapshot = initial
                self._desired = dict(initial.get("q", {}))
                self._record_metrics_locked(time.monotonic(), float(initial.get("sim_time", 0.0)))
            while not self._stop.is_set():
                sim_time = float(self._snapshot.get("sim_time", 0.0))
                with self._lock:
                    targets = self._targets_locked(sim_time)
                if any(not math.isfinite(value) for value in targets.values()):
                    raise RuntimeError("non-finite target generated")
                self.engine.set_targets(targets)
                self.engine.step()
                snapshot = self.engine.snapshot()
                now = time.monotonic()
                with self._lock:
                    self._snapshot = snapshot
                    self._state_seq += 1
                    self._last_step_wall = now
                    self._finish_completed_step_locked(
                        float(snapshot.get("sim_time", 0.0))
                    )
                    self._record_metrics_locked(now, float(snapshot.get("sim_time", 0.0)))

                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    self._stop.wait(delay)
                elif delay < -max(0.05, 10.0 * dt):
                    # Drop accumulated wall-clock lag; never perform a burst of unsafe catch-up steps.
                    next_tick = time.monotonic()
        except Exception as exc:
            with self._lock:
                self._fault = f"{type(exc).__name__}: {exc}"
                self._finish_pending_locked(
                    status="failed",
                    reason=self._fault,
                    sim_time=float(self._snapshot.get("sim_time", 0.0)),
                )
            self._stop.set()
        finally:
            # Physics resources, if any, remain owned by this thread.
            self.engine.close()

    def _render_loop(self) -> None:
        renderer = None
        period = 1.0 / self.config.render_fps
        next_frame = time.monotonic()
        try:
            # Creation and destruction occur on this thread so EGL ownership is valid.
            renderer = self.engine.create_renderer()
            while not self._stop.is_set():
                with self._lock:
                    snapshot = {
                        "sim_time": float(self._snapshot.get("sim_time", 0.0)),
                        "q": dict(self._snapshot.get("q", {})),
                    }
                try:
                    frame = renderer.render_png(snapshot)
                except Exception as exc:
                    with self._lock:
                        self._render_error = f"{type(exc).__name__}: {exc}"
                    # Avoid a hot failure loop while leaving physics alive.
                    self._stop.wait(min(max(period, 0.1), 1.0))
                else:
                    now = time.monotonic()
                    with self._lock:
                        self._frame = frame
                        self._frame_seq += 1
                        self._last_frame_wall = now
                        self._render_error = None
                next_frame += period
                delay = next_frame - time.monotonic()
                if delay > 0.0:
                    self._stop.wait(delay)
                elif delay < -period:
                    # Never issue catch-up renders after an expensive frame.
                    next_frame = time.monotonic()
        except Exception as exc:
            with self._lock:
                self._render_error = f"{type(exc).__name__}: {exc}"
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as exc:
                    with self._lock:
                        self._render_error = f"close {type(exc).__name__}: {exc}"

    def _watchdog_loop(self) -> None:
        period = min(max(self.config.loop_stall_timeout / 4.0, 0.02), 0.25)
        while not self._stop.wait(period):
            now = time.monotonic()
            with self._lock:
                if self._last_step_wall and now - self._last_step_wall > self.config.loop_stall_timeout:
                    self._fault = "physics_loop_stalled"
                    self._finish_pending_locked(
                        status="failed",
                        reason=self._fault,
                        sim_time=float(self._snapshot.get("sim_time", 0.0)),
                    )
                    self._stop.set()
                    return
                if self._current and self._current.started_monotonic is not None:
                    expected = self._action_deadline_seconds(self._current)
                    if now - self._current.started_monotonic > expected:
                        self._cancel_pending_locked(
                            reason="action_deadline_exceeded"
                        )
