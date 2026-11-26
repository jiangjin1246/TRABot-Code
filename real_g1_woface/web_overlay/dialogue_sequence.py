"""Per-turn random Approved-Atom execution for the dialogue MVP.

This module intentionally knows nothing about Gemini semantics.  A turn starts
on the first model output activity, executes one already-validated exact
Catalog plan, and is invalidated as soon as the upstream turn ends.  Every Atom
is submitted as one canonical inline q-trajectory; the persistent MuJoCo
controller creates its minimum-jerk bridge from measured q when that action
actually starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
from typing import Any


BRIDGE_POLICY = "minimum_jerk_v1"
NEUTRAL_ACTION = "joint_release_arm_pose"
TERMINAL_ACTION_STATUSES = frozenset(
    {"completed", "canceled", "replaced", "failed"}
)


class DialogueSequenceError(RuntimeError):
    """The simulator rejected or failed one action in a dialogue sequence."""


def _json_copy(value: Any) -> Any:
    """Thaw a recursively immutable Catalog payload without mutating its source."""

    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_copy(item) for item in value]
    return value


def _exact_atom_record(atom: Any) -> dict[str, Any]:
    return {
        "atom_id": str(atom.atom_id),
        "version": str(atom.version),
        "revision": int(atom.revision),
        "content_sha256": str(atom.content_sha256),
    }


def atom_to_inline_motion(atom: Any) -> dict[str, Any]:
    """Build a new service motion from the exact Catalog Atom control payload.

    The returned object contains q-trajectory data only.  It never references
    preview PNGs and never mutates the recursively frozen Catalog payload.
    """

    payload = atom.atom_payload
    if not isinstance(payload, Mapping):
        raise DialogueSequenceError(f"Catalog Atom {atom.atom_id} payload is invalid")
    control = payload.get("control")
    if not isinstance(control, Mapping):
        raise DialogueSequenceError(f"Catalog Atom {atom.atom_id} has no control object")
    joints = control.get("joints")
    trajectory = control.get("trajectory")
    if (
        not isinstance(joints, Sequence)
        or isinstance(joints, (str, bytes, bytearray))
        or not joints
    ):
        raise DialogueSequenceError(f"Catalog Atom {atom.atom_id} has invalid joints")
    if (
        not isinstance(trajectory, Sequence)
        or isinstance(trajectory, (str, bytes, bytearray))
        or not trajectory
    ):
        raise DialogueSequenceError(
            f"Catalog Atom {atom.atom_id} has no canonical trajectory"
        )

    identity = _exact_atom_record(atom)
    return {
        "name": (
            f"{identity['atom_id']}@{identity['version']}:"
            f"r{identity['revision']}:{identity['content_sha256']}"
        ),
        "joints": [str(joint) for joint in joints],
        "frames": _json_copy(trajectory),
        # The Atom samples are already the desired q path.  The action-start
        # bridge is quintic; interpolating these samples linearly avoids adding
        # a second hidden smoothing policy to the Atom itself.
        "interpolation": "linear",
        "catalog_identity": identity,
    }


@dataclass(frozen=True, slots=True)
class SequenceSettings:
    count: int = 3
    base_seed: int = 42
    log_path: Path | str = Path("runtime/dialogue_atom_turns.jsonl")
    bridge_policy: str = BRIDGE_POLICY
    speed: float = 1.0
    blend_duration: float = 0.0
    neutral_action: str = NEUTRAL_ACTION
    poll_interval: float = 0.05
    action_timeout: float = 45.0

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count != 3:
            raise ValueError("the frozen dialogue MVP requires count=3")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer")
        if self.bridge_policy != BRIDGE_POLICY:
            raise ValueError(f"bridge_policy must be {BRIDGE_POLICY!r}")
        if not math.isfinite(float(self.speed)) or float(self.speed) <= 0.0:
            raise ValueError("speed must be positive")
        if not math.isfinite(float(self.blend_duration)) or float(self.blend_duration) != 0.0:
            raise ValueError("dialogue Atom blend_duration must be zero")
        if not math.isfinite(float(self.poll_interval)) or float(self.poll_interval) < 0.0:
            raise ValueError("poll_interval cannot be negative")
        if not math.isfinite(float(self.action_timeout)) or float(self.action_timeout) <= 0.0:
            raise ValueError("action_timeout must be positive")
        if not isinstance(self.neutral_action, str) or not self.neutral_action:
            raise ValueError("neutral_action must be a non-empty string")
        object.__setattr__(self, "log_path", Path(self.log_path).expanduser())


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_id: str
    source_turn_id: Any
    strategy: str
    seed: int
    atoms: tuple[dict[str, Any], ...]
    action_ids: tuple[str, ...]
    action_records: tuple[dict[str, Any], ...]
    bridge_durations: tuple[float | None, ...]
    playback_elapsed_seconds: float
    wall_elapsed_seconds: float
    status: str
    neutral_action_id: str = ""
    error: str = ""


@dataclass(slots=True)
class TurnHandle:
    source_turn_id: Any
    turn_index: int
    plan: Any
    started_monotonic: float
    task: asyncio.Task[TurnResult] | None = field(default=None, repr=False)
    closed: bool = False
    action_ids: list[str] = field(default_factory=list)
    records: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    sequence_started_monotonic: float | None = None
    playback_elapsed_seconds: float = 0.0
    failure: str = ""
    all_submitted_event: asyncio.Event = field(
        default_factory=asyncio.Event, repr=False
    )

    @property
    def turn_id(self) -> str:
        return str(self.plan.turn_id)

    @property
    def seed(self) -> int:
        return int(self.plan.seed)

    async def wait_atoms_complete(self) -> TurnResult:
        if self.task is None:
            raise RuntimeError("turn task has not been installed")
        return await asyncio.shield(self.task)

    async def wait_all_submitted(self) -> tuple[str, ...]:
        """Wait until Atom 1 started and all three exact actions were accepted."""

        await self.all_submitted_event.wait()
        return tuple(self.action_ids)


class _JsonlTurnLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, payload: Mapping[str, Any]) -> None:
        record = dict(payload)
        record.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        )
        body = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(body)
                stream.write("\n")
                stream.flush()


EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


class DialogueSequenceCoordinator:
    """Run exact planned Atoms and own cancel/neutral for one dialogue bridge."""

    def __init__(
        self,
        *,
        client: Any,
        atom_catalog: Any,
        planner: Any,
        settings: SequenceSettings,
        event_sink: EventSink | None = None,
    ) -> None:
        self.client = client
        self.atom_catalog = atom_catalog
        self.planner = planner
        self.settings = settings
        self.event_sink = event_sink
        self._control_lock = asyncio.Lock()
        self._active: TurnHandle | None = None
        self._turns: dict[Any, TurnHandle] = {}
        self._turn_index = 0
        self._finish_tasks: dict[Any, asyncio.Task[TurnResult]] = {}
        self._cleanup_tail: asyncio.Task[TurnResult] | None = None
        self._log = _JsonlTurnLog(Path(settings.log_path))
        self.last_log_error = ""

    def _append_final_log(self, payload: Mapping[str, Any]) -> None:
        """Final logging is observable but may never suppress safety cleanup."""

        try:
            self._log.append(payload)
        except Exception as exc:
            self.last_log_error = str(exc)

    def start_turn(self, *, source_turn_id: Any) -> TurnHandle:
        """Plan once and schedule a turn from a synchronous model-output hook."""

        asyncio.get_running_loop()
        existing = self._turns.get(source_turn_id)
        if existing is not None and not existing.closed:
            return existing

        if self._active is not None and not self._active.closed:
            self.request_finish(
                source_turn_id=self._active.source_turn_id, reason="superseded"
            )

        turn_index = self._turn_index
        self._turn_index += 1
        turn_id = f"turn_{turn_index + 1:03d}"
        seed = self.settings.base_seed + turn_index
        plan = self.planner.plan(
            turn_id=turn_id,
            atom_catalog=self.atom_catalog,
            count=self.settings.count,
            seed=seed,
        )
        atoms = tuple(plan.atoms)
        if len(atoms) != self.settings.count:
            raise DialogueSequenceError(
                f"Planner returned {len(atoms)} Atoms; expected {self.settings.count}"
            )
        exact_ids = [
            (atom.atom_id, atom.version, atom.revision, atom.content_sha256)
            for atom in atoms
        ]
        if len(set(exact_ids)) != len(exact_ids):
            raise DialogueSequenceError("Planner repeated an exact Atom in one turn")

        handle = TurnHandle(
            source_turn_id=source_turn_id,
            turn_index=turn_index,
            plan=plan,
            started_monotonic=time.monotonic(),
        )
        barrier = self._cleanup_tail
        plan_record = dict(plan.to_log_record())
        # Audit durability is part of admission.  Persist the exact plan before
        # publishing a live handle or creating any task that could POST it.
        # A disk error therefore has zero Atom side effects and is handled by
        # the caller's stop/neutral recovery path.
        self._log.append(
            {
                "event": "turn_planned",
                **plan_record,
                "source_turn_id": source_turn_id,
                "bridge_policy": self.settings.bridge_policy,
                "blend_duration": self.settings.blend_duration,
            }
        )
        self._active = handle
        self._turns[source_turn_id] = handle
        try:
            handle.task = asyncio.create_task(
                self._run_turn(handle, barrier), name=f"dialogue-atoms-{turn_id}"
            )
        except Exception:
            self._turns.pop(source_turn_id, None)
            if self._active is handle:
                self._active = None
            raise
        handle.task.add_done_callback(
            lambda task, current=handle: self._handle_runner_done(current, task)
        )
        return handle

    def _handle_runner_done(
        self, handle: TurnHandle, task: asyncio.Task[TurnResult]
    ) -> None:
        if handle.closed or task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            handle.failure = str(error)
            self.request_finish(source_turn_id=handle.source_turn_id, reason="failed")

    def request_finish(
        self, *, source_turn_id: Any, reason: str
    ) -> asyncio.Task[TurnResult]:
        """Synchronously invalidate a generation, then cancel and neutralize it."""

        existing_task = self._finish_tasks.get(source_turn_id)
        if existing_task is not None:
            return existing_task
        handle = self._turns.get(source_turn_id)
        if handle is None:
            async def no_active_turn() -> TurnResult:
                return TurnResult(
                    turn_id="",
                    source_turn_id=source_turn_id,
                    strategy="random_v0",
                    seed=0,
                    atoms=(),
                    action_ids=(),
                    action_records=(),
                    bridge_durations=(),
                    playback_elapsed_seconds=0.0,
                    wall_elapsed_seconds=0.0,
                    status="no_active_turn",
                )

            return asyncio.create_task(no_active_turn())

        # This happens before any await.  A runner that has not entered its POST
        # lock can no longer submit; a runner already inside waits for its
        # shielded worker thread before releasing that lock to cancel().
        handle.closed = True
        if self._active is handle:
            self._active = None
        if handle.task is not None and not handle.task.done():
            handle.task.cancel()
        finish_task = asyncio.create_task(
            self._finish_turn_impl(handle, str(reason)),
            name=f"dialogue-finish-{handle.turn_id}",
        )
        self._finish_tasks[source_turn_id] = finish_task
        self._cleanup_tail = finish_task
        return finish_task

    async def finish_turn(self, *, source_turn_id: Any, reason: str) -> TurnResult:
        # Shield the actual cleanup: the upstream reset method may cancel its own
        # bookkeeping task, but cancel+neutral must still reach the controller.
        task = self.request_finish(source_turn_id=source_turn_id, reason=reason)
        return await asyncio.shield(task)

    async def report_start_failure(
        self, *, source_turn_id: Any, error: str
    ) -> None:
        if self.event_sink is None:
            return
        payload = {
            "kind": "catalog_atom_motion",
            "matched": True,
            "simulated": True,
            "executed": False,
            "scheduled": False,
            "running": False,
            "completed": False,
            "lifecycle": "failed",
            "action": "random_catalog_sequence",
            "label": "random Catalog sequence",
            "actionRequestId": "",
            "sourceTurnId": source_turn_id,
            "bridgePolicy": self.settings.bridge_policy,
            "error": str(error),
        }
        try:
            returned = self.event_sink(payload)
            if returned is not None:
                await returned
        except Exception:
            return

    def request_recovery(
        self, *, source_turn_id: Any, error: str
    ) -> asyncio.Task[TurnResult]:
        """Stop and neutralize after planning/startup fails before a handle exists."""

        existing = self._finish_tasks.get(source_turn_id)
        if existing is not None:
            return existing
        consumed_index = max(self._turn_index - 1, 0)
        task = asyncio.create_task(
            self._recover_without_plan(
                source_turn_id=source_turn_id,
                turn_index=consumed_index,
                error=str(error),
            ),
            name=f"dialogue-recovery-{source_turn_id}",
        )
        self._finish_tasks[source_turn_id] = task
        self._cleanup_tail = task
        return task

    async def _recover_without_plan(
        self, *, source_turn_id: Any, turn_index: int, error: str
    ) -> TurnResult:
        neutral_action_id = ""
        recovery_error = ""
        started = time.monotonic()
        try:
            async with self._control_lock:
                await self._await_thread_call(self.client.stop_action_sync)
                neutral_result = await self._await_thread_call(
                    self.client.trigger_sync,
                    self.settings.neutral_action,
                    mode="replace",
                    speed=self.settings.speed,
                    blend_duration=self.settings.blend_duration,
                    bridge_policy=self.settings.bridge_policy,
                )
            if not isinstance(neutral_result, Mapping):
                raise DialogueSequenceError("neutral response must be an object")
            raw_id = (
                neutral_result.get("id")
                or neutral_result.get("action_id")
                or neutral_result.get("request_id")
            )
            if raw_id is None:
                raise DialogueSequenceError("neutral action has no id")
            neutral_action_id = str(raw_id)
            await self._wait_neutral(neutral_action_id)
        except Exception as exc:
            recovery_error = str(exc)

        result = TurnResult(
            turn_id=f"turn_{turn_index + 1:03d}",
            source_turn_id=source_turn_id,
            strategy="random_v0",
            seed=self.settings.base_seed + turn_index,
            atoms=(),
            action_ids=(),
            action_records=(),
            bridge_durations=(),
            playback_elapsed_seconds=0.0,
            wall_elapsed_seconds=max(0.0, time.monotonic() - started),
            status="failed",
            neutral_action_id=neutral_action_id,
            error=error if not recovery_error else f"{error}; recovery: {recovery_error}",
        )
        self._append_final_log(
            {
                "event": "turn_finished",
                "turn_id": result.turn_id,
                "source_turn_id": result.source_turn_id,
                "strategy": result.strategy,
                "seed": result.seed,
                "atoms": [],
                "action_ids": [],
                "actions": [],
                "bridge_policy": self.settings.bridge_policy,
                "bridge_durations": [],
                "playback_elapsed_seconds": 0.0,
                "wall_elapsed_seconds": result.wall_elapsed_seconds,
                "status": result.status,
                "neutral_action_id": neutral_action_id,
                "error": result.error,
            }
        )
        return result

    def _is_live(self, handle: TurnHandle) -> bool:
        return self._active is handle and not handle.closed

    def _assert_live(self, handle: TurnHandle) -> None:
        if not self._is_live(handle):
            raise asyncio.CancelledError

    async def _await_thread_call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Cancelling asyncio.to_thread does not stop its OS thread.  Waiting
            # here is what prevents a POST from arriving after controller cancel.
            try:
                await asyncio.shield(worker)
            except Exception:
                pass
            raise

    def _register_submission(
        self, handle: TurnHandle, result: Mapping[str, Any]
    ) -> str:
        if not bool(result.get("accepted", True)):
            raise DialogueSequenceError(
                str(result.get("error") or "simulator rejected sequence action")
            )
        raw_id = result.get("id") or result.get("action_id") or result.get("request_id")
        if raw_id is None:
            raise DialogueSequenceError("simulator accepted an action without an id")
        action_id = str(raw_id)
        if action_id not in handle.action_ids:
            handle.action_ids.append(action_id)
        raw_record = result.get("action")
        if isinstance(raw_record, Mapping):
            handle.records[action_id] = dict(raw_record)
        return action_id

    async def _submit_atom(self, handle: TurnHandle, atom: Any) -> tuple[str, dict[str, Any]]:
        motion = atom_to_inline_motion(atom)
        result: Mapping[str, Any] | None = None
        async with self._control_lock:
            self._assert_live(handle)
            worker = asyncio.create_task(
                asyncio.to_thread(
                    self.client.trigger_sync,
                    motion,
                    mode="enqueue",
                    speed=self.settings.speed,
                    blend_duration=self.settings.blend_duration,
                    bridge_policy=self.settings.bridge_policy,
                )
            )
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                try:
                    result = await asyncio.shield(worker)
                except Exception:
                    result = None
                if isinstance(result, Mapping):
                    self._register_submission(handle, result)
                raise
        if not isinstance(result, Mapping):
            raise DialogueSequenceError("simulator action response must be an object")
        action_id = self._register_submission(handle, result)
        await self._emit_lifecycle(handle, atom, action_id, "scheduled", result)
        return action_id, dict(result)

    async def _emit_lifecycle(
        self,
        handle: TurnHandle,
        atom: Any,
        action_id: str,
        lifecycle: str,
        record: Mapping[str, Any],
    ) -> None:
        if self.event_sink is None:
            return
        payload = {
            "kind": "catalog_atom_motion",
            "matched": True,
            "simulated": True,
            "executed": lifecycle == "completed",
            "scheduled": lifecycle == "scheduled",
            "running": lifecycle == "started",
            "completed": lifecycle == "completed",
            "lifecycle": lifecycle,
            "action": str(atom.atom_id),
            "label": str(atom.atom_id),
            "actionRequestId": action_id,
            "turnId": handle.turn_id,
            "exactAtom": _exact_atom_record(atom),
            "bridgePolicy": self.settings.bridge_policy,
        }
        if record.get("bridge_duration") is not None:
            payload["bridgeDuration"] = record.get("bridge_duration")
        try:
            returned = self.event_sink(payload)
            if returned is not None:
                await returned
        except Exception:
            # Browser status is observational; it must never stop controller
            # cleanup or change the action plan.
            return

    async def _observe_action(
        self,
        handle: TurnHandle,
        atom: Any,
        action_id: str,
        started_event: asyncio.Event,
        queue_ready_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.settings.action_timeout
        started_emitted = False
        while asyncio.get_running_loop().time() < deadline:
            record = await asyncio.to_thread(self.client.action_sync, action_id)
            if not isinstance(record, Mapping):
                raise DialogueSequenceError(f"action {action_id} status is not an object")
            current = dict(record)
            handle.records[action_id] = current
            status = str(current.get("status", ""))
            if status in {"started", "completed"} and not started_emitted:
                started_emitted = True
                started_event.set()
                await self._emit_lifecycle(
                    handle, atom, action_id, "started", current
                )
                # For Atom 1, let the runner fill the two free queue slots before
                # this observer performs its next status request.  This is an
                # event hand-off, not a time gap, and makes the capacity-two
                # enqueue contract deterministic even for extremely short clips.
                if status == "started" and queue_ready_event is not None:
                    await queue_ready_event.wait()
            if status in TERMINAL_ACTION_STATUSES:
                if status != "completed":
                    if not started_event.is_set():
                        started_event.set()
                    await self._emit_lifecycle(
                        handle, atom, action_id, status, current
                    )
                    raise DialogueSequenceError(
                        f"simulator action {action_id} ended with status {status}"
                    )
                await self._emit_lifecycle(
                    handle, atom, action_id, "completed", current
                )
                return current
            await asyncio.sleep(self.settings.poll_interval)
        if not started_event.is_set():
            started_event.set()
        raise DialogueSequenceError(
            f"timed out after {self.settings.action_timeout:.1f}s waiting for {action_id}"
        )

    async def _wait_for_first_started(
        self,
        observation: asyncio.Task[dict[str, Any]],
        started_event: asyncio.Event,
    ) -> None:
        waiter = asyncio.create_task(started_event.wait())
        try:
            done, _ = await asyncio.wait(
                {waiter, observation}, return_when=asyncio.FIRST_COMPLETED
            )
            if observation in done and not started_event.is_set():
                await observation
            await waiter
        finally:
            if not waiter.done():
                waiter.cancel()

    def _atoms_result(self, handle: TurnHandle, *, status: str) -> TurnResult:
        exact_atoms = tuple(_exact_atom_record(atom) for atom in handle.plan.atoms)
        audit_fields = (
            "id",
            "name",
            "status",
            "bridge_policy",
            "bridge_duration",
            "duration",
            "scheduled_wall",
            "started_wall",
            "completed_wall",
            "finished_wall",
            "scheduled_sim",
            "started_sim",
            "completed_sim",
            "finished_sim",
            "reason",
        )
        action_records = tuple(
            {
                key: record.get(key)
                for key in audit_fields
                if key in record
            }
            for action_id in handle.action_ids
            for record in (handle.records.get(action_id, {"id": action_id}),)
        )
        bridge_durations = tuple(
            handle.records.get(action_id, {}).get("bridge_duration")
            for action_id in handle.action_ids
        )
        return TurnResult(
            turn_id=handle.turn_id,
            source_turn_id=handle.source_turn_id,
            strategy=str(handle.plan.strategy),
            seed=handle.seed,
            atoms=exact_atoms,
            action_ids=tuple(handle.action_ids),
            action_records=action_records,
            bridge_durations=bridge_durations,
            playback_elapsed_seconds=handle.playback_elapsed_seconds,
            wall_elapsed_seconds=max(0.0, time.monotonic() - handle.started_monotonic),
            status=status,
            error=handle.failure,
        )

    async def _run_turn(
        self,
        handle: TurnHandle,
        cleanup_barrier: asyncio.Task[TurnResult] | None,
    ) -> TurnResult:
        observations: list[asyncio.Task[dict[str, Any]]] = []
        first_queue_ready = asyncio.Event()
        try:
            if cleanup_barrier is not None and not cleanup_barrier.done():
                await asyncio.shield(cleanup_barrier)
            self._assert_live(handle)
            atoms = tuple(handle.plan.atoms)
            handle.sequence_started_monotonic = time.monotonic()

            first_id, _ = await self._submit_atom(handle, atoms[0])
            first_started = asyncio.Event()
            first_observation = asyncio.create_task(
                self._observe_action(
                    handle,
                    atoms[0],
                    first_id,
                    first_started,
                    first_queue_ready,
                ),
                name=f"observe-{first_id}",
            )
            observations.append(first_observation)
            await self._wait_for_first_started(first_observation, first_started)
            self._assert_live(handle)

            # Queue capacity is two.  Once Atom 1 has actually started, enqueue
            # exactly Atoms 2 and 3 with no artificial inter-action wait.
            for atom in atoms[1:]:
                action_id, _ = await self._submit_atom(handle, atom)
                started = asyncio.Event()
                observations.append(
                    asyncio.create_task(
                        self._observe_action(handle, atom, action_id, started),
                        name=f"observe-{action_id}",
                    )
                )

            first_queue_ready.set()
            handle.all_submitted_event.set()

            await asyncio.gather(*observations)
            if handle.sequence_started_monotonic is not None:
                handle.playback_elapsed_seconds = max(
                    0.0, time.monotonic() - handle.sequence_started_monotonic
                )
            return self._atoms_result(handle, status="atoms_completed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            handle.failure = str(exc)
            raise
        finally:
            if handle.sequence_started_monotonic is not None:
                handle.playback_elapsed_seconds = max(
                    0.0, time.monotonic() - handle.sequence_started_monotonic
                )
            first_queue_ready.set()
            for task in observations:
                if not task.done():
                    task.cancel()
            if observations:
                await asyncio.gather(*observations, return_exceptions=True)

    async def _wait_neutral(self, action_id: str) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.settings.action_timeout
        while asyncio.get_running_loop().time() < deadline:
            record = await asyncio.to_thread(self.client.action_sync, action_id)
            if not isinstance(record, Mapping):
                raise DialogueSequenceError(
                    f"neutral action {action_id} status is not an object"
                )
            status = str(record.get("status", ""))
            if status in TERMINAL_ACTION_STATUSES:
                if status != "completed":
                    raise DialogueSequenceError(
                        f"neutral action {action_id} ended with status {status}"
                    )
                return dict(record)
            await asyncio.sleep(self.settings.poll_interval)
        raise DialogueSequenceError(
            f"timed out after {self.settings.action_timeout:.1f}s waiting for neutral"
        )

    async def _finish_turn_impl(self, handle: TurnHandle, reason: str) -> TurnResult:
        runner_error = ""
        if handle.task is not None:
            try:
                await handle.task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                runner_error = str(exc)
                handle.failure = handle.failure or runner_error

        neutral_action_id = ""
        neutral_error = ""
        try:
            async with self._control_lock:
                await self._await_thread_call(self.client.stop_action_sync)
                # Capture terminal timestamps/status immediately after cancel,
                # before the neutral action adds another history record.
                for action_id in handle.action_ids:
                    try:
                        record = await self._await_thread_call(
                            self.client.action_sync, action_id
                        )
                    except Exception:
                        continue
                    if isinstance(record, Mapping):
                        handle.records[action_id] = dict(record)
                neutral_result = await self._await_thread_call(
                    self.client.trigger_sync,
                    self.settings.neutral_action,
                    mode="replace",
                    speed=self.settings.speed,
                    blend_duration=self.settings.blend_duration,
                    bridge_policy=self.settings.bridge_policy,
                )
            if not isinstance(neutral_result, Mapping):
                raise DialogueSequenceError("neutral response must be an object")
            raw_id = (
                neutral_result.get("id")
                or neutral_result.get("action_id")
                or neutral_result.get("request_id")
            )
            if raw_id is None:
                raise DialogueSequenceError("neutral action has no id")
            neutral_action_id = str(raw_id)
            await self._wait_neutral(neutral_action_id)
        except Exception as exc:
            neutral_error = str(exc)
            handle.failure = handle.failure or neutral_error

        result = self._atoms_result(handle, status=reason)
        result = TurnResult(
            turn_id=result.turn_id,
            source_turn_id=result.source_turn_id,
            strategy=result.strategy,
            seed=result.seed,
            atoms=result.atoms,
            action_ids=result.action_ids,
            action_records=result.action_records,
            bridge_durations=result.bridge_durations,
            playback_elapsed_seconds=result.playback_elapsed_seconds,
            wall_elapsed_seconds=result.wall_elapsed_seconds,
            status=result.status,
            neutral_action_id=neutral_action_id,
            error=result.error,
        )
        self._append_final_log(
            {
                "event": "turn_finished",
                "turn_id": result.turn_id,
                "source_turn_id": result.source_turn_id,
                "strategy": result.strategy,
                "seed": result.seed,
                "atoms": list(result.atoms),
                "action_ids": list(result.action_ids),
                "actions": list(result.action_records),
                "bridge_policy": self.settings.bridge_policy,
                "bridge_durations": list(result.bridge_durations),
                "playback_elapsed_seconds": result.playback_elapsed_seconds,
                "wall_elapsed_seconds": result.wall_elapsed_seconds,
                "status": result.status,
                "neutral_action_id": neutral_action_id,
                "runner_error": runner_error,
                "neutral_error": neutral_error,
                "error": result.error,
            }
        )
        return result


CoordinatorFactory = Callable[[Any], DialogueSequenceCoordinator]


def patch_random_dialogue_bridge_class(
    cls: type[Any], *, coordinator_factory: CoordinatorFactory
) -> bool:
    """Bind random execution to an untouched upstream Gemini bridge class."""

    if getattr(cls, "_mujoco_random_sequence_patched", False):
        return False
    required = (
        "_mark_robot_output_activity",
        "_handle_robot_input_transcription",
        "_handle_robot_output_transcription",
        "_cleanup_robot_turn_after_playback",
        "_reset_robot_turn_state",
    )
    if any(not hasattr(cls, name) for name in required):
        return False

    original_mark = cls._mark_robot_output_activity
    original_reset = cls._reset_robot_turn_state

    def get_coordinator(bridge: Any) -> DialogueSequenceCoordinator:
        coordinator = getattr(bridge, "_mujoco_dialogue_sequence", None)
        if coordinator is None:
            coordinator = coordinator_factory(bridge)
            bridge._mujoco_dialogue_sequence = coordinator
        return coordinator

    def mark_and_start(self: Any) -> Any:
        result = original_mark(self)
        if bool(getattr(self, "robot_turn_complete_received", False)):
            return result
        turn_serial = getattr(self, "robot_turn_serial", 0)
        if getattr(self, "_mujoco_sequence_started_serial", None) == turn_serial:
            return result
        coordinator = get_coordinator(self)
        self._mujoco_sequence_started_serial = turn_serial
        try:
            coordinator.start_turn(source_turn_id=turn_serial)
        except Exception as exc:
            error_text = str(exc)
            self.robot_turn_mode = "random_catalog_failed"
            self.robot_presentation_started = False
            # Register cancel/neutral synchronously before yielding to a possibly
            # slow browser event sink.  turn_complete and a reconnect therefore
            # see this exact cleanup task as the next-turn barrier.
            recovery = coordinator.request_recovery(
                source_turn_id=turn_serial, error=error_text
            )

            async def report_and_recover() -> None:
                await coordinator.report_start_failure(
                    source_turn_id=turn_serial, error=error_text
                )
                await asyncio.shield(recovery)

            self._mujoco_sequence_start_failure_task = asyncio.create_task(
                report_and_recover(),
                name=f"dialogue-start-failure-{turn_serial}",
            )
            return result
        self.robot_turn_mode = "random_catalog"
        self.robot_presentation_started = True
        return result

    async def ignore_legacy_input(self: Any, text: str) -> None:
        return None

    async def ignore_legacy_output(self: Any, text: str) -> None:
        return None

    async def finish_random_turn(self: Any, turn_serial: int) -> None:
        coordinator = getattr(self, "_mujoco_dialogue_sequence", None)
        if coordinator is not None:
            await coordinator.finish_turn(
                source_turn_id=turn_serial, reason="turn_complete"
            )
        if turn_serial == getattr(self, "robot_turn_serial", turn_serial):
            # Call the captured method, not the interruption wrapper below.
            original_reset(self)
            self._mujoco_sequence_started_serial = None

    def reset_and_cancel(self: Any) -> Any:
        turn_serial = getattr(self, "robot_turn_serial", 0)
        coordinator = getattr(self, "_mujoco_dialogue_sequence", None)
        if (
            coordinator is not None
            and getattr(self, "_mujoco_sequence_started_serial", None) == turn_serial
        ):
            coordinator.request_finish(
                source_turn_id=turn_serial, reason="interrupted"
            )
        result = original_reset(self)
        self._mujoco_sequence_started_serial = None
        return result

    cls._mark_robot_output_activity = mark_and_start
    cls._handle_robot_input_transcription = ignore_legacy_input
    cls._handle_robot_output_transcription = ignore_legacy_output
    cls._cleanup_robot_turn_after_playback = finish_random_turn
    cls._reset_robot_turn_state = reset_and_cancel
    cls._mujoco_random_sequence_patched = True
    cls._mujoco_random_sequence_original_mark = original_mark
    cls._mujoco_random_sequence_original_reset = original_reset
    return True


__all__ = [
    "BRIDGE_POLICY",
    "DialogueSequenceCoordinator",
    "DialogueSequenceError",
    "NEUTRAL_ACTION",
    "SequenceSettings",
    "TurnHandle",
    "TurnResult",
    "atom_to_inline_motion",
    "patch_random_dialogue_bridge_class",
]
