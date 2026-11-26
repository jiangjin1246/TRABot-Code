"""Per-Gemini-connection ownership for semantic gesture turns.

This module is the narrow adapter between the patched Live bridge and the
deterministic runtime.  It owns turn identity, tool-call idempotency, and the
first-output timestamp.  Atom choice, budget packing, service admission, and
recovery remain in :mod:`web_overlay.semantic_runtime`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Protocol
import weakref

from .reply_clock import MAX_REPLY_DURATION_US, ReplyClock
from .semantic_catalog import SemanticCatalog, SemanticCatalogRow
from .semantic_motion import (
    inline_motion,
    minimum_jerk_bridge_us,
    recovery_cost_callback,
    transition_cost_callback,
)
from .semantic_planner import PlannerHistory, replan_suffix, update_history
from .semantic_runtime import (
    RUNTIME_FUNCTIONS,
    SUPPORTED_BRIDGE_POLICY,
    SemanticActionStep,
    SemanticRecoveryStep,
    SemanticSuffixExecutor,
    SemanticToolPayload,
    validate_gemini_tool_payload,
)
from .semantic_session import (
    SemanticSessionNeutralGate,
    SemanticSessionNeutralResult,
)


SEMANTIC_REPLY_BUDGET_MULTIPLIER = 3
SEMANTIC_REPLY_BUDGET_POLICY = "gemini_estimate_x3_v1"


class SemanticControllerError(RuntimeError):
    """A turn/tool transition violated the fail-closed controller contract."""


def _effective_reply_budget_us(estimated_duration_us: int) -> int:
    if (
        isinstance(estimated_duration_us, bool)
        or not isinstance(estimated_duration_us, int)
        or estimated_duration_us < 1
    ):
        raise SemanticControllerError(
            "estimated spoken duration must be a positive integer"
        )
    if estimated_duration_us > (
        MAX_REPLY_DURATION_US // SEMANTIC_REPLY_BUDGET_MULTIPLIER
    ):
        raise SemanticControllerError(
            "scaled estimated spoken duration exceeds the reply clock range"
        )
    return estimated_duration_us * SEMANTIC_REPLY_BUDGET_MULTIPLIER


class SemanticExecution(Protocol):
    """Minimal lifecycle exposed by the deterministic runtime adapter."""

    async def run(self) -> Any: ...

    def on_audio_chunk(
        self, *, audio_bytes: bytes, mime_type: str, observed_ns: int
    ) -> Any: ...

    def on_turn_complete(self, *, observed_ns: int) -> Any: ...

    def interrupt(self, *, reason: str, observed_ns: int) -> Any: ...


ValidationFunction = Callable[..., Any]
ExecutionFactory = Callable[..., SemanticExecution]
AuditSink = Callable[[Mapping[str, Any]], Any]


@dataclass(slots=True)
class _TurnState:
    turn_id: int
    output_started_ns: int | None = None
    call_id: str = ""
    payload_sha256: str = ""
    request_id: str = ""
    revision: int = 0
    request: Any = None
    response: dict[str, Any] | None = None
    execution: SemanticExecution | None = None
    task: asyncio.Task[Any] | None = None
    turn_complete: bool = False
    interrupted: bool = False
    invalidated: bool = False
    terminal_reason: str = ""
    runtime_error_code: str = ""


def _payload_digest(payload: Mapping[str, Any]) -> str:
    try:
        body = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SemanticControllerError("tool payload is not canonical finite JSON") from exc
    return hashlib.sha256(body).hexdigest()


def _maybe_schedule(value: Any, *, name: str) -> asyncio.Task[Any] | None:
    if not hasattr(value, "__await__"):
        return None

    async def consume() -> None:
        try:
            await value
        except asyncio.CancelledError:
            raise
        except Exception:
            # Gesture cleanup is auxiliary and must never escape into Live audio.
            return

    return asyncio.create_task(consume(), name=name)


class SemanticDialogueController:
    """Turn-safe Live bridge adapter with one immutable plan per reply.

    The output clock is frozen on the first transcription/audio activity even
    when the Gemini tool call arrives later.  A late call for the *same* active
    turn can therefore execute only against its already-consumed budget.  Calls
    and activity for reset/interrupted generations are rejected or ignored.
    """

    def __init__(
        self,
        *,
        bridge: Any,
        validate_payload: ValidationFunction,
        execution_factory: ExecutionFactory,
        execute: bool,
        force_first_policy_enabled: bool = False,
        session_preparer: Any | None = None,
        simulated: bool = True,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        audit: AuditSink | None = None,
    ) -> None:
        try:
            self._bridge_ref = weakref.ref(bridge)
        except TypeError:
            # Some small test/upstream proxy objects do not expose weakrefs.
            self._bridge_ref = lambda: bridge
        self._validate_payload = validate_payload
        self._execution_factory = execution_factory
        self._execute = bool(execute)
        self._force_first_policy_enabled = bool(force_first_policy_enabled)
        self._session_preparer = session_preparer
        self._simulated = bool(simulated)
        self._monotonic_ns = monotonic_ns
        self._audit = audit
        self._instance_id = secrets.token_hex(12)
        self._revision = 0
        self._turns: dict[int, _TurnState] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._session_start_lock: asyncio.Lock | None = None
        self._session_start_result: SemanticSessionNeutralResult | None = None
        self._session_prepare_task: asyncio.Task[
            SemanticSessionNeutralResult
        ] | None = None

    async def _emit_session_neutral_event(
        self,
        *,
        status: str,
        result: SemanticSessionNeutralResult,
    ) -> None:
        bridge = self._bridge_ref()
        emit = getattr(bridge, "_emit_robot_action_event", None) if bridge else None
        if not callable(emit):
            return
        payload = {
            "matched": True,
            "executed": status == "ready" and result.action_id is not None,
            "simulated": self._simulated,
            "kind": "semantic_session_neutral",
            "status": status,
            "ready": result.ready,
            "reasonCode": result.reason_code,
            "actionId": result.action_id,
            "stableSamples": result.stable_samples,
            "maximumQDriftRad": result.maximum_q_drift_rad,
            "maximumAbsDqRadS": result.maximum_abs_dq_rad_s,
        }
        try:
            await emit(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            # UI delivery is observability only; the cached gate outcome remains
            # authoritative and a failed send can never open a blocked session.
            return

    async def on_session_start(self) -> Mapping[str, Any]:
        """Run the per-connection neutral gate exactly once before Live receive."""

        if self._session_start_result is not None:
            return self._session_start_result.as_dict()
        if self._session_start_lock is None:
            self._session_start_lock = asyncio.Lock()
        async with self._session_start_lock:
            if self._session_start_result is not None:
                return self._session_start_result.as_dict()
            if self._session_preparer is None:
                result = SemanticSessionNeutralResult(
                    True,
                    "session_neutral_disabled",
                )
                self._session_start_result = result
                return result.as_dict()
            task = self._session_prepare_task
            if task is None or task.cancelled():
                preparing = SemanticSessionNeutralResult(
                    False,
                    "session_neutral_preparing",
                )
                await self._emit_session_neutral_event(
                    status="preparing",
                    result=preparing,
                )

                async def prepare_once() -> SemanticSessionNeutralResult:
                    try:
                        candidate = await asyncio.to_thread(
                            self._session_preparer.prepare
                        )
                        if not isinstance(candidate, SemanticSessionNeutralResult):
                            raise SemanticControllerError(
                                "session neutral preparer returned an invalid result"
                            )
                        prepared = candidate
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        prepared = SemanticSessionNeutralResult(
                            False,
                            "session_neutral_internal_error",
                        )
                    # The shared task, not any individual waiter, owns the
                    # terminal cache and UI. A canceled browser start cannot
                    # cause a second neutral POST while this worker continues.
                    self._session_start_result = prepared
                    await self._emit_session_neutral_event(
                        status="ready" if prepared.ready else "blocked",
                        result=prepared,
                    )
                    return prepared

                task = asyncio.create_task(
                    prepare_once(),
                    name=f"semantic-session-neutral-{self._instance_id}",
                )
                self._session_prepare_task = task
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        # Shield the controller-owned task so canceling one start_session
        # waiter cannot cancel the non-cancelable synchronous simulator POST.
        result = await asyncio.shield(task)
        return result.as_dict()

    def _bridge_turn_id(self) -> int:
        bridge = self._bridge_ref()
        if bridge is None:
            return -1
        try:
            return int(getattr(bridge, "robot_turn_serial"))
        except (TypeError, ValueError, AttributeError):
            return -1

    def _is_current(self, turn_id: int) -> bool:
        state = self._turns.get(turn_id)
        return (
            self._bridge_turn_id() == turn_id
            and (state is None or not state.invalidated)
        )

    def _state_for_activity(self, turn_id: int) -> _TurnState | None:
        if not self._is_current(turn_id):
            return None
        state = self._turns.get(turn_id)
        if state is None:
            state = _TurnState(turn_id=turn_id)
            self._turns[turn_id] = state
            self._trim_states()
        if state.interrupted or state.turn_complete:
            return None
        return state

    def _trim_states(self) -> None:
        if len(self._turns) <= 16:
            return
        for turn_id in sorted(self._turns)[:-16]:
            state = self._turns[turn_id]
            if state.task is None or state.task.done():
                del self._turns[turn_id]

    def _freeze_output_start(
        self, state: _TurnState, observed_ns: int | None = None
    ) -> None:
        value = int(self._monotonic_ns()) if observed_ns is None else observed_ns
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SemanticControllerError("output observation clock is invalid")
        if state.output_started_ns is None or value < state.output_started_ns:
            state.output_started_ns = value

    async def accept_tool_plan(
        self,
        *,
        turn_id: int,
        call_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(turn_id, int) or isinstance(turn_id, bool):
            raise SemanticControllerError("turn id is invalid")
        if not isinstance(call_id, str) or not call_id:
            raise SemanticControllerError("tool call id is missing")
        if not isinstance(payload, Mapping):
            raise SemanticControllerError("tool payload must be an object")
        digest = _payload_digest(payload)
        if not self._is_current(turn_id):
            raise SemanticControllerError("tool call belongs to a stale turn")
        session_start = await self.on_session_start()
        if session_start.get("ready") is not True:
            raise SemanticControllerError("semantic session is not ready")

        state = self._turns.get(turn_id)
        if state is None:
            state = _TurnState(turn_id=turn_id)
            self._turns[turn_id] = state
            self._trim_states()
        if state.invalidated or state.interrupted or state.turn_complete:
            raise SemanticControllerError("turn no longer accepts a semantic plan")
        if state.call_id:
            if state.call_id == call_id and state.payload_sha256 == digest:
                assert state.response is not None
                return dict(state.response)
            raise SemanticControllerError("turn already owns a different semantic plan")

        self._revision += 1
        revision = self._revision
        request_id = f"semantic-{self._instance_id}-t{turn_id}-r{revision}"
        request = self._validate_payload(
            payload,
            request_id=request_id,
            revision=revision,
        )
        functions = tuple(getattr(request, "ordered_functions", ()))
        if self._audit is not None:
            self._audit(
                {
                    "schema_version": "semantic_controller_audit_v1",
                    "event": "tool_plan_accepted",
                    "turn_id": turn_id,
                    "call_id_sha256": hashlib.sha256(
                        call_id.encode("utf-8")
                    ).hexdigest(),
                    "payload_sha256": digest,
                    "request_id": request_id,
                    "revision": revision,
                    "ordered_functions": list(functions),
                    "estimated_spoken_duration_us": int(
                        getattr(request, "estimated_spoken_duration_us", 0)
                    ),
                    "output_started_ns": state.output_started_ns,
                    "force_first_policy_enabled": (
                        self._force_first_policy_enabled
                    ),
                }
            )
        state.call_id = call_id
        state.payload_sha256 = digest
        state.request_id = request_id
        state.revision = revision
        state.request = request
        state.response = {
            "accepted": True,
            "function_count": len(functions),
            "execution_mode": "active" if self._execute else "dry_run",
            "force_first_policy_enabled": self._force_first_policy_enabled,
            "request_id": request_id,
            "revision": revision,
        }
        self._start_if_ready(state)
        return dict(state.response)

    def on_output_activity(
        self, *, turn_id: int, observed_ns: int | None = None
    ) -> None:
        state = self._state_for_activity(turn_id)
        if state is None:
            return
        self._freeze_output_start(state, observed_ns)
        self._start_if_ready(state)

    def observe_output_activity(self, *, turn_id: int, observed_ns: int) -> None:
        """Synchronously preserve arrival order without starting runtime work."""

        state = self._state_for_activity(turn_id)
        if state is not None:
            self._freeze_output_start(state, observed_ns)

    def on_audio_chunk(
        self,
        *,
        turn_id: int,
        audio_bytes: bytes,
        mime_type: str,
        observed_ns: int | None = None,
    ) -> None:
        state = self._state_for_activity(turn_id)
        if state is None:
            return
        self._freeze_output_start(state, observed_ns)
        self._start_if_ready(state)
        if state.execution is not None:
            value = state.execution.on_audio_chunk(
                audio_bytes=bytes(audio_bytes),
                mime_type=str(mime_type),
                observed_ns=(
                    int(self._monotonic_ns())
                    if observed_ns is None
                    else observed_ns
                ),
            )
            _maybe_schedule(value, name=f"semantic-runtime-audio-{turn_id}")

    def on_turn_complete(self, *, turn_id: int) -> None:
        state = self._turns.get(turn_id)
        if state is None:
            state = _TurnState(turn_id=turn_id)
            self._turns[turn_id] = state
            self._trim_states()
        if state.invalidated or state.interrupted or state.turn_complete:
            return
        state.turn_complete = True
        state.terminal_reason = "turn_complete"
        if state.execution is not None:
            value = state.execution.on_turn_complete(observed_ns=int(self._monotonic_ns()))
            _maybe_schedule(value, name=f"semantic-runtime-complete-{turn_id}")

    def on_interrupted(self, *, turn_id: int) -> None:
        self._interrupt_state(turn_id, reason="interrupted", invalidate=True)

    def reset_turn(self, *, previous_turn_id: int) -> None:
        self._interrupt_state(previous_turn_id, reason="reset", invalidate=True)

    def _interrupt_state(self, turn_id: int, *, reason: str, invalidate: bool) -> None:
        state = self._turns.get(turn_id)
        if state is None:
            state = _TurnState(turn_id=turn_id)
            self._turns[turn_id] = state
        already_terminal = state.interrupted or state.invalidated
        state.interrupted = True
        state.invalidated = state.invalidated or invalidate
        state.terminal_reason = state.terminal_reason or reason
        interrupt_result: Any = None
        if not already_terminal and state.execution is not None:
            try:
                # Signal execution before any durable audit I/O.  Losing the
                # interrupt-vs-POST boundary would be a safety error; losing an
                # audit record must never undo an already-issued signal.
                interrupt_result = state.execution.interrupt(
                    reason=reason,
                    observed_ns=int(self._monotonic_ns()),
                )
            except Exception:
                interrupt_result = None
        if already_terminal:
            return

        controller_event = {
            "schema_version": "semantic_controller_audit_v1",
            "event": "turn_invalidated",
            "turn_id": turn_id,
            "request_id": state.request_id,
            "revision": state.revision,
            "reason": reason,
            "dropped_functions": list(
                getattr(state.request, "ordered_functions", ())
            ),
        }

        async def persist_interrupt_observability() -> None:
            audit_failed = False
            if hasattr(interrupt_result, "__await__"):
                try:
                    # Production execution performs the runtime's durable
                    # interrupted audit here, after its constant-time signal.
                    await interrupt_result
                except asyncio.CancelledError:
                    raise
                except Exception:
                    audit_failed = True
            if self._audit is not None:
                try:
                    # DurableSemanticAudit fsync is deliberately off the Live
                    # event loop and follows the runtime interrupted record.
                    await asyncio.to_thread(self._audit, controller_event)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    audit_failed = True
            if audit_failed:
                # Never expose sink exception text, and never undo the safety
                # signal.  The stable code makes the failed observability path
                # explicit and keeps this turn fail-closed.
                state.runtime_error_code = "semantic_interrupt_audit_failed"

        if hasattr(interrupt_result, "__await__") or self._audit is not None:
            task = _maybe_schedule(
                persist_interrupt_observability(),
                name=f"semantic-runtime-interrupt-audit-{turn_id}",
            )
            if task is not None:
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    def has_pending(self, *, turn_id: int) -> bool:
        if not self._is_current(turn_id):
            return False
        state = self._turns.get(turn_id)
        return bool(state is not None and state.task is not None and not state.task.done())

    def _start_if_ready(self, state: _TurnState) -> None:
        if (
            state.execution is not None
            or state.request is None
            or state.output_started_ns is None
            or state.invalidated
            or state.interrupted
            or state.turn_complete
            or not self._is_current(state.turn_id)
        ):
            return
        if self._audit is not None:
            self._audit(
                {
                    "schema_version": "semantic_controller_audit_v1",
                    "event": "runtime_scheduled",
                    "turn_id": state.turn_id,
                    "request_id": state.request_id,
                    "revision": state.revision,
                    "output_started_ns": state.output_started_ns,
                    "force_first_policy_enabled": (
                        self._force_first_policy_enabled
                    ),
                }
            )
        execution = self._execution_factory(
            request=state.request,
            turn_id=state.turn_id,
            request_id=state.request_id,
            revision=state.revision,
            output_started_ns=state.output_started_ns,
            execute=self._execute,
        )
        state.execution = execution
        task = asyncio.create_task(
            self._run_execution(state, execution),
            name=f"semantic-runtime-{state.turn_id}-{state.revision}",
        )
        state.task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run_execution(
        self, state: _TurnState, execution: SemanticExecution
    ) -> None:
        try:
            await execution.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            state.runtime_error_code = "semantic_runtime_failed"
            if self._audit is not None:
                try:
                    self._audit(
                        {
                            "schema_version": "semantic_controller_audit_v1",
                            "event": "runtime_failed",
                            "turn_id": state.turn_id,
                            "request_id": state.request_id,
                            "revision": state.revision,
                            "error": state.runtime_error_code,
                        }
                    )
                except Exception:
                    pass


class DurableSemanticAudit:
    """Append one canonical JSON line and fsync before acknowledging it."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def __call__(self, payload: Mapping[str, Any]) -> None:
        record = dict(payload)
        record.setdefault(
            "timestamp_utc",
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        try:
            body = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise SemanticControllerError("audit event is not finite JSON") from exc
        if len(body) > 1_048_576:
            raise SemanticControllerError("audit event exceeds one MiB")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(self.path, flags, 0o600)
            try:
                view = memoryview(body)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short audit write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)


@dataclass(frozen=True, slots=True)
class _NoPlan:
    atoms: tuple[Any, ...] = ()


def _exact_ref(row: SemanticCatalogRow, catalog: SemanticCatalog) -> dict[str, Any]:
    return {
        "entry_id": row.entry_id,
        "atom_id": row.atom_id,
        "version": row.version,
        "revision": row.revision,
        "content_sha256": row.content_sha256,
        "catalog_content_sha256": catalog.content_sha256,
    }


def _row_key(row: SemanticCatalogRow) -> tuple[str, str, int, str]:
    return (row.atom_id, row.version, row.revision, row.content_sha256)


class _SemanticPlannerAdapter:
    """Bind measured-q replans to immutable Catalog rows and real durations."""

    _HISTORY_WINDOW = 24

    def __init__(
        self,
        catalog: SemanticCatalog,
        *,
        tail_guard_us: int,
        allowed_entry_ids: Any | None = None,
    ) -> None:
        self.catalog = catalog
        self.tail_guard_us = tail_guard_us
        if allowed_entry_ids is None:
            self.rows = catalog.rows
            self.by_function = catalog.by_function
            self.allowed_entry_ids: frozenset[str] | None = None
        else:
            if isinstance(allowed_entry_ids, (str, bytes)):
                raise SemanticControllerError(
                    "approved entry ids must be a collection of exact entry ids"
                )
            try:
                normalized_ids = frozenset(allowed_entry_ids)
            except TypeError as exc:
                raise SemanticControllerError(
                    "approved entry ids must be an iterable of strings"
                ) from exc
            if any(
                not isinstance(entry_id, str) or not entry_id
                for entry_id in normalized_ids
            ):
                raise SemanticControllerError(
                    "approved entry ids must contain only non-empty strings"
                )
            catalog_ids = {row.entry_id for row in catalog.rows}
            unknown_ids = normalized_ids - catalog_ids
            if unknown_ids:
                raise SemanticControllerError(
                    "approved entry ids contain entries outside the semantic Catalog"
                )
            self.rows = tuple(
                row for row in catalog.rows if row.entry_id in normalized_ids
            )
            self.by_function = {
                function: tuple(
                    row.entry_id
                    for row in self.rows
                    if row.function == function
                )
                for function in RUNTIME_FUNCTIONS
            }
            missing_functions = tuple(
                function
                for function in RUNTIME_FUNCTIONS
                if not self.by_function[function]
            )
            if missing_functions:
                raise SemanticControllerError(
                    "approved Atom pool does not cover every F17 function: "
                    + ", ".join(missing_functions)
                )
            self.allowed_entry_ids = normalized_ids
        self._persistent_history = PlannerHistory()
        self._history_lock = threading.Lock()
        self._local = threading.local()
        self._recovery_cost = recovery_cost_callback(catalog.neutral_motion)

    def begin(self, interrupt: threading.Event) -> None:
        self._local.interrupt = interrupt
        self._local.selected = []
        self._local.durations = {}

    def finish(self, completed_count: int) -> None:
        selected = tuple(getattr(self._local, "selected", ()))
        if (
            not isinstance(completed_count, int)
            or isinstance(completed_count, bool)
            or completed_count < 0
            or completed_count > len(selected)
        ):
            raise SemanticControllerError(
                "completed count is inconsistent with selected Atom attempts"
            )
        completed = selected[:completed_count]
        if completed:
            with self._history_lock:
                updated = update_history(self._persistent_history, completed)
                self._persistent_history = PlannerHistory(
                    recent_atom_refs=updated.recent_atom_refs[-self._HISTORY_WINDOW :],
                    recent_source_ids=updated.recent_source_ids[-self._HISTORY_WINDOW :],
                    last_unilateral_side=updated.last_unilateral_side,
                    unilateral_run_length=updated.unilateral_run_length,
                    unilateral_balance=updated.unilateral_balance,
                )
        self._local.selected = []
        self._local.durations = {}

    def discard_plan(self, atom: SemanticCatalogRow, reason: str) -> None:
        """Forget exactly the latest selected, never-committed Atom attempt."""

        if not isinstance(atom, SemanticCatalogRow):
            raise SemanticControllerError("discarded plan is not a Catalog Atom")
        if not isinstance(reason, str) or not reason or len(reason) > 128:
            raise SemanticControllerError("discarded plan reason is invalid")
        selected = getattr(self._local, "selected", None)
        durations = getattr(self._local, "durations", None)
        if not isinstance(selected, list) or not isinstance(durations, dict):
            raise SemanticControllerError("planner execution context is missing")
        if not selected or selected[-1] is not atom:
            raise SemanticControllerError(
                "discarded plan is not the exact current uncommitted attempt"
            )
        key = _row_key(atom)
        if key not in durations:
            raise SemanticControllerError("discarded plan duration binding is missing")
        selected.pop()
        durations.pop(key)

    def _planning_context(
        self,
        previous_atom: Any,
        history: tuple[Any, ...],
    ) -> tuple[dict[str, Any], PlannerHistory]:
        if isinstance(previous_atom, Mapping) and isinstance(
            previous_atom.get("q"), Mapping
        ):
            measured_q = dict(previous_atom["q"])
        elif previous_atom is None:
            # Dry-run has no simulator read.  Neutral is the only deterministic
            # non-executable origin and cannot affect live admission.
            measured_q = dict(self.catalog.neutral_q)
        else:
            raise SemanticControllerError("planner did not receive measured q")
        with self._history_lock:
            combined_history = self._persistent_history
        if history:
            combined_history = update_history(combined_history, history)
        return measured_q, combined_history

    def _record_selected(
        self,
        plan: Any,
        measured_q: Mapping[str, Any],
    ) -> None:
        if not plan.atoms:
            return
        row = plan.atoms[0]
        if not isinstance(row, SemanticCatalogRow):
            raise SemanticControllerError("planner returned a non-Catalog Atom")
        bridge_us = minimum_jerk_bridge_us(measured_q, row.entry_q)
        durations = getattr(self._local, "durations", None)
        selected = getattr(self._local, "selected", None)
        if not isinstance(durations, dict) or not isinstance(selected, list):
            raise SemanticControllerError("planner execution context is missing")
        durations[_row_key(row)] = bridge_us + row.duration_us
        selected.append(row)

    def force_first_plan(
        self,
        remaining_functions: tuple[str, ...],
        previous_atom: Any,
        history: tuple[Any, ...],
    ) -> Any:
        """Choose one minimum-cost closed Atom for the first function only."""

        interrupt = getattr(self._local, "interrupt", None)
        if interrupt is not None and interrupt.is_set():
            return _NoPlan()
        if not remaining_functions:
            return _NoPlan()
        first_function = remaining_functions[0]
        if not isinstance(first_function, str) or not first_function:
            raise SemanticControllerError("forced first function is invalid")
        candidate_ids = self.by_function.get(first_function)
        if not candidate_ids:
            return _NoPlan()
        candidates = tuple(
            row for row in self.rows if row.function == first_function
        )
        if not candidates or {row.entry_id for row in candidates} != set(candidate_ids):
            raise SemanticControllerError("forced candidate index drifted from Catalog")
        measured_q, combined_history = self._planning_context(previous_atom, history)
        transition = transition_cost_callback(measured_q)
        minimum_closed_cost_us = min(
            transition(None, row)
            + row.duration_us
            + self._recovery_cost(row)
            + self.tail_guard_us
            for row in candidates
        )
        plan = replan_suffix(
            (first_function,),
            candidates,
            transition_cost_us=transition,
            remaining_budget_us=minimum_closed_cost_us,
            history=combined_history,
            previous_atom=None,
            final_recovery_cost_us=self._recovery_cost,
            tail_guard_us=self.tail_guard_us,
        )
        if len(plan.atoms) != 1:
            raise SemanticControllerError("forced planner did not return exactly one Atom")
        self._record_selected(plan, measured_q)
        return plan

    def __call__(
        self,
        remaining_functions: tuple[str, ...],
        remaining_budget_us: int,
        previous_atom: Any,
        history: tuple[Any, ...],
    ) -> Any:
        interrupt = getattr(self._local, "interrupt", None)
        if interrupt is not None and interrupt.is_set():
            return _NoPlan()
        measured_q, combined_history = self._planning_context(previous_atom, history)
        transition = transition_cost_callback(measured_q)
        plan = replan_suffix(
            remaining_functions,
            self.rows,
            transition_cost_us=transition,
            remaining_budget_us=remaining_budget_us,
            history=combined_history,
            previous_atom=None,
            final_recovery_cost_us=self._recovery_cost,
            tail_guard_us=self.tail_guard_us,
        )
        self._record_selected(plan, measured_q)
        return plan

    def prepare_action(self, row: SemanticCatalogRow) -> SemanticActionStep:
        if not isinstance(row, SemanticCatalogRow):
            raise SemanticControllerError("runtime requested a non-Catalog Atom")
        durations = getattr(self._local, "durations", {})
        estimated_duration_us = durations.get(_row_key(row))
        if not isinstance(estimated_duration_us, int) or estimated_duration_us <= 0:
            raise SemanticControllerError("planned Atom duration binding is missing")
        plan_digest = hashlib.sha256(
            (self.catalog.content_sha256 + ":" + row.entry_id).encode("utf-8")
        ).hexdigest()
        return SemanticActionStep(
            plan_id=f"atom-{plan_digest[:40]}",
            function=row.function,
            atom_ref=_exact_ref(row, self.catalog),
            action_payload={
                "motion": inline_motion(row),
                "speed": 1.0,
                "blend_duration": 0.0,
                "bridge_policy": SUPPORTED_BRIDGE_POLICY,
            },
            estimated_duration_us=estimated_duration_us,
        )

    def prepare_recovery(
        self,
        previous_atom: SemanticCatalogRow,
        remaining_budget_us: int | None,
        history: tuple[Any, ...],
    ) -> SemanticRecoveryStep:
        del remaining_budget_us, history
        if not isinstance(previous_atom, SemanticCatalogRow):
            raise SemanticControllerError("recovery tail is not a Catalog Atom")
        estimated_duration_us = (
            self._recovery_cost(previous_atom) + self.tail_guard_us
        )
        digest = hashlib.sha256(
            (
                self.catalog.content_sha256
                + ":neutral-after:"
                + previous_atom.entry_id
            ).encode("utf-8")
        ).hexdigest()
        return SemanticRecoveryStep(
            recovery_id=f"neutral-{digest[:40]}",
            recovery_ref={
                "target": "catalog_canonical_neutral",
                "catalog_content_sha256": self.catalog.content_sha256,
                "neutral_motion_sha256": self.catalog.neutral_motion_sha256,
                "after": _exact_ref(previous_atom, self.catalog),
                "tail_guard_us": self.tail_guard_us,
            },
            action_payload={
                "motion": inline_motion(self.catalog.neutral_motion),
                "speed": 1.0,
                "blend_duration": 0.0,
                "bridge_policy": SUPPORTED_BRIDGE_POLICY,
            },
            estimated_duration_us=estimated_duration_us,
        )


class _SemanticRuntimeCoordinator:
    def __init__(
        self,
        *,
        catalog: SemanticCatalog,
        client: Any,
        execute: bool,
        force_first: bool,
        audit: AuditSink,
        tail_guard_us: int,
        poll_interval_s: float,
        action_timeout_s: float,
        allowed_entry_ids: Any | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.monotonic_ns = monotonic_ns
        self.force_first = bool(force_first)
        self.client = client
        self.simulated = bool(getattr(client, "simulated", True))
        self.robot_runtime = self._robot_runtime(client)
        self.planner = _SemanticPlannerAdapter(
            catalog,
            tail_guard_us=tail_guard_us,
            allowed_entry_ids=allowed_entry_ids,
        )
        owner = f"semantic-controller-{secrets.token_hex(12)}"
        self.executor = SemanticSuffixExecutor(
            self.planner,
            self.planner.prepare_action,
            prepare_recovery=self.planner.prepare_recovery,
            force_first_plan=(
                self.planner.force_first_plan if force_first else None
            ),
            discard_plan=self.planner.discard_plan,
            client=client,
            dry_run=not execute,
            exclusive_owner=owner if execute else None,
            audit=audit,
            monotonic_ns=monotonic_ns,
            poll_interval_s=poll_interval_s,
            action_timeout_s=action_timeout_s,
        )

    @staticmethod
    def _robot_runtime(client: Any) -> dict[str, Any] | None:
        if bool(getattr(client, "simulated", True)):
            return None
        runtime_method = getattr(client, "robot_runtime", None)
        if callable(runtime_method):
            try:
                runtime = runtime_method()
            except Exception:
                runtime = None
            if isinstance(runtime, Mapping):
                return dict(runtime)
        config = getattr(client, "config", None)
        network = getattr(client, "network_interface", None)
        if not isinstance(network, str) or not network:
            network = getattr(config, "network", "")
        return {
            "enabled": True,
            "simulationMode": False,
            "networkInterface": network if isinstance(network, str) else "",
            "backend": "g1_atom_realtime",
        }

    def execute(
        self,
        request: SemanticToolPayload,
        *,
        output_started_ns: int,
        interrupt: threading.Event,
    ) -> Any:
        self.planner.begin(interrupt)
        result = None
        primary_error: BaseException | None = None
        try:
            begin_request = getattr(
                self.client, "begin_semantic_request", None
            )
            if callable(begin_request):
                begin_request(request)
            if interrupt.is_set():
                return None
            clock = ReplyClock.from_payload(
                request,
                effective_duration_us=_effective_reply_budget_us(
                    request.estimated_spoken_duration_us
                ),
                budget_policy=SEMANTIC_REPLY_BUDGET_POLICY,
                monotonic_ns=self.monotonic_ns,
                started_monotonic_ns=output_started_ns,
            )
            result = self.executor.execute(request, reply_clock=clock)
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                completed_count = (
                    len(result.completed_functions) if result is not None else 0
                )
                self.planner.finish(completed_count)
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                    raise
            finally:
                finish_request = getattr(
                    self.client, "finish_semantic_request", None
                )
                if callable(finish_request):
                    try:
                        finish_request(result, error=primary_error)
                    except BaseException:
                        if primary_error is None:
                            raise

    def signal_interrupt(self, request: SemanticToolPayload, *, reason: str) -> bool:
        return self.executor.signal_interrupt(
            request.request_id,
            request.revision,
            reason=reason,
        )

    def audit_signaled_interrupt(self, request: SemanticToolPayload) -> bool:
        return self.executor.audit_signaled_interrupt(
            request.request_id,
            request.revision,
        )

    def interrupt_active_session_sync(self, *, reason: str) -> Any:
        interrupt_session = getattr(
            self.client, "interrupt_active_session_sync", None
        )
        if not callable(interrupt_session):
            return None
        return interrupt_session(reason)


class _RuntimeTurnExecution:
    def __init__(
        self,
        *,
        coordinator: _SemanticRuntimeCoordinator,
        request: SemanticToolPayload,
        output_started_ns: int,
        bridge: Any,
    ) -> None:
        self.coordinator = coordinator
        self.request = request
        self.output_started_ns = output_started_ns
        self._bridge_ref = weakref.ref(bridge)
        self._interrupt = threading.Event()
        self._result: Any = None

    async def run(self) -> Any:
        self._result = await asyncio.to_thread(
            self.coordinator.execute,
            self.request,
            output_started_ns=self.output_started_ns,
            interrupt=self._interrupt,
        )
        bridge = self._bridge_ref()
        emit = getattr(bridge, "_emit_robot_action_event", None) if bridge else None
        if callable(emit) and self._result is not None:
            try:
                payload = {
                    "matched": True,
                    "executed": bool(self._result.completed_functions),
                    "simulated": self.coordinator.simulated,
                    "kind": "semantic_atom_sequence",
                    "label": "semantic_f17",
                    "requestId": self._result.request_id,
                    "revision": self._result.revision,
                    "status": self._result.status,
                    "plannedFunctions": list(self._result.planned_functions),
                    "completedFunctions": list(self._result.completed_functions),
                    "droppedFunctions": list(self._result.dropped_functions),
                    "actionIds": list(self._result.action_ids),
                    "recoveryActionId": self._result.recovery_action_id,
                    "closedNeutral": self._result.closed_neutral,
                    "forceFirstPolicyEnabled": self.coordinator.force_first,
                    "wouldForceFirstAtom": self._result.would_force_first_atom,
                    "forcedFirstAtom": self._result.forced_first_atom,
                    "plannedOverrunUs": self._result.planned_overrun_us,
                    "playbackSource": (
                        "approved_semantic_atom_catalog"
                        if self.coordinator.simulated
                        else "approved_semantic_atom_catalog_g1"
                    ),
                }
                if not self.coordinator.simulated:
                    payload["robot"] = dict(
                        self.coordinator.robot_runtime or {}
                    )
                await emit(payload)
            except Exception:
                pass
        return self._result

    def on_audio_chunk(
        self, *, audio_bytes: bytes, mime_type: str, observed_ns: int
    ) -> None:
        del audio_bytes, mime_type, observed_ns

    def on_turn_complete(self, *, observed_ns: int) -> None:
        del observed_ns

    def interrupt(self, *, reason: str, observed_ns: int) -> Any:
        del observed_ns
        # The external gate covers the small pre-executor registration window;
        # signal_interrupt linearizes against the runtime's POST boundary once
        # the exact request is active.  Neither operation performs audit I/O.
        self._interrupt.set()
        signaled = self.coordinator.signal_interrupt(self.request, reason=reason)
        interrupt_session = callable(
            getattr(
                self.coordinator.client,
                "interrupt_active_session_sync",
                None,
            )
        )
        if not signaled and not interrupt_session:
            return None

        async def persist_runtime_interrupt() -> bool:
            cleanup_error: BaseException | None = None
            if interrupt_session:
                try:
                    await asyncio.to_thread(
                        self.coordinator.interrupt_active_session_sync,
                        reason=reason,
                    )
                except BaseException as exc:
                    cleanup_error = exc
            audited = False
            if signaled:
                try:
                    audited = await asyncio.to_thread(
                        self.coordinator.audit_signaled_interrupt,
                        self.request,
                    )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None:
                raise cleanup_error
            return audited

        return persist_runtime_interrupt()


def semantic_controller_factory(
    *,
    catalog: SemanticCatalog,
    client: Any,
    config: Any,
) -> Callable[[Any], SemanticDialogueController]:
    """Create per-bridge controllers while validating shared settings once."""

    if tuple(catalog.by_function) != RUNTIME_FUNCTIONS:
        raise SemanticControllerError(
            "Catalog ontology order does not match the runtime F17 enum"
        )
    tail_seconds = float(config.semantic_tail_guard_seconds)
    if not 0.0 <= tail_seconds <= 60.0:
        raise SemanticControllerError("semantic tail guard must be between 0 and 60s")
    tail_guard_us = int(round(tail_seconds * 1_000_000))
    audit = DurableSemanticAudit(config.semantic_sequence_log_path)
    approved_entry_ids = None
    approved_entry_ids_method = getattr(client, "approved_entry_ids", None)
    if callable(approved_entry_ids_method):
        approved_entry_ids = approved_entry_ids_method()
        if approved_entry_ids is None:
            raise SemanticControllerError(
                "approved_entry_ids() must return an iterable of exact entry ids"
            )

    # Validate the optional executable subset at startup, before accepting a
    # Live connection.  Per-bridge planner instances retain independent
    # history, but all use this same exact approved identity set.
    validated_planner = _SemanticPlannerAdapter(
        catalog,
        tail_guard_us=tail_guard_us,
        allowed_entry_ids=approved_entry_ids,
    )
    approved_entry_ids = validated_planner.allowed_entry_ids

    def factory(bridge: Any) -> SemanticDialogueController:
        coordinator = _SemanticRuntimeCoordinator(
            catalog=catalog,
            client=client,
            execute=bool(config.semantic_execute),
            force_first=bool(config.semantic_force_first),
            audit=audit,
            tail_guard_us=tail_guard_us,
            poll_interval_s=float(config.semantic_poll_interval),
            action_timeout_s=float(config.semantic_action_timeout),
            allowed_entry_ids=approved_entry_ids,
        )
        session_preparer = None
        if bool(config.semantic_execute) and bool(
            getattr(config, "semantic_session_neutral_enabled", True)
        ):
            session_preparer = SemanticSessionNeutralGate(
                client=client,
                neutral_motion=catalog.neutral_motion,
                neutral_q=catalog.neutral_q,
                exclusive_owner=f"semantic-session-{secrets.token_hex(12)}",
                poll_interval_s=float(config.semantic_poll_interval),
                timeout_s=float(config.semantic_action_timeout),
            )

        def make_execution(**kwargs: Any) -> _RuntimeTurnExecution:
            request = kwargs["request"]
            if not isinstance(request, SemanticToolPayload):
                raise SemanticControllerError("validated request type drifted")
            return _RuntimeTurnExecution(
                coordinator=coordinator,
                request=request,
                output_started_ns=int(kwargs["output_started_ns"]),
                bridge=bridge,
            )

        return SemanticDialogueController(
            bridge=bridge,
            validate_payload=validate_gemini_tool_payload,
            execution_factory=make_execution,
            execute=bool(config.semantic_execute),
            force_first_policy_enabled=bool(config.semantic_force_first),
            session_preparer=session_preparer,
            simulated=coordinator.simulated,
            audit=audit,
        )

    return factory


__all__ = [
    "DurableSemanticAudit",
    "ExecutionFactory",
    "SEMANTIC_REPLY_BUDGET_MULTIPLIER",
    "SEMANTIC_REPLY_BUDGET_POLICY",
    "SemanticControllerError",
    "SemanticDialogueController",
    "SemanticExecution",
    "semantic_controller_factory",
]
