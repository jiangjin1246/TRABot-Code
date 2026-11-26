"""Gemini Live tool bridge for the deterministic semantic gesture planner.

Gemini is deliberately restricted to an ordered list of semantic functions and
an estimated spoken duration.  This module never lets the model select an Atom,
motion path, execution side, timing anchor, speed, or controller operation.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol
import weakref

from .semantic_runtime import RUNTIME_FUNCTIONS, gemini_tool_parameters_schema


SEMANTIC_TOOL_NAME = "submit_gesture_semantics"
SEMANTIC_FUNCTIONS = RUNTIME_FUNCTIONS
_CONTROL_HANDLER_TASK: contextvars.ContextVar[asyncio.Task[Any] | None] = (
    contextvars.ContextVar("semantic_control_handler_task", default=None)
)


class SemanticDialogueError(RuntimeError):
    """The semantic Live bridge cannot uphold its tool/runtime contract."""


class SemanticTurnController(Protocol):
    """Per-WebSocket controller consumed by the Live bridge patch."""

    async def on_session_start(self) -> Mapping[str, Any]: ...

    async def accept_tool_plan(
        self,
        *,
        turn_id: int,
        call_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def observe_output_activity(self, *, turn_id: int, observed_ns: int) -> None: ...

    def on_output_activity(self, *, turn_id: int, observed_ns: int) -> Any: ...

    def on_audio_chunk(
        self,
        *,
        turn_id: int,
        audio_bytes: bytes,
        mime_type: str,
        observed_ns: int,
    ) -> Any: ...

    def on_turn_complete(self, *, turn_id: int) -> Any: ...

    def on_interrupted(self, *, turn_id: int) -> Any: ...

    def reset_turn(self, *, previous_turn_id: int) -> None: ...

    def has_pending(self, *, turn_id: int) -> bool: ...


ControllerFactory = Callable[[Any], SemanticTurnController]


class _SharedBridgeStartState:
    """Per-StreamSession ownership shared by short-lived bridge instances."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.active_bridge_ref: Callable[[], Any | None] | None = None
        self.failed_bridge: Any | None = None
        self._bridge_refs: list[Callable[[], Any | None]] = []
        self.closing = False
        self.closed = False
        self.ownership_epoch = 0
        self.cleanup_task: asyncio.Task[Any] | None = None
        self.direct_stop_suppressions: dict[
            asyncio.Task[Any],
            tuple[
                Callable[[], Any | None],
                Callable[[], Any | None],
                int,
            ],
        ] = {}

    def active_bridge(self) -> Any | None:
        if self.failed_bridge is not None:
            return self.failed_bridge
        return (
            None
            if self.active_bridge_ref is None
            else self.active_bridge_ref()
        )

    def set_active_bridge(self, bridge: Any | None) -> None:
        current = self.active_bridge()
        if current is bridge and not (
            bridge is None and self.active_bridge_ref is not None
        ):
            return
        self.ownership_epoch += 1
        if bridge is None:
            self.active_bridge_ref = None
            return
        try:
            self.active_bridge_ref = weakref.ref(bridge)
        except TypeError:
            self.active_bridge_ref = lambda: bridge

    def mark_failed_bridge(self, bridge: Any) -> None:
        self.failed_bridge = None
        self.set_active_bridge(bridge)
        self.failed_bridge = bridge

    def clear_failed_bridge(self, bridge: Any | None = None) -> None:
        if bridge is None or self.failed_bridge is bridge:
            self.failed_bridge = None

    @staticmethod
    def _identity_ref(value: Any | None) -> Callable[[], Any | None]:
        if value is None:
            return lambda: None
        try:
            return weakref.ref(value)
        except TypeError:
            return lambda: value

    def register_bridge(self, bridge: Any) -> None:
        live: list[Callable[[], Any | None]] = []
        already_registered = False
        for reference in self._bridge_refs:
            candidate = reference()
            if candidate is None:
                continue
            live.append(reference)
            already_registered = already_registered or candidate is bridge
        if not already_registered:
            live.append(self._identity_ref(bridge))
        self._bridge_refs = live

    def unregister_bridge(self, bridge: Any) -> None:
        self._bridge_refs = [
            reference
            for reference in self._bridge_refs
            if reference() is not None and reference() is not bridge
        ]

    def registered_bridges(self) -> tuple[Any, ...]:
        live_references: list[Callable[[], Any | None]] = []
        bridges: list[Any] = []
        for reference in self._bridge_refs:
            bridge = reference()
            if bridge is None:
                continue
            live_references.append(reference)
            if not any(current is bridge for current in bridges):
                bridges.append(bridge)
        self._bridge_refs = live_references
        return tuple(bridges)

    def clear_registered_bridges(self) -> None:
        self._bridge_refs = []


def load_semantic_prompt(path: Path | str) -> str:
    prompt_path = Path(path).expanduser().resolve()
    try:
        body = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SemanticDialogueError(f"cannot read semantic prompt {prompt_path}: {exc}") from exc
    if not body.strip():
        raise SemanticDialogueError("semantic prompt cannot be empty")
    return body.rstrip()


def semantic_tool(types_module: Any) -> Any:
    """Build the installed google-genai declaration without importing it here."""

    parameters = gemini_tool_parameters_schema()
    declaration = types_module.FunctionDeclaration(
        name=SEMANTIC_TOOL_NAME,
        description=(
            "Declare ordered semantic gesture functions before speaking. The "
            "local deterministic planner chooses and times the actual motions."
        ),
        parameters_json_schema=parameters,
    )
    return types_module.Tool(function_declarations=[declaration])


def patch_live_config_builder(module: Any, *, prompt: str) -> bool:
    """Append the frozen semantic instruction and tool to LiveConnectConfig."""

    if getattr(module, "_semantic_live_config_patched", False):
        return False
    original = getattr(module, "_build_live_config", None)
    types_module = getattr(module, "types", None)
    if original is None or not callable(original) or types_module is None:
        raise SemanticDialogueError("upstream exposes no compatible Live config builder")
    tool = semantic_tool(types_module)

    @functools.wraps(original)
    def build_with_semantics(*args: Any, **kwargs: Any) -> Any:
        call_kwargs = dict(kwargs)
        base_instruction = str(call_kwargs.get("system_instruction") or "").rstrip()
        call_kwargs["system_instruction"] = (
            f"{base_instruction}\n\n{prompt}" if base_instruction else prompt
        )
        config = original(*args, **call_kwargs)
        existing = list(getattr(config, "tools", None) or [])
        if any(
            any(
                getattr(declaration, "name", None) == SEMANTIC_TOOL_NAME
                for declaration in (getattr(item, "function_declarations", None) or [])
            )
            for item in existing
        ):
            raise SemanticDialogueError("semantic tool was already present in Live config")
        # google-genai models are Pydantic v2 objects in the deployed runtime.
        copier = getattr(config, "model_copy", None)
        if callable(copier):
            return copier(update={"tools": [*existing, tool]})
        config.tools = [*existing, tool]
        return config

    module._build_live_config = build_with_semantics
    module._semantic_live_config_patched = True
    module._semantic_live_config_original = original
    return True


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _invoke_controller_hook(
    target: SemanticTurnController,
    method_name: str,
    *,
    logger: Any,
    **kwargs: Any,
) -> Any:
    """Keep auxiliary gesture failures outside the Gemini audio path."""

    try:
        return await _maybe_await(getattr(target, method_name)(**kwargs))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Semantic controller hook {} failed for turn {}: {}",
                method_name,
                kwargs.get("turn_id"),
                type(exc).__name__,
            )
        return None


def _schedule(
    self: Any,
    hook: Callable[[], Any],
    *,
    name: str,
    turn_id: int,
    cancel_on_reset: bool = True,
) -> None:
    async def guarded() -> None:
        if cancel_on_reset and int(getattr(self, "robot_turn_serial", -1)) != turn_id:
            return
        await _maybe_await(hook())

    task = asyncio.create_task(guarded(), name=name)
    task_attribute = (
        "_semantic_activity_tasks" if cancel_on_reset else "_semantic_terminal_tasks"
    )
    tasks = getattr(self, task_attribute, None)
    if tasks is None:
        tasks = set()
        setattr(self, task_attribute, tasks)
    tasks.add(task)

    def discard(done: asyncio.Task[Any]) -> None:
        tasks.discard(done)

    task.add_done_callback(discard)


def _cancel_activity_tasks(self: Any) -> None:
    tasks = getattr(self, "_semantic_activity_tasks", None)
    if not tasks:
        return
    for task in tuple(tasks):
        if not task.done():
            task.cancel()
    tasks.clear()


def _install_runtime_terminal_failure_trace() -> None:
    """Best-effort observation of accepted runtimes that exit before result emit."""
    try:
        from .semantic_controller import _RuntimeTurnExecution
    except Exception:
        return
    cls = _RuntimeTurnExecution
    if cls.__dict__.get("_experiment_terminal_failure_trace_patched", False):
        return
    original = getattr(cls, "run", None)
    if not inspect.iscoroutinefunction(original):
        return

    @functools.wraps(original)
    async def run_with_terminal_trace(self: Any) -> Any:
        try:
            return await original(self)
        except BaseException as exc:
            try:
                bridge_ref = getattr(self, "_bridge_ref", None)
                bridge = bridge_ref() if callable(bridge_ref) else None
                stream = getattr(bridge, "stream_session", None)
                hub = getattr(stream, "_experiment_reply_trace_hub", None)
                request = getattr(self, "request", None)
                request_id = getattr(request, "request_id", None)
                revision = getattr(request, "revision", None)
                if (
                    hub is not None
                    and isinstance(request_id, str)
                    and request_id
                    and isinstance(revision, int)
                    and not isinstance(revision, bool)
                ):
                    hub.observe_semantic_summary_for_semantic(
                        request_id,
                        revision,
                        status=(
                            "cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else "failed"
                        ),
                        execution_summary_complete=False,
                        terminal_summary=True,
                        observed_ns=time.monotonic_ns(),
                    )
            except Exception:
                pass
            raise

    cls.run = run_with_terminal_trace
    cls._experiment_terminal_failure_trace_patched = True
    cls._experiment_terminal_failure_trace_original = original


def patch_semantic_bridge_class(
    cls: type[Any],
    *,
    module: Any,
    controller_factory: ControllerFactory,
) -> bool:
    """Patch only the semantic mode of the upstream Gemini Live bridge."""

    if getattr(cls, "_semantic_dialogue_patched", False):
        return False
    original_start = getattr(cls, "start_session", None)
    original_receive = getattr(cls, "_receive_loop", None)
    original_mark = getattr(cls, "_mark_robot_output_activity", None)
    original_reset = getattr(cls, "_reset_robot_turn_state", None)
    original_pending = getattr(cls, "_robot_has_pending_turn_task", None)
    original_close = getattr(cls, "close", None)
    original_emit_robot_action = getattr(cls, "_emit_robot_action_event", None)
    if not all(
        callable(value)
        for value in (
            original_start,
            original_receive,
            original_mark,
            original_reset,
            original_close,
        )
    ):
        raise SemanticDialogueError("upstream GeminiLiveBridge contract is incompatible")
    safe_send = getattr(module, "_safe_send_json", None)
    types_module = getattr(module, "types", None)
    logger = getattr(module, "logger", None)
    if not callable(safe_send) or types_module is None:
        raise SemanticDialogueError("upstream semantic bridge dependencies are missing")

    def trace_hub(self: Any) -> Any | None:
        stream = getattr(self, "stream_session", None)
        hub = getattr(stream, "_experiment_reply_trace_hub", None)
        return hub if callable(getattr(hub, "ensure_reply", None)) else None

    def ensure_trace_reply(
        self: Any, *, turn_id: int, observed_ns: int, cause: str
    ) -> Any | None:
        try:
            hub = trace_hub(self)
            if hub is None:
                return None
            key = hub.ensure_reply(
                semantic_turn_id=int(turn_id),
                observed_ns=int(observed_ns),
                cause=str(cause)[:64],
            )
            keys = getattr(self, "_experiment_reply_keys", None)
            if not isinstance(keys, dict):
                keys = {}
                self._experiment_reply_keys = keys
            keys[int(turn_id)] = key
            while len(keys) > 8:
                keys.pop(next(iter(keys)))
            return key
        except Exception:
            return None

    def trace_key(self: Any, turn_id: int) -> Any | None:
        keys = getattr(self, "_experiment_reply_keys", None)
        return keys.get(int(turn_id)) if isinstance(keys, dict) else None

    def mark_transcript_event(self: Any, turn_id: int, channel: str) -> None:
        """Remember actual SDK transcription events, independently of seal flags."""
        try:
            observations = getattr(self, "_experiment_transcript_events", None)
            if not isinstance(observations, dict):
                observations = {}
                setattr(self, "_experiment_transcript_events", observations)
            key = int(turn_id)
            channels = observations.setdefault(key, set())
            if isinstance(channels, set):
                channels.add(str(channel))
            while len(observations) > 8:
                observations.pop(next(iter(observations)))
        except Exception:
            pass

    def transcript_event_observed(self: Any, turn_id: int, channel: str) -> bool:
        try:
            observations = getattr(self, "_experiment_transcript_events", None)
            return bool(
                isinstance(observations, dict)
                and str(channel) in observations.get(int(turn_id), set())
            )
        except Exception:
            return False

    def snapshot_trace_reply(
        self: Any, *, turn_id: int, observed_ns: int, complete: bool
    ) -> tuple[Any | None, Any | None]:
        try:
            hub = trace_hub(self)
            key = trace_key(self, turn_id)
            if hub is None or key is None:
                return hub, key
            stream = getattr(self, "stream_session", None)
            user_observed = transcript_event_observed(self, turn_id, "user")
            reply_observed = transcript_event_observed(self, turn_id, "reply")
            hub.observe_transcript(
                key,
                user_text=getattr(
                    stream, "input_transcript", getattr(stream, "input", "")
                ),
                reply_text=getattr(
                    stream, "output_transcript", getattr(stream, "output", "")
                ),
                observed_ns=int(observed_ns),
                user_text_complete=bool(
                    user_observed and (
                        complete
                        or getattr(stream, "input_transcript_sealed", False)
                    )
                ),
                reply_text_complete=bool(
                    reply_observed and (
                        complete
                        or getattr(stream, "output_transcript_sealed", False)
                    )
                ),
                user_text_observed=user_observed,
                reply_text_observed=reply_observed,
            )
            return hub, key
        except Exception:
            return None, None

    def seal_trace_reply(
        self: Any, *, turn_id: int, reason: str, observed_ns: int
    ) -> None:
        try:
            hub, key = snapshot_trace_reply(
                self,
                turn_id=turn_id,
                observed_ns=observed_ns,
                complete=reason in {"completed", "turn_complete"},
            )
            if hub is None or key is None:
                return
            hub.finish_reply(
                key, reason=str(reason)[:64], observed_ns=int(observed_ns)
            )
            keys = getattr(self, "_experiment_reply_keys", None)
            if isinstance(keys, dict) and keys.get(int(turn_id)) == key:
                keys.pop(int(turn_id), None)
        except Exception:
            pass

    def mark_trace_terminal_intent(
        self: Any, *, turn_id: int, observed_ns: int
    ) -> None:
        try:
            hub, key = snapshot_trace_reply(
                self,
                turn_id=turn_id,
                observed_ns=observed_ns,
                complete=True,
            )
            if hub is not None and key is not None:
                hub.mark_terminal_intent(
                    key, reason="turn_complete", observed_ns=int(observed_ns)
                )
        except Exception:
            pass

    def shared_start_state(self: Any) -> _SharedBridgeStartState:
        stream = getattr(self, "stream_session", None)
        current = getattr(stream, "_semantic_shared_start_state", None)
        if current is None:
            current = getattr(self, "_semantic_shared_start_state", None)
        if isinstance(current, _SharedBridgeStartState):
            return current
        current = _SharedBridgeStartState()
        try:
            setattr(stream, "_semantic_shared_start_state", current)
        except (AttributeError, TypeError):
            # The deployed StreamSession is mutable. Keep a bridge-local
            # fallback only for compatibility with restricted test doubles.
            self._semantic_shared_start_state = current
        return current

    def current_task() -> asyncio.Task[Any] | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    def clear_direct_stop_suppression(
        shared: _SharedBridgeStartState,
        task: asyncio.Task[Any] | None = None,
    ) -> None:
        task = current_task() if task is None else task
        if task is not None:
            shared.direct_stop_suppressions.pop(task, None)

    def arm_direct_stop_suppression(
        self: Any,
        shared: _SharedBridgeStartState,
    ) -> None:
        task = current_task()
        active = shared.active_bridge()
        if task is None or active is self:
            return
        token = (
            shared._identity_ref(self),
            shared._identity_ref(active),
            shared.ownership_epoch,
        )
        shared.direct_stop_suppressions[task] = token

        def expire() -> None:
            if shared.direct_stop_suppressions.get(task) is token:
                shared.direct_stop_suppressions.pop(task, None)

        try:
            asyncio.get_running_loop().call_soon(expire)
        except RuntimeError:
            expire()

    async def cancel_task(
        task: Any,
        *,
        clear_owner: tuple[Any, str] | None = None,
    ) -> list[BaseException]:
        if not isinstance(task, asyncio.Task) or task is current_task():
            return []
        errors: list[BaseException] = []
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            errors.append(exc)
        finally:
            if clear_owner is not None:
                owner, attribute = clear_owner
                if getattr(owner, attribute, None) is task:
                    setattr(owner, attribute, None)
        return errors

    async def force_cancel_bridge_tasks(self: Any) -> list[BaseException]:
        errors: list[BaseException] = []
        task_attributes = (
            "session_task",
            "_semantic_failed_start_cleanup_task",
            "robot_turn_cleanup_task",
            "robot_explicit_action_task",
            "robot_default_gesture_task",
            "robot_presentation_loop_task",
        )
        seen: set[int] = set()
        for attribute in task_attributes:
            task = getattr(self, attribute, None)
            if id(task) in seen:
                continue
            seen.add(id(task))
            errors.extend(
                await cancel_task(task, clear_owner=(self, attribute))
            )
        for attribute in ("_semantic_activity_tasks", "_semantic_terminal_tasks"):
            tasks = getattr(self, attribute, None)
            if not tasks:
                continue
            for task in tuple(tasks):
                if id(task) in seen:
                    continue
                seen.add(id(task))
                errors.extend(await cancel_task(task))
            tasks.clear()
        return errors

    async def close_browser_control(self: Any) -> list[BaseException]:
        errors: list[BaseException] = []
        ws = getattr(self, "browser_ws", None)
        if ws is not None and not any(
            getattr(ws, attribute, False) is True
            for attribute in ("closed", "_closed")
        ):
            close_ws = getattr(ws, "close", None)
            if callable(close_ws):
                try:
                    await _maybe_await(close_ws())
                except BaseException as exc:
                    errors.append(exc)
        control_task = getattr(self, "_semantic_control_task", None)
        errors.extend(
            await cancel_task(
                control_task,
                clear_owner=(self, "_semantic_control_task"),
            )
        )
        return errors

    async def retire_bridge(
        self: Any,
        *,
        close_control: bool,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        shared = shared_start_state(self)
        clear_direct_stop_suppression(shared)
        try:
            await _maybe_await(self.close(send_browser_event=False))
        except BaseException as exc:
            errors.append(exc)
            errors.extend(await force_cancel_bridge_tasks(self))
        finally:
            # retire_bridge never has an immediate upstream raw stop call.
            clear_direct_stop_suppression(shared)
        if close_control:
            errors.extend(await close_browser_control(self))
        return errors

    def report_cleanup_errors(
        errors: list[BaseException],
        *,
        context: str,
    ) -> None:
        if logger is None or len(errors) < 2:
            return
        for error in errors[1:]:
            if isinstance(error, asyncio.CancelledError):
                continue
            logger.warning(
                "Additional semantic {} cleanup failure: {}",
                context,
                type(error).__name__,
            )

    @staticmethod
    def clear_pending_controller(self: Any, current: Any) -> None:
        if getattr(self, "_semantic_pending_start_controller", None) is current:
            self._semantic_pending_start_controller = None

    async def cleanup_failed_original_start(self: Any) -> bool:
        async def cleanup() -> bool:
            try:
                await _maybe_await(self.close(send_browser_event=False))
            except BaseException as exc:
                if logger is not None:
                    logger.warning(
                        "Failed to close partially started Gemini session {}: {}",
                        getattr(getattr(self, "stream_session", None), "session_id", "unknown"),
                        type(exc).__name__,
                    )
                return False
            return True

        task = asyncio.create_task(
            cleanup(),
            name="semantic-failed-gemini-start-cleanup",
        )
        self._semantic_failed_start_cleanup_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if getattr(self, "_semantic_failed_start_cleanup_task", None) is task:
                self._semantic_failed_start_cleanup_task = None

    def controller(self: Any) -> SemanticTurnController:
        current = getattr(self, "_semantic_turn_controller", None)
        if current is None:
            current = controller_factory(self)
            self._semantic_turn_controller = current
        return current

    def fresh_controller(self: Any) -> SemanticTurnController:
        current = controller_factory(self)
        self._semantic_turn_controller = current
        return current

    def browser_ws_is_closing(self: Any) -> bool:
        ws = getattr(self, "browser_ws", None)
        if ws is None:
            return True
        for attribute in ("closed", "closing", "_closed", "_closing"):
            if getattr(ws, attribute, False) is True:
                return True
        transports = []
        request = getattr(ws, "_req", None)
        writer = getattr(ws, "_writer", None)
        transports.append(getattr(request, "transport", None))
        transports.append(getattr(writer, "transport", None))
        for transport in transports:
            is_closing = getattr(transport, "is_closing", None)
            if callable(is_closing):
                try:
                    if is_closing():
                        return True
                except Exception:
                    return True
        return False

    @staticmethod
    def public_session_reason(value: Any) -> str:
        return (
            value
            if isinstance(value, str)
            and value
            and len(value) <= 96
            and all(
                character.islower()
                or character.isdigit()
                or character == "_"
                for character in value
            )
            else "session_neutral_hook_failed"
        )

    async def ensure_session_ready(
        self: Any,
        current: SemanticTurnController,
    ) -> bool:
        if browser_ws_is_closing(self):
            return False
        start_result = await _invoke_controller_hook(
            current,
            "on_session_start",
            logger=logger,
        )
        if isinstance(start_result, Mapping) and start_result.get("ready") is True:
            return not browser_ws_is_closing(self)
        reason_code = public_session_reason(
            start_result.get("reason_code")
            if isinstance(start_result, Mapping)
            else None
        )
        if logger is not None:
            logger.warning(
                "Semantic session neutral gate blocked for session {}: {}",
                getattr(
                    getattr(self, "stream_session", None),
                    "session_id",
                    "unknown",
                ),
                reason_code,
            )
        if not browser_ws_is_closing(self):
            try:
                await safe_send(
                    self.browser_ws,
                    {
                        "type": "semantic_session_not_ready",
                        "ready": False,
                        "reasonCode": reason_code,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if logger is not None:
                    logger.debug(
                        "Could not report blocked semantic session {}: {}",
                        getattr(
                            getattr(self, "stream_session", None),
                            "session_id",
                            "unknown",
                        ),
                        type(exc).__name__,
                    )
        return False

    @functools.wraps(original_close)
    async def close_with_semantics(
        self: Any,
        *,
        send_browser_event: bool = True,
    ) -> Any:
        shared = shared_start_state(self)
        clear_direct_stop_suppression(shared)
        lock = getattr(self, "_semantic_close_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._semantic_close_lock = lock
        async with lock:
            # A bridge owns StreamSession generation only after its upstream
            # start_session succeeds (or while a partially-started call is being
            # cleaned up).  Brand-new and already-closed bridge instances must be
            # true no-ops: upstream close() unconditionally calls
            # stream_session.stop_generation(), so a stale control_ws finally
            # could otherwise stop a newer bridge on the same StreamSession.
            if not bool(getattr(self, "_semantic_generation_owner", False)):
                arm_direct_stop_suppression(self, shared)
                return None
            active = shared.active_bridge()
            if active is not None and active is not self:
                self._semantic_generation_owner = False
                arm_direct_stop_suppression(self, shared)
                return None
            try:
                result = await _maybe_await(
                    original_close(
                        self,
                        send_browser_event=send_browser_event,
                    )
                )
            except BaseException:
                shared.mark_failed_bridge(self)
                raise
            self._semantic_generation_owner = False
            shared.clear_failed_bridge(self)
            if shared.active_bridge() is self:
                shared.set_active_bridge(None)
            return result

    @functools.wraps(original_start)
    async def start_with_semantics(self: Any, *args: Any, **kwargs: Any) -> Any:
        shared = shared_start_state(self)
        clear_direct_stop_suppression(shared)
        shared.register_bridge(self)
        handler_task = _CONTROL_HANDLER_TASK.get()
        if handler_task is current_task():
            self._semantic_control_task = handler_task
        async with shared.lock:
            if shared.closing or shared.closed:
                errors = await close_browser_control(self)
                shared.unregister_bridge(self)
                if errors:
                    report_cleanup_errors(errors, context="closed-session control")
                    raise errors[0]
                return None
            # A dead replacement control socket has no claim on the shared
            # StreamSession.  Check before touching the active bridge: closing
            # the old owner would otherwise stop a healthy live generation for
            # a newcomer that can never receive Gemini events.
            if browser_ws_is_closing(self):
                shared.unregister_bridge(self)
                return None
            previous = shared.active_bridge()
            if previous is not None and previous is not self:
                same_control = (
                    getattr(previous, "browser_ws", None)
                    is getattr(self, "browser_ws", None)
                )
                errors = await retire_bridge(
                    previous,
                    close_control=not same_control,
                )
                if errors:
                    # A generation or control-channel close failure must block
                    # replacement until that exact owner can be retired.
                    shared.mark_failed_bridge(previous)
                    report_cleanup_errors(errors, context="replacement")
                    raise errors[0]
                shared.unregister_bridge(previous)
            if shared.closing or shared.closed:
                return None
            shared.set_active_bridge(self)
            # A reused bridge may own a prior Gemini task. A brand-new bridge
            # must not call upstream close() here: it would mark the shared
            # StreamSession stopped for the entire neutral-settling gate.  A
            # failed owner close stays registered so replacement also fails
            # closed and retries that exact cleanup.
            if bool(getattr(self, "_semantic_generation_owner", False)):
                await _maybe_await(self.close(send_browser_event=False))
            current = getattr(
                self,
                "_semantic_pending_start_controller",
                None,
            )
            if current is None:
                current = fresh_controller(self)
                self._semantic_pending_start_controller = current
            else:
                # A prior start waiter may have been canceled while the
                # shielded neutral preparation continued. Reattach that exact
                # controller instead of creating a second gate/POST.
                self._semantic_turn_controller = current
            session_ready = await ensure_session_ready(self, current)
            if shared.closing or shared.closed:
                clear_pending_controller(self, current)
                if shared.active_bridge() is self:
                    shared.set_active_bridge(None)
                return None
            if not session_ready:
                clear_pending_controller(self, current)
                if shared.active_bridge() is self:
                    shared.set_active_bridge(None)
                if browser_ws_is_closing(self):
                    shared.unregister_bridge(self)
                return None
            # From this point cancellation cannot reuse the neutral result:
            # original_start may already have created its Gemini session task.
            clear_pending_controller(self, current)
            # Permit upstream start_session's own initial close() and make any
            # partial start safely cleanable if it raises or is cancelled.
            self._semantic_generation_owner = True
            try:
                result = await _maybe_await(original_start(self, *args, **kwargs))
                self._semantic_generation_owner = True
                shared.register_bridge(self)
                shared.set_active_bridge(self)
                if shared.closing or shared.closed:
                    errors = await retire_bridge(self, close_control=False)
                    if errors:
                        report_cleanup_errors(
                            errors,
                            context="post-start StreamSession",
                        )
                        raise errors[0]
                    return None
                return result
            except BaseException:
                cleanup_succeeded = False
                try:
                    self._semantic_generation_owner = True
                    shared.set_active_bridge(self)
                    # Keep the controller installed through close(): its
                    # patched reset hook invalidates any semantic tasks the
                    # partially started Gemini session may have scheduled.
                    cleanup_succeeded = await cleanup_failed_original_start(self)
                finally:
                    # A failed close means partial Gemini state may still own
                    # the shared StreamSession. Keep it registered so every
                    # replacement retries that close and fails closed instead
                    # of silently starting over the leaked generation.
                    if cleanup_succeeded and shared.active_bridge() is self:
                        shared.set_active_bridge(None)
                    if getattr(self, "_semantic_turn_controller", None) is current:
                        self._semantic_turn_controller = None
                raise

    def install_control_handler_context() -> None:
        if getattr(module, "_semantic_control_handler_context_patched", False):
            return
        original_control_ws = getattr(module, "control_ws", None)
        if not callable(original_control_ws):
            return

        @functools.wraps(original_control_ws)
        async def control_ws_with_semantic_context(
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            task = current_task()
            token = _CONTROL_HANDLER_TASK.set(task)
            try:
                return await _maybe_await(original_control_ws(*args, **kwargs))
            finally:
                _CONTROL_HANDLER_TASK.reset(token)

        module.control_ws = control_ws_with_semantic_context
        module._semantic_control_handler_context_patched = True
        module._semantic_control_handler_context_original = original_control_ws

    def install_stream_session_lifecycle() -> None:
        stream_cls = getattr(module, "StreamSession", None)
        if not isinstance(stream_cls, type) or getattr(
            stream_cls,
            "_semantic_lifecycle_patched",
            False,
        ):
            return
        original_stream_close = getattr(stream_cls, "close", None)
        original_stop_generation = getattr(stream_cls, "stop_generation", None)
        if not callable(original_stream_close) or not callable(
            original_stop_generation
        ):
            raise SemanticDialogueError(
                "upstream StreamSession lifecycle contract is incompatible"
            )

        @functools.wraps(original_stop_generation)
        def stop_generation_with_owner_guard(
            stream: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            shared = getattr(stream, "_semantic_shared_start_state", None)
            task = current_task()
            if isinstance(shared, _SharedBridgeStartState) and task is not None:
                token = shared.direct_stop_suppressions.pop(task, None)
                if token is not None:
                    stale_ref, active_ref, epoch = token
                    stale_bridge = stale_ref()
                    expected_active = active_ref()
                    active = shared.active_bridge()
                    if (
                        stale_bridge is not None
                        and stale_bridge is not active
                        and expected_active is active
                        and epoch == shared.ownership_epoch
                    ):
                        return None
            return original_stop_generation(stream, *args, **kwargs)

        async def cleanup_stream(
            stream: Any,
            shared: _SharedBridgeStartState,
        ) -> Any:
            errors: list[BaseException] = []
            result: Any = None
            stream_close_succeeded = False
            async with shared.lock:
                for bridge in shared.registered_bridges():
                    errors.extend(
                        await retire_bridge(bridge, close_control=True)
                    )
                    bridge._semantic_generation_owner = False
                shared.clear_failed_bridge()
                shared.set_active_bridge(None)
                shared.clear_registered_bridges()
                shared.direct_stop_suppressions.clear()
                try:
                    result = await _maybe_await(original_stream_close(stream))
                    stream_close_succeeded = True
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    shared.closed = stream_close_succeeded
            if errors:
                report_cleanup_errors(errors, context="StreamSession")
                raise errors[0]
            return result

        @functools.wraps(original_stream_close)
        async def close_stream_with_semantics(
            stream: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if args or kwargs:
                raise SemanticDialogueError(
                    "upstream StreamSession.close signature is incompatible"
                )
            shared = getattr(stream, "_semantic_shared_start_state", None)
            if not isinstance(shared, _SharedBridgeStartState):
                shared = _SharedBridgeStartState()
                setattr(stream, "_semantic_shared_start_state", shared)
            # This assignment deliberately happens before waiting on either the
            # existing cleanup task or the ownership lock. A neutral gate that
            # currently owns the lock must observe it before starting Gemini.
            shared.closing = True
            task = shared.cleanup_task
            if task is None:
                if shared.closed:
                    return None
                task = asyncio.create_task(
                    cleanup_stream(stream, shared),
                    name=(
                        "semantic-stream-cleanup-"
                        f"{getattr(stream, 'session_id', 'unknown')}"
                    ),
                )
                shared.cleanup_task = task

                def clear_cleanup_task(done: asyncio.Task[Any]) -> None:
                    if shared.cleanup_task is done:
                        shared.cleanup_task = None

                task.add_done_callback(clear_cleanup_task)
            return await asyncio.shield(task)

        stream_cls.stop_generation = stop_generation_with_owner_guard
        stream_cls.close = close_stream_with_semantics
        stream_cls._semantic_lifecycle_patched = True
        stream_cls._semantic_lifecycle_original_close = original_stream_close
        stream_cls._semantic_lifecycle_original_stop_generation = (
            original_stop_generation
        )

    @functools.wraps(original_reset)
    def reset_with_semantics(self: Any) -> None:
        previous = int(getattr(self, "robot_turn_serial", 0))
        try:
            hub = trace_hub(self)
            key = trace_key(self, previous)
            terminal_intent_pending = bool(
                hub is not None
                and key is not None
                and hub.has_terminal_intent(key)
            )
        except Exception:
            terminal_intent_pending = False
        if not terminal_intent_pending:
            seal_trace_reply(
                self,
                turn_id=previous,
                reason="turn_reset",
                observed_ns=time.monotonic_ns(),
            )
        _cancel_activity_tasks(self)
        current = getattr(self, "_semantic_turn_controller", None)
        try:
            if current is not None:
                current.reset_turn(previous_turn_id=previous)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Semantic controller reset failed for turn {}: {}",
                    previous,
                    type(exc).__name__,
                )
        finally:
            original_reset(self)

    def mark_at(self: Any, observed_ns: int) -> None:
        original_mark(self)
        turn_id = int(getattr(self, "robot_turn_serial", 0))
        ensure_trace_reply(
            self,
            turn_id=turn_id,
            observed_ns=observed_ns,
            cause="gemini_output_activity",
        )
        current = controller(self)
        observer = getattr(current, "observe_output_activity", None)
        if callable(observer):
            try:
                observer(turn_id=turn_id, observed_ns=observed_ns)
            except Exception as exc:
                if logger is not None:
                    logger.warning(
                        "Semantic output observation failed for turn {}: {}",
                        turn_id,
                        type(exc).__name__,
                    )
        _schedule(
            self,
            lambda: _invoke_controller_hook(
                current,
                "on_output_activity",
                logger=logger,
                turn_id=turn_id,
                observed_ns=observed_ns,
            ),
            name=f"semantic-output-{turn_id}",
            turn_id=turn_id,
        )

    @functools.wraps(original_mark)
    def mark_with_semantics(self: Any) -> None:
        mark_at(self, time.monotonic_ns())

    def pending_with_semantics(self: Any) -> bool:
        upstream_pending = bool(original_pending(self)) if callable(original_pending) else False
        current = getattr(self, "_semantic_turn_controller", None)
        turn_id = int(getattr(self, "robot_turn_serial", 0))
        if upstream_pending or current is None:
            return upstream_pending
        try:
            return bool(current.has_pending(turn_id=turn_id))
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Semantic controller pending check failed for turn {}: {}",
                    turn_id,
                    type(exc).__name__,
                )
            return False

    async def no_legacy_semantic_selection(self: Any, text: str) -> None:
        del self, text
        return None

    async def handle_tool_calls(self: Any, session: Any, tool_call: Any) -> None:
        calls = getattr(tool_call, "function_calls", None) or []
        for call in calls:
            call_id = str(getattr(call, "id", "") or "")
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", None)
            turn_id = int(getattr(self, "robot_turn_serial", 0))
            reply_key = ensure_trace_reply(
                self,
                turn_id=turn_id,
                observed_ns=time.monotonic_ns(),
                cause="gemini_tool_call",
            )
            if not call_id:
                await safe_send(
                    self.browser_ws,
                    {
                        "type": "semantic_gesture_plan",
                        "turnId": int(getattr(self, "robot_turn_serial", 0)),
                        "accepted": False,
                        "functionCount": 0,
                        "executionMode": "dry_run",
                        "error": "missing_call_id",
                    },
                )
                if logger is not None:
                    logger.warning("Ignoring semantic tool call without an id")
                continue
            if name != SEMANTIC_TOOL_NAME:
                response_payload: Mapping[str, Any] = {
                    "accepted": False,
                    "error": "unsupported_tool",
                }
            elif not isinstance(args, Mapping):
                response_payload = {"accepted": False, "error": "invalid_arguments"}
            else:
                try:
                    response_payload = await controller(self).accept_tool_plan(
                        turn_id=int(getattr(self, "robot_turn_serial", 0)),
                        call_id=call_id,
                        payload=args,
                    )
                except Exception as exc:  # fail closed: Gemini may still speak without motion
                    if logger is not None:
                        logger.warning(
                            "Semantic gesture plan rejected for turn {}: {}",
                            int(getattr(self, "robot_turn_serial", 0)),
                            type(exc).__name__,
                        )
                    response_payload = {
                        "accepted": False,
                        "error": "semantic_plan_rejected",
                    }
            try:
                previews = getattr(
                    self, "_experiment_semantic_plan_previews", None
                )
                if not isinstance(previews, dict):
                    previews = {}
                    self._experiment_semantic_plan_previews = previews
                preview_status = (
                    "accepted"
                    if response_payload.get("accepted") is True
                    else str(response_payload.get("error", "plan_rejected"))
                )
                previews[turn_id] = {
                    "accepted": response_payload.get("accepted") is True,
                    "status": preview_status,
                    "ordered_functions": (
                        list(args.get("ordered_functions", []))
                        if isinstance(args, Mapping) else []
                    ),
                }
                while len(previews) > 8:
                    previews.pop(next(iter(previews)))

                hub = trace_hub(self)
                if hub is not None and reply_key is not None:
                    request_id = response_payload.get("request_id")
                    revision = response_payload.get("revision")
                    if (
                        response_payload.get("accepted") is True
                        and isinstance(request_id, str) and request_id
                        and isinstance(revision, int) and not isinstance(revision, bool)
                    ):
                        hub.bind_semantic(reply_key, request_id, revision)
                    hub.observe_semantic_summary(
                        reply_key,
                        status=preview_status,
                        ordered_functions=(
                            list(args.get("ordered_functions", []))
                            if isinstance(args, Mapping) else []
                        ),
                        terminal_summary=False,
                        observed_ns=time.monotonic_ns(),
                    )
            except Exception:
                pass

            try:
                await session.send_tool_response(
                    function_responses=[
                        types_module.FunctionResponse(
                            id=call_id,
                            name=name or SEMANTIC_TOOL_NAME,
                            response=dict(response_payload),
                        )
                    ]
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if logger is not None:
                    logger.warning(
                        "Semantic tool response failed for turn {}: {}",
                        int(getattr(self, "robot_turn_serial", 0)),
                        type(exc).__name__,
                    )
                await safe_send(
                    self.browser_ws,
                    {
                        "type": "semantic_tool_response_failed",
                        "turnId": int(getattr(self, "robot_turn_serial", 0)),
                        "error": "tool_response_failed",
                    },
                )
            await safe_send(
                self.browser_ws,
                {
                    "type": "semantic_gesture_plan",
                    "turnId": int(getattr(self, "robot_turn_serial", 0)),
                    "accepted": bool(response_payload.get("accepted")),
                    "functionCount": int(response_payload.get("function_count", 0)),
                    "executionMode": response_payload.get("execution_mode", "dry_run"),
                    "forceFirstPolicyEnabled": bool(
                        response_payload.get("force_first_policy_enabled", False)
                    ),
                    "error": response_payload.get("error", ""),
                },
            )

    @functools.wraps(original_receive)
    async def receive_with_semantics(self: Any, session: Any) -> None:
        # Defense in depth for direct/internal receiver invocations. Normal
        # production flow has already completed this cached gate in start_session.
        shared = shared_start_state(self)
        if shared.closing or shared.closed:
            return
        if not await ensure_session_ready(self, controller(self)):
            return
        if shared.closing or shared.closed:
            return
        while True:
            async for response in session.receive():
                if logger is not None:
                    logger.debug(
                        "Received Gemini Live response for semantic session {}: {}",
                        self.stream_session.session_id,
                        response,
                    )

                if getattr(response, "go_away", None):
                    await safe_send(
                        self.browser_ws,
                        {
                            "type": "status",
                            "level": "warn",
                            "message": (
                                "Gemini requested this connection to end or reconnect soon: "
                                f"{response.go_away}"
                            ),
                        },
                    )

                tool_call = getattr(response, "tool_call", None)
                if tool_call is not None:
                    await handle_tool_calls(self, session, tool_call)

                server_content = getattr(response, "server_content", None)
                if not server_content:
                    continue

                input_transcription = getattr(server_content, "input_transcription", None)
                if input_transcription is not None and hasattr(input_transcription, "text"):
                    mark_transcript_event(
                        self, int(getattr(self, "robot_turn_serial", 0)), "user"
                    )
                    input_text = getattr(input_transcription, "text", "") or ""
                    if input_text:
                        self.stream_session.update_input_transcript(input_text)
                        await safe_send(
                            self.browser_ws,
                            {"type": "input_transcription", "text": input_text},
                        )

                output_transcription = getattr(server_content, "output_transcription", None)
                if output_transcription is not None and hasattr(output_transcription, "text"):
                    mark_transcript_event(
                        self, int(getattr(self, "robot_turn_serial", 0)), "reply"
                    )
                    output_text = getattr(output_transcription, "text", "") or ""
                    if output_text:
                        self._mark_robot_output_activity()
                        self.stream_session.update_output_transcript(output_text)
                        await safe_send(
                            self.browser_ws,
                            {"type": "output_transcription", "text": output_text},
                        )

                model_turn = getattr(server_content, "model_turn", None)
                if model_turn and getattr(model_turn, "parts", None):
                    for part in model_turn.parts:
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and getattr(inline_data, "data", None):
                            audio_bytes = bytes(inline_data.data)
                            observed_ns = time.monotonic_ns()
                            mime_type = str(
                                getattr(inline_data, "mime_type", "audio/pcm;rate=24000")
                            )
                            mark_at(self, observed_ns)
                            self.stream_session.enqueue_gemini_audio(audio_bytes, mime_type)
                            turn_id = int(getattr(self, "robot_turn_serial", 0))
                            _schedule(
                                self,
                                lambda: _invoke_controller_hook(
                                    controller(self),
                                    "on_audio_chunk",
                                    logger=logger,
                                    turn_id=turn_id,
                                    audio_bytes=audio_bytes,
                                    mime_type=mime_type,
                                    observed_ns=observed_ns,
                                ),
                                name=f"semantic-audio-{turn_id}",
                                turn_id=turn_id,
                            )

                if getattr(server_content, "interrupted", False):
                    turn_id = int(getattr(self, "robot_turn_serial", 0))
                    observed_ns = time.monotonic_ns()
                    self.stream_session.mark_interrupted()
                    seal_trace_reply(
                        self,
                        turn_id=turn_id,
                        reason="interrupted",
                        observed_ns=observed_ns,
                    )
                    self._reset_robot_turn_state()
                    await safe_send(self.browser_ws, {"type": "interrupted"})
                    _schedule(
                        self,
                        lambda: _invoke_controller_hook(
                            controller(self),
                            "on_interrupted",
                            logger=logger,
                            turn_id=turn_id,
                        ),
                        name=f"semantic-interrupted-{turn_id}",
                        turn_id=turn_id,
                        cancel_on_reset=False,
                    )

                if getattr(server_content, "turn_complete", False):
                    turn_id = int(getattr(self, "robot_turn_serial", 0))
                    observed_ns = time.monotonic_ns()
                    self.robot_turn_complete_received = True
                    # Queue the final marker while the reply is still active so
                    # its padded/drained media retains the immutable producer key.
                    self.stream_session.mark_turn_complete()
                    try:
                        previews = getattr(
                            self, "_experiment_semantic_plan_previews", None
                        )
                        preview = (
                            previews.get(turn_id)
                            if isinstance(previews, dict) else None
                        )
                        hub = trace_hub(self)
                        key = trace_key(self, turn_id)
                        if (
                            hub is not None
                            and key is not None
                            and (
                                not isinstance(preview, Mapping)
                                or preview.get("accepted") is not True
                            )
                        ):
                            hub.observe_semantic_summary(
                                key,
                                status=(
                                    str(preview.get("status", "no_plan"))
                                    if isinstance(preview, Mapping)
                                    else "no_plan"
                                ),
                                planned_functions=[],
                                completed_functions=[],
                                dropped_functions=(
                                    preview.get("ordered_functions", [])
                                    if isinstance(preview, Mapping) else []
                                ),
                                result_action_ids=[],
                                execution_summary_complete=True,
                                recovery_required=False,
                                closed_neutral=True,
                                terminal_summary=True,
                                observed_ns=observed_ns,
                            )
                    except Exception:
                        pass
                    mark_trace_terminal_intent(
                        self,
                        turn_id=turn_id,
                        observed_ns=observed_ns,
                    )
                    self._cancel_robot_task(self.robot_turn_cleanup_task)
                    self.robot_turn_cleanup_task = asyncio.create_task(
                        self._cleanup_robot_turn_after_playback(turn_id)
                    )
                    await safe_send(self.browser_ws, {"type": "turn_complete"})
                    _schedule(
                        self,
                        lambda: _invoke_controller_hook(
                            controller(self),
                            "on_turn_complete",
                            logger=logger,
                            turn_id=turn_id,
                        ),
                        name=f"semantic-complete-{turn_id}",
                        turn_id=turn_id,
                        cancel_on_reset=False,
                    )

            if logger is not None:
                logger.debug(
                    "Gemini semantic receive iterator completed; re-entering for {}",
                    self.stream_session.session_id,
                )

    if callable(original_emit_robot_action):
        @functools.wraps(original_emit_robot_action)
        async def emit_robot_action_with_trace(
            self: Any, payload: Mapping[str, Any]
        ) -> Any:
            try:
                if (
                    isinstance(payload, Mapping)
                    and payload.get("kind") == "semantic_atom_sequence"
                ):
                    hub = trace_hub(self)
                    request_id = payload.get("requestId")
                    revision = payload.get("revision")
                    if (
                        hub is not None
                        and isinstance(request_id, str) and request_id
                        and isinstance(revision, int) and not isinstance(revision, bool)
                    ):
                        summary: dict[str, Any] = {
                            "status": payload.get("status"),
                            "planned_functions": payload.get(
                                "plannedFunctions", []
                            ),
                            "completed_functions": payload.get(
                                "completedFunctions", []
                            ),
                            "dropped_functions": payload.get(
                                "droppedFunctions", []
                            ),
                            "execution_summary_complete": True,
                            "recovery_required": bool(
                                payload.get("completedFunctions", [])
                            ),
                            "recovery_action_id": payload.get(
                                "recoveryActionId"
                            ),
                            "closed_neutral": payload.get("closedNeutral"),
                            "terminal_summary": True,
                            "observed_ns": time.monotonic_ns(),
                        }
                        if "actionIds" in payload:
                            summary["result_action_ids"] = payload.get(
                                "actionIds"
                            )
                        hub.observe_semantic_summary_for_semantic(
                            request_id, revision, **summary
                        )
            except Exception:
                pass
            return await _maybe_await(
                original_emit_robot_action(self, payload)
            )

        cls._emit_robot_action_event = emit_robot_action_with_trace

    cls.start_session = start_with_semantics
    cls._reset_robot_turn_state = reset_with_semantics
    cls._mark_robot_output_activity = mark_with_semantics
    cls._robot_has_pending_turn_task = pending_with_semantics
    cls._handle_robot_input_transcription = no_legacy_semantic_selection
    cls._handle_robot_output_transcription = no_legacy_semantic_selection
    cls._receive_loop = receive_with_semantics
    cls.close = close_with_semantics
    cls._semantic_dialogue_patched = True
    cls._semantic_dialogue_original_start = original_start
    cls._semantic_dialogue_original_receive = original_receive
    install_control_handler_context()
    install_stream_session_lifecycle()
    _install_runtime_terminal_failure_trace()
    return True
