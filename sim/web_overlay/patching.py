from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import functools
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping

from .atom_catalog import load_catalog
from .compositor import CompositeVideoTrack
from .config import OverlayConfig
from .dialogue_sequence import (
    DialogueSequenceCoordinator,
    SequenceSettings,
    patch_random_dialogue_bridge_class,
)
from .random_planner import random_planner
try:
    from .recording_integration import (
        START_PATH as EXPERIMENT_RECORDING_START_PATH,
        STATUS_PATH as EXPERIMENT_RECORDING_STATUS_PATH,
        STOP_PATH as EXPERIMENT_RECORDING_STOP_PATH,
        install_experiment_recording,
        install_semantic_receipt_observer,
    )
    from .recording_ui import inject_experiment_recording_ui
except Exception:  # optional fail-open sidecar; core overlay must still boot
    EXPERIMENT_RECORDING_START_PATH = "/api/experiment-recording/start"
    EXPERIMENT_RECORDING_STOP_PATH = "/api/experiment-recording/stop"
    EXPERIMENT_RECORDING_STATUS_PATH = "/api/experiment-recording/status/{session_id}"
    install_experiment_recording = None
    install_semantic_receipt_observer = None
    inject_experiment_recording_ui = None
from .sim_client import SimServiceClient


_EXPERIMENT_RECORDING_AVAILABLE = False


@dataclass(slots=True)
class PatchReport:
    queue_track_classes: list[str]
    stream_session_classes: list[str]
    peer_connection_classes: list[str]
    action_symbols: list[str]
    robot_runtime_symbols: list[str]
    html_globals: list[str]
    aiohttp_routes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_namespaces(module: ModuleType) -> Iterable[tuple[str, ModuleType]]:
    yield module.__name__, module
    seen = {id(module)}
    for name, value in vars(module).items():
        if not isinstance(value, ModuleType) or id(value) in seen:
            continue
        if hasattr(value, "QueueStreamTrack") or hasattr(value, "StreamSession") or hasattr(
            value, "trigger_robot_action"
        ):
            seen.add(id(value))
            yield f"{module.__name__}.{name}", value


def _patch_queue_track_class(
    cls: type[Any], client: SimServiceClient, config: OverlayConfig
) -> bool:
    """Fallback for upstreams that do not expose ``RTCPeerConnection``.

    The adapter calls the captured original ``recv`` in a background task.  It
    must not call ``self.recv`` after patching, otherwise the compositor would
    recurse into itself.  Normal deployments use the cleaner PeerConnection
    wrapper below; this path keeps the same independent clock for compatibility.
    """

    if getattr(cls, "_mujoco_overlay_patched", False):
        return False
    original_recv = getattr(cls, "recv", None)
    if original_recv is None or not inspect.iscoroutinefunction(original_recv):
        return False

    class OriginalRecvSource:
        def __init__(self, track: Any) -> None:
            self.track = track

        async def recv(self):
            return await original_recv(self.track)

    @functools.wraps(original_recv)
    async def recv_with_overlay(self: Any):
        if getattr(self, "kind", None) != "video":
            return await original_recv(self)
        composite = getattr(self, "_mujoco_composite_track", None)
        if composite is None:
            composite = CompositeVideoTrack(
                OriginalRecvSource(self),
                client,
                config,
                stop_source=False,
                reply_hub=getattr(self, "_experiment_reply_trace_hub", None),
            )
            self._mujoco_composite_track = composite
        return await composite.recv()

    cls.recv = recv_with_overlay
    cls._mujoco_overlay_patched = True
    cls._mujoco_overlay_original_recv = original_recv
    return True


def _patch_peer_connection_class(
    namespace: ModuleType,
    client: SimServiceClient,
    config: OverlayConfig,
) -> str | None:
    base_cls = getattr(namespace, "RTCPeerConnection", None)
    if not isinstance(base_cls, type) or getattr(
        base_cls, "_mujoco_overlay_patched", False
    ):
        return None

    class OverlayPeerConnection(base_cls):
        _mujoco_overlay_patched = True

        def addTrack(self, track: Any):
            if getattr(track, "kind", None) == "video" and not isinstance(
                track, CompositeVideoTrack
            ):
                track = CompositeVideoTrack(
                    track,
                    client,
                    config,
                    reply_hub=getattr(self, "_experiment_reply_trace_hub", None),
                )
            return super().addTrack(track)

    OverlayPeerConnection.__name__ = base_cls.__name__
    OverlayPeerConnection.__qualname__ = base_cls.__qualname__
    setattr(namespace, "RTCPeerConnection", OverlayPeerConnection)
    return f"{namespace.__name__}.RTCPeerConnection"


PRESENTATION_GESTURE_IDS = (
    "joint_presentation_gesture",
    "joint_presentation_gesture_b",
    "joint_presentation_gesture_c",
)
DEFAULT_PRESENTATION_GESTURE_ID = PRESENTATION_GESTURE_IDS[0]
PRESENTATION_GESTURE_SPECS = {
    PRESENTATION_GESTURE_IDS[0]: {
        "id": PRESENTATION_GESTURE_IDS[0],
        "label": "presentation",
    },
    PRESENTATION_GESTURE_IDS[1]: {
        "id": PRESENTATION_GESTURE_IDS[1],
        "label": "greeting",
    },
    PRESENTATION_GESTURE_IDS[2]: {
        "id": PRESENTATION_GESTURE_IDS[2],
        "label": "thanks",
    },
}
RELEASE_ACTION_ID = "joint_release_arm_pose"
ACTION_LIFECYCLE_POLL_SECONDS = 0.05
ACTION_LIFECYCLE_TIMEOUT_SECONDS = 45.0
DIALOGUE_SEQUENCE_MODES = frozenset(
    {"legacy_abc", "random_catalog", "semantic_f17"}
)


def simulator_robot_runtime() -> dict[str, Any]:
    """Return the camelCase runtime schema expected by the untouched web UI."""

    available = [dict(spec) for spec in PRESENTATION_GESTURE_SPECS.values()]
    available.append({"id": RELEASE_ACTION_ID, "label": "release"})
    return {
        "enabled": True,
        "simulationMode": True,
        "networkInterface": "loopback-only",
        "availableActions": available,
        "reasonCodes": [],
        "reasonLabel": "MuJoCo simulator enabled",
        "backend": "mujoco",
    }


def _random_sequence_coordinator_factory(
    client: SimServiceClient,
    config: OverlayConfig,
) -> Callable[[Any], DialogueSequenceCoordinator]:
    """Validate the exact Catalog once, then create per-connection coordinators."""

    catalog = load_catalog(Path(config.atom_catalog_path).expanduser().resolve())
    settings = SequenceSettings(
        count=config.random_sequence_count,
        base_seed=config.random_sequence_base_seed,
        log_path=Path(config.random_sequence_log_path).expanduser(),
        poll_interval=config.random_sequence_poll_interval,
        action_timeout=config.random_sequence_action_timeout,
    )

    def factory(bridge: Any) -> DialogueSequenceCoordinator:
        async def emit(payload: dict[str, Any]) -> None:
            event = {
                **payload,
                "playbackSource": "approved_atom_catalog",
                "robot": simulator_robot_runtime(),
            }
            await bridge._emit_robot_action_event(event)

        return DialogueSequenceCoordinator(
            client=client,
            atom_catalog=catalog,
            planner=random_planner,
            settings=settings,
            event_sink=emit,
        )

    return factory


def _semantic_sequence_runtime_setup(
    module: ModuleType,
    client: SimServiceClient,
    config: OverlayConfig,
) -> tuple[Callable[[Any], Any], Callable[..., bool]]:
    """Load semantic-only dependencies only when that mode is selected."""

    from .semantic_catalog import load_semantic_catalog
    from .semantic_controller import semantic_controller_factory
    from .semantic_dialogue import (
        load_semantic_prompt,
        patch_live_config_builder,
        patch_semantic_bridge_class,
    )

    if callable(install_semantic_receipt_observer):
        try:
            install_semantic_receipt_observer(module)
        except Exception:
            pass

    catalog = load_semantic_catalog(
        Path(config.semantic_catalog_path).expanduser().resolve()
    )
    prompt = load_semantic_prompt(
        Path(config.semantic_prompt_path).expanduser().resolve()
    )
    factory = semantic_controller_factory(
        catalog=catalog,
        client=client,
        config=config,
    )
    patch_live_config_builder(module, prompt=prompt)
    return factory, patch_semantic_bridge_class


def pick_robot_action(
    text: Any, requested_action: Any = ""
) -> dict[str, str] | None:
    """Match the real bridge API: return a motion spec dictionary or ``None``."""

    if isinstance(requested_action, dict):
        requested_action = requested_action.get("id", "")
    if requested_action == RELEASE_ACTION_ID:
        return {"id": RELEASE_ACTION_ID, "label": "release"}
    if requested_action in PRESENTATION_GESTURE_SPECS:
        return dict(PRESENTATION_GESTURE_SPECS[str(requested_action)])

    normalized = str(text or "").lower()
    if any(word in normalized for word in ("thank", "thanks", "谢谢", "感谢")):
        return dict(PRESENTATION_GESTURE_SPECS[PRESENTATION_GESTURE_IDS[2]])
    if any(
        word in normalized
        for word in ("hello", "welcome", "你好", "您好", "欢迎")
    ):
        return dict(PRESENTATION_GESTURE_SPECS[PRESENTATION_GESTURE_IDS[1]])
    if any(
        word in normalized
        for word in (
            "presentation",
            "present",
            "introduce",
            "introduction",
            "explain",
            "describe",
            "tell me about",
            "demonstrate",
            "演讲",
            "介绍",
            "讲解",
        )
    ):
        return dict(PRESENTATION_GESTURE_SPECS[PRESENTATION_GESTURE_IDS[0]])
    return None


def select_presentation_action(text: Any, explicit_action: Any = None) -> str:
    """Return an ID for the simulator trigger, with a deterministic fallback."""

    if isinstance(explicit_action, dict):
        explicit_action = explicit_action.get("id", "")
    if isinstance(explicit_action, str) and explicit_action:
        # Pass explicit simulator-safe IDs through. The motion service registry
        # remains the allow-list and rejects IDs that have no JSON motion.
        return explicit_action
    spec = pick_robot_action(text, explicit_action)
    return spec["id"] if spec is not None else DEFAULT_PRESENTATION_GESTURE_ID


def _action_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    call_kwargs = dict(kwargs)
    # The existing demo calls trigger_robot_action(text, action_id). Prefer the
    # explicit second argument; if absent, retain its deterministic text picker.
    if len(args) >= 2:
        action = select_presentation_action(args[0], args[1])
    elif args:
        first = args[0]
        if isinstance(first, str) and first not in PRESENTATION_GESTURE_IDS:
            action = select_presentation_action(first)
        else:
            action = first
    else:
        for key in ("motion", "preset", "action", "action_id", "presentation_id"):
            if key in call_kwargs:
                value = call_kwargs.pop(key)
                if key in ("motion", "preset"):
                    action = {key: value}
                else:
                    action = value
                break
        else:
            raise TypeError("trigger_robot_action requires an action, preset, or motion")
    supported = {
        key: call_kwargs[key]
        for key in ("mode", "speed", "blend_duration")
        if key in call_kwargs
    }
    return action, supported


def _compatible_trigger_result(action: Any, result: dict[str, Any]) -> dict[str, Any]:
    label = "inline_motion" if isinstance(action, dict) else str(action)
    return {
        **result,
        "matched": True,
        "executed": bool(result.get("accepted", True)),
        "simulated": True,
        "action": action,
        "label": label,
        "playbackSource": "motion_file",
        "robot": simulator_robot_runtime(),
        "simulatorAction": result,
    }


def _action_request_id(result: Mapping[str, Any]) -> str:
    value = result.get("id") or result.get("action_id") or result.get("request_id")
    return "" if value is None else str(value)


def _state_record_matches(record: Any, request_id: str) -> bool:
    if isinstance(record, Mapping):
        value = record.get("id") or record.get("action_id") or record.get("request_id")
        return value is not None and str(value) == request_id
    return record is not None and str(record) == request_id


def _compact_simulator_status(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep lifecycle WebSocket events small; `/state` also contains every joint."""

    queued = state.get("queued")
    return {
        "action": state.get("action"),
        "lastCompleted": state.get("last_completed"),
        "fault": state.get("fault"),
        "queueDepth": len(queued) if isinstance(queued, list) else None,
    }


def _lifecycle_payload(
    action: Any,
    result: Mapping[str, Any],
    lifecycle: str,
    *,
    browser_executed: bool = False,
    state: Mapping[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    payload = _compatible_trigger_result(action, dict(result))
    payload.update(
        {
            "lifecycle": lifecycle,
            "actionRequestId": _action_request_id(result),
            # The untouched upstream needs `executed=True` in the return value to
            # advance its turn state. Browser lifecycle events override this flag
            # until completion, avoiding three generic "executed" UI messages.
            "executed": browser_executed,
            "scheduled": lifecycle == "scheduled",
            "running": lifecycle == "started",
            "completed": lifecycle == "completed",
            "kind": "simulator_motion",
        }
    )
    if state is not None:
        payload["simulatorStatus"] = _compact_simulator_status(state)
    if error:
        payload["error"] = error
    return payload


async def _monitor_action_lifecycle(
    bridge: Any,
    client: SimServiceClient,
    action: Any,
    submission: Mapping[str, Any],
) -> bool:
    """Emit started/completed after observing the submitted simulator request."""

    request_id = _action_request_id(submission)
    if not request_id:
        payload = _lifecycle_payload(
            action,
            submission,
            "failed",
            error="Simulator accepted the action without returning a request id.",
        )
        await bridge._emit_robot_action_event(payload)
        return False

    loop = asyncio.get_running_loop()
    deadline = loop.time() + ACTION_LIFECYCLE_TIMEOUT_SECONDS
    started_emitted = False
    try:
        while loop.time() < deadline:
            state = await asyncio.to_thread(client.state_sync)
            fault = state.get("fault")
            if fault:
                payload = _lifecycle_payload(
                    action,
                    submission,
                    "failed",
                    state=state,
                    error=f"Simulator fault while running {request_id}: {fault}",
                )
                await bridge._emit_robot_action_event(payload)
                return False

            completed = state.get("last_completed")
            if _state_record_matches(completed, request_id):
                # A short action may start and finish between two 50 ms polls.
                # The completed simulator record still proves it started, so keep
                # the browser event order deterministic.
                if not started_emitted:
                    started = _lifecycle_payload(
                        action, submission, "started", state=state
                    )
                    started["observedFrom"] = "last_completed"
                    await bridge._emit_robot_action_event(started)
                    started_emitted = True

                status = (
                    str(completed.get("status", "completed"))
                    if isinstance(completed, Mapping)
                    else "completed"
                )
                if status != "completed":
                    failed = _lifecycle_payload(
                        action,
                        submission,
                        "failed",
                        state=state,
                        error=f"Simulator action {request_id} ended with status {status}.",
                    )
                    await bridge._emit_robot_action_event(failed)
                    return False

                done = _lifecycle_payload(
                    action,
                    submission,
                    "completed",
                    browser_executed=True,
                    state=state,
                )
                if isinstance(completed, Mapping):
                    done["completion"] = dict(completed)
                await bridge._emit_robot_action_event(done)
                return True

            if not started_emitted and _state_record_matches(
                state.get("action"), request_id
            ):
                started = _lifecycle_payload(
                    action, submission, "started", state=state
                )
                await bridge._emit_robot_action_event(started)
                started_emitted = True

            await asyncio.sleep(ACTION_LIFECYCLE_POLL_SECONDS)

        timed_out = _lifecycle_payload(
            action,
            submission,
            "failed",
            error=(
                f"Timed out after {ACTION_LIFECYCLE_TIMEOUT_SECONDS:.1f}s "
                f"waiting for simulator action {request_id}."
            ),
        )
        await bridge._emit_robot_action_event(timed_out)
        return False
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # preserve the upstream robot_action error contract
        failed = _lifecycle_payload(
            action, submission, "failed", error=str(exc)
        )
        await bridge._emit_robot_action_event(failed)
        return False


def _patch_gemini_bridge_class(cls: type[Any], client: SimServiceClient) -> bool:
    """Replace the one blocking action event with an async lifecycle stream."""

    if getattr(cls, "_mujoco_action_lifecycle_patched", False):
        return False
    original_trigger = getattr(cls, "_trigger_robot_action", None)
    if original_trigger is None or not inspect.iscoroutinefunction(original_trigger):
        return False

    @functools.wraps(original_trigger)
    async def trigger_with_lifecycle(
        self: Any, *, text: str = "", action_id: str = ""
    ) -> dict[str, Any]:
        action = select_presentation_action(text, action_id)
        turn_serial = getattr(self, "robot_turn_serial", None)

        # Preserve the old bridge's serial execution contract without withholding
        # the first scheduled event for the duration of the motion.
        previous = getattr(self, "_mujoco_action_lifecycle_task", None)
        if previous is not None and not previous.done():
            await asyncio.shield(previous)

        # The untouched presentation loop starts 0.3 s after an accepted action.
        # If it waited for an earlier action and the turn ended meanwhile, do not
        # submit a stale follow-up gesture.
        if action in PRESENTATION_GESTURE_IDS and (
            turn_serial != getattr(self, "robot_turn_serial", turn_serial)
            or bool(getattr(self, "robot_turn_complete_received", False))
        ):
            return _lifecycle_payload(action, {}, "cancelled")

        try:
            submission = await asyncio.to_thread(client.trigger_sync, action)
        except Exception as exc:  # same matched/error payload as the upstream path
            failed = _lifecycle_payload(action, {}, "failed", error=str(exc))
            await self._emit_robot_action_event(failed)
            return failed

        if not bool(submission.get("accepted", True)):
            failed = _lifecycle_payload(
                action,
                submission,
                "failed",
                error=str(submission.get("error") or "Simulator rejected the action."),
            )
            await self._emit_robot_action_event(failed)
            return failed

        scheduled = _lifecycle_payload(action, submission, "scheduled")
        await self._emit_robot_action_event(scheduled)
        if not _action_request_id(submission):
            failed = _lifecycle_payload(
                action,
                submission,
                "failed",
                error="Simulator accepted the action without returning a request id.",
            )
            await self._emit_robot_action_event(failed)
            return failed

        monitor = asyncio.create_task(
            _monitor_action_lifecycle(self, client, action, submission),
            name=f"mujoco-action-{_action_request_id(submission) or 'unknown'}",
        )
        self._mujoco_action_lifecycle_task = monitor

        # Compatibility return: upstream uses this private call's `executed`
        # boolean to start its presentation-loop bookkeeping. The browser already
        # received a distinct scheduled event with `executed=False` above.
        accepted = dict(scheduled)
        accepted["executed"] = True
        return accepted

    cls._trigger_robot_action = trigger_with_lifecycle
    cls._mujoco_action_lifecycle_patched = True
    cls._mujoco_action_lifecycle_original_trigger = original_trigger
    return True


def _make_action_replacement(
    original: Callable[..., Any] | None,
    client: SimServiceClient,
) -> Callable[..., Any]:
    # The original real-robot bridge blocks until its subprocess has finished.
    # Preserve that contract so the untouched 0.3 s presentation-loop gap is
    # measured after playback, rather than flooding the simulator queue.
    legacy_trigger = getattr(client, "trigger_legacy_sync", client.trigger_sync)
    if original is not None and inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_trigger(*args: Any, **kwargs: Any):
            action, options = _action_from_call(args, kwargs)
            result = await asyncio.to_thread(legacy_trigger, action, **options)
            return _compatible_trigger_result(action, result)

        return async_trigger

    def sync_trigger(*args: Any, **kwargs: Any):
        action, options = _action_from_call(args, kwargs)
        result = legacy_trigger(action, **options)
        return _compatible_trigger_result(action, result)

    if original is not None:
        sync_trigger = functools.wraps(original)(sync_trigger)
    return sync_trigger


class SimRobotRuntime:
    """Small compatibility facade used in place of the real-robot runtime."""

    simulation_mode = True
    enabled = True

    def __init__(self, client: SimServiceClient):
        self.client = client

    def __call__(self) -> dict[str, Any]:
        return simulator_robot_runtime()

    def trigger_robot_action(self, action: Any, **kwargs: Any) -> dict[str, Any]:
        return self.client.trigger_sync(action, **kwargs)

    run_action = trigger_robot_action
    execute = trigger_robot_action
    submit = trigger_robot_action

    async def trigger_robot_action_async(
        self, action: Any, **kwargs: Any
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.client.trigger_sync, action, **kwargs)

    def release(self) -> dict[str, Any]:
        return self.client.release_sync()

    def stop(self) -> dict[str, Any]:
        return self.client.stop_action_sync()

    cancel = stop

    def status(self) -> dict[str, Any]:
        return self.client.state_sync()

    get_status = status


WIDESCREEN_CSS = """
<style id="mujoco-overlay-widescreen">
  .stage {
    min-width: 0 !important;
  }

  .video-shell {
    width: 100% !important;
    max-width: 1280px !important;
    height: auto !important;
    max-height: none !important;
    aspect-ratio: 8 / 3 !important;
  }

  .video-shell > video.live-player {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    object-position: center center !important;
  }

  @media (max-width: 700px) and (orientation: portrait) {
    .video-shell {
      width: 100vw !important;
      height: auto !important;
      aspect-ratio: 8 / 3 !important;
    }
  }
</style>
""".strip()


ACTION_LIFECYCLE_SCRIPT = """
<script id="mujoco-action-lifecycle-ui">
(() => {
  const upstreamUpdateRobotPanel = window.updateRobotPanel;
  const upstreamSetStatus = window.setStatus;
  if (typeof upstreamUpdateRobotPanel !== "function") {
    return;
  }
  let sessionNeutralOwnsStatus = false;
  if (typeof upstreamSetStatus === "function") {
    window.setStatus = function setMujocoSessionStatus(text, cls = "") {
      if (sessionNeutralOwnsStatus && text === "Session stopped") {
        return;
      }
      upstreamSetStatus(text, cls);
    };
  }

  window.updateRobotPanel = function updateMujocoRobotLifecycle(payload) {
    const lifecycle = payload && payload.lifecycle;
    const sessionNeutral =
      payload &&
      payload.kind === "semantic_session_neutral" &&
      typeof payload.status === "string";
    if (sessionNeutral) {
      const status = payload.status;
      const reasonCode = String(payload.reasonCode || "session_neutral_unknown");
      const reasonLabels = {
        simulator_busy: "Another simulator client is active",
        neutral_admission_conflict: "Simulator state changed during neutral admission",
        neutral_stability_timeout: "Neutral pose did not settle before timeout",
        simulator_action_sequence_changed: "Simulator activity changed while settling",
        simulator_fault: "Simulator reported a fault",
        neutral_action_failed: "Safe-neutral action failed",
        neutral_action_timeout: "Safe-neutral action did not complete before timeout",
        neutral_action_observation_timeout: "Safe-neutral completion could not be verified",
        neutral_target_changed: "Neutral target changed while settling",
      };
      const statusLabels = {
        preparing: "Preparing canonical neutral",
        ready: "Semantic session ready",
        blocked: "Semantic session blocked",
      };
      const label = statusLabels[status] || "Semantic session neutral gate";
      const explanation = reasonLabels[reasonCode] || reasonCode;

      sessionNeutralOwnsStatus = status !== "ready";
      if (status === "preparing") {
        setStatus("Returning robot to canonical neutral before Gemini starts.");
      } else if (status === "ready") {
        setStatus("Canonical neutral is stable. Connecting to Gemini Live.", "ok");
      } else if (status === "blocked") {
        setStatus(`Semantic session blocked: ${explanation}.`, "warn");
      }
      currentTurnHasRobotAction = false;
      setRobotMode("Mode: simulation");
      setRobotAction(`Action: ${label}`);
      setRobotResult(`Result: ${label}.`);
      const details = [`Reason: ${reasonCode}`, explanation];
      if (payload.actionId) {
        details.push(`Neutral action: ${payload.actionId}`);
      }
      if (Number.isInteger(payload.stableSamples) && payload.stableSamples > 0) {
        details.push(`Stable samples: ${payload.stableSamples}`);
      }
      setRobotDetail(`Detail: ${details.join(" | ")}`);
      // The untouched upstream robot_action branch otherwise replaces the
      // precise neutral message with generic "simulating: robot action" text.
      payload.executed = false;
      return;
    }
    const semanticResult =
      payload &&
      payload.kind === "semantic_atom_sequence" &&
      typeof payload.status === "string";
    if (semanticResult) {
      const status = payload.status;
      const planned = Array.isArray(payload.plannedFunctions)
        ? payload.plannedFunctions.map(String)
        : [];
      const completed = Array.isArray(payload.completedFunctions)
        ? payload.completedFunctions.map(String)
        : [];
      const dropped = Array.isArray(payload.droppedFunctions)
        ? payload.droppedFunctions.map(String)
        : [];
      const forceFirstPolicyEnabled = payload.forceFirstPolicyEnabled === true;
      const forcedFirstAtom = payload.forcedFirstAtom === true;
      const plannedOverrunUs =
        Number.isSafeInteger(payload.plannedOverrunUs) && payload.plannedOverrunUs >= 0
          ? payload.plannedOverrunUs
          : 0;
      const mode = payload.simulated ? "simulation" : "real robot";
      const action = completed.length ? completed.join(" → ") : "no motion planned";
      const forcePrefix = forcedFirstAtom
        ? `Force +${(plannedOverrunUs / 1_000_000).toFixed(3)}s · `
        : "";

      // This latch means the current turn has a terminal robot decision, not
      // necessarily that an Atom was submitted.  It prevents later transcript
      // chunks from replacing an explicit no_plan result with generic waiting.
      currentTurnHasRobotAction = status !== "interrupted";
      setRobotMode(`Mode: ${mode}`);
      setRobotAction(`Action: ${forcePrefix}${action}`);
      setRobotResult(`Result: Semantic planner ${status}.`);
      const details = [
        `Planner status: ${status}`,
        `Force-first policy: ${forceFirstPolicyEnabled ? "enabled" : "disabled"}`,
        `Planned: ${planned.length ? planned.join(" → ") : "none"}`,
        `Completed: ${completed.length ? completed.join(" → ") : "none"}`,
        `Dropped: ${dropped.length ? dropped.join(" → ") : "none"}`,
      ];
      if (forcedFirstAtom) {
        details.push(
          `Forced overrun: +${(plannedOverrunUs / 1_000_000).toFixed(3)}s`
        );
      }
      if (payload.recoveryActionId) {
        details.push(`Recovery: ${payload.recoveryActionId}`);
      }
      setRobotDetail(`Detail: ${details.join(" | ")}`);
      return;
    }
    if (!lifecycle) {
      upstreamUpdateRobotPanel(payload);
      return;
    }

    const runtime = payload.robot || {};
    const mode = runtime.simulationMode
      ? "simulation"
      : runtime.enabled
        ? "real robot"
        : "disabled";
    const label = payload.label || payload.action || "robot action";
    const lifecycleLabels = {
      scheduled: "Robot scheduled",
      started: "Robot started",
      completed: "Robot completed",
      failed: "Robot failed",
      cancelled: "Robot cancelled",
    };
    const lifecycleLabel = lifecycleLabels[lifecycle] || `Robot ${lifecycle}`;

    currentTurnHasRobotAction = lifecycle !== "failed" && lifecycle !== "cancelled";
    setRobotMode(`Mode: ${mode}`);
    setRobotAction(`${lifecycleLabel}: ${label}`);
    setRobotResult(`Result: ${lifecycleLabel}.`);

    const details = [`Lifecycle: ${lifecycle}`];
    if (payload.actionRequestId) {
      details.push(`Request: ${payload.actionRequestId}`);
    }
    const status = payload.simulatorStatus || {};
    if (Number.isInteger(status.queueDepth)) {
      details.push(`Queue: ${status.queueDepth}`);
    }
    if (payload.error) {
      details.push(`Error: ${payload.error}`);
    }
    setRobotDetail(`Detail: ${details.join(" | ")}`);

    if (["scheduled", "started", "completed"].includes(lifecycle)) {
      pushFeed(`${lifecycleLabel}: ${label}`, lifecycle === "completed" ? "ok" : "");
      setStatus(`${lifecycleLabel}: ${label}.`, lifecycle === "completed" ? "ok" : "");
      // The untouched switch statement emits a second generic "simulated"
      // message whenever `executed` is true. Lifecycle-aware UI has already
      // rendered the precise state, so suppress only that duplicate branch.
      payload.executed = false;
    }
  };
})();
</script>
""".strip()


_CONTROL_SOCKET_GENERATION_MARKER = "mujoco-control-socket-generation"

_CONTROL_SOCKET_GENERATION_REPLACEMENTS = (
    (
        """    let controlSocket = null;
    let captureAudioContext = null;""",
        """    let controlSocket = null;
    let controlSocketGeneration = 0; // mujoco-control-socket-generation
    let captureAudioContext = null;""",
    ),
    (
        """    async function closeGemini(notifyBackend = true) {
      await stopMic(false);
      if (notifyBackend && controlSocket && controlSocket.readyState === WebSocket.OPEN) {
        controlSocket.send(JSON.stringify({ type: "close_session" }));
      }
      if (controlSocket) {
        controlSocket.onclose = null;
        controlSocket.close();
      }
      stopControlKeepalive();
      controlSocket = null;
      isGeminiConnected = false;
      resetRobotPanel();
      toggleButtons(Boolean(pc && pc.connectionState === "connected"));
    }""",
        """    function isCurrentControlSocket(socket, generation) {
      return controlSocket === socket && controlSocketGeneration === generation;
    }

    async function closeGemini(
      notifyBackend = true,
      expectedSocket = controlSocket,
      expectedGeneration = controlSocketGeneration,
      preserveGeneration = false
    ) {
      if (!isCurrentControlSocket(expectedSocket, expectedGeneration)) {
        return;
      }
      await stopMic(false);
      // stopMic is asynchronous. A newer start may have taken ownership while
      // it was pending, so revalidate before touching any socket or UI state.
      if (!isCurrentControlSocket(expectedSocket, expectedGeneration)) {
        return;
      }
      if (notifyBackend && expectedSocket && expectedSocket.readyState === WebSocket.OPEN) {
        expectedSocket.send(JSON.stringify({ type: "close_session" }));
      }
      if (expectedSocket) {
        expectedSocket.onclose = null;
        expectedSocket.close();
      }
      // WebSocket.close() may synchronously enter host code. Never let that
      // re-entrancy clear a replacement socket.
      if (!isCurrentControlSocket(expectedSocket, expectedGeneration)) {
        return;
      }
      stopControlKeepalive();
      controlSocket = null;
      if (!preserveGeneration) {
        controlSocketGeneration += 1;
      }
      isGeminiConnected = false;
      resetRobotPanel();
      toggleButtons(Boolean(pc && pc.connectionState === "connected"));
    }""",
    ),
    (
        """    async function startGeminiConversation() {
      if (!sessionId || !pc || !["new", "connecting", "connected"].includes(pc.connectionState)) {
        await startConnection();
      }
      resumePlayer();
      await prepareAvatarConfig();
      await closeGemini(false);
      clearPlayerNode();
      $("placeholder").style.display = "flex";
      await prepareMicAccess();
      pendingMicStart = true;
      setInputTranscript("");
      setOutputTranscript("");
      resetRobotPanel();
      setOutputPath("-");
      setSelectedAudio("gemini_live_reply");
      controlSocket = new WebSocket(serverWsUrl);
      pushFeed("Connecting to the Gemini control channel.");""",
        """    async function startGeminiConversation() {
      // Reserve ownership before the first await. Concurrent starts can finish
      // preparation out of order, but only the latest generation may create a
      // control WebSocket or mutate connection UI.
      const socketGeneration = ++controlSocketGeneration;
      const previousSocket = controlSocket;
      await closeGemini(false, previousSocket, socketGeneration, true);
      if (controlSocketGeneration !== socketGeneration) {
        return;
      }
      if (!sessionId || !pc || !["new", "connecting", "connected"].includes(pc.connectionState)) {
        await startConnection();
        if (controlSocketGeneration !== socketGeneration) {
          return;
        }
      }
      resumePlayer();
      await prepareAvatarConfig();
      if (controlSocketGeneration !== socketGeneration) {
        return;
      }
      clearPlayerNode();
      $("placeholder").style.display = "flex";
      await prepareMicAccess();
      if (controlSocketGeneration !== socketGeneration) {
        return;
      }
      pendingMicStart = true;
      setInputTranscript("");
      setOutputTranscript("");
      resetRobotPanel();
      setOutputPath("-");
      setSelectedAudio("gemini_live_reply");
      const socket = new WebSocket(serverWsUrl);
      if (controlSocketGeneration !== socketGeneration) {
        socket.close();
        return;
      }
      controlSocket = socket;
      pushFeed("Connecting to the Gemini control channel.");""",
    ),
    (
        """        controlSocket.onopen = () => {
          pushFeed("Gemini control channel connected.", "ok");
          startControlKeepalive();
          controlSocket.send(JSON.stringify({
            type: "start_session",
            session_id: sessionId,
            api_key: $("api-key").value.trim(),
            model: $("gemini-model").value.trim(),
            voice_name: $("gemini-voice").value,
            system_instruction: $("system-prompt").value.trim(),
            temperature: $("gemini-temperature").value.trim(),
          }));
        };""",
        """        socket.onopen = () => {
          if (!isCurrentControlSocket(socket, socketGeneration)) {
            return;
          }
          pushFeed("Gemini control channel connected.", "ok");
          startControlKeepalive();
          socket.send(JSON.stringify({
            type: "start_session",
            session_id: sessionId,
            api_key: $("api-key").value.trim(),
            model: $("gemini-model").value.trim(),
            voice_name: $("gemini-voice").value,
            system_instruction: $("system-prompt").value.trim(),
            temperature: $("gemini-temperature").value.trim(),
          }));
        };""",
    ),
    (
        """      controlSocket.onmessage = async (event) => {
        const payload = JSON.parse(event.data);""",
        """      socket.onmessage = async (event) => {
        if (!isCurrentControlSocket(socket, socketGeneration)) {
          return;
        }
        const payload = JSON.parse(event.data);""",
    ),
    (
        """          case "session_closed":
            pushFeed(payload.message || "Gemini session closed.");
            await closeGemini(false);
            break;""",
        """          case "session_closed":
            pushFeed(payload.message || "Gemini session closed.");
            await closeGemini(false, socket, socketGeneration);
            break;""",
    ),
    (
        """      controlSocket.onerror = () => {
        pushFeed("Gemini control channel error.", "warn");
        setStatus("Gemini control channel error.", "warn");
      };""",
        """      socket.onerror = () => {
        if (!isCurrentControlSocket(socket, socketGeneration)) {
          return;
        }
        pushFeed("Gemini control channel error.", "warn");
        setStatus("Gemini control channel error.", "warn");
      };""",
    ),
    (
        """      controlSocket.onclose = async () => {
        pushFeed("Gemini control channel closed.");
        await closeGemini(false);
      };""",
        """      socket.onclose = async () => {
        if (!isCurrentControlSocket(socket, socketGeneration)) {
          return;
        }
        pushFeed("Gemini control channel closed.");
        await closeGemini(false, socket, socketGeneration);
      };""",
    ),
)


def _patch_control_socket_generation(html: str) -> tuple[str, bool]:
    """Install an exact, fail-closed ownership guard in the upstream classic script."""

    if _CONTROL_SOCKET_GENERATION_MARKER in html:
        for _old, new in _CONTROL_SOCKET_GENERATION_REPLACEMENTS:
            if html.count(new) != 1:
                raise RuntimeError(
                    "installed control WebSocket generation guard is incomplete"
                )
        return html, False

    control_socket_signatures = (
        "let controlSocket",
        "async function startGeminiConversation",
        "new WebSocket(",
        'case "session_closed"',
    )
    if not all(signature in html for signature in control_socket_signatures):
        return html, False

    for old, _new in _CONTROL_SOCKET_GENERATION_REPLACEMENTS:
        if html.count(old) != 1:
            raise RuntimeError(
                "cannot install control WebSocket generation guard: "
                "upstream classic-script anchor drifted"
            )

    patched = html
    for old, new in _CONTROL_SOCKET_GENERATION_REPLACEMENTS:
        patched = patched.replace(old, new, 1)
    for _old, new in _CONTROL_SOCKET_GENERATION_REPLACEMENTS:
        if patched.count(new) != 1:
            raise RuntimeError(
                "control WebSocket generation guard postcondition failed"
            )
    return patched, True


def _inject_overlay_html(html: str) -> tuple[str, bool]:
    changed = False
    html, control_socket_changed = _patch_control_socket_generation(html)
    changed = changed or control_socket_changed
    lower = html.lower()
    if "mujoco-overlay-widescreen" not in html:
        if "</head>" in lower:
            index = lower.index("</head>")
            html = html[:index] + WIDESCREEN_CSS + "\n" + html[index:]
        else:
            html = WIDESCREEN_CSS + "\n" + html
        changed = True

    if "mujoco-action-lifecycle-ui" not in html:
        lower = html.lower()
        if "</body>" in lower:
            index = lower.index("</body>")
            html = html[:index] + ACTION_LIFECYCLE_SCRIPT + "\n" + html[index:]
        else:
            html = html + "\n" + ACTION_LIFECYCLE_SCRIPT
        changed = True
    if _EXPERIMENT_RECORDING_AVAILABLE and callable(inject_experiment_recording_ui):
        try:
            html, recording_changed = inject_experiment_recording_ui(html)
            changed = changed or recording_changed
        except Exception:
            # A recorder-only page patch failure cannot break conversation UI.
            pass
    return html, changed


def _patch_html_globals(module: ModuleType) -> list[str]:
    patched: list[str] = []
    for name, value in list(vars(module).items()):
        if not isinstance(value, str) or "<html" not in value.lower():
            continue
        value, changed = _inject_overlay_html(value)
        if changed:
            setattr(module, name, value)
            patched.append(f"{module.__name__}.{name}")
    return patched


def _bind_stream_session(cls: type[Any], client: SimServiceClient) -> bool:
    if getattr(cls, "_mujoco_overlay_bound", False):
        return False
    cls._mujoco_overlay_bound = True
    cls._mujoco_overlay_client = client
    return True


def _add_aiohttp_routes(
    app: Any,
    client: SimServiceClient,
    report_provider: Callable[[], dict[str, Any]],
) -> list[str]:
    router = getattr(app, "router", None)
    if router is None:
        return []
    try:
        from aiohttp import web
    except ImportError:
        return []

    @web.middleware
    async def widescreen_html(request: Any, handler: Callable[..., Any]):
        response = await handler(request)
        content_type = getattr(response, "content_type", "")
        body = getattr(response, "body", None)
        if content_type != "text/html" or not isinstance(body, bytes):
            return response
        try:
            html = body.decode(getattr(response, "charset", None) or "utf-8")
        except UnicodeDecodeError:
            return response
        html, changed = _inject_overlay_html(html)
        if changed:
            response.body = html.encode(
                getattr(response, "charset", None) or "utf-8"
            )
        return response

    async def overlay_health(_: Any):
        try:
            sim = await asyncio.to_thread(client.health_sync)
            ok = bool(sim.get("ok", sim.get("status") in ("ok", "healthy")))
        except Exception as exc:
            sim = {"ok": False, "error": str(exc)}
            ok = False
        body = {
            "ok": ok,
            "component": "flashhead-mujoco-overlay",
            "sim": sim,
            "cache": client.cache_status(),
        }
        return web.json_response(body, status=200 if ok else 503)

    async def overlay_status(_: Any):
        try:
            sim = await asyncio.to_thread(client.state_sync)
        except Exception as exc:
            sim = {"available": False, "error": str(exc)}
        return web.json_response(
            {
                "component": "flashhead-mujoco-overlay",
                "sim": sim,
                "cache": client.cache_status(),
                "patches": report_provider(),
            }
        )

    added: list[str] = []
    middlewares = getattr(app, "middlewares", None)
    if middlewares is not None:
        try:
            middlewares.append(widescreen_html)
            added.append("middleware:widescreen-css")
        except (AttributeError, RuntimeError):
            pass
    for path, handler in (
        ("/overlay/health", overlay_health),
        ("/overlay/status", overlay_status),
        ("/sim/status", overlay_status),
    ):
        try:
            router.add_get(path, handler)
        except RuntimeError:
            continue
        added.append(path)
    return added


def install_runtime_patches(
    module: ModuleType,
    client: SimServiceClient,
    config: OverlayConfig,
) -> PatchReport:
    """Patch the imported demo in place; no upstream files are copied or edited."""

    global _EXPERIMENT_RECORDING_AVAILABLE
    report = PatchReport([], [], [], [], [], [], [])
    recording_markers: list[str] = []
    if callable(install_experiment_recording):
        try:
            recording_markers = install_experiment_recording(module)
        except Exception as exc:
            logger = getattr(module, "logger", None)
            if logger is not None:
                try:
                    logger.warning(
                        "Experiment recording disabled after optional patch failure: {}",
                        type(exc).__name__,
                    )
                except Exception:
                    pass
    _EXPERIMENT_RECORDING_AVAILABLE = bool(recording_markers)
    if recording_markers:
        report.action_symbols.extend(recording_markers)
        report.aiohttp_routes.extend(
            (
                EXPERIMENT_RECORDING_START_PATH,
                EXPERIMENT_RECORDING_STOP_PATH,
                EXPERIMENT_RECORDING_STATUS_PATH,
            )
        )
    namespaces = list(_module_namespaces(module))
    sequence_mode = config.dialogue_sequence_mode.strip().lower()
    if sequence_mode not in DIALOGUE_SEQUENCE_MODES:
        raise ValueError(
            "OVERLAY_DIALOGUE_SEQUENCE_MODE must be semantic_f17, "
            "random_catalog, or legacy_abc"
        )
    coordinator_factory = (
        _random_sequence_coordinator_factory(client, config)
        if sequence_mode == "random_catalog"
        else None
    )
    semantic_factory = None
    semantic_bridge_patcher = None
    if sequence_mode == "semantic_f17":
        semantic_factory, semantic_bridge_patcher = _semantic_sequence_runtime_setup(
            module,
            client,
            config,
        )

    # The normal app constructs peer connections only after startup, so wrapping
    # addTrack here gives the video a true independent clock without touching the
    # audio QueueStreamTrack.  Use the QueueStreamTrack adapter only when an
    # upstream does not expose its PeerConnection class.
    patched_pc = _patch_peer_connection_class(module, client, config)
    if patched_pc:
        report.peer_connection_classes.append(patched_pc)

    for prefix, namespace in namespaces:
        track_cls = getattr(namespace, "QueueStreamTrack", None)
        if not report.peer_connection_classes and isinstance(
            track_cls, type
        ) and _patch_queue_track_class(
            track_cls, client, config
        ):
            report.queue_track_classes.append(f"{prefix}.QueueStreamTrack")

        session_cls = getattr(namespace, "StreamSession", None)
        if isinstance(session_cls, type) and _bind_stream_session(session_cls, client):
            report.stream_session_classes.append(f"{prefix}.StreamSession")

        bridge_cls = getattr(namespace, "GeminiLiveBridge", None)
        if isinstance(bridge_cls, type) and _patch_gemini_bridge_class(
            bridge_cls, client
        ):
            report.action_symbols.append(
                f"{prefix}.GeminiLiveBridge._trigger_robot_action"
            )
        if (
            isinstance(bridge_cls, type)
            and coordinator_factory is not None
            and patch_random_dialogue_bridge_class(
                bridge_cls, coordinator_factory=coordinator_factory
            )
        ):
            report.action_symbols.append(
                f"{prefix}.GeminiLiveBridge.random_catalog_sequence"
            )
        if (
            isinstance(bridge_cls, type)
            and semantic_factory is not None
            and semantic_bridge_patcher is not None
            and semantic_bridge_patcher(
                bridge_cls,
                module=module,
                controller_factory=semantic_factory,
            )
        ):
            report.action_symbols.append(
                f"{prefix}.GeminiLiveBridge.semantic_f17_sequence"
            )

        if hasattr(namespace, "trigger_robot_action"):
            original = getattr(namespace, "trigger_robot_action")
            setattr(
                namespace,
                "trigger_robot_action",
                _make_action_replacement(original if callable(original) else None, client),
            )
            report.action_symbols.append(f"{prefix}.trigger_robot_action")

        if hasattr(namespace, "robot_runtime"):
            setattr(namespace, "robot_runtime", SimRobotRuntime(client))
            report.robot_runtime_symbols.append(f"{prefix}.robot_runtime")

    # Install missing globals too: handlers in the main module commonly resolve
    # these names lazily when a response arrives.
    if not hasattr(module, "trigger_robot_action"):
        setattr(module, "trigger_robot_action", _make_action_replacement(None, client))
        report.action_symbols.append(f"{module.__name__}.trigger_robot_action")
    if not hasattr(module, "robot_runtime"):
        setattr(module, "robot_runtime", SimRobotRuntime(client))
        report.robot_runtime_symbols.append(f"{module.__name__}.robot_runtime")

    report.html_globals.extend(_patch_html_globals(module))

    def report_provider() -> dict[str, Any]:
        return report.as_dict()

    for app_name in ("app", "web_app", "application"):
        app = getattr(module, app_name, None)
        if app is not None:
            report.aiohttp_routes.extend(
                _add_aiohttp_routes(app, client, report_provider)
            )
            if report.aiohttp_routes:
                break
    return report
