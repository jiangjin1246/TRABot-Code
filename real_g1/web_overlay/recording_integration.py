from __future__ import annotations

import asyncio
import functools
import inspect
import time
import threading
from types import ModuleType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
import weakref

import numpy as np

from .recording import (
    RecordingConfig,
    RecordingSlotManager,
    SessionRecordingController,
    STATUS_TYPE,
)
from .reply_trace import ReplyTraceHub, install_submission_receipt_observer


START_PATH = "/api/experiment-recording/start"
STOP_PATH = "/api/experiment-recording/stop"
STATUS_PATH = "/api/experiment-recording/status/{session_id}"
_REQUEST_HEADER = "X-Experiment-Recording"
_MAX_JSON_BYTES = 4096
_MEDIA_PUBLISH_CONTEXT = threading.local()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class _ControllerRegistry:
    def __init__(self, config: RecordingConfig) -> None:
        self.config = config
        self.slot_manager = RecordingSlotManager(config.max_active_recordings)
        self._controllers: weakref.WeakSet[SessionRecordingController] = weakref.WeakSet()
        self._hubs: weakref.WeakSet[ReplyTraceHub] = weakref.WeakSet()
        self._semantic_lock = threading.Lock()
        self._pending_semantic: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._pending_semantic_count = 0

    def attach(self, stream: Any) -> SessionRecordingController:
        existing = getattr(stream, "_experiment_recording_controller", None)
        if isinstance(existing, SessionRecordingController):
            return existing
        session_id = str(getattr(stream, "session_id", ""))
        hub = getattr(stream, "_experiment_reply_trace_hub", None)
        if not isinstance(hub, ReplyTraceHub):
            hub = ReplyTraceHub(session_id)
            stream._experiment_reply_trace_hub = hub
        controller = SessionRecordingController(
            self.config,
            session_id=session_id,
            slot_manager=self.slot_manager,
            reply_hub=hub,
        )
        stream._experiment_recording_controller = controller
        pc = getattr(stream, "pc", None)
        if pc is not None:
            pc._experiment_recording_controller = controller
            pc._experiment_reply_trace_hub = hub
        for track_name in ("video_track", "audio_track"):
            track = getattr(stream, track_name, None)
            if track is not None:
                track._experiment_reply_trace_hub = hub
        self._controllers.add(controller)
        self._hubs.add(hub)
        hub.set_semantic_binding_sink(self._semantic_bound)
        return controller

    @staticmethod
    def _deliver_semantic_action(
        hub: ReplyTraceHub,
        request_id: str,
        revision: int,
        event: Mapping[str, Any],
    ) -> None:
        kind = str(event.get("kind", ""))
        role = "recovery" if kind == "recovery" else "semantic"
        record = event.get("action_record")
        if not isinstance(record, Mapping):
            return
        hub.observe_body_for_semantic(
            request_id,
            revision,
            record,
            role,
            ordinal=event.get("ordinal"),
            function=event.get("function"),
            atom_ref=event.get("atom_ref"),
            terminal_status=event.get("terminal_status"),
        )

    def _semantic_bound(
        self, hub: ReplyTraceHub, request_id: str, revision: int
    ) -> None:
        binding = (request_id, revision)
        with self._semantic_lock:
            pending = self._pending_semantic.pop(binding, [])
            self._pending_semantic_count -= len(pending)
        for event in pending:
            try:
                self._deliver_semantic_action(
                    hub, request_id, revision, event
                )
            except Exception:
                pass

    def semantic_observer(
        self, runtime: Any, request: Any
    ) -> Callable[..., None] | None:
        del runtime
        request_id = str(getattr(request, "request_id", ""))
        revision = getattr(request, "revision", 0)
        if (
            not request_id or not isinstance(revision, int)
            or isinstance(revision, bool) or revision < 1
        ):
            return None

        def observe(**event: Any) -> None:
            candidates = [
                hub for hub in tuple(self._hubs)
                if hub.reply_for_semantic(request_id, revision) is not None
            ]
            if len(candidates) == 1:
                self._deliver_semantic_action(
                    candidates[0], request_id, revision, event
                )
                return
            if candidates:
                return
            snapshot = dict(event)
            with self._semantic_lock:
                if self._pending_semantic_count >= 256:
                    return
                self._pending_semantic.setdefault(
                    (request_id, revision), []
                ).append(snapshot)
                self._pending_semantic_count += 1

        return observe

    def wait_all(self, timeout: float) -> bool:
        return self.slot_manager.wait_all(timeout)

    def stop_all(self) -> None:
        for controller in tuple(self._controllers):
            try:
                controller.close_stream()
            except Exception:
                pass
        self.slot_manager.stop_all()

    def abort_all(self, *, force: bool = False) -> None:
        self.slot_manager.abort_all(force=force)


def _controller_for_stream(stream: Any) -> SessionRecordingController | None:
    value = getattr(stream, "_experiment_recording_controller", None)
    return value if isinstance(value, SessionRecordingController) else None


def _install_stream_lifecycle(module: ModuleType, registry: _ControllerRegistry) -> str:
    cls = getattr(module, "StreamSession", None)
    if not isinstance(cls, type):
        raise RuntimeError("experiment recorder requires StreamSession")
    if getattr(cls, "_experiment_recording_lifecycle_patched", False):
        return f"{module.__name__}.StreamSession.experiment_recording"

    original_init = getattr(cls, "__init__", None)
    original_start_generation = getattr(cls, "start_live_generation", None)
    original_close = getattr(cls, "close", None)
    original_launch = getattr(cls, "_launch_live_generation_worker", None)
    original_align = getattr(cls, "_align_frames_to_audio", None)
    original_publish = getattr(cls, "_publish_chunk_media", None)
    original_enqueue = getattr(cls, "_enqueue_from_thread_blocking", None)
    original_publish_audio = getattr(cls, "_publish_audio_packet", None)
    original_publish_video = getattr(cls, "_publish_video_frame", None)
    if not all(callable(value) for value in (original_init, original_start_generation, original_close)):
        raise RuntimeError("StreamSession recording lifecycle contract is incompatible")

    @functools.wraps(original_init)
    def init_with_recorder(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            registry.attach(self)
        except Exception:
            # Recorder setup must never prevent a WebRTC session from existing.
            self._experiment_recording_controller = None

    @functools.wraps(original_start_generation)
    def start_generation_with_recorder(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_start_generation(self, *args, **kwargs)
        try:
            controller = _controller_for_stream(self)
            if controller is not None:
                controller.bind_generation(int(getattr(self, "request_id", 0)))
        except Exception:
            pass
        return result

    media_contract = all(
        callable(value)
        for value in (original_launch, original_align, original_publish, original_enqueue)
    )
    if media_contract:
        @functools.wraps(original_launch)
        def launch_with_provenance(
            self: Any, request_id: int, stream_epoch: int
        ) -> Any:
            try:
                hub = getattr(self, "_experiment_reply_trace_hub", None)
                if isinstance(hub, ReplyTraceHub):
                    generation = int(request_id)
                    hub.bind_generation(generation)
                    self.reply_queue = hub.tagged_queue(self.reply_queue)
                    controller = _controller_for_stream(self)
                    if controller is not None:
                        controller.bind_generation(generation)
            except Exception:
                pass
            return original_launch(self, request_id, stream_epoch)

        @functools.wraps(original_align)
        def align_with_provenance(*args: Any, **kwargs: Any) -> Any:
            result = original_align(*args, **kwargs)
            try:
                context = ReplyTraceHub.producer_context()
                if context is not None:
                    hub, key = context
                    hub.remember_generated_array(result, key)
            except Exception:
                pass
            return result

        @functools.wraps(original_publish)
        def publish_with_provenance(
            self: Any, frames: Any, *args: Any, **kwargs: Any
        ) -> Any:
            try:
                bound = inspect.signature(original_publish).bind_partial(
                    self, frames, *args, **kwargs
                )
                flush_audio = bound.arguments.get("flush_audio") is True
            except Exception:
                flush_audio = kwargs.get("flush_audio") is True
            try:
                hub = getattr(self, "_experiment_reply_trace_hub", None)
                key = (
                    hub.key_for_generated_array(frames)
                    if isinstance(hub, ReplyTraceHub) else None
                )
            except Exception:
                hub = None
                key = None
            previous = getattr(_MEDIA_PUBLISH_CONTEXT, "value", None)
            publish_state = None
            if isinstance(hub, ReplyTraceHub) and key is not None:
                publish_state = {
                    "hub": hub,
                    "key": key,
                    "owner": self,
                    "enqueue_attempts": 0,
                    "enqueue_failed": False,
                }
                _MEDIA_PUBLISH_CONTEXT.value = publish_state
            try:
                if isinstance(hub, ReplyTraceHub) and key is not None:
                    with hub.media_scope(key):
                        result = original_publish(self, frames, *args, **kwargs)
                else:
                    result = original_publish(self, frames, *args, **kwargs)
            except BaseException:
                if publish_state is not None:
                    publish_state["enqueue_failed"] = True
                raise
            finally:
                _MEDIA_PUBLISH_CONTEXT.value = previous
            stop_event = None
            try:
                stop_event = bound.arguments.get("stop_event")
            except Exception:
                stop_event = kwargs.get("stop_event")
            stopped = bool(
                getattr(stop_event, "is_set", lambda: False)()
            ) if stop_event is not None else False
            if (
                flush_audio
                and publish_state is not None
                and publish_state["enqueue_attempts"] > 0
                and not publish_state["enqueue_failed"]
                and not stopped
                and not bool(getattr(self, "closed", False))
                and not inspect.isawaitable(result)
                and result is not False
            ):
                try:
                    hub.mark_media_producer_done(
                        key, observed_ns=time.monotonic_ns()
                    )
                except Exception:
                    pass
            return result

        def _record_publish_result(self: Any, result: Any) -> None:
            publish_state = getattr(_MEDIA_PUBLISH_CONTEXT, "value", None)
            if (
                isinstance(publish_state, dict)
                and publish_state.get("owner") is self
            ):
                publish_state["enqueue_attempts"] += 1
                if result is not True:
                    publish_state["enqueue_failed"] = True

        if callable(original_publish_audio):
            @functools.wraps(original_publish_audio)
            def publish_audio_with_provenance(
                self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                try:
                    result = original_publish_audio(self, *args, **kwargs)
                except BaseException:
                    _record_publish_result(self, False)
                    raise
                _record_publish_result(self, result)
                return result

        if callable(original_publish_video):
            @functools.wraps(original_publish_video)
            def publish_video_with_provenance(
                self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                try:
                    result = original_publish_video(self, *args, **kwargs)
                except BaseException:
                    _record_publish_result(self, False)
                    raise
                _record_publish_result(self, result)
                return result

        @functools.wraps(original_enqueue)
        def enqueue_with_provenance(
            self: Any, track: Any, frame: Any, *args: Any, **kwargs: Any
        ) -> Any:
            tagged_hub = None
            key = None
            kind = ""
            media_seq = None
            tagged = False
            publish_state = getattr(_MEDIA_PUBLISH_CONTEXT, "value", None)
            try:
                if (
                    isinstance(publish_state, dict)
                    and publish_state.get("owner") is self
                    and isinstance(publish_state.get("hub"), ReplyTraceHub)
                ):
                    tagged_hub = publish_state["hub"]
                    key = publish_state.get("key")
                    kind = str(getattr(track, "kind", ""))
                    if kind in {"audio", "video"} and key is not None:
                        media_seq = tagged_hub.next_media_seq(
                            key, media_kind=kind
                        )
                        tagged = (
                            isinstance(media_seq, int)
                            and tagged_hub.tag_frame(
                                frame,
                                key,
                                media_kind=kind,
                                media_seq=media_seq,
                            ) is not None
                        )
                        if not tagged:
                            publish_state["enqueue_failed"] = True
            except Exception:
                tagged_hub = None
                tagged = False
                if isinstance(publish_state, dict):
                    publish_state["enqueue_failed"] = True
            try:
                result = original_enqueue(self, track, frame, *args, **kwargs)
            except BaseException:
                if isinstance(publish_state, dict):
                    publish_state["enqueue_failed"] = True
                if tagged and tagged_hub is not None:
                    try:
                        tagged_hub.discard_frame(frame)
                    except Exception:
                        pass
                raise
            if not result:
                if isinstance(publish_state, dict):
                    publish_state["enqueue_failed"] = True
                if tagged and tagged_hub is not None:
                    try:
                        tagged_hub.discard_frame(frame)
                    except Exception:
                        pass
            elif (
                tagged
                and tagged_hub is not None
                and key is not None
                and isinstance(media_seq, int)
            ):
                try:
                    if not tagged_hub.note_media_enqueued(
                        key, media_kind=kind, media_seq=media_seq
                    ):
                        if isinstance(publish_state, dict):
                            publish_state["enqueue_failed"] = True
                except Exception:
                    if isinstance(publish_state, dict):
                        publish_state["enqueue_failed"] = True
            return result

    @functools.wraps(original_close)
    async def close_with_recorder(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            controller = _controller_for_stream(self)
            if controller is not None:
                controller.close_stream()
        except Exception:
            pass
        return await _maybe_await(original_close(self, *args, **kwargs))

    cls.__init__ = init_with_recorder
    cls.start_live_generation = start_generation_with_recorder
    cls.close = close_with_recorder
    if media_contract:
        cls._launch_live_generation_worker = launch_with_provenance
        cls._align_frames_to_audio = staticmethod(align_with_provenance)
        cls._publish_chunk_media = publish_with_provenance
        cls._enqueue_from_thread_blocking = enqueue_with_provenance
        if callable(original_publish_audio):
            cls._publish_audio_packet = publish_audio_with_provenance
        if callable(original_publish_video):
            cls._publish_video_frame = publish_video_with_provenance
        cls._experiment_reply_media_provenance_patched = True
    cls._experiment_recording_lifecycle_patched = True
    cls._experiment_recording_original_close = original_close
    return f"{module.__name__}.StreamSession.experiment_recording"


class _RecordingSdkSessionProxy:
    def __init__(self, session: Any, bridge: Any) -> None:
        self._session = session
        # The upstream start path increments StreamSession.request_id before it
        # creates the send-loop task, while the outer bridge wrapper publishes
        # its convenience attribute only after start_session returns. Reading
        # the stream first therefore binds reused bridges to the correct task.
        try:
            generation = int(getattr(bridge.stream_session, "request_id", 0))
        except Exception:
            generation = None
        if not isinstance(generation, int) or generation <= 0:
            generation = getattr(bridge, "_experiment_recording_generation", None)
        self._generation = generation
        try:
            self._bridge_ref = weakref.ref(bridge)
        except TypeError:
            self._bridge_ref = lambda: bridge

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def send_realtime_input(self, *args: Any, **kwargs: Any) -> Any:
        controller = None
        offer = None
        audio = kwargs.get("audio")
        if audio is not None:
            try:
                bridge = self._bridge_ref()
                if bridge is not None:
                    controller = _controller_for_stream(
                        getattr(bridge, "stream_session", None)
                    )
                    if controller is not None:
                        offer = controller.active_offer()
            except Exception:
                controller = None
                offer = None
        result = await self._session.send_realtime_input(*args, **kwargs)
        try:
            if audio is not None and controller is not None and offer is not None:
                data = getattr(audio, "data", None)
                if isinstance(self._generation, int) and data:
                    controller.offer_user_pcm16_for(
                        offer,
                        data,
                        generation=self._generation,
                        captured_ns=time.monotonic_ns(),
                    )
        except Exception:
            pass
        return result


def _install_bridge_lifecycle(module: ModuleType) -> str:
    cls = getattr(module, "GeminiLiveBridge", None)
    if not isinstance(cls, type):
        raise RuntimeError("experiment recorder requires GeminiLiveBridge")
    if getattr(cls, "_experiment_recording_bridge_patched", False):
        return f"{module.__name__}.GeminiLiveBridge.experiment_recording"

    original_start = getattr(cls, "start_session", None)
    original_send_audio = getattr(cls, "_send_audio_loop", None)
    original_close = getattr(cls, "close", None)
    if not all(callable(value) for value in (original_start, original_send_audio, original_close)):
        raise RuntimeError("Gemini bridge recording contract is incompatible")

    @functools.wraps(original_start)
    async def start_with_recorder(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await _maybe_await(original_start(self, *args, **kwargs))
        try:
            generation = int(
                getattr(getattr(self, "stream_session", None), "request_id", 0)
            )
            self._experiment_recording_generation = generation
            controller = _controller_for_stream(getattr(self, "stream_session", None))
            if controller is not None:
                controller.bind_generation(generation)
        except Exception:
            pass
        return result

    @functools.wraps(original_send_audio)
    async def send_audio_with_recorder(self: Any, session: Any) -> Any:
        try:
            session_for_send = _RecordingSdkSessionProxy(session, self)
        except Exception:
            # Recorder-side construction is optional. Never prevent the
            # original Gemini loop from receiving its real SDK session.
            session_for_send = session
        return await _maybe_await(
            original_send_audio(self, session_for_send)
        )

    @functools.wraps(original_close)
    async def close_with_recorder(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            generation = getattr(self, "_experiment_recording_generation", None)
        except Exception:
            generation = None
        try:
            return await _maybe_await(original_close(self, *args, **kwargs))
        finally:
            try:
                controller = _controller_for_stream(
                    getattr(self, "stream_session", None)
                )
                if controller is not None and isinstance(generation, int):
                    controller.stop_generation(
                        generation,
                        reason="gemini_generation_closed",
                    )
            except Exception:
                pass

    cls.start_session = start_with_recorder
    cls._send_audio_loop = send_audio_with_recorder
    cls.close = close_with_recorder
    cls._experiment_recording_bridge_patched = True
    cls._experiment_recording_original_start = original_start
    cls._experiment_recording_original_close = original_close
    return f"{module.__name__}.GeminiLiveBridge.experiment_recording"


def _media_time_ns(frame: Any) -> int | None:
    pts = getattr(frame, "pts", None)
    base = getattr(frame, "time_base", None)
    numerator = getattr(base, "numerator", None)
    denominator = getattr(base, "denominator", None)
    if pts is None or numerator is None or not denominator:
        return None
    try:
        return round(int(pts) * int(numerator) * 1e9 / int(denominator))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _frame_timing(frame: Any) -> tuple[int | None, int | None, int | None]:
    try:
        pts = getattr(frame, "pts", None)
        base = getattr(frame, "time_base", None)
        numerator = getattr(base, "numerator", None)
        denominator = getattr(base, "denominator", None)
        return (
            int(pts) if pts is not None else None,
            int(numerator) if numerator is not None else None,
            int(denominator) if denominator else None,
        )
    except (TypeError, ValueError, OverflowError):
        return None, None, None


def _install_peer_connection_taps(module: ModuleType) -> str:
    namespace_cls = getattr(module, "RTCPeerConnection", None)
    base_track = getattr(module, "MediaStreamTrack", None)
    if not isinstance(namespace_cls, type) or not isinstance(base_track, type):
        raise RuntimeError("experiment recorder requires aiortc media classes")
    cls = namespace_cls
    # If an earlier optional install failed and the compositor subsequently
    # replaced the namespace class, patch the first real addTrack base. Then
    # OverlayPeerConnection composes video first and super().addTrack applies
    # the recorder tap outside it, preserving final-composite capture order.
    if namespace_cls.__dict__.get("_mujoco_overlay_patched", False):
        for candidate in namespace_cls.__mro__[1:]:
            if callable(candidate.__dict__.get("addTrack")):
                cls = candidate
                break
    if getattr(cls, "_experiment_recording_tracks_patched", False):
        return f"{module.__name__}.RTCPeerConnection.experiment_recording"
    original_add_track = getattr(cls, "addTrack", None)
    if not callable(original_add_track):
        raise RuntimeError("RTCPeerConnection.addTrack is unavailable")

    class _RecordingTapTrack(base_track):
        kind = ""

        def __init__(self, source: Any, pc: Any) -> None:
            super().__init__()
            self.source = source
            try:
                self._pc_ref = weakref.ref(pc)
            except TypeError:
                self._pc_ref = lambda: pc
            self._tap_stopped = False

        def _controller(self) -> SessionRecordingController | None:
            pc = self._pc_ref()
            value = getattr(pc, "_experiment_recording_controller", None)
            return value if isinstance(value, SessionRecordingController) else None

        def _reply_hub(self) -> ReplyTraceHub | None:
            pc = self._pc_ref()
            value = getattr(pc, "_experiment_reply_trace_hub", None)
            return value if isinstance(value, ReplyTraceHub) else None

        def stop(self) -> None:
            if self._tap_stopped:
                return
            self._tap_stopped = True
            try:
                stop = getattr(self.source, "stop", None)
                if callable(stop):
                    stop()
            finally:
                super().stop()

    class RecordingVideoTapTrack(_RecordingTapTrack):
        kind = "video"

        async def recv(self) -> Any:
            frame = await self.source.recv()
            captured_ns = time.monotonic_ns()
            provenance = None
            try:
                hub = self._reply_hub()
                if hub is not None:
                    provenance = hub.take_frame(frame)
            except Exception:
                provenance = None
            controller = None
            offer = None
            try:
                controller = self._controller()
                offer = controller.active_offer() if controller is not None else None
                if controller is not None and offer is not None:
                    bgr = frame.to_ndarray(format="bgr24")
                    if provenance is None:
                        controller.offer_video_for(
                            offer, bgr, captured_ns=captured_ns
                        )
                    else:
                        pts, time_base_num, time_base_den = _frame_timing(frame)
                        controller.offer_video_for(
                            offer,
                            bgr,
                            captured_ns=captured_ns,
                            reply_key=provenance.reply_key,
                            face_source_seq=provenance.face_source_seq,
                            pts=pts,
                            time_base_num=time_base_num,
                            time_base_den=time_base_den,
                            media_seq=provenance.media_seq,
                        )
            except Exception:
                # Returning the original frame is the non-negotiable path.
                if controller is not None and offer is not None:
                    try:
                        controller.fail_offer(offer, "video_tap_invalid")
                    except Exception:
                        pass
            return frame

    class RecordingAssistantAudioTapTrack(_RecordingTapTrack):
        kind = "audio"

        async def recv(self) -> Any:
            controller_before = None
            generation_before = None
            try:
                controller_before = self._controller()
                generation_before = (
                    controller_before.current_generation()
                    if controller_before is not None
                    else None
                )
            except Exception:
                if controller_before is not None:
                    try:
                        controller_before.fail_active(
                            "assistant_audio_tap_invalid"
                        )
                    except Exception:
                        pass
                controller_before = None
                generation_before = None
            frame = await self.source.recv()
            captured_ns = time.monotonic_ns()
            provenance = None
            try:
                hub = self._reply_hub()
                if hub is not None:
                    provenance = hub.take_frame(frame)
            except Exception:
                provenance = None
            controller = None
            offer = None
            try:
                controller = self._controller()
                offer = controller.active_offer() if controller is not None else None
                if (
                    controller is not None
                    and controller is controller_before
                    and generation_before is not None
                    and controller.current_generation() == generation_before
                    and offer is not None
                    and offer[2] == generation_before
                ):
                    media_ns = _media_time_ns(frame)
                    if media_ns is None:
                        raise ValueError("assistant frame has no PTS")
                    if int(getattr(frame, "sample_rate", 0)) != 16_000:
                        raise ValueError("assistant frame sample rate changed")
                    format_name = str(
                        getattr(getattr(frame, "format", None), "name", "")
                    )
                    layout_name = str(
                        getattr(getattr(frame, "layout", None), "name", "")
                    )
                    if format_name != "s16" or layout_name != "mono":
                        raise ValueError("assistant frame PCM format changed")
                    array = frame.to_ndarray()
                    pcm = np.asarray(array, dtype="<i2").reshape(-1).tobytes()
                    if provenance is None:
                        controller.offer_assistant_pcm16_for(
                            offer,
                            pcm,
                            generation=generation_before,
                            captured_ns=captured_ns,
                            media_time_ns=media_ns,
                        )
                    else:
                        pts, time_base_num, time_base_den = _frame_timing(frame)
                        controller.offer_assistant_pcm16_for(
                            offer,
                            pcm,
                            generation=generation_before,
                            captured_ns=captured_ns,
                            media_time_ns=media_ns,
                            reply_key=provenance.reply_key,
                            pts=pts,
                            time_base_num=time_base_num,
                            time_base_den=time_base_den,
                            media_seq=provenance.media_seq,
                        )
            except Exception:
                if controller is not None and offer is not None:
                    try:
                        controller.fail_offer(
                            offer,
                            "assistant_audio_tap_invalid",
                        )
                    except Exception:
                        pass
            return frame

    @functools.wraps(original_add_track)
    def add_track_with_recorder(self: Any, track: Any) -> Any:
        original_track = track
        try:
            kind = getattr(track, "kind", None)
            if kind == "video" and not isinstance(track, RecordingVideoTapTrack):
                track = RecordingVideoTapTrack(track, self)
            elif kind == "audio" and not isinstance(
                track, RecordingAssistantAudioTapTrack
            ):
                track = RecordingAssistantAudioTapTrack(track, self)
        except Exception:
            track = original_track
        return original_add_track(self, track)

    cls.addTrack = add_track_with_recorder
    cls._experiment_recording_tracks_patched = True
    cls._experiment_recording_original_add_track = original_add_track
    module.RecordingVideoTapTrack = RecordingVideoTapTrack
    module.RecordingAssistantAudioTapTrack = RecordingAssistantAudioTapTrack
    return f"{module.__name__}.RTCPeerConnection.experiment_recording"


def _same_origin(request: Any) -> bool:
    origin = str(request.headers.get("Origin", "") or "")
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme == str(request.scheme)
            and parsed.netloc == str(request.host)
            and parsed.path in {"", "/"}
        )
    except Exception:
        return False


async def _small_json(request: Any) -> dict[str, Any]:
    length = request.content_length
    if length is not None and length > _MAX_JSON_BYTES:
        raise ValueError("request_too_large")
    content = getattr(request, "content", None)
    read_chunk = getattr(content, "read", None)
    if callable(read_chunk):
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_JSON_BYTES:
            chunk = await read_chunk(_MAX_JSON_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(bytes(chunk))
            total += len(chunk)
            if total > _MAX_JSON_BYTES:
                raise ValueError("request_too_large")
        raw = b"".join(chunks)
    else:
        # Test/fallback request objects do not expose aiohttp's bounded stream.
        raw = await request.read()
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError("request_too_large")
    import json

    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid_json_object")
    return value


def _valid_identifier(value: Any, *, maximum: int) -> str | None:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        return None
    if not all(character.isalnum() or character in "-_" for character in value):
        return None
    return value


def _install_build_app_routes(
    module: ModuleType,
    registry: _ControllerRegistry,
) -> str:
    original_build_app = getattr(module, "build_app", None)
    if not callable(original_build_app):
        raise RuntimeError("experiment recorder requires build_app")
    if getattr(module, "_experiment_recording_build_app_patched", False):
        return f"{module.__name__}.build_app.experiment_recording"

    def json_error(web: Any, status: int, code: str, message: str) -> Any:
        return web.json_response(
            {
                "type": STATUS_TYPE,
                "state": "error",
                "active": False,
                "session_id": "",
                "request_id": "",
                "stream_generation": 0,
                "recording_id": "",
                "error_code": code,
                "message": message,
            },
            status=status,
        )

    @functools.wraps(original_build_app)
    def build_app_with_recording(*args: Any, **kwargs: Any) -> Any:
        app = original_build_app(*args, **kwargs)
        from aiohttp import web

        def current_controller(session_id: str) -> SessionRecordingController | None:
            sessions = getattr(module, "SESSIONS", {})
            stream = sessions.get(session_id) if hasattr(sessions, "get") else None
            return _controller_for_stream(stream)

        def current_recordable_stream(session_id: str) -> tuple[Any, SessionRecordingController] | None:
            sessions = getattr(module, "SESSIONS", {})
            stream = sessions.get(session_id) if hasattr(sessions, "get") else None
            controller = _controller_for_stream(stream)
            if stream is None or controller is None or bool(getattr(stream, "closed", False)):
                return None
            pc_state = str(getattr(getattr(stream, "pc", None), "connectionState", ""))
            if pc_state != "connected" or int(getattr(stream, "request_id", 0)) <= 0:
                return None
            if str(getattr(stream, "status", "")) in {
                "idle",
                "prepared",
                "stopped",
                "error",
            }:
                return None
            return stream, controller

        async def start_recording(request: Any) -> Any:
            if not bool(
                getattr(module, "_experiment_recording_routes_available", False)
            ):
                return json_error(
                    web,
                    503,
                    "recording_unavailable",
                    "Experiment recording is unavailable; live conversation is unaffected",
                )
            if (
                request.headers.get(_REQUEST_HEADER) != "1"
                or not _same_origin(request)
            ):
                return json_error(web, 403, "recording_request_forbidden", "Recording request rejected")
            try:
                body = await _small_json(request)
            except Exception as exc:
                status = 413 if str(exc) == "request_too_large" else 400
                return json_error(web, status, "invalid_recording_request", "Invalid recording request")
            session_id = _valid_identifier(body.get("session_id"), maximum=128)
            request_id = _valid_identifier(body.get("request_id"), maximum=96)
            if session_id is None or request_id is None:
                return json_error(web, 400, "invalid_recording_request", "Invalid recording identifiers")
            recordable = current_recordable_stream(session_id)
            if recordable is None:
                return json_error(
                    web,
                    409,
                    "session_not_recordable",
                    "Start and connect Gemini before recording",
                )
            _stream, controller = recordable
            payload = await asyncio.to_thread(controller.start, request_id)
            state = payload.get("state")
            if state == "recording":
                status = 200
            elif state in {"starting", "finalizing"}:
                # An exact idempotent retry may observe the original job while
                # encoder startup or cleanup is still in progress. Return its
                # ownership token as a successful pending response so the UI
                # can poll and, critically, can still stop that recorder.
                status = 202
            else:
                status = 409
            if payload.get("error_code") in {"recording_disabled", "recording_permission_denied", "disk_space_low", "ffmpeg_unavailable"}:
                status = 503
            return web.json_response(payload, status=status)

        async def stop_recording(request: Any) -> Any:
            if (
                request.headers.get(_REQUEST_HEADER) != "1"
                or not _same_origin(request)
            ):
                return json_error(web, 403, "recording_request_forbidden", "Recording request rejected")
            try:
                body = await _small_json(request)
            except Exception as exc:
                status = 413 if str(exc) == "request_too_large" else 400
                return json_error(web, status, "invalid_recording_request", "Invalid recording request")
            session_id = _valid_identifier(body.get("session_id"), maximum=128)
            recording_id = _valid_identifier(body.get("recording_id"), maximum=96)
            if session_id is None or recording_id is None:
                return json_error(web, 400, "invalid_recording_request", "Invalid recording identifiers")
            controller = current_controller(session_id)
            if controller is None:
                return json_error(web, 404, "session_not_found", "WebRTC session was not found")
            payload = controller.stop(recording_id)
            status = 409 if payload.get("error_code") == "recording_owner_mismatch" else 200
            return web.json_response(payload, status=status)

        async def recording_status(request: Any) -> Any:
            if not _same_origin(request):
                return json_error(web, 403, "recording_request_forbidden", "Recording request rejected")
            session_id = _valid_identifier(request.match_info.get("session_id"), maximum=128)
            recording_id = _valid_identifier(request.query.get("recording_id"), maximum=96)
            if session_id is None or recording_id is None:
                return json_error(web, 400, "invalid_recording_request", "Invalid recording identifiers")
            controller = current_controller(session_id)
            if controller is None:
                return json_error(web, 404, "session_not_found", "WebRTC session was not found")
            payload = controller.status(recording_id)
            status = 404 if payload.get("error_code") == "recording_not_found" else 200
            return web.json_response(payload, status=status)

        async def finalize_recorders(_app: Any) -> None:
            registry.stop_all()
            if await asyncio.to_thread(registry.wait_all, 5.0):
                return
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(registry.abort_all), timeout=0.5
                )
            except (TimeoutError, asyncio.TimeoutError):
                pass
            if await asyncio.to_thread(registry.wait_all, 2.0):
                return
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(registry.abort_all, force=True), timeout=0.5
                )
            except (TimeoutError, asyncio.TimeoutError):
                pass
            await asyncio.to_thread(registry.wait_all, 1.0)

        module._experiment_recording_routes_available = False
        try:
            # Register shutdown first so even a later partial route failure has
            # a bounded cleanup path.  The start handler remains disabled until
            # every optional route has been installed successfully.
            app.on_shutdown.append(finalize_recorders)
            app.router.add_post(START_PATH, start_recording)
            app.router.add_post(STOP_PATH, stop_recording)
            app.router.add_get(STATUS_PATH, recording_status)
            module._experiment_recording_routes_available = True
        except Exception as exc:
            logger = getattr(module, "logger", None)
            if logger is not None:
                try:
                    logger.warning(
                        "Experiment recording routes disabled: {}",
                        type(exc).__name__,
                    )
                except Exception:
                    pass
        return app

    module.build_app = build_app_with_recording
    module._experiment_recording_build_app_patched = True
    module._experiment_recording_original_build_app = original_build_app
    return f"{module.__name__}.build_app.experiment_recording"


def install_semantic_receipt_observer(module: ModuleType) -> bool:
    """Install exact terminal action observation only for semantic mode."""
    registry = getattr(module, "_experiment_recording_registry", None)
    if not isinstance(registry, _ControllerRegistry):
        return False
    try:
        from .semantic_runtime import SemanticSuffixExecutor

        return install_submission_receipt_observer(
            SemanticSuffixExecutor, registry.semantic_observer
        )
    except Exception:
        return False


def install_experiment_recording(module: ModuleType) -> list[str]:
    """Install recorder sidecars before semantic ownership patches are applied."""

    existing = getattr(module, "_experiment_recording_registry", None)
    if isinstance(existing, _ControllerRegistry):
        return list(getattr(module, "_experiment_recording_patch_markers", ()))
    config = RecordingConfig.from_env()
    config.validate()
    if not config.enabled:
        module._experiment_recording_patch_markers = ()
        return []
    required = {
        "StreamSession": ("__init__", "start_live_generation", "close"),
        "GeminiLiveBridge": ("start_session", "_send_audio_loop", "close"),
        "RTCPeerConnection": ("addTrack",),
        "MediaStreamTrack": (),
    }
    for class_name, methods in required.items():
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type) or not all(
            callable(getattr(cls, method, None)) for method in methods
        ):
            raise RuntimeError(
                f"experiment recorder upstream contract missing {class_name}"
            )
    if not callable(getattr(module, "build_app", None)):
        raise RuntimeError("experiment recorder upstream contract missing build_app")
    registry = getattr(module, "_experiment_recording_pending_registry", None)
    if not isinstance(registry, _ControllerRegistry):
        registry = _ControllerRegistry(config)
        # If an unexpected optional-patch failure occurs after the first wrapper
        # is installed, a later retry must reuse the registry captured by that
        # wrapper.  Otherwise shutdown and the global encoder slot could split
        # across two registries.
        module._experiment_recording_pending_registry = registry
    markers = [
        _install_stream_lifecycle(module, registry),
        _install_bridge_lifecycle(module),
        _install_peer_connection_taps(module),
        _install_build_app_routes(module, registry),
    ]
    module._experiment_recording_registry = registry
    module._experiment_recording_patch_markers = tuple(markers)
    try:
        delattr(module, "_experiment_recording_pending_registry")
    except AttributeError:
        pass
    return markers
