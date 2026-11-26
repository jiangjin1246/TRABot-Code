from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import select
import shutil
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping
import uuid
import wave

if TYPE_CHECKING:
    from .reply_trace import ReplyKey


STATUS_TYPE = "experiment_recording_status_v1"
_WALL_CLOCK_STEP_THRESHOLD_NS = 50_000_000
_REPLY_STATE_LIMIT = 256
_REPLY_ACTION_LIMIT = 256
_REPLY_EVENT_BYTES_LIMIT = 256 * 1024
_REPLY_METADATA_BYTES_LIMIT = 8 * 1024 * 1024
TERMINAL_STATES = frozenset({"completed", "error"})


def _snapshot_reply_event(value: Any) -> dict[str, Any] | None:
    """Bounded immutable event copy made before producer objects can mutate."""
    remaining_nodes = [4096]
    remaining_bytes = [_REPLY_EVENT_BYTES_LIMIT]
    truncated = [False]

    def copy_text(item: Any, maximum: int) -> str:
        raw = str(item).encode("utf-8", errors="replace")
        budget = max(0, min(int(maximum), remaining_bytes[0]))
        if len(raw) > budget:
            truncated[0] = True
        clipped = raw[:budget]
        # Avoid a partial UTF-8 code point at the boundary.
        text = clipped.decode("utf-8", errors="ignore")
        remaining_bytes[0] -= len(text.encode("utf-8"))
        return text

    def copy_value(item: Any, depth: int = 0) -> Any:
        remaining_nodes[0] -= 1
        if remaining_nodes[0] < 0 or remaining_bytes[0] <= 0 or depth > 8:
            truncated[0] = True
            return "<truncated>"
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, str):
            return copy_text(item, 16_384)
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for index, (key, nested) in enumerate(item.items()):
                if index >= 256 or remaining_nodes[0] <= 0 or remaining_bytes[0] <= 0:
                    result["_truncated"] = True
                    truncated[0] = True
                    break
                result[copy_text(key, 256)] = copy_value(nested, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            result = [copy_value(nested, depth + 1) for nested in item[:256]]
            if len(item) > 256:
                result.append("<truncated>")
                truncated[0] = True
            return result
        return copy_text(item, 16_384)

    snapshot = copy_value(value)
    if not isinstance(snapshot, dict):
        return None
    if truncated[0]:
        snapshot["_reply_metrics_snapshot_truncated"] = True
    return snapshot


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, yes/no, on/off, or 1/0")


@dataclass(frozen=True, slots=True)
class RecordingConfig:
    """Server-owned limits for the opt-in experiment recorder."""

    enabled: bool = True
    output_root: Path = Path("runtime/experiment_recordings")
    ffmpeg_binary: str = "ffmpeg"
    width: int = 640
    height: int = 480
    fps: int = 25
    sample_rate: int = 16_000
    video_queue_size: int = 8
    audio_queue_size: int = 256
    max_audio_queue_bytes: int = 4 * 1024 * 1024
    minimum_free_bytes: int = 10 * 1024 * 1024 * 1024
    max_duration_seconds: float = 2 * 60 * 60
    max_recording_bytes: int = 20 * 1024 * 1024 * 1024
    encoder_wait_seconds: float = 8.0
    pipe_write_timeout_seconds: float = 2.0
    finalize_timeout_seconds: float = 600.0
    max_audio_packet_bytes: int = 32_000
    max_audio_gap_seconds: float = 300.0
    max_video_gap_frames: int = 50
    max_active_recordings: int = 1
    reply_event_queue_size: int = 512
    reply_event_bytes_limit: int = _REPLY_EVENT_BYTES_LIMIT
    reply_metadata_bytes_limit: int = _REPLY_METADATA_BYTES_LIMIT

    @classmethod
    def from_env(cls) -> "RecordingConfig":
        defaults = cls()
        return cls(
            enabled=_env_bool(
                "OVERLAY_EXPERIMENT_RECORDING_ENABLED", defaults.enabled
            ),
            output_root=Path(
                os.environ.get(
                    "OVERLAY_EXPERIMENT_RECORDING_ROOT",
                    str(defaults.output_root),
                )
            ),
            ffmpeg_binary=os.environ.get(
                "OVERLAY_EXPERIMENT_RECORDING_FFMPEG", defaults.ffmpeg_binary
            ),
            minimum_free_bytes=int(
                os.environ.get(
                    "OVERLAY_EXPERIMENT_RECORDING_MIN_FREE_BYTES",
                    defaults.minimum_free_bytes,
                )
            ),
            max_duration_seconds=float(
                os.environ.get(
                    "OVERLAY_EXPERIMENT_RECORDING_MAX_SECONDS",
                    defaults.max_duration_seconds,
                )
            ),
            max_recording_bytes=int(
                os.environ.get(
                    "OVERLAY_EXPERIMENT_RECORDING_MAX_BYTES",
                    defaults.max_recording_bytes,
                )
            ),
            reply_event_bytes_limit=int(
                os.environ.get(
                    "OVERLAY_EXPERIMENT_REPLY_EVENT_MAX_BYTES",
                    defaults.reply_event_bytes_limit,
                )
            ),
            reply_metadata_bytes_limit=int(
                os.environ.get(
                    "OVERLAY_EXPERIMENT_REPLY_METADATA_MAX_BYTES",
                    defaults.reply_metadata_bytes_limit,
                )
            ),
        )

    def validate(self) -> None:
        if self.width != 640 or self.height != 480:
            raise ValueError("experiment recording geometry must be 640x480")
        if self.fps != 25 or self.sample_rate != 16_000:
            raise ValueError("experiment recording media clock must be 25 fps / 16 kHz")
        if not 1 <= self.video_queue_size <= 8:
            raise ValueError("video_queue_size must be in [1, 8]")
        if self.audio_queue_size <= 0 or self.max_audio_queue_bytes <= 0:
            raise ValueError("audio queue limits must be positive")
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes must be non-negative")
        if self.max_duration_seconds <= 0 or not math.isfinite(
            self.max_duration_seconds
        ):
            raise ValueError("max_duration_seconds must be positive and finite")
        if (
            self.max_recording_bytes <= 0
            or self.encoder_wait_seconds <= 0
            or self.pipe_write_timeout_seconds <= 0
            or self.finalize_timeout_seconds <= 0
            or not 2 <= self.max_audio_packet_bytes <= self.max_audio_queue_bytes
            or self.max_audio_gap_seconds <= 0
            or not math.isfinite(self.max_audio_gap_seconds)
            or self.max_video_gap_frames <= 0
            or self.max_active_recordings <= 0
            or not 1 <= self.reply_event_queue_size <= 2048
            or not 4096 <= self.reply_event_bytes_limit <= 1024 * 1024
            or not self.reply_event_bytes_limit <= self.reply_metadata_bytes_limit <= 64 * 1024 * 1024
        ):
            raise ValueError("recording byte/time limits must be positive")


@dataclass(frozen=True, slots=True)
class _VideoPacket:
    captured_ns: int
    bgr: Any
    reply_key: ReplyKey | None = None
    face_source_seq: int | None = None
    pts: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    media_seq: int | None = None


@dataclass(frozen=True, slots=True)
class _AudioPacket:
    captured_ns: int
    pcm16: bytes
    media_time_ns: int | None = None
    reply_key: ReplyKey | None = None
    pts: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    dropped_before: int = 0
    media_seq: int | None = None


@dataclass(frozen=True, slots=True)
class _PcmWriteResult:
    bytes_written: int
    payload_start_sample: int | None
    payload_end_sample_exclusive: int | None


@dataclass(frozen=True, slots=True)
class _VideoWriteResult:
    last_frame: Any
    packet_frame_index: int | None
    written: bool


class _PcmTimelineWriter:
    """Write timestamped mono PCM to a WAV without accumulating the session."""

    def __init__(
        self,
        path: Path,
        *,
        t0_ns: int,
        sample_rate: int,
        timeline: str,
        max_gap_samples: int | None = None,
    ) -> None:
        self.path = path
        self.t0_ns = t0_ns
        self.sample_rate = sample_rate
        self.cursor = 0
        self.timeline = timeline
        self.max_gap_samples = (
            sample_rate * 300 if max_gap_samples is None else int(max_gap_samples)
        )
        if self.max_gap_samples <= 0:
            raise ValueError("max_gap_samples must be positive")
        self._first_packet = True
        self._media_anchor_ns: int | None = None
        self._capture_anchor_ns: int | None = None
        self._last_media_ns: int | None = None
        self._last_capture_ns: int | None = None
        self._media_gap_offset_ns = 0
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)

    def write(self, packet: _AudioPacket) -> _PcmWriteResult:
        pcm = packet.pcm16
        if not pcm:
            return _PcmWriteResult(0, None, None)
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if self.timeline == "continuous_capture":
            # Browser PCM already contains the user's silence. Anchor the first
            # successfully model-bound chunk, then preserve sample continuity;
            # SDK scheduling jitter must not create or trim speech.
            expected = (
                max(
                    0,
                    round(
                        (packet.captured_ns - self.t0_ns)
                        * self.sample_rate
                        / 1e9
                    ),
                )
                if self._first_packet
                else self.cursor
            )
        elif self.timeline == "media_pts":
            if packet.media_time_ns is None:
                raise ValueError("assistant packet is missing media PTS")
            if self._media_anchor_ns is None:
                self._media_anchor_ns = packet.media_time_ns
                self._capture_anchor_ns = packet.captured_ns
            else:
                # Use cumulative drift from the immutable first-packet anchors.
                # Repeated sub-deadband stalls must accumulate; a per-packet
                # threshold silently compresses long real playout gaps.
                assert self._capture_anchor_ns is not None
                media_elapsed = max(0, packet.media_time_ns - self._media_anchor_ns)
                capture_elapsed = max(0, packet.captured_ns - self._capture_anchor_ns)
                observed_gap = max(0, capture_elapsed - media_elapsed)
                if observed_gap > 100_000_000:
                    self._media_gap_offset_ns = max(
                        self._media_gap_offset_ns, observed_gap
                    )
            assert self._capture_anchor_ns is not None
            expected_ns = (
                self._capture_anchor_ns
                - self.t0_ns
                + packet.media_time_ns
                - self._media_anchor_ns
                + self._media_gap_offset_ns
            )
            expected = max(0, round(expected_ns * self.sample_rate / 1e9))
        else:
            raise ValueError("unknown PCM timeline")
        self._first_packet = False
        if packet.media_time_ns is not None:
            self._last_media_ns = packet.media_time_ns
            self._last_capture_ns = packet.captured_ns
        bytes_written = 0
        if expected > self.cursor:
            silence_samples = expected - self.cursor
            if silence_samples > self.max_gap_samples:
                raise ValueError("audio_timeline_discontinuity")
            silence = b"\0\0" * min(silence_samples, self.sample_rate)
            while silence_samples:
                count = min(silence_samples, self.sample_rate)
                self._wav.writeframesraw(silence[: count * 2])
                self.cursor += count
                bytes_written += count * 2
                silence_samples -= count
        elif expected < self.cursor:
            overlap = min(self.cursor - expected, len(pcm) // 2)
            pcm = pcm[overlap * 2 :]
        payload_start = None
        payload_end = None
        if pcm:
            payload_start = self.cursor
            self._wav.writeframesraw(pcm)
            samples = len(pcm) // 2
            self.cursor += samples
            payload_end = self.cursor
            bytes_written += len(pcm)
        return _PcmWriteResult(bytes_written, payload_start, payload_end)

    def close(self) -> None:
        if self.cursor == 0:
            self._wav.writeframesraw(b"\0\0")
            self.cursor = 1
        self._wav.close()


class ExperimentRecorder:
    """A fail-open media sidecar: live paths only copy and offer bounded packets."""

    def __init__(
        self,
        config: RecordingConfig,
        *,
        session_id: str,
        request_id: str,
        stream_generation: int,
        recording_id: str | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_now: Callable[[], float] = time.time,
        popen: Callable[..., Any] = subprocess.Popen,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        reply_hub: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.session_id = session_id
        self.request_id = request_id
        self.stream_generation = int(stream_generation)
        self.recording_id = recording_id or uuid.uuid4().hex
        self._monotonic_ns = monotonic_ns
        self._utc_now = utc_now
        self._popen = popen
        self._disk_usage = disk_usage
        self._reply_hub = reply_hub
        self._lock = threading.RLock()
        self._process_lock = threading.RLock()
        self._state = "idle"
        self._message = ""
        self._error_code = ""
        self._stop_reason = ""
        self._started_ns = 0
        self._started_utc = 0.0
        self._wall_anchor_uncertainty_ns = 0
        self._start_anchor_before_ns = 0
        self._start_anchor_after_ns = 0
        self._capture_stopped_ns = 0
        self._capture_stopped_wall = 0.0
        self._stop_wall_anchor_uncertainty_ns = 0
        self._stop_anchor_before_ns = 0
        self._stop_anchor_after_ns = 0
        self._wall_clock_offset_change_ns = 0
        self._finished_ns = 0
        self._recording_dir: Path | None = None
        self._final_path: Path | None = None
        self._video_queue: queue.Queue[_VideoPacket] = queue.Queue(
            maxsize=config.video_queue_size
        )
        self._user_queue: queue.Queue[_AudioPacket] = queue.Queue(
            maxsize=config.audio_queue_size
        )
        self._assistant_queue: queue.Queue[_AudioPacket] = queue.Queue(
            maxsize=config.audio_queue_size
        )
        self._reply_queue: queue.Queue[Mapping[str, Any]] = queue.Queue(
            maxsize=config.reply_event_queue_size
        )
        self._audio_queued_bytes = 0
        self._accepting = False
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()
        self._ready_event = threading.Event()
        self._finished_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._encoder_process: Any | None = None
        self._mux_process: Any | None = None
        self._dropped_video_frames = 0
        self._video_frames_written = 0
        self._user_samples_written = 0
        self._assistant_samples_written = 0
        self._reply_accumulator: Any | None = None
        self._reply_metrics_accepting = True
        self._reply_metrics_dropped_events = 0
        self._reply_metrics_metadata_bytes = 0
        self._reply_metrics_error = ""
        self._reply_metrics_summary: dict[str, Any] = {
            "schema_version": "experiment_reply_v2",
            "formula_version": "reply_timing_v2_semantic_body_recovery_excluded_v1",
            "path": "",
            "sha256": "",
            "row_count": 0,
            "dropped_events": 0,
            "error": "",
            "partial_path": "",
            "promotion_state": "none",
            "last_good_rows": 0,
        }

    def _owner_matches(
        self, session_id: str, recording_id: str, stream_generation: int | None
    ) -> bool:
        return (
            session_id == self.session_id
            and recording_id == self.recording_id
            and (
                stream_generation is None
                or int(stream_generation) == self.stream_generation
            )
        )

    def set_stream_generation(self, generation: int) -> bool:
        """Bind a pre-Gemini recording to the first live generation only."""

        generation = int(generation)
        with self._lock:
            if self._state not in {"idle", "starting", "recording"}:
                return False
            if self.stream_generation not in {0, generation}:
                return False
            self.stream_generation = generation
            return True

    def start(self) -> bool:
        with self._lock:
            if self._state in {"starting", "recording"}:
                return True
            if self._state != "idle" or not self.config.enabled:
                return False
            self._state = "starting"

        try:
            root = self._prepare_output_root()
            if self._disk_usage(root).free < self.config.minimum_free_bytes:
                raise RuntimeError("disk_space_low")
            with self._lock:
                if self._state != "starting" or self._stop_event.is_set():
                    return False
            stamp = time.strftime(
                "%Y%m%dT%H%M%SZ", time.gmtime(self._utc_now())
            )
            directory = root / f"{stamp}_{self.recording_id}"
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
            anchor_before_ns = self._monotonic_ns()
            started_utc = self._utc_now()
            anchor_after_ns = self._monotonic_ns()
            started_ns = (anchor_before_ns + anchor_after_ns) // 2
            with self._lock:
                self._recording_dir = directory
                self._started_ns = started_ns
                self._started_utc = started_utc
                self._start_anchor_before_ns = anchor_before_ns
                self._start_anchor_after_ns = anchor_after_ns
                anchor_span_ns = max(0, anchor_after_ns - anchor_before_ns)
                self._wall_anchor_uncertainty_ns = (anchor_span_ns + 1) // 2
            if self._reply_hub is not None:
                try:
                    self._reply_hub.set_event_sink(self.offer_reply_event)
                except Exception:
                    with self._lock:
                        self._reply_metrics_error = "reply_sink_attach_failed"
            worker = threading.Thread(
                target=self._worker_main,
                name=f"experiment-recorder-{self.recording_id[:8]}",
                daemon=True,
            )
            with self._lock:
                if self._state != "starting" or self._stop_event.is_set():
                    return False
                self._worker = worker
                # Start while holding the state lock so request_stop cannot see a
                # published-but-not-started worker and strand the global slot.
                worker.start()
            if not self._ready_event.wait(self.config.encoder_wait_seconds):
                self._fail("encoder_start_timeout")
                return False
            return self.is_recording()
        except Exception as exc:
            self._fail(self._error_from_exception(exc))
            return False

    def _prepare_output_root(self) -> Path:
        configured = self.config.output_root.expanduser()
        if configured.is_symlink():
            raise RuntimeError("recording_root_unsafe")
        root = configured.resolve()
        broad_roots = {
            Path(root.anchor),
            Path.home().resolve(),
            Path.cwd().resolve(),
        }
        if root in broad_roots:
            raise RuntimeError("recording_root_unsafe")
        created = False
        try:
            root.mkdir(mode=0o700, parents=False)
            created = True
        except FileExistsError:
            pass
        except FileNotFoundError as exc:
            raise RuntimeError("recording_root_unsafe") from exc
        if created:
            # Only tighten permissions on the dedicated directory created by
            # this recorder. Never chmod an operator-provided existing path.
            os.chmod(root, 0o700)
        info = root.stat()
        if not root.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("recording_root_unsafe")
        return root

    @staticmethod
    def _error_from_exception(exc: BaseException) -> str:
        value = str(exc).strip()
        if value in {
            "disk_space_low",
            "duration_limit",
            "byte_limit",
            "ffmpeg_exited",
            "ffmpeg_pipe_timeout",
            "video_timeline_discontinuity",
            "video_stream_missing",
            "final_mux_failed",
            "final_mux_timeout",
            "finalization_aborted",
            "audio_timeline_discontinuity",
            "recording_root_unsafe",
        } or value.startswith("ffmpeg_exit_"):
            return value
        if isinstance(exc, FileNotFoundError):
            return "ffmpeg_unavailable"
        if isinstance(exc, PermissionError):
            return "recording_permission_denied"
        return f"recording_io_error:{type(exc).__name__}"

    def is_recording(self) -> bool:
        with self._lock:
            return self._state == "recording" and self._accepting

    def is_finished(self) -> bool:
        return self._finished_event.is_set()

    def _mark_capture_stopped_unlocked(self) -> None:
        if self._started_ns and not self._capture_stopped_ns:
            before_ns = self._monotonic_ns()
            stopped_wall = self._utc_now()
            after_ns = self._monotonic_ns()
            stopped_ns = (before_ns + after_ns) // 2
            self._capture_stopped_ns = max(self._started_ns, stopped_ns)
            self._capture_stopped_wall = float(stopped_wall)
            self._stop_anchor_before_ns = before_ns
            self._stop_anchor_after_ns = after_ns
            span_ns = max(0, after_ns - before_ns)
            self._stop_wall_anchor_uncertainty_ns = (span_ns + 1) // 2
            start_offset_ns = round(self._started_utc * 1e9) - self._started_ns
            stop_offset_ns = (
                round(self._capture_stopped_wall * 1e9)
                - self._capture_stopped_ns
            )
            self._wall_clock_offset_change_ns = (
                stop_offset_ns - start_offset_ns
            )

    def offer_reply_event(self, event: Mapping[str, Any]) -> bool:
        """Non-blocking fail-open handoff; accumulator mutation stays on worker."""
        try:
            if not isinstance(event, Mapping):
                return False
            if event.get("kind") == "reply_metrics_error":
                error = str(event.get("error") or "reply_metrics_error")[:128]
                with self._lock:
                    if self._reply_metrics_accepting:
                        self._reply_metrics_dropped_events += 1
                    self._reply_metrics_accepting = False
                    self._reply_metrics_error = error
                return False
            snapshot = _snapshot_reply_event(event)
            if snapshot is None:
                return False
            if snapshot.pop("_reply_metrics_snapshot_truncated", False) is True:
                with self._lock:
                    if self._reply_metrics_accepting:
                        self._reply_metrics_dropped_events += 1
                    self._reply_metrics_accepting = False
                    self._reply_metrics_error = "reply_event_snapshot_limit_exceeded"
                return False
            try:
                event_bytes = len(json.dumps(
                    snapshot, ensure_ascii=False, allow_nan=False,
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8"))
            except Exception:
                event_bytes = self.config.reply_event_bytes_limit + 1
            with self._lock:
                if (
                    not self._reply_metrics_accepting
                    or (not self._accepting and self._state != "starting")
                ):
                    return False
                projected = self._reply_metrics_metadata_bytes + event_bytes
                if (
                    event_bytes > self.config.reply_event_bytes_limit
                    or projected > self.config.reply_metadata_bytes_limit
                ):
                    self._reply_metrics_accepting = False
                    self._reply_metrics_dropped_events += 1
                    self._reply_metrics_error = (
                        "reply_event_bytes_limit_exceeded"
                        if event_bytes > self.config.reply_event_bytes_limit
                        else "reply_metadata_bytes_limit_exceeded"
                    )
                    return False
                try:
                    self._reply_queue.put_nowait(snapshot)
                except queue.Full:
                    self._reply_metrics_accepting = False
                    self._reply_metrics_dropped_events += 1
                    self._reply_metrics_error = "reply_event_queue_full"
                    return False
                self._reply_metrics_metadata_bytes = projected
            return True
        except Exception:
            return False

    def _fail(self, error_code: str) -> None:
        with self._lock:
            if self._state in TERMINAL_STATES:
                return
            self._error_code = error_code
            self._mark_capture_stopped_unlocked()
            self._accepting = False
            self._stop_reason = error_code
            self._stop_event.set()
            worker = self._worker
            if worker is None or not worker.is_alive():
                self._state = "error"
                self._message = "Recording stopped; live conversation is unaffected"
                self._finished_ns = self._monotonic_ns()
                self._discard_pending_media_unlocked()
                self._finished_event.set()
            else:
                self._state = "finalizing"
                self._message = "Recording failed; cleaning up"

    def fail(self, error_code: str) -> bool:
        """Public, non-throwing fault sink for a media tap."""

        try:
            self._fail(str(error_code)[:160] or "recording_error")
        except Exception:
            return False
        return False

    def _limits_ok(self, now_ns: int) -> bool:
        if now_ns - self._started_ns > self.config.max_duration_seconds * 1e9:
            self._fail("duration_limit")
            return False
        return True

    def _current_disk_bytes(self) -> int:
        directory = self._recording_dir
        if directory is None:
            return 0
        total = 0
        try:
            for path in directory.iterdir():
                try:
                    if path.is_file():
                        total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            return self.config.max_recording_bytes + 1
        return total

    @staticmethod
    def _empty_queue(target: queue.Queue[Any]) -> int:
        discarded = 0
        while True:
            try:
                target.get_nowait()
                discarded += 1
            except queue.Empty:
                return discarded

    def _discard_pending_media_unlocked(self) -> None:
        """Release queued media while the recorder state lock is held."""

        self._dropped_video_frames += self._empty_queue(self._video_queue)
        self._empty_queue(self._user_queue)
        self._empty_queue(self._assistant_queue)
        self._empty_queue(self._reply_queue)
        self._audio_queued_bytes = 0

    def _fail_current_offer(
        self,
        error_code: str,
        *,
        session_id: str,
        recording_id: str,
        stream_generation: int | None,
    ) -> bool:
        with self._lock:
            if self._accepting and self._owner_matches(
                session_id, recording_id, stream_generation
            ):
                self._fail(error_code)
        return False

    def push_video_bgr(
        self,
        frame: Any,
        *,
        session_id: str,
        recording_id: str,
        stream_generation: int | None,
        captured_ns: int | None = None,
        reply_key: ReplyKey | None = None,
        face_source_seq: int | None = None,
        pts: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        media_seq: int | None = None,
    ) -> bool:
        try:
            with self._lock:
                if not self._accepting or not self._owner_matches(
                    session_id, recording_id, stream_generation
                ):
                    return False
                now_ns = self._monotonic_ns() if captured_ns is None else int(captured_ns)
                if not self._limits_ok(now_ns):
                    return False
            shape = tuple(getattr(frame, "shape", ()))
            if shape == (self.config.height, 1280, 3):
                frame = frame[:, 320:960]
            if tuple(getattr(frame, "shape", ())) != (
                self.config.height,
                self.config.width,
                3,
            ):
                return self._fail_current_offer(
                    "video_geometry_invalid",
                    session_id=session_id,
                    recording_id=recording_id,
                    stream_generation=stream_generation,
                )
            packet = _VideoPacket(
                now_ns, frame.copy(), reply_key, face_source_seq,
                pts, time_base_num, time_base_den, media_seq,
            )
            # Copying a frame happens outside the state lock. Recheck ownership
            # and enqueue in one critical section so terminal cleanup cannot be
            # followed by a late put from an in-flight media callback.
            with self._lock:
                if not self._accepting or not self._owner_matches(
                    session_id, recording_id, stream_generation
                ):
                    return False
                if not self._limits_ok(now_ns):
                    return False
                try:
                    self._video_queue.put_nowait(packet)
                except queue.Full:
                    self._dropped_video_frames += 1
                    return False
            return True
        except Exception:
            return self._fail_current_offer(
                "video_tap_error",
                session_id=session_id,
                recording_id=recording_id,
                stream_generation=stream_generation,
            )

    def _push_audio(
        self,
        target: queue.Queue[_AudioPacket],
        pcm16: bytes | bytearray | memoryview,
        *,
        session_id: str,
        recording_id: str,
        stream_generation: int,
        sample_rate: int,
        channels: int,
        captured_ns: int | None,
        media_time_ns: int | None = None,
        reply_key: ReplyKey | None = None,
        pts: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        dropped_before: int = 0,
        media_seq: int | None = None,
    ) -> bool:
        try:
            with self._lock:
                if not self._accepting or not self._owner_matches(
                    session_id, recording_id, stream_generation
                ):
                    return False
                if sample_rate != self.config.sample_rate or channels != 1:
                    self._fail("audio_format_invalid")
                    return False
                payload_size = len(pcm16)
                if (
                    payload_size <= 0
                    or payload_size % 2
                    or payload_size > self.config.max_audio_packet_bytes
                ):
                    self._fail("audio_payload_invalid")
                    return False
                now_ns = self._monotonic_ns() if captured_ns is None else int(captured_ns)
                if not self._limits_ok(now_ns):
                    return False
                if self._audio_queued_bytes + payload_size > self.config.max_audio_queue_bytes:
                    self._fail("audio_queue_full")
                    return False
                pcm = bytes(pcm16)
                try:
                    target.put_nowait(_AudioPacket(
                        now_ns, pcm, media_time_ns, reply_key, pts,
                        time_base_num, time_base_den, int(dropped_before),
                        media_seq,
                    ))
                except queue.Full:
                    self._fail("audio_queue_full")
                    return False
                self._audio_queued_bytes += len(pcm)
            return True
        except Exception:
            return self._fail_current_offer(
                "audio_tap_error",
                session_id=session_id,
                recording_id=recording_id,
                stream_generation=stream_generation,
            )

    def push_user_pcm16(self, pcm16: bytes, **metadata: Any) -> bool:
        return self._push_audio(self._user_queue, pcm16, **metadata)

    def push_assistant_pcm16(self, pcm16: bytes, **metadata: Any) -> bool:
        return self._push_audio(self._assistant_queue, pcm16, **metadata)

    def request_stop(
        self,
        *,
        session_id: str,
        recording_id: str,
        stream_generation: int | None = None,
        reason: str = "operator_stop",
    ) -> bool:
        try:
            with self._lock:
                if not self._owner_matches(
                    session_id, recording_id, stream_generation
                ):
                    return False
                if self._state in TERMINAL_STATES or self._state == "finalizing":
                    return True
                self._mark_capture_stopped_unlocked()
                worker = self._worker
                if worker is None and self._state in {"idle", "starting"}:
                    self._accepting = False
                    self._state = "completed"
                    self._message = "Recording stopped before capture began"
                    self._stop_reason = str(reason)[:160]
                    self._finished_ns = self._monotonic_ns()
                    self._discard_pending_media_unlocked()
                    self._stop_event.set()
                    self._ready_event.set()
                    self._finished_event.set()
                    return True
                self._accepting = False
                self._state = "finalizing"
                self._message = "Saving recording"
                self._stop_reason = str(reason)[:160]
                self._stop_event.set()
                return True
        except Exception:
            return False

    def wait_finished(self, timeout: float | None = None) -> bool:
        return self._finished_event.wait(timeout)

    def abort_finalize(
        self,
        *,
        reason: str = "server_shutdown_timeout",
        force: bool = False,
    ) -> None:
        """Bound shutdown without ever signalling a non-recorder process."""

        try:
            self._abort_event.set()
            self._fail(str(reason)[:160] or "finalization_aborted")
            if not self._process_lock.acquire(blocking=False):
                # Never make application shutdown wait for a stuck process
                # spawn. The spawn path observes _abort_event before releasing
                # this lock and kills any child it managed to create.
                return
            try:
                processes = (self._encoder_process, self._mux_process)
            finally:
                self._process_lock.release()
            for process in processes:
                if process is None or process.poll() is not None:
                    continue
                try:
                    (process.kill if force else process.terminate)()
                except Exception:
                    continue
        except Exception:
            pass

    def _open_ffmpeg(self, directory: Path) -> tuple[Any, Any, Path]:
        raw_video = directory / "video.partial.mkv"
        stderr_path = directory / "encoder.log"
        stderr_file = open(stderr_path, "xb", buffering=0)
        os.chmod(stderr_path, 0o600)
        argv = [
            self.config.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.config.width}x{self.config.height}",
            "-framerate",
            str(self.config.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-threads",
            "1",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            str(raw_video),
        ]
        process = None
        try:
            with self._process_lock:
                if self._abort_event.is_set():
                    raise RuntimeError("finalization_aborted")
                process = self._popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    shell=False,
                    close_fds=True,
                )
                self._encoder_process = process
                if self._abort_event.is_set():
                    try:
                        process.kill()
                    finally:
                        raise RuntimeError("finalization_aborted")
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            if process.poll() is not None:
                raise RuntimeError("ffmpeg_exited")
            os.set_blocking(process.stdin.fileno(), False)
            return process, stderr_file, raw_video
        except Exception:
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=2.0)
                except Exception:
                    pass
                with self._process_lock:
                    if self._encoder_process is process:
                        self._encoder_process = None
            try:
                stderr_file.close()
            except Exception:
                pass
            raise

    def _write_pipe_bounded(
        self,
        process: Any,
        payload: bytes,
        *,
        deadline: float | None = None,
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        fd = process.stdin.fileno()
        view = memoryview(payload)
        if deadline is None:
            deadline = time.monotonic() + self.config.pipe_write_timeout_seconds
        offset = 0
        while offset < len(view):
            if process.poll() is not None:
                raise RuntimeError("ffmpeg_exited")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("ffmpeg_pipe_timeout")
            _readable, writable, _errors = select.select([], [fd], [], remaining)
            if not writable:
                raise RuntimeError("ffmpeg_pipe_timeout")
            try:
                count = os.write(fd, view[offset : offset + 256 * 1024])
            except BlockingIOError:
                continue
            if count <= 0:
                raise BrokenPipeError("ffmpeg pipe closed")
            offset += count

    def _write_video_packet(
        self, process: Any, packet: _VideoPacket, last: Any
    ) -> _VideoWriteResult:
        frame_index = max(
            0,
            round((packet.captured_ns - self._started_ns) * self.config.fps / 1e9),
        )
        with self._lock:
            written = self._video_frames_written
        if frame_index < written:
            return _VideoWriteResult(last, None, False)
        if frame_index - written > self.config.max_video_gap_frames:
            raise RuntimeError("video_timeline_discontinuity")
        if last is None:
            last = packet.bgr
        deadline = time.monotonic() + self.config.pipe_write_timeout_seconds
        while written < frame_index:
            self._write_pipe_bounded(
                process,
                last.tobytes(order="C"),
                deadline=deadline,
            )
            written += 1
        packet_index = written
        self._write_pipe_bounded(
            process,
            packet.bgr.tobytes(order="C"),
            deadline=deadline,
        )
        written += 1
        with self._lock:
            self._video_frames_written = written
        return _VideoWriteResult(packet.bgr, packet_index, True)

    @staticmethod
    def _drain_one(target: queue.Queue[Any]) -> Any | None:
        try:
            return target.get_nowait()
        except queue.Empty:
            return None

    def _new_reply_accumulator(self) -> Any | None:
        try:
            from .reply_trace import ReplyMetricsAccumulator

            return ReplyMetricsAccumulator(
                recording_id=self.recording_id,
                session_id=self.session_id,
                started_monotonic_ns=self._started_ns,
                started_wall=self._started_utc,
                wall_anchor_uncertainty_ns=self._wall_anchor_uncertainty_ns,
                sample_rate=self.config.sample_rate,
                fps=self.config.fps,
            )
        except Exception:
            with self._lock:
                self._reply_metrics_accepting = False
                self._reply_metrics_error = "reply_accumulator_unavailable"
            return None

    def _drain_reply_events(self, accumulator: Any | None, limit: int = 64) -> bool:
        did_work = False
        for _ in range(max(0, int(limit))):
            event = self._drain_one(self._reply_queue)
            if event is None:
                break
            did_work = True
            if accumulator is None:
                continue
            try:
                if not accumulator.observe_event(event):
                    with self._lock:
                        self._reply_metrics_accepting = False
                        if not self._reply_metrics_error:
                            self._reply_metrics_error = str(
                                getattr(accumulator, "last_error", "")
                                or "reply_event_rejected"
                            )[:128]
            except Exception:
                with self._lock:
                    self._reply_metrics_accepting = False
                    self._reply_metrics_error = "reply_accumulator_error"
                accumulator = None
        return did_work

    @staticmethod
    def _write_all(fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(fd, view[offset:])
            if count <= 0:
                raise OSError("short sidecar write")
            offset += count

    def _prepare_replies_sidecar(
        self,
        accumulator: Any | None,
        *,
        user_track_end_sample: int,
        assistant_track_end_sample: int,
    ) -> bool:
        directory = self._recording_dir
        if directory is None:
            return False
        if accumulator is None:
            with self._lock:
                error = self._reply_metrics_error or "reply_accumulator_unavailable"
                self._reply_metrics_error = error
                self._reply_metrics_accepting = False
                self._reply_metrics_summary = {
                    "schema_version": "experiment_reply_v2",
                    "formula_version": "reply_timing_v2_semantic_body_recovery_excluded_v1",
                    "path": "",
                    "partial_path": "",
                    "sha256": "",
                    "row_count": 0,
                    "last_good_rows": 0,
                    "partial_integrity": "none",
                    "dropped_events": self._reply_metrics_dropped_events,
                    "error": error,
                    "promotion_state": "blocked",
                    "reply_metrics_durability": "finalize_only",
                    "reply_state_limit": _REPLY_STATE_LIMIT,
                    "reply_actions_per_reply_limit": _REPLY_ACTION_LIMIT,
                    "reply_event_bytes_limit": self.config.reply_event_bytes_limit,
                    "reply_metadata_bytes_limit": self.config.reply_metadata_bytes_limit,
                    "reply_metadata_bytes_accepted": self._reply_metrics_metadata_bytes,
                    "capture_stopped_monotonic_ns": self._capture_stopped_ns,
                }
            return False
        with self._lock:
            cutoff_ns = self._capture_stopped_ns or max(
                self._started_ns, self._monotonic_ns()
            )
            systemic_error = self._reply_metrics_error
            dropped = self._reply_metrics_dropped_events
        capture_span_ns = max(0, cutoff_ns - self._started_ns)
        capture_end_sample = math.ceil(
            capture_span_ns * self.config.sample_rate / 1e9
        )
        capture_end_frame = math.ceil(
            capture_span_ns * self.config.fps / 1e9
        )
        # The MP4 amix uses duration=longest.  Its executable common timeline
        # ends at the earlier of video EOF and the longer audio-track EOF.
        artifact_audio_end_sample = max(
            int(user_track_end_sample), int(assistant_track_end_sample)
        )
        artifact_video_end_frame = int(self._video_frames_written)
        wall_clock_step_detected = (
            abs(self._wall_clock_offset_change_ns)
            > _WALL_CLOCK_STEP_THRESHOLD_NS
        )
        mapping_uncertainty_ns = (
            self._wall_anchor_uncertainty_ns
            + abs(self._wall_clock_offset_change_ns)
            + self._stop_wall_anchor_uncertainty_ns
        )
        mapping_uncertainty_exceeds_threshold = (
            mapping_uncertainty_ns > _WALL_CLOCK_STEP_THRESHOLD_NS
        )
        partial = directory / "replies.partial.jsonl"
        fd: int | None = None
        digest = hashlib.sha256()
        last_good_rows = 0
        prepare_error = ""
        partial_integrity = "none"
        try:
            rows = accumulator.rows(
                recording_end_sample=artifact_audio_end_sample,
                recording_end_frame=artifact_video_end_frame,
                user_track_end_sample=user_track_end_sample,
                assistant_track_end_sample=assistant_track_end_sample,
                capture_end_sample=capture_end_sample,
                capture_end_frame=capture_end_frame,
                wall_clock_step_detected=wall_clock_step_detected,
                stop_wall_anchor_uncertainty_ns=(
                    self._stop_wall_anchor_uncertainty_ns
                ),
                wall_clock_offset_change_ns=(
                    self._wall_clock_offset_change_ns
                ),
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(partial, flags, 0o600)
            os.fchmod(fd, 0o600)
            partial_integrity = "jsonl_prefix"
            for row in rows:
                row["recording_timeline"] = {
                    "start_anchor": {
                        "monotonic_before_ns": self._start_anchor_before_ns,
                        "monotonic_midpoint_ns": self._started_ns,
                        "monotonic_after_ns": self._start_anchor_after_ns,
                        "wall_unix": self._started_utc,
                        "uncertainty_ns": self._wall_anchor_uncertainty_ns,
                    },
                    "stop_anchor": {
                        "monotonic_before_ns": self._stop_anchor_before_ns,
                        "monotonic_midpoint_ns": cutoff_ns,
                        "monotonic_after_ns": self._stop_anchor_after_ns,
                        "wall_unix": self._capture_stopped_wall,
                        "uncertainty_ns": self._stop_wall_anchor_uncertainty_ns,
                    },
                    "wall_clock_offset_change_ns": self._wall_clock_offset_change_ns,
                    "wall_clock_step_threshold_ns": _WALL_CLOCK_STEP_THRESHOLD_NS,
                    "wall_clock_step_detected": wall_clock_step_detected,
                    "mapping_uncertainty_ns": mapping_uncertainty_ns,
                    "mapping_uncertainty_basis": "start_anchor_half_span_plus_abs_offset_change_plus_stop_anchor_half_span",
                    "mapping_uncertainty_exceeds_threshold": (
                        mapping_uncertainty_exceeds_threshold
                    ),
                    "started_monotonic_ns": self._started_ns,
                    "started_wall": self._started_utc,
                    "wall_anchor_uncertainty_ns": self._wall_anchor_uncertainty_ns,
                    "capture_stopped_monotonic_ns": cutoff_ns,
                    "capture_end_sample_exclusive": capture_end_sample,
                    "capture_end_frame_exclusive": capture_end_frame,
                    "artifact_audio_end_sample_exclusive": artifact_audio_end_sample,
                    "artifact_video_end_frame_exclusive": artifact_video_end_frame,
                    "video_track_end_frame_exclusive": self._video_frames_written,
                    "user_track_end_sample_exclusive": user_track_end_sample,
                    "assistant_track_end_sample_exclusive": assistant_track_end_sample,
                    "sample_rate": self.config.sample_rate,
                    "fps": self.config.fps,
                }
                if systemic_error or dropped:
                    quality = row.setdefault("quality", {})
                    reasons = quality.setdefault("reasons", [])
                    token = systemic_error or "reply_event_dropped"
                    if token not in reasons:
                        reasons.append(token)
                    quality["valid"] = False
                    quality["metrics_incomplete"] = True
                payload = json.dumps(
                    row, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8") + b"\n"
                row_offset = os.lseek(fd, 0, os.SEEK_CUR)
                try:
                    self._write_all(fd, payload)
                except Exception:
                    # A short write may leave an invalid second JSON line. Keep
                    # only the last known-complete prefix for forensic recovery.
                    try:
                        os.ftruncate(fd, row_offset)
                        try:
                            os.fsync(fd)
                        except OSError:
                            pass
                        partial_integrity = "jsonl_prefix"
                    except OSError:
                        partial_integrity = "unknown"
                    raise
                digest.update(payload)
                last_good_rows += 1
            partial_integrity = "complete_jsonl"
            os.fsync(fd)
            os.close(fd)
            fd = None
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as exc:
            prepare_error = f"reply_sidecar_error:{type(exc).__name__}"[:160]
            if fd is not None:
                try:
                    os.fsync(fd)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
        combined_error = systemic_error or prepare_error
        if dropped and not combined_error:
            combined_error = "reply_event_dropped"
        exists = partial.is_file()
        reported_good_rows = last_good_rows if partial_integrity != "unknown" else 0
        summary = {
            "schema_version": "experiment_reply_v2",
            "formula_version": "reply_timing_v2_semantic_body_recovery_excluded_v1",
            "path": "",
            "partial_path": "replies.partial.jsonl" if exists else "",
            "sha256": (
                digest.hexdigest()
                if exists and partial_integrity != "unknown" else ""
            ),
            "row_count": reported_good_rows,
            "last_good_rows": reported_good_rows,
            "partial_integrity": partial_integrity if exists else "none",
            "dropped_events": dropped,
            "error": combined_error,
            "promotion_state": (
                "prepared" if exists and not combined_error else
                "error" if prepare_error else
                "blocked" if exists else "none"
            ),
            "wall_anchor_uncertainty_ns": self._wall_anchor_uncertainty_ns,
            "stop_wall_anchor_uncertainty_ns": self._stop_wall_anchor_uncertainty_ns,
            "capture_stopped_monotonic_ns": cutoff_ns,
            "capture_stopped_wall": self._capture_stopped_wall,
            "wall_clock_offset_change_ns": self._wall_clock_offset_change_ns,
            "wall_clock_step_threshold_ns": _WALL_CLOCK_STEP_THRESHOLD_NS,
            "wall_clock_step_detected": wall_clock_step_detected,
            "mapping_uncertainty_ns": mapping_uncertainty_ns,
            "mapping_uncertainty_basis": "start_anchor_half_span_plus_abs_offset_change_plus_stop_anchor_half_span",
            "mapping_uncertainty_exceeds_threshold": (
                mapping_uncertainty_exceeds_threshold
            ),
            "reply_metrics_durability": "finalize_only",
            "reply_state_limit": _REPLY_STATE_LIMIT,
            "reply_actions_per_reply_limit": _REPLY_ACTION_LIMIT,
            "reply_event_bytes_limit": self.config.reply_event_bytes_limit,
            "reply_metadata_bytes_limit": self.config.reply_metadata_bytes_limit,
            "reply_metadata_bytes_accepted": self._reply_metrics_metadata_bytes,
            "artifact_audio_end_sample_exclusive": artifact_audio_end_sample,
            "artifact_video_end_frame_exclusive": artifact_video_end_frame,
            "capture_end_sample_exclusive": capture_end_sample,
            "capture_end_frame_exclusive": capture_end_frame,
        }
        with self._lock:
            if prepare_error:
                self._reply_metrics_error = prepare_error
                self._reply_metrics_accepting = False
            self._reply_metrics_summary = summary
        return summary["promotion_state"] == "prepared"

    def _write_replies_sidecar(
        self,
        accumulator: Any | None,
        *,
        user_track_end_sample: int,
        assistant_track_end_sample: int,
    ) -> bool:
        """Compatibility name: prepare only; publication follows successful mux."""
        return self._prepare_replies_sidecar(
            accumulator,
            user_track_end_sample=user_track_end_sample,
            assistant_track_end_sample=assistant_track_end_sample,
        )

    def _mark_replies_unpublished(self, reason: str) -> None:
        with self._lock:
            summary = dict(self._reply_metrics_summary)
            if summary.get("promotion_state") == "prepared":
                summary["promotion_state"] = "blocked"
                summary["error"] = str(reason)[:160]
                self._reply_metrics_summary = summary

    def _promote_replies_sidecar(self) -> bool:
        directory = self._recording_dir
        if directory is None:
            return False
        partial = directory / "replies.partial.jsonl"
        final = directory / "replies.jsonl"
        with self._lock:
            summary = dict(self._reply_metrics_summary)
        if (
            summary.get("promotion_state") != "prepared"
            or summary.get("error")
            or summary.get("dropped_events")
        ):
            return False
        promoted = False
        try:
            os.replace(partial, final)
            promoted = True
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            summary["path"] = "replies.jsonl"
            summary["partial_path"] = ""
            summary["promotion_state"] = "published"
            with self._lock:
                self._reply_metrics_summary = summary
            return True
        except Exception as exc:
            rollback_ok = not promoted
            if promoted:
                try:
                    os.replace(final, partial)
                    directory_fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    rollback_ok = True
                except Exception:
                    rollback_ok = False
            summary["error"] = f"reply_sidecar_promote_error:{type(exc).__name__}"[:160]
            summary["promotion_state"] = "error" if rollback_ok else "uncertain"
            summary["path"] = ""
            summary["partial_path"] = (
                "replies.partial.jsonl"
                if rollback_ok and partial.is_file() else ""
            )
            with self._lock:
                self._reply_metrics_error = summary["error"]
                self._reply_metrics_summary = summary
            return False

    def _ack_reply_media_placement(
        self,
        reply_key: Any,
        *,
        media_kind: str,
        media_seq: Any,
        face_source_seq: Any = None,
    ) -> None:
        """Fail-open recorder-worker acknowledgement; never touches live media."""
        if (
            reply_key is None
            or isinstance(media_seq, bool)
            or not isinstance(media_seq, int)
            or media_seq < 1
        ):
            return
        try:
            callback = getattr(self._reply_hub, "mark_media_packet_placed", None)
            if callable(callback):
                callback(
                    reply_key,
                    media_kind=media_kind,
                    media_seq=media_seq,
                    face_source_seq=face_source_seq,
                    observed_ns=self._monotonic_ns(),
                )
        except Exception:
            pass

    def _worker_main(self) -> None:
        process = None
        stderr_file = None
        user_writer = None
        assistant_writer = None
        raw_video = None
        error: BaseException | None = None
        accumulator = self._new_reply_accumulator()
        self._reply_accumulator = accumulator
        try:
            directory = self._recording_dir
            if directory is None:
                raise RuntimeError("recording directory was not initialized")
            process, stderr_file, raw_video = self._open_ffmpeg(directory)
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            user_path = directory / "user.partial.wav"
            assistant_path = directory / "assistant.partial.wav"
            user_writer = _PcmTimelineWriter(
                user_path,
                t0_ns=self._started_ns,
                sample_rate=self.config.sample_rate,
                timeline="continuous_capture",
                max_gap_samples=round(
                    self.config.sample_rate * self.config.max_audio_gap_seconds
                ),
            )
            assistant_writer = _PcmTimelineWriter(
                assistant_path,
                t0_ns=self._started_ns,
                sample_rate=self.config.sample_rate,
                timeline="media_pts",
                max_gap_samples=round(
                    self.config.sample_rate * self.config.max_audio_gap_seconds
                ),
            )
            for path in (user_path, assistant_path):
                os.chmod(path, 0o600)
            with self._lock:
                if (
                    self._state == "starting"
                    and not self._stop_event.is_set()
                    and not self._error_code
                ):
                    self._accepting = True
                    self._state = "recording"
                    self._message = "Recording"
            self._ready_event.set()

            last_frame = None
            next_disk_check = 0.0
            while True:
                did_work = self._drain_reply_events(accumulator)
                video = self._drain_one(self._video_queue)
                if video is not None:
                    video_result = self._write_video_packet(process, video, last_frame)
                    last_frame = video_result.last_frame
                    if (
                        accumulator is not None
                        and video_result.packet_frame_index is not None
                        and video.reply_key is not None
                        and video.face_source_seq is not None
                    ):
                        try:
                            accumulator.observe_face(
                                video.reply_key,
                                video_result.packet_frame_index,
                                video.face_source_seq,
                                captured_ns=video.captured_ns,
                                pts=video.pts,
                                time_base_num=video.time_base_num,
                                time_base_den=video.time_base_den,
                                media_seq=video.media_seq,
                            )
                        except Exception:
                            with self._lock:
                                self._reply_metrics_accepting = False
                                self._reply_metrics_error = "reply_accumulator_error"
                    if video_result.written:
                        self._ack_reply_media_placement(
                            video.reply_key,
                            media_kind="video",
                            media_seq=video.media_seq,
                            face_source_seq=video.face_source_seq,
                        )
                    did_work = True
                for target, writer, counter_name in (
                    (self._user_queue, user_writer, "_user_samples_written"),
                    (
                        self._assistant_queue,
                        assistant_writer,
                        "_assistant_samples_written",
                    ),
                ):
                    packet = self._drain_one(target)
                    if packet is None:
                        continue
                    with self._lock:
                        self._audio_queued_bytes = max(
                            0, self._audio_queued_bytes - len(packet.pcm16)
                        )
                    write_result = writer.write(packet)
                    if (
                        accumulator is not None
                        and target is self._assistant_queue
                        and packet.reply_key is not None
                        and write_result.payload_start_sample is not None
                        and write_result.payload_end_sample_exclusive is not None
                    ):
                        try:
                            accumulator.observe_audio(
                                packet.reply_key,
                                write_result.payload_start_sample,
                                write_result.payload_end_sample_exclusive,
                                captured_ns=packet.captured_ns,
                                media_time_ns=packet.media_time_ns,
                                pts=packet.pts,
                                time_base_num=packet.time_base_num,
                                time_base_den=packet.time_base_den,
                                dropped_before=packet.dropped_before,
                                media_seq=packet.media_seq,
                            )
                        except Exception:
                            with self._lock:
                                self._reply_metrics_accepting = False
                                self._reply_metrics_error = "reply_accumulator_error"
                    with self._lock:
                        setattr(
                            self,
                            counter_name,
                            getattr(self, counter_name)
                            + write_result.bytes_written // 2,
                        )
                    if target is self._assistant_queue:
                        self._ack_reply_media_placement(
                            packet.reply_key,
                            media_kind="audio",
                            media_seq=packet.media_seq,
                        )
                    did_work = True

                if process.poll() is not None:
                    raise RuntimeError("ffmpeg_exited")
                now = time.monotonic()
                if now >= next_disk_check:
                    next_disk_check = now + 1.0
                    if self._disk_usage(directory).free < self.config.minimum_free_bytes:
                        raise RuntimeError("disk_space_low")
                    if self._current_disk_bytes() > self.config.max_recording_bytes:
                        raise RuntimeError("byte_limit")
                if (
                    self._monotonic_ns() - self._started_ns
                    > self.config.max_duration_seconds * 1e9
                ):
                    raise RuntimeError("duration_limit")
                if self._stop_event.is_set() and all(
                    target.empty()
                    for target in (
                        self._video_queue,
                        self._user_queue,
                        self._assistant_queue,
                        self._reply_queue,
                    )
                ):
                    break
                if not did_work:
                    self._stop_event.wait(0.01)
        except BaseException as exc:
            error = exc
            try:
                self._fail(self._error_from_exception(exc))
            except Exception:
                pass
        finally:
            self._ready_event.set()
            for writer in (user_writer, assistant_writer):
                if writer is not None:
                    try:
                        writer.close()
                    except BaseException as exc:
                        error = error or exc
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.close()
                    process.wait(timeout=self.config.encoder_wait_seconds)
                    if process.returncode:
                        raise RuntimeError(f"ffmpeg_exit_{process.returncode}")
                except BaseException as exc:
                    error = error or exc
                    try:
                        process.kill()
                        process.wait(timeout=2.0)
                    except BaseException:
                        pass
                finally:
                    with self._process_lock:
                        if self._encoder_process is process:
                            self._encoder_process = None
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except BaseException:
                    pass

        try:
            if self._reply_hub is not None:
                self._reply_hub.set_event_sink(None)
        except Exception:
            pass
        self._drain_reply_events(accumulator, self.config.reply_event_queue_size)
        try:
            self._write_replies_sidecar(
                accumulator,
                user_track_end_sample=(user_writer.cursor if user_writer is not None else 0),
                assistant_track_end_sample=(
                    assistant_writer.cursor if assistant_writer is not None else 0
                ),
            )
        except Exception:
            pass

        with self._lock:
            already_failed = bool(self._error_code)
        if error is None and not already_failed:
            try:
                self._mux_final(raw_video)
            except BaseException as exc:
                error = exc
            else:
                self._promote_replies_sidecar()
        if error is not None or already_failed:
            self._mark_replies_unpublished(
                "media_recording_failed" if error is not None else "recording_failed"
            )
        self._finish(error)

    def _mux_final(self, raw_video: Path | None) -> None:
        directory = self._recording_dir
        if directory is None or raw_video is None or not raw_video.is_file():
            raise RuntimeError("video_stream_missing")
        user_path = directory / "user.partial.wav"
        assistant_path = directory / "assistant.partial.wav"
        partial_mp4 = directory / "session.partial.mp4"
        final_mp4 = directory / "session.mp4"
        # Remux temporarily coexists with the MKV and both WAV sources.  Reserve
        # enough space for an output no larger than those inputs while retaining
        # the operator-configured free-space floor.
        source_bytes = sum(
            path.stat().st_size for path in (raw_video, user_path, assistant_path)
        )
        if (
            self._disk_usage(directory).free
            < self.config.minimum_free_bytes + source_bytes
        ):
            try:
                partial_mp4.unlink()
            except OSError:
                pass
            raise RuntimeError("disk_space_low")
        argv = [
            self.config.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(raw_video),
            "-i",
            str(user_path),
            "-i",
            str(assistant_path),
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,volume=0.5[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(partial_mp4),
        ]
        process = None
        stderr_file = None
        try:
            try:
                partial_mp4.unlink()
            except FileNotFoundError:
                pass
            stderr_path = directory / "encoder.log"
            stderr_file = open(stderr_path, "ab", buffering=0)
            os.chmod(stderr_path, 0o600)
            with self._process_lock:
                if self._abort_event.is_set():
                    raise RuntimeError("finalization_aborted")
                process = self._popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    shell=False,
                    close_fds=True,
                )
                self._mux_process = process
                if self._abort_event.is_set():
                    try:
                        process.kill()
                    finally:
                        raise RuntimeError("finalization_aborted")
            deadline = time.monotonic() + self.config.finalize_timeout_seconds
            while process.poll() is None:
                if self._abort_event.is_set():
                    raise RuntimeError("finalization_aborted")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("final_mux_timeout")
                self._abort_event.wait(min(0.1, remaining))
            if process.returncode != 0:
                raise RuntimeError("final_mux_failed")
        except Exception:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2.0)
                except Exception:
                    try:
                        process.kill()
                        process.wait(timeout=2.0)
                    except Exception:
                        pass
            try:
                partial_mp4.unlink()
            except OSError:
                pass
            raise
        finally:
            if process is not None:
                with self._process_lock:
                    if self._mux_process is process:
                        self._mux_process = None
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except Exception:
                    pass
        os.chmod(partial_mp4, 0o600)
        os.replace(partial_mp4, final_mp4)
        os.chmod(final_mp4, 0o600)
        for partial, final_name in (
            (user_path, "user.wav"),
            (assistant_path, "assistant.wav"),
        ):
            final = directory / final_name
            os.replace(partial, final)
            os.chmod(final, 0o600)
        try:
            raw_video.unlink()
        except OSError:
            pass
        self._final_path = final_mp4

    def _finish(self, error: BaseException | None) -> None:
        with self._lock:
            self._mark_capture_stopped_unlocked()
            self._accepting = False
            self._discard_pending_media_unlocked()
            self._finished_ns = self._monotonic_ns()
            if error is None and not self._error_code:
                self._state = "completed"
                self._message = "Recording saved"
            else:
                self._state = "error"
                if not self._error_code:
                    assert error is not None
                    self._error_code = self._error_from_exception(error)
                self._message = "Recording stopped; live conversation is unaffected"
            snapshot = self._snapshot_unlocked(include_private=True)
        self._write_manifest(snapshot)
        self._finished_event.set()

    def _write_manifest(self, snapshot: dict[str, Any]) -> None:
        directory = self._recording_dir
        if directory is None:
            return
        manifest = directory / "manifest.json"
        partial = directory / "manifest.partial.json"
        safe = dict(snapshot)
        safe.pop("output_path", None)
        try:
            with open(partial, "x", encoding="utf-8") as handle:
                json.dump(safe, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
            os.chmod(partial, 0o600)
            os.replace(partial, manifest)
            os.chmod(manifest, 0o600)
        except Exception:
            pass

    def _snapshot_unlocked(self, *, include_private: bool = False) -> dict[str, Any]:
        duration = 0.0
        if self._started_ns:
            end = self._finished_ns or self._monotonic_ns()
            duration = max(0.0, (end - self._started_ns) / 1e9)
        body: dict[str, Any] = {
            "type": STATUS_TYPE,
            "state": self._state,
            "active": self._state in {"starting", "recording", "finalizing"},
            "session_id": self.session_id,
            "request_id": self.request_id,
            "stream_generation": self.stream_generation,
            "recording_id": self.recording_id,
            "message": self._message,
            "error_code": self._error_code,
            "duration_seconds": round(duration, 6),
            "dropped_video_frames": self._dropped_video_frames,
            "video_frames_written": self._video_frames_written,
            "user_samples_written": self._user_samples_written,
            "assistant_samples_written": self._assistant_samples_written,
        }
        if include_private:
            body.update(
                {
                    "stop_reason": self._stop_reason,
                    "started_at_unix": self._started_utc,
                    "media": {
                        "width": self.config.width,
                        "height": self.config.height,
                        "fps": self.config.fps,
                        "sample_rate": self.config.sample_rate,
                    },
                    "output_path": str(self._final_path or ""),
                    "reply_metrics": dict(self._reply_metrics_summary),
                }
            )
        return body

    def snapshot(self) -> dict[str, Any]:
        try:
            with self._lock:
                return self._snapshot_unlocked()
        except Exception:
            return {
                "type": STATUS_TYPE,
                "state": "error",
                "active": False,
                "session_id": self.session_id,
                "request_id": self.request_id,
                "stream_generation": self.stream_generation,
                "recording_id": self.recording_id,
                "message": "Recording status unavailable",
                "error_code": "status_unavailable",
            }


class RecordingSlotManager:
    """Server-wide encoder admission; a slot is held through finalization."""

    def __init__(self, maximum: int = 1) -> None:
        if maximum <= 0:
            raise ValueError("maximum active recordings must be positive")
        self.maximum = maximum
        self._lock = threading.Lock()
        self._active: dict[str, ExperimentRecorder] = {}

    def try_reserve(self, recorder: ExperimentRecorder) -> bool:
        with self._lock:
            for recording_id, active in tuple(self._active.items()):
                if active.is_finished():
                    del self._active[recording_id]
            current = self._active.get(recorder.recording_id)
            if current is recorder:
                return True
            if len(self._active) >= self.maximum:
                return False
            self._active[recorder.recording_id] = recorder
            return True

    def watch(self, recorder: ExperimentRecorder) -> None:
        def release_after_finish() -> None:
            try:
                recorder.wait_finished(None)
            finally:
                with self._lock:
                    if self._active.get(recorder.recording_id) is recorder:
                        del self._active[recorder.recording_id]

        threading.Thread(
            target=release_after_finish,
            name=f"recording-slot-{recorder.recording_id[:8]}",
            daemon=True,
        ).start()

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def stop_all(self, *, reason: str = "server_shutdown") -> None:
        with self._lock:
            active = tuple(self._active.values())
        for recorder in active:
            try:
                recorder.request_stop(
                    session_id=recorder.session_id,
                    recording_id=recorder.recording_id,
                    reason=reason,
                )
            except Exception:
                continue

    def abort_all(
        self,
        *,
        reason: str = "server_shutdown_timeout",
        force: bool = False,
    ) -> None:
        with self._lock:
            active = tuple(self._active.values())
        for recorder in active:
            recorder.abort_finalize(reason=reason, force=force)

    def wait_all(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            active = tuple(self._active.values())
        for recorder in active:
            if not recorder.wait_finished(max(0.0, deadline - time.monotonic())):
                return False
        return True


class SessionRecordingController:
    """Object-local owner that prevents stale sessions/generations crossing."""

    def __init__(
        self,
        config: RecordingConfig,
        *,
        session_id: str,
        recorder_factory: Callable[..., ExperimentRecorder] = ExperimentRecorder,
        slot_manager: RecordingSlotManager | None = None,
        reply_hub: Any | None = None,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self._factory = recorder_factory
        self._reply_hub = reply_hub
        self._slot_manager = slot_manager or RecordingSlotManager(
            config.max_active_recordings
        )
        self._lock = threading.RLock()
        self._active: ExperimentRecorder | None = None
        self._requests: dict[str, ExperimentRecorder] = {}
        self._stream_generation = 0
        self._closed = False

    def bind_generation(self, generation: int) -> None:
        recorder_to_stop = None
        with self._lock:
            if self._closed:
                return
            generation = int(generation)
            self._stream_generation = generation
            try:
                if self._reply_hub is not None and generation > 0:
                    self._reply_hub.bind_generation(generation)
            except Exception:
                pass
            active = self._active
            if active is not None and not active.set_stream_generation(generation):
                recorder_to_stop = active
        if recorder_to_stop is not None:
            recorder_to_stop.request_stop(
                session_id=self.session_id,
                recording_id=recorder_to_stop.recording_id,
                reason="generation_replaced",
            )

    def start(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            previous = self._requests.get(request_id)
            if previous is not None:
                return self._public_snapshot(previous)
            if self._closed:
                return self._error_snapshot(
                    request_id,
                    "recording_session_closed",
                    "WebRTC session is already closed",
                )
            if not self.config.enabled:
                return self._error_snapshot(
                    request_id, "recording_disabled", "Experiment recording is disabled"
                )
            if self._active is not None and not self._active.is_finished():
                return self._error_snapshot(
                    request_id,
                    "recording_already_active",
                    "Another experiment recording is active for this session",
                )
            if len(self._requests) >= 64:
                for old_request, old_recorder in tuple(self._requests.items()):
                    if old_recorder.is_finished() and old_recorder is not self._active:
                        del self._requests[old_request]
                    if len(self._requests) < 64:
                        break
            if len(self._requests) >= 64:
                return self._error_snapshot(
                    request_id,
                    "recording_history_full",
                    "Recording history is temporarily full",
                )
            try:
                recorder_kwargs: dict[str, Any] = {
                    "session_id": self.session_id,
                    "request_id": request_id,
                    "stream_generation": self._stream_generation,
                }
                if self._reply_hub is not None:
                    recorder_kwargs["reply_hub"] = self._reply_hub
                recorder = self._factory(self.config, **recorder_kwargs)
            except Exception as exc:
                return self._error_snapshot(
                    request_id,
                    ExperimentRecorder._error_from_exception(exc),
                    "Experiment recorder could not be initialized",
                )
            try:
                reserved = self._slot_manager.try_reserve(recorder)
            except Exception:
                reserved = False
            if not reserved:
                return self._error_snapshot(
                    request_id,
                    "recording_capacity_reached",
                    "Another experiment recording is active",
                )
            self._requests[request_id] = recorder
            self._active = recorder
        try:
            recorder.start()
        except Exception as exc:
            recorder.fail(ExperimentRecorder._error_from_exception(exc))
        self._slot_manager.watch(recorder)
        snapshot = recorder.snapshot()
        if snapshot["state"] == "error" and recorder.is_finished():
            with self._lock:
                if self._active is recorder:
                    self._active = None
        return snapshot

    @staticmethod
    def _public_snapshot(recorder: ExperimentRecorder) -> dict[str, Any]:
        snapshot = recorder.snapshot()
        if snapshot.get("state") == "idle" and recorder.is_finished():
            # request_stop publishes its terminal state before setting the
            # finished event.  If those two reads straddled that transition,
            # discard the stale idle snapshot and return the terminal one.
            return recorder.snapshot()
        if snapshot.get("state") == "idle":
            # The first caller has reserved/published this exact job but has not
            # yet entered ExperimentRecorder.start(). All public reads expose a
            # traceable pending job, never a terminal-looking idle recorder.
            snapshot = dict(snapshot)
            snapshot["state"] = "starting"
            snapshot["active"] = True
            snapshot["message"] = "Preparing recorder"
        return snapshot

    def _error_snapshot(self, request_id: str, code: str, message: str) -> dict[str, Any]:
        return {
            "type": STATUS_TYPE,
            "state": "error",
            "active": False,
            "session_id": self.session_id,
            "request_id": request_id,
            "stream_generation": self._stream_generation,
            "recording_id": "",
            "message": message,
            "error_code": code,
        }

    def active_offer(self) -> tuple[ExperimentRecorder, str, int] | None:
        with self._lock:
            recorder = self._active
            if recorder is None or not recorder.is_recording():
                return None
            return recorder, recorder.recording_id, recorder.stream_generation

    def _recorder_for_offer(
        self,
        offer: tuple[ExperimentRecorder, str, int] | None,
    ) -> ExperimentRecorder | None:
        if offer is None:
            return None
        try:
            recorder, recording_id, generation = offer
        except (TypeError, ValueError):
            return None
        with self._lock:
            if (
                self._active is not recorder
                or recorder.recording_id != recording_id
                or recorder.stream_generation != int(generation)
            ):
                return None
        return recorder

    def fail_offer(
        self,
        offer: tuple[ExperimentRecorder, str, int] | None,
        error_code: str,
    ) -> bool:
        """Fail only the exact recorder that observed a tap-side error."""

        try:
            recorder = self._recorder_for_offer(offer)
            if recorder is None or offer is None:
                return False
            _owned_recorder, recording_id, generation = offer
            return recorder._fail_current_offer(
                error_code,
                session_id=self.session_id,
                recording_id=recording_id,
                stream_generation=generation,
            )
        except Exception:
            return False

    def fail_active(self, error_code: str) -> bool:
        try:
            with self._lock:
                recorder = self._active
            if recorder is None or not recorder.is_recording():
                return False
            return recorder.fail(error_code)
        except Exception:
            return False

    def current_generation(self) -> int:
        with self._lock:
            return self._stream_generation

    def offer_video(self, frame: Any, *, captured_ns: int | None = None) -> bool:
        try:
            offer = self.active_offer()
            return self.offer_video_for(offer, frame, captured_ns=captured_ns)
        except Exception:
            return False

    def offer_video_for(
        self,
        offer: tuple[ExperimentRecorder, str, int] | None,
        frame: Any,
        *,
        captured_ns: int | None = None,
        reply_key: ReplyKey | None = None,
        face_source_seq: int | None = None,
        pts: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        media_seq: int | None = None,
    ) -> bool:
        try:
            recorder = self._recorder_for_offer(offer)
            if recorder is None or offer is None:
                return False
            _owned_recorder, recording_id, generation = offer
            return recorder.push_video_bgr(
                frame,
                session_id=self.session_id,
                recording_id=recording_id,
                stream_generation=generation,
                captured_ns=captured_ns,
                reply_key=reply_key,
                face_source_seq=face_source_seq,
                pts=pts,
                time_base_num=time_base_num,
                time_base_den=time_base_den,
                media_seq=media_seq,
            )
        except Exception:
            return False

    def offer_user_pcm16(
        self,
        pcm16: bytes,
        *,
        generation: int,
        captured_ns: int | None = None,
    ) -> bool:
        try:
            offer = self.active_offer()
        except Exception:
            return False
        return self.offer_user_pcm16_for(
            offer,
            pcm16,
            generation=generation,
            captured_ns=captured_ns,
        )

    def offer_user_pcm16_for(
        self,
        offer: tuple[ExperimentRecorder, str, int] | None,
        pcm16: bytes,
        *,
        generation: int,
        captured_ns: int | None = None,
    ) -> bool:
        return self._offer_audio_for(
            offer,
            "user",
            pcm16,
            generation=generation,
            captured_ns=captured_ns,
        )

    def offer_assistant_pcm16(
        self,
        pcm16: bytes,
        *,
        generation: int,
        captured_ns: int | None = None,
        media_time_ns: int | None = None,
    ) -> bool:
        try:
            offer = self.active_offer()
        except Exception:
            return False
        return self.offer_assistant_pcm16_for(
            offer,
            pcm16,
            generation=generation,
            captured_ns=captured_ns,
            media_time_ns=media_time_ns,
        )

    def offer_assistant_pcm16_for(
        self,
        offer: tuple[ExperimentRecorder, str, int] | None,
        pcm16: bytes,
        *,
        generation: int,
        captured_ns: int | None,
        media_time_ns: int | None = None,
        reply_key: ReplyKey | None = None,
        pts: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        dropped_before: int = 0,
        media_seq: int | None = None,
    ) -> bool:
        return self._offer_audio_for(
            offer,
            "assistant",
            pcm16,
            generation=generation,
            captured_ns=captured_ns,
            media_time_ns=media_time_ns,
            reply_key=reply_key,
            pts=pts,
            time_base_num=time_base_num,
            time_base_den=time_base_den,
            dropped_before=dropped_before,
            media_seq=media_seq,
        )

    def _offer_audio_for(
        self,
        offer: tuple[ExperimentRecorder, str, int] | None,
        kind: str,
        pcm16: bytes,
        *,
        generation: int,
        captured_ns: int | None,
        media_time_ns: int | None = None,
        reply_key: ReplyKey | None = None,
        pts: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        dropped_before: int = 0,
        media_seq: int | None = None,
    ) -> bool:
        try:
            recorder = self._recorder_for_offer(offer)
            if recorder is None or offer is None:
                return False
            _owned_recorder, recording_id, active_generation = offer
            if int(generation) != active_generation:
                return False
            push = (
                recorder.push_user_pcm16
                if kind == "user"
                else recorder.push_assistant_pcm16
            )
            return push(
                pcm16,
                session_id=self.session_id,
                recording_id=recording_id,
                stream_generation=active_generation,
                sample_rate=self.config.sample_rate,
                channels=1,
                captured_ns=captured_ns,
                media_time_ns=media_time_ns,
                reply_key=reply_key,
                pts=pts,
                time_base_num=time_base_num,
                time_base_den=time_base_den,
                dropped_before=dropped_before,
                media_seq=media_seq,
            )
        except Exception:
            return False

    def stop(self, recording_id: str, *, reason: str = "operator_stop") -> dict[str, Any]:
        with self._lock:
            recorder = self._active
            if recorder is None or recorder.recording_id != recording_id:
                return self._error_snapshot(
                    "", "recording_owner_mismatch", "Recording ownership token is stale"
                )
        recorder.request_stop(
            session_id=self.session_id,
            recording_id=recording_id,
            reason=reason,
        )
        return self._public_snapshot(recorder)

    def stop_generation(self, generation: int, *, reason: str) -> bool:
        with self._lock:
            recorder = self._active
            if recorder is None or recorder.stream_generation != int(generation):
                return False
        return recorder.request_stop(
            session_id=self.session_id,
            recording_id=recorder.recording_id,
            stream_generation=generation,
            reason=reason,
        )

    def close_stream(self) -> None:
        with self._lock:
            self._closed = True
            recorder = self._active
        if recorder is not None:
            recorder.request_stop(
                session_id=self.session_id,
                recording_id=recorder.recording_id,
                reason="stream_closed",
            )

    def status(self, recording_id: str) -> dict[str, Any]:
        with self._lock:
            recorder = self._active
            if recorder is None or recorder.recording_id != recording_id:
                for candidate in self._requests.values():
                    if candidate.recording_id == recording_id:
                        recorder = candidate
                        break
            if recorder is None:
                return self._error_snapshot(
                    "", "recording_not_found", "Recording was not found"
                )
        return self._public_snapshot(recorder)

    def wait_all(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            recorders = tuple(self._requests.values())
        for recorder in recorders:
            remaining = max(0.0, deadline - time.monotonic())
            if not recorder.wait_finished(remaining):
                return False
        return True
