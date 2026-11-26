from __future__ import annotations

import asyncio
from fractions import Fraction
import math
import time
from typing import Any, Awaitable, Callable

from .config import OverlayConfig
from .sim_client import SimServiceClient, decode_cached_frame


# The conversation camera and the released semantic catalog are fixed: the
# latter drives only the 14 arm joints, while the head is rigidly attached to
# the torso.  The full 640x480 MuJoCo frame is aspect-fitted into the existing
# 1280x480 stream, so these coordinates are the approved upper-body overlay
# scaled back to the native full-body camera view.
_FACE_VIDEO_SOURCE_CROP = (109 / 512, 0.0, 403 / 512, 400 / 512)
_FACE_VIDEO_CENTER_X = 0.5
_FACE_VIDEO_BOTTOM_Y = 102 / 480
_FACE_VIDEO_WIDTH = 47 / 480
_FACE_VIDEO_HEIGHT = 64 / 480


def _numpy():
    import numpy as np

    return np


def _is_bgr_image(image: Any) -> bool:
    return bool(
        image is not None
        and getattr(image, "ndim", None) == 3
        and image.shape[2] == 3
        and image.shape[0] > 0
        and image.shape[1] > 0
    )


def _resize_bgr(image: Any, width: int, height: int):
    np = _numpy()
    src_h, src_w = image.shape[:2]
    try:
        import cv2

        shrinking = width < src_w or height < src_h
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(image, (width, height), interpolation=interpolation)
    except ImportError:
        x_idx = np.minimum(
            (np.arange(width, dtype=np.float64) * src_w / width).astype(int),
            src_w - 1,
        )
        y_idx = np.minimum(
            (np.arange(height, dtype=np.float64) * src_h / height).astype(int),
            src_h - 1,
        )
        return image[y_idx[:, None], x_idx[None, :]]


def letterbox_bgr(image: Any, width: int, height: int):
    """Resize with nearest-neighbour sampling and preserve the aspect ratio."""

    np = _numpy()
    if not _is_bgr_image(image):
        return np.zeros((height, width, 3), dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, min(width, round(src_w * scale)))
    dst_h = max(1, min(height, round(src_h * scale)))
    resized = _resize_bgr(image, dst_w, dst_h)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - dst_h) // 2
    x0 = (width - dst_w) // 2
    panel[y0 : y0 + dst_h, x0 : x0 + dst_w] = resized
    return panel


def _paste_face_video(face_bgr: Any, sim_bgr: Any):
    """Join a cropped face-and-neck portrait to the native robot neck."""

    np = _numpy()
    if not _is_bgr_image(sim_bgr):
        return sim_bgr
    output = sim_bgr.copy()
    if not _is_bgr_image(face_bgr):
        return output

    height, width = output.shape[:2]
    oval_width = max(
        1,
        min(width, height, round(_FACE_VIDEO_WIDTH * height)),
    )
    oval_height = max(
        1,
        min(width, height, round(_FACE_VIDEO_HEIGHT * height)),
    )
    src_height, src_width = face_bgr.shape[:2]
    crop_x0 = max(
        0,
        min(src_width - 1, round(_FACE_VIDEO_SOURCE_CROP[0] * src_width)),
    )
    crop_y0 = max(
        0,
        min(src_height - 1, round(_FACE_VIDEO_SOURCE_CROP[1] * src_height)),
    )
    crop_x1 = max(
        crop_x0 + 1,
        min(src_width, round(_FACE_VIDEO_SOURCE_CROP[2] * src_width)),
    )
    crop_y1 = max(
        crop_y0 + 1,
        min(src_height, round(_FACE_VIDEO_SOURCE_CROP[3] * src_height)),
    )
    face_patch = _resize_bgr(
        face_bgr[crop_y0:crop_y1, crop_x0:crop_x1],
        oval_width,
        oval_height,
    )
    center_x = round(_FACE_VIDEO_CENTER_X * width)
    bottom_y = round(_FACE_VIDEO_BOTTOM_Y * height)
    x0 = max(0, min(width - oval_width, center_x - oval_width // 2))
    y0 = max(0, min(height - oval_height, bottom_y - oval_height))
    x = (
        np.arange(oval_width, dtype=np.float32) + 0.5 - oval_width / 2
    ) / (oval_width / 2)
    y = (
        np.arange(oval_height, dtype=np.float32) + 0.5 - oval_height / 2
    ) / (oval_height / 2)
    radius = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
    alpha = np.clip(
        (1.0 - radius) * (min(oval_width, oval_height) / 3.0) + 0.5,
        0.0,
        1.0,
    )[..., None]
    robot_patch = output[
        y0 : y0 + oval_height, x0 : x0 + oval_width
    ]
    blended = (
        face_patch.astype(np.float32) * alpha
        + robot_patch.astype(np.float32) * (1.0 - alpha)
    )
    output[y0 : y0 + oval_height, x0 : x0 + oval_width] = np.clip(
        blended, 0, 255
    ).astype(np.uint8)
    return output


def compose_bgr(face_bgr: Any, sim_bgr: Any, config: OverlayConfig):
    """Show the full robot and join a face portrait to its native neck.

    The output deliberately retains the established 2*panel_width by
    panel_height stream geometry.  If the simulator frame is unavailable, the
    legacy left-face/right-black fallback is preserved.
    """

    output_width = config.panel_width * 2
    output_height = config.panel_height
    if not _is_bgr_image(sim_bgr):
        np = _numpy()
        left = letterbox_bgr(face_bgr, config.panel_width, output_height)
        right = np.zeros((output_height, config.panel_width, 3), dtype=np.uint8)
        return np.concatenate((left, right), axis=1)

    robot = letterbox_bgr(sim_bgr, output_width, output_height)
    return _paste_face_video(face_bgr, robot)


def _video_frame_from_bgr(image: Any):
    import av

    return av.VideoFrame.from_ndarray(image, format="bgr24")


def compose_av_frame(source_frame: Any, client: SimServiceClient, config: OverlayConfig):
    """Compose one frame while preserving the FlashHead PTS/time base.

    Preserving the source timestamp is what keeps the combined video on the same
    WebRTC clock as the untouched Gemini audio track.
    """

    face_bgr = source_frame.to_ndarray(format="bgr24")
    sim_bgr = None
    cached = client.latest_frame()
    if cached is not None:
        age = time.monotonic() - cached.received_monotonic
        if age <= config.stale_frame_seconds:
            try:
                sim_bgr = decode_cached_frame(cached)
            except Exception:
                sim_bgr = None
    combined = compose_bgr(face_bgr, sim_bgr, config)
    output = _video_frame_from_bgr(combined)
    output.pts = getattr(source_frame, "pts", None)
    output.time_base = getattr(source_frame, "time_base", None)
    return output


try:
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
except ImportError:  # allows config/client tests without installing the full demo
    class MediaStreamError(Exception):  # type: ignore[no-redef]
        pass

    class MediaStreamTrack:  # type: ignore[no-redef]
        kind = "video"

        def stop(self) -> None:
            return None


class CompositeVideoTrack(MediaStreamTrack):
    """A fixed-rate composite track decoupled from the FlashHead producer.

    A background task consumes FlashHead frames and keeps only the newest BGR
    image.  ``recv`` runs on its own clock, so a temporarily empty upstream
    queue repeats that image while still showing the newest MuJoCo frame.  A
    genuine upstream end/error is propagated instead of being hidden forever.
    """

    kind = "video"
    video_clock_rate = 90_000

    def __init__(
        self,
        source: Any,
        client: SimServiceClient,
        config: OverlayConfig,
        *,
        stop_source: bool = True,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        reply_hub: Any | None = None,
    ) -> None:
        super().__init__()
        fps = float(config.composite_fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("composite_fps must be a positive finite number")

        self.source = source
        self.client = client
        self.config = config
        self._stop_source = stop_source
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._reply_hub = reply_hub
        self._latest_face_provenance: tuple[Any, int] | None = None
        self._fps = fps
        self._period = 1.0 / fps
        self._pts_per_frame = Fraction(self.video_clock_rate, 1) / Fraction(str(fps))
        self._time_base = Fraction(1, self.video_clock_rate)

        self._recv_lock = asyncio.Lock()
        self._face_ready = asyncio.Event()
        self._source_task: asyncio.Task[None] | None = None
        self._latest_face_bgr: Any = None
        self._source_ended = False
        self._terminal_error: Exception | None = None
        self._stopped = False

        self._timeline_start: float | None = None
        self._frame_index = 0
        self._pts_base: int | None = None
        self._source_pts_anchor: int | None = None

        self._decoded_sim_source: Any = None
        self._decoded_sim_bgr: Any = None

    def _ensure_started(self) -> None:
        if self._source_task is not None:
            return
        self.client.ensure_frame_polling()
        self._source_task = asyncio.create_task(
            self._read_source(), name="flashhead-face-reader"
        )

    def _remember_source_pts(self, frame: Any) -> None:
        if self._source_pts_anchor is not None:
            return
        pts = getattr(frame, "pts", None)
        time_base = getattr(frame, "time_base", None)
        numerator = getattr(time_base, "numerator", None)
        denominator = getattr(time_base, "denominator", None)
        if pts is None or numerator is None or not denominator:
            return
        try:
            seconds = Fraction(int(pts) * int(numerator), int(denominator))
            self._source_pts_anchor = round(seconds * self.video_clock_rate)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return

    async def _read_source(self) -> None:
        try:
            while not self._stopped:
                frame = await self.source.recv()
                if frame is None:
                    raise MediaStreamError
                self._remember_source_pts(frame)
                provenance = None
                try:
                    hub = self._reply_hub
                    tag = hub.take_frame(frame) if hub is not None else None
                    if tag is not None:
                        source_seq = hub.next_face_source_seq(tag.reply_key)
                        if source_seq is not None:
                            provenance = (
                                tag.reply_key, source_seq, tag.media_seq
                            )
                except Exception:
                    provenance = None
                face_bgr = frame.to_ndarray(format="bgr24")
                copy = getattr(face_bgr, "copy", None)
                self._latest_face_bgr = copy() if callable(copy) else face_bgr
                # Publish image and its exact source identity as one event-loop turn.
                self._latest_face_provenance = provenance
                self._face_ready.set()
        except asyncio.CancelledError:
            raise
        except (MediaStreamError, EOFError, StopAsyncIteration):
            self._terminal_error = MediaStreamError()
        except Exception as exc:
            self._terminal_error = exc
        finally:
            if not self._stopped:
                self._source_ended = True
            self._face_ready.set()

    def _raise_terminal(self) -> None:
        error = self._terminal_error
        if error is not None:
            raise error
        raise MediaStreamError

    async def _wait_for_tick(self) -> int:
        now = self._clock()
        if self._timeline_start is None:
            self._timeline_start = now
            self._pts_base = self._source_pts_anchor or 0

        assert self._timeline_start is not None
        index = self._frame_index
        target = self._timeline_start + index * self._period

        # When the loop is delayed, skip old deadlines rather than emitting a
        # burst of catch-up frames.  This keeps encoder load bounded.
        if now - target >= self._period:
            elapsed_ticks = math.floor(
                (now - self._timeline_start) / self._period + 1e-9
            )
            index = max(index, elapsed_ticks)
            target = self._timeline_start + index * self._period

        delay = target - now
        if delay > 0:
            await self._sleep(delay)

        self._frame_index = index + 1
        assert self._pts_base is not None
        return self._pts_base + round(index * self._pts_per_frame)

    def _latest_sim_frame(self):
        cached = self.client.latest_frame()
        if cached is None:
            return None
        age = self._clock() - cached.received_monotonic
        if age > self.config.stale_frame_seconds:
            return None
        if cached is not self._decoded_sim_source:
            self._decoded_sim_source = cached
            try:
                self._decoded_sim_bgr = decode_cached_frame(cached)
            except Exception:
                self._decoded_sim_bgr = None
        return self._decoded_sim_bgr

    async def recv(self):
        async with self._recv_lock:
            if self._stopped:
                self._raise_terminal()
            self._ensure_started()

            # Give the producer up to one output period to supply the very
            # first face.  Thereafter a missing source frame never gates output.
            if self._latest_face_bgr is None and not self._source_ended:
                try:
                    await asyncio.wait_for(
                        self._face_ready.wait(), timeout=self._period
                    )
                except asyncio.TimeoutError:
                    pass

            if self._stopped or self._source_ended:
                self._raise_terminal()
            pts = await self._wait_for_tick()
            if self._stopped or self._source_ended:
                self._raise_terminal()

            combined = compose_bgr(
                self._latest_face_bgr, self._latest_sim_frame(), self.config
            )
            output = _video_frame_from_bgr(combined)
            output.pts = pts
            output.time_base = self._time_base
            try:
                provenance = self._latest_face_provenance
                hub = self._reply_hub
                if provenance is not None and hub is not None:
                    key, source_seq, media_seq = provenance
                    hub.tag_frame(
                        output,
                        key,
                        media_kind="video",
                        face_source_seq=source_seq,
                        media_seq=media_seq,
                    )
            except Exception:
                pass
            return output

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._face_ready.set()
        if self._source_task is not None and not self._source_task.done():
            self._source_task.cancel()
        if self._stop_source:
            stop = getattr(self.source, "stop", None)
            if callable(stop):
                stop()
        super().stop()
