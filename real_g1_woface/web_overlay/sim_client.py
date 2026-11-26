from __future__ import annotations

import asyncio
from dataclasses import dataclass
import io
import json
import threading
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import OverlayConfig


class SimServiceError(RuntimeError):
    """Simulator transport error with an optional preserved HTTP status."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CachedFrame:
    body: bytes
    content_type: str
    received_monotonic: float


class SimServiceClient:
    """Small stdlib HTTP client with a non-blocking latest-frame cache.

    Network work is performed in worker threads. A WebRTC ``recv`` call therefore
    never waits for MuJoCo; it consumes the most recent decoded frame instead.
    """

    def __init__(self, config: OverlayConfig):
        self.config = config
        self._lock = threading.Lock()
        self._frame: CachedFrame | None = None
        self._last_error: str | None = None
        self._frame_successes = 0
        self._action_successes = 0
        self._poll_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        data = None
        headers = {"Accept": "application/json, image/png, image/jpeg"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self.config.endpoint(path), data=data, headers=headers, method=method
        )
        try:
            with urlopen(request, timeout=self.config.http_timeout) as response:
                return response.read(), response.headers.get_content_type()
        except HTTPError as exc:
            detail = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
                error_payload = json.loads(error_body)
                if isinstance(error_payload, dict):
                    detail = str(
                        error_payload.get("detail")
                        or error_payload.get("error")
                        or error_body
                    )
                else:
                    detail = error_body
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise SimServiceError(
                f"{method} {path}: HTTP {exc.code} {exc.reason}{suffix}",
                status_code=int(exc.code),
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise SimServiceError(f"{method} {path}: {exc}") from exc

    def get_json_sync(self, path: str) -> dict[str, Any]:
        body, _ = self._request("GET", path)
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SimServiceError(f"GET {path}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise SimServiceError(f"GET {path}: expected a JSON object")
        return value

    def health_sync(self) -> dict[str, Any]:
        return self.get_json_sync(self.config.health_path)

    def state_sync(self) -> dict[str, Any]:
        return self.get_json_sync(self.config.state_path)

    def action_sync(self, action_id: str) -> dict[str, Any]:
        encoded = quote(str(action_id), safe="")
        return self.get_json_sync(f"/actions/{encoded}")

    def post_action_sync(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body, _ = self._request("POST", self.config.action_path, payload)
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SimServiceError("action response is not valid JSON") from exc
        if not isinstance(result, dict):
            raise SimServiceError("action response must be a JSON object")
        with self._lock:
            self._action_successes += 1
            self._last_error = None
        return result

    def trigger_sync(
        self,
        action_or_motion: Any,
        *,
        mode: str | None = None,
        speed: float | None = None,
        blend_duration: float | None = None,
        bridge_policy: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(action_or_motion, Mapping):
            raw = dict(action_or_motion)
            if "motion" in raw or "preset" in raw or "release" in raw:
                payload: dict[str, Any] = raw
            elif "joints" in raw and "frames" in raw:
                payload = {"motion": raw}
            else:
                payload = raw
        else:
            payload = {"action": str(action_or_motion)}

        if "release" not in payload:
            payload.setdefault("mode", mode or self.config.default_action_mode)
            payload.setdefault("speed", speed or self.config.default_speed)
            payload.setdefault(
                "blend_duration",
                self.config.default_blend_duration
                if blend_duration is None
                else blend_duration,
            )
            if bridge_policy is not None:
                payload.setdefault("bridge_policy", bridge_policy)
        return self.post_action_sync(payload)

    def wait_for_action_sync(
        self,
        action_id: str,
        *,
        timeout: float = 45.0,
        poll_interval: float = 0.05,
    ) -> dict[str, Any]:
        """Wait for one submitted action without blocking the asyncio loop.

        The untouched robot demo invokes its bridge in ``asyncio.to_thread`` and
        expects that bridge call to return only after physical playback finishes.
        Polling here preserves that contract while the lower-level atom API stays
        non-blocking for the future streaming planner.
        """

        deadline = time.monotonic() + max(float(timeout), 0.1)
        while time.monotonic() < deadline:
            action = self.action_sync(action_id)
            status = str(action.get("status", ""))
            if status in {"completed", "canceled", "replaced", "failed"}:
                if status != "completed":
                    raise SimServiceError(
                        f"simulator action {action_id} ended with status {status}"
                    )
                return action
            time.sleep(max(float(poll_interval), 0.01))
        raise SimServiceError(
            f"timed out after {timeout:.1f}s waiting for simulator action {action_id}"
        )

    def trigger_legacy_sync(
        self,
        action_or_motion: Any,
        *,
        mode: str | None = None,
        speed: float | None = None,
        blend_duration: float | None = None,
        bridge_policy: str | None = None,
        timeout: float = 45.0,
    ) -> dict[str, Any]:
        """Submit and wait, matching the blocking contract of the real G1 CLI."""

        result = self.trigger_sync(
            action_or_motion,
            mode=mode,
            speed=speed,
            blend_duration=blend_duration,
            bridge_policy=bridge_policy,
        )
        action_id = result.get("id")
        if not action_id:
            return result
        completion = self.wait_for_action_sync(str(action_id), timeout=timeout)
        return {**result, "completion": completion}

    def release_sync(self) -> dict[str, Any]:
        return self.post_action_sync({"release": True})

    def stop_action_sync(self) -> dict[str, Any]:
        body, _ = self._request("DELETE", self.config.action_delete_path)
        if not body:
            return {"stopped": True}
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SimServiceError("stop response is not valid JSON") from exc
        return result if isinstance(result, dict) else {"result": result}

    def fetch_frame_sync(self) -> CachedFrame:
        body, content_type = self._request("GET", self.config.frame_path)
        if not body:
            raise SimServiceError("simulator returned an empty frame")
        frame = CachedFrame(body, content_type, time.monotonic())
        with self._lock:
            self._frame = frame
            self._frame_successes += 1
            self._last_error = None
        return frame

    def latest_frame(self) -> CachedFrame | None:
        with self._lock:
            return self._frame

    def cache_status(self) -> dict[str, Any]:
        with self._lock:
            frame = self._frame
            age = None if frame is None else time.monotonic() - frame.received_monotonic
            return {
                "frame_ready": frame is not None,
                "frame_age_seconds": age,
                "frame_fresh": age is not None
                and age <= self.config.stale_frame_seconds,
                "frame_successes": self._frame_successes,
                "action_successes": self._action_successes,
                "last_error": self._last_error,
            }

    def ensure_frame_polling(self) -> None:
        if self._poll_task is not None and not self._poll_task.done():
            return
        asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._poll_task = asyncio.create_task(
            self._poll_frames(), name="mujoco-frame-poller"
        )

    async def _poll_frames(self) -> None:
        interval = 1.0 / max(self.config.frame_poll_hz, 1.0)
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                await asyncio.to_thread(self.fetch_frame_sync)
            except Exception as exc:  # keep WebRTC alive during simulator restarts
                with self._lock:
                    self._last_error = str(exc)
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), remaining)
                except asyncio.TimeoutError:
                    pass

    async def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._poll_task is not None:
            await self._poll_task
        self._poll_task = None


def decode_cached_frame(frame: CachedFrame):
    """Decode PNG/JPEG bytes into BGR numpy data using the existing PyAV stack."""

    import av

    with av.open(io.BytesIO(frame.body), mode="r") as container:
        decoded = next(container.decode(video=0), None)
    if decoded is None:
        raise SimServiceError(f"could not decode {frame.content_type} frame")
    return decoded.to_ndarray(format="bgr24")
