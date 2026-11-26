"""Small dependency-free HTTP facade for :class:`MotionService`."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from typing import Any
from urllib.parse import unquote, urlparse

from .motion import MotionValidationError
from .service import MotionConflictError, MotionService


MAX_BODY_BYTES = 2 * 1024 * 1024


def _payload_sha256(options: dict[str, Any]) -> str:
    core = {key: value for key, value in options.items() if key != "payload_sha256"}
    try:
        body = json.dumps(
            core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MotionValidationError(f"cannot canonicalize finite action payload: {exc}") from exc
    return hashlib.sha256(body).hexdigest()


def create_http_server(
    service: MotionService,
    host: str,
    port: int,
    *,
    cors_origin: str = "*",
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "G1SimService/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return None

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise MotionValidationError("invalid Content-Length") from exc
            if length <= 0 or length > MAX_BODY_BYTES:
                raise MotionValidationError("JSON body is empty or too large")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise MotionValidationError(f"invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise MotionValidationError("request body must be an object")
            return payload

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/health":
                payload = service.health()
                self._json(200 if payload["ok"] else 503, payload)
            elif path in {"/state", "/status"}:
                self._json(200, service.state())
            elif path in {"/action", "/actions"}:
                self._json(200, service.action_state())
            elif path.startswith("/actions/") or path.startswith("/action/"):
                action_id = unquote(path.rsplit("/", 1)[-1])
                action = service.get_action(action_id)
                if action is None:
                    self._json(
                        404,
                        {"error": "action_not_found", "id": action_id},
                    )
                else:
                    self._json(200, action)
            elif path in {"/frame", "/frame.png", "/frame.jpg"}:
                body, sequence = service.frame()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Frame-Sequence", str(sequence))
                self._cors()
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not_found", "path": path})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path in {"/heartbeat", "/health"}:
                    self._json(200, service.heartbeat())
                    return
                if path == "/action/cancel":
                    owner = None
                    if int(self.headers.get("Content-Length", "0") or "0") > 0:
                        owner = self._read_json().get("exclusive_owner")
                    self._json(200, service.cancel(exclusive_owner=owner))
                    return
                if path not in {"/action", "/actions/trigger"}:
                    self._json(404, {"error": "not_found", "path": path})
                    return
                payload = self._read_json()
                nested = payload.get("payload")
                options = dict(nested) if isinstance(nested, dict) else {}
                options.update({key: value for key, value in payload.items() if key != "payload"})
                declared_payload_sha256 = options.get("payload_sha256")
                if declared_payload_sha256 is not None:
                    if not isinstance(declared_payload_sha256, str):
                        raise MotionValidationError("payload_sha256 must be a string")
                    actual_payload_sha256 = _payload_sha256(options)
                    if declared_payload_sha256 != actual_payload_sha256:
                        raise MotionValidationError(
                            "payload_sha256 does not match the canonical action request"
                        )
                if (
                    options.get("release") is True
                    or options.get("command") == "release"
                    or options.get("action") == "release"
                ):
                    self._json(
                        200,
                        service.cancel(
                            reason="release",
                            exclusive_owner=options.get("exclusive_owner"),
                        ),
                    )
                    return
                if "motion" in options:
                    motion: Any = options["motion"]
                elif "preset" in options:
                    motion = str(options["preset"])
                elif "action" in options:
                    motion = str(options["action"])
                elif "path" in options:
                    motion = str(options["path"])
                elif "frames" in options or "keyframes" in options:
                    motion = options
                else:
                    raise MotionValidationError("provide action, preset, path, motion, or release")
                bridge_policy = options.get("bridge_policy")
                if "bridge_policy" in options and not isinstance(bridge_policy, str):
                    raise MotionValidationError("bridge_policy must be a string")
                result = service.submit(
                    motion,
                    mode=str(options.get("mode", "enqueue")),
                    speed=float(options.get("speed", 1.0)),
                    blend_duration=(
                        None
                        if options.get("blend_duration") is None
                        else float(options["blend_duration"])
                    ),
                    bridge_policy=(
                        None
                        if "bridge_policy" not in options
                        else bridge_policy
                    ),
                    action_id=(str(options["id"]) if options.get("id") else None),
                    require_empty=options.get("require_empty", False),
                    expected_action_seq=options.get("expected_action_seq"),
                    exclusive_owner=options.get("exclusive_owner"),
                    payload_sha256=declared_payload_sha256,
                )
                self._json(202, result)
            except MotionConflictError as exc:
                self._json(409, {"error": "action_conflict", "detail": str(exc)})
            except (MotionValidationError, TypeError, ValueError) as exc:
                self._json(400, {"error": "invalid_request", "detail": str(exc)})
            except Exception as exc:
                self._json(500, {"error": "internal_error", "detail": str(exc)})

        def do_DELETE(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/action":
                    self._json(200, service.cancel())
                else:
                    self._json(404, {"error": "not_found", "path": path})
            except MotionConflictError as exc:
                self._json(409, {"error": "action_conflict", "detail": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
