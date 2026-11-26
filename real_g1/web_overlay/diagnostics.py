from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Callable
from urllib.parse import urlsplit

from .config import OverlayConfig
@dataclass(slots=True)
class DiagnosticsServer:
    server: ThreadingHTTPServer
    thread: threading.Thread

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def start_diagnostics_server(
    config: OverlayConfig,
    client: Any,
    patch_status: Callable[[], dict[str, Any]],
) -> DiagnosticsServer:
    component = (
        "flashhead-g1-atom-realtime"
        if bool(getattr(client, "semantic_only", False))
        else "flashhead-mujoco-overlay"
    )

    def observed_state() -> dict[str, Any]:
        # A real-robot planning state read may acquire DDS control. Diagnostics
        # must remain observational, so use the client's non-starting snapshot.
        peek = getattr(client, "peek_state_sync", None)
        return peek() if callable(peek) else client.state_sync()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            return None

        def _send(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, default=str, separators=(",", ":")).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            path = urlsplit(self.path).path
            if path == "/health":
                try:
                    sim = client.health_sync()
                    ok = bool(
                        sim.get("ok", sim.get("status") in ("ok", "healthy"))
                    )
                except Exception as exc:
                    sim = {"ok": False, "error": str(exc)}
                    ok = False
                self._send(
                    200 if ok else 503,
                    {
                        "ok": ok,
                        "component": component,
                        "backend": sim,
                        "cache": client.cache_status(),
                    },
                )
                return
            if path in ("/status", "/sim/status", "/g1/status"):
                try:
                    sim = observed_state()
                except Exception as exc:
                    sim = {"available": False, "error": str(exc)}
                self._send(
                    200,
                    {
                        "component": component,
                        "backend": sim,
                        "cache": client.cache_status(),
                        "patches": patch_status(),
                    },
                )
                return
            self._send(404, {"error": "not found"})

    server = ThreadingHTTPServer(
        (config.diagnostics_host, config.diagnostics_port), Handler
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="overlay-diagnostics",
        daemon=True,
    )
    thread.start()
    return DiagnosticsServer(server, thread)
