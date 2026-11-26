from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Callable
from urllib.parse import urlsplit

from .config import OverlayConfig
from .sim_client import SimServiceClient


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
    client: SimServiceClient,
    patch_status: Callable[[], dict[str, Any]],
) -> DiagnosticsServer:
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
                        "component": "flashhead-mujoco-overlay",
                        "sim": sim,
                        "cache": client.cache_status(),
                    },
                )
                return
            if path in ("/status", "/sim/status"):
                try:
                    sim = client.state_sync()
                except Exception as exc:
                    sim = {"available": False, "error": str(exc)}
                self._send(
                    200,
                    {
                        "component": "flashhead-mujoco-overlay",
                        "sim": sim,
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
