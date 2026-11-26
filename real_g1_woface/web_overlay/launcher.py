from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence

from .config import OverlayConfig
from .diagnostics import start_diagnostics_server
from .faceless import install_faceless_inference_stub
from .g1_session_client import G1SessionClient, G1SessionConfig
from .patching import (
    DEFAULT_PRESENTATION_GESTURE_ID,
    PRESENTATION_GESTURE_IDS,
    install_runtime_patches,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FlashHead/Gemini face app without face "
            "generation (CPU only), with semantic_f17 planning and one "
            "fail-closed G1 Atom session per reply."
        )
    )
    parser.add_argument(
        "--upstream-path",
        default=os.environ.get("FLASHHEAD_PROJECT_ROOT"),
        help="Directory containing webrtc_app_gemini_http.py",
    )
    parser.add_argument(
        "--upstream-module",
        default=os.environ.get(
            "FLASHHEAD_UPSTREAM_MODULE", "webrtc_app_gemini_http"
        ),
    )
    parser.add_argument(
        "--entrypoint",
        default=os.environ.get("FLASHHEAD_UPSTREAM_ENTRYPOINT", "main"),
        help="Callable in the upstream module; falls back to its app object",
    )
    parser.add_argument(
        "--no-diagnostics", action="store_true", help="Do not bind diagnostics port"
    )
    parser.add_argument(
        "upstream_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the original entrypoint",
    )
    return parser


class _DisabledLegacyRobotRuntime:
    """Keep the old presentation bridge inert; semantic_f17 owns real motion."""

    enabled = False
    simulation_mode = True

    def __call__(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "simulationMode": True,
            "networkInterface": "disabled",
            "availableActions": [],
            "reasonCodes": ["semantic_f17_g1_backend_owns_motion"],
            "backend": "disabled_legacy_bridge",
        }

    def trigger_robot_action(self, *_: Any, **__: Any) -> dict[str, Any]:
        return _legacy_disabled_result()

    run_action = trigger_robot_action
    execute = trigger_robot_action
    submit = trigger_robot_action

    async def trigger_robot_action_async(
        self, *_: Any, **__: Any
    ) -> dict[str, Any]:
        return _legacy_disabled_result()

    def release(self) -> dict[str, Any]:
        return {"released": True, "noop": True}

    def stop(self) -> dict[str, Any]:
        return {"stopped": True, "noop": True}

    cancel = stop


def _legacy_disabled_result() -> dict[str, Any]:
    return {
        "matched": False,
        "executed": False,
        "simulated": False,
        "reason": "legacy presentation actions are disabled; semantic_f17 owns G1 motion",
    }


def install_disabled_real_bridge_stub() -> ModuleType:
    """Prevent the upstream module from importing either legacy G1 CLI."""

    stub = ModuleType("flash_head_robot_bridge")
    stub.DEFAULT_PRESENTATION_GESTURE_ID = DEFAULT_PRESENTATION_GESTURE_ID
    stub.PRESENTATION_GESTURE_IDS = PRESENTATION_GESTURE_IDS
    stub.pick_robot_action = lambda *_args, **_kwargs: None
    stub.robot_runtime = _DisabledLegacyRobotRuntime()
    stub.trigger_robot_action = lambda *_args, **_kwargs: _legacy_disabled_result()
    stub.__file__ = "<flashhead-g1-semantic-only bridge>"
    sys.modules[stub.__name__] = stub
    return stub


def load_upstream(path: str, module_name: str) -> ModuleType:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"upstream project directory does not exist: {root}")
    sys.path.insert(0, str(root))
    # These flags disable every legacy direct rt/arm_sdk path. The dedicated
    # session client below is the only component allowed to spawn the native
    # Atom runner.
    os.environ["G1_ACTIONS_ENABLED"] = "0"
    os.environ["G1_SIMULATION_MODE"] = "1"
    os.environ["ROBOT_ENABLED"] = "0"
    install_disabled_real_bridge_stub()
    # No face: upstream imports flash_head.inference at module load, so the
    # CPU stand-in must be registered first. FlashHead weights never load.
    install_faceless_inference_stub()
    print("[g1-realtime] face generation disabled; flash_head.inference is the CPU stand-in")
    return importlib.import_module(module_name)


def _run_app_object(app: Any, passthrough: Sequence[str]) -> Any:
    module_name = app.__class__.__module__
    if module_name.startswith("aiohttp"):
        from aiohttp import web

        return web.run_app(app)
    if callable(getattr(app, "run", None)):
        return app.run()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "upstream exposes an app object but no supported runner; set "
            "--entrypoint or install uvicorn"
        ) from exc
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "1131"))
    return uvicorn.run(app, host=host, port=port)


def run_upstream(
    module: ModuleType, entrypoint_name: str, passthrough: Sequence[str]
) -> Any:
    sys.argv = [str(getattr(module, "__file__", module.__name__))] + list(passthrough)
    entrypoint = getattr(module, entrypoint_name, None)
    if callable(entrypoint):
        result = entrypoint()
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result
    for name in ("app", "web_app", "application"):
        app = getattr(module, name, None)
        if app is not None:
            return _run_app_object(app, passthrough)
    raise SystemExit(
        f"{module.__name__!r} exposes neither {entrypoint_name!r} nor an app object"
    )


def _client(config: OverlayConfig) -> G1SessionClient:
    if config.backend.strip().lower() != "g1_realtime":
        raise SystemExit("OVERLAY_BACKEND must be g1_realtime")
    if config.dialogue_sequence_mode.strip().lower() != "semantic_f17":
        raise SystemExit("real-time G1 mode requires semantic_f17")
    if not config.semantic_execute:
        raise SystemExit("real-time G1 mode requires OVERLAY_SEMANTIC_EXECUTE=1")
    if not config.g1_realtime_enabled:
        raise SystemExit("set G1_REALTIME_ENABLED=1 only when the operator is ready")
    semantic_catalog = Path(config.semantic_catalog_path).expanduser().resolve()
    g1_catalog = Path(config.g1_catalog_path).expanduser().resolve()
    if semantic_catalog != g1_catalog:
        raise SystemExit(
            "OVERLAY_SEMANTIC_CATALOG_PATH and G1_REALTIME_CATALOG must be the same "
            "content-addressed catalog"
        )
    client = G1SessionClient(
        G1SessionConfig(
            runner=Path(config.g1_runner_path),
            catalog=g1_catalog,
            limits=Path(config.g1_limits_path),
            review_store=Path(config.g1_review_store_path),
            network=config.g1_network_interface,
            startup_timeout_s=config.g1_session_startup_timeout,
            action_timeout_s=config.semantic_action_timeout,
            idle_timeout_s=config.g1_session_idle_timeout,
            process_stop_timeout_s=config.g1_process_stop_timeout,
        )
    )
    health = client.health_sync()
    if not bool(health.get("runner_ready")):
        raise SystemExit(f"G1 realtime backend is not ready: {health}")
    print(
        "[g1-realtime] approved pool: "
        f"{health.get('approved_count')} exact Atoms; network={config.g1_network_interface}"
    )
    return client


def main(argv: Sequence[str] | None = None) -> Any:
    args = _parser().parse_args(argv)
    if not args.upstream_path:
        raise SystemExit("set FLASHHEAD_PROJECT_ROOT or pass --upstream-path")
    passthrough = list(args.upstream_args)
    if passthrough[:1] == ["--"]:
        passthrough.pop(0)

    config = OverlayConfig.from_env()
    client = _client(config)
    module = load_upstream(args.upstream_path, args.upstream_module)
    report = install_runtime_patches(module, client, config)
    diagnostics = None
    if not args.no_diagnostics:
        diagnostics = start_diagnostics_server(config, client, report.as_dict)
        host, port = diagnostics.address
        print(f"[g1-realtime] diagnostics http://{host}:{port}/health")
    print(f"[g1-realtime] patches: {report.as_dict()}")
    try:
        return run_upstream(module, args.entrypoint, passthrough)
    finally:
        client.close()
        if diagnostics is not None:
            diagnostics.close()


if __name__ == "__main__":
    main()
