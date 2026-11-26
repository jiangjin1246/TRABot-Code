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
from .patching import (
    DEFAULT_PRESENTATION_GESTURE_ID,
    PRESENTATION_GESTURE_IDS,
    SimRobotRuntime,
    install_runtime_patches,
    pick_robot_action,
    select_presentation_action,
    simulator_robot_runtime,
)
from .sim_client import SimServiceClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing FlashHead Gemini app with a MuJoCo side-by-side "
            "video and simulator-only action backend."
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


def install_sim_bridge_stub(client: SimServiceClient) -> ModuleType:
    """Prevent importing the real G1 bridge in the simulator process."""

    stub = ModuleType("flash_head_robot_bridge")
    runtime = SimRobotRuntime(client)
    stub.DEFAULT_PRESENTATION_GESTURE_ID = DEFAULT_PRESENTATION_GESTURE_ID
    stub.PRESENTATION_GESTURE_IDS = PRESENTATION_GESTURE_IDS
    stub.pick_robot_action = pick_robot_action
    stub.robot_runtime = runtime

    def trigger_robot_action(text: Any = "", action_id: Any = None):
        action = select_presentation_action(text, action_id)
        result = client.trigger_legacy_sync(action)
        return {
            **result,
            "matched": True,
            "executed": bool(result.get("accepted", True)),
            "simulated": True,
            "action": action,
            "label": action,
            "playbackSource": "motion_file",
            "robot": simulator_robot_runtime(),
            "simulatorAction": result,
        }

    stub.trigger_robot_action = trigger_robot_action
    stub.__file__ = "<flashhead-mujoco-overlay simulator bridge>"
    sys.modules[stub.__name__] = stub
    return stub


def load_upstream(
    path: str, module_name: str, client: SimServiceClient
) -> ModuleType:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"upstream project directory does not exist: {root}")
    sys.path.insert(0, str(root))
    # Set dry-run flags as a second guard, then inject the exact five symbols the
    # app imports. The real-robot bridge module is never imported in this process.
    os.environ.setdefault("G1_SIMULATION_MODE", "1")
    os.environ.setdefault("ROBOT_ENABLED", "0")
    install_sim_bridge_stub(client)
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
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
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


def main(argv: Sequence[str] | None = None) -> Any:
    args = _parser().parse_args(argv)
    if not args.upstream_path:
        raise SystemExit("set FLASHHEAD_PROJECT_ROOT or pass --upstream-path")
    passthrough = list(args.upstream_args)
    if passthrough[:1] == ["--"]:
        passthrough.pop(0)

    config = OverlayConfig.from_env()
    client = SimServiceClient(config)
    module = load_upstream(args.upstream_path, args.upstream_module, client)
    report = install_runtime_patches(module, client, config)
    diagnostics = None
    if not args.no_diagnostics:
        diagnostics = start_diagnostics_server(config, client, report.as_dict)
        host, port = diagnostics.address
        print(f"[overlay] diagnostics http://{host}:{port}/health")
    print(f"[overlay] patches: {report.as_dict()}")
    try:
        return run_upstream(module, args.entrypoint, passthrough)
    finally:
        if diagnostics is not None:
            diagnostics.close()


if __name__ == "__main__":
    main()
