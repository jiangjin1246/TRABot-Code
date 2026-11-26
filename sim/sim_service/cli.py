"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading

from .engine import CONVERSATION_CAMERA, FreeCameraSpec, KinematicEngine, MujocoEngine
from .http_api import create_http_server
from .service import MotionService, ServiceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent fixed-base G1 MuJoCo service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--engine", choices=("mujoco", "kinematic"), default="mujoco")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--motion-root", type=Path)
    parser.add_argument("--control-mode", choices=("auto", "torque", "position"), default="auto")
    parser.add_argument("--kp", type=float, default=45.0)
    parser.add_argument("--kd", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera",
        help=(
            "fixed camera name/id; omit (or use conversation/auto) for the "
            "front conversation view, or use scene for MuJoCo's scene default"
        ),
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=CONVERSATION_CAMERA.lookat,
    )
    parser.add_argument("--camera-distance", type=float, default=CONVERSATION_CAMERA.distance)
    parser.add_argument("--camera-azimuth", type=float, default=CONVERSATION_CAMERA.azimuth)
    parser.add_argument("--camera-elevation", type=float, default=CONVERSATION_CAMERA.elevation)
    parser.add_argument("--render-fps", type=float, default=15.0)
    parser.add_argument("--queue-capacity", type=int, default=16)
    parser.add_argument("--max-motion-duration", type=float, default=10.0)
    parser.add_argument("--cors-origin", default="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.engine == "mujoco":
        model = args.model or Path(__file__).with_name("assets") / "fixed_arm.xml"
        camera_spec = FreeCameraSpec(
            lookat=tuple(args.camera_lookat),
            distance=args.camera_distance,
            azimuth=args.camera_azimuth,
            elevation=args.camera_elevation,
        )
        engine = MujocoEngine(
            model,
            kp=args.kp,
            kd=args.kd,
            control_mode=args.control_mode,
            width=args.width,
            height=args.height,
            camera=args.camera,
            camera_spec=camera_spec,
        )
    else:
        engine = KinematicEngine(kp=args.kp, kd=args.kd)
    config = ServiceConfig(
        queue_capacity=args.queue_capacity,
        render_fps=args.render_fps,
        max_motion_duration=args.max_motion_duration,
    )
    service = MotionService(engine, config=config, motion_root=args.motion_root)
    server = create_http_server(service, args.host, args.port, cors_origin=args.cors_origin)
    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    service.start()
    address, port = server.server_address[:2]
    print(f"G1 simulation service listening on http://{address}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
