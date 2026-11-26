#!/usr/bin/env python3
"""Small dependency-free client for testing named and CRISP/BEAT2 motions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urlopen(request, timeout=2.0) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("simulator response is not a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("SIM_SERVICE_URL", "http://127.0.0.1:8765"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health")
    subparsers.add_parser("status")
    subparsers.add_parser("cancel")

    play = subparsers.add_parser("play")
    play.add_argument("motion", help="registered action ID or JSON file")
    play.add_argument("--mode", choices=("enqueue", "replace"), default="replace")
    play.add_argument("--replace", dest="mode", action="store_const", const="replace")
    play.add_argument("--enqueue", dest="mode", action="store_const", const="enqueue")
    play.add_argument("--speed", type=float, default=1.0)
    play.add_argument("--blend", type=float, default=0.25)
    play.add_argument("--wait", action="store_true")
    play.add_argument("--timeout", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "health":
        result = request_json(args.url, "GET", "/health")
    elif args.command == "status":
        result = request_json(args.url, "GET", "/state")
    elif args.command == "cancel":
        result = request_json(args.url, "POST", "/action/cancel")
    else:
        candidate = Path(args.motion).expanduser()
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as stream:
                motion = json.load(stream)
            if not isinstance(motion, dict):
                raise SystemExit("motion JSON must contain an object")
            payload: dict = {"motion": motion}
        else:
            payload = {"action": args.motion}
        payload.update(
            mode=args.mode,
            speed=args.speed,
            blend_duration=args.blend,
        )
        result = request_json(args.url, "POST", "/actions/trigger", payload)
        if args.wait:
            action_id = result.get("id")
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                state = request_json(args.url, "GET", "/state")
                completed = state.get("last_completed") or {}
                if completed.get("id") == action_id:
                    result = {"submitted": result, "completed": completed}
                    break
                time.sleep(0.05)
            else:
                raise SystemExit(f"timed out waiting for action {action_id}")

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
