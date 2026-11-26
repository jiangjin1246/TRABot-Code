#!/usr/bin/env python3
"""Run three production random-Atom turns against an already-running simulator.

This intentionally bypasses Gemini while exercising the production Catalog,
Planner, DialogueSequenceCoordinator, HTTP client, action-start minimum-jerk
controller path, cancellation, and neutral return.  It never starts or stops a
service process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web_overlay.atom_catalog import load_catalog  # noqa: E402
from web_overlay.config import OverlayConfig  # noqa: E402
from web_overlay.dialogue_sequence import (  # noqa: E402
    BRIDGE_POLICY,
    DialogueSequenceCoordinator,
    SequenceSettings,
)
from web_overlay.random_planner import random_planner  # noqa: E402
from web_overlay.sim_client import SimServiceClient  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded three-Atom dialogue lifecycle directly against an "
            "already-running MuJoCo service."
        )
    )
    parser.add_argument(
        "--sim-url",
        default=os.environ.get("OVERLAY_SIM_URL", "http://127.0.0.1:8765"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "atom_catalog_v1.json",
    )
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "e2e_dialogue_atom_turns.jsonl",
    )
    parser.add_argument("--action-timeout", type=float, default=45.0)
    return parser


async def _run(args: argparse.Namespace) -> list[dict]:
    if args.turns != 3:
        raise SystemExit("this frozen MVP acceptance runner requires --turns 3")
    config = OverlayConfig(
        sim_url=args.sim_url,
        http_timeout=max(OverlayConfig().http_timeout, 2.0),
    )
    client = SimServiceClient(config)
    health = await asyncio.to_thread(client.health_sync)
    if not bool(health.get("ok", health.get("status") in {"ok", "healthy"})):
        raise RuntimeError(f"MuJoCo service is not healthy: {health}")

    catalog = load_catalog(args.catalog.resolve())
    settings = SequenceSettings(
        count=3,
        base_seed=args.seed,
        log_path=args.log.resolve(),
        action_timeout=args.action_timeout,
    )
    coordinator = DialogueSequenceCoordinator(
        client=client,
        atom_catalog=catalog,
        planner=random_planner,
        settings=settings,
    )
    run_id = f"e2e-{int(time.time())}-{os.getpid()}"
    results: list[dict] = []

    for index in range(args.turns):
        source_turn_id = f"{run_id}-{index + 1}"
        handle = coordinator.start_turn(source_turn_id=source_turn_id)
        finished = False
        try:
            early_turn_complete = index == 1
            if early_turn_complete:
                submitted = await asyncio.wait_for(
                    handle.wait_all_submitted(), timeout=args.action_timeout
                )
                if len(submitted) != 3:
                    raise RuntimeError(
                        f"{handle.turn_id} submitted {len(submitted)} actions instead of 3"
                    )
            else:
                atoms_result = await handle.wait_atoms_complete()
                if atoms_result.status != "atoms_completed":
                    raise RuntimeError(
                        f"{atoms_result.turn_id} did not complete all Atoms: "
                        f"{atoms_result.status}"
                    )
                if len(atoms_result.action_ids) != 3:
                    raise RuntimeError(
                        f"{atoms_result.turn_id} executed "
                        f"{len(atoms_result.action_ids)} Atom actions instead of 3"
                    )
                if any(
                    duration is None for duration in atoms_result.bridge_durations
                ):
                    raise RuntimeError(
                        f"{atoms_result.turn_id} lacks a controller bridge duration"
                    )
            final = await coordinator.finish_turn(
                source_turn_id=source_turn_id, reason="turn_complete"
            )
            finished = True
            if not final.neutral_action_id or final.error:
                raise RuntimeError(
                    f"{final.turn_id} neutral return failed: {final.error or 'no id'}"
                )
            if len(final.action_ids) != 3:
                raise RuntimeError(
                    f"{final.turn_id} recorded {len(final.action_ids)} actions instead of 3"
                )
            if len({atom["atom_id"] for atom in final.atoms}) != 3:
                raise RuntimeError(f"{final.turn_id} repeated an Atom")
            terminal_statuses = [
                str(action.get("status", "")) for action in final.action_records
            ]
            if any(
                status not in {"completed", "canceled", "replaced", "failed"}
                for status in terminal_statuses
            ):
                raise RuntimeError(
                    f"{final.turn_id} retained nonterminal actions: {terminal_statuses}"
                )
            if early_turn_complete and not any(
                status in {"canceled", "replaced"} for status in terminal_statuses
            ):
                raise RuntimeError(
                    f"{final.turn_id} did not exercise queued-action cancellation"
                )
            if early_turn_complete and (
                not final.bridge_durations
                or final.bridge_durations[0] is None
            ):
                raise RuntimeError(
                    f"{final.turn_id} never observed Atom 1's runtime bridge"
                )
            results.append(
                {
                    "turn_id": final.turn_id,
                    "source_turn_id": source_turn_id,
                    "seed": final.seed,
                    "atom_ids": [atom["atom_id"] for atom in final.atoms],
                    "action_ids": list(final.action_ids),
                    "actions": list(final.action_records),
                    "mode": (
                        "early_turn_complete"
                        if early_turn_complete
                        else "full_atom_playback"
                    ),
                    "bridge_policy": BRIDGE_POLICY,
                    "bridge_durations": list(final.bridge_durations),
                    "playback_elapsed_seconds": final.playback_elapsed_seconds,
                    "wall_elapsed_seconds": final.wall_elapsed_seconds,
                    "neutral_action_id": final.neutral_action_id,
                    "status": final.status,
                }
            )
        finally:
            if not finished:
                await coordinator.finish_turn(
                    source_turn_id=source_turn_id, reason="e2e_abort"
                )

    expected_seeds = [args.seed + index for index in range(args.turns)]
    if [result["seed"] for result in results] != expected_seeds:
        raise RuntimeError("turn seeds are not reproducible")
    return results


def main() -> int:
    args = _parser().parse_args()
    results = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "ok": True,
                "turns": len(results),
                "catalog": str(args.catalog.resolve()),
                "log": str(args.log.resolve()),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
