"""Command line interface for Atom Schema v1 authoring."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from .extractor import ExtractorConfig, extract_sources
from .schema import AtomValidationError, load_atom
from .segmentation import DetectorConfig
from .source import (
    discover_sources,
    load_source_manifest,
    read_source_manifest_metadata,
    select_sources,
    write_source_manifest,
)


def _csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _selected_records(args: argparse.Namespace):
    if args.source_manifest:
        return load_source_manifest(args.source_manifest, source_root=args.source)
    discovered = discover_sources(args.source, recursive=args.recursive)
    return select_sources(
        discovered,
        speakers=_csv(args.speakers),
        sessions=_csv(args.sessions),
        max_speakers=None if args.max_speakers <= 0 else args.max_speakers,
        sources_per_speaker=(
            None if args.sources_per_speaker <= 0 else args.sources_per_speaker
        ),
        max_sources=None if args.max_sources <= 0 else args.max_sources,
        seed=args.seed,
    )


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="PKL file or pkl-results directory")
    parser.add_argument(
        "--source-manifest",
        help="Reuse selected_sources.json; --source becomes the local source root",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--speakers",
        help="Comma-separated speaker IDs or keys such as 1_wayne,7_sophie",
    )
    parser.add_argument("--sessions", default="0,1", help="Comma-separated session IDs")
    parser.add_argument("--max-speakers", type=int, default=10, help="0 selects all speakers")
    parser.add_argument(
        "--sources-per-speaker",
        type=int,
        default=2,
        help="0 keeps every source for each selected speaker",
    )
    parser.add_argument("--max-sources", type=int, default=0, help="0 means no global cap")
    parser.add_argument("--seed", type=int, default=20260915)


def _cmd_sources(args: argparse.Namespace) -> int:
    records = _selected_records(args)
    write_source_manifest(
        args.out,
        records,
        selection={
            "seed": args.seed,
            "speakers": _csv(args.speakers),
            "sessions": _csv(args.sessions),
            "max_speakers": args.max_speakers,
            "sources_per_speaker": args.sources_per_speaker,
            "max_sources": args.max_sources,
        },
    )
    counts = Counter(record.identity.speaker_key for record in records)
    print(f"Selected {len(records)} sources from {len(counts)} speakers -> {args.out}")
    for speaker, count in sorted(counts.items()):
        sessions = sorted(
            {record.identity.session_id for record in records if record.identity.speaker_key == speaker}
        )
        print(f"  {speaker}: {count} source(s), session(s) {','.join(sessions)}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"output directory is not empty: {output}. Use a new run directory to avoid stale artifacts."
        )
    records = _selected_records(args)
    detector = DetectorConfig(
        detection_fps=args.detection_fps,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration,
        motion_snr_min=args.min_motion_snr,
        core_quality_min=args.min_quality,
    )
    config = ExtractorConfig(
        detector=detector,
        output_fps=args.output_fps,
        gain=args.gain,
        arms_only=not args.include_waist,
        candidates_per_minute=args.candidates_per_minute,
        min_candidates_per_source=args.min_candidates_per_source,
        max_candidates_per_source=args.max_candidates_per_source,
        auto_time_scale=not args.no_auto_time_scale,
        max_time_scale=args.max_time_scale,
        max_output_duration_s=args.max_output_duration,
        max_velocity_rad_s=args.max_velocity,
        max_acceleration_rad_s2=args.max_acceleration,
        max_jerk_rad_s3=args.max_jerk,
        position_margin_rad=args.position_margin,
        collision_sample_stride=args.collision_sample_stride,
        safe_cutpoint_velocity_rad_s=args.safe_cutpoint_velocity,
        safe_cutpoint_acceleration_rad_s2=args.safe_cutpoint_acceleration,
        emit_preview_motions=not args.no_preview_motions,
        drop_constraint_failures=args.drop_constraint_failures,
    )
    print(
        "SECURITY: loading trusted local retarget .pkl files; never point this "
        "offline command at untrusted uploads.",
        file=sys.stderr,
    )
    selection_metadata = {
        "seed": args.seed,
        "speakers": _csv(args.speakers),
        "sessions": _csv(args.sessions),
        "max_speakers": args.max_speakers,
        "sources_per_speaker": args.sources_per_speaker,
        "max_sources": args.max_sources,
    }
    if args.source_manifest:
        selection_metadata = read_source_manifest_metadata(args.source_manifest)
    summary = extract_sources(
        records,
        output_dir=output,
        neutral_path=args.neutral,
        scene_path=None if args.no_collision else args.scene,
        config=config,
        selection_metadata=selection_metadata,
    )
    print(
        f"Wrote {summary.candidate_count} Atom v1 candidates from "
        f"{summary.source_count} sources to {summary.output_dir}"
    )
    print(f"Dropped/build failures: {summary.dropped_count}")
    print(f"Run manifest: {summary.manifest_path}")
    return 0


def _iter_json_paths(values: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        else:
            result.append(path)
    return result


def _cmd_validate(args: argparse.Namespace) -> int:
    paths = _iter_json_paths(args.paths)
    if not paths:
        raise ValueError("no atom JSON files found")
    failures = 0
    for path in paths:
        try:
            atom = load_atom(path, verify_hash=not args.skip_hash)
            print(f"OK {path}: {atom['identity']['id']} [{atom['status']}]")
        except (OSError, json.JSONDecodeError, AtomValidationError, ValueError) as exc:
            failures += 1
            print(f"FAIL {path}: {exc}", file=sys.stderr)
    print(f"Validated {len(paths) - failures}/{len(paths)} atom(s)")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m atom_pipeline",
        description="Offline BEAT2-retarget -> phase-aware G1 Atom Schema v1 pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sources = subparsers.add_parser("sources", help="Create a stratified source manifest")
    _add_selection_arguments(sources)
    sources.add_argument("--out", required=True)
    sources.set_defaults(func=_cmd_sources)

    extract = subparsers.add_parser("extract", help="Extract Atom Schema v1 candidates")
    _add_selection_arguments(extract)
    extract.add_argument("--out-dir", required=True)
    extract.add_argument("--neutral", required=True)
    extract.add_argument(
        "--scene",
        default=os.environ.get("MUJOCO_MODEL_PATH"),
        help="MuJoCo scene for joint-range/contact checks",
    )
    extract.add_argument("--no-collision", action="store_true")
    extract.add_argument("--include-waist", action="store_true")
    extract.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help=(
            "Retarget motion amplitude in (0,1]; defaults to 1.0 (full amplitude). "
            "Pass an explicit smaller value for compatibility with conservative runs."
        ),
    )
    extract.add_argument("--detection-fps", type=float, default=25.0)
    extract.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help=(
            "Legacy compatibility setting recorded in the manifest. Extractor 1.3.3 "
            "exports contiguous source-frame knots without resampling."
        ),
    )
    extract.add_argument("--min-duration", type=float, default=0.40)
    extract.add_argument("--max-duration", type=float, default=4.0)
    extract.add_argument("--min-motion-snr", type=float, default=4.0)
    extract.add_argument("--min-quality", type=float, default=0.55)
    extract.add_argument("--candidates-per-minute", type=float, default=6.0)
    extract.add_argument("--min-candidates-per-source", type=int, default=1)
    extract.add_argument("--max-candidates-per-source", type=int, default=12)
    extract.add_argument("--no-auto-time-scale", action="store_true")
    extract.add_argument("--max-time-scale", type=float, default=3.0)
    extract.add_argument(
        "--max-output-duration",
        type=float,
        default=10.0,
        help="Hard preview/runtime duration cap after automatic slowing",
    )
    extract.add_argument("--max-velocity", type=float, default=0.8)
    extract.add_argument("--max-acceleration", type=float, default=4.0)
    extract.add_argument("--max-jerk", type=float, default=30.0)
    extract.add_argument("--position-margin", type=float, default=0.05)
    extract.add_argument("--collision-sample-stride", type=int, default=2)
    extract.add_argument("--safe-cutpoint-velocity", type=float, default=0.15)
    extract.add_argument("--safe-cutpoint-acceleration", type=float, default=1.5)
    extract.add_argument("--no-preview-motions", action="store_true")
    extract.add_argument("--drop-constraint-failures", action="store_true")
    extract.set_defaults(func=_cmd_extract)

    validate = subparsers.add_parser("validate", help="Validate atom JSON and content hashes")
    validate.add_argument("paths", nargs="+")
    validate.add_argument("--skip-hash", action="store_true")
    validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract" and not args.no_collision and not args.scene:
        parser.error("--scene is required unless --no-collision is used")
    try:
        return int(args.func(args))
    except (OSError, ValueError, AtomValidationError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
