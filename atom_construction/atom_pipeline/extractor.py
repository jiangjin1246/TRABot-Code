"""Batch extraction of phase-structured Atom Schema v1 candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from .constraint_contract import build_constraint_profile, canonical_json_sha256
from .joints import G1_29_JOINTS, G1_ARM_JOINTS, G1_UPPER_BODY_JOINTS
from .safety import (
    MotionLimits,
    MujocoCollisionValidator,
    analyze_kinematics,
    build_atom_safety,
    build_motion_constraint_report,
)
from .schema import SCHEMA_PATH, SCHEMA_VERSION, atom_hash, write_atom
from .segmentation import DetectionResult, DetectorConfig, GestureSegment, detect_gesture_segments
from .source import SourceRecord, file_sha256, write_source_manifest


EXTRACTOR_NAME = "beat2-natural-atom-extractor"
EXTRACTOR_VERSION = "1.3.3"
SOURCE_POSE_MAPPING = "preserve_absolute_retarget_pose"
Q_SPACE = "absolute_joint_position_rad"
GAIN_REFERENCE = "control.neutral"
EXPORT_SAMPLING_MODE = "contiguous_source_frame_knots_bit_exact_v2"
IDENTITY_MAPPING_TOKEN = "absolute_retarget_source_frame_knots_v4"
PORT_TOLERANCE_SEMANTICS = "storage_consistency_only_not_runtime_acceptance"


@dataclass(frozen=True, slots=True)
class ExtractorConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output_fps: float = 50.0
    gain: float = 1.0
    arms_only: bool = True
    candidates_per_minute: float = 6.0
    min_candidates_per_source: int = 1
    max_candidates_per_source: int = 12
    auto_time_scale: bool = True
    max_time_scale: float = 3.0
    max_output_duration_s: float = 10.0
    max_velocity_rad_s: float = 0.8
    max_acceleration_rad_s2: float = 4.0
    max_jerk_rad_s3: float = 30.0
    position_margin_rad: float = 0.05
    collision_sample_stride: int = 2
    safe_cutpoint_velocity_rad_s: float = 0.15
    safe_cutpoint_acceleration_rad_s2: float = 1.5
    source_pose_mapping: str = SOURCE_POSE_MAPPING
    export_sampling_mode: str = EXPORT_SAMPLING_MODE
    emit_preview_motions: bool = True
    drop_constraint_failures: bool = False
    neutral_id: str = "safe_neutral.v1"
    fallback_atom_id: str = "safe_neutral.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("output_fps", self.output_fps),
            ("gain", self.gain),
            ("candidates_per_minute", self.candidates_per_minute),
            ("max_time_scale", self.max_time_scale),
            ("max_output_duration_s", self.max_output_duration_s),
            ("max_velocity_rad_s", self.max_velocity_rad_s),
            ("max_acceleration_rad_s2", self.max_acceleration_rad_s2),
            ("max_jerk_rad_s3", self.max_jerk_rad_s3),
            ("safe_cutpoint_velocity_rad_s", self.safe_cutpoint_velocity_rad_s),
            (
                "safe_cutpoint_acceleration_rad_s2",
                self.safe_cutpoint_acceleration_rad_s2,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if not 0.0 < self.gain <= 1.0:
            raise ValueError("gain must be in (0,1]")
        if self.min_candidates_per_source < 0:
            raise ValueError("min_candidates_per_source cannot be negative")
        if self.max_candidates_per_source < 1:
            raise ValueError("max_candidates_per_source must be at least one")
        if self.min_candidates_per_source > self.max_candidates_per_source:
            raise ValueError("min_candidates_per_source cannot exceed the maximum")
        if self.collision_sample_stride < 1:
            raise ValueError("collision_sample_stride must be at least one")
        if self.source_pose_mapping != SOURCE_POSE_MAPPING:
            raise ValueError(
                "Extractor 1.3.3 only emits preserve_absolute_retarget_pose; "
                "legacy neutral-rebased output remains readable but cannot be newly emitted"
            )
        if self.export_sampling_mode != EXPORT_SAMPLING_MODE:
            raise ValueError(
                "Extractor 1.3.3 only emits bit-exact contiguous source-frame knots; "
                "detector-filtered and resampled export paths are not supported"
            )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["detector"] = self.detector.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class RetargetedSequence:
    fps: float
    dof_pos: np.ndarray
    root_pos: np.ndarray | None
    root_rot: np.ndarray | None

    @property
    def duration_s(self) -> float:
        return (len(self.dof_pos) - 1) / self.fps


@dataclass(frozen=True, slots=True)
class ExtractionSummary:
    source_count: int
    candidate_count: int
    dropped_count: int
    output_dir: Path
    manifest_path: Path


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_retargeted_sequence(path: str | Path) -> RetargetedSequence:
    """Load a trusted retarget output and validate its exact G1 representation.

    Pickle can execute code while loading.  This function is intentionally for
    locally generated, trusted retarget results only; the CLI prints the same
    warning and never accepts uploads over the web service.
    """

    source = Path(path)
    with source.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - documented trusted offline input
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: pickle root must be a mapping")
    if "dof_pos" not in payload:
        raise ValueError(f"{source}: missing dof_pos")
    fps_raw = payload.get("fps", 30)
    if isinstance(fps_raw, bool):
        raise ValueError(f"{source}: fps must be numeric")
    fps = float(fps_raw)
    if not math.isfinite(fps) or not 20.0 <= fps <= 120.0:
        raise ValueError(f"{source}: fps must be in [20,120], got {fps}")
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    if dof_pos.ndim != 2 or dof_pos.shape[1] != len(G1_29_JOINTS) or len(dof_pos) < 3:
        raise ValueError(f"{source}: dof_pos must have shape (T,29), T>=3; got {dof_pos.shape}")
    if not np.isfinite(dof_pos).all():
        raise ValueError(f"{source}: dof_pos contains NaN or Inf")

    def optional_array(key: str, width: int) -> np.ndarray | None:
        raw = payload.get(key)
        if raw is None:
            return None
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (len(dof_pos), width) or not np.isfinite(array).all():
            raise ValueError(
                f"{source}: {key} must have shape ({len(dof_pos)},{width}) and be finite"
            )
        return array

    return RetargetedSequence(
        fps=fps,
        dof_pos=dof_pos,
        root_pos=optional_array("root_pos", 3),
        root_rot=optional_array("root_rot", 4),
    )


def load_neutral(path: str | Path, joints: Sequence[str]) -> tuple[dict[str, float], str]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw = payload.get("neutral") if isinstance(payload, dict) else None
    if not isinstance(raw, Mapping) and isinstance(payload, dict):
        frames = payload.get("frames")
        if isinstance(frames, list) and frames and isinstance(frames[0], Mapping):
            raw = frames[0].get("q")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: neutral JSON needs `neutral` or frames[0].q")
    result: dict[str, float] = {}
    for joint in joints:
        if joint in raw:
            value = float(raw[joint])
        elif joint.endswith("_joint") and joint[:-6] in raw:
            value = float(raw[joint[:-6]])
        elif joint.startswith("waist_"):
            value = 0.0
        else:
            raise ValueError(f"{source}: neutral pose is missing {joint}")
        if not math.isfinite(value):
            raise ValueError(f"{source}: neutral value for {joint} is not finite")
        result[joint] = value
    return result, file_sha256(source)


def _joint_weights(joints: Sequence[str]) -> np.ndarray:
    return np.asarray([0.6 if name.startswith("waist_") else 1.0 for name in joints])


def _select_segments(
    result: DetectionResult,
    source_duration_s: float,
    config: ExtractorConfig,
) -> list[GestureSegment]:
    target = int(math.ceil(source_duration_s / 60.0 * config.candidates_per_minute))
    target = max(config.min_candidates_per_source, target)
    target = min(config.max_candidates_per_source, target)
    eligible = sorted(
        result.segments,
        key=lambda item: (
            item.exit_kind == "moving_open",
            -item.kinematic_quality,
            item.start_index,
        ),
    )
    if len(eligible) <= target:
        return eligible

    # Deterministic farthest-first selection balances time position, duration,
    # amplitude and apex type rather than taking only the strongest beats.
    vectors = np.asarray(
        [
            [
                segment.start_index / max(len(result.q) - 1, 1),
                (segment.end_index - segment.start_index) / max(len(result.q) - 1, 1),
                segment.displacement_max,
                segment.kinematic_quality,
                1.0 if segment.primary_apex.kind == "postural" else 0.0,
            ]
            for segment in eligible
        ],
        dtype=np.float64,
    )
    med = np.median(vectors, axis=0)
    scale = np.maximum(np.percentile(vectors, 75, axis=0) - np.percentile(vectors, 25, axis=0), 1e-6)
    normalized = (vectors - med) / scale
    first = max(range(len(eligible)), key=lambda index: (eligible[index].kinematic_quality, -index))
    selected = [first]
    while len(selected) < target:
        remaining = [index for index in range(len(eligible)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(float(np.sum(np.abs(normalized[index] - normalized[old]))) for old in selected),
                eligible[index].kinematic_quality,
                -index,
            ),
        )
        selected.append(next_index)
    return sorted((eligible[index] for index in selected), key=lambda item: item.start_index)


def _handedness(q: np.ndarray, joints: Sequence[str]) -> tuple[str, float, float]:
    movement = np.sum(np.abs(np.diff(q, axis=0)), axis=0)
    left = float(sum(value for value, joint in zip(movement, joints) if joint.startswith("left_")))
    right = float(sum(value for value, joint in zip(movement, joints) if joint.startswith("right_")))
    maximum = max(left, right, 1e-12)
    if left < 0.15 * maximum and right < 0.15 * maximum:
        label = "none"
    elif left >= 0.55 * right and right >= 0.55 * left:
        label = "bilateral"
    elif left > right:
        label = "left"
    else:
        label = "right"
    return label, left, right


def _phase_times(
    segment: GestureSegment,
    detector_fps: float,
    source_fps: float,
    source_frame_start: int,
    source_frame_end: int,
    time_scale: float,
    duration: float,
) -> dict[str, dict[str, float]]:
    def mapped_time(detector_index: int) -> float:
        source_frame = int(round(detector_index / detector_fps * source_fps))
        source_frame = min(max(source_frame, source_frame_start), source_frame_end)
        return min(
            duration,
            max(0.0, (source_frame - source_frame_start) / source_fps * time_scale),
        )

    local: dict[str, list[tuple[float, float]]] = {}
    for phase in segment.phases:
        start = mapped_time(phase.start_index)
        end = max(start, mapped_time(phase.end_index))
        local.setdefault(phase.name, []).append((start, end))
    stroke_ranges = local.get("stroke")
    if not stroke_ranges:
        raise ValueError("detector produced a segment without a stroke")
    stroke_start = stroke_ranges[0][0]
    stroke_end = stroke_ranges[-1][1]
    retract_ranges = local.get("retraction", [])
    retract_start = retract_ranges[0][0] if retract_ranges else duration
    retract_start = max(stroke_end, min(retract_start, duration))
    return {
        "preparation": {"start_s": 0.0, "end_s": stroke_start},
        "stroke": {"start_s": stroke_start, "end_s": stroke_end},
        "hold": {"start_s": stroke_end, "end_s": retract_start},
        "retraction": {"start_s": retract_start, "end_s": duration},
    }


def _safe_cutpoints(
    phases: Mapping[str, Mapping[str, float]],
    duration: float,
    analysis: Any,
    config: ExtractorConfig,
) -> list[float]:
    values = [0.0]
    for name in ("preparation", "stroke", "hold", "retraction"):
        values.append(float(phases[name]["end_s"]))
    values.append(duration)
    ordered = sorted(min(max(float(value), 0.0), duration) for value in values)
    unique: list[float] = []
    for value in ordered:
        if not unique or value > unique[-1] + 1e-12:
            unique.append(value)
    certified: list[float] = []
    for value in unique:
        sample = int(np.argmin(np.abs(analysis.times - value)))
        velocity = float(np.max(np.abs(analysis.dq[sample])))
        acceleration = float(np.max(np.abs(analysis.ddq[sample])))
        if (
            velocity <= config.safe_cutpoint_velocity_rad_s + 1e-9
            and acceleration <= config.safe_cutpoint_acceleration_rad_s2 + 1e-9
        ):
            exact = duration if abs(value - duration) <= 1e-9 else value
            certified.append(exact)
    return certified


def _trajectory_points(
    analysis: Any,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in range(len(analysis.times)):
        result.append(
            {
                "t": float(analysis.times[row]),
                "q": {joint: float(analysis.q[row, col]) for col, joint in enumerate(analysis.joints)},
                "dq": {joint: float(analysis.dq[row, col]) for col, joint in enumerate(analysis.joints)},
                "ddq": {joint: float(analysis.ddq[row, col]) for col, joint in enumerate(analysis.joints)},
            }
        )
    return result


def _motion_limits(config: ExtractorConfig, joint_ranges: Mapping[str, Sequence[float]]) -> MotionLimits:
    joints = G1_ARM_JOINTS if config.arms_only else G1_UPPER_BODY_JOINTS
    return MotionLimits(
        velocity_rad_s=config.max_velocity_rad_s,
        acceleration_rad_s2=config.max_acceleration_rad_s2,
        jerk_rad_s3=config.max_jerk_rad_s3,
        velocity_by_joint={joint: config.max_velocity_rad_s for joint in joints},
        acceleration_by_joint={joint: config.max_acceleration_rad_s2 for joint in joints},
        jerk_by_joint={joint: config.max_jerk_rad_s3 for joint in joints},
        joint_ranges=joint_ranges,
        position_margin_rad=config.position_margin_rad,
        max_time_scale=config.max_time_scale,
    )


def _has_paired_source_video(record: SourceRecord) -> bool:
    """Locate the sibling BEAT2 video while preserving nested source paths.

    ``record.relative_path`` is relative to the configured ``pkl-results``
    root.  Reconstructing that root from the record keeps this helper working
    for both direct discovery and manifest replay without storing an absolute
    dataset path in Atom provenance.
    """

    relative = Path(record.relative_path)
    source_root = record.path
    for _ in relative.parts:
        source_root = source_root.parent
    video_path = source_root.parent / "video-results" / relative.with_suffix(".mp4")
    return video_path.is_file()


def _build_candidate(
    record: SourceRecord,
    sequence: RetargetedSequence,
    source_sha256: str,
    result: DetectionResult,
    segment: GestureSegment,
    candidate_index: int,
    neutral: Mapping[str, float],
    neutral_hash: str,
    config: ExtractorConfig,
    limits: MotionLimits,
    collision_validator: MujocoCollisionValidator | None,
    config_hash: str,
    constraint_profile: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    joints = tuple(G1_ARM_JOINTS if config.arms_only else G1_UPPER_BODY_JOINTS)
    # Detection uses a 25 Hz filtered representation, but export is deliberately
    # sourced from the untouched retarget sequence. Detector indices are mapped
    # once to inclusive source-frame bounds; every contiguous raw knot is kept.
    end_sample = min(segment.end_index, len(result.q) - 1)
    source_frame_start = int(round(segment.start_index / result.fps * sequence.fps))
    source_frame_end = int(round(end_sample / result.fps * sequence.fps))
    source_frame_start = min(max(source_frame_start, 0), len(sequence.dof_pos) - 1)
    source_frame_end = min(max(source_frame_end, 0), len(sequence.dof_pos) - 1)
    if source_frame_end <= source_frame_start:
        source_frame_end = min(len(sequence.dof_pos) - 1, source_frame_start + 1)
    if source_frame_end <= source_frame_start:
        raise ValueError("detected interval has fewer than two source-frame knots")
    source_apex_frame = int(round(segment.primary_apex.index / result.fps * sequence.fps))
    source_apex_frame = min(max(source_apex_frame, source_frame_start), source_frame_end)

    source_columns = [G1_29_JOINTS.index(joint) for joint in joints]
    source_q = sequence.dof_pos[
        source_frame_start : source_frame_end + 1,
        source_columns,
    ]
    neutral_vector = np.asarray([neutral[joint] for joint in joints], dtype=np.float64)
    # ``q`` is always absolute. At gain=1 bypass arithmetic so every q knot is
    # bit-for-bit the loaded float64 retarget source value. Lower gain is the
    # explicit neutral-referenced ablation.
    if config.gain == 1.0:
        target_q = np.array(source_q, dtype=np.float64, copy=True)
    else:
        target_q = neutral_vector + config.gain * (source_q - neutral_vector)
    raw_times = np.arange(len(target_q), dtype=np.float64) / sequence.fps
    effective_scale_cap = min(
        config.max_time_scale,
        config.max_output_duration_s / max(float(raw_times[-1]), 1e-9),
    )
    effective_scale_cap = max(1.0, effective_scale_cap)

    preflight = analyze_kinematics(raw_times, target_q, joints, limits=limits)
    recommended = float(preflight.recommended_time_scale)
    applied_scale = 1.0
    scale_limited = False
    if config.auto_time_scale and recommended > 1.0 + 1e-9:
        applied_scale = min(recommended, effective_scale_cap)
        scale_limited = recommended > effective_scale_cap + 1e-9
    # Retiming changes only source-knot timestamps. q is never interpolated or
    # filtered, so refinement cannot erase high-frequency source information.
    for _ in range(4):
        scaled_times = raw_times * applied_scale
        analysis = analyze_kinematics(scaled_times, target_q, joints, limits=limits)
        remaining = float(analysis.recommended_time_scale)
        if not config.auto_time_scale or remaining <= 1.0 + 1e-6:
            break
        next_scale = min(effective_scale_cap, applied_scale * remaining)
        if next_scale <= applied_scale + 1e-9:
            scale_limited = True
            break
        applied_scale = next_scale
        scale_limited = scale_limited or applied_scale >= effective_scale_cap - 1e-9
    scaled_times = raw_times * applied_scale
    analysis = analyze_kinematics(scaled_times, target_q, joints, limits=limits)
    safety = build_atom_safety(
        scaled_times,
        target_q,
        joints,
        limits=limits,
        collision_validator=collision_validator,
        neutral=neutral,
    )
    detailed_report = build_motion_constraint_report(
        scaled_times,
        target_q,
        joints,
        limits=limits,
        collision_validator=collision_validator,
        neutral=neutral,
    )
    constraint_profile_hash = None
    if constraint_profile is not None:
        constraint_profile_hash = canonical_json_sha256(constraint_profile)
        detailed_report["constraint_profile"] = dict(constraint_profile)
        detailed_report["constraint_profile_sha256"] = constraint_profile_hash
    # Avoid leaking a machine-specific absolute scene path in published reports.
    collision = detailed_report.get("collision")
    if isinstance(collision, dict) and isinstance(collision.get("model"), dict):
        collision["model"]["path"] = Path(str(collision["model"]["path"])).name

    duration = float(analysis.times[-1])
    phases = _phase_times(
        segment,
        result.fps,
        sequence.fps,
        source_frame_start,
        source_frame_end,
        applied_scale,
        duration,
    )
    phase_source_frames = {
        name: {
            "frame_start": min(
                source_frame_end,
                source_frame_start
                + int(round(values["start_s"] / applied_scale * sequence.fps)),
            ),
            "frame_end": min(
                source_frame_end,
                source_frame_start
                + int(round(values["end_s"] / applied_scale * sequence.fps)),
            ),
        }
        for name, values in phases.items()
    }
    safe_cutpoints = _safe_cutpoints(phases, duration, analysis, config)
    apex = min(
        duration,
        max(
            0.0,
            (source_apex_frame - source_frame_start) / sequence.fps * applied_scale,
        ),
    )
    handedness, left_motion, right_motion = _handedness(analysis.q, joints)
    active_joints = [
        joints[index]
        for index in segment.active_joint_indexes
        if 0 <= index < len(joints)
    ]
    if not active_joints:
        active_joints = list(joints)
    tags = ["motion_only", "upper_body", handedness]
    tags.append("single_apex" if len(segment.apices) == 1 else "multiple_apices")
    if phases["hold"]["end_s"] > phases["hold"]["start_s"] + 1e-9:
        tags.append("contains_hold")
    if segment.exit_kind != "returned":
        tags.append(segment.exit_kind)

    identity_digest = hashlib.sha256(
        (
            f"{record.relative_path}\0{source_sha256}\0{source_frame_start}\0"
            f"{source_frame_end}\0{source_apex_frame}\0{IDENTITY_MAPPING_TOKEN}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    atom_id = (
        f"beat2.{record.identity.speaker_key}.s{record.identity.session_id}."
        f"{record.identity.segment_id}.f{source_frame_start}-{source_frame_end}."
        f"{identity_digest}"
    )
    trajectory = _trajectory_points(analysis)
    storage_consistency_tolerance = {"q": 1e-8, "dq": 1e-8, "ddq": 1e-7}
    source_mapping_q = (
        source_q
        if config.gain == 1.0
        else neutral_vector + config.gain * (source_q - neutral_vector)
    )
    source_knot_export_error = float(np.max(np.abs(analysis.q - source_mapping_q)))
    source_knots_equal_original_q = bool(
        config.gain == 1.0 and np.array_equal(analysis.q, source_q)
    )
    start_point, end_point = trajectory[0], trajectory[-1]
    fixed_base_pass = bool(
        safety["metrics"]["passed"]
        and safety["collision"]["checked"]
        and safety["collision"]["passed"]
    )
    graph_eligible = bool(
        fixed_base_pass
        and segment.exit_kind != "moving_open"
        and bool(safe_cutpoints)
        and abs(safe_cutpoints[-1] - duration) <= 1e-9
        and segment.boundary_start_cost <= config.detector.boundary_review_cost
        and segment.boundary_end_cost <= config.detector.boundary_review_cost
    )
    evidence_payload = {
        "validator_version": safety["validator_version"],
        "config_sha256": config_hash,
        "constraint_profile_sha256": constraint_profile_hash,
        "model_root_xml_sha256": (
            detailed_report.get("collision", {}).get("model", {}).get("root_xml_sha256")
            if isinstance(detailed_report.get("collision"), dict)
            else None
        ),
        "dependency_closure_hash_available": False,
        "note": (
            "The root XML hash is not a dependency-closure hash; included XML, meshes, "
            "textures and MuJoCo version must be frozen before publication/certification."
        ),
    }
    evidence_hash = _json_hash(evidence_payload)
    atom: dict[str, Any] = {
        "$schema": SCHEMA_PATH.name,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "id": atom_id,
            "name": f"BEAT2 motion-only candidate {record.identity.sequence_id} #{candidate_index}",
            "revision": 1,
            "description": (
                "Naturally segmented motion-only candidate from an already-retargeted "
                "BEAT2 G1 trajectory. Semantic and prosodic labels are unavailable."
            ),
        },
        "status": "candidate",
        "semantics": {
            "gesture_family": "unknown",
            "communicative_functions": ["unknown"],
            "tags": tags,
            "handedness": handedness,
            "confidence": 0.0,
            "prosody": {
                "alignment": "unknown",
                "target_anchor": "apex",
                "intensity": "unknown",
                "confidence": 0.0,
            },
        },
        "source": {
            "dataset": "BEAT2-retargeted-G1",
            "speaker": record.identity.speaker_key,
            "session": record.identity.session_id,
            "clip": record.identity.sequence_id,
            "input_sha256": source_sha256,
            "source_fps": sequence.fps,
            "frame_start": source_frame_start,
            "frame_end": source_frame_end,
            "time_start": source_frame_start / sequence.fps,
            "time_end": source_frame_end / sequence.fps,
            "relative_path": record.relative_path,
            "extractor": EXTRACTOR_NAME,
            "extractor_version": EXTRACTOR_VERSION,
            "license": "inherit-from-BEAT2-and-retarget-pipeline; verify-before-redistribution",
        },
        "timing": {
            "nominal_duration_s": duration,
            "min_duration_s": duration,
            "max_duration_s": duration / 0.8,
            "phases": phases,
            "anchors": {"apex": apex},
            "safe_cutpoints": safe_cutpoints,
            "min_lead_ms": float(phases["preparation"]["end_s"] * 1000.0),
            "late_policy": "fallback",
        },
        "control": {
            "sample_rate_hz": sequence.fps / applied_scale,
            "joints": list(joints),
            "joint_mask": active_joints,
            "neutral": {"id": config.neutral_id, "content_hash": neutral_hash},
            "pose_representation": {
                "q_space": Q_SPACE,
                "source_mapping": config.source_pose_mapping,
                "gain_reference": GAIN_REFERENCE,
                "sampling_basis": config.export_sampling_mode,
                "detector_filter_affects_export": False,
            },
            "backend_compatibility": ["mujoco-g1-29dof", "unitree-g1-upper-body-review-only"],
            "interpolation": "sampled",
            "tempo_range": {"min": 0.8, "max": 1.0},
            "amplitude_range": {"min": 0.8, "max": 1.0},
            "trajectory": trajectory,
        },
        "entry": {
            "q": start_point["q"],
            "dq": start_point["dq"],
            "ddq": start_point["ddq"],
            "tolerance": storage_consistency_tolerance,
            "tolerance_semantics": PORT_TOLERANCE_SEMANTICS,
        },
        "exit": {
            "q": end_point["q"],
            "dq": end_point["dq"],
            "ddq": end_point["ddq"],
            "tolerance": storage_consistency_tolerance,
            "tolerance_semantics": PORT_TOLERANCE_SEMANTICS,
        },
        "safety": safety,
        "fallback": {
            "atom_id": config.fallback_atom_id,
            "policy": "release_to_neutral",
            "reason_codes": ["late", "constraint_failure", "unsafe_transition", "barge_in"],
        },
        "metadata": {
            "modalities": {
                "motion": True,
                "paired_video": _has_paired_source_video(record),
                "audio": False,
                "timestamped_text": False,
                "official_semantics": False,
            },
            "eligibility": {
                "eligible_for_motion_graph_curation": graph_eligible,
                "usable_for_motion_graph": False,
                "usable_for_semantic_planner": False,
                "usable_for_prosody": False,
                "usable_for_official_eval": False,
                "usable_for_real_robot": False,
            },
            "kinematics": {
                "active_joints": active_joints,
                "left_motion": left_motion,
                "right_motion": right_motion,
                "exit_kind": segment.exit_kind,
                "apex_kind": segment.primary_apex.kind,
                "apex_confidence": segment.primary_apex.confidence,
                "apex_count": len(segment.apices),
                "boundary_start_cost": segment.boundary_start_cost,
                "boundary_end_cost": segment.boundary_end_cost,
                "motion_snr": segment.motion_snr,
                "displacement_max": segment.displacement_max,
                "quality": segment.kinematic_quality,
                "warnings": list(segment.warnings),
            },
            "processing": {
                "config_sha256": config_hash,
                "detector_fps": result.fps,
                "requested_output_fps_legacy": config.output_fps,
                "effective_source_knot_rate_hz": sequence.fps / applied_scale,
                "gain": config.gain,
                "source_pose_mapping": config.source_pose_mapping,
                "trajectory_export": {
                    "sampling_basis": config.export_sampling_mode,
                    "resampling_method": "none",
                    "detector_filter_affects_export": False,
                    "source_frame_start": source_frame_start,
                    "source_frame_end": source_frame_end,
                    "source_apex_frame": source_apex_frame,
                    "phase_source_frames": phase_source_frames,
                    "source_knot_count": len(source_q),
                    "source_knot_max_abs_export_error_rad": source_knot_export_error,
                    "source_knots_equal_original_q": source_knots_equal_original_q,
                    "time_transform": "t_out=(source_frame-source_frame_start)/source_fps*applied_time_scale",
                },
                "source_entry_offset_from_neutral_rad": {
                    joint: float(source_q[0, index] - neutral_vector[index])
                    for index, joint in enumerate(joints)
                },
                "max_abs_source_entry_offset_from_neutral_rad": float(
                    np.max(np.abs(source_q[0] - neutral_vector))
                ),
                "recommended_time_scale_before_scaling": recommended,
                "applied_time_scale": applied_scale,
                "time_scale_limited": scale_limited,
                "effective_time_scale_cap": effective_scale_cap,
                "max_output_duration_s": config.max_output_duration_s,
                "safe_cutpoint_velocity_rad_s": config.safe_cutpoint_velocity_rad_s,
                "safe_cutpoint_acceleration_rad_s2": (
                    config.safe_cutpoint_acceleration_rad_s2
                ),
                "not_for_hardware": True,
            },
        },
        "extensions": {
            "validation_evidence": {
                **evidence_payload,
                "evidence_sha256": evidence_hash,
            },
            "research_claim_boundary": {
                "fixed_base_only": True,
                "balance_checked": False,
                "torque_checked": False,
                "tracking_checked": False,
                "human_reviewed": False,
            }
        },
    }
    preview = {
        "format": "g1_joint_motion_v1",
        "name": atom_id,
        "duration": duration,
        "joints": list(joints),
        "interpolation": "linear",
        "frames": [
            {"t": point["t"], "q": point["q"]}
            for point in trajectory
        ],
        "source_atom_hash": atom_hash(atom),
    }
    return atom, detailed_report, preview


def extract_sources(
    records: Sequence[SourceRecord],
    *,
    output_dir: str | Path,
    neutral_path: str | Path,
    scene_path: str | Path | None = None,
    config: ExtractorConfig | None = None,
    selection_metadata: Mapping[str, Any] | None = None,
) -> ExtractionSummary:
    """Extract and write a deterministic, provenance-rich candidate batch."""

    config = config or ExtractorConfig()
    output = Path(output_dir)
    candidates_dir = output / "candidates"
    reports_dir = output / "reports"
    previews_dir = output / "preview_motions"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if config.emit_preview_motions:
        previews_dir.mkdir(parents=True, exist_ok=True)

    joints = tuple(G1_ARM_JOINTS if config.arms_only else G1_UPPER_BODY_JOINTS)
    neutral, neutral_hash = load_neutral(neutral_path, joints)
    collision_validator = (
        MujocoCollisionValidator(scene_path, sample_stride=config.collision_sample_stride)
        if scene_path is not None
        else None
    )
    joint_ranges: dict[str, tuple[float, float]] = {}
    if collision_validator is not None:
        joint_ranges = collision_validator.joint_ranges(joints)
    limits = _motion_limits(config, joint_ranges)
    constraint_profile = (
        build_constraint_profile(
            limits,
            joints,
            collision_validator,
            neutral_id=config.neutral_id,
            neutral_content_hash=neutral_hash,
        )
        if collision_validator is not None
        else None
    )
    config_payload = config.as_dict()
    config_hash = _json_hash(config_payload)

    source_manifest = output / "selected_sources.json"
    write_source_manifest(
        source_manifest,
        records,
        selection={
            **dict(selection_metadata or {}),
            "config_sha256": config_hash,
        },
    )
    source_results: list[dict[str, Any]] = []
    atom_entries: list[dict[str, Any]] = []
    seen_atom_ids: set[str] = set()
    all_rejections: list[dict[str, Any]] = []
    dropped_count = 0
    for record in records:
        source_sha = file_sha256(record.path)
        sequence = load_retargeted_sequence(record.path)
        columns = [G1_29_JOINTS.index(joint) for joint in joints]
        raw_selected = sequence.dof_pos[:, columns]
        if joint_ranges:
            scales = np.asarray(
                [joint_ranges[joint][1] - joint_ranges[joint][0] for joint in joints],
                dtype=np.float64,
            )
        else:
            scales = np.maximum(
                np.percentile(raw_selected, 95, axis=0) - np.percentile(raw_selected, 5, axis=0),
                0.25,
            )
        detection = detect_gesture_segments(
            raw_selected,
            sequence.fps,
            joint_scales=scales,
            joint_weights=_joint_weights(joints),
            config=config.detector,
        )
        selected_segments = _select_segments(detection, sequence.duration_s, config)
        accepted_for_source = 0
        for candidate_index, segment in enumerate(selected_segments):
            try:
                atom, report, preview = _build_candidate(
                    record,
                    sequence,
                    source_sha,
                    detection,
                    segment,
                    candidate_index,
                    neutral,
                    neutral_hash,
                    config,
                    limits,
                    collision_validator,
                    config_hash,
                    constraint_profile,
                )
                atom_id = str(atom["identity"]["id"])
                if atom_id in seen_atom_ids:
                    raise ValueError(f"duplicate stable atom id in run: {atom_id}")
                seen_atom_ids.add(atom_id)
                if config.drop_constraint_failures and not atom["safety"]["metrics"]["passed"]:
                    dropped_count += 1
                    all_rejections.append({
                        "source": record.relative_path,
                        "candidate_index": candidate_index,
                        "reason": "motion_constraints_failed",
                        "violations": atom["safety"]["violations"],
                    })
                    continue
                atom_path = candidates_dir / f"{atom['identity']['id']}.json"
                written = write_atom(atom, atom_path)
                report["atom_id"] = written["identity"]["id"]
                report["atom_content_hash"] = written["identity"]["content_hash"]
                report_path = reports_dir / f"{written['identity']['id']}.motion_report.json"
                report_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                report_sha256 = file_sha256(report_path)
                preview_path: Path | None = None
                if config.emit_preview_motions:
                    preview["source_atom_hash"] = written["identity"]["content_hash"]
                    preview_path = previews_dir / f"{written['identity']['id']}.json"
                    preview_path.write_text(
                        json.dumps(preview, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                atom_entries.append({
                    "id": written["identity"]["id"],
                    "content_hash": written["identity"]["content_hash"],
                    "speaker": record.identity.speaker_key,
                    "session": record.identity.session_id,
                    "source": record.relative_path,
                    "atom": atom_path.relative_to(output).as_posix(),
                    "report": report_path.relative_to(output).as_posix(),
                    "report_sha256": report_sha256,
                    "preview_motion": (
                        preview_path.relative_to(output).as_posix() if preview_path else None
                    ),
                    "constraint_passed": bool(written["safety"]["metrics"]["passed"]),
                    "collision_checked": bool(written["safety"]["collision"]["checked"]),
                    "motion_graph_curation_eligible": bool(
                        written["metadata"]["eligibility"][
                            "eligible_for_motion_graph_curation"
                        ]
                    ),
                })
                accepted_for_source += 1
            except Exception as exc:  # preserve the rest of a research batch
                dropped_count += 1
                all_rejections.append({
                    "source": record.relative_path,
                    "candidate_index": candidate_index,
                    "reason": "candidate_build_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
        source_rejections = [
            {"source": record.relative_path, **item}
            for item in detection.rejections
        ]
        all_rejections.extend(source_rejections)
        source_results.append({
            **record.as_dict(),
            "input_sha256": source_sha,
            "fps": sequence.fps,
            "duration_s": sequence.duration_s,
            "detected_segments": len(detection.segments),
            "selected_segments": len(selected_segments),
            "written_candidates": accepted_for_source,
            "detector_rejections": len(detection.rejections),
            "thresholds": {
                "energy_floor": detection.floor,
                "energy_sigma": detection.sigma,
                "energy_dynamic_range": detection.dynamic_range,
                "theta_on": detection.theta_on,
                "theta_off": detection.theta_off,
            },
        })

    speaker_counts = Counter(entry["speaker"] for entry in atom_entries)
    session_counts = Counter(entry["session"] for entry in atom_entries)
    run_manifest = {
        "format": "beat2_atom_extraction_run_v1",
        "extractor": {"name": EXTRACTOR_NAME, "version": EXTRACTOR_VERSION},
        "schema_version": SCHEMA_VERSION,
        "config": config_payload,
        "config_sha256": config_hash,
        "constraint_profile": constraint_profile,
        "constraint_profile_sha256": (
            canonical_json_sha256(constraint_profile)
            if constraint_profile is not None else None
        ),
        "neutral": {
            "id": config.neutral_id,
            "content_hash": neutral_hash,
            "filename": Path(neutral_path).name,
        },
        "scene": {
            "enabled": scene_path is not None,
            "filename": Path(scene_path).name if scene_path is not None else None,
            "root_xml_sha256": (
                file_sha256(scene_path) if scene_path is not None else None
            ),
            "dependency_closure_hash_available": False,
        },
        "source_count": len(records),
        "atom_count": len(atom_entries),
        "dropped_candidate_count": dropped_count,
        "speaker_count": len(speaker_counts),
        "atoms_per_speaker": dict(sorted(speaker_counts.items())),
        "atoms_per_session": dict(sorted(session_counts.items())),
        "sources": source_results,
        "atoms": atom_entries,
        "rejections_file": "rejections.json",
        "claim_boundary": {
            "motion_only": True,
            "semantic_labels_available": False,
            "prosody_available": False,
            "fixed_base_collision_only": scene_path is not None,
            "real_robot_certified": False,
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "rejections.json").write_text(
        json.dumps(all_rejections, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ExtractionSummary(
        source_count=len(records),
        candidate_count=len(atom_entries),
        dropped_count=dropped_count,
        output_dir=output,
        manifest_path=manifest_path,
    )
