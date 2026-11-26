"""Pure, provenance-preserving repair of Atom Schema v1 candidates.

Repair is intentionally conservative and fail-closed.  The input mapping is
never mutated; a successful call returns a new candidate revision after:

1. scaling displacement about an explicit neutral/reference pose,
2. applying an endpoint-preserving zero-phase binomial smoother, and
3. stretching time until the configured kinematic limits pass.

The module does not write files, promote status, or claim hardware safety.  A
caller can therefore call :func:`repair_candidate` transactionally
and decide separately whether/where to persist the returned artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from .constraint_contract import build_constraint_profile, canonical_json_sha256

from .safety import (
    CONSERVATIVE_G1_LIMITS,
    MotionLimits,
    MujocoCollisionValidator,
    analyze_kinematics,
    build_atom_safety,
    build_motion_constraint_report,
    time_scale_trajectory,
)
from .schema import SCHEMA_VERSION, atom_hash, validate_atom


REPAIR_NAME = "atom-constraint-repair"
REPAIR_VERSION = "2.0.0"
_PHASE_NAMES = ("preparation", "stroke", "hold", "retraction")
_REPAIR_LIMBS = ("left_arm", "right_arm", "waist")


class AtomRepairError(ValueError):
    """Raised when a requested repair cannot produce a valid safe candidate."""


@dataclass(frozen=True, slots=True)
class RepairRecipe:
    """Deterministic repair settings suitable for hashing and provenance."""

    amplitude_scale: float = 0.85
    smoothing_passes: int = 1
    smoothing_kernel: tuple[float, ...] = (1.0, 4.0, 6.0, 4.0, 1.0)
    retime: bool = True
    max_time_scale: float = 3.0
    max_output_duration_s: float = 10.0
    max_iterations: int = 5
    pass_tolerance: float = 1e-6
    # Absent limbs retain the legacy scalar amplitude_scale. This keeps old
    # callers and serialized recipes meaningful while allowing Auto-Repair to
    # constrain only the limb that needs repair.
    limb_scales: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.amplitude_scale)) or not 0.0 < self.amplitude_scale <= 1.0:
            raise ValueError("amplitude_scale must be finite and in (0, 1]")
        if isinstance(self.smoothing_passes, bool) or self.smoothing_passes < 0:
            raise ValueError("smoothing_passes must be a non-negative integer")
        if len(self.smoothing_kernel) < 1 or len(self.smoothing_kernel) % 2 != 1:
            raise ValueError("smoothing_kernel must have a positive odd length")
        kernel = np.asarray(self.smoothing_kernel, dtype=np.float64)
        if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0) or not np.any(kernel > 0.0):
            raise ValueError("smoothing_kernel must contain finite non-negative weights")
        if not math.isfinite(float(self.max_time_scale)) or self.max_time_scale < 1.0:
            raise ValueError("max_time_scale must be finite and at least one")
        if (
            not math.isfinite(float(self.max_output_duration_s))
            or self.max_output_duration_s <= 0.0
        ):
            raise ValueError("max_output_duration_s must be finite and positive")
        if isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not math.isfinite(float(self.pass_tolerance)) or self.pass_tolerance < 0.0:
            raise ValueError("pass_tolerance must be finite and non-negative")
        if self.limb_scales is not None:
            unknown = sorted(set(self.limb_scales) - set(_REPAIR_LIMBS))
            if unknown:
                raise ValueError(f"unknown limb scale(s): {', '.join(unknown)}")
            for limb, raw in self.limb_scales.items():
                value = float(raw)
                if not math.isfinite(value) or not 0.0 < value <= 1.0:
                    raise ValueError(
                        f"limb_scales[{limb!r}] must be finite and in (0, 1]"
                    )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "amplitude_scale": float(self.amplitude_scale),
            "smoothing": {
                "method": "zero_phase_normalized_convolution_endpoint_locked",
                "passes": int(self.smoothing_passes),
                "kernel": [float(value) for value in self.smoothing_kernel],
            },
            "retime": bool(self.retime),
            "max_time_scale": float(self.max_time_scale),
            "max_output_duration_s": float(self.max_output_duration_s),
            "max_iterations": int(self.max_iterations),
            "pass_tolerance": float(self.pass_tolerance),
        }
        if self.limb_scales is not None:
            result["limb_scales"] = {
                limb: float(self.limb_scales[limb])
                for limb in _REPAIR_LIMBS
                if limb in self.limb_scales
            }
        return result


@dataclass(frozen=True, slots=True)
class RepairResult:
    """New in-memory artifacts returned to an offline caller."""

    atom: dict[str, Any]
    safety_report: dict[str, Any]
    preview_motion: dict[str, Any]
    applied_recipe: dict[str, Any]
    parent_hash: str


def _trajectory_arrays(atom: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    control = atom["control"]
    joints = tuple(str(joint) for joint in control["joints"])
    trajectory = control["trajectory"]
    times = np.asarray([point["t"] for point in trajectory], dtype=np.float64)
    q = np.asarray(
        [[point["q"][joint] for joint in joints] for point in trajectory],
        dtype=np.float64,
    )
    if len(times) < 3:
        raise AtomRepairError("repair requires at least three trajectory samples")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(q)):
        raise AtomRepairError("trajectory contains non-finite values")
    return times, q, joints


def joint_repair_limb(joint: str) -> str | None:
    """Return the independently scalable upper-body limb for a joint."""

    if joint.startswith("left_") and (
        "shoulder" in joint or "elbow" in joint or "wrist" in joint
    ):
        return "left_arm"
    if joint.startswith("right_") and (
        "shoulder" in joint or "elbow" in joint or "wrist" in joint
    ):
        return "right_arm"
    if joint.startswith("waist_"):
        return "waist"
    return None


def _effective_limb_scales(recipe: RepairRecipe) -> dict[str, float]:
    overrides = recipe.limb_scales or {}
    return {
        limb: float(overrides.get(limb, recipe.amplitude_scale))
        for limb in _REPAIR_LIMBS
    }


def _reference_vector(
    atom: Mapping[str, Any],
    q: np.ndarray,
    joints: Sequence[str],
    neutral_pose: Mapping[str, float] | Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, str, list[str]]:
    if neutral_pose is None:
        reference = q[0].copy()
        source = "entry_pose"
        fallback_joints = list(joints)
    elif isinstance(neutral_pose, Mapping):
        missing = [joint for joint in joints if joint not in neutral_pose]
        reference = np.asarray(
            [
                neutral_pose[joint] if joint in neutral_pose else q[0, index]
                for index, joint in enumerate(joints)
            ],
            dtype=np.float64,
        )
        source = (
            "explicit_neutral_pose"
            if not missing
            else "explicit_neutral_pose_with_entry_fallback"
        )
        fallback_joints = missing
    else:
        reference = np.asarray(neutral_pose, dtype=np.float64)
        if reference.shape != (len(joints),):
            raise AtomRepairError("neutral_pose must contain one value per controlled joint")
        source = "explicit_neutral_vector"
        fallback_joints = []
    if not np.all(np.isfinite(reference)):
        raise AtomRepairError("neutral_pose contains non-finite values")
    return reference, source, fallback_joints


def _normalized_smooth_once(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Centered, zero-phase convolution without reflected endpoint overshoot."""

    half = len(kernel) // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    numerator = np.empty_like(values)
    for column in range(values.shape[1]):
        numerator[:, column] = np.convolve(padded[:, column], kernel, mode="valid")
    return numerator / float(np.sum(kernel))


def endpoint_preserving_zero_phase_smooth(
    q: Sequence[Sequence[float]] | np.ndarray,
    *,
    passes: int = 1,
    kernel: Sequence[float] = (1.0, 4.0, 6.0, 4.0, 1.0),
) -> np.ndarray:
    """Smooth every joint without phase lag and restore exact entry/exit poses."""

    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3 or values.shape[1] < 1:
        raise ValueError("q must have shape [samples,joints] with at least three samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("q must contain only finite values")
    if isinstance(passes, bool) or int(passes) != passes or passes < 0:
        raise ValueError("passes must be a non-negative integer")
    weights = np.asarray(kernel, dtype=np.float64)
    if (
        weights.ndim != 1
        or len(weights) < 1
        or len(weights) % 2 != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError("kernel must contain a positive odd number of finite non-negative weights")
    result = values.copy()
    entry, exit_state = values[0].copy(), values[-1].copy()
    for _ in range(int(passes)):
        result = _normalized_smooth_once(result, weights)
        result[0] = entry
        result[-1] = exit_state
    return result


def _scale_timing(atom: dict[str, Any], factor: float) -> None:
    timing = atom["timing"]
    old_duration = float(timing["nominal_duration_s"])
    new_duration = old_duration * factor
    timing["nominal_duration_s"] = new_duration
    timing["min_duration_s"] = float(timing["min_duration_s"]) * factor
    timing["max_duration_s"] = float(timing["max_duration_s"]) * factor
    for phase_name in _PHASE_NAMES:
        phase = timing["phases"][phase_name]
        phase["start_s"] = float(phase["start_s"]) * factor
        phase["end_s"] = float(phase["end_s"]) * factor
    for anchor_name in tuple(timing["anchors"]):
        timing["anchors"][anchor_name] = float(timing["anchors"][anchor_name]) * factor
    timing["safe_cutpoints"] = [float(value) * factor for value in timing["safe_cutpoints"]]
    timing["min_lead_ms"] = float(timing["min_lead_ms"]) * factor

    # Ensure accumulated floating-point products keep exact contract endpoints.
    timing["phases"]["preparation"]["start_s"] = 0.0
    timing["phases"]["retraction"]["end_s"] = new_duration
    if timing["safe_cutpoints"]:
        if abs(timing["safe_cutpoints"][0]) <= 1e-12:
            timing["safe_cutpoints"][0] = 0.0
        if abs(timing["safe_cutpoints"][-1] - new_duration) <= max(1e-12, 1e-9 * new_duration):
            timing["safe_cutpoints"][-1] = new_duration


def _trajectory_points(analysis: Any) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row, time_s in enumerate(analysis.times):
        points.append(
            {
                "t": float(time_s),
                "q": {
                    joint: float(analysis.q[row, column])
                    for column, joint in enumerate(analysis.joints)
                },
                "dq": {
                    joint: float(analysis.dq[row, column])
                    for column, joint in enumerate(analysis.joints)
                },
                "ddq": {
                    joint: float(analysis.ddq[row, column])
                    for column, joint in enumerate(analysis.joints)
                },
            }
        )
    return points


def _json_round_trip(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject non-JSON extension values while returning an isolated deep copy."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise AtomRepairError(f"input atom is not finite JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise AtomRepairError("input atom must be an object")
    return result


def _json_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AtomRepairError(f"repair evidence is not finite JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _safe_cutpoint_thresholds(atom: Mapping[str, Any]) -> tuple[float, float]:
    processing = atom.get("metadata", {}).get("processing", {})
    if not isinstance(processing, Mapping):
        processing = {}
    raw_velocity = processing.get("safe_cutpoint_velocity_rad_s", 0.15)
    raw_acceleration = processing.get("safe_cutpoint_acceleration_rad_s2", 1.5)
    try:
        velocity = float(raw_velocity)
        acceleration = float(raw_acceleration)
    except (TypeError, ValueError) as exc:
        raise AtomRepairError("safe-cutpoint thresholds must be numeric") from exc
    if (
        not math.isfinite(velocity)
        or velocity <= 0.0
        or not math.isfinite(acceleration)
        or acceleration <= 0.0
    ):
        raise AtomRepairError("safe-cutpoint thresholds must be finite and positive")
    return velocity, acceleration


def _recompute_safe_cutpoints(
    atom: Mapping[str, Any],
    analysis: Any,
    *,
    velocity_limit: float,
    acceleration_limit: float,
    exit_kind: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Filter mapped phase/cutpoint candidates against the repaired dynamics."""

    duration = float(analysis.times[-1])
    candidates = [float(value) for value in atom["timing"]["safe_cutpoints"]]
    phases = atom["timing"]["phases"]
    candidates.extend(float(phases[name]["end_s"]) for name in _PHASE_NAMES)
    candidates.extend((0.0, duration))
    candidates = sorted(min(max(value, 0.0), duration) for value in candidates)
    unique: list[float] = []
    for value in candidates:
        if not unique or value > unique[-1] + 1e-12:
            unique.append(value)

    kept: list[float] = []
    dropped: list[dict[str, Any]] = []
    for value in unique:
        sample_index = int(np.argmin(np.abs(analysis.times - value)))
        max_velocity = float(np.max(np.abs(analysis.dq[sample_index])))
        max_acceleration = float(np.max(np.abs(analysis.ddq[sample_index])))
        reasons: list[str] = []
        if max_velocity > velocity_limit + 1e-9:
            reasons.append("velocity")
        if max_acceleration > acceleration_limit + 1e-9:
            reasons.append("acceleration")
        if exit_kind == "moving_open" and abs(value - duration) <= 1e-9:
            reasons.append("moving_open_exit")
        exact = duration if abs(value - duration) <= 1e-9 else (0.0 if abs(value) <= 1e-12 else value)
        if reasons:
            dropped.append(
                {
                    "mapped_time_s": float(exact),
                    "nearest_sample_time_s": float(analysis.times[sample_index]),
                    "max_abs_velocity_rad_s": max_velocity,
                    "max_abs_acceleration_rad_s2": max_acceleration,
                    "reasons": reasons,
                }
            )
        else:
            kept.append(float(exact))
    return kept, dropped


def repair_candidate(
    atom: Mapping[str, Any],
    *,
    recipe: RepairRecipe | None = None,
    limits: MotionLimits | None = None,
    collision_validator: MujocoCollisionValidator | None = None,
    neutral_pose: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
) -> RepairResult:
    """Return a repaired, rehashed candidate without mutating ``atom``.

    ``collision_validator`` is optional for numerical repair/testing.  When it
    is supplied, a collision failure makes the repair fail.  When omitted, the
    result explicitly records that collision was not checked and receives no
    backend certification.
    """

    if not isinstance(atom, Mapping):
        raise AtomRepairError("atom must be a mapping")
    source_atom = _json_round_trip(atom)
    try:
        validate_atom(source_atom)
    except Exception as exc:
        raise AtomRepairError(f"input atom is invalid: {exc}") from exc
    if source_atom["schema_version"] != SCHEMA_VERSION:
        raise AtomRepairError(f"only {SCHEMA_VERSION} atoms can be repaired")
    if source_atom["status"] != "candidate":
        raise AtomRepairError("repair accepts candidate atoms only")

    recipe = recipe or RepairRecipe()
    limits = limits or CONSERVATIVE_G1_LIMITS
    if recipe.max_time_scale > limits.max_time_scale + 1e-12:
        raise AtomRepairError(
            "repair recipe max_time_scale exceeds the frozen MotionLimits cap"
        )
    parent_hash = atom_hash(source_atom)
    supplied_parent_hash = source_atom["identity"].get("content_hash")
    if supplied_parent_hash is not None and supplied_parent_hash != parent_hash:
        # validate_atom normally catches this; retain a local fail-closed guard.
        raise AtomRepairError("parent content_hash does not match canonical atom content")

    times, q, joints = _trajectory_arrays(source_atom)
    parent_duration = float(times[-1])
    parent_phases = deepcopy(source_atom["timing"]["phases"])
    parent_anchors = deepcopy(source_atom["timing"]["anchors"])
    parent_cutpoints = [float(value) for value in source_atom["timing"]["safe_cutpoints"]]
    cutpoint_velocity_limit, cutpoint_acceleration_limit = _safe_cutpoint_thresholds(
        source_atom
    )
    reference, reference_source, reference_fallback_joints = _reference_vector(
        source_atom, q, joints, neutral_pose
    )
    effective_limb_scales = _effective_limb_scales(recipe)
    column_scales = np.asarray(
        [
            effective_limb_scales.get(joint_repair_limb(joint), recipe.amplitude_scale)
            for joint in joints
        ],
        dtype=np.float64,
    )
    repaired_q = reference + (q - reference) * column_scales[None, :]
    repaired_q = endpoint_preserving_zero_phase_smooth(
        repaired_q,
        passes=recipe.smoothing_passes,
        kernel=recipe.smoothing_kernel,
    )

    applied_time_scale = 1.0
    for _ in range(recipe.max_iterations):
        analysis = analyze_kinematics(times, repaired_q, joints, limits=limits)
        dynamic_violations = [
            item
            for item in analysis.report["violations"]
            if item["code"] != "JOINT_POSITION_LIMIT"
        ]
        position_violations = [
            item
            for item in analysis.report["violations"]
            if item["code"] == "JOINT_POSITION_LIMIT"
        ]
        if position_violations:
            raise AtomRepairError(
                "amplitude scaling did not satisfy joint position limits; "
                "temporal retiming cannot repair position violations"
            )
        if not dynamic_violations:
            break
        if not recipe.retime:
            raise AtomRepairError("trajectory violates dynamic limits and retiming is disabled")
        remaining_scale = float(analysis.recommended_time_scale)
        if remaining_scale <= 1.0 + recipe.pass_tolerance:
            break
        next_factor = applied_time_scale * remaining_scale
        if next_factor > recipe.max_time_scale + recipe.pass_tolerance:
            raise AtomRepairError(
                f"required time scale {next_factor:.6g} exceeds repair cap "
                f"{recipe.max_time_scale:.6g}"
            )
        if parent_duration * next_factor > recipe.max_output_duration_s + recipe.pass_tolerance:
            raise AtomRepairError(
                f"required repaired duration {parent_duration * next_factor:.6g}s exceeds "
                f"realtime atom cap {recipe.max_output_duration_s:.6g}s; split the candidate"
            )
        times = time_scale_trajectory(times, remaining_scale)
        applied_time_scale = next_factor
    else:
        raise AtomRepairError("retiming did not converge within max_iterations")

    analysis = analyze_kinematics(times, repaired_q, joints, limits=limits)
    if float(times[-1]) > recipe.max_output_duration_s + recipe.pass_tolerance:
        raise AtomRepairError(
            f"repaired duration {float(times[-1]):.6g}s exceeds realtime atom cap "
            f"{recipe.max_output_duration_s:.6g}s; split the candidate"
        )
    if analysis.report["result"] != "pass":
        raise AtomRepairError("repaired trajectory still violates kinematic constraints")
    safety = build_atom_safety(
        times,
        repaired_q,
        joints,
        limits=limits,
        collision_validator=collision_validator,
        neutral=reference,
    )
    if not safety["metrics"]["kinematics_passed"]:
        raise AtomRepairError("repaired trajectory failed final kinematic validation")
    if collision_validator is not None and not safety["collision"]["passed"]:
        raise AtomRepairError("repaired trajectory failed MuJoCo collision validation")
    detailed_report = build_motion_constraint_report(
        times,
        repaired_q,
        joints,
        limits=limits,
        collision_validator=collision_validator,
        neutral=reference,
    )
    constraint_profile_hash = None
    if collision_validator is not None:
        neutral_binding = source_atom.get("control", {}).get("neutral", {})
        if not isinstance(neutral_binding, Mapping):
            raise AtomRepairError("source Atom lacks a neutral-pose binding")
        constraint_profile = build_constraint_profile(
            limits,
            joints,
            collision_validator,
            neutral_id=neutral_binding.get("id"),
            neutral_content_hash=neutral_binding.get("content_hash"),
        )
        constraint_profile_hash = canonical_json_sha256(constraint_profile)
        detailed_report["constraint_profile"] = constraint_profile
        detailed_report["constraint_profile_sha256"] = constraint_profile_hash

    collision_report = detailed_report.get("collision")
    if isinstance(collision_report, dict) and isinstance(collision_report.get("model"), dict):
        model_path = str(collision_report["model"].get("path", ""))
        if model_path:
            collision_report["model"]["path"] = model_path.replace("\\", "/").rsplit("/", 1)[-1]

    repaired = deepcopy(source_atom)
    repaired["identity"].pop("content_hash", None)
    repaired["identity"]["revision"] = int(source_atom["identity"]["revision"]) + 1
    repaired["status"] = "candidate"
    repaired["control"]["trajectory"] = _trajectory_points(analysis)
    repaired["control"]["sample_rate_hz"] = float(
        (len(times) - 1) / (times[-1] - times[0])
    )
    if applied_time_scale != 1.0:
        _scale_timing(repaired, applied_time_scale)
    repaired["timing"]["nominal_duration_s"] = float(times[-1])
    repaired["control"]["trajectory"][-1]["t"] = float(times[-1])

    old_kinematics = source_atom.get("metadata", {}).get("kinematics", {})
    exit_kind = (
        str(old_kinematics.get("exit_kind", "unknown"))
        if isinstance(old_kinematics, Mapping)
        else "unknown"
    )
    safe_cutpoints, dropped_cutpoints = _recompute_safe_cutpoints(
        repaired,
        analysis,
        velocity_limit=cutpoint_velocity_limit,
        acceleration_limit=cutpoint_acceleration_limit,
        exit_kind=exit_kind,
    )
    repaired["timing"]["safe_cutpoints"] = safe_cutpoints

    start_point = repaired["control"]["trajectory"][0]
    end_point = repaired["control"]["trajectory"][-1]
    for interface_name, point in (("entry", start_point), ("exit", end_point)):
        for vector_name in ("q", "dq", "ddq"):
            repaired[interface_name][vector_name] = deepcopy(point[vector_name])
    repaired["safety"] = safety

    metadata = repaired.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise AtomRepairError("metadata must be an object")
    history = metadata.setdefault("repair_history", [])
    if not isinstance(history, list):
        raise AtomRepairError("metadata.repair_history must be an array")
    applied_recipe = {
        **recipe.as_dict(),
        "reference_pose_source": reference_source,
        "reference_pose_fallback_joints": reference_fallback_joints,
        "scaling_mode": (
            "per_limb" if recipe.limb_scales is not None else "global_scalar"
        ),
        "effective_limb_scales": effective_limb_scales,
        "uncontrolled_joint_scale": float(recipe.amplitude_scale),
        "applied_time_scale": float(applied_time_scale),
        "resulting_duration_s": float(times[-1]),
        "timing_mapping": {
            "method": "uniform_parent_time_mapping",
            "rationale": (
                "Amplitude scaling is affine and smoothing is centered zero-phase; "
                "phase and apex annotations are therefore mapped only by the applied "
                "uniform time scale, not re-detected."
            ),
            "parent_duration_s": parent_duration,
            "parent_phases": parent_phases,
            "parent_anchors": parent_anchors,
            "parent_safe_cutpoints": parent_cutpoints,
            "scale": float(applied_time_scale),
            "safe_cutpoint_filter": {
                "max_velocity_rad_s": cutpoint_velocity_limit,
                "max_acceleration_rad_s2": cutpoint_acceleration_limit,
                "moving_open_exit_is_unsafe": True,
                "kept": safe_cutpoints,
                "dropped": dropped_cutpoints,
            },
        },
    }
    applied_recipe["constraint_profile_sha256"] = constraint_profile_hash
    history.append(
        {
            "repairer": REPAIR_NAME,
            "repairer_version": REPAIR_VERSION,
            "parent_atom_id": str(source_atom["identity"]["id"]),
            "parent_revision": int(source_atom["identity"]["revision"]),
            "parent_content_hash": parent_hash,
            "recipe": deepcopy(applied_recipe),
        }
    )

    endpoint_is_safe = bool(safe_cutpoints) and abs(safe_cutpoints[-1] - float(times[-1])) <= 1e-9
    fixed_base_pass = bool(
        safety["metrics"]["kinematics_passed"]
        and safety["collision"]["checked"]
        and safety["collision"]["passed"]
    )
    metadata["eligibility"] = {
        "eligible_for_motion_graph_curation": bool(
            fixed_base_pass and endpoint_is_safe and exit_kind != "moving_open"
        ),
        "usable_for_motion_graph": False,
        "usable_for_semantic_planner": False,
        "usable_for_prosody": False,
        "usable_for_official_eval": False,
        "usable_for_real_robot": False,
        "requires_human_review": True,
    }
    metadata["processing"] = {
        "stage": "constraint_repair",
        "repairer": REPAIR_NAME,
        "repairer_version": REPAIR_VERSION,
        "parent_content_hash": parent_hash,
        "amplitude_scale": float(recipe.amplitude_scale),
        "scaling_mode": applied_recipe["scaling_mode"],
        "effective_limb_scales": deepcopy(effective_limb_scales),
        "smoothing_method": "zero_phase_normalized_convolution_endpoint_locked",
        "smoothing_passes": int(recipe.smoothing_passes),
        "applied_time_scale_from_parent": float(applied_time_scale),
        "output_sample_rate_hz": float(repaired["control"]["sample_rate_hz"]),
        "safe_cutpoint_velocity_rad_s": cutpoint_velocity_limit,
        "safe_cutpoint_acceleration_rad_s2": cutpoint_acceleration_limit,
        "safe_cutpoints_kept": len(safe_cutpoints),
        "safe_cutpoints_dropped": len(dropped_cutpoints),
        "not_for_hardware": True,
    }

    extensions = repaired.setdefault("extensions", {})
    if not isinstance(extensions, dict):
        raise AtomRepairError("extensions must be an object")
    extensions["repair_provenance"] = {
        "parent_atom_id": str(source_atom["identity"]["id"]),
        "parent_revision": int(source_atom["identity"]["revision"]),
        "parent_content_hash": parent_hash,
        "repairer": REPAIR_NAME,
        "repairer_version": REPAIR_VERSION,
        "recipe": deepcopy(applied_recipe),
        "constraint_profile_sha256": constraint_profile_hash,
    }
    recipe_hash = _json_hash(applied_recipe)
    report_hash = _json_hash(detailed_report)
    collision_model = (
        detailed_report.get("collision", {}).get("model", {})
        if isinstance(detailed_report.get("collision"), Mapping)
        else {}
    )
    evidence_payload = {
        "validator_version": safety["validator_version"],
        "repairer": REPAIR_NAME,
        "repairer_version": REPAIR_VERSION,
        "parent_content_hash": parent_hash,
        "repair_recipe_sha256": recipe_hash,
        "constraint_report_sha256": report_hash,
        "constraint_profile_sha256": constraint_profile_hash,
        "collision_checked": bool(safety["collision"]["checked"]),
        "collision_passed": bool(safety["collision"]["passed"]),
        "model_root_xml_sha256": collision_model.get("root_xml_sha256"),
        "dependency_closure_hash_available": False,
        "note": (
            "Repair evidence supersedes the parent validation evidence. The root XML "
            "hash is not a dependency-closure hash; real-robot safety is not certified."
        ),
    }
    extensions["validation_evidence"] = {
        **evidence_payload,
        "evidence_sha256": _json_hash(evidence_payload),
    }
    extensions["research_claim_boundary"] = {
        "fixed_base_only": True,
        "balance_checked": False,
        "torque_checked": False,
        "tracking_checked": False,
        "human_reviewed_after_repair": False,
        "real_robot_certified": False,
    }

    repaired_hash = atom_hash(repaired)
    repaired["identity"]["content_hash"] = repaired_hash
    try:
        validate_atom(repaired)
    except Exception as exc:
        raise AtomRepairError(f"repaired atom failed Schema v1 validation: {exc}") from exc

    preview = {
        "format": "g1_joint_motion_v1",
        "name": f"{repaired['identity']['id']}.r{repaired['identity']['revision']}",
        "duration": float(times[-1]),
        "joints": list(joints),
        "interpolation": "linear",
        "frames": [
            {"t": point["t"], "q": deepcopy(point["q"])}
            for point in repaired["control"]["trajectory"]
        ],
        "source_atom_hash": repaired_hash,
        "parent_atom_hash": parent_hash,
    }
    return RepairResult(
        atom=repaired,
        safety_report=detailed_report,
        preview_motion=preview,
        applied_recipe=applied_recipe,
        parent_hash=parent_hash,
    )


__all__ = [
    "AtomRepairError",
    "REPAIR_NAME",
    "REPAIR_VERSION",
    "RepairRecipe",
    "RepairResult",
    "endpoint_preserving_zero_phase_smooth",
    "joint_repair_limb",
    "repair_candidate",
]
