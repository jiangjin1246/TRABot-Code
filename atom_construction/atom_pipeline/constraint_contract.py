"""Frozen Validator Contract v2 used by extraction and every repair path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .joints import normalize_joint_names
from .safety import COLLISION_VALIDATOR_VERSION, VALIDATOR_VERSION, MotionLimits, MujocoCollisionValidator

FORMAT = "atom-validator-contract-v2"
PROFILE_ID = "g1-fixed-base-upper-body-v2"
ROOT_POLICY = "all_free_joints_pinned_to_model_qpos0"


class ConstraintContractError(ValueError):
    """Frozen evidence is absent, malformed, or differs from the live model."""


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _number(value: Any, label: str, *, positive: bool = False, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConstraintContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (minimum is not None and result < minimum):
        raise ConstraintContractError(f"{label} is outside its valid range")
    return result


def _object(value: Any, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstraintContractError(f"{label} must be an object")
    missing, unknown = sorted(keys - set(value)), sorted(set(value) - keys)
    if missing or unknown:
        raise ConstraintContractError(f"{label} keys mismatch; missing={missing}, unknown={unknown}")
    return value


def _joint_values(value: Any, label: str, joints: tuple[str, ...], *, ranges: bool = False) -> dict[str, Any]:
    mapping = _object(value, label, set(joints))
    if not ranges:
        return {joint: _number(mapping[joint], f"{label}.{joint}", positive=True) for joint in joints}
    result: dict[str, tuple[float, float]] = {}
    for joint in joints:
        raw = mapping[joint]
        if not isinstance(raw, list) or len(raw) != 2:
            raise ConstraintContractError(f"{label}.{joint} must be [low, high]")
        low, high = _number(raw[0], label), _number(raw[1], label)
        if low >= high:
            raise ConstraintContractError(f"{label}.{joint} must satisfy low < high")
        result[joint] = (low, high)
    return result


def build_constraint_profile(
    limits: MotionLimits,
    joints: Sequence[str],
    validator: MujocoCollisionValidator,
    *,
    neutral_id: str,
    neutral_content_hash: str,
) -> dict[str, Any]:
    """Serialize every kinematic and fixed-base collision input."""
    if not isinstance(neutral_id, str) or not neutral_id:
        raise ConstraintContractError("neutral.id must be a non-empty string")
    if (
        not isinstance(neutral_content_hash, str)
        or len(neutral_content_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in neutral_content_hash)
    ):
        raise ConstraintContractError("neutral.content_hash must be a SHA-256 digest")
    controlled = normalize_joint_names(joints)
    ranges = validator.joint_ranges(controlled)
    if set(ranges) != set(controlled) or set(limits.joint_ranges) != set(controlled):
        raise ConstraintContractError("joint ranges must exactly cover controlled joints")
    for joint in controlled:
        if not all(math.isclose(float(a), float(b), abs_tol=1e-12) for a, b in zip(limits.joint_ranges[joint], ranges[joint])):
            raise ConstraintContractError(f"model range mismatch for {joint}")
    mujoco, _ = validator._load()
    scene = validator.scene_path.resolve()
    return {
        "format": FORMAT,
        "profile_id": PROFILE_ID,
        "validator_version": VALIDATOR_VERSION,
        "controlled_joints": list(controlled),
        "neutral": {
            "id": neutral_id,
            "content_hash": neutral_content_hash.lower(),
        },
        "motion_limits": {
            "velocity_rad_s": float(limits.velocity_rad_s),
            "acceleration_rad_s2": float(limits.acceleration_rad_s2),
            "jerk_rad_s3": float(limits.jerk_rad_s3),
            "velocity_by_joint": {joint: limits.threshold("velocity", joint) for joint in controlled},
            "acceleration_by_joint": {joint: limits.threshold("acceleration", joint) for joint in controlled},
            "jerk_by_joint": {joint: limits.threshold("jerk", joint) for joint in controlled},
            "joint_ranges_rad": {joint: [float(x) for x in ranges[joint]] for joint in controlled},
            "position_margin_rad": float(limits.position_margin_rad),
            "warning_fraction": float(limits.warning_fraction),
            "time_scale_safety_factor": float(limits.time_scale_safety_factor),
            "max_time_scale": float(limits.max_time_scale),
        },
        "collision_policy": {
            "validator_version": COLLISION_VALIDATOR_VERSION,
            "backend_id": validator.backend_id,
            "sample_stride": int(validator.sample_stride),
            "contact_distance_threshold_m": float(validator.contact_distance_threshold_m),
            "baseline_tolerance_m": float(validator.baseline_tolerance_m),
            "allowed_environment_robot_tokens": list(validator.allowed_environment_robot_tokens),
            "root_policy": ROOT_POLICY,
            "model_fingerprint": {
                "filename": scene.name,
                "root_xml_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
                "controlled_joint_ranges_rad": {joint: [float(x) for x in ranges[joint]] for joint in controlled},
                "dependency_closure_hash_available": False,
            },
        },
    }


def motion_limits_from_profile(profile: Mapping[str, Any], joints: Sequence[str]) -> MotionLimits:
    root = _object(profile, "profile", {
        "format", "profile_id", "validator_version", "controlled_joints",
        "neutral", "motion_limits", "collision_policy",
    })
    if root["format"] != FORMAT or root["profile_id"] != PROFILE_ID or root["validator_version"] != VALIDATOR_VERSION:
        raise ConstraintContractError("unsupported constraint profile")
    controlled = normalize_joint_names(joints)
    if root["controlled_joints"] != list(controlled):
        raise ConstraintContractError("controlled_joints mismatch")
    neutral = _object(root["neutral"], "neutral", {"id", "content_hash"})
    neutral_id, neutral_hash = neutral["id"], neutral["content_hash"]
    if not isinstance(neutral_id, str) or not neutral_id:
        raise ConstraintContractError("neutral.id must be a non-empty string")
    if (
        not isinstance(neutral_hash, str)
        or len(neutral_hash) != 64
        or neutral_hash != neutral_hash.lower()
        or any(character not in "0123456789abcdef" for character in neutral_hash)
    ):
        raise ConstraintContractError(
            "neutral.content_hash must be a lowercase SHA-256 digest"
        )
    motion = _object(root["motion_limits"], "motion_limits", {
        "velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3", "velocity_by_joint",
        "acceleration_by_joint", "jerk_by_joint", "joint_ranges_rad", "position_margin_rad",
        "warning_fraction", "time_scale_safety_factor", "max_time_scale",
    })
    limits = MotionLimits(
        velocity_rad_s=_number(motion["velocity_rad_s"], "velocity", positive=True),
        acceleration_rad_s2=_number(motion["acceleration_rad_s2"], "acceleration", positive=True),
        jerk_rad_s3=_number(motion["jerk_rad_s3"], "jerk", positive=True),
        velocity_by_joint=_joint_values(motion["velocity_by_joint"], "velocity_by_joint", controlled),
        acceleration_by_joint=_joint_values(motion["acceleration_by_joint"], "acceleration_by_joint", controlled),
        jerk_by_joint=_joint_values(motion["jerk_by_joint"], "jerk_by_joint", controlled),
        joint_ranges=_joint_values(motion["joint_ranges_rad"], "joint_ranges_rad", controlled, ranges=True),
        position_margin_rad=_number(motion["position_margin_rad"], "position_margin", minimum=0.0),
        warning_fraction=_number(motion["warning_fraction"], "warning_fraction", positive=True),
        time_scale_safety_factor=_number(motion["time_scale_safety_factor"], "time safety", positive=True),
        max_time_scale=_number(motion["max_time_scale"], "max_time_scale", positive=True),
    )
    if limits.max_time_scale < 1.0:
        raise ConstraintContractError("max_time_scale must be at least one")
    return limits



def _require_report_matches_profile(
    report: Mapping[str, Any],
    profile: Mapping[str, Any],
    limits: MotionLimits,
    joints: tuple[str, ...],
) -> None:
    """Prove that r1 was actually evaluated with the advertised snapshot."""
    kinematics = report.get("kinematics")
    if not isinstance(kinematics, Mapping):
        raise ConstraintContractError("r1 report lacks kinematics evidence")
    if kinematics.get("validator") != VALIDATOR_VERSION or kinematics.get("joints") != list(joints):
        raise ConstraintContractError("r1 kinematics validator/joints differ from profile")
    thresholds = kinematics.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ConstraintContractError("r1 report lacks effective thresholds")
    expected_fallback = {
        "velocity_rad_s": float(limits.velocity_rad_s),
        "acceleration_rad_s2": float(limits.acceleration_rad_s2),
        "jerk_rad_s3": float(limits.jerk_rad_s3),
    }
    if thresholds.get("global_fallback") != expected_fallback:
        raise ConstraintContractError("r1 global thresholds differ from profile")
    if thresholds.get("position_margin_rad") != float(limits.position_margin_rad):
        raise ConstraintContractError("r1 position margin differs from profile")
    if thresholds.get("warning_fraction") != float(limits.warning_fraction):
        raise ConstraintContractError("r1 warning fraction differs from profile")
    expected_effective = {
        joint: {
            "velocity_rad_s": limits.threshold("velocity", joint),
            "acceleration_rad_s2": limits.threshold("acceleration", joint),
            "jerk_rad_s3": limits.threshold("jerk", joint),
            "model_position_range_rad": [float(x) for x in limits.joint_ranges[joint]],
        }
        for joint in joints
    }
    if thresholds.get("per_joint_effective") != expected_effective:
        raise ConstraintContractError("r1 per-joint thresholds/ranges differ from profile")
    scaling = kinematics.get("time_scaling")
    if (
        not isinstance(scaling, Mapping)
        or scaling.get("configured_max_factor") != float(limits.max_time_scale)
        or scaling.get("safety_factor") != float(limits.time_scale_safety_factor)
    ):
        raise ConstraintContractError("r1 time-scaling policy differs from profile")
    collision = report.get("collision")
    policy = profile["collision_policy"]
    if not isinstance(collision, Mapping):
        raise ConstraintContractError("r1 report lacks collision evidence")
    expected_collision = {
        "validator": policy["validator_version"],
        "backend": policy["backend_id"],
        "root_policy": policy["root_policy"],
        "sample_stride": policy["sample_stride"],
        "contact_distance_threshold_m": policy["contact_distance_threshold_m"],
        "baseline_tolerance_m": policy["baseline_tolerance_m"],
        "allowed_environment_robot_tokens": policy["allowed_environment_robot_tokens"],
    }
    for key, expected in expected_collision.items():
        if collision.get(key) != expected:
            raise ConstraintContractError(f"r1 collision field {key} differs from profile")
    model = collision.get("model")
    fingerprint = policy["model_fingerprint"]
    if not isinstance(model, Mapping):
        raise ConstraintContractError("r1 collision report lacks model fingerprint")
    if Path(str(model.get("path", ""))).name != fingerprint["filename"]:
        raise ConstraintContractError("r1 collision scene filename differs from profile")
    for key in ("root_xml_sha256", "mujoco_version"):
        if model.get(key) != fingerprint[key]:
            raise ConstraintContractError(f"r1 collision model {key} differs from profile")

def frozen_context_from_report(
    report: Mapping[str, Any] | None,
    joints: Sequence[str],
    scene_path: str | Path,
) -> tuple[MotionLimits, MujocoCollisionValidator, dict[str, Any], str]:
    """Reconstruct exact r1 context; legacy/mismatched evidence fails closed."""
    if not isinstance(report, Mapping) or report.get("validator") != VALIDATOR_VERSION:
        raise ConstraintContractError("Validator Contract v2 report required; re-extract legacy r1")
    profile, expected = report.get("constraint_profile"), report.get("constraint_profile_sha256")
    if not isinstance(profile, Mapping) or not isinstance(expected, str):
        raise ConstraintContractError("report lacks frozen constraint profile/hash")
    frozen = json.loads(json.dumps(profile, allow_nan=False))
    actual = canonical_json_sha256(frozen)
    if actual != expected.lower():
        raise ConstraintContractError("constraint profile SHA-256 mismatch")
    controlled = normalize_joint_names(joints)
    limits = motion_limits_from_profile(frozen, controlled)
    _require_report_matches_profile(report, frozen, limits, controlled)
    collision = _object(frozen["collision_policy"], "collision_policy", {
        "validator_version", "backend_id", "sample_stride", "contact_distance_threshold_m",
        "baseline_tolerance_m", "allowed_environment_robot_tokens", "root_policy", "model_fingerprint",
    })
    if collision["validator_version"] != COLLISION_VALIDATOR_VERSION or collision["root_policy"] != ROOT_POLICY:
        raise ConstraintContractError("collision validator/root policy mismatch")
    tokens, stride = collision["allowed_environment_robot_tokens"], collision["sample_stride"]
    if not isinstance(tokens, list) or len(set(tokens)) != len(tokens) or any(not isinstance(x, str) or not x for x in tokens):
        raise ConstraintContractError("allowed collision tokens are invalid")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ConstraintContractError("sample_stride must be a positive integer")
    fingerprint = collision.get("model_fingerprint")
    if not isinstance(fingerprint, Mapping):
        raise ConstraintContractError("collision model fingerprint is missing")
    scene = Path(scene_path).resolve()
    if fingerprint.get("filename") != scene.name:
        raise ConstraintContractError("current scene filename differs from frozen r1")
    current_root_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
    if fingerprint.get("root_xml_sha256") != current_root_hash:
        raise ConstraintContractError(
            "current scene root XML hash differs from frozen r1"
        )
    validator = MujocoCollisionValidator(
        scene_path,
        sample_stride=stride,
        contact_distance_threshold_m=_number(
            collision["contact_distance_threshold_m"], "contact distance"
        ),
        baseline_tolerance_m=_number(
            collision["baseline_tolerance_m"], "baseline tolerance", minimum=0.0
        ),
        backend_id=collision["backend_id"],
        allowed_environment_robot_tokens=tokens,
    )
    neutral = frozen["neutral"]
    current = build_constraint_profile(
        limits,
        controlled,
        validator,
        neutral_id=neutral["id"],
        neutral_content_hash=neutral["content_hash"],
    )
    if current != frozen or canonical_json_sha256(current) != actual:
        raise ConstraintContractError("current scene/model/ranges/collision policy differ from frozen r1")
    return limits, validator, frozen, actual


__all__ = [
    "ConstraintContractError", "build_constraint_profile", "canonical_json_sha256",
    "frozen_context_from_report", "motion_limits_from_profile",
]
