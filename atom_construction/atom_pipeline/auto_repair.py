"""Bounded deterministic Auto-Repair search for Atom Schema v1.

Trials run only in memory.  The caller may publish the single least-distorting
MuJoCo-validated winner, but this module never writes artifacts or approves an
Atom.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .repair import (
    AtomRepairError,
    RepairRecipe,
    RepairResult,
    joint_repair_limb,
    repair_candidate,
)
from .safety import (
    CONSERVATIVE_G1_LIMITS,
    MotionLimits,
    MujocoCollisionValidator,
    analyze_kinematics,
)
from .schema import atom_hash, validate_atom


AUTO_REPAIR_NAME = "atom-auto-repair-agent"
AUTO_REPAIR_VERSION = "2.0.0"
POLICY_NAME = "minimal_distortion_limb_v2"


class AutoRepairCanceled(RuntimeError):
    """Raised cooperatively between bounded trials."""


@dataclass(frozen=True, slots=True)
class AutoRepairPolicy:
    max_trials: int = 24
    max_collision_trials: int = 8
    amplitude_scales: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.75, 0.65, 0.5)
    smoothing_passes: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    max_time_scale: float = 3.0
    max_output_duration_s: float = 10.0
    max_trajectory_nrmse: float = 0.75
    min_apex_retention: float = 0.35

    def __post_init__(self) -> None:
        if self.max_trials < 1 or self.max_collision_trials < 1:
            raise ValueError("trial limits must be positive")
        if any(not 0.0 < float(value) <= 1.0 for value in self.amplitude_scales):
            raise ValueError("amplitude scales must be in (0,1]")
        if any(isinstance(value, bool) or int(value) < 0 for value in self.smoothing_passes):
            raise ValueError("smoothing passes must be non-negative integers")
        if self.max_time_scale < 1.0 or self.max_output_duration_s <= 0.0:
            raise ValueError("time limits are invalid")


@dataclass(frozen=True, slots=True)
class FailureProfile:
    position: bool
    velocity: bool
    acceleration: bool
    jerk: bool
    collision: bool
    codes: tuple[str, ...]

    @property
    def any_failure(self) -> bool:
        return any((self.position, self.velocity, self.acceleration, self.jerk, self.collision))


@dataclass(frozen=True, slots=True)
class FidelityScore:
    total: float
    trajectory_nrmse: float
    apex_nrmse: float
    endpoint_nrmse: float
    duration_log_ratio: float
    apex_retention: float
    amplitude_scale: float
    smoothing_passes: int
    time_scale: float
    limb_scales: Mapping[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": float(self.total),
            "trajectory_nrmse": float(self.trajectory_nrmse),
            "apex_nrmse": float(self.apex_nrmse),
            "endpoint_nrmse": float(self.endpoint_nrmse),
            "duration_log_ratio": float(self.duration_log_ratio),
            "apex_retention": float(self.apex_retention),
            "amplitude_scale": float(self.amplitude_scale),
            "smoothing_passes": int(self.smoothing_passes),
            "time_scale": float(self.time_scale),
            "limb_scales": (
                {key: float(value) for key, value in self.limb_scales.items()}
                if self.limb_scales is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class AutoRepairResult:
    status: str
    parent_content_hash: str
    failure_profile: FailureProfile
    winner: RepairResult | None
    winner_score: FidelityScore | None
    recipes_attempted: int
    kinematic_feasible: int
    collision_checked: int
    reason_code: str | None
    trial_outcomes: tuple[dict[str, Any], ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "agent": AUTO_REPAIR_NAME,
            "agent_version": AUTO_REPAIR_VERSION,
            "policy": POLICY_NAME,
            "status": self.status,
            "parent_content_hash": self.parent_content_hash,
            "failure_codes": list(self.failure_profile.codes),
            "recipes_attempted": self.recipes_attempted,
            "kinematic_feasible": self.kinematic_feasible,
            "collision_checked": self.collision_checked,
            "reason_code": self.reason_code,
            "winner_fidelity": self.winner_score.as_dict() if self.winner_score else None,
            "trial_outcomes": [deepcopy(item) for item in self.trial_outcomes],
            "claim_boundary": (
                "Fixed-base kinematic/contact repair only; human review remains required "
                "and real-robot safety is not certified."
            ),
        }


def _violations(atom: Mapping[str, Any], report: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if isinstance(report, Mapping):
        kinematics = report.get("kinematics")
        if isinstance(kinematics, Mapping) and isinstance(kinematics.get("violations"), list):
            values = list(kinematics["violations"])
            collision = report.get("collision")
            if isinstance(collision, Mapping) and isinstance(collision.get("violations"), list):
                values.extend(collision["violations"])
            return [item for item in values if isinstance(item, Mapping)]
        if isinstance(report.get("violations"), list):
            return [item for item in report["violations"] if isinstance(item, Mapping)]
    safety = atom.get("safety")
    return [
        item for item in (safety.get("violations", []) if isinstance(safety, Mapping) else [])
        if isinstance(item, Mapping)
    ]


def analyze_failure(atom: Mapping[str, Any], report: Mapping[str, Any] | None = None) -> FailureProfile:
    codes = {str(item.get("code") or "").upper() for item in _violations(atom, report)}
    safety = atom.get("safety") if isinstance(atom.get("safety"), Mapping) else {}
    embedded_collision = (
        safety.get("collision") if isinstance(safety.get("collision"), Mapping) else {}
    )
    report_collision = report.get("collision") if isinstance(report, Mapping) else None
    if isinstance(report_collision, Mapping):
        collision_failed = (
            str(report_collision.get("result", "")).lower() == "fail"
            or bool(report_collision.get("checked"))
            and not bool(report_collision.get("passed"))
        )
    else:
        collision_failed = (
            bool(embedded_collision.get("checked"))
            and not bool(embedded_collision.get("passed"))
        )
    collision_failed = collision_failed or any(
        "COLLISION" in code or "CONTACT" in code for code in codes
    )
    return FailureProfile(
        position=any("POSITION" in code and "JOINT" in code for code in codes),
        velocity=any("VELOCITY" in code for code in codes),
        acceleration=any("ACCELERATION" in code for code in codes),
        jerk=any("JERK" in code for code in codes),
        collision=collision_failed,
        codes=tuple(sorted(code for code in codes if code)),
    )


def generate_recipes(
    profile: FailureProfile,
    policy: AutoRepairPolicy | None = None,
) -> list[RepairRecipe]:
    policy = policy or AutoRepairPolicy()
    if not profile.any_failure:
        return []
    if profile.position or profile.collision:
        amplitudes = policy.amplitude_scales
    else:
        # Dynamic violations are repaired only by smoothing/retiming; amplitude
        # loss cannot be justified when position and collision already pass.
        amplitudes = (1.0,)
    if profile.jerk:
        smooth_order = (1, 2, 3, 4, 5, 0)
    elif profile.acceleration:
        smooth_order = (1, 2, 3, 0, 4, 5)
    else:
        smooth_order = (0, 1, 2, 3, 4, 5)
    smooth_order = tuple(value for value in smooth_order if value in policy.smoothing_passes)
    recipes: list[RepairRecipe] = []
    # Interleave smoothing across amplitudes so a bounded search still reaches
    # every configured amplitude, including the collision-repair floor.
    for passes in smooth_order:
        for amplitude in amplitudes:
            recipes.append(
                RepairRecipe(
                    amplitude_scale=float(amplitude),
                    smoothing_passes=int(passes),
                    retime=True,
                    max_time_scale=policy.max_time_scale,
                    max_output_duration_s=policy.max_output_duration_s,
                )
            )
            if len(recipes) >= policy.max_trials:
                return recipes
    return recipes


def _trajectory(atom: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    control = atom["control"]
    joints = tuple(str(value) for value in control["joints"])
    points = control["trajectory"]
    times = np.asarray([float(point["t"]) for point in points], dtype=np.float64)
    q = np.asarray(
        [[float(point["q"][joint]) for joint in joints] for point in points],
        dtype=np.float64,
    )
    return times, q, joints


def _neutral_vector(
    atom: Mapping[str, Any],
    q: np.ndarray,
    joints: Sequence[str],
    neutral_pose: Mapping[str, float] | Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if neutral_pose is None:
        return q[0].copy()
    if isinstance(neutral_pose, Mapping):
        return np.asarray(
            [
                float(neutral_pose.get(joint, q[0, index]))
                for index, joint in enumerate(joints)
            ],
            dtype=np.float64,
        )
    result = np.asarray(neutral_pose, dtype=np.float64)
    if result.shape != (len(joints),) or not np.all(np.isfinite(result)):
        raise AtomRepairError("neutral_pose must contain one finite value per joint")
    return result


def maximum_position_limb_scales(
    atom: Mapping[str, Any],
    limits: MotionLimits,
    neutral_pose: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Return analytic maximum scales that keep each limb in joint ranges."""

    _, q, joints = _trajectory(atom)
    reference = _neutral_vector(atom, q, joints, neutral_pose)
    result = {"left_arm": 1.0, "right_arm": 1.0, "waist": 1.0}
    for column, joint in enumerate(joints):
        limb = joint_repair_limb(joint)
        if limb is None or joint not in limits.joint_ranges:
            continue
        raw_low, raw_high = limits.joint_ranges[joint]
        low = float(raw_low) + float(limits.position_margin_rad)
        high = float(raw_high) - float(limits.position_margin_rad)
        center = float(reference[column])
        if low >= high or center < low - 1e-12 or center > high + 1e-12:
            raise AtomRepairError(
                f"reference pose for {joint} is outside its effective position range"
            )
        alpha = 1.0
        delta = q[:, column] - center
        positive = delta > 0.0
        negative = delta < 0.0
        if np.any(positive):
            alpha = min(alpha, float(np.min((high - center) / delta[positive])))
        if np.any(negative):
            alpha = min(alpha, float(np.min((low - center) / delta[negative])))
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise AtomRepairError(f"no positive position-feasible scale exists for {joint}")
        if alpha < 1.0:
            alpha = max(1e-12, alpha * (1.0 - 1e-9))
        result[limb] = min(result[limb], alpha)
    return result


def _collision_limbs(report: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(report, Mapping):
        return set()
    items: list[Mapping[str, Any]] = []
    for key in ("violations", "contact_pairs"):
        values = report.get(key)
        if isinstance(values, list):
            items.extend(value for value in values if isinstance(value, Mapping))
    collision = report.get("collision")
    if isinstance(collision, Mapping):
        for key in ("violations", "contact_pairs"):
            values = collision.get(key)
            if isinstance(values, list):
                items.extend(value for value in values if isinstance(value, Mapping))
    limbs: set[str] = set()
    for item in items:
        names: list[str] = []
        for key in ("joint", "body_names", "geom_names"):
            value = item.get(key)
            if isinstance(value, str):
                names.append(value.lower())
            elif isinstance(value, Sequence):
                names.extend(str(part).lower() for part in value)
        joined = " ".join(names)
        if "left" in joined and any(
            token in joined for token in ("arm", "shoulder", "elbow", "wrist", "hand")
        ):
            limbs.add("left_arm")
        if "right" in joined and any(
            token in joined for token in ("arm", "shoulder", "elbow", "wrist", "hand")
        ):
            limbs.add("right_arm")
        # A torso is commonly the passive collision counterpart of an arm;
        # only explicit waist geometry identifies waist actuation as involved.
        if "waist" in joined:
            limbs.add("waist")
    return limbs


def _smooth_order(profile: FailureProfile, policy: AutoRepairPolicy) -> tuple[int, ...]:
    if profile.jerk:
        preferred = (1, 2, 3, 4, 5, 0)
    elif profile.acceleration:
        preferred = (1, 2, 3, 0, 4, 5)
    else:
        preferred = (0, 1, 2, 3, 4, 5)
    return tuple(value for value in preferred if value in policy.smoothing_passes)


def _adaptive_recipes(
    atom: Mapping[str, Any],
    profile: FailureProfile,
    policy: AutoRepairPolicy,
    limits: MotionLimits,
    neutral_pose: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    collision_limbs: set[str],
) -> list[RepairRecipe]:
    base = (
        maximum_position_limb_scales(atom, limits, neutral_pose)
        if profile.position
        else {"left_arm": 1.0, "right_arm": 1.0, "waist": 1.0}
    )
    factors = policy.amplitude_scales if profile.collision else (1.0,)
    recipes: list[RepairRecipe] = []
    seen: set[tuple[Any, ...]] = set()
    # Interleave amplitude buckets within each smoothing level so the bounded
    # kinematic budget reaches every collision scale and multiple smoothings.
    for passes in _smooth_order(profile, policy):
        for factor in factors:
            scales = dict(base)
            for limb in collision_limbs:
                scales[limb] *= float(factor)
            key = (int(passes),) + tuple(round(scales[limb], 12) for limb in (
                "left_arm", "right_arm", "waist"
            ))
            if key in seen:
                continue
            seen.add(key)
            limb_scales = scales if (profile.position or profile.collision) else None
            recipes.append(
                RepairRecipe(
                    amplitude_scale=1.0,
                    limb_scales=limb_scales,
                    smoothing_passes=int(passes),
                    retime=True,
                    max_time_scale=policy.max_time_scale,
                    max_output_duration_s=policy.max_output_duration_s,
                )
            )
            if len(recipes) >= policy.max_trials:
                return recipes
    return recipes


def _interpolate(times: np.ndarray, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.empty((len(target), q.shape[1]), dtype=np.float64)
    for index in range(q.shape[1]):
        values[:, index] = np.interp(target, times, q[:, index])
    return values


def _apex_time(atom: Mapping[str, Any], duration: float) -> float:
    anchors = atom.get("timing", {}).get("anchors", {})
    for key in ("apex_s", "apex", "stroke_apex_s"):
        value = anchors.get(key) if isinstance(anchors, Mapping) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return min(duration, max(0.0, float(value)))
    return 0.5 * duration


def score_fidelity(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    recipe: RepairRecipe,
) -> FidelityScore:
    parent_t, parent_q, parent_joints = _trajectory(parent)
    child_t, child_q, child_joints = _trajectory(child)
    if parent_joints != child_joints:
        raise ValueError("child joint order differs from parent")
    parent_duration, child_duration = float(parent_t[-1]), float(child_t[-1])
    u = np.linspace(0.0, 1.0, max(64, len(parent_t), len(child_t)))
    parent_sample = _interpolate(parent_t, parent_q, u * parent_duration)
    child_sample = _interpolate(child_t, child_q, u * child_duration)
    ranges = np.ptp(parent_sample, axis=0)
    active = ranges > 1e-4
    denominator = float(np.linalg.norm(ranges[active])) if np.any(active) else 1.0
    trajectory_nrmse = (
        float(np.linalg.norm((child_sample - parent_sample)[:, active]))
        / max(1e-9, math.sqrt(len(u)) * denominator)
        if np.any(active) else 0.0
    )
    parent_apex = _interpolate(
        parent_t, parent_q, np.asarray([_apex_time(parent, parent_duration)])
    )[0]
    child_apex = _interpolate(
        child_t, child_q, np.asarray([_apex_time(child, child_duration)])
    )[0]
    apex_nrmse = float(np.linalg.norm(child_apex - parent_apex) / max(1e-9, denominator))
    endpoint_nrmse = float(
        (np.linalg.norm(child_q[0] - parent_q[0]) + np.linalg.norm(child_q[-1] - parent_q[-1]))
        / max(1e-9, 2.0 * denominator)
    )
    duration_log_ratio = abs(math.log(child_duration / parent_duration))
    parent_energy = float(np.linalg.norm(parent_apex - parent_q[0]))
    child_energy = float(np.linalg.norm(child_apex - child_q[0]))
    apex_retention = min(2.0, child_energy / parent_energy) if parent_energy > 1e-9 else 1.0
    total = (
        0.45 * trajectory_nrmse + 0.25 * apex_nrmse
        + 0.15 * duration_log_ratio + 0.10 * endpoint_nrmse
        + 0.05 * min(1.0, recipe.smoothing_passes / 5.0)
    )
    recorded_limb_scales = (
        {key: float(value) for key, value in recipe.limb_scales.items()}
        if recipe.limb_scales is not None else None
    )
    effective_amplitude = (
        min(recorded_limb_scales.values())
        if recorded_limb_scales else float(recipe.amplitude_scale)
    )
    return FidelityScore(
        total, trajectory_nrmse, apex_nrmse, endpoint_nrmse, duration_log_ratio,
        apex_retention, effective_amplitude, recipe.smoothing_passes,
        child_duration / parent_duration, recorded_limb_scales,
    )


def _cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise AutoRepairCanceled("auto-repair canceled")


def _passes(result: RepairResult, require_collision: bool) -> bool:
    safety = result.atom.get("safety", {})
    metrics = safety.get("metrics", {}) if isinstance(safety, Mapping) else {}
    collision = safety.get("collision", {}) if isinstance(safety, Mapping) else {}
    return bool(metrics.get("kinematics_passed", metrics.get("passed", False))) and (
        not require_collision
        or (bool(collision.get("checked")) and bool(collision.get("passed")))
    )


def _fidelity_order_key(
    value: tuple[FidelityScore, RepairRecipe],
) -> tuple[float, float, float, int]:
    fidelity, recipe = value
    return (
        fidelity.total,
        -fidelity.apex_retention,
        fidelity.time_scale,
        recipe.smoothing_passes,
    )


def _scale_signature(recipe: RepairRecipe) -> tuple[float, float, float]:
    scales = recipe.limb_scales or {}
    return tuple(
        round(float(scales.get(limb, recipe.amplitude_scale)), 12)
        for limb in ("left_arm", "right_arm", "waist")
    )


def _collision_trial_order(
    feasible: Sequence[tuple[FidelityScore, RepairRecipe]],
    budget: int,
    collision_limbs: set[str],
) -> list[tuple[FidelityScore, RepairRecipe]]:
    """Cover every available amplitude bucket within the collision budget.

    The least-distorting bucket gets the first trial and, when the budget is
    larger than the bucket count, the first extra smoothing trial.  One trial
    is then reserved for every remaining amplitude bucket before further
    within-bucket variants are considered.
    """

    groups: dict[tuple[float, float, float], list[tuple[FidelityScore, RepairRecipe]]] = {}
    for item in feasible:
        groups.setdefault(_scale_signature(item[1]), []).append(item)
    for items in groups.values():
        items.sort(key=_fidelity_order_key)
    def collision_scale(items: list[tuple[FidelityScore, RepairRecipe]]) -> float:
        recipe = items[0][1]
        scales = recipe.limb_scales or {}
        targets = collision_limbs or set(scales)
        return min(
            float(scales.get(limb, recipe.amplitude_scale))
            for limb in targets
        )

    buckets = sorted(
        groups.values(),
        key=lambda items: (
            -collision_scale(items),
            _fidelity_order_key(items[0]),
        ),
    )
    if not buckets or budget < 1:
        return []

    selected: list[tuple[FidelityScore, RepairRecipe]] = [buckets[0].pop(0)]
    extra_before_coverage = max(0, budget - len(buckets))
    while extra_before_coverage and buckets[0] and len(selected) < budget:
        selected.append(buckets[0].pop(0))
        extra_before_coverage -= 1
    for bucket in buckets[1:]:
        if len(selected) >= budget:
            break
        selected.append(bucket.pop(0))
    remaining = sorted(
        (item for bucket in buckets for item in bucket),
        key=_fidelity_order_key,
    )
    selected.extend(remaining[: max(0, budget - len(selected))])
    return selected


def search_best_repair(
    atom: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
    policy: AutoRepairPolicy | None = None,
    limits: MotionLimits | None = None,
    collision_validator: MujocoCollisionValidator | None = None,
    neutral_pose: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    require_collision: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> AutoRepairResult:
    validate_atom(atom)
    parent = deepcopy(dict(atom))
    parent_hash = atom_hash(parent)
    requested_profile = analyze_failure(parent, report)
    if require_collision and collision_validator is None:
        return AutoRepairResult(
            "unrepairable", parent_hash, requested_profile, None, None, 0, 0, 0,
            "COLLISION_VALIDATOR_REQUIRED",
        )

    policy = policy or AutoRepairPolicy()
    limits_were_supplied = limits is not None
    limits = limits or CONSERVATIVE_G1_LIMITS
    if policy.max_time_scale > limits.max_time_scale:
        policy = replace(
            policy, max_time_scale=float(limits.max_time_scale)
        )
    parent_times, parent_q, parent_joints = _trajectory(parent)
    _cancel(cancel_check)

    if not limits_were_supplied and collision_validator is not None:
        if not hasattr(collision_validator, "joint_ranges"):
            return AutoRepairResult(
                "unrepairable", parent_hash, requested_profile, None, None, 0, 0, 0,
                "MODEL_JOINT_RANGES_UNAVAILABLE",
            )
        try:
            model_ranges = collision_validator.joint_ranges(parent_joints)
        except Exception as exc:
            return AutoRepairResult(
                "unrepairable", parent_hash, requested_profile, None, None, 0, 0, 0,
                "MODEL_JOINT_RANGES_UNAVAILABLE",
                ({"status": "range_load_failed", "reason": str(exc)[:240]},),
            )
        limits = replace(limits, joint_ranges=model_ranges)

    try:
        fresh_kinematics = analyze_kinematics(
            parent_times, parent_q, parent_joints, limits=limits
        )
    except Exception as exc:
        return AutoRepairResult(
            "unrepairable", parent_hash, requested_profile, None, None, 0, 0, 0,
            "KINEMATIC_VALIDATION_FAILED",
            ({"status": "raw_validation_failed", "reason": str(exc)[:240]},),
        )

    fresh_collision: Mapping[str, Any] | None = None
    if collision_validator is not None:
        if not hasattr(collision_validator, "validate"):
            return AutoRepairResult(
                "unrepairable", parent_hash, requested_profile, None, None, 0, 0, 0,
                "COLLISION_VALIDATOR_INVALID",
            )
        try:
            neutral = _neutral_vector(parent, parent_q, parent_joints, neutral_pose)
            raw_collision = collision_validator.validate(
                parent_times, parent_q, parent_joints, neutral=neutral
            )
            if not isinstance(raw_collision, Mapping):
                raise TypeError("collision validator returned a non-mapping report")
            fresh_collision = raw_collision
        except Exception as exc:
            return AutoRepairResult(
                "unrepairable", parent_hash, requested_profile, None, None, 0, 0, 0,
                "COLLISION_VALIDATION_FAILED",
                ({"status": "raw_collision_failed", "reason": str(exc)[:240]},),
            )

    fresh_report: dict[str, Any] = {"kinematics": fresh_kinematics.report}
    if fresh_collision is not None:
        fresh_report["collision"] = fresh_collision
    elif not require_collision:
        # Numerical-only callers explicitly opted out of collision repair; do
        # not resurrect stale embedded collision failures in the fresh profile.
        fresh_report["collision"] = {"result": "pass", "violations": []}
    profile = analyze_failure(parent, fresh_report)

    missing_position_ranges = sorted(
        set(parent_joints) - set(limits.joint_ranges)
    )
    if (requested_profile.position or profile.position) and missing_position_ranges:
        return AutoRepairResult(
            "unrepairable", parent_hash, profile, None, None, 0, 0, 0,
            "POSITION_RANGES_REQUIRED",
            ({
                "status": "range_missing",
                "joints": missing_position_ranges,
            },),
        )

    collision_passed = bool(
        fresh_collision is not None
        and (
            str(fresh_collision.get("result", "")).lower() == "pass"
            or bool(fresh_collision.get("checked"))
            and bool(fresh_collision.get("passed"))
        )
    )
    if fresh_kinematics.passed and (not require_collision or collision_passed):
        return AutoRepairResult(
            "not_needed", parent_hash, profile, None, None, 0, 0, 0,
            "ALREADY_PASSES",
        )

    involved_collision_limbs = (
        _collision_limbs(report) | _collision_limbs(fresh_collision)
    )
    if profile.collision and not involved_collision_limbs:
        return AutoRepairResult(
            "unrepairable", parent_hash, profile, None, None, 0, 0, 0,
            "COLLISION_LIMB_UNRESOLVED",
            ({
                "status": "collision_limb_unresolved",
                "collision_report": deepcopy(dict(fresh_collision or {})),
            },),
        )

    trials: list[dict[str, Any]] = []
    feasible: list[tuple[FidelityScore, RepairRecipe]] = []
    try:
        recipes = _adaptive_recipes(
            parent, profile, policy, limits, neutral_pose, involved_collision_limbs
        )
    except AtomRepairError as exc:
        return AutoRepairResult(
            "unrepairable", parent_hash, profile, None, None, 0, 0, 0,
            "NO_POSITION_FEASIBLE_SCALE",
            ({"status": "analytic_position_failed", "reason": str(exc)[:240]},),
        )
    for recipe in recipes:
        _cancel(cancel_check)
        try:
            result = repair_candidate(
                parent, recipe=recipe, limits=limits,
                collision_validator=None, neutral_pose=neutral_pose,
            )
            fidelity = score_fidelity(parent, result.atom, recipe)
            if not _passes(result, False):
                raise AtomRepairError("kinematic constraints remain violated")
            if (
                fidelity.trajectory_nrmse > policy.max_trajectory_nrmse
                or fidelity.apex_retention < policy.min_apex_retention
            ):
                trials.append({"status": "fidelity_rejected", "recipe": recipe.as_dict()})
                continue
            feasible.append((fidelity, recipe))
            trials.append({
                "status": "kinematic_feasible",
                "recipe": recipe.as_dict(),
                "fidelity": fidelity.as_dict(),
            })
        except AtomRepairError as exc:
            trials.append({
                "status": "kinematic_failed",
                "recipe": recipe.as_dict(),
                "reason": str(exc)[:240],
            })
    feasible.sort(key=_fidelity_order_key)
    if not feasible:
        return AutoRepairResult(
            "unrepairable", parent_hash, profile, None, None, len(recipes), 0, 0,
            "NO_KINEMATIC_RECIPE", tuple(trials),
        )
    collision_candidates = (
        _collision_trial_order(
            feasible, policy.max_collision_trials, involved_collision_limbs
        )
        if profile.collision
        else feasible[: policy.max_collision_trials]
    )
    checked = 0
    for fidelity, recipe in collision_candidates:
        _cancel(cancel_check)
        checked += 1
        try:
            result = repair_candidate(
                parent, recipe=recipe, limits=limits,
                collision_validator=collision_validator, neutral_pose=neutral_pose,
            )
            if _passes(result, require_collision):
                return AutoRepairResult(
                    "repaired", parent_hash, profile, result, fidelity, len(recipes),
                    len(feasible), checked, None, tuple(trials),
                )
        except AtomRepairError as exc:
            trials.append({
                "status": "collision_failed",
                "recipe": recipe.as_dict(),
                "reason": str(exc)[:240],
            })
    return AutoRepairResult(
        "unrepairable", parent_hash, profile, None, None, len(recipes), len(feasible),
        checked, "NO_COLLISION_SAFE_RECIPE", tuple(trials),
    )


__all__ = [
    "AUTO_REPAIR_NAME", "AUTO_REPAIR_VERSION", "POLICY_NAME",
    "AutoRepairCanceled", "AutoRepairPolicy", "AutoRepairResult",
    "FailureProfile", "FidelityScore", "analyze_failure", "generate_recipes",
    "maximum_position_limb_scales", "score_fidelity", "search_best_repair",
]
