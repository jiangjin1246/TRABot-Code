"""Automatic motion-constraint reports for G1 gesture atoms.

The numerical validator intentionally depends only on NumPy.  MuJoCo is an
optional, lazily imported second stage used to sweep a trajectory through a
fixed-root model and report contacts that are new (or materially deeper) than
the neutral pose.  Passing the numerical stage alone must therefore not be
interpreted as MuJoCo backend certification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .joints import (
    G1_29_JOINTS,
    G1_ARM_JOINTS,
    G1_LEG_JOINTS,
    G1_WAIST_JOINTS,
    canonical_joint_name,
    joint_aliases,
    normalize_joint_mapping,
    normalize_joint_names,
)


VALIDATOR_VERSION = "atom-motion-constraints-v2"
COLLISION_VALIDATOR_VERSION = "mujoco-fixed-root-collision-v2"


def _profile(
    leg: float,
    waist: float,
    shoulder: float,
    elbow: float,
    wrist: float,
) -> dict[str, float]:
    values = {joint: leg for joint in G1_LEG_JOINTS}
    values.update({joint: waist for joint in G1_WAIST_JOINTS})
    for joint in G1_ARM_JOINTS:
        if "shoulder" in joint:
            values[joint] = shoulder
        elif "elbow" in joint:
            values[joint] = elbow
        else:
            values[joint] = wrist
    return values


def _normalize_positive_limits(values: Mapping[str, float], label: str) -> dict[str, float]:
    normalized = normalize_joint_mapping(values)
    result: dict[str, float] = {}
    for joint, raw in normalized.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}[{joint!r}] must be finite and positive")
        result[joint] = value
    return result


def _normalize_ranges(
    ranges: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, float]]:
    normalized = normalize_joint_mapping(ranges)
    result: dict[str, tuple[float, float]] = {}
    for joint, raw in normalized.items():
        if len(raw) != 2:
            raise ValueError(f"joint range for {joint!r} must contain [low, high]")
        low, high = float(raw[0]), float(raw[1])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"joint range for {joint!r} must be finite with low < high")
        result[joint] = (low, high)
    return result


@dataclass(frozen=True)
class MotionLimits:
    """Conservative defaults plus explicit per-joint overrides.

    These thresholds are pipeline gates, not manufacturer-rated hardware
    limits.  They are deliberately conservative for conversational upper-body
    motion and should be versioned with each experiment configuration.
    """

    velocity_rad_s: float = 0.8
    acceleration_rad_s2: float = 4.0
    jerk_rad_s3: float = 30.0
    velocity_by_joint: Mapping[str, float] = field(
        default_factory=lambda: _profile(0.5, 0.5, 0.8, 1.0, 1.2)
    )
    acceleration_by_joint: Mapping[str, float] = field(
        default_factory=lambda: _profile(2.5, 2.5, 4.0, 5.0, 6.0)
    )
    jerk_by_joint: Mapping[str, float] = field(
        default_factory=lambda: _profile(20.0, 20.0, 30.0, 40.0, 50.0)
    )
    joint_ranges: Mapping[str, Sequence[float]] = field(default_factory=dict)
    position_margin_rad: float = 0.05
    warning_fraction: float = 0.85
    time_scale_safety_factor: float = 1.02
    max_time_scale: float = 3.0

    def __post_init__(self) -> None:
        for label, value in (
            ("velocity_rad_s", self.velocity_rad_s),
            ("acceleration_rad_s2", self.acceleration_rad_s2),
            ("jerk_rad_s3", self.jerk_rad_s3),
            ("max_time_scale", self.max_time_scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if not 0.0 < float(self.warning_fraction) < 1.0:
            raise ValueError("warning_fraction must be between zero and one")
        if not math.isfinite(float(self.position_margin_rad)) or self.position_margin_rad < 0.0:
            raise ValueError("position_margin_rad must be finite and non-negative")
        if (
            not math.isfinite(float(self.time_scale_safety_factor))
            or self.time_scale_safety_factor < 1.0
        ):
            raise ValueError("time_scale_safety_factor must be finite and at least one")
        object.__setattr__(
            self,
            "velocity_by_joint",
            _normalize_positive_limits(self.velocity_by_joint, "velocity_by_joint"),
        )
        object.__setattr__(
            self,
            "acceleration_by_joint",
            _normalize_positive_limits(self.acceleration_by_joint, "acceleration_by_joint"),
        )
        object.__setattr__(
            self,
            "jerk_by_joint",
            _normalize_positive_limits(self.jerk_by_joint, "jerk_by_joint"),
        )
        object.__setattr__(self, "joint_ranges", _normalize_ranges(self.joint_ranges))

    def threshold(self, metric: str, joint: str) -> float:
        canonical = canonical_joint_name(joint)
        if metric == "velocity":
            return float(self.velocity_by_joint.get(canonical, self.velocity_rad_s))
        if metric == "acceleration":
            return float(
                self.acceleration_by_joint.get(canonical, self.acceleration_rad_s2)
            )
        if metric == "jerk":
            return float(self.jerk_by_joint.get(canonical, self.jerk_rad_s3))
        raise KeyError(f"unsupported metric: {metric!r}")


CONSERVATIVE_G1_LIMITS = MotionLimits()


@dataclass(frozen=True)
class KinematicAnalysis:
    """Numerical arrays and their JSON-serializable constraint report."""

    times: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    jerk: np.ndarray
    joints: tuple[str, ...]
    required_time_scale: float
    recommended_time_scale: float
    report: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.report["result"] == "pass"


def _coerce_trajectory(
    times: Sequence[float] | np.ndarray,
    q: Sequence[Sequence[float]] | np.ndarray,
    joints: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    canonical_joints = normalize_joint_names(joints)
    time_values = np.asarray(times, dtype=np.float64)
    positions = np.asarray(q, dtype=np.float64)
    if time_values.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    if positions.ndim != 2:
        raise ValueError("q must have shape [samples, joints]")
    if len(time_values) != positions.shape[0]:
        raise ValueError("times and q must contain the same number of samples")
    if positions.shape[1] != len(canonical_joints):
        raise ValueError("q must contain one column per joint")
    if len(time_values) < 2:
        raise ValueError("a trajectory needs at least two samples")
    if not np.all(np.isfinite(time_values)) or not np.all(np.isfinite(positions)):
        raise ValueError("times and q must contain only finite values")
    if time_values[0] < 0.0:
        raise ValueError("trajectory time cannot start before zero")
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return time_values, positions, canonical_joints


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    edge_order = 2 if len(times) >= 3 else 1
    return np.asarray(
        np.gradient(values, times, axis=0, edge_order=edge_order),
        dtype=np.float64,
    )


def _violation(
    code: str,
    joint: str,
    metric: str,
    value: float,
    limit: float,
    time_s: float,
    **details: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "code": code,
        "severity": "error",
        "joint": joint,
        "metric": metric,
        "observed": float(value),
        "limit": float(limit),
        "time_s": float(time_s),
    }
    item.update(details)
    return item


def _warning(
    code: str,
    joint: str,
    metric: str,
    value: float,
    limit: float,
    time_s: float,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "warning",
        "joint": joint,
        "metric": metric,
        "observed": float(value),
        "limit": float(limit),
        "time_s": float(time_s),
    }


def analyze_kinematics(
    times: Sequence[float] | np.ndarray,
    q: Sequence[Sequence[float]] | np.ndarray,
    joints: Sequence[str],
    *,
    limits: MotionLimits = CONSERVATIVE_G1_LIMITS,
    joint_ranges: Mapping[str, Sequence[float]] | None = None,
) -> KinematicAnalysis:
    """Compute derivatives, constraint violations and a safe time-scale hint.

    Velocity scales with ``1/s``, acceleration with ``1/s**2`` and jerk with
    ``1/s**3`` when time is stretched by ``s``.  The returned factor is the
    maximum of the corresponding inverse-power ratios.  Position-limit errors
    cannot be repaired through temporal scaling and remain hard failures.
    """

    time_values, positions, canonical_joints = _coerce_trajectory(times, q, joints)
    velocities = _derivative(positions, time_values)
    accelerations = _derivative(velocities, time_values)
    jerks = _derivative(accelerations, time_values)

    ranges = dict(limits.joint_ranges)
    if joint_ranges is not None:
        ranges.update(_normalize_ranges(joint_ranges))

    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    per_joint: dict[str, dict[str, Any]] = {}
    scale_candidates: list[dict[str, Any]] = []

    metric_arrays = {
        "velocity": velocities,
        "acceleration": accelerations,
        "jerk": jerks,
    }
    metric_codes = {
        "velocity": "JOINT_VELOCITY_LIMIT",
        "acceleration": "JOINT_ACCELERATION_LIMIT",
        "jerk": "JOINT_JERK_LIMIT",
    }
    metric_units = {
        "velocity": "rad/s",
        "acceleration": "rad/s^2",
        "jerk": "rad/s^3",
    }
    metric_powers = {"velocity": 1.0, "acceleration": 2.0, "jerk": 3.0}

    for column, joint in enumerate(canonical_joints):
        joint_observed: dict[str, Any] = {
            "position_min_rad": float(np.min(positions[:, column])),
            "position_max_rad": float(np.max(positions[:, column])),
        }
        if joint in ranges:
            raw_low, raw_high = ranges[joint]
            low = raw_low + limits.position_margin_rad
            high = raw_high - limits.position_margin_rad
            if low >= high:
                raise ValueError(
                    f"position margin leaves no valid range for {joint!r}: "
                    f"[{raw_low}, {raw_high}]"
                )
            joint_observed["allowed_position_rad"] = [float(low), float(high)]
            below_index = int(np.argmin(positions[:, column]))
            above_index = int(np.argmax(positions[:, column]))
            observed_low = float(positions[below_index, column])
            observed_high = float(positions[above_index, column])
            if observed_low < low:
                violations.append(
                    _violation(
                        "JOINT_POSITION_LIMIT",
                        joint,
                        "position",
                        observed_low,
                        low,
                        time_values[below_index],
                        side="low",
                        model_limit_rad=float(raw_low),
                        required_margin_rad=float(limits.position_margin_rad),
                    )
                )
            if observed_high > high:
                violations.append(
                    _violation(
                        "JOINT_POSITION_LIMIT",
                        joint,
                        "position",
                        observed_high,
                        high,
                        time_values[above_index],
                        side="high",
                        model_limit_rad=float(raw_high),
                        required_margin_rad=float(limits.position_margin_rad),
                    )
                )

        for metric, values in metric_arrays.items():
            absolute = np.abs(values[:, column])
            peak_index = int(np.argmax(absolute))
            peak = float(absolute[peak_index])
            threshold = limits.threshold(metric, joint)
            ratio = peak / threshold
            scale = ratio ** (1.0 / metric_powers[metric])
            joint_observed[f"max_abs_{metric}_{metric_units[metric]}"] = peak
            joint_observed[f"{metric}_limit_{metric_units[metric]}"] = threshold
            scale_candidates.append(
                {
                    "joint": joint,
                    "metric": metric,
                    "ratio_to_limit": float(ratio),
                    "required_time_scale": float(scale),
                }
            )
            if ratio > 1.0 + 1e-9:
                violations.append(
                    _violation(
                        metric_codes[metric],
                        joint,
                        metric,
                        peak,
                        threshold,
                        time_values[peak_index],
                        unit=metric_units[metric],
                    )
                )
            elif ratio >= limits.warning_fraction:
                warnings.append(
                    _warning(
                        f"NEAR_{metric_codes[metric]}",
                        joint,
                        metric,
                        peak,
                        threshold,
                        time_values[peak_index],
                    )
                )
        per_joint[joint] = joint_observed

    scale_candidates.sort(key=lambda item: item["required_time_scale"], reverse=True)
    raw_required_scale = max(1.0, scale_candidates[0]["required_time_scale"])
    recommended_scale = (
        raw_required_scale * limits.time_scale_safety_factor
        if raw_required_scale > 1.0 + 1e-9
        else 1.0
    )
    position_failures = sum(
        item["code"] == "JOINT_POSITION_LIMIT" for item in violations
    )
    global_observed = {
        "max_abs_velocity_rad/s": float(np.max(np.abs(velocities))),
        "max_abs_acceleration_rad/s^2": float(np.max(np.abs(accelerations))),
        "max_abs_jerk_rad/s^3": float(np.max(np.abs(jerks))),
    }
    effective_thresholds = {
        joint: {
            "velocity_rad_s": limits.threshold("velocity", joint),
            "acceleration_rad_s2": limits.threshold("acceleration", joint),
            "jerk_rad_s3": limits.threshold("jerk", joint),
            **(
                {"model_position_range_rad": list(ranges[joint])}
                if joint in ranges
                else {}
            ),
        }
        for joint in canonical_joints
    }
    report: dict[str, Any] = {
        "validator": VALIDATOR_VERSION,
        "result": "pass" if not violations else "fail",
        "sample_count": int(len(time_values)),
        "duration_s": float(time_values[-1] - time_values[0]),
        "joints": list(canonical_joints),
        "thresholds": {
            "global_fallback": {
                "velocity_rad_s": float(limits.velocity_rad_s),
                "acceleration_rad_s2": float(limits.acceleration_rad_s2),
                "jerk_rad_s3": float(limits.jerk_rad_s3),
            },
            "position_margin_rad": float(limits.position_margin_rad),
            "warning_fraction": float(limits.warning_fraction),
            "per_joint_effective": effective_thresholds,
        },
        "observed": {"global": global_observed, "per_joint": per_joint},
        "time_scaling": {
            "required_factor": float(raw_required_scale),
            "recommended_factor": float(recommended_scale),
            "configured_max_factor": float(limits.max_time_scale),
            "safety_factor": float(limits.time_scale_safety_factor),
            "within_configured_max": bool(recommended_scale <= limits.max_time_scale + 1e-9),
            "position_failures_are_fixable": position_failures == 0,
            "limiting_constraints": scale_candidates[:5],
        },
        "violation_count": len(violations),
        "warning_count": len(warnings),
        "violations": violations,
        "warnings": warnings,
    }
    return KinematicAnalysis(
        times=time_values,
        q=positions,
        dq=velocities,
        ddq=accelerations,
        jerk=jerks,
        joints=canonical_joints,
        required_time_scale=float(raw_required_scale),
        recommended_time_scale=float(recommended_scale),
        report=report,
    )


def required_time_scale(
    times: Sequence[float] | np.ndarray,
    q: Sequence[Sequence[float]] | np.ndarray,
    joints: Sequence[str],
    *,
    limits: MotionLimits = CONSERVATIVE_G1_LIMITS,
) -> float:
    """Return the safety-factored temporal stretch recommended for dynamics."""

    return analyze_kinematics(times, q, joints, limits=limits).recommended_time_scale


def time_scale_trajectory(
    times: Sequence[float] | np.ndarray,
    factor: float,
) -> np.ndarray:
    """Stretch timestamps around the first sample without modifying positions."""

    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("times must contain at least two finite samples")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("times must be strictly increasing")
    factor_value = float(factor)
    if not math.isfinite(factor_value) or factor_value <= 0.0:
        raise ValueError("time-scale factor must be finite and positive")
    return values[0] + (values - values[0]) * factor_value


class MujocoUnavailableError(RuntimeError):
    """Raised when optional MuJoCo collision validation cannot be loaded."""


class MujocoCollisionValidator:
    """Sweep poses through a MuJoCo model with every free root held at qpos0.

    A contact is reported when it is absent at neutral, or when its distance is
    more than ``baseline_tolerance_m`` deeper than at neutral.  Expected
    foot/ankle contacts with the world are allowed by default.  This is a pose
    sweep, not a dynamic balance or torque-feasibility certificate.
    """

    def __init__(
        self,
        scene_path: str | Path,
        *,
        sample_stride: int = 1,
        contact_distance_threshold_m: float = 0.0,
        baseline_tolerance_m: float = 0.002,
        backend_id: str = "mujoco-g1-29dof",
        allowed_environment_robot_tokens: Sequence[str] = (
            "foot",
            "ankle",
        ),
        module_loader: Callable[[], Any] | None = None,
    ) -> None:
        if int(sample_stride) < 1:
            raise ValueError("sample_stride must be at least one")
        if not math.isfinite(float(contact_distance_threshold_m)):
            raise ValueError("contact_distance_threshold_m must be finite")
        if not math.isfinite(float(baseline_tolerance_m)) or baseline_tolerance_m < 0.0:
            raise ValueError("baseline_tolerance_m must be finite and non-negative")
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("backend_id must be a non-empty string")
        self.scene_path = Path(scene_path).expanduser()
        self.sample_stride = int(sample_stride)
        self.contact_distance_threshold_m = float(contact_distance_threshold_m)
        self.baseline_tolerance_m = float(baseline_tolerance_m)
        self.backend_id = backend_id.strip()
        self.allowed_environment_robot_tokens = tuple(
            str(token).lower() for token in allowed_environment_robot_tokens
        )
        self._module_loader = module_loader or (lambda: importlib.import_module("mujoco"))
        self._mujoco: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._mujoco, self._model
        model_path = self.scene_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene does not exist: {model_path}")
        try:
            mujoco = self._module_loader()
        except (ImportError, ModuleNotFoundError) as exc:
            raise MujocoUnavailableError(
                "MuJoCo collision validation requires the `mujoco` Python package"
            ) from exc
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        return mujoco, self._model

    def _joint_record(self, canonical: str) -> tuple[int, int]:
        mujoco, model = self._load()
        joint_id = -1
        for alias in joint_aliases(canonical):
            joint_id = int(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, alias)
            )
            if joint_id >= 0:
                break
        if joint_id < 0:
            raise ValueError(f"MuJoCo scene does not contain joint {canonical!r}")
        joint_type = int(model.jnt_type[joint_id])
        supported = {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }
        if joint_type not in supported:
            raise ValueError(f"controlled joint {canonical!r} is not scalar")
        return joint_id, int(model.jnt_qposadr[joint_id])

    def joint_ranges(
        self,
        joints: Sequence[str] = G1_29_JOINTS,
    ) -> dict[str, tuple[float, float]]:
        """Read limited scalar-joint ranges from the configured model."""

        _, model = self._load()
        result: dict[str, tuple[float, float]] = {}
        for canonical in normalize_joint_names(joints):
            joint_id, _ = self._joint_record(canonical)
            if int(model.jnt_limited[joint_id]):
                low, high = model.jnt_range[joint_id]
                result[canonical] = (float(low), float(high))
        return result

    @staticmethod
    def _name(mujoco: Any, model: Any, object_type: Any, object_id: int, prefix: str) -> str:
        name = mujoco.mj_id2name(model, object_type, int(object_id))
        return str(name) if name else f"{prefix}:{int(object_id)}"

    def _contact_snapshot(self, data: Any) -> dict[tuple[int, int], dict[str, Any]]:
        mujoco, model = self._load()
        contacts: dict[tuple[int, int], dict[str, Any]] = {}
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            key = tuple(sorted((geom1, geom2)))
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            kind = "environment" if body1 == 0 or body2 == 0 else "self"
            if body1 == body2:
                continue
            geom_names = (
                self._name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom1, "geom"),
                self._name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom2, "geom"),
            )
            body_names = (
                self._name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body1, "body"),
                self._name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body2, "body"),
            )
            robot_side_names = [
                geom_names[index].lower() + " " + body_names[index].lower()
                for index, body in enumerate((body1, body2))
                if body != 0
            ]
            allowed = kind == "environment" and any(
                token in name
                for name in robot_side_names
                for token in self.allowed_environment_robot_tokens
            )
            distance = float(contact.dist)
            previous = contacts.get(key)
            if previous is None or distance < previous["distance_m"]:
                contacts[key] = {
                    "key": key,
                    "kind": kind,
                    "geom_names": geom_names,
                    "body_names": body_names,
                    "distance_m": distance,
                    "allowed": allowed,
                }
        return contacts

    def validate(
        self,
        times: Sequence[float] | np.ndarray,
        q: Sequence[Sequence[float]] | np.ndarray,
        joints: Sequence[str],
        *,
        neutral: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable new-contact report for a fixed-root sweep."""

        time_values, positions, canonical_joints = _coerce_trajectory(times, q, joints)
        mujoco, model = self._load()
        records = [self._joint_record(joint) for joint in canonical_joints]
        data = mujoco.MjData(model)
        root_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()

        if neutral is None:
            neutral_values = positions[0]
        elif isinstance(neutral, Mapping):
            neutral_map = normalize_joint_mapping(neutral)
            neutral_values = np.asarray(
                [neutral_map.get(joint, positions[0, index]) for index, joint in enumerate(canonical_joints)],
                dtype=np.float64,
            )
        else:
            neutral_values = np.asarray(neutral, dtype=np.float64)
            if neutral_values.shape != (len(canonical_joints),):
                raise ValueError("neutral sequence must contain one value per joint")
        if not np.all(np.isfinite(neutral_values)):
            raise ValueError("neutral must contain only finite values")

        def set_pose(row: np.ndarray) -> None:
            data.qpos[:] = root_qpos
            data.qvel[:] = 0.0
            for (_, qpos_address), value in zip(records, row):
                data.qpos[qpos_address] = float(value)
            mujoco.mj_forward(model, data)

        set_pose(neutral_values)
        baseline = self._contact_snapshot(data)
        baseline_distances = {
            key: item["distance_m"]
            for key, item in baseline.items()
            if not item["allowed"]
            and item["distance_m"] <= self.contact_distance_threshold_m
        }

        indices = list(range(0, len(time_values), self.sample_stride))
        if indices[-1] != len(time_values) - 1:
            indices.append(len(time_values) - 1)
        pair_records: dict[tuple[int, int], dict[str, Any]] = {}
        max_self_per_sample = 0
        max_environment_per_sample = 0
        minimum_distance: float | None = None
        for sample_index in indices:
            set_pose(positions[sample_index])
            snapshot = self._contact_snapshot(data)
            self_count = 0
            environment_count = 0
            for key, contact in snapshot.items():
                distance = contact["distance_m"]
                if contact["allowed"] or distance > self.contact_distance_threshold_m:
                    continue
                baseline_distance = baseline_distances.get(key)
                if (
                    baseline_distance is not None
                    and distance >= baseline_distance - self.baseline_tolerance_m
                ):
                    continue
                if contact["kind"] == "self":
                    self_count += 1
                else:
                    environment_count += 1
                minimum_distance = (
                    distance if minimum_distance is None else min(minimum_distance, distance)
                )
                record = pair_records.get(key)
                if record is None:
                    record = {
                        "kind": contact["kind"],
                        "geom_names": list(contact["geom_names"]),
                        "body_names": list(contact["body_names"]),
                        "first_sample": int(sample_index),
                        "first_time_s": float(time_values[sample_index]),
                        "min_distance_m": float(distance),
                        "occurrences": 0,
                        "baseline_distance_m": (
                            float(baseline_distance) if baseline_distance is not None else None
                        ),
                    }
                    pair_records[key] = record
                record["occurrences"] += 1
                record["min_distance_m"] = min(record["min_distance_m"], float(distance))
            max_self_per_sample = max(max_self_per_sample, self_count)
            max_environment_per_sample = max(max_environment_per_sample, environment_count)

        contact_pairs = sorted(
            pair_records.values(),
            key=lambda item: (item["first_time_s"], item["kind"], item["geom_names"]),
        )
        violations = [
            {
                "code": "SELF_COLLISION" if item["kind"] == "self" else "ENVIRONMENT_COLLISION",
                "severity": "error",
                **item,
            }
            for item in contact_pairs
        ]
        free_root_count = sum(
            int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
            for joint_id in range(int(model.njnt))
        )
        resolved_path = self.scene_path.resolve()
        model_sha256 = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        return {
            "validator": COLLISION_VALIDATOR_VERSION,
            "backend": self.backend_id,
            "result": "pass" if not violations else "fail",
            "model": {
                "path": str(resolved_path),
                "root_xml_sha256": model_sha256,
                "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
            },
            "root_policy": "all_free_joints_pinned_to_model_qpos0",
            "allowed_environment_robot_tokens": list(
                self.allowed_environment_robot_tokens
            ),
            "free_root_count": int(free_root_count),
            "trajectory_sample_count": int(len(time_values)),
            "evaluated_sample_count": int(len(indices)),
            "sample_stride": int(self.sample_stride),
            "contact_distance_threshold_m": float(self.contact_distance_threshold_m),
            "baseline_tolerance_m": float(self.baseline_tolerance_m),
            "neutral_baseline_contact_count": int(len(baseline_distances)),
            "new_or_deeper_contact_pair_count": int(len(contact_pairs)),
            "max_new_self_contacts_per_sample": int(max_self_per_sample),
            "max_new_environment_contacts_per_sample": int(max_environment_per_sample),
            "minimum_contact_distance_m": (
                float(minimum_distance) if minimum_distance is not None else None
            ),
            "contact_pairs": contact_pairs,
            "violation_count": len(violations),
            "violations": violations,
            "warnings": [
                {
                    "code": "POSE_SWEEP_ONLY",
                    "severity": "warning",
                    "message": (
                        "Fixed-root contact sweep does not certify balance, actuator torque, "
                        "tracking error, or real-robot safety."
                    ),
                }
            ],
        }


def build_motion_constraint_report(
    times: Sequence[float] | np.ndarray,
    q: Sequence[Sequence[float]] | np.ndarray,
    joints: Sequence[str],
    *,
    limits: MotionLimits = CONSERVATIVE_G1_LIMITS,
    joint_ranges: Mapping[str, Sequence[float]] | None = None,
    collision_validator: MujocoCollisionValidator | None = None,
    neutral: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Combine kinematics and optional collision checking into one gate report."""

    ranges = joint_ranges
    collision: dict[str, Any] | None = None
    collision_error: dict[str, Any] | None = None
    if collision_validator is not None:
        try:
            if ranges is None:
                ranges = collision_validator.joint_ranges(joints)
            collision = collision_validator.validate(times, q, joints, neutral=neutral)
        except (FileNotFoundError, MujocoUnavailableError) as exc:
            collision_error = {
                "code": "MUJOCO_COLLISION_UNAVAILABLE",
                "severity": "warning",
                "message": str(exc),
            }

    kinematics = analyze_kinematics(
        times,
        q,
        joints,
        limits=limits,
        joint_ranges=ranges,
    )
    collision_passed = collision is not None and collision["result"] == "pass"
    all_passed = kinematics.passed and (collision is None or collision_passed)
    warnings: list[dict[str, Any]] = []
    if collision_validator is None:
        warnings.append(
            {
                "code": "MUJOCO_COLLISION_NOT_RUN",
                "severity": "warning",
                "message": "Kinematics passed without a MuJoCo contact sweep.",
            }
        )
    if collision_error is not None:
        warnings.append(collision_error)
    certified_backends = (
        [str(collision["backend"])]
        if kinematics.passed and collision_passed and collision is not None
        else []
    )
    return {
        "validator": VALIDATOR_VERSION,
        "result": "pass" if all_passed else "fail",
        "coverage": {
            "kinematics": True,
            "joint_ranges": bool(ranges),
            "mujoco_collision": collision is not None,
            "dynamic_balance": False,
            "actuator_torque": False,
            "real_robot": False,
        },
        "eligible_for_backend_certification": bool(certified_backends),
        "certified_backends": certified_backends,
        "kinematics": kinematics.report,
        "collision": collision,
        "warnings": warnings,
    }


def _schema_diagnostic(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project a detailed diagnostic onto Atom Schema v1's strict shape."""

    code = str(item.get("code", "UNKNOWN_CONSTRAINT"))
    joint = item.get("joint")
    metric = item.get("metric")
    if "message" in item:
        message = str(item["message"])
    elif joint is not None and metric is not None:
        message = f"{joint} {metric} constraint was reached or exceeded."
    else:
        message = code.replace("_", " ").lower().capitalize() + "."
    result: dict[str, Any] = {"code": code, "message": message}
    if joint is not None:
        result["joint"] = str(joint)
    for key in ("time_s", "observed", "limit"):
        if key in item and item[key] is not None:
            value = item[key]
            result[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
    return result


def build_atom_safety(
    times: Sequence[float] | np.ndarray,
    q: Sequence[Sequence[float]] | np.ndarray,
    joints: Sequence[str],
    *,
    limits: MotionLimits = CONSERVATIVE_G1_LIMITS,
    joint_ranges: Mapping[str, Sequence[float]] | None = None,
    collision_validator: MujocoCollisionValidator | None = None,
    neutral: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    max_tracking_error_rad: float = 0.12,
    min_collision_distance_m: float = 0.0,
    max_torque_saturation_fraction: float = 0.10,
) -> dict[str, Any]:
    """Build the strict ``safety`` object embedded by Atom Schema v1.

    Rich contact pairs, per-joint effective thresholds, and time-scale
    contributors remain available from :func:`build_motion_constraint_report`
    for a sidecar audit report.  This compact projection deliberately records
    when collision, tracking-error, or torque checks were not performed.
    """

    tracking_limit = float(max_tracking_error_rad)
    distance_limit = float(min_collision_distance_m)
    torque_limit = float(max_torque_saturation_fraction)
    if not math.isfinite(tracking_limit) or tracking_limit < 0.0:
        raise ValueError("max_tracking_error_rad must be finite and non-negative")
    if not math.isfinite(distance_limit) or distance_limit < 0.0:
        raise ValueError("min_collision_distance_m must be finite and non-negative")
    if not math.isfinite(torque_limit) or not 0.0 <= torque_limit <= 1.0:
        raise ValueError("max_torque_saturation_fraction must be between zero and one")

    detailed = build_motion_constraint_report(
        times,
        q,
        joints,
        limits=limits,
        joint_ranges=joint_ranges,
        collision_validator=collision_validator,
        neutral=neutral,
    )
    kinematics = detailed["kinematics"]
    canonical_joints = tuple(kinematics["joints"])
    effective = kinematics["thresholds"]["per_joint_effective"]
    velocity_ceiling = max(item["velocity_rad_s"] for item in effective.values())
    acceleration_ceiling = max(item["acceleration_rad_s2"] for item in effective.values())
    jerk_ceiling = max(item["jerk_rad_s3"] for item in effective.values())

    per_joint: dict[str, dict[str, float]] = {}
    for joint in canonical_joints:
        observed = kinematics["observed"]["per_joint"][joint]
        per_joint[joint] = {
            "q_min": float(observed["position_min_rad"]),
            "q_max": float(observed["position_max_rad"]),
            "max_abs_dq": float(observed["max_abs_velocity_rad/s"]),
            "max_abs_ddq": float(observed["max_abs_acceleration_rad/s^2"]),
            "max_abs_jerk": float(observed["max_abs_jerk_rad/s^3"]),
        }

    collision = detailed["collision"]
    if collision is None:
        collision_summary = {
            "checked": False,
            "passed": False,
            "self_collision_count": 0,
            "environment_collision_count": 0,
            "min_distance_m": None,
        }
    else:
        collision_summary = {
            "checked": True,
            "passed": collision["result"] == "pass",
            "self_collision_count": sum(
                item["kind"] == "self" for item in collision["contact_pairs"]
            ),
            "environment_collision_count": sum(
                item["kind"] == "environment" for item in collision["contact_pairs"]
            ),
            "min_distance_m": collision["minimum_contact_distance_m"],
        }

    violations = [_schema_diagnostic(item) for item in kinematics["violations"]]
    if collision is not None:
        violations.extend(_schema_diagnostic(item) for item in collision["violations"])
    warnings = [_schema_diagnostic(item) for item in kinematics["warnings"]]
    warnings.extend(_schema_diagnostic(item) for item in detailed["warnings"])
    if collision is not None:
        warnings.extend(_schema_diagnostic(item) for item in collision["warnings"])

    scaling = kinematics["time_scaling"]
    return {
        "thresholds": {
            "joint_position_margin_rad": float(limits.position_margin_rad),
            "max_velocity_rad_s": float(velocity_ceiling),
            "max_acceleration_rad_s2": float(acceleration_ceiling),
            "max_jerk_rad_s3": float(jerk_ceiling),
            "max_tracking_error_rad": tracking_limit,
            "min_collision_distance_m": distance_limit,
            "max_torque_saturation_fraction": torque_limit,
        },
        "metrics": {
            "passed": detailed["result"] == "pass",
            "base_mode": "fixed",
            "sample_count": int(kinematics["sample_count"]),
            "duration_s": float(kinematics["duration_s"]),
            "required_time_scale": float(scaling["required_factor"]),
            "recommended_time_scale": float(scaling["recommended_factor"]),
            "within_max_time_scale": bool(scaling["within_configured_max"]),
            "kinematics_passed": kinematics["result"] == "pass",
            "position_limits_checked": bool(detailed["coverage"]["joint_ranges"]),
            "collision_checked": bool(collision_summary["checked"]),
            "collision_passed": bool(collision_summary["passed"]),
            "tracking_error_checked": False,
            "torque_checked": False,
            "threshold_profile": "g1-fixed-base-upper-body-v2",
        },
        "per_joint": per_joint,
        "violations": violations,
        "warnings": warnings,
        "collision": collision_summary,
        "validator_version": VALIDATOR_VERSION,
        "certified_backends": list(detailed["certified_backends"]),
    }
