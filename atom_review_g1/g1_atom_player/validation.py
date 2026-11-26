from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .catalog import AtomRecord, EXPECTED_JOINTS


class LimitsError(ValueError):
    pass


def load_limits(path: str | Path) -> dict[str, Any]:
    limits_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(limits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LimitsError(f"cannot read limits {limits_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "g1_atom_limits_v1":
        raise LimitsError("limits file has an unsupported schema")
    joints = payload.get("joints")
    if not isinstance(joints, dict) or set(joints) != set(EXPECTED_JOINTS):
        raise LimitsError("limits.joints must contain exactly the canonical 14 joints")
    for joint in EXPECTED_JOINTS:
        item = joints[joint]
        if not isinstance(item, dict):
            raise LimitsError(f"limits.joints.{joint} must be an object")
        try:
            minimum = float(item["min_rad"])
            maximum = float(item["max_rad"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise LimitsError(f"limits.joints.{joint} has invalid min/max") from exc
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise LimitsError(f"limits.joints.{joint} has invalid min/max")
    if "max_atom_segment_velocity_rad_s" in payload:
        raise LimitsError(
            "max_atom_segment_velocity_rad_s is a removed legacy Atom-body gate"
        )
    margin = payload.get("position_margin_rad", 0.0)
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        raise LimitsError("position_margin_rad must be numeric")
    if not math.isfinite(float(margin)) or float(margin) < 0:
        raise LimitsError("position_margin_rad must be finite and non-negative")
    for joint in EXPECTED_JOINTS:
        minimum = float(joints[joint]["min_rad"])
        maximum = float(joints[joint]["max_rad"])
        if minimum + float(margin) > maximum - float(margin):
            raise LimitsError(f"limits.joints.{joint} has no range after margin")
    return payload


def validate_atom(atom: AtomRecord, limits: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    joint_limits = limits["joints"]
    margin = float(limits.get("position_margin_rad", 0.0))
    for frame_index, frame in enumerate(atom.motion["frames"]):
        for joint in EXPECTED_JOINTS:
            q = float(frame["q"][joint])
            minimum = float(joint_limits[joint]["min_rad"]) + margin
            maximum = float(joint_limits[joint]["max_rad"]) - margin
            if q < minimum or q > maximum:
                errors.append(
                    {
                        "code": "joint_position_limit",
                        "joint": joint,
                        "frame": frame_index,
                        "t": float(frame["t"]),
                        "q": q,
                        "allowed": [minimum, maximum],
                    }
                )
                if len(errors) >= 50:
                    break
        if len(errors) >= 50:
            break

    return {
        "ok": not errors,
        "entry_id": atom.entry_id,
        "atom_id": atom.atom_id,
        "motion_sha256": atom.motion_sha256,
        "duration": atom.duration,
        "frame_count": atom.frame_count,
        "max_abs_q": atom.max_abs_q,
        "max_segment_velocity": atom.max_segment_velocity,
        "bridge_from_neutral": atom.bridge_from_neutral,
        "bridge_to_neutral": atom.bridge_to_neutral,
        "limits_source": limits.get("source", ""),
        "atom_segment_velocity_gate_enabled": False,
        "errors": errors,
        "warnings": warnings,
        "unchecked": [
            "atom_segment_velocity_limit",
            "self_collision",
            "torque",
            "tracking_error",
            "acceleration",
            "jerk",
        ],
    }
