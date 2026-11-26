"""Deterministic transition/recovery accounting for semantic Atom plans."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence


MINIMUM_JERK_FPS = 25
MINIMUM_JERK_MIN_TICKS = 2
MINIMUM_JERK_PEAK_FACTOR = 1.875


class SemanticMotionError(ValueError):
    """A catalog motion or measured pose cannot be safely costed."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SemanticMotionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise SemanticMotionError(f"{label} must be finite")
    return result


def _q_from_frame(motion: Any, frame: Any, label: str) -> dict[str, float]:
    joints = _field(motion, "joints")
    q = _field(frame, "q")
    if isinstance(q, Mapping):
        result = {str(joint): _finite(value, f"{label}.{joint}") for joint, value in q.items()}
    elif (
        isinstance(joints, Sequence)
        and not isinstance(joints, (str, bytes, bytearray))
        and isinstance(q, Sequence)
        and not isinstance(q, (str, bytes, bytearray))
        and len(joints) == len(q)
    ):
        result = {
            str(joint): _finite(value, f"{label}.{joint}")
            for joint, value in zip(joints, q, strict=True)
        }
    else:
        raise SemanticMotionError(f"{label} has no valid joint-position mapping")
    if not result:
        raise SemanticMotionError(f"{label} cannot be empty")
    return result


def motion_endpoints(value: Any) -> tuple[dict[str, float], dict[str, float]]:
    explicit_entry = _field(value, "entry_q")
    explicit_exit = _field(value, "exit_q")
    if isinstance(explicit_entry, Mapping) and isinstance(explicit_exit, Mapping):
        return (
            {str(key): _finite(item, f"entry_q.{key}") for key, item in explicit_entry.items()},
            {str(key): _finite(item, f"exit_q.{key}") for key, item in explicit_exit.items()},
        )
    motion = _field(value, "motion", value)
    frames = _field(motion, "frames")
    if (
        not isinstance(frames, Sequence)
        or isinstance(frames, (str, bytes, bytearray))
        or not frames
    ):
        raise SemanticMotionError("motion.frames must be a non-empty sequence")
    return (
        _q_from_frame(motion, frames[0], "motion first frame"),
        _q_from_frame(motion, frames[-1], "motion last frame"),
    )


def motion_duration_us(value: Any) -> int:
    explicit = _field(value, "duration_us")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return explicit
    motion = _field(value, "motion", value)
    frames = _field(motion, "frames")
    if not isinstance(frames, Sequence) or not frames:
        raise SemanticMotionError("motion has no frames")
    start = _finite(_field(frames[0], "t"), "motion start time")
    end = _finite(_field(frames[-1], "t"), "motion end time")
    duration = end - start
    if duration <= 0:
        raise SemanticMotionError("motion duration must be positive")
    return int(math.ceil(duration * 1_000_000 - 1e-9))


def _lookup_pose_value(pose: Mapping[str, Any], joint: str) -> float:
    aliases = (joint, f"{joint}_joint") if not joint.endswith("_joint") else (joint, joint[:-6])
    for alias in aliases:
        if alias in pose:
            return _finite(pose[alias], f"measured pose {alias}")
    raise SemanticMotionError(f"measured pose is missing controlled joint {joint!r}")


def minimum_jerk_bridge_us(
    from_q: Mapping[str, Any],
    to_q: Mapping[str, Any],
) -> int:
    if not isinstance(from_q, Mapping) or not isinstance(to_q, Mapping) or not to_q:
        raise SemanticMotionError("bridge endpoints must be non-empty mappings")
    maximum_delta = max(
        abs(_finite(target, f"target {joint}") - _lookup_pose_value(from_q, str(joint)))
        for joint, target in to_q.items()
    )
    required = MINIMUM_JERK_PEAK_FACTOR * maximum_delta
    if not math.isfinite(required):
        raise SemanticMotionError("minimum-jerk duration is not finite")
    ticks = max(
        MINIMUM_JERK_MIN_TICKS,
        int(math.ceil(required * MINIMUM_JERK_FPS - 1e-12)),
    )
    return ticks * (1_000_000 // MINIMUM_JERK_FPS)


def transition_cost_callback(initial_q: Mapping[str, Any]):
    frozen_initial = dict(initial_q)

    def cost(previous: Any | None, candidate: Any) -> int:
        candidate_entry, _ = motion_endpoints(candidate)
        if previous is None:
            origin = frozen_initial
        elif isinstance(previous, Mapping) and "q" in previous and len(previous) <= 8:
            raw = previous.get("q")
            if not isinstance(raw, Mapping):
                raise SemanticMotionError("opaque previous measured state has invalid q")
            origin = raw
        else:
            _, origin = motion_endpoints(previous)
        return minimum_jerk_bridge_us(origin, candidate_entry)

    return cost


def recovery_cost_callback(neutral_motion: Any):
    neutral_entry, _ = motion_endpoints(neutral_motion)
    neutral_duration = motion_duration_us(neutral_motion)

    def cost(previous: Any) -> int:
        _, origin = motion_endpoints(previous)
        return minimum_jerk_bridge_us(origin, neutral_entry) + neutral_duration

    return cost


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [thaw_json(item) for item in value]
    return deepcopy(value)


def inline_motion(value: Any) -> dict[str, Any]:
    motion = _field(value, "motion", value)
    result = thaw_json(motion)
    if not isinstance(result, dict):
        raise SemanticMotionError("inline motion must be an object")
    # Re-parse endpoints and duration before exposing executable bytes.
    motion_endpoints(result)
    motion_duration_us(result)
    return result


__all__ = [
    "SemanticMotionError",
    "inline_motion",
    "minimum_jerk_bridge_us",
    "motion_duration_us",
    "motion_endpoints",
    "recovery_cost_callback",
    "thaw_json",
    "transition_cost_callback",
]
