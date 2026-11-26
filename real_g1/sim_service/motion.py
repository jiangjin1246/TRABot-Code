"""Motion JSON parsing and deterministic interpolation.

Two formats are accepted so the service can consume both the original FlashHead
bridge files and the newer ``crisp_g1`` gesture candidates.

Bridge format::

    {"joints": ["j1"], "frames": [{"t": 0.0, "q": [0.0]}, ...]}

CRISP format::

    {"neutral": {"j1": 0.0},
     "keyframes": [{"time": 0.0, "targets": {"j1": 0.0}}, ...]}
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


class MotionValidationError(ValueError):
    """Raised when a motion cannot be played safely or unambiguously."""


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MotionValidationError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise MotionValidationError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class Oscillation:
    joint: str
    center: float
    amplitude: float
    frequency_hz: float
    start: float
    end: float
    phase: float = 0.0
    ramp: float = 0.12

    def value(self, t: float) -> float | None:
        if t < self.start or t > self.end:
            return None
        local = t - self.start
        envelope = 1.0
        if self.ramp > 0.0:
            envelope = min(
                1.0,
                max(0.0, local / self.ramp),
                max(0.0, (self.end - t) / self.ramp),
            )
        return self.center + self.amplitude * envelope * math.sin(
            2.0 * math.pi * self.frequency_hz * local + self.phase
        )


@dataclass(frozen=True)
class MotionClip:
    name: str
    joints: tuple[str, ...]
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    interpolation: str = "smoothstep"
    oscillations: tuple[Oscillation, ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise MotionValidationError("motion name cannot be empty")
        if not self.joints or len(set(self.joints)) != len(self.joints):
            raise MotionValidationError("joints must be a non-empty unique list")
        if len(self.times) != len(self.values) or not self.times:
            raise MotionValidationError("times and values must contain the same frames")
        if self.times[0] != 0.0:
            raise MotionValidationError("normalized motion must start at t=0")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise MotionValidationError("frame times must be strictly increasing")
        if any(len(row) != len(self.joints) for row in self.values):
            raise MotionValidationError("every frame must contain one value per joint")
        if self.interpolation not in {"linear", "smoothstep"}:
            raise MotionValidationError("interpolation must be linear or smoothstep")

    @property
    def duration(self) -> float:
        return self.times[-1]

    def sample(self, t: float) -> dict[str, float]:
        """Sample the clip. Values outside its duration clamp to an endpoint."""
        t = min(max(float(t), 0.0), self.duration)
        if len(self.times) == 1 or t <= 0.0:
            row = self.values[0]
        elif t >= self.duration:
            row = self.values[-1]
        else:
            right = bisect_right(self.times, t)
            left = right - 1
            span = self.times[right] - self.times[left]
            alpha = (t - self.times[left]) / span
            if self.interpolation == "smoothstep":
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            row = tuple(
                a + (b - a) * alpha
                for a, b in zip(self.values[left], self.values[right])
            )
        result = dict(zip(self.joints, row))
        for oscillation in self.oscillations:
            value = oscillation.value(t)
            if value is not None:
                result[oscillation.joint] = value
        return result


def _parse_oscillations(
    items: Any,
    neutral: Mapping[str, float],
    duration: float,
) -> tuple[Oscillation, ...]:
    if items is None:
        return ()
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise MotionValidationError("oscillations must be a list")
    result: list[Oscillation] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            raise MotionValidationError(f"oscillations[{index}] must be an object")
        joint = raw.get("joint", raw.get("actuator"))
        if not isinstance(joint, str) or not joint:
            raise MotionValidationError(f"oscillations[{index}].joint is required")
        center = _finite_float(raw.get("center", neutral.get(joint, 0.0)), "center")
        amplitude = _finite_float(raw.get("amplitude", 0.0), "amplitude")
        frequency = _finite_float(
            raw.get("frequency_hz", raw.get("frequency", raw.get("hz", 1.0))),
            "frequency_hz",
        )
        start = _finite_float(raw.get("start", raw.get("start_time", 0.0)), "start")
        end = _finite_float(raw.get("end", raw.get("end_time", duration)), "end")
        phase = _finite_float(raw.get("phase", 0.0), "phase")
        ramp = _finite_float(raw.get("ramp", raw.get("ramp_time", 0.12)), "ramp")
        if frequency < 0.0 or amplitude < 0.0 or ramp < 0.0:
            raise MotionValidationError("oscillation frequency/amplitude/ramp cannot be negative")
        if start < 0.0 or end < start or end > duration + 1e-9:
            raise MotionValidationError("oscillation time range is outside the motion")
        result.append(Oscillation(joint, center, amplitude, frequency, start, end, phase, ramp))
    return tuple(result)


def _parse_bridge(payload: Mapping[str, Any], name: str, source: str | None) -> MotionClip:
    joints_raw = payload.get("joints")
    frames = payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
        raise MotionValidationError("frames must be a non-empty list")
    if joints_raw is None and isinstance(frames[0], Mapping) and isinstance(frames[0].get("q"), Mapping):
        # Some bridge files omit ``joints`` because each frame is self-labelled.
        joints_raw = list(frames[0]["q"])
    if not isinstance(joints_raw, Sequence) or isinstance(joints_raw, (str, bytes)):
        raise MotionValidationError("joints must be a list")
    joints = tuple(str(joint) for joint in joints_raw)
    if not joints or any(not joint for joint in joints) or len(set(joints)) != len(joints):
        raise MotionValidationError("joints must be a non-empty unique list of names")

    absolute_times: list[float] = []
    rows: list[tuple[float, ...]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise MotionValidationError(f"frames[{index}] must be an object")
        time_value = _finite_float(frame.get("t", frame.get("time")), f"frames[{index}].t")
        q = frame.get("q")
        if isinstance(q, Mapping):
            missing = [joint for joint in joints if joint not in q]
            if missing:
                raise MotionValidationError(
                    f"frames[{index}].q is missing joints: {', '.join(missing)}"
                )
            rows.append(
                tuple(_finite_float(q[joint], f"frames[{index}].q.{joint}") for joint in joints)
            )
        elif isinstance(q, Sequence) and not isinstance(q, (str, bytes)) and len(q) == len(joints):
            rows.append(tuple(_finite_float(value, f"frames[{index}].q") for value in q))
        else:
            raise MotionValidationError(f"frames[{index}].q must match joints")
        absolute_times.append(time_value)
    if absolute_times[0] < 0.0:
        raise MotionValidationError("first frame time cannot be negative")
    if any(b <= a for a, b in zip(absolute_times, absolute_times[1:])):
        raise MotionValidationError("frame times must be strictly increasing")
    origin = absolute_times[0]
    times = tuple(value - origin for value in absolute_times)
    declared_duration = payload.get("duration")
    if declared_duration is not None:
        duration = _finite_float(declared_duration, "duration")
        if duration < times[-1] - 1e-9:
            raise MotionValidationError("duration cannot be shorter than the last frame")
        if duration > times[-1] + 1e-9:
            times = times + (duration,)
            rows.append(rows[-1])
    interpolation = str(payload.get("interpolation", "smoothstep"))
    oscillations = _parse_oscillations(payload.get("oscillations"), {}, times[-1])
    return MotionClip(name, joints, times, tuple(rows), interpolation, oscillations, source)


def _parse_crisp(payload: Mapping[str, Any], name: str, source: str | None) -> MotionClip:
    raw_neutral = payload.get("neutral", {})
    raw_frames = payload.get("keyframes")
    if not isinstance(raw_neutral, Mapping):
        raise MotionValidationError("neutral must be an object")
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)) or not raw_frames:
        raise MotionValidationError("keyframes must be a non-empty list")

    neutral = {str(key): _finite_float(value, f"neutral.{key}") for key, value in raw_neutral.items()}
    joint_order = list(neutral)
    absolute_times: list[float] = []
    target_frames: list[dict[str, float]] = []
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, Mapping):
            raise MotionValidationError(f"keyframes[{index}] must be an object")
        absolute_times.append(
            _finite_float(frame.get("time", frame.get("t")), f"keyframes[{index}].time")
        )
        raw_targets = frame.get("targets", {})
        if not isinstance(raw_targets, Mapping):
            raise MotionValidationError(f"keyframes[{index}].targets must be an object")
        targets: dict[str, float] = {}
        for key, value in raw_targets.items():
            joint = str(key)
            if joint not in joint_order:
                joint_order.append(joint)
            targets[joint] = _finite_float(value, f"keyframes[{index}].targets.{joint}")
        target_frames.append(targets)

    raw_oscillations = payload.get("oscillations")
    if isinstance(raw_oscillations, Sequence) and not isinstance(raw_oscillations, (str, bytes)):
        for raw in raw_oscillations:
            if isinstance(raw, Mapping):
                joint = raw.get("joint", raw.get("actuator"))
                if isinstance(joint, str) and joint and joint not in joint_order:
                    joint_order.append(joint)
    if not joint_order:
        raise MotionValidationError("motion does not target any joints")
    if absolute_times[0] < 0.0 or any(b <= a for a, b in zip(absolute_times, absolute_times[1:])):
        raise MotionValidationError("keyframe times must be non-negative and strictly increasing")

    current = {joint: neutral.get(joint, 0.0) for joint in joint_order}
    rows: list[tuple[float, ...]] = []
    for targets in target_frames:
        current.update(targets)
        rows.append(tuple(current[joint] for joint in joint_order))
    origin = absolute_times[0]
    times = tuple(value - origin for value in absolute_times)
    declared_duration = payload.get("duration")
    if declared_duration is not None:
        duration = _finite_float(declared_duration, "duration")
        if duration < times[-1] - 1e-9:
            raise MotionValidationError("duration cannot be shorter than the last keyframe")
        if duration > times[-1] + 1e-9:
            times = times + (duration,)
            rows.append(rows[-1])
    interpolation = str(payload.get("interpolation", "smoothstep"))
    oscillations = _parse_oscillations(raw_oscillations, neutral, times[-1])
    return MotionClip(
        name,
        tuple(joint_order),
        times,
        tuple(rows),
        interpolation,
        oscillations,
        source,
    )


def parse_motion(
    payload: Mapping[str, Any],
    *,
    name: str | None = None,
    source: str | None = None,
) -> MotionClip:
    """Parse and validate either supported motion format."""
    if not isinstance(payload, Mapping):
        raise MotionValidationError("motion must be a JSON object")
    clip_name = name or str(payload.get("name", payload.get("id", "inline-motion")))
    if "frames" in payload:
        return _parse_bridge(payload, clip_name, source)
    if "keyframes" in payload:
        return _parse_crisp(payload, clip_name, source)
    raise MotionValidationError("motion needs either frames or keyframes")
