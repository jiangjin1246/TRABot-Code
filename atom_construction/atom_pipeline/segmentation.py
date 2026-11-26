"""Natural-boundary gesture segmentation and phase/apex detection.

The implementation intentionally uses NumPy only.  Every threshold is exposed
through :class:`DetectorConfig` so an experiment can freeze and hash it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    detection_fps: float = 25.0
    min_duration_s: float = 0.40
    ideal_min_duration_s: float = 0.70
    ideal_max_duration_s: float = 2.80
    max_duration_s: float = 4.00
    on_hold_s: float = 0.12
    off_hold_s: float = 0.16
    merge_gap_s: float = 0.12
    hold_merge_max_s: float = 1.00
    boundary_search_s: float = 0.32
    boundary_window_s: float = 0.08
    boundary_review_cost: float = 0.35
    boundary_reject_cost: float = 0.50
    min_stroke_s: float = 0.12
    min_hold_s: float = 0.12
    min_apex_separation_s: float = 0.20
    motion_snr_min: float = 4.0
    displacement_min: float = 0.04
    core_quality_min: float = 0.55

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhaseInterval:
    name: str
    start_index: int
    end_index: int


@dataclass(frozen=True, slots=True)
class Apex:
    index: int
    kind: str
    confidence: float


@dataclass(frozen=True, slots=True)
class GestureSegment:
    start_index: int
    end_index: int
    phases: tuple[PhaseInterval, ...]
    primary_apex: Apex
    apices: tuple[Apex, ...]
    active_joint_indexes: tuple[int, ...]
    exit_kind: str
    boundary_start_cost: float
    boundary_end_cost: float
    motion_snr: float
    displacement_max: float
    kinematic_quality: float
    warnings: tuple[str, ...] = ()

    def duration(self, fps: float) -> float:
        return (self.end_index - self.start_index) / float(fps)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    fps: float
    times: np.ndarray
    q: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    energy: np.ndarray
    floor: float
    sigma: float
    dynamic_range: float
    theta_on: float
    theta_off: float
    segments: tuple[GestureSegment, ...]
    rejections: tuple[dict[str, object], ...]


def _seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    return max(minimum, int(round(seconds * fps)))


def _resample(q: np.ndarray, source_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = (len(q) - 1) / source_fps
    sample_count = max(2, int(math.floor(duration * target_fps + 1e-9)) + 1)
    target_times = np.arange(sample_count, dtype=np.float64) / target_fps
    if target_times[-1] < duration - 0.5 / target_fps:
        target_times = np.append(target_times, duration)
    source_times = np.arange(len(q), dtype=np.float64) / source_fps
    result = np.empty((len(target_times), q.shape[1]), dtype=np.float64)
    for index in range(q.shape[1]):
        result[:, index] = np.interp(target_times, source_times, q[:, index])
    return target_times, result


def _median3(q: np.ndarray) -> np.ndarray:
    padded = np.pad(q, ((1, 1), (0, 0)), mode="reflect")
    return np.median(np.stack((padded[:-2], padded[1:-1], padded[2:])), axis=0)


def _binomial5(q: np.ndarray) -> np.ndarray:
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    padded = np.pad(q, ((2, 2), (0, 0)), mode="reflect")
    result = np.empty_like(q)
    for index in range(q.shape[1]):
        result[:, index] = np.convolve(padded[:, index], kernel, mode="valid")
    return result


def preprocess_trajectory(
    q: np.ndarray,
    source_fps: float,
    target_fps: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError(f"q must have shape (T,J) with T>=3, got {values.shape}")
    if not math.isfinite(source_fps) or not 20.0 <= source_fps <= 120.0:
        raise ValueError(f"source fps must be in [20,120], got {source_fps}")
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target fps must be positive")
    if not np.isfinite(values).all():
        raise ValueError("q contains NaN or Inf")
    times, resampled = _resample(values, float(source_fps), float(target_fps))
    smooth = _binomial5(_median3(resampled))
    edge_order = 2 if len(times) >= 3 else 1
    velocity = np.gradient(smooth, times, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=edge_order)
    return times, smooth, velocity, acceleration


def normalized_motion_energy(
    velocity: np.ndarray,
    joint_scales: np.ndarray,
    joint_weights: np.ndarray | None = None,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    scales = np.maximum(np.asarray(joint_scales, dtype=np.float64), 0.25)
    if scales.shape != (velocity.shape[1],):
        raise ValueError("joint_scales must contain one value per joint")
    weights = (
        np.ones(velocity.shape[1], dtype=np.float64)
        if joint_weights is None
        else np.asarray(joint_weights, dtype=np.float64)
    )
    if weights.shape != scales.shape or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("joint_weights must be non-negative and contain one positive value")
    return np.sqrt(np.sum(weights * np.square(velocity / scales), axis=1) / np.sum(weights))


def adaptive_thresholds(energy: np.ndarray) -> tuple[float, float, float, float, float]:
    energy = np.asarray(energy, dtype=np.float64)
    floor = float(np.percentile(energy, 20))
    median = float(np.percentile(energy, 50))
    quiet = energy[energy <= median]
    quiet_median = float(np.median(quiet)) if len(quiet) else floor
    sigma = 1.4826 * float(np.median(np.abs(quiet - quiet_median))) if len(quiet) else 0.0
    dynamic_range = max(0.0, float(np.percentile(energy, 95)) - floor)
    on_delta = float(np.clip(max(3.0 * sigma, 0.20 * dynamic_range), 0.20 * dynamic_range, 0.55 * dynamic_range))
    off_delta = float(np.clip(max(1.5 * sigma, 0.08 * dynamic_range), 0.08 * dynamic_range, 0.25 * dynamic_range))
    return floor, sigma, dynamic_range, floor + on_delta, floor + off_delta


def _hysteresis_islands(
    energy: np.ndarray,
    theta_on: float,
    theta_off: float,
    on_frames: int,
    off_frames: int,
    merge_frames: int,
) -> list[tuple[int, int]]:
    islands: list[tuple[int, int]] = []
    active = False
    on_count = 0
    off_count = 0
    start = 0
    for index, value in enumerate(energy):
        if not active:
            on_count = on_count + 1 if value >= theta_on else 0
            if on_count >= on_frames:
                start = index - on_frames + 1
                active = True
                off_count = 0
        else:
            off_count = off_count + 1 if value <= theta_off else 0
            if off_count >= off_frames:
                end = index - off_frames + 1
                if end > start:
                    islands.append((start, end))
                active = False
                on_count = 0
    if active:
        islands.append((start, len(energy)))

    merged: list[tuple[int, int]] = []
    for start, end in islands:
        if merged and start - merged[-1][1] <= merge_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _merge_held_pose_islands(
    islands: list[tuple[int, int]],
    q: np.ndarray,
    scales: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> list[tuple[int, int]]:
    """Join an outward stroke and its later retraction across a stable hold.

    Energy-only hysteresis sees those as two events.  They belong to one atom
    when the quiet gap holds a displaced pose and the following event moves
    substantially back toward the pose preceding the first event.
    """

    if not islands:
        return []
    maximum_gap = _seconds_to_frames(config.hold_merge_max_s, fps)
    context = _seconds_to_frames(0.20, fps)
    result: list[tuple[int, int]] = [islands[0]]
    for next_start, next_end in islands[1:]:
        previous_start, previous_end = result[-1]
        gap = next_start - previous_end
        merge = False
        if 0 <= gap <= maximum_gap:
            reference = np.median(q[max(0, previous_start - context):previous_start + 1], axis=0)
            hold = np.median(q[previous_end:next_start + 1], axis=0)
            next_exit = np.median(q[max(next_start, next_end - context):next_end], axis=0)
            hold_distance = float(np.sqrt(np.mean(np.square((hold - reference) / scales))))
            exit_distance = float(np.sqrt(np.mean(np.square((next_exit - reference) / scales))))
            merge = (
                hold_distance >= config.displacement_min
                and exit_distance <= max(0.55 * hold_distance, 0.5 * config.displacement_min)
            )
        if merge:
            result[-1] = (previous_start, next_end)
        else:
            result.append((next_start, next_end))
    return result


def _local_pose_variation(q: np.ndarray, center: int, half_window: int, scales: np.ndarray) -> float:
    left = max(0, center - half_window)
    right = min(len(q), center + half_window + 1)
    if right - left < 2:
        return 0.0
    span = np.ptp(q[left:right], axis=0) / scales
    return float(np.sqrt(np.mean(np.square(span))))


def _boundary_costs(
    q: np.ndarray,
    acceleration: np.ndarray,
    energy: np.ndarray,
    floor: float,
    dynamic_range: float,
    scales: np.ndarray,
    half_window: int,
) -> np.ndarray:
    e_norm = np.clip((energy - floor) / max(dynamic_range, 1e-9), 0.0, 1.0)
    accel_energy = np.sqrt(np.mean(np.square(acceleration / scales), axis=1))
    accel_scale = max(float(np.percentile(accel_energy, 95)), 1e-9)
    a_norm = np.clip(accel_energy / accel_scale, 0.0, 1.0)
    pose = np.asarray(
        [_local_pose_variation(q, index, half_window, scales) for index in range(len(q))],
        dtype=np.float64,
    )
    pose_norm = np.clip(pose / max(float(np.percentile(pose, 95)), 1e-9), 0.0, 1.0)
    return 0.60 * e_norm + 0.20 * a_norm + 0.20 * pose_norm


def _natural_boundary(cost: np.ndarray, anchor: int, radius: int, before: bool) -> int:
    if before:
        left, right = max(0, anchor - radius), min(len(cost), anchor + 1)
    else:
        left, right = max(0, anchor), min(len(cost), anchor + radius + 1)
    if right <= left:
        return min(max(anchor, 0), len(cost) - 1)
    return left + int(np.argmin(cost[left:right]))


def _local_extrema(values: np.ndarray, kind: str, radius: int = 1) -> list[int]:
    result: list[int] = []
    index = radius
    while index < len(values) - radius:
        left = index
        right = index
        while right + 1 < len(values) and values[right + 1] == values[index]:
            right += 1
        center = (left + right) // 2
        lo = max(0, left - radius)
        hi = min(len(values), right + radius + 1)
        neighborhood = values[lo:hi]
        good = values[index] >= np.max(neighborhood) if kind == "max" else values[index] <= np.min(neighborhood)
        if good and (lo < left or right + 1 < hi):
            result.append(center)
        index = right + 1
    return result


def _split_interval(
    start: int,
    end: int,
    costs: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> list[tuple[int, int]]:
    max_frames = _seconds_to_frames(config.max_duration_s, fps)
    min_frames = _seconds_to_frames(config.min_duration_s, fps)
    if end - start <= max_frames:
        return [(start, end)]
    minima = [
        index
        for index in _local_extrema(costs[start:end], "min", radius=_seconds_to_frames(0.12, fps))
        if costs[start + index] <= config.boundary_reject_cost
    ]
    nodes = [start] + [start + value for value in minima] + [end]
    nodes = sorted(set(nodes))
    best: dict[int, tuple[float, list[int]]] = {start: (0.0, [start])}
    ideal = 0.5 * (config.ideal_min_duration_s + config.ideal_max_duration_s)
    for right in nodes[1:]:
        choices: list[tuple[float, list[int]]] = []
        for left in nodes:
            if left >= right or left not in best:
                continue
            duration_frames = right - left
            if duration_frames < min_frames or duration_frames > max_frames:
                continue
            duration = duration_frames / fps
            duration_cost = 0.08 * abs(duration - ideal) / max(ideal, 1e-9)
            boundary_cost = 0.0 if right == end else float(costs[right])
            old_cost, old_path = best[left]
            choices.append((old_cost + boundary_cost + duration_cost, old_path + [right]))
        if choices:
            best[right] = min(choices, key=lambda item: item[0])
    if end not in best:
        return []
    path = best[end][1]
    return list(zip(path, path[1:]))


def _active_joints(q: np.ndarray, velocity: np.ndarray, scales: np.ndarray, fps: float) -> tuple[int, ...]:
    contribution = np.sum(np.abs(velocity) / scales, axis=0) / fps
    maximum = float(np.max(contribution)) if len(contribution) else 0.0
    eligible = np.flatnonzero(contribution >= 0.10 * maximum) if maximum > 0.0 else np.asarray([], dtype=int)
    if len(eligible) == 0:
        return (int(np.argmax(contribution)),)
    ordered = eligible[np.argsort(contribution[eligible])[::-1]]
    target = 0.85 * float(np.sum(contribution[eligible]))
    chosen: list[int] = []
    running = 0.0
    for index in ordered:
        chosen.append(int(index))
        running += float(contribution[index])
        if running >= target:
            break
    return tuple(sorted(chosen))


def _prominence(values: np.ndarray, index: int, radius: int) -> float:
    left = values[max(0, index - radius):index + 1]
    right = values[index:min(len(values), index + radius + 1)]
    if not len(left) or not len(right):
        return 0.0
    return float(values[index] - max(float(np.min(left)), float(np.min(right))))


def _nms(candidates: list[Apex], min_separation: int) -> tuple[Apex, ...]:
    selected: list[Apex] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.index)):
        if all(abs(candidate.index - old.index) >= min_separation for old in selected):
            selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.index))


def _phase_segment(
    start: int,
    end: int,
    q: np.ndarray,
    velocity: np.ndarray,
    energy: np.ndarray,
    floor: float,
    sigma: float,
    dynamic_range: float,
    scales: np.ndarray,
    costs: np.ndarray,
    fps: float,
    config: DetectorConfig,
) -> GestureSegment | dict[str, object]:
    local_q = q[start:end]
    local_v = velocity[start:end]
    local_e = energy[start:end]
    if len(local_q) < _seconds_to_frames(config.min_duration_s, fps):
        return {"reason": "duration_too_short", "start_index": start, "end_index": end}
    active = _active_joints(local_q, local_v, scales, fps)
    active_columns = np.asarray(active, dtype=int)
    q_ref = np.median(local_q[:max(2, _seconds_to_frames(0.12, fps))], axis=0)
    displacement = np.sqrt(
        np.mean(
            np.square(
                (local_q[:, active_columns] - q_ref[active_columns])
                / scales[active_columns]
            ),
            axis=1,
        )
    )
    d_max = float(np.max(displacement))
    radius = _seconds_to_frames(0.12, fps)
    prominence_radius = _seconds_to_frames(0.48, fps)
    candidates: list[Apex] = []
    # A communicative apex is frequently a stationary pointing/open-hand
    # plateau. Its prominence can be near zero even though the pose is the
    # semantic apex. Detect the first stable sample on the high-displacement
    # plateau before considering kinetic peaks.
    high_pose = displacement >= 0.85 * max(d_max, 1e-9)
    quiet_pose = local_e <= floor + max(1.5 * sigma, 0.15 * dynamic_range)
    plateau_mask = high_pose & quiet_pose
    minimum_plateau = _seconds_to_frames(config.min_hold_s, fps)
    run_start: int | None = None
    for index, active_flag in enumerate(plateau_mask):
        if active_flag and run_start is None:
            run_start = index
        at_end = index == len(plateau_mask) - 1
        if run_start is not None and ((not active_flag) or at_end):
            run_end = index if not active_flag else index + 1
            if run_end - run_start >= minimum_plateau:
                plateau_index = run_start
                confidence = float(np.clip(
                    0.70 * displacement[plateau_index] / max(d_max, 1e-9)
                    + 0.20
                    + 0.10 * (run_end - run_start) / max(minimum_plateau, 1),
                    0.0,
                    1.0,
                ))
                candidates.append(Apex(start + plateau_index, "postural", confidence))
                break
            run_start = None
    for index in _local_extrema(displacement, "max", radius=radius):
        prominence = _prominence(displacement, index, prominence_radius)
        if prominence < max(0.015, 0.12 * max(d_max, 1e-9)):
            continue
        reversal = 0.0
        if 0 < index < len(displacement) - 1:
            before = displacement[index] - displacement[max(0, index - radius)]
            after = displacement[min(len(displacement) - 1, index + radius)] - displacement[index]
            reversal = 1.0 if before >= 0.0 and after <= 0.0 else 0.0
        quiet = 1.0 - float(np.clip((local_e[index] - floor) / max(dynamic_range, 1e-9), 0.0, 1.0))
        confidence = float(np.clip(
            0.50 * displacement[index] / max(d_max, 1e-9)
            + 0.25 * prominence / max(d_max, 1e-9)
            + 0.15 * reversal
            + 0.10 * quiet,
            0.0,
            1.0,
        ))
        if confidence >= 0.60:
            candidates.append(Apex(start + index, "postural", confidence))

    if not candidates:
        peak = int(np.argmax(local_e))
        e_prominence = _prominence(local_e, peak, prominence_radius)
        if e_prominence < 0.18 * max(dynamic_range, 1e-9):
            return {
                "reason": "no_reliable_apex",
                "start_index": start,
                "end_index": end,
                "displacement_max": d_max,
            }
        confidence = float(np.clip(
            0.70 * (local_e[peak] - floor) / max(dynamic_range, 1e-9)
            + 0.30 * e_prominence / max(dynamic_range, 1e-9),
            0.0,
            1.0,
        ))
        candidates = [Apex(start + peak, "kinetic", confidence)]

    apices = _nms(candidates, _seconds_to_frames(config.min_apex_separation_s, fps))
    primary = max(apices, key=lambda item: item.confidence)
    primary_local = primary.index - start
    # For a postural apex, the stroke is the outward motion leading to it. A
    # faster retraction must not redefine the gesture apex or stroke.
    search_end = primary_local + 1 if primary.kind == "postural" else len(local_e)
    kinetic_peak = int(np.argmax(local_e[:search_end]))
    stroke_threshold = floor + max(
        0.35 * (float(local_e[kinetic_peak]) - floor),
        0.15 * dynamic_range,
    )
    stroke_left = kinetic_peak
    while stroke_left > 0 and local_e[stroke_left - 1] >= stroke_threshold:
        stroke_left -= 1
    stroke_right = kinetic_peak + 1
    while stroke_right < len(local_e) and local_e[stroke_right] >= stroke_threshold:
        stroke_right += 1
    stroke_left = min(stroke_left, primary.index - start)
    stroke_right = max(stroke_right, primary.index - start + 1)
    min_stroke = _seconds_to_frames(config.min_stroke_s, fps)
    if stroke_right - stroke_left < min_stroke:
        padding = min_stroke - (stroke_right - stroke_left)
        stroke_left = max(0, stroke_left - padding // 2)
        stroke_right = min(len(local_e), stroke_right + padding - padding // 2)
    if stroke_right - stroke_left < min_stroke:
        return {"reason": "stroke_too_short", "start_index": start, "end_index": end}

    hold_threshold = floor + float(np.clip(
        max(1.5 * sigma, 0.10 * dynamic_range),
        0.10 * dynamic_range,
        0.22 * dynamic_range,
    ))
    hold_start = stroke_right
    hold_end = hold_start
    min_hold = _seconds_to_frames(config.min_hold_s, fps)
    while hold_end < len(local_e):
        high_pose = displacement[hold_end] >= 0.70 * max(float(displacement[primary.index - start]), 1e-9)
        if local_e[hold_end] <= hold_threshold and high_pose:
            hold_end += 1
        else:
            break
    if hold_end - hold_start < min_hold:
        hold_end = hold_start

    phases: list[PhaseInterval] = []
    if stroke_left > 0:
        phases.append(PhaseInterval("preparation", start, start + stroke_left))
    phases.append(PhaseInterval("stroke", start + stroke_left, start + stroke_right))
    if hold_end > hold_start:
        phases.append(PhaseInterval("hold", start + hold_start, start + hold_end))
    remainder_start = hold_end if hold_end > hold_start else stroke_right
    exit_distance = float(displacement[-1])
    exit_speed = float(local_e[-1])
    if remainder_start < len(local_e) and exit_distance <= max(0.30 * d_max, 0.04):
        phases.append(PhaseInterval("retraction", start + remainder_start, end))
        exit_kind = "returned"
    elif exit_speed <= floor + 0.25 * dynamic_range:
        if remainder_start < len(local_e):
            phases.append(PhaseInterval("hold", start + remainder_start, end))
        exit_kind = "stable_new_pose"
    else:
        exit_kind = "moving_open"

    warnings: list[str] = []
    if len(apices) > 1:
        warnings.append("multiple_apices_review_required")
    if exit_kind == "moving_open":
        warnings.append("moving_open_exit_not_core_eligible")
    boundary_start = float(costs[start])
    boundary_end = float(costs[min(end - 1, len(costs) - 1)])
    snr = dynamic_range / max(sigma, 1e-9)
    boundary_quietness = 1.0 - min(1.0, (boundary_start + boundary_end) / max(2.0 * config.boundary_reject_cost, 1e-9))
    quality = float(np.clip(
        0.25 * boundary_quietness
        + 0.20 * min(1.0, snr / 10.0)
        + 0.25 * primary.confidence
        + 0.20
        + 0.10 * (1.0 if exit_kind != "moving_open" else 0.0),
        0.0,
        1.0,
    ))
    return GestureSegment(
        start_index=start,
        end_index=end,
        phases=tuple(phases),
        primary_apex=primary,
        apices=apices,
        active_joint_indexes=active,
        exit_kind=exit_kind,
        boundary_start_cost=boundary_start,
        boundary_end_cost=boundary_end,
        motion_snr=snr,
        displacement_max=d_max,
        kinematic_quality=quality,
        warnings=tuple(warnings),
    )


def detect_gesture_segments(
    q: np.ndarray,
    source_fps: float,
    *,
    joint_scales: Sequence[float] | None = None,
    joint_weights: Sequence[float] | None = None,
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Detect phase-structured motion atoms from a retargeted trajectory."""

    config = config or DetectorConfig()
    times, smooth, velocity, acceleration = preprocess_trajectory(
        q,
        source_fps,
        config.detection_fps,
    )
    if joint_scales is None:
        scales = np.maximum(np.percentile(smooth, 95, axis=0) - np.percentile(smooth, 5, axis=0), 0.25)
    else:
        scales = np.maximum(np.asarray(joint_scales, dtype=np.float64), 0.25)
    weights = None if joint_weights is None else np.asarray(joint_weights, dtype=np.float64)
    energy = normalized_motion_energy(velocity, scales, weights)
    floor, sigma, dynamic_range, theta_on, theta_off = adaptive_thresholds(energy)
    rejections: list[dict[str, object]] = []
    if dynamic_range <= 1e-9 or dynamic_range / max(sigma, 1e-9) < config.motion_snr_min:
        rejections.append({
            "reason": "motion_snr_too_low",
            "motion_snr": dynamic_range / max(sigma, 1e-9),
        })
        return DetectionResult(
            config.detection_fps, times, smooth, velocity, acceleration, energy,
            floor, sigma, dynamic_range, theta_on, theta_off, (), tuple(rejections),
        )

    islands = _hysteresis_islands(
        energy,
        theta_on,
        theta_off,
        _seconds_to_frames(config.on_hold_s, config.detection_fps),
        _seconds_to_frames(config.off_hold_s, config.detection_fps),
        _seconds_to_frames(config.merge_gap_s, config.detection_fps),
    )
    islands = _merge_held_pose_islands(
        islands,
        smooth,
        scales,
        config.detection_fps,
        config,
    )
    costs = _boundary_costs(
        smooth,
        acceleration,
        energy,
        floor,
        dynamic_range,
        scales,
        _seconds_to_frames(config.boundary_window_s, config.detection_fps),
    )
    search = _seconds_to_frames(config.boundary_search_s, config.detection_fps)
    intervals: list[tuple[int, int]] = []
    for island_start, island_end in islands:
        start = _natural_boundary(costs, island_start, search, before=True)
        end_anchor = min(len(costs) - 1, max(island_end, start + 1))
        end = _natural_boundary(costs, end_anchor, search, before=False) + 1
        if costs[start] > config.boundary_reject_cost or costs[end - 1] > config.boundary_reject_cost:
            rejections.append({
                "reason": "boundary_cost_too_high",
                "start_index": start,
                "end_index": end,
                "start_cost": float(costs[start]),
                "end_cost": float(costs[end - 1]),
            })
            continue
        split = _split_interval(start, end, costs, config.detection_fps, config)
        if not split:
            rejections.append({
                "reason": "continuous_unsegmented",
                "start_index": start,
                "end_index": end,
            })
        intervals.extend(split)

    segments: list[GestureSegment] = []
    for start, end in intervals:
        result = _phase_segment(
            start, end, smooth, velocity, energy, floor, sigma, dynamic_range,
            scales, costs, config.detection_fps, config,
        )
        if isinstance(result, dict):
            rejections.append(result)
            continue
        if result.displacement_max < config.displacement_min and result.primary_apex.kind != "kinetic":
            rejections.append({
                "reason": "displacement_too_small",
                "start_index": start,
                "end_index": end,
                "displacement_max": result.displacement_max,
            })
            continue
        if result.kinematic_quality < config.core_quality_min:
            rejections.append({
                "reason": "kinematic_quality_too_low",
                "start_index": start,
                "end_index": end,
                "kinematic_quality": result.kinematic_quality,
            })
            continue
        segments.append(result)

    return DetectionResult(
        config.detection_fps,
        times,
        smooth,
        velocity,
        acceleration,
        energy,
        floor,
        sigma,
        dynamic_range,
        theta_on,
        theta_off,
        tuple(segments),
        tuple(rejections),
    )
