from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_JOINTS = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

MINIMUM_JERK_PEAK_SPEED_RAD_S = 1.0
MINIMUM_JERK_DURATION_FPS = 25.0
MINIMUM_JERK_MIN_FRAME_INTERVALS = 2


class CatalogError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def minimum_jerk_duration(maximum_delta: float) -> float:
    required = 1.875 * float(maximum_delta) / MINIMUM_JERK_PEAK_SPEED_RAD_S
    ticks = max(
        MINIMUM_JERK_MIN_FRAME_INTERVALS,
        int(math.ceil(required * MINIMUM_JERK_DURATION_FPS - 1e-12)),
    )
    return ticks / MINIMUM_JERK_DURATION_FPS


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CatalogError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CatalogError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CatalogError(f"{label} must be finite")
    return parsed


def _validate_motion(motion: Any, label: str) -> tuple[float, float, int]:
    if not isinstance(motion, Mapping):
        raise CatalogError(f"{label} must be an object")
    if motion.get("format") != "g1_joint_motion_v1":
        raise CatalogError(f"{label}.format must be g1_joint_motion_v1")
    if motion.get("interpolation") != "linear":
        raise CatalogError(f"{label}.interpolation must be linear")
    joints = motion.get("joints")
    if tuple(joints or ()) != EXPECTED_JOINTS:
        raise CatalogError(f"{label}.joints is not the canonical 14-joint order")
    frames = motion.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise CatalogError(f"{label}.frames must contain at least two frames")

    previous_t: float | None = None
    max_abs_q = 0.0
    max_segment_velocity = 0.0
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise CatalogError(f"{label}.frames[{frame_index}] must be an object")
        t = _finite_float(frame.get("t"), f"{label}.frames[{frame_index}].t")
        q = frame.get("q")
        if not isinstance(q, Mapping) or set(q) != set(EXPECTED_JOINTS):
            raise CatalogError(
                f"{label}.frames[{frame_index}].q must contain exactly 14 joints"
            )
        current_q: dict[str, float] = {}
        for joint in EXPECTED_JOINTS:
            current_q[joint] = _finite_float(
                q[joint], f"{label}.frames[{frame_index}].q.{joint}"
            )
            max_abs_q = max(max_abs_q, abs(current_q[joint]))
        if previous_t is None:
            if abs(t) > 1e-9:
                raise CatalogError(f"{label} must start at t=0")
        else:
            if t <= previous_t:
                raise CatalogError(f"{label} frame times must be strictly increasing")
            dt = t - previous_t
            previous_q = frames[frame_index - 1]["q"]
            for joint in EXPECTED_JOINTS:
                velocity = abs(current_q[joint] - float(previous_q[joint])) / dt
                max_segment_velocity = max(max_segment_velocity, velocity)
        previous_t = t

    duration = _finite_float(motion.get("duration"), f"{label}.duration")
    if abs(duration - float(previous_t)) > 1e-6:
        raise CatalogError(f"{label}.duration does not match its final frame")
    return max_abs_q, max_segment_velocity, len(frames)


@dataclass(frozen=True, slots=True)
class AtomRecord:
    index: int
    entry_id: str
    atom_id: str
    function: str
    version: str
    revision: int
    content_sha256: str
    duration: float
    frame_count: int
    max_abs_q: float
    max_segment_velocity: float
    bridge_from_neutral: float
    bridge_to_neutral: float
    motion_sha256: str
    motion: Mapping[str, Any]
    raw: Mapping[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "entry_id": self.entry_id,
            "atom_id": self.atom_id,
            "function": self.function,
            "version": self.version,
            "revision": self.revision,
            "content_sha256": self.content_sha256,
            "motion_sha256": self.motion_sha256,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "max_abs_q": self.max_abs_q,
            "max_segment_velocity": self.max_segment_velocity,
            "bridge_from_neutral": self.bridge_from_neutral,
            "bridge_to_neutral": self.bridge_to_neutral,
        }


@dataclass(frozen=True, slots=True)
class AtomCatalog:
    path: Path
    file_sha256: str
    entries: tuple[AtomRecord, ...]
    neutral_motion: Mapping[str, Any]
    neutral_motion_sha256: str
    by_entry_id: Mapping[str, AtomRecord]
    by_atom_id: Mapping[str, AtomRecord]

    def resolve(self, identity: str) -> AtomRecord:
        if identity in self.by_entry_id:
            return self.by_entry_id[identity]
        if identity in self.by_atom_id:
            return self.by_atom_id[identity]
        raise KeyError(identity)


def load_catalog(path: str | Path) -> AtomCatalog:
    catalog_path = Path(path).expanduser().resolve()
    try:
        body = catalog_path.read_bytes()
        payload = json.loads(body)
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read catalog {catalog_path}: {exc}") from exc
    file_sha256 = hashlib.sha256(body).hexdigest()
    expected_name = f"semantic_atom_catalog_v1.{file_sha256}.json"
    if catalog_path.name != expected_name:
        raise CatalogError(
            f"catalog filename is not its content address; expected {expected_name}"
        )
    if not isinstance(payload, Mapping):
        raise CatalogError("catalog root must be an object")

    neutral = payload.get("neutral")
    if not isinstance(neutral, Mapping):
        raise CatalogError("catalog.neutral must be an object")
    neutral_motion = neutral.get("motion")
    _validate_motion(neutral_motion, "catalog.neutral.motion")
    neutral_q = neutral.get("q")
    if not isinstance(neutral_q, Mapping) or set(neutral_q) != set(EXPECTED_JOINTS):
        raise CatalogError("catalog.neutral.q must contain exactly 14 joints")
    parsed_neutral_q = {
        joint: _finite_float(neutral_q[joint], f"catalog.neutral.q.{joint}")
        for joint in EXPECTED_JOINTS
    }

    rows = payload.get("entries")
    if not isinstance(rows, list) or not rows:
        raise CatalogError("catalog.entries must be a non-empty array")

    entries: list[AtomRecord] = []
    entry_ids: set[str] = set()
    atom_ids: set[str] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise CatalogError(f"catalog.entries[{position - 1}] must be an object")
        entry_id = str(row.get("entry_id", ""))
        atom_id = str(row.get("atom_id", ""))
        content_sha256 = str(row.get("content_sha256", ""))
        if len(entry_id) != 64 or len(content_sha256) != 64 or not atom_id:
            raise CatalogError(f"entry {position} has invalid identity fields")
        if entry_id in entry_ids or atom_id in atom_ids:
            raise CatalogError(f"entry {position} duplicates an identity")
        entry_ids.add(entry_id)
        atom_ids.add(atom_id)

        motion = row.get("motion")
        max_abs_q, max_segment_velocity, frame_count = _validate_motion(
            motion, f"catalog.entries[{position - 1}].motion"
        )
        assert isinstance(motion, Mapping)
        frames = motion["frames"]
        first_q = frames[0]["q"]
        last_q = frames[-1]["q"]
        from_delta = max(
            abs(float(first_q[joint]) - parsed_neutral_q[joint])
            for joint in EXPECTED_JOINTS
        )
        to_delta = max(
            abs(float(last_q[joint]) - parsed_neutral_q[joint])
            for joint in EXPECTED_JOINTS
        )
        duration = float(motion["duration"])
        declared_us = row.get("duration_us")
        if declared_us is not None and abs(float(declared_us) / 1_000_000 - duration) > 1e-6:
            raise CatalogError(f"entry {position} duration_us does not match motion")
        entries.append(
            AtomRecord(
                index=int(row.get("index", position)),
                entry_id=entry_id,
                atom_id=atom_id,
                function=str(row.get("function", "")),
                version=str(row.get("version", "")),
                revision=int(row.get("revision", 0)),
                content_sha256=content_sha256,
                duration=duration,
                frame_count=frame_count,
                max_abs_q=max_abs_q,
                max_segment_velocity=max_segment_velocity,
                bridge_from_neutral=minimum_jerk_duration(from_delta),
                bridge_to_neutral=minimum_jerk_duration(to_delta),
                motion_sha256=canonical_sha256(motion),
                motion=motion,
                raw=row,
            )
        )

    entries.sort(key=lambda item: item.index)
    return AtomCatalog(
        path=catalog_path,
        file_sha256=file_sha256,
        entries=tuple(entries),
        neutral_motion=neutral_motion,
        neutral_motion_sha256=canonical_sha256(neutral_motion),
        by_entry_id={entry.entry_id: entry for entry in entries},
        by_atom_id={entry.atom_id: entry for entry in entries},
    )
