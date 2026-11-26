"""Canonical Unitree G1 joint names used by the atom pipeline.

The BEAT2 retargeting files and the existing motion JSON files use names such
as ``left_elbow`` while MuJoCo models commonly append ``_joint``.  Keeping one
canonical vocabulary at the dataset boundary avoids silently assigning a
trajectory column to the wrong joint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar


G1_29_JOINTS: tuple[str, ...] = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
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

G1_LEG_JOINTS: tuple[str, ...] = G1_29_JOINTS[:12]
G1_WAIST_JOINTS: tuple[str, ...] = G1_29_JOINTS[12:15]
G1_ARM_JOINTS: tuple[str, ...] = G1_29_JOINTS[15:]
G1_UPPER_BODY_JOINTS: tuple[str, ...] = G1_WAIST_JOINTS + G1_ARM_JOINTS
G1_JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(G1_29_JOINTS)}

_G1_SET = frozenset(G1_29_JOINTS)
_ValueT = TypeVar("_ValueT")


def canonical_joint_name(name: str, *, strict: bool = True) -> str:
    """Return the no-suffix G1 name for a dataset or MuJoCo joint name.

    Only the harmless ``_joint`` transport alias is normalized.  Fuzzy name
    matching is deliberately avoided because a wrong column-to-joint mapping
    is a safety error, not a spelling correction.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("joint name must be a non-empty string")
    cleaned = name.strip()
    candidate = cleaned[:-6] if cleaned.endswith("_joint") else cleaned
    if candidate in _G1_SET:
        return candidate
    if strict:
        raise KeyError(f"unknown G1 joint: {name!r}")
    return candidate


def joint_aliases(name: str) -> tuple[str, str]:
    """Return the canonical and MuJoCo-suffixed aliases for ``name``."""

    canonical = canonical_joint_name(name)
    return canonical, f"{canonical}_joint"


def normalize_joint_names(names: Iterable[str]) -> tuple[str, ...]:
    """Normalize a joint list and reject duplicate columns after aliasing."""

    result = tuple(canonical_joint_name(name) for name in names)
    if not result:
        raise ValueError("joint list cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError("joint list contains duplicate canonical names")
    return result


def normalize_joint_mapping(
    values: Mapping[str, _ValueT],
    *,
    strict: bool = True,
) -> dict[str, _ValueT]:
    """Normalize mapping keys while detecting ambiguous duplicate aliases."""

    result: dict[str, _ValueT] = {}
    sources: dict[str, str] = {}
    for raw_name, value in values.items():
        canonical = canonical_joint_name(raw_name, strict=strict)
        if canonical in result:
            raise ValueError(
                f"joint aliases {sources[canonical]!r} and {raw_name!r} both map "
                f"to {canonical!r}"
            )
        result[canonical] = value
        sources[canonical] = raw_name
    return result


def resolve_available_joint(name: str, available: Iterable[str]) -> str | None:
    """Resolve a canonical/``_joint`` name against names exposed by a backend."""

    canonical = canonical_joint_name(name)
    by_canonical: dict[str, str] = {}
    for candidate in available:
        try:
            key = canonical_joint_name(candidate)
        except (KeyError, ValueError):
            continue
        if key in by_canonical and by_canonical[key] != candidate:
            raise ValueError(f"backend exposes ambiguous aliases for {key!r}")
        by_canonical[key] = candidate
    return by_canonical.get(canonical)
