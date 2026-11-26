"""Fail-closed compiler and loader for the F17 semantic Atom catalog.

The compiler joins the formal human-label export with the exact Approved Atom
inventory and execution-side sidecar.  It validates every executable Atom,
fixed-base report, and preview binding, then embeds the exact preview motion in
a deterministic content-addressed artifact.  Runtime consumers therefore do
not need to trust mutable source-run paths.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from web_overlay.semantic_ontology import (
    EXCLUDED_FUNCTION,
    ONTOLOGY_SCHEMA_VERSION,
    OntologyValidationError,
    SemanticOntology,
    load_ontology,
)


CATALOG_SCHEMA_VERSION = "semantic_atom_catalog_v1"
ATOM_SCHEMA_VERSION = "robot.gesture_atom.v1"
INVENTORY_SCHEMA_VERSION = "approved_atom_inventory_v1"
FINAL_LABEL_SCHEMA_VERSION = "f18_human_label_v1"
REMOVED_SCHEMA_VERSION = "f18_removed_action_v1"
SIDE_SCHEMA_VERSION = "f18_exact_motion_sides_v1"
MOTION_FORMAT = "g1_joint_motion_v1"
MUJOCO_BACKEND = "mujoco-g1-29dof"
SIDE_ALGORITHM = "g1_joint_total_variation_handedness_v1"
SIDE_BILATERAL_MIN_RATIO = 0.55
SIDE_NONE_MAX_FRACTION = 0.15

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPAIR_VERSION = re.compile(r"^[0-9a-f]{32}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MAX_INPUT_BYTES = 32 * 1024 * 1024
_MAX_ASSET_BYTES = 64 * 1024 * 1024
_MAX_CATALOG_BYTES = 256 * 1024 * 1024

_INVENTORY_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "catalog_id",
        "source_run_dir",
        "review_export_dir",
        "annotation_snapshot",
        "run_manifest",
        "review_manifest",
        "approved_count",
        "atoms",
    }
)
_INVENTORY_ROW_FIELDS = frozenset(
    {
        "atom_id",
        "version",
        "revision",
        "content_sha256",
        "approved",
        "approval_event_id",
        "decision_source_event_id",
        "latest_annotation_event_id",
        "approval_timestamp",
        "fixed_base_pass",
        "variant",
        "atom_file",
        "atom_file_sha256",
        "preview_file",
        "preview_file_sha256",
        "report_file",
        "report_file_sha256",
        "clip_file",
        "clip_sha256",
        "duration_s",
        "source",
        "source_window",
        "semantics",
        "index",
    }
)
_FINAL_FIELDS = frozenset(
    {
        "schema_version",
        "index",
        "atom_id",
        "version",
        "revision",
        "content_sha256",
        "label",
        "human_review",
    }
)
_REMOVED_FIELDS = _FINAL_FIELDS - {"label"}
_HUMAN_REVIEW_FIELDS = frozenset(
    {
        "decision",
        "event_id",
        "qwen_basis",
        "qwen_label",
        "review_note",
        "reviewed_at_utc",
        "reviewer",
    }
)
_SIDE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "catalog_id",
        "item_count",
        "dataset_manifest_sha256",
        "approved_inventory_sha256",
        "algorithm",
        "items",
    }
)
_SIDE_ALGORITHM_FIELDS = frozenset(
    {
        "algorithm_id",
        "definition",
        "unit",
        "bilateral_min_ratio",
        "none_max_fraction",
        "coordinate_frame",
    }
)
_SIDE_ROW_FIELDS = frozenset(
    {
        "index",
        "atom_id",
        "version",
        "revision",
        "content_sha256",
        "approval_event_id",
        "atom_file_sha256",
        "variant",
        "stored_handedness",
        "stored_left_path_rad",
        "stored_right_path_rad",
        "exact_trajectory_handedness",
        "left_path_rad",
        "right_path_rad",
        "left_share",
        "balance_ratio",
        "handedness_differs_from_stored",
        "motion_totals_differ_from_stored",
    }
)
_PREVIEW_FIELDS = frozenset(
    {
        "format",
        "name",
        "duration",
        "joints",
        "interpolation",
        "frames",
        "source_atom_hash",
        "parent_atom_hash",
    }
)
_PREVIEW_REQUIRED_FIELDS = _PREVIEW_FIELDS - {"parent_atom_hash"}
_CATALOG_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "ontology",
        "inputs",
        "source",
        "counts",
        "neutral",
        "entries",
        "by_function",
    }
)
_CATALOG_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "index",
        "atom_id",
        "version",
        "revision",
        "content_sha256",
        "function",
        "planner_category",
        "duration_us",
        "execution_side",
        "left_path_rad",
        "right_path_rad",
        "source_id",
        "source",
        "approved",
        "removed",
        "provenance",
        "motion",
    }
)
_CATALOG_SOURCE_FIELDS = frozenset({"catalog_id", "source_run_dir"})
_ENTRY_SOURCE_FIELDS = frozenset(
    {
        "dataset",
        "speaker",
        "session",
        "clip",
        "relative_path",
        "input_sha256",
        "frame_start",
        "frame_end",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "approval_event_id",
        "review_event_id",
        "review_decision",
        "variant",
        "atom_file",
        "atom_file_sha256",
        "preview_file",
        "preview_file_sha256",
        "report_file",
        "report_file_sha256",
    }
)


class SemanticCatalogValidationError(ValueError):
    """The formal review snapshot or compiled catalog is inconsistent."""


@dataclass(frozen=True, slots=True)
class SemanticCatalogRow:
    entry_id: str
    index: int
    atom_id: str
    version: str
    revision: int
    content_sha256: str
    function: str
    planner_category: str
    duration_us: int
    execution_side: str
    left_path_rad: float
    right_path_rad: float
    source_id: str
    approved: bool
    removed: bool
    source: Mapping[str, Any] = field(repr=False, compare=False)
    provenance: Mapping[str, Any] = field(repr=False, compare=False)
    motion: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def exact_tuple(self) -> tuple[str, str, int, str]:
        return (self.atom_id, self.version, self.revision, self.content_sha256)

    @property
    def entry_q(self) -> Mapping[str, float]:
        return self.motion["frames"][0]["q"]

    @property
    def exit_q(self) -> Mapping[str, float]:
        return self.motion["frames"][-1]["q"]


@dataclass(frozen=True, slots=True)
class SemanticCatalog:
    schema_version: str
    ontology_version: str
    ontology_sha256: str
    ontology: Mapping[str, Any]
    rows: tuple[SemanticCatalogRow, ...]
    by_function: Mapping[str, tuple[str, ...]]
    retained_count: int
    removed_count: int
    source_count: int
    neutral_motion_sha256: str
    neutral_duration_us: int
    neutral_q: Mapping[str, float]
    neutral_motion: Mapping[str, Any] = field(repr=False, compare=False)
    inputs: Mapping[str, str]
    content_sha256: str
    _canonical_bytes: bytes = field(repr=False, compare=False)


def _fail(label: str, detail: str) -> None:
    raise SemanticCatalogValidationError(f"{label}: {detail}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(label, f"invalid finite JSON in {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(label, "must be a JSON object")
    return value


def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        _fail(label, f"cannot stat {path}: {exc}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail(label, f"must be a regular non-symlink file: {path}")
    if before.st_size > max_bytes:
        _fail(label, f"exceeds {max_bytes} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        _fail(label, f"cannot safely open {path}: {exc}")
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _fail(label, "file changed while opening")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if len(raw) > max_bytes or len(raw) != after.st_size:
            _fail(label, "file changed while reading or exceeds size limit")
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(label, "file changed while reading")
        return raw
    finally:
        os.close(fd)


def _read_json(path: Path, label: str, *, max_bytes: int = _MAX_INPUT_BYTES) -> tuple[bytes, Mapping[str, Any]]:
    raw = _read_regular(path, label=label, max_bytes=max_bytes)
    return raw, _decode_json(raw, path, label)


def _read_jsonl(path: Path, label: str) -> tuple[bytes, list[Mapping[str, Any]]]:
    raw = _read_regular(path, label=label, max_bytes=_MAX_INPUT_BYTES)
    if raw and not raw.endswith(b"\n"):
        _fail(label, "must end with a newline")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            _fail(label, f"blank line at {line_number}")
        rows.append(_decode_json(line, path, f"{label}[{line_number}]") )
    return raw, rows


def _strict(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        _fail(label, f"missing field(s): {', '.join(missing)}")
    if unknown:
        _fail(label, f"unknown field(s): {', '.join(unknown)}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(label, "must be a non-empty string")
    return value


def _token(value: Any, label: str) -> str:
    result = _string(value, label)
    if _TOKEN.fullmatch(result) is None:
        _fail(label, "contains unsupported characters")
    return result


def _version(value: Any, label: str) -> str:
    result = _string(value, label)
    if result != "candidate" and _REPAIR_VERSION.fullmatch(result) is None:
        _fail(label, "must be candidate or a lowercase 32-character repair id")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label)
    if _SHA256.fullmatch(result) is None:
        _fail(label, "must be a lowercase SHA-256 digest")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(label, "must be a positive integer")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(label, "must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail(label, "must be finite and non-negative")
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail("catalog", f"cannot canonicalize finite JSON: {exc}")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _atom_hash(atom: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(atom))
    identity = payload.get("identity")
    if isinstance(identity, dict):
        identity.pop("content_hash", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _exact_tuple(row: Mapping[str, Any], label: str) -> tuple[str, str, int, str]:
    return (
        _token(row.get("atom_id"), f"{label}.atom_id"),
        _version(row.get("version"), f"{label}.version"),
        _positive_int(row.get("revision"), f"{label}.revision"),
        _sha256(row.get("content_sha256"), f"{label}.content_sha256"),
    )


def _safe_asset(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative or "\\" in relative:
        _fail(label, "unsafe relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(label, f"unsafe relative path: {relative}")
    current = root
    for index, part in enumerate(pure.parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            _fail(label, f"missing path component in {relative}: {exc}")
        if stat.S_ISLNK(info.st_mode):
            _fail(label, f"symlink is forbidden: {relative}")
        if index < len(pure.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            _fail(label, f"non-directory path component: {relative}")
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        _fail(label, f"path escapes source run: {relative}: {exc}")
    return current


def _motion_frames(
    motion: Mapping[str, Any],
    *,
    label: str,
    require_interpolation: bool,
) -> tuple[tuple[str, ...], list[dict[str, Any]], float]:
    required = {"format", "name", "duration", "joints", "frames"}
    optional = {"interpolation"}
    if require_interpolation:
        required |= {"interpolation", "source_atom_hash"}
        optional = {"parent_atom_hash"}
    missing = sorted(required - set(motion))
    unknown = sorted(set(motion) - required - optional)
    if missing:
        _fail(label, f"missing field(s): {', '.join(missing)}")
    if unknown:
        _fail(label, f"unknown field(s): {', '.join(unknown)}")
    if motion["format"] != MOTION_FORMAT:
        _fail(f"{label}.format", f"must equal {MOTION_FORMAT}")
    _string(motion["name"], f"{label}.name")
    if "interpolation" in motion:
        interpolation = _string(motion["interpolation"], f"{label}.interpolation")
        if require_interpolation and interpolation != "linear":
            _fail(f"{label}.interpolation", "must equal linear")
    duration = _nonnegative_number(motion["duration"], f"{label}.duration")
    joints_raw = motion["joints"]
    if (
        not isinstance(joints_raw, list)
        or not joints_raw
        or any(not isinstance(joint, str) or not joint for joint in joints_raw)
        or len(joints_raw) != len(set(joints_raw))
    ):
        _fail(f"{label}.joints", "must be a unique non-empty string array")
    joints = tuple(joints_raw)
    frames_raw = motion["frames"]
    if not isinstance(frames_raw, list) or len(frames_raw) < 2:
        _fail(f"{label}.frames", "must contain at least two frames")
    frames: list[dict[str, Any]] = []
    previous_t = -math.inf
    for index, raw_frame in enumerate(frames_raw):
        frame_label = f"{label}.frames[{index}]"
        if not isinstance(raw_frame, Mapping) or set(raw_frame) != {"t", "q"}:
            _fail(frame_label, "must contain exactly t and q")
        t = _nonnegative_number(raw_frame["t"], f"{frame_label}.t")
        if index == 0 and t != 0.0:
            _fail(f"{frame_label}.t", "first frame must start at zero")
        if t <= previous_t:
            _fail(f"{frame_label}.t", "must be strictly increasing")
        previous_t = t
        q_raw = raw_frame["q"]
        if not isinstance(q_raw, Mapping) or set(q_raw) != set(joints):
            _fail(f"{frame_label}.q", "must exactly match motion joints")
        q: dict[str, float] = {}
        for joint in joints:
            raw_value = q_raw[joint]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                _fail(f"{frame_label}.q.{joint}", "must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                _fail(f"{frame_label}.q.{joint}", "must be finite")
            q[joint] = value
        frames.append({"t": t, "q": q})
    if not math.isclose(frames[-1]["t"], duration, rel_tol=0.0, abs_tol=1e-9):
        _fail(f"{label}.duration", "must equal final frame time")
    return joints, frames, duration


def _validate_atom_trajectory(atom: Mapping[str, Any], exact: tuple[str, str, int, str], label: str) -> tuple[tuple[str, ...], list[dict[str, Any]], float]:
    if atom.get("schema_version") != ATOM_SCHEMA_VERSION:
        _fail(f"{label}.schema_version", f"must equal {ATOM_SCHEMA_VERSION}")
    identity = atom.get("identity")
    if not isinstance(identity, Mapping) or (
        identity.get("id"),
        identity.get("revision"),
        identity.get("content_hash"),
    ) != (exact[0], exact[2], exact[3]):
        _fail(label, "identity does not match exact tuple")
    if _atom_hash(atom) != exact[3]:
        _fail(label, "canonical content hash mismatch")
    control = atom.get("control")
    if not isinstance(control, Mapping):
        _fail(f"{label}.control", "must be an object")
    joints_raw = control.get("joints")
    trajectory_raw = control.get("trajectory")
    if (
        not isinstance(joints_raw, list)
        or not joints_raw
        or any(not isinstance(joint, str) or not joint for joint in joints_raw)
        or len(joints_raw) != len(set(joints_raw))
    ):
        _fail(f"{label}.control.joints", "must be unique non-empty strings")
    joints = tuple(joints_raw)
    if not isinstance(trajectory_raw, list) or len(trajectory_raw) < 2:
        _fail(f"{label}.control.trajectory", "must contain at least two samples")
    frames: list[dict[str, Any]] = []
    previous_t = -math.inf
    for index, sample in enumerate(trajectory_raw):
        sample_label = f"{label}.control.trajectory[{index}]"
        if not isinstance(sample, Mapping):
            _fail(sample_label, "must be an object")
        t = _nonnegative_number(sample.get("t"), f"{sample_label}.t")
        if index == 0 and t != 0.0:
            _fail(f"{sample_label}.t", "first sample must start at zero")
        if t <= previous_t:
            _fail(f"{sample_label}.t", "must be strictly increasing")
        previous_t = t
        q_raw = sample.get("q")
        if not isinstance(q_raw, Mapping) or set(q_raw) != set(joints):
            _fail(f"{sample_label}.q", "must exactly match control.joints")
        q: dict[str, float] = {}
        for joint in joints:
            raw_value = q_raw[joint]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                _fail(f"{sample_label}.q.{joint}", "must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                _fail(f"{sample_label}.q.{joint}", "must be finite")
            q[joint] = value
        frames.append({"t": t, "q": q})
    timing = atom.get("timing")
    if not isinstance(timing, Mapping):
        _fail(f"{label}.timing", "must be an object")
    duration = _nonnegative_number(
        timing.get("nominal_duration_s"), f"{label}.timing.nominal_duration_s"
    )
    if not math.isclose(frames[-1]["t"], duration, rel_tol=0.0, abs_tol=1e-9):
        _fail(f"{label}.timing.nominal_duration_s", "must equal final trajectory time")
    return joints, frames, duration


def _validate_fixed_base(atom: Mapping[str, Any], report: Mapping[str, Any], label: str) -> None:
    safety = atom.get("safety")
    metrics = safety.get("metrics") if isinstance(safety, Mapping) else None
    collision = safety.get("collision") if isinstance(safety, Mapping) else None
    violations = safety.get("violations") if isinstance(safety, Mapping) else None
    certified = safety.get("certified_backends") if isinstance(safety, Mapping) else None
    report_collision = report.get("collision")
    report_backends = report.get("certified_backends")
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("passed") is not True
        or metrics.get("base_mode") != "fixed"
        or metrics.get("kinematics_passed") is not True
        or metrics.get("position_limits_checked") is not True
        or metrics.get("collision_checked") is not True
        or not isinstance(collision, Mapping)
        or collision.get("checked") is not True
        or collision.get("passed") is not True
        or not isinstance(violations, list)
        or violations
        or not isinstance(certified, list)
        or MUJOCO_BACKEND not in certified
        or report.get("result") != "pass"
        or not isinstance(report_collision, Mapping)
        or report_collision.get("result") != "pass"
        or report.get("eligible_for_backend_certification") is not True
        or not isinstance(report_backends, list)
        or MUJOCO_BACKEND not in report_backends
    ):
        _fail(label, "is not an exact fixed-base PASS")


def _execution_metrics(joints: Sequence[str], frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = 0.0
    right = 0.0
    for previous, current in zip(frames, frames[1:]):
        for joint in joints:
            delta = abs(float(current["q"][joint]) - float(previous["q"][joint]))
            if joint.startswith("left_"):
                left += delta
            elif joint.startswith("right_"):
                right += delta
    maximum = max(left, right, 1e-12)
    if left < SIDE_NONE_MAX_FRACTION * maximum and right < SIDE_NONE_MAX_FRACTION * maximum:
        handedness = "none"
    elif left >= SIDE_BILATERAL_MIN_RATIO * right and right >= SIDE_BILATERAL_MIN_RATIO * left:
        handedness = "bilateral"
    else:
        handedness = "left" if left > right else "right"
    total = left + right
    return {
        "handedness": handedness,
        "left_path_rad": left,
        "right_path_rad": right,
        "left_share": left / total if total > 0.0 else 0.5,
        "balance_ratio": min(left, right) / max(left, right) if max(left, right) > 0.0 else 1.0,
    }


def _validate_human_review(review: Any, label: str, decisions: set[str]) -> Mapping[str, Any]:
    if not isinstance(review, Mapping):
        _fail(label, "must be an object")
    _strict(review, _HUMAN_REVIEW_FIELDS, label)
    decision = _string(review["decision"], f"{label}.decision")
    if decision not in decisions:
        _fail(f"{label}.decision", f"must be one of {sorted(decisions)}")
    _token(review["event_id"], f"{label}.event_id")
    _string(review["reviewed_at_utc"], f"{label}.reviewed_at_utc")
    _string(review["reviewer"], f"{label}.reviewer")
    for key in ("qwen_basis", "qwen_label", "review_note"):
        if not isinstance(review[key], str):
            _fail(f"{label}.{key}", "must be a string")
    return review


def _ontology_payload(ontology: SemanticOntology) -> dict[str, Any]:
    return {
        "schema_version": ontology.schema_version,
        "sha256": ontology.content_sha256,
        "label_scope": ontology.label_scope,
        "derived_from": ontology.derived_from,
        "excluded_functions": list(ontology.excluded_functions),
        "functions": [
            {
                "function": item.function,
                "planner_category": item.planner_category,
                "common_families": list(item.common_families),
                "definition_zh": item.definition_zh,
            }
            for item in ontology.functions
        ],
        "separation_rule": ontology.separation_rule,
    }


def _validate_preview(
    preview: Mapping[str, Any],
    *,
    exact: tuple[str, str, int, str],
    atom_joints: tuple[str, ...],
    atom_frames: list[dict[str, Any]],
    atom_duration: float,
    label: str,
) -> dict[str, Any]:
    missing = sorted(_PREVIEW_REQUIRED_FIELDS - set(preview))
    unknown = sorted(set(preview) - _PREVIEW_FIELDS)
    if missing:
        _fail(label, f"missing field(s): {', '.join(missing)}")
    if unknown:
        _fail(label, f"unknown field(s): {', '.join(unknown)}")
    joints, frames, duration = _motion_frames(
        preview, label=label, require_interpolation=True
    )
    if preview["source_atom_hash"] != exact[3]:
        _fail(f"{label}.source_atom_hash", "does not match exact Atom hash")
    parent = preview.get("parent_atom_hash")
    if parent is not None:
        _sha256(parent, f"{label}.parent_atom_hash")
    if joints != atom_joints or frames != atom_frames:
        _fail(label, "preview trajectory does not exactly match Atom q trajectory")
    if not math.isclose(duration, atom_duration, rel_tol=0.0, abs_tol=1e-9):
        _fail(f"{label}.duration", "does not match exact Atom duration")
    return {
        "format": MOTION_FORMAT,
        "name": preview["name"],
        "duration": duration,
        "joints": list(joints),
        "interpolation": preview["interpolation"],
        "frames": frames,
        "source_atom_hash": exact[3],
        "parent_atom_hash": parent,
    }


def _source_payload(source: Any, label: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(source, Mapping):
        _fail(label, "must be an object")
    required = (
        "dataset",
        "speaker",
        "session",
        "clip",
        "relative_path",
        "input_sha256",
        "frame_start",
        "frame_end",
    )
    for key in required:
        if key not in source:
            _fail(label, f"missing field: {key}")
    dataset = _string(source["dataset"], f"{label}.dataset")
    speaker = _string(source["speaker"], f"{label}.speaker")
    session = _string(source["session"], f"{label}.session")
    clip = _string(source["clip"], f"{label}.clip")
    relative_path = _string(source["relative_path"], f"{label}.relative_path")
    input_sha = _sha256(source["input_sha256"], f"{label}.input_sha256")
    start = source["frame_start"]
    end = source["frame_end"]
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        _fail(f"{label}.frame_start", "must be a non-negative integer")
    if isinstance(end, bool) or not isinstance(end, int) or end < start:
        _fail(f"{label}.frame_end", "must be an integer not before frame_start")
    payload = {
        "dataset": dataset,
        "speaker": speaker,
        "session": session,
        "clip": clip,
        "relative_path": relative_path,
        "input_sha256": input_sha,
        "frame_start": start,
        "frame_end": end,
    }
    source_id = f"{dataset}:{speaker}:s{session}:{clip}:f{start}-{end}:{input_sha[:12]}"
    return source_id, payload


def compile_semantic_catalog(
    *,
    ontology_path: str | Path,
    final_labels_path: str | Path,
    removed_actions_path: str | Path,
    approved_inventory_path: str | Path,
    exact_sidecar_path: str | Path,
    neutral_motion_path: str | Path,
    expected_retained: int | None = None,
    expected_removed: int | None = None,
) -> SemanticCatalog:
    """Compile one deterministic catalog after validating every exact binding."""

    try:
        ontology = load_ontology(ontology_path)
    except OntologyValidationError as exc:
        raise SemanticCatalogValidationError(f"ontology: {exc}") from exc
    final_raw, final_rows = _read_jsonl(Path(final_labels_path), "final_labels")
    removed_raw, removed_rows = _read_jsonl(Path(removed_actions_path), "removed_actions")
    inventory_raw, inventory = _read_json(Path(approved_inventory_path), "approved_inventory")
    side_raw, sidecar = _read_json(Path(exact_sidecar_path), "exact_sidecar")
    neutral_raw, neutral = _read_json(Path(neutral_motion_path), "neutral_motion")

    if not final_rows:
        _fail("final_labels", "must not be empty")
    if expected_retained is not None and len(final_rows) != expected_retained:
        _fail("final_labels", f"count {len(final_rows)} does not match expected {expected_retained}")
    if expected_removed is not None and len(removed_rows) != expected_removed:
        _fail("removed_actions", f"count {len(removed_rows)} does not match expected {expected_removed}")

    neutral_joints, neutral_frames, neutral_duration = _motion_frames(
        neutral, label="neutral_motion", require_interpolation=False
    )
    if neutral_frames[0]["q"] != neutral_frames[-1]["q"]:
        _fail("neutral_motion", "first and final q must be identical")
    neutral_q = neutral_frames[0]["q"]
    neutral_interpolation = neutral.get("interpolation", "linear")
    if neutral_interpolation != "linear":
        _fail("neutral_motion.interpolation", "must equal linear")
    neutral_duration_us = round(neutral_duration * 1_000_000)
    if neutral_duration_us < 1:
        _fail("neutral_motion.duration", "rounds to less than one microsecond")
    normalized_neutral_motion = {
        "format": MOTION_FORMAT,
        "name": neutral["name"],
        "duration": neutral_duration,
        "joints": list(neutral_joints),
        "interpolation": neutral_interpolation,
        "frames": neutral_frames,
    }

    labels_by_index: dict[int, Mapping[str, Any]] = {}
    review_bindings: dict[int, tuple[str, str, int, str]] = {}
    for row_number, row in enumerate(final_rows, start=1):
        label = f"final_labels[{row_number}]"
        _strict(row, _FINAL_FIELDS, label)
        if row["schema_version"] != FINAL_LABEL_SCHEMA_VERSION:
            _fail(f"{label}.schema_version", f"must equal {FINAL_LABEL_SCHEMA_VERSION}")
        index = _positive_int(row["index"], f"{label}.index")
        if index in review_bindings:
            _fail(label, f"duplicate review index {index}")
        exact = _exact_tuple(row, label)
        function = _string(row["label"], f"{label}.label")
        if function == EXCLUDED_FUNCTION:
            _fail(f"{label}.label", "present_explain is explicitly removed from dialogue_function_17_v1")
        if function not in ontology.by_name:
            _fail(f"{label}.label", f"is not in {ONTOLOGY_SCHEMA_VERSION}: {function}")
        _validate_human_review(
            row["human_review"], f"{label}.human_review", {"approve_qwen", "modify_label"}
        )
        labels_by_index[index] = row
        review_bindings[index] = exact
    for row_number, row in enumerate(removed_rows, start=1):
        label = f"removed_actions[{row_number}]"
        _strict(row, _REMOVED_FIELDS, label)
        if row["schema_version"] != REMOVED_SCHEMA_VERSION:
            _fail(f"{label}.schema_version", f"must equal {REMOVED_SCHEMA_VERSION}")
        index = _positive_int(row["index"], f"{label}.index")
        if index in review_bindings:
            _fail("review partition", f"index {index} appears in both retained and removed rows")
        _validate_human_review(row["human_review"], f"{label}.human_review", {"remove_atom"})
        review_bindings[index] = _exact_tuple(row, label)

    _strict(inventory, _INVENTORY_ROOT_FIELDS, "approved_inventory")
    if inventory["schema_version"] != INVENTORY_SCHEMA_VERSION:
        _fail("approved_inventory.schema_version", f"must equal {INVENTORY_SCHEMA_VERSION}")
    catalog_id = _token(inventory["catalog_id"], "approved_inventory.catalog_id")
    source_value = _string(inventory["source_run_dir"], "approved_inventory.source_run_dir")
    source_run = Path(source_value)
    if not source_run.is_absolute() or source_run.is_symlink():
        _fail("approved_inventory.source_run_dir", "must be an absolute non-symlink directory")
    try:
        source_run = source_run.resolve(strict=True)
    except OSError as exc:
        _fail("approved_inventory.source_run_dir", f"does not exist: {exc}")
    if not source_run.is_dir():
        _fail("approved_inventory.source_run_dir", "must be a directory")
    inventory_rows = inventory["atoms"]
    if not isinstance(inventory_rows, list) or not inventory_rows:
        _fail("approved_inventory.atoms", "must be a non-empty array")
    if inventory["approved_count"] != len(inventory_rows):
        _fail("approved_inventory.approved_count", "does not match atoms length")
    expected_indices = set(range(1, len(inventory_rows) + 1))
    if set(review_bindings) != expected_indices:
        missing = sorted(expected_indices - set(review_bindings))
        extra = sorted(set(review_bindings) - expected_indices)
        _fail("review partition", f"must exactly cover inventory; missing={missing}, extra={extra}")

    _strict(sidecar, _SIDE_ROOT_FIELDS, "exact_sidecar")
    if sidecar["schema_version"] != SIDE_SCHEMA_VERSION:
        _fail("exact_sidecar.schema_version", f"must equal {SIDE_SCHEMA_VERSION}")
    if sidecar["catalog_id"] != catalog_id:
        _fail("exact_sidecar.catalog_id", "does not match approved inventory")
    if sidecar["item_count"] != len(inventory_rows):
        _fail("exact_sidecar.item_count", "does not match approved inventory")
    if sidecar["approved_inventory_sha256"] != hashlib.sha256(inventory_raw).hexdigest():
        _fail("exact_sidecar.approved_inventory_sha256", "inventory hash mismatch")
    _sha256(sidecar["dataset_manifest_sha256"], "exact_sidecar.dataset_manifest_sha256")
    algorithm = sidecar["algorithm"]
    if not isinstance(algorithm, Mapping):
        _fail("exact_sidecar.algorithm", "must be an object")
    _strict(algorithm, _SIDE_ALGORITHM_FIELDS, "exact_sidecar.algorithm")
    if (
        algorithm["algorithm_id"] != SIDE_ALGORITHM
        or algorithm["unit"] != "aggregate_joint_radians"
        or algorithm["coordinate_frame"] != "robot_anatomical_joint_names"
        or algorithm["bilateral_min_ratio"] != SIDE_BILATERAL_MIN_RATIO
        or algorithm["none_max_fraction"] != SIDE_NONE_MAX_FRACTION
    ):
        _fail("exact_sidecar.algorithm", "unsupported execution-side algorithm contract")
    side_rows = sidecar["items"]
    if not isinstance(side_rows, list) or len(side_rows) != len(inventory_rows):
        _fail("exact_sidecar.items", "must exactly match approved inventory length")

    output_entries: list[dict[str, Any]] = []
    seen_exact: set[tuple[str, str, int, str]] = set()
    seen_atom_ids: set[str] = set()
    for offset, (inventory_row, side_row) in enumerate(zip(inventory_rows, side_rows), start=1):
        inventory_label = f"approved_inventory.atoms[{offset - 1}]"
        side_label = f"exact_sidecar.items[{offset - 1}]"
        if not isinstance(inventory_row, Mapping):
            _fail(inventory_label, "must be an object")
        if not isinstance(side_row, Mapping):
            _fail(side_label, "must be an object")
        _strict(inventory_row, _INVENTORY_ROW_FIELDS, inventory_label)
        _strict(side_row, _SIDE_ROW_FIELDS, side_label)
        if inventory_row["index"] != offset or side_row["index"] != offset:
            _fail("catalog inputs", "inventory and sidecar indices must be contiguous and ordered")
        exact = _exact_tuple(inventory_row, inventory_label)
        if exact in seen_exact or exact[0] in seen_atom_ids:
            _fail(inventory_label, "duplicate exact tuple or atom_id")
        seen_exact.add(exact)
        seen_atom_ids.add(exact[0])
        if review_bindings[offset] != exact:
            _fail("review partition", f"exact tuple mismatch at index {offset}")
        if _exact_tuple(side_row, side_label) != exact:
            _fail(side_label, "exact tuple does not match approved inventory")
        if inventory_row["approved"] is not True:
            _fail(f"{inventory_label}.approved", "must be literal true")
        if inventory_row["fixed_base_pass"] is not True:
            _fail(f"{inventory_label}.fixed_base_pass", "must be literal true")
        approval_event = _token(
            inventory_row["approval_event_id"], f"{inventory_label}.approval_event_id"
        )
        variant = _string(inventory_row["variant"], f"{inventory_label}.variant")
        expected_variant = "raw" if exact[1] == "candidate" else "repaired"
        if (
            variant != expected_variant
            or (variant == "raw" and exact[2] != 1)
            or (variant == "repaired" and exact[2] < 2)
        ):
            _fail(f"{inventory_label}.variant", "does not match exact version/revision")
        if (
            side_row["approval_event_id"] != approval_event
            or side_row["variant"] != variant
            or side_row["atom_file_sha256"] != inventory_row["atom_file_sha256"]
        ):
            _fail(side_label, "provenance does not match approved inventory")

        assets: dict[str, tuple[Path, bytes, Mapping[str, Any]]] = {}
        for kind in ("atom", "preview", "report"):
            relative = inventory_row[f"{kind}_file"]
            path = _safe_asset(source_run, relative, f"{inventory_label}.{kind}_file")
            raw = _read_regular(path, label=f"{inventory_label}.{kind}_file", max_bytes=_MAX_ASSET_BYTES)
            expected_sha = _sha256(
                inventory_row[f"{kind}_file_sha256"], f"{inventory_label}.{kind}_file_sha256"
            )
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                _fail(f"{inventory_label}.{kind}_file_sha256", "asset hash mismatch")
            assets[kind] = (path, raw, _decode_json(raw, path, f"{inventory_label}.{kind}_file"))
        atom = assets["atom"][2]
        report = assets["report"][2]
        atom_joints, atom_frames, atom_duration = _validate_atom_trajectory(
            atom, exact, inventory_label
        )
        if atom_joints != neutral_joints:
            _fail(inventory_label, "Atom joints do not exactly match canonical neutral joints")
        _validate_fixed_base(atom, report, inventory_label)
        duration = _nonnegative_number(
            inventory_row["duration_s"], f"{inventory_label}.duration_s"
        )
        if not math.isclose(duration, atom_duration, rel_tol=0.0, abs_tol=1e-9):
            _fail(f"{inventory_label}.duration_s", "does not match exact Atom")
        preview = _validate_preview(
            assets["preview"][2],
            exact=exact,
            atom_joints=atom_joints,
            atom_frames=atom_frames,
            atom_duration=atom_duration,
            label=f"{inventory_label}.preview",
        )
        computed_side = _execution_metrics(atom_joints, atom_frames)
        for key in ("left_path_rad", "right_path_rad", "left_share", "balance_ratio"):
            observed = _nonnegative_number(side_row[key], f"{side_label}.{key}")
            if not math.isclose(observed, computed_side[key], rel_tol=0.0, abs_tol=1e-10):
                _fail(f"{side_label}.{key}", "does not match exact Atom trajectory")
        if side_row["exact_trajectory_handedness"] != computed_side["handedness"]:
            _fail(f"{side_label}.exact_trajectory_handedness", "does not match exact Atom trajectory")
        stored_handedness = side_row["stored_handedness"]
        atom_semantics = atom.get("semantics")
        if (
            not isinstance(atom_semantics, Mapping)
            or atom_semantics.get("handedness") != stored_handedness
        ):
            _fail(f"{side_label}.stored_handedness", "does not match Atom semantics")
        stored_left = _nonnegative_number(
            side_row["stored_left_path_rad"], f"{side_label}.stored_left_path_rad"
        )
        stored_right = _nonnegative_number(
            side_row["stored_right_path_rad"], f"{side_label}.stored_right_path_rad"
        )
        expected_handedness_differs = stored_handedness != computed_side["handedness"]
        expected_motion_differs = (
            abs(stored_left - computed_side["left_path_rad"]) > 1e-10
            or abs(stored_right - computed_side["right_path_rad"]) > 1e-10
        )
        if side_row["handedness_differs_from_stored"] is not expected_handedness_differs:
            _fail(f"{side_label}.handedness_differs_from_stored", "is inconsistent")
        if side_row["motion_totals_differ_from_stored"] is not expected_motion_differs:
            _fail(f"{side_label}.motion_totals_differ_from_stored", "is inconsistent")

        if offset not in labels_by_index:
            continue
        label_row = labels_by_index[offset]
        function = label_row["label"]
        function_def = ontology.by_name[function]
        source_id, source_payload = _source_payload(
            inventory_row["source"], f"{inventory_label}.source"
        )
        duration_us = round(atom_duration * 1_000_000)
        if duration_us < 1:
            _fail(f"{inventory_label}.duration_s", "rounds to less than one microsecond")
        review = label_row["human_review"]
        entry_identity = {
            "ontology_version": ontology.schema_version,
            "function": function,
            "atom_id": exact[0],
            "version": exact[1],
            "revision": exact[2],
            "content_sha256": exact[3],
        }
        entry_id = hashlib.sha256(_canonical_bytes(entry_identity)).hexdigest()
        output_entries.append(
            {
                "entry_id": entry_id,
                "index": offset,
                "atom_id": exact[0],
                "version": exact[1],
                "revision": exact[2],
                "content_sha256": exact[3],
                "function": function,
                "planner_category": function_def.planner_category,
                "duration_us": duration_us,
                "execution_side": computed_side["handedness"],
                "left_path_rad": computed_side["left_path_rad"],
                "right_path_rad": computed_side["right_path_rad"],
                "source_id": source_id,
                "source": source_payload,
                "approved": True,
                "removed": False,
                "provenance": {
                    "approval_event_id": approval_event,
                    "review_event_id": review["event_id"],
                    "review_decision": review["decision"],
                    "variant": variant,
                    "atom_file": inventory_row["atom_file"],
                    "atom_file_sha256": inventory_row["atom_file_sha256"],
                    "preview_file": inventory_row["preview_file"],
                    "preview_file_sha256": inventory_row["preview_file_sha256"],
                    "report_file": inventory_row["report_file"],
                    "report_file_sha256": inventory_row["report_file_sha256"],
                },
                "motion": preview,
            }
        )

    by_function: dict[str, list[str]] = {
        function: [] for function in ontology.function_names
    }
    for entry in output_entries:
        by_function[entry["function"]].append(entry["entry_id"])
    payload = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "ontology": _ontology_payload(ontology),
        "inputs": {
            "final_labels_sha256": hashlib.sha256(final_raw).hexdigest(),
            "removed_actions_sha256": hashlib.sha256(removed_raw).hexdigest(),
            "approved_inventory_sha256": hashlib.sha256(inventory_raw).hexdigest(),
            "exact_sidecar_sha256": hashlib.sha256(side_raw).hexdigest(),
            "neutral_motion_sha256": hashlib.sha256(neutral_raw).hexdigest(),
        },
        "source": {
            "catalog_id": catalog_id,
            "source_run_dir": str(source_run),
        },
        "counts": {
            "source": len(inventory_rows),
            "retained": len(output_entries),
            "removed": len(removed_rows),
        },
        "neutral": {
            "motion_sha256": hashlib.sha256(neutral_raw).hexdigest(),
            "duration_us": neutral_duration_us,
            "joints": list(neutral_joints),
            "q": neutral_q,
            "motion": normalized_neutral_motion,
        },
        "entries": output_entries,
        "by_function": by_function,
    }
    raw = _canonical_bytes(payload) + b"\n"
    return _catalog_from_payload(payload, raw)


def _catalog_from_payload(payload: Mapping[str, Any], raw: bytes) -> SemanticCatalog:
    _strict(payload, _CATALOG_ROOT_FIELDS, "catalog")
    if payload["schema_version"] != CATALOG_SCHEMA_VERSION:
        _fail("catalog.schema_version", f"must equal {CATALOG_SCHEMA_VERSION}")
    ontology = payload["ontology"]
    if not isinstance(ontology, Mapping):
        _fail("catalog.ontology", "must be an object")
    ontology_fields = frozenset(
        {
            "schema_version",
            "sha256",
            "label_scope",
            "derived_from",
            "excluded_functions",
            "functions",
            "separation_rule",
        }
    )
    _strict(ontology, ontology_fields, "catalog.ontology")
    if ontology["schema_version"] != ONTOLOGY_SCHEMA_VERSION:
        _fail("catalog.ontology.schema_version", f"must equal {ONTOLOGY_SCHEMA_VERSION}")
    if ontology["label_scope"] != "planner_function":
        _fail("catalog.ontology.label_scope", "must equal planner_function")
    _string(ontology["derived_from"], "catalog.ontology.derived_from")
    _string(ontology["separation_rule"], "catalog.ontology.separation_rule")
    ontology_sha = _sha256(ontology["sha256"], "catalog.ontology.sha256")
    if ontology["excluded_functions"] != [EXCLUDED_FUNCTION]:
        _fail("catalog.ontology.excluded_functions", "must explicitly contain only present_explain")
    function_rows = ontology["functions"]
    if not isinstance(function_rows, list) or len(function_rows) != 17:
        _fail("catalog.ontology.functions", "must contain exactly 17 functions")
    functions: list[str] = []
    categories: dict[str, str] = {}
    for index, row in enumerate(function_rows):
        label = f"catalog.ontology.functions[{index}]"
        if not isinstance(row, Mapping) or set(row) != {
            "function",
            "planner_category",
            "common_families",
            "definition_zh",
        }:
            _fail(label, "has invalid fields")
        function = _string(row["function"], f"{label}.function")
        if function == EXCLUDED_FUNCTION or function in functions:
            _fail(label, "contains an excluded or duplicate function")
        functions.append(function)
        categories[function] = _string(row["planner_category"], f"{label}.planner_category")
        families = row["common_families"]
        if (
            not isinstance(families, list)
            or not families
            or any(not isinstance(family, str) or not family for family in families)
            or len(families) != len(set(families))
        ):
            _fail(f"{label}.common_families", "must be a unique non-empty string array")
        _string(row["definition_zh"], f"{label}.definition_zh")

    inputs = payload["inputs"]
    expected_inputs = {
        "final_labels_sha256",
        "removed_actions_sha256",
        "approved_inventory_sha256",
        "exact_sidecar_sha256",
        "neutral_motion_sha256",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        _fail("catalog.inputs", "has invalid fields")
    for key, value in inputs.items():
        _sha256(value, f"catalog.inputs.{key}")
    counts = payload["counts"]
    if not isinstance(counts, Mapping) or set(counts) != {"source", "retained", "removed"}:
        _fail("catalog.counts", "has invalid fields")
    for key in ("source", "retained", "removed"):
        if isinstance(counts[key], bool) or not isinstance(counts[key], int) or counts[key] < 0:
            _fail(f"catalog.counts.{key}", "must be a non-negative integer")
    if counts["source"] != counts["retained"] + counts["removed"]:
        _fail("catalog.counts", "source must equal retained plus removed")
    source_root = payload["source"]
    if not isinstance(source_root, Mapping):
        _fail("catalog.source", "must be an object")
    _strict(source_root, _CATALOG_SOURCE_FIELDS, "catalog.source")
    _token(source_root["catalog_id"], "catalog.source.catalog_id")
    source_run_value = _string(source_root["source_run_dir"], "catalog.source.source_run_dir")
    if not Path(source_run_value).is_absolute():
        _fail("catalog.source.source_run_dir", "must be an absolute provenance path")
    neutral = payload["neutral"]
    if not isinstance(neutral, Mapping) or set(neutral) != {
        "motion_sha256",
        "duration_us",
        "joints",
        "q",
        "motion",
    }:
        _fail("catalog.neutral", "has invalid fields")
    neutral_sha = _sha256(neutral["motion_sha256"], "catalog.neutral.motion_sha256")
    if neutral_sha != inputs["neutral_motion_sha256"]:
        _fail("catalog.neutral.motion_sha256", "does not match input hash")
    neutral_joints = neutral["joints"]
    neutral_q_raw = neutral["q"]
    if (
        not isinstance(neutral_joints, list)
        or not neutral_joints
        or len(neutral_joints) != len(set(neutral_joints))
        or not isinstance(neutral_q_raw, Mapping)
        or set(neutral_q_raw) != set(neutral_joints)
    ):
        _fail("catalog.neutral", "joints/q binding is invalid")
    neutral_q = {
        joint: _finite_number(neutral_q_raw[joint], f"catalog.neutral.q.{joint}")
        for joint in neutral_joints
    }
    neutral_duration_us = _positive_int(
        neutral["duration_us"], "catalog.neutral.duration_us"
    )
    neutral_motion_raw = neutral["motion"]
    if not isinstance(neutral_motion_raw, Mapping):
        _fail("catalog.neutral.motion", "must be an object")
    neutral_motion_joints, neutral_motion_frames, neutral_motion_duration = _motion_frames(
        neutral_motion_raw,
        label="catalog.neutral.motion",
        require_interpolation=False,
    )
    if neutral_motion_raw.get("interpolation") != "linear":
        _fail("catalog.neutral.motion.interpolation", "must equal linear")
    if (
        neutral_motion_joints != tuple(neutral_joints)
        or neutral_motion_frames[0]["q"] != neutral_q
        or neutral_motion_frames[-1]["q"] != neutral_q
        or round(neutral_motion_duration * 1_000_000) != neutral_duration_us
    ):
        _fail("catalog.neutral.motion", "does not match canonical neutral q/duration")

    entries = payload["entries"]
    if not isinstance(entries, list) or len(entries) != counts["retained"]:
        _fail("catalog.entries", "length does not match retained count")
    rows: list[SemanticCatalogRow] = []
    seen_entry_ids: set[str] = set()
    seen_exact: set[tuple[str, str, int, str]] = set()
    seen_indices: set[int] = set()
    previous_index = 0
    for offset, entry in enumerate(entries):
        label = f"catalog.entries[{offset}]"
        if not isinstance(entry, Mapping):
            _fail(label, "must be an object")
        _strict(entry, _CATALOG_ENTRY_FIELDS, label)
        entry_id = _sha256(entry["entry_id"], f"{label}.entry_id")
        exact = _exact_tuple(entry, label)
        function = _string(entry["function"], f"{label}.function")
        if function not in categories or entry["planner_category"] != categories[function]:
            _fail(label, "function/category is not in embedded ontology")
        expected_entry_id = hashlib.sha256(
            _canonical_bytes(
                {
                    "ontology_version": ONTOLOGY_SCHEMA_VERSION,
                    "function": function,
                    "atom_id": exact[0],
                    "version": exact[1],
                    "revision": exact[2],
                    "content_sha256": exact[3],
                }
            )
        ).hexdigest()
        if entry_id != expected_entry_id:
            _fail(f"{label}.entry_id", "does not match exact tuple and function")
        if entry_id in seen_entry_ids or exact in seen_exact:
            _fail(label, "duplicate entry_id or exact tuple")
        seen_entry_ids.add(entry_id)
        seen_exact.add(exact)
        if entry["approved"] is not True or entry["removed"] is not False:
            _fail(label, "planner rows must be approved=true and removed=false")
        duration_us = _positive_int(entry["duration_us"], f"{label}.duration_us")
        side = entry["execution_side"]
        if side not in {"left", "right", "bilateral", "none"}:
            _fail(f"{label}.execution_side", "has an unsupported value")
        left = _nonnegative_number(entry["left_path_rad"], f"{label}.left_path_rad")
        right = _nonnegative_number(entry["right_path_rad"], f"{label}.right_path_rad")
        source_id = _string(entry["source_id"], f"{label}.source_id")
        if not isinstance(entry["source"], Mapping) or not isinstance(entry["provenance"], Mapping):
            _fail(label, "source and provenance must be objects")
        source_id_expected, normalized_source = _source_payload(
            entry["source"], f"{label}.source"
        )
        _strict(entry["source"], _ENTRY_SOURCE_FIELDS, f"{label}.source")
        if source_id != source_id_expected or dict(entry["source"]) != normalized_source:
            _fail(f"{label}.source_id", "does not match normalized source provenance")
        provenance = entry["provenance"]
        _strict(provenance, _PROVENANCE_FIELDS, f"{label}.provenance")
        _token(provenance["approval_event_id"], f"{label}.provenance.approval_event_id")
        _token(provenance["review_event_id"], f"{label}.provenance.review_event_id")
        if provenance["review_decision"] not in {"approve_qwen", "modify_label"}:
            _fail(f"{label}.provenance.review_decision", "has unsupported value")
        if provenance["variant"] not in {"raw", "repaired"}:
            _fail(f"{label}.provenance.variant", "has unsupported value")
        for kind in ("atom", "preview", "report"):
            relative = provenance[f"{kind}_file"]
            if (
                not isinstance(relative, str)
                or not relative
                or PurePosixPath(relative).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            ):
                _fail(f"{label}.provenance.{kind}_file", "must be a safe relative POSIX path")
            _sha256(
                provenance[f"{kind}_file_sha256"],
                f"{label}.provenance.{kind}_file_sha256",
            )
        motion = entry["motion"]
        if not isinstance(motion, Mapping):
            _fail(f"{label}.motion", "must be an object")
        motion_joints, motion_frames, motion_duration = _motion_frames(
            motion, label=f"{label}.motion", require_interpolation=True
        )
        if motion_joints != tuple(neutral_joints):
            _fail(f"{label}.motion.joints", "must match canonical neutral joints")
        if motion["source_atom_hash"] != exact[3]:
            _fail(f"{label}.motion.source_atom_hash", "does not match exact tuple")
        parent_hash = motion.get("parent_atom_hash")
        if parent_hash is not None:
            _sha256(parent_hash, f"{label}.motion.parent_atom_hash")
        if round(motion_duration * 1_000_000) != duration_us:
            _fail(f"{label}.duration_us", "does not match inline motion")
        metrics = _execution_metrics(motion_joints, motion_frames)
        if (
            metrics["handedness"] != side
            or not math.isclose(metrics["left_path_rad"], left, rel_tol=0.0, abs_tol=1e-10)
            or not math.isclose(metrics["right_path_rad"], right, rel_tol=0.0, abs_tol=1e-10)
        ):
            _fail(label, "execution side does not match inline motion")
        index = _positive_int(entry["index"], f"{label}.index")
        if index in seen_indices or index <= previous_index:
            _fail(f"{label}.index", "must be unique and strictly increasing")
        seen_indices.add(index)
        previous_index = index
        rows.append(
            SemanticCatalogRow(
                entry_id=entry_id,
                index=index,
                atom_id=exact[0],
                version=exact[1],
                revision=exact[2],
                content_sha256=exact[3],
                function=function,
                planner_category=categories[function],
                duration_us=duration_us,
                execution_side=side,
                left_path_rad=left,
                right_path_rad=right,
                source_id=source_id,
                approved=True,
                removed=False,
                source=_deep_freeze(entry["source"]),
                provenance=_deep_freeze(entry["provenance"]),
                motion=_deep_freeze(entry["motion"]),
            )
        )
    by_function_raw = payload["by_function"]
    if not isinstance(by_function_raw, Mapping) or set(by_function_raw) != set(functions):
        _fail("catalog.by_function", "must contain every ontology function exactly once")
    rebuilt: dict[str, list[str]] = {function: [] for function in functions}
    for row in rows:
        rebuilt[row.function].append(row.entry_id)
    by_function: dict[str, tuple[str, ...]] = {}
    for function in functions:
        values = by_function_raw[function]
        if not isinstance(values, list) or values != rebuilt[function]:
            _fail(f"catalog.by_function.{function}", "does not match catalog entries")
        by_function[function] = tuple(values)
    content_sha = hashlib.sha256(raw).hexdigest()
    return SemanticCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        ontology_version=ONTOLOGY_SCHEMA_VERSION,
        ontology_sha256=ontology_sha,
        ontology=_deep_freeze(ontology),
        rows=tuple(rows),
        by_function=MappingProxyType(by_function),
        retained_count=counts["retained"],
        removed_count=counts["removed"],
        source_count=counts["source"],
        neutral_motion_sha256=neutral_sha,
        neutral_duration_us=neutral_duration_us,
        neutral_q=MappingProxyType(neutral_q),
        neutral_motion=_deep_freeze(neutral_motion_raw),
        inputs=MappingProxyType(dict(inputs)),
        content_sha256=content_sha,
        _canonical_bytes=raw,
    )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(label, "must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(label, "must be finite")
    return result


def publish_semantic_catalog(catalog: SemanticCatalog, output_dir: str | Path) -> Path:
    """Atomically publish one immutable file named by its exact byte SHA-256."""

    output = Path(output_dir)
    if output.exists() and output.is_symlink():
        _fail("output_dir", "symlink is forbidden")
    try:
        output.mkdir(parents=True, exist_ok=True, mode=0o750)
        root = output.resolve(strict=True)
    except OSError as exc:
        _fail("output_dir", f"cannot create or resolve {output}: {exc}")
    if not root.is_dir():
        _fail("output_dir", "must be a directory")
    expected_sha = hashlib.sha256(catalog._canonical_bytes).hexdigest()
    if expected_sha != catalog.content_sha256:
        _fail("catalog", "in-memory canonical bytes no longer match content SHA")
    target = root / f"{CATALOG_SCHEMA_VERSION}.{expected_sha}.json"
    if target.exists() or target.is_symlink():
        existing = _read_regular(target, label="published catalog", max_bytes=_MAX_CATALOG_BYTES)
        if existing != catalog._canonical_bytes:
            _fail("published catalog", "content-addressed path exists with different bytes")
        return target
    fd, temporary_name = tempfile.mkstemp(prefix=".semantic_catalog.tmp.", dir=root)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o444)
        view = memoryview(catalog._canonical_bytes)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                _fail("published catalog", "short write")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _read_regular(target, label="published catalog", max_bytes=_MAX_CATALOG_BYTES)
            if existing != catalog._canonical_bytes:
                _fail("published catalog", "concurrent content mismatch")
        temporary.unlink(missing_ok=True)
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def load_semantic_catalog(path: str | Path) -> SemanticCatalog:
    """Load an immutable, self-contained planner catalog and verify its address."""

    catalog_path = Path(path)
    raw = _read_regular(catalog_path, label="catalog", max_bytes=_MAX_CATALOG_BYTES)
    content_sha = hashlib.sha256(raw).hexdigest()
    expected_name = f"{CATALOG_SCHEMA_VERSION}.{content_sha}.json"
    if catalog_path.name != expected_name:
        _fail("catalog", f"filename must be the content address {expected_name}")
    payload = _decode_json(raw, catalog_path, "catalog")
    if _canonical_bytes(payload) + b"\n" != raw:
        _fail("catalog", "bytes must be canonical JSON with one trailing newline")
    return _catalog_from_payload(payload, raw)


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "SemanticCatalog",
    "SemanticCatalogRow",
    "SemanticCatalogValidationError",
    "compile_semantic_catalog",
    "load_semantic_catalog",
    "publish_semantic_catalog",
]
