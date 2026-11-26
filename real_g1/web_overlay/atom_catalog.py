"""Fail-closed loader for the hand-seeded Approved Atom catalog.

The catalog stores identity tokens only.  Motion data is always loaded from the
canonical Atom JSON in the frozen source run; preview PNG/video artifacts are
never considered executable trajectories.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CATALOG_SCHEMA_VERSION = "atom_catalog_v1"
ATOM_SCHEMA_VERSION = "robot.gesture_atom.v1"
MUJOCO_BACKEND = "mujoco-g1-29dof"

_ROOT_FIELDS = frozenset({"schema_version", "source_run_dir", "atoms"})
_ATOM_FIELDS = frozenset(
    {
        "atom_id",
        "version",
        "revision",
        "content_sha256",
        "approved",
        "primary_category",
        "tags",
        "label_source",
    }
)
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPAIR_VERSION = re.compile(r"^[0-9a-f]{32}$")


class CatalogValidationError(ValueError):
    """The catalog or one of its exact source-run bindings is invalid."""


@dataclass(frozen=True, slots=True)
class CatalogAtom:
    atom_id: str
    version: str
    revision: int
    content_sha256: str
    approved: bool
    primary_category: str
    tags: tuple[str, ...]
    label_source: str
    atom_path: Path
    report_path: Path
    approval_event_id: str
    atom_payload: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def first_q(self) -> dict[str, float]:
        trajectory = self.atom_payload["control"]["trajectory"]
        return dict(trajectory[0]["q"])


@dataclass(frozen=True, slots=True)
class AtomCatalog:
    schema_version: str
    source_run_dir: Path
    atoms: tuple[CatalogAtom, ...]
    annotations_sha256: str

    def approved_atoms(self) -> tuple[CatalogAtom, ...]:
        """Return only entries bound to literal ``approved=true`` evidence."""

        return tuple(atom for atom in self.atoms if atom.approved is True)


def _fail(label: str, detail: str) -> None:
    raise CatalogValidationError(f"{label}: {detail}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _decode_json(raw: bytes, path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(label, f"cannot read valid finite JSON from {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail(label, "must be a JSON object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(label, f"cannot read valid finite JSON from {path}: {exc}")
    return _decode_json(raw, path, label)


def _strict_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
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
    if _SAFE_TOKEN.fullmatch(result) is None:
        _fail(label, "contains unsupported characters")
    return result


def _version(value: Any, label: str) -> str:
    result = _string(value, label)
    if result != "candidate" and _REPAIR_VERSION.fullmatch(result) is None:
        _fail(label, "must equal candidate or a 32-character lowercase repair id")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label)
    if _LOWER_SHA256.fullmatch(result) is None:
        _fail(label, "must be a lowercase SHA-256 digest")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(label, "must be a positive integer")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        body = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail("Atom", f"cannot canonicalize finite JSON: {exc}")
    return hashlib.sha256(body).hexdigest()


def _atom_hash(atom: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(atom))
    identity = payload.get("identity")
    if isinstance(identity, dict):
        identity.pop("content_hash", None)
    return _canonical_sha256(payload)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _safe_evidence_bytes(
    run_root: Path, relative: str, label: str
) -> tuple[Path, bytes]:
    """Read one regular run artifact without following any symlink component."""

    if not isinstance(relative, str) or not relative:
        _fail(label, "unsafe evidence path: must be a non-empty relative path")
    relative_path = Path(relative)
    parts = relative_path.parts
    if relative_path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        _fail(label, "unsafe evidence path: must be a canonical relative path")
    candidate = run_root.joinpath(*parts)
    try:
        canonical = candidate.resolve(strict=True)
        canonical.relative_to(run_root)
    except FileNotFoundError:
        _fail(label, f"unsafe evidence path does not exist: {candidate}")
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(label, f"unsafe evidence path cannot be resolved within source_run_dir: {exc}")
    if canonical != candidate:
        _fail(label, "unsafe evidence path contains a symlink component")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not nofollow or not directory:
        _fail(label, "unsafe evidence path: platform lacks no-follow directory opens")

    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        root_fd = os.open(run_root, os.O_RDONLY | directory | cloexec)
        directory_fds.append(root_fd)
        parent_fd = root_fd
        for component in parts[:-1]:
            parent_fd = os.open(
                component,
                os.O_RDONLY | directory | nofollow | cloexec,
                dir_fd=parent_fd,
            )
            directory_fds.append(parent_fd)
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | nofollow | cloexec,
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            _fail(label, "unsafe evidence path is not a regular file")
        with os.fdopen(file_fd, "rb", closefd=True) as stream:
            file_fd = None
            raw = stream.read()
        current = candidate.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
        ):
            _fail(label, "unsafe evidence path changed during read")
    except CatalogValidationError:
        raise
    except OSError as exc:
        _fail(label, f"unsafe evidence path cannot be opened without symlinks: {exc}")
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)
    return candidate, raw


def _read_evidence_json(
    run_root: Path, relative: str, label: str
) -> tuple[Path, bytes, Mapping[str, Any]]:
    path, raw = _safe_evidence_bytes(run_root, relative, label)
    return path, raw, _decode_json(raw, path, label)


def _read_events(raw: bytes, path: Path) -> tuple[tuple[Mapping[str, Any], ...], str]:
    events: list[Mapping[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        _fail("annotations", f"is not UTF-8: {exc}")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            _fail("annotations", f"line {line_number} is invalid JSON: {exc}")
        if not isinstance(event, Mapping):
            _fail("annotations", f"line {line_number} must be an object")
        events.append(event)
    return tuple(events), hashlib.sha256(raw).hexdigest()


def _expected_atom_relative(
    *,
    atom_id: str,
    version: str,
    revision: int,
    content_hash: str,
) -> str:
    if version == "candidate":
        relative = f"candidates/{atom_id}.json"
    else:
        relative = (
            f"curation/repaired/{atom_id}/"
            f"r{revision}.{content_hash[:12]}.{version}/atom.json"
        )
    return relative


def _exact_repair_event(
    events: Sequence[Mapping[str, Any]],
    *,
    atom_id: str,
    version: str,
    revision: int,
    content_hash: str,
    atom_relative: str,
) -> Mapping[str, Any]:
    matches = [
        event
        for event in events
        if event.get("event_type") == "repair"
        and event.get("event_id") == version
        and event.get("atom_id") == atom_id
    ]
    if len(matches) != 1:
        _fail(f"Atom {atom_id}", "must have exactly one exact repair event")
    event = matches[0]
    auto_repair = event.get("auto_repair")
    if (
        event.get("revision") != revision
        or event.get("content_hash") != content_hash
        or event.get("atom") != atom_relative
        or not isinstance(auto_repair, Mapping)
        or auto_repair.get("status") != "repaired"
    ):
        _fail(f"Atom {atom_id}", "repair event does not match catalog identity")
    return event


def _approval_event(
    events: Sequence[Mapping[str, Any]],
    *,
    atom_id: str,
    version: str,
    revision: int,
    content_hash: str,
) -> Mapping[str, Any]:
    latest: Mapping[str, Any] | None = None
    for event in events:
        if (
            event.get("event_type") == "annotation"
            and event.get("atom_id") == atom_id
            and event.get("reviewed_version") == version
            and event.get("reviewed_revision") == revision
            and event.get("reviewed_content_hash") == content_hash
            and event.get("decision") is not None
        ):
            latest = event
    if latest is None or latest.get("decision") != "approve":
        _fail(f"Atom {atom_id}", "exact version is not currently approved")
    event_id = latest.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        _fail(f"Atom {atom_id}", "approval event has no event_id")
    return latest


def _manifest_record(
    manifest: Mapping[str, Any], atom_id: str
) -> Mapping[str, Any]:
    records = manifest.get("atoms")
    if not isinstance(records, list):
        _fail("run_manifest.atoms", "must be an array")
    matches = [record for record in records if isinstance(record, Mapping) and record.get("id") == atom_id]
    if len(matches) != 1:
        _fail(f"Atom {atom_id}", "must have exactly one run_manifest record")
    return matches[0]


def _validate_trajectory(atom: Mapping[str, Any], atom_id: str) -> dict[str, float]:
    control = atom.get("control")
    if not isinstance(control, Mapping):
        _fail(f"Atom {atom_id}.control", "must be an object")
    joints = control.get("joints")
    if (
        not isinstance(joints, list)
        or not joints
        or any(not isinstance(joint, str) or not joint for joint in joints)
        or len(set(joints)) != len(joints)
    ):
        _fail(f"Atom {atom_id}.control.joints", "must be a non-empty unique string array")
    trajectory = control.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        _fail(f"Atom {atom_id}.control.trajectory", "must contain at least two frames")

    previous_time: float | None = None
    expected_joints = set(joints)
    for index, frame in enumerate(trajectory):
        if not isinstance(frame, Mapping):
            _fail(f"Atom {atom_id}.control.trajectory[{index}]", "must be an object")
        raw_time = frame.get("t")
        if isinstance(raw_time, bool) or not isinstance(raw_time, (int, float)):
            _fail(f"Atom {atom_id}.control.trajectory[{index}].t", "must be finite numeric")
        time_value = float(raw_time)
        if not math.isfinite(time_value) or time_value < 0.0:
            _fail(f"Atom {atom_id}.control.trajectory[{index}].t", "must be finite and non-negative")
        if previous_time is None and time_value != 0.0:
            _fail(f"Atom {atom_id}.control.trajectory[0].t", "must equal 0.0")
        if previous_time is not None and time_value <= previous_time:
            _fail(f"Atom {atom_id}.control.trajectory[{index}].t", "must increase strictly")
        previous_time = time_value

        q = frame.get("q")
        if not isinstance(q, Mapping) or set(q) != expected_joints:
            _fail(f"Atom {atom_id}.control.trajectory[{index}].q", "must exactly match control.joints")
        for joint, raw_value in q.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
                _fail(
                    f"Atom {atom_id}.control.trajectory[{index}].q.{joint}",
                    "must be finite numeric",
                )
    return {joint: float(trajectory[0]["q"][joint]) for joint in joints}


def _validate_fixed_base_pass(
    atom: Mapping[str, Any], report: Mapping[str, Any], atom_id: str
) -> None:
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
        _fail(f"Atom {atom_id}", "is not an exact fixed-base PASS")


def _load_catalog_atom(
    raw: Mapping[str, Any],
    *,
    index: int,
    run_root: Path,
    events: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> CatalogAtom:
    label = f"atoms[{index}]"
    _strict_fields(raw, _ATOM_FIELDS, label)
    atom_id = _token(raw["atom_id"], f"{label}.atom_id")
    version = _version(raw["version"], f"{label}.version")
    revision = _positive_integer(raw["revision"], f"{label}.revision")
    content_hash = _sha256(raw["content_sha256"], f"{label}.content_sha256")
    if raw["approved"] is not True:
        _fail(f"{label}.approved", "must be literal true")
    category = _string(raw["primary_category"], f"{label}.primary_category")
    tags_value = raw["tags"]
    if (
        not isinstance(tags_value, list)
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags_value)
        or len(set(tags_value)) != len(tags_value)
    ):
        _fail(f"{label}.tags", "must be a unique string array")
    label_source = _string(raw["label_source"], f"{label}.label_source")
    if label_source != "manual_seed":
        _fail(f"{label}.label_source", "must equal manual_seed")

    atom_relative = _expected_atom_relative(
        atom_id=atom_id,
        version=version,
        revision=revision,
        content_hash=content_hash,
    )
    atom_path, _atom_raw, atom = _read_evidence_json(
        run_root, atom_relative, f"Atom {atom_id}"
    )
    if version == "candidate":
        record = _manifest_record(manifest, atom_id)
        expected_report = f"reports/{atom_id}.motion_report.json"
        if (
            revision != 1
            or record.get("content_hash") != content_hash
            or record.get("atom") != atom_relative
            or record.get("report") != expected_report
            or record.get("constraint_passed") is not True
            or record.get("collision_checked") is not True
        ):
            _fail(f"Atom {atom_id}", "candidate run_manifest binding is not an exact PASS")
        expected_report_sha = _sha256(record.get("report_sha256"), f"Atom {atom_id} report_sha256")
        repair_event = None
    else:
        repair_event = _exact_repair_event(
            events,
            atom_id=atom_id,
            version=version,
            revision=revision,
            content_hash=content_hash,
            atom_relative=atom_relative,
        )
        report_relative = repair_event.get("report")
        expected_report = atom_path.with_name("motion_report.json").relative_to(run_root).as_posix()
        if report_relative != expected_report:
            _fail(f"Atom {atom_id}", "repair report path does not match exact bundle")
    report_path, report_raw, report = _read_evidence_json(
        run_root, expected_report, f"Atom {atom_id} report"
    )
    if version == "candidate" and hashlib.sha256(report_raw).hexdigest() != expected_report_sha:
        _fail(f"Atom {atom_id}", "candidate report hash mismatch")

    if atom.get("schema_version") != ATOM_SCHEMA_VERSION:
        _fail(f"Atom {atom_id}.schema_version", f"must equal {ATOM_SCHEMA_VERSION}")
    identity = atom.get("identity")
    if not isinstance(identity, Mapping):
        _fail(f"Atom {atom_id}.identity", "must be an object")
    if (
        identity.get("id") != atom_id
        or identity.get("revision") != revision
        or identity.get("content_hash") != content_hash
    ):
        _fail(f"Atom {atom_id}", "identity does not match catalog exact version")
    if _atom_hash(atom) != content_hash:
        _fail(f"Atom {atom_id}", "canonical content hash mismatch")
    _validate_trajectory(atom, atom_id)

    if version == "candidate":
        if report.get("atom_id") != atom_id or report.get("atom_content_hash") != content_hash:
            _fail(f"Atom {atom_id}", "candidate report identity mismatch")
    else:
        expected_report_sha = _sha256(
            repair_event.get("report_sha256"), f"Atom {atom_id} report_sha256"
        )
        if _canonical_sha256(report) != expected_report_sha:
            _fail(f"Atom {atom_id}", "repair report hash mismatch")
    _validate_fixed_base_pass(atom, report, atom_id)

    approval = _approval_event(
        events,
        atom_id=atom_id,
        version=version,
        revision=revision,
        content_hash=content_hash,
    )
    return CatalogAtom(
        atom_id=atom_id,
        version=version,
        revision=revision,
        content_sha256=content_hash,
        approved=True,
        primary_category=category,
        tags=tuple(tags_value),
        label_source=label_source,
        atom_path=atom_path,
        report_path=report_path,
        approval_event_id=str(approval["event_id"]),
        atom_payload=_deep_freeze(atom),
    )


def load_catalog(path: str | Path) -> AtomCatalog:
    """Load and validate one ``atom_catalog_v1`` snapshot.

    Every catalog entry must still be the exact human-approved, fixed-base PASS
    version in its source run at load time.  Any ambiguity or mismatch rejects
    the entire catalog rather than silently dropping an entry.
    """

    catalog_path = Path(path)
    raw = _read_json(catalog_path, "catalog")
    _strict_fields(raw, _ROOT_FIELDS, "catalog")
    if raw["schema_version"] != CATALOG_SCHEMA_VERSION:
        _fail("catalog.schema_version", f"must equal {CATALOG_SCHEMA_VERSION}")
    source_value = _string(raw["source_run_dir"], "catalog.source_run_dir")
    source_path = Path(source_value)
    if not source_path.is_absolute():
        _fail("catalog.source_run_dir", "must be an absolute path")
    try:
        run_root = source_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        _fail("catalog.source_run_dir", f"does not exist: {exc}")
    if not run_root.is_dir():
        _fail("catalog.source_run_dir", "must be a directory")

    entries = raw["atoms"]
    if not isinstance(entries, list) or not entries:
        _fail("catalog.atoms", "must be a non-empty array")
    if any(not isinstance(entry, Mapping) for entry in entries):
        _fail("catalog.atoms", "every item must be an object")
    atom_ids: set[str] = set()
    for index, entry in enumerate(entries):
        _strict_fields(entry, _ATOM_FIELDS, f"atoms[{index}]")
        atom_id = _token(entry["atom_id"], f"atoms[{index}].atom_id")
        if atom_id in atom_ids:
            _fail("catalog.atoms", "duplicate atom_id entries are forbidden")
        atom_ids.add(atom_id)

    annotations_path, annotations_raw = _safe_evidence_bytes(
        run_root, "curation/annotations.jsonl", "annotations"
    )
    events, annotations_sha = _read_events(annotations_raw, annotations_path)
    _manifest_path, _manifest_raw, manifest = _read_evidence_json(
        run_root, "run_manifest.json", "run_manifest"
    )
    atoms = tuple(
        _load_catalog_atom(
            entry,
            index=index,
            run_root=run_root,
            events=events,
            manifest=manifest,
        )
        for index, entry in enumerate(entries)
    )
    return AtomCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        source_run_dir=run_root,
        atoms=atoms,
        annotations_sha256=annotations_sha,
    )


__all__ = [
    "AtomCatalog",
    "CatalogAtom",
    "CatalogValidationError",
    "load_catalog",
]
