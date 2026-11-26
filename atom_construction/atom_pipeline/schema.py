"""Validation and deterministic I/O for the Robot Gesture Atom v1 contract.

The JSON Schema next to this module is the language-neutral contract.  This
module deliberately uses only the Python standard library so extraction and
runtime tools do not need ``jsonschema``.  In addition to structural checks it
enforces cross-field invariants which JSON Schema cannot express conveniently:
phase ordering, trajectory dimensions, endpoint interfaces, certification
evidence, and content-hash integrity.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "robot.gesture_atom.v1"
ATOM_STATUS_VALUES = frozenset({"candidate", "curated", "certified"})
SCHEMA_PATH = Path(__file__).with_name("schema") / "atom_v1.schema.json"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PHASE_NAMES = ("preparation", "stroke", "hold", "retraction")
_VECTOR_NAMES = ("q", "dq", "ddq")


class AtomValidationError(ValueError):
    """Raised when an atom violates the v1 structural or semantic contract."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def load_json_schema() -> dict[str, Any]:
    """Return a fresh copy of the bundled Draft 2020-12 JSON Schema."""

    with SCHEMA_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):  # protects against accidental packaging damage
        raise RuntimeError(f"Atom schema is not a JSON object: {SCHEMA_PATH}")
    return value


def _fail(path: str, message: str) -> None:
    raise AtomValidationError(path, message)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    if missing:
        _fail(path, f"missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        _fail(path, f"unknown field(s): {', '.join(unknown)}")


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        _fail(path, "must be a non-empty string" if not allow_empty else "must be a string")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if minimum is not None and result < minimum:
        _fail(path, f"must be >= {minimum}")
    if exclusive_minimum is not None and result <= exclusive_minimum:
        _fail(path, f"must be > {exclusive_minimum}")
    if maximum is not None and result > maximum:
        _fail(path, f"must be <= {maximum}")
    return result


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _enum(value: Any, path: str, choices: set[str] | frozenset[str]) -> str:
    result = _string(value, path)
    if result not in choices:
        _fail(path, f"must be one of: {', '.join(sorted(choices))}")
    return result


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path)
    if _SHA256_RE.fullmatch(result) is None:
        _fail(path, "must be exactly 64 hexadecimal characters")
    return result.lower()


def _string_list(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    if nonempty and not value:
        _fail(path, "must not be empty")
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        _fail(path, "must contain unique values")
    return result


def _vector(value: Any, path: str, joints: list[str]) -> dict[str, float]:
    mapping = _object(value, path)
    expected = set(joints)
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing joints: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown joints: {', '.join(unknown)}")
        _fail(path, "; ".join(details))
    return {joint: _number(mapping[joint], f"{path}.{joint}") for joint in joints}


def _validate_identity(atom: Mapping[str, Any]) -> None:
    identity = _object(atom["identity"], "$.identity")
    _strict_keys(
        identity,
        "$.identity",
        required={"id", "name", "revision"},
        optional={"description", "content_hash"},
    )
    atom_id = _string(identity["id"], "$.identity.id")
    if _ID_RE.fullmatch(atom_id) is None:
        _fail("$.identity.id", "contains characters outside [A-Za-z0-9._:-]")
    _string(identity["name"], "$.identity.name")
    _integer(identity["revision"], "$.identity.revision", minimum=1)
    if "description" in identity:
        _string(identity["description"], "$.identity.description", allow_empty=True)
    if "content_hash" in identity:
        _sha256(identity["content_hash"], "$.identity.content_hash")


def _validate_semantics(atom: Mapping[str, Any]) -> None:
    semantics = _object(atom["semantics"], "$.semantics")
    _strict_keys(
        semantics,
        "$.semantics",
        required={
            "gesture_family",
            "communicative_functions",
            "tags",
            "handedness",
            "confidence",
            "prosody",
        },
    )
    _string(semantics["gesture_family"], "$.semantics.gesture_family")
    _string_list(
        semantics["communicative_functions"],
        "$.semantics.communicative_functions",
        nonempty=True,
    )
    _string_list(semantics["tags"], "$.semantics.tags")
    _enum(
        semantics["handedness"],
        "$.semantics.handedness",
        {"left", "right", "bilateral", "none", "mixed"},
    )
    _number(semantics["confidence"], "$.semantics.confidence", minimum=0.0, maximum=1.0)

    prosody = _object(semantics["prosody"], "$.semantics.prosody")
    _strict_keys(
        prosody,
        "$.semantics.prosody",
        required={"alignment", "target_anchor", "intensity", "confidence"},
        optional={"accent_types"},
    )
    _enum(
        prosody["alignment"],
        "$.semantics.prosody.alignment",
        {"accent", "phrase_boundary", "turn_boundary", "free", "unknown"},
    )
    if prosody["target_anchor"] != "apex":
        _fail("$.semantics.prosody.target_anchor", "must equal 'apex'")
    _enum(
        prosody["intensity"],
        "$.semantics.prosody.intensity",
        {"low", "medium", "high", "adaptive", "unknown"},
    )
    _number(
        prosody["confidence"],
        "$.semantics.prosody.confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if "accent_types" in prosody:
        _string_list(prosody["accent_types"], "$.semantics.prosody.accent_types")


def _validate_source(atom: Mapping[str, Any]) -> None:
    source = _object(atom["source"], "$.source")
    _strict_keys(
        source,
        "$.source",
        required={
            "dataset",
            "speaker",
            "session",
            "clip",
            "input_sha256",
            "source_fps",
            "frame_start",
            "frame_end",
            "time_start",
            "time_end",
        },
        optional={"relative_path", "extractor", "extractor_version", "license"},
    )
    for field in ("dataset", "speaker", "session", "clip"):
        _string(source[field], f"$.source.{field}")
    _sha256(source["input_sha256"], "$.source.input_sha256")
    _number(source["source_fps"], "$.source.source_fps", exclusive_minimum=0.0)
    frame_start = _integer(source["frame_start"], "$.source.frame_start", minimum=0)
    frame_end = _integer(source["frame_end"], "$.source.frame_end", minimum=0)
    if frame_end < frame_start:
        _fail("$.source.frame_end", "must be >= frame_start")
    time_start = _number(source["time_start"], "$.source.time_start", minimum=0.0)
    time_end = _number(source["time_end"], "$.source.time_end", minimum=0.0)
    if time_end <= time_start:
        _fail("$.source.time_end", "must be > time_start")
    if "relative_path" in source:
        raw_path = _string(source["relative_path"], "$.source.relative_path")
        if raw_path.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_RE.match(raw_path):
            _fail("$.source.relative_path", "must be a relative, portable path")
        portable = PurePosixPath(raw_path.replace("\\", "/"))
        if ".." in portable.parts:
            _fail("$.source.relative_path", "must not traverse a parent directory")
    for field in ("extractor", "extractor_version", "license"):
        if field in source:
            _string(source[field], f"$.source.{field}", allow_empty=True)


def _validate_timing(atom: Mapping[str, Any]) -> tuple[float, dict[str, tuple[float, float]]]:
    timing = _object(atom["timing"], "$.timing")
    _strict_keys(
        timing,
        "$.timing",
        required={
            "nominal_duration_s",
            "min_duration_s",
            "max_duration_s",
            "phases",
            "anchors",
            "safe_cutpoints",
            "min_lead_ms",
            "late_policy",
        },
    )
    nominal = _number(
        timing["nominal_duration_s"],
        "$.timing.nominal_duration_s",
        exclusive_minimum=0.0,
    )
    minimum = _number(
        timing["min_duration_s"],
        "$.timing.min_duration_s",
        exclusive_minimum=0.0,
    )
    maximum = _number(
        timing["max_duration_s"],
        "$.timing.max_duration_s",
        exclusive_minimum=0.0,
    )
    if not minimum <= nominal <= maximum:
        _fail(
            "$.timing",
            "duration invariant must satisfy min_duration_s <= nominal_duration_s <= max_duration_s",
        )

    raw_phases = _object(timing["phases"], "$.timing.phases")
    _strict_keys(raw_phases, "$.timing.phases", required=set(_PHASE_NAMES))
    phases: dict[str, tuple[float, float]] = {}
    previous_end: float | None = None
    for name in _PHASE_NAMES:
        path = f"$.timing.phases.{name}"
        interval = _object(raw_phases[name], path)
        _strict_keys(interval, path, required={"start_s", "end_s"})
        start = _number(interval["start_s"], f"{path}.start_s", minimum=0.0)
        end = _number(interval["end_s"], f"{path}.end_s", minimum=0.0)
        if end < start:
            _fail(f"{path}.end_s", "must be >= start_s")
        if end > nominal + 1e-9:
            _fail(f"{path}.end_s", "must not exceed nominal_duration_s")
        if previous_end is not None and start < previous_end - 1e-9:
            _fail(f"{path}.start_s", "phases must be ordered and non-overlapping")
        phases[name] = (start, end)
        previous_end = end
    if not math.isclose(phases["preparation"][0], 0.0, abs_tol=1e-9):
        _fail("$.timing.phases.preparation.start_s", "must start at 0")
    if not math.isclose(phases["retraction"][1], nominal, abs_tol=1e-9):
        _fail("$.timing.phases.retraction.end_s", "must equal nominal_duration_s")

    anchors = _object(timing["anchors"], "$.timing.anchors")
    _strict_keys(
        anchors,
        "$.timing.anchors",
        required={"apex"},
        optional={"semantic_onset", "semantic_offset"},
    )
    parsed_anchors = {
        name: _number(value, f"$.timing.anchors.{name}", minimum=0.0, maximum=nominal)
        for name, value in anchors.items()
    }
    stroke_start, stroke_end = phases["stroke"]
    if not stroke_start - 1e-9 <= parsed_anchors["apex"] <= stroke_end + 1e-9:
        _fail("$.timing.anchors.apex", "must lie inside the stroke interval")
    if (
        "semantic_onset" in parsed_anchors
        and "semantic_offset" in parsed_anchors
        and parsed_anchors["semantic_offset"] < parsed_anchors["semantic_onset"]
    ):
        _fail("$.timing.anchors.semantic_offset", "must be >= semantic_onset")

    cutpoints_raw = timing["safe_cutpoints"]
    if not isinstance(cutpoints_raw, list):
        _fail("$.timing.safe_cutpoints", "must be an array")
    cutpoints = [
        _number(value, f"$.timing.safe_cutpoints[{index}]", minimum=0.0, maximum=nominal)
        for index, value in enumerate(cutpoints_raw)
    ]
    if any(right <= left for left, right in zip(cutpoints, cutpoints[1:])):
        _fail("$.timing.safe_cutpoints", "must be strictly increasing and unique")
    _number(timing["min_lead_ms"], "$.timing.min_lead_ms", minimum=0.0)
    _enum(
        timing["late_policy"],
        "$.timing.late_policy",
        {"skip", "fallback", "compress", "truncate_preparation", "execute_late"},
    )
    return nominal, phases


def _validate_range(
    value: Any,
    path: str,
    *,
    min_floor: float,
    min_exclusive: bool = False,
) -> tuple[float, float]:
    mapping = _object(value, path)
    _strict_keys(mapping, path, required={"min", "max"})
    if min_exclusive:
        minimum = _number(mapping["min"], f"{path}.min", exclusive_minimum=min_floor)
    else:
        minimum = _number(mapping["min"], f"{path}.min", minimum=min_floor)
    maximum = _number(mapping["max"], f"{path}.max")
    if maximum < minimum:
        _fail(f"{path}.max", "must be >= min")
    return minimum, maximum


def _validate_control(atom: Mapping[str, Any], duration: float) -> tuple[list[str], list[dict[str, Any]]]:
    control = _object(atom["control"], "$.control")
    _strict_keys(
        control,
        "$.control",
        required={
            "sample_rate_hz",
            "joints",
            "joint_mask",
            "neutral",
            "backend_compatibility",
            "interpolation",
            "tempo_range",
            "amplitude_range",
            "trajectory",
        },
        optional={"pose_representation"},
    )
    _number(control["sample_rate_hz"], "$.control.sample_rate_hz", exclusive_minimum=0.0)
    joints = _string_list(control["joints"], "$.control.joints", nonempty=True)
    joint_mask = _string_list(control["joint_mask"], "$.control.joint_mask", nonempty=True)
    unknown_mask = sorted(set(joint_mask) - set(joints))
    if unknown_mask:
        _fail("$.control.joint_mask", f"contains joints absent from control.joints: {', '.join(unknown_mask)}")

    neutral = _object(control["neutral"], "$.control.neutral")
    _strict_keys(neutral, "$.control.neutral", required={"id", "content_hash"})
    _string(neutral["id"], "$.control.neutral.id")
    if "pose_representation" in control:
        representation = _object(
            control["pose_representation"], "$.control.pose_representation"
        )
        _strict_keys(
            representation,
            "$.control.pose_representation",
            required={"q_space", "source_mapping", "gain_reference"},
            optional={"sampling_basis", "detector_filter_affects_export"},
        )
        _enum(
            representation["q_space"],
            "$.control.pose_representation.q_space",
            {"absolute_joint_position_rad"},
        )
        source_mapping = _enum(
            representation["source_mapping"],
            "$.control.pose_representation.source_mapping",
            {"preserve_absolute_retarget_pose", "neutral_rebased_delta"},
        )
        gain_reference = _enum(
            representation["gain_reference"],
            "$.control.pose_representation.gain_reference",
            {"control.neutral", "source_entry"},
        )
        expected_reference = (
            "control.neutral"
            if source_mapping == "preserve_absolute_retarget_pose"
            else "source_entry"
        )
        if gain_reference != expected_reference:
            _fail(
                "$.control.pose_representation.gain_reference",
                f"must equal {expected_reference!r} for source_mapping {source_mapping!r}",
            )
        sampling_basis_present = "sampling_basis" in representation
        detector_export_present = "detector_filter_affects_export" in representation
        if sampling_basis_present != detector_export_present:
            _fail(
                "$.control.pose_representation",
                "sampling_basis and detector_filter_affects_export must be declared together",
            )
        if sampling_basis_present:
            _enum(
                representation["sampling_basis"],
                "$.control.pose_representation.sampling_basis",
                {"contiguous_source_frame_knots", "contiguous_source_frame_knots_bit_exact", "contiguous_source_frame_knots_bit_exact_v2"},
            )
            detector_affects_export = _boolean(
                representation["detector_filter_affects_export"],
                "$.control.pose_representation.detector_filter_affects_export",
            )
            if detector_affects_export:
                _fail(
                    "$.control.pose_representation.detector_filter_affects_export",
                    "must be false for contiguous source-frame knot export",
                )
    _sha256(neutral["content_hash"], "$.control.neutral.content_hash")
    _string_list(
        control["backend_compatibility"],
        "$.control.backend_compatibility",
        nonempty=True,
    )
    _enum(
        control["interpolation"],
        "$.control.interpolation",
        {"linear", "cubic", "quintic", "sampled"},
    )
    _validate_range(control["tempo_range"], "$.control.tempo_range", min_floor=0.0, min_exclusive=True)
    _validate_range(control["amplitude_range"], "$.control.amplitude_range", min_floor=0.0)

    raw_trajectory = control["trajectory"]
    if not isinstance(raw_trajectory, list) or len(raw_trajectory) < 2:
        _fail("$.control.trajectory", "must contain at least two samples")
    trajectory: list[dict[str, Any]] = []
    previous_time: float | None = None
    for index, raw_point in enumerate(raw_trajectory):
        path = f"$.control.trajectory[{index}]"
        point = _object(raw_point, path)
        _strict_keys(point, path, required={"t", "q", "dq", "ddq"})
        time_s = _number(point["t"], f"{path}.t", minimum=0.0)
        if previous_time is not None and time_s <= previous_time:
            _fail(f"{path}.t", "trajectory times must be strictly increasing")
        previous_time = time_s
        parsed: dict[str, Any] = {"t": time_s}
        for vector_name in _VECTOR_NAMES:
            parsed[vector_name] = _vector(point[vector_name], f"{path}.{vector_name}", joints)
        trajectory.append(parsed)
    if not math.isclose(trajectory[0]["t"], 0.0, abs_tol=1e-9):
        _fail("$.control.trajectory[0].t", "must equal 0")
    if not math.isclose(trajectory[-1]["t"], duration, abs_tol=1e-9):
        _fail("$.control.trajectory[-1].t", "must equal timing.nominal_duration_s")
    return joints, trajectory


def _validate_interface(
    atom: Mapping[str, Any],
    field: str,
    joints: list[str],
    endpoint: Mapping[str, Any],
) -> None:
    path = f"$.{field}"
    interface = _object(atom[field], path)
    _strict_keys(
        interface,
        path,
        required={"q", "dq", "ddq", "tolerance"},
        optional={"tolerance_semantics"},
    )
    state = {
        vector_name: _vector(interface[vector_name], f"{path}.{vector_name}", joints)
        for vector_name in _VECTOR_NAMES
    }
    tolerance = _object(interface["tolerance"], f"{path}.tolerance")
    _strict_keys(tolerance, f"{path}.tolerance", required=set(_VECTOR_NAMES))
    tolerances = {
        name: _number(tolerance[name], f"{path}.tolerance.{name}", minimum=0.0)
        for name in _VECTOR_NAMES
    }
    if "tolerance_semantics" in interface:
        _enum(
            interface["tolerance_semantics"],
            f"{path}.tolerance_semantics",
            {"storage_consistency_only_not_runtime_acceptance"},
        )
    for vector_name in _VECTOR_NAMES:
        for joint in joints:
            delta = abs(state[vector_name][joint] - endpoint[vector_name][joint])
            if delta > tolerances[vector_name] + 1e-12:
                _fail(
                    f"{path}.{vector_name}.{joint}",
                    f"differs from the trajectory endpoint by {delta:g}, above tolerance {tolerances[vector_name]:g}",
                )


def _validate_thresholds(value: Any) -> None:
    path = "$.safety.thresholds"
    thresholds = _object(value, path)
    required = {
        "joint_position_margin_rad",
        "max_velocity_rad_s",
        "max_acceleration_rad_s2",
        "max_jerk_rad_s3",
        "max_tracking_error_rad",
        "min_collision_distance_m",
        "max_torque_saturation_fraction",
    }
    _strict_keys(thresholds, path, required=required)
    _number(thresholds["joint_position_margin_rad"], f"{path}.joint_position_margin_rad", minimum=0.0)
    for field in ("max_velocity_rad_s", "max_acceleration_rad_s2", "max_jerk_rad_s3"):
        _number(thresholds[field], f"{path}.{field}", exclusive_minimum=0.0)
    _number(thresholds["max_tracking_error_rad"], f"{path}.max_tracking_error_rad", minimum=0.0)
    _number(thresholds["min_collision_distance_m"], f"{path}.min_collision_distance_m", minimum=0.0)
    _number(
        thresholds["max_torque_saturation_fraction"],
        f"{path}.max_torque_saturation_fraction",
        minimum=0.0,
        maximum=1.0,
    )


def _validate_metric_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    _number(value, path)


def _validate_diagnostics(value: Any, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    result = []
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        item = _object(raw, item_path)
        _strict_keys(
            item,
            item_path,
            required={"code", "message"},
            optional={"joint", "time_s", "observed", "limit"},
        )
        _string(item["code"], f"{item_path}.code")
        _string(item["message"], f"{item_path}.message")
        if "joint" in item:
            _string(item["joint"], f"{item_path}.joint")
        if "time_s" in item:
            _number(item["time_s"], f"{item_path}.time_s", minimum=0.0)
        for field in ("observed", "limit"):
            if field in item:
                _validate_metric_value(item[field], f"{item_path}.{field}")
        result.append(item)
    return result


def _validate_safety(atom: Mapping[str, Any], joints: list[str], duration: float) -> None:
    safety = _object(atom["safety"], "$.safety")
    _strict_keys(
        safety,
        "$.safety",
        required={
            "thresholds",
            "metrics",
            "per_joint",
            "violations",
            "warnings",
            "collision",
            "validator_version",
            "certified_backends",
        },
        optional={"validated_at"},
    )
    _validate_thresholds(safety["thresholds"])

    metrics = _object(safety["metrics"], "$.safety.metrics")
    if "passed" not in metrics or "base_mode" not in metrics:
        _fail("$.safety.metrics", "must include passed and base_mode")
    metrics_passed = _boolean(metrics["passed"], "$.safety.metrics.passed")
    _enum(metrics["base_mode"], "$.safety.metrics.base_mode", {"fixed", "floating", "hardware"})
    for name, value in metrics.items():
        if name not in {"passed", "base_mode"}:
            _validate_metric_value(value, f"$.safety.metrics.{name}")

    per_joint = _object(safety["per_joint"], "$.safety.per_joint")
    missing = sorted(set(joints) - set(per_joint))
    unknown = sorted(set(per_joint) - set(joints))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing joints: {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown joints: {', '.join(unknown)}")
        _fail("$.safety.per_joint", "; ".join(detail))
    required_joint_fields = {"q_min", "q_max", "max_abs_dq", "max_abs_ddq", "max_abs_jerk"}
    optional_joint_fields = {"max_abs_tracking_error", "max_abs_torque"}
    for joint in joints:
        path = f"$.safety.per_joint.{joint}"
        report = _object(per_joint[joint], path)
        _strict_keys(report, path, required=required_joint_fields, optional=optional_joint_fields)
        q_min = _number(report["q_min"], f"{path}.q_min")
        q_max = _number(report["q_max"], f"{path}.q_max")
        if q_max < q_min:
            _fail(f"{path}.q_max", "must be >= q_min")
        for field in required_joint_fields - {"q_min", "q_max"} | optional_joint_fields:
            if field in report:
                _number(report[field], f"{path}.{field}", minimum=0.0)

    violations = _validate_diagnostics(safety["violations"], "$.safety.violations")
    _validate_diagnostics(safety["warnings"], "$.safety.warnings")
    if metrics_passed and violations:
        _fail("$.safety.metrics.passed", "cannot be true while violations is non-empty")

    collision = _object(safety["collision"], "$.safety.collision")
    _strict_keys(
        collision,
        "$.safety.collision",
        required={
            "checked",
            "passed",
            "self_collision_count",
            "environment_collision_count",
            "min_distance_m",
        },
    )
    collision_checked = _boolean(collision["checked"], "$.safety.collision.checked")
    collision_passed = _boolean(collision["passed"], "$.safety.collision.passed")
    self_count = _integer(
        collision["self_collision_count"],
        "$.safety.collision.self_collision_count",
        minimum=0,
    )
    environment_count = _integer(
        collision["environment_collision_count"],
        "$.safety.collision.environment_collision_count",
        minimum=0,
    )
    if collision["min_distance_m"] is not None:
        _number(collision["min_distance_m"], "$.safety.collision.min_distance_m")
    if collision_passed and not collision_checked:
        _fail("$.safety.collision.passed", "cannot be true when checked is false")
    if collision_passed and (self_count or environment_count):
        _fail("$.safety.collision", "a passing collision report must have zero collision counts")
    collision_distance = collision["min_distance_m"]
    required_distance = float(safety["thresholds"]["min_collision_distance_m"])
    if collision_checked and collision_distance is not None:
        observed_distance = float(collision_distance)
        if collision_passed and observed_distance < required_distance - 1e-12:
            _fail(
                "$.safety.collision.min_distance_m",
                "a passing collision report cannot be below "
                "safety.thresholds.min_collision_distance_m",
            )

    velocity_limit = float(safety["thresholds"]["max_velocity_rad_s"])
    acceleration_limit = float(safety["thresholds"]["max_acceleration_rad_s2"])
    jerk_limit = float(safety["thresholds"]["max_jerk_rad_s3"])
    observed_exceeds_threshold = any(
        float(report[metric]) > limit + 1e-12
        for report in per_joint.values()
        for metric, limit in (
            ("max_abs_dq", velocity_limit),
            ("max_abs_ddq", acceleration_limit),
            ("max_abs_jerk", jerk_limit),
        )
    )
    if metrics_passed and observed_exceeds_threshold:
        _fail(
            "$.safety.metrics.passed",
            "cannot be true when per_joint dynamics exceed declared safety thresholds",
        )

    _string(safety["validator_version"], "$.safety.validator_version")
    certified_backends = _string_list(
        safety["certified_backends"],
        "$.safety.certified_backends",
    )
    compatible_backends = set(atom["control"]["backend_compatibility"])
    unsupported = sorted(set(certified_backends) - compatible_backends)
    if unsupported:
        _fail(
            "$.safety.certified_backends",
            f"contains backends absent from control.backend_compatibility: {', '.join(unsupported)}",
        )
    if "validated_at" in safety:
        _string(safety["validated_at"], "$.safety.validated_at")

    status = atom["status"]
    if status == "certified":
        if not metrics_passed:
            _fail("$.status", "certified atoms require safety.metrics.passed=true")
        if violations:
            _fail("$.status", "certified atoms cannot contain safety violations")
        if not collision_checked or not collision_passed:
            _fail("$.status", "certified atoms require a checked and passing collision report")
        if not certified_backends:
            _fail("$.status", "certified atoms require at least one certified backend")

    # Diagnostic timestamps must refer to this atom, not the uncropped source.
    for list_name in ("violations", "warnings"):
        for index, item in enumerate(safety[list_name]):
            if "time_s" in item and float(item["time_s"]) > duration + 1e-9:
                _fail(f"$.safety.{list_name}[{index}].time_s", "must not exceed atom duration")


def _validate_fallback(atom: Mapping[str, Any]) -> None:
    fallback = _object(atom["fallback"], "$.fallback")
    _strict_keys(
        fallback,
        "$.fallback",
        required={"atom_id", "policy"},
        optional={"reason_codes"},
    )
    _string(fallback["atom_id"], "$.fallback.atom_id")
    _enum(
        fallback["policy"],
        "$.fallback.policy",
        {"replace", "release_to_neutral", "hold", "skip"},
    )
    if "reason_codes" in fallback:
        _string_list(fallback["reason_codes"], "$.fallback.reason_codes")


def canonical_atom_json(atom: Mapping[str, Any]) -> str:
    """Serialize an atom deterministically, excluding its self-referential hash.

    The returned string has sorted keys, no insignificant whitespace, UTF-8
    characters left unescaped, and no NaN/Infinity values.  Only
    ``identity.content_hash`` is removed; source and neutral hashes remain part
    of the content identity.
    """

    if not isinstance(atom, Mapping):
        raise AtomValidationError("$", "must be an object")
    payload = deepcopy(dict(atom))
    identity = payload.get("identity")
    if isinstance(identity, dict):
        identity.pop("content_hash", None)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AtomValidationError("$", f"must contain only finite JSON values: {exc}") from exc


def atom_hash(atom: Mapping[str, Any]) -> str:
    """Return the lowercase SHA-256 hex digest of canonical atom content."""

    canonical = canonical_atom_json(atom).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_atom(atom: Mapping[str, Any], *, verify_hash: bool = True) -> None:
    """Validate one atom, raising :class:`AtomValidationError` on failure.

    Validation never mutates the input and never promotes ``status``.  In
    particular, a candidate with a successful fixed-base rollout remains a
    candidate until an explicit curation/certification step changes it.
    """

    root = _object(atom, "$")
    required = {
        "schema_version",
        "identity",
        "status",
        "semantics",
        "source",
        "timing",
        "control",
        "entry",
        "exit",
        "safety",
        "fallback",
    }
    _strict_keys(root, "$", required=required, optional={"$schema", "metadata", "extensions"})
    if root["schema_version"] != SCHEMA_VERSION:
        _fail("$.schema_version", f"must equal '{SCHEMA_VERSION}'")
    if "$schema" in root:
        _string(root["$schema"], "$.$schema")
    _validate_identity(root)
    _enum(root["status"], "$.status", ATOM_STATUS_VALUES)
    _validate_semantics(root)
    _validate_source(root)
    duration, _ = _validate_timing(root)
    joints, trajectory = _validate_control(root, duration)
    _validate_interface(root, "entry", joints, trajectory[0])
    _validate_interface(root, "exit", joints, trajectory[-1])
    _validate_safety(root, joints, duration)
    _validate_fallback(root)
    for field in ("metadata", "extensions"):
        if field in root:
            _object(root[field], f"$.{field}")

    # This also rejects non-JSON extension values and NaN/Infinity anywhere.
    canonical_atom_json(root)
    supplied_hash = root["identity"].get("content_hash")
    if verify_hash and supplied_hash is not None:
        expected = atom_hash(root)
        if not hmac.compare_digest(str(supplied_hash).lower(), expected):
            _fail("$.identity.content_hash", f"does not match canonical content hash {expected}")


def load_atom(path: str | os.PathLike[str], *, verify_hash: bool = True) -> dict[str, Any]:
    """Load and validate an atom JSON file."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            atom = json.load(stream)
    except json.JSONDecodeError as exc:
        raise AtomValidationError("$", f"invalid JSON in {source}: {exc.msg}") from exc
    if not isinstance(atom, dict):
        raise AtomValidationError("$", "must be an object")
    validate_atom(atom, verify_hash=verify_hash)
    return atom


def write_atom(
    atom: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    fill_content_hash: bool = True,
) -> dict[str, Any]:
    """Validate and atomically write an atom, returning the written copy.

    By default ``identity.content_hash`` is replaced with the canonical hash.
    This is the only field this function fills; notably it never changes
    ``status`` or safety results.
    """

    if not isinstance(atom, Mapping):
        raise AtomValidationError("$", "must be an object")
    result = deepcopy(dict(atom))
    validate_atom(result, verify_hash=False)
    if fill_content_hash:
        identity = result.get("identity")
        if not isinstance(identity, dict):  # already caught above; keeps typing honest
            raise AtomValidationError("$.identity", "must be an object")
        identity["content_hash"] = atom_hash(result)
    validate_atom(result, verify_hash=fill_content_hash)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return result


# Concise aliases for callers which treat this module as a serializer backend.
validate = validate_atom
load = load_atom
write = write_atom
hash_atom = atom_hash


__all__ = [
    "ATOM_STATUS_VALUES",
    "AtomValidationError",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "atom_hash",
    "canonical_atom_json",
    "hash_atom",
    "load",
    "load_atom",
    "load_json_schema",
    "validate",
    "validate_atom",
    "write",
    "write_atom",
]
