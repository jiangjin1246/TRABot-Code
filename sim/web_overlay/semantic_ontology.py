"""Strict loader for the frozen 17-function dialogue ontology."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping


ONTOLOGY_SCHEMA_VERSION = "dialogue_function_17_v1"
EXCLUDED_FUNCTION = "present_explain"
EXPECTED_FUNCTION_COUNT = 17

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "label_scope",
        "derived_from",
        "excluded_functions",
        "functions",
        "separation_rule",
    }
)
_FUNCTION_FIELDS = frozenset(
    {"function", "planner_category", "common_families", "definition_zh"}
)
_EXCLUDED_FIELDS = frozenset({"function", "reason"})
_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_ONTOLOGY_BYTES = 256 * 1024


class OntologyValidationError(ValueError):
    """Raised when the frozen ontology is missing, unsafe, or ambiguous."""


@dataclass(frozen=True, slots=True)
class DialogueFunction:
    function: str
    planner_category: str
    common_families: tuple[str, ...]
    definition_zh: str


@dataclass(frozen=True, slots=True)
class SemanticOntology:
    schema_version: str
    label_scope: str
    derived_from: str
    functions: tuple[DialogueFunction, ...]
    excluded_functions: tuple[str, ...]
    separation_rule: str
    content_sha256: str
    by_name: Mapping[str, DialogueFunction]

    @property
    def function_names(self) -> tuple[str, ...]:
        return tuple(item.function for item in self.functions)


def _fail(label: str, detail: str) -> None:
    raise OntologyValidationError(f"{label}: {detail}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_regular(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        _fail("ontology", f"cannot stat {path}: {exc}")
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        _fail("ontology", f"must be a regular non-symlink file: {path}")
    if before.st_size > _MAX_ONTOLOGY_BYTES:
        _fail("ontology", f"file exceeds {_MAX_ONTOLOGY_BYTES} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        _fail("ontology", f"cannot safely open {path}: {exc}")
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _fail("ontology", "file changed while opening")
        raw = b""
        while len(raw) <= _MAX_ONTOLOGY_BYTES:
            chunk = os.read(fd, min(64 * 1024, _MAX_ONTOLOGY_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(fd)
        if len(raw) > _MAX_ONTOLOGY_BYTES or len(raw) != after.st_size:
            _fail("ontology", "file changed while reading or exceeds size limit")
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
            _fail("ontology", "file changed while reading")
        return raw
    finally:
        os.close(fd)


def _decode(raw: bytes, path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail("ontology", f"invalid finite JSON in {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail("ontology", "root must be an object")
    return value


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
        _fail(label, "must be a lowercase underscore token")
    return result


def load_ontology(path: str | Path) -> SemanticOntology:
    """Load exactly ``dialogue_function_17_v1`` and reject ontology drift."""

    ontology_path = Path(path)
    raw = _read_regular(ontology_path)
    value = _decode(raw, ontology_path)
    _strict(value, _ROOT_FIELDS, "ontology")
    if value["schema_version"] != ONTOLOGY_SCHEMA_VERSION:
        _fail("ontology.schema_version", f"must equal {ONTOLOGY_SCHEMA_VERSION}")
    if value["label_scope"] != "planner_function":
        _fail("ontology.label_scope", "must equal planner_function")
    derived_from = _string(value["derived_from"], "ontology.derived_from")
    separation_rule = _string(value["separation_rule"], "ontology.separation_rule")

    excluded_raw = value["excluded_functions"]
    if not isinstance(excluded_raw, list) or len(excluded_raw) != 1:
        _fail(
            "ontology.excluded_functions",
            "must contain exactly the explicit present_explain exclusion",
        )
    excluded: list[str] = []
    for index, item in enumerate(excluded_raw):
        label = f"ontology.excluded_functions[{index}]"
        if not isinstance(item, Mapping):
            _fail(label, "must be an object")
        _strict(item, _EXCLUDED_FIELDS, label)
        function = _token(item["function"], f"{label}.function")
        _string(item["reason"], f"{label}.reason")
        excluded.append(function)
    if tuple(excluded) != (EXCLUDED_FUNCTION,):
        _fail(
            "ontology.excluded_functions",
            "present_explain must be the sole explicitly excluded function",
        )

    functions_raw = value["functions"]
    if not isinstance(functions_raw, list) or len(functions_raw) != EXPECTED_FUNCTION_COUNT:
        _fail(
            "ontology.functions",
            f"must contain exactly {EXPECTED_FUNCTION_COUNT} functions",
        )
    functions: list[DialogueFunction] = []
    seen: set[str] = set()
    for index, item in enumerate(functions_raw):
        label = f"ontology.functions[{index}]"
        if not isinstance(item, Mapping):
            _fail(label, "must be an object")
        _strict(item, _FUNCTION_FIELDS, label)
        function = _token(item["function"], f"{label}.function")
        if function in seen:
            _fail(label, f"duplicate function {function}")
        if function == EXCLUDED_FUNCTION:
            _fail(label, "present_explain is explicitly removed from this ontology")
        seen.add(function)
        families_raw = item["common_families"]
        if (
            not isinstance(families_raw, list)
            or not families_raw
            or any(not isinstance(family, str) or not family.strip() for family in families_raw)
            or len(families_raw) != len(set(families_raw))
        ):
            _fail(f"{label}.common_families", "must be a unique non-empty string array")
        functions.append(
            DialogueFunction(
                function=function,
                planner_category=_token(
                    item["planner_category"], f"{label}.planner_category"
                ),
                common_families=tuple(families_raw),
                definition_zh=_string(item["definition_zh"], f"{label}.definition_zh"),
            )
        )
    return SemanticOntology(
        schema_version=ONTOLOGY_SCHEMA_VERSION,
        label_scope="planner_function",
        derived_from=derived_from,
        functions=tuple(functions),
        excluded_functions=tuple(excluded),
        separation_rule=separation_rule,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        by_name=MappingProxyType({item.function: item for item in functions}),
    )


__all__ = [
    "DialogueFunction",
    "EXCLUDED_FUNCTION",
    "ONTOLOGY_SCHEMA_VERSION",
    "OntologyValidationError",
    "SemanticOntology",
    "load_ontology",
]
