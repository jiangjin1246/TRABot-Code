"""Identity-only client for one native G1 realtime Atom session.

The semantic planner still prepares inline motions so its timing and audit
hashes remain unchanged.  This adapter deliberately never forwards those
joint arrays to the native process: the process receives only an exact catalog
identity and reloads the pinned motion itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Mapping


STATUS_SCHEMA_VERSION = "g1_atom_motion_cli_status_v1"
BACKEND_NAME = "g1_realtime_session"
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NETWORK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_SEMANTIC_REF_FIELDS = frozenset(
    {
        "entry_id",
        "atom_id",
        "version",
        "revision",
        "content_sha256",
        "catalog_content_sha256",
    }
)
_RECOVERY_REF_FIELDS = frozenset(
    {
        "target",
        "catalog_content_sha256",
        "neutral_motion_sha256",
        "after",
        "tail_guard_us",
    }
)


class G1SessionError(RuntimeError):
    """The native session could not satisfy the requested operation."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class G1SessionConflictError(G1SessionError):
    """Atomic admission failed before an NDJSON command was written."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


class G1SessionProtocolError(G1SessionError):
    """The native process violated the frozen NDJSON status contract."""


@dataclass(frozen=True, slots=True)
class G1SessionConfig:
    runner: Path
    catalog: Path
    limits: Path
    network: str
    review_store: Path | None = None
    startup_timeout_s: float = 8.0
    action_timeout_s: float = 45.0
    idle_timeout_s: float = 15.0
    process_stop_timeout_s: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "runner", _resolved_path(self.runner, "runner"))
        object.__setattr__(self, "catalog", _resolved_path(self.catalog, "catalog"))
        object.__setattr__(self, "limits", _resolved_path(self.limits, "limits"))
        if self.review_store is not None:
            object.__setattr__(
                self,
                "review_store",
                _resolved_path(self.review_store, "review_store"),
            )
        if not isinstance(self.network, str) or _NETWORK_RE.fullmatch(self.network) is None:
            raise ValueError("network must be a safe non-empty interface name")
        _timeout(self.startup_timeout_s, "startup_timeout_s", 0.1, 600.0)
        _timeout(self.action_timeout_s, "action_timeout_s", 0.1, 600.0)
        _timeout(self.idle_timeout_s, "idle_timeout_s", 1.0, 3600.0)
        _timeout(
            self.process_stop_timeout_s,
            "process_stop_timeout_s",
            0.1,
            60.0,
        )


@dataclass(frozen=True, slots=True)
class _CatalogIdentity:
    entry_id: str
    atom_id: str
    version: str
    revision: int
    content_sha256: str
    motion_sha256: str


def _resolved_path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be a filesystem path")
    return Path(value).expanduser().resolve()


def _timeout(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
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
        raise G1SessionError(f"value is not finite canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _flat_payload_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("payload_sha256", None)
    return _canonical_sha256(body)


def _lower_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise G1SessionError(f"{label} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise G1SessionError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise G1SessionError(f"{label} must be a positive integer")
    return result


def _safe_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN_RE.fullmatch(value) is None:
        raise G1SessionError(f"{label} must be a safe protocol token")
    return value


def _finite_q(value: Any, label: str = "measured_q") -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise G1SessionProtocolError(f"{label} must be a non-empty object")
    result: dict[str, float] = {}
    for joint, raw in value.items():
        if not isinstance(joint, str) or not joint:
            raise G1SessionProtocolError(f"{label} contains an invalid joint name")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise G1SessionProtocolError(f"{label}.{joint} must be numeric")
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise G1SessionProtocolError(f"{label}.{joint} must be finite")
        result[joint] = numeric
    return result


def _load_catalog(path: Path) -> tuple[str, str, dict[str, _CatalogIdentity]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load G1 catalog {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "semantic_atom_catalog_v1":
        raise ValueError("G1 catalog has an unsupported schema")
    rows = payload.get("entries")
    neutral = payload.get("neutral")
    if not isinstance(rows, list) or not rows or not isinstance(neutral, Mapping):
        raise ValueError("G1 catalog must contain entries and neutral")
    neutral_motion = neutral.get("motion")
    if not isinstance(neutral_motion, Mapping):
        raise ValueError("G1 catalog neutral motion is missing")
    # This field intentionally hashes the original canonical-neutral source
    # file, not the normalized inline ``neutral.motion`` object embedded in the
    # semantic catalog.  SemanticCatalog exposes this declared, catalog-bound
    # value in recovery_ref, so using a newly canonicalized inline-motion hash
    # here would reject every genuine recovery despite identical motion data.
    neutral_motion_sha256 = neutral.get("motion_sha256")
    if (
        not isinstance(neutral_motion_sha256, str)
        or _SHA256_RE.fullmatch(neutral_motion_sha256) is None
    ):
        raise ValueError("G1 catalog neutral.motion_sha256 is invalid")
    inputs = payload.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("neutral_motion_sha256") != neutral_motion_sha256
    ):
        raise ValueError(
            "G1 catalog neutral.motion_sha256 does not match inputs authority"
        )
    catalog_sha256 = hashlib.sha256(raw).hexdigest()
    identities: dict[str, _CatalogIdentity] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"G1 catalog entry {index} must be an object")
        entry_id = row.get("entry_id")
        atom_id = row.get("atom_id")
        version = row.get("version")
        revision = row.get("revision")
        content_sha256 = row.get("content_sha256")
        motion = row.get("motion")
        if not isinstance(entry_id, str) or _SHA256_RE.fullmatch(entry_id) is None:
            raise ValueError(f"G1 catalog entry {index} has an invalid entry_id")
        if entry_id in identities:
            raise ValueError(f"G1 catalog duplicates entry_id {entry_id}")
        if not isinstance(atom_id, str) or not atom_id:
            raise ValueError(f"G1 catalog entry {index} has an invalid atom_id")
        if not isinstance(version, str) or not version or len(version) > 128:
            raise ValueError(f"G1 catalog entry {index} has an invalid version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError(f"G1 catalog entry {index} has an invalid revision")
        if not isinstance(content_sha256, str) or _SHA256_RE.fullmatch(content_sha256) is None:
            raise ValueError(f"G1 catalog entry {index} has an invalid content_sha256")
        if row.get("approved") is not True or row.get("removed") is not False:
            raise ValueError(f"G1 catalog entry {index} is not an executable approved row")
        if not isinstance(motion, Mapping):
            raise ValueError(f"G1 catalog entry {index} has no inline motion")
        identities[entry_id] = _CatalogIdentity(
            entry_id=entry_id,
            atom_id=atom_id,
            version=version,
            revision=revision,
            content_sha256=content_sha256,
            motion_sha256=_canonical_sha256(motion),
        )
    return catalog_sha256, neutral_motion_sha256, identities


class G1SessionClient:
    """Simulator-shaped facade over one fail-closed native G1 process."""

    simulated = False
    semantic_only = True
    real_enabled = True

    def __init__(
        self,
        config: G1SessionConfig,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, G1SessionConfig):
            raise TypeError("config must be a G1SessionConfig")
        if not callable(popen_factory) or not callable(monotonic):
            raise TypeError("popen_factory and monotonic must be callable")
        self.config = config
        self._popen_factory = popen_factory
        self._real_popen = popen_factory is subprocess.Popen
        self._monotonic = monotonic
        (
            self._catalog_sha256,
            self._neutral_motion_sha256,
            self._catalog_entries,
        ) = _load_catalog(config.catalog)

        self._condition = threading.Condition(threading.RLock())
        self._process: Any | None = None
        self._reader: threading.Thread | None = None
        self._closed = False
        self._bound_request: tuple[str, int, str] | None = None
        self._owner_thread: int | None = None
        self._session_id: str | None = None
        self._started_once = False
        self._ready = False
        self._session_completed = False
        self._expected_stop = False
        self._fault: str | None = None
        self._native_error: dict[str, Any] | None = None
        self._q: dict[str, float] | None = None
        self._action_seq = 0
        self._state_seq = 0
        self._current_action_id: str | None = None
        self._actions: dict[str, dict[str, Any]] = {}
        self._accepted_ids: set[str] = set()
        self._last_event: str | None = None
        self._last_returncode: int | None = None
        self._last_interrupt_reason: str | None = None
        self._snapshot_counter = 0
        self._pending_snapshot_id: str | None = None
        self._pending_snapshot_action_seq: int | None = None
        self._last_snapshot_id: str | None = None

    # -- Public, read-only diagnostics -------------------------------------------------

    def approved_entry_ids(self) -> tuple[str, ...]:
        """Return entries allowed by the optional current manual-review store."""

        if self.config.review_store is None:
            return tuple(self._catalog_entries)
        path = self.config.review_store
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise G1SessionError(f"cannot read review store {path}: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != "g1_atom_visual_reviews_v1":
            raise G1SessionError("review store has an unsupported schema")
        reviews = payload.get("reviews")
        if not isinstance(reviews, Mapping):
            raise G1SessionError("review store reviews must be an object")
        approved: list[str] = []
        for entry_id, identity in self._catalog_entries.items():
            record = reviews.get(entry_id)
            if not isinstance(record, Mapping) or record.get("status") != "approved":
                continue
            if record.get("current") is False:
                continue
            if (
                record.get("catalog_file_sha256") != self._catalog_sha256
                or record.get("entry_id") != identity.entry_id
                or record.get("atom_id") != identity.atom_id
                or record.get("version") != identity.version
                or record.get("revision") != identity.revision
                or record.get("content_sha256") != identity.content_sha256
                or record.get("motion_sha256") != identity.motion_sha256
            ):
                continue
            preview_sha = record.get("preview_clip_sha256")
            if not isinstance(preview_sha, str) or _SHA256_RE.fullmatch(preview_sha) is None:
                continue
            approved.append(entry_id)
        return tuple(approved)

    def health_sync(self) -> dict[str, Any]:
        """Report readiness without opening DDS or spawning the runner."""

        try:
            approved_count = len(self.approved_entry_ids())
            approvals_ok = approved_count > 0
        except G1SessionError:
            approved_count = 0
            approvals_ok = False
        runner_ready = (
            self.config.runner.is_file()
            and os.access(self.config.runner, os.X_OK)
            and self.config.catalog.is_file()
            and self.config.limits.is_file()
            and approvals_ok
        )
        with self._condition:
            session = self._session_status_locked()
        return {
            "ok": runner_ready,
            "backend": BACKEND_NAME,
            "session": session,
            "runner_ready": runner_ready,
            "approved_count": approved_count,
            "real_enabled": True,
            "network": self.config.network,
        }

    def cache_status(self) -> dict[str, Any]:
        with self._condition:
            process = self._process
            return {
                "backend": BACKEND_NAME,
                "session": self._session_status_locked(),
                "session_id": self._session_id,
                "pid": (
                    getattr(process, "pid", None)
                    if process is not None and self._process_is_alive(process)
                    else None
                ),
                "ready": self._ready,
                "active_action_id": self._current_action_id,
                "action_seq": self._action_seq,
                "fault": self._fault,
                "last_event": self._last_event,
                "returncode": self._last_returncode,
            }

    def robot_runtime(self) -> dict[str, Any]:
        """Return the legacy crop-face UI's runtime metadata shape."""

        return {
            "enabled": self.real_enabled,
            "simulationMode": False,
            "networkInterface": self.config.network,
            "backend": "g1_atom_realtime",
            "availableActions": [],
            "reasonCodes": [],
        }

    def peek_state_sync(self) -> dict[str, Any]:
        """Return cached state only; this method never starts the native process."""

        with self._condition:
            return self._state_locked()

    # -- Semantic request lifecycle ---------------------------------------------------

    def begin_semantic_request(self, request: Any) -> dict[str, Any]:
        request_id, revision, fingerprint = self._request_identity(request)
        owner = threading.get_ident()
        with self._condition:
            if self._closed:
                raise G1SessionConflictError("G1 session client is closed")
            if self._bound_request is not None:
                raise G1SessionConflictError("another semantic request is already bound")
            if self._process_alive_locked():
                raise G1SessionConflictError("a native G1 session is still active")
            digest = hashlib.sha256(
                _canonical_bytes(
                    {
                        "request_id": request_id,
                        "revision": revision,
                        "fingerprint_sha256": fingerprint,
                    }
                )
            ).hexdigest()
            self._bound_request = (request_id, revision, fingerprint)
            self._owner_thread = owner
            self._session_id = f"g1sem-{digest[:40]}"
            self._started_once = False
            self._ready = False
            self._session_completed = False
            self._expected_stop = False
            self._fault = None
            self._native_error = None
            self._q = None
            self._action_seq = 0
            self._state_seq += 1
            self._current_action_id = None
            self._actions = {}
            self._accepted_ids = set()
            self._last_event = "request_bound"
            self._last_returncode = None
            self._last_interrupt_reason = None
            self._pending_snapshot_id = None
            self._pending_snapshot_action_seq = None
            self._last_snapshot_id = None
            return {
                "session_id": self._session_id,
                "request_id": request_id,
                "revision": revision,
            }

    def finish_semantic_request(
        self,
        result_or_none: Any,
        error: BaseException | None = None,
    ) -> None:
        del result_or_none
        self._require_bound_owner()
        with self._condition:
            process = self._process
            recovery_completed = any(
                record.get("kind") == "recovery"
                and record.get("status") == "completed"
                for record in self._actions.values()
            )
            graceful = error is None and (
                self._session_completed or recovery_completed
            )
        if process is not None and self._process_is_alive(process):
            if graceful:
                try:
                    process.wait(timeout=self.config.process_stop_timeout_s)
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            if self._process_is_alive(process):
                self._stop_process(process, reason="semantic_request_finished")
        with self._condition:
            self._bound_request = None
            self._owner_thread = None
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            process = self._process
        if process is not None and self._process_is_alive(process):
            self._stop_process(process, reason="client_closed")
        with self._condition:
            self._bound_request = None
            self._owner_thread = None
            self._condition.notify_all()

    # -- Simulator-shaped execution API ----------------------------------------------

    def state_sync(self) -> dict[str, Any]:
        self._require_bound_owner()
        with self._condition:
            if not self._started_once:
                self._start_process_locked()
            deadline = self._monotonic() + self.config.startup_timeout_s
            while not self._ready:
                self._raise_fault_or_exit_locked("waiting for session_ready")
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    process = self._process
                    self._fault = "timed out waiting for native session_ready"
                    self._condition.notify_all()
                    break
                self._condition.wait(min(remaining, 0.1))
            if not self._ready:
                process = self._process
            else:
                if self._current_action_id is None and not self._session_completed:
                    return self._snapshot_state_locked()
                return self._state_locked()
        if process is not None and self._process_is_alive(process):
            self._stop_process(process, reason="startup_timeout")
        raise G1SessionError(self._fault or "native session did not become ready")

    def post_semantic_action_sync(
        self,
        flat: Mapping[str, Any],
        kind: str,
        atom_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_bound_owner()
        if not isinstance(flat, Mapping):
            raise G1SessionError("flat action payload must be an object")
        if kind not in {"semantic", "recovery"}:
            raise G1SessionError("semantic action kind must be semantic or recovery")
        action_id = _safe_token(flat.get("id"), "flat.id")
        if flat.get("mode") != "enqueue" or flat.get("require_empty") is not True:
            raise G1SessionError("G1 actions require mode=enqueue and require_empty=true")
        expected_seq = _nonnegative_int(
            flat.get("expected_action_seq"), "flat.expected_action_seq"
        )
        payload_sha256 = _lower_sha256(
            flat.get("payload_sha256"), "flat.payload_sha256"
        )
        if _flat_payload_sha256(flat) != payload_sha256:
            raise G1SessionError("flat.payload_sha256 does not match the exact flat payload")

        if kind == "semantic":
            identity = self._validate_semantic_ref(atom_ref)
            command: dict[str, Any] = {
                "type": "append",
                "session_id": self._session_id,
                "action_id": action_id,
                "entry_id": identity.entry_id,
                "expect_version": identity.version,
                "expect_revision": identity.revision,
                "expect_content_sha256": identity.content_sha256,
                "expected_action_seq": expected_seq,
                "payload_sha256": payload_sha256,
            }
            entry_id: str | None = identity.entry_id
        else:
            self._validate_recovery_ref(atom_ref)
            command = {
                "type": "end_turn",
                "session_id": self._session_id,
                "action_id": action_id,
                "expected_action_seq": expected_seq,
                "payload_sha256": payload_sha256,
            }
            entry_id = None

        with self._condition:
            self._require_ready_idle_locked(expected_seq, action_id)
            process = self._process
            if process is None or process.stdin is None:
                raise G1SessionError("native session stdin is unavailable")
            self._actions[action_id] = {
                "id": action_id,
                "status": "scheduled",
                "payload_sha256": payload_sha256,
                "kind": kind,
                **({"entry_id": entry_id} if entry_id is not None else {}),
            }
            line = json.dumps(
                command,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if len(line.encode("utf-8")) > 64 * 1024:
                self._actions.pop(action_id, None)
                raise G1SessionError("native session command exceeds 64 KiB")
            try:
                process.stdin.write(line + "\n")
                process.stdin.flush()
            except Exception as exc:
                self._actions[action_id]["status"] = "failed"
                self._fault = f"cannot write native session command: {exc}"
                self._condition.notify_all()
                raise G1SessionError(self._fault) from exc

            deadline = self._monotonic() + self.config.action_timeout_s
            while action_id not in self._accepted_ids:
                record = self._actions[action_id]
                if record["status"] in {"failed", "canceled"}:
                    raise G1SessionError(
                        self._fault or f"native action {action_id} was not accepted"
                    )
                self._raise_fault_or_exit_locked(f"waiting for action_accepted {action_id}")
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    self._fault = f"timed out waiting for action_accepted {action_id}"
                    record["status"] = "failed"
                    self._condition.notify_all()
                    raise G1SessionError(self._fault)
                self._condition.wait(min(remaining, 0.1))
            return {
                "accepted": True,
                "id": action_id,
                "payload_sha256": payload_sha256,
                "action_seq": self._action_seq,
            }

    def action_sync(self, action_id: str) -> dict[str, Any]:
        self._require_bound_owner()
        action_id = _safe_token(action_id, "action_id")
        with self._condition:
            record = self._actions.get(action_id)
            if record is None:
                raise G1SessionError(f"unknown native action id {action_id}")
            return dict(record)

    def interrupt_active_session_sync(self, reason: str = "interrupt") -> bool:
        """Signal fail-safe release; intentionally callable from another thread."""

        clean_reason = str(reason).strip()[:256] or "interrupt"
        with self._condition:
            process = self._process
            if process is None or not self._process_is_alive(process):
                return False
            self._last_interrupt_reason = clean_reason
            self._expected_stop = True
            self._ready = False
            current = self._current_action_id
            if current is not None and current in self._actions:
                self._actions[current]["status"] = "canceled"
            self._current_action_id = None
            self._condition.notify_all()
        self._stop_process(process, reason=clean_reason)
        return True

    # -- Validation -------------------------------------------------------------------

    @staticmethod
    def _request_identity(request: Any) -> tuple[str, int, str]:
        if isinstance(request, Mapping):
            request_id = request.get("request_id")
            revision = request.get("revision")
            fingerprint = request.get("fingerprint_sha256")
        else:
            request_id = getattr(request, "request_id", None)
            revision = getattr(request, "revision", None)
            fingerprint = getattr(request, "fingerprint_sha256", None)
        request_id = _safe_token(request_id, "request.request_id")
        revision = _positive_int(revision, "request.revision")
        fingerprint = _lower_sha256(fingerprint, "request.fingerprint_sha256")
        return request_id, revision, fingerprint

    def _validate_semantic_ref(
        self,
        atom_ref: Mapping[str, Any],
        *,
        require_approved: bool = True,
    ) -> _CatalogIdentity:
        if not isinstance(atom_ref, Mapping) or set(atom_ref) != _SEMANTIC_REF_FIELDS:
            raise G1SessionError("semantic atom_ref does not have the exact identity shape")
        if atom_ref.get("catalog_content_sha256") != self._catalog_sha256:
            raise G1SessionError("semantic atom_ref catalog hash mismatch")
        entry_id = _lower_sha256(atom_ref.get("entry_id"), "atom_ref.entry_id")
        identity = self._catalog_entries.get(entry_id)
        if identity is None:
            raise G1SessionError("semantic atom_ref entry_id is absent from the catalog")
        expected = {
            "entry_id": identity.entry_id,
            "atom_id": identity.atom_id,
            "version": identity.version,
            "revision": identity.revision,
            "content_sha256": identity.content_sha256,
            "catalog_content_sha256": self._catalog_sha256,
        }
        if dict(atom_ref) != expected:
            raise G1SessionError("semantic atom_ref does not match the exact catalog row")
        if require_approved and entry_id not in self.approved_entry_ids():
            raise G1SessionError("semantic atom_ref is not in the current approved pool")
        return identity

    def _validate_recovery_ref(self, atom_ref: Mapping[str, Any]) -> None:
        if not isinstance(atom_ref, Mapping) or set(atom_ref) != _RECOVERY_REF_FIELDS:
            raise G1SessionError("recovery atom_ref does not have the exact identity shape")
        if atom_ref.get("target") != "catalog_canonical_neutral":
            raise G1SessionError("recovery atom_ref target is not canonical neutral")
        if atom_ref.get("catalog_content_sha256") != self._catalog_sha256:
            raise G1SessionError("recovery atom_ref catalog hash mismatch")
        if atom_ref.get("neutral_motion_sha256") != self._neutral_motion_sha256:
            raise G1SessionError("recovery atom_ref neutral motion hash mismatch")
        _nonnegative_int(atom_ref.get("tail_guard_us"), "recovery atom_ref.tail_guard_us")
        self._validate_semantic_ref(atom_ref.get("after"), require_approved=False)

    def _require_bound_owner(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            if self._closed:
                raise G1SessionConflictError("G1 session client is closed")
            if self._bound_request is None or self._owner_thread is None:
                raise G1SessionConflictError("no semantic request is bound")
            if owner != self._owner_thread:
                raise G1SessionConflictError(
                    "semantic request operations must stay on their bound thread"
                )

    def _require_ready_idle_locked(self, expected_seq: int, action_id: str) -> None:
        if not self._ready or self._fault is not None:
            raise G1SessionConflictError("native G1 session is not ready")
        if self._session_completed or not self._process_alive_locked():
            raise G1SessionConflictError("native G1 session is no longer active")
        if self._current_action_id is not None:
            raise G1SessionConflictError("native G1 session is not empty")
        if expected_seq != self._action_seq:
            raise G1SessionConflictError(
                "flat.expected_action_seq does not match native session state"
            )
        if action_id in self._actions:
            raise G1SessionConflictError("native action id was already used")

    def _snapshot_state_locked(self) -> dict[str, Any]:
        """CAS-read a fresh measured pose without changing native action_seq."""

        if self._current_action_id is not None:
            return self._state_locked()
        if self._pending_snapshot_id is not None:
            raise G1SessionConflictError("another native state snapshot is pending")
        process = self._process
        if process is None or process.stdin is None or not self._process_is_alive(process):
            raise G1SessionError("native session is unavailable for a state snapshot")
        self._snapshot_counter += 1
        request_id = f"snapshot-{self._snapshot_counter}"
        _safe_token(request_id, "snapshot.request_id")
        expected_seq = self._action_seq
        command = {
            "type": "snapshot",
            "session_id": self._session_id,
            "request_id": request_id,
            "expected_action_seq": expected_seq,
        }
        self._pending_snapshot_id = request_id
        self._pending_snapshot_action_seq = expected_seq
        line = json.dumps(
            command,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except Exception as exc:
            self._pending_snapshot_id = None
            self._pending_snapshot_action_seq = None
            self._ready = False
            self._fault = f"cannot write native state snapshot request: {exc}"
            self._condition.notify_all()
            self._send_signal(process, signal.SIGINT)
            raise G1SessionError(self._fault) from exc

        deadline = self._monotonic() + self.config.action_timeout_s
        while self._last_snapshot_id != request_id:
            self._raise_fault_or_exit_locked(
                f"waiting for state_snapshot {request_id}"
            )
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                self._pending_snapshot_id = None
                self._pending_snapshot_action_seq = None
                self._ready = False
                self._fault = f"timed out waiting for state_snapshot {request_id}"
                self._condition.notify_all()
                self._send_signal(process, signal.SIGINT)
                raise G1SessionError(self._fault)
            self._condition.wait(min(remaining, 0.1))
        return self._state_locked()

    # -- Native process and event parser ---------------------------------------------

    def _start_process_locked(self) -> None:
        if self._session_id is None or self._bound_request is None:
            raise G1SessionConflictError("no semantic request is bound")
        if self._started_once:
            raise G1SessionConflictError("native session cannot be restarted")
        self._started_once = True
        idle = format(float(self.config.idle_timeout_s), ".12g")
        command = [
            str(self.config.runner),
            f"--catalog={self.config.catalog}",
            f"--limits={self.config.limits}",
            "--execute",
            f"--network={self.config.network}",
            "--realtime-session",
            f"--session-id={self._session_id}",
            f"--session-idle-timeout={idle}",
        ]
        try:
            process = self._popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            self._fault = f"cannot start native G1 session: {exc}"
            raise G1SessionError(self._fault) from exc
        if process.stdin is None or process.stdout is None:
            self._fault = "native G1 session did not expose stdin/stdout pipes"
            try:
                process.terminate()
            except Exception:
                pass
            raise G1SessionError(self._fault)
        self._process = process
        self._last_event = "process_started"
        self._reader = threading.Thread(
            target=self._read_stdout,
            args=(process,),
            name=f"g1-session-reader-{self._session_id[-8:]}",
            daemon=True,
        )
        self._reader.start()

    def _read_stdout(self, process: Any) -> None:
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = raw.rstrip("\n")
                if line.endswith("\r"):
                    raise G1SessionProtocolError("native status lines must use LF, not CRLF")
                if not line:
                    raise G1SessionProtocolError("native emitted an empty status line")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise G1SessionProtocolError("native emitted invalid NDJSON") from exc
                self._handle_event(event)
        except Exception as exc:
            with self._condition:
                if self._process is process and self._fault is None:
                    self._fault = f"{type(exc).__name__}: {exc}"[:512]
                    self._fail_current_locked()
                    self._condition.notify_all()
            if self._process_is_alive(process):
                self._send_signal(process, signal.SIGINT)
        finally:
            try:
                returncode = process.wait()
            except Exception:
                returncode = process.poll()
            with self._condition:
                if self._process is process:
                    self._last_returncode = returncode
                    if (
                        not self._expected_stop
                        and not self._session_completed
                        and self._fault is None
                    ):
                        self._fault = (
                            f"native G1 session exited unexpectedly with returncode {returncode}"
                        )
                        self._fail_current_locked()
                    self._condition.notify_all()

    def _handle_event(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise G1SessionProtocolError("native status must be a JSON object")
        if value.get("schema_version") != STATUS_SCHEMA_VERSION:
            raise G1SessionProtocolError("native status schema_version mismatch")
        event = value.get("event")
        if not isinstance(event, str) or not event:
            raise G1SessionProtocolError("native status event is missing")
        with self._condition:
            self._last_event = event
            self._state_seq += 1
            if event == "session_ready":
                self._event_session_ready_locked(value)
            elif event in {"action_accepted", "action_started", "action_completed"}:
                self._event_action_locked(event, value)
            elif event == "state_snapshot":
                self._event_snapshot_locked(value)
            elif event == "completed":
                self._require_event_session_locked(value)
                self._session_completed = True
                self._ready = False
            elif event == "error":
                code = value.get("code")
                message = value.get("message")
                if not isinstance(code, str) or not code or not isinstance(message, str):
                    raise G1SessionProtocolError("native error status is malformed")
                self._native_error = {"code": code, "message": message}
                self._fault = f"native {code}: {message}"[:512]
                self._ready = False
                self._fail_current_locked()
            elif event == "fail_safe_release":
                # A following error status carries the authoritative fault.  An
                # expected SIGINT may emit only this status before exiting.
                if self._session_id is not None and "session_id" in value:
                    self._require_event_session_locked(value)
            self._condition.notify_all()

    def _event_session_ready_locked(self, value: Mapping[str, Any]) -> None:
        self._require_event_session_locked(value)
        if self._ready or self._q is not None:
            raise G1SessionProtocolError("native emitted duplicate session_ready")
        action_seq = _nonnegative_int(value.get("action_seq"), "session_ready.action_seq")
        measured_q = _finite_q(value.get("measured_q"))
        self._action_seq = action_seq
        self._q = measured_q
        self._ready = True

    def _event_snapshot_locked(self, value: Mapping[str, Any]) -> None:
        self._require_event_session_locked(value)
        request_id = _safe_token(
            value.get("request_id"), "state_snapshot.request_id"
        )
        if self._pending_snapshot_id is None or request_id != self._pending_snapshot_id:
            raise G1SessionProtocolError(
                "native state_snapshot does not match the pending request_id"
            )
        action_seq = _nonnegative_int(
            value.get("action_seq"), "state_snapshot.action_seq"
        )
        if action_seq != self._pending_snapshot_action_seq:
            raise G1SessionProtocolError(
                "native state_snapshot action_seq does not match snapshot CAS"
            )
        self._q = _finite_q(value.get("measured_q"))
        self._action_seq = action_seq
        self._last_snapshot_id = request_id
        self._pending_snapshot_id = None
        self._pending_snapshot_action_seq = None

    def _event_action_locked(self, event: str, value: Mapping[str, Any]) -> None:
        self._require_event_session_locked(value)
        action_id = _safe_token(value.get("action_id"), f"{event}.action_id")
        payload_sha256 = _lower_sha256(
            value.get("payload_sha256"), f"{event}.payload_sha256"
        )
        action_seq = _nonnegative_int(value.get("action_seq"), f"{event}.action_seq")
        record = self._actions.get(action_id)
        if record is None:
            raise G1SessionProtocolError(f"native {event} referenced an unknown action")
        if record.get("payload_sha256") != payload_sha256:
            raise G1SessionProtocolError(f"native {event} payload_sha256 mismatch")
        if action_seq < self._action_seq:
            raise G1SessionProtocolError(f"native {event} action_seq moved backwards")
        entry_id = value.get("entry_id")
        expected_entry = record.get("entry_id")
        if expected_entry is not None and entry_id != expected_entry:
            raise G1SessionProtocolError(f"native {event} entry_id mismatch")
        if expected_entry is None and entry_id not in (None, ""):
            raise G1SessionProtocolError(f"native recovery {event} unexpectedly named an entry")

        if event == "action_accepted":
            if action_id in self._accepted_ids or self._current_action_id is not None:
                raise G1SessionProtocolError("native action_accepted order is invalid")
            self._accepted_ids.add(action_id)
            self._current_action_id = action_id
            record["status"] = "scheduled"
        elif event == "action_started":
            if action_id not in self._accepted_ids or self._current_action_id != action_id:
                raise G1SessionProtocolError("native action_started preceded acceptance")
            record["status"] = "started"
        else:
            if action_id not in self._accepted_ids or self._current_action_id != action_id:
                raise G1SessionProtocolError("native action_completed order is invalid")
            kind = value.get("kind")
            if kind != record.get("kind"):
                raise G1SessionProtocolError("native action_completed kind mismatch")
            measured_q = _finite_q(value.get("measured_q"))
            self._q = measured_q
            record["status"] = "completed"
            self._current_action_id = None
        self._action_seq = action_seq

    def _require_event_session_locked(self, value: Mapping[str, Any]) -> None:
        if value.get("session_id") != self._session_id:
            raise G1SessionProtocolError("native status session_id mismatch")

    # -- Process cleanup --------------------------------------------------------------

    def _stop_process(self, process: Any, *, reason: str) -> None:
        with self._condition:
            if self._process is process:
                self._expected_stop = True
                self._ready = False
                self._last_interrupt_reason = str(reason)[:256]
        self._send_signal(process, signal.SIGINT)
        try:
            process.wait(timeout=self.config.process_stop_timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            if not self._process_is_alive(process):
                return
        self._send_signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=self.config.process_stop_timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            if not self._process_is_alive(process):
                return
        try:
            process.kill()
        except Exception:
            return
        try:
            process.wait(timeout=2.0)
        except Exception:
            pass

    def _send_signal(self, process: Any, sig: signal.Signals) -> None:
        try:
            if self._real_popen:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except (ProcessLookupError, AttributeError):
            pass

    @staticmethod
    def _process_is_alive(process: Any) -> bool:
        try:
            return process.poll() is None
        except Exception:
            return False

    def _process_alive_locked(self) -> bool:
        return self._process is not None and self._process_is_alive(self._process)

    def _raise_fault_or_exit_locked(self, context: str) -> None:
        if self._fault is not None:
            raise G1SessionError(self._fault)
        if self._process is None:
            raise G1SessionError(f"native process disappeared while {context}")
        returncode = self._process.poll()
        if returncode is not None and not self._session_completed:
            # stdout is ordered before EOF.  Give the reader the chance to
            # consume a final structured `error` record instead of replacing
            # it with a less useful generic return-code failure.
            if self._reader is not None and self._reader.is_alive():
                return
            raise G1SessionError(
                f"native process exited with returncode {returncode} while {context}"
            )

    def _fail_current_locked(self) -> None:
        for record in self._actions.values():
            if record.get("status") not in {
                "completed",
                "canceled",
                "failed",
            }:
                record["status"] = "failed"
        self._current_action_id = None

    def _session_status_locked(self) -> str:
        if self._closed:
            return "closed"
        if self._fault is not None:
            return "faulted"
        if self._session_completed:
            return "completed"
        if self._current_action_id is not None:
            return "running"
        if self._ready:
            return "ready"
        if self._started_once and self._process_alive_locked():
            return "starting"
        if self._bound_request is not None:
            return "bound"
        return "idle"

    def _state_locked(self) -> dict[str, Any]:
        current = None
        if self._current_action_id is not None:
            record = self._actions.get(self._current_action_id)
            current = None if record is None else dict(record)
        return {
            "current": current,
            "queued": [],
            "fault": self._fault,
            "action_seq": self._action_seq,
            "q": {} if self._q is None else dict(self._q),
            "seq": self._state_seq,
        }


__all__ = [
    "BACKEND_NAME",
    "G1SessionClient",
    "G1SessionConfig",
    "G1SessionConflictError",
    "G1SessionError",
    "G1SessionProtocolError",
    "STATUS_SCHEMA_VERSION",
]
