from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from .catalog import EXPECTED_JOINTS, AtomCatalog, AtomRecord, load_catalog
from .config import AppConfig
from .reviews import ReviewStore
from .validation import load_limits, validate_atom


PINNED_PREVIEW_MANIFEST_SHA256 = (
    "31b79368c9ef142163901717a596ce580be88b0d4a38c96efe8d520b808a3296"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: str, **fields: Any) -> None:
        record = {"timestamp": _utc_now(), "event": event, **fields}
        body = json.dumps(
            record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(body + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def _strict_verify(config: AppConfig, catalog: AtomCatalog) -> None:
    source = config.sim_source_dir
    if not source.is_dir():
        raise RuntimeError(f"strict validator source is missing: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from web_overlay.semantic_catalog import load_semantic_catalog  # type: ignore

    verified = load_semantic_catalog(config.catalog_path)
    if verified.content_sha256 != catalog.file_sha256:
        raise RuntimeError("strict catalog validator returned a different SHA-256")
    if len(verified.rows) != len(catalog.entries):
        raise RuntimeError("strict catalog validator returned a different entry count")


@dataclass(frozen=True, slots=True)
class PreviewAsset:
    path: Path
    sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _preview_map(root: Path, catalog: AtomCatalog) -> dict[str, PreviewAsset]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != PINNED_PREVIEW_MANIFEST_SHA256:
            raise RuntimeError("preview manifest does not match the pinned reviewed release")
        payload = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError):
        return {}
    rows: Any = None
    if isinstance(payload, Mapping):
        for key in ("atoms", "entries", "clips"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    if not isinstance(rows, list):
        return {}
    resolved_root = root.resolve()
    result: dict[str, PreviewAsset] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        atom_id = row.get("atom_id")
        relative = row.get("clip_file") or row.get("video_file") or row.get("clip")
        if not isinstance(atom_id, str) or not isinstance(relative, str):
            continue
        atom = catalog.by_atom_id.get(atom_id)
        if atom is None:
            continue
        try:
            revision = int(row.get("revision", -1))
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            str(row.get("version", "")) != atom.version
            or revision != atom.revision
            or str(row.get("content_sha256", "")) != atom.content_sha256
        ):
            continue
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        declared_sha256 = str(row.get("clip_sha256", ""))
        if candidate.is_file() and len(declared_sha256) == 64:
            actual_sha256 = _file_sha256(candidate)
            if actual_sha256 != declared_sha256:
                raise RuntimeError(f"preview clip SHA-256 mismatch for {atom_id}")
            if atom_id in result:
                raise RuntimeError(f"duplicate exact preview for {atom_id}")
            result[atom_id] = PreviewAsset(path=candidate, sha256=actual_sha256)
    missing = [atom.atom_id for atom in catalog.entries if atom.atom_id not in result]
    if missing:
        raise RuntimeError(
            f"exact previews are missing for {len(missing)} catalog Atoms; first={missing[0]}"
        )
    return result


class RunManager:
    def __init__(self, config: AppConfig, audit: AuditLog) -> None:
        self.config = config
        self.audit = audit
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._closing = False
        self._state: dict[str, Any] = {
            "status": "idle",
            "mode": "single",
            "entry_id": "",
            "atom_id": "",
            "entry_ids": [],
            "atom_ids": [],
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "output": [],
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def start(self, atom: AtomRecord) -> dict[str, Any]:
        return self._start((atom,), mode="single")

    def start_sequence(self, atoms: tuple[AtomRecord, ...]) -> dict[str, Any]:
        if not 2 <= len(atoms) <= 3:
            raise ValueError("a sequence must contain exactly 2 or 3 Atoms")
        entry_ids = [atom.entry_id for atom in atoms]
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("a sequence cannot contain duplicate entry_ids")
        return self._start(atoms, mode="sequence")

    @staticmethod
    def _identity(atom: AtomRecord) -> dict[str, Any]:
        return {
            "entry_id": atom.entry_id,
            "atom_id": atom.atom_id,
            "version": atom.version,
            "revision": atom.revision,
            "content_sha256": atom.content_sha256,
            "motion_sha256": atom.motion_sha256,
        }

    def _start(self, atoms: tuple[AtomRecord, ...], *, mode: str) -> dict[str, Any]:
        if not atoms:
            raise ValueError("at least one Atom is required")
        identities = [self._identity(atom) for atom in atoms]
        first = atoms[0]
        with self._lock:
            if self._closing:
                raise RuntimeError("service is shutting down; no new Atom can start")
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("another Atom is already running")
            # The durable intent record is written before any process can open DDS.
            # If the disk is unavailable, no robot process is started.
            if mode == "single":
                self.audit.append("real_play_start_requested", **identities[0])
            else:
                self.audit.append(
                    "real_sequence_start_requested",
                    sequence_length=len(atoms),
                    entries=identities,
                )
            command = [
                str(self.config.runner_path),
                f"--catalog={self.config.catalog_path}",
            ]
            for atom in atoms:
                command.extend(
                    [
                        f"--entry-id={atom.entry_id}",
                        f"--expect-version={atom.version}",
                        f"--expect-revision={atom.revision}",
                        f"--expect-content-sha256={atom.content_sha256}",
                    ]
                )
            command.extend(
                [
                    f"--limits={self.config.limits_path}",
                    "--execute",
                    f"--network={self.config.network_interface}",
                ]
            )
            command.extend(f"--confirm-entry-id={atom.entry_id}" for atom in atoms)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._process = process
            self._state = {
                "status": "running",
                "mode": mode,
                "entry_id": first.entry_id,
                "atom_id": first.atom_id,
                "entry_ids": [atom.entry_id for atom in atoms],
                "atom_ids": [atom.atom_id for atom in atoms],
                "entries": identities,
                "sequence_length": len(atoms),
                "pid": process.pid,
                "started_at": _utc_now(),
                "finished_at": None,
                "returncode": None,
                "output": [],
            }
            threading.Thread(target=self._collect, args=(process,), daemon=True).start()
            try:
                if mode == "single":
                    self.audit.append("real_play_started", **identities[0], pid=process.pid)
                else:
                    self.audit.append(
                        "real_sequence_started",
                        sequence_length=len(atoms),
                        entries=identities,
                        pid=process.pid,
                    )
            except Exception:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                raise
            return self.snapshot()

    def _collect(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            with self._lock:
                output = self._state.setdefault("output", [])
                output.append(line)
                if len(output) > 300:
                    del output[:-300]
        returncode = process.wait()
        with self._lock:
            if self._process is not process:
                return
            self._state["status"] = "completed" if returncode == 0 else "failed"
            self._state["returncode"] = returncode
            self._state["finished_at"] = _utc_now()
            snapshot = self.snapshot()
            self._process = None
        if snapshot.get("mode") == "sequence":
            self.audit.append(
                "real_sequence_finished",
                sequence_length=snapshot["sequence_length"],
                entries=snapshot["entries"],
                returncode=returncode,
                status=snapshot["status"],
            )
        else:
            self.audit.append(
                "real_play_finished",
                entry_id=snapshot["entry_id"],
                atom_id=snapshot["atom_id"],
                returncode=returncode,
                status=snapshot["status"],
            )

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError("no Atom is running")
            os.killpg(process.pid, signal.SIGINT)
            self._state["status"] = "canceling"
            snapshot = self.snapshot()
        try:
            if snapshot.get("mode") == "sequence":
                self.audit.append(
                    "real_sequence_cancel_requested",
                    sequence_length=snapshot["sequence_length"],
                    entries=snapshot["entries"],
                    pid=process.pid,
                )
            else:
                self.audit.append(
                    "real_play_cancel_requested",
                    entry_id=snapshot["entry_id"],
                    atom_id=snapshot["atom_id"],
                    pid=process.pid,
                )
        except Exception as exc:
            with self._lock:
                self._state.setdefault("output", []).append(
                    f"audit warning after cancel signal: {exc}"
                )
        return self.snapshot()

    def begin_shutdown(self) -> None:
        """Atomically reject new runs before the HTTP accept loop is stopped."""
        with self._lock:
            self._closing = True

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            process = self._process
            entry_id = self._state.get("entry_id", "")
            atom_id = self._state.get("atom_id", "")
            if process is not None and process.poll() is None:
                self._state["status"] = "canceling"
        if process is None or process.poll() is not None:
            return
        # Safety takes precedence over logging: signal the process group before
        # touching disk, so an audit failure cannot leave a DDS owner orphaned.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self.audit.append(
                "service_shutdown_cancel_requested",
                entry_id=entry_id,
                atom_id=atom_id,
                pid=process.pid,
            )
        except Exception as exc:
            with self._lock:
                self._state.setdefault("output", []).append(
                    f"audit warning after shutdown signal: {exc}"
                )
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Do not let the web process exit and orphan an active DDS owner.
            process.wait()


class AtomPlayerApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.catalog = load_catalog(config.catalog_path)
        _strict_verify(config, self.catalog)
        self.limits = load_limits(config.limits_path)
        self.previews = _preview_map(config.preview_root, self.catalog)
        self.reviews = ReviewStore(
            config.review_store_path,
            self.catalog,
            {atom_id: asset.sha256 for atom_id, asset in self.previews.items()},
        )
        self.audit = AuditLog(config.audit_log_path)
        self.runs = RunManager(config, self.audit)
        self.csrf_token = secrets.token_urlsafe(32)
        self.index_html = (config.root / "web" / "index.html").read_bytes()

    def preview_integrity_ok(self, atom: AtomRecord) -> bool:
        asset = self.previews.get(atom.atom_id)
        if asset is None:
            return False
        try:
            body = asset.path.read_bytes()
        except OSError:
            return False
        return hashlib.sha256(body).hexdigest() == asset.sha256

    def require_preview_integrity(self, atom: AtomRecord) -> PreviewAsset:
        asset = self.previews.get(atom.atom_id)
        if asset is None:
            raise RuntimeError("exact preview is unavailable")
        if not self.preview_integrity_ok(atom):
            raise RuntimeError("preview content changed after startup; review and real play are blocked")
        return asset

    def atom_payload(self, atom: AtomRecord, *, verify_preview: bool = False) -> dict[str, Any]:
        validation = validate_atom(atom, self.limits)
        review = self.reviews.get(atom)
        frames = atom.motion["frames"]
        first_pose = frames[0]["q"]
        last_pose = frames[-1]["q"]
        preview_integrity: bool | None = None
        if verify_preview:
            preview_integrity = self.preview_integrity_ok(atom)
            if not preview_integrity:
                review = {**review, "status": "stale", "current": False}
        return {
            **atom.public_dict(),
            "review": review,
            "validation": validation,
            "first_q": [float(first_pose[joint]) for joint in EXPECTED_JOINTS],
            "last_q": [float(last_pose[joint]) for joint in EXPECTED_JOINTS],
            "preview_available": atom.atom_id in self.previews and preview_integrity is not False,
            "preview_integrity_ok": preview_integrity,
            "preview_clip_sha256": (
                self.previews[atom.atom_id].sha256 if atom.atom_id in self.previews else None
            ),
        }

    def prepare_sequence(self, entry_ids: Any) -> tuple[AtomRecord, ...]:
        if not isinstance(entry_ids, list):
            raise ValueError("entry_ids must be a JSON array")
        if not 2 <= len(entry_ids) <= 3:
            raise ValueError("entry_ids must contain exactly 2 or 3 entries")
        if any(
            not isinstance(entry_id, str)
            or len(entry_id) != 64
            or any(character not in "0123456789abcdef" for character in entry_id)
            for entry_id in entry_ids
        ):
            raise ValueError("each entry_id must be a lowercase 64-character SHA-256")
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("entry_ids must not contain duplicates")

        atoms: list[AtomRecord] = []
        for position, entry_id in enumerate(entry_ids, start=1):
            atom = self.catalog.by_entry_id.get(entry_id)
            if atom is None:
                raise LookupError(f"unknown Atom at sequence position {position}")
            try:
                self.require_preview_integrity(atom)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"sequence position {position} ({atom.atom_id}) failed preview integrity: {exc}"
                ) from exc
            review = self.reviews.get(atom)
            if not review.get("current") or review.get("status") != "approved":
                raise RuntimeError(
                    f"sequence position {position} ({atom.atom_id}) is not a current approved Atom"
                )
            validation = validate_atom(atom, self.limits)
            if not validation["ok"]:
                raise RuntimeError(
                    f"sequence position {position} ({atom.atom_id}) does not pass the configured hardware gate"
                )
            atoms.append(atom)
        return tuple(atoms)

    def native_validate(self, atom: AtomRecord) -> dict[str, Any]:
        python_result = validate_atom(atom, self.limits)
        result: dict[str, Any] = {"python": python_result, "native": None}
        if not self.config.runner_path.is_file() or not os.access(self.config.runner_path, os.X_OK):
            result["native"] = {"ok": False, "error": "native runner is not built"}
            return result
        command = [
            str(self.config.runner_path),
            f"--catalog={self.config.catalog_path}",
            f"--entry-id={atom.entry_id}",
            f"--expect-version={atom.version}",
            f"--expect-revision={atom.revision}",
            f"--expect-content-sha256={atom.content_sha256}",
            f"--limits={self.config.limits_path}",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        result["native"] = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-10000:],
        }
        return result


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "G1AtomPlayer/1"

    @property
    def app(self) -> AtomPlayerApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _request_host_ok(self) -> bool:
        port = self.app.config.port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        host = self.headers.get("Host", "")
        if host not in allowed_hosts:
            self._error(HTTPStatus.FORBIDDEN, "invalid Host header")
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{item}" for item in allowed_hosts}:
            self._error(HTTPStatus.FORBIDDEN, "cross-origin request rejected")
            return False
        return True

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def _body(self) -> dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if size < 0 or size > 64 * 1024:
            raise ValueError("request body is too large")
        raw = self.rfile.read(size)
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _resolve_atom(self, encoded: str) -> AtomRecord:
        identity = unquote(encoded)
        try:
            return self.app.catalog.resolve(identity)
        except KeyError as exc:
            raise LookupError("unknown Atom") from exc

    def do_GET(self) -> None:  # noqa: N802
        if not self._request_host_ok():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                body = self.app.index_html
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "catalog_sha256": self.app.catalog.file_sha256,
                        "preview_manifest_sha256": PINNED_PREVIEW_MANIFEST_SHA256,
                        "atom_count": len(self.app.catalog.entries),
                        "preview_count": len(self.app.previews),
                        "real_enabled": self.app.config.real_enabled,
                        "network_interface": self.app.config.network_interface,
                        "runner_ready": self.app.config.runner_path.is_file()
                        and os.access(self.app.config.runner_path, os.X_OK),
                        "csrf_token": self.app.csrf_token,
                        "run": self.app.runs.snapshot(),
                    },
                )
                return
            if path == "/api/atoms":
                query = parse_qs(parsed.query)
                needle = query.get("q", [""])[0].casefold()
                function = query.get("function", [""])[0]
                review_filter = query.get("review", [""])[0]
                rows = []
                for atom in self.app.catalog.entries:
                    review = self.app.reviews.get(atom)
                    if needle and needle not in f"{atom.index} {atom.atom_id} {atom.function}".casefold():
                        continue
                    if function and atom.function != function:
                        continue
                    if review_filter and review.get("status") != review_filter:
                        continue
                    rows.append(self.app.atom_payload(atom))
                self._json(HTTPStatus.OK, {"ok": True, "atoms": rows, "count": len(rows)})
                return
            if path == "/api/run":
                self._json(HTTPStatus.OK, {"ok": True, "run": self.app.runs.snapshot()})
                return
            if path.startswith("/api/atoms/"):
                suffix = path[len("/api/atoms/") :]
                if suffix.endswith("/video"):
                    atom = self._resolve_atom(suffix[: -len("/video")])
                    video = self.app.previews.get(atom.atom_id)
                    if video is None:
                        self._error(HTTPStatus.NOT_FOUND, "preview is unavailable")
                        return
                    self._file(video)
                    return
                atom = self._resolve_atom(suffix)
                self._json(
                    HTTPStatus.OK,
                    {"ok": True, "atom": self.app.atom_payload(atom, verify_preview=True)},
                )
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except LookupError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _file(self, asset: PreviewAsset) -> None:
        # Buffer and hash the exact bytes that will be returned.  This avoids a
        # hash/open race if a preview path is replaced while the service runs.
        body = asset.path.read_bytes()
        if hashlib.sha256(body).hexdigest() != asset.sha256:
            self._error(
                HTTPStatus.CONFLICT,
                "preview content changed after startup; reload the reviewed release",
            )
            return
        size = len(body)
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw = range_header[6:].split(",", 1)[0]
            first, _, last = raw.partition("-")
            if first:
                start = int(first)
            if last:
                end = min(int(last), size - 1)
            if start < 0 or start >= size or end < start:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header(
            "Content-Type", mimetypes.guess_type(asset.path.name)[0] or "application/octet-stream"
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("ETag", f'"sha256-{asset.sha256}"')
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self.wfile.write(body[start : end + 1])

    def do_POST(self) -> None:  # noqa: N802
        if not self._request_host_ok():
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            return
        if not secrets.compare_digest(
            self.headers.get("X-G1-Atom-Token", ""), self.app.csrf_token
        ):
            self._error(HTTPStatus.FORBIDDEN, "missing or invalid request token")
            return
        path = urlparse(self.path).path
        try:
            payload = self._body()
            if path == "/api/run/cancel":
                self._json(HTTPStatus.ACCEPTED, {"ok": True, "run": self.app.runs.cancel()})
                return
            if path == "/api/run/sequence":
                if not self.app.config.real_enabled:
                    self._error(HTTPStatus.FORBIDDEN, "real mode is disabled in config.env")
                    return
                if set(payload) != {"entry_ids"}:
                    raise ValueError("sequence request body must contain only entry_ids")
                atoms = self.app.prepare_sequence(payload["entry_ids"])
                if not self.app.config.runner_path.is_file() or not os.access(
                    self.app.config.runner_path, os.X_OK
                ):
                    self._error(HTTPStatus.SERVICE_UNAVAILABLE, "native runner is not ready")
                    return
                self._json(
                    HTTPStatus.ACCEPTED,
                    {"ok": True, "run": self.app.runs.start_sequence(atoms)},
                )
                return
            if not path.startswith("/api/atoms/"):
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            suffix = path[len("/api/atoms/") :]
            identity, separator, operation = suffix.rpartition("/")
            if not separator:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            atom = self._resolve_atom(identity)
            if operation == "review":
                status = str(payload.get("status", ""))
                if status == "approved":
                    if payload.get("watched") is not True:
                        raise ValueError("approved review requires watched=true")
                    self.app.require_preview_integrity(atom)
                result = self.app.reviews.set(
                    atom,
                    status=status,
                    note=str(payload.get("note", "")),
                    reviewer=str(payload.get("reviewer", "")),
                )
                self.app.audit.append(
                    "visual_review_updated",
                    entry_id=atom.entry_id,
                    atom_id=atom.atom_id,
                    status=result["status"],
                    reviewer=result["reviewer"],
                )
                self._json(HTTPStatus.OK, {"ok": True, "review": result})
                return
            if operation == "validate":
                result = self.app.native_validate(atom)
                self.app.audit.append(
                    "validation_requested",
                    entry_id=atom.entry_id,
                    atom_id=atom.atom_id,
                    passed=bool(result["python"]["ok"] and result["native"] and result["native"]["ok"]),
                )
                self._json(HTTPStatus.OK, {"ok": True, "validation": result})
                return
            if operation == "play":
                if not self.app.config.real_enabled:
                    self._error(HTTPStatus.FORBIDDEN, "real mode is disabled in config.env")
                    return
                self.app.require_preview_integrity(atom)
                review = self.app.reviews.get(atom)
                if not review.get("current") or review.get("status") != "approved":
                    self._error(HTTPStatus.CONFLICT, "current Atom version is not visually approved")
                    return
                validation = validate_atom(atom, self.app.limits)
                if not validation["ok"]:
                    self._error(HTTPStatus.CONFLICT, "Atom does not pass the configured hardware gate")
                    return
                if payload.get("confirm_atom_id") != atom.atom_id:
                    self._error(HTTPStatus.BAD_REQUEST, "confirm_atom_id must exactly match the selected Atom")
                    return
                if not self.app.config.runner_path.is_file() or not os.access(
                    self.app.config.runner_path, os.X_OK
                ):
                    self._error(HTTPStatus.SERVICE_UNAVAILABLE, "native runner is not ready")
                    return
                self._json(HTTPStatus.ACCEPTED, {"ok": True, "run": self.app.runs.start(atom)})
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except LookupError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: AtomPlayerApp):
        self.app = app
        super().__init__(address, ApiHandler)


def serve(config: AppConfig) -> None:
    app = AtomPlayerApp(config)
    server = AppServer((config.host, config.port), app)
    shutdown_started = threading.Event()

    def request_shutdown(_signum: int, _frame: Any) -> None:
        # BaseServer.shutdown() must run from a thread other than the one that
        # is executing serve_forever(), otherwise it deadlocks.  Background
        # shells may also start Python with SIGINT ignored, so install handlers
        # for both operator interrupts and service-stop SIGTERM explicitly.
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        app.runs.begin_shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        json.dumps(
            {
                "event": "server_started",
                "url": f"http://{config.host}:{config.port}",
                "atom_count": len(app.catalog.entries),
                "catalog_sha256": app.catalog.file_sha256,
                "preview_count": len(app.previews),
                "real_enabled": config.real_enabled,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        app.runs.shutdown()
