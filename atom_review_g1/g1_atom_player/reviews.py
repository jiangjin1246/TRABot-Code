from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping

from .catalog import AtomCatalog, AtomRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    """Content-addressed visual-review decisions.

    A decision automatically becomes stale if the catalog file, Atom identity,
    or canonical motion changes.
    """

    def __init__(
        self,
        path: str | Path,
        catalog: AtomCatalog,
        preview_sha256_by_atom: Mapping[str, str],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.catalog = catalog
        self.preview_sha256_by_atom = dict(preview_sha256_by_atom)
        self._lock = threading.RLock()
        self._payload = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": "g1_atom_visual_reviews_v1", "reviews": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read review store {self.path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != "g1_atom_visual_reviews_v1":
            raise ValueError("review store has an unsupported schema")
        if not isinstance(payload.get("reviews"), dict):
            raise ValueError("review store reviews must be an object")
        return payload

    def _current(self, atom: AtomRecord, record: Mapping[str, Any]) -> bool:
        return (
            record.get("catalog_file_sha256") == self.catalog.file_sha256
            and record.get("entry_id") == atom.entry_id
            and record.get("atom_id") == atom.atom_id
            and record.get("version") == atom.version
            and record.get("revision") == atom.revision
            and record.get("content_sha256") == atom.content_sha256
            and record.get("motion_sha256") == atom.motion_sha256
            and record.get("preview_clip_sha256")
            == self.preview_sha256_by_atom.get(atom.atom_id)
        )

    def get(self, atom: AtomRecord) -> dict[str, Any]:
        with self._lock:
            raw = self._payload["reviews"].get(atom.entry_id)
            if not isinstance(raw, Mapping):
                return {"status": "pending", "current": True, "note": ""}
            result = dict(raw)
            current = self._current(atom, raw)
            result["current"] = current
            if not current:
                result["status"] = "stale"
            return result

    def set(
        self,
        atom: AtomRecord,
        *,
        status: str,
        note: str = "",
        reviewer: str = "",
    ) -> dict[str, Any]:
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError("status must be pending, approved, or rejected")
        preview_sha256 = self.preview_sha256_by_atom.get(atom.atom_id)
        if status == "approved" and not preview_sha256:
            raise ValueError("an exact verified preview is required before approval")
        record = {
            "status": status,
            "note": str(note).strip()[:2000],
            "reviewer": str(reviewer).strip()[:200],
            "reviewed_at": _utc_now(),
            "catalog_file_sha256": self.catalog.file_sha256,
            "entry_id": atom.entry_id,
            "atom_id": atom.atom_id,
            "version": atom.version,
            "revision": atom.revision,
            "content_sha256": atom.content_sha256,
            "motion_sha256": atom.motion_sha256,
            "preview_clip_sha256": preview_sha256,
        }
        with self._lock:
            self._payload["reviews"][atom.entry_id] = record
            self._write_locked()
        return {**record, "current": True}

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            self._payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
