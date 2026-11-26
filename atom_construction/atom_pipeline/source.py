"""Source discovery, provenance and speaker/session-stratified sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable, Sequence


_BEAT2_STEM = re.compile(
    r"^(?P<speaker_id>\d+)_(?P<speaker_name>[^_]+)_"
    r"(?P<session>\d+)_(?P<segment_a>\d+)_(?P<segment_b>\d+)$"
)


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Identity recoverable from the retarget batch filename."""

    speaker_id: str
    speaker_name: str
    session_id: str
    segment_id: str
    sequence_id: str

    @property
    def speaker_key(self) -> str:
        return f"{self.speaker_id}_{self.speaker_name}"


@dataclass(frozen=True, slots=True)
class SourceRecord:
    path: Path
    relative_path: str
    identity: SourceIdentity

    def as_dict(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "speaker_id": self.identity.speaker_id,
            "speaker_name": self.identity.speaker_name,
            "speaker_key": self.identity.speaker_key,
            "session_id": self.identity.session_id,
            "segment_id": self.identity.segment_id,
            "sequence_id": self.identity.sequence_id,
        }


def parse_source_identity(path: str | Path) -> SourceIdentity:
    """Parse ``speaker_name_session_segment_segment`` without guessing demographics."""

    stem = Path(path).stem
    match = _BEAT2_STEM.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"{Path(path).name!r} does not match "
            "<speaker_id>_<speaker_name>_<session>_<segment>_<segment>.pkl"
        )
    values = match.groupdict()
    if values["segment_a"] != values["segment_b"]:
        raise ValueError(f"{Path(path).name!r} has inconsistent segment suffixes")
    return SourceIdentity(
        speaker_id=values["speaker_id"],
        speaker_name=values["speaker_name"],
        session_id=values["session"],
        segment_id=values["segment_a"],
        sequence_id=stem,
    )


def discover_sources(
    source: str | Path,
    *,
    recursive: bool = False,
    strict_names: bool = True,
) -> list[SourceRecord]:
    """Discover retarget PKLs and retain portable paths relative to ``source``."""

    root = Path(source).expanduser().resolve()
    if root.is_file():
        paths = [root]
        relative_root = root.parent
    elif root.is_dir():
        paths = sorted(root.rglob("*.pkl") if recursive else root.glob("*.pkl"))
        relative_root = root
    else:
        raise FileNotFoundError(root)

    records: list[SourceRecord] = []
    errors: list[str] = []
    for path in paths:
        try:
            identity = parse_source_identity(path)
        except ValueError as exc:
            if strict_names:
                errors.append(str(exc))
            continue
        records.append(
            SourceRecord(
                path=path,
                relative_path=path.relative_to(relative_root).as_posix(),
                identity=identity,
            )
        )
    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"invalid BEAT2 source names ({len(errors)}): {preview}")
    if not records:
        raise ValueError(f"no matching .pkl files found under {root}")
    return records


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**31 - 1, value


def _stable_shuffle(items: Sequence[SourceRecord], seed: int, salt: str) -> list[SourceRecord]:
    result = sorted(items, key=lambda item: item.relative_path)
    salt_value = int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:8], "big")
    random.Random(seed ^ salt_value).shuffle(result)
    return result


def select_sources(
    records: Sequence[SourceRecord],
    *,
    speakers: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    max_speakers: int | None = 10,
    sources_per_speaker: int | None = 2,
    max_sources: int | None = None,
    seed: int = 20260915,
) -> list[SourceRecord]:
    """Select sources fairly across speaker and session.

    Selection is deterministic.  It round-robins sessions inside each speaker,
    then speakers globally; this prevents the lexicographic single-speaker bug
    in the original extractor and makes session 1 reachable despite containing
    far fewer, much longer files.
    """

    speaker_filter = {str(item) for item in speakers or ()}
    session_filter = {str(item) for item in sessions or ()}
    filtered = [
        item
        for item in records
        if (not speaker_filter or item.identity.speaker_id in speaker_filter
            or item.identity.speaker_key in speaker_filter)
        and (not session_filter or item.identity.session_id in session_filter)
    ]
    if not filtered:
        raise ValueError("source filters selected no records")

    by_speaker: dict[str, list[SourceRecord]] = {}
    for item in filtered:
        by_speaker.setdefault(item.identity.speaker_key, []).append(item)
    speaker_keys = sorted(
        by_speaker,
        key=lambda key: (_natural_key(key.split("_", 1)[0]), key),
    )
    # Do not make a capped pilot silently prefer low numeric speaker IDs.
    # The deterministic shuffle is recorded through ``seed`` in the manifest.
    random.Random(seed).shuffle(speaker_keys)
    if max_speakers is not None:
        if max_speakers <= 0:
            max_speakers = None
        else:
            speaker_keys = speaker_keys[:max_speakers]

    queues: dict[str, list[SourceRecord]] = {}
    for speaker_key in speaker_keys:
        by_session: dict[str, list[SourceRecord]] = {}
        for item in by_speaker[speaker_key]:
            by_session.setdefault(item.identity.session_id, []).append(item)
        session_queues = {
            key: _stable_shuffle(value, seed, f"{speaker_key}:{key}")
            for key, value in by_session.items()
        }
        ordered: list[SourceRecord] = []
        session_keys = sorted(session_queues, key=_natural_key)
        while any(session_queues.values()):
            for session_key in session_keys:
                queue = session_queues[session_key]
                if queue:
                    ordered.append(queue.pop())
        if sources_per_speaker is not None and sources_per_speaker > 0:
            ordered = ordered[:sources_per_speaker]
        queues[speaker_key] = ordered

    selected: list[SourceRecord] = []
    while any(queues.values()):
        for speaker_key in speaker_keys:
            queue = queues[speaker_key]
            if queue:
                selected.append(queue.pop(0))
                if max_sources is not None and max_sources > 0 and len(selected) >= max_sources:
                    return selected
    return selected


def load_source_manifest(
    path: str | Path,
    *,
    source_root: str | Path,
) -> list[SourceRecord]:
    """Rehydrate a portable source manifest against an explicit local root."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "beat2_retarget_source_manifest_v1":
        raise ValueError(f"{manifest_path}: unsupported source manifest")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{manifest_path}: sources must be a non-empty list")
    root = Path(source_root).expanduser().resolve()
    records: list[SourceRecord] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict) or not isinstance(raw.get("relative_path"), str):
            raise ValueError(f"{manifest_path}: sources[{index}] is invalid")
        relative = Path(raw["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{manifest_path}: sources[{index}] path is not portable")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{manifest_path}: sources[{index}] escapes source_root") from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        identity = parse_source_identity(resolved)
        records.append(SourceRecord(resolved, relative.as_posix(), identity))
    return records


def read_source_manifest_metadata(path: str | Path) -> dict[str, object]:
    """Return immutable replay provenance for an existing source manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "beat2_retarget_source_manifest_v1":
        raise ValueError(f"{manifest_path}: unsupported source manifest")
    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError(f"{manifest_path}: selection must be an object")
    return {
        "replayed_source_manifest": manifest_path.name,
        "replayed_source_manifest_sha256": file_sha256(manifest_path),
        "original_selection": selection,
    }


def write_source_manifest(
    path: str | Path,
    records: Sequence[SourceRecord],
    *,
    selection: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "beat2_retarget_source_manifest_v1",
        "count": len(records),
        "selection": selection or {},
        "sources": [record.as_dict() for record in records],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
