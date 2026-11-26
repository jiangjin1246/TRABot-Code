"""Fail-open per-reply provenance and recording-side metric aggregation."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import json
import math
import queue
import re
import threading
from typing import Any, Callable, Mapping
import weakref

_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_PRIVATE_KEY = "_experiment_reply_key"
_FRAME_TAG_MAGIC = object()
_MISSING = object()
_CONTEXT = threading.local()
_SUBMISSION_CONTEXT = threading.local()
_ARRAY_LIMIT = 256
_FRAME_LIMIT = 16
_TEXT_LIMIT = 16_384
_JSON_MAX_DEPTH = 8
_JSON_MAX_ITEMS = 256
_REPLY_STATE_LIMIT = 256
_ACTIONS_PER_REPLY_LIMIT = 256
_SEMANTIC_BINDING_LIMIT = 512
_PENDING_SEMANTIC_SUMMARY_LIMIT = 64


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bounded_text(value: Any) -> tuple[str, bool]:
    text = str(value or "")
    return text[:_TEXT_LIMIT], len(text) > _TEXT_LIMIT


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _json_snapshot(value: Any, *, depth: int = 0) -> Any:
    """Deep, bounded, strict-JSON snapshot; never retain planner-owned objects."""
    if depth > _JSON_MAX_DEPTH:
        return "<depth-limit>"
    if value is None or isinstance(value, (str, bool)):
        return value[:_TEXT_LIMIT] if isinstance(value, str) else value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _JSON_MAX_ITEMS:
                result["_truncated"] = True
                break
            key = str(raw_key)[:256]
            result[key] = _json_snapshot(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _json_snapshot(item, depth=depth + 1)
            for item in value[:_JSON_MAX_ITEMS]
        ]
        if len(value) > _JSON_MAX_ITEMS:
            result.append("<items-truncated>")
        return result
    return str(value)[:_TEXT_LIMIT]


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplyKey:
    session_id: str
    stream_generation: int
    reply_seq: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or _SESSION_RE.fullmatch(self.session_id) is None:
            raise ValueError("session_id has an invalid shape")
        _positive(self.stream_generation, "stream_generation")
        _positive(self.reply_seq, "reply_seq")

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "stream_generation": self.stream_generation,
            "reply_seq": self.reply_seq,
        }


@dataclass(frozen=True, slots=True)
class FrameProvenance:
    reply_key: ReplyKey
    face_source_seq: int | None = None
    media_kind: str | None = None
    media_seq: int | None = None


def _key_from_event(event: Mapping[str, Any]) -> ReplyKey | None:
    raw = event.get("reply_key")
    if isinstance(raw, ReplyKey):
        return raw
    if not isinstance(raw, Mapping):
        return None
    try:
        return ReplyKey(str(raw["session_id"]), int(raw["stream_generation"]), int(raw["reply_seq"]))
    except (KeyError, TypeError, ValueError):
        return None


class TaggedReplyQueue(queue.Queue):
    """A Queue-compatible private tag carrier; upstream ignores the extra key."""

    def __init__(self, hub: "ReplyTraceHub", maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._hub_ref = weakref.ref(hub)

    @classmethod
    def from_queue(cls, source: queue.Queue[Any], hub: "ReplyTraceHub") -> "TaggedReplyQueue":
        target = cls(hub, int(getattr(source, "maxsize", 0)))
        while True:
            try:
                target.put_nowait(source.get_nowait())
            except queue.Empty:
                return target

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        hub = self._hub_ref()
        if isinstance(item, dict) and _PRIVATE_KEY not in item and hub is not None:
            if str(item.get("kind", "")) in {"audio", "turn_complete", "interrupted"}:
                key = hub.current_reply()
                if key is not None:
                    item = {**item, _PRIVATE_KEY: key}
        super().put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        # The prior item's context ends before this potentially blocking wait.
        _CONTEXT.value = None
        item = super().get(block=block, timeout=timeout)
        key = item.get(_PRIVATE_KEY) if isinstance(item, dict) else None
        hub = self._hub_ref()
        _CONTEXT.value = (hub, key) if hub is not None and isinstance(key, ReplyKey) else None
        return item


class ReplyTraceHub:
    """Exact generation/reply FSM plus bounded array/frame identity maps."""

    def __init__(self, session_id: str, *, event_sink: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        if not isinstance(session_id, str) or _SESSION_RE.fullmatch(session_id) is None:
            raise ValueError("session_id has an invalid shape")
        self.session_id = session_id
        self._sink = event_sink
        self._lock = threading.RLock()
        self._generation = 0
        self._sequence = 0
        self._active: ReplyKey | None = None
        self._known: set[ReplyKey] = set()
        self._finished: set[ReplyKey] = set()
        self._terminal_intents: set[ReplyKey] = set()
        self._terminal_intent_details: dict[ReplyKey, tuple[str, int]] = {}
        self._turns: dict[ReplyKey, int] = {}
        self._starts: dict[ReplyKey, tuple[int, str]] = {}
        self._semantic: dict[tuple[str, int], ReplyKey] = {}
        self._pending_semantic_summaries: OrderedDict[
            tuple[str, int], dict[str, Any]
        ] = OrderedDict()
        self._semantic_binding_sink: Callable[["ReplyTraceHub", str, int], Any] | None = None
        self._arrays: OrderedDict[int, tuple[weakref.ReferenceType[Any], ReplyKey]] = OrderedDict()
        self._frames: OrderedDict[int, tuple[Any, FrameProvenance]] = OrderedDict()
        self._face_sequences: dict[ReplyKey, int] = {}
        self._media_sequences: dict[ReplyKey, dict[str, int]] = {}
        self._media_fences: dict[ReplyKey, dict[str, Any]] = {}
        self._media_drain_committed: set[ReplyKey] = set()

    def _prune_finished_for_room_locked(self) -> bool:
        if len(self._known) < _REPLY_STATE_LIMIT:
            return True
        candidates = [
            key for key in self._finished if key != self._active
        ]
        if not candidates:
            return False
        victim = min(
            candidates,
            key=lambda key: (key.stream_generation, key.reply_seq),
        )
        self._known.discard(victim)
        self._finished.discard(victim)
        self._terminal_intents.discard(victim)
        self._terminal_intent_details.pop(victim, None)
        self._turns.pop(victim, None)
        self._starts.pop(victim, None)
        self._face_sequences.pop(victim, None)
        self._media_sequences.pop(victim, None)
        self._media_fences.pop(victim, None)
        self._media_drain_committed.discard(victim)
        self._semantic = {
            binding: owner
            for binding, owner in self._semantic.items()
            if owner != victim
        }
        return len(self._known) < _REPLY_STATE_LIMIT

    def set_event_sink(self, sink: Callable[[Mapping[str, Any]], Any] | None) -> None:
        replay: list[dict[str, Any]] = []
        with self._lock:
            self._sink = sink
            active = self._active
            if sink is not None and active is not None:
                replay.append({
                    "kind": "reply_started",
                    "reply_key": active.as_dict(),
                    "semantic_turn_id": self._turns.get(active),
                    "observed_ns": self._starts.get(active, (0, "unknown"))[0],
                    "cause": self._starts.get(active, (0, "unknown"))[1],
                    "started_before_recording": True,
                })
                for (request_id, revision), key in self._semantic.items():
                    if key == active:
                        replay.append({
                            "kind": "semantic_bound",
                            "reply_key": active.as_dict(),
                            "request_id": request_id,
                            "revision": revision,
                        })
                intent = self._terminal_intent_details.get(active)
                if intent is not None:
                    replay.append({
                        "kind": "reply_terminal_intent",
                        "reply_key": active.as_dict(),
                        "reason": intent[0],
                        "observed_ns": intent[1],
                    })
                fence = self._media_fences.get(active)
                if isinstance(fence, dict) and fence.get("producer_done") is True:
                    replay.append({
                        "kind": "media_producer_done",
                        "reply_key": active.as_dict(),
                        "observed_ns": int(fence.get("producer_done_ns", 0)),
                        "expected_last_audio_seq": int(
                            fence.get("expected_last_audio_seq", 0)
                        ),
                        "expected_last_video_seq": int(
                            fence.get("expected_last_video_seq", 0)
                        ),
                    })
        for event in replay:
            try:
                sink(dict(event))  # type: ignore[misc]
            except Exception:
                pass

    def _emit(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            sink = self._sink
        if sink is not None:
            try:
                sink(dict(event))
            except Exception:
                pass

    def bind_generation(self, generation: int) -> None:
        generation = _positive(generation, "generation")
        with self._lock:
            if generation == self._generation:
                return
            stale = self._active
            self._generation = generation
            self._sequence = 0
            self._active = None
            self._known.clear()
            self._finished.clear()
            self._terminal_intents.clear()
            self._terminal_intent_details.clear()
            self._turns.clear()
            self._starts.clear()
            self._semantic.clear()
            self._pending_semantic_summaries.clear()
            self._arrays.clear()
            self._frames.clear()
            self._face_sequences.clear()
            self._media_sequences.clear()
            self._media_fences.clear()
            self._media_drain_committed.clear()
        if stale is not None:
            self._emit({
                "kind": "reply_terminal",
                "reply_key": stale.as_dict(),
                "reason": "generation_replaced",
                "observed_ns": None,
                "timestamp_known": False,
            })

    def ensure_reply(self, *, semantic_turn_id: int, observed_ns: int, cause: str) -> ReplyKey:
        if isinstance(semantic_turn_id, bool) or not isinstance(semantic_turn_id, int):
            raise ValueError("semantic_turn_id must be an integer")
        _nonnegative(observed_ns, "observed_ns")
        if not isinstance(cause, str) or not cause or len(cause) > 64:
            raise ValueError("cause has an invalid shape")
        replaced: ReplyKey | None = None
        with self._lock:
            if self._generation < 1:
                raise RuntimeError("reply generation is not bound")
            key = self._active
            if key is not None and self._turns.get(key) != semantic_turn_id:
                # A late terminal callback cannot merge two explicit robot turns.
                replaced = key
                self._finished.add(key)
                self._active = None
                key = None
            created = key is None
            pruned_for_limit = False
            if created:
                at_limit = len(self._known) >= _REPLY_STATE_LIMIT
                if not self._prune_finished_for_room_locked():
                    raise RuntimeError("reply provenance state limit reached")
                pruned_for_limit = at_limit
                self._sequence += 1
                key = ReplyKey(self.session_id, self._generation, self._sequence)
                self._active = key
                self._known.add(key)
                self._turns[key] = semantic_turn_id
                self._starts[key] = (observed_ns, cause)
        assert key is not None
        if replaced is not None:
            self._emit({"kind": "reply_terminal", "reply_key": replaced.as_dict(), "reason": "semantic_turn_replaced", "observed_ns": observed_ns})
        if pruned_for_limit:
            self._emit({
                "kind": "reply_metrics_error",
                "error": "reply_trace_state_limit_exceeded",
            })
        if created:
            self._emit({"kind": "reply_started", "reply_key": key.as_dict(), "semantic_turn_id": semantic_turn_id, "observed_ns": observed_ns, "cause": cause})
        return key

    def current_reply(self) -> ReplyKey | None:
        with self._lock:
            return self._active

    def finish_reply(self, key: ReplyKey, *, reason: str, observed_ns: int) -> bool:
        try:
            _nonnegative(observed_ns, "observed_ns")
        except ValueError:
            return False
        if (
            not isinstance(reason, str)
            or not reason
            or len(reason) > 64
            or reason == "turn_complete"
        ):
            return False
        with self._lock:
            if (not isinstance(key, ReplyKey) or key.session_id != self.session_id or
                    key.stream_generation != self._generation or key not in self._known or key in self._finished):
                return False
            self._finished.add(key)
            self._media_fences.pop(key, None)
            if self._active == key:
                self._active = None
        self._emit({"kind": "reply_terminal", "reply_key": key.as_dict(), "reason": reason, "observed_ns": observed_ns})
        return True

    def mark_terminal_intent(
        self, key: ReplyKey, *, reason: str, observed_ns: int
    ) -> bool:
        """Record a Gemini terminal marker without ending media provenance."""
        try:
            _nonnegative(observed_ns, "observed_ns")
        except ValueError:
            return False
        if not isinstance(reason, str) or not reason or len(reason) > 64:
            return False
        with self._lock:
            if (
                not isinstance(key, ReplyKey)
                or key.session_id != self.session_id
                or key.stream_generation != self._generation
                or key not in self._known
                or key in self._finished
            ):
                return False
            self._terminal_intents.add(key)
            self._terminal_intent_details[key] = (reason, observed_ns)
        self._emit({
            "kind": "reply_terminal_intent",
            "reply_key": key.as_dict(),
            "reason": reason,
            "observed_ns": observed_ns,
        })
        self._complete_media_fence_if_ready(key, observed_ns=observed_ns)
        return True

    def _media_fence_ready_locked(self, key: ReplyKey) -> bool:
        fence = self._media_fences.get(key)
        if not isinstance(fence, dict) or fence.get("producer_done") is not True:
            return False
        expected_audio = int(fence.get("expected_last_audio_seq", 0))
        expected_video = int(fence.get("expected_last_video_seq", 0))
        if int(fence.get("placed_last_audio_seq", 0)) < expected_audio:
            return False
        if int(fence.get("placed_last_video_seq", 0)) < expected_video:
            return False
        # A generated video sequence is complete only when the same source
        # sequence appeared in an actually written final composite frame.
        if expected_video > 0 and int(
            fence.get("last_face_source_seq_placed", 0)
        ) < 1:
            return False
        return (
            key in self._terminal_intents
            and key not in self._finished
            and key not in self._media_drain_committed
        )

    def _complete_media_fence_if_ready(
        self, key: ReplyKey, *, observed_ns: int
    ) -> bool:
        with self._lock:
            ready = self._media_fence_ready_locked(key)
        return self.mark_media_drained(key, observed_ns=observed_ns) if ready else False

    def next_media_seq(self, key: ReplyKey, *, media_kind: str) -> int | None:
        """Allocate a per-reply sequence before tagging an upstream frame."""
        if media_kind not in {"audio", "video"}:
            return None
        with self._lock:
            if key not in self._known or key in self._finished:
                return None
            counters = self._media_sequences.setdefault(
                key, {"audio": 0, "video": 0}
            )
            value = int(counters.get(media_kind, 0)) + 1
            counters[media_kind] = value
            return value

    def note_media_enqueued(
        self, key: ReplyKey, *, media_kind: str, media_seq: int
    ) -> bool:
        """Freeze only sequence numbers whose upstream enqueue succeeded."""
        if (
            media_kind not in {"audio", "video"}
            or isinstance(media_seq, bool)
            or not isinstance(media_seq, int)
            or media_seq < 1
        ):
            return False
        with self._lock:
            if (
                not isinstance(key, ReplyKey)
                or key.stream_generation != self._generation
                or key not in self._known
                or key in self._finished
            ):
                return False
            fence = self._media_fences.setdefault(key, {})
            name = f"expected_last_{media_kind}_seq"
            previous = int(fence.get(name, 0))
            if fence.get("producer_done") is True or media_seq != previous + 1:
                return False
            fence[name] = media_seq
            ready = self._media_fence_ready_locked(key)
            observed_ns = int(fence.get("last_observed_ns", 0))
        return (
            self.mark_media_drained(key, observed_ns=observed_ns)
            if ready else True
        )

    def note_video_source_consumed(
        self, key: ReplyKey, *, face_source_seq: int
    ) -> bool:
        """Record a genuinely fresh compositor source, never a cached repeat."""
        if isinstance(face_source_seq, bool) or not isinstance(face_source_seq, int) or face_source_seq < 1:
            return False
        with self._lock:
            if key not in self._known or key in self._finished:
                return False
            fence = self._media_fences.setdefault(key, {})
            fence["video_source_consumed"] = int(
                fence.get("video_source_consumed", 0)
            ) + 1
            fence["last_face_source_seq_consumed"] = max(
                int(fence.get("last_face_source_seq_consumed", 0)),
                face_source_seq,
            )
        return True

    def mark_media_producer_done(
        self, key: ReplyKey, *, observed_ns: int
    ) -> bool:
        """Close producer counts; completion still waits for recorder placement."""
        try:
            _nonnegative(observed_ns, "observed_ns")
        except ValueError:
            return False
        with self._lock:
            if key not in self._known or key in self._finished:
                return False
            fence = self._media_fences.setdefault(key, {})
            if fence.get("producer_done") is True:
                return True
            fence["producer_done"] = True
            fence["producer_done_ns"] = observed_ns
            fence["last_observed_ns"] = observed_ns
            expected_audio = int(fence.get("expected_last_audio_seq", 0))
            expected_video = int(fence.get("expected_last_video_seq", 0))
            ready = self._media_fence_ready_locked(key)
        self._emit({
            "kind": "media_producer_done",
            "reply_key": key.as_dict(),
            "observed_ns": observed_ns,
            "expected_last_audio_seq": expected_audio,
            "expected_last_video_seq": expected_video,
        })
        return (
            self.mark_media_drained(key, observed_ns=observed_ns)
            if ready else True
        )

    def mark_media_packet_placed(
        self,
        key: ReplyKey,
        *,
        media_kind: str,
        media_seq: int,
        observed_ns: int,
        face_source_seq: int | None = None,
    ) -> bool:
        """Acknowledge contiguous exact sequences after recorder placement."""
        if (
            media_kind not in {"audio", "video"}
            or isinstance(media_seq, bool)
            or not isinstance(media_seq, int)
            or media_seq < 1
        ):
            return False
        try:
            _nonnegative(observed_ns, "observed_ns")
        except ValueError:
            return False
        error_reason = ""
        with self._lock:
            if key not in self._known or key in self._finished:
                return False
            fence = self._media_fences.setdefault(key, {})
            expected = int(fence.get(f"expected_last_{media_kind}_seq", 0))
            if fence.get("producer_done") is True and media_seq > expected:
                return False
            name = f"placed_last_{media_kind}_seq"
            previous = int(fence.get(name, 0))
            if media_seq <= previous:
                return True
            if media_seq != previous + 1:
                error_reason = f"{media_kind}_media_sequence_gap"
            elif media_kind == "video":
                previous_face = int(
                    fence.get("last_face_source_seq_placed", 0)
                )
                if (
                    isinstance(face_source_seq, bool)
                    or not isinstance(face_source_seq, int)
                    or face_source_seq <= previous_face
                ):
                    error_reason = "video_fresh_source_provenance_missing"
                else:
                    fence[name] = media_seq
                    fence["last_face_source_seq_placed"] = face_source_seq
            else:
                fence[name] = media_seq
            fence["last_observed_ns"] = observed_ns
            ready = not error_reason and self._media_fence_ready_locked(key)
        if error_reason:
            self._emit({
                "kind": "provenance_drop",
                "reply_key": key.as_dict(),
                "media_kind": media_kind,
                "reason": error_reason,
            })
            return False
        return (
            self.mark_media_drained(key, observed_ns=observed_ns)
            if ready else True
        )

    def mark_media_drained(self, key: ReplyKey, *, observed_ns: int) -> bool:
        """Finish only after an exact producer/recorder placement fence."""
        try:
            _nonnegative(observed_ns, "observed_ns")
        except ValueError:
            return False
        with self._lock:
            if (
                not isinstance(key, ReplyKey)
                or key.session_id != self.session_id
                or key.stream_generation != self._generation
                or key not in self._known
                or key in self._finished
                or key not in self._terminal_intents
                or key in self._media_drain_committed
                or not self._media_fence_ready_locked(key)
            ):
                return False
            self._media_drain_committed.add(key)
            fence_snapshot = {
                name: int(self._media_fences.get(key, {}).get(name, 0))
                for name in (
                    "expected_last_audio_seq", "expected_last_video_seq",
                    "placed_last_audio_seq", "placed_last_video_seq",
                    "last_face_source_seq_placed",
                )
            }
        self._emit({
            "kind": "media_drained",
            "reply_key": key.as_dict(),
            "observed_ns": observed_ns,
            **fence_snapshot,
        })
        return self.finish_reply(key, reason="completed", observed_ns=observed_ns)

    def has_terminal_intent(self, key: ReplyKey) -> bool:
        with self._lock:
            return (
                isinstance(key, ReplyKey)
                and key.stream_generation == self._generation
                and key in self._terminal_intents
                and key not in self._finished
            )

    def set_semantic_binding_sink(
        self,
        sink: Callable[["ReplyTraceHub", str, int], Any] | None,
    ) -> None:
        with self._lock:
            self._semantic_binding_sink = sink

    def bind_semantic(self, key: ReplyKey, request_id: str, revision: int) -> bool:
        if not isinstance(request_id, str) or not request_id or len(request_id) > 192:
            return False
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            return False
        with self._lock:
            if (not isinstance(key, ReplyKey) or key.session_id != self.session_id or
                    key.stream_generation != self._generation or key not in self._known):
                return False
            binding = (request_id, revision)
            previous = self._semantic.get(binding)
            if previous is not None and previous != key:
                return False
            binding_limit = (
                previous is None
                and len(self._semantic) >= _SEMANTIC_BINDING_LIMIT
            )
            if not binding_limit:
                self._semantic[binding] = key
            pending_summary = (
                None if binding_limit
                else self._pending_semantic_summaries.pop(binding, None)
            )
        if binding_limit:
            self._emit({
                "kind": "reply_metrics_error",
                "error": "semantic_binding_limit_exceeded",
            })
            return False
        self._emit({"kind": "semantic_bound", "reply_key": key.as_dict(), "request_id": request_id, "revision": revision})
        if pending_summary is not None:
            self.observe_semantic_summary(key, **pending_summary)
        with self._lock:
            binding_sink = self._semantic_binding_sink
        if binding_sink is not None:
            try:
                binding_sink(self, request_id, revision)
            except Exception:
                pass
        return True

    def observe_transcript(
        self,
        key: ReplyKey,
        *,
        user_text: Any,
        reply_text: Any,
        observed_ns: int,
        user_text_complete: bool,
        reply_text_complete: bool,
        user_text_observed: bool = True,
        reply_text_observed: bool = True,
    ) -> bool:
        with self._lock:
            if key not in self._known:
                return False
        raw_user = str(user_text or "")
        raw_reply = str(reply_text or "")
        clean_user, user_truncated = _bounded_text(raw_user)
        clean_reply, reply_truncated = _bounded_text(raw_reply)
        self._emit({
            "kind": "transcript_snapshot",
            "reply_key": key.as_dict(),
            "user_text": clean_user,
            "reply_text": clean_reply,
            "user_text_sha256": _sha256_text(raw_user),
            "reply_text_sha256": _sha256_text(raw_reply),
            "user_text_truncated": user_truncated,
            "reply_text_truncated": reply_truncated,
            "user_text_observed": bool(user_text_observed),
            "reply_text_observed": bool(reply_text_observed),
            "observed_ns": observed_ns,
            "user_text_complete": bool(user_text_complete),
            "reply_text_complete": bool(reply_text_complete),
        })
        return True

    def observe_semantic_summary(
        self, key: ReplyKey, **summary: Any
    ) -> bool:
        with self._lock:
            if key not in self._known:
                return False
        event = {
            "kind": "semantic_summary",
            "reply_key": key.as_dict(),
            **_json_snapshot(summary),
        }
        self._emit(event)
        return True

    def observe_semantic_summary_for_semantic(
        self, request_id: str, revision: int, **summary: Any
    ) -> bool:
        if (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 192
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            return False
        key = self.reply_for_semantic(request_id, revision)
        if key is not None:
            return self.observe_semantic_summary(key, **summary)
        snapshot = _json_snapshot(summary)
        if not isinstance(snapshot, Mapping):
            return False
        binding = (request_id, revision)
        with self._lock:
            # A terminal summary is authoritative over a later preview.
            previous = self._pending_semantic_summaries.get(binding)
            if (
                isinstance(previous, Mapping)
                and previous.get("terminal_summary") is True
                and snapshot.get("terminal_summary") is not True
            ):
                return True
            if (
                binding not in self._pending_semantic_summaries
                and len(self._pending_semantic_summaries)
                >= _PENDING_SEMANTIC_SUMMARY_LIMIT
            ):
                pending_limit = True
            else:
                pending_limit = False
                self._pending_semantic_summaries[binding] = dict(snapshot)
                self._pending_semantic_summaries.move_to_end(binding)
        if pending_limit:
            self._emit({
                "kind": "reply_metrics_error",
                "error": "pending_semantic_summary_limit_exceeded",
            })
            return False
        return True

    def reply_for_semantic(self, request_id: str, revision: int) -> ReplyKey | None:
        try:
            with self._lock:
                return self._semantic.get((str(request_id), int(revision)))
        except Exception:
            return None

    def observe_body_for_semantic(
        self,
        request_id: str,
        revision: int,
        action_record: Mapping[str, Any],
        role: str,
        **metadata: Any,
    ) -> bool:
        key = self.reply_for_semantic(request_id, revision)
        if key is None or not isinstance(action_record, Mapping):
            return False
        event = {
            "kind": "body_action",
            "reply_key": key.as_dict(),
            "action_record": _json_snapshot(action_record),
            "role": str(role),
        }
        for name in ("ordinal", "function", "atom_ref", "terminal_status"):
            if name in metadata:
                event[name] = _json_snapshot(metadata[name])
        self._emit(event)
        return True

    def tagged_queue(self, source: queue.Queue[Any]) -> TaggedReplyQueue:
        return source if isinstance(source, TaggedReplyQueue) else TaggedReplyQueue.from_queue(source, self)

    @staticmethod
    def producer_context() -> tuple["ReplyTraceHub", ReplyKey] | None:
        value = getattr(_CONTEXT, "value", None)
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], ReplyTraceHub) and isinstance(value[1], ReplyKey):
            return value
        return None

    def remember_generated_array(self, value: Any, key: ReplyKey) -> None:
        try:
            reference = weakref.ref(value)
        except TypeError:
            return
        evicted: ReplyKey | None = None
        with self._lock:
            self._arrays[id(value)] = (reference, key)
            self._arrays.move_to_end(id(value))
            while len(self._arrays) > _ARRAY_LIMIT:
                _, (_, evicted) = self._arrays.popitem(last=False)
        if evicted is not None:
            self._emit({
                "kind": "provenance_drop",
                "reply_key": evicted.as_dict(),
                "media_kind": "generated_array",
                "reason": "array_identity_evicted",
            })

    def key_for_generated_array(self, value: Any) -> ReplyKey | None:
        with self._lock:
            tagged = self._arrays.pop(id(value), None)
        if tagged is None:
            return None
        reference, key = tagged
        return key if reference() is value else None


    def next_face_source_seq(self, key: ReplyKey) -> int | None:
        """Allocate only when the compositor consumed a genuinely fresh source."""
        with self._lock:
            if key not in self._known:
                return None
            value = self._face_sequences.get(key, 0) + 1
            self._face_sequences[key] = value
        self.note_video_source_consumed(key, face_source_seq=value)
        return value

    def tag_frame(
        self,
        frame: Any,
        key: ReplyKey,
        *,
        media_kind: str,
        face_source_seq: int | None = None,
        media_seq: int | None = None,
    ) -> FrameProvenance | None:
        clean_media_kind = str(media_kind)[:32] or "unknown"
        with self._lock:
            if key not in self._known:
                return None
            tag = FrameProvenance(
                key, face_source_seq, clean_media_kind,
                media_seq
                if isinstance(media_seq, int) and not isinstance(media_seq, bool) and media_seq > 0
                else None,
            )
        # PyAV frames are neither weak-referenceable nor open to arbitrary
        # attributes, but their documented opaque slot retains Python objects.
        # Carrying the tag on the frame avoids holding a queue-worth of RGB data.
        try:
            previous = getattr(frame, "opaque", _MISSING)
            setattr(frame, "opaque", (_FRAME_TAG_MAGIC, tag, previous))
            return tag
        except Exception:
            pass
        evicted: FrameProvenance | None = None
        with self._lock:
            try:
                owner: Any = weakref.ref(frame)
            except TypeError:
                owner = frame
            self._frames[id(frame)] = (owner, tag)
            self._frames.move_to_end(id(frame))
            while len(self._frames) > _FRAME_LIMIT:
                _, (_, evicted) = self._frames.popitem(last=False)
        if evicted is not None:
            self._emit({
                "kind": "provenance_drop",
                "reply_key": evicted.reply_key.as_dict(),
                "media_kind": evicted.media_kind or "unknown",
                "reason": "frame_identity_evicted",
            })
        return tag

    def take_frame(self, frame: Any) -> FrameProvenance | None:
        try:
            carried = getattr(frame, "opaque", None)
            if (
                isinstance(carried, tuple) and len(carried) == 3
                and carried[0] is _FRAME_TAG_MAGIC
                and isinstance(carried[1], FrameProvenance)
            ):
                previous = carried[2]
                setattr(frame, "opaque", None if previous is _MISSING else previous)
                return carried[1]
        except Exception:
            pass
        with self._lock:
            tagged = self._frames.pop(id(frame), None)
        if tagged is None:
            return None
        owned, tag = tagged
        actual = owned() if isinstance(owned, weakref.ReferenceType) else owned
        return tag if actual is frame else None

    def discard_frame(self, frame: Any) -> None:
        try:
            carried = getattr(frame, "opaque", None)
            if isinstance(carried, tuple) and len(carried) == 3 and carried[0] is _FRAME_TAG_MAGIC:
                previous = carried[2]
                setattr(frame, "opaque", None if previous is _MISSING else previous)
        except Exception:
            pass
        with self._lock:
            self._frames.pop(id(frame), None)

    @staticmethod
    def take_producer_context() -> tuple["ReplyTraceHub", ReplyKey] | None:
        value = ReplyTraceHub.producer_context()
        _CONTEXT.value = None
        return value

    @contextmanager
    def media_scope(self, key: ReplyKey):
        """Propagate one producer identity through synchronous AV publication."""
        previous = getattr(_CONTEXT, "value", None)
        with self._lock:
            valid = (
                isinstance(key, ReplyKey)
                and key.session_id == self.session_id
                and key.stream_generation == self._generation
                and key in self._known
            )
        _CONTEXT.value = (self, key) if valid else None
        try:
            yield
        finally:
            _CONTEXT.value = previous


def install_submission_receipt_observer(
    runtime_cls: type[Any],
    observer_resolver: Callable[[Any, Any], Callable[..., Any] | None],
) -> bool:
    """Observe exact verified terminal records without changing runtime control flow."""
    if not isinstance(runtime_cls, type) or not callable(observer_resolver):
        raise TypeError("runtime class and observer resolver are required")
    if runtime_cls.__dict__.get("_experiment_reply_receipt_observer_patched", False):
        return True
    original = getattr(runtime_cls, "_submit_and_wait", None)
    if not callable(original):
        raise TypeError("runtime has no _submit_and_wait method")

    terminal_statuses = frozenset({"completed", "canceled", "cancelled", "replaced", "failed"})

    def notify(context: Any, record: Any, status: Any) -> None:
        """Optional notification; every exception is isolated from execution."""
        try:
            if not isinstance(context, dict) or not isinstance(record, Mapping):
                return
            normalized_status = str(status or record.get("status", "")).lower()
            if normalized_status not in terminal_statuses:
                return
            action_id = str(record.get("id", ""))[:192]
            dedup_key = action_id or f"missing-id:{normalized_status}"
            notified = context["notified"]
            if dedup_key in notified:
                return
            notified.add(dedup_key)
            if not context["resolver_called"]:
                context["resolver_called"] = True
                context["observer"] = observer_resolver(
                    context["runtime"], context["request"]
                )
            observer = context.get("observer")
            if not callable(observer):
                return
            record_snapshot = _json_snapshot(record)
            observer(
                request_id=context["request_id"],
                revision=context["revision"],
                kind=context["kind"],
                ordinal=context["ordinal"],
                function=context["function"],
                atom_ref=_json_snapshot(context["atom_ref"]),
                action_record=(
                    record_snapshot if isinstance(record_snapshot, dict) else {}
                ),
                terminal_status=normalized_status,
            )
        except Exception:
            pass

    verify_descriptor = runtime_cls.__dict__.get("_verify_action_record")
    verify_original: Any = None
    if isinstance(verify_descriptor, staticmethod):
        verify_original = verify_descriptor.__func__

        @functools.wraps(verify_original)
        def verified(value: Any, *args: Any, **kwargs: Any) -> Any:
            status = verify_original(value, *args, **kwargs)
            notify(getattr(_SUBMISSION_CONTEXT, "value", None), value, status)
            return status

        runtime_cls._verify_action_record = staticmethod(verified)
    elif callable(verify_descriptor):
        verify_original = verify_descriptor

        @functools.wraps(verify_original)
        def verified_method(instance: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            status = verify_original(instance, value, *args, **kwargs)
            notify(getattr(_SUBMISSION_CONTEXT, "value", None), value, status)
            return status

        runtime_cls._verify_action_record = verified_method

    @functools.wraps(original)
    def observed(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_SUBMISSION_CONTEXT, "value", None)
        atom_snapshot = _json_snapshot(kwargs.get("atom_ref", {}))
        revision = getattr(request, "revision", 0)
        context = {
            "runtime": self,
            "request": request,
            "request_id": str(getattr(request, "request_id", ""))[:192],
            "revision": (
                int(revision)
                if isinstance(revision, int) and not isinstance(revision, bool)
                else 0
            ),
            "kind": str(kwargs.get("kind", ""))[:64],
            "ordinal": (
                int(kwargs.get("ordinal", 0))
                if isinstance(kwargs.get("ordinal", 0), int)
                and not isinstance(kwargs.get("ordinal", 0), bool)
                else 0
            ),
            "function": (
                None if kwargs.get("semantic_function") is None
                else str(kwargs.get("semantic_function"))[:128]
            ),
            "atom_ref": atom_snapshot if isinstance(atom_snapshot, dict) else {},
            "notified": set(),
            "resolver_called": False,
            "observer": None,
        }
        _SUBMISSION_CONTEXT.value = context
        try:
            result = original(self, request, *args, **kwargs)
            record = getattr(result, "record", None)
            if isinstance(record, Mapping):
                notify(context, record, record.get("status", "completed"))
            return result
        finally:
            _SUBMISSION_CONTEXT.value = previous

    runtime_cls._submit_and_wait = observed
    runtime_cls._experiment_reply_receipt_observer_patched = True
    runtime_cls._experiment_reply_receipt_observer_original = original
    runtime_cls._experiment_reply_receipt_observer_original_verify = verify_original
    return True


@dataclass(slots=True)
class _ReplyState:
    key: ReplyKey
    semantic_turn_id: int | None = None
    semantic_request_id: str = ""
    semantic_revision: int = 0
    semantic_status: str = ""
    semantic_summary_observed: bool = False
    semantic_preview_observed: bool = False
    semantic_preview_observed_ns: int | None = None
    semantic_terminal_summary_observed: bool = False
    semantic_summary_observed_ns: int | None = None
    closed_neutral: bool | None = None
    first_output_ns: int | None = None
    reply_start_cause: str = ""
    started_before_recording: bool = False
    terminal_reason: str = ""
    terminal_ns: int | None = None
    terminal_intent_reason: str = ""
    terminal_intent_ns: int | None = None
    media_drained_ns: int | None = None
    media_fence: dict[str, Any] = field(default_factory=dict)
    user_text: str = ""
    user_text_sha256: str = ""
    user_text_truncated: bool = False
    reply_text: str = ""
    reply_text_sha256: str = ""
    reply_text_truncated: bool = False
    transcript_observed_ns: int | None = None
    user_text_observed: bool = False
    reply_text_observed: bool = False
    user_text_complete: bool = False
    reply_text_complete: bool = False
    ordered_functions: list[str] = field(default_factory=list)
    planned_functions: list[str] = field(default_factory=list)
    completed_functions: list[str] = field(default_factory=list)
    dropped_functions: list[str] = field(default_factory=list)
    result_action_ids: list[str] = field(default_factory=list)
    result_action_ids_observed: bool = False
    result_action_ids_valid: bool = True
    execution_summary_complete: bool | None = None
    selected_atoms: list[dict[str, Any]] = field(default_factory=list)
    selected_atoms_sha256: str = ""
    recovery_required: bool | None = None
    recovery_action_id: str | None = None
    audio_start: int | None = None
    audio_end: int | None = None
    audio_first_meta: dict[str, Any] = field(default_factory=dict)
    audio_last_meta: dict[str, Any] = field(default_factory=dict)
    face_frame: int | None = None
    face_source_seq: int | None = None
    face_meta: dict[str, Any] = field(default_factory=dict)
    face_last_frame: int | None = None
    face_last_source_seq: int | None = None
    face_last_meta: dict[str, Any] = field(default_factory=dict)
    face_fresh_source_count: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)
    duplicate_action_ids: list[str] = field(default_factory=list)
    provenance_errors: list[str] = field(default_factory=list)


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[list[float]] = []
    for start, end in sorted(x for x in intervals if x[1] > x[0]):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return [(x[0], x[1]) for x in result]


def _tiou(speech: tuple[float, float] | None, body: list[tuple[float, float]]) -> float | None:
    if speech is None or speech[1] <= speech[0]:
        return None
    body = _union(body)
    if not body:
        return None
    overlap = sum(
        max(0.0, min(speech[1], end) - max(speech[0], start))
        for start, end in body
    )
    union = speech[1] - speech[0] + sum(end - start for start, end in body) - overlap
    return None if union <= 0 else 100.0 * overlap / union


class ReplyMetricsAccumulator:
    """Single-recorder-worker aggregate using actual written sample/frame indices."""

    def __init__(self, *, recording_id: str, session_id: str, started_monotonic_ns: int,
                 started_wall: float, wall_anchor_uncertainty_ns: int, sample_rate: int, fps: int) -> None:
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError("recording_id is required")
        if not isinstance(session_id, str) or _SESSION_RE.fullmatch(session_id) is None:
            raise ValueError("session_id has an invalid shape")
        self.recording_id = recording_id
        self.session_id = session_id
        self.started_monotonic_ns = _nonnegative(started_monotonic_ns, "started_monotonic_ns")
        wall = _finite(started_wall)
        if wall is None:
            raise ValueError("started_wall must be finite")
        self.started_wall = wall
        self.wall_anchor_uncertainty_ns = _nonnegative(
            wall_anchor_uncertainty_ns, "wall_anchor_uncertainty_ns"
        )
        self.sample_rate = _positive(sample_rate, "sample_rate")
        self.fps = _positive(fps, "fps")
        self._states: dict[ReplyKey, _ReplyState] = {}
        self.last_error = ""

    def _state(self, key: ReplyKey, create: bool = False) -> _ReplyState | None:
        if not isinstance(key, ReplyKey) or key.session_id != self.session_id:
            return None
        result = self._states.get(key)
        if result is None and create:
            if len(self._states) >= _REPLY_STATE_LIMIT:
                self.last_error = "reply_state_limit_exceeded"
                return None
            result = self._states[key] = _ReplyState(key)
        return result

    def observe_event(self, event: Mapping[str, Any]) -> bool:
        try:
            key = _key_from_event(event) if isinstance(event, Mapping) else None
            if key is None or key.session_id != self.session_id:
                return False
            kind = str(event.get("kind", ""))
            recognized = {
                "reply_started", "semantic_bound", "reply_terminal",
                "reply_terminal_intent", "media_producer_done", "media_drained",
                "transcript_snapshot", "semantic_summary", "body_action",
                "provenance_drop",
            }
            state = self._state(key, kind in recognized)
            if state is None:
                return False
            if kind == "reply_started":
                turn = event.get("semantic_turn_id")
                if isinstance(turn, int) and not isinstance(turn, bool):
                    state.semantic_turn_id = turn
                observed = event.get("observed_ns")
                if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                    state.first_output_ns = observed if state.first_output_ns is None else min(state.first_output_ns, observed)
                if not state.reply_start_cause:
                    state.reply_start_cause = str(event.get("cause", ""))[:64]
                state.started_before_recording = (
                    state.started_before_recording
                    or event.get("started_before_recording") is True
                )
                return True
            if kind == "semantic_bound":
                request_id, revision = event.get("request_id"), event.get("revision")
                if not isinstance(request_id, str) or not request_id or isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                    return False
                state.semantic_request_id, state.semantic_revision = request_id[:192], revision
                return True
            if kind == "reply_terminal_intent":
                reason = event.get("reason")
                if not isinstance(reason, str) or not reason:
                    return False
                state.terminal_intent_reason = state.terminal_intent_reason or reason[:64]
                observed = event.get("observed_ns")
                if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                    state.terminal_intent_ns = observed
                return True
            if kind == "media_producer_done":
                observed = event.get("observed_ns")
                if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
                    return False
                state.media_fence.update({
                    "producer_done_monotonic_ns": observed,
                    "expected_last_audio_seq": max(
                        0, int(event.get("expected_last_audio_seq", 0))
                    ),
                    "expected_last_video_seq": max(
                        0, int(event.get("expected_last_video_seq", 0))
                    ),
                })
                return True
            if kind == "media_drained":
                observed = event.get("observed_ns")
                if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
                    return False
                state.media_drained_ns = observed
                for name in (
                    "expected_last_audio_seq", "expected_last_video_seq",
                    "placed_last_audio_seq", "placed_last_video_seq",
                    "last_face_source_seq_placed",
                ):
                    value = event.get(name)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        state.media_fence[name] = value
                state.media_fence["drained_monotonic_ns"] = observed
                return True
            if kind == "reply_terminal":
                reason = event.get("reason")
                if not isinstance(reason, str) or not reason:
                    return False
                state.terminal_reason = state.terminal_reason or reason[:64]
                observed = event.get("observed_ns")
                if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                    state.terminal_ns = observed
                return True
            if kind == "transcript_snapshot":
                raw_user = str(event.get("user_text") or "")
                raw_reply = str(event.get("reply_text") or "")
                state.user_text, state.user_text_truncated = _bounded_text(raw_user)
                state.reply_text, state.reply_text_truncated = _bounded_text(raw_reply)
                state.user_text_sha256 = _sha256_text(raw_user)
                state.reply_text_sha256 = _sha256_text(raw_reply)
                state.user_text_observed = (
                    event.get("user_text_observed") is True
                    if "user_text_observed" in event else "user_text" in event
                )
                state.reply_text_observed = (
                    event.get("reply_text_observed") is True
                    if "reply_text_observed" in event else "reply_text" in event
                )
                if "user_text_truncated" in event:
                    state.user_text_truncated = event.get("user_text_truncated") is True
                if "reply_text_truncated" in event:
                    state.reply_text_truncated = event.get("reply_text_truncated") is True
                if isinstance(event.get("user_text_sha256"), str):
                    state.user_text_sha256 = str(event["user_text_sha256"])[:64]
                if isinstance(event.get("reply_text_sha256"), str):
                    state.reply_text_sha256 = str(event["reply_text_sha256"])[:64]
                observed = event.get("observed_ns")
                if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                    state.transcript_observed_ns = observed
                state.user_text_complete = event.get("user_text_complete") is True
                state.reply_text_complete = event.get("reply_text_complete") is True
                return True
            if kind == "semantic_summary":
                is_terminal_summary = event.get("terminal_summary") is True
                state.semantic_summary_observed = True
                if not is_terminal_summary:
                    state.semantic_preview_observed = True
                    preview_observed = event.get("observed_ns")
                    if isinstance(preview_observed, int) and not isinstance(preview_observed, bool) and preview_observed >= 0:
                        state.semantic_preview_observed_ns = preview_observed
                if state.semantic_terminal_summary_observed:
                    if not is_terminal_summary and not state.ordered_functions:
                        functions = event.get("ordered_functions")
                        if isinstance(functions, (list, tuple)):
                            state.ordered_functions = [
                                str(x)[:128]
                                for x in functions[:_JSON_MAX_ITEMS]
                            ]
                    return True
                if is_terminal_summary:
                    state.semantic_terminal_summary_observed = True
                observed = event.get("observed_ns")
                if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                    state.semantic_summary_observed_ns = observed
                if "status" in event:
                    state.semantic_status = str(event.get("status", ""))[:128]
                if "ordered_functions" in event:
                    functions = event.get("ordered_functions")
                    if isinstance(functions, (list, tuple)):
                        clean_functions = [
                            str(x)[:128] for x in functions[:_JSON_MAX_ITEMS]
                        ]
                        if is_terminal_summary:
                            state.planned_functions = clean_functions
                            if not state.ordered_functions:
                                state.ordered_functions = list(clean_functions)
                        else:
                            state.ordered_functions = clean_functions
                if "planned_functions" in event and is_terminal_summary:
                    planned = event.get("planned_functions")
                    if isinstance(planned, (list, tuple)):
                        state.planned_functions = [
                            str(x)[:128] for x in planned[:_JSON_MAX_ITEMS]
                        ]
                if "selected_atoms" in event:
                    atoms = event.get("selected_atoms")
                    snapshot = _json_snapshot(atoms)
                    if isinstance(snapshot, list) and snapshot:
                        state.selected_atoms = [
                            dict(x) for x in snapshot if isinstance(x, Mapping)
                        ]
                        state.selected_atoms_sha256 = _json_sha256(state.selected_atoms)
                if "completed_functions" in event:
                    completed = event.get("completed_functions")
                    if isinstance(completed, (list, tuple)):
                        state.completed_functions = [
                            str(x)[:128] for x in completed[:_JSON_MAX_ITEMS]
                        ]
                if "dropped_functions" in event:
                    dropped = event.get("dropped_functions")
                    if isinstance(dropped, (list, tuple)):
                        state.dropped_functions = [
                            str(x)[:128] for x in dropped[:_JSON_MAX_ITEMS]
                        ]
                if is_terminal_summary and "result_action_ids" in event:
                    result_ids = event.get("result_action_ids")
                    state.result_action_ids_observed = True
                    state.result_action_ids_valid = bool(
                        isinstance(result_ids, (list, tuple))
                        and len(result_ids) <= _JSON_MAX_ITEMS
                        and all(
                            isinstance(value, str) and bool(value)
                            for value in result_ids
                        )
                    )
                    state.result_action_ids = (
                        [
                            value[:192]
                            for value in result_ids[:_JSON_MAX_ITEMS]
                            if isinstance(value, str) and value
                        ]
                        if isinstance(result_ids, (list, tuple))
                        else []
                    )
                if is_terminal_summary and "execution_summary_complete" in event:
                    summary_complete = event.get("execution_summary_complete")
                    if isinstance(summary_complete, bool):
                        state.execution_summary_complete = summary_complete
                if "recovery_required" in event:
                    required = event.get("recovery_required")
                    if isinstance(required, bool):
                        state.recovery_required = required
                if "recovery_action_id" in event:
                    recovery_id = event.get("recovery_action_id")
                    state.recovery_action_id = (
                        recovery_id[:192]
                        if isinstance(recovery_id, str) and recovery_id else None
                    )
                neutral = event.get("closed_neutral", event.get("closedNeutral"))
                if isinstance(neutral, bool):
                    state.closed_neutral = neutral
                return True
            if kind == "provenance_drop":
                reason = str(event.get("reason", "provenance_drop"))[:128]
                media_kind = str(event.get("media_kind", "unknown"))[:32]
                token = f"provenance_drop:{media_kind}:{reason}"
                if token not in state.provenance_errors:
                    state.provenance_errors.append(token)
                return True
            if kind == "body_action":
                return self.observe_body(
                    key,
                    event.get("action_record", {}),
                    str(event.get("role", "")),
                    ordinal=event.get("ordinal"),
                    function=event.get("function"),
                    atom_ref=event.get("atom_ref"),
                    terminal_status=event.get("terminal_status"),
                )
            return False
        except Exception:
            return False

    def observe_text(self, key: ReplyKey, *, user_text: Any, reply_text: Any) -> bool:
        return self.observe_event({
            "kind": "transcript_snapshot", "reply_key": key.as_dict(),
            "user_text": user_text, "reply_text": reply_text,
        }) if isinstance(key, ReplyKey) else False

    def observe_plan(self, key: ReplyKey, *, status: Any,
                     ordered_functions: Any, selected_atoms: Any,
                     recovery_required: bool | None = None,
                     recovery_action_id: str | None = None,
                     result_action_ids: Any = _MISSING,
                     execution_summary_complete: bool | None = None) -> bool:
        if not isinstance(key, ReplyKey):
            return False
        event = {
            "kind": "semantic_summary", "reply_key": key.as_dict(),
            "status": status, "ordered_functions": ordered_functions,
            "selected_atoms": selected_atoms,
            "terminal_summary": True,
        }
        if recovery_required is not None:
            event["recovery_required"] = recovery_required
        if recovery_action_id is not None:
            event["recovery_action_id"] = recovery_action_id
        if result_action_ids is not _MISSING:
            event["result_action_ids"] = result_action_ids
        if execution_summary_complete is not None:
            event["execution_summary_complete"] = execution_summary_complete
        return self.observe_event(event)

    def observe_audio(self, key: ReplyKey, start_sample: int,
                      end_sample_exclusive: int, **metadata: Any) -> bool:
        try:
            start = _nonnegative(start_sample, "start_sample")
            end = _nonnegative(end_sample_exclusive, "end_sample_exclusive")
        except ValueError:
            return False
        state = self._state(key)
        if state is None or end <= start:
            return False
        allowed = {
            "media_time_ns", "pts", "time_base_num", "time_base_den",
            "captured_ns", "dropped_before", "media_seq",
        }
        clean = {
            name: _json_snapshot(value) for name, value in metadata.items()
            if name in allowed
        }
        if state.audio_start is None or start < state.audio_start:
            state.audio_start, state.audio_first_meta = start, clean
        if state.audio_end is None or end > state.audio_end:
            state.audio_end, state.audio_last_meta = end, clean
        return True

    def observe_face(self, key: ReplyKey, frame_index: int,
                     source_seq: int, **metadata: Any) -> bool:
        try:
            frame = _nonnegative(frame_index, "frame_index")
            source = _positive(source_seq, "source_seq")
        except ValueError:
            return False
        state = self._state(key)
        if state is None:
            return False
        allowed = {
            "pts", "time_base_num", "time_base_den", "captured_ns", "media_seq"
        }
        clean = {
            name: _json_snapshot(value) for name, value in metadata.items()
            if name in allowed
        }
        if state.face_frame is None or frame < state.face_frame:
            state.face_frame, state.face_source_seq = frame, source
            state.face_meta = clean
        if state.face_last_source_seq is None:
            state.face_last_source_seq = source
            state.face_last_frame = frame
            state.face_last_meta = clean
            state.face_fresh_source_count = 1
        elif source > state.face_last_source_seq:
            state.face_last_source_seq = source
            state.face_last_frame = frame
            state.face_last_meta = clean
            state.face_fresh_source_count += 1
        return True

    def observe_body(
        self,
        key: ReplyKey,
        action_record: Mapping[str, Any],
        role: str,
        *,
        ordinal: Any = None,
        function: Any = None,
        atom_ref: Any = None,
        terminal_status: Any = None,
    ) -> bool:
        state = self._state(key)
        if (
            state is None or not isinstance(action_record, Mapping)
            or role not in {"semantic", "recovery"}
        ):
            return False
        action_id = action_record.get("id")
        if not isinstance(action_id, str) or not action_id:
            return False
        status = str(terminal_status or action_record.get("status", "")).lower()
        started = _finite(action_record.get("started_wall"))
        completed = _finite(
            action_record.get("completed_wall", action_record.get("finished_wall"))
        )
        failed_statuses = {
            "failed", "error", "cancelled", "canceled", "aborted", "rejected",
            "replaced",
        }
        failed = status in failed_statuses
        completed_interval = (
            status == "completed" and started is not None and completed is not None
            and completed > started
        )
        timing_invalid = status == "completed" and (
            started is None or completed is None or completed <= started
        )
        if not completed_interval and not failed and not timing_invalid:
            return False
        if any(item["id"] == action_id for item in state.actions):
            clean_duplicate = action_id[:192]
            if clean_duplicate not in state.duplicate_action_ids:
                state.duplicate_action_ids.append(clean_duplicate)
            return not timing_invalid
        if len(state.actions) >= _ACTIONS_PER_REPLY_LIMIT:
            self.last_error = "reply_actions_per_reply_limit_exceeded"
            return False
        names = (
            "status", "payload_sha256", "scheduled_wall", "started_wall",
            "completed_wall", "finished_wall", "scheduled_sim", "started_sim",
            "completed_sim", "finished_sim", "bridge_duration", "duration", "speed",
        )
        clean_ordinal = (
            int(ordinal)
            if isinstance(ordinal, int) and not isinstance(ordinal, bool)
            else None
        )
        clean_function = (
            str(function)[:128] if isinstance(function, str) and function else None
        )
        atom_snapshot = _json_snapshot(atom_ref)
        if not isinstance(atom_snapshot, Mapping):
            atom_snapshot = {}
        item: dict[str, Any] = {
            "id": action_id[:192],
            "role": role[:64],
            "terminal_status": status or None,
            "ordinal": clean_ordinal,
            "function": clean_function,
            "atom_ref": dict(atom_snapshot),
            "raw_completed_interval": completed_interval,
            "included_in_body": completed_interval,
            "unexpected_recovery": False,
            "start_s": None,
            "end_s": None,
            "mapped_start_monotonic_ns": None,
            "mapped_end_monotonic_ns": None,
        }
        for name in names:
            if name in action_record:
                item[name] = _json_snapshot(action_record[name])
        if completed_interval:
            assert started is not None and completed is not None
            item["start_s"] = started - self.started_wall
            item["end_s"] = completed - self.started_wall
            item["mapped_start_monotonic_ns"] = (
                self.started_monotonic_ns
                + round((started - self.started_wall) * 1e9)
            )
            item["mapped_end_monotonic_ns"] = (
                self.started_monotonic_ns
                + round((completed - self.started_wall) * 1e9)
            )
        state.actions.append(item)
        if role == "semantic":
            selected = {
                "ordinal": clean_ordinal,
                "function": clean_function,
                "action_id": action_id[:192],
                "terminal_status": status or None,
                "atom_ref": dict(atom_snapshot),
            }
            state.selected_atoms = [
                value for value in state.selected_atoms
                if value.get("action_id") != action_id[:192]
            ]
            state.selected_atoms.append(selected)
            state.selected_atoms.sort(
                key=lambda value: (
                    value.get("ordinal")
                    if isinstance(value.get("ordinal"), int) else 1 << 30,
                    str(value.get("action_id", "")),
                )
            )
            state.selected_atoms_sha256 = _json_sha256(state.selected_atoms)
        # Invalid completed timestamps remain in the forensic action list, while
        # False preserves the observation API's rejection contract.
        return not timing_invalid

    def rows(
        self,
        *,
        recording_end_sample: int,
        recording_end_frame: int,
        user_track_end_sample: int | None = None,
        assistant_track_end_sample: int | None = None,
        capture_end_sample: int | None = None,
        capture_end_frame: int | None = None,
        wall_clock_step_detected: bool = False,
        stop_wall_anchor_uncertainty_ns: int = 0,
        wall_clock_offset_change_ns: int = 0,
    ) -> list[dict[str, Any]]:
        try:
            end_sample = _nonnegative(recording_end_sample, "recording_end_sample")
            end_frame = _nonnegative(recording_end_frame, "recording_end_frame")
        except ValueError:
            return []

        def ceil_div(value: int, divisor: int) -> int:
            return -(-value // divisor)

        def merge_ns(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
            merged: list[list[int]] = []
            for interval_start, interval_end in sorted(
                value for value in values if value[1] > value[0]
            ):
                if not merged or interval_start > merged[-1][1]:
                    merged.append([interval_start, interval_end])
                else:
                    merged[-1][1] = max(merged[-1][1], interval_end)
            return [(value[0], value[1]) for value in merged]

        def overlap_union_ns(
            speech_value: tuple[int, int] | None,
            body_value: list[tuple[int, int]],
        ) -> tuple[int | None, int | None, float | None]:
            if speech_value is None or speech_value[1] <= speech_value[0]:
                return None, None, None
            merged = merge_ns(body_value)
            if not merged:
                return None, None, None
            intersection = sum(
                max(
                    0,
                    min(speech_value[1], body_end)
                    - max(speech_value[0], body_start),
                )
                for body_start, body_end in merged
            )
            union = (
                speech_value[1]
                - speech_value[0]
                + sum(body_end - body_start for body_start, body_end in merged)
                - intersection
            )
            return (
                intersection,
                union,
                None if union <= 0 else 100.0 * intersection / union,
            )

        clean_user_end = (
            end_sample
            if user_track_end_sample is None
            else max(0, int(user_track_end_sample))
        )
        clean_assistant_end = (
            end_sample
            if assistant_track_end_sample is None
            else max(0, int(assistant_track_end_sample))
        )
        clean_capture_sample = (
            end_sample
            if capture_end_sample is None
            else max(0, int(capture_end_sample))
        )
        clean_capture_frame = (
            end_frame
            if capture_end_frame is None
            else max(0, int(capture_end_frame))
        )
        try:
            clean_stop_anchor_uncertainty_ns = _nonnegative(
                stop_wall_anchor_uncertainty_ns,
                "stop_wall_anchor_uncertainty_ns",
            )
        except ValueError:
            clean_stop_anchor_uncertainty_ns = 0
        clean_wall_clock_offset_change_ns = (
            int(wall_clock_offset_change_ns)
            if isinstance(wall_clock_offset_change_ns, int)
            and not isinstance(wall_clock_offset_change_ns, bool)
            else 0
        )
        mapping_uncertainty_ns = (
            self.wall_anchor_uncertainty_ns
            + abs(clean_wall_clock_offset_change_ns)
            + clean_stop_anchor_uncertainty_ns
        )
        mapping_uncertainty_threshold_ns = 50_000_000
        mapping_uncertainty_exceeds_threshold = (
            mapping_uncertainty_ns > mapping_uncertainty_threshold_ns
        )
        result: list[dict[str, Any]] = []
        recording_end_s = min(
            end_sample / self.sample_rate,
            end_frame / self.fps,
        )
        for key in sorted(
            self._states, key=lambda value: (value.stream_generation, value.reply_seq)
        ):
            state = self._states[key]
            audio_start = (
                None
                if state.audio_start is None
                else state.audio_start / self.sample_rate
            )
            audio_end = (
                None
                if state.audio_end is None
                else state.audio_end / self.sample_rate
            )
            speech_ns = (
                (
                    state.audio_start * 1_000_000_000 // self.sample_rate,
                    ceil_div(
                        state.audio_end * 1_000_000_000, self.sample_rate
                    ),
                )
                if state.audio_start is not None
                and state.audio_end is not None
                and state.audio_end > state.audio_start
                else None
            )
            face_s = (
                None if state.face_frame is None else state.face_frame / self.fps
            )
            face_ns = (
                None
                if state.face_frame is None
                else state.face_frame * 1_000_000_000 // self.fps
            )
            face_last_s = (
                None
                if state.face_last_frame is None
                else state.face_last_frame / self.fps
            )
            action_copies = [dict(value) for value in state.actions]
            semantic_failure_statuses = {
                "failed", "error", "cancelled", "canceled", "aborted"
            }
            result_scope_known = bool(
                state.semantic_terminal_summary_observed
                and state.execution_summary_complete is True
                and state.semantic_status.lower()
                not in semantic_failure_statuses
                and state.result_action_ids_observed
                and state.result_action_ids_valid
            )
            declared_result_action_ids = set(state.result_action_ids)
            declared_semantic_sequence = [
                action_id for action_id in state.result_action_ids
                if not (
                    state.recovery_action_id
                    and action_id == state.recovery_action_id
                )
            ]
            declared_semantic_action_ids = set(declared_semantic_sequence)
            for value in action_copies:
                raw_completed = value.get("raw_completed_interval") is True
                action_id = str(value.get("id", ""))
                if value.get("role") == "recovery":
                    exact_recovery = bool(
                        state.recovery_action_id
                        and action_id == state.recovery_action_id
                    )
                    recovery_matches_result = bool(
                        exact_recovery
                        and state.recovery_required is True
                        and state.result_action_ids.count(action_id) == 1
                        and bool(state.result_action_ids)
                        and state.result_action_ids[-1] == action_id
                        and value.get("ordinal")
                        == len(declared_semantic_sequence) + 1
                    )
                    # Recovery remains fully reconciled and forensic, but
                    # it is not part of the semantic-body timing population.
                    value["included_in_body"] = False
                    value["body_exclusion_basis"] = (
                        "exact_recovery_excluded_from_semantic_body_metrics"
                        if exact_recovery
                        else "recovery_role_excluded_from_semantic_body_metrics"
                    )
                    value["unexpected_recovery"] = (
                        raw_completed and not exact_recovery
                    )
                else:
                    semantic_matches_result = False
                    if (
                        action_id in declared_semantic_action_ids
                        and declared_semantic_sequence.count(action_id) == 1
                    ):
                        expected_index = (
                            declared_semantic_sequence.index(action_id) + 1
                        )
                        semantic_matches_result = bool(
                            value.get("ordinal") == expected_index
                            and expected_index <= len(state.completed_functions)
                            and value.get("function")
                            == state.completed_functions[expected_index - 1]
                        )
                    value["included_in_body"] = bool(
                        raw_completed
                        and (
                            not result_scope_known
                            or semantic_matches_result
                        )
                    )
                value["unexpected_result_action"] = bool(
                    raw_completed
                    and result_scope_known
                    and action_id not in declared_result_action_ids
                )
            actions = sorted(
                action_copies,
                key=lambda value: (
                    float(value["start_s"])
                    if value.get("start_s") is not None
                    else math.inf,
                    str(value["id"]),
                ),
            )
            # Clip forensics retain the exact recovery's observed raw span even
            # though that span is excluded from semantic body metrics.
            recovery_intervals = [
                (float(value["start_s"]), float(value["end_s"]))
                for value in actions
                if value.get("role") == "recovery"
                and state.recovery_action_id
                and value.get("id") == state.recovery_action_id
                and isinstance(value.get("start_s"), (int, float))
                and not isinstance(value.get("start_s"), bool)
                and isinstance(value.get("end_s"), (int, float))
                and not isinstance(value.get("end_s"), bool)
                and math.isfinite(float(value["start_s"]))
                and math.isfinite(float(value["end_s"]))
                and float(value["end_s"]) > float(value["start_s"])
            ]
            recovery_start_candidate = min(
                (value[0] for value in recovery_intervals), default=None
            )
            recovery_end_candidate = max(
                (value[1] for value in recovery_intervals), default=None
            )
            intervals_ns = merge_ns([
                (
                    int(value["mapped_start_monotonic_ns"])
                    - self.started_monotonic_ns,
                    int(value["mapped_end_monotonic_ns"])
                    - self.started_monotonic_ns,
                )
                for value in actions
                if value.get("included_in_body") is True
                and isinstance(value.get("mapped_start_monotonic_ns"), int)
                and isinstance(value.get("mapped_end_monotonic_ns"), int)
            ])
            intervals = [
                (interval_start / 1e9, interval_end / 1e9)
                for interval_start, interval_end in intervals_ns
            ]
            body_start_ns = min(
                (value[0] for value in intervals_ns), default=None
            )
            body_end_ns = max(
                (value[1] for value in intervals_ns), default=None
            )
            body_start = (
                None if body_start_ns is None else body_start_ns / 1e9
            )
            body_end = None if body_end_ns is None else body_end_ns / 1e9
            envelope_ns = (
                [(body_start_ns, body_end_ns)]
                if body_start_ns is not None
                and body_end_ns is not None
                and body_end_ns > body_start_ns
                else []
            )

            body_invalid_reasons: list[str] = []
            failure_statuses = {
                "failed", "error", "cancelled", "canceled", "aborted", "rejected"
            }
            for action in actions:
                status = str(
                    action.get("terminal_status") or action.get("status", "")
                ).lower()
                action_id = str(action["id"])
                if status == "replaced":
                    body_invalid_reasons.append(f"action_replaced:{action_id}")
                elif status in failure_statuses:
                    body_invalid_reasons.append(f"action_failed:{action_id}")
                if (
                    status == "completed"
                    and action.get("raw_completed_interval") is not True
                ):
                    body_invalid_reasons.append(
                        f"action_timing_invalid:{action_id}"
                    )
            completed_receipts = [
                action for action in action_copies
                if str(
                    action.get("terminal_status") or action.get("status", "")
                ).lower() == "completed"
            ]
            completed_receipt_ids = [
                str(action.get("id", "")) for action in completed_receipts
            ]
            expected_semantic_action_ids: list[str] = []
            action_reconciliation_reasons: list[str] = []
            semantic_execution_failed = bool(
                state.execution_summary_complete is False
                or state.semantic_status.lower() in semantic_failure_statuses
            )
            if semantic_execution_failed:
                action_reconciliation_reasons.append(
                    "semantic_execution_failed"
                )
            if (
                state.semantic_terminal_summary_observed
                and state.execution_summary_complete is None
            ):
                action_reconciliation_reasons.append(
                    "execution_summary_completeness_missing"
                )
            successful_execution_summary = bool(
                state.semantic_terminal_summary_observed
                and state.execution_summary_complete is True
                and not semantic_execution_failed
            )
            if successful_execution_summary:
                if not state.result_action_ids_observed:
                    action_reconciliation_reasons.append(
                        "result_action_ids_missing"
                    )
                elif not state.result_action_ids_valid:
                    action_reconciliation_reasons.append(
                        "result_action_ids_malformed"
                    )
                else:
                    result_action_ids = list(state.result_action_ids)
                    result_action_id_set = set(result_action_ids)
                    seen_result_ids: set[str] = set()
                    for action_id in result_action_ids:
                        if action_id in seen_result_ids:
                            action_reconciliation_reasons.append(
                                f"duplicate_result_action_id:{action_id}"
                            )
                        seen_result_ids.add(action_id)
                    expected_semantic_action_ids = [
                        action_id for action_id in result_action_ids
                        if not (
                            state.recovery_action_id
                            and action_id == state.recovery_action_id
                        )
                    ]
                    if (
                        state.recovery_action_id
                        and state.recovery_action_id not in result_action_id_set
                    ):
                        action_reconciliation_reasons.append(
                            "recovery_action_id_not_in_result"
                        )
                    elif (
                        state.recovery_action_id
                        and (
                            not result_action_ids
                            or result_action_ids[-1]
                            != state.recovery_action_id
                        )
                    ):
                        action_reconciliation_reasons.append(
                            "recovery_action_id_not_last"
                        )
                    if state.recovery_action_id and not state.recovery_required:
                        action_reconciliation_reasons.append(
                            f"unexpected_recovery_action_id:{state.recovery_action_id}"
                        )
                    if len(state.completed_functions) != len(
                        expected_semantic_action_ids
                    ):
                        action_reconciliation_reasons.append(
                            "completed_function_count_mismatch"
                        )
                    completed_by_id = {
                        str(action.get("id", "")): action
                        for action in completed_receipts
                    }
                    for index, action_id in enumerate(
                        expected_semantic_action_ids, start=1
                    ):
                        action = completed_by_id.get(action_id)
                        if action is None:
                            action_reconciliation_reasons.append(
                                f"missing_completed_action_receipt:{action_id}"
                            )
                            continue
                        if action.get("role") != "semantic":
                            action_reconciliation_reasons.append(
                                f"action_role_mismatch:{action_id}"
                            )
                        if action.get("ordinal") != index:
                            action_reconciliation_reasons.append(
                                f"action_order_mismatch:{action_id}"
                            )
                        if index <= len(state.completed_functions) and (
                            action.get("function")
                            != state.completed_functions[index - 1]
                        ):
                            action_reconciliation_reasons.append(
                                f"action_function_mismatch:{action_id}"
                            )
                    if state.recovery_action_id:
                        recovery = completed_by_id.get(
                            state.recovery_action_id
                        )
                        if recovery is None:
                            if state.recovery_required:
                                action_reconciliation_reasons.append(
                                    f"missing_completed_action_receipt:{state.recovery_action_id}"
                                )
                        else:
                            if recovery.get("role") != "recovery":
                                action_reconciliation_reasons.append(
                                    f"action_role_mismatch:{state.recovery_action_id}"
                                )
                            if recovery.get("ordinal") != (
                                len(expected_semantic_action_ids) + 1
                            ):
                                action_reconciliation_reasons.append(
                                    f"action_order_mismatch:{state.recovery_action_id}"
                                )
                    for action in completed_receipts:
                        action_id = str(action.get("id", ""))
                        if action_id not in result_action_id_set:
                            action_reconciliation_reasons.append(
                                f"unexpected_completed_action_receipt:{action_id}"
                            )
                    for action_id in state.duplicate_action_ids:
                        action_reconciliation_reasons.append(
                            f"duplicate_action_receipt:{action_id}"
                        )
                    if any(
                        reason.startswith("action_order_mismatch:")
                        for reason in action_reconciliation_reasons
                    ):
                        action_reconciliation_reasons.append(
                            "action_order_mismatch"
                        )
                    for action in action_copies:
                        action_id = str(action.get("id", ""))
                        action["expected_result_action"] = (
                            action_id in result_action_id_set
                        )
                        action["expected_role"] = (
                            "recovery"
                            if state.recovery_action_id
                            and action_id == state.recovery_action_id
                            else (
                                "semantic"
                                if action_id in result_action_id_set
                                else None
                            )
                        )
            body_invalid_reasons.extend(action_reconciliation_reasons)
            has_semantic_body = any(
                action.get("role") == "semantic" for action in actions
            )
            semantic_expected = bool(
                state.semantic_preview_observed
                or state.semantic_request_id
                or state.ordered_functions
                or state.planned_functions
                or has_semantic_body
            )
            if semantic_expected and not state.semantic_terminal_summary_observed:
                body_invalid_reasons.append("semantic_summary_missing")
            recovery_complete = bool(
                state.recovery_action_id
                and any(
                    action.get("role") == "recovery"
                    and action.get("id") == state.recovery_action_id
                    and action.get("raw_completed_interval") is True
                    for action in actions
                )
            )
            if state.recovery_required:
                if not state.recovery_action_id:
                    body_invalid_reasons.extend(
                        ("missing_recovery", "recovery_action_id_missing")
                    )
                elif not recovery_complete:
                    body_invalid_reasons.extend(
                        (
                            "missing_recovery",
                            f"missing_recovery:{state.recovery_action_id}",
                        )
                    )
            if state.closed_neutral is False:
                body_invalid_reasons.append("neutral_not_closed")
            if wall_clock_step_detected and actions:
                body_invalid_reasons.append("wall_clock_step_detected")
            if mapping_uncertainty_exceeds_threshold and actions:
                body_invalid_reasons.append(
                    "mapping_uncertainty_exceeds_threshold"
                )
            body_invalid_reasons.extend(state.provenance_errors)
            body_invalid_reasons = list(dict.fromkeys(body_invalid_reasons))

            terminal_reason = state.terminal_reason or "recording_stopped"
            normally_completed = (
                terminal_reason == "completed"
                and state.media_drained_ns is not None
            )
            right_censored = not normally_completed
            left_censored = state.started_before_recording
            censor_reasons: list[str] = []
            if left_censored:
                censor_reasons.append("reply_started_before_recording")
            if right_censored:
                if terminal_reason in {"completed", "turn_complete"}:
                    censor_reasons.append("media_not_drained")
                else:
                    censor_reasons.append(terminal_reason)

            face_delay_candidate = (
                None
                if face_ns is None or speech_ns is None
                else (face_ns - speech_ns[0]) / 1e6
            )
            body_delay_candidate = (
                None
                if body_start_ns is None or speech_ns is None
                else (body_start_ns - speech_ns[0]) / 1e6
            )
            action_i, action_u, action_tiou_candidate = overlap_union_ns(
                speech_ns, intervals_ns
            )
            env_i, env_u, env_tiou_candidate = overlap_union_ns(
                speech_ns, envelope_ns
            )

            face_reason = (
                ";".join(state.provenance_errors)
                if state.provenance_errors
                else (
                    "reply_started_before_recording"
                    if left_censored
                    else (
                        None
                        if face_delay_candidate is not None
                        else "missing_speech_or_face_boundary"
                    )
                )
            )
            body_reason = (
                ";".join(body_invalid_reasons)
                if body_invalid_reasons
                else (
                    "reply_started_before_recording"
                    if left_censored
                    else (
                        None
                        if body_delay_candidate is not None
                        else "missing_speech_or_body_boundary"
                    )
                )
            )
            tiou_reason = (
                ";".join(body_invalid_reasons)
                if body_invalid_reasons
                else (
                    f"reply_terminal:{terminal_reason}"
                    if right_censored
                    else (
                        "reply_started_before_recording"
                        if left_censored
                        else (
                            None
                            if action_tiou_candidate is not None
                            else "missing_speech_or_body_interval"
                        )
                    )
                )
            )
            face_valid = face_reason is None
            body_valid = body_reason is None
            tiou_valid = tiou_reason is None
            face_delay = face_delay_candidate if face_valid else None
            body_delay = body_delay_candidate if body_valid else None
            action_tiou = action_tiou_candidate if tiou_valid else None
            env_tiou = env_tiou_candidate if tiou_valid else None

            starts = [
                value
                for value in (
                    audio_start, body_start, recovery_start_candidate, face_s
                )
                if value is not None
            ]
            ends = [
                value
                for value in (audio_end, body_end, recovery_end_candidate)
                if value is not None
            ]
            if face_last_s is not None:
                ends.append(face_last_s + 1.0 / self.fps)
            elif face_s is not None:
                ends.append(face_s + 1.0 / self.fps)
            clip = None
            if starts and ends:
                base_start, base_end = min(starts), max(ends)
                requested_start, requested_end = (
                    base_start - 0.5,
                    base_end + 0.5,
                )
                clip_start_s = max(0.0, min(recording_end_s, requested_start))
                clip_end_s = max(
                    clip_start_s, min(recording_end_s, requested_end)
                )
                clip = {
                    "boundary_basis": "common_reply_envelope_with_roll",
                    "clip_pre_roll_s": 0.5,
                    "clip_post_roll_s": 0.5,
                    "base_start_s": base_start,
                    "base_end_s": base_end,
                    "requested_start_s": requested_start,
                    "requested_end_s": requested_end,
                    "start_s": clip_start_s,
                    "end_s": clip_end_s,
                    "artifact_audio_end_sample_exclusive": end_sample,
                    "user_track_end_sample_exclusive": clean_user_end,
                    "assistant_track_end_sample_exclusive": clean_assistant_end,
                    "video_track_end_frame_exclusive": end_frame,
                    "capture_end_sample_exclusive": clean_capture_sample,
                    "capture_end_frame_exclusive": clean_capture_frame,
                    "artifact_common_end_s": recording_end_s,
                    "capture_theoretical_end_s": min(
                        clean_capture_sample / self.sample_rate,
                        clean_capture_frame / self.fps,
                    ),
                    "audio_start_sample": max(
                        0,
                        min(
                            end_sample,
                            math.floor(clip_start_s * self.sample_rate),
                        ),
                    ),
                    "audio_end_sample_exclusive": max(
                        0,
                        min(
                            end_sample,
                            math.ceil(clip_end_s * self.sample_rate),
                        ),
                    ),
                    "video_start_frame": max(
                        0,
                        min(end_frame, math.floor(clip_start_s * self.fps)),
                    ),
                    "video_end_frame_exclusive": max(
                        0,
                        min(end_frame, math.ceil(clip_end_s * self.fps)),
                    ),
                    "left_censored": (
                        left_censored or requested_start < 0.0
                    ),
                    "right_censored": (
                        right_censored or requested_end > recording_end_s
                    ),
                }

            annotation_reasons: list[str] = []
            for prefix, observed, complete, truncated in (
                (
                    "user_text",
                    state.user_text_observed,
                    state.user_text_complete,
                    state.user_text_truncated,
                ),
                (
                    "reply_text",
                    state.reply_text_observed,
                    state.reply_text_complete,
                    state.reply_text_truncated,
                ),
            ):
                if not observed:
                    annotation_reasons.append(f"{prefix}_missing")
                elif truncated:
                    annotation_reasons.append(f"{prefix}_truncated")
                elif not complete:
                    annotation_reasons.append(f"{prefix}_incomplete")
            if semantic_expected and not state.semantic_terminal_summary_observed:
                annotation_reasons.append("semantic_summary_missing")
            if (
                state.semantic_terminal_summary_observed
                and state.execution_summary_complete is not True
            ):
                annotation_reasons.append(
                    "semantic_execution_summary_incomplete"
                )

            reasons: list[str] = []
            for value in (
                *body_invalid_reasons,
                *censor_reasons,
                face_reason,
                body_reason,
                tiou_reason,
            ):
                if value and value not in reasons:
                    reasons.append(value)
            metrics_incomplete = not (
                face_valid and body_valid and tiou_valid
            )
            valid = not metrics_incomplete and not reasons
            formula_definition = {
                "interval_convention": "half_open_integer_nanoseconds",
                "timing_domain": "server_output",
                "speech": "assistant_output_payload_envelope_including_padding_and_writer_gaps",
                "speech_interval": "S=[first_payload_sample,last_payload_sample_exclusive)",
                "speech_not_acoustic": "not physical playout, VAD, or acoustic speech onset/end",
                "delay_sign": "modality_onset_minus_assistant_payload_start;negative_means_modality_precedes_assistant_payload",
                "face_delay_signed": "face_first_new_frame_onset minus assistant_payload_start; negative means face precedes payload",
                "body_delay_signed": "semantic_body_envelope_start minus assistant_payload_start; negative means semantic body precedes payload",
                "body_membership_basis": "completed_semantic_actions_only;exact_recovery_excluded",
                "absolute_delay": "absolute value of corresponding signed delay",
                "body_interval_coordinates": "intervals_union_ns are offsets from recording start; mapped_*_monotonic_ns are absolute monotonic timestamps",
                "mapping_uncertainty": "start_anchor_half_span + abs(stop_minus_start_wall_monotonic_offset_change) + stop_anchor_half_span",
                "wall_clock_mapping_assumption": "no_unobserved_wall_clock_excursion_between_start_and_stop_anchors",
                "mapping_uncertainty_threshold_ns": mapping_uncertainty_threshold_ns,
                "b_union": "merge(completed semantic actions only; exact recovery excluded)",
                "b_env": "B_env=[min(B_union starts),max(B_union ends)); exact recovery excluded",
                "intersection_ns": "measure(S intersection B)",
                "union_ns": "measure(S union B)",
                "tiou_percent": "100 * intersection_ns / union_ns",
            }
            metrics = {
                "boundary_basis": "server_output_recording_timeline",
                "formula_version": "reply_timing_v2_semantic_body_recovery_excluded_v1",
                "formula_definition": formula_definition,
                "face_delay_candidate_ms": face_delay_candidate,
                "face_delay_ms": face_delay,
                "face_delay_abs_ms": (
                    None if face_delay is None else abs(face_delay)
                ),
                "face_delay_valid": face_valid,
                "face_delay_reason": face_reason,
                "body_delay_candidate_ms": body_delay_candidate,
                "body_delay_ms": body_delay,
                "body_delay_abs_ms": (
                    None if body_delay is None else abs(body_delay)
                ),
                "body_delay_valid": body_valid,
                "body_delay_reason": body_reason,
                "speech_body_action_union_intersection_ns": action_i,
                "speech_body_action_union_union_ns": action_u,
                "speech_body_action_union_tiou_percent_candidate": (
                    action_tiou_candidate
                ),
                "speech_body_action_union_tiou_percent": action_tiou,
                "speech_body_action_union_tiou_valid": tiou_valid,
                "speech_body_action_union_tiou_reason": tiou_reason,
                "speech_body_envelope_intersection_ns": env_i,
                "speech_body_envelope_union_ns": env_u,
                "speech_body_envelope_tiou_percent_candidate": (
                    env_tiou_candidate
                ),
                "speech_body_envelope_tiou_percent": env_tiou,
                "speech_body_envelope_tiou_valid": tiou_valid,
                "speech_body_envelope_tiou_reason": tiou_reason,
                "reasons": {
                    "face_delay_ms": face_reason,
                    "body_delay_ms": body_reason,
                    "speech_body_action_union_tiou_percent": tiou_reason,
                    "speech_body_envelope_tiou_percent": tiou_reason,
                },
            }
            row = {
                "schema_version": "experiment_reply_v2",
                "recording_id": self.recording_id,
                "reply_index": key.reply_seq,
                "turn_id": state.semantic_turn_id,
                "identity": {
                    "recording_id": self.recording_id,
                    **key.as_dict(),
                    "semantic_turn_id": state.semantic_turn_id,
                    "semantic_request_id": state.semantic_request_id,
                    "semantic_revision": state.semantic_revision or None,
                },
                "reply_started": {
                    "observed_monotonic_ns": state.first_output_ns,
                    "cause": state.reply_start_cause or None,
                    "started_before_recording": state.started_before_recording,
                },
                "text": {
                    "user_text": state.user_text,
                    "user_text_sha256": (
                        state.user_text_sha256 or _sha256_text("")
                    ),
                    "user_text_truncated": state.user_text_truncated,
                    "reply_text": state.reply_text,
                    "reply_text_sha256": (
                        state.reply_text_sha256 or _sha256_text("")
                    ),
                    "reply_text_truncated": state.reply_text_truncated,
                    "observed_monotonic_ns": state.transcript_observed_ns,
                    "user_text_observed": state.user_text_observed,
                    "reply_text_observed": state.reply_text_observed,
                    "user_text_complete": state.user_text_complete,
                    "reply_text_complete": state.reply_text_complete,
                },
                "semantics": {
                    "status": state.semantic_status or None,
                    "summary_observed": state.semantic_terminal_summary_observed,
                    "preview_observed": state.semantic_preview_observed,
                    "preview_observed_monotonic_ns": state.semantic_preview_observed_ns,
                    "summary_observed_monotonic_ns": (
                        state.semantic_summary_observed_ns
                    ),
                    "ordered_functions": list(state.ordered_functions),
                    "planned_functions": list(state.planned_functions),
                    "completed_functions": list(state.completed_functions),
                    "dropped_functions": list(state.dropped_functions),
                    "result_action_ids": list(state.result_action_ids),
                    "result_action_ids_observed": state.result_action_ids_observed,
                    "execution_summary_complete": state.execution_summary_complete,
                    "expected_semantic_action_ids": list(
                        expected_semantic_action_ids
                    ),
                    "action_reconciliation_valid": not bool(
                        action_reconciliation_reasons
                    ),
                    "action_reconciliation_reasons": list(
                        action_reconciliation_reasons
                    ),
                    "completed_receipt_action_ids": list(
                        completed_receipt_ids
                    ),
                    "selected_atoms_basis": "terminal_observed_submitted_semantic_action_receipts",
                    "selected_atoms": _json_snapshot(state.selected_atoms),
                    "selected_atoms_sha256": (
                        state.selected_atoms_sha256 or _json_sha256([])
                    ),
                    "recovery_required": state.recovery_required,
                    "recovery_action_id": state.recovery_action_id,
                    "closed_neutral": state.closed_neutral,
                },
                "terminal": {
                    "intent_reason": state.terminal_intent_reason or None,
                    "intent_observed_monotonic_ns": state.terminal_intent_ns,
                    "media_drained_monotonic_ns": state.media_drained_ns,
                    "reason": terminal_reason,
                    "observed_monotonic_ns": state.terminal_ns,
                    "media_fence": dict(state.media_fence),
                },
                "audio_handoff": {
                    "boundary_basis": "server_webrtc_audio_payload_written",
                    "start_sample": state.audio_start,
                    "end_sample_exclusive": state.audio_end,
                    "start_s": audio_start,
                    "end_s": audio_end,
                    "first_packet": dict(state.audio_first_meta),
                    "last_packet": dict(state.audio_last_meta),
                },
                "face": {
                    "boundary_basis": "final_composite_frame_written",
                    "first_new_frame_index": state.face_frame,
                    "first_new_frame_s": face_s,
                    "source_seq": state.face_source_seq,
                    "last_new_frame_index": state.face_last_frame,
                    "last_new_frame_s": face_last_s,
                    "last_source_seq": state.face_last_source_seq,
                    "fresh_source_count": state.face_fresh_source_count,
                    "first_packet": dict(state.face_meta),
                    "last_packet": dict(state.face_last_meta),
                    **dict(state.face_meta),
                },
                "body": {
                    "body_membership_basis": "completed_semantic_actions_only;exact_recovery_excluded",
                    "boundary_basis": "simulator_action_wall_mapped",
                    "start_s": body_start,
                    "end_s": body_end,
                    "semantic_action_end_s": body_end,
                    "mapping_uncertainty_ns": mapping_uncertainty_ns,
                    "mapping_uncertainty_ms": mapping_uncertainty_ns / 1e6,
                    "mapping_uncertainty_basis": "start_anchor_half_span_plus_abs_offset_change_plus_stop_anchor_half_span",
                    "start_wall_anchor_uncertainty_ns": (
                        self.wall_anchor_uncertainty_ns
                    ),
                    "stop_wall_anchor_uncertainty_ns": (
                        clean_stop_anchor_uncertainty_ns
                    ),
                    "wall_clock_offset_change_ns": (
                        clean_wall_clock_offset_change_ns
                    ),
                    "mapping_uncertainty_threshold_ns": (
                        mapping_uncertainty_threshold_ns
                    ),
                    "mapping_uncertainty_exceeds_threshold": (
                        mapping_uncertainty_exceeds_threshold
                    ),
                    "intervals_s": intervals,
                    "intervals_union_ns": [
                        [interval_start, interval_end]
                        for interval_start, interval_end in intervals_ns
                    ],
                    "envelope_ns": (
                        None
                        if not envelope_ns
                        else [envelope_ns[0][0], envelope_ns[0][1]]
                    ),
                    "actions": actions,
                },
                "clip": clip,
                "metrics": metrics,
                "quality": {
                    "valid": valid,
                    "metrics_incomplete": metrics_incomplete,
                    "left_censored": left_censored,
                    "right_censored": right_censored,
                    "reasons": reasons,
                    "annotation_complete": not annotation_reasons,
                    "annotation_reasons": annotation_reasons,
                    "speech_definition": "assistant_output_payload_envelope_including_padding_and_writer_gaps",
                    "physical_playout_measured": False,
                    "vad_applied": False,
                    "user_audio_boundary": "post_sdk_send_approx",
                },
                "speech_start_s": audio_start,
                "speech_end_s": audio_end,
                "face_first_new_frame_s": face_s,
                "face_last_new_frame_s": face_last_s,
                "body_start_s": body_start,
                "body_end_s": body_end,
                "semantic_action_end_s": body_end,
                "clip_start_s": None if clip is None else clip["start_s"],
                "clip_end_s": None if clip is None else clip["end_s"],
                "valid_for_metrics": valid,
                "failure_reason": None if not reasons else ";".join(reasons),
            }
            result.append(row)
        return result


__all__ = ["FrameProvenance", "ReplyKey", "ReplyMetricsAccumulator", "ReplyTraceHub", "TaggedReplyQueue", "install_submission_receipt_observer"]
