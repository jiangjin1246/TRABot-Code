"""Monotonic reply-duration budget used by the semantic gesture runtime.

Wall-clock timestamps are deliberately absent from this module.  A clock is
bound to one trusted request/revision and becomes invalid if its injected
monotonic source ever moves backwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Callable


MAX_CLOCK_NS = (1 << 63) - 1
MIN_REPLY_DURATION_US = 100_000
# This is an integer/overflow guard, not a product-level reply limit.  The
# actual admissible maximum is lower when added to a non-zero monotonic epoch.
MAX_REPLY_DURATION_US = MAX_CLOCK_NS // 1_000
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ReplyClockError(ValueError):
    """The reply clock or one of its exact bindings is invalid."""


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReplyClockError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ReplyClockSnapshot:
    request_id: str
    revision: int
    started_monotonic_ns: int
    deadline_monotonic_ns: int
    observed_monotonic_ns: int
    elapsed_ns: int
    remaining_ns: int
    expired: bool

    @property
    def elapsed_us(self) -> int:
        return self.elapsed_ns // 1_000

    @property
    def remaining_us(self) -> int:
        # Floor is conservative: a sub-microsecond remainder is not advertised
        # to a planner as usable time.
        return self.remaining_ns // 1_000

    @property
    def remaining_seconds(self) -> float:
        return self.remaining_ns / 1_000_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "revision": self.revision,
            "started_monotonic_ns": self.started_monotonic_ns,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "elapsed_ns": self.elapsed_ns,
            "remaining_ns": self.remaining_ns,
            "elapsed_us": self.elapsed_us,
            "remaining_us": self.remaining_us,
            "expired": self.expired,
        }


class ReplyClock:
    """Thread-safe, monotonic budget for one semantic reply revision."""

    def __init__(
        self,
        request_id: str,
        revision: int,
        duration_us: int,
        *,
        effective_duration_us: int | None = None,
        budget_policy: str = "gemini_estimate_v1",
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        started_monotonic_ns: int | None = None,
    ) -> None:
        if not isinstance(request_id, str) or _REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ReplyClockError("request_id has an invalid shape")
        self.request_id = request_id
        self.revision = _positive_int(revision, "revision")
        self.duration_us = _positive_int(duration_us, "duration_us")
        if self.duration_us < MIN_REPLY_DURATION_US:
            raise ReplyClockError(
                f"duration_us is below the {MIN_REPLY_DURATION_US} minimum"
            )
        if self.duration_us > MAX_REPLY_DURATION_US:
            raise ReplyClockError(
                f"duration_us exceeds the {MAX_REPLY_DURATION_US} maximum"
            )
        if effective_duration_us is None:
            effective = self.duration_us
        else:
            effective = _positive_int(
                effective_duration_us, "effective_duration_us"
            )
        if effective < MIN_REPLY_DURATION_US:
            raise ReplyClockError(
                f"effective_duration_us is below the {MIN_REPLY_DURATION_US} minimum"
            )
        if effective > MAX_REPLY_DURATION_US:
            raise ReplyClockError(
                f"effective_duration_us exceeds the {MAX_REPLY_DURATION_US} maximum"
            )
        if (
            not isinstance(budget_policy, str)
            or not budget_policy
            or len(budget_policy) > 64
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", budget_policy) is None
        ):
            raise ReplyClockError("budget_policy has an invalid shape")
        self.effective_duration_us = effective
        self.budget_policy = budget_policy
        if not callable(monotonic_ns):
            raise ReplyClockError("monotonic_ns must be callable")
        self._monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        observed = self._read_monotonic()
        if started_monotonic_ns is None:
            started = observed
        else:
            if (
                isinstance(started_monotonic_ns, bool)
                or not isinstance(started_monotonic_ns, int)
                or started_monotonic_ns < 0
                or started_monotonic_ns > MAX_CLOCK_NS
            ):
                raise ReplyClockError(
                    "started_monotonic_ns must be a non-negative clock integer"
                )
            if started_monotonic_ns > observed:
                raise ReplyClockError(
                    "started_monotonic_ns cannot be in the monotonic future"
                )
            started = started_monotonic_ns
        duration_ns = self.effective_duration_us * 1_000
        if started > MAX_CLOCK_NS - duration_ns:
            raise ReplyClockError("reply deadline overflows the clock range")
        self.started_monotonic_ns = started
        self.deadline_monotonic_ns = started + duration_ns
        # Construction itself samples the source.  A later observation below
        # this value is a backwards clock even when a trusted earlier start was
        # supplied for a late-arriving tool call.
        self._last_observed_ns = observed

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        effective_duration_us: int | None = None,
        budget_policy: str = "gemini_estimate_v1",
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        started_monotonic_ns: int | None = None,
    ) -> "ReplyClock":
        try:
            request_id = payload.request_id
            revision = payload.revision
            duration_us = payload.estimated_spoken_duration_us
        except AttributeError as exc:
            raise ReplyClockError("payload does not expose the semantic clock binding") from exc
        return cls(
            request_id,
            revision,
            duration_us,
            effective_duration_us=effective_duration_us,
            budget_policy=budget_policy,
            monotonic_ns=monotonic_ns,
            started_monotonic_ns=started_monotonic_ns,
        )

    def _read_monotonic(self) -> int:
        value = self._monotonic_ns()
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > MAX_CLOCK_NS
        ):
            raise ReplyClockError("monotonic_ns returned an invalid integer")
        return value

    def assert_binding(self, request_id: str, revision: int, duration_us: int) -> None:
        if (
            request_id != self.request_id
            or revision != self.revision
            or duration_us != self.duration_us
        ):
            raise ReplyClockError("reply clock binding does not match request/revision/duration")

    def snapshot(self) -> ReplyClockSnapshot:
        with self._lock:
            observed = self._read_monotonic()
            if observed < self._last_observed_ns:
                raise ReplyClockError("monotonic clock moved backwards")
            self._last_observed_ns = observed
            elapsed = observed - self.started_monotonic_ns
            remaining = max(self.deadline_monotonic_ns - observed, 0)
            return ReplyClockSnapshot(
                request_id=self.request_id,
                revision=self.revision,
                started_monotonic_ns=self.started_monotonic_ns,
                deadline_monotonic_ns=self.deadline_monotonic_ns,
                observed_monotonic_ns=observed,
                elapsed_ns=elapsed,
                remaining_ns=remaining,
                expired=observed >= self.deadline_monotonic_ns,
            )

    def remaining_seconds(self) -> float:
        return self.snapshot().remaining_seconds


__all__ = [
    "MAX_CLOCK_NS",
    "MAX_REPLY_DURATION_US",
    "MIN_REPLY_DURATION_US",
    "ReplyClock",
    "ReplyClockError",
    "ReplyClockSnapshot",
]
