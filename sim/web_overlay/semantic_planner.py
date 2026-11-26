"""Pure, deterministic semantic Atom prefix planner.

The planner deliberately knows nothing about HTTP, MuJoCo, or the live dialogue
loop.  Callers provide an already validated/immutable catalogue, an ordered list
of semantic functions, and pure cost callbacks.  Durations and all costs are
integer microseconds; speed is therefore fixed at 1.0 and no per-Atom duration
policy is applied here.

Every returned non-empty plan satisfies this strict budget equation::

    motions + entry/inter-Atom transitions + final recovery + tail guard
        <= budget

The search never skips or reorders a requested function.  Its first objective is
the longest contiguous prefix, followed by the deterministic tie-breaks encoded
in ``_score_state`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import heapq
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


FIT_BUCKET_US = 500_000
# Frozen latency/quality trade-off.  Primary prefix length is solved exactly by
# a separate DP and therefore does not depend on this secondary beam width.
SECONDARY_BEAM_WIDTH = 32
_MISSING = object()
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VALID_SIDES = frozenset(("left", "right", "bilateral", "none"))
_UNILATERAL_SIDES = frozenset(("left", "right"))


class PlannerValidationError(ValueError):
    """Raised when planner input violates the fail-closed catalogue contract."""


@dataclass(frozen=True, order=True)
class ExactAtomRef:
    """Content-addressed identity of one immutable approved Atom revision."""

    atom_id: str
    version: str
    revision: int
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.atom_id, str) or not self.atom_id:
            raise PlannerValidationError("ExactAtomRef.atom_id must be non-empty")
        if not isinstance(self.version, str) or not self.version:
            raise PlannerValidationError("ExactAtomRef.version must be non-empty")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 0
        ):
            raise PlannerValidationError(
                "ExactAtomRef.revision must be a non-negative integer"
            )
        if (
            not isinstance(self.content_sha256, str)
            or _SHA256_RE.fullmatch(self.content_sha256) is None
        ):
            raise PlannerValidationError(
                "ExactAtomRef.content_sha256 must be 64 lowercase hexadecimal characters"
            )


@dataclass(frozen=True)
class PlannerHistory:
    """Immutable cross-reply history used only by deterministic tie-breaks.

    The planner does not trim these tuples because the runtime owns its cooldown
    window policy.  Pass an already bounded history if only recent replies should
    count.
    """

    recent_atom_refs: tuple[ExactAtomRef, ...] = ()
    recent_source_ids: tuple[str, ...] = ()
    last_unilateral_side: str | None = None
    unilateral_run_length: int = 0
    unilateral_balance: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.recent_atom_refs, tuple) or any(
            not isinstance(ref, ExactAtomRef) for ref in self.recent_atom_refs
        ):
            raise PlannerValidationError(
                "history.recent_atom_refs must be a tuple of ExactAtomRef values"
            )
        if not isinstance(self.recent_source_ids, tuple) or any(
            not isinstance(source, str) or not source
            for source in self.recent_source_ids
        ):
            raise PlannerValidationError(
                "history.recent_source_ids must be a tuple of non-empty strings"
            )
        if self.last_unilateral_side not in (None, "left", "right"):
            raise PlannerValidationError(
                "history.last_unilateral_side must be left, right, or None"
            )
        _require_nonnegative_int(
            self.unilateral_run_length, "history.unilateral_run_length"
        )
        if not isinstance(self.unilateral_balance, int) or isinstance(
            self.unilateral_balance, bool
        ):
            raise PlannerValidationError("history.unilateral_balance must be an integer")
        if self.last_unilateral_side is None and self.unilateral_run_length != 0:
            raise PlannerValidationError(
                "history.unilateral_run_length must be zero when no side is active"
            )
        if self.last_unilateral_side is not None and self.unilateral_run_length == 0:
            raise PlannerValidationError(
                "history.unilateral_run_length must be positive when a side is active"
            )


@dataclass(frozen=True)
class PlanResult:
    """Stable output contract consumed by the dialogue/runtime integration."""

    atoms: tuple[Any, ...]
    exact_refs: tuple[ExactAtomRef, ...]
    covered_functions: tuple[str, ...]
    remaining_functions: tuple[str, ...]
    used_budget_us: int
    unused_budget_us: int
    prefix_length: int
    fit_bucket: int
    motion_and_transition_us: int
    final_recovery_us: int
    tail_guard_us: int
    history_after: PlannerHistory
    secondary_search_complete: bool
    secondary_beam_width: int


@dataclass(frozen=True)
class _Candidate:
    original: Any
    exact_ref: ExactAtomRef
    function: str
    duration_us: int
    side: str
    source_id: str | None


@dataclass(frozen=True)
class _SearchState:
    candidates: tuple[_Candidate, ...]
    path_cost_us: int
    atom_repeat_penalty: int
    source_repeat_penalty: int
    last_unilateral_side: str | None
    unilateral_run_length: int
    max_unilateral_run: int
    same_unilateral_edges: int
    unilateral_balance: int


@dataclass(frozen=True)
class _ScoredPlan:
    state: _SearchState
    used_budget_us: int
    final_recovery_us: int
    rank: tuple[Any, ...]


class _CostOracle:
    """Memoize pure edge/terminal callbacks by exact Atom identity."""

    def __init__(
        self,
        transition: TransitionCost,
        recovery: FinalRecoveryCost | None,
        previous_atom: Any | None,
    ) -> None:
        self._transition = transition
        self._recovery = recovery
        self._previous_atom = previous_atom
        self._initial: dict[ExactAtomRef, int] = {}
        self._edges: dict[tuple[ExactAtomRef, ExactAtomRef], int] = {}
        self._terminal: dict[ExactAtomRef, int] = {}

    def transition(self, previous: _Candidate | None, candidate: _Candidate) -> int:
        if previous is None:
            if candidate.exact_ref not in self._initial:
                self._initial[candidate.exact_ref] = _call_nonnegative_cost(
                    self._transition,
                    (self._previous_atom, candidate.original),
                    "transition_cost_us",
                    candidate.exact_ref,
                )
            return self._initial[candidate.exact_ref]
        key = (previous.exact_ref, candidate.exact_ref)
        if key not in self._edges:
            self._edges[key] = _call_nonnegative_cost(
                self._transition,
                (previous.original, candidate.original),
                "transition_cost_us",
                candidate.exact_ref,
            )
        return self._edges[key]

    def recovery(self, candidate: _Candidate) -> int:
        if self._recovery is None:
            return 0
        if candidate.exact_ref not in self._terminal:
            self._terminal[candidate.exact_ref] = _call_nonnegative_cost(
                self._recovery,
                (candidate.original,),
                "final_recovery_cost_us",
                candidate.exact_ref,
            )
        return self._terminal[candidate.exact_ref]


TransitionCost = Callable[[Any | None, Any], int]
FinalRecoveryCost = Callable[[Any], int]


def plan_prefix(
    functions: Sequence[str],
    catalog: Iterable[Any],
    *,
    transition_cost_us: TransitionCost,
    budget_us: int,
    history: PlannerHistory | None = None,
    previous_atom: Any | None = None,
    final_recovery_cost_us: FinalRecoveryCost | None = None,
    tail_guard_us: int = 0,
) -> PlanResult:
    """Select the best exact Atoms for the longest feasible function prefix.

    ``transition_cost_us`` is called with ``(previous, candidate)``.  For the
    first candidate, ``previous`` is the caller-provided ``previous_atom``; this
    may be an Atom or an opaque measured-state object understood by the callback.
    Later calls receive the previously selected original catalogue entry.

    ``final_recovery_cost_us`` is evaluated for the last Atom of each prospective
    plan.  It is separate from ``transition_cost_us`` so a runtime can account for
    measured-state entry, inter-Atom bridges, and the final return-to-neutral gate
    without hiding any cost outside the strict budget.
    """

    ordered_functions = _normalise_functions(functions)
    candidates = _normalise_catalog(catalog)
    _require_nonnegative_int(budget_us, "budget_us")
    _require_nonnegative_int(tail_guard_us, "tail_guard_us")
    if not callable(transition_cost_us):
        raise PlannerValidationError("transition_cost_us must be callable")
    if final_recovery_cost_us is not None and not callable(final_recovery_cost_us):
        raise PlannerValidationError("final_recovery_cost_us must be callable")
    if history is None:
        history = PlannerHistory()
    elif not isinstance(history, PlannerHistory):
        raise PlannerValidationError("history must be a PlannerHistory value")

    candidates_by_function: dict[str, tuple[_Candidate, ...]] = {}
    catalogue_functions = {candidate.function for candidate in candidates}
    for function in ordered_functions:
        if function not in catalogue_functions:
            raise PlannerValidationError(f"unknown function: {function!r}")
    for function in sorted(catalogue_functions):
        matching = tuple(
            sorted(
                (candidate for candidate in candidates if candidate.function == function),
                key=lambda candidate: (candidate.duration_us, candidate.exact_ref),
            )
        )
        candidates_by_function[function] = matching

    initial = _SearchState(
        candidates=(),
        path_cost_us=0,
        atom_repeat_penalty=0,
        source_repeat_penalty=0,
        last_unilateral_side=history.last_unilateral_side,
        unilateral_run_length=history.unilateral_run_length,
        max_unilateral_run=history.unilateral_run_length,
        same_unilateral_edges=0,
        unilateral_balance=history.unilateral_balance,
    )
    history_atom_counts = _counts(history.recent_atom_refs)
    history_source_counts = _counts(history.recent_source_ids)
    oracle = _CostOracle(
        transition_cost_us,
        final_recovery_cost_us,
        previous_atom,
    )
    target_depth, witness = _longest_feasible_prefix(
        ordered_functions,
        candidates_by_function,
        oracle=oracle,
        budget_us=budget_us,
        tail_guard_us=tail_guard_us,
    )
    best, secondary_search_complete = _optimise_target_prefix(
        initial,
        ordered_functions,
        candidates_by_function,
        target_depth=target_depth,
        witness=witness,
        oracle=oracle,
        budget_us=budget_us,
        tail_guard_us=tail_guard_us,
        history_atom_counts=history_atom_counts,
        history_source_counts=history_source_counts,
    )
    selected = best.state.candidates
    atoms = tuple(candidate.original for candidate in selected)
    exact_refs = tuple(candidate.exact_ref for candidate in selected)
    covered = ordered_functions[: len(selected)]
    remaining = ordered_functions[len(selected) :]
    history_after = _append_history(history, selected)
    unused = budget_us - best.used_budget_us

    return PlanResult(
        atoms=atoms,
        exact_refs=exact_refs,
        covered_functions=covered,
        remaining_functions=remaining,
        used_budget_us=best.used_budget_us,
        unused_budget_us=unused,
        prefix_length=len(selected),
        fit_bucket=unused // FIT_BUCKET_US,
        motion_and_transition_us=best.state.path_cost_us,
        final_recovery_us=best.final_recovery_us,
        tail_guard_us=tail_guard_us if selected else 0,
        history_after=history_after,
        secondary_search_complete=secondary_search_complete,
        secondary_beam_width=SECONDARY_BEAM_WIDTH,
    )


def replan_suffix(
    remaining_functions: Sequence[str],
    catalog: Iterable[Any],
    *,
    transition_cost_us: TransitionCost,
    remaining_budget_us: int,
    history: PlannerHistory | None = None,
    previous_atom: Any | None = None,
    final_recovery_cost_us: FinalRecoveryCost | None = None,
    tail_guard_us: int = 0,
) -> PlanResult:
    """Pure runtime replan entry point for an unexecuted function suffix."""

    return plan_prefix(
        remaining_functions,
        catalog,
        transition_cost_us=transition_cost_us,
        budget_us=remaining_budget_us,
        history=history,
        previous_atom=previous_atom,
        final_recovery_cost_us=final_recovery_cost_us,
        tail_guard_us=tail_guard_us,
    )


def update_history(
    history: PlannerHistory, atoms: Sequence[Any]
) -> PlannerHistory:
    """Return history with selected catalogue entries appended, without mutation."""

    if not isinstance(history, PlannerHistory):
        raise PlannerValidationError("history must be a PlannerHistory value")
    candidates = tuple(_normalise_entry(atom, index) for index, atom in enumerate(atoms))
    return _append_history(history, candidates)


def _longest_feasible_prefix(
    functions: tuple[str, ...],
    candidates_by_function: Mapping[str, tuple[_Candidate, ...]],
    *,
    oracle: _CostOracle,
    budget_us: int,
    tail_guard_us: int,
) -> tuple[int, tuple[_Candidate, ...]]:
    """Find the exact longest prefix in polynomial time.

    Transition cost is Markovian by contract: it depends on the previous exact
    Atom (or initial measured state) and the candidate.  Therefore, at a fixed
    depth, a higher-cost path ending at the same exact Atom can never make a
    deeper prefix feasible when its lower-cost counterpart cannot.  Keeping the
    minimum-cost path per last Atom is consequently an exact dominance rule for
    the primary longest-prefix objective.
    """

    # exact ref -> (motion/transition cost, deterministic witness path)
    layer: dict[ExactAtomRef, tuple[int, tuple[_Candidate, ...]]] = {}
    longest = 0
    witness: tuple[_Candidate, ...] = ()

    for depth, function in enumerate(functions):
        next_layer: dict[ExactAtomRef, tuple[int, tuple[_Candidate, ...]]] = {}
        options = candidates_by_function[function]
        if depth == 0:
            predecessors: tuple[tuple[int, tuple[_Candidate, ...]], ...] = ((0, ()),)
        else:
            predecessors = tuple(layer.values())

        for previous_cost, previous_path in predecessors:
            previous = previous_path[-1] if previous_path else None
            for candidate in options:
                cost = (
                    previous_cost
                    + oracle.transition(previous, candidate)
                    + candidate.duration_us
                )
                if cost > budget_us:
                    continue
                path = previous_path + (candidate,)
                existing = next_layer.get(candidate.exact_ref)
                path_key = tuple(item.exact_ref for item in path)
                if existing is None or (cost, path_key) < (
                    existing[0],
                    tuple(item.exact_ref for item in existing[1]),
                ):
                    next_layer[candidate.exact_ref] = (cost, path)

        if not next_layer:
            break

        feasible: list[tuple[int, tuple[ExactAtomRef, ...], tuple[_Candidate, ...]]] = []
        for cost, path in next_layer.values():
            used = cost + oracle.recovery(path[-1]) + tail_guard_us
            if used <= budget_us:
                feasible.append(
                    (used, tuple(item.exact_ref for item in path), path)
                )
        if feasible:
            _, _, witness = min(feasible)
            longest = depth + 1
        layer = next_layer

    return longest, witness


def _optimise_target_prefix(
    initial: _SearchState,
    functions: tuple[str, ...],
    candidates_by_function: Mapping[str, tuple[_Candidate, ...]],
    *,
    target_depth: int,
    witness: tuple[_Candidate, ...],
    oracle: _CostOracle,
    budget_us: int,
    tail_guard_us: int,
    history_atom_counts: Mapping[ExactAtomRef, int],
    history_source_counts: Mapping[str, int],
) -> tuple[_ScoredPlan, bool]:
    """Optimise secondary objectives with a frozen deterministic beam.

    The exact dynamic programme above has already fixed the globally longest
    prefix and supplies a feasible witness.  The witness prefix is forcibly
    retained at every truncated layer, so this bounded secondary search can
    never shorten the primary result.  ``secondary_search_complete`` tells the
    caller whether all secondary alternatives were enumerated (small case) or
    the frozen beam policy was used (large case).
    """

    if target_depth == 0:
        scored = _score_state(
            initial,
            budget_us=budget_us,
            oracle=oracle,
            tail_guard_us=tail_guard_us,
        )
        assert scored is not None
        return scored, True

    candidate_by_ref = {
        candidate.exact_ref: candidate
        for options in candidates_by_function.values()
        for candidate in options
    }

    @lru_cache(maxsize=None)
    def minimum_tail_cost(next_depth: int, last_ref: ExactAtomRef) -> int:
        last = candidate_by_ref[last_ref]
        if next_depth == target_depth:
            return oracle.recovery(last) + tail_guard_us
        return min(
            oracle.transition(last, candidate)
            + candidate.duration_us
            + minimum_tail_cost(next_depth + 1, candidate.exact_ref)
            for candidate in candidates_by_function[functions[next_depth]]
        )

    beam: tuple[_SearchState, ...] = (initial,)
    witness_refs = tuple(candidate.exact_ref for candidate in witness)
    search_complete = True

    for depth in range(target_depth):
        children: list[_SearchState] = []
        for state in beam:
            previous = state.candidates[-1] if state.candidates else None
            selected_atom_counts = _counts(
                candidate.exact_ref for candidate in state.candidates
            )
            selected_source_counts = _counts(
                candidate.source_id
                for candidate in state.candidates
                if candidate.source_id is not None
            )
            for candidate in candidates_by_function[functions[depth]]:
                path_cost = (
                    state.path_cost_us
                    + oracle.transition(previous, candidate)
                    + candidate.duration_us
                )
                if path_cost > budget_us:
                    continue
                child = _extend_state(
                    state,
                    candidate,
                    path_cost_us=path_cost,
                    history_atom_counts=history_atom_counts,
                    history_source_counts=history_source_counts,
                    selected_atom_counts=selected_atom_counts,
                    selected_source_counts=selected_source_counts,
                )
                # This exact partial path must still be able to reach the target
                # and close through recovery+guard within the budget.
                if (
                    child.path_cost_us
                    + minimum_tail_cost(depth + 1, candidate.exact_ref)
                    <= budget_us
                ):
                    children.append(child)

        if not children:
            raise RuntimeError("planner lost its exact feasible-prefix witness")

        if depth + 1 == target_depth:
            scored_children = (
                _score_state(
                    child,
                    budget_us=budget_us,
                    oracle=oracle,
                    tail_guard_us=tail_guard_us,
                )
                for child in children
            )
            feasible = tuple(item for item in scored_children if item is not None)
            if not feasible:
                raise RuntimeError("planner target prefix has no closed feasible path")
            return min(feasible, key=lambda item: item.rank), search_complete

        key = lambda state: _partial_rank(
            state,
            future_cost_us=minimum_tail_cost(
                depth + 1, state.candidates[-1].exact_ref
            ),
            budget_us=budget_us,
        )
        if len(children) <= SECONDARY_BEAM_WIDTH:
            beam = tuple(sorted(children, key=key))
            continue

        search_complete = False
        kept = heapq.nsmallest(SECONDARY_BEAM_WIDTH, children, key=key)
        witness_prefix = witness_refs[: depth + 1]
        kept_paths = {
            tuple(candidate.exact_ref for candidate in state.candidates)
            for state in kept
        }
        if witness_prefix not in kept_paths:
            witness_state = next(
                (
                    state
                    for state in children
                    if tuple(
                        candidate.exact_ref for candidate in state.candidates
                    )
                    == witness_prefix
                ),
                None,
            )
            if witness_state is None:
                raise RuntimeError("planner witness disappeared during beam pruning")
            kept.append(witness_state)
        beam = tuple(sorted(kept, key=key))

    raise RuntimeError("planner secondary search ended before its target depth")


def _extend_state(
    state: _SearchState,
    candidate: _Candidate,
    *,
    path_cost_us: int,
    history_atom_counts: Mapping[ExactAtomRef, int],
    history_source_counts: Mapping[str, int],
    selected_atom_counts: Mapping[ExactAtomRef, int],
    selected_source_counts: Mapping[str, int],
) -> _SearchState:
    side_state = _advance_side(state, candidate.side)
    atom_repeat = (
        state.atom_repeat_penalty
        + history_atom_counts.get(candidate.exact_ref, 0)
        + selected_atom_counts.get(candidate.exact_ref, 0)
    )
    source_repeat = state.source_repeat_penalty
    if candidate.source_id is not None:
        source_repeat += history_source_counts.get(candidate.source_id, 0)
        source_repeat += selected_source_counts.get(candidate.source_id, 0)
    return _SearchState(
        candidates=state.candidates + (candidate,),
        path_cost_us=path_cost_us,
        atom_repeat_penalty=atom_repeat,
        source_repeat_penalty=source_repeat,
        last_unilateral_side=side_state[0],
        unilateral_run_length=side_state[1],
        max_unilateral_run=side_state[2],
        same_unilateral_edges=side_state[3],
        unilateral_balance=side_state[4],
    )


def _partial_rank(
    state: _SearchState, *, future_cost_us: int, budget_us: int
) -> tuple[Any, ...]:
    forecast_unused = budget_us - state.path_cost_us - future_cost_us
    return (
        forecast_unused // FIT_BUCKET_US,
        state.atom_repeat_penalty,
        state.max_unilateral_run,
        state.same_unilateral_edges,
        abs(state.unilateral_balance),
        forecast_unused,
        state.source_repeat_penalty,
        tuple(candidate.exact_ref for candidate in state.candidates),
    )


def _score_state(
    state: _SearchState,
    *,
    budget_us: int,
    oracle: _CostOracle,
    tail_guard_us: int,
) -> _ScoredPlan | None:
    if not state.candidates:
        recovery = 0
        guard = 0
    else:
        recovery = oracle.recovery(state.candidates[-1])
        guard = tail_guard_us
    used = state.path_cost_us + recovery + guard
    if used > budget_us:
        return None
    unused = budget_us - used
    exact_path = tuple(candidate.exact_ref for candidate in state.candidates)
    rank = (
        -len(state.candidates),
        unused // FIT_BUCKET_US,
        state.atom_repeat_penalty,
        state.max_unilateral_run,
        state.same_unilateral_edges,
        abs(state.unilateral_balance),
        unused,
        state.source_repeat_penalty,
        exact_path,
    )
    return _ScoredPlan(
        state=state,
        used_budget_us=used,
        final_recovery_us=recovery,
        rank=rank,
    )


def _normalise_functions(functions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(functions, (str, bytes)) or not isinstance(functions, Sequence):
        raise PlannerValidationError("functions must be an ordered sequence of strings")
    ordered = tuple(functions)
    for index, function in enumerate(ordered):
        if not isinstance(function, str) or not function:
            raise PlannerValidationError(
                f"functions[{index}] must be a non-empty string"
            )
    return ordered


def _normalise_catalog(catalog: Iterable[Any]) -> tuple[_Candidate, ...]:
    if isinstance(catalog, (str, bytes, Mapping)):
        raise PlannerValidationError("catalog must be an iterable of Atom entries")
    try:
        entries = tuple(catalog)
    except TypeError as exc:
        raise PlannerValidationError("catalog must be iterable") from exc

    normalised = tuple(_normalise_entry(entry, index) for index, entry in enumerate(entries))
    seen: set[ExactAtomRef] = set()
    for candidate in normalised:
        if candidate.exact_ref in seen:
            raise PlannerValidationError(
                f"duplicate exact Atom tuple: {candidate.exact_ref!r}"
            )
        seen.add(candidate.exact_ref)
    return tuple(sorted(normalised, key=lambda candidate: candidate.exact_ref))


def _normalise_entry(entry: Any, index: int) -> _Candidate:
    prefix = f"catalog[{index}]"
    removed = _field(entry, "removed", default=False)
    if removed is not False:
        if removed is True:
            raise PlannerValidationError(f"{prefix} is removed")
        raise PlannerValidationError(f"{prefix}.removed must be boolean")
    approved = _field(entry, "approved", default=_MISSING)
    if approved is _MISSING:
        status = _field(entry, "status", default=_MISSING)
        approved = status == "approved" if status is not _MISSING else _MISSING
    if approved is not True:
        raise PlannerValidationError(f"{prefix} is not explicitly approved")

    atom_id = _required_nonempty_string(entry, "atom_id", prefix)
    version = _required_nonempty_string(entry, "version", prefix)
    revision = _field(entry, "revision", default=_MISSING)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise PlannerValidationError(f"{prefix}.revision must be a non-negative integer")
    content_sha256 = _required_nonempty_string(entry, "content_sha256", prefix)
    if _SHA256_RE.fullmatch(content_sha256) is None:
        raise PlannerValidationError(
            f"{prefix}.content_sha256 must be 64 lowercase hexadecimal characters"
        )
    function = _required_nonempty_string(entry, "function", prefix)
    duration = _field(entry, "duration_us", default=_MISSING)
    if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        raise PlannerValidationError(f"{prefix}.duration_us must be a positive integer")
    side = _field(entry, "execution_side", default=_MISSING)
    if side is _MISSING:
        side = _field(entry, "handedness", default=_MISSING)
    if side not in _VALID_SIDES:
        raise PlannerValidationError(
            f"{prefix}.execution_side must be one of {sorted(_VALID_SIDES)!r}"
        )
    source_id = _field(entry, "source_id", default=None)
    if source_id is not None and (not isinstance(source_id, str) or not source_id):
        raise PlannerValidationError(f"{prefix}.source_id must be a non-empty string or None")

    return _Candidate(
        original=entry,
        exact_ref=ExactAtomRef(atom_id, version, revision, content_sha256),
        function=function,
        duration_us=duration,
        side=side,
        source_id=source_id,
    )


def _append_history(
    history: PlannerHistory, candidates: Sequence[_Candidate]
) -> PlannerHistory:
    side = history.last_unilateral_side
    run = history.unilateral_run_length
    balance = history.unilateral_balance
    for candidate in candidates:
        if candidate.side in _UNILATERAL_SIDES:
            if candidate.side == side:
                run += 1
            else:
                side = candidate.side
                run = 1
            balance += 1 if candidate.side == "left" else -1
        else:
            side = None
            run = 0
    return PlannerHistory(
        recent_atom_refs=history.recent_atom_refs
        + tuple(candidate.exact_ref for candidate in candidates),
        recent_source_ids=history.recent_source_ids
        + tuple(
            candidate.source_id
            for candidate in candidates
            if candidate.source_id is not None
        ),
        last_unilateral_side=side,
        unilateral_run_length=run,
        unilateral_balance=balance,
    )


def _advance_side(
    state: _SearchState, side: str
) -> tuple[str | None, int, int, int, int]:
    if side not in _UNILATERAL_SIDES:
        return (
            None,
            0,
            state.max_unilateral_run,
            state.same_unilateral_edges,
            state.unilateral_balance,
        )
    same = side == state.last_unilateral_side
    run = state.unilateral_run_length + 1 if same else 1
    return (
        side,
        run,
        max(state.max_unilateral_run, run),
        state.same_unilateral_edges + (1 if same else 0),
        state.unilateral_balance + (1 if side == "left" else -1),
    )


def _call_nonnegative_cost(
    callback: Callable[..., Any],
    args: tuple[Any, ...],
    name: str,
    exact_ref: ExactAtomRef,
) -> int:
    value = callback(*args)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlannerValidationError(
            f"{name} returned an invalid cost for {exact_ref!r}; "
            "expected a non-negative integer microsecond value"
        )
    return value


def _field(entry: Any, name: str, *, default: Any) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _required_nonempty_string(entry: Any, name: str, prefix: str) -> str:
    value = _field(entry, name, default=_MISSING)
    if not isinstance(value, str) or not value:
        raise PlannerValidationError(f"{prefix}.{name} must be a non-empty string")
    return value


def _counts(values: Iterable[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _require_nonnegative_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PlannerValidationError(f"{name} must be a non-negative integer")


__all__ = [
    "ExactAtomRef",
    "FIT_BUCKET_US",
    "PlanResult",
    "PlannerHistory",
    "PlannerValidationError",
    "SECONDARY_BEAM_WIDTH",
    "plan_prefix",
    "replan_suffix",
    "update_history",
]
