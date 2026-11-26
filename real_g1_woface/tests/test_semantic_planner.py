from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import unittest

from web_overlay.semantic_planner import (
    ExactAtomRef,
    PlannerHistory,
    PlannerValidationError,
    plan_prefix,
    replan_suffix,
)


def atom(
    atom_id: str,
    function: str,
    duration_us: int,
    *,
    side: str = "bilateral",
    source_id: str | None = None,
    removed: bool = False,
) -> dict[str, object]:
    return {
        "atom_id": atom_id,
        "version": "candidate",
        "revision": 1,
        "content_sha256": atom_id[0] * 64,
        "function": function,
        "duration_us": duration_us,
        "execution_side": side,
        "source_id": source_id or f"source:{atom_id}",
        "approved": True,
        "removed": removed,
    }


def zero_transition(_previous: object, _candidate: object) -> int:
    return 0


class SemanticPlannerTests(unittest.TestCase):
    def test_third_atom_one_microsecond_over_budget_is_truncated(self) -> None:
        catalog = (
            atom("a", "f1", 1_000_000),
            atom("b", "f2", 1_000_000),
            atom("c", "f3", 1_000_000),
        )

        plan = plan_prefix(
            ("f1", "f2", "f3"),
            catalog,
            transition_cost_us=zero_transition,
            budget_us=2_999_999,
        )

        self.assertEqual(plan.covered_functions, ("f1", "f2"))
        self.assertEqual([entry["atom_id"] for entry in plan.atoms], ["a", "b"])
        self.assertEqual(plan.used_budget_us, 2_000_000)
        self.assertEqual(plan.remaining_functions, ("f3",))
        self.assertEqual(plan.history_after.recent_atom_refs, plan.exact_refs)

    def test_same_function_short_variant_automatically_packs_longer_prefix(self) -> None:
        catalog = (
            atom("a", "emphasize", 1_500_000),
            atom("b", "emphasize", 1_000_000),
        )

        plan = plan_prefix(
            ("emphasize", "emphasize"),
            catalog,
            transition_cost_us=zero_transition,
            budget_us=2_000_000,
        )

        self.assertEqual(plan.prefix_length, 2)
        self.assertEqual([entry["atom_id"] for entry in plan.atoms], ["b", "b"])
        self.assertEqual(plan.unused_budget_us, 0)

    def test_prefix_length_is_monotonic_as_budget_increases(self) -> None:
        catalog = tuple(
            atom(name, function, 1_000_000)
            for name, function in (("a", "f1"), ("b", "f2"), ("c", "f3"))
        )
        budgets = (0, 999_999, 1_000_000, 1_999_999, 2_000_000, 3_000_000)
        lengths = [
            plan_prefix(
                ("f1", "f2", "f3"),
                catalog,
                transition_cost_us=zero_transition,
                budget_us=budget,
            ).prefix_length
            for budget in budgets
        ]

        self.assertEqual(lengths, [0, 0, 1, 1, 2, 3])
        self.assertEqual(lengths, sorted(lengths))

    def test_side_objective_avoids_unilateral_run_and_bilateral_resets_it(self) -> None:
        left = atom("a", "f", 1_000_000, side="left")
        right = atom("b", "f", 1_000_000, side="right")
        bilateral = atom("c", "f", 1_000_000, side="bilateral")
        history = PlannerHistory(
            last_unilateral_side="left",
            unilateral_run_length=1,
        )

        two_sided = plan_prefix(
            ("f", "f"),
            (left, right),
            transition_cost_us=zero_transition,
            budget_us=2_000_000,
            history=history,
        )
        with_bilateral = plan_prefix(
            ("f", "f"),
            (left, bilateral),
            transition_cost_us=zero_transition,
            budget_us=2_000_000,
            history=history,
        )

        self.assertEqual(
            [entry["execution_side"] for entry in two_sided.atoms],
            ["right", "left"],
        )
        self.assertEqual(
            [entry["execution_side"] for entry in with_bilateral.atoms],
            ["bilateral", "left"],
        )

    def test_500ms_fit_bucket_lets_repeat_win_before_exact_slack(self) -> None:
        long = atom("a", "f", 1_000_000)
        short = atom("b", "f", 900_000)

        plan = plan_prefix(
            ("f", "f"),
            (long, short),
            transition_cost_us=zero_transition,
            budget_us=2_000_000,
        )

        # Both 0 us and 100 ms slack occupy fit bucket zero.  The distinct pair
        # therefore beats the exact-fit repeated Atom; exact slack is considered
        # only after repeat and side objectives.
        self.assertEqual([entry["atom_id"] for entry in plan.atoms], ["a", "b"])
        self.assertEqual(plan.fit_bucket, 0)
        self.assertEqual(plan.unused_budget_us, 100_000)

    def test_removed_catalog_entry_and_unknown_function_are_rejected(self) -> None:
        with self.assertRaisesRegex(PlannerValidationError, "removed"):
            plan_prefix(
                ("f",),
                (atom("a", "f", 1_000_000, removed=True),),
                transition_cost_us=zero_transition,
                budget_us=1_000_000,
            )

        with self.assertRaisesRegex(PlannerValidationError, "unknown function"):
            plan_prefix(
                ("unknown",),
                (atom("a", "f", 1_000_000),),
                transition_cost_us=zero_transition,
                budget_us=1_000_000,
            )

    def test_catalog_order_and_python_hash_seed_do_not_change_plan(self) -> None:
        catalog = (
            atom("a", "f", 1_000_000, source_id="s1"),
            atom("b", "f", 1_000_000, source_id="s2"),
            atom("c", "f", 1_000_000, source_id="s3"),
        )
        expected = ["a", "b"]
        for permutation in itertools.permutations(catalog):
            plan = plan_prefix(
                ("f", "f"),
                permutation,
                transition_cost_us=zero_transition,
                budget_us=2_000_000,
            )
            self.assertEqual([entry["atom_id"] for entry in plan.atoms], expected)

        script = """
import json
from web_overlay.semantic_planner import plan_prefix
catalog = json.loads({catalog!r})
plan = plan_prefix(("f", "f"), catalog, transition_cost_us=lambda p, c: 0, budget_us=2000000)
print(json.dumps([entry["atom_id"] for entry in plan.atoms]))
""".format(catalog=json.dumps(catalog))
        outputs = []
        for seed in ("1", "7", "123"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    cwd=os.getcwd(),
                    env=environment,
                    text=True,
                ).strip()
            )
        self.assertEqual(outputs, ['["a", "b"]'] * 3)

    def test_history_repeat_source_novelty_and_suffix_replan_are_pure(self) -> None:
        a = atom("a", "f", 1_000_000, source_id="old")
        b = atom("b", "f", 1_000_000, source_id="new")
        history = PlannerHistory(
            recent_atom_refs=(ExactAtomRef("a", "candidate", 1, "a" * 64),),
            recent_source_ids=("old",),
        )

        first = plan_prefix(
            ("f",),
            (a, b),
            transition_cost_us=zero_transition,
            budget_us=1_000_000,
            history=history,
        )
        suffix = replan_suffix(
            ("f", "f"),
            (a, b),
            transition_cost_us=zero_transition,
            remaining_budget_us=1_999_999,
            history=history,
            previous_atom=first.atoms[0],
        )

        self.assertEqual(first.atoms[0]["atom_id"], "b")
        self.assertEqual(suffix.prefix_length, 1)
        self.assertEqual(suffix.atoms[0]["atom_id"], "b")
        self.assertEqual(a["duration_us"], 1_000_000)
        self.assertEqual(b["duration_us"], 1_000_000)

    def test_final_recovery_and_tail_guard_are_part_of_strict_fit(self) -> None:
        catalog = (
            atom("a", "f1", 1_000_000),
            atom("b", "f2", 1_000_000),
        )

        plan = plan_prefix(
            ("f1", "f2"),
            catalog,
            transition_cost_us=zero_transition,
            final_recovery_cost_us=lambda last: 300_000,
            tail_guard_us=200_000,
            budget_us=2_499_999,
        )

        self.assertEqual(plan.prefix_length, 1)
        self.assertEqual(plan.motion_and_transition_us, 1_000_000)
        self.assertEqual(plan.final_recovery_us, 300_000)
        self.assertEqual(plan.tail_guard_us, 200_000)
        self.assertEqual(plan.used_budget_us, 1_500_000)

    def test_initial_inter_atom_terminal_costs_and_empty_plan_accounting(self) -> None:
        initial_state = object()
        catalog = (
            atom("a", "f1", 1_000_000),
            atom("b", "f2", 1_000_000),
        )
        calls = []

        def transition(previous, candidate):
            calls.append((previous, candidate))
            return 100_000 if previous is initial_state else 200_000

        plan = plan_prefix(
            ("f1", "f2"),
            catalog,
            transition_cost_us=transition,
            previous_atom=initial_state,
            final_recovery_cost_us=lambda last: 300_000,
            tail_guard_us=50_000,
            budget_us=2_650_000,
        )
        empty = plan_prefix(
            (),
            catalog,
            transition_cost_us=transition,
            previous_atom=initial_state,
            final_recovery_cost_us=lambda last: 300_000,
            tail_guard_us=50_000,
            budget_us=0,
        )

        self.assertEqual(plan.prefix_length, 2)
        self.assertEqual(plan.motion_and_transition_us, 2_300_000)
        self.assertEqual(plan.final_recovery_us, 300_000)
        self.assertEqual(plan.used_budget_us, 2_650_000)
        self.assertTrue(any(previous is initial_state for previous, _ in calls))
        self.assertEqual(empty.prefix_length, 0)
        self.assertEqual(empty.used_budget_us, 0)
        self.assertEqual(empty.tail_guard_us, 0)

    def test_fixed_beam_bounds_worst_combinatorics_without_losing_prefix(self) -> None:
        catalog = tuple(
            atom(f"a{index:02d}", "f", 100_000 + index * 1_000)
            for index in range(17)
        )
        transition_calls = 0

        def transition(_previous, _candidate):
            nonlocal transition_calls
            transition_calls += 1
            return 0

        plan = plan_prefix(
            ("f",) * 12,
            catalog,
            transition_cost_us=transition,
            budget_us=1_392_000,
        )

        self.assertEqual(plan.prefix_length, 12)
        self.assertFalse(plan.secondary_search_complete)
        # Pure costs are memoized by initial/exact edge, rather than called once
        # for each of the theoretical 17**12 paths.
        self.assertLessEqual(transition_calls, 17 + 17 * 17)

    def test_no_single_atom_duration_cap_beyond_total_reply_budget(self) -> None:
        long_atom = atom("a", "f", 10_000_000)
        plan = plan_prefix(
            ("f",),
            (long_atom,),
            transition_cost_us=zero_transition,
            budget_us=10_000_000,
        )
        self.assertEqual(plan.prefix_length, 1)
        self.assertIs(plan.atoms[0], long_atom)


if __name__ == "__main__":
    unittest.main()
