from __future__ import annotations

import unittest

from web_overlay.semantic_motion import (
    minimum_jerk_bridge_us,
    recovery_cost_callback,
    transition_cost_callback,
)


def row(name: str, first: float, last: float, duration_us: int = 1_000_000):
    return {
        "atom_id": name,
        "duration_us": duration_us,
        "motion": {
            "joints": ["left_elbow"],
            "frames": [
                {"t": 0.0, "q": {"left_elbow": first}},
                {"t": duration_us / 1_000_000, "q": {"left_elbow": last}},
            ],
        },
    }


class SemanticMotionTests(unittest.TestCase):
    def test_minimum_jerk_cost_matches_service_tick_contract_and_joint_alias(self):
        self.assertEqual(
            minimum_jerk_bridge_us(
                {"left_elbow_joint": 0.0}, {"left_elbow": 0.0}
            ),
            80_000,
        )
        self.assertEqual(
            minimum_jerk_bridge_us(
                {"left_elbow_joint": 0.0}, {"left_elbow": 0.1}
            ),
            200_000,
        )
        self.assertEqual(
            minimum_jerk_bridge_us(
                {"left_elbow_joint": 0.0}, {"left_elbow": 0.64}
            ),
            1_200_000,
        )

    def test_transition_uses_measured_pose_then_previous_exact_exit(self):
        first = row("a", 0.1, 0.3)
        second = row("b", 0.5, 0.6)
        cost = transition_cost_callback({"left_elbow_joint": 0.0})

        self.assertEqual(cost(None, first), 200_000)
        self.assertEqual(cost(first, second), 400_000)

    def test_recovery_includes_bridge_and_complete_neutral_motion(self):
        previous = row("a", 0.0, 0.3)
        neutral = {
            "duration_us": 800_000,
            "motion": {
                "joints": ["left_elbow"],
                "frames": [
                    {"t": 0.0, "q": {"left_elbow": 0.0}},
                    {"t": 0.8, "q": {"left_elbow": 0.0}},
                ],
            },
        }

        self.assertEqual(recovery_cost_callback(neutral)(previous), 1_400_000)


if __name__ == "__main__":
    unittest.main()
