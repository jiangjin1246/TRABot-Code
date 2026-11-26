from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from atom_pipeline.joints import (
    G1_29_JOINTS,
    G1_ARM_JOINTS,
    canonical_joint_name,
    normalize_joint_mapping,
    normalize_joint_names,
)
from atom_pipeline.safety import (
    MotionLimits,
    MujocoCollisionValidator,
    analyze_kinematics,
    build_atom_safety,
    build_motion_constraint_report,
    time_scale_trajectory,
)
from atom_pipeline.schema import validate_atom
from tests.test_schema import sample_atom


def permissive_limits(**overrides: object) -> MotionLimits:
    values: dict[str, object] = {
        "velocity_rad_s": 100.0,
        "acceleration_rad_s2": 100.0,
        "jerk_rad_s3": 100.0,
        "velocity_by_joint": {},
        "acceleration_by_joint": {},
        "jerk_by_joint": {},
        "time_scale_safety_factor": 1.0,
    }
    values.update(overrides)
    return MotionLimits(**values)


class G1JointTests(unittest.TestCase):
    def test_canonical_order_and_aliases(self) -> None:
        self.assertEqual(len(G1_29_JOINTS), 29)
        self.assertEqual(len(G1_ARM_JOINTS), 14)
        self.assertEqual(canonical_joint_name("left_elbow_joint"), "left_elbow")
        self.assertEqual(
            normalize_joint_names(["left_elbow_joint", "right_elbow"]),
            ("left_elbow", "right_elbow"),
        )

    def test_duplicate_aliases_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_joint_mapping({"left_elbow": 0.0, "left_elbow_joint": 0.0})

    def test_unknown_joint_is_not_fuzzily_matched(self) -> None:
        with self.assertRaises(KeyError):
            canonical_joint_name("left_elbo")


class KinematicSafetyTests(unittest.TestCase):
    def test_constant_motion_passes_and_is_json_serializable(self) -> None:
        times = np.linspace(0.0, 1.0, 11)
        q = np.zeros((len(times), 2))
        report = build_motion_constraint_report(
            times,
            q,
            ["left_elbow", "right_elbow_joint"],
            limits=permissive_limits(),
        )
        self.assertEqual(report["result"], "pass")
        self.assertFalse(report["eligible_for_backend_certification"])
        self.assertEqual(report["certified_backends"], [])
        self.assertEqual(report["kinematics"]["violation_count"], 0)
        json.dumps(report)

    def test_velocity_violation_produces_exact_time_scale(self) -> None:
        times = np.asarray([0.0, 0.5, 1.0])
        q = np.asarray([[0.0], [1.0], [2.0]])
        limits = permissive_limits(velocity_rad_s=1.0)
        analysis = analyze_kinematics(times, q, ["left_elbow"], limits=limits)
        self.assertFalse(analysis.passed)
        self.assertAlmostEqual(analysis.required_time_scale, 2.0, places=8)
        self.assertAlmostEqual(analysis.recommended_time_scale, 2.0, places=8)
        self.assertIn(
            "JOINT_VELOCITY_LIMIT",
            {item["code"] for item in analysis.report["violations"]},
        )

        scaled_times = time_scale_trajectory(times, analysis.recommended_time_scale)
        scaled = analyze_kinematics(scaled_times, q, ["left_elbow"], limits=limits)
        self.assertTrue(scaled.passed)
        self.assertAlmostEqual(scaled.report["duration_s"], 2.0, places=8)

    def test_position_margin_is_a_hard_non_temporal_failure(self) -> None:
        times = np.asarray([0.0, 0.5, 1.0])
        q = np.asarray([[0.0], [0.96], [0.0]])
        limits = permissive_limits(position_margin_rad=0.1)
        analysis = analyze_kinematics(
            times,
            q,
            ["left_elbow"],
            limits=limits,
            joint_ranges={"left_elbow_joint": (-1.0, 1.0)},
        )
        position_errors = [
            item
            for item in analysis.report["violations"]
            if item["code"] == "JOINT_POSITION_LIMIT"
        ]
        self.assertEqual(len(position_errors), 1)
        self.assertEqual(position_errors[0]["side"], "high")
        self.assertFalse(
            analysis.report["time_scaling"]["position_failures_are_fixable"]
        )

    def test_dynamic_scale_uses_inverse_power_laws(self) -> None:
        times = np.linspace(0.0, 1.0, 9)
        q = (times**2)[:, None]
        limits = permissive_limits(acceleration_rad_s2=0.5)
        analysis = analyze_kinematics(times, q, ["left_elbow"], limits=limits)
        self.assertAlmostEqual(analysis.required_time_scale, 2.0, places=6)

    def test_atom_safety_projection_matches_strict_field_names(self) -> None:
        times = np.linspace(0.0, 1.0, 11)
        q = np.zeros((len(times), 1))
        safety = build_atom_safety(
            times,
            q,
            ["left_elbow"],
            limits=permissive_limits(),
            joint_ranges={"left_elbow": (-2.0, 2.0)},
        )
        self.assertEqual(
            set(safety),
            {
                "thresholds",
                "metrics",
                "per_joint",
                "violations",
                "warnings",
                "collision",
                "validator_version",
                "certified_backends",
            },
        )
        self.assertEqual(
            set(safety["per_joint"]["left_elbow"]),
            {"q_min", "q_max", "max_abs_dq", "max_abs_ddq", "max_abs_jerk"},
        )
        self.assertTrue(safety["metrics"]["passed"])
        self.assertEqual(
            safety["metrics"]["threshold_profile"], "g1-fixed-base-upper-body-v2"
        )
        self.assertFalse(safety["collision"]["checked"])
        self.assertFalse(safety["collision"]["passed"])
        self.assertEqual(safety["certified_backends"], [])
        json.dumps(safety)

    def test_atom_safety_projection_validates_inside_schema_v1(self) -> None:
        atom = sample_atom()
        trajectory = atom["control"]["trajectory"]
        joints = atom["control"]["joints"]
        times = np.asarray([point["t"] for point in trajectory])
        q = np.asarray([[point["q"][joint] for joint in joints] for point in trajectory])
        atom["safety"] = build_atom_safety(
            times,
            q,
            joints,
            limits=permissive_limits(
                velocity_rad_s=1_000.0,
                acceleration_rad_s2=1_000.0,
                jerk_rad_s3=1_000.0,
            ),
            joint_ranges={joint: (-2.0, 2.0) for joint in joints},
        )
        validate_atom(atom)

    def test_invalid_trajectory_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            analyze_kinematics(
                [0.0, 1.0, 0.5],
                [[0.0], [0.0], [0.0]],
                ["left_elbow"],
            )
        with self.assertRaises(ValueError):
            analyze_kinematics(
                [0.0, 1.0],
                [[0.0], [float("nan")]],
                ["left_elbow"],
            )


class OptionalMujocoTests(unittest.TestCase):
    def test_constructor_does_not_load_mujoco(self) -> None:
        calls: list[bool] = []

        def loader() -> object:
            calls.append(True)
            raise AssertionError("lazy loader was invoked by construction")

        validator = MujocoCollisionValidator(
            Path("not_loaded.xml"),
            module_loader=loader,
        )
        self.assertEqual(calls, [])
        self.assertEqual(validator.sample_stride, 1)


if __name__ == "__main__":
    unittest.main()
