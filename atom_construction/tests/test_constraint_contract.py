from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from atom_pipeline.constraint_contract import (
    ConstraintContractError,
    build_constraint_profile,
    canonical_json_sha256,
    frozen_context_from_report,
    motion_limits_from_profile,
)
from atom_pipeline.safety import MotionLimits, build_motion_constraint_report


JOINTS = ("left_shoulder_pitch", "right_shoulder_pitch")


class FakeMujocoModule:
    __version__ = "3.test"


class FakeValidator:
    sample_stride = 2
    contact_distance_threshold_m = -0.001
    baseline_tolerance_m = 0.003
    backend_id = "fake-g1"
    allowed_environment_robot_tokens = ()
    scene_path: Path

    def __init__(self, scene: Path) -> None:
        self.scene_path = scene

    def _load(self):
        return FakeMujocoModule(), object()

    def joint_ranges(self, joints):
        return {joint: (-1.0, 1.0) for joint in joints}


def profile_and_report(root: Path):
    scene = root / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    limits = MotionLimits(
        velocity_rad_s=0.8,
        acceleration_rad_s2=4.0,
        jerk_rad_s3=30.0,
        velocity_by_joint={joint: 0.8 for joint in JOINTS},
        acceleration_by_joint={joint: 4.0 for joint in JOINTS},
        jerk_by_joint={joint: 30.0 for joint in JOINTS},
        joint_ranges={joint: (-1.0, 1.0) for joint in JOINTS},
        position_margin_rad=0.05,
        warning_fraction=0.85,
        time_scale_safety_factor=1.02,
        max_time_scale=3.0,
    )
    validator = FakeValidator(scene)
    profile = build_constraint_profile(
        limits,
        JOINTS,
        validator,
        neutral_id="safe_neutral.v1",
        neutral_content_hash="a" * 64,
    )
    report = build_motion_constraint_report(
        [0.0, 0.1, 0.2],
        [[0.0, 0.0], [0.01, -0.01], [0.0, 0.0]],
        JOINTS,
        limits=limits,
        joint_ranges=limits.joint_ranges,
    )
    report["collision"] = {
        "validator": profile["collision_policy"]["validator_version"],
        "backend": "fake-g1",
        "result": "pass",
        "model": {
            "path": "scene.xml",
            "root_xml_sha256": profile["collision_policy"]["model_fingerprint"]["root_xml_sha256"],
            "mujoco_version": "3.test",
        },
        "root_policy": profile["collision_policy"]["root_policy"],
        "sample_stride": 2,
        "contact_distance_threshold_m": -0.001,
        "baseline_tolerance_m": 0.003,
        "allowed_environment_robot_tokens": [],
    }
    report["constraint_profile"] = profile
    report["constraint_profile_sha256"] = canonical_json_sha256(profile)
    return scene, limits, validator, profile, report


class ConstraintContractTests(unittest.TestCase):
    def test_complete_profile_round_trip_includes_time_collision_and_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            _, limits, _, profile, _ = profile_and_report(Path(directory))
            rebuilt = motion_limits_from_profile(profile, JOINTS)
        self.assertEqual(dict(rebuilt.joint_ranges), dict(limits.joint_ranges))
        self.assertEqual(rebuilt.time_scale_safety_factor, 1.02)
        self.assertEqual(rebuilt.max_time_scale, 3.0)
        self.assertEqual(profile["collision_policy"]["allowed_environment_robot_tokens"], [])
        self.assertEqual(
            profile["neutral"],
            {"id": "safe_neutral.v1", "content_hash": "a" * 64},
        )

    def test_valid_frozen_context_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _, profile, report = profile_and_report(Path(directory))
            with patch(
                "atom_pipeline.constraint_contract.MujocoCollisionValidator",
                side_effect=lambda path, **kwargs: FakeValidator(Path(path)),
            ):
                _, _, frozen, profile_hash = frozen_context_from_report(
                    report, JOINTS, scene
                )
        self.assertEqual(frozen, profile)
        self.assertEqual(profile_hash, canonical_json_sha256(profile))

    def test_legacy_and_tampered_reports_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, validator, profile, report = profile_and_report(Path(directory))
            with self.assertRaisesRegex(ConstraintContractError, "v2"):
                frozen_context_from_report({"validator": "atom-motion-constraints-v1"}, JOINTS, scene)
            tampered = deepcopy(report)
            tampered["constraint_profile"]["motion_limits"]["jerk_by_joint"][JOINTS[0]] = 50.0
            with self.assertRaisesRegex(ConstraintContractError, "SHA-256"):
                frozen_context_from_report(tampered, JOINTS, scene)

    def test_report_thresholds_must_match_advertised_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, validator, profile, report = profile_and_report(Path(directory))
            tampered = deepcopy(report)
            tampered["kinematics"]["thresholds"]["per_joint_effective"][JOINTS[0]]["jerk_rad_s3"] = 50.0
            with self.assertRaisesRegex(ConstraintContractError, "per-joint"):
                frozen_context_from_report(tampered, JOINTS, scene)

    def test_time_safety_factor_tamper_fails_when_required_factor_is_one(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _, _, report = profile_and_report(Path(directory))
            self.assertEqual(
                report["kinematics"]["time_scaling"]["required_factor"], 1.0
            )
            tampered = deepcopy(report)
            tampered["kinematics"]["time_scaling"]["safety_factor"] = 1.10
            with self.assertRaisesRegex(ConstraintContractError, "time-scaling"):
                frozen_context_from_report(tampered, JOINTS, scene)

    def test_current_scene_hash_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _, _, report = profile_and_report(Path(directory))
            scene.write_text("<mujoco model='changed'/>", encoding="utf-8")
            with self.assertRaisesRegex(ConstraintContractError, "differ"):
                frozen_context_from_report(report, JOINTS, scene)


if __name__ == "__main__":
    unittest.main()

