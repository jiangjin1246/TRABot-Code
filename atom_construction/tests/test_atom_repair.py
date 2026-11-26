from __future__ import annotations

from copy import deepcopy
import json
import unittest

import numpy as np

from atom_pipeline.repair import (
    AtomRepairError,
    RepairRecipe,
    endpoint_preserving_zero_phase_smooth,
    repair_candidate,
)
from atom_pipeline.safety import MotionLimits
from atom_pipeline.schema import atom_hash, validate_atom
from tests.test_schema import sample_atom


def limits(
    *,
    velocity: float = 100.0,
    acceleration: float = 100.0,
    jerk: float = 100.0,
    ranges: dict[str, tuple[float, float]] | None = None,
    safety_factor: float = 1.0,
) -> MotionLimits:
    return MotionLimits(
        velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration,
        jerk_rad_s3=jerk,
        velocity_by_joint={},
        acceleration_by_joint={},
        jerk_by_joint={},
        joint_ranges=ranges or {},
        time_scale_safety_factor=safety_factor,
        max_time_scale=20.0,
    )


def dense_candidate(*, duration: float = 1.0) -> dict:
    atom = sample_atom()
    joints = atom["control"]["joints"]
    times = np.linspace(0.0, duration, 21)
    left = np.sin(np.pi * times / duration)
    right = 0.8 * np.sin(np.pi * times / duration)
    q = np.column_stack((0.2 + left, 0.2 + right))
    dq = np.gradient(q, times, axis=0, edge_order=2)
    ddq = np.gradient(dq, times, axis=0, edge_order=2)
    atom["control"]["sample_rate_hz"] = 20.0 / duration
    atom["control"]["trajectory"] = [
        {
            "t": float(times[row]),
            "q": {joint: float(q[row, col]) for col, joint in enumerate(joints)},
            "dq": {joint: float(dq[row, col]) for col, joint in enumerate(joints)},
            "ddq": {joint: float(ddq[row, col]) for col, joint in enumerate(joints)},
        }
        for row in range(len(times))
    ]
    atom["timing"]["nominal_duration_s"] = duration
    atom["timing"]["min_duration_s"] = 0.75 * duration
    atom["timing"]["max_duration_s"] = 1.3 * duration
    atom["timing"]["phases"] = {
        "preparation": {"start_s": 0.0, "end_s": 0.2 * duration},
        "stroke": {"start_s": 0.2 * duration, "end_s": 0.6 * duration},
        "hold": {"start_s": 0.6 * duration, "end_s": 0.7 * duration},
        "retraction": {"start_s": 0.7 * duration, "end_s": duration},
    }
    atom["timing"]["anchors"] = {
        "apex": 0.5 * duration,
        "semantic_onset": 0.2 * duration,
        "semantic_offset": 0.7 * duration,
    }
    atom["timing"]["safe_cutpoints"] = [0.0, 0.2 * duration, 0.7 * duration, duration]
    atom["timing"]["min_lead_ms"] = 200.0 * duration
    for interface, point in (
        (atom["entry"], atom["control"]["trajectory"][0]),
        (atom["exit"], atom["control"]["trajectory"][-1]),
    ):
        for vector in ("q", "dq", "ddq"):
            interface[vector] = deepcopy(point[vector])
    atom["safety"]["metrics"]["passed"] = True
    atom["safety"]["violations"] = []
    atom["identity"].pop("content_hash", None)
    validate_atom(atom)
    return atom


class SmoothingTests(unittest.TestCase):
    def test_smoothing_is_symmetric_and_endpoint_locked(self) -> None:
        q = np.zeros(( nine := 9, 1))
        q[nine // 2, 0] = 1.0
        smoothed = endpoint_preserving_zero_phase_smooth(q, passes=1)
        np.testing.assert_allclose(smoothed[:, 0], smoothed[::-1, 0])
        np.testing.assert_array_equal(smoothed[0], q[0])
        np.testing.assert_array_equal(smoothed[-1], q[-1])
        self.assertEqual(int(np.argmax(smoothed[:, 0])), nine // 2)

    def test_zero_passes_returns_copy(self) -> None:
        q = np.arange(12, dtype=np.float64).reshape(6, 2)
        result = endpoint_preserving_zero_phase_smooth(q, passes=0)
        np.testing.assert_array_equal(result, q)
        self.assertIsNot(result, q)


class RepairCandidateTests(unittest.TestCase):
    def test_repair_is_pure_revisions_and_hashes_with_provenance(self) -> None:
        parent = dense_candidate()
        before = deepcopy(parent)
        parent_hash = atom_hash(parent)
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(amplitude_scale=0.5, smoothing_passes=1),
            limits=limits(),
        )

        self.assertEqual(parent, before)
        self.assertEqual(result.atom["identity"]["id"], parent["identity"]["id"])
        self.assertEqual(
            result.atom["identity"]["revision"], parent["identity"]["revision"] + 1
        )
        self.assertEqual(result.atom["status"], "candidate")
        self.assertEqual(result.parent_hash, parent_hash)
        self.assertEqual(result.atom["identity"]["content_hash"], atom_hash(result.atom))
        provenance = result.atom["extensions"]["repair_provenance"]
        self.assertEqual(provenance["parent_content_hash"], parent_hash)
        self.assertEqual(provenance["recipe"]["amplitude_scale"], 0.5)
        self.assertEqual(len(result.atom["metadata"]["repair_history"]), 1)
        self.assertEqual(
            provenance["recipe"]["timing_mapping"]["method"],
            "uniform_parent_time_mapping",
        )
        self.assertFalse(result.atom["safety"]["collision"]["checked"])
        self.assertEqual(result.atom["safety"]["certified_backends"], [])
        validate_atom(result.atom)
        json.dumps(result.safety_report, allow_nan=False)

    def test_amplitude_is_scaled_about_entry_pose(self) -> None:
        parent = dense_candidate()
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(amplitude_scale=0.5, smoothing_passes=0),
            limits=limits(),
        )
        parent_q = parent["control"]["trajectory"][10]["q"]["left_elbow"]
        child_q = result.atom["control"]["trajectory"][10]["q"]["left_elbow"]
        entry = parent["entry"]["q"]["left_elbow"]
        self.assertAlmostEqual(child_q - entry, 0.5 * (parent_q - entry), places=10)

    def test_per_limb_scale_changes_only_target_arm_and_records_provenance(self) -> None:
        parent = dense_candidate()
        before = deepcopy(parent)
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(
                amplitude_scale=1.0,
                limb_scales={"left_arm": 0.5},
                smoothing_passes=0,
            ),
            limits=limits(),
        )
        middle = 10
        entry = parent["control"]["trajectory"][0]["q"]
        parent_q = parent["control"]["trajectory"][middle]["q"]
        child_q = result.atom["control"]["trajectory"][middle]["q"]
        self.assertAlmostEqual(
            child_q["left_elbow"] - entry["left_elbow"],
            0.5 * (parent_q["left_elbow"] - entry["left_elbow"]),
        )
        self.assertEqual(child_q["right_elbow"], parent_q["right_elbow"])
        self.assertEqual(parent, before)
        self.assertEqual(result.applied_recipe["scaling_mode"], "per_limb")
        self.assertEqual(
            result.applied_recipe["effective_limb_scales"],
            {"left_arm": 0.5, "right_arm": 1.0, "waist": 1.0},
        )
        self.assertEqual(
            result.atom["extensions"]["repair_provenance"]["recipe"][
                "effective_limb_scales"
            ]["left_arm"],
            0.5,
        )

    def test_partial_neutral_mapping_falls_back_to_entry_and_records_joints(self) -> None:
        parent = dense_candidate()
        joints = parent["control"]["joints"]
        provided_joint, missing_joint = joints[:2]
        neutral = {provided_joint: 0.05}
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(amplitude_scale=0.5, smoothing_passes=0),
            limits=limits(),
            neutral_pose=neutral,
        )

        entry = parent["control"]["trajectory"][0]["q"][missing_joint]
        middle = parent["control"]["trajectory"][10]["q"][missing_joint]
        repaired = result.atom["control"]["trajectory"][10]["q"][missing_joint]
        self.assertAlmostEqual(repaired, entry + 0.5 * (middle - entry), places=10)
        self.assertEqual(
            result.applied_recipe["reference_pose_source"],
            "explicit_neutral_pose_with_entry_fallback",
        )
        self.assertIn(
            missing_joint,
            result.applied_recipe["reference_pose_fallback_joints"],
        )
        self.assertNotIn(
            provided_joint,
            result.applied_recipe["reference_pose_fallback_joints"],
        )

    def test_repair_rejects_motion_that_only_passes_beyond_realtime_duration_cap(self) -> None:
        parent = dense_candidate()
        with self.assertRaisesRegex(AtomRepairError, "realtime atom cap"):
            repair_candidate(
                parent,
                recipe=RepairRecipe(
                    amplitude_scale=1.0,
                    smoothing_passes=0,
                    max_time_scale=10.0,
                    max_output_duration_s=1.1,
                ),
                limits=limits(velocity=2.0, acceleration=1_000.0, jerk=1_000.0),
            )

    def test_retiming_updates_every_time_field_and_derivatives(self) -> None:
        parent = dense_candidate()
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(
                amplitude_scale=1.0,
                smoothing_passes=0,
                max_time_scale=10.0,
            ),
            limits=limits(velocity=2.0, acceleration=1_000.0, jerk=1_000.0),
        )
        factor = result.applied_recipe["applied_time_scale"]
        self.assertGreater(factor, 1.0)
        child = result.atom
        self.assertAlmostEqual(child["timing"]["nominal_duration_s"], factor)
        self.assertAlmostEqual(child["timing"]["anchors"]["apex"], 0.5 * factor)
        self.assertAlmostEqual(
            child["timing"]["phases"]["stroke"]["start_s"], 0.2 * factor
        )
        # This sine path has non-zero endpoint velocity. Retiming enough for the
        # motion limits does not make it satisfy the stricter safe-cutpoint
        # threshold, so the old end marker must be dropped rather than trusted.
        self.assertEqual(child["timing"]["safe_cutpoints"], [])
        self.assertGreater(
            child["metadata"]["processing"]["safe_cutpoints_dropped"], 0
        )
        self.assertAlmostEqual(child["timing"]["min_lead_ms"], 200.0 * factor)
        self.assertEqual(
            child["entry"]["dq"], child["control"]["trajectory"][0]["dq"]
        )
        self.assertEqual(
            child["exit"]["ddq"], child["control"]["trajectory"][-1]["ddq"]
        )
        self.assertTrue(child["safety"]["metrics"]["kinematics_passed"])
        validate_atom(child)

    def test_retimed_motion_with_held_endpoints_keeps_safe_start_and_end(self) -> None:
        parent = dense_candidate()
        trajectory = parent["control"]["trajectory"]
        times = np.asarray([point["t"] for point in trajectory])
        profile = np.zeros(len(times))
        profile[5:16] = np.sin(np.linspace(0.0, np.pi, 11))
        q = np.column_stack((0.2 + profile, 0.2 + 0.8 * profile))
        dq = np.gradient(q, times, axis=0, edge_order=2)
        ddq = np.gradient(dq, times, axis=0, edge_order=2)
        joints = parent["control"]["joints"]
        for row, point in enumerate(trajectory):
            point["q"] = {joint: float(q[row, col]) for col, joint in enumerate(joints)}
            point["dq"] = {joint: float(dq[row, col]) for col, joint in enumerate(joints)}
            point["ddq"] = {joint: float(ddq[row, col]) for col, joint in enumerate(joints)}
        for interface, point in ((parent["entry"], trajectory[0]), (parent["exit"], trajectory[-1])):
            for vector in ("q", "dq", "ddq"):
                interface[vector] = deepcopy(point[vector])
        validate_atom(parent)

        result = repair_candidate(
            parent,
            recipe=RepairRecipe(
                amplitude_scale=1.0,
                smoothing_passes=0,
                max_time_scale=10.0,
            ),
            limits=limits(velocity=2.0, acceleration=1_000.0, jerk=1_000.0),
        )
        factor = result.applied_recipe["applied_time_scale"]
        duration = result.atom["timing"]["nominal_duration_s"]
        self.assertGreater(factor, 1.0)
        self.assertIn(0.0, result.atom["timing"]["safe_cutpoints"])
        self.assertIn(duration, result.atom["timing"]["safe_cutpoints"])

    def test_safe_cutpoints_are_refiltered_and_moving_open_exit_is_dropped(self) -> None:
        parent = dense_candidate()
        parent.setdefault("metadata", {})["kinematics"] = {"exit_kind": "moving_open"}
        parent["metadata"]["processing"] = {
            "safe_cutpoint_velocity_rad_s": 100.0,
            "safe_cutpoint_acceleration_rad_s2": 100.0,
        }
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(amplitude_scale=0.8, smoothing_passes=0),
            limits=limits(),
        )
        duration = result.atom["timing"]["nominal_duration_s"]
        self.assertIn(0.0, result.atom["timing"]["safe_cutpoints"])
        self.assertNotIn(duration, result.atom["timing"]["safe_cutpoints"])
        dropped = result.applied_recipe["timing_mapping"]["safe_cutpoint_filter"]["dropped"]
        self.assertTrue(any("moving_open_exit" in item["reasons"] for item in dropped))
        self.assertFalse(
            result.atom["metadata"]["eligibility"]["eligible_for_motion_graph_curation"]
        )

    def test_repair_replaces_stale_evidence_and_processing(self) -> None:
        parent = dense_candidate()
        parent.setdefault("extensions", {})["validation_evidence"] = {
            "evidence_sha256": "stale-parent-evidence",
            "model_root_xml_sha256": "stale-model",
        }
        parent.setdefault("metadata", {})["processing"] = {
            "applied_time_scale": 99.0,
            "safe_cutpoint_velocity_rad_s": 100.0,
            "safe_cutpoint_acceleration_rad_s2": 100.0,
        }
        result = repair_candidate(
            parent,
            recipe=RepairRecipe(amplitude_scale=0.8, smoothing_passes=0),
            limits=limits(),
        )
        evidence = result.atom["extensions"]["validation_evidence"]
        self.assertNotEqual(evidence["evidence_sha256"], "stale-parent-evidence")
        self.assertIn("repair_recipe_sha256", evidence)
        self.assertIn("constraint_report_sha256", evidence)
        processing = result.atom["metadata"]["processing"]
        self.assertNotIn("applied_time_scale", processing)
        self.assertEqual(processing["stage"], "constraint_repair")
        self.assertEqual(processing["applied_time_scale_from_parent"], 1.0)
        boundary = result.atom["extensions"]["research_claim_boundary"]
        self.assertFalse(boundary["human_reviewed_after_repair"])

    def test_preview_is_direct_bridge_shape(self) -> None:
        result = repair_candidate(
            dense_candidate(),
            recipe=RepairRecipe(amplitude_scale=0.8, smoothing_passes=0),
            limits=limits(),
        )
        preview = result.preview_motion
        self.assertEqual(preview["joints"], result.atom["control"]["joints"])
        self.assertEqual(preview["frames"][0]["t"], 0.0)
        self.assertEqual(preview["frames"][-1]["t"], preview["duration"])
        self.assertEqual(
            preview["source_atom_hash"], result.atom["identity"]["content_hash"]
        )
        self.assertEqual(preview["parent_atom_hash"], result.parent_hash)

    def test_non_candidate_invalid_and_nonconvergent_repairs_fail_closed(self) -> None:
        curated = dense_candidate()
        curated["status"] = "curated"
        with self.assertRaisesRegex(AtomRepairError, "candidate atoms only"):
            repair_candidate(curated, limits=limits())

        invalid = dense_candidate()
        invalid["control"]["trajectory"][2]["q"]["left_elbow"] = float("nan")
        with self.assertRaisesRegex(AtomRepairError, "finite JSON"):
            repair_candidate(invalid, limits=limits())

        parent = dense_candidate()
        with self.assertRaisesRegex(AtomRepairError, "exceeds repair cap"):
            repair_candidate(
                parent,
                recipe=RepairRecipe(
                    amplitude_scale=1.0,
                    smoothing_passes=0,
                    max_time_scale=1.1,
                ),
                limits=limits(velocity=0.1, acceleration=1_000.0, jerk=1_000.0),
            )

    def test_position_failure_cannot_be_hidden_by_retiming(self) -> None:
        parent = dense_candidate()
        with self.assertRaisesRegex(AtomRepairError, "position limits"):
            repair_candidate(
                parent,
                recipe=RepairRecipe(amplitude_scale=1.0, smoothing_passes=0),
                limits=limits(ranges={"left_elbow": (-0.5, 0.5), "right_elbow": (-2.0, 2.0)}),
            )


if __name__ == "__main__":
    unittest.main()
