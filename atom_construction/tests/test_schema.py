from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from atom_pipeline.schema import (
    SCHEMA_VERSION,
    AtomValidationError,
    atom_hash,
    load_atom,
    load_json_schema,
    validate_atom,
    write_atom,
)


ZERO_HASH = "0" * 64


def sample_atom(*, status: str = "candidate") -> dict:
    joints = ["left_elbow", "right_elbow"]
    q0 = {"left_elbow": 0.2, "right_elbow": 0.2}
    q1 = {"left_elbow": 0.7, "right_elbow": 0.55}
    q2 = {"left_elbow": 0.2, "right_elbow": 0.2}
    zero = {joint: 0.0 for joint in joints}
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "id": "beat2.demo.0001",
            "name": "Bilateral explanatory beat",
            "revision": 1,
            "description": "Compact co-speech beat used by schema tests.",
        },
        "status": status,
        "semantics": {
            "gesture_family": "beat",
            "communicative_functions": ["emphasis", "explanation"],
            "tags": ["co_speech", "bilateral"],
            "handedness": "bilateral",
            "confidence": 0.82,
            "prosody": {
                "alignment": "accent",
                "target_anchor": "apex",
                "accent_types": ["pitch_accent"],
                "intensity": "adaptive",
                "confidence": 0.73,
            },
        },
        "source": {
            "dataset": "BEAT2",
            "speaker": "speaker_10",
            "session": "session_0",
            "clip": "clip_0001",
            "input_sha256": ZERO_HASH,
            "source_fps": 30.0,
            "frame_start": 120,
            "frame_end": 150,
            "time_start": 4.0,
            "time_end": 5.0,
            "relative_path": "pkl-results/clip_0001.pkl",
            "extractor": "phase_extractor",
            "extractor_version": "1.0.0",
        },
        "timing": {
            "nominal_duration_s": 1.0,
            "min_duration_s": 0.75,
            "max_duration_s": 1.3,
            "phases": {
                "preparation": {"start_s": 0.0, "end_s": 0.2},
                "stroke": {"start_s": 0.2, "end_s": 0.6},
                "hold": {"start_s": 0.6, "end_s": 0.7},
                "retraction": {"start_s": 0.7, "end_s": 1.0},
            },
            "anchors": {
                "apex": 0.45,
                "semantic_onset": 0.2,
                "semantic_offset": 0.7,
            },
            "safe_cutpoints": [0.0, 0.2, 0.7, 1.0],
            "min_lead_ms": 180.0,
            "late_policy": "fallback",
        },
        "control": {
            "sample_rate_hz": 50.0,
            "joints": joints,
            "joint_mask": joints,
            "neutral": {"id": "walking_arms_down.v1", "content_hash": ZERO_HASH},
            "backend_compatibility": ["mujoco-g1-29dof", "unitree-g1-upper-body"],
            "interpolation": "quintic",
            "tempo_range": {"min": 0.75, "max": 1.3},
            "amplitude_range": {"min": 0.7, "max": 1.0},
            "trajectory": [
                {"t": 0.0, "q": dict(q0), "dq": dict(zero), "ddq": dict(zero)},
                {
                    "t": 0.5,
                    "q": dict(q1),
                    "dq": {"left_elbow": 0.25, "right_elbow": 0.2},
                    "ddq": {"left_elbow": -0.1, "right_elbow": -0.1},
                },
                {"t": 1.0, "q": dict(q2), "dq": dict(zero), "ddq": dict(zero)},
            ],
        },
        "entry": {
            "q": dict(q0),
            "dq": dict(zero),
            "ddq": dict(zero),
            "tolerance": {"q": 0.01, "dq": 0.02, "ddq": 0.1},
        },
        "exit": {
            "q": dict(q2),
            "dq": dict(zero),
            "ddq": dict(zero),
            "tolerance": {"q": 0.01, "dq": 0.02, "ddq": 0.1},
        },
        "safety": {
            "thresholds": {
                "joint_position_margin_rad": 0.05,
                "max_velocity_rad_s": 2.5,
                "max_acceleration_rad_s2": 8.0,
                "max_jerk_rad_s3": 80.0,
                "max_tracking_error_rad": 0.15,
                "min_collision_distance_m": 0.02,
                "max_torque_saturation_fraction": 0.05,
            },
            "metrics": {
                "passed": True,
                "base_mode": "fixed",
                "max_velocity_rad_s": 1.1,
                "max_acceleration_rad_s2": 4.2,
            },
            "per_joint": {
                joint: {
                    "q_min": -1.0,
                    "q_max": 1.5,
                    "max_abs_dq": 1.1,
                    "max_abs_ddq": 4.2,
                    "max_abs_jerk": 30.0,
                    "max_abs_tracking_error": 0.03,
                }
                for joint in joints
            },
            "violations": [],
            "warnings": [
                {"code": "FIXED_BASE_ONLY", "message": "Only fixed-base rollout has been checked."}
            ],
            "collision": {
                "checked": True,
                "passed": True,
                "self_collision_count": 0,
                "environment_collision_count": 0,
                "min_distance_m": 0.08,
            },
            "validator_version": "atom-safety/1.0.0",
            "certified_backends": [],
        },
        "fallback": {
            "atom_id": "safe_neutral.v1",
            "policy": "release_to_neutral",
            "reason_codes": ["late", "unsafe_transition", "backend_unavailable"],
        },
        "metadata": {"reviewer": "test"},
        "extensions": {"experiment": {"split": "dev"}},
    }


class AtomSchemaTests(unittest.TestCase):
    def test_valid_candidate_is_not_promoted_by_validation(self) -> None:
        atom = sample_atom()
        validate_atom(atom)
        self.assertEqual(atom["status"], "candidate")

    def test_schema_resource_has_expected_contract(self) -> None:
        schema = load_json_schema()
        self.assertEqual(schema["properties"]["schema_version"]["const"], SCHEMA_VERSION)
        self.assertEqual(schema["properties"]["status"]["enum"], ["candidate", "curated", "certified"])
        self.assertTrue({"entry", "exit", "safety", "fallback"}.issubset(schema["required"]))

    def test_hash_is_order_independent_and_excludes_itself(self) -> None:
        atom = sample_atom()
        first = atom_hash(atom)
        reordered = dict(reversed(list(atom.items())))
        reordered["identity"] = dict(atom["identity"], content_hash="f" * 64)
        self.assertEqual(atom_hash(reordered), first)
        reordered["timing"]["min_lead_ms"] = 181.0
        self.assertNotEqual(atom_hash(reordered), first)

    def test_write_fills_hash_and_load_verifies_round_trip(self) -> None:
        atom = sample_atom()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atom.json"
            written = write_atom(atom, path)
            loaded = load_atom(path)
        self.assertEqual(loaded, written)
        self.assertEqual(written["identity"]["content_hash"], atom_hash(written))
        self.assertNotIn("content_hash", atom["identity"])
        self.assertEqual(written["status"], "candidate")

    def test_load_rejects_tampering_after_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atom.json"
            write_atom(sample_atom(), path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["identity"]["name"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AtomValidationError, "content_hash"):
                load_atom(path)

    def test_rejects_non_phase_aware_or_misaligned_trajectory(self) -> None:
        cases = []

        apex_outside_stroke = sample_atom()
        apex_outside_stroke["timing"]["anchors"]["apex"] = 0.8
        cases.append((apex_outside_stroke, "anchors.apex"))

        missing_joint = sample_atom()
        del missing_joint["control"]["trajectory"][1]["dq"]["right_elbow"]
        cases.append((missing_joint, "missing joints"))

        endpoint_mismatch = sample_atom()
        endpoint_mismatch["exit"]["q"]["left_elbow"] = 0.8
        cases.append((endpoint_mismatch, "trajectory endpoint"))

        unsorted_cutpoints = sample_atom()
        unsorted_cutpoints["timing"]["safe_cutpoints"] = [0.0, 0.7, 0.2, 1.0]
        cases.append((unsorted_cutpoints, "strictly increasing"))

        for atom, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(AtomValidationError, message):
                validate_atom(atom)

    def test_source_path_is_portable_and_status_is_explicit(self) -> None:
        atom = sample_atom()
        atom["source"]["relative_path"] = "/home/user/private/input.pkl"
        with self.assertRaisesRegex(AtomValidationError, "relative, portable path"):
            validate_atom(atom)

        atom = sample_atom()
        atom["status"] = "validated"
        with self.assertRaisesRegex(AtomValidationError, "candidate, certified, curated"):
            validate_atom(atom)

    def test_certified_requires_full_safety_evidence(self) -> None:
        atom = sample_atom(status="certified")
        with self.assertRaisesRegex(AtomValidationError, "certified backend"):
            validate_atom(atom)

        atom["safety"]["certified_backends"] = ["mujoco-g1-29dof"]
        validate_atom(atom)

        atom["safety"]["collision"]["checked"] = False
        atom["safety"]["collision"]["passed"] = False
        with self.assertRaisesRegex(AtomValidationError, "checked and passing collision"):
            validate_atom(atom)

    def test_unknown_core_fields_and_non_finite_values_are_rejected(self) -> None:
        atom = sample_atom()
        atom["control"]["magic"] = True
        with self.assertRaisesRegex(AtomValidationError, "unknown field"):
            validate_atom(atom)

        atom = sample_atom()
        atom["extensions"]["bad"] = float("nan")
        with self.assertRaisesRegex(AtomValidationError, "finite JSON"):
            validate_atom(atom)

    def test_passing_safety_must_match_declared_thresholds(self) -> None:
        atom = sample_atom()
        atom["safety"]["collision"]["min_distance_m"] = -1.0
        with self.assertRaisesRegex(AtomValidationError, "min_collision_distance"):
            validate_atom(atom)

        atom = sample_atom()
        atom["safety"]["per_joint"]["left_elbow"]["max_abs_dq"] = 9.0
        with self.assertRaisesRegex(AtomValidationError, "per_joint dynamics"):
            validate_atom(atom)

    def test_pose_representation_is_explicit_but_legacy_optional(self) -> None:
        legacy = sample_atom()
        self.assertNotIn("pose_representation", legacy["control"])
        validate_atom(legacy)

        variants = (
            {
                "q_space": "absolute_joint_position_rad",
                "source_mapping": "preserve_absolute_retarget_pose",
                "gain_reference": "control.neutral",
            },
            {
                "q_space": "absolute_joint_position_rad",
                "source_mapping": "neutral_rebased_delta",
                "gain_reference": "source_entry",
            },
        )
        for representation in variants:
            atom = sample_atom()
            atom["control"]["pose_representation"] = representation
            validate_atom(atom)

        raw_knots = sample_atom()
        raw_knots["control"]["pose_representation"] = {
            "q_space": "absolute_joint_position_rad",
            "source_mapping": "preserve_absolute_retarget_pose",
            "gain_reference": "control.neutral",
            "sampling_basis": "contiguous_source_frame_knots_bit_exact_v2",
            "detector_filter_affects_export": False,
        }
        validate_atom(raw_knots)

        for port_name in ("entry", "exit"):
            raw_knots[port_name]["tolerance_semantics"] = (
                "storage_consistency_only_not_runtime_acceptance"
            )
        validate_atom(raw_knots)

    def test_raw_knot_representation_rejects_incomplete_or_filtered_claim(self) -> None:
        atom = sample_atom()
        atom["control"]["pose_representation"] = {
            "q_space": "absolute_joint_position_rad",
            "source_mapping": "preserve_absolute_retarget_pose",
            "gain_reference": "control.neutral",
            "sampling_basis": "contiguous_source_frame_knots_bit_exact_v2",
        }
        with self.assertRaisesRegex(AtomValidationError, "declared together"):
            validate_atom(atom)

        atom["control"]["pose_representation"]["detector_filter_affects_export"] = True
        with self.assertRaisesRegex(AtomValidationError, "must be false"):
            validate_atom(atom)

    def test_pose_representation_rejects_cross_field_mismatch(self) -> None:
        atom = sample_atom()
        atom["control"]["pose_representation"] = {
            "q_space": "absolute_joint_position_rad",
            "source_mapping": "preserve_absolute_retarget_pose",
            "gain_reference": "source_entry",
        }
        with self.assertRaisesRegex(AtomValidationError, "must equal 'control.neutral'"):
            validate_atom(atom)

    def test_json_schema_representation_variants_match_python_contract(self) -> None:
        schema = load_json_schema()
        variants = schema["$defs"]["poseRepresentation"]["oneOf"]
        observed = {
            (
                item["properties"]["q_space"]["const"],
                item["properties"]["source_mapping"]["const"],
                item["properties"]["gain_reference"]["const"],
            )
            for item in variants
        }
        self.assertEqual(
            observed,
            {
                (
                    "absolute_joint_position_rad",
                    "preserve_absolute_retarget_pose",
                    "control.neutral",
                ),
                (
                    "absolute_joint_position_rad",
                    "neutral_rebased_delta",
                    "source_entry",
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
