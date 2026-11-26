from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from atom_pipeline.auto_repair import (
    AutoRepairCanceled,
    AutoRepairPolicy,
    FailureProfile,
    analyze_failure,
    generate_recipes,
    maximum_position_limb_scales,
    search_best_repair,
)
from atom_pipeline.repair import AtomRepairError, RepairResult
from atom_pipeline.schema import atom_hash
from tests.test_atom_repair import dense_candidate, limits


def failing_candidate() -> dict:
    atom = dense_candidate()
    atom["safety"]["metrics"]["passed"] = False
    atom["safety"]["metrics"]["kinematics_passed"] = False
    atom["safety"]["collision"] = {
        "checked": True,
        "passed": False,
        "self_collision_count": 1,
        "environment_collision_count": 0,
        "min_distance_m": -0.01,
    }
    atom["safety"]["violations"] = [
        {
            "code": "JOINT_JERK_LIMIT",
            "message": "jerk exceeded",
            "joint": "left_elbow",
            "time_s": 0.5,
            "observed": 60.0,
            "limit": 30.0,
        },
        {"code": "SELF_COLLISION", "message": "contact"},
    ]
    return atom


class FreshCollisionValidator:
    def __init__(self, *, passed: bool) -> None:
        self.passed = passed

    def joint_ranges(self, joints):
        return {joint: (-10.0, 10.0) for joint in joints}

    def validate(self, *args, **kwargs):
        contact = {
            "kind": "self",
            "body_names": ["left_hand", "torso"],
            "geom_names": ["left_hand_geom", "torso_geom"],
        }
        violation = {"code": "SELF_COLLISION", **contact}
        return {
            "result": "pass" if self.passed else "fail",
            "backend": "fake-mujoco",
            "violations": [] if self.passed else [violation],
            "warnings": [],
            "contact_pairs": [] if self.passed else [contact],
            "minimum_contact_distance_m": None if self.passed else -0.01,
        }


class AutoRepairPolicyTests(unittest.TestCase):
    def test_failure_profile_and_recipe_order_are_violation_aware(self) -> None:
        profile = analyze_failure(failing_candidate())
        self.assertTrue(profile.jerk)
        self.assertTrue(profile.collision)
        recipes = generate_recipes(profile)
        self.assertLessEqual(len(recipes), 24)
        self.assertEqual(recipes[0].amplitude_scale, 1.0)
        self.assertEqual(recipes[0].smoothing_passes, 1)
        self.assertIn(0.5, {recipe.amplitude_scale for recipe in recipes})

    def test_velocity_only_does_not_start_with_amplitude_loss(self) -> None:
        profile = FailureProfile(False, True, False, False, False, ("JOINT_VELOCITY_LIMIT",))
        recipes = generate_recipes(profile)
        self.assertTrue(recipes)
        self.assertTrue(all(recipe.amplitude_scale == 1.0 for recipe in recipes))
        self.assertTrue(all(recipe.limb_scales is None for recipe in recipes))
        self.assertEqual(recipes[0].smoothing_passes, 0)

    def test_analytic_position_scale_is_per_limb_and_maximal(self) -> None:
        atom = dense_candidate()
        position_limits = limits(
            ranges={
                "left_elbow": (-2.0, 0.75),
                "right_elbow": (-2.0, 2.0),
            }
        )
        scales = maximum_position_limb_scales(atom, position_limits)
        # Effective high is 0.70, neutral/entry is 0.20, peak is 1.20.
        self.assertAlmostEqual(scales["left_arm"], 0.5, places=8)
        self.assertEqual(scales["right_arm"], 1.0)
        self.assertEqual(scales["waist"], 1.0)


class AutoRepairSearchTests(unittest.TestCase):
    @staticmethod
    def fake_repair(atom, *, recipe, collision_validator=None, **_kwargs):
        child = deepcopy(atom)
        joints = child["control"]["joints"]
        entry = child["control"]["trajectory"][0]["q"]
        for point in child["control"]["trajectory"]:
            for joint in joints:
                point["q"][joint] = entry[joint] + recipe.amplitude_scale * (
                    point["q"][joint] - entry[joint]
                )
        child["safety"]["metrics"]["passed"] = True
        child["safety"]["metrics"]["kinematics_passed"] = True
        checked = collision_validator is not None
        child["safety"]["collision"] = {
            "checked": checked,
            "passed": checked and recipe.smoothing_passes >= 2,
            "self_collision_count": 0 if checked and recipe.smoothing_passes >= 2 else 1,
            "environment_collision_count": 0,
            "min_distance_m": 0.01 if checked and recipe.smoothing_passes >= 2 else -0.01,
        }
        child["safety"]["violations"] = []
        applied = {
            **recipe.as_dict(),
            "applied_time_scale": 1.0,
            "resulting_duration_s": child["timing"]["nominal_duration_s"],
        }
        return RepairResult(child, {}, {}, applied, atom_hash(atom))

    def test_search_keeps_trials_in_memory_and_selects_first_collision_pass(self) -> None:
        parent = failing_candidate()
        before = deepcopy(parent)
        with patch("atom_pipeline.auto_repair.repair_candidate", side_effect=self.fake_repair):
            result = search_best_repair(
                parent,
                limits=limits(jerk=1.0),
                collision_validator=FreshCollisionValidator(passed=False),
                require_collision=True,
            )
        self.assertEqual(parent, before)
        self.assertEqual(result.status, "repaired")
        self.assertIsNotNone(result.winner)
        self.assertEqual(result.winner_score.smoothing_passes, 2)
        self.assertEqual(result.collision_checked, 2)
        self.assertEqual(result.summary()["policy"], "minimal_distortion_limb_v2")

    def test_left_collision_search_only_scales_left_arm(self) -> None:
        parent = failing_candidate()
        detailed_report = {
            "kinematics": {"violations": []},
            "collision": {
                "violations": [
                    {
                        "code": "SELF_COLLISION",
                        "body_names": ["left_hand", "torso"],
                    }
                ]
            },
        }
        attempted = []

        def capture(atom, *, recipe, collision_validator=None, **kwargs):
            attempted.append(recipe)
            result = self.fake_repair(
                atom,
                recipe=recipe,
                collision_validator=collision_validator,
                **kwargs,
            )
            if collision_validator is not None:
                result.atom["safety"]["collision"]["passed"] = True
            return result

        with patch("atom_pipeline.auto_repair.repair_candidate", side_effect=capture):
            result = search_best_repair(
                parent,
                report=detailed_report,
                limits=limits(),
                collision_validator=FreshCollisionValidator(passed=False),
                require_collision=True,
            )
        self.assertEqual(result.status, "repaired")
        self.assertTrue(attempted)
        self.assertTrue(any(recipe.limb_scales["left_arm"] < 1.0 for recipe in attempted))
        for recipe in attempted:
            self.assertEqual(recipe.amplitude_scale, 1.0)
            self.assertEqual(recipe.limb_scales["right_arm"], 1.0)
            self.assertEqual(recipe.limb_scales["waist"], 1.0)
        self.assertIsNotNone(result.winner_score.limb_scales)

    def test_collision_budget_reaches_every_configured_amplitude_bucket(self) -> None:
        parent = failing_candidate()
        detailed_report = {
            "kinematics": {"violations": []},
            "collision": {
                "violations": [
                    {
                        "code": "SELF_COLLISION",
                        "body_names": ["left_hand", "torso"],
                    }
                ]
            },
        }
        collision_recipes = []

        def always_collision(atom, *, recipe, collision_validator=None, **kwargs):
            result = self.fake_repair(
                atom,
                recipe=recipe,
                collision_validator=collision_validator,
                **kwargs,
            )
            if collision_validator is not None:
                collision_recipes.append(recipe)
                result.atom["safety"]["collision"]["passed"] = False
            return result

        policy = AutoRepairPolicy(max_collision_trials=8)
        with patch(
            "atom_pipeline.auto_repair.repair_candidate",
            side_effect=always_collision,
        ):
            result = search_best_repair(
                parent,
                report=detailed_report,
                policy=policy,
                limits=limits(),
                collision_validator=FreshCollisionValidator(passed=False),
                require_collision=True,
            )
        self.assertEqual(result.reason_code, "NO_COLLISION_SAFE_RECIPE")
        self.assertEqual(len(collision_recipes), 8)
        tried_left = {
            round(recipe.limb_scales["left_arm"], 8)
            for recipe in collision_recipes
        }
        self.assertEqual(
            tried_left,
            {round(value, 8) for value in policy.amplitude_scales},
        )
        self.assertTrue(
            all(recipe.limb_scales["right_arm"] == 1.0 for recipe in collision_recipes)
        )

    def test_collision_validator_is_required_and_cancel_is_cooperative(self) -> None:
        parent = failing_candidate()
        no_validator = search_best_repair(parent, limits=limits(), require_collision=True)
        self.assertEqual(no_validator.reason_code, "COLLISION_VALIDATOR_REQUIRED")
        with self.assertRaises(AutoRepairCanceled):
            search_best_repair(
                parent,
                limits=limits(),
                collision_validator=FreshCollisionValidator(passed=False),
                cancel_check=lambda: True,
            )

    def test_default_limits_load_model_ranges_before_position_repair(self) -> None:
        parent = failing_candidate()
        parent["safety"]["violations"] = [
            {
                "code": "JOINT_POSITION_LIMIT",
                "message": "position exceeded",
                "joint": "left_elbow",
                "observed": 1.2,
                "limit": 0.7,
                "time_s": 0.5,
            }
        ]

        class RangeValidator:
            def __init__(self):
                self.range_calls = 0

            def joint_ranges(self, joints):
                self.range_calls += 1
                return {
                    "left_elbow": (-2.0, 0.75),
                    "right_elbow": (-2.0, 2.0),
                }

            def validate(self, *args, **kwargs):
                return {"result": "pass", "violations": []}

        validator = RangeValidator()
        seen_limits = []

        def capture(atom, *, recipe, limits, collision_validator=None, **kwargs):
            seen_limits.append(limits)
            result = self.fake_repair(
                atom,
                recipe=recipe,
                collision_validator=collision_validator,
                **kwargs,
            )
            if collision_validator is not None:
                result.atom["safety"]["collision"]["passed"] = True
            return result

        with patch("atom_pipeline.auto_repair.repair_candidate", side_effect=capture):
            result = search_best_repair(
                parent,
                collision_validator=validator,
                require_collision=True,
            )
        self.assertEqual(result.status, "repaired")
        self.assertEqual(validator.range_calls, 1)
        self.assertTrue(seen_limits)
        self.assertEqual(seen_limits[0].joint_ranges["left_elbow"], (-2.0, 0.75))
        self.assertAlmostEqual(
            result.winner.applied_recipe["limb_scales"]["left_arm"],
            0.5,
            places=8,
        )

    def test_stale_embedded_pass_does_not_bypass_fresh_collision_failure(self) -> None:
        atom = dense_candidate()
        atom["safety"]["metrics"]["passed"] = True
        atom["safety"]["metrics"]["kinematics_passed"] = True
        atom["safety"]["collision"] = {
            "checked": True,
            "passed": True,
            "self_collision_count": 0,
            "environment_collision_count": 0,
            "min_distance_m": 0.02,
        }
        result = search_best_repair(
            atom,
            limits=limits(),
            collision_validator=FreshCollisionValidator(passed=False),
        )
        self.assertNotEqual(result.status, "not_needed")
        self.assertEqual(result.reason_code, "NO_COLLISION_SAFE_RECIPE")
        self.assertGreater(result.collision_checked, 0)

    def test_position_repair_requires_ranges_for_every_controlled_joint(self) -> None:
        parent = failing_candidate()
        parent["safety"]["violations"] = [
            {
                "code": "JOINT_POSITION_LIMIT",
                "message": "position exceeded",
                "joint": "left_elbow",
                "observed": 1.2,
                "limit": 0.7,
                "time_s": 0.5,
            }
        ]
        partial = limits(ranges={"left_elbow": (-2.0, 0.75)})
        result = search_best_repair(
            parent,
            limits=partial,
            collision_validator=FreshCollisionValidator(passed=True),
        )
        self.assertEqual(result.reason_code, "POSITION_RANGES_REQUIRED")
        self.assertEqual(
            result.trial_outcomes[0]["joints"],
            ["right_elbow"],
        )

    def test_unattributed_fresh_collision_fails_closed(self) -> None:
        class UnattributedValidator(FreshCollisionValidator):
            def validate(self, *args, **kwargs):
                return {
                    "result": "fail",
                    "violations": [{"code": "SELF_COLLISION"}],
                }

        result = search_best_repair(
            failing_candidate(),
            report={"kinematics": {"violations": []}},
            limits=limits(),
            collision_validator=UnattributedValidator(passed=False),
        )
        self.assertEqual(result.reason_code, "COLLISION_LIMB_UNRESOLVED")
        self.assertEqual(result.recipes_attempted, 0)

    def test_passing_atom_is_not_needlessly_repaired(self) -> None:
        atom = dense_candidate()
        atom["safety"]["metrics"]["kinematics_passed"] = True
        atom["safety"]["collision"] = {
            "checked": True,
            "passed": True,
            "self_collision_count": 0,
            "environment_collision_count": 0,
            "min_distance_m": 0.02,
        }
        result = search_best_repair(
            atom, limits=limits(), collision_validator=FreshCollisionValidator(passed=True)
        )
        self.assertEqual(result.status, "not_needed")
        self.assertEqual(result.recipes_attempted, 0)

    def test_frozen_strict_jerk_does_not_become_not_needed_under_new_defaults(self) -> None:
        parent = dense_candidate()
        strict = limits(
            jerk=20.0,
            ranges={
                "left_elbow": (-10.0, 10.0),
                "right_elbow": (-10.0, 10.0),
            },
        )
        report = {
            "kinematics": {
                "violations": [
                    {
                        "code": "JOINT_JERK_LIMIT",
                        "joint": "left_elbow",
                        "observed": 27.0,
                        "limit": 20.0,
                    }
                ]
            }
        }
        with patch(
            "atom_pipeline.auto_repair.repair_candidate",
            side_effect=AtomRepairError("stop after not-needed gate"),
        ):
            result = search_best_repair(
                parent,
                report=report,
                limits=strict,
                collision_validator=FreshCollisionValidator(passed=True),
            )
        self.assertNotEqual(result.status, "not_needed")
        self.assertGreater(result.recipes_attempted, 0)

    def test_policy_time_cap_is_clamped_to_frozen_limits(self) -> None:
        parent = failing_candidate()
        frozen = limits(jerk=1.0)
        frozen = type(frozen)(
            velocity_rad_s=frozen.velocity_rad_s,
            acceleration_rad_s2=frozen.acceleration_rad_s2,
            jerk_rad_s3=frozen.jerk_rad_s3,
            velocity_by_joint=frozen.velocity_by_joint,
            acceleration_by_joint=frozen.acceleration_by_joint,
            jerk_by_joint=frozen.jerk_by_joint,
            joint_ranges=frozen.joint_ranges,
            position_margin_rad=frozen.position_margin_rad,
            warning_fraction=frozen.warning_fraction,
            time_scale_safety_factor=frozen.time_scale_safety_factor,
            max_time_scale=1.5,
        )
        observed_caps = []
        def capture(atom, *, recipe, **kwargs):
            observed_caps.append(recipe.max_time_scale)
            raise AtomRepairError("captured")
        with patch("atom_pipeline.auto_repair.repair_candidate", side_effect=capture):
            result = search_best_repair(
                parent,
                policy=AutoRepairPolicy(max_time_scale=3.0),
                limits=frozen,
                collision_validator=FreshCollisionValidator(passed=True),
            )
        self.assertTrue(observed_caps)
        self.assertTrue(all(value <= 1.5 for value in observed_caps))



if __name__ == "__main__":
    unittest.main()
