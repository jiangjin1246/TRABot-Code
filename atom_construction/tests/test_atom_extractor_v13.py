from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from atom_pipeline.extractor import ExtractorConfig, extract_sources
from atom_pipeline.schema import load_atom
from atom_pipeline.source import discover_sources


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


class AbsolutePoseExtractionTests(unittest.TestCase):
    def test_absolute_entry_is_preserved_and_gain_is_about_neutral(self) -> None:
        fps = 30
        times = np.arange(0.0, 6.0, 1.0 / fps)
        dof_pos = np.zeros((len(times), 29), dtype=np.float64)
        dof_pos[:, 22] = -0.543210987654321
        dof_pos[:, 23] = -1.00
        dof_pos[:, 25] = 0.15
        # This source-frame-only alternating component is intentionally above
        # the detector branch bandwidth. It must survive export bit-for-bit.
        dof_pos[:, 24] = 0.031 * np.where(np.arange(len(times)) % 2 == 0, -1.0, 1.0)
        outward = _minimum_jerk((times - 1.8) / 0.5)
        inward = _minimum_jerk((times - 2.8) / 0.6)
        dof_pos[:, 28] = -0.40 + 0.50 * (outward - inward)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "pkl-results"
            sources.mkdir()
            source = sources / "21_ayana_1_12_12.pkl"
            with source.open("wb") as stream:
                pickle.dump({"fps": fps, "dof_pos": dof_pos}, stream)
            records = discover_sources(sources)
            neutral_path = Path(__file__).parents[2] / "catalog" / "safe_neutral.json"
            neutral_payload = json.loads(neutral_path.read_text(encoding="utf-8"))
            neutral = neutral_payload.get("neutral") or neutral_payload["frames"][0]["q"]

            atoms = {}
            for gain in (1.0, 0.5):
                output = root / f"run-{gain}"
                summary = extract_sources(
                    records,
                    output_dir=output,
                    neutral_path=neutral_path,
                    config=ExtractorConfig(
                        gain=gain,
                        candidates_per_minute=20.0,
                        max_candidates_per_source=1,
                    ),
                )
                self.assertEqual(summary.candidate_count, 1)
                atoms[gain] = load_atom(next((output / "candidates").glob("*.json")))

        full = atoms[1.0]
        half = atoms[0.5]
        representation = {
            "q_space": "absolute_joint_position_rad",
            "source_mapping": "preserve_absolute_retarget_pose",
            "gain_reference": "control.neutral",
            "sampling_basis": "contiguous_source_frame_knots_bit_exact_v2",
            "detector_filter_affects_export": False,
        }
        self.assertEqual(full["control"]["pose_representation"], representation)
        self.assertEqual(half["control"]["pose_representation"], representation)
        self.assertAlmostEqual(full["entry"]["q"]["right_shoulder_pitch"], -0.543210987654321, places=12)
        self.assertNotAlmostEqual(
            full["entry"]["q"]["right_shoulder_pitch"],
            neutral["right_shoulder_pitch"],
            places=3,
        )
        expected_half_entry = neutral["right_shoulder_pitch"] + 0.5 * (
            full["entry"]["q"]["right_shoulder_pitch"]
            - neutral["right_shoulder_pitch"]
        )
        self.assertAlmostEqual(
            half["entry"]["q"]["right_shoulder_pitch"],
            expected_half_entry,
            places=6,
        )
        self.assertAlmostEqual(
            full["metadata"]["processing"]["source_entry_offset_from_neutral_rad"][
                "right_shoulder_pitch"
            ],
            -0.543210987654321 - neutral["right_shoulder_pitch"],
            places=6,
        )

        source_start = full["source"]["frame_start"]
        source_end = full["source"]["frame_end"]
        trajectory = full["control"]["trajectory"]
        self.assertEqual(len(trajectory), source_end - source_start + 1)
        raw_high_frequency = dof_pos[source_start : source_end + 1, 24]
        exported_high_frequency = np.asarray(
            [point["q"]["right_shoulder_yaw"] for point in trajectory]
        )
        np.testing.assert_array_equal(exported_high_frequency, raw_high_frequency)
        self.assertEqual(
            full["entry"]["q"]["right_shoulder_yaw"],
            float(dof_pos[source_start, 24]),
        )
        export = full["metadata"]["processing"]["trajectory_export"]
        self.assertEqual(export["resampling_method"], "none")
        self.assertFalse(export["detector_filter_affects_export"])
        self.assertTrue(export["source_knots_equal_original_q"])
        self.assertEqual(export["source_knot_max_abs_export_error_rad"], 0.0)
        applied_scale = full["metadata"]["processing"]["applied_time_scale"]
        self.assertAlmostEqual(full["control"]["sample_rate_hz"], fps / applied_scale)
        np.testing.assert_allclose(
            np.diff([point["t"] for point in trajectory]),
            applied_scale / fps,
            rtol=0.0,
            atol=1e-12,
        )

        def joint_range(atom: dict, joint: str) -> float:
            values = [point["q"][joint] for point in atom["control"]["trajectory"]]
            return max(values) - min(values)

        full_range = joint_range(full, "right_wrist_yaw")
        half_range = joint_range(half, "right_wrist_yaw")
        self.assertGreater(full_range, 0.40)
        self.assertAlmostEqual(half_range / full_range, 0.5, places=3)

        for atom in (full, half):
            trajectory = atom["control"]["trajectory"]
            for port, point in ((atom["entry"], trajectory[0]), (atom["exit"], trajectory[-1])):
                for vector in ("q", "dq", "ddq"):
                    self.assertEqual(port[vector], point[vector])
                self.assertEqual(
                    port["tolerance_semantics"],
                    "storage_consistency_only_not_runtime_acceptance",
                )
            self.assertEqual(
                list(atom["timing"]["phases"]),
                ["preparation", "stroke", "hold", "retraction"],
            )
            stroke = atom["timing"]["phases"]["stroke"]
            apex = atom["timing"]["anchors"]["apex"]
            self.assertLessEqual(stroke["start_s"], apex)
            self.assertLessEqual(apex, stroke["end_s"])

    def test_v13_refuses_to_emit_legacy_rebased_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "only emits"):
            ExtractorConfig(source_pose_mapping="neutral_rebased_delta")


if __name__ == "__main__":
    unittest.main()
