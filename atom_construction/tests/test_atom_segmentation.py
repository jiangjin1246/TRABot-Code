from __future__ import annotations

import unittest

import numpy as np

from atom_pipeline.segmentation import DetectorConfig, detect_gesture_segments


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def synthetic_gesture(fps: int = 30) -> tuple[np.ndarray, int]:
    times = np.arange(0.0, 8.0, 1.0 / fps)
    q = np.zeros((len(times), 14), dtype=np.float64)
    q[:, 0] = 0.8 * _minimum_jerk((times - 2.0) / 0.6)
    q[:, 0] -= 0.8 * _minimum_jerk((times - 3.0) / 0.7)
    q[:, 3] = 0.5 * _minimum_jerk((times - 2.1) / 0.5)
    q[:, 3] -= 0.5 * _minimum_jerk((times - 3.1) / 0.6)
    return q, fps


class AtomSegmentationTests(unittest.TestCase):
    def test_hold_connects_outward_stroke_and_retraction(self) -> None:
        q, fps = synthetic_gesture()
        result = detect_gesture_segments(q, fps)
        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertEqual(segment.primary_apex.kind, "postural")
        self.assertEqual(segment.exit_kind, "returned")
        self.assertGreaterEqual(segment.kinematic_quality, 0.55)
        names = [phase.name for phase in segment.phases]
        self.assertEqual(names, ["preparation", "stroke", "hold", "retraction"])
        self.assertLess(segment.start_index / result.fps, 2.0)
        self.assertGreater(segment.end_index / result.fps, 3.5)
        self.assertEqual(result.rejections, ())

    def test_static_sequence_has_no_false_gesture(self) -> None:
        q = np.zeros((300, 14), dtype=np.float64)
        result = detect_gesture_segments(q, 30)
        self.assertEqual(result.segments, ())
        self.assertEqual(result.rejections[0]["reason"], "motion_snr_too_low")

    def test_asymmetric_motion_uses_stationary_plateau_as_postural_apex(self) -> None:
        fps = 30
        times = np.arange(0.0, 7.0, 1.0 / fps)
        q = np.zeros((len(times), 14), dtype=np.float64)
        q[:, 0] = 0.8 * _minimum_jerk((times - 1.5) / 1.2)
        q[:, 0] -= 0.8 * _minimum_jerk((times - 3.2) / 0.3)
        result = detect_gesture_segments(q, fps)
        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertEqual(segment.primary_apex.kind, "postural")
        apex_time = segment.primary_apex.index / result.fps
        self.assertGreaterEqual(apex_time, 2.5)
        self.assertLess(apex_time, 3.25)
        hold = next(phase for phase in segment.phases if phase.name == "hold")
        self.assertGreaterEqual((hold.end_index - hold.start_index) / result.fps, 0.35)

    def test_invalid_values_fail_closed(self) -> None:
        q, fps = synthetic_gesture()
        q[30, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            detect_gesture_segments(q, fps, config=DetectorConfig())


if __name__ == "__main__":
    unittest.main()
