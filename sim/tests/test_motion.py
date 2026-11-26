import math

import unittest

from sim_service.motion import MotionValidationError, parse_motion


class MotionTests(unittest.TestCase):
    def test_bridge_motion_samples_smoothstep(self) -> None:
        clip = parse_motion(
        {
            "name": "test",
            "joints": ["j1"],
            "frames": [{"t": 2.0, "q": [0.0]}, {"t": 4.0, "q": [2.0]}],
        }
    )
        self.assertEqual(clip.times, (0.0, 2.0))
        self.assertEqual(clip.duration, 2.0)
        self.assertEqual(clip.sample(-1.0), {"j1": 0.0})
        self.assertEqual(clip.sample(1.0), {"j1": 1.0})
        self.assertEqual(clip.sample(9.0), {"j1": 2.0})


    def test_bridge_dict_q_and_declared_duration(self) -> None:
        clip = parse_motion(
            {
                "joints": ["j2", "j1"],
                "duration": 3.6,
                "frames": [
                    {"t": 0.0, "q": {"j1": 1.0, "j2": 2.0}},
                    {"t": 2.35, "q": {"j1": 3.0, "j2": 4.0}},
                ],
            }
        )
        self.assertEqual(clip.times, (0.0, 2.35, 3.6))
        self.assertEqual(clip.sample(3.6), {"j2": 4.0, "j1": 3.0})

    def test_crisp_motion_carries_unspecified_targets_and_duration(self) -> None:
        clip = parse_motion(
        {
            "neutral": {"j1": 0.2, "j2": -0.2},
            "duration": 2.0,
            "keyframes": [
                {"time": 0.0, "targets": {"j1": 0.4}},
                {"time": 1.0, "targets": {"j2": 0.8}},
            ],
        },
        name="crisp",
    )
        self.assertEqual(clip.duration, 2.0)
        self.assertEqual(clip.sample(0.0), {"j1": 0.4, "j2": -0.2})
        self.assertEqual(clip.sample(1.0), {"j1": 0.4, "j2": 0.8})
        self.assertEqual(clip.sample(2.0), {"j1": 0.4, "j2": 0.8})


    def test_oscillation_has_zero_ramped_endpoints(self) -> None:
        clip = parse_motion(
        {
            "neutral": {"j1": 0.5},
            "keyframes": [
                {"time": 0.0, "targets": {"j1": 0.5}},
                {"time": 1.0, "targets": {"j1": 0.5}},
            ],
            "oscillations": [
                {"joint": "j1", "center": 0.5, "amplitude": 0.2, "frequency_hz": 1.0}
            ],
        }
    )
        self.assertAlmostEqual(clip.sample(0.0)["j1"], 0.5)
        self.assertAlmostEqual(clip.sample(1.0)["j1"], 0.5)
        self.assertTrue(math.isfinite(clip.sample(0.25)["j1"]))


    def test_invalid_motion_rejected(self) -> None:
        payloads = [
            {},
            {"joints": ["j1"], "frames": []},
            {"joints": ["j1"], "frames": [{"t": 0, "q": [float("nan")]}]},
            {"joints": ["j1"], "frames": [{"t": 1, "q": [0]}, {"t": 1, "q": [1]}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(MotionValidationError):
                parse_motion(payload)
