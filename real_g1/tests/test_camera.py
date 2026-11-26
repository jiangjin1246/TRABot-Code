from types import SimpleNamespace
import unittest

from sim_service.cli import build_parser
from sim_service.engine import CONVERSATION_CAMERA, FreeCameraSpec, build_render_camera


class _FakeCamera:
    def __init__(self):
        self.type = None
        self.fixedcamid = 99
        self.trackbodyid = 99
        self.lookat = [0.0, 0.0, 0.0]
        self.distance = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0


class CameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mujoco = SimpleNamespace(
            MjvCamera=_FakeCamera,
            mjtCamera=SimpleNamespace(mjCAMERA_FREE=7),
        )

    def test_default_is_explicit_centered_conversation_camera(self) -> None:
        camera = build_render_camera(self.mujoco, None)
        self.assertEqual(camera.type, 7)
        self.assertEqual(camera.fixedcamid, -1)
        self.assertEqual(camera.trackbodyid, -1)
        self.assertEqual(tuple(camera.lookat), (0.0, 0.0, 0.62))
        self.assertEqual(camera.distance, 2.2)
        self.assertEqual(camera.azimuth, 180.0)
        self.assertEqual(camera.elevation, -6.0)

    def test_fixed_camera_and_scene_default_remain_available(self) -> None:
        self.assertEqual(build_render_camera(self.mujoco, "tracking"), "tracking")
        self.assertEqual(build_render_camera(self.mujoco, "2"), 2)
        self.assertEqual(build_render_camera(self.mujoco, "scene"), -1)

    def test_cli_camera_defaults_match_engine_defaults(self) -> None:
        args = build_parser().parse_args(["--engine", "kinematic"])
        self.assertEqual(tuple(args.camera_lookat), CONVERSATION_CAMERA.lookat)
        self.assertEqual(args.camera_distance, CONVERSATION_CAMERA.distance)
        self.assertEqual(args.camera_azimuth, CONVERSATION_CAMERA.azimuth)
        self.assertEqual(args.camera_elevation, CONVERSATION_CAMERA.elevation)

    def test_invalid_camera_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FreeCameraSpec(distance=0.0)
        with self.assertRaises(ValueError):
            FreeCameraSpec(elevation=90.0)


if __name__ == "__main__":
    unittest.main()
