from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sim_service.engine import KinematicEngine, MujocoEngine
from sim_service.http_api import create_http_server
from sim_service.motion import MotionValidationError
from sim_service.service import (
    LEGACY_CUBIC_BRIDGE_POLICY,
    MINIMUM_JERK_BRIDGE_POLICY,
    MotionService,
    ServiceConfig,
    minimum_jerk_bridge_duration,
)
from web_overlay.config import OverlayConfig
from web_overlay.sim_client import SimServiceClient


JOINT = "right_elbow_joint"
OTHER_JOINT = "left_elbow_joint"


def motion(*, first: float = 0.04, last: float = 0.14, duration: float = 0.1):
    return {
        "name": "minimum-jerk-test",
        "joints": [JOINT],
        "interpolation": "linear",
        "frames": [
            {"t": 0.0, "q": [first]},
            {"t": duration, "q": [last]},
        ],
    }


class _RecordingClient(SimServiceClient):
    def __init__(self) -> None:
        super().__init__(OverlayConfig())
        self.payload = None

    def post_action_sync(self, payload):
        self.payload = dict(payload)
        return {"accepted": True, "id": "recorded"}


class MinimumJerkBridgeTests(unittest.TestCase):
    def test_duration_matches_fixed_speed_tick_rule(self) -> None:
        self.assertEqual(minimum_jerk_bridge_duration(0.0), 0.08)
        self.assertEqual(minimum_jerk_bridge_duration(0.1), 0.2)
        self.assertEqual(minimum_jerk_bridge_duration(0.64), 1.2)

    def test_nonfinite_derived_duration_is_a_validation_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "duration"):
            minimum_jerk_bridge_duration(1e308)

    def test_large_bridge_is_rejected_over_http_without_faulting_service(self) -> None:
        service = MotionService(
            KinematicEngine(joints=(JOINT,), timestep=0.002),
            config=ServiceConfig(render_fps=0, max_motion_duration=10.0),
        )
        server = create_http_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        service.start()
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and JOINT not in service.state()["q"]:
                time.sleep(0.005)
            self.assertIn(JOINT, service.state()["q"])
            request = Request(
                base + "/actions/trigger",
                data=json.dumps(
                    {
                        "motion": motion(first=1e308, last=1e308),
                        "bridge_policy": MINIMUM_JERK_BRIDGE_POLICY,
                        "blend_duration": 0,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 400)
            self.assertTrue(service.health()["ok"])
            self.assertIsNone(service.health()["fault"])
        finally:
            server.shutdown()
            server.server_close()
            service.stop()
            thread.join(1)

    def test_start_time_bridge_validation_fails_only_that_queued_action(self) -> None:
        engine = KinematicEngine(
            joints=(JOINT, OTHER_JOINT),
            timestep=0.002,
        )
        # The queue is deliberately built before the first engine snapshot.
        # At actual action start JOINT is far enough away to exceed the bound.
        engine.q[JOINT] = -100.0
        engine.target[JOINT] = -100.0
        service = MotionService(
            engine,
            config=ServiceConfig(render_fps=0, max_motion_duration=10.0),
        )
        bad = service.submit(
            motion(first=1.0, last=1.1),
            action_id="too-large-at-start",
            bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
            blend_duration=0.0,
        )
        good = service.submit(
            {
                "name": "safe-after-invalid",
                "joints": [OTHER_JOINT],
                "interpolation": "linear",
                "frames": [
                    {"t": 0.0, "q": [0.0]},
                    {"t": 0.05, "q": [0.05]},
                ],
            },
            action_id="safe-after-invalid",
            bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
            blend_duration=0.0,
        )
        service.start()
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if service.get_action(good["id"])["status"] == "completed":
                    break
                time.sleep(0.005)
            self.assertEqual(service.get_action(bad["id"])["status"], "failed")
            self.assertEqual(service.get_action(good["id"])["status"], "completed")
            self.assertTrue(service.health()["ok"])
            self.assertIsNone(service.health()["fault"])
        finally:
            service.stop()

    def test_missing_actual_q_fails_action_without_stopping_service(self) -> None:
        class MissingQEngine(KinematicEngine):
            def snapshot(self):
                value = super().snapshot()
                value["q"] = {}
                return value

        service = MotionService(
            MissingQEngine(joints=(JOINT,), timestep=0.002),
            config=ServiceConfig(render_fps=0),
        )
        accepted = service.submit(
            motion(),
            action_id="missing-actual-q",
            bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
            blend_duration=0.0,
        )
        service.start()
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if service.get_action(accepted["id"])["status"] == "failed":
                    break
                time.sleep(0.005)
            record = service.get_action(accepted["id"])
            self.assertEqual(record["status"], "failed")
            self.assertIn("missing controlled joint", record["reason"])
            self.assertTrue(service.health()["ok"])
            self.assertIsNone(service.health()["fault"])
        finally:
            service.stop()

    def test_start_captures_actual_q_and_updates_500hz_quintic_target(self) -> None:
        service = MotionService(
            KinematicEngine(joints=(JOINT,), timestep=0.002),
            config=ServiceConfig(render_fps=0),
        )
        service.submit(
            motion(),
            action_id="mj-action",
            bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
            blend_duration=0.0,
        )
        # Scheduling did not freeze a stale pose. The current measured pose is
        # changed immediately before the action actually starts.
        service._snapshot = {
            "sim_time": 7.0,
            "q": {JOINT: 0.0},
            "dq": {JOINT: 0.0},
            "target": {JOINT: 0.0},
        }

        first_target = service._targets_locked(7.0)
        self.assertEqual(first_target[JOINT], 0.0)
        record = service.get_action("mj-action")
        self.assertEqual(record["bridge_policy"], MINIMUM_JERK_BRIDGE_POLICY)
        self.assertEqual(record["bridge_duration"], 0.08)

        target_500hz = service._targets_locked(7.002)[JOINT]
        normalized_time = 0.002 / 0.08
        alpha = (
            10.0 * normalized_time**3
            - 15.0 * normalized_time**4
            + 6.0 * normalized_time**5
        )
        self.assertAlmostEqual(target_500hz, 0.04 * alpha, places=12)

        # The bridge ends exactly on clip.sample(0); the next controller tick
        # advances the Atom itself with no gap or duplicate bridge action.
        self.assertAlmostEqual(service._targets_locked(7.08)[JOINT], 0.04, places=12)
        self.assertAlmostEqual(service._targets_locked(7.082)[JOINT], 0.042, places=12)

    def test_legacy_cubic_blend_remains_the_default(self) -> None:
        service = MotionService(
            KinematicEngine(joints=(JOINT,), timestep=0.002),
            config=ServiceConfig(render_fps=0, default_blend_duration=0.2),
        )
        service.submit(motion(first=1.0), action_id="legacy-action")
        service._snapshot = {
            "sim_time": 3.0,
            "q": {JOINT: 0.0},
            "dq": {JOINT: 0.0},
            "target": {JOINT: 0.0},
        }
        service._targets_locked(3.0)
        self.assertAlmostEqual(service._targets_locked(3.1)[JOINT], 0.5)
        record = service.get_action("legacy-action")
        self.assertEqual(record["bridge_policy"], LEGACY_CUBIC_BRIDGE_POLICY)
        self.assertEqual(record["bridge_duration"], 0.2)

    def test_policy_values_and_zero_blend_contract_are_strict(self) -> None:
        service = MotionService(KinematicEngine(joints=(JOINT,)))
        with self.assertRaisesRegex(MotionValidationError, "bridge_policy"):
            service.submit(motion(), bridge_policy="minimum_jerk_v2", blend_duration=0.0)
        with self.assertRaisesRegex(MotionValidationError, "blend_duration=0"):
            service.submit(
                motion(),
                bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
                blend_duration=0.1,
            )

    def test_watchdog_budget_includes_actual_minimum_jerk_duration(self) -> None:
        service = MotionService(
            KinematicEngine(joints=(JOINT,), timestep=0.002),
            config=ServiceConfig(render_fps=0, action_overrun_grace=0.25),
        )
        service.submit(
            motion(first=1.0),
            bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
            blend_duration=0.0,
        )
        service._snapshot = {
            "sim_time": 0.0,
            "q": {JOINT: 0.0},
            "dq": {JOINT: 0.0},
            "target": {JOINT: 0.0},
        }
        service._targets_locked(0.0)
        action = service._current
        self.assertIsNotNone(action)
        self.assertAlmostEqual(
            service._action_deadline_seconds(action),
            action.bridge_duration + action.clip.duration / action.speed + 0.25,
        )

    def test_minimum_jerk_runs_through_the_physics_controller_step_path(self) -> None:
        class ControllerPathEngine(KinematicEngine):
            def __init__(self) -> None:
                super().__init__(joints=(JOINT,), timestep=0.002)
                self.commanded_targets = []
                self.controller_steps = 0

            def set_targets(self, targets) -> None:
                self.commanded_targets.append(float(targets[JOINT]))
                super().set_targets(targets)

            def step(self) -> None:
                self.controller_steps += 1
                super().step()

        engine = ControllerPathEngine()
        service = MotionService(engine, config=ServiceConfig(render_fps=0))
        service.start()
        try:
            accepted = service.submit(
                motion(),
                bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
                blend_duration=0.0,
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if service.get_action(accepted["id"])["status"] == "completed":
                    break
                time.sleep(0.005)
            self.assertEqual(service.get_action(accepted["id"])["status"], "completed")
        finally:
            service.stop()
        self.assertGreater(engine.controller_steps, 40)
        self.assertTrue(any(0.0 < value < 0.04 for value in engine.commanded_targets))

    def test_mujoco_controller_step_still_calls_mj_step(self) -> None:
        class FakeMujoco:
            def __init__(self) -> None:
                self.step_calls = 0
                self.forward_calls = 0

            def mj_step(self, model, data) -> None:
                self.step_calls += 1

            def mj_forward(self, model, data) -> None:
                self.forward_calls += 1

        engine = MujocoEngine.__new__(MujocoEngine)
        engine._owner_thread = None
        engine._free_roots = []
        engine._actuators = [
            {
                "actuator_id": 0,
                "qpos_adr": 0,
                "dof_adr": 0,
                "joint": JOINT,
            }
        ]
        engine.control_mode = "position"
        engine.model = SimpleNamespace(actuator_ctrllimited=[False])
        engine.data = SimpleNamespace(qpos=[0.0], qvel=[0.0], ctrl=[0.0])
        engine.target = {JOINT: 0.25}
        engine.mujoco = FakeMujoco()

        engine.step()

        self.assertEqual(engine.data.ctrl[0], 0.25)
        self.assertEqual(engine.mujoco.step_calls, 1)
        self.assertEqual(engine.mujoco.forward_calls, 1)

    def test_client_forwards_explicit_policy_and_zero_blend(self) -> None:
        client = _RecordingClient()
        client.trigger_sync(
            motion(),
            bridge_policy=MINIMUM_JERK_BRIDGE_POLICY,
            blend_duration=0.0,
        )
        self.assertEqual(client.payload["bridge_policy"], MINIMUM_JERK_BRIDGE_POLICY)
        self.assertEqual(client.payload["blend_duration"], 0.0)

    def test_http_forwards_policy_and_get_reports_actual_duration(self) -> None:
        service = MotionService(
            KinematicEngine(joints=(JOINT,), timestep=0.002),
            config=ServiceConfig(render_fps=0),
        )
        server = create_http_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        service.start()
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            body = json.dumps(
                {
                    "motion": motion(),
                    "bridge_policy": MINIMUM_JERK_BRIDGE_POLICY,
                    "blend_duration": 0,
                }
            ).encode()
            request = Request(
                base + "/actions/trigger",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request) as response:
                accepted = json.load(response)
            self.assertEqual(
                accepted["action"]["bridge_policy"], MINIMUM_JERK_BRIDGE_POLICY
            )
            action_url = base + "/actions/" + accepted["id"]
            deadline = time.monotonic() + 1.0
            record = None
            while time.monotonic() < deadline:
                with urlopen(action_url) as response:
                    record = json.load(response)
                if record["started_wall"] is not None:
                    break
                time.sleep(0.005)
            self.assertIsNotNone(record)
            self.assertEqual(record["bridge_duration"], 0.08)

            invalid = Request(
                base + "/actions/trigger",
                data=json.dumps(
                    {
                        "motion": motion(),
                        "bridge_policy": "unknown",
                        "blend_duration": 0,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as context:
                urlopen(invalid)
            self.assertEqual(context.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            service.stop()
            thread.join(1)


if __name__ == "__main__":
    unittest.main()
