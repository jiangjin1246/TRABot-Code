import time
import threading

import unittest

from sim_service.engine import KinematicEngine, SnapshotRenderer
from sim_service.motion import MotionValidationError
from sim_service.service import MotionConflictError, MotionService, ServiceConfig


def wait_until(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def motion(name="move", value=0.4, duration=0.08):
    return {
        "name": name,
        "joints": ["right_elbow_joint"],
        "frames": [
            {"t": 0.0, "q": [0.0]},
            {"t": duration, "q": [value]},
        ],
    }


class ServiceTests(unittest.TestCase):
    def test_action_lifecycle_has_queryable_timestamps(self) -> None:
        service = MotionService(
            KinematicEngine(timestep=0.002),
            config=ServiceConfig(
                render_fps=0,
                default_blend_duration=0.0,
            ),
        )
        accepted = service.submit(
            motion(duration=0.25),
            action_id="lifecycle-action",
        )
        self.assertEqual(accepted["status"], "scheduled")
        self.assertEqual(accepted["action"]["status"], "scheduled")
        self.assertIsNone(accepted["action"]["started_wall"])
        self.assertIsNone(accepted["action"]["completed_wall"])
        self.assertEqual(service.state()["queued"][0]["status"], "scheduled")

        service.start()
        try:
            self.assertTrue(
                wait_until(
                    lambda: (
                        service.get_action("lifecycle-action") or {}
                    ).get("status")
                    == "started"
                )
            )
            started = service.get_action("lifecycle-action")
            self.assertIsNotNone(started)
            self.assertEqual(started["status"], "started")
            self.assertIsNotNone(started["started_wall"])
            self.assertIsNotNone(started["started_sim"])
            self.assertLessEqual(started["scheduled_wall"], started["started_wall"])

            self.assertTrue(
                wait_until(
                    lambda: (
                        service.get_action("lifecycle-action") or {}
                    ).get("status")
                    == "completed"
                )
            )
            completed = service.get_action("lifecycle-action")
            self.assertIsNotNone(completed)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["completed_wall"], completed["finished_wall"])
            self.assertEqual(completed["completed_sim"], completed["finished_sim"])
            self.assertLessEqual(completed["started_wall"], completed["completed_wall"])
            self.assertLessEqual(completed["started_sim"], completed["completed_sim"])
            self.assertEqual(
                service.action_state()["last_completed"]["id"],
                "lifecycle-action",
            )
            self.assertIsNone(service.get_action("missing-action"))
        finally:
            service.stop()

    def test_service_plays_motion_and_updates_state(self) -> None:
        engine = KinematicEngine(timestep=0.002)
        service = MotionService(
        engine,
        config=ServiceConfig(render_fps=30, default_blend_duration=0.01),
    )
        service.start()
        try:
            result = service.submit(motion())
            self.assertTrue(result["accepted"])
            self.assertTrue(wait_until(lambda: service.action_state()["last_completed"] is not None))
            state = service.state()
            self.assertEqual(state["last_completed"]["status"], "completed")
            self.assertAlmostEqual(state["target"]["right_elbow_joint"], 0.4)
            self.assertTrue(service.health()["ok"])
            frame, sequence = service.frame()
            self.assertTrue(frame.startswith(b"\x89PNG"))
            self.assertGreater(sequence, 0)
            self.assertIsNotNone(service.health()["frame_age_s"])
        finally:
            service.stop()


    def test_queue_capacity_replace_and_cancel(self) -> None:
        service = MotionService(
        KinematicEngine(timestep=0.005),
        config=ServiceConfig(queue_capacity=1, default_blend_duration=0.0),
    )
        service.start()
        try:
            first = service.submit(motion(duration=0.5), action_id="first")
            self.assertTrue(wait_until(lambda: service.action_state()["current"] is not None))
            queued = service.submit(
                motion("queued", 0.2, 0.5),
                action_id="queued",
            )
            with self.assertRaisesRegex(MotionValidationError, "queue is full"):
                service.submit(motion("overflow", 0.1, 0.5))
            replacement = service.submit(
                motion("replacement", 0.1, 0.5),
                mode="replace",
                action_id="replacement",
            )
            self.assertTrue(replacement["accepted"])
            self.assertEqual(service.get_action(first["id"])["status"], "replaced")
            self.assertEqual(service.get_action(queued["id"])["status"], "replaced")
            canceled = service.cancel()
            self.assertGreaterEqual(canceled["canceled"], 1)
            self.assertEqual(
                service.get_action(replacement["id"])["status"],
                "canceled",
            )
            self.assertIsNone(service.action_state()["current"])
            self.assertEqual(service.action_state()["queued"], [])
        finally:
            service.stop()

    def test_duplicate_action_id_is_rejected(self) -> None:
        service = MotionService(KinematicEngine())
        service.submit(motion(), action_id="stable-id")
        with self.assertRaisesRegex(MotionValidationError, "already exists"):
            service.submit(motion(), action_id="stable-id")

    def test_atomic_require_empty_and_action_seq_prevent_queue_race(self) -> None:
        service = MotionService(KinematicEngine())
        initial = service.action_state()["action_seq"]
        first = service.submit(
            motion("semantic-first", duration=0.4),
            action_id="semantic-first",
            require_empty=True,
            expected_action_seq=initial,
            exclusive_owner="semantic:session-1",
            payload_sha256="a" * 64,
        )

        self.assertGreater(first["action_seq"], initial)
        self.assertEqual(first["action"]["exclusive_owner"], "semantic:session-1")
        self.assertEqual(first["action"]["payload_sha256"], "a" * 64)
        with self.assertRaisesRegex(MotionConflictError, "expected_action_seq"):
            service.submit(
                motion("stale"),
                expected_action_seq=initial,
                require_empty=True,
                exclusive_owner="semantic:session-1",
            )
        with self.assertRaisesRegex(MotionConflictError, "reserved"):
            service.submit(motion("foreign"))
        with self.assertRaisesRegex(MotionConflictError, "reserved"):
            service.cancel()
        with self.assertRaisesRegex(MotionConflictError, "require_empty"):
            service.submit(
                motion("same-owner-queued"),
                require_empty=True,
                exclusive_owner="semantic:session-1",
            )
        self.assertEqual(len(service.action_state()["queued"]), 1)
        canceled = service.cancel(exclusive_owner="semantic:session-1")
        self.assertEqual(canceled["canceled"], 1)

    def test_exclusive_owner_releases_after_own_action_completes(self) -> None:
        service = MotionService(
            KinematicEngine(timestep=0.002),
            config=ServiceConfig(render_fps=0, default_blend_duration=0.0),
        )
        service.submit(
            motion("owned", duration=0.05),
            action_id="owned",
            require_empty=True,
            expected_action_seq=service.action_state()["action_seq"],
            exclusive_owner="semantic:session-1",
        )
        service.start()
        try:
            self.assertTrue(
                wait_until(
                    lambda: (service.get_action("owned") or {}).get("status")
                    == "completed"
                )
            )
            state = service.action_state()
            accepted = service.submit(
                motion("legacy-after"),
                require_empty=True,
                expected_action_seq=state["action_seq"],
            )
            self.assertTrue(accepted["accepted"])
        finally:
            service.stop()

    def test_watchdog_deadline_cancels_owned_action_without_owner_gate_crash(self) -> None:
        service = MotionService(
            KinematicEngine(timestep=0.002),
            config=ServiceConfig(
                render_fps=0,
                default_blend_duration=0.0,
                loop_stall_timeout=0.2,
                action_overrun_grace=0.01,
            ),
        )
        service.submit(
            motion("owned-timeout", duration=5.0),
            action_id="owned-timeout",
            require_empty=True,
            expected_action_seq=service.action_state()["action_seq"],
            exclusive_owner="semantic:deadline-owner",
        )
        service.start()
        try:
            self.assertTrue(
                wait_until(
                    lambda: (service.get_action("owned-timeout") or {}).get("status")
                    == "started"
                )
            )
            with service._lock:
                self.assertIsNotNone(service._current)
                service._current.started_monotonic = time.monotonic() - 100.0
            self.assertTrue(
                wait_until(
                    lambda: (service.get_action("owned-timeout") or {}).get("status")
                    == "canceled"
                )
            )
            record = service.get_action("owned-timeout")
            self.assertEqual(record["reason"], "action_deadline_exceeded")
            self.assertIsNone(service.action_state()["current"])
            self.assertEqual(service.action_state()["queued"], [])
            self.assertTrue(service._watchdog_thread.is_alive())
            self.assertTrue(service.running)
            # The external owner gate remains intact for client-triggered cancel.
            service.submit(
                motion("still-owned", duration=0.5),
                action_id="still-owned",
                exclusive_owner="semantic:deadline-owner",
            )
            with self.assertRaisesRegex(MotionConflictError, "reserved"):
                service.cancel()
            service.cancel(exclusive_owner="semantic:deadline-owner")
        finally:
            service.stop()


    def test_unknown_joint_and_bad_speed_are_rejected(self) -> None:
        service = MotionService(KinematicEngine())
        with self.assertRaisesRegex(MotionValidationError, "no controllable joint"):
            service.submit(
            {
                "joints": ["missing"],
                "frames": [{"t": 0, "q": [0]}, {"t": 1, "q": [1]}],
            }
        )
        with self.assertRaisesRegex(MotionValidationError, "speed"):
            service.submit(motion(), speed=100)

    def test_slow_renderer_does_not_throttle_physics(self) -> None:
        class SlowRenderer(SnapshotRenderer):
            def __init__(self) -> None:
                self.owner = threading.get_ident()
                self.render_count = 0
                self.closed_on = None

            def render_png(self, snapshot) -> bytes:
                self.assert_owner()
                self.render_count += 1
                time.sleep(0.08)
                return b"\x89PNG-slow"

            def close(self) -> None:
                self.assert_owner()
                self.closed_on = threading.get_ident()

            def assert_owner(self) -> None:
                if threading.get_ident() != self.owner:
                    raise RuntimeError("renderer changed threads")

        class SlowRenderEngine(KinematicEngine):
            def __init__(self) -> None:
                super().__init__(timestep=0.002)
                self.renderer = None
                self.physics_thread_id = None

            def step(self) -> None:
                self.physics_thread_id = self.physics_thread_id or threading.get_ident()
                super().step()

            def create_renderer(self) -> SnapshotRenderer:
                self.renderer = SlowRenderer()
                return self.renderer

        engine = SlowRenderEngine()
        service = MotionService(engine, config=ServiceConfig(render_fps=100, metrics_window=1.0))
        service.start()
        try:
            time.sleep(0.38)
            health = service.health()
            # The old coupled design produced ~5 steps here; decoupled physics
            # remains close to 500 Hz even though each render consumes 80 ms.
            self.assertGreater(health["state_seq"], 100)
            self.assertGreater(health["step_hz"], 250)
            self.assertGreater(health["sim_real_time_factor"], 0.5)
            self.assertLess(health["frame_fps"], 30)
            self.assertNotEqual(engine.physics_thread_id, engine.renderer.owner)
        finally:
            service.stop()
        self.assertEqual(engine.renderer.owner, engine.renderer.closed_on)
