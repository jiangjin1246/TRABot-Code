import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sim_service.engine import KinematicEngine
from sim_service.http_api import create_http_server
from sim_service.service import MotionService, ServiceConfig


class HttpApiTests(unittest.TestCase):
    def test_http_health_action_state_frame_and_release(self) -> None:
        service = MotionService(
            KinematicEngine(timestep=0.002),
            config=ServiceConfig(default_blend_duration=0.0),
        )
        server = create_http_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        service.start()
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/health") as response:
                self.assertTrue(json.load(response)["ok"])

            body = json.dumps(
                {
                    "motion": {
                        "name": "http-motion",
                        "joints": ["right_elbow_joint"],
                        "frames": [
                            {"t": 0, "q": {"right_elbow_joint": 0}},
                            {"t": 0.05, "q": {"right_elbow_joint": 0.2}},
                        ],
                    }
                }
            ).encode()
            request = Request(
                base + "/actions/trigger",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request) as response:
                accepted = json.load(response)
                self.assertEqual(response.status, 202)
                self.assertTrue(accepted["accepted"])
                self.assertEqual(accepted["status"], "scheduled")
                self.assertEqual(accepted["action"]["status"], "scheduled")
                self.assertIsNotNone(accepted["scheduled_wall"])

            action_url = base + "/actions/" + accepted["id"]
            deadline = time.monotonic() + 1.5
            lifecycle = None
            while time.monotonic() < deadline:
                with urlopen(action_url) as response:
                    lifecycle = json.load(response)
                if lifecycle["status"] == "completed":
                    break
                time.sleep(0.01)
            self.assertIsNotNone(lifecycle)
            self.assertEqual(lifecycle["status"], "completed")
            self.assertLessEqual(
                lifecycle["scheduled_wall"],
                lifecycle["started_wall"],
            )
            self.assertLessEqual(
                lifecycle["started_wall"],
                lifecycle["completed_wall"],
            )

            with urlopen(base + "/action") as response:
                actions = json.load(response)
                self.assertEqual(actions["last_completed"]["id"], accepted["id"])

            with self.assertRaises(HTTPError) as context:
                urlopen(base + "/actions/not-present")
            self.assertEqual(context.exception.code, 404)

            with urlopen(base + "/status") as response:
                self.assertIn("q", json.load(response))
            with urlopen(base + "/frame") as response:
                self.assertEqual(response.headers["Content-Type"], "image/png")
                self.assertTrue(response.read().startswith(b"\x89PNG"))

            release = Request(
                base + "/action",
                data=b'{"release":true}',
                headers={"Content-Type": "application/json"},
            )
            with urlopen(release) as response:
                self.assertEqual(json.load(response)["mode"], "hold")

            bad = Request(
                base + "/action",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as context:
                urlopen(bad)
            self.assertEqual(context.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            service.stop()
            thread.join(1)
