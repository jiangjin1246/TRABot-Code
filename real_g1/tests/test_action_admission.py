from __future__ import annotations

import json
import hashlib
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sim_service.engine import KinematicEngine
from sim_service.http_api import create_http_server
from sim_service.service import MotionService, ServiceConfig


def motion(name: str) -> dict:
    return {
        "name": name,
        "joints": ["right_elbow_joint"],
        "frames": [
            {"t": 0.0, "q": [0.0]},
            {"t": 0.5, "q": [0.2]},
        ],
        "interpolation": "linear",
    }


class AtomicAdmissionHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = MotionService(
            KinematicEngine(timestep=0.002),
            config=ServiceConfig(render_fps=0, default_blend_duration=0.0),
        )
        self.server = create_http_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.service.stop()
        self.thread.join(1)

    def post(self, payload: dict) -> tuple[int, dict]:
        request = Request(
            self.base + "/actions/trigger",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def test_http_forwards_atomic_preconditions_and_returns_409(self) -> None:
        with urlopen(self.base + "/actions") as response:
            action_seq = json.load(response)["action_seq"]
        first_payload = {
                "motion": motion("owned"),
                "id": "semantic-owned",
                "require_empty": True,
                "expected_action_seq": action_seq,
                "exclusive_owner": "semantic:session-1",
        }
        first_payload["payload_sha256"] = hashlib.sha256(
            json.dumps(
                first_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        status, accepted = self.post(first_payload)
        self.assertEqual(status, 202)
        self.assertEqual(
            accepted["action"]["payload_sha256"], first_payload["payload_sha256"]
        )

        status, mismatch = self.post(
            {
                "motion": motion("bad-hash"),
                "payload_sha256": "0" * 64,
            }
        )
        self.assertEqual(status, 400)
        self.assertIn("payload_sha256", mismatch["detail"])

        status, conflict = self.post(
            {
                "motion": motion("foreign"),
                "require_empty": True,
                "expected_action_seq": action_seq,
            }
        )
        self.assertEqual(status, 409)
        self.assertEqual(conflict["error"], "action_conflict")
        self.assertEqual(len(self.service.action_state()["queued"]), 1)


if __name__ == "__main__":
    unittest.main()
