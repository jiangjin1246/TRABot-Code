from __future__ import annotations

import json
from io import BytesIO
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from web_overlay.config import OverlayConfig
from web_overlay.sim_client import SimServiceClient, SimServiceError


class _RecordingClient(SimServiceClient):
    def __init__(self, config):
        super().__init__(config)
        self.requests = []

    def _request(self, method, path, payload=None):
        self.requests.append((method, path, payload))
        if path == "/health":
            return json.dumps({"ok": True}).encode(), "application/json"
        if path == "/state":
            return (
                json.dumps({"mode": "idle", "sim_time": 1.25}).encode(),
                "application/json",
            )
        if path == "/frame":
            return b"fake-png", "image/png"
        if method == "DELETE":
            return json.dumps({"stopped": True}).encode(), "application/json"
        return (
            json.dumps(
                {"accepted": True, "id": "test-1", "queue_depth": 1}
            ).encode(),
            "application/json",
        )


class _CompletingClient(_RecordingClient):
    def __init__(self, config):
        super().__init__(config)
        self.action_reads = 0

    def _request(self, method, path, payload=None):
        if path != "/actions/test-1":
            return super()._request(method, path, payload)
        self.requests.append((method, path, payload))
        self.action_reads += 1
        if self.action_reads == 1:
            state = {"id": "test-1", "name": "wave", "status": "started"}
        else:
            state = {"id": "test-1", "name": "wave", "status": "completed"}
        return json.dumps(state).encode(), "application/json"


class SimClientTest(unittest.TestCase):
    def setUp(self):
        self.client = _RecordingClient(OverlayConfig())

    def test_health_state_and_frame_cache(self):
        self.assertTrue(self.client.health_sync()["ok"])
        self.assertEqual(self.client.state_sync()["mode"], "idle")
        frame = self.client.fetch_frame_sync()
        self.assertEqual(frame.content_type, "image/png")
        self.assertEqual(frame.body, b"fake-png")
        self.assertTrue(self.client.cache_status()["frame_fresh"])

    def test_named_action_is_translated_to_sim_payload(self):
        result = self.client.trigger_sync("wave", mode="replace", speed=0.8)
        self.assertTrue(result["accepted"])
        _, path, payload = self.client.requests[-1]
        self.assertEqual(path, "/actions/trigger")
        self.assertEqual(payload["action"], "wave")
        self.assertEqual(payload["mode"], "replace")
        self.assertEqual(payload["speed"], 0.8)

    def test_full_motion_and_release_are_supported(self):
        motion = {"joints": ["j0"], "frames": [{"t": 0, "q": {"j0": 0}}]}
        self.client.trigger_sync(motion)
        self.assertEqual(self.client.requests[-1][2]["motion"], motion)
        self.client.release_sync()
        self.assertEqual(self.client.requests[-1][2], {"release": True})

    def test_legacy_trigger_waits_for_completion(self):
        client = _CompletingClient(OverlayConfig())
        result = client.trigger_legacy_sync("wave")
        self.assertEqual(result["completion"]["status"], "completed")
        self.assertEqual(client.action_reads, 2)

    def test_http_error_includes_server_detail(self):
        error = HTTPError(
            "http://127.0.0.1:8765/actions/trigger",
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":"invalid_request","detail":"action queue is full"}'),
        )
        with patch("web_overlay.sim_client.urlopen", side_effect=error):
            with self.assertRaisesRegex(SimServiceError, "action queue is full"):
                SimServiceClient(OverlayConfig()).trigger_sync("wave")


if __name__ == "__main__":
    unittest.main()
