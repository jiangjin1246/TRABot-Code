from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from web_overlay.g1_session_client import (
    G1SessionClient,
    G1SessionConfig,
    G1SessionConflictError,
    G1SessionError,
    STATUS_SCHEMA_VERSION,
)


ENTRY_ID = "a" * 64
ATOM_ID = "beat.test.s0.1.f1-2.deadbeef"
VERSION = "f" * 32
CONTENT_SHA256 = "b" * 64
PREVIEW_SHA256 = "c" * 64
NEUTRAL_SOURCE_SHA256 = "e" * 64
Q0 = {"left_shoulder_pitch": 0.0, "right_shoulder_pitch": 0.0}
Q1 = {"left_shoulder_pitch": 0.1, "right_shoulder_pitch": -0.1}


def _canonical_sha256(value):
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


class _QueueOutput:
    def __init__(self):
        self.items = queue.Queue()

    def push(self, value):
        self.items.put(value)

    def close(self):
        self.items.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.items.get(timeout=2.0)
        if value is None:
            raise StopIteration
        return value


class _FakeStdin:
    def __init__(self, process):
        self.process = process
        self.buffer = ""

    def write(self, value):
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.process.on_command(json.loads(line))
        return len(value)

    def flush(self):
        return None


class _FakeProcess:
    _next_pid = 80000

    def __init__(self, command, *, behavior="normal", auto_ready=True):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.command = list(command)
        self.behavior = behavior
        self.stdout = _QueueOutput()
        self.stdin = _FakeStdin(self)
        self.returncode = None
        self.done = threading.Event()
        self.signals = []
        self.commands = []
        self.action_seq = 0
        self.session_id = next(
            item.split("=", 1)[1]
            for item in command
            if item.startswith("--session-id=")
        )
        if auto_ready:
            self.emit(
                "session_ready",
                session_id=self.session_id,
                action_seq=0,
                measured_q=Q0,
            )

    def emit(self, event, **fields):
        self.stdout.push(
            json.dumps(
                {
                    "schema_version": STATUS_SCHEMA_VERSION,
                    "event": event,
                    **fields,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    def on_command(self, command):
        self.commands.append(command)
        if command["type"] == "snapshot":
            if self.behavior != "snapshot_timeout":
                self.emit(
                    "state_snapshot",
                    session_id=self.session_id,
                    request_id=command["request_id"],
                    action_seq=(
                        self.action_seq + 1
                        if self.behavior == "snapshot_wrong_seq"
                        else self.action_seq
                    ),
                    measured_q=Q0 if self.action_seq == 0 else Q1,
                )
            return
        if self.behavior == "native_error":
            self.emit("error", code="session_protocol_error", message="bad command")
            self._exit(5)
            return
        self.action_seq += 1
        common = {
            "session_id": self.session_id,
            "action_id": command["action_id"],
            "action_seq": self.action_seq,
            "payload_sha256": command["payload_sha256"],
        }
        if command["type"] == "append":
            common["entry_id"] = command["entry_id"]
            kind = "semantic"
        else:
            kind = "recovery"
        self.emit("action_accepted", **common)
        self.action_seq += 1
        common["action_seq"] = self.action_seq
        self.emit("action_started", **common)
        self.action_seq += 1
        common["action_seq"] = self.action_seq
        self.emit("action_completed", kind=kind, measured_q=Q1, **common)
        if command["type"] == "end_turn":
            if self.behavior == "delayed_session_completed":
                threading.Thread(
                    target=self._complete_session_after_delay,
                    daemon=True,
                ).start()
            else:
                self.emit("completed", session_id=self.session_id)
                self._exit(0)

    def _complete_session_after_delay(self):
        time.sleep(0.2)
        self.emit("completed", session_id=self.session_id)
        self._exit(0)

    def _exit(self, returncode):
        self.returncode = returncode
        self.done.set()
        self.stdout.close()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self._exit(130 if sig == signal.SIGINT else 143)

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.signals.append(signal.SIGKILL)
        self._exit(137)


class _Factory:
    def __init__(self, *, behavior="normal", auto_ready=True):
        self.behavior = behavior
        self.auto_ready = auto_ready
        self.calls = []
        self.processes = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        process = _FakeProcess(
            command,
            behavior=self.behavior,
            auto_ready=self.auto_ready,
        )
        self.processes.append(process)
        return process


class G1SessionClientTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = self.root / "runner"
        self.runner.write_text("#!/bin/sh\n", encoding="utf-8")
        self.runner.chmod(0o755)
        self.limits = self.root / "limits.json"
        self.limits.write_text("{}\n", encoding="utf-8")
        self.motion = {
            "format": "g1_joint_motion_v1",
            "duration": 0.1,
            "joints": list(Q0),
            "interpolation": "linear",
            "frames": [{"t": 0.0, "q": Q0}, {"t": 0.1, "q": Q1}],
            "source_atom_hash": CONTENT_SHA256,
        }
        neutral_motion = {
            **self.motion,
            "frames": [{"t": 0.0, "q": Q0}, {"t": 0.1, "q": Q0}],
        }
        catalog_payload = {
            "schema_version": "semantic_atom_catalog_v1",
            "inputs": {"neutral_motion_sha256": NEUTRAL_SOURCE_SHA256},
            "neutral": {
                "motion_sha256": NEUTRAL_SOURCE_SHA256,
                "motion": neutral_motion,
            },
            "entries": [
                {
                    "entry_id": ENTRY_ID,
                    "atom_id": ATOM_ID,
                    "version": VERSION,
                    "revision": 1,
                    "content_sha256": CONTENT_SHA256,
                    "approved": True,
                    "removed": False,
                    "motion": self.motion,
                }
            ],
        }
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(
            json.dumps(catalog_payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        self.catalog_sha = hashlib.sha256(self.catalog.read_bytes()).hexdigest()
        self.embedded_neutral_motion_sha = _canonical_sha256(neutral_motion)
        self.neutral_sha = NEUTRAL_SOURCE_SHA256
        self.review_store = self.root / "reviews.json"
        self.review_store.write_text(
            json.dumps(
                {
                    "schema": "g1_atom_visual_reviews_v1",
                    "reviews": {
                        ENTRY_ID: {
                            "status": "approved",
                            "catalog_file_sha256": self.catalog_sha,
                            "entry_id": ENTRY_ID,
                            "atom_id": ATOM_ID,
                            "version": VERSION,
                            "revision": 1,
                            "content_sha256": CONTENT_SHA256,
                            "motion_sha256": _canonical_sha256(self.motion),
                            "preview_clip_sha256": PREVIEW_SHA256,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def client(
        self,
        factory=None,
        *,
        review=True,
        startup_timeout=1.0,
        action_timeout=1.0,
    ):
        factory = factory or _Factory()
        config = G1SessionConfig(
            runner=self.runner,
            catalog=self.catalog,
            limits=self.limits,
            network="eno2",
            review_store=self.review_store if review else None,
            startup_timeout_s=startup_timeout,
            action_timeout_s=action_timeout,
            idle_timeout_s=15.0,
        )
        return G1SessionClient(config, popen_factory=factory), factory

    @staticmethod
    def request(request_id="req-1", revision=1):
        return {
            "request_id": request_id,
            "revision": revision,
            "fingerprint_sha256": "d" * 64,
        }

    def atom_ref(self):
        return {
            "entry_id": ENTRY_ID,
            "atom_id": ATOM_ID,
            "version": VERSION,
            "revision": 1,
            "content_sha256": CONTENT_SHA256,
            "catalog_content_sha256": self.catalog_sha,
        }

    def recovery_ref(self):
        return {
            "target": "catalog_canonical_neutral",
            "catalog_content_sha256": self.catalog_sha,
            "neutral_motion_sha256": self.neutral_sha,
            "after": self.atom_ref(),
            "tail_guard_us": 250000,
        }

    @staticmethod
    def flat(action_id, action_seq):
        payload = {
            "motion": {
                "format": "g1_joint_motion_v1",
                "frames": [{"t": 0.0, "q": {"never_forward": 123.0}}],
            },
            "speed": 1.0,
            "blend_duration": 0.0,
            "bridge_policy": "minimum_jerk_v1",
            "id": action_id,
            "mode": "enqueue",
            "require_empty": True,
            "expected_action_seq": action_seq,
            "exclusive_owner": "owner-1",
        }
        payload["payload_sha256"] = _canonical_sha256(payload)
        return payload

    def test_health_peek_and_first_state_are_spawn_safe(self):
        client, factory = self.client()
        self.assertEqual(client.health_sync()["approved_count"], 1)
        self.assertTrue(client.health_sync()["runner_ready"])
        self.assertEqual(client.peek_state_sync()["q"], {})
        self.assertEqual(factory.calls, [])
        client.begin_semantic_request(self.request())
        self.assertEqual(factory.calls, [])

        state = client.state_sync()
        self.assertEqual(state["current"], None)
        self.assertEqual(state["queued"], [])
        self.assertEqual(state["fault"], None)
        self.assertEqual(state["action_seq"], 0)
        self.assertEqual(state["q"], Q0)
        command = factory.calls[0][0]
        self.assertIn("--realtime-session", command)
        self.assertIn("--network=eno2", command)
        self.assertIn("--session-idle-timeout=15", command)
        snapshot = factory.processes[0].commands[0]
        self.assertEqual(
            set(snapshot),
            {"type", "session_id", "request_id", "expected_action_seq"},
        )
        self.assertEqual(snapshot["type"], "snapshot")
        self.assertEqual(snapshot["expected_action_seq"], 0)
        self.assertEqual(client.robot_runtime()["simulationMode"], False)
        client.finish_semantic_request(None)
        self.assertIn(signal.SIGINT, factory.processes[0].signals)

    def test_append_and_recovery_forward_only_exact_identity(self):
        client, factory = self.client()
        client.begin_semantic_request(self.request())
        state = client.state_sync()

        semantic_flat = self.flat("sem-action-1", state["action_seq"])
        response = client.post_semantic_action_sync(
            semantic_flat,
            kind="semantic",
            atom_ref=self.atom_ref(),
        )
        self.assertTrue(response["accepted"])
        self.assertEqual(client.action_sync("sem-action-1")["status"], "completed")
        append = factory.processes[0].commands[1]
        self.assertEqual(set(append), {
            "type", "session_id", "action_id", "entry_id", "expect_version",
            "expect_revision", "expect_content_sha256", "expected_action_seq",
            "payload_sha256",
        })
        self.assertNotIn("motion", append)
        self.assertNotIn("never_forward", json.dumps(append))

        state = client.state_sync()
        recovery_flat = self.flat("sem-recovery-1", state["action_seq"])
        recovery = client.post_semantic_action_sync(
            recovery_flat,
            kind="recovery",
            atom_ref=self.recovery_ref(),
        )
        self.assertTrue(recovery["accepted"])
        self.assertEqual(client.action_sync("sem-recovery-1")["status"], "completed")
        end_turn = factory.processes[0].commands[3]
        self.assertEqual(set(end_turn), {
            "type", "session_id", "action_id", "expected_action_seq", "payload_sha256",
        })
        self.assertEqual(end_turn["type"], "end_turn")
        client.finish_semantic_request({"closed_neutral": True})
        self.assertNotIn(signal.SIGINT, factory.processes[0].signals)

    def test_compare_and_submit_rejects_stale_seq_without_writing(self):
        client, factory = self.client()
        client.begin_semantic_request(self.request())
        state = client.state_sync()
        flat = self.flat("stale-action", state["action_seq"] + 1)
        with self.assertRaises(G1SessionConflictError) as caught:
            client.post_semantic_action_sync(flat, "semantic", self.atom_ref())
        self.assertEqual(caught.exception.status_code, 409)
        self.assertFalse(
            any(command["type"] == "append" for command in factory.processes[0].commands)
        )
        client.finish_semantic_request(None)

    def test_request_and_thread_binding_reject_concurrency(self):
        client, _ = self.client()
        client.begin_semantic_request(self.request())
        with self.assertRaises(G1SessionConflictError):
            client.begin_semantic_request(self.request("req-2"))

        errors = []

        def foreign_state():
            try:
                client.state_sync()
            except Exception as exc:  # asserted below
                errors.append(exc)

        thread = threading.Thread(target=foreign_state)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], G1SessionConflictError)
        client.finish_semantic_request(None)

    def test_interrupt_and_exception_finish_release_the_process(self):
        client, factory = self.client()
        client.begin_semantic_request(self.request())
        client.state_sync()
        result = []
        thread = threading.Thread(
            target=lambda: result.append(client.interrupt_active_session_sync("barge-in"))
        )
        thread.start()
        thread.join()
        self.assertEqual(result, [True])
        self.assertIn(signal.SIGINT, factory.processes[0].signals)
        client.finish_semantic_request(None)

        client2, factory2 = self.client()
        client2.begin_semantic_request(self.request("req-error"))
        client2.state_sync()
        client2.finish_semantic_request(None, error=RuntimeError("planner failed"))
        self.assertIn(signal.SIGINT, factory2.processes[0].signals)

    def test_native_error_fails_action_and_is_cleaned_up(self):
        factory = _Factory(behavior="native_error")
        client, _ = self.client(factory)
        client.begin_semantic_request(self.request())
        state = client.state_sync()
        flat = self.flat("bad-native-action", state["action_seq"])
        with self.assertRaisesRegex(G1SessionError, "session_protocol_error"):
            client.post_semantic_action_sync(flat, "semantic", self.atom_ref())
        self.assertEqual(client.action_sync("bad-native-action")["status"], "failed")
        client.finish_semantic_request(None, error=RuntimeError("native failed"))

    def test_recovery_completion_race_waits_for_native_completed_without_sigint(self):
        factory = _Factory(behavior="delayed_session_completed")
        client, _ = self.client(factory)
        client.begin_semantic_request(self.request("recovery-race"))
        state = client.state_sync()
        semantic = self.flat("race-semantic", state["action_seq"])
        client.post_semantic_action_sync(semantic, "semantic", self.atom_ref())
        deadline = time.monotonic() + 1.0
        while client.action_sync("race-semantic")["status"] != "completed":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        state = client.state_sync()
        recovery = self.flat("race-recovery", state["action_seq"])
        client.post_semantic_action_sync(recovery, "recovery", self.recovery_ref())
        while client.action_sync("race-recovery")["status"] != "completed":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        process = factory.processes[0]
        self.assertIsNone(process.poll())

        client.finish_semantic_request({"closed_neutral": True})

        self.assertEqual(process.poll(), 0)
        self.assertNotIn(signal.SIGINT, process.signals)

    def test_review_store_must_match_current_catalog_identity(self):
        payload = json.loads(self.review_store.read_text(encoding="utf-8"))
        payload["reviews"][ENTRY_ID]["content_sha256"] = "0" * 64
        self.review_store.write_text(json.dumps(payload), encoding="utf-8")
        client, _ = self.client()
        self.assertEqual(client.approved_entry_ids(), ())
        self.assertFalse(client.health_sync()["runner_ready"])

    def test_recovery_uses_declared_source_neutral_hash_not_inline_motion_hash(self):
        self.assertNotEqual(self.neutral_sha, self.embedded_neutral_motion_sha)
        client, _ = self.client()
        client.begin_semantic_request(self.request("neutral-hash"))
        state = client.state_sync()
        semantic = self.flat("neutral-hash-semantic", state["action_seq"])
        client.post_semantic_action_sync(semantic, "semantic", self.atom_ref())
        deadline = time.monotonic() + 1.0
        while client.action_sync("neutral-hash-semantic")["status"] != "completed":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        state = client.state_sync()
        recovery = self.flat("neutral-hash-recovery", state["action_seq"])

        wrong_ref = self.recovery_ref()
        wrong_ref["neutral_motion_sha256"] = self.embedded_neutral_motion_sha
        with self.assertRaisesRegex(G1SessionError, "neutral motion hash mismatch"):
            client.post_semantic_action_sync(recovery, "recovery", wrong_ref)

        response = client.post_semantic_action_sync(
            recovery,
            "recovery",
            self.recovery_ref(),
        )
        self.assertTrue(response["accepted"])
        client.finish_semantic_request({"closed_neutral": True})

    def test_startup_timeout_sends_fail_safe_interrupt(self):
        factory = _Factory(auto_ready=False)
        client, _ = self.client(factory, startup_timeout=0.1)
        client.begin_semantic_request(self.request())
        with self.assertRaisesRegex(G1SessionError, "session_ready"):
            client.state_sync()
        self.assertIn(signal.SIGINT, factory.processes[0].signals)

    def test_idle_state_reads_use_monotonic_snapshot_ids_and_timeout_closed(self):
        client, factory = self.client()
        client.begin_semantic_request(self.request())
        first = client.state_sync()
        second = client.state_sync()
        self.assertEqual(first["q"], Q0)
        self.assertEqual(second["q"], Q0)
        snapshots = [
            command
            for command in factory.processes[0].commands
            if command["type"] == "snapshot"
        ]
        self.assertEqual(len(snapshots), 2)
        self.assertNotEqual(snapshots[0]["request_id"], snapshots[1]["request_id"])
        self.assertLess(
            int(snapshots[0]["request_id"].split("-")[-1]),
            int(snapshots[1]["request_id"].split("-")[-1]),
        )
        client.finish_semantic_request(None)

        timeout_factory = _Factory(behavior="snapshot_timeout")
        timeout_client, _ = self.client(
            timeout_factory,
            action_timeout=0.1,
        )
        timeout_client.begin_semantic_request(self.request("snapshot-timeout"))
        with self.assertRaisesRegex(G1SessionError, "state_snapshot"):
            timeout_client.state_sync()
        self.assertIn(signal.SIGINT, timeout_factory.processes[0].signals)

        bad_factory = _Factory(behavior="snapshot_wrong_seq")
        bad_client, _ = self.client(bad_factory)
        bad_client.begin_semantic_request(self.request("snapshot-wrong-seq"))
        with self.assertRaisesRegex(G1SessionError, "action_seq"):
            bad_client.state_sync()
        self.assertIn(signal.SIGINT, bad_factory.processes[0].signals)


if __name__ == "__main__":
    unittest.main()
