from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import patch

from web_overlay.config import OverlayConfig
from web_overlay.dialogue_sequence import (
    DialogueSequenceCoordinator,
    SequenceSettings,
    atom_to_inline_motion,
    patch_random_dialogue_bridge_class,
)
from web_overlay.patching import install_runtime_patches


def _control(name: str) -> dict:
    return {
        "joints": ["left_elbow", "right_elbow"],
        "trajectory": [
            {
                "t": 0.0,
                "q": {"left_elbow": 0.1, "right_elbow": -0.1},
                "dq": {"left_elbow": 0.0, "right_elbow": 0.0},
            },
            {
                "t": 0.08,
                "q": {"left_elbow": 0.2, "right_elbow": -0.2},
                "dq": {"left_elbow": 0.0, "right_elbow": 0.0},
            },
        ],
        "interpolation": "linear",
        "source": name,
    }


@dataclass(frozen=True)
class _Atom:
    atom_id: str
    version: str
    revision: int
    content_sha256: str
    atom_payload: dict


class _Catalog:
    def __init__(self, atoms):
        self.atoms = tuple(atoms)

    def approved_atoms(self):
        return self.atoms


@dataclass(frozen=True)
class _Plan:
    turn_id: str
    strategy: str
    seed: int
    atoms: tuple[_Atom, ...]

    def to_log_record(self):
        return {
            "turn_id": self.turn_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "atoms": [
                {
                    "atom_id": atom.atom_id,
                    "version": atom.version,
                    "revision": atom.revision,
                    "content_sha256": atom.content_sha256,
                }
                for atom in self.atoms
            ],
        }


class _Planner:
    def __init__(self):
        self.calls = []

    def plan(self, *, turn_id, atom_catalog, count, seed):
        self.calls.append((turn_id, count, seed))
        atoms = list(atom_catalog.approved_atoms())
        offset = seed % len(atoms)
        selected = tuple((atoms[offset:] + atoms[:offset])[:count])
        return _Plan(turn_id, "random_v0", seed, selected)


class _FailingPlanner:
    def plan(self, **_kwargs):
        raise RuntimeError("planner unavailable")


def _atoms(count=5):
    return tuple(
        _Atom(
            atom_id=f"atom_{index:03d}",
            version=f"version_{index}",
            revision=2,
            content_sha256=(f"{index:x}" * 64)[:64],
            atom_payload={"control": _control(f"atom_{index:03d}")},
        )
        for index in range(1, count + 1)
    )


class _FakeClient:
    """A capacity-two, action-specific lifecycle fake.

    It deliberately does not expose `/state.last_completed`; the coordinator
    must observe each request through `action_sync(id)`.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.timeline = []
        self.records = {}
        self.polls = {}
        self.stop_count = 0

    def trigger_sync(self, action, **kwargs):
        with self.lock:
            request_id = f"action-{len(self.calls) + 1}"
            label = action if isinstance(action, str) else action["name"]
            self.calls.append((label, dict(kwargs), request_id))
            self.timeline.append(f"post:{label}")
            self.records[request_id] = {
                "id": request_id,
                "name": label,
                "status": "scheduled",
                "bridge_policy": kwargs.get("bridge_policy"),
                "bridge_duration": None,
                "scheduled_wall": time.time(),
                "started_wall": None,
                "completed_wall": None,
                "finished_wall": None,
            }
            self.polls[request_id] = 0
            return {"accepted": True, "id": request_id, "action": dict(self.records[request_id])}

    def action_sync(self, action_id):
        with self.lock:
            record = self.records[action_id]
            self.polls[action_id] += 1
            index = int(action_id.split("-")[1])
            is_neutral = record["name"] == "joint_release_arm_pose"
            predecessor_complete = index == 1 or self.records[f"action-{index - 1}"]["status"] == "completed"
            if is_neutral or predecessor_complete:
                if record["status"] == "scheduled":
                    record["status"] = "started"
                    record["bridge_duration"] = 0.04 * index
                    record["started_wall"] = time.time()
                elif record["status"] == "started":
                    record["status"] = "completed"
                    record["completed_wall"] = time.time()
                    record["finished_wall"] = record["completed_wall"]
            self.timeline.append(f"get:{action_id}:{record['status']}")
            return dict(record)

    def stop_action_sync(self):
        with self.lock:
            self.stop_count += 1
            self.timeline.append("stop")
            for record in self.records.values():
                if record["status"] not in {"completed", "failed", "canceled", "replaced"}:
                    record["status"] = "canceled"
                    record["finished_wall"] = time.time()
            return {"stopped": True}


class _BlockingSecondPostClient(_FakeClient):
    def __init__(self):
        super().__init__()
        self.second_post_entered = threading.Event()
        self.release_second_post = threading.Event()

    def trigger_sync(self, action, **kwargs):
        is_atom = not isinstance(action, str)
        with self.lock:
            atom_posts = sum(
                1 for label, _, _ in self.calls if label != "joint_release_arm_pose"
            )
        if is_atom and atom_posts == 1:
            self.timeline.append("post:second:begin")
            self.second_post_entered.set()
            if not self.release_second_post.wait(timeout=2.0):
                raise RuntimeError("test did not release the second POST")
            self.timeline.append("post:second:end")
        return super().trigger_sync(action, **kwargs)


class DialogueSequenceCoordinatorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.log_path = Path(self.tempdir.name) / "turns.jsonl"
        self.catalog = _Catalog(_atoms())
        self.planner = _Planner()
        self.client = _FakeClient()
        self.events = []

        async def event_sink(payload):
            self.events.append(dict(payload))

        self.coordinator = DialogueSequenceCoordinator(
            client=self.client,
            atom_catalog=self.catalog,
            planner=self.planner,
            settings=SequenceSettings(
                count=3,
                base_seed=42,
                log_path=self.log_path,
                poll_interval=0.0,
                action_timeout=2.0,
            ),
            event_sink=event_sink,
        )

    async def test_repeated_model_output_plans_once_and_enqueues_after_first_started(self):
        first = self.coordinator.start_turn(source_turn_id=17)
        repeated = self.coordinator.start_turn(source_turn_id=17)
        self.assertIs(first, repeated)

        result = await first.wait_atoms_complete()

        self.assertEqual(result.status, "atoms_completed")
        self.assertEqual(self.planner.calls, [("turn_001", 3, 42)])
        atom_calls = self.client.calls[:3]
        self.assertEqual(len(atom_calls), 3)
        self.assertEqual(len({call[0] for call in atom_calls}), 3)
        for label, kwargs, _ in atom_calls:
            self.assertNotIn(label, {"joint_presentation_gesture", "joint_presentation_gesture_b", "joint_presentation_gesture_c"})
            self.assertEqual(kwargs["bridge_policy"], "minimum_jerk_v1")
            self.assertEqual(kwargs["blend_duration"], 0.0)
            self.assertEqual(kwargs["mode"], "enqueue")

        first_started = self.client.timeline.index("get:action-1:started")
        second_post = next(index for index, value in enumerate(self.client.timeline) if value.startswith("post:atom_") and index > 0)
        third_post = next(index for index, value in enumerate(self.client.timeline[second_post + 1 :], second_post + 1) if value.startswith("post:atom_"))
        first_completed = self.client.timeline.index("get:action-1:completed")
        self.assertLess(first_started, second_post)
        self.assertLess(third_post, first_completed)
        self.assertEqual(
            [event["lifecycle"] for event in self.events].count("completed"),
            3,
        )

    async def test_turn_complete_waits_for_inflight_post_then_cancels_and_returns_neutral(self):
        client = _BlockingSecondPostClient()
        coordinator = DialogueSequenceCoordinator(
            client=client,
            atom_catalog=self.catalog,
            planner=_Planner(),
            settings=SequenceSettings(
                count=3,
                base_seed=7,
                log_path=self.log_path,
                poll_interval=0.0,
                action_timeout=2.0,
            ),
        )
        handle = coordinator.start_turn(source_turn_id=3)
        entered = await asyncio.to_thread(client.second_post_entered.wait, 1.0)
        self.assertTrue(entered)

        finish_task = coordinator.request_finish(source_turn_id=3, reason="turn_complete")
        await asyncio.sleep(0)
        self.assertNotIn("stop", client.timeline)

        client.release_second_post.set()
        final = await asyncio.wait_for(asyncio.shield(finish_task), timeout=2.0)

        with self.assertRaises(asyncio.CancelledError):
            await handle.wait_atoms_complete()
        self.assertEqual(final.status, "turn_complete")
        self.assertGreater(final.playback_elapsed_seconds, 0.0)
        self.assertIn("post:second:end", client.timeline)
        self.assertLess(client.timeline.index("post:second:end"), client.timeline.index("stop"))
        atom_posts = [call for call in client.calls if call[0] != "joint_release_arm_pose"]
        self.assertEqual(len(atom_posts), 2)
        self.assertEqual(client.calls[-1][0], "joint_release_arm_pose")
        self.assertEqual(client.calls[-1][1]["bridge_policy"], "minimum_jerk_v1")
        self.assertEqual(client.calls[-1][1]["blend_duration"], 0.0)
        self.assertLess(client.timeline.index("stop"), client.timeline.index("post:joint_release_arm_pose"))

    async def test_three_turns_have_seeded_unique_plans_neutral_and_complete_logs(self):
        final_results = []
        for source_turn in (101, 102, 103):
            handle = self.coordinator.start_turn(source_turn_id=source_turn)
            await handle.wait_atoms_complete()
            final_results.append(
                await self.coordinator.finish_turn(
                    source_turn_id=source_turn, reason="turn_complete"
                )
            )

        self.assertEqual([result.seed for result in final_results], [42, 43, 44])
        self.assertEqual(self.client.stop_count, 3)
        neutrals = [call for call in self.client.calls if call[0] == "joint_release_arm_pose"]
        self.assertEqual(len(neutrals), 3)
        for result in final_results:
            self.assertEqual(len(result.atoms), 3)
            self.assertEqual(len({atom["atom_id"] for atom in result.atoms}), 3)
            self.assertEqual(len(result.action_ids), 3)
            self.assertEqual(len(result.action_records), 3)
            self.assertEqual(len(result.bridge_durations), 3)
            self.assertGreaterEqual(result.wall_elapsed_seconds, 0.0)
            for action in result.action_records:
                self.assertEqual(action["status"], "completed")
                self.assertIsNotNone(action["scheduled_wall"])
                self.assertIsNotNone(action["started_wall"])
                self.assertIsNotNone(action["completed_wall"])
                self.assertIsNotNone(action["finished_wall"])

        records = [json.loads(line) for line in self.log_path.read_text().splitlines()]
        final_records = [record for record in records if record["event"] == "turn_finished"]
        self.assertEqual(len(final_records), 3)
        self.assertEqual([record["seed"] for record in final_records], [42, 43, 44])
        for record in final_records:
            self.assertEqual(record["bridge_policy"], "minimum_jerk_v1")
            self.assertEqual(record["status"], "turn_complete")
            self.assertEqual(len(record["atoms"]), 3)
            self.assertEqual(len(record["action_ids"]), 3)
            self.assertEqual(len(record["actions"]), 3)
            self.assertEqual(len(record["bridge_durations"]), 3)

    async def test_reconnect_turn_waits_for_previous_cancel_and_neutral_barrier(self):
        client = _BlockingSecondPostClient()
        coordinator = DialogueSequenceCoordinator(
            client=client,
            atom_catalog=self.catalog,
            planner=_Planner(),
            settings=SequenceSettings(
                count=3,
                base_seed=20,
                log_path=self.log_path,
                poll_interval=0.0,
                action_timeout=2.0,
            ),
        )
        first = coordinator.start_turn(source_turn_id="connection-a")
        entered = await asyncio.to_thread(client.second_post_entered.wait, 1.0)
        self.assertTrue(entered)

        old_finish = coordinator.request_finish(
            source_turn_id="connection-a", reason="interrupted"
        )
        second = coordinator.start_turn(source_turn_id="connection-b")
        await asyncio.sleep(0)
        self.assertEqual(len(client.calls), 1)

        client.release_second_post.set()
        await asyncio.wait_for(asyncio.shield(old_finish), timeout=2.0)
        await asyncio.wait_for(second.wait_atoms_complete(), timeout=2.0)

        neutral_index = next(
            index
            for index, call in enumerate(client.calls)
            if call[0] == "joint_release_arm_pose"
        )
        first_new_turn_index = neutral_index + 1
        self.assertLess(neutral_index, first_new_turn_index)
        self.assertNotEqual(
            client.calls[first_new_turn_index][0], "joint_release_arm_pose"
        )
        with self.assertRaises(asyncio.CancelledError):
            await first.wait_atoms_complete()
        await coordinator.finish_turn(
            source_turn_id="connection-b", reason="turn_complete"
        )

    async def test_duplicate_normal_cleanup_submits_neutral_only_once(self):
        handle = self.coordinator.start_turn(source_turn_id=55)
        await handle.wait_atoms_complete()
        first = self.coordinator.request_finish(
            source_turn_id=55, reason="turn_complete"
        )
        repeated = self.coordinator.request_finish(
            source_turn_id=55, reason="turn_complete"
        )
        self.assertIs(first, repeated)
        await asyncio.gather(asyncio.shield(first), asyncio.shield(repeated))
        neutrals = [
            call for call in self.client.calls if call[0] == "joint_release_arm_pose"
        ]
        self.assertEqual(len(neutrals), 1)

    async def test_startup_failure_recovery_stops_and_returns_neutral(self):
        coordinator = DialogueSequenceCoordinator(
            client=self.client,
            atom_catalog=self.catalog,
            planner=_FailingPlanner(),
            settings=SequenceSettings(
                log_path=self.log_path,
                poll_interval=0.0,
                action_timeout=2.0,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "planner unavailable"):
            coordinator.start_turn(source_turn_id="bad-plan")
        recovery = coordinator.request_recovery(
            source_turn_id="bad-plan", error="planner unavailable"
        )
        result = await asyncio.wait_for(asyncio.shield(recovery), timeout=2.0)

        self.assertEqual(result.status, "failed")
        self.assertEqual(self.client.stop_count, 1)
        self.assertEqual(self.client.calls[-1][0], "joint_release_arm_pose")
        self.assertTrue(result.neutral_action_id)
        self.assertIn("planner unavailable", result.error)

    def test_atom_payload_is_exact_canonical_q_trajectory_without_preview_frames(self):
        atom = _atoms(1)[0]
        motion = atom_to_inline_motion(atom)
        self.assertEqual(motion["joints"], atom.atom_payload["control"]["joints"])
        self.assertEqual(motion["frames"], atom.atom_payload["control"]["trajectory"])
        self.assertEqual(motion["interpolation"], "linear")
        self.assertIn(atom.version, motion["name"])
        self.assertIn(atom.content_sha256, motion["name"])
        self.assertNotIn("png", json.dumps(motion).lower())

    def test_runner_contains_no_legacy_fixed_point_three_second_gap(self):
        source = inspect.getsource(DialogueSequenceCoordinator)
        self.assertNotIn("0.3", source)
        self.assertNotIn("PRESENTATION_GESTURE_LOOP_GAP_SECONDS", source)

    def test_frozen_mvp_settings_reject_nonfinite_values_and_non_three_count(self):
        with self.assertRaises(ValueError):
            SequenceSettings(count=4)
        for field_name, value in (
            ("speed", float("nan")),
            ("speed", float("inf")),
            ("poll_interval", float("nan")),
            ("action_timeout", float("inf")),
        ):
            with self.subTest(field=field_name, value=value):
                with self.assertRaises(ValueError):
                    SequenceSettings(**{field_name: value})


class _BridgeCoordinator:
    def __init__(self):
        self.starts = []
        self.finishes = []

    def start_turn(self, *, source_turn_id):
        self.starts.append(source_turn_id)
        return object()

    async def finish_turn(self, *, source_turn_id, reason):
        self.finishes.append((source_turn_id, reason))
        return object()

    def request_finish(self, *, source_turn_id, reason):
        async def finish():
            self.finishes.append((source_turn_id, reason))
        return asyncio.create_task(finish())


class _FailingBridgeCoordinator(_BridgeCoordinator):
    def __init__(self):
        super().__init__()
        self.start_attempts = []
        self.failure_events = []
        self.recoveries = []

    def start_turn(self, *, source_turn_id):
        self.start_attempts.append(source_turn_id)
        raise RuntimeError("planner unavailable")

    async def report_start_failure(self, *, source_turn_id, error):
        self.failure_events.append((source_turn_id, error))

    def request_recovery(self, *, source_turn_id, error):
        async def recover():
            self.recoveries.append((source_turn_id, error))
            return object()

        return asyncio.create_task(recover())


def _bridge_class():
    class Bridge:
        def __init__(self):
            self.robot_turn_serial = 7
            self.robot_turn_complete_received = False
            self.mark_count = 0
            self.old_input_count = 0
            self.old_output_count = 0
            self.old_cleanup_count = 0

        def _mark_robot_output_activity(self):
            self.mark_count += 1

        async def _handle_robot_input_transcription(self, text):
            self.old_input_count += 1

        async def _handle_robot_output_transcription(self, text):
            self.old_output_count += 1

        async def _cleanup_robot_turn_after_playback(self, turn_serial):
            self.old_cleanup_count += 1

        def _reset_robot_turn_state(self):
            self.robot_turn_serial += 1
            self.robot_turn_complete_received = False

    return Bridge


class RandomDialogueBridgePatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_model_output_activity_starts_once_and_bypasses_legacy_abc(self):
        bridge_cls = _bridge_class()
        coordinator = _BridgeCoordinator()
        changed = patch_random_dialogue_bridge_class(
            bridge_cls, coordinator_factory=lambda _bridge: coordinator
        )
        self.assertTrue(changed)
        bridge = bridge_cls()

        bridge._mark_robot_output_activity()
        bridge._mark_robot_output_activity()
        await bridge._handle_robot_input_transcription("please present")
        await bridge._handle_robot_output_transcription("ordinary response without a presentation keyword")

        self.assertEqual(bridge.mark_count, 2)
        self.assertEqual(coordinator.starts, [7])
        self.assertEqual(bridge.old_input_count, 0)
        self.assertEqual(bridge.old_output_count, 0)

        bridge.robot_turn_complete_received = True
        await bridge._cleanup_robot_turn_after_playback(7)
        self.assertEqual(coordinator.finishes, [(7, "turn_complete")])
        self.assertEqual(bridge.robot_turn_serial, 8)
        self.assertEqual(bridge.old_cleanup_count, 0)

    async def test_interrupted_reset_invalidates_sequence_and_requests_neutral(self):
        bridge_cls = _bridge_class()
        coordinator = _BridgeCoordinator()
        patch_random_dialogue_bridge_class(
            bridge_cls, coordinator_factory=lambda _bridge: coordinator
        )
        bridge = bridge_cls()
        bridge._mark_robot_output_activity()

        bridge._reset_robot_turn_state()
        await asyncio.sleep(0)

        self.assertEqual(coordinator.starts, [7])
        self.assertIn((7, "interrupted"), coordinator.finishes)
        self.assertEqual(bridge.robot_turn_serial, 8)

    async def test_planner_failure_is_observable_and_does_not_escape_receive_hook(self):
        bridge_cls = _bridge_class()
        coordinator = _FailingBridgeCoordinator()
        patch_random_dialogue_bridge_class(
            bridge_cls, coordinator_factory=lambda _bridge: coordinator
        )
        bridge = bridge_cls()

        # This synchronous hook runs inside Gemini's receive iterator.  A Planner
        # failure must not propagate and tear down the voice session.
        bridge._mark_robot_output_activity()
        bridge._mark_robot_output_activity()
        await asyncio.wait_for(
            asyncio.shield(bridge._mujoco_sequence_start_failure_task), timeout=1.0
        )

        self.assertEqual(coordinator.start_attempts, [7])
        self.assertEqual(
            coordinator.failure_events, [(7, "planner unavailable")]
        )
        self.assertEqual(coordinator.recoveries, [(7, "planner unavailable")])
        self.assertEqual(bridge.robot_turn_mode, "random_catalog_failed")
        self.assertFalse(bridge.robot_presentation_started)

    async def test_slow_failure_event_cannot_delay_recovery_barrier_into_next_turn(self):
        client = _FakeClient()
        failure_event_started = asyncio.Event()
        release_failure_event = asyncio.Event()

        async def slow_event_sink(payload):
            if payload.get("lifecycle") == "failed":
                failure_event_started.set()
                await release_failure_event.wait()

        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        coordinator = DialogueSequenceCoordinator(
            client=client,
            atom_catalog=_Catalog(_atoms()),
            planner=_FailingPlanner(),
            settings=SequenceSettings(
                log_path=Path(tempdir.name) / "turns.jsonl",
                poll_interval=0.0,
                action_timeout=2.0,
            ),
            event_sink=slow_event_sink,
        )
        bridge_cls = _bridge_class()
        patch_random_dialogue_bridge_class(
            bridge_cls, coordinator_factory=lambda _bridge: coordinator
        )
        bridge = bridge_cls()

        bridge._mark_robot_output_activity()
        failure_task = bridge._mujoco_sequence_start_failure_task
        await asyncio.wait_for(failure_event_started.wait(), timeout=1.0)

        # turn_complete arrives while browser failure reporting is still blocked.
        await asyncio.wait_for(
            bridge._cleanup_robot_turn_after_playback(7), timeout=1.0
        )
        self.assertEqual(bridge.robot_turn_serial, 8)
        self.assertEqual(client.stop_count, 1)
        self.assertEqual(client.calls[-1][0], "joint_release_arm_pose")

        coordinator.planner = _Planner()
        bridge._mark_robot_output_activity()
        next_turn = coordinator.start_turn(source_turn_id=8)
        await asyncio.wait_for(next_turn.wait_all_submitted(), timeout=1.0)
        stop_count_before_slow_event_returns = client.stop_count

        release_failure_event.set()
        await asyncio.wait_for(asyncio.shield(failure_task), timeout=1.0)
        await asyncio.sleep(0)
        self.assertEqual(client.stop_count, stop_count_before_slow_event_returns)

        await asyncio.wait_for(next_turn.wait_atoms_complete(), timeout=1.0)
        await bridge._cleanup_robot_turn_after_playback(8)

    async def test_plan_log_failure_has_zero_atom_posts_and_neutral_barrier(self):
        client = _FakeClient()
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        coordinator = DialogueSequenceCoordinator(
            client=client,
            atom_catalog=_Catalog(_atoms()),
            planner=_Planner(),
            settings=SequenceSettings(
                log_path=Path(tempdir.name) / "turns.jsonl",
                poll_interval=0.0,
                action_timeout=2.0,
            ),
        )
        original_append = coordinator._log.append

        def fail_log(_payload):
            raise OSError("audit disk full")

        coordinator._log.append = fail_log
        bridge_cls = _bridge_class()
        patch_random_dialogue_bridge_class(
            bridge_cls, coordinator_factory=lambda _bridge: coordinator
        )
        bridge = bridge_cls()

        bridge._mark_robot_output_activity()
        await asyncio.wait_for(
            asyncio.shield(bridge._mujoco_sequence_start_failure_task), timeout=1.0
        )
        self.assertEqual(client.stop_count, 1)
        self.assertEqual(
            [call[0] for call in client.calls], ["joint_release_arm_pose"]
        )
        self.assertIn("audit disk full", coordinator.last_log_error)

        await bridge._cleanup_robot_turn_after_playback(7)
        coordinator._log.append = original_append
        bridge._mark_robot_output_activity()
        next_turn = coordinator.start_turn(source_turn_id=8)
        await asyncio.wait_for(next_turn.wait_all_submitted(), timeout=1.0)

        neutral_post = client.timeline.index("post:joint_release_arm_pose")
        next_atom_post = next(
            index
            for index, value in enumerate(client.timeline)
            if index > neutral_post and value.startswith("post:atom_")
        )
        self.assertLess(neutral_post, next_atom_post)
        await asyncio.wait_for(next_turn.wait_atoms_complete(), timeout=1.0)
        await bridge._cleanup_robot_turn_after_playback(8)


class RandomDialogueConfigurationTest(unittest.TestCase):
    def test_environment_selects_random_catalog_and_exact_mvp_defaults(self):
        with patch.dict(
            os.environ,
            {
                "OVERLAY_DIALOGUE_SEQUENCE_MODE": "random_catalog",
                "OVERLAY_ATOM_CATALOG_PATH": "/tmp/exact-catalog.json",
                "OVERLAY_RANDOM_SEQUENCE_COUNT": "3",
                "OVERLAY_RANDOM_SEQUENCE_BASE_SEED": "70",
                "OVERLAY_RANDOM_SEQUENCE_LOG_PATH": "/tmp/turns.jsonl",
            },
            clear=False,
        ):
            config = OverlayConfig.from_env()
        self.assertEqual(config.dialogue_sequence_mode, "random_catalog")
        self.assertEqual(config.atom_catalog_path, "/tmp/exact-catalog.json")
        self.assertEqual(config.random_sequence_count, 3)
        self.assertEqual(config.random_sequence_base_seed, 70)
        self.assertEqual(config.random_sequence_log_path, "/tmp/turns.jsonl")

    def test_install_switches_random_patch_on_and_legacy_fallback_off(self):
        random_module = ModuleType("fake_random_dialogue_upstream")
        random_module.GeminiLiveBridge = _bridge_class()
        client = _FakeClient()
        random_config = OverlayConfig(
            dialogue_sequence_mode="random_catalog",
            atom_catalog_path="/does/not/matter/in/mock.json",
        )
        with patch("web_overlay.patching.load_catalog", return_value=self):
            report = install_runtime_patches(random_module, client, random_config)
        self.assertTrue(
            random_module.GeminiLiveBridge._mujoco_random_sequence_patched
        )
        self.assertIn(
            "fake_random_dialogue_upstream.GeminiLiveBridge.random_catalog_sequence",
            report.action_symbols,
        )

        legacy_module = ModuleType("fake_legacy_dialogue_upstream")
        legacy_module.GeminiLiveBridge = _bridge_class()
        install_runtime_patches(legacy_module, _FakeClient(), OverlayConfig())
        self.assertFalse(
            getattr(
                legacy_module.GeminiLiveBridge,
                "_mujoco_random_sequence_patched",
                False,
            )
        )


if __name__ == "__main__":
    unittest.main()
