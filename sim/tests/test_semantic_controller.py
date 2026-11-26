from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from web_overlay.config import OverlayConfig
from web_overlay.semantic_controller import (
    DurableSemanticAudit,
    SemanticControllerError,
    SemanticDialogueController,
    _SemanticPlannerAdapter,
    semantic_controller_factory,
)
from web_overlay.semantic_catalog import SemanticCatalog, SemanticCatalogRow
from web_overlay.semantic_runtime import RUNTIME_FUNCTIONS
from web_overlay.semantic_planner import PlannerHistory, update_history


@dataclass(frozen=True)
class _Request:
    ordered_functions: tuple[str, ...]
    estimated_spoken_duration_s: float
    request_id: str
    revision: int


class _Execution:
    def __init__(self, kwargs, gate):
        self.kwargs = kwargs
        self.gate = gate
        self.started = asyncio.Event()
        self.audio = []
        self.completed = []
        self.interruptions = []

    async def run(self):
        self.started.set()
        await self.gate.wait()

    def on_audio_chunk(self, **kwargs):
        self.audio.append(kwargs)

    def on_turn_complete(self, **kwargs):
        self.completed.append(kwargs)

    def interrupt(self, **kwargs):
        self.interruptions.append(kwargs)


class _Harness:
    def __init__(self):
        self.bridge = SimpleNamespace(robot_turn_serial=7)
        self.now = 1_000_000_000
        self.validations = []
        self.executions = []
        self.gate = asyncio.Event()

        def validate(payload, *, request_id, revision):
            self.validations.append((dict(payload), request_id, revision))
            return _Request(
                ordered_functions=tuple(payload["ordered_functions"]),
                estimated_spoken_duration_s=float(
                    payload["estimated_spoken_duration_s"]
                ),
                request_id=request_id,
                revision=revision,
            )

        def make_execution(**kwargs):
            execution = _Execution(kwargs, self.gate)
            self.executions.append(execution)
            return execution

        self.controller = SemanticDialogueController(
            bridge=self.bridge,
            validate_payload=validate,
            execution_factory=make_execution,
            execute=True,
            monotonic_ns=lambda: self.now,
        )

    @property
    def payload(self):
        return {
            "ordered_functions": ["turn_open", "emphasize"],
            "estimated_spoken_duration_s": 5.0,
        }


def _catalog() -> SemanticCatalog:
    motion = {
        "format": "g1_joint_motion_v1",
        "name": "approved-emphasize",
        "duration": 0.1,
        "joints": ["left_elbow"],
        "interpolation": "linear",
        "frames": [
            {"t": 0.0, "q": {"left_elbow": 0.0}},
            {"t": 0.1, "q": {"left_elbow": 0.1}},
        ],
        "source_atom_hash": "a" * 64,
        "parent_atom_hash": None,
    }
    row = SemanticCatalogRow(
        entry_id="entry-emphasize",
        index=1,
        atom_id="atom-emphasize",
        version="0123456789abcdef0123456789abcdef",
        revision=1,
        content_sha256="a" * 64,
        function="emphasize",
        planner_category="emphasis",
        duration_us=100_000,
        execution_side="left",
        left_path_rad=0.1,
        right_path_rad=0.0,
        source_id="fixture:1",
        approved=True,
        removed=False,
        source={},
        provenance={},
        motion=motion,
    )
    neutral = {
        "format": "g1_joint_motion_v1",
        "name": "canonical-neutral",
        "duration": 0.8,
        "joints": ["left_elbow"],
        "interpolation": "linear",
        "frames": [
            {"t": 0.0, "q": {"left_elbow": 0.0}},
            {"t": 0.8, "q": {"left_elbow": 0.0}},
        ],
        "source_atom_hash": "b" * 64,
        "parent_atom_hash": None,
    }
    return SemanticCatalog(
        schema_version="semantic_atom_catalog_v1",
        ontology_version="dialogue_function_17_v1",
        ontology_sha256="c" * 64,
        ontology={},
        rows=(row,),
        by_function={
            function: (row.entry_id,) if function == "emphasize" else ()
            for function in RUNTIME_FUNCTIONS
        },
        retained_count=1,
        removed_count=0,
        source_count=1,
        neutral_motion_sha256="d" * 64,
        neutral_duration_us=800_000,
        neutral_q={"left_elbow": 0.0},
        neutral_motion=neutral,
        inputs={},
        content_sha256="e" * 64,
        _canonical_bytes=b"{}\n",
    )


def _force_choice_catalog() -> tuple[
    SemanticCatalog,
    SemanticCatalogRow,
    SemanticCatalogRow,
]:
    base = _catalog()
    original = base.rows[0]
    short_motion = {
        **dict(original.motion),
        "name": "forced-short-right",
        "duration": 0.1,
        "frames": [
            {"t": 0.0, "q": {"left_elbow": 0.0}},
            {"t": 0.1, "q": {"left_elbow": 0.1}},
        ],
        "source_atom_hash": "1" * 64,
    }
    long_motion = {
        **dict(original.motion),
        "name": "forced-long-left",
        "duration": 0.9,
        "frames": [
            {"t": 0.0, "q": {"left_elbow": 0.0}},
            {"t": 0.9, "q": {"left_elbow": 0.1}},
        ],
        "source_atom_hash": "2" * 64,
    }
    short = replace(
        original,
        entry_id="entry-forced-short",
        atom_id="atom-forced-short",
        version="11111111111111111111111111111111",
        content_sha256="1" * 64,
        duration_us=100_000,
        execution_side="right",
        left_path_rad=0.0,
        right_path_rad=0.1,
        source_id="fixture:short",
        motion=short_motion,
    )
    long = replace(
        original,
        entry_id="entry-forced-long",
        atom_id="atom-forced-long",
        version="22222222222222222222222222222222",
        content_sha256="2" * 64,
        duration_us=900_000,
        execution_side="left",
        left_path_rad=0.1,
        right_path_rad=0.0,
        source_id="fixture:long",
        motion=long_motion,
    )
    catalog = replace(
        base,
        rows=(long, short),
        by_function={
            function: (
                (long.entry_id, short.entry_id)
                if function == "emphasize"
                else ()
            )
            for function in RUNTIME_FUNCTIONS
        },
        retained_count=2,
        source_count=2,
        content_sha256="f" * 64,
    )
    return catalog, short, long


class _LiveClient:
    def __init__(self):
        self.posts = []
        self.seq = 0
        self.q = {"left_elbow": 0.0}

    def state_sync(self):
        return {
            "action": None,
            "queued": [],
            "action_seq": self.seq,
            "fault": None,
            "q": dict(self.q),
        }

    def post_action_sync(self, payload):
        body = dict(payload)
        self.posts.append(body)
        self.seq += 1
        frames = body["motion"]["frames"]
        self.q = dict(frames[-1]["q"])
        return {
            "accepted": True,
            "id": body["id"],
            "payload_sha256": body["payload_sha256"],
        }

    def action_sync(self, action_id):
        payload = next(item for item in self.posts if item["id"] == action_id)
        return {
            "id": action_id,
            "status": "completed",
            "payload_sha256": payload["payload_sha256"],
        }


class _BlockingPlanningClient(_LiveClient):
    def __init__(self):
        super().__init__()
        self.state_entered = threading.Event()
        self.state_release = threading.Event()
        self._blocked_once = False

    def state_sync(self):
        if not self._blocked_once:
            self._blocked_once = True
            self.state_entered.set()
            if not self.state_release.wait(timeout=2.0):
                raise TimeoutError("test planning state gate was not released")
        return super().state_sync()


class _ForbiddenStartupClient:
    def __init__(self):
        self.calls = []

    def state_sync(self):
        self.calls.append("state")
        raise AssertionError("semantic construction must not read simulator state")

    def post_action_sync(self, payload):
        self.calls.append(("post", payload))
        raise AssertionError("semantic construction must not submit an action")

    def action_sync(self, action_id):
        self.calls.append(("action", action_id))
        raise AssertionError("semantic construction must not query an action")


class _DriftTwiceThenStableClient(_LiveClient):
    def __init__(self):
        super().__init__()
        self._planning_and_admission_q = [0.0, 0.02, 0.04, 0.04]
        self.semantic_terminal_entered = threading.Event()
        self.semantic_terminal_release = threading.Event()
        self._blocked_semantic_terminal = False

    def state_sync(self):
        if self._planning_and_admission_q:
            self.q = {
                "left_elbow": self._planning_and_admission_q.pop(0),
            }
        return super().state_sync()

    def action_sync(self, action_id):
        payload = next(item for item in self.posts if item["id"] == action_id)
        if (
            payload["motion"]["name"] == "approved-emphasize"
            and not self._blocked_semantic_terminal
        ):
            self._blocked_semantic_terminal = True
            self.semantic_terminal_entered.set()
            if not self.semantic_terminal_release.wait(timeout=2.0):
                raise TimeoutError("test semantic terminal gate was not released")
        return super().action_sync(action_id)


class _Bridge:
    def __init__(self):
        self.robot_turn_serial = 1
        self.events = []

    async def _emit_robot_action_event(self, payload):
        self.events.append(dict(payload))


class SemanticControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_force_first_policy_is_explicit_default_off_and_env_enabled(self):
        self.assertFalse(OverlayConfig().semantic_force_first)
        with patch.dict(
            "os.environ",
            {"OVERLAY_SEMANTIC_FORCE_FIRST": "1"},
        ):
            self.assertTrue(OverlayConfig.from_env().semantic_force_first)
        root = Path(__file__).resolve().parents[1]
        common = (root / "scripts" / "common.sh").read_text(encoding="utf-8")
        runner = (root / "scripts" / "run_web.sh").read_text(encoding="utf-8")
        self.assertIn(
            'OVERLAY_SEMANTIC_FORCE_FIRST="${OVERLAY_SEMANTIC_FORCE_FIRST:-1}"',
            common,
        )
        self.assertIn(
            'OVERLAY_SEMANTIC_FORCE_FIRST="${OVERLAY_SEMANTIC_FORCE_FIRST}"',
            runner,
        )

    async def test_force_first_plan_minimizes_complete_closed_loop_cost(self):
        catalog, short, _long = _force_choice_catalog()
        planner = _SemanticPlannerAdapter(catalog, tail_guard_us=100_000)
        planner.begin(threading.Event())
        measured = {"q": {"left_elbow": 0.0}}

        normal = planner(
            ("emphasize",),
            1,
            measured,
            (),
        )
        self.assertEqual(normal.atoms, ())
        forced = planner.force_first_plan(
            ("emphasize", "deny"),
            measured,
            (),
        )

        self.assertEqual(forced.atoms, (short,))
        self.assertEqual(forced.covered_functions, ("emphasize",))
        self.assertEqual(forced.remaining_functions, ())
        action = planner.prepare_action(short)
        recovery = planner.prepare_recovery(short, 1, ())
        self.assertEqual(
            forced.used_budget_us,
            action.estimated_duration_us + recovery.estimated_duration_us,
        )
        planner.finish(1)
        self.assertEqual(
            planner._persistent_history.recent_source_ids,
            ("fixture:short",),
        )

    async def test_force_first_plan_uses_history_side_ties_and_requires_candidate(self):
        catalog, short, long = _force_choice_catalog()
        tied_long_motion = {
            **dict(long.motion),
            "duration": 0.1,
            "frames": [
                {"t": 0.0, "q": {"left_elbow": 0.0}},
                {"t": 0.1, "q": {"left_elbow": 0.1}},
            ],
        }
        tied_long = replace(
            long,
            duration_us=100_000,
            motion=tied_long_motion,
        )
        tied_catalog = replace(
            catalog,
            rows=(tied_long, short),
        )
        planner = _SemanticPlannerAdapter(tied_catalog, tail_guard_us=100_000)
        planner._persistent_history = update_history(
            PlannerHistory(),
            (tied_long,),
        )
        planner.begin(threading.Event())
        measured = {"q": {"left_elbow": 0.0}}

        forced = planner.force_first_plan(("emphasize",), measured, ())
        self.assertEqual(forced.atoms, (short,))
        planner.discard_plan(short, "test_uncommitted_force")
        missing = planner.force_first_plan(("affirm",), measured, ())
        self.assertEqual(missing.atoms, ())
        self.assertEqual(planner._local.selected, [])
        planner.finish(0)

    async def test_production_factory_and_controller_construction_are_io_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = SimpleNamespace(
                semantic_tail_guard_seconds=0.1,
                semantic_sequence_log_path=str(Path(temporary) / "semantic.jsonl"),
                semantic_execute=True,
                semantic_force_first=False,
                semantic_poll_interval=0.001,
                semantic_action_timeout=1.0,
            )
            client = _ForbiddenStartupClient()

            factory = semantic_controller_factory(
                catalog=_catalog(),
                client=client,
                config=config,
            )
            controller = factory(_Bridge())

            self.assertIsInstance(controller, SemanticDialogueController)
            self.assertEqual(client.calls, [])
            self.assertFalse(Path(config.semantic_sequence_log_path).exists())

    async def test_tool_first_waits_for_first_output_before_starting_runtime(self):
        harness = _Harness()

        response = await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="call-1",
            payload=harness.payload,
        )

        self.assertTrue(response["accepted"])
        self.assertEqual(harness.executions, [])
        harness.now = 1_250_000_000
        harness.controller.on_output_activity(turn_id=7)
        await harness.executions[0].started.wait()
        self.assertEqual(
            harness.executions[0].kwargs["output_started_ns"],
            1_250_000_000,
        )
        self.assertTrue(harness.controller.has_pending(turn_id=7))
        harness.gate.set()
        await asyncio.sleep(0)

    async def test_out_of_order_scheduled_hooks_keep_earliest_sync_arrival(self):
        harness = _Harness()
        await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="call-reordered-hooks",
            payload=harness.payload,
        )
        harness.controller.observe_output_activity(
            turn_id=7, observed_ns=700_000_000
        )
        harness.controller.observe_output_activity(
            turn_id=7, observed_ns=900_000_000
        )

        # Simulate event-loop scheduling in the opposite order from arrival.
        harness.controller.on_output_activity(
            turn_id=7, observed_ns=900_000_000
        )
        harness.controller.on_output_activity(
            turn_id=7, observed_ns=700_000_000
        )
        await harness.executions[0].started.wait()

        self.assertEqual(
            harness.executions[0].kwargs["output_started_ns"],
            700_000_000,
        )
        harness.gate.set()
        await asyncio.sleep(0)

    async def test_audio_first_freezes_old_clock_for_late_tool(self):
        harness = _Harness()
        harness.controller.on_audio_chunk(
            turn_id=7,
            audio_bytes=b"first audio",
            mime_type="audio/pcm;rate=24000",
            observed_ns=900_000_000,
        )
        harness.now = 3_500_000_000

        await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="late-call",
            payload=harness.payload,
        )
        await harness.executions[0].started.wait()

        self.assertEqual(
            harness.executions[0].kwargs["output_started_ns"],
            900_000_000,
        )
        harness.gate.set()
        await asyncio.sleep(0)

    async def test_exact_duplicate_is_idempotent_and_conflicts_fail_closed(self):
        harness = _Harness()
        first = await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="call-1",
            payload=harness.payload,
        )
        duplicate = await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="call-1",
            payload=harness.payload,
        )

        self.assertEqual(duplicate, first)
        self.assertEqual(len(harness.validations), 1)
        with self.assertRaises(SemanticControllerError):
            await harness.controller.accept_tool_plan(
                turn_id=7,
                call_id="call-1",
                payload={
                    "ordered_functions": ["deny"],
                    "estimated_spoken_duration_s": 5.0,
                },
            )
        with self.assertRaises(SemanticControllerError):
            await harness.controller.accept_tool_plan(
                turn_id=7,
                call_id="call-2",
                payload=harness.payload,
            )

    async def test_reset_rejects_late_tool_and_ignores_stale_activity(self):
        harness = _Harness()
        harness.controller.on_output_activity(turn_id=7)
        harness.controller.reset_turn(previous_turn_id=7)
        harness.bridge.robot_turn_serial = 8

        harness.controller.on_output_activity(turn_id=7)
        self.assertEqual(harness.executions, [])
        with self.assertRaises(SemanticControllerError):
            await harness.controller.accept_tool_plan(
                turn_id=7,
                call_id="late-old-call",
                payload=harness.payload,
            )

    async def test_interrupt_drops_pending_without_cancelling_active_runtime(self):
        harness = _Harness()
        await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="call-1",
            payload=harness.payload,
        )
        harness.controller.on_output_activity(turn_id=7)
        execution = harness.executions[0]
        await execution.started.wait()

        harness.controller.on_interrupted(turn_id=7)

        self.assertEqual(len(execution.interruptions), 1)
        self.assertFalse(execution.kwargs.get("cancelled", False))
        self.assertFalse(execution.started.is_set() and execution.gate.is_set())
        self.assertFalse(harness.controller.has_pending(turn_id=7))
        with self.assertRaises(SemanticControllerError):
            await harness.controller.accept_tool_plan(
                turn_id=7,
                call_id="late-after-interrupt",
                payload=harness.payload,
            )
        harness.gate.set()
        await asyncio.sleep(0)

    async def test_interrupt_signal_precedes_slow_durable_audit(self):
        harness = _Harness()
        await harness.controller.accept_tool_plan(
            turn_id=7,
            call_id="call-slow-audit",
            payload=harness.payload,
        )
        harness.controller.on_output_activity(turn_id=7)
        execution = harness.executions[0]
        await execution.started.wait()
        audit_entered = threading.Event()
        audit_release = threading.Event()

        def slow_audit(payload):
            if payload.get("event") == "turn_invalidated":
                audit_entered.set()
                audit_release.wait(timeout=1.0)

        harness.controller._audit = slow_audit
        loop = asyncio.get_running_loop()
        before = loop.time()
        harness.controller.reset_turn(previous_turn_id=7)
        elapsed = loop.time() - before

        # reset_turn performs only the safety signal on the Live event loop.
        self.assertLess(elapsed, 0.1)
        self.assertEqual(len(execution.interruptions), 1)
        self.assertTrue(
            await asyncio.to_thread(audit_entered.wait, 1.0),
            "slow audit was not reached",
        )

        # The event loop remains responsive while fsync-like audit work blocks.
        tick = asyncio.Event()
        loop.call_soon(tick.set)
        await asyncio.wait_for(tick.wait(), timeout=0.1)
        audit_release.set()
        for task in tuple(harness.controller._background_tasks):
            if "interrupt-audit" in task.get_name():
                await task
        harness.gate.set()
        await asyncio.sleep(0)

    async def test_production_signal_blocks_post_before_deferred_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit_path = Path(temporary) / "semantic.jsonl"
            config = SimpleNamespace(
                semantic_tail_guard_seconds=0.1,
                semantic_sequence_log_path=str(audit_path),
                semantic_execute=True,
                semantic_force_first=False,
                semantic_poll_interval=0.001,
                semantic_action_timeout=1.0,
            )
            client = _BlockingPlanningClient()
            bridge = _Bridge()
            controller = semantic_controller_factory(
                catalog=_catalog(),
                client=client,
                config=config,
            )(bridge)
            response = await controller.accept_tool_plan(
                turn_id=1,
                call_id="interrupt-before-post",
                payload={
                    "ordered_functions": ["emphasize"],
                    "estimated_spoken_duration_s": 5.0,
                },
            )
            self.assertFalse(response["force_first_policy_enabled"])
            controller.on_output_activity(turn_id=1)
            self.assertTrue(
                await asyncio.to_thread(client.state_entered.wait, 1.0),
                "runtime never reached the blocked planning state read",
            )

            state = controller._turns[1]
            execution = state.execution
            self.assertIsNotNone(execution)
            runtime_audit_entered = threading.Event()
            runtime_audit_release = threading.Event()
            controller_audit_entered = threading.Event()
            controller_audit_release = threading.Event()

            def slow_runtime_audit(payload):
                if payload.get("event") == "interrupted":
                    runtime_audit_entered.set()
                    runtime_audit_release.wait(timeout=2.0)

            def slow_controller_audit(payload):
                if payload.get("event") == "turn_invalidated":
                    controller_audit_entered.set()
                    controller_audit_release.wait(timeout=2.0)

            runtime_executor = execution.coordinator.executor
            runtime_executor.audit = slow_runtime_audit
            controller._audit = slow_controller_audit
            original_signal = runtime_executor.signal_interrupt
            signal_seen = threading.Event()

            def observed_signal(request_id, revision, *, reason="interrupt"):
                result = original_signal(request_id, revision, reason=reason)
                if result:
                    signal_seen.set()
                return result

            runtime_executor.signal_interrupt = observed_signal
            loop = asyncio.get_running_loop()
            before = loop.time()
            controller.reset_turn(previous_turn_id=1)
            elapsed = loop.time() - before

            self.assertLess(elapsed, 0.1)
            self.assertTrue(signal_seen.is_set())
            self.assertTrue(execution._interrupt.is_set())
            audit_wait_started = loop.time()
            self.assertTrue(
                await asyncio.to_thread(runtime_audit_entered.wait, 1.0),
                "deferred runtime audit was not started",
            )
            self.assertLess(loop.time() - audit_wait_started, 0.5)
            self.assertFalse(controller_audit_entered.is_set())

            # Let the stale planning read return.  The already-linearized
            # signal must win before the simulator POST boundary, even while
            # the runtime audit is still blocked on durable I/O.
            client.state_release.set()
            await asyncio.sleep(0.05)
            self.assertEqual(client.posts, [])

            runtime_audit_release.set()
            self.assertTrue(
                await asyncio.to_thread(controller_audit_entered.wait, 1.0),
                "controller audit did not follow the runtime interrupt audit",
            )
            self.assertEqual(client.posts, [])
            controller_audit_release.set()
            if state.task is not None:
                await asyncio.wait_for(state.task, timeout=1.0)
            for task in tuple(controller._background_tasks):
                await asyncio.wait_for(task, timeout=1.0)
            self.assertEqual(client.posts, [])

    async def test_admission_replans_discard_attempts_and_commit_history_at_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = SimpleNamespace(
                semantic_tail_guard_seconds=0.1,
                semantic_sequence_log_path=str(Path(temporary) / "semantic.jsonl"),
                semantic_execute=True,
                semantic_force_first=False,
                semantic_poll_interval=0.001,
                semantic_action_timeout=1.0,
            )
            client = _DriftTwiceThenStableClient()
            bridge = _Bridge()
            controller = semantic_controller_factory(
                catalog=_catalog(),
                client=client,
                config=config,
            )(bridge)
            await controller.accept_tool_plan(
                turn_id=1,
                call_id="bounded-admission-replan",
                payload={
                    "ordered_functions": ["emphasize"],
                    "estimated_spoken_duration_s": 5.0,
                },
            )
            controller.on_output_activity(turn_id=1)
            state = controller._turns[1]
            execution = state.execution
            self.assertIsNotNone(execution)
            planner = execution.coordinator.planner
            original_discard = execution.coordinator.executor.discard_plan
            discard_observations = []

            def observed_discard(atom, reason):
                original_discard(atom, reason)
                discard_observations.append(
                    (reason, len(planner._local.selected))
                )

            execution.coordinator.executor.discard_plan = observed_discard
            self.assertTrue(
                await asyncio.to_thread(client.semantic_terminal_entered.wait, 1.0),
                "the stable Atom was not submitted",
            )

            # Two stale selections have been removed, exactly one Atom POST was
            # admitted, and cross-reply history remains uncommitted until the
            # own action reaches terminal and the request closes neutral.
            self.assertEqual(
                discard_observations,
                [("admission_q_drift", 0), ("admission_q_drift", 0)],
            )
            self.assertEqual(len(client.posts), 1)
            self.assertEqual(
                planner._persistent_history.recent_atom_refs,
                (),
            )

            client.semantic_terminal_release.set()
            if state.task is not None:
                await asyncio.wait_for(state.task, timeout=1.0)

            self.assertEqual(len(client.posts), 2)
            self.assertEqual(
                [item["motion"]["name"] for item in client.posts],
                ["approved-emphasize", "canonical-neutral"],
            )
            self.assertEqual(
                len(planner._persistent_history.recent_atom_refs),
                1,
            )
            self.assertEqual(
                planner._persistent_history.recent_source_ids,
                ("fixture:1",),
            )

    async def test_turn_complete_before_output_leaves_tombstone_for_late_tool(self):
        harness = _Harness()
        harness.controller.on_turn_complete(turn_id=7)

        with self.assertRaises(SemanticControllerError):
            await harness.controller.accept_tool_plan(
                turn_id=7,
                call_id="late-after-complete",
                payload=harness.payload,
            )
        harness.controller.on_output_activity(turn_id=7)
        self.assertEqual(harness.executions, [])

    async def test_production_adapter_executes_exact_atom_then_catalog_neutral(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit_path = Path(temporary) / "semantic.jsonl"
            config = SimpleNamespace(
                semantic_tail_guard_seconds=0.1,
                semantic_sequence_log_path=str(audit_path),
                semantic_execute=True,
                semantic_force_first=False,
                semantic_poll_interval=0.001,
                semantic_action_timeout=1.0,
            )
            client = _LiveClient()
            bridge = _Bridge()
            controller = semantic_controller_factory(
                catalog=_catalog(),
                client=client,
                config=config,
            )(bridge)

            response = await controller.accept_tool_plan(
                turn_id=1,
                call_id="production-call",
                payload={
                    "ordered_functions": ["emphasize"],
                    "estimated_spoken_duration_s": 5.0,
                },
            )
            controller.on_output_activity(turn_id=1)
            for _ in range(200):
                if not controller.has_pending(turn_id=1):
                    break
                await asyncio.sleep(0.005)

            self.assertTrue(response["accepted"])
            self.assertEqual(response["execution_mode"], "active")
            self.assertFalse(controller.has_pending(turn_id=1))
            self.assertEqual(len(client.posts), 2)
            self.assertEqual(client.posts[0]["motion"]["name"], "approved-emphasize")
            self.assertEqual(client.posts[1]["motion"]["name"], "canonical-neutral")
            self.assertTrue(all(item["speed"] == 1.0 for item in client.posts))
            self.assertTrue(all(item["require_empty"] for item in client.posts))
            self.assertTrue(all(item["mode"] == "enqueue" for item in client.posts))
            self.assertEqual(bridge.events[-1]["completedFunctions"], ["emphasize"])
            self.assertTrue(bridge.events[-1]["closedNeutral"])
            self.assertFalse(bridge.events[-1]["forcedFirstAtom"])
            self.assertEqual(bridge.events[-1]["plannedOverrunUs"], 0)

            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            accepted = [row for row in records if row["event"] == "tool_plan_accepted"]
            self.assertEqual(len(accepted), 1)
            self.assertFalse(accepted[0]["force_first_policy_enabled"])
            events = [record["event"] for record in records]
            self.assertIn("tool_plan_accepted", events)
            self.assertIn("plan_selected", events)
            self.assertEqual(
                [
                    record["kind"]
                    for record in records
                    if record["event"] == "action_submitted"
                ],
                ["semantic", "recovery"],
            )
            self.assertIn("request_finished", events)

    async def test_force_first_policy_executes_one_closed_atom_and_drops_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit_path = Path(temporary) / "semantic.jsonl"
            config = SimpleNamespace(
                semantic_tail_guard_seconds=0.1,
                semantic_sequence_log_path=str(audit_path),
                semantic_execute=True,
                semantic_force_first=True,
                semantic_poll_interval=0.001,
                semantic_action_timeout=1.0,
            )
            client = _LiveClient()
            bridge = _Bridge()
            controller = semantic_controller_factory(
                catalog=_catalog(),
                client=client,
                config=config,
            )(bridge)

            response = await controller.accept_tool_plan(
                turn_id=1,
                call_id="force-one-short-reply",
                payload={
                    "ordered_functions": ["emphasize", "emphasize"],
                    "estimated_spoken_duration_s": 0.1,
                },
            )
            self.assertTrue(response["force_first_policy_enabled"])
            controller.on_output_activity(turn_id=1)
            state = controller._turns[1]
            if state.task is not None:
                await asyncio.wait_for(state.task, timeout=1.0)

            self.assertEqual(
                [item["motion"]["name"] for item in client.posts],
                ["approved-emphasize", "canonical-neutral"],
            )
            event = bridge.events[-1]
            self.assertEqual(event["status"], "forced_first_completed")
            self.assertTrue(event["forceFirstPolicyEnabled"])
            self.assertTrue(event["wouldForceFirstAtom"])
            self.assertTrue(event["forcedFirstAtom"])
            self.assertGreater(event["plannedOverrunUs"], 0)
            self.assertEqual(event["plannedFunctions"], ["emphasize"])
            self.assertEqual(event["completedFunctions"], ["emphasize"])
            self.assertEqual(event["droppedFunctions"], ["emphasize"])
            self.assertTrue(event["closedNeutral"])

            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            accepted = [row for row in records if row["event"] == "tool_plan_accepted"]
            self.assertEqual(len(accepted), 1)
            self.assertTrue(accepted[0]["force_first_policy_enabled"])
            self.assertEqual(
                len([row for row in records if row["event"] == "force_one_precommit"]),
                1,
            )
            finished = [row for row in records if row["event"] == "request_finished"]
            self.assertEqual(len(finished), 1)
            self.assertTrue(finished[0]["forced_first_atom"])
            self.assertEqual(
                finished[0]["planned_overrun_us"],
                event["plannedOverrunUs"],
            )

    async def test_force_first_policy_off_preserves_budget_no_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit_path = Path(temporary) / "semantic.jsonl"
            config = SimpleNamespace(
                semantic_tail_guard_seconds=0.1,
                semantic_sequence_log_path=str(audit_path),
                semantic_execute=True,
                semantic_force_first=False,
                semantic_poll_interval=0.001,
                semantic_action_timeout=1.0,
            )
            client = _LiveClient()
            bridge = _Bridge()
            controller = semantic_controller_factory(
                catalog=_catalog(),
                client=client,
                config=config,
            )(bridge)

            await controller.accept_tool_plan(
                turn_id=1,
                call_id="force-one-disabled",
                payload={
                    "ordered_functions": ["emphasize", "emphasize"],
                    "estimated_spoken_duration_s": 0.1,
                },
            )
            controller.on_output_activity(turn_id=1)
            state = controller._turns[1]
            if state.task is not None:
                await asyncio.wait_for(state.task, timeout=1.0)

            self.assertEqual(client.posts, [])
            event = bridge.events[-1]
            self.assertEqual(event["status"], "no_plan")
            self.assertFalse(event["forceFirstPolicyEnabled"])
            self.assertFalse(event["wouldForceFirstAtom"])
            self.assertFalse(event["forcedFirstAtom"])
            self.assertEqual(event["plannedOverrunUs"], 0)
            self.assertEqual(event["droppedFunctions"], ["emphasize", "emphasize"])
            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotIn("force_one_requested", [row["event"] for row in records])

    def test_durable_audit_appends_canonical_fsynced_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            audit = DurableSemanticAudit(path)
            audit({"schema_version": "test", "event": "one", "value": 1})
            audit({"schema_version": "test", "event": "two", "value": 2})

            raw = path.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            rows = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual([row["event"] for row in rows], ["one", "two"])
            self.assertTrue(all("timestamp_utc" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
