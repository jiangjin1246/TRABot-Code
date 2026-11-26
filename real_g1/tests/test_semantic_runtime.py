from __future__ import annotations

import math
import threading
import unittest

from web_overlay.reply_clock import ReplyClock, ReplyClockError
from web_overlay.semantic_runtime import (
    ADMISSION_Q_DRIFT_REASON,
    MAX_ADMISSION_REPLAN_ATTEMPTS,
    RUNTIME_FUNCTIONS,
    AuditCallbackError,
    SemanticActionStep,
    SemanticConflictError,
    SemanticExecutionError,
    SemanticRecoveryStep,
    SemanticSuffixExecutor,
    ToolPayloadValidationError,
    canonical_action_payload_sha256,
    gemini_tool_parameters_schema,
    validate_gemini_tool_payload,
)


class _Monotonic:
    def __init__(self, value_ns: int = 1_000_000_000):
        self.value_ns = value_ns

    def __call__(self) -> int:
        return self.value_ns

    def advance(self, seconds: float) -> None:
        self.value_ns += int(seconds * 1_000_000_000)


class _Plan:
    def __init__(self, atoms):
        self.atoms = tuple(atoms)
        self.prefix_length = len(self.atoms)
        self.remaining_functions = ()


class _Atom:
    def __init__(self, function: str, serial: int):
        self.function = function
        self.serial = serial


class _Planner:
    def __init__(self):
        self.calls = []
        self.poison = object()

    def __call__(self, remaining_functions, remaining_budget_us, previous_atom, history):
        self.calls.append(
            (
                tuple(remaining_functions),
                remaining_budget_us,
                previous_atom,
                tuple(history),
            )
        )
        if not remaining_functions:
            return _Plan(())
        atom = _Atom(remaining_functions[0], len(self.calls))
        # The executor must never prepare or submit this stale suffix element.
        return _Plan((atom, self.poison))


class _Client:
    def __init__(self, statuses=("scheduled", "completed")):
        self.statuses = tuple(statuses)
        self.posts = []
        self.reads = []
        self.state_reads = 0
        self.action_reads = {}
        self.forbidden_calls = []

    def state_sync(self):
        self.state_reads += 1
        return {
            "action": None,
            "queued": [],
            "action_seq": len(self.posts),
            "fault": None,
            "q": {"left_elbow": 0.0},
        }

    def post_action_sync(self, payload):
        body = dict(payload)
        self.posts.append(body)
        return {"accepted": True, "id": body["id"], "queue_depth": 1}

    def action_sync(self, action_id):
        self.reads.append(action_id)
        reads = self.action_reads.get(action_id, 0)
        self.action_reads[action_id] = reads + 1
        index = min(reads, len(self.statuses) - 1)
        payload = next(item for item in self.posts if item["id"] == action_id)
        return {
            "id": action_id,
            "status": self.statuses[index],
            "payload_sha256": payload["payload_sha256"],
        }

    def stop_action_sync(self):
        self.forbidden_calls.append("stop")

    def release_sync(self):
        self.forbidden_calls.append("release")


class _SemanticHookClient(_Client):
    def __init__(self, statuses=("completed",)):
        super().__init__(statuses=statuses)
        self.semantic_posts = []
        self.fallback_posts = 0

    def post_action_sync(self, payload):
        self.fallback_posts += 1
        raise AssertionError("semantic hook must take precedence over fallback POST")

    def post_semantic_action_sync(self, payload, *, kind, atom_ref):
        body = dict(payload)
        self.posts.append(body)
        self.semantic_posts.append((kind, dict(atom_ref), body))
        return {
            "accepted": True,
            "id": body["id"],
            "payload_sha256": body["payload_sha256"],
        }


class _Http409(RuntimeError):
    status_code = 409


class _ConflictClient(_Client):
    def post_action_sync(self, payload):
        self.posts.append(dict(payload))
        raise _Http409("HTTP 409 action admission conflict")

    def action_sync(self, action_id):
        self.reads.append(action_id)
        raise RuntimeError("HTTP 404 action_not_found")


class _BlockingStateClient(_Client):
    def __init__(self):
        super().__init__(statuses=("completed",))
        self.state_entered = threading.Event()
        self.state_release = threading.Event()

    def state_sync(self):
        self.state_entered.set()
        self.state_release.wait(timeout=1.0)
        return super().state_sync()


class _ResetOnStateReadClient(_Client):
    def __init__(self, fail_on_read):
        super().__init__(statuses=("completed",))
        self.fail_on_read = fail_on_read
        self.state_entered = threading.Event()
        self.state_release = threading.Event()

    def state_sync(self):
        next_read = self.state_reads + 1
        if next_read == self.fail_on_read:
            self.state_reads = next_read
            self.state_entered.set()
            self.state_release.wait(timeout=1.0)
            self.fail_on_read = None
            raise ConnectionResetError("native session reset during snapshot")
        return super().state_sync()


class _BlockingPostClient(_Client):
    def __init__(self):
        super().__init__(statuses=("completed",))
        self.post_entered = threading.Event()
        self.post_release = threading.Event()

    def post_action_sync(self, payload):
        body = dict(payload)
        self.posts.append(body)
        self.post_entered.set()
        self.post_release.wait(timeout=1.0)
        return {"accepted": True, "id": body["id"], "queue_depth": 1}


class _InvalidPostResponseClient(_Client):
    def __init__(self):
        super().__init__(statuses=("scheduled", "completed"))

    def post_action_sync(self, payload):
        body = dict(payload)
        self.posts.append(body)
        # Simulate a server/proxy contract failure after durable admission.
        return {"accepted": False, "id": body["id"]}


class _EventuallyVisibleResponseLossClient(_Client):
    def __init__(self):
        super().__init__(statuses=("scheduled", "completed"))
        self.missing_reads = 2

    def post_action_sync(self, payload):
        body = dict(payload)
        self.posts.append(body)
        if len(self.posts) == 1:
            raise RuntimeError("transport timeout after request body was sent")
        return {"accepted": True, "id": body["id"], "queue_depth": 1}

    def action_sync(self, action_id):
        if self.missing_reads:
            self.missing_reads -= 1
            self.reads.append(action_id)
            raise RuntimeError("HTTP 404 action_not_yet_visible")
        return super().action_sync(action_id)


class _BadActionHashClient(_Client):
    def action_sync(self, action_id):
        record = super().action_sync(action_id)
        record["payload_sha256"] = "0" * 64
        return record


class _FailRecoveryClient(_Client):
    def __init__(self):
        super().__init__(statuses=("completed",))

    def action_sync(self, action_id):
        record = super().action_sync(action_id)
        if len(self.posts) >= 2 and action_id == self.posts[1]["id"]:
            record["status"] = "failed"
        return record


class _TransientActionReadClient(_Client):
    def __init__(self):
        super().__init__(statuses=("scheduled", "completed"))
        self.transient_failures_remaining = 1
        self.transient_read_failures = 0

    def action_sync(self, action_id):
        if self.transient_failures_remaining:
            self.transient_failures_remaining -= 1
            self.transient_read_failures += 1
            raise RuntimeError("temporary observation transport error")
        return super().action_sync(action_id)


class _DriftingQClient(_Client):
    def __init__(self):
        super().__init__(statuses=("completed",))
        self._q_by_state_read = (
            0.10,  # first planning snapshot
            0.10,  # first fresh admission snapshot
            0.70,  # second planning snapshot after the first Atom completed
            0.70,  # second fresh admission snapshot
            0.72,  # neutral recovery admission snapshot
        )

    def state_sync(self):
        state = super().state_sync()
        index = min(self.state_reads - 1, len(self._q_by_state_read) - 1)
        state["q"] = {"left_elbow": self._q_by_state_read[index]}
        return state


class _AdmissionQDriftClient(_Client):
    def state_sync(self):
        state = super().state_sync()
        state["q"] = {
            "left_elbow": 0.10 if self.state_reads == 1 else 0.11
        }
        return state


class _AdmissionMicroDriftClient(_Client):
    def state_sync(self):
        state = super().state_sync()
        state["q"] = {"left_elbow": 0.10 + self.state_reads * 1e-6}
        return state


class _NonConvergentQClient(_Client):
    def __init__(self):
        super().__init__(statuses=("completed",))
        self.unstable = True

    def state_sync(self):
        state = super().state_sync()
        state["q"] = {
            "left_elbow": (
                0.10 * self.state_reads if self.unstable else 0.77
            )
        }
        return state


class _SecondAtomNonConvergentQClient(_Client):
    def __init__(self):
        super().__init__(statuses=("completed",))

    def state_sync(self):
        state = super().state_sync()
        state["q"] = {
            "left_elbow": (
                0.0
                if self.state_reads <= 2
                else 0.10 * (self.state_reads - 2)
            )
        }
        return state


def _prepare(atom):
    if not isinstance(atom, _Atom):
        raise AssertionError("executor touched a non-first stale planner step")
    return SemanticActionStep(
        plan_id=f"plan-{atom.serial}",
        function=atom.function,
        atom_ref={"atom_id": f"atom-{atom.serial}", "revision": 1},
        action_payload={
            "motion": {
                "format": "g1_joint_motion_v1",
                "joints": ["left_elbow"],
                "frames": [
                    {"t": 0.0, "q": {"left_elbow": 0.0}},
                    {"t": 0.1, "q": {"left_elbow": 0.1}},
                ],
            },
            "speed": 1.0,
            "blend_duration": 0.0,
            "bridge_policy": "minimum_jerk_v1",
        },
        estimated_duration_us=200_000,
    )


def _prepare_recovery(previous_atom, remaining_budget_us, history):
    if previous_atom is None or tuple(history)[-1] is not previous_atom:
        raise AssertionError("recovery did not receive the completed prefix tail")
    return SemanticRecoveryStep(
        recovery_id=f"neutral-after-{previous_atom.serial}",
        recovery_ref={"target": "safe_neutral", "after": previous_atom.serial},
        action_payload={
            "motion": {
                "format": "g1_joint_motion_v1",
                "joints": ["left_elbow"],
                "frames": [
                    {"t": 0.0, "q": {"left_elbow": 0.1}},
                    {"t": 0.1, "q": {"left_elbow": 0.0}},
                ],
            },
            "speed": 1.0,
            "blend_duration": 0.0,
            "bridge_policy": "minimum_jerk_v1",
        },
        estimated_duration_us=150_000,
    )


class ToolPayloadTest(unittest.TestCase):
    def test_runtime_enum_is_the_ordered_retained_f17_set(self):
        self.assertEqual(
            RUNTIME_FUNCTIONS,
            (
                "emphasize",
                "enumerate",
                "contrast_compare",
                "refer_self",
                "refer_other",
                "refer_external",
                "depict_size_shape",
                "depict_direction_motion",
                "turn_open",
                "turn_yield",
                "greet_farewell",
                "thank",
                "affirm",
                "deny",
                "certainty",
                "uncertainty",
                "enthusiasm",
            ),
        )
        self.assertEqual(len(RUNTIME_FUNCTIONS), 17)
        self.assertNotIn("present_explain", RUNTIME_FUNCTIONS)

    def test_schema_and_validator_use_only_the_two_model_owned_fields(self):
        schema = gemini_tool_parameters_schema()
        self.assertEqual(
            schema["required"],
            ["ordered_functions", "estimated_spoken_duration_s"],
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            tuple(schema["properties"]["ordered_functions"]["items"]["enum"]),
            RUNTIME_FUNCTIONS,
        )
        self.assertNotIn("maxItems", schema["properties"]["ordered_functions"])
        self.assertEqual(
            schema["properties"]["estimated_spoken_duration_s"]["minimum"], 0.1
        )
        self.assertNotIn(
            "maximum", schema["properties"]["estimated_spoken_duration_s"]
        )
        result = validate_gemini_tool_payload(
            {
                "ordered_functions": ["emphasize", "deny", "emphasize"],
                "estimated_spoken_duration_s": 1.0000001,
            },
            request_id="turn-7",
            revision=3,
        )
        self.assertEqual(result.request_id, "turn-7")
        self.assertEqual(result.revision, 3)
        self.assertEqual(
            result.ordered_functions, ("emphasize", "deny", "emphasize")
        )
        self.assertEqual(result.estimated_spoken_duration_us, 1_000_001)

    def test_payload_validation_is_strict_and_fail_closed(self):
        valid = {
            "ordered_functions": ["emphasize"],
            "estimated_spoken_duration_s": 2.5,
        }
        invalid = [
            ({**valid, "request_id": "model-spoof"}, "exactly"),
            ({"ordered_functions": ["present_explain"], "estimated_spoken_duration_s": 1}, "function"),
            ({"ordered_functions": "emphasize", "estimated_spoken_duration_s": 1}, "array"),
            ({"ordered_functions": ["emphasize"], "estimated_spoken_duration_s": True}, "duration"),
            ({"ordered_functions": ["emphasize"], "estimated_spoken_duration_s": math.inf}, "finite"),
            ({"ordered_functions": ["emphasize"], "estimated_spoken_duration_s": 0}, "at least"),
            ({"ordered_functions": ["emphasize"], "estimated_spoken_duration_s": 0.09}, "at least"),
            ({"ordered_functions": ["emphasize"], "estimated_spoken_duration_s": 1e20}, "clock range"),
            ({"ordered_functions": ["emphasize"], "estimated_spoken_duration_s": 10**400}, "clock range"),
        ]
        for raw, message in invalid:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ToolPayloadValidationError, message):
                    validate_gemini_tool_payload(raw, request_id="turn-1", revision=1)
        with self.assertRaisesRegex(ToolPayloadValidationError, "request_id"):
            validate_gemini_tool_payload(valid, request_id="bad request", revision=1)
        with self.assertRaisesRegex(ToolPayloadValidationError, "revision"):
            validate_gemini_tool_payload(valid, request_id="turn-1", revision=True)

        long_reply = validate_gemini_tool_payload(
            {
                "ordered_functions": ["emphasize"] * 13,
                "estimated_spoken_duration_s": 3_600,
            },
            request_id="turn-long",
            revision=1,
        )
        self.assertEqual(len(long_reply.ordered_functions), 13)
        self.assertEqual(long_reply.estimated_spoken_duration_us, 3_600_000_000)


class ReplyClockTest(unittest.TestCase):
    def test_snapshot_and_remaining_are_monotonic_and_clamped(self):
        monotonic = _Monotonic()
        clock = ReplyClock(
            request_id="turn-1",
            revision=2,
            duration_us=1_000_000,
            monotonic_ns=monotonic,
        )
        first = clock.snapshot()
        self.assertEqual(first.elapsed_us, 0)
        self.assertEqual(first.remaining_us, 1_000_000)
        self.assertFalse(first.expired)
        monotonic.advance(0.25)
        second = clock.snapshot()
        self.assertEqual(second.elapsed_us, 250_000)
        self.assertEqual(second.remaining_us, 750_000)
        self.assertAlmostEqual(clock.remaining_seconds(), 0.75)
        monotonic.advance(1.0)
        final = clock.snapshot()
        self.assertEqual(final.remaining_us, 0)
        self.assertTrue(final.expired)

    def test_clock_rejects_binding_mismatch_and_backward_observation(self):
        monotonic = _Monotonic()
        clock = ReplyClock("turn-1", 2, 100_000, monotonic_ns=monotonic)
        clock.assert_binding("turn-1", 2, 100_000)
        with self.assertRaisesRegex(ReplyClockError, "binding"):
            clock.assert_binding("turn-1", 3, 100_000)
        monotonic.advance(0.01)
        clock.snapshot()
        monotonic.value_ns -= 1
        with self.assertRaisesRegex(ReplyClockError, "backwards"):
            clock.snapshot()

    def test_late_tool_uses_frozen_output_start_and_consumes_elapsed_budget(self):
        monotonic = _Monotonic()
        output_started_ns = monotonic.value_ns
        monotonic.advance(2.0)
        payload = validate_gemini_tool_payload(
            {
                "ordered_functions": ["emphasize"],
                "estimated_spoken_duration_s": 5.0,
            },
            request_id="turn-late-tool",
            revision=1,
        )
        clock = ReplyClock.from_payload(
            payload,
            monotonic_ns=monotonic,
            started_monotonic_ns=output_started_ns,
        )
        snapshot = clock.snapshot()
        self.assertEqual(snapshot.started_monotonic_ns, output_started_ns)
        self.assertEqual(snapshot.elapsed_us, 2_000_000)
        self.assertEqual(snapshot.remaining_us, 3_000_000)
        with self.assertRaisesRegex(ReplyClockError, "future"):
            ReplyClock.from_payload(
                payload,
                monotonic_ns=monotonic,
                started_monotonic_ns=monotonic.value_ns + 1,
            )


class SemanticSuffixExecutorTest(unittest.TestCase):
    def _request(
        self,
        *,
        request_id="turn-1",
        revision=1,
        functions=("emphasize", "deny"),
        seconds=5.0,
    ):
        return validate_gemini_tool_payload(
            {
                "ordered_functions": list(functions),
                "estimated_spoken_duration_s": seconds,
            },
            request_id=request_id,
            revision=revision,
        )

    @staticmethod
    def _execute_live(executor, request):
        clock = ReplyClock.from_payload(
            request,
            monotonic_ns=executor._monotonic_ns,
        )
        return executor.execute(request, reply_clock=clock)

    def test_default_dry_run_plans_only_first_step_and_never_calls_client(self):
        planner = _Planner()
        client = _Client()
        audits = []
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            audit=audits.append,
        )
        result = executor.execute(self._request())
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.planned_functions, ("emphasize",))
        self.assertEqual(result.completed_functions, ())
        self.assertEqual(result.pending_functions, ("emphasize", "deny"))
        self.assertEqual(client.posts, [])
        self.assertEqual(client.reads, [])
        self.assertEqual(len(planner.calls), 1)
        self.assertIn("dry_run", [event["event"] for event in audits])

    def test_live_executor_submits_one_then_replans_suffix_after_own_terminal(self):
        monotonic = _Monotonic()
        planner = _Planner()
        client = _Client()
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
            sleep=lambda seconds: monotonic.advance(seconds),
            poll_interval_s=0.01,
        )
        request = self._request()
        clock = ReplyClock.from_payload(request, monotonic_ns=monotonic)
        result = executor.execute(request, reply_clock=clock)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.completed_functions, ("emphasize", "deny"))
        self.assertEqual(result.pending_functions, ())
        self.assertEqual(len(client.posts), 3)
        self.assertTrue(all(item["mode"] == "enqueue" for item in client.posts))
        self.assertTrue(all("id" in item for item in client.posts))
        self.assertEqual(len(set(item["id"] for item in client.posts)), 3)
        self.assertNotIn("replace", [item.get("mode") for item in client.posts])
        self.assertEqual(client.forbidden_calls, [])
        self.assertEqual(client.state_reads, 5)
        for index, item in enumerate(client.posts):
            self.assertTrue(item["require_empty"])
            self.assertEqual(item["expected_action_seq"], index)
            self.assertRegex(item["exclusive_owner"], r"^semturn-[0-9a-f]{40}$")
            self.assertEqual(item["bridge_policy"], "minimum_jerk_v1")
            self.assertEqual(
                item["payload_sha256"], canonical_action_payload_sha256(item)
            )
        self.assertEqual(len({item["exclusive_owner"] for item in client.posts}), 1)
        self.assertEqual([call[0] for call in planner.calls], [("emphasize", "deny"), ("deny",)])
        self.assertEqual(planner.calls[0][2], {"q": {"left_elbow": 0.0}})
        self.assertEqual(planner.calls[1][2], {"q": {"left_elbow": 0.0}})
        self.assertEqual(tuple(atom.function for atom in planner.calls[1][3]), ("emphasize",))
        self.assertTrue(all(read in result.action_ids for read in client.reads))
        self.assertEqual(result.recovery_action_id, client.posts[-1]["id"])
        self.assertTrue(result.closed_neutral)

    def test_optional_semantic_post_hook_receives_kind_and_exact_reference(self):
        client = _SemanticHookClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )

        result = self._execute_live(
            executor,
            self._request(functions=("affirm",)),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(client.fallback_posts, 0)
        self.assertEqual(
            [kind for kind, _atom_ref, _body in client.semantic_posts],
            ["semantic", "recovery"],
        )
        self.assertEqual(
            client.semantic_posts[0][1],
            {"atom_id": "atom-1", "revision": 1},
        )
        self.assertEqual(
            client.semantic_posts[1][1],
            {"target": "safe_neutral", "after": 1},
        )

    def test_each_suffix_replan_uses_a_fresh_measured_q_snapshot(self):
        planner = _Planner()
        client = _DriftingQClient()
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            sleep=lambda _: None,
        )
        result = self._execute_live(executor, self._request())
        self.assertEqual(result.status, "completed")
        self.assertEqual(
            [call[2] for call in planner.calls],
            [
                {"q": {"left_elbow": 0.10}},
                {"q": {"left_elbow": 0.70}},
            ],
        )
        self.assertEqual(client.state_reads, 5)

    def test_q_drift_discards_stale_plan_and_replans_from_admission_snapshot(self):
        client = _AdmissionQDriftClient(statuses=("completed",))
        planner = _Planner()
        discarded = []
        audits = []
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            discard_plan=lambda atom, reason: discarded.append((atom, reason)),
            exclusive_owner="semantic-test-owner",
            audit=audits.append,
        )
        result = self._execute_live(
            executor, self._request(functions=("affirm",))
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.planned_functions, ("affirm",))
        self.assertEqual(result.completed_functions, ("affirm",))
        self.assertEqual(len(client.posts), 2)  # fresh Atom, then neutral recovery
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(planner.calls[0][2], {"q": {"left_elbow": 0.10}})
        self.assertEqual(planner.calls[1][2], {"q": {"left_elbow": 0.11}})
        self.assertEqual(len(discarded), 1)
        self.assertEqual(discarded[0][0].serial, 1)
        self.assertEqual(discarded[0][0].function, "affirm")
        self.assertEqual(discarded[0][1], ADMISSION_Q_DRIFT_REASON)
        discarded_events = [event for event in audits if event["event"] == "plan_discarded"]
        self.assertEqual(len(discarded_events), 1)
        self.assertEqual(discarded_events[0]["reason"], ADMISSION_Q_DRIFT_REASON)
        self.assertFalse(executor.status()["safe_stopped"])

    def test_nonconvergent_q_skips_turn_without_post_or_permanent_safe_stop(self):
        client = _NonConvergentQClient()
        planner = _Planner()
        discarded = []
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            discard_plan=lambda atom, reason: discarded.append((atom, reason)),
            exclusive_owner="semantic-test-owner",
        )
        first = self._execute_live(
            executor, self._request(functions=("affirm",))
        )
        self.assertEqual(first.status, "no_stable_admission")
        self.assertEqual(first.planned_functions, ())
        self.assertEqual(first.completed_functions, ())
        self.assertEqual(first.dropped_functions, ("affirm",))
        self.assertTrue(first.closed_neutral)
        self.assertEqual(client.posts, [])
        self.assertEqual(len(planner.calls), MAX_ADMISSION_REPLAN_ATTEMPTS + 1)
        self.assertEqual(len(discarded), MAX_ADMISSION_REPLAN_ATTEMPTS + 1)
        self.assertTrue(
            all(reason == ADMISSION_Q_DRIFT_REASON for _, reason in discarded)
        )
        self.assertFalse(executor.status()["safe_stopped"])

        # Instability is scoped to one turn; a later stable turn remains usable.
        client.unstable = False
        second = self._execute_live(
            executor,
            self._request(
                request_id="turn-2", functions=("deny",)
            ),
        )
        self.assertEqual(second.status, "completed")
        self.assertEqual(second.completed_functions, ("deny",))
        self.assertEqual(len(client.posts), 2)
        self.assertFalse(executor.status()["safe_stopped"])

    def test_nonconvergent_next_atom_closes_completed_prefix_with_recovery(self):
        client = _SecondAtomNonConvergentQClient()
        discarded = []
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            discard_plan=lambda atom, reason: discarded.append((atom, reason)),
            exclusive_owner="semantic-test-owner",
        )
        result = self._execute_live(executor, self._request())
        self.assertEqual(result.status, "partial_unstable_admission")
        self.assertEqual(result.planned_functions, ("emphasize",))
        self.assertEqual(result.completed_functions, ("emphasize",))
        self.assertEqual(result.dropped_functions, ("deny",))
        self.assertTrue(result.closed_neutral)
        self.assertEqual(len(client.posts), 2)
        self.assertEqual(result.action_ids, tuple(item["id"] for item in client.posts))
        self.assertEqual(result.recovery_action_id, client.posts[-1]["id"])
        self.assertEqual(
            client.posts[-1]["motion"]["frames"][-1]["q"]["left_elbow"],
            0.0,
        )
        self.assertEqual(len(discarded), MAX_ADMISSION_REPLAN_ATTEMPTS + 1)
        self.assertFalse(executor.status()["safe_stopped"])

    def test_micro_q_drift_reserves_one_tick_and_executes(self):
        client = _AdmissionMicroDriftClient(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        result = self._execute_live(
            executor, self._request(functions=("affirm",))
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(client.posts), 2)
        self.assertTrue(result.closed_neutral)

    def test_live_mode_requires_the_frozen_reply_clock(self):
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=_Client(statuses=("completed",)),
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "explicit ReplyClock"):
            executor.execute(self._request(functions=("affirm",)))
        self.assertEqual(executor.client.posts, [])

    def test_force_one_executes_and_closes_only_the_first_function_over_budget(self):
        monotonic = _Monotonic()
        planner_calls = []
        force_calls = []
        recovery_budgets = []
        audits = []

        def no_fit(remaining, budget, previous, history):
            planner_calls.append((remaining, budget, previous, history))
            return _Plan(())

        def force_first(remaining, previous, history):
            force_calls.append((remaining, previous, history))
            return _Plan((_Atom(remaining[0], 101),))

        def forced_recovery(previous, remaining_budget_us, history):
            recovery_budgets.append(remaining_budget_us)
            return _prepare_recovery(previous, 1, history)

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            no_fit,
            _prepare,
            force_first_plan=force_first,
            client=client,
            dry_run=False,
            prepare_recovery=forced_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audits.append,
            monotonic_ns=monotonic,
        )
        request = self._request(functions=("affirm", "deny"), seconds=0.1)
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )

        self.assertEqual(result.status, "forced_first_completed")
        self.assertTrue(result.forced_first_atom)
        self.assertEqual(result.planned_overrun_us, 290_000)
        self.assertEqual(result.completed_functions, ("affirm",))
        self.assertEqual(result.dropped_functions, ("deny",))
        self.assertTrue(result.closed_neutral)
        self.assertEqual(len(client.posts), 2)
        self.assertEqual(result.recovery_action_id, client.posts[-1]["id"])
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(len(force_calls), 1)
        self.assertEqual(force_calls[0][0], ("affirm", "deny"))
        self.assertEqual(force_calls[0][1], {"q": {"left_elbow": 0.0}})
        self.assertEqual(force_calls[0][2], ())
        self.assertEqual(recovery_budgets, [None])
        precommit = [event for event in audits if event["event"] == "force_one_precommit"]
        self.assertEqual(len(precommit), 1)
        self.assertTrue(precommit[0]["would_force"])
        self.assertTrue(precommit[0]["forced_overrun"])
        self.assertEqual(precommit[0]["closed_cost_us"], 390_000)
        self.assertEqual(precommit[0]["decision_budget_us"], 100_000)
        self.assertEqual(precommit[0]["planned_overrun_us"], 290_000)

    def test_normally_planned_first_atom_converts_to_force_after_prepare_expiry(self):
        monotonic = _Monotonic()
        planner = _Planner()
        force_calls = []

        def should_not_reselect(*args):
            force_calls.append(args)
            raise AssertionError("the already selected first Atom must be retained")

        def slow_prepare(atom):
            step = _prepare(atom)
            monotonic.advance(0.2)
            return step

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            planner,
            slow_prepare,
            force_first_plan=should_not_reselect,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
        )
        request = self._request(functions=("affirm", "deny"), seconds=0.1)
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )

        self.assertEqual(result.status, "forced_first_completed")
        self.assertTrue(result.forced_first_atom)
        self.assertEqual(result.planned_overrun_us, 390_000)
        self.assertEqual(result.completed_functions, ("affirm",))
        self.assertEqual(result.dropped_functions, ("deny",))
        self.assertEqual(force_calls, [])
        self.assertEqual(len(client.posts), 2)

    def test_force_one_no_candidate_is_distinct_and_posts_nothing(self):
        audits = []
        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda *args: _Plan(()),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audits.append,
        )
        result = self._execute_live(
            executor, self._request(functions=("thank",), seconds=0.1)
        )
        self.assertEqual(result.status, "no_candidate")
        self.assertFalse(result.forced_first_atom)
        self.assertEqual(result.planned_overrun_us, 0)
        self.assertTrue(result.closed_neutral)
        self.assertEqual(client.posts, [])
        decisions = [event for event in audits if event["event"] == "force_one_decision"]
        self.assertEqual(len(decisions), 1)
        self.assertFalse(decisions[0]["would_force"])
        self.assertEqual(decisions[0]["reason"], "no_candidate")
        self.assertFalse(executor.status()["safe_stopped"])

    def test_interrupt_after_force_selection_still_wins_before_any_post(self):
        holder = {}

        def interrupting_prepare(atom):
            holder["executor"].signal_interrupt(
                "turn-1", 1, reason="barge_in_before_forced_post"
            )
            return _prepare(atom)

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            interrupting_prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 301),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        holder["executor"] = executor
        result = self._execute_live(
            executor, self._request(functions=("affirm",), seconds=0.1)
        )
        self.assertEqual(result.status, "interrupted")
        self.assertFalse(result.forced_first_atom)
        self.assertEqual(client.posts, [])

    def test_interrupt_during_force_precommit_audit_wins_before_any_post(self):
        audit_entered = threading.Event()
        audit_release = threading.Event()
        holder = {}

        def blocking_audit(event):
            if event["event"] == "force_one_precommit":
                audit_entered.set()
                audit_release.wait(timeout=1.0)

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 351),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=blocking_audit,
        )
        request = self._request(functions=("affirm",), seconds=0.1)

        def run():
            holder["result"] = self._execute_live(executor, request)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(audit_entered.wait(timeout=1.0))
        self.assertEqual(client.posts, [])
        self.assertTrue(
            executor.signal_interrupt(
                "turn-1", 1, reason="barge_in_during_force_precommit_audit"
            )
        )
        audit_release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        result = holder["result"]
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(result.completed_functions, ())
        self.assertFalse(result.forced_first_atom)
        self.assertEqual(client.posts, [])

    def test_force_one_keeps_atomic_admission_conflicts_fail_closed(self):
        client = _ConflictClient()
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 401),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticConflictError, "HTTP 409"):
            self._execute_live(
                executor, self._request(functions=("affirm",), seconds=0.1)
            )
        self.assertEqual(len(client.posts), 1)
        self.assertTrue(executor.status()["safe_stopped"])

    def test_force_one_converts_same_atom_when_submit_intent_crosses_deadline(self):
        monotonic = _Monotonic()
        audits = []
        force_calls = []

        def audit(event):
            audits.append(event)
            if event["event"] == "submit_intent" and not event["force_one"]:
                monotonic.advance(0.2)

        def should_not_reselect(*args):
            force_calls.append(args)
            raise AssertionError("precommit conversion must retain the selected Atom")

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            force_first_plan=should_not_reselect,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audit,
            monotonic_ns=monotonic,
        )
        request = self._request(functions=("affirm", "deny"), seconds=0.4)
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )
        self.assertEqual(result.status, "forced_first_completed")
        self.assertTrue(result.forced_first_atom)
        self.assertEqual(result.planned_overrun_us, 190_000)
        self.assertEqual(force_calls, [])
        self.assertEqual(len(client.posts), 2)
        intents = [event for event in audits if event["event"] == "submit_intent"]
        self.assertEqual([event["force_one"] for event in intents], [False, True, False])
        forced_precommit = [
            event for event in audits if event["event"] == "force_one_precommit"
        ]
        self.assertEqual(len(forced_precommit), 1)
        self.assertEqual(forced_precommit[0]["function"], "affirm")
        self.assertEqual(forced_precommit[0]["closed_cost_us"], 390_000)
        self.assertEqual(forced_precommit[0]["decision_budget_us"], 200_000)

    def test_force_one_action_terminal_failure_never_blindly_recovers(self):
        client = _Client(statuses=("failed",))
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 451),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "status failed"):
            self._execute_live(
                executor, self._request(functions=("affirm",), seconds=0.1)
            )
        self.assertEqual(len(client.posts), 1)
        self.assertTrue(executor.status()["safe_stopped"])

    def test_force_one_recovery_failure_is_not_reported_as_closed(self):
        client = _FailRecoveryClient()
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 461),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "status failed"):
            self._execute_live(
                executor, self._request(functions=("affirm",), seconds=0.1)
            )
        self.assertEqual(len(client.posts), 2)
        self.assertTrue(executor.status()["safe_stopped"])

    def test_force_one_dry_run_reports_would_force_without_posting(self):
        monotonic = _Monotonic()
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 501),)
            ),
            prepare_recovery=_prepare_recovery,
            monotonic_ns=monotonic,
        )
        result = executor.execute(
            self._request(functions=("affirm", "deny"), seconds=0.1)
        )
        self.assertEqual(result.status, "dry_run_forced_first")
        self.assertTrue(result.would_force_first_atom)
        self.assertFalse(result.forced_first_atom)
        self.assertEqual(result.planned_overrun_us, 290_000)
        self.assertEqual(result.planned_functions, ("affirm",))
        self.assertEqual(result.completed_functions, ())
        self.assertEqual(result.pending_functions, ("affirm",))
        self.assertEqual(result.dropped_functions, ("deny",))
        self.assertFalse(
            set(result.pending_functions) & set(result.dropped_functions)
        )
        self.assertEqual(result.action_ids, ())

    def test_dry_run_matches_live_guard_only_same_atom_force_decision(self):
        def should_not_reselect(*args):
            raise AssertionError("same-Atom conversion must not call force_first_plan")

        dry_monotonic = _Monotonic()
        dry_executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            force_first_plan=should_not_reselect,
            prepare_recovery=_prepare_recovery,
            monotonic_ns=dry_monotonic,
        )
        dry_request = self._request(
            request_id="dry-guard",
            functions=("affirm", "deny"),
            seconds=0.36,
        )
        dry_result = dry_executor.execute(dry_request)

        live_monotonic = _Monotonic()
        live_executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            force_first_plan=should_not_reselect,
            client=_Client(statuses=("completed",)),
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=live_monotonic,
        )
        live_request = self._request(
            request_id="live-guard",
            functions=("affirm", "deny"),
            seconds=0.36,
        )
        live_result = live_executor.execute(
            live_request,
            reply_clock=ReplyClock.from_payload(
                live_request, monotonic_ns=live_monotonic
            ),
        )

        self.assertEqual(dry_result.status, "dry_run_forced_first")
        self.assertTrue(dry_result.would_force_first_atom)
        self.assertFalse(dry_result.forced_first_atom)
        self.assertEqual(dry_result.planned_overrun_us, 30_000)
        self.assertEqual(dry_result.pending_functions, ("affirm",))
        self.assertEqual(dry_result.dropped_functions, ("deny",))
        self.assertEqual(live_result.status, "forced_first_completed")
        self.assertEqual(
            dry_result.planned_overrun_us, live_result.planned_overrun_us
        )

    def test_dry_run_same_atom_force_accounts_for_prepare_crossing_deadline(self):
        monotonic = _Monotonic()

        def slow_prepare(atom):
            step = _prepare(atom)
            monotonic.advance(0.2)
            return step

        executor = SemanticSuffixExecutor(
            _Planner(),
            slow_prepare,
            force_first_plan=lambda *args: (_ for _ in ()).throw(
                AssertionError("same-Atom conversion must not reselect")
            ),
            prepare_recovery=_prepare_recovery,
            monotonic_ns=monotonic,
        )
        result = executor.execute(
            self._request(
                request_id="dry-prepare-expiry",
                functions=("affirm", "deny"),
                seconds=0.4,
            )
        )
        self.assertEqual(result.status, "dry_run_forced_first")
        self.assertTrue(result.would_force_first_atom)
        self.assertFalse(result.forced_first_atom)
        self.assertEqual(result.planned_overrun_us, 190_000)
        self.assertEqual(result.pending_functions, ("affirm",))
        self.assertEqual(result.dropped_functions, ("deny",))

    def test_force_one_is_allowed_when_reply_clock_expired_before_first_plan(self):
        monotonic = _Monotonic()
        planner_budgets = []

        def no_fit(remaining, budget, previous, history):
            planner_budgets.append(budget)
            return _Plan(())

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            no_fit,
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 551),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
        )
        request = self._request(functions=("affirm",), seconds=0.1)
        clock = ReplyClock.from_payload(request, monotonic_ns=monotonic)
        monotonic.advance(0.2)
        result = executor.execute(request, reply_clock=clock)
        self.assertEqual(planner_budgets, [0])
        self.assertEqual(result.status, "forced_first_completed")
        self.assertEqual(result.planned_overrun_us, 390_000)
        self.assertEqual(len(client.posts), 2)

    def test_natural_expiry_during_forced_atom_cannot_skip_neutral_recovery(self):
        monotonic = _Monotonic()
        client = _Client(statuses=("completed",))
        original_action_sync = client.action_sync
        advanced = False

        def expire_during_atom(action_id):
            nonlocal advanced
            record = original_action_sync(action_id)
            if len(client.posts) == 1 and not advanced:
                advanced = True
                monotonic.advance(1.0)
            return record

        client.action_sync = expire_during_atom
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (_Atom(remaining[0], 561),)
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
        )
        request = self._request(functions=("affirm", "deny"), seconds=0.1)
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )
        self.assertEqual(result.status, "forced_first_completed")
        self.assertEqual(result.completed_functions, ("affirm",))
        self.assertEqual(result.dropped_functions, ("deny",))
        self.assertTrue(result.closed_neutral)
        self.assertEqual(len(client.posts), 2)

    def test_force_first_planner_must_return_zero_or_exactly_one_atom(self):
        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            lambda *args: _Plan(()),
            _prepare,
            force_first_plan=lambda remaining, previous, history: _Plan(
                (
                    _Atom(remaining[0], 571),
                    _Atom(remaining[0], 572),
                )
            ),
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "exactly one Atom"):
            self._execute_live(
                executor, self._request(functions=("affirm",), seconds=0.1)
            )
        self.assertEqual(client.posts, [])
        self.assertTrue(executor.status()["safe_stopped"])

    def test_force_one_is_never_reused_for_a_later_suffix(self):
        planner_calls = []
        force_calls = []

        def first_only(remaining, budget, previous, history):
            planner_calls.append((remaining, budget, previous, history))
            if len(planner_calls) == 1:
                return _Plan((_Atom(remaining[0], 601),))
            return _Plan(())

        def forbidden_force(*args):
            force_calls.append(args)
            raise AssertionError("force-one must never apply to a later suffix")

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            first_only,
            _prepare,
            force_first_plan=forbidden_force,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        result = self._execute_live(executor, self._request())
        self.assertEqual(result.status, "partial_no_plan")
        self.assertFalse(result.would_force_first_atom)
        self.assertFalse(result.forced_first_atom)
        self.assertEqual(result.completed_functions, ("emphasize",))
        self.assertEqual(result.dropped_functions, ("deny",))
        self.assertEqual(force_calls, [])
        self.assertEqual(len(client.posts), 2)  # first Atom plus neutral recovery

    def test_slow_planner_cannot_submit_from_a_stale_reply_budget(self):
        monotonic = _Monotonic()
        planner = _Planner()

        def slow_plan(*args):
            monotonic.advance(2.0)
            return planner(*args)

        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            slow_plan,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
            sleep=lambda seconds: monotonic.advance(seconds),
        )
        request = self._request(functions=("affirm",), seconds=1.0)
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )
        self.assertEqual(result.status, "no_safe_prefix")
        self.assertEqual(client.posts, [])
        self.assertTrue(result.closed_neutral)

    def test_expiry_after_an_admitted_atom_still_runs_reserved_recovery(self):
        monotonic = _Monotonic()
        client = _Client(statuses=("completed",))
        original_action = client.action_sync
        first_semantic_read = True

        def slow_first_action(action_id):
            nonlocal first_semantic_read
            if first_semantic_read:
                first_semantic_read = False
                monotonic.advance(2.0)
            return original_action(action_id)

        client.action_sync = slow_first_action
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
            sleep=lambda seconds: monotonic.advance(seconds),
        )
        request = self._request(functions=("affirm",), seconds=1.0)
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )
        self.assertEqual(result.status, "expired")
        self.assertEqual(result.completed_functions, ("affirm",))
        self.assertEqual(len(client.posts), 2)
        self.assertEqual(result.recovery_action_id, client.posts[-1]["id"])
        self.assertTrue(result.closed_neutral)

    def test_interrupt_drops_pending_but_observes_active_to_terminal_without_cancel(self):
        monotonic = _Monotonic()
        planner = _Planner()
        client = _Client(statuses=("started", "completed"))
        first_read = threading.Event()
        continue_read = threading.Event()
        original_action = client.action_sync

        def blocking_action(action_id):
            if not first_read.is_set():
                first_read.set()
                continue_read.wait(timeout=1.0)
            return original_action(action_id)

        client.action_sync = blocking_action
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
            sleep=lambda seconds: monotonic.advance(seconds),
            poll_interval_s=0.01,
        )
        request = self._request(functions=("emphasize", "deny", "affirm"))
        clock = ReplyClock.from_payload(request, monotonic_ns=monotonic)
        holder = {}

        def run():
            holder["result"] = executor.execute(request, reply_clock=clock)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(first_read.wait(timeout=1.0))
        self.assertTrue(executor.interrupt("turn-1", 1, reason="barge_in"))
        continue_read.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        result = holder["result"]
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(result.completed_functions, ("emphasize",))
        self.assertEqual(result.dropped_functions, ("deny", "affirm"))
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(client.forbidden_calls, [])
        self.assertEqual(len(planner.calls), 1)
        self.assertFalse(executor.status()["safe_stopped"])
        next_result = self._execute_live(
            executor,
            self._request(request_id="turn-2", functions=("affirm",)),
        )
        self.assertEqual(next_result.status, "completed")
        self.assertEqual(len(client.posts), 3)

    def test_signal_interrupt_is_nonblocking_at_the_committed_post_boundary(self):
        client = _BlockingPostClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            sleep=lambda _: None,
        )
        request = self._request(functions=("affirm", "deny"))
        clock = ReplyClock.from_payload(request)
        holder = {}

        def run():
            holder["result"] = executor.execute(request, reply_clock=clock)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(client.post_entered.wait(timeout=1.0))
        # This must not wait for the blocked network POST or call the audit sink.
        self.assertTrue(
            executor.signal_interrupt("turn-1", 1, reason="event_loop_barge_in")
        )
        client.post_release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(holder["result"].status, "interrupted")
        self.assertEqual(len(client.posts), 1)
        self.assertIsNone(holder["result"].recovery_action_id)

    def test_interrupt_audit_is_explicitly_deferred_from_signal(self):
        interrupted_audits = []

        def audit(event):
            if event["event"] == "interrupted":
                interrupted_audits.append(event)
                raise RuntimeError("slow durable sink failed")

        client = _BlockingStateClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audit,
        )
        request = self._request(functions=("affirm",))
        holder = {}

        def run():
            holder["result"] = executor.execute(
                request, reply_clock=ReplyClock.from_payload(request)
            )

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(client.state_entered.wait(timeout=1.0))
        self.assertTrue(executor.signal_interrupt("turn-1", 1, reason="barge_in"))
        self.assertEqual(interrupted_audits, [])
        with self.assertRaisesRegex(AuditCallbackError, "slow durable sink"):
            executor.audit_signaled_interrupt("turn-1", 1)
        self.assertEqual(len(interrupted_audits), 1)
        client.state_release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(holder["result"].status, "interrupted")

    def test_finalizing_result_rejects_a_late_interrupt_signal(self):
        finish_entered = threading.Event()
        finish_release = threading.Event()

        def audit(event):
            if event["event"] == "request_finished":
                finish_entered.set()
                finish_release.wait(timeout=1.0)

        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=_Client(statuses=("completed",)),
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audit,
            sleep=lambda _: None,
        )
        request = self._request(functions=("affirm",))
        holder = {}

        def run():
            holder["result"] = self._execute_live(executor, request)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(finish_entered.wait(timeout=1.0))
        self.assertFalse(executor.signal_interrupt("turn-1", 1, reason="too_late"))
        finish_release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(holder["result"].status, "completed")

    def test_request_revision_is_idempotent_and_stale_or_different_replay_fails(self):
        planner = _Planner()
        client = _Client(statuses=("completed",))
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            sleep=lambda _: None,
        )
        request = self._request(functions=("affirm",))
        first = self._execute_live(executor, request)
        second = self._execute_live(executor, request)
        self.assertIs(first, second)
        self.assertEqual(len(client.posts), 2)
        with self.assertRaisesRegex(SemanticConflictError, "different content"):
            self._execute_live(executor, self._request(functions=("deny",)))
        newer = validate_gemini_tool_payload(
            {"ordered_functions": ["deny"], "estimated_spoken_duration_s": 1.0},
            request_id="turn-1",
            revision=2,
        )
        self.assertEqual(self._execute_live(executor, newer).status, "completed")
        self.assertNotEqual(
            client.posts[0]["exclusive_owner"], client.posts[2]["exclusive_owner"]
        )
        with self.assertRaisesRegex(SemanticConflictError, "stale revision"):
            self._execute_live(executor, request)

    def test_interrupt_winning_before_atomic_post_drops_suffix_without_submission(self):
        client = _BlockingStateClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        request = self._request(functions=("emphasize", "deny"))
        holder = {}

        def run():
            holder["result"] = self._execute_live(executor, request)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(client.state_entered.wait(timeout=1.0))
        self.assertTrue(executor.interrupt("turn-1", 1, reason="new_generation"))
        client.state_release.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(holder["result"].status, "interrupted")
        self.assertEqual(
            holder["result"].dropped_functions, ("emphasize", "deny")
        )
        self.assertEqual(client.posts, [])
        self.assertEqual(client.forbidden_calls, [])

    def test_interrupt_state_reset_before_post_does_not_poison_next_turn(self):
        for failed_state_read in (1, 2):
            with self.subTest(
                phase="planning" if failed_state_read == 1 else "admission"
            ):
                client = _ResetOnStateReadClient(failed_state_read)
                executor = SemanticSuffixExecutor(
                    _Planner(),
                    _prepare,
                    client=client,
                    dry_run=False,
                    prepare_recovery=_prepare_recovery,
                    exclusive_owner="semantic-test-owner",
                    sleep=lambda _: None,
                )
                request = self._request(functions=("emphasize", "deny"))
                holder = {}

                def run():
                    try:
                        holder["result"] = self._execute_live(executor, request)
                    except BaseException as exc:  # surface worker failures below
                        holder["error"] = exc

                worker = threading.Thread(target=run)
                worker.start()
                self.assertTrue(client.state_entered.wait(timeout=1.0))
                self.assertTrue(
                    executor.interrupt("turn-1", 1, reason="native_session_reset")
                )
                client.state_release.set()
                worker.join(timeout=2.0)

                self.assertFalse(worker.is_alive())
                self.assertNotIn("error", holder)
                self.assertEqual(holder["result"].status, "interrupted")
                self.assertEqual(
                    holder["result"].dropped_functions,
                    ("emphasize", "deny"),
                )
                self.assertEqual(client.posts, [])
                self.assertFalse(executor.status()["safe_stopped"])
                self.assertNotIn(("turn-1", 1), executor._failed_keys)

                next_result = self._execute_live(
                    executor,
                    self._request(
                        request_id="turn-2", functions=("affirm",)
                    ),
                )
                self.assertEqual(next_result.status, "completed")
                self.assertEqual(len(client.posts), 2)

    def test_admission_409_is_not_retried_and_globally_safe_stops_executor(self):
        client = _ConflictClient()
        planner = _Planner()
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticConflictError, "HTTP 409"):
            self._execute_live(executor, self._request(functions=("affirm",)))
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(len(planner.calls), 1)
        self.assertTrue(executor.status()["safe_stopped"])
        newer = validate_gemini_tool_payload(
            {"ordered_functions": ["deny"], "estimated_spoken_duration_s": 1.0},
            request_id="turn-2",
            revision=1,
        )
        with self.assertRaisesRegex(SemanticConflictError, "safe-stopped"):
            self._execute_live(executor, newer)
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(client.forbidden_calls, [])

    def test_invalid_post_response_drains_exact_own_action_without_repost(self):
        client = _InvalidPostResponseClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            sleep=lambda _: None,
        )
        with self.assertRaisesRegex(SemanticExecutionError, "submission response"):
            self._execute_live(
                executor, self._request(functions=("affirm",))
            )
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(len(client.reads), 2)
        self.assertTrue(executor.status()["safe_stopped"])

    def test_response_loss_retries_only_exact_id_until_eventually_terminal(self):
        monotonic = _Monotonic()
        client = _EventuallyVisibleResponseLossClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
            sleep=lambda seconds: monotonic.advance(seconds),
            poll_interval_s=0.01,
        )
        request = self._request(functions=("affirm",))
        result = executor.execute(
            request,
            reply_clock=ReplyClock.from_payload(request, monotonic_ns=monotonic),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(client.posts), 2)  # Atom once, then recovery once
        self.assertEqual(client.missing_reads, 0)
        self.assertGreaterEqual(len(client.reads), 4)
        self.assertTrue(result.closed_neutral)

    def test_own_action_payload_hash_mismatch_stops_before_replan_or_recovery(self):
        client = _BadActionHashClient(statuses=("completed",))
        planner = _Planner()
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "payload_sha256 mismatch"):
            self._execute_live(
                executor, self._request(functions=("affirm", "deny"))
            )
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(len(planner.calls), 1)
        self.assertTrue(executor.status()["safe_stopped"])
        self.assertEqual(client.forbidden_calls, [])

    def test_transient_own_action_read_is_retried_without_reposting(self):
        monotonic = _Monotonic()
        client = _TransientActionReadClient()
        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
            sleep=lambda seconds: monotonic.advance(seconds),
            poll_interval_s=0.01,
        )
        result = self._execute_live(executor, self._request(functions=("affirm",)))
        self.assertEqual(result.status, "completed")
        self.assertEqual(client.transient_read_failures, 1)
        self.assertEqual(len(client.posts), 2)  # one Atom and one recovery only
        self.assertEqual(len({item["id"] for item in client.posts}), 2)

    def test_post_submit_audit_failure_still_observes_own_terminal_then_fails(self):
        client = _Client(statuses=("scheduled", "completed"))

        def audit(event):
            if event["event"] == "action_submitted":
                raise RuntimeError("audit sink unavailable after POST")

        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audit,
            sleep=lambda _: None,
        )
        with self.assertRaisesRegex(AuditCallbackError, "after POST"):
            self._execute_live(executor, self._request(functions=("affirm",)))
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(len(client.reads), 2)
        self.assertTrue(executor.status()["safe_stopped"])
        self.assertEqual(client.forbidden_calls, [])

    def test_action_speed_must_match_the_planner_unit_speed_contract(self):
        client = _Client(statuses=("completed",))

        def wrong_speed(atom):
            step = _prepare(atom)
            payload = step.action_payload
            payload["speed"] = 1.25
            return SemanticActionStep(
                plan_id=step.plan_id,
                function=step.function,
                atom_ref=step.atom_ref,
                action_payload=payload,
                estimated_duration_us=step.estimated_duration_us,
            )

        executor = SemanticSuffixExecutor(
            _Planner(),
            wrong_speed,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "exactly 1.0"):
            self._execute_live(executor, self._request(functions=("affirm",)))
        self.assertEqual(client.posts, [])

    def test_wrong_function_unsafe_payload_and_failed_terminal_do_not_replan(self):
        request = self._request(functions=("emphasize", "deny"))

        wrong = SemanticSuffixExecutor(
            lambda *_: _Plan((_Atom("affirm", 1),)),
            _prepare,
            client=_Client(),
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "does not match"):
            self._execute_live(wrong, request)
        self.assertEqual(wrong.client.posts, [])

        planner = _Planner()
        unsafe_client = _Client()

        def unsafe(atom):
            step = _prepare(atom)
            return SemanticActionStep(
                plan_id=step.plan_id,
                function=step.function,
                atom_ref=step.atom_ref,
                action_payload={"motion": step.action_payload["motion"], "mode": "replace"},
                estimated_duration_us=step.estimated_duration_us,
            )

        unsafe = SemanticSuffixExecutor(
            planner,
            unsafe,
            client=unsafe_client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
        )
        with self.assertRaisesRegex(SemanticExecutionError, "action payload"):
            self._execute_live(unsafe, request)
        self.assertEqual(unsafe_client.posts, [])

        failed_planner = _Planner()
        failed_client = _Client(statuses=("failed",))
        failed = SemanticSuffixExecutor(
            failed_planner,
            _prepare,
            client=failed_client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            sleep=lambda _: None,
        )
        with self.assertRaisesRegex(SemanticExecutionError, "failed"):
            self._execute_live(failed, request)
        self.assertEqual(len(failed_client.posts), 1)
        self.assertEqual(len(failed_planner.calls), 1)
        self.assertEqual(failed_client.forbidden_calls, [])

    def test_audit_failure_before_submit_is_fail_closed(self):
        client = _Client()

        def audit(event):
            if event["event"] == "submit_intent":
                raise RuntimeError("disk full")

        executor = SemanticSuffixExecutor(
            _Planner(),
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            audit=audit,
        )
        with self.assertRaisesRegex(AuditCallbackError, "disk full"):
            self._execute_live(executor, self._request())
        self.assertEqual(client.posts, [])

    def test_expired_reply_is_noop(self):
        monotonic = _Monotonic()
        request = self._request(seconds=0.1)
        clock = ReplyClock.from_payload(request, monotonic_ns=monotonic)
        monotonic.advance(0.2)
        planner = _Planner()
        client = _Client()
        executor = SemanticSuffixExecutor(
            planner,
            _prepare,
            client=client,
            dry_run=False,
            prepare_recovery=_prepare_recovery,
            exclusive_owner="semantic-test-owner",
            monotonic_ns=monotonic,
        )
        result = executor.execute(request, reply_clock=clock)
        self.assertEqual(result.status, "expired")
        self.assertEqual(result.dropped_functions, ("emphasize", "deny"))
        self.assertEqual(planner.calls, [])
        self.assertEqual(client.posts, [])


if __name__ == "__main__":
    unittest.main()
