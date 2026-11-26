from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from web_overlay.semantic_dialogue import (
    SEMANTIC_FUNCTIONS,
    SEMANTIC_TOOL_NAME,
    patch_live_config_builder,
    patch_semantic_bridge_class,
    semantic_tool,
)
from web_overlay.semantic_runtime import (
    RUNTIME_FUNCTIONS,
    gemini_tool_parameters_schema,
)


class _FunctionDeclaration:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.parameters_json_schema = kwargs.get("parameters_json_schema")


class _Tool:
    def __init__(self, **kwargs):
        self.function_declarations = kwargs.get("function_declarations")


class _FunctionResponse:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Config:
    def __init__(self, instruction):
        self.instruction = instruction
        self.tools = []

    def model_copy(self, *, update):
        copied = _Config(self.instruction)
        copied.tools = list(update["tools"])
        return copied


class _Controller:
    def __init__(self):
        self.plans = []
        self.activities = []
        self.audio = []
        self.completed = []
        self.interrupted = []
        self.resets = []

    async def accept_tool_plan(self, **kwargs):
        self.plans.append(kwargs)
        return {
            "accepted": True,
            "function_count": len(kwargs["payload"]["ordered_functions"]),
            "execution_mode": "dry_run",
            "force_first_policy_enabled": True,
        }

    def on_output_activity(self, **kwargs):
        self.activities.append(kwargs)

    def on_audio_chunk(self, **kwargs):
        self.audio.append(kwargs)

    def on_turn_complete(self, **kwargs):
        self.completed.append(kwargs)

    def on_interrupted(self, **kwargs):
        self.interrupted.append(kwargs)

    def reset_turn(self, **kwargs):
        self.resets.append(kwargs)

    def has_pending(self, **kwargs):
        return False


class _RejectingController(_Controller):
    async def accept_tool_plan(self, **kwargs):
        del kwargs
        raise RuntimeError("private simulator detail must not leave the server")


class _FailingHooksController(_Controller):
    def on_output_activity(self, **kwargs):
        del kwargs
        raise RuntimeError("output hook failed")

    def on_audio_chunk(self, **kwargs):
        del kwargs
        raise RuntimeError("audio hook failed")

    def on_turn_complete(self, **kwargs):
        del kwargs
        raise RuntimeError("complete hook failed")


class _SlowAudioController(_Controller):
    def __init__(self):
        super().__init__()
        self.audio_started = asyncio.Event()
        self.audio_release = asyncio.Event()

    async def on_audio_chunk(self, **kwargs):
        self.audio_started.set()
        await self.audio_release.wait()
        self.audio.append(kwargs)


class _ObservingController(_Controller):
    def __init__(self):
        super().__init__()
        self.observations = []

    def observe_output_activity(self, **kwargs):
        self.observations.append(kwargs)


class _BrokenStateController(_Controller):
    def reset_turn(self, **kwargs):
        del kwargs
        raise RuntimeError("reset failed")

    def has_pending(self, **kwargs):
        del kwargs
        raise RuntimeError("pending failed")


class _Stream:
    session_id = "stream-1"

    def __init__(self):
        self.input = ""
        self.output = ""
        self.audio = []
        self.turn_complete = False

    def update_input_transcript(self, value):
        self.input += value

    def update_output_transcript(self, value):
        self.output += value

    def enqueue_gemini_audio(self, body, mime):
        self.audio.append((body, mime))

    def mark_interrupted(self):
        pass

    def mark_turn_complete(self):
        self.turn_complete = True


def _bridge_class():
    class Bridge:
        def __init__(self):
            self.stream_session = _Stream()
            self.browser_ws = object()
            self.robot_turn_serial = 4
            self.robot_turn_complete_received = False
            self.robot_turn_cleanup_task = None

        async def _receive_loop(self, session):
            raise AssertionError("original receiver must be replaced")

        def _mark_robot_output_activity(self):
            pass

        def _reset_robot_turn_state(self):
            self.robot_turn_serial += 1

        def _robot_has_pending_turn_task(self):
            return False

        def _cancel_robot_task(self, task):
            if task is not None:
                task.cancel()

        async def _cleanup_robot_turn_after_playback(self, turn_id):
            del turn_id

        async def _handle_robot_input_transcription(self, text):
            raise AssertionError(text)

        async def _handle_robot_output_transcription(self, text):
            raise AssertionError(text)

    return Bridge


class _Session:
    def __init__(self, responses):
        self.responses = responses
        self.tool_responses = []
        self.receive_count = 0

    async def receive(self):
        self.receive_count += 1
        if self.receive_count > 1:
            raise asyncio.CancelledError()
        for response in self.responses:
            yield response

    async def send_tool_response(self, *, function_responses):
        self.tool_responses.extend(function_responses)


class _FailingResponseSession(_Session):
    async def send_tool_response(self, *, function_responses):
        del function_responses
        raise ValueError("SDK rejected response")


class SemanticDialogueTests(unittest.IsolatedAsyncioTestCase):
    def module(self):
        module = ModuleType("fake_semantic_upstream")
        module.types = SimpleNamespace(
            FunctionDeclaration=_FunctionDeclaration,
            Tool=_Tool,
            FunctionResponse=_FunctionResponse,
        )
        module.sent = []

        async def safe_send(_ws, value):
            module.sent.append(value)

        module._safe_send_json = safe_send
        module.logger = None
        return module

    def test_tool_schema_is_exact_explicit_17_and_excludes_removed_class(self):
        module = self.module()
        value = semantic_tool(module.types)
        declaration = value.function_declarations[0]
        schema = declaration.parameters_json_schema

        self.assertEqual(declaration.name, SEMANTIC_TOOL_NAME)
        self.assertEqual(
            tuple(schema["properties"]["ordered_functions"]["items"]["enum"]),
            SEMANTIC_FUNCTIONS,
        )
        self.assertEqual(len(SEMANTIC_FUNCTIONS), 17)
        self.assertNotIn("present_explain", SEMANTIC_FUNCTIONS)
        self.assertIs(SEMANTIC_FUNCTIONS, RUNTIME_FUNCTIONS)
        self.assertEqual(schema, gemini_tool_parameters_schema())
        ordered = schema["properties"]["ordered_functions"]
        duration = schema["properties"]["estimated_spoken_duration_s"]
        self.assertNotIn("maxItems", ordered)
        self.assertEqual(duration["minimum"], 0.1)
        self.assertNotIn("maximum", duration)

    def test_live_config_appends_prompt_and_preserves_existing_tools(self):
        module = self.module()
        old_tool = object()

        def original(*, system_instruction, **_kwargs):
            value = _Config(system_instruction)
            value.tools = [old_tool]
            return value

        module._build_live_config = original
        self.assertTrue(patch_live_config_builder(module, prompt="SEMANTIC PROMPT"))

        value = module._build_live_config(
            system_instruction="BASE", voice_name="voice", temperature=0
        )

        self.assertEqual(value.instruction, "BASE\n\nSEMANTIC PROMPT")
        self.assertIs(value.tools[0], old_tool)
        self.assertEqual(value.tools[1].function_declarations[0].name, SEMANTIC_TOOL_NAME)
        self.assertFalse(patch_live_config_builder(module, prompt="ignored"))

    def test_frozen_prompt_discloses_explicit_force_first_budget_policy(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "gemini_semantic_gesture_v1.txt"
        )
        prompt = " ".join(prompt_path.read_text(encoding="utf-8").split())
        for expected in (
            "normally fit complete actions inside the reply duration",
            "force-first policy is explicitly enabled",
            "one complete Atom for the first function may exceed",
            "recover to canonical neutral",
            "drop the remaining suffix",
        ):
            self.assertIn(expected, prompt)

    async def test_receiver_accepts_tool_then_observes_audio_without_legacy_selector(self):
        module = self.module()
        controller = _Controller()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="call-1",
                    name=SEMANTIC_TOOL_NAME,
                    args={
                        "ordered_functions": ["turn_open", "emphasize"],
                        "estimated_spoken_duration_s": 8.0,
                    },
                )
            ]
        )
        server_content = SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text="Hello"),
            model_turn=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        inline_data=SimpleNamespace(
                            data=b"\x00\x00" * 24,
                            mime_type="audio/pcm;rate=24000",
                        )
                    )
                ]
            ),
            interrupted=False,
            turn_complete=True,
        )
        session = _Session(
            [
                SimpleNamespace(tool_call=tool_call, server_content=None, go_away=None),
                SimpleNamespace(tool_call=None, server_content=server_content, go_away=None),
            ]
        )

        task = asyncio.create_task(bridge._receive_loop(session))
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(len(controller.plans), 1)
        self.assertEqual(controller.plans[0]["turn_id"], 4)
        self.assertEqual(len(session.tool_responses), 1)
        self.assertTrue(session.tool_responses[0].response["accepted"])
        self.assertEqual(bridge.stream_session.output, "Hello")
        self.assertEqual(len(controller.audio), 1)
        self.assertEqual(controller.completed, [{"turn_id": 4}])
        self.assertTrue(bridge.stream_session.turn_complete)
        self.assertTrue(
            any(event.get("type") == "semantic_gesture_plan" for event in module.sent)
        )
        browser_plan = next(
            event for event in module.sent if event.get("type") == "semantic_gesture_plan"
        )
        self.assertTrue(browser_plan["forceFirstPolicyEnabled"])

    async def test_reset_cancels_scheduled_activity_for_the_old_turn(self):
        module = self.module()
        controller = _Controller()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()

        bridge._mark_robot_output_activity()
        bridge._reset_robot_turn_state()
        await asyncio.sleep(0)

        self.assertEqual(bridge.robot_turn_serial, 5)
        self.assertEqual(controller.activities, [])
        self.assertEqual(controller.resets, [{"previous_turn_id": 4}])

    async def test_controller_hook_failures_do_not_block_gemini_audio_or_completion(self):
        module = self.module()
        controller = _FailingHooksController()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        content = SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text="Hello"),
            model_turn=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        inline_data=SimpleNamespace(
                            data=b"\x00\x00" * 24,
                            mime_type="audio/pcm;rate=24000",
                        )
                    )
                ]
            ),
            interrupted=False,
            turn_complete=True,
        )
        session = _Session(
            [SimpleNamespace(tool_call=None, server_content=content, go_away=None)]
        )

        with self.assertRaises(asyncio.CancelledError):
            await bridge._receive_loop(session)

        self.assertEqual(len(bridge.stream_session.audio), 1)
        self.assertTrue(bridge.stream_session.turn_complete)
        self.assertEqual(bridge.stream_session.output, "Hello")

    async def test_tool_exception_returns_only_stable_public_error_code(self):
        module = self.module()
        controller = _RejectingController()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="call-private",
                    name=SEMANTIC_TOOL_NAME,
                    args={
                        "ordered_functions": ["emphasize"],
                        "estimated_spoken_duration_s": 2.0,
                    },
                )
            ]
        )
        session = _Session(
            [SimpleNamespace(tool_call=tool_call, server_content=None, go_away=None)]
        )

        with self.assertRaises(asyncio.CancelledError):
            await bridge._receive_loop(session)

        response = session.tool_responses[0].response
        self.assertEqual(response["error"], "semantic_plan_rejected")
        self.assertNotIn("detail", response)
        browser_event = next(
            event for event in module.sent if event.get("type") == "semantic_gesture_plan"
        )
        self.assertNotIn("detail", browser_event)
        self.assertNotIn("private", str(browser_event))

    async def test_slow_audio_hook_runs_after_audio_is_enqueued(self):
        module = self.module()
        controller = _SlowAudioController()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        content = SimpleNamespace(
            input_transcription=None,
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        inline_data=SimpleNamespace(
                            data=b"audio-now",
                            mime_type="audio/pcm;rate=24000",
                        )
                    )
                ]
            ),
            interrupted=False,
            turn_complete=False,
        )
        session = _Session(
            [SimpleNamespace(tool_call=None, server_content=content, go_away=None)]
        )

        with self.assertRaises(asyncio.CancelledError):
            await bridge._receive_loop(session)

        self.assertEqual(
            bridge.stream_session.audio,
            [(b"audio-now", "audio/pcm;rate=24000")],
        )
        await asyncio.wait_for(controller.audio_started.wait(), timeout=1.0)
        self.assertEqual(controller.audio, [])
        controller.audio_release.set()
        await asyncio.sleep(0)
        self.assertEqual(len(controller.audio), 1)

    async def test_missing_tool_call_id_is_not_sent_to_sdk_and_audio_continues(self):
        module = self.module()

        class StrictFunctionResponse(_FunctionResponse):
            def __init__(self, **kwargs):
                if not kwargs.get("id"):
                    raise ValueError("id is required")
                super().__init__(**kwargs)

        module.types.FunctionResponse = StrictFunctionResponse
        controller = _Controller()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        missing_id = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id=None,
                    name=SEMANTIC_TOOL_NAME,
                    args={
                        "ordered_functions": ["emphasize"],
                        "estimated_spoken_duration_s": 1.0,
                    },
                )
            ]
        )
        content = SimpleNamespace(
            input_transcription=None,
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        inline_data=SimpleNamespace(
                            data=b"audio",
                            mime_type="audio/pcm;rate=24000",
                        )
                    )
                ]
            ),
            interrupted=False,
            turn_complete=False,
        )
        session = _Session(
            [
                SimpleNamespace(tool_call=missing_id, server_content=None, go_away=None),
                SimpleNamespace(tool_call=None, server_content=content, go_away=None),
            ]
        )

        with self.assertRaises(asyncio.CancelledError):
            await bridge._receive_loop(session)

        self.assertEqual(session.tool_responses, [])
        self.assertEqual(len(bridge.stream_session.audio), 1)
        warning = next(
            event
            for event in module.sent
            if event.get("type") == "semantic_gesture_plan"
        )
        self.assertEqual(warning["error"], "missing_call_id")

    async def test_sdk_tool_response_failure_does_not_stop_later_audio(self):
        module = self.module()
        controller = _Controller()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        tool_call = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="call-sdk-fails",
                    name=SEMANTIC_TOOL_NAME,
                    args={
                        "ordered_functions": ["emphasize"],
                        "estimated_spoken_duration_s": 1.0,
                    },
                )
            ]
        )
        content = SimpleNamespace(
            input_transcription=None,
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        inline_data=SimpleNamespace(
                            data=b"audio-after-tool-error",
                            mime_type="audio/pcm;rate=24000",
                        )
                    )
                ]
            ),
            interrupted=False,
            turn_complete=False,
        )
        session = _FailingResponseSession(
            [
                SimpleNamespace(tool_call=tool_call, server_content=None, go_away=None),
                SimpleNamespace(tool_call=None, server_content=content, go_away=None),
            ]
        )

        with self.assertRaises(asyncio.CancelledError):
            await bridge._receive_loop(session)

        self.assertEqual(len(bridge.stream_session.audio), 1)
        self.assertTrue(
            any(
                event.get("type") == "semantic_tool_response_failed"
                for event in module.sent
            )
        )

    async def test_reset_and_pending_failures_are_isolated_from_upstream_state(self):
        module = self.module()
        controller = _BrokenStateController()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        bridge._semantic_turn_controller = controller

        self.assertFalse(bridge._robot_has_pending_turn_task())
        bridge._reset_robot_turn_state()
        self.assertEqual(bridge.robot_turn_serial, 5)

    async def test_output_anchor_is_captured_before_slow_browser_send(self):
        module = self.module()
        send_started = asyncio.Event()
        send_release = asyncio.Event()

        async def slow_send(_ws, value):
            module.sent.append(value)
            if value.get("type") == "output_transcription":
                send_started.set()
                await send_release.wait()

        module._safe_send_json = slow_send
        controller = _ObservingController()
        bridge_cls = _bridge_class()
        patch_semantic_bridge_class(
            bridge_cls,
            module=module,
            controller_factory=lambda _bridge: controller,
        )
        bridge = bridge_cls()
        content = SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text="anchored"),
            model_turn=None,
            interrupted=False,
            turn_complete=False,
        )
        session = _Session(
            [SimpleNamespace(tool_call=None, server_content=content, go_away=None)]
        )

        with patch("web_overlay.semantic_dialogue.time.monotonic_ns", return_value=123456):
            task = asyncio.create_task(bridge._receive_loop(session))
            await asyncio.wait_for(send_started.wait(), timeout=1.0)
            self.assertEqual(
                controller.observations,
                [{"turn_id": 4, "observed_ns": 123456}],
            )
            await asyncio.sleep(0)
            self.assertEqual(
                controller.activities,
                [{"turn_id": 4, "observed_ns": 123456}],
            )
            send_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
