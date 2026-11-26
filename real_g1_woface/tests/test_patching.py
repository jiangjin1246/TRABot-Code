from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from web_overlay.config import OverlayConfig
from web_overlay.patching import (
    ACTION_LIFECYCLE_SCRIPT,
    PRESENTATION_GESTURE_IDS,
    RELEASE_ACTION_ID,
    WIDESCREEN_CSS,
    _add_aiohttp_routes,
    install_runtime_patches,
    pick_robot_action,
    select_presentation_action,
)


class _Client:
    def __init__(self):
        self.calls = []

    def trigger_sync(self, action, **kwargs):
        raise AssertionError("legacy wrapper must wait for simulator completion")

    def trigger_legacy_sync(self, action, **kwargs):
        self.calls.append((action, kwargs))
        return {"accepted": True, "completion": {"status": "completed"}}

    def state_sync(self):
        return {"mode": "idle"}

    def release_sync(self):
        return {"released": True}

    def stop_action_sync(self):
        return {"stopped": True}


class _QueueStreamTrack:
    kind = "video"

    async def recv(self):
        return object()


class _PeerConnection:
    def __init__(self):
        self.added = []

    def addTrack(self, track):
        self.added.append(track)
        return track


class _StreamSession:
    pass


def _lifecycle_bridge_class():
    class _GeminiLiveBridge:
        def __init__(self):
            self.events = []
            self.robot_turn_serial = 1
            self.robot_turn_complete_received = False

        async def _emit_robot_action_event(self, payload):
            self.events.append(dict(payload))

        async def _trigger_robot_action(self, *, text="", action_id=""):
            raise AssertionError("upstream blocking trigger must be replaced")

    return _GeminiLiveBridge


class _LifecycleClient:
    def __init__(self, states=None, trigger_error=None):
        self.calls = []
        self.states = list(states or [])
        self.trigger_error = trigger_error

    def trigger_sync(self, action, **kwargs):
        self.calls.append((action, kwargs))
        if self.trigger_error is not None:
            raise self.trigger_error
        return {"accepted": True, "id": "request-1", "queue_depth": 0}

    def state_sync(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        if self.states:
            return self.states[0]
        return {
            "action": None,
            "queued": [],
            "last_completed": {
                "id": "request-1",
                "status": "completed",
            },
            "fault": None,
        }

    def release_sync(self):
        return {"released": True}

    def stop_action_sync(self):
        return {"stopped": True}


class _SemanticOnlyClient(_LifecycleClient):
    semantic_only = True
    simulated = False


class _SerialLifecycleClient(_LifecycleClient):
    def __init__(self):
        super().__init__()
        self.finish_first = False

    def trigger_sync(self, action, **kwargs):
        self.calls.append((action, kwargs))
        request_id = f"request-{len(self.calls)}"
        return {"accepted": True, "id": request_id, "queue_depth": 0}

    def state_sync(self):
        if len(self.calls) == 1 and not self.finish_first:
            return {
                "action": {"id": "request-1"},
                "queued": [],
                "last_completed": None,
                "fault": None,
            }
        request_id = f"request-{len(self.calls)}"
        return {
            "action": None,
            "queued": [],
            "last_completed": {"id": request_id, "status": "completed"},
            "fault": None,
        }


class PatchingTest(unittest.TestCase):
    def test_semantic_mode_loads_exact_assets_and_installs_live_bridge(self):
        module = ModuleType("fake_semantic_mode_upstream")
        module.GeminiLiveBridge = type("GeminiLiveBridge", (), {})
        client = _LifecycleClient()
        catalog = object()
        controller_factory = object()
        config = OverlayConfig(
            dialogue_sequence_mode="semantic_f17",
            semantic_catalog_path="catalogs/content-addressed.json",
            semantic_prompt_path="prompts/semantic.txt",
        )

        with (
            patch(
                "web_overlay.semantic_catalog.load_semantic_catalog",
                return_value=catalog,
            ) as load_catalog,
            patch(
                "web_overlay.semantic_dialogue.load_semantic_prompt",
                return_value="SEMANTIC PROMPT",
            ) as load_prompt,
            patch(
                "web_overlay.semantic_controller.semantic_controller_factory",
                return_value=controller_factory,
            ) as make_controller,
            patch(
                "web_overlay.semantic_dialogue.patch_live_config_builder",
                return_value=True,
            ) as patch_config,
            patch(
                "web_overlay.semantic_dialogue.patch_semantic_bridge_class",
                return_value=True,
            ) as patch_bridge,
        ):
            report = install_runtime_patches(module, client, config)

        load_catalog.assert_called_once_with(
            Path(config.semantic_catalog_path).expanduser().resolve()
        )
        load_prompt.assert_called_once_with(
            Path(config.semantic_prompt_path).expanduser().resolve()
        )
        make_controller.assert_called_once_with(
            catalog=catalog,
            client=client,
            config=config,
        )
        patch_config.assert_called_once_with(module, prompt="SEMANTIC PROMPT")
        patch_bridge.assert_called_once_with(
            module.GeminiLiveBridge,
            module=module,
            controller_factory=controller_factory,
        )
        self.assertEqual(client.calls, [])
        self.assertIn(
            "fake_semantic_mode_upstream.GeminiLiveBridge.semantic_f17_sequence",
            report.action_symbols,
        )

    def test_legacy_and_random_modes_never_load_semantic_assets(self):
        for mode in ("legacy_abc", "random_catalog"):
            with self.subTest(mode=mode):
                module = ModuleType(f"fake_{mode}_upstream")
                config = OverlayConfig(dialogue_sequence_mode=mode)
                with (
                    patch(
                        "web_overlay.patching._semantic_sequence_runtime_setup",
                        side_effect=AssertionError("semantic runtime was touched"),
                    ) as semantic_setup,
                    patch(
                        "web_overlay.patching._random_sequence_coordinator_factory",
                        return_value=lambda _bridge: object(),
                    ),
                ):
                    install_runtime_patches(module, _LifecycleClient(), config)
                semantic_setup.assert_not_called()

    def test_peer_connection_wraps_video_in_independent_composite_track(self):
        from web_overlay.compositor import CompositeVideoTrack

        module = ModuleType("fake_peer_upstream")
        module.RTCPeerConnection = _PeerConnection
        module.QueueStreamTrack = _QueueStreamTrack
        report = install_runtime_patches(module, _LifecycleClient(), OverlayConfig())

        connection = module.RTCPeerConnection()
        source = _QueueStreamTrack()
        added = connection.addTrack(source)

        self.assertIsInstance(added, CompositeVideoTrack)
        self.assertIs(added.source, source)
        self.assertIn(
            "fake_peer_upstream.RTCPeerConnection",
            report.peer_connection_classes,
        )
        self.assertEqual(report.queue_track_classes, [])

    def test_widescreen_css_fits_inside_stage_without_viewport_overflow(self):
        self.assertIn(".stage", WIDESCREEN_CSS)
        self.assertIn("width: 100% !important", WIDESCREEN_CSS)
        self.assertIn("object-fit: contain !important", WIDESCREEN_CSS)
        self.assertNotIn("min(96vw", WIDESCREEN_CSS)
        self.assertNotIn("min(98vw", WIDESCREEN_CSS)

    def test_lifecycle_ui_names_all_browser_states(self):
        for text in (
            "Robot scheduled",
            "Robot started",
            "Robot completed",
            "Robot failed",
        ):
            self.assertIn(text, ACTION_LIFECYCLE_SCRIPT)
        self.assertIn("pushFeed(`${lifecycleLabel}: ${label}`", ACTION_LIFECYCLE_SCRIPT)

    def test_semantic_result_ui_precedes_fallback_and_shows_forced_overrun(self):
        semantic_branch = ACTION_LIFECYCLE_SCRIPT.index(
            'payload.kind === "semantic_atom_sequence"'
        )
        generic_fallback = ACTION_LIFECYCLE_SCRIPT.index("if (!lifecycle)")
        self.assertLess(semantic_branch, generic_fallback)
        for expected in (
            "payload.forcedFirstAtom === true",
            "payload.plannedOverrunUs",
            "payload.forceFirstPolicyEnabled === true",
            "Force-first policy: ${forceFirstPolicyEnabled ? \"enabled\" : \"disabled\"}",
            "no motion planned",
            "Forced overrun: +${(plannedOverrunUs / 1_000_000).toFixed(3)}s",
            "currentTurnHasRobotAction = status !== \"interrupted\"",
        ):
            self.assertIn(expected, ACTION_LIFECYCLE_SCRIPT)

        module = ModuleType("fake_html_upstream")
        module.HTML_PAGE = (
            "<html><head></head><body><script>"
            "function updateRobotPanel(payload) {}"
            "</script></body></html>"
        )
        install_runtime_patches(module, _LifecycleClient(), OverlayConfig())
        self.assertIn("mujoco-overlay-widescreen", module.HTML_PAGE)
        self.assertIn("mujoco-action-lifecycle-ui", module.HTML_PAGE)
        self.assertLess(
            module.HTML_PAGE.index("mujoco-action-lifecycle-ui"),
            module.HTML_PAGE.index("</body>"),
        )

    def test_semantic_only_installs_only_idempotent_control_event_cases(self):
        module = ModuleType("fake_g1_semantic_only_upstream")
        module.HTML_PAGE = """<html><head></head><body><script>
function updateRobotPanel(payload) {}
function pushFeed(message, level = "") {}
function upstreamTriggerRobotAction() {}
switch (payload.type) {
          case "pong":
            break;
          default:
            pushFeed(`Received unknown event: ${payload.type || "unknown"}`);
            break;
}
</script></body></html>"""
        module.RTCPeerConnection = _PeerConnection
        module.QueueStreamTrack = _QueueStreamTrack
        module.StreamSession = _StreamSession
        original_trigger = lambda *_args, **_kwargs: "legacy"
        original_runtime = object()
        module.trigger_robot_action = original_trigger
        module.robot_runtime = original_runtime

        report = install_runtime_patches(
            module,
            _SemanticOnlyClient(),
            OverlayConfig(video_overlay_enabled=True),
        )

        page = module.HTML_PAGE
        for event_name in (
            "semantic_gesture_plan",
            "semantic_tool_response_failed",
            "semantic_session_not_ready",
        ):
            self.assertEqual(page.count(f'case "{event_name}"'), 1)
            self.assertLess(
                page.index(f'case "{event_name}"'), page.index("default:")
            )
        self.assertIn('accepted ? "accepted" : "rejected"', page)
        self.assertIn("Semantic gesture plan could not be acknowledged", page)
        self.assertIn("Semantic motion session is not ready", page)
        self.assertNotIn("mujoco-overlay-widescreen", page)
        self.assertNotIn("mujoco-action-lifecycle-ui", page)
        self.assertNotIn("mujoco-experiment-recording-ui", page)
        self.assertNotIn("mujoco-control-socket-generation", page)
        self.assertIs(module.RTCPeerConnection, _PeerConnection)
        self.assertIs(module.QueueStreamTrack, _QueueStreamTrack)
        self.assertIs(module.StreamSession, _StreamSession)
        self.assertIs(module.trigger_robot_action, original_trigger)
        self.assertIs(module.robot_runtime, original_runtime)
        self.assertEqual(report.peer_connection_classes, [])
        self.assertEqual(report.queue_track_classes, [])
        self.assertEqual(report.stream_session_classes, [])
        self.assertEqual(report.action_symbols, [])
        self.assertEqual(report.robot_runtime_symbols, [])

        first_page = module.HTML_PAGE
        second_report = install_runtime_patches(
            module,
            _SemanticOnlyClient(),
            OverlayConfig(video_overlay_enabled=True),
        )
        self.assertEqual(module.HTML_PAGE, first_page)
        self.assertEqual(second_report.html_globals, [])

    def test_semantic_only_html_middleware_applies_the_same_narrow_patch(self):
        class _Router:
            def __init__(self):
                self.routes = []

            def add_get(self, path, handler):
                self.routes.append((path, handler))

        class _Response:
            content_type = "text/html"
            charset = "utf-8"

            def __init__(self, body):
                self.body = body

        fake_aiohttp = ModuleType("aiohttp")
        fake_aiohttp.web = SimpleNamespace(
            middleware=lambda function: function,
            json_response=lambda body, status=200: (body, status),
        )
        app = SimpleNamespace(router=_Router(), middlewares=[])
        with patch.dict(sys.modules, {"aiohttp": fake_aiohttp}):
            added = _add_aiohttp_routes(
                app, _SemanticOnlyClient(), lambda: {}
            )

        self.assertIn("middleware:semantic-control-events", added)
        self.assertEqual(len(app.middlewares), 1)
        response = _Response(
            b"""<html><head></head><body><script>
switch (payload.type) {
          case \"pong\":
            break;
          default:
            pushFeed(`Received unknown event: ${payload.type || \"unknown\"}`);
            break;
}
</script></body></html>"""
        )

        async def handler(_request):
            return response

        patched = asyncio.run(app.middlewares[0](None, handler))
        page = patched.body.decode("utf-8")
        for event_name in (
            "semantic_gesture_plan",
            "semantic_tool_response_failed",
            "semantic_session_not_ready",
        ):
            self.assertEqual(page.count(f'case "{event_name}"'), 1)
        self.assertNotIn("mujoco-overlay-widescreen", page)
        self.assertNotIn("mujoco-action-lifecycle-ui", page)
        self.assertNotIn("mujoco-experiment-recording-ui", page)
        self.assertNotIn("mujoco-control-socket-generation", page)

    def test_pick_robot_action_returns_bridge_compatible_spec(self):
        self.assertIsNone(pick_robot_action("ordinary conversation"))
        self.assertEqual(
            pick_robot_action("anything", PRESENTATION_GESTURE_IDS[0]),
            {"id": PRESENTATION_GESTURE_IDS[0], "label": "presentation"},
        )
        self.assertEqual(
            pick_robot_action("欢迎大家")["id"], PRESENTATION_GESTURE_IDS[1]
        )
        self.assertEqual(
            pick_robot_action("thank you")["id"], PRESENTATION_GESTURE_IDS[2]
        )
        self.assertEqual(
            pick_robot_action("接下来进行讲解")["id"],
            PRESENTATION_GESTURE_IDS[0],
        )
        self.assertEqual(
            pick_robot_action("introduce Al Ain Museum")["id"],
            PRESENTATION_GESTURE_IDS[0],
        )
        self.assertEqual(
            select_presentation_action("ordinary conversation"),
            PRESENTATION_GESTURE_IDS[0],
        )
        self.assertEqual(
            select_presentation_action("", RELEASE_ACTION_ID),
            RELEASE_ACTION_ID,
        )
        self.assertEqual(
            pick_robot_action("", RELEASE_ACTION_ID),
            {"id": RELEASE_ACTION_ID, "label": "release"},
        )

    def test_async_trigger_and_runtime_are_replaced(self):
        module = ModuleType("fake_upstream")

        async def old_trigger(_):
            raise AssertionError("real bridge must not be called")

        module.trigger_robot_action = old_trigger
        module.robot_runtime = object()
        module.QueueStreamTrack = _QueueStreamTrack
        module.StreamSession = _StreamSession
        client = _Client()
        report = install_runtime_patches(module, client, OverlayConfig())

        result = asyncio.run(
            module.trigger_robot_action("hello there", None, mode="replace")
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["executed"])
        self.assertTrue(result["simulated"])
        self.assertTrue(result["robot"]["enabled"])
        self.assertTrue(result["robot"]["simulationMode"])
        self.assertEqual(result["robot"]["backend"], "mujoco")
        self.assertEqual(
            client.calls,
            [("joint_presentation_gesture_b", {"mode": "replace"})],
        )
        self.assertTrue(module.robot_runtime.simulation_mode)
        self.assertEqual(module.robot_runtime()["backend"], "mujoco")
        self.assertIn("fake_upstream.QueueStreamTrack", report.queue_track_classes)
        self.assertIn("fake_upstream.StreamSession", report.stream_session_classes)


class ActionLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_browser_gets_scheduled_started_completed_without_waiting(self):
        module = ModuleType("fake_lifecycle_upstream")
        module.GeminiLiveBridge = _lifecycle_bridge_class()
        client = _LifecycleClient(
            states=[
                {
                    "action": None,
                    "queued": [{"id": "request-1"}],
                    "last_completed": None,
                    "fault": None,
                },
                {
                    "action": {
                        "id": "request-1",
                        "name": PRESENTATION_GESTURE_IDS[0],
                    },
                    "queued": [],
                    "last_completed": None,
                    "fault": None,
                },
                {
                    "action": None,
                    "queued": [],
                    "last_completed": {
                        "id": "request-1",
                        "name": PRESENTATION_GESTURE_IDS[0],
                        "status": "completed",
                    },
                    "fault": None,
                },
            ]
        )
        report = install_runtime_patches(module, client, OverlayConfig())
        bridge = module.GeminiLiveBridge()

        with patch(
            "web_overlay.patching.ACTION_LIFECYCLE_POLL_SECONDS", 0
        ):
            accepted = await bridge._trigger_robot_action(
                text="introduce the museum",
                action_id=PRESENTATION_GESTURE_IDS[0],
            )

            # POST acceptance returns immediately. State polling and the motion's
            # completion remain in a background task.
            self.assertEqual([event["lifecycle"] for event in bridge.events], ["scheduled"])
            self.assertFalse(bridge._mujoco_action_lifecycle_task.done())
            self.assertTrue(accepted["executed"])
            self.assertEqual(client.calls, [(PRESENTATION_GESTURE_IDS[0], {})])

            completed = await bridge._mujoco_action_lifecycle_task

        self.assertTrue(completed)
        self.assertEqual(
            [event["lifecycle"] for event in bridge.events],
            ["scheduled", "started", "completed"],
        )
        self.assertFalse(bridge.events[0]["executed"])
        self.assertFalse(bridge.events[1]["executed"])
        self.assertTrue(bridge.events[2]["executed"])
        self.assertEqual(bridge.events[2]["actionRequestId"], "request-1")
        self.assertIn(
            "fake_lifecycle_upstream.GeminiLiveBridge._trigger_robot_action",
            report.action_symbols,
        )

    async def test_post_error_is_still_a_matched_robot_action_error(self):
        module = ModuleType("fake_error_upstream")
        module.GeminiLiveBridge = _lifecycle_bridge_class()
        client = _LifecycleClient(trigger_error=RuntimeError("sim unavailable"))
        install_runtime_patches(module, client, OverlayConfig())
        bridge = module.GeminiLiveBridge()

        result = await bridge._trigger_robot_action(
            action_id=PRESENTATION_GESTURE_IDS[0]
        )

        self.assertTrue(result["matched"])
        self.assertFalse(result["executed"])
        self.assertEqual(result["lifecycle"], "failed")
        self.assertIn("sim unavailable", result["error"])
        self.assertEqual(len(bridge.events), 1)
        self.assertEqual(bridge.events[0]["lifecycle"], "failed")

    async def test_second_presentation_waits_for_first_completion(self):
        module = ModuleType("fake_serial_upstream")
        module.GeminiLiveBridge = _lifecycle_bridge_class()
        client = _SerialLifecycleClient()
        install_runtime_patches(module, client, OverlayConfig())
        bridge = module.GeminiLiveBridge()

        with patch(
            "web_overlay.patching.ACTION_LIFECYCLE_POLL_SECONDS", 0.001
        ):
            await bridge._trigger_robot_action(
                action_id=PRESENTATION_GESTURE_IDS[0]
            )
            second = asyncio.create_task(
                bridge._trigger_robot_action(
                    action_id=PRESENTATION_GESTURE_IDS[1]
                )
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(client.calls), 1)
            self.assertFalse(second.done())

            client.finish_first = True
            second_result = await asyncio.wait_for(second, timeout=1.0)
            self.assertTrue(second_result["executed"])
            self.assertEqual(len(client.calls), 2)
            await asyncio.wait_for(
                bridge._mujoco_action_lifecycle_task, timeout=1.0
            )

        scheduled = [
            event for event in bridge.events if event["lifecycle"] == "scheduled"
        ]
        self.assertEqual(
            [event["actionRequestId"] for event in scheduled],
            ["request-1", "request-2"],
        )


if __name__ == "__main__":
    unittest.main()
