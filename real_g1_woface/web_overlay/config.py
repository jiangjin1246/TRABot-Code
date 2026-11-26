from __future__ import annotations

from dataclasses import dataclass
import os


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Runtime settings. Environment variables intentionally use one prefix."""

    backend: str = "mujoco"
    video_overlay_enabled: bool = True
    sim_url: str = "http://127.0.0.1:8765"
    frame_path: str = "/frame"
    state_path: str = "/state"
    health_path: str = "/health"
    action_path: str = "/actions/trigger"
    action_delete_path: str = "/action"
    panel_width: int = 640
    panel_height: int = 480
    composite_fps: float = 25.0
    frame_poll_hz: float = 25.0
    http_timeout: float = 0.6
    stale_frame_seconds: float = 1.5
    default_action_mode: str = "enqueue"
    default_speed: float = 1.0
    default_blend_duration: float = 0.25
    diagnostics_host: str = "127.0.0.1"
    diagnostics_port: int = 8791
    # Keep the dataclass default backward-compatible for isolated callers and
    # tests. scripts/run_web.sh explicitly selects random_catalog for the MVP;
    # operators can set legacy_abc to restore the untouched upstream selector.
    dialogue_sequence_mode: str = "legacy_abc"
    atom_catalog_path: str = "atom_catalog_v1.json"
    random_sequence_count: int = 3
    random_sequence_base_seed: int = 42
    random_sequence_log_path: str = "runtime/dialogue_atom_turns.jsonl"
    random_sequence_poll_interval: float = 0.05
    random_sequence_action_timeout: float = 45.0
    semantic_catalog_path: str = (
        "../catalog/"
        "semantic_atom_catalog_v1.6725cb58a85b60a1b885d51a407c1ad7a5abea7d309658b2105cc0d86e264e5b.json"
    )
    semantic_prompt_path: str = "prompts/gemini_semantic_gesture_v1.txt"
    semantic_execute: bool = False
    semantic_force_first: bool = False
    semantic_sequence_log_path: str = "runtime/dialogue_semantic_f17_turns.jsonl"
    semantic_tail_guard_seconds: float = 0.35
    semantic_poll_interval: float = 0.05
    semantic_action_timeout: float = 45.0
    semantic_session_neutral_enabled: bool = True
    g1_realtime_enabled: bool = False
    g1_runner_path: str = "g1_bridge/build/g1_atom_motion_cli"
    g1_catalog_path: str = ""
    g1_limits_path: str = "config/g1_limits.json"
    g1_review_store_path: str = "runtime/reviews.json"
    g1_network_interface: str = "eno2"
    g1_session_startup_timeout: float = 8.0
    g1_session_idle_timeout: float = 15.0
    g1_process_stop_timeout: float = 8.0

    @classmethod
    def from_env(cls) -> "OverlayConfig":
        defaults = cls()
        return cls(
            backend=os.environ.get("OVERLAY_BACKEND", defaults.backend),
            video_overlay_enabled=_env_bool(
                "OVERLAY_VIDEO_ENABLED", defaults.video_overlay_enabled
            ),
            sim_url=os.environ.get("OVERLAY_SIM_URL", defaults.sim_url),
            frame_path=os.environ.get("OVERLAY_FRAME_PATH", defaults.frame_path),
            state_path=os.environ.get("OVERLAY_STATE_PATH", defaults.state_path),
            health_path=os.environ.get("OVERLAY_HEALTH_PATH", defaults.health_path),
            action_path=os.environ.get("OVERLAY_ACTION_PATH", defaults.action_path),
            action_delete_path=os.environ.get(
                "OVERLAY_ACTION_DELETE_PATH", defaults.action_delete_path
            ),
            panel_width=_env_int("OVERLAY_PANEL_WIDTH", defaults.panel_width),
            panel_height=_env_int("OVERLAY_PANEL_HEIGHT", defaults.panel_height),
            composite_fps=_env_float(
                "OVERLAY_COMPOSITE_FPS", defaults.composite_fps
            ),
            frame_poll_hz=_env_float(
                "OVERLAY_FRAME_POLL_HZ", defaults.frame_poll_hz
            ),
            http_timeout=_env_float("OVERLAY_HTTP_TIMEOUT", defaults.http_timeout),
            stale_frame_seconds=_env_float(
                "OVERLAY_STALE_FRAME_SECONDS", defaults.stale_frame_seconds
            ),
            default_action_mode=os.environ.get(
                "OVERLAY_ACTION_MODE", defaults.default_action_mode
            ),
            default_speed=_env_float(
                "OVERLAY_ACTION_SPEED", defaults.default_speed
            ),
            default_blend_duration=_env_float(
                "OVERLAY_BLEND_DURATION", defaults.default_blend_duration
            ),
            diagnostics_host=os.environ.get(
                "OVERLAY_DIAGNOSTICS_HOST", defaults.diagnostics_host
            ),
            diagnostics_port=_env_int(
                "OVERLAY_DIAGNOSTICS_PORT", defaults.diagnostics_port
            ),
            dialogue_sequence_mode=os.environ.get(
                "OVERLAY_DIALOGUE_SEQUENCE_MODE", defaults.dialogue_sequence_mode
            ),
            atom_catalog_path=os.environ.get(
                "OVERLAY_ATOM_CATALOG_PATH", defaults.atom_catalog_path
            ),
            random_sequence_count=_env_int(
                "OVERLAY_RANDOM_SEQUENCE_COUNT", defaults.random_sequence_count
            ),
            random_sequence_base_seed=_env_int(
                "OVERLAY_RANDOM_SEQUENCE_BASE_SEED",
                defaults.random_sequence_base_seed,
            ),
            random_sequence_log_path=os.environ.get(
                "OVERLAY_RANDOM_SEQUENCE_LOG_PATH",
                defaults.random_sequence_log_path,
            ),
            random_sequence_poll_interval=_env_float(
                "OVERLAY_RANDOM_SEQUENCE_POLL_INTERVAL",
                defaults.random_sequence_poll_interval,
            ),
            random_sequence_action_timeout=_env_float(
                "OVERLAY_RANDOM_SEQUENCE_ACTION_TIMEOUT",
                defaults.random_sequence_action_timeout,
            ),
            semantic_catalog_path=os.environ.get(
                "OVERLAY_SEMANTIC_CATALOG_PATH", defaults.semantic_catalog_path
            ),
            semantic_prompt_path=os.environ.get(
                "OVERLAY_SEMANTIC_PROMPT_PATH", defaults.semantic_prompt_path
            ),
            semantic_execute=_env_bool(
                "OVERLAY_SEMANTIC_EXECUTE", defaults.semantic_execute
            ),
            semantic_force_first=_env_bool(
                "OVERLAY_SEMANTIC_FORCE_FIRST", defaults.semantic_force_first
            ),
            semantic_sequence_log_path=os.environ.get(
                "OVERLAY_SEMANTIC_LOG_PATH", defaults.semantic_sequence_log_path
            ),
            semantic_tail_guard_seconds=_env_float(
                "OVERLAY_SEMANTIC_TAIL_GUARD_SECONDS",
                defaults.semantic_tail_guard_seconds,
            ),
            semantic_poll_interval=_env_float(
                "OVERLAY_SEMANTIC_POLL_INTERVAL",
                defaults.semantic_poll_interval,
            ),
            semantic_action_timeout=_env_float(
                "OVERLAY_SEMANTIC_ACTION_TIMEOUT",
                defaults.semantic_action_timeout,
            ),
            semantic_session_neutral_enabled=_env_bool(
                "OVERLAY_SEMANTIC_SESSION_NEUTRAL_ENABLED",
                defaults.semantic_session_neutral_enabled,
            ),
            g1_realtime_enabled=_env_bool(
                "G1_REALTIME_ENABLED", defaults.g1_realtime_enabled
            ),
            g1_runner_path=os.environ.get(
                "G1_REALTIME_RUNNER", defaults.g1_runner_path
            ),
            g1_catalog_path=os.environ.get(
                "G1_REALTIME_CATALOG", defaults.g1_catalog_path
            ),
            g1_limits_path=os.environ.get(
                "G1_REALTIME_LIMITS", defaults.g1_limits_path
            ),
            g1_review_store_path=os.environ.get(
                "G1_REALTIME_REVIEW_STORE", defaults.g1_review_store_path
            ),
            g1_network_interface=os.environ.get(
                "G1_REALTIME_NETWORK_INTERFACE",
                defaults.g1_network_interface,
            ),
            g1_session_startup_timeout=_env_float(
                "G1_REALTIME_SESSION_STARTUP_TIMEOUT",
                defaults.g1_session_startup_timeout,
            ),
            g1_session_idle_timeout=_env_float(
                "G1_REALTIME_SESSION_IDLE_TIMEOUT",
                defaults.g1_session_idle_timeout,
            ),
            g1_process_stop_timeout=_env_float(
                "G1_REALTIME_PROCESS_STOP_TIMEOUT",
                defaults.g1_process_stop_timeout,
            ),
        )

    def endpoint(self, path: str) -> str:
        return f"{self.sim_url.rstrip('/')}/{path.lstrip('/')}"
