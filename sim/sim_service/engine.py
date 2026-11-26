"""Physics-engine adapters.

Only the simulation thread calls the mutable physics methods. The renderer owns
a separate ``MjData`` and consumes copied snapshots; HTTP threads never touch
either mutable MuJoCo state object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import threading
from typing import Any, Mapping
import zlib


G1_UPPER_BODY_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class FreeCameraSpec:
    """Stable free-camera framing for the fixed-base conversational G1."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.62)
    distance: float = 2.2
    azimuth: float = 180.0
    elevation: float = -6.0

    def __post_init__(self) -> None:
        values = (*self.lookat, self.distance, self.azimuth, self.elevation)
        if len(self.lookat) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError("camera values must be finite and lookat must contain three values")
        if self.distance <= 0.0:
            raise ValueError("camera distance must be positive")
        if not -89.0 <= self.elevation <= 89.0:
            raise ValueError("camera elevation must be between -89 and 89 degrees")


CONVERSATION_CAMERA = FreeCameraSpec()


def build_render_camera(
    mujoco: Any,
    requested: str | int | None,
    spec: FreeCameraSpec = CONVERSATION_CAMERA,
) -> Any:
    """Build the stable conversation view or preserve an explicit camera.

    ``None``, ``conversation`` and ``auto`` select the front view. ``scene``
    restores MuJoCo's scene-statistic-derived free camera. Other names and IDs
    retain the existing fixed-camera behavior.
    """

    if requested not in (None, "conversation", "auto"):
        if requested == "scene":
            return -1
        if isinstance(requested, str) and requested.lstrip("-").isdigit():
            return int(requested)
        return requested
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = spec.lookat
    camera.distance = spec.distance
    camera.azimuth = spec.azimuth
    camera.elevation = spec.elevation
    return camera


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """Encode tightly packed RGB bytes as a basic PNG without Pillow/OpenCV."""
    if width <= 0 or height <= 0 or len(rgb) != width * height * 3:
        raise ValueError("invalid RGB image dimensions")

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[row * stride : (row + 1) * stride] for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 3))
        + chunk(b"IEND", b"")
    )


def placeholder_png(width: int = 64, height: int = 48) -> bytes:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            base = (y * width + x) * 3
            shade = 24 + int(18 * y / max(height - 1, 1))
            pixels[base : base + 3] = bytes((shade, shade + 4, shade + 10))
    return encode_png(width, height, bytes(pixels))


class SnapshotRenderer(ABC):
    """Renderer owned from creation through destruction by one render thread."""

    @abstractmethod
    def render_png(self, snapshot: Mapping[str, Any]) -> bytes: ...

    def close(self) -> None:
        return None


class PhysicsEngine(ABC):
    @property
    @abstractmethod
    def timestep(self) -> float: ...

    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def resolve_joint(self, name: str) -> str | None: ...

    @abstractmethod
    def set_targets(self, targets: Mapping[str, float]) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]: ...

    @abstractmethod
    def create_renderer(self) -> SnapshotRenderer: ...

    def close(self) -> None:
        return None


class _StaticSnapshotRenderer(SnapshotRenderer):
    def __init__(self, frame: bytes) -> None:
        self.frame = frame

    def render_png(self, snapshot: Mapping[str, Any]) -> bytes:
        del snapshot
        return self.frame


class KinematicEngine(PhysicsEngine):
    """Dependency-free deterministic engine for API development and tests."""

    def __init__(
        self,
        joints: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
        *,
        timestep: float = 0.005,
        kp: float = 45.0,
        kd: float = 8.0,
        max_acceleration: float = 30.0,
    ) -> None:
        self._joint_names = tuple(joints)
        self._timestep = float(timestep)
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_acceleration = float(max_acceleration)
        self.q = {joint: 0.0 for joint in joints}
        self.dq = {joint: 0.0 for joint in joints}
        self.target = dict(self.q)
        self.time = 0.0
        self._frame = placeholder_png()

    @property
    def timestep(self) -> float:
        return self._timestep

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def resolve_joint(self, name: str) -> str | None:
        if name in self.q:
            return name
        with_suffix = f"{name}_joint"
        if with_suffix in self.q:
            return with_suffix
        if name.endswith("_joint") and name[:-6] in self.q:
            return name[:-6]
        return None

    def set_targets(self, targets: Mapping[str, float]) -> None:
        for joint, value in targets.items():
            resolved = self.resolve_joint(joint)
            if resolved is not None:
                self.target[resolved] = float(value)

    def step(self) -> None:
        dt = self._timestep
        for joint in self._joint_names:
            acceleration = self.kp * (self.target[joint] - self.q[joint]) - self.kd * self.dq[joint]
            acceleration = min(max(acceleration, -self.max_acceleration), self.max_acceleration)
            self.dq[joint] += acceleration * dt
            self.q[joint] += self.dq[joint] * dt
        self.time += dt

    def snapshot(self) -> dict[str, Any]:
        return {
            "sim_time": self.time,
            "q": dict(self.q),
            "dq": dict(self.dq),
            "target": dict(self.target),
        }

    def create_renderer(self) -> SnapshotRenderer:
        return _StaticSnapshotRenderer(self._frame)


class _MujocoSnapshotRenderer(SnapshotRenderer):
    """Off-screen renderer with state and GL context isolated from physics."""

    def __init__(self, engine: "MujocoEngine") -> None:
        self._owner_thread = threading.get_ident()
        self.mujoco = engine.mujoco
        self.np = engine.np
        self.model = engine.model  # Shared read-only after service startup.
        self.width = engine.width
        self.height = engine.height
        self.camera = build_render_camera(
            self.mujoco,
            engine.camera,
            engine.camera_spec,
        )
        self.records = tuple(
            (record["joint"], record["qpos_adr"]) for record in engine._actuators
        )
        self.data = self.mujoco.MjData(self.model)
        self.data.qpos[:] = engine._initial_qpos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        # Renderer creation allocates the EGL/OSMesa context on this thread.
        self.renderer = self.mujoco.Renderer(
            self.model,
            height=self.height,
            width=self.width,
        )

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("renderer and GL context must stay on the render thread")

    def render_png(self, snapshot: Mapping[str, Any]) -> bytes:
        self._assert_owner()
        q = snapshot.get("q", {})
        if isinstance(q, Mapping):
            for joint, qpos_adr in self.records:
                value = q.get(joint)
                if value is not None:
                    self.data.qpos[qpos_adr] = float(value)
        sim_time = snapshot.get("sim_time")
        if sim_time is not None:
            self.data.time = float(sim_time)
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        image = self.renderer.render()
        contiguous = self.np.ascontiguousarray(image, dtype=self.np.uint8)
        return encode_png(self.width, self.height, contiguous.tobytes())

    def close(self) -> None:
        self._assert_owner()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class MujocoEngine(PhysicsEngine):
    """MuJoCo PD engine with the floating root pinned to its initial pose."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        kp: float = 45.0,
        kd: float = 2.5,
        control_mode: str = "auto",
        width: int = 640,
        height: int = 480,
        camera: str | int | None = None,
        camera_spec: FreeCameraSpec = CONVERSATION_CAMERA,
        timestep: float | None = None,
    ) -> None:
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("MuJoCo mode needs `pip install -e .`") from exc
        if control_mode not in {"auto", "torque", "position"}:
            raise ValueError("control_mode must be auto, torque, or position")
        self.mujoco = mujoco
        self.np = np
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).expanduser().resolve()))
        if timestep is not None:
            self.model.opt.timestep = float(timestep)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.kp = float(kp)
        self.kd = float(kd)
        self.control_mode = control_mode
        self.width = int(width)
        self.height = int(height)
        self.camera = camera
        self.camera_spec = camera_spec
        self._owner_thread: int | None = None

        self._actuators: list[dict[str, Any]] = []
        self._aliases: dict[str, str] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if not joint_name:
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            record = {
                "actuator_id": actuator_id,
                "joint_id": joint_id,
                "joint": joint_name,
                "qpos_adr": qpos_adr,
                "dof_adr": dof_adr,
            }
            self._actuators.append(record)
            self._aliases[joint_name] = joint_name
            self._aliases[joint_name.removesuffix("_joint")] = joint_name
            if actuator_name:
                self._aliases[actuator_name] = joint_name
                self._aliases[actuator_name.removesuffix("_joint")] = joint_name

        self._joint_names = tuple(record["joint"] for record in self._actuators)
        self.target = {
            record["joint"]: float(self.data.qpos[record["qpos_adr"]]) for record in self._actuators
        }
        self._free_roots: list[tuple[int, int, Any]] = []
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                pose = self.data.qpos[qpos_adr : qpos_adr + 7].copy()
                self._free_roots.append((qpos_adr, dof_adr, pose))
        # Immutable seed copied by each independent rendering MjData.
        self._initial_qpos = self.data.qpos.copy()

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def resolve_joint(self, name: str) -> str | None:
        return self._aliases.get(name)

    def _claim_thread(self) -> None:
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("MuJoCo state may only be accessed by the simulation thread")

    def set_targets(self, targets: Mapping[str, float]) -> None:
        self._claim_thread()
        for name, value in targets.items():
            resolved = self.resolve_joint(name)
            if resolved is not None:
                self.target[resolved] = float(value)

    def _pin_roots(self) -> None:
        for qpos_adr, dof_adr, pose in self._free_roots:
            self.data.qpos[qpos_adr : qpos_adr + 7] = pose
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

    def _is_position_actuator(self, actuator_id: int) -> bool:
        if self.control_mode == "position":
            return True
        if self.control_mode == "torque":
            return False
        return int(self.model.actuator_biastype[actuator_id]) != 0

    def step(self) -> None:
        self._claim_thread()
        self._pin_roots()
        for record in self._actuators:
            aid = record["actuator_id"]
            q = float(self.data.qpos[record["qpos_adr"]])
            dq = float(self.data.qvel[record["dof_adr"]])
            target = self.target[record["joint"]]
            if self._is_position_actuator(aid):
                control = target
            else:
                control = self.kp * (target - q) - self.kd * dq
            if int(self.model.actuator_ctrllimited[aid]):
                low, high = self.model.actuator_ctrlrange[aid]
                control = min(max(control, float(low)), float(high))
            self.data.ctrl[aid] = control
        self.mujoco.mj_step(self.model, self.data)
        self._pin_roots()
        self.mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> dict[str, Any]:
        self._claim_thread()
        q = {
            record["joint"]: float(self.data.qpos[record["qpos_adr"]])
            for record in self._actuators
        }
        dq = {
            record["joint"]: float(self.data.qvel[record["dof_adr"]])
            for record in self._actuators
        }
        return {"sim_time": float(self.data.time), "q": q, "dq": dq, "target": dict(self.target)}

    def create_renderer(self) -> SnapshotRenderer:
        return _MujocoSnapshotRenderer(self)
