# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-effector control and RGB-D perception tools for LIBERO RL rollouts."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.libero.env_client import LiberoEnvClient
from robots.libero_rl import geometry
from rpent.robots.components.sam3_client import Sam3Client
from rpent.session import EnvState, StepRecord
from rpent.tools import ToolResult, tool

Camera = Literal["agentview", "wrist"]
#: Simulator camera rendered for each tool camera name.
SIM_CAMERAS: dict[str, str] = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
IMAGE_SIZE = 1024
#: LIBERO OSC_POSE output per normalized action unit.
POSITION_SCALE_M = 0.05
ROTATION_SCALE_RAD = 0.5
#: Minimum valid points for a box or mask median.
MIN_REGION_POINTS = 8

Pixel = Annotated[list[int], Field(min_length=2, max_length=2)]
Step = Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class ControlConfig:
    """Backend servo limits; planners never set these."""

    position_step_m: float = 0.03
    rotation_step_rad: float = 0.12
    position_tolerance_m: float = 0.004
    rotation_tolerance_rad: float = 0.02
    settle_speed_m: float = 0.001
    settle_speed_rad: float = 0.005
    max_motion_steps: int = 300
    stall_steps: int = 15
    stall_progress_m: float = 0.002
    stall_progress_rad: float = 0.01
    opening_tolerance: float = 0.08
    max_gripper_steps: int = 40
    settle_steps: int = 3
    settle_opening: float = 0.01
    segment_min_score: float = 0.2


@dataclass(frozen=True)
class Measurement:
    """Measured end-effector pose and normalized gripper opening."""

    xyz: np.ndarray
    rotation: np.ndarray
    opening: float

    def state(self) -> dict[str, Any]:
        yaw = geometry.yaw_of(self.rotation)
        return {
            "pose": {
                "xyz": [round(float(value), 4) for value in self.xyz],
                "yaw": None if yaw is None else round(yaw, 3),
                "pitch": round(geometry.pitch_of(self.rotation), 3),
            },
            "opening": round(self.opening, 3),
        }


class LiberoRLPrimitives:
    """Servo one LIBERO arm from measured state and query recorded RGB-D views."""

    def __init__(
        self,
        *,
        env: LiberoEnvClient,
        sam3_client: Sam3Client,
        state: EnvState,
        check_cancelled: Callable[[], None],
        config: ControlConfig = ControlConfig(),
    ) -> None:
        self.env = env
        self._sam3 = sam3_client
        self._state = state
        self._check_cancelled = check_cancelled
        self._config = config
        # The client resets on connect, leaving the gripper fully open.
        self._open_gap = self._finger_gap()
        self.env_steps = 0
        self.frames = [np.asarray(env.last_obs["main_images"])]
        self._masks: dict[str, tuple[int, str, np.ndarray]] = {}
        self._mask_ids = itertools.count(1)

    @property
    def ended(self) -> bool:
        return self.env.terminated or self.env.truncated

    def _finger_gap(self) -> float:
        states = np.asarray(self.env.last_obs["states"], dtype=np.float64)
        return float(abs(states[6]) + abs(states[7]))

    def measure(self) -> Measurement:
        """Read the pose and opening from the latest environment observation."""
        states = np.asarray(self.env.last_obs["states"], dtype=np.float64)
        return Measurement(
            xyz=states[:3],
            rotation=Rotation.from_rotvec(states[3:6]).as_matrix(),
            opening=min(1.0, self._finger_gap() / self._open_gap),
        )

    def _step(self, action: np.ndarray) -> None:
        self._check_cancelled()
        self.env.step(action)
        self.env_steps += 1
        self.frames.append(np.asarray(self.env.last_obs["main_images"]))

    def _pose_action(
        self, current: Measurement, xyz: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Return one bounded servo action and the remaining pose errors."""
        position_error = xyz - current.xyz
        rotation_error = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        action = np.zeros(7)
        action[:3] = _clip_norm(position_error, self._config.position_step_m)
        action[:3] /= POSITION_SCALE_M
        action[3:6] = _clip_norm(rotation_error, self._config.rotation_step_rad)
        action[3:6] /= ROTATION_SCALE_RAD
        return (
            action,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(rotation_error)),
        )

    def _ended_reason(self) -> str:
        return "terminated" if self.env.terminated else "truncated"

    @tool
    def move_to(
        self,
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None,
        target_yaw: float | None = None,
        target_pitch: float | None = None,
    ) -> ToolResult:
        """Move the end effector to an absolute pose and stop when it settles. Omitted targets keep their measured values, and the gripper keeps its last set_gripper target. Motion follows a straight line without collision avoidance. Returns status reached, partial with a reason, or error, with the measured pose and opening.

        Args:
            xyz: Target position [x, y, z], world frame, meters.
            target_yaw: Target heading of the gripper x-axis in the world xy-plane, atan2(R[1,0], R[0,0]), radians.
            target_pitch: Target tilt of the gripper approach axis about the world x-axis, atan2(R[1,2], -R[2,2]), radians; 0 points straight down.
        """
        if self.ended:
            return _error("episode_ended")
        if xyz is None and target_yaw is None and target_pitch is None:
            return _error("no_target")
        start = self.measure()
        target = start.xyz if xyz is None else np.asarray(xyz, dtype=np.float64)
        try:
            goal = geometry.orientation_goal(start.rotation, target_yaw, target_pitch)
        except ValueError as exc:
            return _error(f"orientation_unreachable: {exc}")

        config = self._config
        errors: list[tuple[float, float]] = []
        reason = None
        previous = start
        for steps in itertools.count():
            current = self.measure()
            action, distance, angle = self._pose_action(current, target, goal)
            errors.append((distance, angle))
            settled = (
                np.linalg.norm(current.xyz - previous.xyz) < config.settle_speed_m
                and Rotation.from_matrix(
                    current.rotation @ previous.rotation.T
                ).magnitude()
                < config.settle_speed_rad
            )
            previous = current
            within = (
                distance < config.position_tolerance_m
                and angle < config.rotation_tolerance_rad
            )
            stalled = _stalled(errors, config)
            # Within tolerance, residual jitter without progress is arrival.
            if within and (settled or stalled or steps >= config.max_motion_steps):
                break
            if self.ended:
                reason = self._ended_reason()
            elif steps >= config.max_motion_steps:
                reason = "max_steps"
            elif stalled:
                reason = "stalled"
            if reason is not None:
                break
            # Gripper 0 keeps the last finger command.
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "position_error_m": round(distance, 4),
                "rotation_error_rad": round(angle, 3),
            }
        )

    @tool
    def set_gripper(
        self, opening: Annotated[float, Field(ge=0.0, le=1.0)]
    ) -> ToolResult:
        """Drive the gripper to a normalized opening while holding the end-effector pose. Closing on an object stops early and reports the measured opening; contact does not imply a stable grasp. The target stays in effect during later moves.

        Args:
            opening: Target opening from 0 (closed) to 1 (fully open).
        """
        if self.ended:
            return _error("episode_ended")
        config = self._config
        hold = self.measure()
        previous, still, reason = hold.opening, 0, None
        for steps in itertools.count():
            current = self.measure()
            error = opening - current.opening
            still = (
                still + 1
                if abs(current.opening - previous) < config.settle_opening
                else 0
            )
            previous = current.opening
            settled = still >= config.settle_steps
            within = abs(error) <= config.opening_tolerance
            if within and settled:
                break
            if self.ended:
                reason = self._ended_reason()
            elif steps >= config.max_gripper_steps:
                reason = "max_steps"
            elif settled and error < 0:
                # Opening may first unwind a squeezed command before fingers move.
                reason = "blocked"
            if reason is not None:
                break
            action, _, _ = self._pose_action(current, hold.xyz, hold.rotation)
            # Native gripper input: -1 opens, +1 closes, 0 keeps the command.
            action[6] = 0.0 if within else float(-np.sign(error))
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "opening_error": round(error, 3),
            }
        )

    def record_observation(
        self,
        *,
        command: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        elapsed_s: float | None = None,
    ) -> StepRecord:
        """Record the measured state with camera images and world maps."""
        with self._state.record_step(
            state={
                **self.measure().state(),
                "finger_gap_m": round(self._finger_gap(), 5),
            },
            terminated=self.env.terminated,
            truncated=self.env.truncated,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={"env_steps": self.env_steps},
        ) as step:
            for camera, sim_camera in SIM_CAMERAS.items():
                rgb, depth = self.env.render_camera(
                    sim_camera, IMAGE_SIZE, IMAGE_SIZE, depth=True
                )
                meta = self.env.get_camera_meta(sim_camera, IMAGE_SIZE, IMAGE_SIZE)
                if meta is None:
                    raise RuntimeError(f"{sim_camera} camera metadata is unavailable")
                # Flip rows so pixels match the calibration frame.
                world = geometry.world_from_depth(
                    geometry.metric_depth(depth, meta)[::-1], meta
                )
                self._save(f"{camera}.png", np.asarray(rgb)[::-1], step)
                self._save(f"{camera}_world.npz", world.astype(np.float16), step)
        return self._state.get(step)

    def _save(self, name: str, value: Any, step: int) -> None:
        if self._state.save(name, value, step=step) is None:
            raise RuntimeError(f"failed to save observation artifact {name}")

    def images(self, step: int, cameras: list[str]) -> list[bytes]:
        """Return the recorded PNG bytes for ``cameras`` at ``step``."""
        return [
            self._state.load_bytes(f"{camera}.png", step=step) for camera in cameras
        ]

    @tool(readonly=True)
    def view_env(
        self, camera: Literal["agentview", "wrist", "all"] = "all"
    ) -> ToolResult:
        """Return the latest camera images without advancing the environment. Images share the pixel frame used by segment and back_project.

        Args:
            camera: agentview, wrist, or all.
        """
        step = self._state.latest_step
        cameras = list(SIM_CAMERAS) if camera == "all" else [camera]
        return ToolResult(
            data={
                "step": step,
                "images": [
                    {"camera": name, "size": [IMAGE_SIZE, IMAGE_SIZE]}
                    for name in cameras
                ],
            },
            images=self.images(step, cameras),
        )

    @tool(readonly=True)
    def segment(
        self,
        prompt: str | None = None,
        point: Pixel | None = None,
        camera: Camera = "agentview",
        step: Step | None = None,
    ) -> ToolResult:
        """Segment one object with SAM3 in a recorded image and measure its visible surface. Provide prompt or point, not both. Returns found, score, mask_id, box [row_min, col_min, row_max, col_max] (max exclusive), and world-frame geometry of the visible points: world_xyz (per-axis median), top_z_m (95th-percentile z), size_xyz_m (5th to 95th percentile extent), and axis_yaw (major xy direction in [-pi/2, pi/2), null when the surface is not elongated).

        Args:
            prompt: Text naming the object.
            point: One pixel [row, col] on the object; requires step.
            camera: agentview or wrist.
            step: Observation step; prompt defaults to the latest.
        """
        if (prompt is None) == (point is None):
            return _error("provide exactly one of prompt or point")
        if point is not None and step is None:
            return _error("point requires step")
        record = self._record(step)
        if record is None:
            return _error(f"unknown step {step}")
        index = record.step_idx
        world = self._state.load(f"{camera}_world.npz", step=index)
        if point is not None and not _inside(point, world.shape):
            return _error("point outside the image")
        try:
            segmentation = self._sam3.segment(
                self._state.load_bytes(f"{camera}.png", step=index),
                text_prompt=prompt,
                point=point,
                min_score=self._config.segment_min_score,
            )
        except Exception as exc:
            return _error(f"segmentation_failed: {exc}")
        data: dict[str, Any] = {
            "step": index,
            "camera": camera,
            "found": segmentation.found,
            "score": None
            if segmentation.score is None
            else round(segmentation.score, 3),
        }
        if not segmentation.found:
            return ToolResult(data=data)

        mask = segmentation.mask
        mask_id = f"m_{index}_{camera}_{next(self._mask_ids)}"
        self._masks[mask_id] = (index, camera, mask)
        data["mask_id"] = mask_id
        data["box"] = (
            geometry.box_from_xyxy(segmentation.box, mask.shape)
            if segmentation.box is not None
            else geometry.box_from_mask(mask)
        )
        surface = geometry.surface_geometry(geometry.valid_points(world[mask]))
        if surface is None:
            data.update(
                dict.fromkeys(("world_xyz", "top_z_m", "size_xyz_m", "axis_yaw")),
                geometry_unavailable="insufficient depth points",
            )
        else:
            data.update(surface)
        image = self._state.load(f"{camera}.png", step=index)
        self._state.save(f"{mask_id}.png", _overlay(image, mask), step=index)
        return ToolResult(data=data)

    @tool(readonly=True)
    def back_project(
        self,
        pixel: Pixel | None = None,
        box: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None,
        mask_id: str | None = None,
        camera: Camera | None = None,
        step: Step | None = None,
        z_range_m: Annotated[list[float], Field(min_length=2, max_length=2)]
        | None = None,
    ) -> ToolResult:
        """Return the world-frame position of a pixel's surface point, or the per-axis median of the valid points in a box or segment mask. Box regions include any background they cover.

        Args:
            pixel: [row, col]; requires step.
            box: [row_min, col_min, row_max, col_max], max exclusive; requires step.
            mask_id: A mask_id from segment; it fixes the step and camera.
            camera: agentview or wrist for pixel and box (default agentview).
            step: Observation step for pixel and box.
            z_range_m: For box or mask, keep points with world z in [low, high].
        """
        if sum(value is not None for value in (pixel, box, mask_id)) != 1:
            return _error("provide exactly one of pixel, box, or mask_id")
        mask = None
        if mask_id is not None:
            if camera is not None or step is not None:
                return _error("mask_id fixes the step and camera")
            if mask_id not in self._masks:
                return _error(f"unknown mask_id {mask_id}")
            step, camera, mask = self._masks[mask_id]
        elif step is None:
            return _error("pixel and box require step")
        elif self._record(step) is None:
            return _error(f"unknown step {step}")
        world = self._state.load(f"{camera or 'agentview'}_world.npz", step=step)

        if pixel is not None:
            if z_range_m is not None:
                return _error("z_range_m applies to box and mask")
            if not _inside(pixel, world.shape):
                return _error("pixel outside the image")
            points = geometry.valid_points(world[pixel[0], pixel[1]])
        else:
            if box is not None:
                row_min, col_min = max(box[0], 0), max(box[1], 0)
                points = world[row_min : box[2], col_min : box[3]]
            else:
                points = world[mask]
            points = geometry.valid_points(points)
            if z_range_m is not None:
                low, high = z_range_m
                points = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
        needed = 1 if pixel is not None else MIN_REGION_POINTS
        if len(points) < needed:
            return _error(f"insufficient depth: {len(points)} valid points")
        return ToolResult(
            data={
                "step": step,
                "world_xyz": [round(float(v), 4) for v in np.median(points, axis=0)],
            }
        )

    @tool(readonly=True)
    def finish(self) -> ToolResult:
        """End the episode and report the environment's own success flags."""
        return ToolResult(
            data={
                "_finish": True,
                "success": self.env.terminated,
                "terminated": self.env.terminated,
                "truncated": self.env.truncated,
            }
        )

    def _record(self, step: int | None) -> StepRecord | None:
        try:
            return self._state.get(-1 if step is None else step)
        except KeyError:
            return None


def _error(message: str) -> ToolResult:
    return ToolResult(data={"error": message})


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


def _stalled(errors: list[tuple[float, float]], config: ControlConfig) -> bool:
    """Whether position and rotation errors stopped shrinking."""
    if len(errors) <= config.stall_steps:
        return False
    (old_distance, old_angle), (distance, angle) = (
        errors[-1 - config.stall_steps],
        errors[-1],
    )
    return (
        old_distance - distance < config.stall_progress_m
        and old_angle - angle < config.stall_progress_rad
    )


def _inside(pixel: list[int], shape: tuple[int, ...]) -> bool:
    return 0 <= pixel[0] < shape[0] and 0 <= pixel[1] < shape[1]


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the mask red for log inspection."""
    overlay = np.asarray(image)[..., :3].copy()
    overlay[mask] = (0.55 * overlay[mask] + [0.45 * 255, 0, 0]).astype(np.uint8)
    return overlay
