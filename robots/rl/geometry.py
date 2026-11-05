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

"""End-effector angles and visible-surface geometry for the LIBERO RL tools."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

#: Minimum number of valid points for surface statistics.
MIN_SURFACE_POINTS = 10
#: Major/minor xy variance ratio required to report a surface direction.
_AXIS_RATIO = 2.0
_MIN_HEADING = 1e-6


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_of(rotation: np.ndarray) -> float | None:
    """Return the gripper x-axis heading in the world xy-plane.

    Returns None when the x-axis is vertical and the heading is undefined.
    """
    if math.hypot(rotation[0, 0], rotation[1, 0]) < _MIN_HEADING:
        return None
    return math.atan2(rotation[1, 0], rotation[0, 0])


def pitch_of(rotation: np.ndarray) -> float:
    """Return the approach-axis tilt about the world x-axis; 0 points down."""
    return math.atan2(rotation[1, 2], -rotation[2, 2])


def orientation_goal(
    rotation: np.ndarray,
    target_yaw: float | None,
    target_pitch: float | None,
) -> np.ndarray:
    """Return the rotation nearest ``rotation`` with the requested yaw and pitch.

    Omitted targets keep their measured values. The goal applies world-frame
    corrections about z then x to ``rotation``, solved by Newton iteration.

    Raises:
        ValueError: The yaw is undefined or the targets cannot be met together.
    """
    if target_yaw is None and target_pitch is None:
        return rotation
    yaw = yaw_of(rotation)
    if yaw is None:
        raise ValueError("yaw is undefined while the gripper x-axis is vertical")
    targets = np.array(
        [
            yaw if target_yaw is None else target_yaw,
            pitch_of(rotation) if target_pitch is None else target_pitch,
        ]
    )

    def corrected(delta: np.ndarray) -> np.ndarray:
        return Rotation.from_euler("zx", delta).as_matrix() @ rotation

    def residual(delta: np.ndarray) -> np.ndarray:
        goal = corrected(delta)
        yaw = yaw_of(goal)
        if yaw is None:
            raise ValueError("yaw becomes undefined at the requested pitch")
        return np.array(
            [wrap_angle(targets[0] - yaw), wrap_angle(targets[1] - pitch_of(goal))]
        )

    delta = np.zeros(2)
    for _ in range(20):
        error = residual(delta)
        if np.abs(error).max() < 1e-7:
            return corrected(delta)
        jacobian = np.column_stack(
            [(error - residual(delta + step)) / 1e-6 for step in np.eye(2) * 1e-6]
        )
        delta += np.linalg.lstsq(jacobian, error, rcond=None)[0]
    raise ValueError("the requested yaw and pitch cannot be reached together")


def metric_depth(depth: np.ndarray, camera_meta: dict[str, Any]) -> np.ndarray:
    """Convert a normalized MuJoCo depth buffer to meters."""
    buffer = np.asarray(depth, dtype=np.float64)
    if buffer.ndim == 3:
        buffer = buffer[..., 0]
    near, far = camera_meta["depth_near"], camera_meta["depth_far"]
    return near / (1.0 - buffer * (1.0 - near / far))


def world_from_depth(depth_m: np.ndarray, camera_meta: dict[str, Any]) -> np.ndarray:
    """Back-project a metric depth image to world-frame points."""
    intrinsic = np.asarray(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.asarray(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    rows, cols = np.mgrid[0 : depth_m.shape[0], 0 : depth_m.shape[1]]
    camera_points = np.stack(
        [
            (cols - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * depth_m / intrinsic[1, 1],
            depth_m,
            np.ones_like(depth_m),
        ],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


def valid_points(points: np.ndarray) -> np.ndarray:
    """Keep finite, nonzero world points as an ``(N, 3)`` float array."""
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    keep = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 1e-6)
    return flat[keep]


def surface_geometry(points: np.ndarray) -> dict[str, Any] | None:
    """Summarize visible surface points, or return None when too few exist."""
    if len(points) < MIN_SURFACE_POINTS:
        return None
    low, high = np.percentile(points, [5, 95], axis=0)
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    variances, axes = np.linalg.eigh(xy.T @ xy / len(xy))
    axis_yaw = None
    if variances[1] > _AXIS_RATIO * variances[0]:
        heading = math.atan2(axes[1, 1], axes[0, 1])
        axis_yaw = round((heading + math.pi / 2) % math.pi - math.pi / 2, 3)
    return {
        "world_xyz": _rounded(np.median(points, axis=0)),
        "top_z_m": round(float(high[2]), 4),
        "size_xyz_m": _rounded(high - low),
        "axis_yaw": axis_yaw,
    }


def box_from_xyxy(box: list[float], shape: tuple[int, ...]) -> list[int]:
    """Convert a SAM3 ``[x0, y0, x1, y1]`` box to an exclusive row/col box."""
    height, width = shape[:2]
    x0, y0, x1, y1 = box
    return [
        min(max(math.floor(y0), 0), height),
        min(max(math.floor(x0), 0), width),
        min(max(math.ceil(y1), 0), height),
        min(max(math.ceil(x1), 0), width),
    ]


def box_from_mask(mask: np.ndarray) -> list[int]:
    """Return the exclusive row/col bounds of a nonempty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return [int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1]


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in values]
