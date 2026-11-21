# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Shared mounting coordinates and measured Aloha finger-pad references."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl.coordinates import Coordinates


def pose_matrix(pose: Any) -> np.ndarray:
    """Convert a world xyz/wxyz pose to a rigid transform."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Expected seven finite xyz/wxyz values")
    if not np.isclose(np.linalg.norm(value[3:]), 1, atol=1e-4):
        raise ValueError("Expected a unit wxyz quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def mounting_calibration(left: Any, right: Any, table: Any) -> dict[str, Any]:
    """Construct forward/left/up axes at the midpoint of both mounting bases."""
    left, right, table = (np.asarray(v, dtype=np.float64) for v in (left, right, table))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (left, right, table)):
        raise ValueError("Mount and table positions must be finite xyz vectors")
    lateral = left - right
    lateral[2] = 0
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError("Two distinguishable arm mounting centres are required")
    lateral /= np.linalg.norm(lateral)
    up = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack((np.cross(lateral, up), lateral, up))
    return {
        "base_xyz": ((left + right) / 2).tolist(),
        "base_rotation": rotation.tolist(),
        "table_top_xyz": table.tolist(),
        "left_mount_xyz": left.tolist(),
        "right_mount_xyz": right.tolist(),
    }


@dataclass(frozen=True)
class RobotwinFrame:
    """Episode-fixed shared mounting frame for the two finger-pad TCPs."""

    coordinates: Coordinates
    offsets: dict[str, np.ndarray]
    calibration: dict[str, Any]

    @classmethod
    def from_calibration(cls, calibration: dict[str, Any]) -> "RobotwinFrame":
        coordinates = Coordinates.from_control("general", calibration)
        offsets = {}
        for side in ("left", "right"):
            offset = np.asarray(
                calibration["native_tcp_to_pad"][side], dtype=np.float64
            )
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("Native TCP to pad offsets must be finite xyz vectors")
            offsets[side] = offset
        return cls(coordinates, offsets, calibration)

    def world_pad_pose(self, arm: str, native_tcp: Any) -> np.ndarray:
        """Return the finger-pad pose in simulator world coordinates."""
        pose = pose_matrix(native_tcp)
        pose[:3, 3] += pose[:3, :3] @ self.offsets[arm]
        return pose

    def model_pose(self, arm: str, native_tcp: Any) -> dict[str, list[float]]:
        """Describe the measured pad centre in the model-facing frame."""
        pose = self.world_pad_pose(arm, native_tcp)
        rotation = self.coordinates.orientation(pose[:3, :3])
        return {
            "xyz": self.coordinates.points(pose[:3, 3]).tolist(),
            "orientation_rpy": Rotation.from_matrix(rotation).as_euler("xyz").tolist(),
        }

    def world_target(self, arm: str, xyz: Any, rpy: Any, current: Any) -> np.ndarray:
        """Map an absolute pad target back to the native world TCP."""
        start = self.model_pose(arm, current)
        position = np.asarray(start["xyz"] if xyz is None else xyz, dtype=np.float64)
        angles = np.asarray(
            start["orientation_rpy"] if rpy is None else rpy, dtype=np.float64
        )
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (position, angles)):
            raise ValueError(
                "xyz and orientation_rpy must contain three finite numbers"
            )
        rotation = self.coordinates.world_orientation(
            Rotation.from_euler("xyz", angles).as_matrix()
        )
        native_position = (
            self.coordinates.world_position(position) - rotation @ self.offsets[arm]
        )
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return np.r_[native_position, quaternion]

    def metadata(self) -> dict[str, Any]:
        return {
            **self.coordinates.metadata(),
            "native_tcp_to_pad": {
                side: offset.tolist() for side, offset in self.offsets.items()
            },
            "calibration": self.calibration,
        }


def robotwin_calibration(task: Any) -> dict[str, Any]:
    """Read mounting links and finger collision centres from the live robot."""
    robot = task.robot
    mounts, offsets, links = {}, {}, {}
    for side in ("left", "right"):
        joints = getattr(robot, f"{side}_arm_joints")
        mount = joints[0].parent_link
        mounts[side] = np.asarray(mount.get_pose().p, dtype=np.float64)
        finger_centres = []
        names = []
        for joint, _, _ in getattr(robot, f"{side}_gripper"):
            link = joint.child_link
            names.append(link.get_name())
            points = []
            for shape in link.get_collision_shapes():
                if hasattr(shape, "vertices"):
                    vertices = np.asarray(shape.vertices, dtype=np.float64)
                elif hasattr(shape, "half_size"):
                    half = np.asarray(shape.half_size)
                    vertices = np.array([-half, half])
                else:
                    continue
                local = shape.get_local_pose().to_transformation_matrix()
                points.append(vertices @ local[:3, :3].T + local[:3, 3])
            if not points:
                raise ValueError(
                    f"Missing finger collision geometry for {side}/{link.get_name()}"
                )
            vertices = np.concatenate(points)
            centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            world = link.get_pose().to_transformation_matrix()
            finger_centres.append(world[:3, 3] + world[:3, :3] @ centre)
        if len(finger_centres) != 2:
            raise ValueError(
                "Aloha calibration requires two coupled finger links per arm"
            )
        pad = np.mean(finger_centres, axis=0)
        tcp = pose_matrix(getattr(robot, f"get_{side}_tcp_pose")())
        offsets[side] = (tcp[:3, :3].T @ (pad - tcp[:3, 3])).tolist()
        links[side] = {"mount": mount.get_name(), "fingers": names}
    table = np.asarray(task.table.get_pose().p, dtype=np.float64)
    native = pose_matrix(robot.get_left_tcp_pose())
    return {
        "coordinate_calibration": mounting_calibration(
            mounts["left"], mounts["right"], table
        ),
        "pad_tcp_xyz": (native[:3, 3] + native[:3, :3] @ offsets["left"]).tolist(),
        "native_tcp_to_pad": offsets,
        "links": links,
        "pad_measurement": "midpoint_of_finger_collision_bbox_centres",
    }
