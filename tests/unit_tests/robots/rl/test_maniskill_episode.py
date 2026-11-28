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

"""ManiSkill worker schemas, seeded cameras, and native action termination."""

import asyncio
import io
import json
import os
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from robots.rl.perception import RecordedPerception
from robots.rl.server import (
    SimPoolManager,
    _WorkerSessionHost,
)
from robots.rl.toolkit import episode_tools


@pytest.mark.parametrize(
    ("profile", "allowed", "rejected"),
    [
        (
            "arm",
            {"action": "set_gripper", "opening": 0},
            {"action": "move_lift", "delta_cm": 2},
        ),
        (
            "pen",
            {"action": "move_to", "xyz": [2, 4, 8]},
            {"action": "set_gripper", "opening": 0},
        ),
        (
            "dual",
            {"action": "set_gripper", "arm": "left", "opening": 0},
            {"action": "set_gripper", "opening": 0},
        ),
    ],
)
def test_schema_follows_active_robot(profile, allowed, rejected):
    perception = RecordedPerception()
    perception.tool_capabilities = {
        "rgb_cameras": ("base", "wrist"),
        "depth_cameras": ("base",),
        "segment": False,
        "gripper": profile != "pen",
        "dual_arm": profile == "dual",
    }
    tools = {t.name: t for t in episode_tools(None, "maniskill", perception=perception)}
    assert set(tools) == {"view_env", "back_project", "execute_actions"}
    assert tools["view_env"].input_schema["properties"]["camera"]["enum"] == [
        "base",
        "wrist",
        "all",
    ]
    assert tools["back_project"].input_schema["properties"]["camera"]["enum"] == [
        "base",
        None,
    ]
    batch = tools["execute_actions"]
    assert batch.input_schema["properties"]["observe"]["enum"] == [
        "none",
        "all",
        "base",
        "wrist",
    ]
    batch.validate_arguments({"actions": [allowed], "observe": "wrist"})
    with pytest.raises(ValueError):
        batch.validate_arguments({"actions": [rejected]})


def test_recycling_closes_maniskill_recorder():
    async def scenario():
        manager = SimPoolManager(
            environment="maniskill",
            gpus=[0],
            num_workers=1,
            sim_python="python",
            sam3_endpoint=None,
            max_episodes_per_worker=1,
        )
        worker = Mock(
            episodes=0, replaced=False, alive=True, busy=False, rpc=AsyncMock()
        )
        worker.proc.poll.return_value = None
        manager._sessions["s"] = worker
        manager._replace_worker = AsyncMock()
        try:
            assert (await manager.close_session("s"))["recycled"]
            worker.rpc.assert_awaited_once_with({"op": "close"})
            await asyncio.sleep(0)
            manager._replace_worker.assert_awaited_once_with(worker)
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.environ.get("RPENT_TEST_MANISKILL") != "1",
    reason="requires a ManiSkill GPU runtime",
)
def test_native_worker_seed_camera_and_timeout(tmp_path):
    host = _WorkerSessionHost("maniskill", None, 120, None, None, 0)
    req = {
        "task": {"task_name": "PickCube-v1", "seed": 0},
        "max_episode_steps": 1,
        "image_size": 256,
        "record_artifacts": True,
        "output_dir": str(tmp_path),
    }
    try:
        opened = host.open(req)
        assert opened["init_state_id"] == 0 and not opened["done"]
        assert len(opened["images"]) == len(opened["image_cameras"]) > 0
        measured = json.loads(opened["initial_observation"])["state"]["pose"]["xyz"]
        rgb = host.execute({"name": "view_env", "arguments": {"camera": "base_camera"}})
        assert Image.open(io.BytesIO(rgb["images"][0])).size == (256, 256)
        target = [*measured[:2], measured[2] + 4]
        move = {"action": "move_to", "xyz": target}
        result = host.execute(
            {"name": "execute_actions", "arguments": {"actions": [move, move]}}
        )
        assert result["done"] and result["truncated"] and not result["success"]
        receipt = json.loads(result["text"])
        assert receipt["executed_count"] == 1
        assert receipt["results"][0]["env_steps"] == 1
        assert len(host._toolkit._primitives.frames) == 2
        reopened = host.open({**req, "record_artifacts": False})
        assert (
            json.loads(reopened["initial_observation"])["state"]["pose"]["xyz"]
            == measured
        )
        prim = host._toolkit._primitives
        prim.step_controllers()
        assert "sensor_data" not in prim._obs
        rgb, points, _ = prim.capture("base_camera")
        assert rgb.shape == points.shape == (1024, 1024, 3)
    finally:
        host.close()


@pytest.mark.skipif(
    os.environ.get("RPENT_TEST_MANISKILL") != "1",
    reason="requires a native GPU renderer",
)
def test_physx_back_projection_and_grasp_axes():
    import numpy as np

    from robots.rl.envs.maniskill import LocalManiSkillEnv, transform
    from robots.rl.toolkit import RLToolkit

    env = LocalManiSkillEnv("PickCube-v1", 0, render_device=0)
    try:
        tk = RLToolkit(
            runtime_kwargs={"env": env, "max_episode_steps": 50, "sam3_client": None},
            environment="maniskill",
            image_size=512,
        )
        prim = tk._primitives
        arm = prim.arms["arm"]
        mount = transform(arm.mount.pose)
        forward = mount[:3, 0].copy()
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        axes = np.column_stack([forward, np.cross(up, forward), up])
        np.testing.assert_allclose(prim.coordinates.rotation, axes, atol=1e-6)
        first, second = [transform(link.pose)[:3, 3] for link in arm.fingertips]
        closing = first - second
        wrist = transform(arm.joints[arm.arm_names[-1]].child_link.pose)[:3, 3]
        approach = (first + second) / 2 - wrist
        tool_axes = arm._tcp()[:3, :3] @ arm.tool_rotation
        assert tool_axes[:, 0] @ closing / np.linalg.norm(closing) > 0.99
        assert tool_axes[:, 2] @ approach > 0
        tk.include_jaw_mm = False
        assert "jaw_mm" not in tk._robot_state()
        tk.include_jaw_mm = True
        assert tk._robot_state()["jaw_mm"] > 0
        robot = arm.robot
        initial = robot.get_qpos().clone()
        closed = initial.clone()
        for name in arm.agent.gripper_joint_names:
            closed[0, robot.active_joints_map[name].active_index.item()] = 0
        robot.set_qpos(closed)
        assert tk._robot_state()["jaw_mm"] == pytest.approx(0, abs=0.5)
        robot.set_qpos(initial)
        native = env.scene.unwrapped
        camera = native._sensors["base_camera"].camera
        pose = transform(camera.get_global_pose())
        intr = np.asarray(camera.get_intrinsic_matrix())[0]
        count = 0
        for row in (260, 300, 340):
            # Sample the exposed tabletop on both sides of the robot.
            for col in (160, 350):
                x, y, _ = np.linalg.solve(intr, [col * 2 + 1, row * 2 + 1, 1])
                direction = pose[:3, :3] @ [1, -x, -y]
                direction /= np.linalg.norm(direction)
                hit = native.scene.px.raycast(pose[:3, 3], direction, 4.0)
                if hit is None or "table" not in hit.component.entity.name:
                    continue
                answer = tk.execute_tool(
                    "back_project",
                    {"pixel": [row, col], "camera": "base_camera", "step": 0},
                )
                expected = axes.T @ (hit.position - mount[:3, 3])
                assert (
                    np.linalg.norm(np.array(answer.data["world_xyz"]) / 100 - expected)
                    < 0.002
                )
                count += 1
        assert count == 6
    finally:
        env.close()


@pytest.mark.skipif(
    os.environ.get("RPENT_TEST_MANISKILL") != "1",
    reason="requires a native GPU renderer",
)
@pytest.mark.parametrize(
    "task",
    [
        "TwoRobotPickCube-v1",
        "PickCubeSO100-v1",
        "PickCubeWidowXAI-v1",
        "UnitreeG1TransportBox-v1",
        "SO100GraspCube-v1",
        "InsertFlower-v1",
        "DrawTriangle-v1",
        "DrawSVG-v1",
    ],
)
def test_native_tool_axes(task):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from robots.rl.envs.maniskill import LocalManiSkillEnv, transform
    from robots.rl.toolkit import RLToolkit

    env = LocalManiSkillEnv(task, 0, render_device=0)
    try:
        toolkit = RLToolkit(
            environment="maniskill",
            runtime_kwargs={"env": env, "max_episode_steps": 50, "sam3_client": None},
        )
        prim = toolkit._primitives
        if task == "SO100GraspCube-v1":
            rgb, points, _ = prim.capture("base_camera")
            assert rgb.shape == points.shape == (1024, 1024, 3)
            assert np.isfinite(points).all(axis=-1).mean() > 0.1
        if task == "InsertFlower-v1":
            workspace = (
                transform(env.scene.unwrapped.flower.pose)[:2, 3]
                - prim.coordinates.origin[:2]
            )
            assert prim.coordinates.rotation[:2, 0] @ workspace > 0
        if task.startswith("Draw"):
            assert prim._display_shape((768, 1024)) == (384, 512)
            assert prim._obs["sensor_data"]["base_camera"]["rgb"].shape[:2] == (
                768,
                1024,
            )
            prim.display_size = 2048
            assert prim._display_shape((768, 1024)) == (768, 1024)
        for side, arm in prim.arms.items():
            if hasattr(arm, "open_position"):
                # The SO100 has a hinged finger. Compare the calibrated open jaws.
                qpos = arm.robot.get_qpos().clone()
                qpos[0, arm.finger_index] = float(arm.open_position)
                for name, relation in (
                    getattr(getattr(arm.gripper, "config", None), "mimic", {}) or {}
                ).items():
                    index = int(np.asarray(arm.joints[name].active_index).item())
                    qpos[0, index] = float(
                        arm.open_position * relation.get("multiplier", 1)
                        + relation.get("offset", 0)
                    )
                arm.robot.set_qpos(qpos)
            axes = arm._tcp()[:3, :3] @ arm.tool_rotation
            np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-5)
            if not arm.has_gripper:
                shaft = (
                    transform(arm.ee.pose)[:3, 3]
                    - transform(arm.robot.links_map["panda_hand"].pose)[:3, 3]
                )
                shaft = axes.T @ (shaft / np.linalg.norm(shaft))
                np.testing.assert_allclose(
                    Rotation.from_euler("xyz", [np.pi, 0, 0.7]).apply(shaft),
                    [0, 0, -1],
                    atol=1e-5,
                )
                continue
            first, second = [transform(tip.pose)[:3, 3] for tip in arm.fingertips]
            if hasattr(arm, "joints"):
                wrist = arm.joints[arm.arm_names[-1]].child_link
            else:
                palm = "base" if side == "arm" else f"{side}_palm_link"
                wrist = arm.robot.links_map[palm]
            closing = first - second
            approach = (first + second) / 2 - transform(wrist.pose)[:3, 3]
            assert abs(axes[:, 0] @ closing) / np.linalg.norm(closing) > 0.99
            assert axes[:, 2] @ approach > 0
    finally:
        env.close()
