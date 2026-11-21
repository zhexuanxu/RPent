# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""The RoboTwin episode prompt and tools share one model-facing frame."""

from types import SimpleNamespace

import numpy as np

from robots.rl.prompt_bundle import system_prompt
from robots.rl.remote import remote_options
from robots.rl.toolkit import RLToolkit
from robots.robotwin.general import mounting_calibration
from rpent.utils import logging as rpent_logging
from tests.unit_tests.robots.rl.test_backend_contracts import Twin

OPTIONS = {
    "environment": "robotwin",
    "workflow": "episode",
    "world_cord": "general",
    "action_mode": "batch",
}


def test_robotwin_episode_contract_and_prompt():
    options = remote_options(OPTIONS)
    assert options["max_actions_per_call"] == 3
    prompt = system_prompt(OPTIONS)
    for phrase in (
        "execute_actions",
        "orientation_rpy",
        "metres",
        "left_wrist",
        "right_wrist",
        "back_project",
        "partial",
    ):
        assert phrase in prompt
    assert "quaternions" not in prompt


def test_robotwin_episode_tools_report_and_execute_in_the_same_frame(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rpent_logging, "_output_dir", tmp_path)
    env = Twin()
    offset = [0.02, 0.0, 0.0]
    calibrate = {
        "coordinate_calibration": mounting_calibration(
            [-0.3, -0.65, 0.78], [0.3, -0.65, 0.78], [0, 0, 0.74]
        ),
        "pad_tcp_xyz": [0.02, 0, 0.4],
        "native_tcp_to_pad": {"left": offset, "right": offset},
    }
    native_call = env._client.call
    env._client.call = lambda name: (
        calibrate if name == "env.coordinate_calibration" else native_call(name)
    )
    env.get_task_language = lambda: "Move both arms forward."
    env.get_camera_meta = lambda _: {
        "extrinsic_cv": np.eye(4),
        "intrinsic_K": np.eye(3),
    }
    env.render_camera = lambda camera, depth: (
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.ones((4, 6)),
    )
    toolkit = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": None},
        dashboard_events=SimpleNamespace(emit=lambda _event: None),
        memory=None,
        observation_mode="adaptive",
        output_dir=tmp_path,
        **OPTIONS,
    )
    initial = toolkit.initial_observation().data
    left = initial["state"]["left"]["pose"]
    np.testing.assert_allclose(left["xyz"], [0.65, -0.02, -0.38])
    assert initial["coordinates"]["tcp"] == "finger_pad_midpoint"
    assert {tool.name for tool in toolkit.list_tools()} == {
        "view_env",
        "segment",
        "back_project",
        "finish",
        "execute_actions",
    }
    images = toolkit.execute_tool("view_env", {"camera": "all"})
    assert len(images.images) == 3
    point = toolkit.execute_tool(
        "back_project", {"pixel": [0, 0], "step": 0, "camera": "head"}
    )
    np.testing.assert_allclose(point.data["world_xyz"], [0.65, 0, 0.22])

    move = {
        "action": "move_to",
        "arm": "left",
        "xyz": [0.67, -0.02, -0.38],
        "other_xyz": [0.68, -0.02, -0.38],
    }
    invalid = toolkit.execute_tool(
        "execute_actions", {"actions": [move, {"action": "set_gripper", "opening": 0}]}
    )
    assert invalid.is_error and not env.plans
    too_long = toolkit.execute_tool("execute_actions", {"actions": [move] * 4})
    assert too_long.is_error and not env.plans
    grip = {"action": "set_gripper", "arm": "right", "opening": 1, "other_opening": 1}
    result = toolkit.execute_tool("execute_actions", {"actions": [move, grip]})
    assert result.data["executed_count"] == 2
    assert result.data["results"][0]["status"] == "reached"
    np.testing.assert_allclose(env.plans[0][1][:3], [0, 0.02, 0.3], atol=1e-9)
    np.testing.assert_allclose(env.plans[1][1][:3], [0, 0.03, 0.3], atol=1e-9)
    np.testing.assert_allclose(
        result.data["final_state"]["right"]["pose"]["xyz"], move["other_xyz"]
    )
    env.failed_arm = "right"
    stopped = toolkit.execute_tool("execute_actions", {"actions": [move, grip]})
    assert stopped.data["executed_count"] == 1
    assert stopped.data["stop_reason"] == "plan_failed"
