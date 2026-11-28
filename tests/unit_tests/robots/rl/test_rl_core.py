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

"""Unit tests for RPent refactored robots.rl modules."""

from __future__ import annotations

import numpy as np
import pytest

from robots.rl.conversation import (
    CompactionConfig,
    EpisodeConversation,
    count_image_tokens,
    file_id_for,
)
from robots.rl.geometry import (
    encode_png,
    map_box,
    map_pixel,
    surface_geometry,
    world_from_depth,
)
from robots.rl.toolkit import (
    ActionBatch,
    DualArmBatch,
    MobileBatch,
    PositionActionBatch,
    episode_tools,
    render_prompts,
)
from robots.rl.units import Centimeters, validate_relations


def test_centimeters_roundtrip():
    cm = Centimeters()
    args_m = cm.payload(
        {"xyz": [12.34, -56.78, 90.12], "orientation_rpy": [10.0, -20.0, 30.0]},
        to_model=False,
    )
    assert args_m["xyz"] == pytest.approx([0.1234, -0.5678, 0.9012])
    assert args_m["orientation_rpy"] == pytest.approx([10.0, -20.0, 30.0])
    out = cm.payload({"xyz": [0.1234, -0.5678, 0.9012], "step": 1}, to_model=True)
    assert out["xyz"] == pytest.approx([12.34, -56.78, 90.12])
    assert out["step"] == 1
    exact = cm.payload(
        {
            "xyz": [0.6014, -0.0044, 0.1501],
            "position_error_m": 0.0034,
            "finger_gap_m": 0.10034,
        },
        to_model=True,
    )
    assert exact["xyz"] == [60.14, -0.44, 15.01]
    assert exact["position_error_cm"] == 0.34
    assert exact["finger_gap_cm"] == 10.034
    with pytest.raises(ValueError):
        validate_relations("back_project", {"box": [50, 10, 20, 40]})


def test_geometry_backproject_and_png():
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[40:80, 50:90] = (200, 100, 50)
    png = encode_png(rgb)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(file_id_for(png)) == 6
    assert count_image_tokens(png) > 0
    resized = encode_png(rgb, size=64)
    assert resized[:8] == b"\x89PNG\r\n\x1a\n"
    assert count_image_tokens(resized) == 4
    assert map_pixel([32, 32], (64, 64), (128, 128)) == [65, 65]
    assert map_box([16, 16, 48, 48], (64, 64), (128, 128)) == [32, 32, 96, 96]

    depth = np.full((64, 64), 1.5, dtype=np.float32)
    meta = {
        "intrinsic_K": [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsic_cam2world": np.eye(4).tolist(),
    }
    pts = world_from_depth(depth, meta)
    assert pts.shape == (64, 64, 3)
    assert pts[32, 32] == pytest.approx([0.0, 0.0, 1.5], abs=1e-5)
    geom = surface_geometry(pts.reshape(-1, 3))
    assert geom is not None
    assert geom["world_xyz"][2] == pytest.approx(1.5, abs=1e-3)


@pytest.mark.parametrize(
    "env",
    ["libero", "robotwin", "robocasa", "robodojo", "metaworld", "robolab", "maniskill"],
)
def test_prompts_all_envs(env: str):
    sys_p, usr_m = render_prompts(env, reasoning=True, max_actions_per_call=3)
    assert sys_p.strip()
    assert usr_m.strip()


def test_action_batch_declarations():
    for cls in (ActionBatch, DualArmBatch, MobileBatch, PositionActionBatch):
        assert hasattr(cls(None), "execute_actions")


def test_metaworld_tool_declarations():
    tools = {t.name: t for t in episode_tools(None, "metaworld")}
    assert tools["view_env"].input_schema["properties"]["camera"]["enum"] == [
        "corner2",
        "gripperPOV",
        "all",
    ]
    desc = tools["execute_actions"].description
    assert "validated before any movement" in desc
    assert "Stop immediately" in desc
    assert "view_env" in desc
    action = {"action": "move_to", "xyz": [1, 2, 3]}
    tools["execute_actions"].validate_arguments({"actions": [action]})
    with pytest.raises(ValueError):
        tools["execute_actions"].validate_arguments(
            {"actions": [{**action, "orientation_rpy": [0, 0, 0]}]}
        )


def test_conversation_validation_retry_and_compaction():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "view_env",
                "description": "Render views",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cameras": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["cameras"],
                },
            },
        }
    ]
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text='{"step": 0}',
        tools=tools,
        environment="libero",
        compaction=CompactionConfig(
            max_messages=6, keep_messages=2, context_window=262144
        ),
    )
    assert conv.wire_messages()[0]["role"] == "system"
    assert conv.wire_messages()[1]["role"] == "user"

    conv.append_assistant(
        content=None,
        reasoning_content="thinking",
        tool_calls=[
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "view_env", "arguments": "{}"},
            }
        ],
    )
    err_kind, err_msg, args, should_retry = conv.validate_or_retry(
        [{"id": "c0", "name": "view_env", "arguments": "{}"}]
    )
    assert err_kind is not None and should_retry is True and args is None
    assert "Fix the errors and try again." in conv.wire_messages()[-1]["content"]

    png = encode_png(np.zeros((32, 32, 3), dtype=np.uint8))
    for i in range(4):
        cid = f"call_{i}"
        conv.append_assistant(
            content=None,
            tool_calls=[
                {
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": "view_env",
                        "arguments": '{"cameras": ["agentview"]}',
                    },
                }
            ],
        )
        conv.append_tool_result(
            tool_name="view_env", tool_call_id=cid, text='{"ok": true}', images=[png]
        )

    assert conv.maybe_compact() is True
    msgs = conv.wire_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "TASK" in msgs[1]["content"]
    assert len(conv.messages) < 8


def test_binary_execute_framing_and_zero_base64_conversation():
    from robots.rl.server import pack_execute_response, unpack_execute_response

    png1 = encode_png(np.full((32, 32, 3), 10, dtype=np.uint8))
    png2 = encode_png(np.full((32, 32, 3), 200, dtype=np.uint8))
    meta = {
        "text": '{"ok": true}',
        "done": False,
        "success": False,
        "truncated": False,
        "infrastructure_error": None,
        "sam3_service_errors": [],
    }
    packed = pack_execute_response(meta, [png1, png2])
    out_meta, out_imgs = unpack_execute_response(packed)
    assert out_meta == meta
    assert out_imgs == [png1, png2]

    # Empty image list roundtrip
    out_meta0, out_imgs0 = unpack_execute_response(pack_execute_response(meta, []))
    assert out_meta0 == meta
    assert out_imgs0 == []

    # EpisodeConversation keeps raw bytes without eager Base64, caches OpenAI wire once
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text="{}",
        tools=[],
        environment="libero",
    )
    conv.append_tool_result(
        tool_name="view_env", tool_call_id="c1", text="{}", images=[png1, png2]
    )
    lightweight = conv.wire_messages(inline_base64=False)
    urls = [
        part["image_url"]["url"]
        for msg in lightweight
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part.get("type") == "image_url"
    ]
    assert len(urls) == 2
    assert all(u.startswith("rpent-image://") for u in urls)
    assert conv.images() == [png1, png2]

    openai_msgs_a = conv.wire_messages(inline_base64=True)
    openai_msgs_b = conv.wire_messages(inline_base64=True)
    assert openai_msgs_a[-1] is openai_msgs_b[-1]  # cached across turns
    assert openai_msgs_a[-1]["content"][2]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_sim_pool_replace_worker_is_idempotent():
    import asyncio
    from unittest.mock import MagicMock

    from robots.rl.server import SimPoolManager, _WorkerHandle

    async def _run():
        mgr = SimPoolManager(
            environment="libero",
            gpus=[0],
            num_workers=1,
            sim_python="python",
            sam3_endpoint=None,
        )
        spawned: list[int | None] = []

        async def _fake_spawn(gpu: int | None):
            spawned.append(gpu)
            return MagicMock()

        mgr._spawn_worker = _fake_spawn  # type: ignore[method-assign]
        proc = MagicMock()
        proc.poll.return_value = 0
        handle = _WorkerHandle(
            worker_id=0,
            gpu=0,
            proc=proc,
            reader=MagicMock(),
            writer=MagicMock(),
        )
        mgr._workers.add(handle)
        await asyncio.gather(
            mgr._replace_worker(handle),
            mgr._replace_worker(handle),
            mgr._replace_worker(handle),
        )
        assert handle.replaced is True
        assert handle not in mgr._workers
        assert spawned == [0]

    asyncio.run(_run())


def test_harness_v2_centimeters_preserves_jaw_mm():
    cm = Centimeters()
    out = cm.payload(
        {
            "opening": 0.45,
            "jaw_mm": 36.2,
            "position_error_m": 0.0025,
            "left": {"opening": 0.5, "jaw_mm": 44.0, "pose": {"xyz": [0.1, -0.2, 0.3]}},
        },
        to_model=True,
    )
    assert out["jaw_mm"] == 36.2
    assert out["position_error_cm"] == 0.25
    assert out["left"]["jaw_mm"] == 44.0
    assert out["left"]["pose"]["xyz"] == [10.0, -20.0, 30.0]


@pytest.mark.parametrize(
    ("env", "expected_observe"),
    [
        ("libero", ["none", "all", "agentview", "wrist"]),
        ("robotwin", ["none", "all", "head", "left_wrist", "right_wrist"]),
        ("robocasa", ["none", "all", "agentview", "agentview_right", "wrist"]),
        ("robodojo", ["none", "all", "cam_head", "cam_left_wrist", "cam_right_wrist"]),
        ("robolab", ["none", "all", "over_shoulder_left_camera", "wrist_cam"]),
        ("metaworld", ["none", "all", "corner2", "gripperPOV"]),
    ],
)
def test_harness_v2_observe_enum_and_prompts_all_6_envs(
    env: str, expected_observe: list[str]
):
    from robots.rl.toolkit import openai_tool_schemas

    tools = {t.name: t for t in episode_tools(None, env)}
    assert set(tools) == {"view_env", "segment", "back_project", "execute_actions"}
    observe_schema = tools["execute_actions"].input_schema["properties"]["observe"]
    assert observe_schema["enum"] == expected_observe

    # Schema rejects invalid observe values via EpisodeConversation.validate_or_retry
    sys_p, usr_m = render_prompts(
        env, reasoning=True, max_actions_per_call=3, include_jaw_mm=True
    )
    assert "Images are supplied only through view_env" not in sys_p
    assert "step=0" in sys_p
    assert "observe" in sys_p
    assert "jaw_mm" in sys_p
    sys_no_jaw, _ = render_prompts(env, include_jaw_mm=False)
    assert "jaw_mm" not in sys_no_jaw

    conv = EpisodeConversation(
        system_prompt=sys_p,
        user_message=usr_m,
        initial_observation_text="{}",
        tools=openai_tool_schemas(list(tools.values())),
        environment=env,
    )
    bad_action = (
        {"action": "set_gripper", "arm": "left", "opening": 0}
        if env in {"robotwin", "robodojo"}
        else {"action": "set_gripper", "opening": 0}
    )
    err_kind, _, _, _ = conv.validate_or_retry(
        [
            {
                "id": "c1",
                "name": "execute_actions",
                "arguments": {
                    "actions": [bad_action],
                    "observe": "invalid_camera_name",
                },
            }
        ]
    )
    assert err_kind == "tool_arguments"
    ok_kind, _, parsed, _ = conv.validate_or_retry(
        [
            {
                "id": "c2",
                "name": "execute_actions",
                "arguments": {
                    "actions": [bad_action],
                    "observe": expected_observe[-1],
                },
            }
        ]
    )
    assert ok_kind is None and parsed is not None
    assert parsed["observe"] == expected_observe[-1]


def test_harness_v2_f1_f2_f3_f5_toolkit_across_all_6_envs(monkeypatch):
    import io
    import json

    from PIL import Image

    from robots.rl.geometry import Coordinates
    from robots.rl.perception import RecordedPerception
    from robots.rl.toolkit import RLToolkit
    from robots.rl.units import CAMERAS
    from rpent.tools import ToolResult

    env_specs = {
        "libero": ((1024, 1024), (512, 512), False, 42.5),
        "robocasa": ((1024, 1024), (512, 512), False, 51.0),
        "metaworld": ((1024, 1024), (512, 512), False, 35.3),
        "robolab": ((720, 1280), (432, 768), False, 64.2),
        "robotwin": ((240, 320), (240, 320), True, {"left": 22.4, "right": 58.1}),
        "robodojo": ((480, 640), (480, 640), True, {"left": 18.0, "right": 88.0}),
    }

    for env_name, (native_hw, expected_hw, is_dual, expected_jaw) in env_specs.items():
        cams = CAMERAS[env_name]

        class _FakeEnv:
            terminated = False
            truncated = False

            def get_task_language(self) -> str:
                return f"Do {env_name} task"

        class _FakePrimitives(RecordedPerception):
            cameras = cams

            def __init__(self, *, env, sam3_client, state, check_cancelled, **kwargs):
                self._init_perception(
                    env=env,
                    sam3_client=sam3_client,
                    state=state,
                    check_cancelled=check_cancelled,
                    max_episode_steps=100,
                )
                self.coordinates = Coordinates(
                    origin=np.zeros(3),
                    rotation=np.eye(3),
                    table_height_m=0.0,
                    version=0,
                )
                self._capture_calls = 0

            def _display_shape(self, native_shape: tuple[int, int]) -> tuple[int, int]:
                if env_name == "robolab":
                    if (
                        self.display_size is None
                        or max(native_shape) <= self.display_size
                    ):
                        return native_shape
                    h, w = native_shape
                    scale = self.display_size / max(native_shape)
                    return max(1, round(h * scale)), max(1, round(w * scale))
                return super()._display_shape(native_shape)

            def _refresh_coordinates(self):
                pass

            def native_success(self) -> bool:
                return bool(self.env.terminated)

            def measure(self):
                class _M:
                    def state(self):
                        return {
                            "pose": {
                                "xyz": [0.1, 0.2, 0.3],
                                "orientation_rpy": [3.142, 0.0, 0.0],
                            },
                            "opening": 0.5,
                        }

                return _M()

            def jaw_mm(self):
                return expected_jaw

            def robot_state(self):
                if is_dual:
                    return {
                        "left": {
                            "pose": {
                                "xyz": [0.1, 0.2, 0.3],
                                "orientation_rpy": [3.142, 0.0, 0.0],
                            },
                            "opening": 0.5,
                        },
                        "right": {
                            "pose": {
                                "xyz": [0.1, -0.2, 0.3],
                                "orientation_rpy": [3.142, 0.0, 0.0],
                            },
                            "opening": 0.8,
                        },
                    }
                return {
                    "pose": {
                        "xyz": [0.1, 0.2, 0.3],
                        "orientation_rpy": [3.142, 0.0, 0.0],
                    },
                    "opening": 0.5,
                    "finger_gap_m": 0.05,
                }

            def capture(self, camera: str):
                self._capture_calls += 1
                h, w = native_hw
                rgb = np.full((h, w, 3), 120, dtype=np.uint8)
                world = np.ones((h, w, 3), dtype=np.float32)
                return rgb, world, {"height": h, "width": w}

            def move_to(self, **kwargs):
                self.env_steps += 5
                return ToolResult(
                    data={
                        "status": "reached",
                        "env_steps": 5,
                        "position_error_m": 0.0012,
                        "rotation_error_rad": 0.004,
                        **({"arm": kwargs["arm"]} if "arm" in kwargs else {}),
                    }
                )

            def set_gripper(self, **kwargs):
                self.env_steps += 2
                return ToolResult(
                    data={
                        "status": "reached",
                        "env_steps": 2,
                        "opening_error": 0.01,
                        **({"arm": kwargs["arm"]} if "arm" in kwargs else {}),
                    }
                )

            def move_base(self, **kwargs):
                self.env_steps += 10
                self.coordinates = Coordinates(
                    origin=np.array([0.1, 0.0, 0.0]),
                    rotation=np.eye(3),
                    table_height_m=0.0,
                    version=self.coordinates.version + 1,
                )
                return ToolResult(
                    data={
                        "status": "commanded",
                        "env_steps": 10,
                        "displacement_initial_base_m": [0.1, 0.0, 0.0],
                        "yaw_change_rad": 0.0,
                        "mount_height_change_m": 0.0,
                        "translation_error_m": 0.002,
                    }
                )

        monkeypatch.setattr(
            "robots.rl.toolkit.primitive_owner", lambda _: _FakePrimitives
        )

        # 1) Default include_jaw_mm=False and image_size=None (F2 + F3 off + F5 opening)
        tk_off = RLToolkit(
            runtime_kwargs={"env": _FakeEnv(), "sam3_client": None},
            environment=env_name,
        )
        obs_off = tk_off.initial_observation()
        assert "coordinates" in obs_off.data
        assert obs_off.data["position_unit"] == "cm"
        assert len(obs_off.images) == len(cams)
        assert [item["camera"] for item in obs_off.data["images"]] == list(cams)
        for img_bytes, item in zip(obs_off.images, obs_off.data["images"], strict=True):
            assert item["size"] == list(expected_hw)
            with Image.open(io.BytesIO(img_bytes)) as pil:
                assert pil.size == (expected_hw[1], expected_hw[0])
        if is_dual:
            assert "jaw_mm" not in obs_off.data["state"]["left"]
            assert "jaw_mm" not in obs_off.data["state"]["right"]
        else:
            assert "jaw_mm" not in obs_off.data["state"]

        # 2) include_jaw_mm=True (F1 + F3 on + F5 observe)
        tk = RLToolkit(
            runtime_kwargs={"env": _FakeEnv(), "sam3_client": None},
            environment=env_name,
            include_jaw_mm=True,
        )
        obs = tk.initial_observation()
        assert "coordinates" in obs.data
        if is_dual:
            assert obs.data["state"]["left"]["jaw_mm"] == expected_jaw["left"]
            assert obs.data["state"]["right"]["jaw_mm"] == expected_jaw["right"]
        else:
            assert obs.data["state"]["jaw_mm"] == expected_jaw
            assert "finger_gap_cm" not in obs.data["state"]

        # Subsequent view_env omits coordinates and position_unit
        view = tk.execute_tool("view_env", {"camera": cams[0]})
        assert "coordinates" not in view.data
        assert "position_unit" not in view.data

        # K=3 execute_actions with observe="none" is compact (<700 chars) and omits coordinates
        if is_dual:
            actions = [
                {"action": "move_to", "arm": "left", "xyz": [10, 20, 30]},
                {"action": "set_gripper", "arm": "left", "opening": 0},
                {"action": "move_to", "arm": "right", "xyz": [10, -20, 30]},
            ]
        else:
            actions = [
                {"action": "move_to", "xyz": [10, 20, 30]},
                {"action": "set_gripper", "opening": 0},
                {"action": "move_to", "xyz": [10, 20, 35]},
            ]
        res_none = tk.execute_tool("execute_actions", {"actions": actions})
        assert "coordinates" not in res_none.data
        assert res_none.images == []
        assert "observation" not in res_none.data
        assert len(res_none.data["results"]) == 3
        for item in res_none.data["results"]:
            assert "pose" not in item and "terminated" not in item
        assert res_none.data["results"][0]["position_error_cm"] == pytest.approx(0.12)
        serialized = json.dumps(res_none.data, separators=(",", ":"))
        limit = 800 if is_dual else 700
        assert len(serialized) < limit, (
            f"{env_name} payload too large ({len(serialized)}): {serialized}"
        )

        # execute_actions with observe=<single_cam> returns latest step image
        target_cam = cams[-1]
        res_obs = tk.execute_tool(
            "execute_actions",
            {"actions": [actions[0]], "observe": target_cam},
        )
        assert len(res_obs.images) == 1
        assert res_obs.data["observation"]["camera"] == target_cam
        assert (
            res_obs.data["observation"]["step"] == res_obs.data["final_state"]["step"]
        )
        assert res_obs.data["observation"]["images"] == [
            {"camera": target_cam, "size": list(expected_hw)}
        ]

        # Invalid observe rejected cleanly by execute_tool
        bad_obs = tk.execute_tool(
            "execute_actions",
            {"actions": [actions[0]], "observe": "not_a_camera"},
        )
        assert "error" in bad_obs.data

        # RoboCasa frame_version change triggers fresh coordinates once (no nested duplicate)
        if env_name == "robocasa":
            moved = tk.execute_tool(
                "execute_actions",
                {
                    "actions": [
                        {
                            "action": "move_base",
                            "vx": 10.0,
                            "vy": 0.0,
                            "yaw_rate": 0.0,
                            "duration_s": 1.0,
                        }
                    ]
                },
            )
            assert "coordinates" in moved.data
            assert moved.data["coordinates"]["frame_version"] == 1
            assert "coordinates" not in moved.data["final_state"]
            after_move = tk.execute_tool("view_env", {"camera": cams[0]})
            assert "coordinates" not in after_move.data

        # Terminated episode skips observe image rendering
        tk._primitives.env.terminated = True
        res_term = tk.execute_tool(
            "execute_actions",
            {"actions": [actions[0]], "observe": target_cam},
        )
        assert res_term.images == []
        assert "observation" not in res_term.data


def test_harness_v2_dynamic_compaction_and_fallback():
    png0 = encode_png(np.full((512, 512, 3), 10, dtype=np.uint8))
    pngs = [
        encode_png(np.full((512, 512, 3), 20 + i * 20, dtype=np.uint8))
        for i in range(6)
    ]
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text='{"step": 0, "images": [{"camera": "agentview", "size": [512, 512]}]}',
        tools=[],
        environment="libero",
        initial_images=[png0],
        initial_image_cameras=["agentview"],
        compaction=CompactionConfig(
            strategy="dynamic",
            max_messages=None,
            max_tokens=1500,
            keep_messages=4,
            keep_recent_images=2,
            keep_recent_thinking=2,
            preserve_opening_images=True,
        ),
    )
    # Opening message carries step=0 image
    assert conv.images() == [png0]
    assert '<observation step="0" camera="agentview"' in str(conv.wire_messages()[1])

    for i, img in enumerate(pngs):
        cid = f"call_{i}"
        conv.append_assistant(
            content=None,
            reasoning_content=f"long reasoning step {i} " + ("x" * 200),
            tool_calls=[
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": "view_env", "arguments": "{}"},
                }
            ],
        )
        conv.append_tool_result(
            tool_name="view_env",
            tool_call_id=cid,
            text=f'{{"step": {i + 1}}}',
            images=[img],
        )

    old_wire_snapshot = [dict(m) for m in conv.wire_messages()]
    old_first_tool_user_msg = conv.messages[2].wire[1]
    assert conv.maybe_compact() is True

    # Old wire dicts must NOT have been mutated in place
    assert old_first_tool_user_msg["content"][2]["type"] == "image_url"
    assert len(old_wire_snapshot) > 0

    # All 1 + 12 logical messages preserved (no text dropped)
    assert len(conv.messages) == 13
    # Opening image + last 2 intermediate images preserved (3 total)
    assert conv.images() == [png0, pngs[-2], pngs[-1]]
    image_url_parts = [
        part
        for msg in conv.wire_messages()
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part.get("type") == "image_url"
    ]
    assert len(image_url_parts) == len(conv.images()) == 3

    # Older intermediate images replaced with [image omitted]
    omitted_count = sum(
        1
        for msg in conv.wire_messages()
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part == {"type": "text", "text": "[image omitted]"}
    )
    assert omitted_count == 4

    # Only last 2 assistant messages retain reasoning_content
    assistants = [m for m in conv.messages if m.kind == "assistant"]
    assert all("reasoning_content" not in m.wire[0] for m in assistants[:-2])
    assert all("reasoning_content" in m.wire[0] for m in assistants[-2:])

    # Extreme pure-text overflow triggers base fallback cleanly (no infinite loop)
    conv.compaction = CompactionConfig(
        strategy="dynamic",
        max_messages=None,
        max_tokens=10,
        keep_messages=4,
        keep_recent_images=2,
        keep_recent_thinking=2,
        preserve_opening_images=True,
    )
    assert conv.maybe_compact() is True
    assert len(conv.messages) <= 6
    assert conv.messages[0].kind == "opening"
    assert conv.messages[0].images == (png0,)
    # Lossless full_messages retains every turn and image across compaction events
    assert len(conv.full_messages) == 13
    assert len(conv.compaction_events) == 2
    assert len(conv.all_images_by_id()) == 7


def test_rollout_schedule_sharding_interleaving_and_passes():
    from robots.rl.rollout import build_episode_schedule, parse_shard_spec

    rows = [
        {
            "query_id": f"robotwin-task_{t}-seed{s}",
            "answer": f'{{"environment": "robotwin", "task_name": "task_{t}", "seed": {s}}}',
        }
        for t in range(4)
        for s in range(6)
    ]
    assert parse_shard_spec("0/2") == (0, 2)
    s0 = build_episode_schedule(
        rows, interleave_tasks=True, shard_index=0, num_shards=2, target_count=12
    )
    s1 = build_episode_schedule(
        rows, interleave_tasks=True, shard_index=1, num_shards=2, target_count=12
    )
    # Both shards cover 100% of all 4 tasks with zero seed overlap
    assert {item["task_group"] for item in s0} == {f"task_{t}" for t in range(4)}
    assert {item["task_group"] for item in s1} == {f"task_{t}" for t in range(4)}
    assert {item["query_id"] for item in s0}.isdisjoint(
        {item["query_id"] for item in s1}
    )
    # First 4 scheduled episodes cover all 4 distinct tasks (round-robin interleaved)
    assert [item["task_group"] for item in s0[:4]] == [
        "task_0",
        "task_1",
        "task_2",
        "task_3",
    ]
    # Cycling beyond shard length increments pass_idx cleanly
    s0_cycled = build_episode_schedule(
        rows, interleave_tasks=True, shard_index=0, num_shards=2, target_count=15
    )
    assert len(s0_cycled) == 15
    assert s0_cycled[0]["episode_key"].endswith("__r0")
    assert s0_cycled[12]["episode_key"].endswith("__r1")
    assert len({item["episode_key"] for item in s0_cycled}) == 15


def test_rollout_atomic_save_retry_and_sft_export(tmp_path):
    import asyncio
    import json

    import httpx

    from robots.rl.rollout import (
        EndpointPool,
        _load_completed_episodes,
        detect_provider,
        run_episode,
        split_inline_thinking,
    )
    from robots.rl.server import BINARY_EXECUTE_MEDIA_TYPE, pack_execute_response
    from robots.rl.sft_export import convert_trajectory

    assert split_inline_thinking("<think>\nplan\n</think>\ndone", None) == (
        "done",
        "plan",
    )
    assert (
        detect_provider(
            "auto",
            model_name="deepseek-v4.1-flash",
            model_urls=["http://172.27.116.194:18080/maas/v1"],
            api_key="dummy",
        )
        == "openai"
    )
    assert (
        detect_provider(
            "auto",
            model_name="Qwen3.8-27B",
            model_urls=["http://127.0.0.1:30000/v1"],
            api_key=None,
        )
        == "sglang"
    )

    pool = EndpointPool(["http://h1:30000/v1", "http://h1:30001/v1"])
    u1 = pool.acquire()
    u2 = pool.acquire()
    assert u1 != u2
    pool.release(u1)
    assert pool.acquire() == u1

    png0 = encode_png(np.full((32, 32, 3), 15, dtype=np.uint8))
    png1 = encode_png(np.full((32, 32, 3), 180, dtype=np.uint8))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_actions",
                "description": "Execute actions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {"type": "array"},
                        "observe": {"type": "string"},
                    },
                    "required": ["actions"],
                },
            },
        }
    ]

    llm_calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/sessions/open":
            meta = {
                "session_id": "s1",
                "environment": "libero",
                "system_prompt": "SYS",
                "user_message": "TASK",
                "initial_observation": '{"step": 0, "images": [{"camera": "agentview"}]}',
                "tools": tools,
                "image_cameras": ["agentview"],
                "init_state_id": 0,
                "done": False,
                "success": False,
                "truncated": False,
            }
            return httpx.Response(
                200,
                headers={"content-type": BINARY_EXECUTE_MEDIA_TYPE},
                content=pack_execute_response(meta, [png0]),
            )
        if path == "/v1/chat/completions":
            llm_calls["n"] += 1
            if llm_calls["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0.01"}, json={})
            if llm_calls["n"] == 2:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "reasoning_content": "Move toward target first.",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "execute_actions",
                                                "arguments": '{"actions": [{"action": "move_to"}], "observe": "agentview"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "Close gripper to finish.",
                                "tool_calls": [
                                    {
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "execute_actions",
                                            "arguments": '{"actions": [{"action": "set_gripper"}]}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 150, "completion_tokens": 35},
                },
            )
        if path == "/sessions/s1/execute":
            body = json.loads(request.content)
            if body["arguments"].get("observe"):
                return httpx.Response(
                    200,
                    headers={"content-type": BINARY_EXECUTE_MEDIA_TYPE},
                    content=pack_execute_response(
                        {
                            "text": '{"step": 5}',
                            "done": False,
                            "success": False,
                            "truncated": False,
                        },
                        [png1],
                    ),
                )
            return httpx.Response(
                200,
                headers={"content-type": BINARY_EXECUTE_MEDIA_TYPE},
                content=pack_execute_response(
                    {
                        "text": '{"step": 10, "success": true}',
                        "done": True,
                        "success": True,
                        "truncated": False,
                    },
                    [],
                ),
            )
        if path == "/sessions/s1/close":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"Unexpected path {path}")

    async def _run():
        ep_dir = tmp_path / "episodes" / "libero-t0-s0__r0"
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
            return await run_episode(
                model_url="http://mock/v1",
                model_name="deepseek-v4.1-flash",
                sim_url="http://sim",
                task={"environment": "libero", "task_id": 0, "seed": 0},
                output_dir=ep_dir,
                episode_key="libero-t0-s0__r0",
                query_id="libero-t0-s0",
                api_key="test-key",
                provider="openai",
                max_llm_retries=3,
                client=client,
            )

    outcome = asyncio.run(_run())
    assert outcome["success"] is True
    assert outcome["turns"] == 2
    assert llm_calls["n"] == 3  # 1x 429 retry + 2 turns

    ep_dir = tmp_path / "episodes" / "libero-t0-s0__r0"
    traj_path = ep_dir / "trajectory.json"
    assert traj_path.is_file()
    assert "test-key" not in traj_path.read_text(encoding="utf-8")

    completed = _load_completed_episodes(tmp_path)
    assert "libero-t0-s0__r0" in completed
    assert completed["libero-t0-s0__r0"]["success"] is True

    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    # Export to ms_swift format
    ms_sample, drop = convert_trajectory(
        traj, traj_dir=ep_dir, export_format="ms_swift"
    )
    assert drop is None and ms_sample is not None
    assert isinstance(ms_sample["tools"], str)
    assert len(ms_sample["images"]) == 2
    assert sum(m["content"].count("<image>") for m in ms_sample["messages"]) == 2
    # Trailing tool_response after final execute_actions trimmed cleanly
    assert ms_sample["messages"][-1]["role"] == "tool_call"
    assert ms_sample["messages"][2] == {
        "role": "assistant",
        "content": "<think>\nMove toward target first.\n</think>\n",
    }
    assert ms_sample["messages"][4]["role"] == "tool_response"
    assert ms_sample["messages"][5]["role"] == "user"
    assert "<tool_response>" in ms_sample["messages"][5]["content"]

    # Export to qwen_hf format
    hf_sample, hf_drop = convert_trajectory(
        traj, traj_dir=ep_dir, export_format="qwen_hf"
    )
    assert hf_drop is None and hf_sample is not None
    assert isinstance(
        hf_sample["messages"][2]["tool_calls"][0]["function"]["arguments"], dict
    )
    assert hf_sample["messages"][2]["reasoning_content"] == "Move toward target first."

    # Verify HTTP 400 context-length error parser & automatic compaction recovery
    from robots.rl.rollout import _parse_context_length_error

    parsed_ctx = _parse_context_length_error(
        '{"message":"Requested token count exceeds the model\'s maximum context length of 65536 tokens. '
        'You requested a total of 71436 tokens: 55052 tokens from the input messages and 16384 tokens for the completion."}'
    )
    assert parsed_ctx is not None
    assert parsed_ctx.input_tokens == 55052
    assert parsed_ctx.max_context_tokens == 65536
    assert _parse_context_length_error('{"message":"invalid json"}') is None
