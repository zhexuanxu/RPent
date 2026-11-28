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

"""Unit tests for Harness v2 conversation, compaction chain-splitting, rollout, server, and SFT export."""

from __future__ import annotations

import io
import json
from pathlib import Path
import struct

from PIL import Image
import pytest

from robots.rl.conversation import (
    CompactionConfig,
    EpisodeConversation,
    count_image_tokens,
    count_image_tokens_from_size,
    find_safe_cutoff,
)
from robots.rl.rollout import (
    EndpointPool,
    _normalize_tool_calls_for_turn,
    build_episode_schedule,
    split_inline_thinking,
)
from robots.rl.server import (
    SimPoolRouter,
    pack_execute_response,
    unpack_execute_response,
)
from robots.rl.sft_export import convert_trajectory_to_samples


def _make_png(width: int = 64, height: int = 64, color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "segment",
            "description": "Segment object",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": ["string", "null"]},
                    "camera": {"type": ["string", "null"]},
                    "step": {"type": ["integer", "null"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "back_project",
            "description": "Back-project box",
            "parameters": {
                "type": "object",
                "properties": {
                    "box": {"type": ["array", "null"]},
                    "z_range_cm": {"type": ["array", "null"]},
                },
            },
        },
    },
]


def test_smart_resize_image_token_counts_match_qwen3_5() -> None:
    assert count_image_tokens_from_size(512, 512) == 256
    assert count_image_tokens_from_size(1024, 1024) == 1024
    assert count_image_tokens_from_size(320, 240) == 80
    assert count_image_tokens_from_size(768, 432) == 336
    assert count_image_tokens_from_size(640, 480) == 300
    assert count_image_tokens(_make_png(320, 240)) == 80


def test_validate_rejects_nan_and_short_box() -> None:
    conv = EpisodeConversation(
        system_prompt="sys",
        user_message="task",
        initial_observation_text="{}",
        tools=SAMPLE_TOOLS,
        environment="libero",
    )
    err_kind, err_msg, args, should_retry = conv.validate_or_retry(
        [{"id": "call_0_0", "name": "back_project", "arguments": '{"box": [1, 2, 3]}'}]
    )
    assert err_kind == "tool_arguments"
    assert args is None
    assert should_retry is True
    assert "4 elements" in (err_msg or "")

    err_kind2, _, _, _ = conv.validate_or_retry(
        [{"id": "call_1_0", "name": "back_project", "arguments": '{"box": [NaN, 0, 10, 10]}'}]
    )
    assert err_kind2 == "tool_arguments"


def test_compaction_records_clean_chain_snapshots_and_sft_masks_prefix(tmp_path: Path) -> None:
    img0 = _make_png(512, 512, (1, 2, 3))
    img1 = _make_png(512, 512, (4, 5, 6))
    img2 = _make_png(512, 512, (7, 8, 9))
    img3 = _make_png(512, 512, (10, 11, 12))

    cfg = CompactionConfig(
        strategy="dynamic",
        max_messages=None,
        max_tokens=1000,
        keep_messages=4,
        keep_recent_images=1,
        keep_recent_thinking=1,
        preserve_opening_images=True,
    )
    conv = EpisodeConversation(
        system_prompt="sys",
        user_message="open task",
        initial_observation_text='{"images":[{"camera":"agentview"}]}',
        tools=SAMPLE_TOOLS,
        environment="libero",
        compaction=cfg,
        initial_images=[img0],
        initial_image_cameras=["agentview"],
    )

    # Turn 0 (Chain 0)
    conv.append_assistant(
        content="Locating target.",
        reasoning_content="Thought 0 " + ("x" * 200),
        tool_calls=[
            {"id": "call_0_0", "type": "function", "function": {"name": "segment", "arguments": '{"prompt": "mug", "step": null}'}}
        ],
    )
    conv.append_tool_result(tool_name="segment", tool_call_id="call_0_0", text='{"ok":true}', images=[img1])

    # Turn 1 (Chain 0)
    conv.append_assistant(
        content=None,
        reasoning_content="Thought 1 " + ("y" * 200),
        tool_calls=[
            {"id": "call_1_0", "type": "function", "function": {"name": "segment", "arguments": '{"prompt": "plate"}'}}
        ],
    )
    conv.append_tool_result(tool_name="segment", tool_call_id="call_1_0", text='{"ok":true}', images=[img2])

    # Trigger compaction between Chain 0 and Chain 1
    assert conv.maybe_compact(current_tokens=1500) is True
    assert len(conv.compaction_events) == 1

    # Turn 2 (Chain 1: generated after compaction!)
    conv.append_assistant(
        content="Final step.",
        reasoning_content="Thought 2 after compaction",
        tool_calls=[
            {"id": "call_2_0", "type": "function", "function": {"name": "segment", "arguments": '{"prompt": "bowl"}'}}
        ],
    )
    conv.append_tool_result(tool_name="segment", tool_call_id="call_2_0", text='{"done":true}', images=[img3])

    chains = conv.export_chains()
    assert len(chains) == 2
    assert chains[0]["chain_id"] == 0
    assert chains[0]["turn_indices"] == [0, 1]
    assert chains[0]["prefix_wire_count"] == 2
    assert chains[1]["chain_id"] == 1
    assert chains[1]["turn_indices"] == [2]
    assert chains[1]["prefix_wire_count"] > 2

    # Save PNGs and verify SFT chain-split export
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    image_map: dict[str, str] = {}
    for fid, raw in conv.all_images_by_id().items():
        (img_dir / f"{fid}.png").write_bytes(raw)
        image_map[fid] = f"images/{fid}.png"

    traj = {
        "schema_version": 3,
        "episode_key": "ep0",
        "query_id": "q0",
        "environment": "libero",
        "model": {"name": "deepseek-v4.1-flash"},
        "outcome": {"success": True, "turns": 3, "termination_reason": "success"},
        "tools": SAMPLE_TOOLS,
        "images": image_map,
        "compaction_events": conv.compaction_events,
        "chains": chains,
        "full_messages": conv.full_wire_messages(),
        "compacted_messages": conv.wire_messages(),
    }

    samples, drops = convert_trajectory_to_samples(
        traj, traj_dir=tmp_path, export_format="ms_swift", include_metadata=True
    )
    assert not drops
    assert len(samples) == 2

    # Chain 0: turns 0 & 1 supervised; top-level null ("step": null) stripped!
    s0 = samples[0][0]
    assert s0["metadata"]["chain_id"] == 0
    assert s0["metadata"]["supervised_turns"] == 2
    assert s0["metadata"]["prefix_masked_turns"] == 0
    asst_s0 = [m for m in s0["messages"] if m["role"] == "assistant"]
    tc_s0 = [m for m in s0["messages"] if m["role"] == "tool_call"]
    assert all(m.get("loss") is not False for m in asst_s0 + tc_s0)
    assert "step" not in json.loads(tc_s0[0]["content"])["arguments"]

    # Chain 1: carried-over turns 0 & 1 have loss=False on BOTH assistant and tool_call;
    # turn 2 (post-compaction) is supervised (loss omitted), and [image omitted] appears in prefix!
    s1 = samples[1][0]
    assert s1["metadata"]["chain_id"] == 1
    assert s1["metadata"]["prefix_masked_turns"] == 2
    assert s1["metadata"]["supervised_turns"] == 1
    asst_s1 = [m for m in s1["messages"] if m["role"] == "assistant"]
    tc_s1 = [m for m in s1["messages"] if m["role"] == "tool_call"]
    assert [m.get("loss") for m in asst_s1] == [False, False, None]
    assert [m.get("loss") for m in tc_s1] == [False, False, None]
    assert any("[image omitted]" in m["content"] for m in s1["messages"] if m["role"] == "user")


def test_sft_export_masks_retry_and_empty_think_and_blocks_special_tokens(tmp_path: Path) -> None:
    traj = {
        "schema_version": 3,
        "episode_key": "ep_retry",
        "query_id": "q1",
        "environment": "libero",
        "model": {"name": "deepseek-v4.1-flash"},
        "outcome": {"success": False, "turns": 4, "termination_reason": "response_truncated"},
        "tools": SAMPLE_TOOLS,
        "images": {},
        "compaction_events": [],
        "chains": [
            {
                "chain_id": 0,
                "prefix_wire_count": 2,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "open"},
                    # Turn 0: invalid parallel tool call -> Validation feedback (must be masked!)
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "bad parallel",
                        "tool_calls": [
                            {"id": "call_00_a", "type": "function", "function": {"name": "segment", "arguments": '{"prompt":"a"}'}},
                            {"id": "call_00_b", "type": "function", "function": {"name": "segment", "arguments": '{"prompt":"b"}'}},
                        ],
                    },
                    {"role": "user", "content": "Validation feedback:\nUse one tool call per response.\n\nFix the errors and try again."},
                    # Turn 1: empty thinking -> must be masked!
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_00_c", "type": "function", "function": {"name": "segment", "arguments": '{"prompt":"a"}'}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_00_c", "content": '["ok"]'},
                    # Turn 2: valid recovery turn -> supervised!
                    {
                        "role": "assistant",
                        "content": "Done.",
                        "reasoning_content": "Valid thought",
                    },
                    # Turn 3: truncated final turn -> dropped by drop_truncated_final_turn!
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Infinite loop...",
                    },
                ],
            }
        ],
    }

    samples, drops = convert_trajectory_to_samples(traj, traj_dir=tmp_path, include_metadata=True)
    assert not drops
    assert len(samples) == 1
    s = samples[0][0]
    assts = [m for m in s["messages"] if m["role"] == "assistant"]
    tcs = [m for m in s["messages"] if m["role"] == "tool_call"]
    assert len(assts) == 3  # truncated Turn 3 dropped!
    assert [m.get("loss") for m in assts] == [False, False, None]
    assert [m.get("loss") for m in tcs] == [False, False, False]

    # Contamination guard rejects stray </think> or <image> in assistant content
    bad_traj = json.loads(json.dumps(traj))
    bad_traj["chains"][0]["messages"][6]["content"] = "Done.\n</think>\n"
    bad_samples, bad_drops = convert_trajectory_to_samples(bad_traj, traj_dir=tmp_path)
    assert not bad_samples
    assert any("content_special_token_contamination:</think>" in k for k in bad_drops)


def test_rollout_helpers_and_server_binary_unpack() -> None:
    # Deterministic call IDs
    wire, parsed = _normalize_tool_calls_for_turn(
        [{"id": "call_00_random", "type": "function", "function": {"name": "segment", "arguments": "{}"}}],
        turn_idx=3,
    )
    assert wire[0]["id"] == "call_3_0"
    assert parsed[0]["id"] == "call_3_0"

    # split_inline_thinking handles empty string reasoning_content
    c, r = split_inline_thinking("<think>\nhi\n</think>\nhello", "")
    assert c == "hello" and r == "hi"
    c2, r2 = split_inline_thinking("   ", "")
    assert c2 is None and r2 is None

    # Disjoint sharding & multi-suite LIBERO task grouping
    rows = [
        {"query_id": "s1-t0-0", "answer": json.dumps({"suite": "libero_spatial", "task": 0, "seed": 0})},
        {"query_id": "s2-t0-0", "answer": json.dumps({"suite": "libero_object", "task": 0, "seed": 0})},
    ]
    sched0 = build_episode_schedule(rows, shard_index=0, num_shards=2)
    sched1 = build_episode_schedule(rows, shard_index=1, num_shards=2)
    assert {x["query_id"] for x in sched0}.isdisjoint({x["query_id"] for x in sched1})

    # EndpointPool failover transfers inflight accurately
    pool = EndpointPool(["http://a:30000/v1", "http://b:30000/v1"])
    u1 = pool.acquire()
    u2 = pool.failover(u1)
    assert u2 != u1
    assert pool._inflight[u1] == 0 and pool._inflight[u2] == 1
    pool.release(u2)
    assert all(v == 0 for v in pool._inflight.values())

    # Binary execute response bounds checking
    payload = pack_execute_response({"ok": True}, [b"12345"])
    meta, imgs = unpack_execute_response(payload)
    assert meta == {"ok": True} and imgs == [b"12345"]
    with pytest.raises(ValueError):
        unpack_execute_response(payload[:-2])

    # SimPoolRouter skips unhealthy backends
    router = SimPoolRouter(environment="libero", backends=["http://a:1", "http://b:2"])
    router._unhealthy = {"http://a:1"}
    router._reserved["http://b:2"] = 5
    assert router._pick_backend() == "http://b:2"
