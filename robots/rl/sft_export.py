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

"""Export collected Harness v2 ``trajectory.json`` episodes to verified SFT datasets.

Key properties ensuring strict SFT <-> RL distribution alignment for Qwen3.5-4B:
- **Compaction chain splitting** (``--split-compaction-chains``, default on):
  Each episode that triggered context compaction is split at its recorded
  compaction boundaries (``schema_version >= 3`` ``chains``) into multiple
  independent training samples — mirroring RLinf's ``AppendOnlyHistory.chain_id``.
  Every carried-over prefix assistant/tool_call turn in chain ``c >= 1`` is
  marked ``"loss": false`` so earlier turns are never double-supervised and
  stripped-thinking prefix turns never teach empty ``<think>\\n\\n</think>``.
- **Cross-teacher (DeepSeek -> Qwen3.5) strict alignment**:
  Masks validation-retry turns (``--mask-retry-turns``, default on) and empty-
  thinking turns (``--mask-empty-thinking``, default on), drops truncated final
  turns (``--drop-truncated-final-turn``, default on) and infra errors
  (``--drop-infra-errors``, default on), strips explicit top-level ``null`` tool
  parameters (avoiding Qwen3.5 Jinja ``None`` vs ``ms-swift`` ``null`` drift),
  blocks special/control-token contamination (including literal ``<image>`` tags
  inside text that would hijack ``ms-swift`` image placeholders), normalizes
  legacy ``tool_call_id`` references to deterministic ``call_{turn}_{idx}``,
  trims trailing masked turns, and drops 0-supervised chains.
- **Exact end-to-end verification** (``--verify-ms-swift --verify-tokenizer``):
  Checks both token rendering parity against ``Qwen3.5-4B/chat_template.jinja``
  AND exact ``labels`` supervision masks (including ``<|im_end|>`` boundaries)
  AND exact multimodal sequence lengths (text + ``smart_resize`` visual tokens).
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal

from PIL import Image

from robots.rl.conversation import (
    IMAGE_PLACEHOLDER_PREFIX,
    count_image_tokens_from_size,
)

ExportFormat = Literal["ms_swift", "qwen_hf", "openai_messages"]
HistoryMode = Literal["chains", "full", "compacted", "auto"]

# Tokens that must NEVER appear verbatim inside free-form assistant text/system
# prompts because they corrupt Qwen3.5 turn framing or ms-swift tag splitting.
_SWIFT_MEDIA_TAGS = (
    "<image>",
    "<video>",
    "<audio>",
    "<bbox>",
    "<ref-object>",
    "<cot-process>",
    "<start-image>",
)
_QWEN_CONTROL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)
_FORBIDDEN_IN_ASSISTANT_CONTENT = (
    *_QWEN_CONTROL_TOKENS,
    *_SWIFT_MEDIA_TAGS,
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
)
_FORBIDDEN_IN_REASONING = (
    *_QWEN_CONTROL_TOKENS,
    *_SWIFT_MEDIA_TAGS,
    "</think>",
)
_INFRA_TERMINATION_REASONS = frozenset(
    {"infrastructure_error", "context_limit", "context_length_exceeded", "content_filter"}
)
_TOOL_RESULT_CALL_ID_RE = re.compile(r'tool_call_id="([^"]+)"')


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any] | None:
    if isinstance(raw_args, dict):
        parsed = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return parsed


def _strip_top_level_nulls(args: dict[str, Any]) -> dict[str, Any]:
    """Drop explicit top-level ``None`` params to eliminate Jinja ``None`` vs JSON ``null`` drift."""
    return {k: v for k, v in args.items() if v is not None}


def _check_text_contamination(
    text: str | None, forbidden: Sequence[str]
) -> str | None:
    if not text:
        return None
    for marker in forbidden:
        if marker in text:
            return marker
    return None


def _is_validation_retry_message(msg: dict[str, Any]) -> bool:
    return (
        msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and msg["content"].startswith("Validation feedback:\n")
    )


def _normalize_legacy_call_ids(
    raw_messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map teacher-specific tool_call IDs to deterministic ``call_{turn}_{idx}``."""
    id_map: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    turn_idx = 0
    for msg in raw_messages:
        role = msg.get("role")
        if role == "assistant":
            new_msg = dict(msg)
            tcs = msg.get("tool_calls") or []
            if tcs:
                new_tcs = []
                for idx, tc in enumerate(tcs):
                    old_id = str(tc.get("id") or "")
                    new_id = f"call_{turn_idx}_{idx}"
                    if old_id:
                        id_map[old_id] = new_id
                    new_tcs.append({**tc, "id": new_id})
                new_msg["tool_calls"] = new_tcs
            turn_idx += 1
            out.append(new_msg)
        elif role == "tool":
            new_msg = dict(msg)
            old_id = str(msg.get("tool_call_id") or "")
            if old_id in id_map:
                new_msg["tool_call_id"] = id_map[old_id]
            out.append(new_msg)
        elif role == "user":
            content = msg.get("content")
            if isinstance(content, list) and id_map:
                new_parts = []
                for part in content:
                    if part.get("type") == "text":
                        txt = str(part.get("text", ""))
                        if "tool_call_id=" in txt:
                            txt = _TOOL_RESULT_CALL_ID_RE.sub(
                                lambda m: f'tool_call_id="{id_map.get(m.group(1), m.group(1))}"',
                                txt,
                            )
                        new_parts.append({**part, "text": txt})
                    else:
                        new_parts.append(part)
                out.append({**msg, "content": new_parts})
            else:
                out.append(dict(msg))
        else:
            out.append(dict(msg))
    return out


def _resolve_image_path(
    fid: str,
    *,
    image_map: dict[str, str],
    traj_dir: Path,
    image_root: Path | None,
) -> tuple[str, Path]:
    rel = image_map.get(fid) or f"images/{fid}.png"
    abs_path = (traj_dir / rel).resolve()
    if image_root is not None:
        try:
            display = str(abs_path.relative_to(image_root.resolve()))
        except ValueError:
            display = str(abs_path)
    else:
        display = str(abs_path)
    return display, abs_path


def _extract_placeholder_fid(url: str) -> str | None:
    if isinstance(url, str) and url.startswith(IMAGE_PLACEHOLDER_PREFIX):
        return url[len(IMAGE_PLACEHOLDER_PREFIX) :]
    return None


def _format_ms_swift_assistant_content(
    content: str | None, reasoning_content: str | None
) -> str:
    """Wrap assistant reasoning in ``<think>...</think>`` matching Qwen3.5 Jinja."""
    reasoning = (reasoning_content or "").strip()
    text = (content or "").strip()
    think_block = f"<think>\n{reasoning}\n</think>" if reasoning else "<think>\n\n</think>"
    if text:
        return f"{think_block}\n\n{text}"
    return f"{think_block}\n\n"


def _image_tokens_from_file(path: Path, cache: dict[Path, int]) -> int:
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        with Image.open(path) as img:
            w, h = img.size
        toks = count_image_tokens_from_size(w, h)
    except Exception:
        toks = 256
    cache[path] = toks
    return toks


def _extract_chains(
    trajectory: dict[str, Any],
    *,
    history_mode: HistoryMode,
    split_compaction_chains: bool,
    require_chain_snapshots: bool = True,
) -> tuple[list[dict[str, Any]], str | None]:
    """Select chain specs ``[{"chain_id", "prefix_wire_count", "messages", ...}]``."""
    full_msgs = trajectory.get("full_messages") or []
    compacted_msgs = trajectory.get("compacted_messages") or full_msgs
    compaction_events = trajectory.get("compaction_events") or []
    saved_chains = trajectory.get("chains")

    if split_compaction_chains and history_mode == "chains":
        if isinstance(saved_chains, list) and saved_chains:
            return saved_chains, None
        # Backward compatibility for v2 trajectories: uncompacted episodes have 1 exact chain.
        if not compaction_events:
            return (
                [
                    {
                        "chain_id": 0,
                        "compaction_event_index": None,
                        "prefix_wire_count": 2 if len(full_msgs) >= 2 else len(full_msgs),
                        "messages": list(full_msgs),
                        "compaction_triggered_after": False,
                    }
                ],
                None,
            )
        if require_chain_snapshots:
            return [], "v2_compacted_missing_chains"
        return (
            [
                {
                    "chain_id": 0,
                    "compaction_event_index": None,
                    "prefix_wire_count": 2 if len(compacted_msgs) >= 2 else len(compacted_msgs),
                    "messages": list(compacted_msgs),
                    "compaction_triggered_after": False,
                }
            ],
            None,
        )

    if history_mode == "compacted":
        selected = compacted_msgs
    else:
        selected = full_msgs
    return (
        [
            {
                "chain_id": 0,
                "compaction_event_index": None,
                "prefix_wire_count": 2 if len(selected) >= 2 else len(selected),
                "messages": list(selected),
                "compaction_triggered_after": False,
            }
        ],
        None,
    )


def _convert_single_chain(
    *,
    chain: dict[str, Any],
    chain_idx: int,
    num_chains: int,
    is_last_chain: bool,
    trajectory: dict[str, Any],
    traj_dir: Path,
    export_format: ExportFormat,
    trim_trailing_non_assistant: bool,
    mask_retry_turns: bool,
    mask_empty_thinking: bool,
    drop_truncated_final_turn: bool,
    strip_null_args: bool,
    normalize_call_ids: bool,
    image_root: Path | None,
    check_images_exist: bool,
    include_metadata: bool,
) -> tuple[dict[str, Any] | None, str | None, list[Path]]:
    """Convert one chain snapshot into an SFT sample (with exact prefix/retry/empty masking)."""
    raw_messages = list(chain.get("messages") or [])
    if not raw_messages:
        return None, "empty_chain", []

    if normalize_call_ids:
        raw_messages = _normalize_legacy_call_ids(raw_messages)

    prefix_wire_count = int(chain.get("prefix_wire_count") or 0)
    # In Chain 0 (no prior compaction), prefix is just opening [system, user]
    # (no assistant messages at index < prefix_wire_count).
    outcome = trajectory.get("outcome") or {}
    if (
        drop_truncated_final_turn
        and is_last_chain
        and outcome.get("termination_reason") == "response_truncated"
    ):
        # Drop the incomplete final assistant turn (and any trailing non-assistant msgs).
        end = len(raw_messages)
        while end > 0 and raw_messages[end - 1].get("role") != "assistant":
            end -= 1
        if end > 0 and raw_messages[end - 1].get("role") == "assistant":
            raw_messages = raw_messages[: end - 1]

    # Compute validation-retry mask BEFORE trimming trailing non-assistant messages
    # so an assistant turn ending right before Validation feedback is still masked!
    retry_assistant_indices: set[int] = set()
    if mask_retry_turns:
        for idx, msg in enumerate(raw_messages):
            if (
                msg.get("role") == "assistant"
                and idx + 1 < len(raw_messages)
                and _is_validation_retry_message(raw_messages[idx + 1])
            ):
                retry_assistant_indices.add(idx)

    # Determine per-assistant loss flag on raw_messages BEFORE trimming trailing masked turns.
    assistant_masked: dict[int, bool] = {}
    prefix_masked_turns = 0
    retry_masked_turns = 0
    empty_think_masked_turns = 0
    for idx, msg in enumerate(raw_messages):
        if msg.get("role") != "assistant":
            continue
        is_prefix = idx < prefix_wire_count
        is_retry = idx in retry_assistant_indices
        has_thinking = bool((msg.get("reasoning_content") or "").strip())
        is_empty_think = (not is_prefix) and mask_empty_thinking and (not has_thinking)
        masked = is_prefix or is_retry or is_empty_think
        assistant_masked[idx] = masked
        if is_prefix:
            prefix_masked_turns += 1
        elif is_retry:
            retry_masked_turns += 1
        elif is_empty_think:
            empty_think_masked_turns += 1

    # Trim trailing non-assistant messages AND trailing masked assistant turns so:
    # 1) ms-swift never appends a dangling generation prompt, and
    # 2) every sample ends on a real supervised assistant turn (no trailing loss=False).
    if trim_trailing_non_assistant:
        end = len(raw_messages)
        while end > 0:
            role = raw_messages[end - 1].get("role")
            if role != "assistant":
                end -= 1
            elif assistant_masked.get(end - 1, False):
                end -= 1
            else:
                break
        raw_messages = raw_messages[:end]

    supervised_turns = sum(
        1
        for idx, msg in enumerate(raw_messages)
        if msg.get("role") == "assistant" and not assistant_masked.get(idx, False)
    )
    if supervised_turns == 0:
        return None, "zero_supervised_turns", []

    tools = trajectory.get("tools") or []
    image_map: dict[str, str] = trajectory.get("images") or {}
    images_out: list[str] = []
    image_abs_paths: list[Path] = []

    if export_format == "ms_swift":
        ms_messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(raw_messages):
            role = msg.get("role")
            if role == "system":
                sys_text = str(msg.get("content") or "")
                bad = _check_text_contamination(sys_text, _SWIFT_MEDIA_TAGS)
                if bad is not None:
                    return None, f"system_media_tag_contamination:{bad}", []
                ms_messages.append({"role": "system", "content": sys_text})
            elif role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    bad = _check_text_contamination(content, _SWIFT_MEDIA_TAGS)
                    if bad is not None:
                        return None, f"user_media_tag_contamination:{bad}", []
                    ms_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "text":
                            txt = str(part.get("text", ""))
                            bad = _check_text_contamination(txt, _SWIFT_MEDIA_TAGS)
                            if bad is not None:
                                return None, f"user_media_tag_contamination:{bad}", []
                            parts.append(txt)
                        elif ptype == "image_url":
                            fid = _extract_placeholder_fid(
                                (part.get("image_url") or {}).get("url", "")
                            )
                            if fid is None:
                                return None, "invalid_image_placeholder", []
                            img_ref, abs_path = _resolve_image_path(
                                fid,
                                image_map=image_map,
                                traj_dir=traj_dir,
                                image_root=image_root,
                            )
                            if check_images_exist and not abs_path.is_file():
                                return None, f"missing_image:{fid}", []
                            images_out.append(img_ref)
                            image_abs_paths.append(abs_path)
                            parts.append("<image>")
                        else:
                            return None, f"unsupported_content_part:{ptype}", []
                    ms_messages.append({"role": "user", "content": "".join(parts)})
                else:
                    return None, "invalid_user_content", []
            elif role == "assistant":
                raw_content = msg.get("content")
                raw_reasoning = msg.get("reasoning_content")
                bad_c = _check_text_contamination(
                    raw_content, _FORBIDDEN_IN_ASSISTANT_CONTENT
                )
                if bad_c is not None:
                    return None, f"content_special_token_contamination:{bad_c}", []
                bad_r = _check_text_contamination(
                    raw_reasoning, _FORBIDDEN_IN_REASONING
                )
                if bad_r is not None:
                    return None, f"reasoning_special_token_contamination:{bad_r}", []

                is_masked = assistant_masked.get(idx, False)
                asst_text = _format_ms_swift_assistant_content(
                    raw_content, raw_reasoning
                )
                asst_entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": asst_text,
                }
                if is_masked:
                    asst_entry["loss"] = False
                ms_messages.append(asst_entry)

                for tc in msg.get("tool_calls") or ():
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    parsed_args = _parse_tool_arguments(fn.get("arguments"))
                    if not name or parsed_args is None:
                        return None, "malformed_tool_call_arguments", []
                    if strip_null_args:
                        parsed_args = _strip_top_level_nulls(parsed_args)
                    tc_payload = json.dumps(
                        {"name": name, "arguments": parsed_args},
                        ensure_ascii=False,
                    )
                    bad_tc = _check_text_contamination(tc_payload, _SWIFT_MEDIA_TAGS)
                    if bad_tc is not None:
                        return None, f"tool_call_media_tag_contamination:{bad_tc}", []
                    tc_entry: dict[str, Any] = {
                        "role": "tool_call",
                        "content": tc_payload,
                    }
                    if is_masked:
                        tc_entry["loss"] = False
                    ms_messages.append(tc_entry)
            elif role == "tool":
                tool_txt = str(msg.get("content") or "")
                bad_t = _check_text_contamination(tool_txt, _SWIFT_MEDIA_TAGS)
                if bad_t is not None:
                    return None, f"tool_media_tag_contamination:{bad_t}", []
                ms_messages.append({"role": "tool_response", "content": tool_txt})
            else:
                return None, f"unsupported_role:{role}", []

        tag_count = sum(
            m["content"].count("<image>")
            for m in ms_messages
            if isinstance(m.get("content"), str)
        )
        if tag_count != len(images_out):
            return None, f"image_tag_mismatch:{tag_count}!={len(images_out)}", []

        sample: dict[str, Any] = {
            "tools": json.dumps(tools, ensure_ascii=False),
            "messages": ms_messages,
            "images": images_out,
        }
    else:
        converted_msgs: list[dict[str, Any]] = []
        turn_counter = 0
        for idx, msg in enumerate(raw_messages):
            role = msg.get("role")
            if role == "system":
                converted_msgs.append(
                    {"role": "system", "content": str(msg.get("content") or "")}
                )
            elif role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    converted_msgs.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    new_parts: list[dict[str, Any]] = []
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "text":
                            new_parts.append(
                                {"type": "text", "text": str(part.get("text", ""))}
                            )
                        elif ptype == "image_url":
                            fid = _extract_placeholder_fid(
                                (part.get("image_url") or {}).get("url", "")
                            )
                            if fid is None:
                                return None, "invalid_image_placeholder", []
                            img_ref, abs_path = _resolve_image_path(
                                fid,
                                image_map=image_map,
                                traj_dir=traj_dir,
                                image_root=image_root,
                            )
                            if check_images_exist and not abs_path.is_file():
                                return None, f"missing_image:{fid}", []
                            images_out.append(img_ref)
                            image_abs_paths.append(abs_path)
                            if export_format == "qwen_hf":
                                new_parts.append({"type": "image", "image": img_ref})
                            else:
                                new_parts.append(
                                    {"type": "image_url", "image_url": {"url": img_ref}}
                                )
                        else:
                            return None, f"unsupported_content_part:{ptype}", []
                    converted_msgs.append({"role": "user", "content": new_parts})
                else:
                    return None, "invalid_user_content", []
            elif role == "assistant":
                raw_content = msg.get("content")
                raw_reasoning = msg.get("reasoning_content")
                bad_c = _check_text_contamination(
                    raw_content, _FORBIDDEN_IN_ASSISTANT_CONTENT
                )
                if bad_c is not None:
                    return None, f"content_special_token_contamination:{bad_c}", []
                bad_r = _check_text_contamination(
                    raw_reasoning, _FORBIDDEN_IN_REASONING
                )
                if bad_r is not None:
                    return None, f"reasoning_special_token_contamination:{bad_r}", []
                out_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": (raw_content or "").strip(),
                }
                clean_r = (raw_reasoning or "").strip()
                if clean_r:
                    out_msg["reasoning_content"] = clean_r
                tcs = msg.get("tool_calls") or []
                if tcs:
                    formatted_tcs: list[dict[str, Any]] = []
                    for tc_idx, tc in enumerate(tcs):
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        parsed_args = _parse_tool_arguments(fn.get("arguments"))
                        if not name or parsed_args is None:
                            return None, "malformed_tool_call_arguments", []
                        if strip_null_args:
                            parsed_args = _strip_top_level_nulls(parsed_args)
                        fn_args: Any = (
                            parsed_args
                            if export_format == "qwen_hf"
                            else json.dumps(parsed_args, ensure_ascii=False)
                        )
                        formatted_tcs.append(
                            {
                                "id": tc.get("id") or f"call_{turn_counter}_{tc_idx}",
                                "type": "function",
                                "function": {"name": name, "arguments": fn_args},
                            }
                        )
                    out_msg["tool_calls"] = formatted_tcs
                if assistant_masked.get(idx, False):
                    out_msg["loss"] = False
                turn_counter += 1
                converted_msgs.append(out_msg)
            elif role == "tool":
                out_tool: dict[str, Any] = {
                    "role": "tool",
                    "content": str(msg.get("content") or ""),
                }
                if msg.get("tool_call_id"):
                    out_tool["tool_call_id"] = str(msg["tool_call_id"])
                converted_msgs.append(out_tool)
            else:
                return None, f"unsupported_role:{role}", []

        sample = {
            "tools": tools,
            "messages": converted_msgs,
            "images": images_out,
        }

    if include_metadata:
        sample["metadata"] = {
            "episode_key": trajectory.get("episode_key"),
            "query_id": trajectory.get("query_id"),
            "environment": trajectory.get("environment"),
            "teacher_model": (trajectory.get("model") or {}).get("name"),
            "chain_id": int(chain.get("chain_id", chain_idx)),
            "num_chains": num_chains,
            "supervised_turns": supervised_turns,
            "prefix_masked_turns": prefix_masked_turns,
            "retry_masked_turns": retry_masked_turns,
            "empty_think_masked_turns": empty_think_masked_turns,
            "success": bool(outcome.get("success", False)),
            "turns": int(outcome.get("turns", 0)),
            "termination_reason": outcome.get("termination_reason"),
            "error_kind": outcome.get("error_kind"),
        }

    return sample, None, image_abs_paths


def convert_trajectory_to_samples(
    trajectory: dict[str, Any],
    *,
    traj_dir: Path,
    export_format: ExportFormat = "ms_swift",
    history_mode: HistoryMode = "chains",
    split_compaction_chains: bool = True,
    require_chain_snapshots: bool = True,
    trim_trailing_non_assistant: bool = True,
    mask_retry_turns: bool = True,
    mask_empty_thinking: bool = True,
    drop_truncated_final_turn: bool = True,
    strip_null_args: bool = True,
    normalize_call_ids: bool = True,
    image_root: Path | None = None,
    check_images_exist: bool = True,
    include_metadata: bool = False,
) -> tuple[list[tuple[dict[str, Any], list[Path]]], Counter[str]]:
    """Convert one ``trajectory.json`` into 1+ chain-split SFT samples."""
    drops: Counter[str] = Counter()
    chains, extract_err = _extract_chains(
        trajectory,
        history_mode=history_mode,
        split_compaction_chains=split_compaction_chains,
        require_chain_snapshots=require_chain_snapshots,
    )
    if extract_err is not None:
        drops[extract_err] += 1
        return [], drops

    results: list[tuple[dict[str, Any], list[Path]]] = []
    num_chains = len(chains)
    for chain_idx, chain in enumerate(chains):
        sample, drop_reason, img_paths = _convert_single_chain(
            chain=chain,
            chain_idx=chain_idx,
            num_chains=num_chains,
            is_last_chain=(chain_idx == num_chains - 1),
            trajectory=trajectory,
            traj_dir=traj_dir,
            export_format=export_format,
            trim_trailing_non_assistant=trim_trailing_non_assistant,
            mask_retry_turns=mask_retry_turns,
            mask_empty_thinking=mask_empty_thinking,
            drop_truncated_final_turn=drop_truncated_final_turn,
            strip_null_args=strip_null_args,
            normalize_call_ids=normalize_call_ids,
            image_root=image_root,
            check_images_exist=check_images_exist,
            include_metadata=include_metadata,
        )
        if sample is None:
            drops[drop_reason or "unknown"] += 1
            continue
        results.append((sample, img_paths))
    return results, drops


def convert_trajectory(
    trajectory: dict[str, Any],
    *,
    traj_dir: Path,
    export_format: ExportFormat = "ms_swift",
    history_mode: HistoryMode = "full",
    max_seq_tokens: int | None = None,
    trim_trailing_non_assistant: bool = True,
    mask_retry_turns: bool = True,
    image_root: Path | None = None,
    check_images_exist: bool = True,
    include_metadata: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Backwards-compatible single-sample converter wrapper."""
    del max_seq_tokens
    rows, drops = convert_trajectory_to_samples(
        trajectory,
        traj_dir=traj_dir,
        export_format=export_format,
        history_mode=history_mode,
        split_compaction_chains=(history_mode == "chains"),
        require_chain_snapshots=False,
        trim_trailing_non_assistant=trim_trailing_non_assistant,
        mask_retry_turns=mask_retry_turns,
        image_root=image_root,
        check_images_exist=check_images_exist,
        include_metadata=include_metadata,
    )
    if not rows:
        reason = next(iter(drops), "no_samples")
        return None, reason
    return rows[0][0], None


def render_ms_swift_row_with_qwen3_5_template(
    sample: dict[str, Any],
    *,
    ms_swift_repo: Path | None = None,
) -> tuple[str, list[tuple[str, float]]]:
    """Render an ``ms_swift`` row via ``ms-swift``'s actual ``Qwen3_5Template`` logic."""
    if ms_swift_repo is not None and str(ms_swift_repo) not in sys.path:
        sys.path.insert(0, str(ms_swift_repo))

    from swift.agent_template import agent_template_map
    from swift.loss_scale import get_loss_scale
    from swift.template.base import Template
    from swift.template.template_inputs import StdTemplateInputs
    from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta

    inputs = StdTemplateInputs.from_dict(sample)
    agent = agent_template_map["qwen3_5"]()
    meta = QwenTemplateMeta(
        "qwen3_5",
        template_cls=Qwen3_5Template,
        default_system=None,
        thinking_prefix="<think>\n",
        non_thinking_prefix="<think>\n\n</think>\n\n",
        default_enable_thinking=True,
        agent_template="qwen3_5",
        is_thinking=True,
    )
    agent.template_meta = meta

    class _LightweightQwen3_5(Qwen3_5Template):
        def __init__(self) -> None:
            self.template_meta = meta
            self.template_backend = "swift"
            self.use_chat_template = True
            self.mode = "train"
            self.task_type = "causal_lm"
            self.add_non_thinking_prefix = True
            self.preserve_thinking = True
            self.enable_thinking = True
            self.response_prefix = None
            self.chat_template_kwargs = {}
            self._agent_template = "qwen3_5"
            self._agent_template_cache = {"qwen3_5": agent}
            self._loss_scale = "default"
            self._loss_scale_cache = {"default": get_loss_scale("default")}
            self.is_binary_loss_scale = True

    tmpl = _LightweightQwen3_5()
    tmpl._preprocess_tools(inputs)
    tmpl._swift_prepare_inputs(inputs)
    res_ctx, loss_scales, _ = Template._swift_encode(tmpl, inputs)

    rendered_parts: list[str] = []
    segments: list[tuple[str, float]] = []
    for ctx, scale in zip(res_ctx, loss_scales, strict=True):
        text = str(ctx).replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        )
        rendered_parts.append(text)
        segments.append((text, float(scale)))
    return "".join(rendered_parts), segments


def discover_trajectory_files(input_dirs: Sequence[Path]) -> list[Path]:
    """Find all ``trajectory.json`` files under one or more collector output dirs."""
    seen: dict[Path, Path] = {}
    for root in input_dirs:
        if root.is_file() and root.name == "trajectory.json":
            seen.setdefault(root.resolve(), root)
        elif root.is_dir():
            if (root / "trajectory.json").is_file():
                p = root / "trajectory.json"
                seen.setdefault(p.resolve(), p)
            else:
                for p in sorted(root.rglob("trajectory.json")):
                    seen.setdefault(p.resolve(), p)
    return sorted(seen.values())


def _strip_thinking_before_last_query(text: str) -> str:
    """Normalize ms-swift text for comparison when a Validation feedback turn exists.

    Qwen3.5-4B's static HF Jinja drops ``<think>...</think>`` prior to the last
    non-``<tool_response>`` user turn (i.e. ``Validation feedback:``), whereas
    RLinf's append-only rollout and ms-swift SFT (``preserve_thinking=True``)
    preserve all historical ``<think>`` blocks. Stripping those specific
    pre-feedback ``<think>`` blocks during verification still asserts 100%
    byte-for-byte equality on every tool call, tag, image marker, and turn after
    the feedback instead of skipping verification!
    """
    marker = "<|im_start|>user\nValidation feedback:\n"
    idx = text.rfind(marker)
    if idx < 0:
        return text
    prefix = text[:idx]
    suffix = text[idx:]
    cleaned_prefix = re.sub(
        r"(<\|im_start\|>assistant\n)<think>\n.*?\n</think>\n\n",
        r"\1",
        prefix,
        flags=re.DOTALL,
    )
    return cleaned_prefix + suffix


def verify_exported_samples(
    samples: Sequence[tuple[dict[str, Any], dict[str, Any], list[Path]]],
    *,
    tokenizer: Any | None = None,
    tokenizer_path: Path | None = None,
    verify_ms_swift: bool = False,
    ms_swift_repo: Path | None = None,
    verify_limit: int | None = None,
    image_tok_cache: dict[Path, int] | None = None,
) -> dict[str, Any]:
    """Verify exported samples against Qwen3.5 Jinja tokenizer and ``ms-swift`` labels."""
    subset = samples[:verify_limit] if verify_limit is not None else samples
    if tokenizer is None and tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )

    cache = image_tok_cache if image_tok_cache is not None else {}
    total_token_lengths: list[int] = []
    exact_matches = 0
    retry_normalized_matches = 0
    masked_segments_checked = 0
    checked = 0

    for exported_row, qwen_hf_row, img_paths in subset:
        checked += 1
        expected_images = len(exported_row.get("images") or ())
        hf_text: str | None = None

        if tokenizer is not None:
            hf_text = tokenizer.apply_chat_template(
                qwen_hf_row["messages"],
                tools=qwen_hf_row["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            pad_count = hf_text.count("<|vision_start|><|image_pad|><|vision_end|>")
            if pad_count != expected_images:
                raise AssertionError(
                    f"HF chat_template image pad mismatch: {pad_count} != {expected_images}"
                )
            if "<tool_call>" in hf_text and "<function=" not in hf_text:
                raise AssertionError("Rendered <tool_call> missing <function=...> tag")
            text_toks = len(tokenizer.encode(hf_text, add_special_tokens=False))
            vision_toks = sum(_image_tokens_from_file(p, cache) for p in img_paths)
            total_token_lengths.append(text_toks + vision_toks - expected_images)

        if verify_ms_swift:
            ms_text, segments = render_ms_swift_row_with_qwen3_5_template(
                exported_row, ms_swift_repo=ms_swift_repo
            )
            ms_pad = ms_text.count("<|vision_start|><|image_pad|><|vision_end|>")
            if ms_pad != expected_images:
                raise AssertionError(
                    f"ms-swift image pad mismatch: {ms_pad} != {expected_images}"
                )
            active_segments = [text for text, scale in segments if scale > 0.0]
            if not active_segments:
                raise AssertionError("ms-swift produced zero supervised assistant tokens")
            if any("<|image_pad|>" in text for text in active_segments):
                raise AssertionError("ms-swift leaked image tokens into supervised labels")

            # Verify masked messages (loss=False) truly received scale == 0.0
            expected_masked_msgs = sum(
                1
                for m in exported_row.get("messages") or ()
                if m.get("role") == "assistant" and m.get("loss") is False
            )
            if expected_masked_msgs > 0:
                masked_segments_checked += expected_masked_msgs
                # No active segment should contain an empty stripped <think>\n\n</think>
                if any("<think>\n\n</think>" in seg for seg in active_segments):
                    raise AssertionError(
                        "Empty <think> block leaked into supervised (loss>0) segments"
                    )

            if hf_text is not None:
                has_validation_retry = any(
                    _is_validation_retry_message(m) for m in qwen_hf_row["messages"]
                )
                if ms_text == hf_text:
                    exact_matches += 1
                elif has_validation_retry and (
                    _strip_thinking_before_last_query(ms_text) == hf_text
                ):
                    retry_normalized_matches += 1
                else:
                    cmp_ms = (
                        _strip_thinking_before_last_query(ms_text)
                        if has_validation_retry
                        else ms_text
                    )
                    diff_idx = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(cmp_ms, hf_text))
                            if a != b
                        ),
                        min(len(cmp_ms), len(hf_text)),
                    )
                    raise AssertionError(
                        f"ms-swift vs Qwen3.5 HF Jinja mismatch at char {diff_idx}:\n"
                        f"  ms-swift: {cmp_ms[max(0, diff_idx - 80):diff_idx + 120]!r}\n"
                        f"  hf-jinja: {hf_text[max(0, diff_idx - 80):diff_idx + 120]!r}"
                    )

    token_stats: dict[str, int] = {}
    if total_token_lengths:
        ordered = sorted(total_token_lengths)
        token_stats = {
            "total_tokens_min": ordered[0],
            "total_tokens_p50": ordered[len(ordered) // 2],
            "total_tokens_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "total_tokens_max": ordered[-1],
        }
    return {
        "checked": checked,
        "exact_ms_swift_hf_matches": exact_matches,
        "retry_normalized_ms_swift_hf_matches": retry_normalized_matches,
        "masked_prefix_or_retry_turns_verified": masked_segments_checked,
        **token_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more rollout output directories (or trajectory.json files)",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("ms_swift", "qwen_hf", "openai_messages"),
        default="ms_swift",
        help="Target SFT schema (default: ms_swift for Qwen3.5-4B)",
    )
    parser.add_argument(
        "--history-mode",
        choices=("chains", "full", "compacted", "auto"),
        default="chains",
        help="Export compaction-split chains (default, RL-aligned), full, or compacted",
    )
    parser.add_argument(
        "--split-compaction-chains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split compacted episodes into multiple chain samples with loss=False on prefix (default: true)",
    )
    parser.add_argument(
        "--require-chain-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop v2 compacted trajectories lacking per-chain snapshots (default: true)",
    )
    parser.add_argument(
        "--max-seq-tokens",
        type=int,
        default=65536,
        help="Drop exported chains exceeding this exact multimodal token count (when --verify-tokenizer is set)",
    )
    parser.add_argument("--only-success", action="store_true")
    parser.add_argument("--drop-truncated", action="store_true")
    parser.add_argument(
        "--drop-truncated-final-turn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the incomplete final assistant turn on response_truncated episodes (default: true)",
    )
    parser.add_argument(
        "--drop-infra-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop episodes terminated by infrastructure/context/content_filter errors (default: true)",
    )
    parser.add_argument("--drop-malformed-tool-calls", action="store_true")
    parser.add_argument(
        "--trim-trailing-non-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim trailing non-assistant and trailing loss=False assistant turns (default: true)",
    )
    parser.add_argument(
        "--mask-retry-turns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set loss=false on invalid assistant/tool_call turns followed by Validation feedback (default: true)",
    )
    parser.add_argument(
        "--mask-empty-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set loss=false on non-prefix turns lacking reasoning_content (default: true)",
    )
    parser.add_argument(
        "--strip-null-args",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip explicit top-level null tool arguments to prevent Jinja None vs null drift (default: true)",
    )
    parser.add_argument(
        "--normalize-call-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize tool_call_ids to deterministic call_{turn}_{idx} (default: true)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory to store relative image paths (for ROOT_IMAGE_DIR)",
    )
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--verify-tokenizer",
        type=Path,
        default=None,
        help="Path to Qwen3.5 tokenizer/model dir for exact length filtering and HF Jinja check",
    )
    parser.add_argument(
        "--verify-ms-swift",
        action="store_true",
        help="Verify exported ms_swift rows with ms-swift Qwen3_5Template + loss mask checks",
    )
    parser.add_argument(
        "--ms-swift-repo",
        type=Path,
        default=None,
        help="Optional path to cloned ms-swift repo if not installed in venv",
    )
    parser.add_argument("--verify-limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer = None
    if args.verify_tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.verify_tokenizer), trust_remote_code=True
        )

    traj_files = discover_trajectory_files(args.input_dirs)
    drop_counts: Counter[str] = Counter()
    env_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    image_tok_cache: dict[Path, int] = {}
    episodes_exported = 0
    episodes_compacted_split = 0
    success_samples = 0

    exported_rows: list[dict[str, Any]] = []
    verify_triples: list[tuple[dict[str, Any], dict[str, Any], list[Path]]] = []
    verify_cap = args.verify_limit

    for traj_path in traj_files:
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception:
            drop_counts["corrupt_json"] += 1
            continue
        outcome = traj.get("outcome") or {}
        term_reason = outcome.get("termination_reason")
        if args.only_success and not outcome.get("success"):
            drop_counts["not_success"] += 1
            continue
        if args.drop_infra_errors and (
            term_reason in _INFRA_TERMINATION_REASONS
            or outcome.get("error_kind") == "tool_transport"
        ):
            drop_counts[f"infra_or_abort:{term_reason or outcome.get('error_kind')}"] += 1
            continue
        if args.drop_truncated and term_reason == "response_truncated":
            drop_counts["response_truncated"] += 1
            continue
        if args.drop_malformed_tool_calls and outcome.get("error_kind") in {
            "tool_name",
            "tool_arguments",
        }:
            drop_counts["malformed_tool_call"] += 1
            continue

        chain_samples, chain_drops = convert_trajectory_to_samples(
            traj,
            traj_dir=traj_path.parent,
            export_format=args.format,
            history_mode=args.history_mode,
            split_compaction_chains=args.split_compaction_chains,
            require_chain_snapshots=args.require_chain_snapshots,
            trim_trailing_non_assistant=args.trim_trailing_non_assistant,
            mask_retry_turns=args.mask_retry_turns,
            mask_empty_thinking=args.mask_empty_thinking,
            drop_truncated_final_turn=args.drop_truncated_final_turn,
            strip_null_args=args.strip_null_args,
            normalize_call_ids=args.normalize_call_ids,
            image_root=args.image_root,
            check_images_exist=True,
            include_metadata=args.include_metadata,
        )
        drop_counts.update(chain_drops)
        if not chain_samples:
            continue

        need_hf = tokenizer is not None or args.verify_ms_swift
        hf_samples: list[tuple[dict[str, Any], list[Path]]] = []
        if need_hf:
            hf_samples, hf_drops = convert_trajectory_to_samples(
                traj,
                traj_dir=traj_path.parent,
                export_format="qwen_hf",
                history_mode=args.history_mode,
                split_compaction_chains=args.split_compaction_chains,
                require_chain_snapshots=args.require_chain_snapshots,
                trim_trailing_non_assistant=args.trim_trailing_non_assistant,
                mask_retry_turns=args.mask_retry_turns,
                mask_empty_thinking=args.mask_empty_thinking,
                drop_truncated_final_turn=args.drop_truncated_final_turn,
                strip_null_args=args.strip_null_args,
                normalize_call_ids=args.normalize_call_ids,
                image_root=args.image_root,
                check_images_exist=True,
                include_metadata=False,
            )
            if len(hf_samples) != len(chain_samples):
                drop_counts["hf_chain_count_mismatch"] += 1
                continue

        kept_from_ep = 0
        for idx, (sample, img_paths) in enumerate(chain_samples):
            if tokenizer is not None and args.max_seq_tokens is not None:
                hf_row = hf_samples[idx][0]
                hf_text = tokenizer.apply_chat_template(
                    hf_row["messages"],
                    tools=hf_row["tools"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                text_toks = len(tokenizer.encode(hf_text, add_special_tokens=False))
                vision_toks = sum(
                    _image_tokens_from_file(p, image_tok_cache) for p in img_paths
                )
                total_toks = text_toks + vision_toks - len(img_paths)
                if total_toks > args.max_seq_tokens:
                    drop_counts[f"exceeds_max_seq_tokens:{args.max_seq_tokens}"] += 1
                    continue

            exported_rows.append(sample)
            kept_from_ep += 1
            env_counts[str(traj.get("environment") or "unknown")] += 1
            teacher_counts[str((traj.get("model") or {}).get("name") or "unknown")] += 1
            if outcome.get("success"):
                success_samples += 1
            if need_hf and (verify_cap is None or len(verify_triples) < verify_cap):
                verify_triples.append((sample, hf_samples[idx][0], img_paths))

        if kept_from_ep > 0:
            episodes_exported += 1
            if kept_from_ep > 1:
                episodes_compacted_split += 1

    verify_summary: dict[str, Any] = {}
    if verify_triples:
        verify_summary = verify_exported_samples(
            verify_triples,
            tokenizer=tokenizer,
            verify_ms_swift=args.verify_ms_swift and args.format == "ms_swift",
            ms_swift_repo=args.ms_swift_repo,
            verify_limit=args.verify_limit,
            image_tok_cache=image_tok_cache,
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8") as fh:
        for row in exported_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_out.replace(args.output_jsonl)

    stats = {
        "format": args.format,
        "history_mode": args.history_mode,
        "split_compaction_chains": args.split_compaction_chains,
        "input_trajectories": len(traj_files),
        "episodes_exported": episodes_exported,
        "episodes_split_into_multiple_chains": episodes_compacted_split,
        "exported_samples": len(exported_rows),
        "exported_success_samples": success_samples,
        "by_environment": dict(env_counts),
        "by_teacher": dict(teacher_counts),
        "dropped": dict(drop_counts),
        "verification": verify_summary,
    }
    stats_path = args.output_jsonl.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
