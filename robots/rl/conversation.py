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

"""Pydantic-AI-faithful Qwen conversation formatting, validation, and compaction.

Reproduces the exact wire format of ``QwenChatModel`` + ``WorkflowCapability`` +
``RequestAwareSlidingWindow`` without requiring pydantic-ai at runtime:

- Opening request counts as 1 logical message (``[system, opening_user]``).
- Text-only tool results serialize as a compact 1-element JSON array ``["{...}"]``.
- Image-bearing tool results emit ``role="tool"`` with ``See file <sha1[:6]>.``
  references plus a trailing ``role="user"`` message wrapped in
  ``<tool_response>\n...\n</tool_response>`` with ``<tool_result ...>`` tags
  so Qwen3.5's chat template keeps multi-step reasoning blocks intact.
- Raw PNG bytes live on ``LogicalMessage.images`` without eager Base64 encoding;
  lightweight ``rpent-image://<file_id>`` placeholders stay on ``wire`` for
  templating and prefix comparisons, and OpenAI JSON data URIs are materialized
  lazily once per message only when requested.
- Validation retries format as ``Validation feedback:\n<msg>\n\nFix the errors and try again.``.
- Sliding-window / dynamic compaction records per-chain snapshots at every
  compaction boundary so SFT distillation splits compacted episodes into
  distribution-matched training chains mirroring RLinf's ``AppendOnlyHistory``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from jsonschema import Draft202012Validator
from PIL import Image

from robots.rl.units import argument_hint, validate_relations

TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"
IMAGE_PLACEHOLDER_PREFIX = "rpent-image://"
PATCH_SIZE = 16
MERGE_SIZE = 2
MIN_IMAGE_PIXELS = 65536      # Qwen3.5 preprocessor_config.json shortest_edge
MAX_IMAGE_PIXELS = 16777216   # Qwen3.5 preprocessor_config.json longest_edge
DEFAULT_IMAGE_TOKENS = 1024


def file_id_for(png_bytes: bytes) -> str:
    """Match pydantic-ai's 6-character SHA-1 identifier for binary tool files."""
    return hashlib.sha1(png_bytes).hexdigest()[:6]


def smart_resize_dimensions(
    height: int,
    width: int,
    *,
    factor: int = PATCH_SIZE * MERGE_SIZE,
    min_pixels: int = MIN_IMAGE_PIXELS,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[int, int]:
    """Match Qwen3.5 HF ``Qwen2VLImageProcessor`` ``smart_resize`` dimensions."""
    if height <= 0 or width <= 0:
        return factor, factor
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def count_image_tokens_from_size(width: int, height: int) -> int:
    """Return exact Qwen3.5 merged visual token count for an image of ``(width, height)``."""
    block = PATCH_SIZE * MERGE_SIZE
    h_bar, w_bar = smart_resize_dimensions(height, width, factor=block)
    return (h_bar // block) * (w_bar // block)


def count_image_tokens(png_bytes: bytes) -> int:
    """Return exact Qwen3.5 vision tokens for one image (32x32 merged patch grid)."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            width, height = img.size
    except Exception:
        return DEFAULT_IMAGE_TOKENS
    return count_image_tokens_from_size(width, height)


@dataclass(frozen=True)
class LogicalMessage:
    """One Pydantic-AI-equivalent logical message unit for compaction.

    ``wire`` stores lightweight template-ready dicts without embedded Base64
    image strings; raw PNG payloads are kept directly in ``images``.
    """

    kind: str  # "opening", "assistant", "tool", "retry"
    wire: tuple[dict[str, Any], ...]
    tool_call_ids: tuple[str, ...] = ()
    images: tuple[bytes, ...] = ()
    image_tokens: int = 0


def _parse_opening_cameras(
    initial_observation_text: str, count: int, explicit: Sequence[str] = ()
) -> list[str]:
    if len(explicit) == count:
        return [str(c) for c in explicit]
    try:
        parsed = json.loads(initial_observation_text)
        imgs = parsed.get("images")
        if isinstance(imgs, list) and len(imgs) == count:
            return [str(item.get("camera", f"cam_{i}")) for i, item in enumerate(imgs)]
    except Exception:
        pass
    return [f"cam_{i}" for i in range(count)]


def opening_message(
    system_prompt: str,
    user_message: str,
    initial_observation_text: str,
    initial_images: Sequence[bytes] = (),
    initial_image_cameras: Sequence[str] = (),
) -> LogicalMessage:
    """Build the opening [system, user] logical message (with optional step=0 images)."""
    opening_text = f"{user_message}\n\n{initial_observation_text}"
    if not initial_images:
        return LogicalMessage(
            kind="opening",
            wire=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": opening_text},
            ),
        )
    raw_images = tuple(bytes(img) for img in initial_images)
    fids = [file_id_for(img) for img in raw_images]
    cameras = _parse_opening_cameras(
        initial_observation_text, len(raw_images), initial_image_cameras
    )
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": f"{opening_text}\n"}]
    total_img_tokens = 0
    for img, fid, cam in zip(raw_images, fids, cameras, strict=True):
        total_img_tokens += count_image_tokens(img)
        user_parts.extend(
            [
                {
                    "type": "text",
                    "text": f'<observation step="0" camera="{cam}" file_id="{fid}">',
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{IMAGE_PLACEHOLDER_PREFIX}{fid}"},
                },
                {"type": "text", "text": "</observation>"},
            ]
        )
    return LogicalMessage(
        kind="opening",
        wire=(
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ),
        images=raw_images,
        image_tokens=total_img_tokens,
    )


def assistant_message(
    *,
    content: str | None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> LogicalMessage:
    """Build an assistant logical message matching QwenChatModel & RLinf wire layout."""
    clean_content = (
        content.strip() if isinstance(content, str) and content.strip() else None
    )
    clean_reasoning = (
        reasoning_content.strip()
        if isinstance(reasoning_content, str) and reasoning_content.strip()
        else None
    )
    msg: dict[str, Any] = {"role": "assistant", "content": clean_content}
    if clean_reasoning is not None:
        msg["reasoning_content"] = clean_reasoning
    call_ids: list[str] = []
    if tool_calls:
        msg["tool_calls"] = tool_calls
        call_ids = [str(call.get("id") or "") for call in tool_calls]
    return LogicalMessage(kind="assistant", wire=(msg,), tool_call_ids=tuple(call_ids))


def tool_return_message(
    *,
    tool_name: str,
    tool_call_id: str,
    text: str,
    images: Sequence[bytes] = (),
) -> LogicalMessage:
    """Build a tool return logical message (with optional framed Qwen image turn)."""
    if not images:
        payload = json.dumps([text], separators=(",", ":"), ensure_ascii=False)
        return LogicalMessage(
            kind="tool",
            wire=({"role": "tool", "tool_call_id": tool_call_id, "content": payload},),
            tool_call_ids=(tool_call_id,),
        )

    raw_images = tuple(bytes(img) for img in images)
    fids = [file_id_for(img) for img in raw_images]
    tool_payload = json.dumps(
        [text, *(f"See file {fid}." for fid in fids)],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": TOOL_RESPONSE_OPEN}]
    total_img_tokens = 0
    for img, fid in zip(raw_images, fids, strict=True):
        total_img_tokens += count_image_tokens(img)
        user_parts.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f'<tool_result tool_name="{tool_name}" '
                        f'tool_call_id="{tool_call_id}" file_id="{fid}">'
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{IMAGE_PLACEHOLDER_PREFIX}{fid}"},
                },
                {"type": "text", "text": "</tool_result>"},
            ]
        )
    user_parts.append({"type": "text", "text": TOOL_RESPONSE_CLOSE})
    return LogicalMessage(
        kind="tool",
        wire=(
            {"role": "tool", "tool_call_id": tool_call_id, "content": tool_payload},
            {"role": "user", "content": user_parts},
        ),
        tool_call_ids=(tool_call_id,),
        images=raw_images,
        image_tokens=total_img_tokens,
    )


def retry_message(feedback: str) -> LogicalMessage:
    """Build Pydantic AI's RetryPromptPart user feedback message."""
    return LogicalMessage(
        kind="retry",
        wire=(
            {
                "role": "user",
                "content": f"Validation feedback:\n{feedback}\n\nFix the errors and try again.",
            },
        ),
    )


def _materialize_openai_wire(msg: LogicalMessage) -> list[dict[str, Any]]:
    """Expand ``rpent-image://`` placeholders to Base64 data URIs once for JSON APIs."""
    if not msg.images:
        return list(msg.wire)
    img_iter = iter(msg.images)
    out: list[dict[str, Any]] = []
    for item in msg.wire:
        content = item.get("content")
        if not isinstance(content, list):
            out.append(item)
            continue
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                raw = next(img_iter)
                b64 = base64.b64encode(raw).decode("ascii")
                new_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            else:
                new_parts.append(part)
        out.append({**item, "content": new_parts})
    return out


def validate_tool_calls(
    calls: Sequence[dict[str, Any]],
    tool_schemas: Mapping[str, dict[str, Any]],
    environment: str,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Validate one turn's parsed tool calls exactly like WorkflowCapability.

    Returns ``(error_kind, error_message, parsed_arguments)``.
    When ``error_kind is None`` and ``calls`` is non-empty, ``parsed_arguments``
    holds the validated argument mapping for ``calls[0]``.
    """
    if not calls:
        return None, None, None
    if len(calls) > 1:
        return (
            "tool_arguments",
            "Use one tool call per response. No tool executed.",
            None,
        )
    call = calls[0]
    name = call["name"]
    if name not in tool_schemas:
        return "tool_name", f"Unknown tool name: {name}. No tool executed.", None
    raw_args = call["arguments"]
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        # Reject NaN / Infinity which Python json.loads accepts by default.
        json.dumps(arguments, allow_nan=False)
        errors = list(Draft202012Validator(tool_schemas[name]).iter_errors(arguments))
        if errors:
            error = errors[0]
            location = ".".join(str(item) for item in error.absolute_path)
            raise ValueError(f"{location or name}: {error.message}")
        validate_relations(name, arguments)
    except (AssertionError, ValueError, TypeError, IndexError) as exc:
        hint = argument_hint(name, environment)
        return (
            "tool_arguments",
            f"Invalid tool arguments: {exc}. {hint} No tool executed.",
            None,
        )
    return None, None, arguments


@dataclass(frozen=True)
class CompactionConfig:
    """History compaction settings supporting ``base`` sliding window and ``dynamic`` pruning."""

    max_messages: int | None = 80
    keep_messages: int = 10
    context_window: int | None = 262144
    max_fraction: float | None = 0.7
    max_tokens: int | None = None
    preserve_first_user_message: bool = True
    strategy: Literal["base", "dynamic"] = "base"
    keep_recent_images: int = 4
    keep_recent_thinking: int = 4
    preserve_opening_images: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> CompactionConfig | None:
        if data is None:
            return None
        strategy = str(data.get("strategy", "base"))
        if strategy not in {"base", "dynamic"}:
            raise ValueError(f"Unsupported compaction strategy: {strategy!r}")
        return cls(
            max_messages=data.get("max_messages", 80),
            keep_messages=int(data.get("keep_messages", 10)),
            context_window=data.get("context_window", 262144),
            max_fraction=data.get("max_fraction", 0.7),
            max_tokens=data.get("max_tokens"),
            preserve_first_user_message=bool(
                data.get("preserve_first_user_message", True)
            ),
            strategy=strategy,  # type: ignore[arg-type]
            keep_recent_images=int(data.get("keep_recent_images", 4)),
            keep_recent_thinking=int(data.get("keep_recent_thinking", 4)),
            preserve_opening_images=bool(data.get("preserve_opening_images", True)),
        )

    @property
    def token_trigger(self) -> int | None:
        if self.max_tokens is not None:
            return int(self.max_tokens)
        if self.max_fraction is not None and self.context_window is not None:
            return int(self.context_window * self.max_fraction)
        return None


def _strip_message_images(msg: LogicalMessage) -> LogicalMessage:
    """Return a new LogicalMessage replacing every image_url part with [image omitted]."""
    if not msg.images:
        return msg
    new_wire: list[dict[str, Any]] = []
    for item in msg.wire:
        content = item.get("content")
        if not isinstance(content, list):
            new_wire.append(dict(item))
            continue
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                new_parts.append({"type": "text", "text": "[image omitted]"})
            else:
                new_parts.append(dict(part))
        new_wire.append({**item, "content": new_parts})
    return LogicalMessage(
        kind=msg.kind,
        wire=tuple(new_wire),
        tool_call_ids=msg.tool_call_ids,
        images=(),
        image_tokens=0,
    )


def _strip_message_thinking(msg: LogicalMessage) -> LogicalMessage:
    """Return a new LogicalMessage with assistant reasoning_content removed."""
    if msg.kind != "assistant" or not any(
        item.get("reasoning_content") for item in msg.wire
    ):
        return msg
    new_wire = tuple(
        {k: v for k, v in item.items() if k != "reasoning_content"}
        for item in msg.wire
    )
    return LogicalMessage(
        kind=msg.kind,
        wire=new_wire,
        tool_call_ids=msg.tool_call_ids,
        images=msg.images,
        image_tokens=msg.image_tokens,
    )


def find_safe_cutoff(messages: Sequence[LogicalMessage], keep_messages: int) -> int:
    """Return earliest index >= 0 keeping at least ``keep_messages`` without splitting tool/retry pairs."""
    if keep_messages <= 0 or len(messages) <= keep_messages:
        return 0
    cutoff = len(messages) - keep_messages
    call_index: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        if msg.kind == "assistant":
            for cid in msg.tool_call_ids:
                if cid:
                    call_index[cid] = idx
    changed = True
    while changed and cutoff > 0:
        changed = False
        # Never leave a Validation feedback retry message orphaned at the boundary.
        while cutoff > 0 and messages[cutoff].kind == "retry":
            cutoff -= 1
            changed = True
        for msg in messages[cutoff:]:
            if msg.kind == "tool":
                for cid in msg.tool_call_ids:
                    owner = call_index.get(cid)
                    if owner is not None and owner < cutoff:
                        cutoff = owner
                        changed = True
    return cutoff


def estimate_logical_tokens(
    messages: Sequence[LogicalMessage],
    tools: Sequence[dict[str, Any]] | None = None,
    count_text: Callable[[str], int] | None = None,
) -> int:
    """Token estimate aligned 1-to-1 with RLinf ``EpisodeConversation.estimate_tokens``."""
    text_chunks: list[str] = []
    chars = 0
    img_tokens = 0
    if tools:
        dumped_tools = json.dumps(tools, ensure_ascii=False)
        chars += len(dumped_tools)
        if count_text is not None:
            text_chunks.append(dumped_tools)
    for msg in messages:
        img_tokens += msg.image_tokens
        for wire in msg.wire:
            content = wire.get("content")
            if isinstance(content, str):
                chars += len(content)
                if count_text is not None:
                    text_chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        txt = str(part.get("text", ""))
                        chars += len(txt)
                        if count_text is not None:
                            text_chunks.append(txt)
            rc = wire.get("reasoning_content")
            if rc:
                chars += len(rc)
                if count_text is not None:
                    text_chunks.append(rc)
            if wire.get("tool_calls"):
                dumped_tc = json.dumps(wire["tool_calls"], ensure_ascii=False)
                chars += len(dumped_tc)
                if count_text is not None:
                    text_chunks.append(dumped_tc)
    text_tokens = (
        count_text("\n".join(text_chunks))
        if count_text is not None
        else math.ceil(chars / 4)
    )
    return text_tokens + img_tokens


@dataclass
class EpisodeConversation:
    """Stateful conversation manager shared by RPent standalone rollout and RLinf."""

    system_prompt: str
    user_message: str
    initial_observation_text: str
    tools: list[dict[str, Any]]
    environment: str
    compaction: CompactionConfig | None = None
    max_output_retries: int = 1
    initial_images: Sequence[bytes] = ()
    initial_image_cameras: Sequence[str] = ()
    messages: list[LogicalMessage] = field(default_factory=list)
    full_messages: list[LogicalMessage] = field(default_factory=list)
    compaction_events: list[dict[str, Any]] = field(default_factory=list)
    output_retries: int = 0

    def __post_init__(self) -> None:
        self._tool_schemas = {
            t["function"]["name"]: t["function"]["parameters"]
            for t in self.tools
            if t.get("type") == "function"
        }
        self._openai_wire_cache: dict[int, list[dict[str, Any]]] = {}
        if not self.messages:
            self.messages = [
                opening_message(
                    self.system_prompt,
                    self.user_message,
                    self.initial_observation_text,
                    initial_images=self.initial_images,
                    initial_image_cameras=self.initial_image_cameras,
                )
            ]
        if not self.full_messages:
            self.full_messages = list(self.messages)

        # Compaction-chain tracking (mirrors RLinf AppendOnlyHistory.chain_id).
        # Chain 0 starts with [opening]; every compaction seals the prior chain
        # (if it generated >=1 assistant turn) and begins a new chain whose
        # carried-over prefix is masked (loss=False) during SFT export.
        self._completed_chains: list[dict[str, Any]] = []
        self._chain_prefix_logical_count: int = len(self.messages)
        self._chain_compaction_event_idx: int | None = None
        self._chain_turn_indices: list[int] = []
        self._total_assistant_turns: int = 0

    def images(self) -> list[bytes]:
        """Return all raw binary PNG images across active messages in order."""
        return [img for msg in self.messages for img in msg.images]

    def total_image_tokens(self) -> int:
        return sum(msg.image_tokens for msg in self.messages)

    def estimate_tokens(
        self, count_text: Callable[[str], int] | None = None
    ) -> int:
        return estimate_logical_tokens(self.messages, self.tools, count_text)

    def all_images_by_id(self) -> dict[str, bytes]:
        """Return deduplicated ``{file_id: png_bytes}`` across lossless ``full_messages``."""
        out: dict[str, bytes] = {}
        for msg in self.full_messages:
            for img in msg.images:
                fid = file_id_for(img)
                if fid not in out:
                    out[fid] = img
        return out

    def _wire_from(
        self, logical_messages: Sequence[LogicalMessage], *, inline_base64: bool
    ) -> list[dict[str, Any]]:
        if not inline_base64:
            return [item for msg in logical_messages for item in msg.wire]
        out: list[dict[str, Any]] = []
        for msg in logical_messages:
            if not msg.images:
                out.extend(msg.wire)
                continue
            key = id(msg)
            cached = self._openai_wire_cache.get(key)
            if cached is None:
                cached = _materialize_openai_wire(msg)
                self._openai_wire_cache[key] = cached
            out.extend(cached)
        return out

    def wire_messages(self, *, inline_base64: bool = False) -> list[dict[str, Any]]:
        """Flatten active logical messages into template dicts (or Base64-inlined OpenAI dicts)."""
        return self._wire_from(self.messages, inline_base64=inline_base64)

    def full_wire_messages(self, *, inline_base64: bool = False) -> list[dict[str, Any]]:
        """Flatten lossless uncompacted logical messages into template dicts."""
        return self._wire_from(self.full_messages, inline_base64=inline_base64)

    def _snapshot_chain(
        self,
        logical_messages: Sequence[LogicalMessage],
        *,
        chain_id: int,
        compaction_event_index: int | None,
        prefix_logical_count: int,
        turn_indices: Sequence[int],
        compaction_triggered_after: bool,
    ) -> dict[str, Any]:
        clamped_prefix = min(prefix_logical_count, len(logical_messages))
        prefix_wire_count = sum(
            len(msg.wire) for msg in logical_messages[:clamped_prefix]
        )
        return {
            "chain_id": chain_id,
            "compaction_event_index": compaction_event_index,
            "prefix_logical_count": clamped_prefix,
            "prefix_wire_count": prefix_wire_count,
            "turn_indices": list(turn_indices),
            "compaction_triggered_after": compaction_triggered_after,
            "messages": self._wire_from(logical_messages, inline_base64=False),
        }

    def export_chains(self) -> list[dict[str, Any]]:
        """Return ordered compaction-split chain snapshots for SFT export."""
        chains = list(self._completed_chains)
        if self._chain_turn_indices or not chains:
            chains.append(
                self._snapshot_chain(
                    self.messages,
                    chain_id=len(chains),
                    compaction_event_index=self._chain_compaction_event_idx,
                    prefix_logical_count=self._chain_prefix_logical_count,
                    turn_indices=self._chain_turn_indices,
                    compaction_triggered_after=False,
                )
            )
        return chains

    def _compact_base(self, cfg: CompactionConfig) -> bool:
        cutoff = find_safe_cutoff(self.messages, cfg.keep_messages)
        if cutoff <= 0:
            return False
        trimmed = list(self.messages[cutoff:])
        if cfg.preserve_first_user_message:
            opening = self.messages[0]
            if not trimmed or trimmed[0] is not opening:
                trimmed = [opening, *trimmed]
        if trimmed == self.messages:
            return False
        self.messages = trimmed
        active_ids = {id(m) for m in trimmed}
        self._openai_wire_cache = {
            k: v for k, v in self._openai_wire_cache.items() if k in active_ids
        }
        return True

    def _compact_dynamic(
        self,
        cfg: CompactionConfig,
        *,
        current_tokens: int | None,
        count_text: Callable[[str], int] | None,
        over_messages: bool,
    ) -> bool:
        est_before = estimate_logical_tokens(self.messages, self.tools, count_text)
        start_img_idx = 1 if cfg.preserve_opening_images else 0
        image_indices = [
            i for i in range(start_img_idx, len(self.messages)) if self.messages[i].images
        ]
        keep_img = max(0, cfg.keep_recent_images)
        drop_img_set = (
            set(image_indices[:-keep_img]) if keep_img > 0 else set(image_indices)
        )

        think_indices = [
            i
            for i in range(1, len(self.messages))
            if self.messages[i].kind == "assistant"
            and any(bool(w.get("reasoning_content")) for w in self.messages[i].wire)
        ]
        keep_think = max(0, cfg.keep_recent_thinking)
        drop_think_set = (
            set(think_indices[:-keep_think]) if keep_think > 0 else set(think_indices)
        )

        changed = False
        updated: list[LogicalMessage] = []
        for idx, msg in enumerate(self.messages):
            new_msg = msg
            if idx in drop_img_set:
                new_msg = _strip_message_images(new_msg)
            if idx in drop_think_set:
                new_msg = _strip_message_thinking(new_msg)
            if new_msg is not msg:
                changed = True
            updated.append(new_msg)

        if changed:
            self.messages = updated
            active_ids = {id(m) for m in updated}
            self._openai_wire_cache = {
                k: v for k, v in self._openai_wire_cache.items() if k in active_ids
            }

        limit = cfg.token_trigger
        est_after = estimate_logical_tokens(self.messages, self.tools, count_text)
        saved = max(0, est_before - est_after)
        effective_after = (
            max(0, current_tokens - saved) if current_tokens is not None else est_after
        )
        still_over_tokens = limit is not None and effective_after > limit
        if not changed or still_over_tokens or over_messages:
            fell_back = self._compact_base(cfg)
            return changed or fell_back
        return True

    def maybe_compact(
        self,
        *,
        current_tokens: int | None = None,
        count_text: Callable[[str], int] | None = None,
    ) -> bool:
        """Apply base sliding-window or dynamic batch compaction when budget is exceeded."""
        cfg = self.compaction
        if cfg is None:
            return False
        over_messages = (
            cfg.max_messages is not None and len(self.messages) > cfg.max_messages
        )
        limit = cfg.token_trigger
        if limit is not None:
            tokens = (
                current_tokens
                if current_tokens is not None
                else estimate_logical_tokens(self.messages, self.tools, count_text)
            )
            over_tokens = tokens > limit
        else:
            over_tokens = False
        if not (over_messages or over_tokens):
            return False

        pre_compact_msgs = list(self.messages)
        before_len = len(pre_compact_msgs)
        before_images = sum(len(m.images) for m in pre_compact_msgs)
        if cfg.strategy == "dynamic":
            compacted = self._compact_dynamic(
                cfg,
                current_tokens=current_tokens,
                count_text=count_text,
                over_messages=over_messages,
            )
        else:
            compacted = self._compact_base(cfg)
        if compacted:
            event_idx = len(self.compaction_events)
            # Seal the pre-compaction chain if it contains at least one generated turn;
            # consecutive compactions with 0 turns in between collapse into the same prefix.
            if self._chain_turn_indices:
                self._completed_chains.append(
                    self._snapshot_chain(
                        pre_compact_msgs,
                        chain_id=len(self._completed_chains),
                        compaction_event_index=self._chain_compaction_event_idx,
                        prefix_logical_count=self._chain_prefix_logical_count,
                        turn_indices=self._chain_turn_indices,
                        compaction_triggered_after=True,
                    )
                )
            self._chain_prefix_logical_count = len(self.messages)
            self._chain_compaction_event_idx = event_idx
            self._chain_turn_indices = []
            self.compaction_events.append(
                {
                    "event_index": event_idx,
                    "strategy": cfg.strategy,
                    "trigger_tokens": current_tokens,
                    "logical_messages_before": before_len,
                    "logical_messages_after": len(self.messages),
                    "images_before": before_images,
                    "images_after": sum(len(m.images) for m in self.messages),
                    "full_messages_count": len(self.full_messages),
                }
            )
        return compacted

    def append_assistant(
        self,
        *,
        content: str | None,
        reasoning_content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        msg = assistant_message(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
        self.messages.append(msg)
        self.full_messages.append(msg)
        self._chain_turn_indices.append(self._total_assistant_turns)
        self._total_assistant_turns += 1

    def append_tool_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        text: str,
        images: Sequence[bytes] = (),
    ) -> None:
        self.output_retries = 0
        msg = tool_return_message(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            text=text,
            images=images,
        )
        self.messages.append(msg)
        self.full_messages.append(msg)

    def validate_or_retry(
        self, calls: Sequence[dict[str, Any]]
    ) -> tuple[str | None, str | None, dict[str, Any] | None, bool]:
        """Validate calls; append retry feedback if within retry budget.

        Returns ``(error_kind, error_msg, arguments, should_retry)``.
        """
        err_kind, err_msg, args = validate_tool_calls(
            calls, self._tool_schemas, self.environment
        )
        if err_kind is None:
            self.output_retries = 0
            return None, None, args, False
        self.output_retries += 1
        if self.output_retries <= self.max_output_retries:
            msg = retry_message(err_msg or "")
            self.messages.append(msg)
            self.full_messages.append(msg)
            return err_kind, err_msg, None, True
        return err_kind, err_msg, None, False
