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

"""Standalone async rollout, evaluation, and resumable SFT trajectory collector.

Aligned 1-to-1 with RLinf Harness v2 RL rollout (``RPentAgentLoopWorker``):
- Identical ``EpisodeConversation`` wire format, step-0 opening images,
  ``jaw_mm`` feedback, ``execute_actions(..., observe=...)``, deterministic
  ``call_{turn_idx}_{idx}`` IDs, and ``base``/``dynamic`` context compaction.
- Exact 2-stage compaction triggering (pre-turn budget trigger at 48,000 +
  hard ``context_window - max_tokens`` guard with at most 1 forced compaction
  and strict ``context_length_exceeded`` termination, never clamping
  ``max_tokens``). Optional ``--compaction-tokenizer`` counts exact target
  student (Qwen3.5) text + visual tokens even when collecting from non-Qwen
  teachers such as DeepSeek.
- Compaction-chain snapshots saved in ``trajectory.json`` (``schema_version: 3``)
  so ``robots.rl.sft_export`` splits compacted episodes into independent
  distribution-matched training chains mirroring RLinf's ``AppendOnlyHistory``.
- Never retries non-idempotent simulator ``/execute`` calls mid-episode and
  never persists ``infrastructure_error`` episodes as completed trajectories;
  infra failures log to ``infra_errors.jsonl`` so ``--resume`` cleanly retries.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Literal

import httpx

from robots.rl.conversation import (
    CompactionConfig,
    EpisodeConversation,
    estimate_logical_tokens,
)
from robots.rl.server import BINARY_EXECUTE_MEDIA_TYPE, unpack_execute_response
from rpent.utils.logging import get_logger

logger = get_logger("rl_rollout")

RETRYABLE_LLM_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
FAILOVER_LLM_STATUS = frozenset({500, 502, 503, 504})
ProviderMode = Literal["auto", "sglang", "openai"]


class InfraEpisodeError(RuntimeError):
    """Typed infrastructure failure that aborts the episode without saving it."""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def parse_model_urls(
    model_url: str | None = None, model_urls_file: Path | None = None
) -> list[str]:
    """Parse one or more OpenAI-compatible ``/v1`` base URLs."""
    urls: list[str] = []
    if model_url:
        for item in model_url.split(","):
            cleaned = item.strip().rstrip("/")
            if cleaned:
                urls.append(cleaned)
    if model_urls_file is not None:
        for line in model_urls_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip().rstrip("/")
            if cleaned and not cleaned.startswith("#"):
                urls.append(cleaned)
    deduped = list(dict.fromkeys(urls))
    if not deduped:
        raise ValueError("Provide at least one URL via --model-url or --model-urls-file")
    return deduped


def detect_provider(
    provider: ProviderMode,
    *,
    model_name: str,
    model_urls: Sequence[str],
    api_key: str | None,
) -> Literal["sglang", "openai"]:
    """Select wire protocol flavor (``sglang`` vs pure ``openai``)."""
    if provider in {"sglang", "openai"}:
        return provider
    lower_model = model_name.lower()
    lower_urls = " ".join(u.lower() for u in model_urls)
    if api_key or "deepseek" in lower_model or any(
        hint in lower_urls
        for hint in ("infini-ai", "deepseek", "openai.com", "openrouter", "maas")
    ):
        return "openai"
    return "sglang"


def split_inline_thinking(
    content: str | None, reasoning_content: str | None
) -> tuple[str | None, str | None]:
    """Extract inline ``<think>...</think>`` when a server omits ``reasoning_content``."""
    clean_rc = (
        reasoning_content.strip()
        if isinstance(reasoning_content, str) and reasoning_content.strip()
        else None
    )
    if clean_rc is not None:
        clean_c = content.strip() if isinstance(content, str) and content.strip() else None
        return clean_c, clean_rc
    if not isinstance(content, str):
        return None, None
    if "</think>" not in content:
        clean_c = content.strip() or None
        return clean_c, None
    before, _, after = content.partition("</think>")
    if "<think>" in before:
        reasoning = before.rsplit("<think>", 1)[-1].strip()
    else:
        reasoning = before.strip()
    rest = after.strip()
    return (rest or None), (reasoning or None)


def sanitize_episode_key(raw: str) -> str:
    """Convert an arbitrary query identifier into a safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip()).strip("._-")
    return cleaned or "episode"


def _task_group_key(task: dict[str, Any], query_id: str) -> str:
    suite = task.get("suite")
    suite_prefix = f"{str(suite).strip()}::" if suite is not None and str(suite).strip() else ""
    for field in ("task_name", "task_id", "bddl_file_name", "task"):
        val = task.get(field)
        if val is not None and str(val).strip():
            base = str(val).strip()
            if suite_prefix and not base.startswith(suite_prefix):
                return f"{suite_prefix}{base}"
            return base
    return query_id


def parse_shard_spec(spec: str | None) -> tuple[int, int]:
    """Parse ``i/N`` (0-based shard index and total shard count)."""
    if not spec:
        return 0, 1
    parts = spec.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid --shard value {spec!r}; expected '<index>/<total>'")
    shard_idx, num_shards = int(parts[0]), int(parts[1])
    if num_shards <= 0 or not (0 <= shard_idx < num_shards):
        raise ValueError(
            f"Invalid --shard {spec!r}: require 0 <= index < total and total > 0"
        )
    return shard_idx, num_shards


def build_episode_schedule(
    rows: Sequence[dict[str, Any]],
    *,
    interleave_tasks: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
    target_count: int | None = None,
    limit: int | None = None,
    pass_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic episode assignments with strictly disjoint intra-task sharding.

    Sharding is applied *inside* each task group before round-robin interleaving
    so shards never collide on the same seed/init state. When ``target_count``
    exceeds the shard length, additional passes (``r1``, ``r2``, ...) cycle
    cleanly with distinct ``episode_key`` values.
    """
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        task = (
            json.loads(row["answer"])
            if isinstance(row.get("answer"), str)
            else dict(row)
        )
        raw_qid = row.get("query_id")
        if not raw_qid:
            env_name = task.get("environment", "env")
            tname = _task_group_key(task, f"ep_{idx:05d}")
            seed = task.get("seed", task.get("init_state_id", idx))
            raw_qid = f"{env_name}-{tname}-s{seed}-i{idx}"
        parsed.append(
            {
                "source_idx": idx,
                "query_id": str(raw_qid),
                "task_group": _task_group_key(task, str(raw_qid)),
                "task": task,
            }
        )

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in parsed:
        groups.setdefault(item["task_group"], []).append(item)

    sharded_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for gname, items in groups.items():
        if num_shards > 1:
            subset = items[shard_index::num_shards]
        else:
            subset = list(items)
        if subset:
            sharded_groups[gname] = subset

    if interleave_tasks:
        base_order: list[dict[str, Any]] = []
        max_len = max((len(v) for v in sharded_groups.values()), default=0)
        for pos in range(max_len):
            for items in sharded_groups.values():
                if pos < len(items):
                    base_order.append(items[pos])
    else:
        base_order = [item for items in sharded_groups.values() for item in items]

    if not base_order:
        return []

    total = target_count if target_count is not None else len(base_order)
    if limit is not None:
        total = min(total, limit)

    schedule: list[dict[str, Any]] = []
    for seq_idx in range(total):
        pass_idx = pass_offset + (seq_idx // len(base_order))
        item = base_order[seq_idx % len(base_order)]
        ep_key = sanitize_episode_key(f"{item['query_id']}__r{pass_idx}")
        schedule.append(
            {
                "schedule_idx": seq_idx,
                "pass_idx": pass_idx,
                "episode_key": ep_key,
                "query_id": item["query_id"],
                "task_group": item["task_group"],
                "task": dict(item["task"]),
            }
        )
    return schedule


class EndpointPool:
    """Sticky least-loaded endpoint router maximizing SGLang RadixCache hits."""

    def __init__(self, urls: Sequence[str]) -> None:
        self.urls = [u.rstrip("/") for u in urls]
        self._inflight: dict[str, int] = {u: 0 for u in self.urls}

    def acquire(self) -> str:
        min_load = min(self._inflight.values())
        candidates = [u for u, load in self._inflight.items() if load == min_load]
        chosen = random.choice(candidates) if len(candidates) > 1 else candidates[0]
        self._inflight[chosen] += 1
        return chosen

    def release(self, url: str) -> None:
        if url in self._inflight:
            self._inflight[url] = max(0, self._inflight[url] - 1)

    def failover(self, current_url: str) -> str:
        """Atomically move one in-flight slot from ``current_url`` to a least-loaded peer."""
        others = [u for u in self.urls if u != current_url]
        if not others:
            return current_url
        min_load = min(self._inflight[u] for u in others)
        candidates = [u for u in others if self._inflight[u] == min_load]
        next_url = random.choice(candidates) if len(candidates) > 1 else candidates[0]
        self.release(current_url)
        self._inflight[next_url] += 1
        return next_url

    def alternate(self, exclude_url: str) -> str:
        others = [u for u in self.urls if u != exclude_url]
        if not others:
            return exclude_url
        min_load = min(self._inflight[u] for u in others)
        candidates = [u for u in others if self._inflight[u] == min_load]
        return random.choice(candidates) if len(candidates) > 1 else candidates[0]


def _normalize_for_hf_template(raw: dict[str, Any]) -> dict[str, Any]:
    """Mirror RLinf ``history._normalize_message`` for exact Qwen3.5 token counting."""
    msg = dict(raw)
    if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
        msg.pop("reasoning_content", None)
    if msg.get("content") is None:
        msg["content"] = ""
    if isinstance(msg["content"], list) or not msg.get("tool_calls"):
        return msg
    calls, malformed = [], []
    for call in msg["tool_calls"]:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if isinstance(args, dict):
            calls.append({**call, "function": {**fn, "arguments": args}})
        else:
            malformed.append(json.dumps(call, ensure_ascii=False))
    if malformed:
        msg["content"] = msg["content"] + "\n" + "\n".join(malformed)
    msg["tool_calls"] = calls
    return msg


class CompactionTokenCounter:
    """Exact Qwen3.5 multimodal prompt token counter (text + smart_resize vision tokens)."""

    def __init__(self, tokenizer_path: Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer_path = Path(tokenizer_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path), trust_remote_code=True
        )
        self._lock = threading.Lock()

    def count_prompt_tokens(self, conv: EpisodeConversation) -> int:
        wire_msgs = [_normalize_for_hf_template(m) for m in conv.wire_messages(inline_base64=False)]
        with self._lock:
            text = self._tokenizer.apply_chat_template(
                wire_msgs,
                tools=conv.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        # Each <|image_pad|> placeholder occupies 1 token in text_ids and expands to
        # msg.image_tokens merged visual tokens in SGLang/HF Qwen3VLProcessor.
        num_images = len(conv.images())
        return len(text_ids) + conv.total_image_tokens() - num_images


def _retry_delay_s(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(0.5, min(90.0, float(retry_after)))
        except ValueError:
            pass
    base = min(60.0, 2.0 * (2**attempt))
    return base + random.uniform(0.0, 1.5)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _save_episode_atomic(
    *,
    ep_dir: Path,
    episode_key: str,
    query_id: str,
    pass_idx: int,
    conv: EpisodeConversation,
    outcome: dict[str, Any],
    model_name: str,
    provider: str,
    model_url: str,
    rollout_config: dict[str, Any],
    usage_totals: dict[str, int],
) -> Path:
    """Persist deduplicated PNGs + atomic ``trajectory.json`` (v3 with chains) and ``outcome.json``."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    img_dir = ep_dir / "images"
    images_by_id = conv.all_images_by_id()
    image_relpaths: dict[str, str] = {}
    if images_by_id:
        img_dir.mkdir(parents=True, exist_ok=True)
        for fid, png_bytes in images_by_id.items():
            img_path = img_dir / f"{fid}.png"
            if not img_path.exists():
                tmp_img = img_dir / f".{fid}.png.tmp.{os.getpid()}.{time.monotonic_ns()}"
                tmp_img.write_bytes(png_bytes)
                os.replace(tmp_img, img_path)
            image_relpaths[fid] = f"images/{fid}.png"

    chains = conv.export_chains()
    trajectory = {
        "schema_version": 3,
        "episode_key": episode_key,
        "query_id": query_id,
        "pass_idx": pass_idx,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "environment": conv.environment,
        "model": {
            "name": model_name,
            "provider": provider,
            "endpoint": model_url,
        },
        "rollout_config": rollout_config,
        "outcome": outcome,
        "usage": usage_totals,
        "tools": conv.tools,
        "images": image_relpaths,
        "compaction_events": conv.compaction_events,
        "chains": chains,
        "full_messages": conv.full_wire_messages(inline_base64=False),
        "compacted_messages": conv.wire_messages(inline_base64=False),
    }
    traj_path = ep_dir / "trajectory.json"
    _atomic_write_json(traj_path, trajectory)
    _atomic_write_json(ep_dir / "outcome.json", outcome)
    return traj_path


def _load_completed_episodes(
    output_dir: Path, *, save_trajectory: bool = True
) -> dict[str, dict[str, Any]]:
    """Scan ``manifest.jsonl`` and episode files for ``--resume``."""
    completed: dict[str, dict[str, Any]] = {}
    marker_name = "trajectory.json" if save_trajectory else "outcome.json"
    manifest = output_dir / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                key = rec.get("episode_key")
                if key:
                    marker_path = output_dir / "episodes" / key / marker_name
                    if marker_path.exists():
                        completed[str(key)] = rec
            except json.JSONDecodeError:
                continue

    episodes_root = output_dir / "episodes"
    if episodes_root.is_dir():
        for marker_path in episodes_root.glob(f"*/{marker_name}"):
            key = marker_path.parent.name
            if key in completed:
                continue
            try:
                data = json.loads(marker_path.read_text(encoding="utf-8"))
                out = data.get("outcome") if save_trajectory else data
                out = out or {}
                completed[key] = {
                    "episode_key": key,
                    "query_id": data.get("query_id") or out.get("query_id"),
                    "environment": data.get("environment") or out.get("environment"),
                    "success": bool(out.get("success", False)),
                    "turns": int(out.get("turns", 0)),
                    "termination_reason": out.get("termination_reason"),
                    "error_kind": out.get("error_kind"),
                    "elapsed_s": out.get("elapsed_s", 0.0),
                    "trajectory_path": (
                        str(marker_path.relative_to(output_dir))
                        if save_trajectory
                        else None
                    ),
                }
            except Exception:
                continue
    return completed


class ContextLengthExceededError(RuntimeError):
    """Raised when an OpenAI/SGLang endpoint rejects a prompt+max_tokens > context."""

    def __init__(
        self,
        message: str,
        *,
        input_tokens: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.max_context_tokens = max_context_tokens


def _parse_context_length_error(body: str) -> ContextLengthExceededError | None:
    lower = body.lower()
    if not any(
        hint in lower
        for hint in (
            "maximum context length",
            "context_length",
            "exceeds the model",
            "too many tokens",
            "prompt is too long",
            "input length",
        )
    ):
        return None
    max_ctx: int | None = None
    in_toks: int | None = None
    m_max = re.search(r"maximum context length of (\d+)", body)
    if m_max:
        max_ctx = int(m_max.group(1))
    m_in = re.search(r"(\d+) tokens from the input", body)
    if m_in:
        in_toks = int(m_in.group(1))
    return ContextLengthExceededError(
        body[:400], input_tokens=in_toks, max_context_tokens=max_ctx
    )


async def _post_llm_with_retry(
    client: httpx.AsyncClient,
    *,
    model_url: str,
    endpoint_pool: EndpointPool | None,
    payload: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
) -> tuple[dict[str, Any], str]:
    """Call ``/chat/completions`` with exponential backoff and multi-replica failover."""
    active_url = model_url.rstrip("/")
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                f"{active_url}/chat/completions", json=payload, headers=headers
            )
            if resp.status_code in RETRYABLE_LLM_STATUS and attempt < max_retries:
                delay = _retry_delay_s(attempt, resp.headers.get("Retry-After"))
                if (
                    resp.status_code in FAILOVER_LLM_STATUS
                    and endpoint_pool is not None
                    and len(endpoint_pool.urls) > 1
                ):
                    next_url = endpoint_pool.failover(active_url)
                    if next_url != active_url:
                        logger.warning(
                            "LLM HTTP %d on %s; failing over to %s",
                            resp.status_code,
                            active_url,
                            next_url,
                        )
                        active_url = next_url
                logger.warning(
                    "LLM HTTP %d from %s (attempt %d/%d); retrying in %.1fs",
                    resp.status_code,
                    active_url,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if resp.status_code == 400:
                ctx_err = _parse_context_length_error(resp.text)
                if ctx_err is not None:
                    raise ctx_err
                raise httpx.HTTPStatusError(
                    f"HTTP 400 from {active_url}: {resp.text[:300]}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("choices"):
                raise ValueError(f"Empty choices in LLM response: {data}")
            return data, active_url
        except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _retry_delay_s(attempt)
            if endpoint_pool is not None and attempt >= 1 and len(endpoint_pool.urls) > 1:
                next_url = endpoint_pool.failover(active_url)
                if next_url != active_url:
                    logger.warning(
                        "LLM transport error on %s (%s); failing over to %s",
                        active_url,
                        type(exc).__name__,
                        next_url,
                    )
                    active_url = next_url
            logger.warning(
                "LLM error from %s (%s: %s; attempt %d/%d); retrying in %.1fs",
                active_url,
                type(exc).__name__,
                exc,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _normalize_tool_calls_for_turn(
    raw_tool_calls: Sequence[dict[str, Any]], turn_idx: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize tool calls to deterministic ``call_{turn_idx}_{idx}`` IDs (RLinf parity)."""
    wire_calls: list[dict[str, Any]] = []
    parsed_calls: list[dict[str, Any]] = []
    for idx, tc in enumerate(raw_tool_calls):
        cid = f"call_{turn_idx}_{idx}"
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            name = ""
            arguments: Any = ""
        else:
            name = str(fn.get("name") or "")
            arguments = fn.get("arguments", "")
        wire_calls.append(
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        parsed_calls.append({"id": cid, "name": name, "arguments": arguments})
    return wire_calls, parsed_calls


async def run_episode(
    *,
    model_url: str,
    model_name: str,
    sim_url: str,
    task: dict[str, Any],
    max_turns: int = 50,
    max_tokens: int = 16384,
    context_window: int = 65536,
    max_episode_steps: int = 10000,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    thinking: bool = True,
    reasoning_effort: str | None = None,
    compaction: CompactionConfig | None = None,
    token_counter: CompactionTokenCounter | None = None,
    image_size: int | None = None,
    max_actions_per_call: int = 3,
    include_jaw_mm: bool = True,
    record_artifacts: bool = False,
    output_dir: Path | None = None,
    request_timeout_s: float = 600.0,
    api_key: str | None = None,
    provider: ProviderMode = "auto",
    max_llm_retries: int = 8,
    max_infra_retries: int = 3,
    save_trajectory: bool = True,
    episode_key: str | None = None,
    query_id: str | None = None,
    pass_idx: int = 0,
    client: httpx.AsyncClient | None = None,
    endpoint_pool: EndpointPool | None = None,
) -> tuple[dict[str, Any], str]:
    """Run one standalone Harness v2 episode aligned 1-to-1 with RLinf.

    Returns ``(outcome_dict, final_model_url)`` so callers release the active
    endpoint slot accurately after any failover.
    """
    started = time.perf_counter()
    resolved_provider = detect_provider(
        provider,
        model_name=model_name,
        model_urls=[model_url],
        api_key=api_key,
    )
    llm_headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        llm_headers["Authorization"] = f"Bearer {api_key}"

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=request_timeout_s,
            trust_env=False,
            limits=httpx.Limits(keepalive_expiry=30.0),
        )

    active_model_url = model_url.rstrip("/")
    try:
        base_seed = task.get("seed", task.get("init_state_id"))
        seeds = [base_seed, *(task.get("fallback_seeds") or ())]
        opened: dict[str, Any] | None = None
        initial_images: list[bytes] = []
        last_exc: Exception | None = None

        for infra_attempt in range(max_infra_retries + 1):
            for seed in seeds:
                attempt_task = dict(task)
                if seed is not None:
                    attempt_task["seed"] = seed
                    if "init_state_id" in attempt_task:
                        attempt_task["init_state_id"] = seed
                try:
                    open_resp = await client.post(
                        f"{sim_url.rstrip('/')}/sessions/open",
                        json={
                            "task": attempt_task,
                            "max_episode_steps": max_episode_steps,
                            "image_size": image_size,
                            "max_actions_per_call": max_actions_per_call,
                            "reasoning": thinking,
                            "include_jaw_mm": include_jaw_mm,
                            "record_artifacts": record_artifacts,
                            "output_dir": (
                                str(output_dir)
                                if (record_artifacts and output_dir)
                                else None
                            ),
                        },
                        headers={"Accept": BINARY_EXECUTE_MEDIA_TYPE},
                    )
                    open_resp.raise_for_status()
                    if BINARY_EXECUTE_MEDIA_TYPE in open_resp.headers.get(
                        "content-type", ""
                    ):
                        opened, initial_images = unpack_execute_response(
                            open_resp.content
                        )
                    else:
                        opened = open_resp.json()
                        initial_images = [
                            base64.b64decode(b64)
                            for b64 in opened.get("images_b64") or ()
                        ]
                    break
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    last_exc = exc
            if opened is not None:
                break
            if infra_attempt < max_infra_retries:
                await asyncio.sleep(_retry_delay_s(infra_attempt))

        if opened is None:
            raise InfraEpisodeError(
                f"SimPoolServer open failed: {last_exc}", kind="env_setup"
            ) from last_exc
        sid = opened["session_id"]

        conv = EpisodeConversation(
            system_prompt=opened["system_prompt"],
            user_message=opened["user_message"],
            initial_observation_text=opened["initial_observation"],
            tools=opened["tools"],
            environment=opened["environment"],
            compaction=compaction,
            initial_images=initial_images,
            initial_image_cameras=opened.get("image_cameras") or (),
        )
        done = bool(opened["done"])
        success = bool(opened["success"])
        truncated = bool(opened["truncated"])
        reason = "success" if success else "planner_completed"
        error_kind: str | None = None
        error_msg: str | None = None
        turns = 0
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        last_turn_tokens: int | None = None
        last_msg_count = len(conv.messages)

        async def _measure_prompt_tokens() -> int | None:
            if token_counter is not None:
                return await asyncio.to_thread(token_counter.count_prompt_tokens, conv)
            if last_turn_tokens is not None:
                delta_msgs = conv.messages[last_msg_count:]
                return last_turn_tokens + estimate_logical_tokens(delta_msgs)
            return None

        try:
            while not done and turns < max_turns:
                # Stage 1: Proactive compaction aligned with RLinf token trigger.
                est_current_tokens = await _measure_prompt_tokens()
                if conv.maybe_compact(current_tokens=est_current_tokens):
                    last_turn_tokens = None

                # Stage 2: Exact RLinf budget guard (seq_limit - prompt_len >= max_tokens).
                # Allows at most 1 forced compaction; never clamps max_tokens.
                resp_json: dict[str, Any] | None = None
                for overflow_attempt in range(2):
                    prompt_tokens = await _measure_prompt_tokens()
                    if prompt_tokens is None and token_counter is None:
                        prompt_tokens = conv.estimate_tokens()
                    if (
                        prompt_tokens is not None
                        and context_window - prompt_tokens < max_tokens
                    ):
                        if overflow_attempt == 0 and conv.maybe_compact(
                            current_tokens=max(prompt_tokens, context_window)
                        ):
                            last_turn_tokens = None
                            continue
                        error_kind = "context_length_exceeded"
                        reason = "context_length_exceeded"
                        error_msg = (
                            f"Prompt tokens ({prompt_tokens}) leave less than "
                            f"max_tokens={max_tokens} in context_window={context_window}"
                        )
                        resp_json = None
                        break

                    payload: dict[str, Any] = {
                        "model": model_name,
                        "messages": conv.wire_messages(inline_base64=True),
                        "tools": conv.tools,
                        "parallel_tool_calls": False,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    if resolved_provider == "sglang":
                        payload["top_k"] = top_k
                        chat_kwargs: dict[str, Any] = {
                            "enable_thinking": thinking,
                            "preserve_thinking": True,
                        }
                        if reasoning_effort is not None:
                            chat_kwargs["reasoning_effort"] = reasoning_effort
                        payload["chat_template_kwargs"] = chat_kwargs
                    try:
                        resp_json, active_model_url = await _post_llm_with_retry(
                            client,
                            model_url=active_model_url,
                            endpoint_pool=endpoint_pool,
                            payload=payload,
                            headers=llm_headers,
                            max_retries=max_llm_retries,
                        )
                        break
                    except ContextLengthExceededError as ctx_exc:
                        forced_tokens = max(
                            ctx_exc.input_tokens or context_window,
                            context_window,
                        )
                        if overflow_attempt == 0 and conv.maybe_compact(
                            current_tokens=forced_tokens
                        ):
                            last_turn_tokens = None
                            logger.info(
                                "Context length exceeded on %s (in=%s max=%s); "
                                "forced 1 compaction and retrying turn %d",
                                episode_key or sid,
                                ctx_exc.input_tokens,
                                ctx_exc.max_context_tokens,
                                turns + 1,
                            )
                            continue
                        error_kind = "context_length_exceeded"
                        reason = "context_length_exceeded"
                        error_msg = str(ctx_exc)
                        resp_json = None
                        break

                if resp_json is None:
                    break

                usage = resp_json.get("usage") or {}
                prompt_toks = int(usage.get("prompt_tokens") or 0)
                comp_toks = int(usage.get("completion_tokens") or 0)
                usage_totals["prompt_tokens"] += prompt_toks
                usage_totals["completion_tokens"] += comp_toks
                prompt_details = usage.get("prompt_tokens_details") or {}
                cached_toks = int(
                    prompt_details.get("cached_tokens")
                    or usage.get("prompt_cache_hit_tokens")
                    or 0
                )
                usage_totals["cached_tokens"] += cached_toks
                if prompt_toks > 0:
                    last_turn_tokens = prompt_toks + comp_toks
                    last_msg_count = len(conv.messages) + 1

                turn_idx = turns  # 0-based, matching RLinf agent_loop.py:263
                choice = resp_json["choices"][0]
                turns += 1
                msg = choice["message"]
                finish = choice.get("finish_reason")
                raw_tool_calls = msg.get("tool_calls") or []
                wire_tool_calls, parsed_calls = _normalize_tool_calls_for_turn(
                    raw_tool_calls, turn_idx
                )
                clean_content, clean_reasoning = split_inline_thinking(
                    msg.get("content"), msg.get("reasoning_content")
                )
                conv.append_assistant(
                    content=clean_content,
                    reasoning_content=clean_reasoning,
                    tool_calls=wire_tool_calls or None,
                )
                if finish == "length":
                    error_kind, reason = "response_truncated", "response_truncated"
                    error_msg = "Token limit reached before completing response"
                    break
                if finish not in {None, "stop", "tool_calls", "function_call"}:
                    error_kind, reason = str(finish), str(finish)
                    error_msg = f"Abnormal finish_reason={finish!r}"
                    break
                if not parsed_calls:
                    reason = "planner_completed"
                    break

                err_kind, err_msg, args, should_retry = conv.validate_or_retry(
                    parsed_calls
                )
                if err_kind is not None:
                    if should_retry:
                        continue
                    error_kind, error_msg, reason = err_kind, err_msg, err_kind
                    break

                # Never retry non-idempotent /execute mid-episode (matches RLinf):
                # any failure aborts cleanly as InfraEpisodeError so --resume retries.
                try:
                    exec_resp = await client.post(
                        f"{sim_url.rstrip('/')}/sessions/{sid}/execute",
                        json={"name": parsed_calls[0]["name"], "arguments": args},
                        headers={"Accept": BINARY_EXECUTE_MEDIA_TYPE},
                    )
                    exec_resp.raise_for_status()
                    if BINARY_EXECUTE_MEDIA_TYPE in exec_resp.headers.get(
                        "content-type", ""
                    ):
                        executed, images = unpack_execute_response(exec_resp.content)
                    else:
                        executed = exec_resp.json()
                        images = [
                            base64.b64decode(b64)
                            for b64 in executed.get("images_b64") or ()
                        ]
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    raise InfraEpisodeError(
                        f"Tool {parsed_calls[0]['name']} transport error on {sim_url}: {exc}",
                        kind="tool_transport",
                    ) from exc

                if executed.get("infrastructure_error"):
                    raise InfraEpisodeError(
                        str(executed["infrastructure_error"]),
                        kind="tool_transport",
                    )

                conv.append_tool_result(
                    tool_name=parsed_calls[0]["name"],
                    tool_call_id=parsed_calls[0]["id"],
                    text=executed["text"],
                    images=images,
                )
                success = bool(executed["success"])
                truncated = bool(executed["truncated"])
                done = bool(executed["done"])
            else:
                if not done and turns >= max_turns:
                    reason = "request_limit"
                elif success:
                    reason = "success"
                elif truncated:
                    reason = "environment_step_limit"
        finally:
            try:
                await client.post(
                    f"{sim_url.rstrip('/')}/sessions/{sid}/close", json={}
                )
            except Exception:
                pass
    finally:
        if owns_client:
            await client.aclose()

    if turns == 0:
        raise InfraEpisodeError(
            f"Episode {episode_key or sid} ended with 0 sampled turns ({reason})",
            kind="env_setup",
        )

    chains = conv.export_chains()
    outcome = {
        "episode_key": episode_key,
        "query_id": query_id,
        "pass_idx": pass_idx,
        "environment": conv.environment,
        "task": task,
        "init_state_id": opened.get("init_state_id"),
        "success": success,
        "truncated": truncated,
        "turns": turns,
        "num_chains": len(chains),
        "termination_reason": "success" if success else reason,
        "error_kind": error_kind,
        "error": error_msg,
        "compaction_count": len(conv.compaction_events),
        "num_images": len(conv.all_images_by_id()),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }

    if output_dir is not None:
        if save_trajectory:
            rollout_config = {
                "max_turns": max_turns,
                "max_tokens": max_tokens,
                "context_window": context_window,
                "max_episode_steps": max_episode_steps,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if resolved_provider == "sglang" else None,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
                "compaction_tokenizer": (
                    str(token_counter.tokenizer_path)
                    if token_counter is not None
                    else None
                ),
                "image_size": image_size,
                "max_actions_per_call": max_actions_per_call,
                "include_jaw_mm": include_jaw_mm,
                "compaction": (
                    {
                        "strategy": compaction.strategy,
                        "max_messages": compaction.max_messages,
                        "max_tokens": compaction.max_tokens,
                        "context_window": compaction.context_window,
                        "keep_messages": compaction.keep_messages,
                        "keep_recent_images": compaction.keep_recent_images,
                        "keep_recent_thinking": compaction.keep_recent_thinking,
                        "preserve_opening_images": compaction.preserve_opening_images,
                    }
                    if compaction is not None
                    else None
                ),
            }
            _save_episode_atomic(
                ep_dir=output_dir,
                episode_key=episode_key or output_dir.name,
                query_id=query_id or (episode_key or output_dir.name),
                pass_idx=pass_idx,
                conv=conv,
                outcome=outcome,
                model_name=model_name,
                provider=resolved_provider,
                model_url=active_model_url,
                rollout_config=rollout_config,
                usage_totals=usage_totals,
            )
        else:
            _atomic_write_json(output_dir / "outcome.json", outcome)

    return outcome, active_model_url


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-url",
        default=None,
        help="OpenAI-compatible /v1 URL (comma-separated for multi-replica routing)",
    )
    parser.add_argument(
        "--model-urls-file",
        type=Path,
        default=None,
        help="Optional newline-delimited file of /v1 URLs for sticky load balancing",
    )
    parser.add_argument("--model-name", default="policy")
    parser.add_argument(
        "--provider",
        choices=("auto", "sglang", "openai"),
        default="auto",
        help="API dialect: sglang (adds top_k/chat_template_kwargs), openai, or auto",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Bearer API key (prefer --api-key-env to avoid shell history)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding optional Bearer API key",
    )
    parser.add_argument("--sim-url", required=True, help="SimPoolServer HTTP base URL")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Total trajectories to collect (cycles passes r0, r1... if > shard rows)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Optional intra-task shard '<index>/<total>' (e.g. 0/2 vs 1/2 across teachers)",
    )
    parser.add_argument(
        "--interleave-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round-robin across tasks so every task is covered evenly (default: true)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes already saved in output_dir/episodes/*",
    )
    parser.add_argument(
        "--save-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full message trajectories + chain snapshots + PNGs per episode for SFT",
    )
    parser.add_argument("--pass-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--context-window",
        type=int,
        default=65536,
        help="Hard context limit matching RLinf runner.seq_length (default: 65536)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Simulator step limit passed to /sessions/open (default: 10000)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reasoning/thinking mode (default: true)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("xhigh", "medium", "low"),
        default=None,
        help="Optional reasoning_effort override for Qwen3.8 SGLang (default: model default)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override per-environment default VLM display size (default: env-specific)",
    )
    parser.add_argument("--max-actions-per-call", type=int, default=3)
    parser.add_argument(
        "--include-jaw-mm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose measured physical gripper jaw_mm (mm) in robot state feedback",
    )
    parser.add_argument(
        "--context-management",
        choices=("base", "dynamic"),
        default="dynamic",
        help="History compaction strategy: dynamic (RL-aligned default) or base",
    )
    parser.add_argument(
        "--compaction-max-tokens",
        type=int,
        default=None,
        help="Token trigger for history compaction (default: 48000 for dynamic, aligned with RLinf)",
    )
    parser.add_argument("--keep-messages", type=int, default=10)
    parser.add_argument("--keep-recent-images", type=int, default=4)
    parser.add_argument("--keep-recent-thinking", type=int, default=4)
    parser.add_argument(
        "--preserve-opening-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve step-0 opening images during dynamic compaction (default: true)",
    )
    parser.add_argument(
        "--compaction-tokenizer",
        type=Path,
        default=None,
        help="Optional HF tokenizer dir (e.g. Qwen3.5-4B) for exact Qwen token compaction counting",
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-llm-retries", type=int, default=8)
    parser.add_argument("--max-infra-retries", type=int, default=3)
    parser.add_argument("--record-artifacts", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_urls = parse_model_urls(args.model_url, args.model_urls_file)
    api_key = args.api_key or (
        os.environ.get(args.api_key_env) if args.api_key_env else None
    )
    resolved_provider = detect_provider(
        args.provider,
        model_name=args.model_name,
        model_urls=model_urls,
        api_key=api_key,
    )
    shard_idx, num_shards = parse_shard_spec(args.shard)

    rows = [
        json.loads(line)
        for line in args.tasks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schedule = build_episode_schedule(
        rows,
        interleave_tasks=args.interleave_tasks,
        shard_index=shard_idx,
        num_shards=num_shards,
        target_count=args.target_count,
        limit=args.limit,
        pass_offset=args.pass_offset,
    )

    compaction_max_tokens = args.compaction_max_tokens
    compaction_max_messages: int | None = 80
    if args.context_management == "dynamic":
        compaction_max_messages = None
        if compaction_max_tokens is None:
            compaction_max_tokens = 48000
    compaction = CompactionConfig(
        strategy=args.context_management,
        max_messages=compaction_max_messages,
        keep_messages=args.keep_messages,
        context_window=args.context_window,
        max_tokens=compaction_max_tokens,
        keep_recent_images=args.keep_recent_images,
        keep_recent_thinking=args.keep_recent_thinking,
        preserve_opening_images=args.preserve_opening_images,
    )

    token_counter = (
        CompactionTokenCounter(args.compaction_tokenizer)
        if args.compaction_tokenizer is not None
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = (
        _load_completed_episodes(args.output_dir, save_trajectory=args.save_trajectory)
        if args.resume
        else {}
    )
    pending = [item for item in schedule if item["episode_key"] not in completed]

    logger.info(
        "Rollout starting: model=%s provider=%s endpoints=%d sim=%s concurrency=%d "
        "scheduled=%d resumed=%d pending=%d shard=%d/%d compaction=%s(max_tok=%s,ctx=%d,tok_counter=%s)",
        args.model_name,
        resolved_provider,
        len(model_urls),
        args.sim_url,
        args.concurrency,
        len(schedule),
        len(schedule) - len(pending),
        len(pending),
        shard_idx,
        num_shards,
        compaction.strategy,
        compaction.token_trigger,
        args.context_window,
        bool(token_counter),
    )

    pool = EndpointPool(model_urls)
    manifest_path = args.output_dir / "manifest.jsonl"
    infra_err_path = args.output_dir / "infra_errors.jsonl"
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    stats = {
        "done": len(schedule) - len(pending),
        "solved": sum(
            1
            for item in schedule
            if item["episode_key"] in completed
            and completed[item["episode_key"]].get("success")
        ),
        "infra_errors": 0,
        "turns_sum": 0,
        "new_done": 0,
        "started_at": time.perf_counter(),
    }

    async def _run_all() -> list[dict[str, Any]]:
        limits = httpx.Limits(
            max_connections=max(64, args.concurrency * 4),
            max_keepalive_connections=max(32, args.concurrency * 2),
            keepalive_expiry=30.0,
        )
        async with httpx.AsyncClient(
            timeout=args.request_timeout_s,
            trust_env=False,
            limits=limits,
        ) as client:

            async def _one(item: dict[str, Any]) -> dict[str, Any] | None:
                ep_key = item["episode_key"]
                ep_dir = args.output_dir / "episodes" / ep_key
                async with sem:
                    chosen_url = pool.acquire()
                    active_url = chosen_url
                    try:
                        outcome, active_url = await run_episode(
                            model_url=chosen_url,
                            model_name=args.model_name,
                            sim_url=args.sim_url,
                            task=item["task"],
                            max_turns=args.max_turns,
                            max_tokens=args.max_tokens,
                            context_window=args.context_window,
                            max_episode_steps=args.max_episode_steps,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            thinking=args.thinking,
                            reasoning_effort=args.reasoning_effort,
                            compaction=compaction,
                            token_counter=token_counter,
                            image_size=args.image_size,
                            max_actions_per_call=args.max_actions_per_call,
                            include_jaw_mm=args.include_jaw_mm,
                            record_artifacts=args.record_artifacts,
                            output_dir=ep_dir,
                            request_timeout_s=args.request_timeout_s,
                            api_key=api_key,
                            provider=resolved_provider,
                            max_llm_retries=args.max_llm_retries,
                            max_infra_retries=args.max_infra_retries,
                            save_trajectory=args.save_trajectory,
                            episode_key=ep_key,
                            query_id=item["query_id"],
                            pass_idx=item["pass_idx"],
                            client=client,
                            endpoint_pool=pool,
                        )
                    except Exception as exc:
                        err_rec = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "episode_key": ep_key,
                            "query_id": item["query_id"],
                            "endpoint": active_url,
                            "error_type": type(exc).__name__,
                            "error_kind": getattr(exc, "kind", None),
                            "error": str(exc),
                        }
                        async with write_lock:
                            stats["infra_errors"] += 1
                            with infra_err_path.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                        logger.error(
                            "Episode %s failed with infra error (%s: %s); skipped so --resume can retry",
                            ep_key,
                            type(exc).__name__,
                            exc,
                        )
                        return None
                    finally:
                        pool.release(active_url)

                manifest_rec = {
                    "episode_key": ep_key,
                    "query_id": item["query_id"],
                    "pass_idx": item["pass_idx"],
                    "environment": outcome["environment"],
                    "success": bool(outcome["success"]),
                    "turns": int(outcome["turns"]),
                    "num_chains": int(outcome.get("num_chains", 1)),
                    "compaction_count": int(outcome.get("compaction_count", 0)),
                    "termination_reason": outcome["termination_reason"],
                    "error_kind": outcome["error_kind"],
                    "elapsed_s": outcome["elapsed_s"],
                    "trajectory_path": (
                        f"episodes/{ep_key}/trajectory.json"
                        if args.save_trajectory
                        else None
                    ),
                }
                async with write_lock:
                    with manifest_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(manifest_rec, ensure_ascii=False) + "\n")
                    completed[ep_key] = manifest_rec
                    stats["done"] += 1
                    stats["new_done"] += 1
                    stats["turns_sum"] += int(outcome["turns"])
                    if outcome["success"]:
                        stats["solved"] += 1
                    if stats["new_done"] % 10 == 0 or stats["done"] == len(schedule):
                        elapsed = max(1e-6, time.perf_counter() - stats["started_at"])
                        rate_hr = 3600.0 * stats["new_done"] / elapsed
                        logger.info(
                            "Progress %d/%d (solved=%d %.1f%%, avg_turns=%.1f, "
                            "throughput=%.1f ep/hr, infra_errors=%d)",
                            stats["done"],
                            len(schedule),
                            stats["solved"],
                            100.0 * stats["solved"] / max(stats["done"], 1),
                            stats["turns_sum"] / max(stats["new_done"], 1),
                            rate_hr,
                            stats["infra_errors"],
                        )
                return outcome

            raw_results = await asyncio.gather(*(_one(item) for item in pending))
            return [r for r in raw_results if r is not None]

    new_results = asyncio.run(_run_all())
    all_records = [
        completed[item["episode_key"]]
        for item in schedule
        if item["episode_key"] in completed
    ]
    solved = sum(bool(r.get("success")) for r in all_records)
    summary = {
        "model_name": args.model_name,
        "provider": resolved_provider,
        "scheduled": len(schedule),
        "episodes_completed": len(all_records),
        "episodes_new": len(new_results),
        "infra_errors": stats["infra_errors"],
        "solved": solved,
        "success_rate": solved / max(len(all_records), 1),
        "results": all_records,
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    logger.info(
        "Rollout complete: %d/%d completed (%d solved, %.1f%%, %d infra errors)",
        len(all_records),
        len(schedule),
        solved,
        100.0 * summary["success_rate"],
        stats["infra_errors"],
    )


if __name__ == "__main__":
    main()
