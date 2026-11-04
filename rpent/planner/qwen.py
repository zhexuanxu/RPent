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

"""Qwen chat-completions adapter: thinking on every turn, tool images kept in the tool loop.

Qwen3.5 and Qwen3.8 are served through SGLang's OpenAI-compatible endpoint and
through RLinf's policy endpoint. Both render the model's own chat template, and
two of its rules decide whether earlier reasoning survives:

* Qwen3.5 keeps an assistant turn's ``<think>`` block only when the turn comes
  after the last ``user`` message whose text is not framed by
  ``<tool_response>`` tags. pydantic-ai delivers a tool's images as a trailing
  ``user`` message, so every image-returning tool would end the tool loop and
  drop all reasoning before it. :class:`QwenChatModel` frames that message.
* ``reasoning_effort: "none"`` makes SGLang render the prompt with
  ``enable_thinking=false``. :func:`qwen_model_settings` never sends it and
  states ``enable_thinking`` explicitly instead.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

#: Qwen's recommended thinking-mode sampling; also what a model's
#: ``generation_config.json`` supplies when it ships one.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"


def is_qwen_model(model_name: str) -> bool:
    """Whether a served model name denotes a Qwen checkpoint."""
    return model_name.split("/")[-1].lower().startswith("qwen")


def qwen_model_settings(
    *,
    max_tokens: int,
    reasoning_effort: str = "none",
    thinking: bool = True,
    parallel_tool_calls: bool = False,
) -> OpenAIChatModelSettings:
    """Request settings for a Qwen chat endpoint.

    ``reasoning_effort`` is sent only as a real tier; ``"none"`` leaves the
    field out because SGLang maps it to ``enable_thinking=false``. Thinking is
    stated explicitly through ``chat_template_kwargs``; ``preserve_thinking``
    is honoured by the Qwen3.8 template and ignored by Qwen3.5.
    Optional ``RPENT_PRESENCE_PENALTY`` and ``RPENT_REPETITION_PENALTY``
    environment overrides affect sampling only; absent overrides preserve defaults.
    """
    from pydantic_ai.profiles.openai import OPENAI_REASONING_EFFORT_MAP

    settings = OpenAIChatModelSettings(
        max_tokens=max_tokens,
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        parallel_tool_calls=parallel_tool_calls,
        extra_body={
            "top_k": SAMPLING["top_k"],
            "chat_template_kwargs": {
                "enable_thinking": thinking,
                "preserve_thinking": True,
            },
        },
    )
    for name, minimum in [("presence_penalty", -2.0), ("repetition_penalty", 0.0)]:
        variable = "RPENT_" + name.upper()
        raw = os.environ.get(variable)
        if raw is None:
            continue
        value = float(raw)
        valid = minimum <= value <= 2.0
        if name == "repetition_penalty":
            valid = 0.0 < value <= 2.0
        if not valid:
            raise ValueError(f"invalid {variable}: {raw}")
        if name == "presence_penalty":
            settings[name] = value
        else:
            settings["extra_body"][name] = value
    if reasoning_effort != "none":
        settings["openai_reasoning_effort"] = OPENAI_REASONING_EFFORT_MAP.get(
            reasoning_effort, reasoning_effort
        )
    return settings


def _carries_tool_files(message: ModelRequest) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        and part.model_response_str_and_user_content()[1]
        for part in message.parts
    )


def _as_tool_response(user_message: dict[str, Any]) -> dict[str, Any]:
    """Frame a user message so the Qwen template reads it as a tool response."""
    content = user_message["content"]
    if isinstance(content, str):
        framed: Any = f"{TOOL_RESPONSE_OPEN}{content}{TOOL_RESPONSE_CLOSE}"
    else:
        framed = [
            {"type": "text", "text": TOOL_RESPONSE_OPEN},
            *content,
            {"type": "text", "text": TOOL_RESPONSE_CLOSE},
        ]
    return {**user_message, "content": framed}


class QwenChatModel(OpenAIChatModel):
    """``OpenAIChatModel`` whose tool-returned images stay inside the tool loop.

    The trailing ``user`` message pydantic-ai builds from a tool's files is
    framed by ``<tool_response>`` tags, so the Qwen3.5 template does not take
    it for a new user query and keeps the reasoning of every earlier turn.
    """

    request_trace = None

    async def _completions_create(
        self, messages, stream, model_settings, model_request_parameters
    ):
        """Keep the post-compaction native request beside the exact HTTP payload."""
        trace = self.request_trace
        if trace is None:
            return await super()._completions_create(
                messages, stream, model_settings, model_request_parameters
            )
        from rpent.planner.request_trace import reset_native

        token = trace.native(messages, model_settings, model_request_parameters)
        try:
            return await super()._completions_create(
                messages, stream, model_settings, model_request_parameters
            )
        finally:
            reset_native(token)

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterator[Any]:
        mapped = [item async for item in super()._map_user_message(message)]
        if mapped and mapped[-1].get("role") == "user" and _carries_tool_files(message):
            mapped[-1] = _as_tool_response(mapped[-1])
        for item in mapped:
            yield item
