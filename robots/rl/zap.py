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

"""One streamed tool decision per independent, bounded observation snapshot."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from pydantic_ai import BinaryContent, ThinkingPart, ToolCallPart
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from rpent.planner.base import Planner, PlannerResult, cancel_and_wait, execute_tool
from rpent.planner.request_trace import write_json


class ZapLoop(Planner):
    """Reuse the existing provider, sampling and wire recorder without a history loop."""

    def __init__(
        self, *, model, settings: dict, timeout_s: float, thinking, output_dir: Path
    ) -> None:
        self.model, self.settings = model, settings
        self.timeout_s, self.thinking = timeout_s, thinking
        self.output_dir = output_dir

    @classmethod
    def from_api(cls, planner, *, output_dir: Path) -> ZapLoop:
        """Keep the constructed API model and generation configuration unchanged."""
        from rpent.planner.api_loop import ApiAgentLoop, _build_model_settings

        if not isinstance(planner, ApiAgentLoop):
            raise ValueError("zap requires an API planner")
        return cls(
            model=planner.model,
            settings=planner.model_settings
            or _build_model_settings(
                planner.model,
                planner.max_tokens,
                planner.reasoning_effort,
                thinking=planner.thinking,
                parallel_tool_calls=False,
            ),
            timeout_s=planner.timeout_s,
            thinking=planner._thinking_level(),
            output_dir=output_dir,
        )

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction=None,
    ) -> PlannerResult:
        """Run the snapshot episode on a new event loop."""
        if input_queue is not None or dashboard_interaction is not None:
            raise ValueError("zap is a non-interactive workflow")
        return asyncio.run(
            self.solve_async(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
            )
        )

    async def solve_async(
        self, *, system_prompt: str, user_message: str, toolkit, max_turns: int
    ) -> PlannerResult:
        """Stream complete replies, execute one call and retain all audit evidence."""
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        messages = []
        stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_input_tokens": 0,
            "turns_used": 0,
            "tool_calls": 0,
        }
        finish, error, error_kind = None, None, None
        started = time.perf_counter()

        async def episode():
            nonlocal finish, error, error_kind
            while stats["turns_used"] < max_turns:
                turn = stats["turns_used"] + 1
                observe_started = time.perf_counter()
                observation = await asyncio.to_thread(toolkit.observe_snapshot)
                observation_seconds = time.perf_counter() - observe_started
                observation.data["budget"] = {
                    "request": turn,
                    "requests_remaining_including_current": max_turns - turn + 1,
                    "seconds_remaining": max(
                        0.0, self.timeout_s - (time.perf_counter() - started)
                    ),
                }
                # Bootstrap is enforced at the tool-set boundary; invalid replies
                # never advance it and never execute a sibling action.
                required = (
                    "change_query"
                    if not toolkit.snapshot.queries
                    else "update_todo"
                    if not toolkit.snapshot.todo
                    else None
                )
                definitions = {
                    definition.name: definition
                    for definition in toolkit.list_tools()
                    if required is None or definition.name == required
                }
                text = f"{user_message}\n\n{json.dumps(observation.data, ensure_ascii=False, allow_nan=False)}"
                if required:
                    text += f"\nThis request must call {required}."
                request = [
                    ModelRequest(
                        parts=[
                            SystemPromptPart(system_prompt),
                            UserPromptPart(
                                [
                                    text,
                                    *[
                                        BinaryContent(
                                            data=image, media_type="image/png"
                                        )
                                        for image in observation.images
                                    ],
                                ]
                            ),
                        ]
                    )
                ]
                parameters = ModelRequestParameters(
                    function_tools=[
                        ToolDefinition(
                            name=definition.name,
                            description=definition.description,
                            parameters_json_schema=definition.input_schema,
                            strict=False,
                        )
                        for definition in definitions.values()
                    ],
                    allow_text_output=False,
                    thinking=self.thinking,
                )
                stats["turns_used"] += 1
                begin = time.perf_counter()
                async with self.model.request_stream(
                    request, self.settings, parameters
                ) as stream:
                    async for _event in stream:
                        pass
                    response = stream.get()
                trace = getattr(self.model, "request_trace", None)
                if trace is not None:
                    trace.response(response)
                stats["total_input_tokens"] += response.usage.input_tokens
                stats["total_output_tokens"] += response.usage.output_tokens
                stats["total_cached_input_tokens"] += response.usage.cache_read_tokens
                content, calls = [], []
                for part in response.parts:
                    if isinstance(part, (TextPart, ThinkingPart)):
                        content.append(
                            {
                                "type": "thinking"
                                if isinstance(part, ThinkingPart)
                                else "text",
                                "text": part.content,
                            }
                        )
                    elif isinstance(part, ToolCallPart):
                        calls.append(part)
                        content.append(
                            {
                                "type": "tool_use",
                                "id": part.tool_call_id,
                                "name": part.tool_name,
                                "input": part.args,
                            }
                        )
                messages.extend(
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": content},
                    ]
                )
                write_json(
                    self.output_dir / "snapshots" / f"{turn:04d}.json",
                    {
                        "turn": turn,
                        "snapshot": observation.data,
                        "observation_seconds": observation_seconds,
                        "model_seconds": time.perf_counter() - begin,
                        "elapsed_seconds": time.perf_counter() - started,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "finish_reason": response.finish_reason,
                        "state": response.state,
                    },
                )
                if response.state != "complete" or response.finish_reason == "length":
                    error, error_kind = (
                        "planner response truncated",
                        "response_truncated",
                    )
                    return
                if len(calls) != 1 or calls[0].tool_name not in definitions:
                    toolkit.snapshot.last_result = {
                        "error": "Emit exactly one available tool call; no action executed."
                    }
                    continue
                call = calls[0]
                try:
                    arguments = call.args_as_dict()
                    definitions[call.tool_name].args_schema.model_validate(
                        arguments, strict=True
                    )
                except (ValueError, TypeError) as exc:
                    toolkit.snapshot.last_result = {
                        "error": f"Invalid tool arguments; no action executed: {exc}"
                    }
                    continue
                stats["tool_calls"] += 1
                result = await execute_tool(toolkit, call.tool_name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "name": call.tool_name,
                        "tool_call_id": call.tool_call_id,
                        "content": result.to_text(),
                    }
                )
                if result.data.get("_finish"):
                    finish = result.data
                    return

        try:
            async with asyncio.timeout(self.timeout_s):
                await episode()
        except TimeoutError:
            await cancel_and_wait(toolkit.cancel_active_and_wait)
            error, error_kind = (
                f"planner timed out after {self.timeout_s:g}s",
                "episode_timeout",
            )
        except asyncio.CancelledError:
            await cancel_and_wait(toolkit.cancel_active_and_wait)
            raise
        except Exception as exc:
            error, error_kind = f"{type(exc).__name__}: {exc}", "model_error"
        return PlannerResult(
            finish_result=finish,
            messages=messages,
            stats=stats,
            error=error,
            error_kind=error_kind,
        )
