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

"""Shared adaptive tools with either conversation or snapshot observations."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from robots.rl.prompt_bundle import validate_workflow
from robots.rl.tools import LiberoRLPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolResult, iter_tools
from rpent.tools.toolkit import Toolkit


class RLToolkit(Toolkit):
    """Expose end-effector, perception, and finish tools.

    Multiturn returns images through view_env; ZAP attaches fresh views per request.
    """

    planner_builtin_tools = False

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        observation_mode: str,
        output_dir: Path | str,
        environment: str = "libero",
        workflow: str = "multiturn",
    ) -> None:
        if observation_mode != "adaptive":
            raise ValueError("rl supports only adaptive observations")
        validate_workflow(environment, workflow)
        state = EnvState(output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._images_sent = 0
        self._images_lock = threading.Lock()
        self.environment, self.workflow = environment, workflow
        primitive_type = LiberoRLPrimitives
        if environment != "libero":
            from robots.rl.backends import RoboCasaPrimitives, RoboTwinPrimitives

            primitive_type = {
                "robocasa": RoboCasaPrimitives,
                "robotwin": RoboTwinPrimitives,
            }[environment]
        self._primitives = primitive_type(
            state=state, check_cancelled=self.raise_if_cancelled, **runtime_kwargs
        )
        self._task = self._primitives.env.get_task_language()
        self._publish_step(self._primitives.record_observation())
        self._recorded_env_steps = 0
        self.snapshot = None
        if workflow == "zap":
            from robots.rl.snapshot import Snapshot

            self.snapshot = Snapshot(
                self._primitives,
                environment=environment,
                task=self._task,
                max_episode_steps=self._primitives.env._client.call("env.get_env_meta")[
                    "max_episode_steps"
                ],
            )
        for definition in iter_tools(self._primitives):
            if workflow == "zap" and definition.name in {
                "view_env",
                "segment",
                "back_project",
            }:
                continue
            self.add_tool(definition)
        if self.snapshot is not None:
            for definition in iter_tools(self.snapshot):
                self.add_tool(definition)

    def observe_snapshot(self) -> ToolResult:
        """Return a complete current observation without accumulated history."""
        if self.snapshot is None:
            raise ValueError("snapshot observations require --workflow zap")
        observation = self.snapshot.observe(self._robot_state())
        self._count_images(observation)
        return observation

    def _register_common_tools(self) -> None:
        """Register no file tools; rollouts observe only through robot tools."""

    def _robot_state(self) -> dict[str, Any]:
        if self.environment != "libero":
            return {
                **self._primitives.robot_state(),
                "terminated": self._primitives.native_success(),
                "truncated": self._primitives.env_steps
                >= self._primitives._max_episode_steps,
            }
        env = self._primitives.env
        return {
            **self._primitives.measure().state(),
            "terminated": env.terminated,
            "truncated": env.truncated,
        }

    def initial_observation(self) -> ToolResult:
        """Return the task and measured initial state for the opening message."""
        step = self._state.latest_step
        observation = ToolResult(
            data={
                "task": self._task,
                "observation_mode": "adaptive",
                "state": {"step": step, **self._robot_state()},
            },
        )
        self._count_images(observation)
        self._state.save(
            "planner_audit.json",
            {
                "observation_mode": "adaptive",
                "tools": [
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "input_schema": definition.input_schema,
                    }
                    for definition in self.list_tools()
                ],
                "initial_observation": observation.data,
                "initial_images": len(observation.images),
            },
            step=None,
        )
        return observation

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Run one tool and count the images returned to the planner."""
        result = super().execute_tool(name, input_dict)
        if self.snapshot is not None:
            self.snapshot.last_result = {
                "tool": name,
                "arguments": input_dict,
                "result": result.data,
            }
        self._count_images(result)
        return result

    def _count_images(self, result: ToolResult) -> None:
        with self._images_lock:
            self._images_sent += len(result.images)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Return the measured outcome of a robot action.

        A new observation is recorded only when the action stepped the
        environment; otherwise the latest step number is reused.
        """
        advanced = self._primitives.env_steps > self._recorded_env_steps
        step = self._state.latest_step + 1 if advanced else self._state.latest_step
        if "error" in result:
            data: dict[str, Any] = {"status": "error", "error": result["error"]}
        else:
            data = {"status": result["status"]}
            if "reason" in result:
                data["reason"] = result["reason"]
        if self.environment != "libero":
            # Native adapters also report per-arm errors and physical limits.
            data.update(result)
        data["step"] = step
        if advanced or data["status"] != "error":
            data.update(self._robot_state())
        if advanced:
            # Log the payload the planner receives, not the primitive's internals.
            self._primitives.record_observation(
                command=command, result=data, elapsed_s=elapsed_s
            )
            self._recorded_env_steps = self._primitives.env_steps
        return ToolResult(data=data)

    def solved(self) -> bool:
        """Return the environment's latched success flag."""
        if self.environment != "libero":
            return self._primitives.native_success()
        return self._primitives.env.terminated

    def close(self) -> None:
        """Save the episode video and the rollout's observation counts."""
        super().close()
        self._state.save("episode.mp4", self._primitives.frames, step=None, fps=20)
        self._state.save(
            "rollout.json",
            {
                "observation_mode": "adaptive",
                "workflow": self.workflow,
                "images_sent": self._images_sent,
                "env_steps": self._primitives.env_steps,
                "observations": self._state.latest_step + 1,
            },
            step=None,
        )
