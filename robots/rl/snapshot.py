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

"""Explicit bounded plan and exact-frame perception for snapshot requests."""

from __future__ import annotations

import io
from typing import Annotated

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from robots.rl import geometry
from rpent.tools import ToolResult, tool

Query = Annotated[str, Field(min_length=1, max_length=120)]
TodoItem = Annotated[str, Field(min_length=1, max_length=320)]


class Snapshot:
    """Keep only the model's current queries, todo and last execution result."""

    def __init__(
        self, primitives, *, environment: str, task: str, max_episode_steps: int = 10000
    ) -> None:
        self.primitives = primitives
        self.environment = environment
        self.task = task
        self.max_episode_steps = max_episode_steps
        self.queries: list[str] = []
        self.todo: list[str] = []
        self.generation = 0
        self.last_result: dict | None = None
        self._cache_key: tuple[int, int] | None = None
        self._perception: tuple[list, list[bytes]] = ([], [])

    @tool(readonly=True)
    def change_query(
        self, queries: Annotated[list[Query], Field(min_length=1, max_length=8)]
    ) -> ToolResult:
        """Replace the perception targets with short visible-object descriptions. These queries run automatically on every current camera frame. Include destinations and relevant parts as well as manipulated objects. All matching instances are reported; a missing detection is not evidence of absence. This changes perception only and never moves the robot.

        Args:
            queries: Full replacement, 1 to 8 object or part descriptions; no coordinates or task scripts.
        """
        if any(not query.strip() for query in queries):
            return ToolResult(data={"error": "queries must not be blank"})
        self.queries = [query.strip() for query in queries]
        self.generation += 1
        return ToolResult(
            data={
                "status": "updated",
                "query_generation": self.generation,
                "queries": self.queries,
            }
        )

    @tool(readonly=True)
    def update_todo(
        self, items: Annotated[list[TodoItem], Field(min_length=1, max_length=8)]
    ) -> ToolResult:
        """Replace the entire persistent todo with a compact working plan. Earlier reasoning is absent from the next request: retain the goal, verified progress, immediate next subgoal and any uncertainty needed to continue. Separate observations from intentions; do not mark an intended action as completed. Prefer a few useful items over a motion transcript. This records your plan only; it does not execute it or determine task success.

        Args:
            items: Complete replacement plan, 1 to 8 items of at most 320 characters each.
        """
        if any(not item.strip() for item in items):
            return ToolResult(data={"error": "todo items must not be blank"})
        self.todo = [item.strip() for item in items]
        return ToolResult(data={"status": "updated", "todo": self.todo})

    def observe(self, robot_state: dict) -> ToolResult:
        """Use recorded RGB/depth from one step; never mix perception across steps."""
        primitive = self.primitives
        step = primitive._state.latest_step
        key = (step, self.generation)
        if key != self._cache_key:
            cameras = (
                ("agentview", "wrist")
                if self.environment == "libero"
                else primitive.cameras
            )
            detections, images = [], []
            for camera in cameras:
                image = primitive._state.load(f"{camera}.png", step=step)
                raw = primitive.images(step, [camera])[0]
                world = primitive._state.load(f"{camera}_world.npz", step=step)
                view = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
                draw = ImageDraw.Draw(view)
                groups = (
                    primitive._sam3.segment_queries(raw, self.queries)
                    if self.queries
                    else []
                )
                rows = []
                for query, matches in zip(self.queries, groups, strict=True):
                    for match in matches:
                        if not match.found or match.mask is None:
                            continue
                        if match.mask.shape != world.shape[:2]:
                            raise ValueError(
                                "SAM3 mask and recorded depth dimensions differ"
                            )
                        box = geometry.box_from_mask(match.mask)
                        r0, c0, r1, c1 = box
                        identifier = f"{camera}:{len(rows) + 1}"
                        rows.append(
                            {
                                "id": identifier,
                                "query": query,
                                "score": match.score,
                                "box_rc": box,
                                "surface": geometry.surface_geometry(
                                    geometry.valid_points(world[match.mask])
                                ),
                                "box_surface": geometry.surface_geometry(
                                    geometry.valid_points(world[r0:r1, c0:c1])
                                ),
                            }
                        )
                        draw.rectangle((c0, r0, c1 - 1, r1 - 1), outline="red", width=3)
                        draw.text((c0, max(0, r0 - 12)), identifier, fill="red")
                buffer = io.BytesIO()
                view.save(buffer, format="PNG")
                encoded = buffer.getvalue()
                primitive._save(
                    f"zap-q{self.generation}-{camera}.png", np.asarray(view), step
                )
                detections.append({"camera": camera, "step": step, "objects": rows})
                images.append(encoded)
            self._cache_key, self._perception = key, (detections, images)
        detections, images = self._perception
        return ToolResult(
            data={
                "task": self.task,
                "step": step,
                "state": robot_state,
                "env_steps": primitive.env_steps,
                "env_steps_remaining": max(
                    0, self.max_episode_steps - primitive.env_steps
                ),
                "query_generation": self.generation,
                "queries": self.queries,
                "perception": detections,
                "todo": self.todo,
                "last_result": self.last_result,
            },
            images=list(images),
        )
