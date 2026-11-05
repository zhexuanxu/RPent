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

"""LIBERO RL robot spec, toolkit factory, and runtime hooks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero_rl.prompt_bundle import (
    OBSERVATION_MODES,
    PROMPT_VARIANTS,
    system_prompt,
    user_prompt,
)
from rpent.dashboard.events import DashboardEventSink
from rpent.evaluation import RunFinalizationContext, write_json_atomic
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import make_rpc_client, wait_for_endpoint

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient

SUPPORTED_PLANNERS = ("api", "claude_code")


def get_robot_spec() -> RobotSpec:
    """Return the LIBERO RL identity, prompts, and runner hooks."""
    return RobotSpec(
        name="libero_rl",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        # Every tool drives or observes the one arm, so calls run one at a time.
        sequential_tools=True,
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        validate_args=_validate_args,
        prepare_memory=_skip_memory,
        finalize_run=_finalize_run,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the adaptive LIBERO RL toolkit."""
    from robots.libero_rl.toolkit import LiberoRLToolkit

    return LiberoRLToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        # Required by Toolkit; this toolkit registers no file tools to read it.
        memory=MemoryManager(get_memory_dir("libero_rl")),
        observation_mode=config.prompt_vars["observation_mode"],
        output_dir=config.output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register the LIBERO RL task, observation, and runtime flags."""
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="adaptive",
        help="Adaptive observations: camera images are returned by view_env.",
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPT_VARIANTS,
        default="simple",
        help="System prompt variant: simple is the bare tool contract; base adds "
        "the robot, frames and execution; base_gripper adds finger geometry; "
        "base_gripper_camera adds camera geometry.",
    )
    parser.add_argument(
        "--suite", required=not use_dashboard, help="e.g. libero_object"
    )
    parser.add_argument("--task", type=int, required=not use_dashboard)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument(
        "--libero-type",
        default=None,
        choices=["standard", "pro", "plus"],
        help="LIBERO variant (LIBERO_TYPE when unset).",
    )
    parser.add_argument(
        "--env-endpoint",
        default=None,
        help="[protocol://]host:port of an existing env_server; spawned when unset.",
    )
    parser.add_argument(
        "--sam3-endpoint",
        default=None,
        help="[protocol://]host:port of an existing SAM3 server; spawned when unset.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device for the spawned servers.",
    )


def _validate_args(args: argparse.Namespace) -> None:
    """Reject options that would change the fixed rollout setup unnoticed."""
    if args.planner not in SUPPORTED_PLANNERS:
        raise ValueError(
            f"libero_rl supports --planner {' or '.join(SUPPORTED_PLANNERS)}; "
            f"{args.planner} cannot restrict its built-in tools"
        )
    if args.dashboard or args.tui:
        raise ValueError("libero_rl runs non-interactive rollouts")
    if args.memory_dir is not None or args.memory_repo is not None:
        raise ValueError("libero_rl loads no memory; remove --memory-dir/--memory-repo")
    if args.observation_mode != "adaptive":
        raise ValueError("libero_rl supports only adaptive observations")
    if args.no_images:
        raise ValueError("libero_rl requires images from view_env; remove --no-images")


def _skip_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Rollouts start without guides, memory, or recipes."""


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Derive the run identifiers and the planner-visible prompt variables."""
    if args.observation_mode != "adaptive":
        raise ValueError("libero_rl supports only adaptive observations")
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")
    cell = f"{args.suite}_t{args.task}_s{args.seed}_{args.observation_mode}"
    output_dir = args.output_dir or (
        get_repo_root() / "logs" / f"{datetime.now():%Y%m%d-%H:%M:%S}_{cell}"
    )
    return RunConfig(
        recipe_tag=cell.removeprefix("libero_"),
        output_dir=Path(output_dir),
        prompt_vars={"observation_mode": args.observation_mode, "prompt": args.prompt},
        task_desc={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "observation_mode": args.observation_mode,
            "prompt": args.prompt,
        },
    )


def _spawn_server(
    args: argparse.Namespace,
    output_dir: Path,
    name: str,
    script: Path,
    script_args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> tuple[ProcessDaemon, RpcClient]:
    log = output_dir / f"{name}.log"
    log.write_text("")
    cuda = (
        ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []
    )
    daemon = ProcessDaemon(
        name=name,
        cmd=[
            sys.executable,
            str(script),
            *script_args,
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--parent-watch",
            *cuda,
        ],
        env_overrides=env_overrides,
        log_path=str(log),
    )
    daemon.start()
    try:
        endpoint = wait_for_endpoint(daemon)
    except BaseException:
        daemon.stop()
        raise
    return daemon, make_rpc_client(endpoint)


def _start_env(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)
    from rpent.utils.config import get_libero_type

    # The LIBERO env server and client are shared with robots.libero.
    return _spawn_server(
        args,
        output_dir,
        "env_server",
        get_repo_root() / "robots" / "libero" / "env_server.py",
        [
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--seed",
            str(args.seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
        ],
        {
            "LIBERO_TYPE": args.libero_type or get_libero_type(),
            "MUJOCO_GL": "egl",
            "ROBOT_PLATFORM": "LIBERO",
        },
    )


def _start_sam3(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon | None, RpcClient]:
    if args.sam3_endpoint is not None:
        return None, make_rpc_client(args.sam3_endpoint)
    return _spawn_server(
        args,
        output_dir,
        "sam3_server",
        get_repo_root() / "rpent" / "robots" / "components" / "sam3_server.py",
        [],
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Start or connect the environment and SAM3 services."""
    from robots.libero.env_client import LiberoEnvClient
    from rpent.robots.components.sam3_client import Sam3Client

    starters = {"env": _start_env, "sam3": _start_sam3}
    connectors = {
        "env": lambda rpc: {
            "env": LiberoEnvClient(
                rpc,
                expected_meta={
                    "suite": args.suite,
                    "task": args.task,
                    "seed": args.seed,
                    "max_episode_steps": args.max_episode_steps,
                },
            )
        },
        "sam3": lambda rpc: {"sam3_client": Sam3Client(rpc)},
    }
    selected = set(starters) if components is None else set(components)
    if unknown := selected.difference(starters):
        raise ValueError(f"unknown libero_rl runtime components: {sorted(unknown)}")

    owned: dict[str, ProcessDaemon] = {}
    pending = {
        name: try_spawn_server(
            owned, dashboard_events, name, partial(starter, args, output_dir)
        )
        for name, starter in starters.items()
        if name in selected
    }
    runtime_kwargs: dict[str, Any] = {}
    for name, (daemon, rpc) in pending.items():
        runtime_kwargs.update(
            try_wait_server(
                owned,
                dashboard_events,
                name,
                rpc,
                daemon,
                300.0,
                post_fn=partial(connectors[name], rpc),
            )
        )
    return list(owned.values()), runtime_kwargs


def _finalize_run(context: RunFinalizationContext) -> Path:
    """Write the native success flag beside the rollout's observation counts."""
    rollout_path = context.output_dir / "rollout.json"
    rollout = json.loads(rollout_path.read_text()) if rollout_path.is_file() else {}
    return write_json_atomic(
        context.output_dir / "result.json",
        {
            "robot": context.robot_name,
            "task": dict(context.task_desc),
            "success": context.environment_success,
            "agent_error": context.agent_error,
            "elapsed_s": round(context.elapsed_s, 1),
            "planner": context.planner,
            "model": context.model,
            "max_turns": context.max_turns,
            "stats": dict(context.stats),
            **rollout,
        },
    )
