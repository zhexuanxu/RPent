#!/usr/bin/env python3
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

"""Collect audited episodes with bounded concurrency and resumable attempts."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from rpent.evaluation import write_json_atomic as atomic
from rpent.evaluation.trajectory import audit_episode

EVAL_DIR = Path(__file__).resolve().parent
PROMPTS = ("simple", "base", "base_gripper", "base_gripper_camera")
CONFIGS = ("adaptive",)
LOCK = threading.Lock()
STOP = threading.Event()


def seed_values(value: str) -> list[int]:
    """Parse distinct nonnegative seeds and inclusive ranges."""
    result = []
    for part in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise argparse.ArgumentTypeError("Expected seeds such as 0,3-66")
        bounds = part.split("-")
        start, end = int(bounds[0]), int(bounds[-1])
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError(
                "Seeds must be nonnegative, ordered ranges"
            )
        result.extend(range(start, end + 1))
    return list(dict.fromkeys(result))


def completed(output: Path) -> dict | None:
    """Return an audited sealed attempt, ignoring partial or damaged attempts."""
    for path in sorted(output.glob("attempt-*/COMPLETED.json")):
        try:
            result, _ = audit_episode(path.parent, verify_assets=True)
            return {
                "directory": str(path.parent.resolve()),
                "success": result["success"],
                "task": result["task"],
                "stats": result["stats"],
            }
        except (OSError, ValueError, KeyError, StopIteration):
            continue
    return None


def interrupt(signum: int, frame: object) -> None:
    """Ask workers to stop their own process groups."""
    STOP.set()


def stop_process(process: subprocess.Popen) -> None:
    """Terminate the owned process group, escalating if graceful close stalls."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(args: argparse.Namespace) -> None:
    """Collect the manifest without mixing configurations or repeating sealed jobs."""
    batch = Path(args.batch_dir).resolve()
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / ".collector.lock").open("a") as ownership:
        try:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A collector already owns {batch}") from error
        collect(args, batch)


def collect(args: argparse.Namespace, batch: Path) -> None:
    """Run one exclusively owned batch."""
    STOP.clear()
    if (
        args.infrastructure_retries < 0
        or args.episode_timeout <= 0
        or args.start_interval < 0
    ):
        raise ValueError(
            "Retry count and start interval must be nonnegative; timeout must be positive"
        )
    jobs = []
    for line in Path(args.jobs).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        mode, suite, task, seed = line.split()
        if mode not in CONFIGS:
            raise ValueError(f"Unknown observation mode: {mode}")
        if not re.fullmatch(r"[a-zA-Z0-9_]+", suite) or int(task) < 0 or int(seed) < 0:
            raise ValueError(f"Invalid task manifest entry: {line}")
        for item in args.seeds if args.seeds is not None else [int(seed)]:
            jobs.append((mode, suite, int(task), item))
    jobs = list(dict.fromkeys(jobs))
    identity = {
        "jobs": jobs,
        "prompt": args.prompt,
        "model": os.environ["RPENT_MODEL"],
        "parameters": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith("RPENT_")
            and k not in ("RPENT_BASE_URL", "RPENT_OUTPUT_DIR", "RPENT_SAM3_ENDPOINT")
        },
        "launcher_sha256": hashlib.sha256(
            (EVAL_DIR / "run-libero-rl.sh").read_bytes()
            + (EVAL_DIR / "planner.sh").read_bytes()
        ).hexdigest(),
    }
    # A directory cannot silently mix seeds, prompts or generation configurations.
    manifest = batch / "manifest.json"
    canonical = json.loads(json.dumps(identity))
    if manifest.exists() and json.loads(manifest.read_text()) != canonical:
        raise ValueError(
            "Resume configuration differs from manifest.json; use a new output directory"
        )
    atomic(manifest, canonical)
    endpoints = (
        args.endpoints.split(",") if args.endpoints else [os.environ["RPENT_BASE_URL"]]
    )
    slots = [gpu for gpu in args.gpus.split(",") for _ in range(args.per_gpu)]
    if not slots or args.per_gpu < 1:
        raise ValueError("At least one environment slot is required")
    pending: queue.Queue = queue.Queue()
    records = []
    for job in jobs:
        mode, suite, task, seed = job
        output = batch / f"{suite}_t{task}_s{seed}_{mode}-{args.prompt}"
        row = completed(output)
        if row:
            records.append(row)
        else:
            pending.put((job, output))
    atomic(batch / "valid-index.json", {"episodes": records})
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    started = time.time()

    def worker(index: int, gpu: str) -> None:
        if STOP.wait(index * args.start_interval):
            return
        while not STOP.is_set():
            try:
                (mode, suite, task, seed), output = pending.get_nowait()
            except queue.Empty:
                return
            output.mkdir(exist_ok=True)
            for retry in range(args.infrastructure_retries + 1):
                if STOP.is_set():
                    return
                number = (
                    max(
                        [int(p.name.split("-")[-1]) for p in output.glob("attempt-*")]
                        + [0]
                    )
                    + 1
                )
                attempt = output / f"attempt-{number:03d}"
                attempt.mkdir()
                command = [
                    "bash",
                    str(EVAL_DIR / "run-libero-rl.sh"),
                    "--suite",
                    suite,
                    "--task",
                    str(task),
                    "--seed",
                    str(seed),
                    "--cuda-device",
                    gpu,
                    "--observation-mode",
                    mode,
                    "--prompt",
                    args.prompt,
                ]
                environment = {
                    **os.environ,
                    "RPENT_OUTPUT_DIR": str(attempt),
                    "RPENT_BASE_URL": endpoints[index % len(endpoints)],
                }
                begin = time.time()
                with (attempt / "console.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    while process.poll() is None:
                        if STOP.is_set() or time.time() - begin >= args.episode_timeout:
                            stop_process(process)
                            break
                        time.sleep(0.5)
                    code = process.returncode
                status = {
                    "exit_code": code,
                    "elapsed_seconds": time.time() - begin,
                    "retry": retry,
                    "gpu": gpu,
                    "endpoint": environment["RPENT_BASE_URL"],
                }
                if STOP.is_set():
                    atomic(attempt / "INTERRUPTED.json", status)
                    return
                try:
                    if code:
                        raise ValueError(f"Worker exited {code}")
                    result, turns = audit_episode(attempt, verify_assets=True)
                    status.update(success=result["success"], turns=len(turns))
                    atomic(attempt / "COMPLETED.json", status)
                    row = completed(output)
                    if row is None:
                        raise ValueError("Sealed episode failed its second audit")
                    with LOCK:
                        records.append(row)
                        atomic(batch / "valid-index.json", {"episodes": records})
                    print(
                        f"Completed {output.name}: success={result['success']}",
                        flush=True,
                    )
                    break
                except (OSError, ValueError, KeyError, StopIteration) as error:
                    atomic(attempt / "INVALID.json", {**status, "error": str(error)})
                    print(f"Invalid {output.name}: {error}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, index, gpu) for index, gpu in enumerate(slots)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            STOP.set()
            raise
    atomic(
        batch / "state.json",
        {
            "status": "paused" if STOP.is_set() else "finished",
            "expected": len(jobs),
            "valid": len(records),
            "successes": sum(x["success"] for x in records),
            "elapsed_seconds": time.time() - started,
        },
    )
    if not STOP.is_set() and len(records) != len(jobs):
        raise RuntimeError("Some episodes failed audit; all attempts are retained")


def summary(args: argparse.Namespace) -> None:
    """Print audited sealed episodes without changing a running collector."""
    batch = Path(args.batch_dir).resolve()
    rows = [
        row
        for output in sorted(batch.iterdir())
        if output.is_dir()
        if (row := completed(output)) is not None
    ]
    report = {
        "valid": len(rows),
        "successes": sum(x["success"] for x in rows),
        "episodes": rows,
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    """Run the collection or summary command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("run")
    collection.add_argument("batch_dir")
    collection.add_argument("--jobs", required=True)
    collection.add_argument("--seeds", type=seed_values)
    collection.add_argument("--gpus", required=True)
    collection.add_argument("--per-gpu", type=int, default=1)
    collection.add_argument("--prompt", choices=PROMPTS, default="simple")
    collection.add_argument(
        "--endpoints", help="Comma-separated local OpenAI base URLs"
    )
    collection.add_argument("--infrastructure-retries", type=int, default=1)
    collection.add_argument("--episode-timeout", type=int, default=14400)
    collection.add_argument(
        "--start-interval",
        type=float,
        default=1,
        help="Seconds between initial worker starts",
    )
    report = commands.add_parser("summary")
    report.add_argument("batch_dir")
    args = parser.parse_args()
    {"run": run, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
