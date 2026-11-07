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

"""Task failures count as evaluations without becoming valid training samples."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "rl_batch_outcomes", Path(__file__).parents[4] / "tools/rl/run_batch.py"
)
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)


@pytest.mark.parametrize(
    "kind", ["response_truncated", "context_limit", "tool_arguments", "episode_timeout"]
)
def test_normal_terminal_errors_are_not_retried_as_infrastructure(tmp_path, kind):
    (tmp_path / "transcript_api.json").write_text(json.dumps({"error_kind": kind}))
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"terminated": False}]})
    )
    result = batch.terminal_failure(tmp_path)
    assert result["success"] is False and result["training_usable"] is False
    assert result["termination_reason"] == kind


def test_infrastructure_failure_is_not_a_task_outcome(tmp_path):
    (tmp_path / "transcript_api.json").write_text(
        json.dumps({"error_kind": "model_error"})
    )
    assert batch.terminal_failure(tmp_path) is None


def test_deadline_without_final_result_is_retained_but_not_training_data(tmp_path):
    (tmp_path / "states.json").write_text(
        json.dumps({"steps": [{"terminated": False}]})
    )
    native = tmp_path / "api_trace/native"
    native.mkdir(parents=True)
    (native / "000001.json").write_text("{}")
    result = batch.terminal_failure(tmp_path, deadline=True, job={"seed": 1})
    assert result["task"] == {"seed": 1} and not result["training_usable"]


def test_simulator_startup_timeout_is_infrastructure_failure(tmp_path):
    assert batch.terminal_failure(tmp_path, deadline=True, job={"seed": 1}) is None


def test_resume_rejects_a_modified_evaluation(tmp_path):
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    transcript = attempt / "transcript_api.json"
    transcript.write_text("original response")
    (attempt / "EVALUATED.json").write_text(
        json.dumps(
            {
                "outcome": {"success": False, "training_usable": False},
                "hashes": {
                    transcript.name: hashlib.sha256(transcript.read_bytes()).hexdigest()
                },
            }
        )
    )
    assert batch.evaluated(tmp_path)["success"] is False
    transcript.write_text("damaged response")
    assert batch.evaluated(tmp_path) is None


@pytest.mark.parametrize("environment", ["robocasa", "robotwin"])
def test_zap_batch_rejects_native_jobs_before_writing_manifest(tmp_path, environment):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            [{"environment": environment, "task_name": "example_task", "seed": 1}]
        )
    )
    output = tmp_path / "batch"
    output.mkdir()
    args = argparse.Namespace(
        infrastructure_retries=1,
        episode_timeout=14400,
        start_interval=0,
        max_initializing_per_gpu=1,
        jobs=jobs,
        seeds=None,
        workflow="zap",
    )
    with pytest.raises(ValueError, match="does not support --workflow zap"):
        batch.collect(args, output)
    assert not (output / "manifest.json").exists()
