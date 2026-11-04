# LIBERO RL rollout

[简体中文](README_zh.md)

Run adaptive LIBERO-PRO collection with a local Qwen policy and shared SAM3.
The simulator and policy server use separate Python environments. The setup
below works from a source checkout and does not require a cluster-specific
activation script. RLinf and ms-swift provide their own training launchers.

## Install the environments

Use Linux x86_64, Python 3.12, Git and [uv](https://docs.astral.sh/uv/).
The simulator needs NVIDIA EGL libraries; the model server needs a working
driver compatible with its CUDA 13 wheels. Install the NVIDIA driver through
your host/platform administrator. On Ubuntu, the remaining system packages are:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libegl1 libopengl0 libgl1 libglib2.0-0
git clone --branch rpent-ppo https://github.com/zhexuanxu/RPent.git RPent
cd RPent
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
source tools/libero_rl_eval/env.sh
```

This recipe targets the published `zhexuanxu/RPent:rpent-ppo` and
`zhexuanxu/RLinf:rpent-ppo` feature branches. Upstream `main` does not yet
contain `robots/libero_rl`; after integration, use a checkout containing that
module and its matching RLinf runtime.
Run the requirements command from the checkout root. It installs RPent in
editable mode, the supported RLinf runtime, SAM3 and the LIBERO-PRO simulator;
this workflow does not need OpenPi/Pi0.5 or an action-model checkpoint. The
requirements use that matching RLinf branch and the public RPent SAM3/LIBERO
forks. They do not select the older `rlinf` extra pinned to `release/v0.4`.
Record their resolved revisions and `uv pip freeze` output with an experiment.

Create the model-serving environment in a **new terminal**, from the same
checkout. SGLang 0.5.23 pins its own Torch, Transformers and CUDA dependencies:

```bash
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install --prerelease=allow "sglang==0.5.23" "torchvision==0.29.2"
```

Do not inherit `LD_LIBRARY_PATH` or `PYTHONPATH` from an RL/SFT environment
when opening the serving terminal. Activating a venv changes Python/PATH; it
does not undo another environment's CUDA library overrides. An existing site
profile may be used explicitly in place of standard venv activation when that
host needs its own CUDA/EGL driver paths; none is sourced by these launchers.

| Component | RPent / simulator | Model serving |
| --- | --- | --- |
| Python | 3.12 | 3.12 |
| Torch / CUDA | 2.11.0 / 12.8 | 2.14.1 / 13.0 |
| Transformers | 4.57.6 | 5.19.0 |
| Policy server | HTTP client | SGLang 0.5.23, kernel 0.4.10 |
| Planner | pydantic-ai-slim 2.53.0, harness 0.53.0 | Qwen tool/reasoning parsers |
| Simulator | MuJoCo 3.3.0, robosuite 1.5.2, NumPy 1.26.4 | — |
| Segmentation | SAM3 0.1.0, torchvision 0.26.0 | — |

These are the compatible stack versions used for the existing rollout
validation. `requirements/libero-rl.txt` fixes the main simulator versions;
it is not a complete transitive lockfile. The serving wheels require their
matching CUDA driver even when an older CUDA toolkit is installed on the host.

## Prepare models and simulator assets

In the RPent environment, source `env.sh` before downloads and every rollout
session. To choose shared storage, export `RPENT_DATA_ROOT=/your/data/root`
**before** sourcing it. By default it uses `$XDG_DATA_HOME/rpent`, or
`$HOME/.local/share/rpent` when `XDG_DATA_HOME` is unset. Explicit
`SAM3_CHECKPOINT_PATH`, `LIBERO_CONFIG_PATH` and `LIBERO_PRO_ASSET_PATH`
take precedence over these defaults.

Download the complete Qwen directory, including its tokenizer, processor and
chat template. SAM3 access must be granted on its model page before login:

```bash
source .venv/bin/activate
source tools/libero_rl_eval/env.sh
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
hf download Qwen/Qwen3.8-27B --local-dir "$RPENT_DATA_ROOT/models/Qwen3.8-27B"
hf auth login
hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_PRO_ASSET_PATH"
env -u LIBERO_PRO_ASSET_PATH liberopro-download-assets --assets-dir "$LIBERO_PRO_ASSET_PATH" --skip-existing --no-cache
```

For a network that uses the HF mirror, set `HF_ENDPOINT=https://hf-mirror.com`
before these commands. Clear proxy variables only if your network requires
direct access. `hf download` replaces any dependency on a private `hfd.sh`.
The downloader's `LIBERO_PRO_ASSET_PATH` fallback selects an existing asset
tree, so the download command temporarily removes that variable.

BDDL files and initialization states ship with `rpent-liberopro`. Generate
the configuration using that installed package, while keeping the downloaded
assets in your data directory:

```bash
python - <<'PYCONFIG'
import os
from pathlib import Path
import yaml
import liberopro.liberopro as libero
config = libero.get_default_path_dict()
config["assets"] = str(Path(os.environ["LIBERO_PRO_ASSET_PATH"]).resolve())
config["datasets"] = str(Path(os.environ["RPENT_DATA_ROOT"]) / "libero-pro/datasets")
path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
path.write_text(yaml.safe_dump(config))
print(path.read_text())
PYCONFIG
```

Check that the printed assets, BDDL and initialization-state directories
exist. Keep the `benchmark_root` at the installed package path: RLinf checks
it when loading a simulator. No demonstration dataset download is needed for
these rollouts.

## Start services and one episode

Use separate terminals from the checkout root. First activate **SGLang** and
start Qwen3.8-27B on GPU 2–3 (TP2, MTP enabled):

```bash
source .venv-sglang/bin/activate
MODEL_PATH="$HOME/.local/share/rpent/models/Qwen3.8-27B" SERVE_GPUS=2,3 bash tools/libero_rl_eval/serve.sh
```

Set `MODEL_PATH` to the actual downloaded directory when using another data
root. `curl http://127.0.0.1:30080/v1/models` lists the model when ready.
In a second terminal, start shared SAM3 on GPU 0:

```bash
source .venv/bin/activate
source tools/libero_rl_eval/env.sh
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112
```

In a third terminal, run the simulator on GPU 1:

```bash
source .venv/bin/activate
source tools/libero_rl_eval/env.sh
export RPENT_MODEL=openai-chat:qwen3.8-27b
export RPENT_BASE_URL=http://127.0.0.1:30080/v1
export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
bash tools/libero_rl_eval/run-libero-rl.sh --suite libero_object --task 0 --seed 0 --cuda-device 1
```

Collection supports **adaptive only**: initial/action observations contain
measured state, and `view_env` returns requested camera images. It retains
multimodal tool results and historical thinking. Without `RPENT_SAM3_ENDPOINT`,
each episode starts its own SAM3 on its simulator GPU. Use a small
`RPENT_MAX_TURNS` budget for installation checks; benchmark collection must
keep its intended budgets.

## Important settings

`serve.sh` accepts additional SGLang arguments after its environment settings.
Its listener is local (`127.0.0.1`). The planner and server both default to a
262144-token context.

| Service variable | Default | Meaning |
| --- | --- | --- |
| `RPENT_SGLANG_PYTHON` | `python` | Interpreter from the active SGLang environment; does not activate CUDA libraries |
| `SERVE_GPUS` | `2,3` | Visible physical GPUs; their count sets TP |
| `MODEL_PATH` | Required | Local checkpoint directory |
| `MODEL_NAME` | `qwen3.8-27b` | Served name; match `RPENT_MODEL` after `openai-chat:` |
| `SERVE_PORT` | `30080` | Local model port |
| `SERVE_MEM_FRACTION` | `0.8` | SGLang static memory fraction |
| `SERVE_CONCURRENCY` | `8` | Maximum running model requests |
| `ENABLE_MTP` | `1` | EAGLE/MTP: 3 steps, top-k 1, 4 draft tokens; set 0 for a model without MTP weights |

| Episode variable / flag | Default | Meaning |
| --- | --- | --- |
| `RPENT_MODEL`, `RPENT_BASE_URL` | Required | Served model and its `/v1` endpoint |
| `RPENT_SAM3_ENDPOINT` | Unset | Existing shared SAM3 endpoint |
| `RPENT_PROMPT` | `simple` | `simple`, `base`, `base_gripper`, `base_gripper_camera` |
| `RPENT_REASONING_EFFORT` | `xhigh` for Qwen3.8; `none` otherwise | `none` omits the effort tier while thinking stays enabled |
| `RPENT_MAX_TURNS` | `100` | Model requests per episode |
| `RPENT_MAX_EPISODE_STEPS` | `10000` | Simulator-step budget |
| `RPENT_MAX_TOKENS` | `32768` | Output-token budget per request, including thinking |
| `RPENT_CONTEXT_WINDOW` | `262144` | Planner context budget |
| `RPENT_COMPACTION_FRACTION` | `0.7` | History-compaction threshold |
| `RPENT_PLANNER_TIMEOUT_S` | `7200` | Planner wall-clock budget; set explicitly for a different limit |
| `RPENT_PRESENCE_PENALTY`, `RPENT_REPETITION_PENALTY` | Unset | Optional explicit sampling overrides |
| `RPENT_OUTPUT_DIR` | `logs/libero-rl-adaptive-<prompt>-<time>` | Episode output directory |
| `--suite`, `--task`, `--seed` | Suite/task required; seed 0 | LIBERO task and initialization seed |
| `--cuda-device` | Unset | Simulator GPU, independent of the shared SAM3 GPU |

Thinking sampling is temperature 1.0, top-p 0.95, top-k 20. To reproduce the
SFT-checkpoint evaluation penalties, explicitly export
`RPENT_PRESENCE_PENALTY=1.5` and `RPENT_REPETITION_PENALTY=1.0`.
`--observation-mode adaptive` remains accepted; image/text modes are rejected.
Other arguments to `run-libero-rl.sh` pass through to `rpent`.

## Batch collection, saving and recovery

Use the model/SAM3 exports above, then launch one resumable queue:

```bash
python tools/libero_rl_eval/run_batch.py run "$RPENT_DATA_ROOT/rollouts/new-run" --jobs tools/libero_rl_eval/jobs/panel40.txt --seeds 3-66 --gpus 1 --per-gpu 4
```

A jobs line is `adaptive libero_object 0 0`. `panel40.txt` contains task 0–9
from each of `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`.
Choose environment concurrency for the available simulator memory and startup
load; the model-service concurrency is a separate limit.

| Batch argument | Default | Meaning |
| --- | --- | --- |
| `--jobs` | Required | Four-field task manifest |
| `--seeds` | Manifest seeds | Comma-separated seeds/ranges, e.g. `0,3-66` |
| `--gpus`, `--per-gpu` | GPUs required; 1 worker/GPU | Environment GPUs and worker count per GPU |
| `--prompt` | `simple` | One prompt variant for the batch |
| `--endpoints` | `RPENT_BASE_URL` | Comma-separated policy `/v1` endpoints |
| `--start-interval` | `1` second | Spacing between initial worker starts |
| `--episode-timeout` | `14400` seconds | Outer process timeout per episode |
| `--infrastructure-retries` | `1` | Additional attempts after an error/audit failure per invocation |

Each job keeps independent `attempt-NNN` directories. The audit checks native
success, complete HTTP/native responses, tools and PNG SHA-256 before writing
`COMPLETED.json` and atomically updating `valid-index.json`. A normally completed
unsuccessful task remains valid data. Invalid attempts retain `INVALID.json`;
stopped attempts retain `INTERRUPTED.json`.

SIGINT/SIGTERM stop owned episode process groups and preserve attempts. Repeat
the identical command to resume; sealed episodes are re-audited and skipped.
A directory lock prevents duplicate collectors, and the manifest rejects changed
tasks, model, prompt or generation settings. The read-only summary re-audits
sealed episodes without overwriting a live collector's index:

```bash
python tools/libero_rl_eval/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/new-run"
```

Episode artifacts include `result.json`, `states.json`, transcripts, video and
`api_trace/`. The trace retains actual post-compaction requests, original PNG
bytes, native messages, complete streaming responses and compaction events.
The shared reader is `rpent.evaluation.trajectory`; dataset encoding and SFT
belong to `ms-swift/examples/rpent`.

## Offline installation

Prepare separate **complete** wheelhouses for the two environments on a
networked Linux x86_64/Python 3.12 host. First follow the online installation
above. In each environment, build its requirements into its own wheelhouse:

```bash
source .venv/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-rpent --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements/libero-rl.txt setuptools wheel
source .venv-sglang/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-sglang "sglang==0.5.23" "torchvision==0.29.2"
```

Transfer the checkout, both wheelhouses, model directories and simulator assets
to the offline host. Match OS, CPU architecture, Python ABI and CUDA drivers;
extensions without a prebuilt wheel also need their build toolchain on the
preparation host. A few cached wheels are insufficient to reconstruct a full
environment. On the offline host, create fresh venvs and install each prepared
closure separately. `--no-deps` prevents Git URL metadata from contacting the
network; all dependency wheels must already be present.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --no-index --no-deps wheelhouse-rpent/*.whl
uv pip install --no-index --no-deps --no-build-isolation -e .
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install --no-index --no-deps wheelhouse-sglang/*.whl
```

The offline Python interpreter must be installed locally before `uv venv`.
Run `uv pip check` in each environment. Re-run the configuration-generation
step after relocating the simulator, since its YAML contains absolute paths.
With all assets present, export `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` in runtime terminals. Keep environment installation
separate from active training processes.

## RLinf and SFT

Use RLinf's `examples/agent/rpent/README.md` for its own environment and
`run_train.sh` configuration. Set `tools.rpent.python` to this RPent venv's
interpreter and pass the same LIBERO/SAM3 paths to its episode subprocesses.
Those subprocesses need the RPent CUDA/EGL environment; an RL environment's
`PYTHONPATH` or Torch/NVIDIA library paths cannot substitute for it.
SGLang serving remains in its separate environment.

For trajectory encoding and multimodal SFT, see
`ms-swift/examples/rpent/README.md`. The saved post-compaction requests,
image bytes/order, visible historical thinking and assistant loss masks must
be preserved when preparing training data.
