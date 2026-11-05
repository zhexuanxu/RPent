# Shared adaptive robot workflows

[简体中文](README_zh.md)

Use `--robot rl --simulator libero|robocasa|robotwin --workflow multiturn|zap`.
The default is LIBERO with the existing adaptive multiturn interface. No VLA,
task recipe or implicit file memory is loaded. `robots/libero_rl` has moved to
`robots/rl`; `tools/libero_rl_eval` has moved to `tools/rl`.

Run adaptive LIBERO-PRO collection with a local Qwen policy and shared SAM3.
The simulator and policy server use separate Python environments. The setup
below works from a source checkout and does not require a cluster-specific
activation script. RLinf and ms-swift provide their own training launchers.

## Install the environments

The commands below install LIBERO, the planner and SAM3. New simulators run in
separate processes and environments so their NumPy, SciPy and Torch pins do not
conflict. The policy server is shared by all three simulators.

Use Linux x86_64, Python 3.12, Git and [uv](https://docs.astral.sh/uv/).
The simulator needs NVIDIA EGL libraries; the model server needs a working
driver compatible with its CUDA 13 wheels. Install the NVIDIA driver through
your host/platform administrator. On Ubuntu, the remaining system packages are:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libegl1 libopengl0 libgl1 libglib2.0-0
# Run from the root of the checkout containing robots/rl.
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
source tools/rl/env.sh
```

Use a checkout containing `robots/rl`; the older `rpent-ppo` checkout uses
the `robots/libero_rl` module name. The LIBERO requirement file selects
`zhexuanxu/RLinf:rpent-ppo` for its supported environment runtime.
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

### RoboCasa and RoboTwin simulator processes

From this checkout, install only the native simulator dependencies; neither
RLDX nor LingBot weights are needed. RoboTwin requires a CUDA toolkit with
`nvcc`, a C++ compiler, and its RLinf Python environment adapter:

```bash
uv venv --python 3.11 .venv-robocasa
uv pip install --python .venv-robocasa/bin/python -r requirements/rl-robocasa.txt
uv venv --python 3.11 .venv-robotwin
uv pip install --python .venv-robotwin/bin/python torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-robotwin/bin/python -r requirements/rl-robotwin.txt
RLINF_ROOT=/path/to/RLinf .venv-robotwin/bin/python - <<'PYRLINF'
import os
import sysconfig
from pathlib import Path
root = Path(os.environ["RLINF_ROOT"]).resolve()
assert (root / "rlinf/envs/sim/robotwin/robotwin_env.py").is_file()
Path(sysconfig.get_path("purelib"), "rpent_robotwin_rlinf_source.pth").write_text(str(root) + "\n")
PYRLINF
uv pip install --python .venv-robotwin/bin/python ninja wheel "setuptools>=69.5.1,<75.9"
uv pip install --python .venv-robotwin/bin/python --no-build-isolation 'git+https://github.com/NVlabs/curobo.git@v0.7.8'
```

RoboCasa365 needs the pinned robosuite development revision in
`requirements/rl-robocasa.txt`: the PyPI 1.5.2 release lacks its
`load_model_on_init` interface. Keep it isolated from LIBERO's robosuite.

Install NVIDIA Vulkan as well as EGL for RoboTwin. If the driver libraries are
outside system search paths, set `LD_LIBRARY_PATH` and `VK_ICD_FILENAMES` to
the matching driver distribution before starting the simulator. Verify the
actual GPU allocation in a native smoke test; CUDA and Vulkan enumeration
can differ. Activate the planner environment for rollout commands, and select
each simulator's interpreter explicitly:

```bash
export RPENT_ROBOCASA_PYTHON="$PWD/.venv-robocasa/bin/python"
export RPENT_ROBOTWIN_PYTHON="$PWD/.venv-robotwin/bin/python"
```

For RoboTwin assets, run its versioned downloader in the simulator environment:

```bash
.venv-robotwin/bin/robotwin-download-assets --output /path/to/data/robotwin
export ROBOTWIN_ASSETS_PATH=/path/to/data/robotwin
```

The path is the runtime root containing `.robotwin-assets.json` and `assets/`,
not the nested `assets/` directory. The pinned runtime validates the snapshot.
`demo_randomized` is the default task configuration. The 100-episode manifest
uses the first published valid seed for each RoboTwin task, not a single
shared seed that might be invalid for some tasks.

For RoboCasa, the installed package owns `robocasa/models/assets`. To put
these assets on shared storage, copy its bundled files there and replace that
environment's asset directory with a symlink **before** downloading:

```bash
export ROBOCASA_ASSETS_PATH=/path/to/data/robocasa/assets
.venv-robocasa/bin/python - <<'PY'
import importlib.util, os, shutil
from pathlib import Path
source = Path(importlib.util.find_spec('robocasa').origin).parent / 'models/assets'
target = Path(os.environ['ROBOCASA_ASSETS_PATH']).expanduser().resolve()
if source.resolve() != target:
    shutil.copytree(source, target, dirs_exist_ok=True)
    source.rename(source.with_name('assets.bundled'))
    source.symlink_to(target, target_is_directory=True)
PY
.venv-robocasa/bin/python -m robocasa.scripts.download_kitchen_assets --type all
```

The native downloader asks before downloading. No demonstration dataset is
required. Keep the existing bundled scene/configuration files when relocating
assets. `ROBOCASA_ASSETS_PATH` alone does not redirect this package version.

## Shared tools and snapshot workflow

The ZAP option is an experimental harness inspired by
[the paper's interface and explicit todo](https://arxiv.org/html/2610.35817v3#A1).
It uses the original served model; it does not reproduce the paper's trained
SFT/RL policy. Independent single-request snapshots are our design choice,
not a claim about the unpublished implementation. Compare native success and
matched-task elapsed time before selecting it for collection.

All environments use measured state and native success. LIBERO's multiturn
prompt, six tools, camera layout and control/compression behavior are retained.
RoboCasa adds `move_base(vx, vy, yaw_rate, duration_s)` with local-frame velocity
targets (m/s, rad/s) followed by a stop. It returns measured displacement; native
friction or contact can prevent small commands from producing the target speed. RoboTwin adds an `arm` selector to motion/gripper
tools and optional paired targets for synchronous execution. Its visible TCPs
are labelled LEFT (cyan) and RIGHT (yellow), using measured poses and camera
calibration; anatomical left/right need not match image-left/right. New native
pose tools use world metres and quaternion `[w,x,y,z]`.

`--workflow zap` uses a small direct request loop over the existing model/wire
adapter, without an Agent/Harness conversation or compaction. Every request
contains the current task, robot description, fresh images, exact-frame SAM3
boxes and surface/box back-projections, measured state, current todo and last
tool result. Old reasoning, images and tool history are not replayed. The first
request can only call `change_query`; the next can only call `update_todo`.
Subsequent requests execute exactly one valid tool call; a multi-call reply
executes nothing and consumes a request from the same budget.

The ZAP tool set is `move_to`, `set_gripper`, `change_query`, `update_todo`,
`finish`, plus `move_base` on mobile robots. There is no `view_env`, `segment`
or `back_project` call. Queries and the todo are bounded explicit replacements;
they never accumulate a transcript. Task prompt files live under
`robots/rl/prompts/tasks/<env>/`; their default `{{task_language}}` inserts the
native task instruction without a task-specific solution.

Examples, after starting the model and shared SAM3 as described below:

```bash
bash tools/rl/run.sh --simulator robocasa --task-name OpenDrawer --split target --seed 1 --cuda-device 1
bash tools/rl/run.sh --simulator robotwin --task-name adjust_bottle --task-config demo_randomized --seed 100000 --cuda-device 1
bash tools/rl/run.sh --simulator libero --workflow zap --suite libero_object --task 0 --seed 1 --cuda-device 1
python tools/rl/run_batch.py run /path/to/logs/new-100 \
  --jobs tools/rl/jobs/robocasa-robotwin100.json --gpus 0,1 --per-gpu 50 \
  --start-interval 0 --max-initializing-per-gpu 8 \
  --endpoints http://127.0.0.1:30080/v1,http://127.0.0.1:30081/v1,http://127.0.0.1:30082/v1
```

The JSON manifest contains 50 RoboCasa Target50 tasks at seed 1 and 50
RoboTwin tasks. Each GPU hosts 25 of each type. `--per-gpu 50` admits 100
episodes; `--max-initializing-per-gpu 8` limits cold starts, not steady episode
concurrency. Size these values from real native memory/capacity checks.
ZAP development uses `jobs/libero-dev10.json`; the final fixed panel is
`jobs/panel40.txt`. Use fresh batch directories for different prompts or
workflows. The manifest pins production source, prompts and planner versions.
Complete request/response records remain in `api_trace/`; ZAP additionally
writes per-request snapshot/timing metadata under `snapshots/`.

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
source tools/rl/env.sh
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
MODEL_PATH="$HOME/.local/share/rpent/models/Qwen3.8-27B" SERVE_GPUS=2,3 bash tools/rl/serve.sh
```

Set `MODEL_PATH` to the actual downloaded directory when using another data
root. `curl http://127.0.0.1:30080/v1/models` lists the model when ready.
In a second terminal, start shared SAM3 on GPU 0:

```bash
source .venv/bin/activate
source tools/rl/env.sh
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112
```

In a third terminal, run the simulator on GPU 1:

```bash
source .venv/bin/activate
source tools/rl/env.sh
export RPENT_MODEL=openai-chat:qwen3.8-27b
export RPENT_BASE_URL=http://127.0.0.1:30080/v1
export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
bash tools/rl/run.sh --suite libero_object --task 0 --seed 0 --cuda-device 1
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
| `RPENT_OUTPUT_DIR` | `logs/rl-adaptive-<prompt>-<time>` | Episode output directory |
| `--suite`, `--task`, `--seed` | Suite/task required; seed 0 | LIBERO task and initialization seed |
| `--cuda-device` | Unset | Simulator GPU, independent of the shared SAM3 GPU |

Thinking sampling is temperature 1.0, top-p 0.95, top-k 20. To reproduce the
SFT-checkpoint evaluation penalties, explicitly export
`RPENT_PRESENCE_PENALTY=1.5` and `RPENT_REPETITION_PENALTY=1.0`.
`--observation-mode adaptive` remains accepted; image/text modes are rejected.
Other arguments to `run.sh` pass through to `rpent`.

## Batch collection, saving and recovery

Use the model/SAM3 exports above, then launch one resumable queue:

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/new-run" --jobs tools/rl/jobs/panel40.txt --seeds 3-66 --gpus 1 --per-gpu 4
```

A jobs line is `adaptive libero_object 0 0`. `panel40.txt` contains task 0–9
from each of `libero_object`, `libero_goal`, `libero_spatial`, `libero_10`.
Choose environment concurrency for the available simulator memory and startup
load; the model-service concurrency is a separate limit.

| Batch argument | Default | Meaning |
| --- | --- | --- |
| `--jobs` | Required | JSON tasks or the four-field LIBERO manifest |
| `--seeds` | Manifest seeds | Comma-separated seeds/ranges, e.g. `0,3-66` |
| `--gpus`, `--per-gpu` | GPUs required; 1 worker/GPU | Environment GPUs and worker count per GPU |
| `--prompt` | `simple` | One prompt variant for the batch |
| `--workflow` | `multiturn` | Conversation or independent ZAP snapshots |
| `--max-initializing-per-gpu` | `8` | Concurrent cold starts, separate from episode slots |
| `--endpoints` | `RPENT_BASE_URL` | Comma-separated policy `/v1` endpoints |
| `--start-interval` | `1` second | Spacing between initial worker starts |
| `--episode-timeout` | `14400` seconds | Outer process timeout per episode |
| `--infrastructure-retries` | `1` | Additional attempts after an error/audit failure per invocation |

Each job keeps independent `attempt-NNN` directories. The audit checks native
success, complete HTTP/native responses, tools and PNG SHA-256 before writing
`COMPLETED.json` and atomically updating `valid-index.json`. A normally completed
unsuccessful task remains valid data. Invalid attempts retain `INVALID.json`;
stopped attempts retain `INTERRUPTED.json`.

Normal output/context-limit and episode-timeout failures are sealed as
`EVALUATED.json`, counted once in `outcomes-index.json`, and excluded from the
training index. They are not retried as infrastructure failures. Interrupted
or corrupt attempts remain separate; an incomplete response is never labelled
a complete training output. `valid-index.json` contains only audited records.

SIGINT/SIGTERM stop owned episode process groups and preserve attempts. Repeat
the identical command to resume; sealed episodes are re-audited and skipped.
A directory lock prevents duplicate collectors, and the manifest rejects changed
tasks, model, prompt or generation settings. The read-only summary re-audits
sealed episodes without overwriting a live collector's index:

```bash
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/new-run"
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

An RLinf configuration using this checkout must point its episode module to
`robots.rl.rl_episode`; a configuration pointing to `robots.libero_rl.rl_episode`
belongs to the older checkout. Preserve the native task fields when migrating.

For trajectory encoding and multimodal SFT, see
`ms-swift/examples/rpent/README.md`. The saved post-compaction requests,
image bytes/order, visible historical thinking and assistant loss masks must
be preserved when preparing training data.
