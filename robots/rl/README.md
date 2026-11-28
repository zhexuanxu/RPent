# RPent Unified RL Simulator Layer (`robots/rl`)

`robots/rl` provides a compact, uniform tool-calling interface (Harness v2), warm HTTP simulator pools, and standalone rollout/evaluation harness across all supported manipulation simulators (`libero`, `robotwin`, `robocasa`, `robodojo`, `metaworld`, `maniskill`, and `robolab`).

## Architecture

```
RLinf AgentLoop / Standalone Rollout
   │  HTTP (binary octet-stream /sessions/open & /execute, JSON /close & /pool/*)
   ▼
robots/rl/server.py  (FastAPI + SimPoolManager / SimPoolRouter over Unix-socket warm workers)
   ├── robots/rl/toolkit.py      (RLToolkit, ordered action-batch execution, observe=, jaw_mm)
   ├── robots/rl/perception.py   (RecordedPerception, SAM3 segmentation, RGB-D back-projection)
   ├── robots/rl/envs/
   │     ├── libero.py           (LocalLiberoEnv + LiberoRLPrimitives)
   │     ├── robotwin.py         (LocalRoboTwinEnv + RoboTwinRLPrimitives)
   │     ├── metaworld.py        (LocalMetaWorldEnv + MetaWorldRLPrimitives)
   │     ├── maniskill.py        (LocalManiSkillEnv, ManiSkillRLPrimitives)
   │     ├── robocasa.py         (RoboCasaRLPrimitives)
   │     ├── robodojo.py         (RoboDojoRLPrimitives)
   │     └── robolab.py          (RoboLabRLPrimitives)
   ├── robots/rl/conversation.py (EpisodeConversation, validation retries, base/dynamic compaction)
   ├── robots/rl/geometry.py     (camera projection, back-projection, PNG encode/overlay)
   ├── robots/rl/units.py        (centimetre/metre translation, per-env default resolutions)
   └── robots/rl/prompts/        (shared & per-simulator prompt templates)
```

## Key Design & Harness v2 Protocol Features

1. **Isolated per-service environments (`tools/rl/serve.py`)**:
   A single YAML file supervises the SAM3 GPU pool and one warm `SimPoolManager` per simulator, each running under its own isolated Python venv (`envs/sam3`, `envs/libero`, `envs/robotwin`, `envs/robodojo`, `envs/robolab`, `envs/metaworld`). Reference templates:
   - `tools/rl/sim_services.example.yaml` — single-node 8-GPU layout (GPUs 0–3 RL, GPUs 4–7 sims + SAM3).
   - `tools/rl/sim_services_5env_h100.example.yaml` — dedicated 8× 80 GB H100 5-Env + 8× SAM3 node (`144` concurrent slots) paired with 6-node 27B Async GRPO.
   - `tools/rl/sim_services_4env_sft_h100.example.yaml` — dedicated 8× 80 GB H100 4-Env + 8× SAM3 SFT collection node (`208` concurrent slots across `libero`, `robotwin`, `robolab`, `metaworld`).
   - `tools/rl/sim_services_48gpu.example.yaml` — multi-node `SimPoolRouter` + sleepable collocated pools layout.
2. **Full-concurrency warm worker pools with crash detection & retry**:
   Each `SimPoolManager` pre-warms `workers` processes pinned round-robin across configured GPUs, matches handshakes by explicit `--worker-id`, exposes `workers_target` / `workers_total` / `workers_idle` / `active_sessions` on `GET /health`, polls `proc.poll()` every `0.5s` during startup to catch transient simulator crashes immediately (retrying up to `3×`), and guards `_replace_worker()` with an idempotent `replaced` flag so concurrent cohort aborts never leak or double-respawn workers.
3. **On-demand / deferred simulator rendering (`record_artifacts=False`)**:
   - **LIBERO**: `LocalLiberoEnv(..., record_artifacts=False)` sets `use_camera_obs=False, camera_depths=False` during physics stepping and renders RGB-D on demand only when observations are requested.
   - **RoboTwin**: `LocalRoboTwinEnv(..., record_artifacts=False)` skips per-step offscreen rendering during `reset()` and `chunk_step()`, batching all 3 cameras into a single SAPIEN `scene.update_render()` pass.
   - **RoboLab / MetaWorld**: Captures views lazily on the first perception request at each step when `record_artifacts=False`.
   - Setting `record_artifacts=True` preserves full frame capture and MP4/rollout artifacts for evaluation.
4. **Zero-Base64 binary transport**:
   When a client sends `Accept: application/vnd.rpent.execute+octet-stream`, both `POST /sessions/open` and `POST /sessions/{sid}/execute` stream a length-prefixed binary frame (`[u32 meta_len][utf8 JSON meta][u32 num_images][u32 img_len][raw PNG bytes]...`), with Base64 JSON fallback when not requested. `EpisodeConversation` stores lightweight `rpent-image://<fid>` placeholders plus raw PNG `bytes` references without eager Base64 encoding or deep copies across turns.
5. **Harness v2 protocol across all 6 simulators (`libero`, `robocasa`, `robotwin`, `robodojo`, `robolab`, `metaworld`)**:
   - **Compact feedback (F1)**: `coordinates` and `position_unit` are emitted only on `initial_observation` and whenever `frame_version` changes (e.g., RoboCasa `move_base`/`move_lift`); `execute_actions` returns compact per-item outcomes and a single top-level `final_state` with uniform rounding (`xyz` 4dp m → 2dp cm, `orientation_rpy` 3dp rad, `opening` 3dp).
   - **Per-environment default display resolution (F2)**: When `image_size=None` (default), `libero` / `robocasa` / `metaworld` display `512×512` (preserving native `1024×1024` RGB-D for SAM3/depth), `robolab` displays `768×432` (preserving native `1280×720`), and `robotwin` (`320×240`) / `robodojo` (`640×480`) remain native.
   - **Optional physical `jaw_mm` (F3)**: `--include-jaw-mm` / `OpenRequest.include_jaw_mm` exposes measured inner-pad clearance in millimetres across all 6 simulators (`state["jaw_mm"]` for single-arm, `state[arm]["jaw_mm"]` for dual-arm; unaffected by centimetre scaling).
   - **Dynamic batch compaction (F4)**: `--context-management dynamic` triggers at the token high-water mark to strip older intermediate images (`[image omitted]`) and older assistant `reasoning_content` in one batch while preserving `messages[0]` (including step=0 opening images) and full action/tool text, falling back to `base` sliding window only if still over budget.
   - **Step-0 opening images & `execute_actions(..., observe=...)` (F5)**: `POST /sessions/open` streams `step=0` all-camera PNGs via binary framing through both `SimPoolManager` and `SimPoolRouter`; `execute_actions` accepts optional `observe` (`"none"`, `"all"`, or a declared camera name) to capture post-batch RGB in one turn while keeping `view_env` for motion-free inspection.

## Deploying Simulator & SAM3 Pools

Copy and edit one of the templates under `tools/rl/`, then run:

```bash
ulimit -n 1048576
PYTHONPATH=/path/to/RPent \
  python tools/rl/serve.py --config /path/to/sim_services.yaml
```

> **PYTHONPATH isolation rule**: Keep global `env.PYTHONPATH` in `serve.py` YAML restricted to `RPent` only, and scope `RLinf` exclusively under `simulators.libero.env.PYTHONPATH`. Putting workspace `RLinf` in global `PYTHONPATH` shadows RoboDojo's installed `rlinf.envs.sim.robodojo` package.

Verify pool health:

```bash
for port in 28561 28562 28563 28564 28565; do
  echo "=== :$port ===" && curl -s "http://127.0.0.1:${port}/health" && echo
done
```

### Dedicated 8× 80 GB H100 5-Env Node Sizing (Paired with 27B Async GRPO)

When running large-model (e.g., 27B) multi-node collocated async GRPO (6 RL nodes = 48 GPUs) paired with 1 dedicated 8× 80 GB H100 simulator + SAM3 node (`tools/rl/sim_services_5env_h100.example.yaml`), size the pools around IsaacSim's per-worker VRAM footprint:

| Service | Port | Workers | GPU Mapping | VRAM Profile |
| :--- | :---: | :---: | :--- | :--- |
| **SAM3 Pool** (`max_pending=2048`) | `28520` | `8` | `0..7` (1/GPU) | `~2.5 GiB` / GPU |
| **LIBERO** (`libero_type=pro`, EGL) | `28561` | `40` | `0..7` (5/GPU) | Lightweight EGL |
| **RoboTwin** (SAPIEN / Vulkan) | `28562` | `32` | `0..7` (4/GPU) | Lightweight Vulkan |
| **RoboDojo** (IsaacSim 5.0) | `28563` | `16` | `0..7` (2/GPU) | `~11.5–12.5 GiB` / worker |
| **RoboLab** (IsaacSim 5.0) | `28564` | `16` | `4..7, 0..3` (2/GPU, staggered) | `~11.5–12.5 GiB` / worker |
| **MetaWorld** (MuJoCo EGL) | `28565` | `40` | `0..7` (5/GPU) | Lightweight EGL |
| **Total** | | **144 sim + 8 SAM3** | **8× 80 GB H100** | **Peak `~58–65 GiB / 81.5 GiB` / GPU** (`~16–23 GiB` headroom) |

#### Pre-Flight Checklist for a Dedicated 5-Env Simulator Node

1. **8-GPU Simultaneous Vulkan Pre-Flight**: A single GPU with a wedged Vulkan context can pass CUDA/NCCL/EGL checks yet segfault IsaacSim with `VkResult: ERROR_DEVICE_LOST` (`EXIT=139`). Always verify simultaneous `SimulationApp` startup across all 8 GPUs before designating a node as the simulator host; assign any Vulkan-impaired node to pure CUDA RL training instead.
2. **Air-Gapped Offline IsaacSim Nucleus Alias (`omniverse.toml`)**: IsaacSim USD scenes (notably RoboDojo) embed S3 Nucleus URLs (`https://omniverse-content-production.s3-us-west-2.amazonaws.com/...`). Even when `--/persistent/isaac/asset_root/*` is set, USD asset references hang for `300s` in air-gapped clusters unless **node-local** `/root/.nvidia-omniverse/config/omniverse.toml` exists on every simulator node:
   ```toml
   [aliases]
   "https://omniverse-content-production.s3-us-west-2.amazonaws.com/" = "file:///path/to/rpent_ppo/assets/isaac_nucleus/"
   "http://omniverse-content-production.s3-us-west-2.amazonaws.com/" = "file:///path/to/rpent_ppo/assets/isaac_nucleus/"
   ```
   Stage `omniverse.toml` on shared storage and copy it to `/root/.nvidia-omniverse/config/omniverse.toml` on every node during cluster provisioning.
3. **Native Graphics Libraries (Never Mix CUDA Forward-Compat)**: Keep simulator `LD_LIBRARY_PATH` on native host libraries (`/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64`). Prepending `/usr/local/cuda/compat` (required for PyTorch/SGLang on `535.x` drivers) breaks Vulkan/EGL inside simulator processes.

### Simulator-Specific Setup Notes

#### MetaWorld

Install `requirements/rl-metaworld.txt` in an isolated Python 3.11 environment and enable headless EGL rendering (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`):

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m robots.rl.server serve \
  --env metaworld \
  --workers 16 \
  --gpus 4 5 6 7 \
  --port 28565 \
  --sam3-endpoint http://127.0.0.1:28510
```

Open a session with an MT50 task name (e.g. `pick-place-v3`) and a native task index `0..49` in `seed`. The fixed-wrist Sawyer enforces a strict 4-DoF `move_to` schema (`xyz`-only, rejecting `orientation_rpy`) plus `set_gripper`, reports calibrated live `pose.orientation_rpy`, provides `corner2` and `gripperPOV` views, and clamps `max_episode_steps` to MetaWorld's native 500-step horizon.

#### ManiSkill

Install the native ManiSkill worker on Linux with CUDA and Vulkan:

```bash
python -m pip install -r requirements/rl-maniskill.txt
```

Set `MS_ASSET_DIR` to the installed ManiSkill assets and start the shared SAM3 service with its checkpoint. The worker uses this service for the `segment` tool. Enable the `maniskill` pool in `tools/rl/sim_services.example.yaml`, or start a single worker:

```bash
python -m robots.rl.server serve --env maniskill --workers 1 --gpus 0 \
  --port 28567 --sam3-endpoint http://127.0.0.1:28510
```

The 25 supported RGB-D tasks are listed in [the task catalog](envs/maniskill_tasks.json).

#### RoboLab DROID

Use Linux x86-64, Python 3.11, and a CUDA 12.8-compatible NVIDIA driver with Vulkan support (with Git LFS and FFmpeg installed):

```bash
git clone https://github.com/NVlabs/RoboLab.git /path/to/RoboLab
git -C /path/to/RoboLab checkout b6cfcf6ccd2113c4cd216d35b33024a7d5691da
git -C /path/to/RoboLab lfs pull
python3.11 -m venv /path/to/envs/robolab
/path/to/envs/robolab/bin/python -m pip install \
  -r requirements/rl-robolab.txt -e /path/to/RoboLab -e .
```

Install RoboLab in editable mode so tasks resolve assets beside the source. The [DROID task catalog](../robolab/task_catalog.json) lists 120 task IDs and native instructions (`"task": {"task_name": "BananaInBowlTask", "seed": 0}`). Export `OMNI_KIT_ACCEPT_EULA=Y` (exact `"Y"`) in the simulator environment. Each worker keeps one Isaac application warm and rebuilds a seeded USD scene per episode.

Integration test:

```bash
RPENT_RUN_ROBOLAB_INTEGRATION=1 RPENT_ROBOLAB_GPU=4 \
  /path/to/envs/robolab/bin/python -m pytest -s -q tests/e2e_tests/robolab
```

## Standalone Evaluation & SFT Data Collection

Run standalone episodes or large-scale resumable SFT trajectory collection against any OpenAI-compatible serving endpoint (local SGLang continuous-batching replicas or remote APIs) with the exact same Harness v2 conversation, dynamic compaction, and binary transport.

See the full teammate guide in **[`SFT_DATA_COLLECTION.md`](./SFT_DATA_COLLECTION.md)** for multi-node dual-teacher collection, interrupt/resume, sharding, `robots/rl/sft_export.py` (`ms-swift` / `Qwen3.5-4B` format verification), and `swift sft` training flags.

Quick example:

```bash
python -m robots.rl.rollout \
  --model-url http://127.0.0.1:30000/v1 \
  --model-name Qwen3.8-27B \
  --sim-url http://127.0.0.1:28561 \
  --tasks-jsonl /path/to/data/libero/eval.jsonl \
  --output-dir /path/to/eval_out \
  --concurrency 8 \
  --max-turns 50 \
  --max-tokens 16384 \
  --context-management dynamic \
  --include-jaw-mm \
  --request-timeout-s 1200 \
  --resume
```

## Multi-Node Routers & Sleepable Collocated Pools

For topologies that aggregate simulator pools across multiple hosts or collocate simulators on RL training GPUs (`tools/rl/sim_services_48gpu.example.yaml`), `robots/rl/server.py` provides:

- **`SimPoolRouter` (`python -m robots.rl.server router`)**: Fronts multiple remote `SimPoolManager` HTTP backends with capacity-weighted least-loaded placement (`_reserved / _capacity`), transparent binary `/open` and `/execute` passthrough, aggregated `GET /health`, and fan-out `/cohorts/{id}/close` and `/pool/{sleep,wake}` endpoints.
- **Sleepable collocated pools (`--sleepable`)**: Collocated RL+simulator nodes mark `sleepable: true` so RLinf can call `POST /pool/sleep` before Megatron backward to release VRAM and `POST /pool/wake` after weight sync, while dedicated simulator nodes (`sleepable: false`) stay continuously warm.
- **High-concurrency SAM3 queue (`max_pending: 2048`)**: `HttpRpcServer.request_queue_size` is set to `2048` so hundreds of concurrent visual episodes never hit socket backlog drops.

## Production Issues & Fixes Reference

### 1. Multi-Simulator 5-Env & IsaacSim Runtime Pitfalls

| Symptom / Failure | Root Cause | Fix & Operational Guidance |
| :--- | :--- | :--- |
| **RoboDojo fails importing `rlinf.envs.sim.robodojo.assets` when co-deployed with LIBERO** | Putting workspace `RLinf` in global `PYTHONPATH` shadows RoboDojo's installed `rlinf.envs.sim.robodojo` package. | Restrict global `env.PYTHONPATH` in `serve.py` YAML to `RPent` only; scope `RLinf` strictly under `simulators.libero.env.PYTHONPATH`. |
| **IsaacSim 5.0 / 5.1 crashes on NVIDIA `535.x` drivers (`createDLSSContext` segfault / driver version error)** | `535.x` drivers fail Kit's driver check and segfault inside NGX/DLSS initialization. | Export `OMNI_KIT_ACCEPT_EULA=Y`, `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, native `LD_LIBRARY_PATH` (no `/usr/local/cuda/compat`), and `RPENT_ISAAC_KIT_ARGS="--/rtx/verifyDriverVersion/enabled=false --/ngx/enabled=false --/rtx/post/aa/op=0 --/rtx-transient/dlssg/enabled=false"`. |
| **RoboDojo `/sessions/open` hangs `300s` (`did not announce its endpoint`) on air-gapped nodes** | USD scenes embed S3 Nucleus URLs that stall unless redirected offline via node-local `omniverse.toml`. | Install `/root/.nvidia-omniverse/config/omniverse.toml` mapping S3 Nucleus URLs to the local `isaac_nucleus/` mirror on every simulator host, and pass `--/persistent/isaac/asset_root/{default,cloud,nvidia}` in `RPENT_ISAAC_KIT_ARGS`. |
| **RoboDojo worker fails on open (`missing expected_meta`, `curobo.yml`, or `Eval_Layout/arx_x5/<seed>/` not found)** | `RoboDojoEnvClient` needs `expected_meta` and `ROBODOJO_ASSETS_ROOT`; `curobo.yml` holds absolute paths; `seed ∈ {0,1,2}` selects the layout directory while `layout ∈ [0..19]` selects the JSON file. | Run `update_embodiment_config_path.py` after relocating assets, generate task JSONLs with `seed ∈ {0,1,2}` and `layout ∈ [0..19]`, and let `robots/rl/server.py` wire `expected_meta` and `ROBODOJO_ASSETS_ROOT`. |
| **IsaacSim cold-start segfault stalls `SimPoolManager.start()` for `240s`** | Waiting solely on the readiness future missed immediate child exits. | `robots/rl/server.py` polls `proc.poll()` every `0.5s` for immediate crash detection and retries failed spawns up to `3×`. |
| **Single-GPU Vulkan `VkResult: ERROR_DEVICE_LOST` (`EXIT=139`)** | A GPU with a wedged Vulkan driver state can still pass CUDA/NCCL/EGL checks while crashing every `SimulationApp` launch. | Pre-flight all 8 GPUs with simultaneous `SimulationApp` checks; keep Vulkan-faulty nodes on pure CUDA RL and assign only 8/8 Vulkan-healthy nodes to IsaacSim pools. |
| **Dedicated 5-Env simulator node OOMs when oversubscribing IsaacSim workers** | Each active IsaacSim worker (`robodojo` / `robolab`) peaks at `~11.5–12.5 GiB` VRAM. | Cap IsaacSim at `4` workers/GPU (`16 RoboDojo + 16 RoboLab` staggered across 8× 80 GB GPUs) alongside `40 LIBERO + 40 MetaWorld + 32 RoboTwin + 8 SAM3` (`~58–65 GiB / 81.5 GiB` peak). |
| **RoboLab `WhiteMugInCenterOfTableTask` crashes on open with `AttributeError: ... has no attribute 'bowl'`** | Upstream RoboLab declares `bowl` / `banana` in `contact_object_list` even though `objects_around_table.usda` lacks those scene prims. | `_install_contact_object_guard()` in `robots/robolab/episode_server.py` wraps `create_contact_sensors` before task registration and filters out phantom `contact_object_list` names absent from `env_cfg.scene`. |
| **`httpx.ReadTimeout` in `robots/rl/rollout.py` under heavy 27B thinking concurrency** | Large thinking models (`max_tokens=16384`) serving 10+ concurrent multimodal turns can exceed the default `600s` client timeout. | Increase `--request-timeout-s` (e.g., `1200`–`3600`) and inspect `<ep_dir>/outcome.json` or logs to separate LLM latency from simulator pool errors. |

### 2. Multi-Node & Cross-Cluster Transport Pitfalls

| Symptom / Failure | Root Cause | Fix Implemented |
| :--- | :--- | :--- |
| **`EMFILE: Too many open files` / `SIGABRT`** under 100–384 concurrent sessions | Default container `ulimit -n 1024` is exhausted by concurrent HTTP sockets, Unix sockets, and Ray/Gloo channels. | `robots/rl/server.py` and `tools/rl/serve.py` automatically raise `RLIMIT_NOFILE` to `min(hard, 1048576)`; launch scripts also set `ulimit -n 1048576`. |
| **`httpx.RemoteProtocolError: Server disconnected without sending a response`** | `uvicorn`'s default `timeout_keep_alive=5` raced with `httpx`'s `keepalive_expiry=5.0`, and transport errors surfaced as HTTP `500`. | Raised uvicorn `timeout_keep_alive=300` in `server.py`, set `SimPoolRouter` `keepalive_expiry=30.0`, and mapped `httpx.TransportError` to HTTP `503` for clean client retries. |
| **SAM3 RPC connection drops during rollout bursts** | Default `ThreadingHTTPServer.request_queue_size = 256` overflowed during simultaneous segmentation bursts. | Raised `HttpRpcServer.request_queue_size = 2048` in `rpent/utils/rpc/http_rpc.py` and set `sam3.max_pending: 2048`. |
| **Proxy interception on internal VPC / loopback HTTP** | Container `HTTP_PROXY` variables hijacked internal simulator pool traffic. | Set `trust_env=False` in `SimPoolRouter` / clients and explicitly clear `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` + set `NO_PROXY` in `serve.py` YAML. |
| **Intermittent 0-byte reads on shared BeeGFS/NFS under high concurrency** | Concurrent opens of MuJoCo meshes and Python packages across dozens of workers trigger distributed-FS short-reads. | Stage `envs/`, `assets/`, and repos into node-local RAM tmpfs (`/dev/shm/rpent_ppo`, symlinked at `/tmp/rpent_ppo`) when running on affected shared filesystems. |

## Unit Tests

```bash
pytest tests/unit_tests/robots/rl/test_rl_core.py \
       tests/unit_tests/robots/rl/test_sim_pool_multinode.py \
       tests/unit_tests/robots/rl/test_metaworld_episode.py \
       tests/unit_tests/robots/robolab/ -v
```
