# Harness v2 SFT Data Collection, Compaction Chain-Split Export & `ms-swift` Distillation Guide

This guide documents the end-to-end, RL-aligned SFT distillation pipeline for **Harness v2** (`robots/rl/rollout.py` -> `robots/rl/sft_export.py` -> `zhexuanxu/ms-swift` -> `RLinf` Async GRPO).

Every collection, compaction, tokenization, and vision setting is strictly aligned with RLinf Harness v2 RL training (`max_turns=50`, `max_tokens=16384`, `context_window=65536`, `context_management=dynamic`, `compaction_max_tokens=48000`, `keep_messages=10`, `keep_recent_images=4`, `keep_recent_thinking=4`, `preserve_opening_images=True`, `max_episode_steps=10000`, `include_jaw_mm=True`, deterministic `call_{turn_idx}_{idx}` tool call IDs, step-0 opening images, and `execute_actions(..., observe=...)`).

---

## 1. End-to-End Pipeline Overview

```text
  Teacher Models (Parallel)                 Dedicated Sim + SAM3 Nodes
┌─────────────────────────────────┐      ┌──────────────────────────────┐
│ Local Qwen3.8-27B (6× 8xH100)  │      │ Sim Node A (8× H100, 160 slots)│
│  24× TP=2 SGLang replicas       │◄────►│  64 LIBERO + 16 RoboTwin      │
│  Continuous batching + Radix    │      │  16 RoboLab + 64 MetaWorld    │
└─────────────────────────────────┘      │  + 8 SAM3 (port 28520)        │
                                         └──────────────────────────────┘
┌─────────────────────────────────┐      ┌──────────────────────────────┐
│ Remote API (deepseek-v4.1-flash)│      │ Sim Node B (8× H100, 160 slots)│
│  Via jump-host HTTPS reverse    │◄────►│  64 LIBERO + 16 RoboTwin      │
│  proxy (air-gapped VPC safe)    │      │  16 RoboLab + 64 MetaWorld    │
└─────────────────────────────────┘      │  + 8 SAM3 (port 28520)        │
                  │                                     │
                  ▼   python -m robots.rl.rollout --compaction-tokenizer Qwen3.5-4B
         Per-Episode Atomic Storage (schema_version: 3, --resume safe)
         <output_dir>/
           ├── manifest.jsonl
           ├── infra_errors.jsonl     (infra/transport failures skipped for --resume)
           ├── summary.json
           └── episodes/<episode_key>/
                 ├── trajectory.json   (chains[], full_messages, compacted_messages, tools, outcome)
                 ├── outcome.json
                 └── images/<fid>.png  (deduplicated content-addressed PNGs)
                  │
                  ▼   python -m robots.rl.sft_export --format ms_swift
         Compaction-Chain-Split Verified ms-swift JSONL:
           - Chain 0: Opening + Turns 0..t0-1 (loss=true)
           - Chain k: Compacted prefix 0..t_{k-1}-1 (loss=false!) + Turns t_{k-1}..t_k-1 (loss=true)
           - Masks validation retries & empty thinking, strips null args & truncated final turns
                  │
                  ▼   zhexuanxu/ms-swift (branch rpent-qwen3.5-sft)
         Qwen3.5-4B SFT Checkpoint (100% template, loss-mask, and pixel-processor aligned with RLinf)
```

---

## 2. Step 1 — Start Simulator + SAM3 Pools

Copy `tools/rl/sim_services_4env_sft_h100.example.yaml`, adjust paths, and start on each 8× 80 GB H100 simulator node:

```bash
ulimit -n 1048576
PYTHONPATH=/path/to/RPent \
  python /path/to/RPent/tools/rl/serve.py \
  --config /path/to/sim_services_4env_sft.yaml
```

Verify all 4 environment pools are healthy (`workers_idle == workers_target`):

```bash
for port in 28561 28562 28564 28565; do
  echo "=== :$port ===" && curl -s "http://127.0.0.1:${port}/health" && echo
done
```

> **Pre-flight reminders**: See [`robots/rl/README.md`](./README.md#pre-flight-checklist-for-a-dedicated-5-env-simulator-node) for 8-GPU Vulkan verification, offline `/root/.nvidia-omniverse/config/omniverse.toml`, and native `LD_LIBRARY_PATH`.

---

## 3. Step 2 — Serve Teacher Models

### Option A: Local `Qwen3.8-27B` SGLang Continuous-Batching Replicas (6× 8× H100 = 24× `TP=2`)

Serving `Qwen3.8-27B` with **`TP=2`** gives 4 independent replicas per 8-GPU node (24 replicas across 6 nodes) with generous KV/RadixCache headroom for 64k multimodal contexts:

```bash
export PATH="/opt/venv/reason/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
export SGLANG_MM_BUFFER_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in 0 1 2 3; do
  g0=$((i * 2))
  g1=$((i * 2 + 1))
  port=$((30000 + i))
  CUDA_VISIBLE_DEVICES="${g0},${g1}" \
  nohup /opt/venv/reason/bin/python -m sglang.launch_server \
    --model-path /path/to/models/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B \
    --host 0.0.0.0 --port "${port}" \
    --tp 2 \
    --context-length 65536 \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --attention-backend triton \
    --trust-remote-code \
    > "/path/to/logs/sglang_$(hostname)_${port}.log" 2>&1 < /dev/null &
done
```

### Option B: Remote OpenAI-Compatible API (`deepseek-v4.1-flash`) on Air-Gapped Clusters

When GPU nodes cannot reach the public internet directly, run a minimal VPC-restricted `nginx` HTTPS reverse proxy on an internet-connected jump host:

```nginx
server {
    listen <JUMP_VPC_IP>:18080;
    allow <CLUSTER_VPC_CIDR>;
    deny all;

    client_max_body_size 64m;
    proxy_connect_timeout 60s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;

    location /maas/v1/ {
        proxy_pass https://cloud.infini-ai.com/maas/v1/;
        proxy_ssl_server_name on;
        proxy_http_version 1.1;
        proxy_set_header Host cloud.infini-ai.com;
        proxy_set_header Connection "";
        proxy_set_header Authorization $http_authorization;
    }
}
```

Pass your API key **only via environment variable** (`export DEEPSEEK_API_KEY=...`) — never commit keys to configs or scripts.

---

## 4. Step 3 — Run Resumable RL-Aligned SFT Collection

Always pass **`--compaction-tokenizer /path/to/models/Qwen3.5-4B`** for **both** Qwen3.8-27B and DeepSeek-V4.1-Flash collectors. This counts exact target student (Qwen3.5-4B) text + `smart_resize` visual tokens so both teachers trigger dynamic compaction at the exact RLinf `48,000 / 65,536` boundary and record clean per-chain snapshots (`schema_version: 3`).

### Local `Qwen3.8-27B` Collectors (`--shard 0/2`, Full Sim Concurrency)

```bash
PYTHONPATH=/path/to/RPent python -m robots.rl.rollout \
  --model-urls-file /path/to/qwen27b_endpoints.txt \
  --model-name Qwen3.8-27B \
  --provider sglang \
  --sim-url http://<SIM_NODE_A_IP>:28561 \
  --tasks-jsonl /path/to/data/libero/train.jsonl \
  --output-dir /path/to/sft_data/qwen3.8-27b/libero \
  --shard 0/2 \
  --interleave-tasks \
  --target-count 1000 \
  --concurrency 64 \
  --max-turns 50 \
  --max-tokens 16384 \
  --context-window 65536 \
  --max-episode-steps 10000 \
  --context-management dynamic \
  --compaction-max-tokens 48000 \
  --keep-messages 10 \
  --keep-recent-images 4 \
  --keep-recent-thinking 4 \
  --preserve-opening-images \
  --compaction-tokenizer /path/to/models/Qwen3.5-4B \
  --include-jaw-mm \
  --request-timeout-s 1200 \
  --resume
```

Repeat per environment on Sim Node A (`robotwin` `:28562 --concurrency 16`, `robolab` `:28564 --concurrency 16`, `metaworld` `:28565 --concurrency 64`).

### Remote `deepseek-v4.1-flash` Collectors (`--shard 1/2`, 128 Total Concurrency)

```bash
export DEEPSEEK_API_KEY="your-api-key"

PYTHONPATH=/path/to/RPent python -m robots.rl.rollout \
  --model-url http://<JUMP_VPC_IP>:18080/maas/v1 \
  --model-name deepseek-v4.1-flash \
  --provider openai \
  --api-key-env DEEPSEEK_API_KEY \
  --sim-url http://<SIM_NODE_B_IP>:28561 \
  --tasks-jsonl /path/to/data/libero/train.jsonl \
  --output-dir /path/to/sft_data/deepseek-v4.1-flash/libero \
  --shard 1/2 \
  --interleave-tasks \
  --target-count 1000 \
  --concurrency 48 \
  --max-turns 50 \
  --max-tokens 16384 \
  --context-window 65536 \
  --max-episode-steps 10000 \
  --temperature 0.7 \
  --context-management dynamic \
  --compaction-max-tokens 48000 \
  --keep-messages 10 \
  --keep-recent-images 4 \
  --keep-recent-thinking 4 \
  --preserve-opening-images \
  --compaction-tokenizer /path/to/models/Qwen3.5-4B \
  --include-jaw-mm \
  --max-llm-retries 8 \
  --request-timeout-s 600 \
  --resume
```

(On Sim Node B with `64 LIBERO + 16 RoboTwin + 16 RoboLab + 64 MetaWorld`: e.g. `48 + 16 + 16 + 48 = 128` concurrent requests.)

---

## 5. Step 4 — Compaction-Chain-Split Export & Automated Verification

Run `robots/rl/sft_export.py` to split compacted episodes at every compaction boundary into independent `ms-swift` training rows, apply strict DeepSeek -> Qwen3.5 sanitization, and verify both token rendering and `labels` loss masking:

```bash
# Recommended distillation dataset: successful trajectories, chain-split on compaction
PYTHONPATH=/path/to/RPent python -m robots.rl.sft_export \
  --input-dirs \
    /path/to/sft_data/qwen3.8-27b \
    /path/to/sft_data/deepseek-v4.1-flash \
  --output-jsonl /path/to/sft_exports/rpent_4env_ms_swift_success.jsonl \
  --format ms_swift \
  --history-mode chains \
  --split-compaction-chains \
  --max-seq-tokens 65536 \
  --only-success \
  --drop-infra-errors \
  --drop-truncated-final-turn \
  --mask-retry-turns \
  --mask-empty-thinking \
  --strip-null-args \
  --normalize-call-ids \
  --image-root /path/to/sft_data \
  --verify-tokenizer /path/to/models/Qwen3.5-4B \
  --verify-ms-swift \
  --ms-swift-repo /path/to/repos/ms-swift
```

### Why Compaction Chain-Splitting & Loss Masking Prevent SFT -> RL Mismatch

1. **1-to-1 Mirror of RLinf's `AppendOnlyHistory.chain_id`**:
   - **Chain 0** contains `[opening, turns 0..t0-1]` with `loss=true` on all valid generated turns.
   - **Chain `k >= 1`** contains the **exact dynamically compacted prefix** (older images replaced with `[image omitted]`, older `reasoning_content` stripped to `<think>\n\n</think>`) marked **`"loss": false`** on every carried-over `assistant` + `tool_call` message, followed by the newly generated post-compaction turns (`loss=true`).
   - Every valid turn across the entire episode is supervised **exactly once** (never double-counted, never teaching empty `<think>\n\n</think>`), conditioned on the exact prefix distribution the policy sees in RLinf RL, and naturally bounded within `<= 65536` tokens.
2. **Strict Cross-Teacher (DeepSeek -> Qwen3.5) Sanitization**:
   - Masks invalid validation-retry turns (`--mask-retry-turns`) and empty-thinking turns (`--mask-empty-thinking`) with `"loss": false` while keeping recovery turns supervised.
   - Drops truncated final turns (`--drop-truncated-final-turn`) and infra/abort trajectories (`--drop-infra-errors`).
   - Strips explicit top-level `null` tool arguments (`--strip-null-args`) so Qwen3.5 Jinja (`None`) and `ms-swift` (`null`) never diverge.
   - Blocks Qwen control tokens (`<|im_start|>`, `<|im_end|>`, `</think>`, `<tool_call>`, etc.) and `ms-swift` media placeholders (`<image>`, `<video>`, `<bbox>`, etc.) from leaking inside prose text.
   - Trims trailing non-assistant and trailing `loss: false` turns, dropping any 0-supervised chain so every sample ends cleanly on a supervised turn.

---

## 6. Step 5 — Training `Qwen3.5-4B` with `zhexuanxu/ms-swift` (`rpent-qwen3.5-sft`)

Use our forked `zhexuanxu/ms-swift` (`rpent-qwen3.5-sft` branch), which fixes 7 subtle Qwen3.5 multimodal agent SFT bugs (`SUFFIX` EOS leak on trailing `loss=False`, Arrow parallel `tool_call` `loss=None` shadowing, `<tool_response>` user follow-up misdetection, 0-label microbatch `NaN`, SP `eval_loss` all-reduce, `SWIFT_USE_HF_IMAGE_PROCESSOR=1` pixel parity with RLinf SGLang, and clean `config.json` `mtp_num_hidden_layers=0`/`use_cache=True`/`chat_template.jinja` checkpoint export).

See `examples/train/multimodal/rpent_qwen3_5_4b_sft.sh` and `README_RPENT.md` in `zhexuanxu/ms-swift` for full single-node and multi-node recipes:

```bash
# Critical Qwen3.5-4B 65536 constraints:
# 1. sequence_parallel_size MUST be <= 4 (Qwen3.5-4B has num_key_value_heads=4; sp=8 crashes GatedDeltaNet!)
# 2. CELOSS_PARALLEL_SIZE=2048 is mandatory when sp > 1 (avoids 24 GB logits OOM on 248,320 vocab)
# 3. SWIFT_USE_HF_IMAGE_PROCESSOR=1 guarantees bit-identical pixel_values/image_grid_thw with RLinf SGLang
# 4. --lazy_tokenize false ensures --truncation_strategy delete truly drops oversized rows instead of resampling
# 5. DeepSpeed ZeRO-2 (or FSDP2) preserves FP32 linear_attn.A_log / norm.weight cleanly

ROOT_IMAGE_DIR=/path/to/sft_data \
SWIFT_USE_HF_IMAGE_PROCESSOR=1 \
MIN_PIXELS=65536 \
MAX_PIXELS=16777216 \
CELOSS_PARALLEL_SIZE=2048 \
NPROC_PER_NODE=8 \
swift sft \
  --model /path/to/models/Qwen3.5-4B \
  --template qwen3_5 \
  --dataset /path/to/sft_exports/rpent_4env_ms_swift_success.jsonl \
  --train_type full \
  --torch_dtype bfloat16 \
  --max_length 65536 \
  --truncation_strategy delete \
  --lazy_tokenize false \
  --loss_scale default \
  --preserve_thinking true \
  --freeze_vit true \
  --freeze_aligner true \
  --sequence_parallel_size 4 \
  --attn_impl flash_attn \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --deepspeed zero2 \
  --output_dir /path/to/checkpoints/qwen3.5-4b-rpent-4env-sft
```

The resulting checkpoint can be pointed directly at `RLinf` via `RPENT_MODEL_PATH=/path/to/checkpoints/qwen3.5-4b-rpent-4env-sft/checkpoint-XXX` with zero conversion or template drift.
