# 统一 adaptive 机器人工作流

[English](README.md)

入口为 `--robot rl --simulator libero|robocasa|robotwin --workflow multiturn|zap`。
默认仍是 LIBERO 的 adaptive multiturn 接口，不加载 VLA、任务解法或隐式文件记忆。
原 `robots/libero_rl` 已迁至 `robots/rl`，原 `tools/libero_rl_eval` 已迁至 `tools/rl`。

本指南使用本地 Qwen 策略和共享 SAM3，采集 adaptive LIBERO-PRO 轨迹。
仿真器与策略服务使用独立 Python 环境，安装和启动均从源码仓库完成，
无需集群专用激活脚本。RLinf 和 ms-swift 使用各自的训练入口。

## 安装环境

下文安装 LIBERO、planner 和 SAM3。两个新仿真器在独立进程及环境中运行，
避免 NumPy、SciPy、Torch 的版本要求冲突；三个仿真器共享模型服务。

使用 Linux x86_64、Python 3.12、Git 和 [uv](https://docs.astral.sh/uv/)。
仿真器需要 NVIDIA EGL 库；模型服务需要支持其 CUDA 13 wheel 的驱动。
NVIDIA 驱动由机器或平台管理员安装。Ubuntu 的其余系统依赖如下：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libegl1 libopengl0 libgl1 libglib2.0-0
# 从包含 robots/rl 的源码工作区根目录开始。
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
source tools/rl/env.sh
```

使用包含 `robots/rl` 的工作区；较早的 `rpent-ppo` 工作区仍使用
`robots/libero_rl` 模块名。LIBERO 依赖文件选择
`zhexuanxu/RLinf:rpent-ppo` 作为已支持的环境运行时。
依赖安装命令须在仓库根目录执行。它以 editable 方式安装 RPent，并安装
支持的 RLinf 运行时、SAM3 和 LIBERO-PRO 仿真器；本流程不需要 OpenPi/Pi0.5
或动作模型权重。依赖文件选择对应 RLinf 分支及公开的 RPent SAM3/LIBERO forks，
不选择固定到旧 `release/v0.4` 的 `rlinf` extra。
实验中应记录这些依赖实际解析到的提交，以及 `uv pip freeze` 输出。

在同一仓库的“新终端”中创建模型服务环境。SGLang 0.5.23 自身锁定所需
Torch、Transformers 和 CUDA 依赖：

```bash
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install --prerelease=allow "sglang==0.5.23" "torchvision==0.29.2"
```

模型服务终端不要继承 RL/SFT 环境的 `LD_LIBRARY_PATH` 或 `PYTHONPATH`。
激活 venv 只切换 Python/PATH，不会撤销其他环境的 CUDA 库覆盖。
机器确实需要专用 CUDA/EGL 驱动路径时，可以显式使用已有站点 profile
代替标准 venv 激活；这些启动脚本不会自动加载站点 profile。

| 组件 | RPent / 仿真环境 | 模型服务环境 |
| --- | --- | --- |
| Python | 3.12 | 3.12 |
| Torch / CUDA | 2.11.0 / 12.8 | 2.14.1 / 13.0 |
| Transformers | 4.57.6 | 5.19.0 |
| 策略服务 | HTTP 客户端 | SGLang 0.5.23、kernel 0.4.10 |
| 规划器 | pydantic-ai-slim 2.53.0、harness 0.53.0 | Qwen 工具和推理解析器 |
| 仿真器 | MuJoCo 3.3.0、robosuite 1.5.2、NumPy 1.26.4 | — |
| 分割 | SAM3 0.1.0、torchvision 0.26.0 | — |

这些兼容版本用于已有 rollout 验证。`requirements/libero-rl.txt`
固定主要仿真依赖版本，并非完整的传递依赖锁文件。
即使机器装有较旧 CUDA toolkit，服务端 wheel 仍需要兼容其版本的 CUDA 驱动。

### RoboCasa 与 RoboTwin 仿真进程

在当前源码工作区中仅安装原生仿真依赖，不需要 RLDX 或 LingBot 权重。
RoboTwin 还需要带 `nvcc` 的 CUDA toolkit、C++ 编译器及 RLinf 的环境适配代码：

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

RoboCasa365 需要 `requirements/rl-robocasa.txt` 固定的 robosuite 开发版本；
PyPI 的 1.5.2 缺少 `load_model_on_init` 接口。不要替换 LIBERO 环境的 robosuite。

RoboTwin 需要 NVIDIA Vulkan 和 EGL。驱动库不在系统搜索路径时，在启动前将
`LD_LIBRARY_PATH`、`VK_ICD_FILENAMES` 指向匹配的驱动发行包，并用原生 smoke
测试核对实际 GPU 分配；CUDA 与 Vulkan 的设备编号可能不同。运行 rollout 时
激活 planner 环境，并分别指定两个仿真器的解释器：

```bash
export RPENT_ROBOCASA_PYTHON="$PWD/.venv-robocasa/bin/python"
export RPENT_ROBOTWIN_PYTHON="$PWD/.venv-robotwin/bin/python"
.venv-robotwin/bin/robotwin-download-assets --output /path/to/data/robotwin
export ROBOTWIN_ASSETS_PATH=/path/to/data/robotwin
```

RoboTwin 路径是同时包含 `.robotwin-assets.json` 和 `assets/` 的运行时根目录，
不是内部的 `assets/` 目录。原生运行时会校验快照版本。
默认使用 `demo_randomized`。100 条任务清单为每个 RoboTwin task 选择已发布的
第一个有效 seed，不为所有任务强行使用同一个可能无效的 seed。

RoboCasa 通过安装包内的 `robocasa/models/assets` 读取资产。需要共享存储时，
先复制包内自带文件，再将该环境的资产目录改为符号链接，然后下载：

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

原生下载器会在下载前询问；不需要下载 demonstration 数据集。迁移资产时保留
包内原有场景和配置文件。仅设置 `ROBOCASA_ASSETS_PATH` 不会重定向该版本的读取路径。

## 共享工具与 snapshot 工作流

ZAP 选项参考[论文的工具接口与显式 todo](https://arxiv.org/html/2610.35817v3#A1)。
这里使用原始模型，没有复现论文的 SFT/RL 训练权重。每轮独立 snapshot 是本实现的设计选择，
不声称等同未公开的实现。是否采用，应由原生成功率和固定同任务的耗时比较决定。

三个环境均使用实测状态和原生成功判据。LIBERO multiturn 的提示词、六个工具、
相机及控制与压缩行为保留。RoboCasa 增加
`move_base(vx, vy, yaw_rate, duration_s)`，指定底盘局部坐标系的目标速度（m/s、rad/s），
到时停止，并返回实际位移；原生摩擦或接触会使小速度指令达不到目标速度。RoboTwin 的移动、夹爪工具增加 `arm`，可通过另一侧的可选目标同步控制双臂。
图像中的实测 TCP 使用青色 LEFT、黄色 RIGHT 标注；左右是机器人的解剖学方向，
可能与图像左右相反。新环境的姿态工具使用世界坐标米制位置及 `[w,x,y,z]` 四元数。

`--workflow zap` 使用现有模型转换与请求记录适配器，但直接循环请求，不建立
Agent/Harness 对话，也不做 context 压缩。每个请求只包含当前任务、机器人说明、
当前图像、同帧 SAM3 框与表面/矩形 back-projection、实测状态、当前 todo 和上一次
工具结果。旧 reasoning、图像和工具历史不会重放。第一轮只能调用 `change_query`，
下一轮只能调用 `update_todo`；之后每轮只执行一个有效工具调用，多调用回复不会
执行任何动作，并正常消耗一次请求预算。

ZAP 工具为 `move_to`、`set_gripper`、`change_query`、`update_todo`、`finish`，
移动机器人再增加 `move_base`，不提供 `view_env`、`segment`、`back_project`。
query 和 todo 都是有界、显式的完整替换，不积累对话。每任务提示文件位于
`robots/rl/prompts/tasks/<env>/`，默认 `{{task_language}}` 仅插入原生指令，不添加任务解法。

按后文启动模型和共享 SAM3 后，可运行：

```bash
bash tools/rl/run.sh --simulator robocasa --task-name OpenDrawer --split target --seed 1 --cuda-device 1
bash tools/rl/run.sh --simulator robotwin --task-name adjust_bottle --task-config demo_randomized --seed 100000 --cuda-device 1
bash tools/rl/run.sh --simulator libero --workflow zap --suite libero_object --task 0 --seed 1 --cuda-device 1
python tools/rl/run_batch.py run /path/to/logs/new-100 \
  --jobs tools/rl/jobs/robocasa-robotwin100.json --gpus 0,1 --per-gpu 50 \
  --start-interval 0 --max-initializing-per-gpu 8 \
  --endpoints http://127.0.0.1:30080/v1,http://127.0.0.1:30081/v1,http://127.0.0.1:30082/v1
```

JSON 清单包含 50 个 seed 1 的 RoboCasa Target50 task 和 50 个 RoboTwin task，
每卡分配两类任务各 25 个。`--per-gpu 50` 提供共 100 个 episode 槽位；
`--max-initializing-per-gpu 8` 仅限制冷启动，不限制稳态 episode 并发。
并发参数必须经过真实原生环境的显存和容量验证。
ZAP 开发集是 `jobs/libero-dev10.json`，最终固定 40 条清单是 `jobs/panel40.txt`。
不同提示词或工作流使用新的 batch 目录；manifest 固定生产源码、提示词和 planner
依赖版本。`api_trace/` 保留完整请求及响应，ZAP 另将逐请求 snapshot 与计时写入
`snapshots/`。

## 准备模型与仿真资源

在 RPent 环境中，下载和每次运行前都 source `env.sh`。
如需共享存储，先设置 `RPENT_DATA_ROOT=/your/data/root`，再 source。
默认数据目录为 `$XDG_DATA_HOME/rpent`；没有设置 `XDG_DATA_HOME` 时使用
`$HOME/.local/share/rpent`。显式设置的 `SAM3_CHECKPOINT_PATH`、
`LIBERO_CONFIG_PATH` 和 `LIBERO_PRO_ASSET_PATH` 优先于这些默认值。

下载完整 Qwen 目录，包括 tokenizer、processor 和 chat template。
SAM3 须先在模型页面获得访问权限，再登录下载：

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

需要 HF 镜像时，在命令前设置 `HF_ENDPOINT=https://hf-mirror.com`。
只有网络要求直连时才清除代理变量。`hf download` 不依赖私有 `hfd.sh`。
下载器会将 `LIBERO_PRO_ASSET_PATH` 解释为已有资源树，因此下载命令临时移除
该变量，目标目录仍通过参数显式传入。

BDDL 和初始状态文件由 `rpent-liberopro` 安装包提供。根据实际安装位置生成
配置，同时将下载的大资源保留在自己的数据目录中：

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

确认打印出的 assets、BDDL 和初始状态目录均存在。`benchmark_root` 应保留为
实际安装的包路径，RLinf 会在加载仿真器时检查它。此类 rollout 不需要下载
demonstration 数据集。

## 启动服务和单条轨迹

以下各段在仓库根目录的独立终端执行。先激活“SGLang”环境，在 GPU 2–3
启动 Qwen3.8-27B，默认 TP2 并启用 MTP：

```bash
source .venv-sglang/bin/activate
MODEL_PATH="$HOME/.local/share/rpent/models/Qwen3.8-27B" SERVE_GPUS=2,3 bash tools/rl/serve.sh
```

使用其他数据根目录时，将 `MODEL_PATH` 改为实际下载路径。
`curl http://127.0.0.1:30080/v1/models` 列出模型后表示服务就绪。
在第二个终端中，在 GPU 0 启动共享 SAM3：

```bash
source .venv/bin/activate
source tools/rl/env.sh
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112
```

在第三个终端中，在 GPU 1 运行仿真：

```bash
source .venv/bin/activate
source tools/rl/env.sh
export RPENT_MODEL=openai-chat:qwen3.8-27b
export RPENT_BASE_URL=http://127.0.0.1:30080/v1
export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
bash tools/rl/run.sh --suite libero_object --task 0 --seed 0 --cuda-device 1
```

采集仅支持 **adaptive**：初始和动作观测包含测量状态，`view_env` 返回
指定相机图片。多模态工具结果和历史 thinking 均保留。
不设置 `RPENT_SAM3_ENDPOINT` 时，每条轨迹在自己的仿真 GPU 上启动独立 SAM3。
安装检查可设置较小的 `RPENT_MAX_TURNS`；正式采集须保持实验所需预算。

## 关键参数

`serve.sh` 可以在环境变量设置之后追加 SGLang 参数，服务监听本机
`127.0.0.1`。规划器和模型服务默认上下文都为 262144 token。

| 服务变量 | 默认值 | 含义 |
| --- | --- | --- |
| `RPENT_SGLANG_PYTHON` | `python` | 已激活 SGLang 环境的解释器；不负责切换 CUDA 库 |
| `SERVE_GPUS` | `2,3` | 可见物理 GPU，数量决定 TP |
| `MODEL_PATH` | 必填 | 本地权重目录 |
| `MODEL_NAME` | `qwen3.8-27b` | 服务模型名称，须与 `RPENT_MODEL` 的 `openai-chat:` 后缀一致 |
| `SERVE_PORT` | `30080` | 本地模型端口 |
| `SERVE_MEM_FRACTION` | `0.8` | SGLang 静态显存比例 |
| `SERVE_CONCURRENCY` | `8` | 最大同时运行的模型请求数 |
| `ENABLE_MTP` | `1` | EAGLE/MTP：3 steps、top-k 1、4 draft tokens；没有 MTP 权重的模型设为 0 |

| 轨迹变量 / 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `RPENT_MODEL`、`RPENT_BASE_URL` | 必填 | 服务模型和 `/v1` endpoint |
| `RPENT_SAM3_ENDPOINT` | 未设置 | 已启动的共享 SAM3 endpoint |
| `RPENT_PROMPT` | `simple` | `simple`、`base`、`base_gripper`、`base_gripper_camera` |
| `RPENT_REASONING_EFFORT` | Qwen3.8 为 `xhigh`，其他模型为 `none` | `none` 不发送 effort 等级，仍启用 thinking |
| `RPENT_MAX_TURNS` | `100` | 单条轨迹的模型请求数 |
| `RPENT_MAX_EPISODE_STEPS` | `10000` | 仿真步预算 |
| `RPENT_MAX_TOKENS` | `32768` | 每次请求的输出预算，包含 thinking |
| `RPENT_CONTEXT_WINDOW` | `262144` | 规划器上下文预算 |
| `RPENT_COMPACTION_FRACTION` | `0.7` | 历史压缩阈值 |
| `RPENT_PLANNER_TIMEOUT_S` | `7200` | 规划器总超时；需要其他限制时显式设置 |
| `RPENT_PRESENCE_PENALTY`、`RPENT_REPETITION_PENALTY` | 未设置 | 可显式覆盖的采样参数 |
| `RPENT_OUTPUT_DIR` | `logs/rl-adaptive-<prompt>-<time>` | 单条轨迹输出目录 |
| `--suite`、`--task`、`--seed` | suite/task 必填，seed 0 | LIBERO 任务及初始状态 seed |
| `--cuda-device` | 未设置 | 仿真 GPU，与共享 SAM3 的 GPU 独立 |

Thinking 采样参数为 temperature 1.0、top-p 0.95、top-k 20。复现 SFT checkpoint
评测的 penalty 时，显式设置 `RPENT_PRESENCE_PENALTY=1.5` 和
`RPENT_REPETITION_PENALTY=1.0`。`--observation-mode adaptive` 仍可使用，
image/text 模式会被拒绝。`run.sh` 的其他参数会直接传给 `rpent`。

## 批量采集、保存与恢复

沿用上面的模型和 SAM3 环境变量，启动一个支持恢复的任务队列：

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/new-run" --jobs tools/rl/jobs/panel40.txt --seeds 3-66 --gpus 1 --per-gpu 4
```

清单每行形如 `adaptive libero_object 0 0`。`panel40.txt` 包含
`libero_object`、`libero_goal`、`libero_spatial`、`libero_10` 各自的 task 0–9。
环境并发应结合仿真显存及初始化负载设置，模型服务并发是独立的限制。

| Batch 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--jobs` | 必填 | JSON 任务清单或 LIBERO 四字段清单 |
| `--seeds` | 使用清单的 seed | 逗号分隔的 seed 或区间，例如 `0,3-66` |
| `--gpus`、`--per-gpu` | GPU 必填，每 GPU 1 个 worker | 环境 GPU 和每卡 worker 数 |
| `--prompt` | `simple` | 整批使用一个 prompt 变体 |
| `--workflow` | `multiturn` | 对话循环或每轮独立的 ZAP snapshot |
| `--max-initializing-per-gpu` | `8` | 每卡同时冷启动数，独立于轨迹槽位 |
| `--endpoints` | `RPENT_BASE_URL` | 逗号分隔的策略 `/v1` endpoint |
| `--start-interval` | `1` 秒 | 初次启动 worker 的间隔 |
| `--episode-timeout` | `14400` 秒 | 单条轨迹的外层进程超时 |
| `--infrastructure-retries` | `1` | 每次启动中，错误或审计失败后的额外尝试数 |

每个任务独立保存 `attempt-NNN`。审计会检查 native success、完整 HTTP/原生响应、
工具调用和 PNG SHA-256，通过后写入 `COMPLETED.json` 并原子更新
`valid-index.json`。正常完成但没有成功的任务仍是有效数据；异常尝试保留
`INVALID.json`，中断尝试保留 `INTERRUPTED.json`。

正常达到输出、上下文或单条超时限制的失败写入 `EVALUATED.json`，在
`outcomes-index.json` 中计数一次，不当作基础设施故障重试，也不加入训练索引。
中断或损坏的尝试单独保留；不完整响应不会标成完整训练输出。
`valid-index.json` 只包含通过审计的记录。

SIGINT/SIGTERM 会终止本批次拥有的轨迹进程组并保留所有尝试。重复相同命令即可
恢复，已封存轨迹会重新审计后跳过。目录锁防止重复采集器，manifest 会拒绝任务、
模型、prompt 或生成参数变化。只读 summary 会重新审计已封存数据，不覆盖运行中
采集器的索引：

```bash
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/new-run"
```

单条轨迹包含 `result.json`、`states.json`、transcript、视频和 `api_trace/`。
Trace 保存压缩后实际请求、原始 PNG 字节、原生消息、完整流式响应和压缩事件。
通用读取模块为 `rpent.evaluation.trajectory`，数据编码及 SFT 入口位于
`ms-swift/examples/rpent`。

## 离线安装

在可联网的 Linux x86_64 / Python 3.12 机器上，为两个环境分别准备“完整”
wheelhouse。先按前文完成在线安装，然后在各自环境中构建对应依赖：

```bash
source .venv/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-rpent --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements/libero-rl.txt setuptools wheel
source .venv-sglang/bin/activate
uv pip install pip wheel
python -m pip wheel --pre --wheel-dir wheelhouse-sglang "sglang==0.5.23" "torchvision==0.29.2"
```

将仓库、两套 wheelhouse、模型目录和仿真资源传到离线机器。
操作系统、CPU 架构、Python ABI 和 CUDA 驱动须匹配；没有预编译 wheel
的扩展还需要在准备机器上安装构建工具链。少量缓存 wheel 不能重建完整环境。
离线机器上分别创建新 venv 并安装各自完整依赖；`--no-deps` 避免包内 Git URL
元数据触发联网，因此所有依赖 wheel 都必须已经准备好。

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --no-index --no-deps wheelhouse-rpent/*.whl
uv pip install --no-index --no-deps --no-build-isolation -e .
uv venv --python 3.12 .venv-sglang
source .venv-sglang/bin/activate
uv pip install --no-index --no-deps wheelhouse-sglang/*.whl
```

离线执行 `uv venv` 前，机器上须已安装对应 Python 解释器。
在两个环境中分别运行 `uv pip check`。迁移仿真器后重新执行配置生成步骤，
因为 YAML 中保存的是绝对路径。全部资源就绪后，可在运行终端设置
`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
环境安装和维护须与正在运行的训练进程分开。

## 将 planner 与仿真器部署在不同机器

Adaptive multiturn 支持让 planner 与模型服务部署在一起，将原生仿真器、
动作内部控制循环和共享 SAM3 留在后端机器。两端须使用相同的控制和工具源码、
任务 prompt，并均安装 planner 环境。后端额外准备独立的原生仿真环境和资产，
按前文设置 `RPENT_ROBOCASA_PYTHON`、`RPENT_ROBOTWIN_PYTHON`、
`ROBOTWIN_ASSETS_PATH` 和 `RPENT_RLINF_ROOT`。

后端先启动共享 SAM3，再启动工具管理器。GPU 容量包含初始化中的 episode，
须根据该机器的真实驻留容量测试设置：

```bash
python -m robots.rl.remote_manager \
  --output-dir /path/to/new-backend-run \
  --gpu-slots 0:4,1:6 --max-initializing-per-gpu 2 \
  --sam3-endpoint http://127.0.0.1:8112 --port 28090
```

服务默认只监听 loopback。先建立私网路由或 SSH 转发，再在 planner 机器运行：

```bash
export RPENT_REMOTE_TOOLKIT_ENDPOINT=http://127.0.0.1:28090
unset RPENT_SAM3_ENDPOINT
bash tools/rl/run.sh --simulator robocasa --task-name PickPlaceCounterToCabinet \
  --split target --seed 1 --workflow multiturn --cuda-device 1
```

这里的 `--cuda-device` 指向后端 GPU。后端负责环境和 SAM3 连接，不要再传
`--env-endpoint` 或 `--sam3-endpoint`。批量入口继承同一远程端点，仍通过
`--endpoints` 分配模型请求。此远程接口只支持 adaptive multiturn。
握手会拒绝源码摘要、工具声明、任务或预算不一致的连接。

每条后端 episode 有独立且归管理器所有的 worker，保留配置、日志、原生状态、
图片和视频。变化的产物通过 SHA-256 校验后，以原始字节同步到 planner 目录；
相关文件写入后才发布 `states.json`。`remote-runtime.json` 记录 worker
身份和源码，`remote-transfer.jsonl` 记录文件哈希和传输量。Planner transcript
和 wire trace 留在 planner 机器。传输失败会终止 episode，不重放结果不明的
动作。模型推理期间由心跳续租；planner 失联后，已就绪的后端 worker 会在
五分钟后释放，文件继续保留。

## RLinf 与 SFT

RLinf 的独立环境和 `run_train.sh` 配置见它自己的
`examples/agent/rpent/README.md`。将 `tools.rpent.python` 指向这个 RPent
venv 的解释器，并将相同 LIBERO/SAM3 路径传给 episode 子进程。
子进程还需要 RPent 对应的 CUDA/EGL 环境，不能使用 RL 环境的 `PYTHONPATH`
或 Torch/NVIDIA 库路径替代。SGLang 始终在独立环境中运行。

轨迹编码和多模态 SFT 见 `ms-swift/examples/rpent/README.md`。
训练数据须保留压缩后实际请求、原始图片字节及顺序、可见历史 thinking，
以及指定 assistant 输出的 loss mask。

使用当前工作区的 RLinf 配置须将 episode 模块指向 `robots.rl.rl_episode`；
`robots.libero_rl.rl_episode` 属于旧工作区。迁移时保留原生任务字段。
