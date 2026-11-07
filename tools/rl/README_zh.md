# Adaptive VLM 机器人工作流

[English](README.md) · [从零安装](INSTALL_zh.md) · [参数与部署参考](REFERENCE_zh.md)

VLM 通过末端控制和视觉工具直接操作机器人。统一入口为 `--robot rl`，
显式选择仿真器和工作流：

| 仿真器 | `--simulator` | Multiturn | ZAP |
| --- | --- | --- | --- |
| LIBERO / LIBERO-PRO | `libero` | 支持 | 支持，实验性 |
| RoboCasa365 | `robocasa` | 支持 | 不支持 |
| RoboTwin | `robotwin` | 支持 | 不支持 |

Multiturn 在请求之间保留对话和工具结果，必要时压缩上下文；LIBERO ZAP 每轮
使用独立 snapshot。两者都不加载任务解法、文件记忆或 VLA 策略。成功与否由原生
仿真器判定。ZAP 参考了[这篇论文](https://arxiv.org/abs/2610.35817)，属于实验性
实现，尚不保证成功率或速度。

## 安装与配置

先完成 [INSTALL_zh.md](INSTALL_zh.md)：包括系统条件、隔离 Python 环境、
模型/SAM3 下载和各仿真器资产。复制公共配置，再按自己的磁盘和服务地址修改：

```bash
# 在 checkout 根目录执行。
cp tools/rl/config.example.sh .rl-env.sh
```

下面的示例布局是 GPU 0 放 SAM3，GPU 1 放一个仿真环境，GPU 2–3 放 Qwen3.8-27B
TP2。这些只是设备编号，不要求特定机器。已有的兼容模型服务可以替代本地部署。
显存需求取决于模型、场景和并发数；请从一个 episode 开始。

## 启动公共服务

分别打开终端，工作目录都设为 checkout 根目录。先启动模型：

```bash
source .venv-sglang/bin/activate
source .rl-env.sh
SERVE_GPUS=2,3 bash tools/rl/serve.sh
```

等待 `curl --fail http://127.0.0.1:30080/v1/models` 列出 `qwen3.8-27b`。
第二个终端启动唯一的共享 SAM3：

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112
```

等日志打印监听地址后，在第三个终端激活 planner，加载相同配置：

```bash
source .venv/bin/activate
source .rl-env.sh
source tools/rl/env.sh
```

设置 `RPENT_SAM3_ENDPOINT` 后，episode 共用该服务；如果不设置，每条 episode
会在自己的仿真 GPU 上启动一个 SAM3。

## 运行一个任务

以下命令分别执行，不要同时启动。`--cuda-device` 指定仿真 GPU，模型和 SAM3
分别使用配置中指定的服务。

LIBERO-PRO multiturn：

```bash
RPENT_OUTPUT_DIR=logs/libero-multiturn bash tools/rl/run.sh \
  --simulator libero --workflow multiturn \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

RoboCasa Target50 multiturn，需要先安装 RoboCasa：

```bash
RPENT_OUTPUT_DIR=logs/robocasa-multiturn bash tools/rl/run.sh \
  --simulator robocasa --workflow multiturn \
  --task-name PickPlaceCounterToCabinet --split target --seed 1 --cuda-device 1
```

RoboTwin multiturn，需要先安装 RoboTwin：

```bash
RPENT_OUTPUT_DIR=logs/robotwin-multiturn bash tools/rl/run.sh \
  --simulator robotwin --workflow multiturn \
  --task-name click_alarmclock --task-config demo_randomized \
  --seed 100000 --cuda-device 1
```

RoboTwin 的任务、preset、seed 组合须在公开评测清单中有效；
`robots/robotwin/eval/demo_randomized.json` 保存各任务的有效 seed。
`tools/rl/jobs/robocasa-robotwin100.json` 也列出 RoboCasa/RoboTwin 示例，
它只是任务清单，并不表示需要并发启动 100 个环境。LIBERO 的 task 索引从 0 开始，
属于指定 suite；`panel40.txt` 和 `libero-dev10.json` 是示例任务清单。

LIBERO-PRO ZAP 使用相同服务和任务参数：

```bash
RPENT_OUTPUT_DIR=logs/libero-zap bash tools/rl/run.sh \
  --simulator libero --workflow zap \
  --suite libero_object --task 0 --seed 1 --cuda-device 1
```

每次运行使用新的输出目录。默认预算为 100 次模型请求、10000 仿真步、7200 秒
planner 超时，单次最多输出 32768 tokens，上下文为 262144 tokens。批量启动器
另有 14400 秒的 episode 超时。完整覆盖项见[参数参考](REFERENCE_zh.md#关键参数)。
示例配置显式设置 presence penalty 1.5、repetition penalty 1.0；若不导出这两个变量，
启动器会保留服务端默认值。

## 工作流与工具

Multiturn 首次输入为任务描述和实测机器人状态。`view_env` 返回图像，`segment`
与 `back_project` 使用匹配的已记录帧定位可见表面。`move_to`、`set_gripper`、
`finish` 是共用工具名。RoboCasa 增加 `move_base`，输入局部平面速度与偏航角速度。
RoboTwin 使用 `arm` 参数、表示机器人解剖学左右臂的 LEFT/RIGHT 图像标记，并可
同时指定另一只臂或夹爪的目标。坐标和姿态遵循各机器人的工具约定。

ZAP 每轮自动提供当前图像、同帧 SAM3 框与 3D 测量、机器人状态、当前 todo 和
上次执行结果，不拼接历史对话或 reasoning，也不做历史压缩。第一轮调用
`change_query`，第二轮调用 `update_todo`；之后每轮只执行一个工具。
`update_todo` 整体替换模型可见的 todo。ZAP 不暴露 `view_env`、`segment`、
`back_project`。

任务文本直接取自仿真器的 `get_task_language()`。公共 prompt 文件只描述机器人和
工作流，不存在逐任务 prompt 文件或隐藏任务解法。`--prompt` 的 `simple`、`base`、
`base_gripper`、`base_gripper_camera` 四种选项适用于 LIBERO multiturn；其他支持
组合使用公共机器人与工作流约定。

## 查看结果

`result.json` 记录原生成功标记、错误、轮数和 tokens；`states.json` 保存实测状态
及原生终止标记。进程正常退出不等于任务成功。每条轨迹还保留相机/深度数据、
`episode.mp4`、`rollout.json`、原生对话和 `api_trace/`，包括实际请求、原始图片字节
与完整模型响应。遇到异常先看 `result.json`、`env_server.log`、`sam3_server.log`
（episode 自己启动 SAM3 时）及 transcript 中最后一次工具结果。

需要可审计、可恢复的批量采集时，先从每 GPU 一个驻留环境开始：

```bash
python tools/rl/run_batch.py run "$RPENT_DATA_ROOT/rollouts/libero-dev10" \
  --jobs tools/rl/jobs/libero-dev10.json --gpus 1 --per-gpu 1 \
  --max-initializing-per-gpu 1 --workflow multiturn
python tools/rl/run_batch.py summary "$RPENT_DATA_ROOT/rollouts/libero-dev10"
```

ZAP 使用仅含 LIBERO 的清单，并改为 `--workflow zap`。重试、审计、断点恢复、
[离线安装](REFERENCE_zh.md#离线安装)、[远程仿真器](REFERENCE_zh.md#将-planner-与仿真器部署在不同机器)
和 [RLinf/SFT 集成](REFERENCE_zh.md#rlinf-与-sft) 见参考文档。

## 代码结构

| 模块 | 职责 |
| --- | --- |
| `robots/rl/robot_spec.py` | CLI、配置、服务生命周期、planner 选择 |
| `robots/rl/prompt_bundle.py`、`prompts/*.md` | 支持组合与可复用的机器人/工作流约定 |
| `robots/rl/toolkit.py` | 工具注册、原生任务文本、episode 产物 |
| `robots/rl/tools.py`、`geometry.py` | LIBERO 动作原语及共享视觉几何 |
| `robots/rl/backends.py`、`server.py` | RoboCasa/RoboTwin 动作适配与隔离原生服务 |
| `robots/rl/zap.py`、`snapshot.py` | 独立请求循环、可见 todo 与视觉 snapshot |
| `robots/rl/remote.py`、`remote_manager.py` | 远程 multiturn worker 与文件传输 |
| `robots/rl/rl_episode.py` | RL trainer 的单 episode 接口 |
| `tools/rl/run.sh`、`run_batch.py` | 单条轨迹及可审计批量启动器 |

原生仿真 facade 保留在 `robots/libero`、`robots/robocasa`、`robots/robotwin`，
共享工作流复用它们，而不加载各自的 VLA toolkit。扩展新仿真器时增加原生适配、
机器人约定和明确的工作流支持，无需再复制一套 planner。CPU 回归测试位于
`tests/unit_tests/robots/rl`。
