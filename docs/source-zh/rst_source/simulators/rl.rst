Adaptive 机器人工作流
================================================================

``rl`` 为 LIBERO、RoboCasa、RoboTwin 共享简洁的 adaptive 工作流。
使用 ``--robot rl --simulator libero|robocasa|robotwin`` 选择环境，
使用 ``--workflow multiturn|zap`` 选择循环；不加载 VLA 或任务解法记忆。
每任务提示文件位于 ``robots/rl/prompts/tasks/<env>/``。

RoboCasa 增加物理单位的底盘速度与原生姿态控制；RoboTwin 增加解剖学左右臂选择、
相机标注及双臂同步指令。
:download:`安装运行指南 <../../../../tools/rl/README_zh.md>`
说明隔离仿真环境、资产准备和 100 条任务清单。

ZAP snapshot 循环
----------------------------------------------------------------

每轮输入仅含当前图像及同帧 SAM3 测量、机器人状态、当前 todo 和上次工具结果。
不携带历史 reasoning 或对话，也不做 context 压缩。第一轮调用 ``change_query``，
第二轮调用 ``update_todo``，之后每轮仅执行一个工具。query 和 todo 显式、有界地
完整替换，不使用隐式总结或文件记忆。启动方式、工具及完整轨迹说明见安装运行指南。

远程仿真后端
----------------------------------------------------------------

``--remote-toolkit-endpoint`` 或 ``RPENT_REMOTE_TOOLKIT_ENDPOINT`` 可将
adaptive multiturn 连接到另一台机器上的
``python -m robots.rl.remote_manager``。动作循环、原生渲染和 SAM3 留在
后端，planner 使用原有工具声明、结果和原始图片字节。此时
``--cuda-device`` 选择后端 GPU，环境和 SAM3 连接均由后端负责。
源码、任务与预算校验、每 GPU 容量、产物哈希校验及租约清理详见前文安装指南。

以下说明保留的 LIBERO multiturn 接口。

LIBERO multiturn 概览
----------------------------------------------------------------

每条轨迹启动独立的仿真环境，多个环境可以共用一个 SAM3 分割服务。规划器持续执行，直到调用 ``finish``、耗尽模型请求或环境步预算，或触发超时。成功由最后一个状态中的 LIBERO ``terminated`` 标志决定，不取决于规划器的回答。

.. list-table:: 接口
   :header-rows: 1
   :widths: 30 70

   * - 组件
     - 行为
   * - 工具
     - ``move_to``、``set_gripper``、``view_env``、
       ``segment``、``back_project``、``finish``
   * - 观测
     - 固定使用 adaptive 观测：``view_env`` 返回两个相机图像，
       初始观测和动作返回包含测量状态。
   * - Prompt
     - ``simple``、``base``、``base_gripper``、``base_gripper_camera``
   * - 默认预算
     - 100 次模型请求、10000 个环境步、每次请求最多 32768 个输出 token、
       262144 上下文、70% 历史压缩阈值、7200 秒规划器超时。
       批量入口另设 14400 秒单条轨迹超时。

安装与资源准备
--------------

使用 Linux x86_64、Python 3.12、NVIDIA EGL 和兼容的 CUDA 驱动。在 RPent 仓库根目录创建独立仿真环境：

.. code-block:: bash

   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   source tools/rl/env.sh

:download:`完整安装指南 <../../../../tools/rl/README_zh.md>` 包含独立 SGLang 环境、模型和 SAM3 下载、LIBERO-PRO 资源、配置生成及离线安装。运行轨迹前先完成这些步骤。source ``env.sh`` 前设置 ``RPENT_DATA_ROOT`` 可选择数据根目录，默认使用 ``$XDG_DATA_HOME/rpent``，未设置 XDG 时使用 ``$HOME/.local/share/rpent``。显式资源和权重变量优先于这些默认值。本流程不依赖集群专用激活脚本或 Pi0.5 权重。

运行一个任务
------------

在新终端激活 SGLang 并启动模型服务。``serve.sh`` 使用当前解释器（或 ``RPENT_SGLANG_PYTHON``），必须显式设置 ``MODEL_PATH``。默认服务名称为 Qwen3.8-27B，使用 GPU 2–3、TP2、MTP 和端口 30080。可以通过 ``SERVE_GPUS``、``SERVE_PORT``、``MODEL_PATH``、``MODEL_NAME``、``SERVE_MEM_FRACTION`` 和 ``SERVE_CONCURRENCY`` 覆盖服务设置。没有 MTP 权重的模型使用 ``ENABLE_MTP=0``。

.. code-block:: bash

   source .venv-sglang/bin/activate
   MODEL_PATH=/path/to/Qwen3.8-27B SERVE_GPUS=2,3 bash tools/rl/serve.sh

``curl http://127.0.0.1:30080/v1/models`` 列出模型时，表示服务已经就绪。在第二个终端激活 RPent 环境，启动共享 SAM3：

.. code-block:: bash

   source .venv/bin/activate
   source tools/rl/env.sh
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112

在第三个终端激活 RPent，在 GPU 1 上运行单条轨迹：

.. code-block:: bash

   source .venv/bin/activate
   source tools/rl/env.sh
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
   bash tools/rl/run.sh \
     --suite libero_object --task 0 --seed 0 --cuda-device 1

``--cuda-device`` 指定仿真 GPU，SAM3 endpoint 则独立指定共享服务。不设置 endpoint 时，每条轨迹会在仿真 GPU 上启动自己的 SAM3 服务。额外的命令行参数会传给 ``rpent``。

查看结果
--------

默认输出目录为 RPent 仓库下的 ``logs/rl-<mode>-<prompt>-<time>``，可以通过 ``RPENT_OUTPUT_DIR`` 修改。每条轨迹包含：

* ``result.json``：native success、错误、轮数和 token 用量；
* ``states.json``：测量状态和原生 ``terminated`` 标志；
* ``transcript_*.json``：包含 thinking 和工具调用的原生消息；
* ``rollout.json``、``episode.mp4`` 及相机和状态文件；
* ``api_trace/``：压缩后的原生消息、实际 HTTP 请求、带 SHA-256 的原始 PNG、
  完整网络响应、流式元数据和压缩事件。

进程正常退出不能证明任务成功，也不能证明训练记录完整。``rpent.evaluation.trajectory.audit_episode`` 检查响应完整性、网络响应与原生消息一致性、native success，并可校验所有图片哈希。批量入口通过审计后才发布该条轨迹。

批量采集任务和 Seed
-------------------

``jobs/panel40.txt`` 包含 ``libero_object``、``libero_goal``、``libero_spatial`` 和 ``libero_10`` 各自的 task 0–9，默认 adaptive 模式、seed 0。每行为 ``adaptive suite task seed``，``--seeds`` 可以用逗号分隔的列表或闭区间覆盖清单中的 seed。

.. code-block:: bash

   python tools/rl/run_batch.py run \
     "$RPENT_DATA_ROOT/rollouts/new-run" \
     --jobs tools/rl/jobs/panel40.txt \
     --seeds 3-66 --gpus 1 --per-gpu 4

模型和共享 SAM3 使用单条轨迹示例中的环境变量。``--endpoints URL,URL`` 可以将 worker 分配给其他本地模型服务。``--start-interval`` 控制初次启动 worker 的间隔；设置 ``--per-gpu`` 时须考虑仿真 GPU 的可用显存和初始化负载。

每个任务独立保存各次 ``attempt-NNN``。正常任务失败也是有效数据。基础设施异常和审计失败会保留 ``INVALID.json``，默认重试一次；中断的尝试保存 ``INTERRUPTED.json``。只有通过审计的轨迹才会获得 ``COMPLETED.json``，并写入原子更新的 ``valid-index.json``，暂停采集时仍可使用已完成的数据。SIGINT/SIGTERM 会终止本批次的轨迹进程组，重复相同命令即可恢复。目录锁防止两个采集进程重复启动同一队列；恢复时 manifest 会拒绝任务、模型、prompt 或生成参数变化。

.. code-block:: bash

   python tools/rl/run_batch.py summary \
     "$RPENT_DATA_ROOT/rollouts/new-run"

summary 会重新审计已封存的轨迹，报告有效数量和原生成功数量。数据编码和训练入口位于 ``ms-swift/examples/rpent``。

规划器配置
----------

``planner.sh`` 使用 Qwen thinking 的采样参数：temperature 1.0、top_p 0.95、top_k 20。Qwen3.8 默认使用 ``reasoning_effort=xhigh``；其他 Qwen 模型不发送 effort 等级，但仍启用 thinking。模型服务使用 ``qwen3_coder`` 解析工具调用，使用 ``qwen3`` 解析推理。

.. list-table:: 可覆盖的参数
   :header-rows: 1
   :widths: 45 55

   * - 变量
     - 默认值
   * - ``RPENT_PROMPT``
     - ``simple``
   * - ``RPENT_MAX_TURNS`` / ``RPENT_MAX_EPISODE_STEPS``
     - 100 / 10000
   * - ``RPENT_MAX_TOKENS`` / ``RPENT_CONTEXT_WINDOW``
     - 32768 / 262144
   * - ``RPENT_COMPACTION_FRACTION``
     - 0.7
   * - ``RPENT_REASONING_EFFORT``
     - Qwen3.8 为 ``xhigh``，其他模型为 ``none``
   * - ``RPENT_PLANNER_TIMEOUT_S``
     - 7200
   * - ``RPENT_PRESENCE_PENALTY`` / ``RPENT_REPETITION_PENALTY``
     - 未设置，可显式覆盖采样参数

历史 thinking 通过 ``reasoning_content`` 保留，并由模型自身的 chat template 渲染。RPent 用 ``<tool_response>`` 标签包裹工具图片的 user 消息，让 Qwen3.5 保留此前的 thinking；Qwen3.8 还会设置 ``preserve_thinking=true``。上下文管理会主动改变可见历史，最终实际请求保存在 ``api_trace`` 中。达到输出 token 上限的回答属于不完整训练输出，无法通过轨迹审计。

Prompt 定义位于 ``robots/rl/prompt_bundle.py``：``simple`` 包含工具契约；``base`` 增加机器人坐标系和执行方法；``base_gripper`` 增加手指几何；``base_gripper_camera`` 进一步增加相机几何。

RLinf 训练
----------

激活 RLinf 环境后，使用该仓库自己的入口和静态配置。``tools.rpent.python`` 指定 RPent 解释器。RPent 负责仿真和工具，RLinf 负责策略生成、actor/critic 优化、checkpoint 以及序列和历史配置。

.. code-block:: bash

   cd /path/to/RLinf
   source /path/to/rlinf-venv/bin/activate
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_local

RLinf 环境、RPent 子进程 CUDA/EGL 设置、共享 SAM3 启动和单步 smoke 配置见该仓库示例 README。独立采集的环境变量不会覆盖 RLinf 配方。比较结果时，需对齐 ``agentloop.prompt``、``agentloop.chat_template_kwargs``、``agentloop.history_compaction``、``algorithm.sampling_params`` 和 ``runner.seq_length``。
