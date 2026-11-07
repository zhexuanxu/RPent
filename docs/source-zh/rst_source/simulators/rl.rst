Adaptive 机器人工作流
================================================================

使用 ``--robot rl``，让 VLM 通过末端控制和视觉工具操作机器人。LIBERO、RoboCasa、
RoboTwin 共享 multiturn planner，LIBERO 还支持实验性的 ZAP snapshot 循环。
成功与否由原生仿真器判定，无需 VLA 权重或任务解法记忆。

.. list-table:: 支持的组合
   :header-rows: 1

   * - 仿真器
     - Multiturn
     - ZAP
   * - ``libero`` （含 LIBERO-PRO）
     - 支持
     - 实验性
   * - ``robocasa``
     - 支持
     - 不支持
   * - ``robotwin``
     - 支持
     - 不支持

安装与资源
----------------------------------------------------------------

:download:`安装指南 <../../../../tools/rl/INSTALL_zh.md>` 从新的 checkout 开始，
覆盖系统库、隔离 Python 环境、模型、SAM3、仿真资产与排错。所有路径由自己的
checkout 和数据目录推导，不依赖集群路径。复制 ``tools/rl/config.example.sh``
为 ``.rl-env.sh`` ，再按实际资源修改。

公共环境运行 planner、SAM3、LIBERO；RoboCasa、RoboTwin 各用隔离仿真环境，
模型服务单独安装。Linux 上需要 NVIDIA EGL；RoboTwin 还需要 Vulkan 及编译
cuRobo 的 CUDA toolkit。先准备资产、检查导入，再运行任务。

运行一个任务
----------------------------------------------------------------

:download:`首次运行指南 <../../../../tools/rl/README_zh.md>` 提供完整的服务启动和
四条任务命令：LIBERO multiturn、LIBERO ZAP、RoboCasa multiturn、RoboTwin
multiturn。先从一个 episode 开始。

配置并启动模型和共享 SAM3 后，在 checkout 根目录运行：

.. code-block:: bash

   source .venv/bin/activate
   source .rl-env.sh
   source tools/rl/env.sh
   bash tools/rl/run.sh --simulator libero --workflow multiturn \
     --suite libero_object --task 0 --seed 1 --cuda-device 1

``--cuda-device`` 选择仿真 GPU，与模型/SAM3 的 GPU 独立。RoboCasa 使用
``--task-name`` 和 ``--split`` ；RoboTwin 使用 ``--task-name`` 、
``--task-config`` 及公布的有效 seed。首次运行指南给出完整示例和任务清单位置。

LIBERO multiturn 概览
----------------------------------------------------------------

Multiturn 保留对话和工具结果，必要时压缩上下文。 ``view_env`` 返回已记录图像，
``segment`` 和 ``back_project`` 测量匹配帧中的可见表面。共用动作工具名为
``move_to`` 、 ``set_gripper`` 、 ``finish`` 。RoboCasa 增加局部平面速度及偏航
角速度工具 ``move_base`` ；RoboTwin 增加机器人解剖学左右臂选择、图像标记与
双臂同步指令。任务文本直接来自仿真器，公共 prompt 只描述机器人与工作流。

ZAP snapshot 循环
----------------------------------------------------------------

LIBERO ZAP 每轮发送当前图像、同帧 SAM3 框与 3D 测量、机器人状态、可见 todo
及上次工具结果，不带历史 reasoning 或对话，不做历史压缩。第一轮调用
``change_query`` ，第二轮调用 ``update_todo`` ，之后每轮仅执行一个工具。
不暴露 ``view_env`` 、 ``segment`` 、 ``back_project`` 。该实现仍为实验性，
尚无成功率或速度保证。

规划器配置
----------------------------------------------------------------

默认预算：100 次模型请求、10000 仿真步、单次输出 32768 tokens、262144-token
上下文、multiturn 在 70% 时压缩历史、7200 秒 planner 超时。批量启动器另有
14400 秒 episode 超时。LIBERO multiturn 保留 ``simple`` 、 ``base`` 、
``base_gripper`` 、 ``base_gripper_camera`` 四种 prompt 变体。

示例配置显式设置 presence penalty 1.5、repetition penalty 1.0；不导出时，启动器
保留服务端默认值。Thinking 采样为 temperature 1.0、top-p 0.95、top-k 20，
Qwen3.8 默认 effort 为 ``xhigh`` 。
:download:`参考文档 <../../../../tools/rl/REFERENCE_zh.md>` 列出服务、planner、
批量采集的完整参数，以及实际请求的记录方式。

查看结果
----------------------------------------------------------------

``result.json`` 保存原生成功标记、错误、轮数和 tokens， ``states.json`` 保存
实测状态及原生终止标记。相机/深度数据、 ``episode.mp4`` 、 ``rollout.json`` 、
原生 transcript 和 ``api_trace/`` 保留执行证据。进程正常退出并不证明任务成功或
轨迹可用于训练；批量启动器审计完整请求/响应及图像哈希后才发布记录。

采集任务与 seed
----------------------------------------------------------------

参考文档说明有界并发、valid/evaluated 索引、基础设施重试与中断恢复。使用新的
批次目录，先从每 GPU 一个 episode 开始，再根据实测场景显存增加并发。
``tools/rl/jobs`` 提供示例任务/seed 清单。

远程仿真后端
----------------------------------------------------------------

Adaptive multiturn 可通过 ``python -m robots.rl.remote_manager`` 和
``RPENT_REMOTE_TOOLKIT_ENDPOINT`` 将动作、渲染、SAM3 放到另一台后端机器。
两侧源码必须一致。参考文档覆盖两侧安装、私有网络连接、GPU 容量、源码/任务/
预算校验、文件哈希与租约清理。远程 ZAP 暂不支持。

RLinf 训练
----------------------------------------------------------------

参考文档保留离线安装、RLinf/SFT 集成说明。RLinf 负责策略训练和 checkpoint，
RPent 负责仿真器与工具执行；episode 入口为 ``robots.rl.rl_episode`` 。
数据编码位于 ``ms-swift/examples/rpent`` ，应保留实际请求与图片顺序。
