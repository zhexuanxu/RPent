LIBERO RL Rollouts
==================

``libero_rl`` runs LIBERO-PRO with a language-model policy and six robot tools.
It uses no action model, task guides or task memory. The same robot interface
is used by standalone RPent collection and RLinf training. This page covers
collection; the RLinf repository owns the training configuration and launcher.

Overview
--------

Each episode starts a fresh simulator. A shared SAM3 service can provide
segmentation for several episodes. The planner acts until ``finish``, the
request or simulator-step budget, or a timeout. Native success is the last
state's LIBERO ``terminated`` flag; the planner's answer does not determine it.

.. list-table:: Interface
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Behavior
   * - Tools
     - ``move_to``, ``set_gripper``, ``view_env``,
       ``segment``, ``back_project``, ``finish``
   * - Observation
     - Adaptive observations: ``view_env`` returns the two camera images;
       initial and action observations contain measured state.
   * - Prompts
     - ``simple``, ``base``, ``base_gripper``, ``base_gripper_camera``
   * - Default budgets
     - 100 model requests, 10000 simulator steps, 32768 output tokens per
       request, 262144-token context, history compaction at 70%,
       7200-second planner timeout. The batch launcher also enforces a
       14400-second episode timeout.

Installation and Resources
--------------------------

Use Linux x86_64, Python 3.12, NVIDIA EGL and a CUDA-compatible driver.
Install the simulator in its own environment from the RPent checkout root:

.. code-block:: bash

   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   source tools/libero_rl_eval/env.sh

The :download:`complete setup guide <../../../../tools/libero_rl_eval/README.md>`
covers a separate SGLang environment, model/SAM3 downloads, LIBERO-PRO assets,
configuration generation and offline installation. Complete those steps before
running an episode. Set ``RPENT_DATA_ROOT`` before sourcing ``env.sh`` to choose
the data root; its default is ``$XDG_DATA_HOME/rpent``, falling back to
``$HOME/.local/share/rpent``. Explicit asset/checkpoint variables take precedence.
No cluster-specific activation script or Pi0.5 checkpoint is required.

Run a Task
----------

In a new terminal, activate SGLang and start a model service. ``serve.sh``
uses that interpreter (or ``RPENT_SGLANG_PYTHON``) and requires ``MODEL_PATH``.
Its defaults are the name Qwen3.8-27B, GPU 2–3, TP2, MTP and port 30080.
``SERVE_GPUS``, ``SERVE_PORT``, ``MODEL_PATH``, ``MODEL_NAME``,
``SERVE_MEM_FRACTION`` and ``SERVE_CONCURRENCY`` override the service settings.
For a model without MTP weights, use ``ENABLE_MTP=0``.

.. code-block:: bash

   source .venv-sglang/bin/activate
   MODEL_PATH=/path/to/Qwen3.8-27B SERVE_GPUS=2,3 bash tools/libero_rl_eval/serve.sh

The service is ready when ``curl http://127.0.0.1:30080/v1/models`` lists the
model. In a second terminal, activate RPent and start shared SAM3:

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112

In a third terminal, activate RPent and run one episode on GPU 1:

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
   bash tools/libero_rl_eval/run-libero-rl.sh \
     --suite libero_object --task 0 --seed 0 --cuda-device 1

``--cuda-device`` selects the simulator GPU. The SAM3 endpoint selects the
shared service independently. Omitting it starts an episode-owned SAM3 server
on the simulator GPU. Extra command-line arguments pass through to ``rpent``.

View Results
------------

The default output is ``logs/libero-rl-<mode>-<prompt>-<time>`` in the RPent
checkout. Set ``RPENT_OUTPUT_DIR`` to choose another directory. Each episode
contains:

* ``result.json``: native success, errors, turns and token usage;
* ``states.json``: measured states and native ``terminated`` flags;
* ``transcript_*.json``: native messages with thinking and tool calls;
* ``rollout.json``, ``episode.mp4`` and camera/state artifacts;
* ``api_trace/``: post-compaction native messages, actual HTTP request payloads,
  original PNG bytes with SHA-256, complete wire responses, stream metadata
  and compaction events.

A clean exit proves neither task success nor a complete training record.
``rpent.evaluation.trajectory.audit_episode`` checks response completion,
wire/native agreement, native success and optionally every saved image hash.
The batch launcher applies these checks before publishing an episode.

Collect Tasks and Seeds
-----------------------

``jobs/panel40.txt`` lists task 0–9 from ``libero_object``, ``libero_goal``,
``libero_spatial`` and ``libero_10`` in adaptive mode at seed 0. Each manifest
line has ``adaptive suite task seed``. ``--seeds`` overrides the listed
seed with a comma-separated list or inclusive ranges.

.. code-block:: bash

   python tools/libero_rl_eval/run_batch.py run \
     "$RPENT_DATA_ROOT/rollouts/new-run" \
     --jobs tools/libero_rl_eval/jobs/panel40.txt \
     --seeds 3-66 --gpus 1 --per-gpu 4

The model and shared SAM3 variables from the single-episode example also apply
here. ``--endpoints URL,URL`` distributes workers over additional local model
services. ``--start-interval`` staggers initial worker starts; size
``--per-gpu`` for the simulator GPU's available memory and initialization load.

Each job has separate ``attempt-NNN`` directories. A normal unsuccessful task
is valid data. An infrastructure error or failed audit is retained with
``INVALID.json`` and retried once by default. An interrupted attempt gets
``INTERRUPTED.json``. Only audited episodes receive ``COMPLETED.json`` and an
entry in the atomically updated ``valid-index.json``. Completed data is usable
when the batch is paused. SIGINT/SIGTERM terminate owned episode process groups;
repeat the same command to resume. A directory lock prevents two collectors
from launching the same queue. The manifest rejects changed task, model,
prompt or generation settings on resume.

.. code-block:: bash

   python tools/libero_rl_eval/run_batch.py summary \
     "$RPENT_DATA_ROOT/rollouts/new-run"

The summary re-audits sealed episodes and reports valid episodes and native
successes. Dataset preparation and training are in
``ms-swift/examples/rpent``.

Planner Configuration
---------------------

``planner.sh`` supplies Qwen thinking sampling: temperature 1.0, top_p 0.95,
top_k 20. Qwen3.8 uses ``reasoning_effort=xhigh`` by default; other Qwen models
send no effort tier, while thinking remains enabled. The service uses
``qwen3_coder`` for tool calls and ``qwen3`` for reasoning.

.. list-table:: Overrides
   :header-rows: 1
   :widths: 45 55

   * - Variable
     - Default
   * - ``RPENT_PROMPT``
     - ``simple``
   * - ``RPENT_MAX_TURNS`` / ``RPENT_MAX_EPISODE_STEPS``
     - 100 / 10000
   * - ``RPENT_MAX_TOKENS`` / ``RPENT_CONTEXT_WINDOW``
     - 32768 / 262144
   * - ``RPENT_COMPACTION_FRACTION``
     - 0.7
   * - ``RPENT_REASONING_EFFORT``
     - ``xhigh`` for Qwen3.8; ``none`` for other models
   * - ``RPENT_PLANNER_TIMEOUT_S``
     - 7200
   * - ``RPENT_PRESENCE_PENALTY`` / ``RPENT_REPETITION_PENALTY``
     - Unset; optional explicit sampling overrides

Earlier thinking is preserved in ``reasoning_content`` and rendered by the
model's own chat template. RPent wraps tool-image user messages with
``<tool_response>`` tags so Qwen3.5 retains earlier thinking. Qwen3.8 also uses
``preserve_thinking=true``. Only context management intentionally changes the
visible history; the actual resulting request is retained in ``api_trace``.
A reply ending at the output-token limit remains an incomplete training
output and fails the trajectory audit.

The prompt definitions live in ``robots/libero_rl/prompt_bundle.py``.
``simple`` contains the tool contract; ``base`` adds robot frames and an
execution method; ``base_gripper`` adds finger geometry; and
``base_gripper_camera`` also adds camera geometry.

RLinf Training
--------------

Activate RLinf and use its own launcher and static configuration. Configure
``tools.rpent.python`` to point to the RPent interpreter. RPent owns the
simulator and tools; RLinf owns policy generation, actor/critic optimization,
checkpointing and its sequence/history configuration.

.. code-block:: bash

   cd /path/to/RLinf
   source /path/to/rlinf-venv/bin/activate
   bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_local

Its example README documents its environment, the RPent subprocess CUDA/EGL
setup, shared SAM3 startup and the one-step smoke
configuration. Standalone environment variables do not override the RLinf
recipe. Match ``agentloop.prompt``,
``agentloop.chat_template_kwargs``, ``agentloop.history_compaction``,
``algorithm.sampling_params`` and ``runner.seq_length`` when comparing runs.
