Adaptive Robot Workflows
========================

Run a VLM through end-effector and perception tools with ``--robot rl``.
LIBERO, RoboCasa and RoboTwin share the multiturn planner; LIBERO also supports
an experimental ZAP snapshot loop. Native simulator success determines the
outcome. No VLA weights or task-recipe memory are required.

.. list-table:: Supported combinations
   :header-rows: 1

   * - Simulator
     - Multiturn
     - ZAP
   * - ``libero`` (including LIBERO-PRO)
     - Supported
     - Experimental
   * - ``robocasa``
     - Supported
     - Unsupported
   * - ``robotwin``
     - Supported
     - Unsupported

Installation and Resources
--------------------------

The :download:`installation guide <../../../../tools/rl/INSTALL.md>` starts
from a fresh checkout and covers system libraries, isolated Python environments,
models, SAM3, simulator assets and troubleshooting. It uses your checkout and
data directory, with no cluster-specific paths. Copy
``tools/rl/config.example.sh`` to ``.rl-env.sh`` and customize its settings.

The common environment runs the planner, SAM3 and LIBERO. RoboCasa and RoboTwin
use separate simulator environments; the model server has its own environment.
Linux NVIDIA EGL is required; RoboTwin also requires Vulkan and a CUDA toolkit
for cuRobo. Install assets and verify imports before running episodes.

Run a Task
----------

The :download:`first-task guide <../../../../tools/rl/README.md>` provides
complete service startup and four commands: LIBERO multiturn, LIBERO ZAP,
RoboCasa multiturn and RoboTwin multiturn. Start with one episode.

After configuring and starting the model and shared SAM3 services, from the
checkout root:

.. code-block:: bash

   source .venv/bin/activate
   source .rl-env.sh
   source tools/rl/env.sh
   bash tools/rl/run.sh --simulator libero --workflow multiturn \
     --suite libero_object --task 0 --seed 1 --cuda-device 1

``--cuda-device`` selects the simulator GPU independently of model/SAM3 GPUs.
RoboCasa uses ``--task-name`` and ``--split``; RoboTwin uses ``--task-name``,
``--task-config`` and a published valid seed. The first-task guide contains
working examples and links to task manifests.

LIBERO multiturn overview
-------------------------

Multiturn retains conversation and tool results, with context compaction.
``view_env`` returns recorded images. ``segment`` and ``back_project`` measure
visible surfaces on a matching frame. The shared motion tool names are
``move_to``, ``set_gripper`` and ``finish``. RoboCasa adds ``move_base`` with
local planar velocities and yaw rate. RoboTwin adds anatomical arm selection,
labelled views and paired commands. Native task language is read directly
from the simulator; shared prompts describe robots and workflows.

ZAP snapshot loop
-----------------

LIBERO ZAP sends the current images, same-frame SAM3 boxes/3D measurements,
robot state, visible todo and last execution result on each request. Earlier
reasoning and conversation are absent; no history compaction runs. The first
request calls ``change_query``, the second ``update_todo``; later requests
execute exactly one tool. There is no ``view_env``, ``segment`` or
``back_project`` tool. This experimental implementation has no established
success-rate or speed guarantee.

Planner Configuration
---------------------

The default budgets are 100 model requests, 10000 simulator steps,
32768 output tokens per request, a 262144-token context, multiturn compaction
at 70%, and a 7200-second planner timeout. The batch launcher also enforces a
14400-second episode timeout. LIBERO multiturn retains ``simple``, ``base``,
``base_gripper`` and ``base_gripper_camera`` prompt variants.

The example configuration explicitly sets presence penalty 1.5 and repetition
penalty 1.0; the launcher leaves them unset otherwise. Thinking sampling uses
temperature 1.0, top-p 0.95 and top-k 20. Qwen3.8 defaults to ``xhigh`` effort.
The :download:`reference <../../../../tools/rl/REFERENCE.md>` lists all service,
planner and batch overrides and describes the actual saved requests.

View Results
------------

``result.json`` records native success, errors, turns and tokens.
``states.json`` retains measured states and native termination flags.
Camera/depth artifacts, ``episode.mp4``, ``rollout.json``, native transcripts
and ``api_trace/`` retain the execution evidence. A clean exit alone proves
neither success nor a complete training record. The batch launcher audits
complete requests/responses and image hashes before publishing records.

Collect Tasks and Seeds
-----------------------

The reference describes bounded collection, valid/evaluated indexes,
infrastructure retries and interruption/resume behavior. Use a fresh batch
directory, start with one episode per GPU, and size concurrency from measured
scene memory. ``tools/rl/jobs`` contains example task/seed manifests.

Remote simulator host
---------------------

Adaptive multiturn can place motion, rendering and SAM3 on a separate backend
using ``python -m robots.rl.remote_manager`` and
``RPENT_REMOTE_TOOLKIT_ENDPOINT``. Both hosts need matching source. The
reference covers installation on each host, private routing, GPU capacities,
source/task/budget checks, file hashes and lease cleanup. Remote ZAP is
unsupported.

RLinf Training
--------------

The reference preserves offline installation and RLinf/SFT integration details.
RLinf owns policy training/checkpoints; RPent owns simulator/tool execution.
Its episode entry is ``robots.rl.rl_episode``. Dataset encoding belongs to
``ms-swift/examples/rpent`` and must preserve saved requests and image order.
