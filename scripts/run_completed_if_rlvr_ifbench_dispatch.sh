#!/bin/bash
# Dispatch lm-eval IFBench jobs for IF-RLVR training runs (same RESULTS_ROOT as
# run_completed_open_instruct_ifbench_sbatch.sh: ifbench_completed_open_instruct).
#
# A run is eligible only when training wrote a final HF tree with .checkpoint_complete
# (no intermediate DeepSpeed global_step* evals). Set ONLY_FINAL=0 on the wrapper to allow
# step checkpoints again.
# Entries that already have RESULTS_ROOT/.../.eval_complete are skipped unless FORCE=1.
# Run folders whose basename ends with "test" are always skipped (smoke/debug runs).
#
# Typical usage (login node, Unity scratch defaults):
#   bash lm-evaluation-harness/scripts/run_completed_if_rlvr_ifbench_dispatch.sh
#
# Preview only:
#   bash lm-evaluation-harness/scripts/run_completed_if_rlvr_ifbench_dispatch.sh --list
#
# Narrow to recent experiment folders:
#   RUN_GLOB='Qwen3.5-9B_IF_multi_constraints*' bash .../run_completed_if_rlvr_ifbench_dispatch.sh
#
# Environment overrides (see run_completed_open_instruct_ifbench_sbatch.sh for full list):
#   SCRATCH_ROOT, OUTPUT_ROOT, RESULTS_ROOT, RUN_GLOB, ONLY_FINAL, EVAL_STEP, PREFER_FINAL,
#   STEP_POLICY, FORCE, MAX_PARALLEL, WANDB_PROJECT, WANDB_ENTITY, WANDB_TRAINING_PROJECT,
#   TASKS, LIMIT

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WRAPPER="${WRAPPER:-${SCRIPT_DIR}/run_completed_open_instruct_ifbench_sbatch.sh}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/eval_results/ifbench_completed_open_instruct}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH_ROOT}/eval_manifests}"

# Match grpo_fast defaults / docs unless overridden.
WANDB_PROJECT="${WANDB_PROJECT:-open_instruct_internal}"
WANDB_TRAINING_PROJECT="${WANDB_TRAINING_PROJECT:-${WANDB_PROJECT}}"

# IF-RLVR checkpoints usually land as HF final.
PREFER_FINAL="${PREFER_FINAL:-1}"
ONLY_FINAL="${ONLY_FINAL:-1}"

export PROJECT_ROOT
export SCRATCH_ROOT
export OUTPUT_ROOT
export RESULTS_ROOT
export MANIFEST_DIR
export WANDB_PROJECT
export WANDB_TRAINING_PROJECT
export PREFER_FINAL
export ONLY_FINAL

exec bash "${WRAPPER}" "$@"
