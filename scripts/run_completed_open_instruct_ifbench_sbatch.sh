#!/bin/bash
#SBATCH --job-name=ifbench-eval
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [ifbench-eval] $*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

usage() {
  cat <<'USAGE'
Evaluate completed open-instruct checkpoints on IFBench with SLURM.

Default login-node usage:
  bash lm-evaluation-harness/scripts/run_completed_open_instruct_ifbench_sbatch.sh

Useful modes:
  --list          Print the discovered eval manifest and do not submit jobs.
  --manifest PATH Submit a SLURM array for an existing manifest TSV (skip auto-discovery).
  --run-manifest  Run one manifest entry. Used internally by the SLURM array.

Common environment overrides:
  OUTPUT_ROOT=/scratch4/.../outputs
  RUN_GLOB='*IF_multi_constraints*'
  EVAL_STEP=1000
  PREFER_FINAL=1
  ONLY_FINAL=1            # skip DeepSpeed global_step*; require HF final + .checkpoint_complete
  STEP_POLICY=exact        # exact or at_or_after
  FORCE=1                 # re-run entries with .eval_complete markers
  MAX_PARALLEL=0          # 0 = no SLURM array throttle; N>0 caps concurrent tasks at N
  EVAL_SLURM_GPUS=1       # SLURM --gpus per array task
  DATA_PARALLEL_SIZE=1    # vLLM data_parallel_size (use with EVAL_SLURM_GPUS)
  EVAL_SLURM_PARTITION=gpu,gpu-preempt
  WANDB_PROJECT=open_instruct_internal
  WANDB_ENTITY=<entity>
  WANDB_STRICT_TRAINING_CONFIG=1
  TASKS=ifeval,ifbench
  LIMIT=10                # quick smoke test
USAGE
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HARNESS_ROOT="${HARNESS_ROOT:-${PROJECT_ROOT}/lm-evaluation-harness}"
HARNESS_UV_PYTHON="${HARNESS_UV_PYTHON:-3.12}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/outputs}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/eval_results/ifbench_completed_open_instruct}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH_ROOT}/eval_manifests}"
HARNESS_UV_PROJECT_ENVIRONMENT="${HARNESS_UV_PROJECT_ENVIRONMENT:-${SCRATCH_ROOT}/venvs/lm-evaluation-harness-py312}"

RUN_GLOB="${RUN_GLOB:-*}"
EVAL_STEP="${EVAL_STEP:-1000}"
PREFER_FINAL="${PREFER_FINAL:-1}"
ONLY_FINAL="${ONLY_FINAL:-0}"
STEP_POLICY="${STEP_POLICY:-exact}"
FORCE="${FORCE:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"

EVAL_SLURM_PARTITION="${EVAL_SLURM_PARTITION:-gpu,gpu-preempt}"
EVAL_SLURM_JOB_NAME="${EVAL_SLURM_JOB_NAME:-ifbench-eval}"
EVAL_SLURM_GPUS="${EVAL_SLURM_GPUS:-1}"
EVAL_SLURM_CONSTRAINT="${EVAL_SLURM_CONSTRAINT:-a100-80g}"
EVAL_SLURM_CPUS_PER_GPU="${EVAL_SLURM_CPUS_PER_GPU:-10}"
EVAL_SLURM_MEM="${EVAL_SLURM_MEM:-100G}"
EVAL_SLURM_TIME="${EVAL_SLURM_TIME:-24:00:00}"

CUDA_MODULE="${CUDA_MODULE:-cuda/13.1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
THINK_END_TOKEN="${THINK_END_TOKEN:-auto}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32000}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
GEN_TOP_P="${GEN_TOP_P:-0.95}"
GEN_TOP_K="${GEN_TOP_K:-auto}"
GEN_MIN_P="${GEN_MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"
LIMIT="${LIMIT:-}"
TASKS="${TASKS:-ifeval,ifbench,aime25,aime26,hendrycks_math500,humaneval_instruct,humaneval_plus_instruct,mbpp_instruct,mbpp_plus_instruct}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-open_instruct_internal}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TRAINING_PROJECT="${WANDB_TRAINING_PROJECT:-${WANDB_PROJECT}}"
WANDB_TRAINING_ENTITY="${WANDB_TRAINING_ENTITY:-${WANDB_ENTITY}}"
WANDB_MIRROR_TRAINING_CONFIG="${WANDB_MIRROR_TRAINING_CONFIG:-1}"
WANDB_STRICT_TRAINING_CONFIG="${WANDB_STRICT_TRAINING_CONFIG:-1}"
WANDB_API_TIMEOUT="${WANDB_API_TIMEOUT:-180}"
WANDB_TAGS="${WANDB_TAGS:-}"

EVAL_RUN_FAMILY="${EVAL_RUN_FAMILY:-training_checkpoint}"
EVAL_BENCHMARK_GROUP="${EVAL_BENCHMARK_GROUP:-ifbench}"
EVAL_DATASET_KEY="${EVAL_DATASET_KEY:-}"
EVAL_MODEL_SOURCE="${EVAL_MODEL_SOURCE:-}"
BASE_MODEL_SEED="${BASE_MODEL_SEED:-0}"

MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-}"
GEN_KWARGS_EXTRA="${GEN_KWARGS_EXTRA:-}"
LMEVAL_EXTRA_ARGS="${LMEVAL_EXTRA_ARGS:-}"

MODE="submit"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --list)
      MODE="list"
      shift
      ;;
    --manifest)
      [ -n "${2:-}" ] || die "--manifest requires a path"
      MANIFEST_PATH="$(readlink -f "$2")"
      [ -f "${MANIFEST_PATH}" ] || die "Manifest not found: ${MANIFEST_PATH}"
      MODE="submit-existing"
      shift 2
      ;;
    --run-manifest)
      MODE="run-manifest"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

# Only switch to worker mode inside a SLURM array task. A leaked SLURM_JOB_ID on the
# login node must not turn a submit invocation into a local run-all.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "${MODE}" = "submit" ]; then
  MODE="run-manifest"
fi

setup_common_dirs() {
  mkdir -p "${PROJECT_ROOT}/logs" "${RESULTS_ROOT}" "${MANIFEST_DIR}"
}

setup_slurm_cuda_visible_devices() {
  local want="${DATA_PARALLEL_SIZE:-${EVAL_SLURM_GPUS:-1}}"
  local -a ids=()
  local count=0
  local entry

  [ "${want}" -gt 1 ] || return 0

  if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    IFS=',' read -ra _slurm_job_gpus <<< "${SLURM_JOB_GPUS}"
    for entry in "${_slurm_job_gpus[@]}"; do
      ids+=("${entry%%:*}")
    done
  elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
    IFS=',' read -ra ids <<< "${SLURM_STEP_GPUS}"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra ids <<< "${CUDA_VISIBLE_DEVICES}"
  fi

  count="${#ids[@]}"
  if [ "${count}" -lt "${want}" ] && [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE}" -ge "${want}" ]; then
    ids=()
    local i
    for ((i = 0; i < want; i++)); do
      ids+=("${i}")
    done
    count="${want}"
  fi

  if [ "${count}" -ge "${want}" ]; then
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${ids[*]:0:want}")"
    log "Set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} for data_parallel_size=${want}"
    return 0
  fi

  die "Need ${want} visible GPU(s) for data_parallel_size=${want}; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
}

setup_job_env() {
  setup_common_dirs
  local task_cache_root

  if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE}"
  fi
  if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  fi

  setup_slurm_cuda_visible_devices

  task_cache_root="${SLURM_TMPDIR:-/tmp}/${USER:-user}/ifbench_eval_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
  export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${task_cache_root}/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_ROOT}/cache/triton}"
  export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${task_cache_root}/evaluate}"
  export HF_METRICS_CACHE="${HF_METRICS_CACHE:-${task_cache_root}/metrics}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${task_cache_root}/modules}"
  export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
  export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
  export VLLM_USE_V1="${VLLM_USE_V1:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
  export PYTHONUNBUFFERED=1

  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TRITON_CACHE_DIR}" "${HF_EVALUATE_CACHE}" "${HF_METRICS_CACHE}" "${HF_MODULES_CACHE}"
}

sanitize_for_path() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  echo "${value}"
}

is_hf_model_dir() {
  local dir="$1"
  [ -d "${dir}" ] || return 1
  [ -f "${dir}/config.json" ] || return 1
  [ -f "${dir}/tokenizer.json" ] || [ -f "${dir}/tokenizer_config.json" ] || return 1
  compgen -G "${dir}/model*.safetensors" >/dev/null || compgen -G "${dir}/pytorch_model*.bin" >/dev/null || return 1
}

is_ds_checkpoint_dir() {
  local dir="$1"
  [ -d "${dir}" ] || return 1
  [ -f "${dir}/mp_rank_00_model_states.pt" ] || [ -f "${dir}/zero_pp_rank_0_mp_rank_00_model_states.pt" ] || return 1
  compgen -G "${dir}/*_optim_states.pt" >/dev/null || return 1
}

is_ds_universal_checkpoint_dir() {
  local dir="$1"
  [ -d "${dir}" ] || return 1
  [ -d "${dir}/zero" ] || return 1
  [ -f "${dir}/mp_rank_00_model_states.pt" ] || return 1
}

find_final_hf_dir() {
  local run_dir="$1"
  local marker dir selected=""

  shopt -s nullglob
  for marker in "${run_dir}"/*/.checkpoint_complete; do
    dir="${marker%/.checkpoint_complete}"
    if is_hf_model_dir "${dir}"; then
      selected="${dir}"
    fi
  done
  shopt -u nullglob

  [ -n "${selected}" ] && echo "${selected}"
}

find_step_checkpoint_dir() {
  local run_dir="$1"
  local exact="${run_dir}/global_step${EVAL_STEP}"
  if is_ds_checkpoint_dir "${exact}"; then
    echo "${exact}"
    return 0
  fi

  if [ "${STEP_POLICY}" != "at_or_after" ]; then
    return 1
  fi

  local d name n best="" best_n=0
  shopt -s nullglob
  for d in "${run_dir}"/global_step*; do
    name="${d##*/}"
    n="${name#global_step}"
    [[ "${n}" =~ ^[0-9]+$ ]] || continue
    if (( n >= EVAL_STEP )) && is_ds_checkpoint_dir "${d}"; then
      if [ -z "${best}" ] || (( n < best_n )); then
        best="${d}"
        best_n="${n}"
      fi
    fi
  done
  shopt -u nullglob

  [ -n "${best}" ] && echo "${best}"
}

infer_model_name() {
  local run_dir="$1"
  local name
  name="$(basename "${run_dir}")"

  if [ -n "${MODEL_NAME:-}" ]; then
    echo "${MODEL_NAME}"
    return 0
  fi

  case "${name}" in
    *Qwen3.5-0.8B*) echo "Qwen/Qwen3.5-0.8B" ;;
    *Qwen3.5-2B*) echo "Qwen/Qwen3.5-2B" ;;
    *Qwen3.5-4B*) echo "Qwen/Qwen3.5-4B" ;;
    *Qwen3.5-9B*) echo "Qwen/Qwen3.5-9B" ;;
    *Qwen3-0.6B*) echo "Qwen/Qwen3-0.6B" ;;
    *Qwen3-1.7B*) echo "Qwen/Qwen3-1.7B" ;;
    *Qwen3-4B*) echo "Qwen/Qwen3-4B" ;;
    *Qwen3-8B*) echo "Qwen/Qwen3-8B" ;;
    *gemma-4-E2B*|*gemma_4_e2b*|*Gemma_4_E2B*) echo "google/gemma-4-E2B-it" ;;
    *Llama-3.1-Tulu-3-8B*|*tulu3.1_8b*|*tulu3_8b*) echo "allenai/Llama-3.1-Tulu-3-8B-DPO" ;;
    *) return 1 ;;
  esac
}

find_training_run_name() {
  local run_dir="$1"
  local final_dir="${2:-}"
  local d selected=""

  if [ -n "${final_dir}" ]; then
    basename "${final_dir}"
    return 0
  fi

  shopt -s nullglob
  for d in "${run_dir}"/*__*__*; do
    [ -d "${d}" ] && selected="${d}"
  done
  shopt -u nullglob

  if [ -n "${selected}" ]; then
    basename "${selected}"
  else
    basename "${run_dir}"
  fi
}

append_candidate() {
  local manifest="$1"
  local run_dir="$2"
  local final_dir="" step_dir="" kind="" checkpoint_ref="" hf_dir="" model_name="-"
  local checkpoint_tag train_run_name eval_name result_dir result_slug run_slug

  [ -d "${run_dir}" ] || return 0
  [ "$(basename "${run_dir}")" != "autotune" ] || return 0
  case "$(basename "${run_dir}")" in
    *test)
      log "Skipping $(basename "${run_dir}"): run name ends with 'test'"
      return 0
      ;;
  esac

  final_dir="$(find_final_hf_dir "${run_dir}" || true)"
  if [ "${ONLY_FINAL}" = "1" ]; then
    if [ -z "${final_dir}" ]; then
      return 0
    fi
    kind="hf_final"
    checkpoint_ref="${final_dir}"
    hf_dir="${final_dir}"
    checkpoint_tag="final"
    model_name="$(infer_model_name "${run_dir}" || echo "-")"
  elif [ "${PREFER_FINAL}" = "1" ] && [ -n "${final_dir}" ]; then
    kind="hf_final"
    checkpoint_ref="${final_dir}"
    hf_dir="${final_dir}"
    checkpoint_tag="final"
    model_name="$(infer_model_name "${run_dir}" || echo "-")"
  else
    step_dir="$(find_step_checkpoint_dir "${run_dir}" || true)"
    if [ -n "${step_dir}" ]; then
      kind="deepspeed"
      checkpoint_ref="${step_dir}"
      checkpoint_tag="$(basename "${step_dir}")"
      hf_dir="${run_dir}/hf_${checkpoint_tag}"
      if ! model_name="$(infer_model_name "${run_dir}")"; then
        log "Skipping $(basename "${run_dir}"): cannot infer --model-name for DeepSpeed export. Set MODEL_NAME to override."
        return 0
      fi
    elif [ -n "${final_dir}" ]; then
      kind="hf_final"
      checkpoint_ref="${final_dir}"
      hf_dir="${final_dir}"
      checkpoint_tag="final"
      model_name="$(infer_model_name "${run_dir}" || echo "-")"
    else
      return 0
    fi
  fi

  train_run_name="$(find_training_run_name "${run_dir}" "${final_dir}")"
  eval_name="eval_${train_run_name}_${checkpoint_tag}_ifbench"
  run_slug="$(sanitize_for_path "$(basename "${run_dir}")")"
  result_slug="$(sanitize_for_path "${eval_name}")"
  result_dir="${RESULTS_ROOT}/${run_slug}/${result_slug}"

  if [ "${FORCE}" != "1" ] && [ -f "${result_dir}/.eval_complete" ]; then
    log "Skipping ${eval_name}: ${result_dir}/.eval_complete exists"
    return 0
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${kind}" "${run_dir}" "${checkpoint_ref}" "${hf_dir}" "${model_name}" \
    "${checkpoint_tag}" "${train_run_name}" "${eval_name}" "${result_dir}" >> "${manifest}"
}

build_manifest() {
  local manifest="$1"
  local run_dir

  : > "${manifest}"
  shopt -s nullglob
  for run_dir in "${OUTPUT_ROOT}"/${RUN_GLOB}; do
    append_candidate "${manifest}" "${run_dir}"
  done
  shopt -u nullglob
}

manifest_count() {
  awk 'END { print NR + 0 }' "$1"
}

print_manifest() {
  local manifest="$1"
  local i=0 kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir

  while IFS=$'\t' read -r kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir; do
    i=$((i + 1))
    printf '%d\t%s\t%s\t%s\t%s\t%s\n' "${i}" "${kind}" "$(basename "${run_dir}")" "${checkpoint_tag}" "${model_name}" "${eval_name}"
  done < "${manifest}"
}

resolve_slurm_constraint() {
  local constraint="${EVAL_SLURM_CONSTRAINT}"
  if [ "${EVAL_SLURM_GPUS}" -gt 1 ]; then
    case "${constraint}" in
      *mpi*) ;;
      *) constraint="mpi&${constraint}" ;;
    esac
  fi
  echo "${constraint}"
}

submit_manifest() {
  local manifest="$1"
  local count array_arg slurm_constraint

  slurm_constraint="$(resolve_slurm_constraint)"
  count="$(manifest_count "${manifest}")"
  if [ "${count}" -eq 0 ]; then
    log "No completed eval candidates found under ${OUTPUT_ROOT}/${RUN_GLOB}"
    return 0
  fi

  array_arg="1-${count}"
  if [ "${MAX_PARALLEL}" != "0" ]; then
    array_arg="${array_arg}%${MAX_PARALLEL}"
  fi

  log "Submitting ${count} eval task(s) from manifest ${manifest}"
  printf '%s\n' "${manifest}" > "${MANIFEST_DIR}/active_manifest.path"
  (
    cd "${PROJECT_ROOT}"
    export MANIFEST_PATH="${manifest}"
    export PROJECT_ROOT="${PROJECT_ROOT}"
    export HARNESS_ROOT="${HARNESS_ROOT}"
    sbatch \
      --job-name="${EVAL_SLURM_JOB_NAME}" \
      --partition="${EVAL_SLURM_PARTITION}" \
      --gpus="${EVAL_SLURM_GPUS}" \
      --constraint="${slurm_constraint}" \
      --cpus-per-gpu="${EVAL_SLURM_CPUS_PER_GPU}" \
      --mem="${EVAL_SLURM_MEM}" \
      --time="${EVAL_SLURM_TIME}" \
      --array="${array_arg}" \
      --export=ALL \
      "${SCRIPT_PATH}" --run-manifest
  )
}

ensure_multimodal_processor_files() {
  local hf_dir="$1"
  local model_name="$2"

  [ -f "${hf_dir}/config.json" ] || return 0
  grep -q 'Gemma4ForConditionalGeneration' "${hf_dir}/config.json" 2>/dev/null || return 0
  [ -f "${hf_dir}/processor_config.json" ] && return 0
  [ -n "${model_name}" ] && [ "${model_name}" != "-" ] || die "MODEL_NAME is required to patch Gemma4 processor files in ${hf_dir}"

  log "Copying processor_config.json from ${model_name} into ${hf_dir} (required by vLLM Gemma4 multimodal init)"
  (
    cd "${PROJECT_ROOT}"
    uv run python - <<'PY' "${model_name}" "${hf_dir}"
import shutil
import sys
from huggingface_hub import hf_hub_download

model_name, hf_dir = sys.argv[1:3]
src = hf_hub_download(model_name, "processor_config.json")
shutil.copy2(src, f"{hf_dir}/processor_config.json")
print(f"Wrote {hf_dir}/processor_config.json", flush=True)
PY
  )
}

ensure_gemma4_vllm_weights() {
  local hf_dir="$1"
  local model_name="$2"

  [ -f "${hf_dir}/config.json" ] || return 0
  grep -q 'Gemma4ForConditionalGeneration' "${hf_dir}/config.json" 2>/dev/null || return 0
  [ -f "${hf_dir}/model.safetensors" ] || return 0
  [ -n "${model_name}" ] && [ "${model_name}" != "-" ] || die "MODEL_NAME is required to patch Gemma4 weights in ${hf_dir}"

  log "Patching Gemma4 HF weights for vLLM (fill missing keys from ${model_name}): ${hf_dir}"
  (
    cd "${PROJECT_ROOT}"
    uv run python scripts/patch_gemma4_hf_export_for_vllm.py \
      --hf-dir "${hf_dir}" \
      --base-model-name "${model_name}" >&2
  )
}

ensure_hf_model() {
  local kind="$1"
  local checkpoint_ref="$2"
  local hf_dir="$3"
  local model_name="$4"

  if [ "${kind}" = "hf_hub" ]; then
    [ -n "${checkpoint_ref}" ] || die "Missing Hugging Face model id"
    echo "${checkpoint_ref}"
    return 0
  fi

  if [ "${kind}" = "hf_final" ]; then
    is_hf_model_dir "${hf_dir}" || die "Final model directory is incomplete: ${hf_dir}"
    echo "${hf_dir}"
    return 0
  fi

  if is_hf_model_dir "${hf_dir}"; then
    if grep -q '"language_model\.' "${hf_dir}/model.safetensors.index.json" 2>/dev/null; then
      log "Remapping HF export keys for vLLM compatibility: ${hf_dir}"
      (
        cd "${PROJECT_ROOT}"
        uv run python scripts/remap_hf_export_keys_for_vllm.py \
          --hf-dir "${hf_dir}" \
          --universal-zero-dir "${checkpoint_ref}/zero" \
          --dtype "${DTYPE}" >&2
      )
    fi
    log "Using existing HF export: ${hf_dir}"
    echo "${hf_dir}"
    return 0
  fi

  [ "${model_name}" != "-" ] || die "MODEL_NAME is required for DeepSpeed export of ${checkpoint_ref}"
  if is_ds_universal_checkpoint_dir "${checkpoint_ref}"; then
    log "Converting universal checkpoint ${checkpoint_ref} to HF format at ${hf_dir} with base model ${model_name}"
    (
      cd "${PROJECT_ROOT}"
      uv run python scripts/export_universal_to_hf.py \
        --model-name "${model_name}" \
        --checkpoint-dir "${checkpoint_ref}" \
        --output-dir "${hf_dir}" \
        --dtype "${DTYPE}" >&2
    )
  else
    log "Converting ZeRO checkpoint ${checkpoint_ref} to HF format at ${hf_dir} with base model ${model_name}"
    (
      cd "${PROJECT_ROOT}"
      uv run python scripts/export_zero_to_hf.py \
        --model-name "${model_name}" \
        --checkpoint-dir "${checkpoint_ref}" \
        --output-dir "${hf_dir}" >&2
    )
  fi

  is_hf_model_dir "${hf_dir}" || die "HF export did not produce a complete model directory: ${hf_dir}"
  echo "${hf_dir}"
}

write_wandb_config_json() {
  local output_json="$1"
  local kind="$2"
  local run_dir="$3"
  local checkpoint_ref="$4"
  local hf_dir="$5"
  local model_name="$6"
  local checkpoint_tag="$7"
  local train_run_name="$8"
  local eval_name="$9"
  local result_dir="${10}"
  local effective_think_end_token="${11}"
  local effective_gen_top_k="${12}"

  (
    cd "${HARNESS_ROOT}"
    UV_PROJECT_ENVIRONMENT="${HARNESS_UV_PROJECT_ENVIRONMENT}" \
      uv run --python "${HARNESS_UV_PYTHON}" --extra vllm --extra ifbench --extra ifeval --extra hf --extra wandb python - \
      "${output_json}" \
      "${WANDB_TRAINING_PROJECT}" \
      "${WANDB_TRAINING_ENTITY}" \
      "${train_run_name}" \
      "${WANDB_MIRROR_TRAINING_CONFIG}" \
      "${WANDB_STRICT_TRAINING_CONFIG}" \
      "${kind}" \
      "${run_dir}" \
      "${checkpoint_ref}" \
      "${hf_dir}" \
      "${model_name}" \
      "${checkpoint_tag}" \
      "${eval_name}" \
      "${result_dir}" \
      "${effective_think_end_token}" \
      "${effective_gen_top_k}" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    output_json,
    training_project,
    training_entity,
    train_run_name,
    mirror_training_config,
    strict_training_config,
    kind,
    run_dir,
    checkpoint_ref,
    hf_dir,
    model_name,
    checkpoint_tag,
    eval_name,
    result_dir,
    effective_think_end_token,
    effective_gen_top_k,
) = sys.argv[1:]


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def find_training_run():
    import wandb

    api = wandb.Api(timeout=int(os.environ.get("WANDB_API_TIMEOUT", "180")))
    entity = training_entity or api.default_entity
    if not entity:
        raise RuntimeError("WANDB_TRAINING_ENTITY is unset and wandb.Api().default_entity is empty")

    path = f"{entity}/{training_project}"
    filters_to_try = [
        {"display_name": train_run_name},
        {"name": train_run_name},
        {"config.run_name": train_run_name},
    ]
    errors = []
    for filt in filters_to_try:
        try:
            runs = list(api.runs(path, filters=filt, per_page=20))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{filt}: {exc}")
            continue
        exact = [
            run
            for run in runs
            if run.name == train_run_name
            or run.id == train_run_name
            or (run.config or {}).get("run_name") == train_run_name
        ]
        if exact:
            return exact[0]

    if truthy(os.environ.get("WANDB_SCAN_FOR_TRAINING_RUN", "0")):
        try:
            for run in api.runs(path, per_page=int(os.environ.get("WANDB_SCAN_PER_PAGE", "400"))):
                if run.name == train_run_name or run.id == train_run_name or (run.config or {}).get("run_name") == train_run_name:
                    return run
        except Exception as exc:  # noqa: BLE001
            errors.append(f"project scan: {exc}")

    error_text = "; ".join(errors)
    raise RuntimeError(
        f"Could not find W&B training run {train_run_name!r} in {path}. "
        f"Set WANDB_TRAINING_PROJECT/WANDB_TRAINING_ENTITY if needed. {error_text}"
    )


config = {}
training_run = None
if truthy(mirror_training_config):
    try:
        training_run = find_training_run()
        config.update(training_run.config or {})
        print(
            f"Mirroring W&B config from training run "
            f"{training_run.entity}/{training_run.project}/{training_run.id} ({training_run.name})",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        if truthy(strict_training_config):
            raise SystemExit(str(exc)) from exc
        print(f"WARNING: {exc}; writing eval-only W&B config", file=sys.stderr)

def model_key(model_id: str) -> str:
    return (
        model_id.replace("/", "__")
        .replace(".", "p")
        .replace("-", "_")
        .replace(" ", "_")
    )


def model_display(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def model_family(model_id: str) -> str:
    leaf = model_id.rsplit("/", 1)[-1].lower()
    if leaf.startswith("qwen3.5"):
        return "qwen3.5"
    if leaf.startswith("qwen3"):
        return "qwen3"
    if leaf.startswith("gemma-4"):
        return "gemma-4"
    return leaf.split("-", 1)[0]


def model_vendor(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else ""


def model_size_b(model_id: str) -> float | None:
    import re

    leaf = model_id.rsplit("/", 1)[-1]
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)B", leaf, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Gemma 4 effective sizes are encoded as E2B/E4B.
    m = re.search(r"E([0-9]+(?:\.[0-9]+)?)B", leaf, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def training_global_step(checkpoint_tag_value: str) -> int | None:
    import re

    m = re.search(r"global_step(\d+)", checkpoint_tag_value)
    return int(m.group(1)) if m else None


eval_run_family = os.environ.get("EVAL_RUN_FAMILY", "training_checkpoint")
is_base_model = kind == "hf_hub" or eval_run_family == "base_model"
training_step = training_global_step(checkpoint_tag)
config.setdefault("model_name_or_path", model_name if model_name != "-" else hf_dir)
resolved_model_id = model_name if model_name != "-" else hf_dir
if is_base_model:
    size_b = model_size_b(resolved_model_id)
    config.update(
        {
            "run_name": eval_name,
            "dataset_mixer_list": [],
            "learning_rate": None,
            "seed": int(os.environ.get("BASE_MODEL_SEED", "0")),
            "model_name_or_path": resolved_model_id,
            "eval_model_key": model_key(resolved_model_id),
            "eval_model_display_name": model_display(resolved_model_id),
            "eval_model_family": model_family(resolved_model_id),
            "eval_model_vendor": model_vendor(resolved_model_id),
            "eval_model_size_b": size_b,
            "eval_approach_kind": "base_model",
            "eval_training_dataset_key": "none",
        }
    )

config.update(
    {
        "eval_job_type": "eval",
        "eval_run_family": eval_run_family,
        "eval_is_base_model": is_base_model,
        "eval_model_source": os.environ.get("EVAL_MODEL_SOURCE") or ("hf_hub" if kind == "hf_hub" else kind),
        "eval_benchmark_group": os.environ.get("EVAL_BENCHMARK_GROUP", "ifbench"),
        "eval_dataset_key": os.environ.get("EVAL_DATASET_KEY", ""),
        "eval_name": eval_name,
        "eval_tasks": os.environ.get("TASKS", ""),
        "eval_checkpoint_kind": kind,
        "eval_checkpoint_tag": checkpoint_tag,
        "eval_training_global_step": training_step,
        "eval_checkpoint_path": checkpoint_ref,
        "eval_hf_model_path": hf_dir,
        "eval_results_dir": result_dir,
        "eval_output_root": run_dir,
        "eval_base_model": model_name,
        "eval_max_gen_toks": int(os.environ.get("MAX_GEN_TOKS", "32000")),
        "eval_gen_temperature": float(os.environ.get("GEN_TEMPERATURE", "1.0")),
        "eval_gen_top_p": float(os.environ.get("GEN_TOP_P", "0.95")),
        "eval_gen_top_k": int(effective_gen_top_k),
        "eval_gen_min_p": float(os.environ.get("GEN_MIN_P", "0.0")),
        "eval_presence_penalty": float(os.environ.get("PRESENCE_PENALTY", "1.5")),
        "eval_repetition_penalty": float(os.environ.get("REPETITION_PENALTY", "1.0")),
        "eval_think_end_token": effective_think_end_token,
        "eval_batch_size": os.environ.get("BATCH_SIZE", "auto"),
        "eval_num_fewshot": int(os.environ.get("NUM_FEWSHOT", "0")),
        "eval_apply_chat_template": True,
        "eval_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "eval_slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "training_run_name": train_run_name,
        "eval_wandb_tags": os.environ.get("WANDB_TAGS", ""),
    }
)

if training_run is not None:
    config.update(
        {
            "training_wandb_entity": training_run.entity,
            "training_wandb_project": training_run.project,
            "training_wandb_run_id": training_run.id,
            "training_wandb_run_path": f"{training_run.entity}/{training_run.project}/{training_run.id}",
            "training_wandb_run_url": training_run.url,
        }
    )

Path(output_json).parent.mkdir(parents=True, exist_ok=True)
Path(output_json).write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
PY
  )
}

resolve_think_end_token() {
  local model_name="$1"
  if [ "${THINK_END_TOKEN}" != "auto" ]; then
    echo "${THINK_END_TOKEN}"
    return 0
  fi

  case "${model_name}" in
    google/gemma-4-*) echo "<channel|>" ;;
    *) echo "</think>" ;;
  esac
}

resolve_gen_top_k() {
  local model_name="$1"
  if [ "${GEN_TOP_K}" != "auto" ]; then
    echo "${GEN_TOP_K}"
    return 0
  fi

  case "${model_name}" in
    google/gemma-4-*) echo "64" ;;
    *) echo "20" ;;
  esac
}

run_eval() {
  local kind="$1"
  local run_dir="$2"
  local checkpoint_ref="$3"
  local hf_dir="$4"
  local model_name="$5"
  local checkpoint_tag="$6"
  local train_run_name="$7"
  local eval_name="$8"
  local result_dir="$9"
  local resolved_hf_dir effective_think_end_token effective_gen_top_k

  setup_job_env
  resolved_hf_dir="$(ensure_hf_model "${kind}" "${checkpoint_ref}" "${hf_dir}" "${model_name}")"
  ensure_multimodal_processor_files "${resolved_hf_dir}" "${model_name}"
  ensure_gemma4_vllm_weights "${resolved_hf_dir}" "${model_name}"
  effective_think_end_token="$(resolve_think_end_token "${model_name}")"
  effective_gen_top_k="$(resolve_gen_top_k "${model_name}")"
  mkdir -p "${result_dir}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${kind}" "${run_dir}" "${checkpoint_ref}" "${resolved_hf_dir}" "${model_name}" \
    "${checkpoint_tag}" "${train_run_name}" "${eval_name}" "${result_dir}" > "${result_dir}/eval_candidate.tsv"

  local model_args=(
    "pretrained=${resolved_hf_dir}"
    "dtype=${DTYPE}"
    "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    "data_parallel_size=${DATA_PARALLEL_SIZE}"
    "max_gen_toks=${MAX_GEN_TOKS}"
  )
  if [ -n "${effective_think_end_token}" ]; then
    model_args+=("think_end_token=${effective_think_end_token}")
  fi
  if [ -n "${MODEL_ARGS_EXTRA}" ]; then
    local extra_model_args=()
    read -r -a extra_model_args <<< "${MODEL_ARGS_EXTRA}"
    model_args+=("${extra_model_args[@]}")
  fi

  local gen_kwargs=(
    "temperature=${GEN_TEMPERATURE}"
    "top_p=${GEN_TOP_P}"
    "top_k=${effective_gen_top_k}"
    "min_p=${GEN_MIN_P}"
    "presence_penalty=${PRESENCE_PENALTY}"
    "repetition_penalty=${REPETITION_PENALTY}"
    "max_gen_toks=${MAX_GEN_TOKS}"
  )
  if [ -n "${GEN_KWARGS_EXTRA}" ]; then
    local extra_gen_kwargs=()
    read -r -a extra_gen_kwargs <<< "${GEN_KWARGS_EXTRA}"
    gen_kwargs+=("${extra_gen_kwargs[@]}")
  fi

  local cmd=(
    env "UV_PROJECT_ENVIRONMENT=${HARNESS_UV_PROJECT_ENVIRONMENT}" uv run --python "${HARNESS_UV_PYTHON}" --extra vllm --extra ifbench --extra ifeval --extra hf --extra wandb lm_eval
    --model vllm
    --model_args "${model_args[@]}"
    --tasks "${TASKS}"
    --gen_kwargs "${gen_kwargs[@]}"
    --apply_chat_template
    --num_fewshot "${NUM_FEWSHOT}"
    --batch_size "${BATCH_SIZE}"
    --confirm_run_unsafe_code
    -o "${result_dir}"
  )

  if [ "${LOG_SAMPLES}" = "1" ]; then
    cmd+=(--log_samples)
  fi
  if [ -n "${LIMIT}" ]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [ "${WANDB_ENABLED}" = "1" ]; then
    local wandb_args=(
      "project=${WANDB_PROJECT}"
      "job_type=eval"
      "name=${eval_name}"
      "group=${train_run_name}"
    )
    if [ -n "${WANDB_ENTITY}" ]; then
      wandb_args+=("entity=${WANDB_ENTITY}")
    fi
    # lm_eval parses wandb_args as comma-separated key=value pairs; tag values cannot contain commas.
    if [ -n "${WANDB_TAGS}" ]; then
      case "${WANDB_TAGS}" in
        *,*)
          log "WANDB_TAGS contains commas; add eval_wandb_tags to wandb config instead of --wandb_args tags"
          ;;
        *)
          wandb_args+=("tags=${WANDB_TAGS}")
          ;;
      esac
    fi

    local wandb_config_json_file wandb_config_json
    wandb_config_json_file="${result_dir}/wandb_config.json"
    write_wandb_config_json \
      "${wandb_config_json_file}" \
      "${kind}" \
      "${run_dir}" \
      "${checkpoint_ref}" \
      "${resolved_hf_dir}" \
      "${model_name}" \
      "${checkpoint_tag}" \
      "${train_run_name}" \
      "${eval_name}" \
      "${result_dir}" \
      "${effective_think_end_token}" \
      "${effective_gen_top_k}"
    wandb_config_json="$(<"${wandb_config_json_file}")"

    cmd+=(
      --wandb_args "${wandb_args[@]}"
      --wandb_config_args "${wandb_config_json}"
    )
  fi
  if [ -n "${LMEVAL_EXTRA_ARGS}" ]; then
    local extra_lmeval_args=()
    read -r -a extra_lmeval_args <<< "${LMEVAL_EXTRA_ARGS}"
    cmd+=("${extra_lmeval_args[@]}")
  fi

  log "Running ${eval_name}"
  log "Model path: ${resolved_hf_dir}"
  log "Results path: ${result_dir}"
  (
    cd "${HARNESS_ROOT}"
    "${cmd[@]}"
  )
  touch "${result_dir}/.eval_complete"
  log "Completed ${eval_name}"
}

resolve_manifest_path() {
  if [ -n "${MANIFEST_PATH:-}" ]; then
    echo "${MANIFEST_PATH}"
    return 0
  fi

  if [ -f "${MANIFEST_DIR}/active_manifest.path" ]; then
    cat "${MANIFEST_DIR}/active_manifest.path"
    return 0
  fi

  die "MANIFEST_PATH is not set and ${MANIFEST_DIR}/active_manifest.path is missing"
}

run_manifest_task() {
  local index="${SLURM_ARRAY_TASK_ID:-${TASK_INDEX:-1}}"
  local line kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir

  MANIFEST_PATH="$(resolve_manifest_path)"
  [ -f "${MANIFEST_PATH}" ] || die "Manifest not found: ${MANIFEST_PATH}"

  line="$(sed -n "${index}p" "${MANIFEST_PATH}")"
  [ -n "${line}" ] || die "No manifest entry at index ${index}"

  IFS=$'\t' read -r kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir <<< "${line}"
  run_eval "${kind}" "${run_dir}" "${checkpoint_ref}" "${hf_dir}" "${model_name}" "${checkpoint_tag}" "${train_run_name}" "${eval_name}" "${result_dir}"
}

run_all_from_manifest() {
  local manifest="$1"
  local kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir

  while IFS=$'\t' read -r kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir; do
    run_eval "${kind}" "${run_dir}" "${checkpoint_ref}" "${hf_dir}" "${model_name}" "${checkpoint_tag}" "${train_run_name}" "${eval_name}" "${result_dir}"
  done < "${manifest}"
}

main() {
  setup_common_dirs

  case "${STEP_POLICY}" in
    exact|at_or_after) ;;
    *) die "STEP_POLICY must be exact or at_or_after, got ${STEP_POLICY}" ;;
  esac

  case "${ONLY_FINAL}" in
    0|1) ;;
    *) die "ONLY_FINAL must be 0 or 1, got ${ONLY_FINAL}" ;;
  esac

  if [ "${MODE}" = "run-manifest" ]; then
    run_manifest_task
    return 0
  fi

  if [ "${MODE}" = "submit-existing" ]; then
    submit_manifest "${MANIFEST_PATH}"
    return 0
  fi

  local manifest
  manifest="${MANIFEST_PATH:-${MANIFEST_DIR}/ifbench_completed_$(date -u +'%Y%m%dT%H%M%SZ').tsv}"
  build_manifest "${manifest}"

  if [ "${MODE}" = "list" ]; then
    print_manifest "${manifest}"
    log "Manifest: ${manifest}"
    return 0
  fi

  if [ "${MODE}" = "run-all" ]; then
    run_all_from_manifest "${manifest}"
    return 0
  fi

  submit_manifest "${manifest}"
}

main "$@"
