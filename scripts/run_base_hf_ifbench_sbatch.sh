#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Evaluate base Hugging Face models on the IFBench eval suite via SLURM.

Default usage:
  bash lm-evaluation-harness/scripts/run_base_hf_ifbench_sbatch.sh

Useful modes:
  --list                 Write and print the base-model manifest only.
  --write-manifest-only  Write the manifest and print its path.

Environment overrides:
  SCRATCH_ROOT=/scratch4/workspace/...
  RESULTS_ROOT=/scratch4/.../eval_results/ifbench_base_models
  MANIFEST_PATH=/scratch4/.../eval_manifests/ifbench_base_models.tsv
  MAX_PARALLEL=4
  WANDB_PROJECT=open_instruct_internal
  WANDB_ENTITY=mohdelgaar
USAGE
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WRAPPER="${WRAPPER:-${SCRIPT_DIR}/run_completed_open_instruct_ifbench_sbatch.sh}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch4/workspace/mohamed_elgaar_student_uml_edu-rl-curriculum}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH_ROOT}/eval_results/ifbench_base_models}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH_ROOT}/eval_manifests}"
MANIFEST_PATH="${MANIFEST_PATH:-${MANIFEST_DIR}/ifbench_base_models_$(date -u +'%Y%m%dT%H%M%SZ').tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

EVAL_SLURM_PARTITION="${EVAL_SLURM_PARTITION:-gpu,gpu-preempt}"
EVAL_SLURM_CONSTRAINT="${EVAL_SLURM_CONSTRAINT:-a100-80g}"
EVAL_SLURM_CPUS_PER_GPU="${EVAL_SLURM_CPUS_PER_GPU:-10}"
EVAL_SLURM_MEM="${EVAL_SLURM_MEM:-100G}"
EVAL_SLURM_TIME="${EVAL_SLURM_TIME:-24:00:00}"

MODE="submit"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --list)
      MODE="list"
      shift
      ;;
    --write-manifest-only)
      MODE="write-manifest-only"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

sanitize_for_path() {
  local value="$1"
  value="${value//\//__}"
  value="${value//:/_}"
  value="${value// /_}"
  value="${value//./p}"
  value="${value//-/_}"
  echo "${value}"
}

write_manifest() {
  local model_id slug train_run_name eval_name result_dir

  mkdir -p "${MANIFEST_DIR}" "${RESULTS_ROOT}" "${PROJECT_ROOT}/logs"
  : > "${MANIFEST_PATH}"

  for model_id in \
    "Qwen/Qwen3-0.6B" \
    "Qwen/Qwen3-1.7B" \
    "Qwen/Qwen3.5-0.8B" \
    "Qwen/Qwen3.5-2B" \
    "Qwen/Qwen3.5-9B" \
    "google/gemma-4-E2B-it" \
    "google/gemma-4-E4B-it"; do
    slug="$(sanitize_for_path "${model_id}")"
    train_run_name="base_${slug}"
    eval_name="eval_base_${slug}_ifbench"
    result_dir="${RESULTS_ROOT}/${slug}/${eval_name}"
    if [ "${FORCE:-0}" != "1" ] && [ -f "${result_dir}/.eval_complete" ]; then
      continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "hf_hub" \
      "-" \
      "${model_id}" \
      "${model_id}" \
      "${model_id}" \
      "base_hf" \
      "${train_run_name}" \
      "${eval_name}" \
      "${result_dir}" >> "${MANIFEST_PATH}"
  done
}

print_manifest() {
  local i=0 kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir
  while IFS=$'\t' read -r kind run_dir checkpoint_ref hf_dir model_name checkpoint_tag train_run_name eval_name result_dir; do
    i=$((i + 1))
    printf '%d\t%s\t%s\t%s\n' "${i}" "${kind}" "${model_name}" "${eval_name}"
  done < "${MANIFEST_PATH}"
}

submit_manifest() {
  local count array_arg
  count="$(awk 'END {print NR + 0}' "${MANIFEST_PATH}")"
  if [ "${count}" -eq 0 ]; then
    echo "No base-model evals to submit; all result markers already exist under ${RESULTS_ROOT}" >&2
    return 0
  fi

  array_arg="1-${count}"
  if [ "${MAX_PARALLEL}" != "0" ]; then
    array_arg="${array_arg}%${MAX_PARALLEL}"
  fi

  (
    cd "${PROJECT_ROOT}"
    sbatch \
      --job-name=ifbench-base-eval \
      --partition="${EVAL_SLURM_PARTITION}" \
      --gpus=1 \
      --constraint="${EVAL_SLURM_CONSTRAINT}" \
      --cpus-per-gpu="${EVAL_SLURM_CPUS_PER_GPU}" \
      --mem="${EVAL_SLURM_MEM}" \
      --time="${EVAL_SLURM_TIME}" \
      --array="${array_arg}" \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",HARNESS_ROOT="${PROJECT_ROOT}/lm-evaluation-harness",MANIFEST_PATH="${MANIFEST_PATH}",RESULTS_ROOT="${RESULTS_ROOT}",WANDB_MIRROR_TRAINING_CONFIG=0,WANDB_STRICT_TRAINING_CONFIG=0,EVAL_RUN_FAMILY=base_model,EVAL_MODEL_SOURCE=hf_hub,EVAL_DATASET_KEY=base_model,EVAL_BENCHMARK_GROUP=ifbench,BASE_MODEL_SEED=0,WANDB_TAGS=base_model \
      "${WRAPPER}" --run-manifest
  )
}

write_manifest
case "${MODE}" in
  list)
    print_manifest
    echo "Manifest: ${MANIFEST_PATH}" >&2
    ;;
  write-manifest-only)
    echo "${MANIFEST_PATH}"
    ;;
  submit)
    print_manifest
    echo "Manifest: ${MANIFEST_PATH}" >&2
    submit_manifest
    ;;
esac
