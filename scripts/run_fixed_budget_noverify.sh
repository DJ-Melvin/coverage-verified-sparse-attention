#!/bin/bash
# Fixed-budget no-verification controls at matched effective compute.
#
# Purpose:
#   Backs the matched-effective-compute comparison (verification versus
#   budget). The selector, kernels and paged pipeline are held fixed, the
#   coverage gate is disabled, and the uniform budget is raised toward
#   CVSA-c4's measured effective read (~12%), so the effect of verification
#   is measured at matched compute rather than against a 5% control only.
#
#   fixed_{B} = CVSABackend with coverage_threshold=0.0
#             = always accept = pure sparse, NO verification, B% budget
#   (see eval/longbench_cvsa.py make_backend, fixed_ branch).
#
# Methods (one run, shared samples): dense,fixed_5,fixed_10,fixed_12
#   dense is rerun here so every fixed_* row is scored on exactly the same
#   n=50 samples per task as its reference.
#
# Output:
#   results/camera_ready/fixed_noverify_<llama|mistral|qwen>.json
#
# Usage:
#   # Smoke (1 task, 5 samples, ~20 min):
#   mkdir -p logs
#   MAX_SAMPLES=5 TASKS=hotpotqa OUT=results/smoke_fixed_llama.json \
#     sbatch scripts/run_fixed_budget_noverify.sh llama
#
#   # Full (16 LongBench tasks, 50 samples per task):
#   sbatch scripts/run_fixed_budget_noverify.sh llama
#   sbatch scripts/run_fixed_budget_noverify.sh mistral
#   sbatch scripts/run_fixed_budget_noverify.sh qwen

#SBATCH --job-name=cvsa-fixed-noverify
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/fixed_noverify_%x_%j.out
#SBATCH --error=logs/fixed_noverify_%x_%j.err

set -euo pipefail

MODEL_KEY="${1:-}"
case "${MODEL_KEY}" in
  llama)
    MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
    SHORT=llama
    ;;
  mistral)
    MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
    SHORT=mistral
    ;;
  qwen)
    MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
    SHORT=qwen
    ;;
  *)
    echo "Usage: sbatch $0 {llama|mistral|qwen}" >&2
    exit 2
    ;;
esac

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
mkdir -p logs results/camera_ready

# Optional: ENV_BIN=/path/to/env/bin selects a specific Python environment.
ENV_BIN=${ENV_BIN:-}
if [ -n "${ENV_BIN:-}" ]; then export PATH="${ENV_BIN}:$PATH"; fi

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# The repository root (cvsa/ and eval/) is on PYTHONPATH.
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
PYTHON=${PYTHON:-python}

# dense:    reference for the Gap column + shared sample basis (rerun so the
#           fixed_* rows use the exact same n=50 samples).
# fixed_5:  no-verification control at the 5% base budget.
# fixed_10: no-verification control at the baselines' 10% nominal budget.
# fixed_12: no-verification control at CVSA-c4's effective budget (~12%).
METHODS=${METHODS:-dense,fixed_5,fixed_10,fixed_12}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/fixed_noverify_${SHORT}.json}

echo "=== Fixed-budget no-verification controls: ${SHORT} (${MODEL}) ==="
echo "Methods: ${METHODS}"
echo "Samples per task: ${MAX_SAMPLES}"
echo "Tasks override: ${TASKS:-<canonical 16-task>}"
echo "Output: ${OUT}"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! "${PYTHON}" -c "import torch, transformers, datasets; print('torch=' + torch.__version__)" ; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  echo "Override with: ENV_BIN=/path/to/env/bin sbatch $0 ${MODEL_KEY}" >&2
  exit 1
fi

"${PYTHON}" eval/longbench_cvsa.py \
  --model "${MODEL}" \
  --methods "${METHODS}" \
  --max-samples "${MAX_SAMPLES}" \
  ${TASKS_FLAG} \
  --output "${OUT}"

echo "Done: $(date)"
echo "Output: ${OUT}"
