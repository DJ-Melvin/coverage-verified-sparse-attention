#!/bin/bash
# Ungated controls with four-step refresh and dense layer 0.
#
# Purpose:
#   Backs the additional controls in the dense-layer / recovery ablation
#   appendix. The matched-compute controls of run_fixed_budget_noverify.sh and
#   run_static_policy_ablation.sh (fixed_12, staticpol_dL0) re-select pages
#   every 16 steps, and fixed_12 runs layer 0 sparse, whereas CVSA-c4 refreshes
#   every M=4 steps and keeps layer 0 dense. These controls align both, so the
#   remaining difference to CVSA-c4 is the coverage gate:
#     fixed_9_r4_dL0    uniform 9% base budget, re-selected every 4 steps,
#                       layer 0 dense, no verification
#     fixed_10_r4_dL0   same at 10%, to bracket CVSA-c4's effective read
#     staticpol_r4_dL0  offline per-(layer, head) budgets, re-selected every
#                       4 steps, layer 0 dense, no verification
#   r{N} and dL{L} are parsed in eval/longbench_cvsa.py make_backend.
#   The CVSA-c4 reference row is the deterministic CVSA-c4 run on the same
#   first 50 samples per task (run_exact_vs_ar_canonical.sh).
#
# Requires:
#   results/camera_ready/static_budget_<model>.json, the offline budget table
#   built by scripts/run_static_policy_ablation.sh (Step 2).
#
# Output:
#   results/camera_ready/matched_compute_aligned_<llama|mistral|qwen>.json
#
# Usage:
#   # Smoke (1 task, 5 samples):
#   mkdir -p logs
#   MAX_SAMPLES=5 TASKS=hotpotqa OUT=results/smoke_aligned_llama.json \
#     sbatch scripts/run_matched_compute_aligned.sh llama
#
#   # Full (16 LongBench tasks, 50 samples per task):
#   sbatch scripts/run_matched_compute_aligned.sh llama
#   sbatch scripts/run_matched_compute_aligned.sh mistral
#   sbatch scripts/run_matched_compute_aligned.sh qwen

#SBATCH --job-name=cvsa-mc-aligned
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc_aligned_%x_%j.out
#SBATCH --error=logs/mc_aligned_%x_%j.err

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
export CVSA_STATIC_BUDGET_TABLE=${CVSA_STATIC_BUDGET_TABLE:-${ROOT}/results/camera_ready/static_budget_${SHORT}.json}
PYTHON=${PYTHON:-python}

METHODS=${METHODS:-fixed_9_r4_dL0,fixed_10_r4_dL0,staticpol_r4_dL0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/matched_compute_aligned_${SHORT}.json}

echo "=== Controls with four-step refresh and dense layer 0: ${SHORT} (${MODEL}) ==="
echo "Methods: ${METHODS}"
echo "Static table: ${CVSA_STATIC_BUDGET_TABLE}"
echo "Samples per task: ${MAX_SAMPLES}"
echo "Tasks override: ${TASKS:-<canonical 16-task>}"
echo "Output: ${OUT}"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ",${METHODS}," == *",staticpol"* ]] && [ ! -f "${CVSA_STATIC_BUDGET_TABLE}" ]; then
  echo "Missing ${CVSA_STATIC_BUDGET_TABLE}; build it with scripts/run_static_policy_ablation.sh ${MODEL_KEY}." >&2
  exit 1
fi
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
