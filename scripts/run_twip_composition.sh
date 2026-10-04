#!/bin/bash
# Pre-hoc top-p selection (Twilight principle) with and without the coverage
# gate, in the same pipeline.
#
# Purpose:
#   Backs the pre-hoc top-p (adaptive budget) rows of the selector-agnostic
#   verification and traffic-accounting tables. The pre-hoc top-p principle
#   of Twilight is instantiated inside the CVSA pipeline, so both rows share
#   the scorer, kernels, samples and run:
#
#   twip_85_dL0                  — pre-hoc top-p: per-head budget = smallest
#                                  page set reaching 0.85 cumulative mass on
#                                  ESTIMATED Quest min/max weights; no
#                                  verification.
#   cvsa_5_90_c4_ar20_dL0_tp85   — the SAME selector + CVSA's coverage gate
#                                  and recovery.
#   dense                        — sample-basis reference.
#
# Implementation notes:
#   - The estimator here is fp16 min/max, stronger than the 4-bit estimator
#     of the released Twilight implementation.
#   - Top-p is applied at page granularity (Twilight prunes with its own
#     kernels); the portable principle is per-head top-p on estimates.
#   - No budget cap (Twilight has none), so effective_budget_pct in cvsa_stats
#     should be read for both sparse rows.
#   For the audit of the released Twilight code itself, see
#   scripts/run_twilight_longbench.sh.
#
# Output:
#   results/camera_ready/twip_composition_<llama|mistral|qwen>.json
#
# Usage:
#   MAX_SAMPLES=5 TASKS=hotpotqa sbatch scripts/run_twip_composition.sh llama   # smoke
#   sbatch scripts/run_twip_composition.sh llama                                # full

#SBATCH --job-name=cvsa-twip
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/twip_%x_%j.out
#SBATCH --error=logs/twip_%x_%j.err

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

METHODS=${METHODS:-dense,twip_85_dL0,cvsa_5_90_c4_ar20_dL0_tp85}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/twip_composition_${SHORT}.json}

echo "=== Twilight-composition (twip): ${SHORT} (${MODEL}) ==="
echo "Methods: ${METHODS}"
echo "Samples per task: ${MAX_SAMPLES}"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! "${PYTHON}" -c "import torch, transformers, datasets; print('torch=' + torch.__version__)" ; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
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
