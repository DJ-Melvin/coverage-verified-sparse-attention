#!/bin/bash
# Fully lightweight end-to-end CVSA (sketch-estimated gate normalizer).
#
# Purpose:
#   Backs the fully lightweight end-to-end configuration (selector-agnostic
#   appendix, fully lightweight table). Method suffix `selq_ln` makes the
#   Quest min/max sketch supply BOTH the page selection and the coverage
#   gate's all-context normalizer, so no full-key scan runs anywhere on the
#   path (selection, gate, and recovery expansion all use the sketch).
#
#   The gate then checks the selection against what the indexer can see: it
#   still catches flat-score and insufficient-budget steps and routes
#   recovery there, but indexer estimation bias itself is not observable.
#   It is read against the selq rows of scripts/run_selector_agnostic.sh
#   (same selector with the exact normalizer) on the same samples/harness.
#
#   tau: sketch scores are upper bounds with a different spread from exact
#   scores, so the second argument selects tau (80/85/90/95 -> 0.80..0.95);
#   running 90 and 80 brackets the operating range.
#
# Method: cvsa_5_<tau>_c4_ar20_dL0_selq_ln
#
# Output:
#   results/camera_ready/lightnorm_<model>_t<tau>.json
#   Quality per task plus cvsa_stats (per_head_accept, rescore_rate,
#   effective_budget_pct).
#
# Usage:
#   # Smoke (2 samples, 1 task):
#   MAX_SAMPLES=2 TASKS=hotpotqa sbatch scripts/run_lightnorm.sh llama 90
#   # Full:
#   sbatch scripts/run_lightnorm.sh llama 90
#   sbatch scripts/run_lightnorm.sh llama 80

#SBATCH --job-name=cvsa-lightnorm
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/lightnorm_%x_%j.out
#SBATCH --error=logs/lightnorm_%x_%j.err

set -euo pipefail

MODEL_KEY="${1:-}"
TAU_KEY="${2:-90}"
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
    echo "Usage: sbatch $0 {llama|mistral|qwen} [tau: 90|80]" >&2
    exit 2
    ;;
esac
case "${TAU_KEY}" in
  90|80|85|95) ;;
  *)
    echo "tau must be one of 80/85/90/95, got ${TAU_KEY}" >&2
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

METHODS=${METHODS:-cvsa_5_${TAU_KEY}_c4_ar20_dL0_selq_ln}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/lightnorm_${SHORT}_t${TAU_KEY}.json}

echo "=== Fully lightweight CVSA (selq_ln): ${SHORT}, tau=0.${TAU_KEY} ==="
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
