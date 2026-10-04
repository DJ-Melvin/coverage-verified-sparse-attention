#!/bin/bash
# Coverage gate under a lightweight Quest-style page selector.
#
# Purpose:
#   Backs the selector-agnostic gate experiment (Quest-style min/max indexer
#   row of the selector-agnostic verification table). The page SELECTOR is
#   swapped to Quest-style per-page key min/max upper bounds (Tang et al.,
#   ICML 2024) while the coverage gate and recovery mechanism are unchanged
#   (method suffix `selq`, see cvsa/backend.py `selector="quest"`).
#   The gate verifies the realized exact attention mass of whatever the
#   selector proposed.
#
#   Scope: the exact scorer still runs for the gate's normalizer here, so this
#   measures gate behavior under a lightweight selector, not an end-to-end
#   latency result for that selector (see scripts/run_lightnorm.sh for the
#   fully lightweight configuration).
#
# Methods (paired, same run/samples):
#   dense                       — sample-basis reference
#   cvsa_5_90_c4_ar20_dL0       — CVSA-c4, default exact-scorer selector
#   cvsa_5_90_c4_ar20_dL0_selq  — identical config, Quest min/max selector
#
# Output:
#   results/camera_ready/selector_agnostic_<llama|mistral|qwen>.json
#   Quality per method, plus cvsa_stats (per_head_accept, all_accept,
#   rescore/recovery rates, effective_budget_pct) for each sparse method.
#
# Usage:
#   # Smoke (1 task, 5 samples):
#   MAX_SAMPLES=5 TASKS=hotpotqa sbatch scripts/run_selector_agnostic.sh llama
#   # Full:
#   sbatch scripts/run_selector_agnostic.sh llama
#   sbatch scripts/run_selector_agnostic.sh mistral
#   sbatch scripts/run_selector_agnostic.sh qwen

#SBATCH --job-name=cvsa-selq
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/selq_%x_%j.out
#SBATCH --error=logs/selq_%x_%j.err

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

METHODS=${METHODS:-dense,cvsa_5_90_c4_ar20_dL0,cvsa_5_90_c4_ar20_dL0_selq}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/selector_agnostic_${SHORT}.json}

echo "=== Selector-agnostic validation (selq): ${SHORT} (${MODEL}) ==="
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
