#!/bin/bash
# Static offline per-(layer, head) budget policy vs CVSA's per-step
# verify-and-recover routing, at matched effective compute.
#
# Purpose:
#   Backs the offline per-(layer, head) budget control in the
#   matched-effective-compute comparison (dense-layer / recovery ablation
#   appendix). The static policy assigns larger budgets to heads identified
#   from offline coverage statistics (DuoAttention-style); only the per-step
#   routing is frozen, the selector, kernels and samples are unchanged.
#
#   Step 1  Calibrate: Recovery Atlas (per-(layer, head) recovery rates at
#           5% budget, tau=0.90) on 4 LongBench tasks x 5 prompts
#           (eval/diagnostics_head.py). Calibration prompts come from the
#           evaluation distribution, which favors the static policy.
#           Skipped if ATLAS already exists.
#   Step 2  Build the static table (eval/build_static_budget_table.py):
#           b[l,h] = 5% + r[l,h] * 15%, rescaled so the non-dense-layer mean
#           is TARGET_MEAN_PCT (default 9.0%); with dense layer 0 this gives
#           an effective read close to CVSA-c4's.
#   Step 3  Evaluate on LongBench (16 tasks x 50 samples), one run:
#             dense          - sample-basis reference
#             fixed_12       - uniform static 12% budget, no verification
#             staticpol_dL0  - structured static budget from the table above
#
# Output:
#   results/camera_ready/atlas_calib_<model>.pt      (calibration atlas)
#   results/camera_ready/static_budget_<model>.json  (budget table)
#   results/camera_ready/static_policy_<model>.json  (LongBench scores)
#
# Usage:
#   # Smoke (1 task, 5 samples):
#   MAX_SAMPLES=5 TASKS=hotpotqa sbatch scripts/run_static_policy_ablation.sh llama
#   # Full:
#   sbatch scripts/run_static_policy_ablation.sh llama
#   sbatch scripts/run_static_policy_ablation.sh mistral
#   sbatch scripts/run_static_policy_ablation.sh qwen

#SBATCH --job-name=cvsa-staticpol
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/staticpol_%x_%j.out
#SBATCH --error=logs/staticpol_%x_%j.err

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

ATLAS=${ATLAS:-results/camera_ready/atlas_calib_${SHORT}.pt}
TABLE=${TABLE:-results/camera_ready/static_budget_${SHORT}.json}
# 9.0% over 31 non-dense layers ~= 8.7% total attention read; + dL0 2.9%
# ~= 11.6% effective, matched to CVSA-c4's 11.7%.
TARGET_MEAN_PCT=${TARGET_MEAN_PCT:-9.0}
METHODS=${METHODS:-dense,fixed_12,staticpol_dL0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/static_policy_${SHORT}.json}

echo "=== Static-policy ablation: ${SHORT} (${MODEL}) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! "${PYTHON}" -c "import torch, transformers, datasets; print('torch=' + torch.__version__)" ; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  exit 1
fi

# ── Step 1: offline calibration atlas (skipped if present) ─────────────
if [ -f "${ATLAS}" ]; then
  echo "[1/3] Calibration atlas exists: ${ATLAS} (skipping)"
else
  echo "[1/3] Collecting calibration atlas -> ${ATLAS}"
  "${PYTHON}" eval/diagnostics_head.py \
    --model "${MODEL}" \
    --num-samples 5 \
    --max-decode 128 \
    --max-context 8000 \
    --budgets 0.05 \
    --tau-values 0.90 \
    --output "${ATLAS}"
fi

# ── Step 2: build static per-(layer, head) budget table ────────────────
echo "[2/3] Building static budget table -> ${TABLE}"
"${PYTHON}" eval/build_static_budget_table.py \
  --atlas "${ATLAS}" \
  --output "${TABLE}" \
  --calib-budget 0.05 \
  --tau 0.90 \
  --base 0.05 \
  --ar 0.20 \
  --dense-layers 0 \
  --target-mean-pct "${TARGET_MEAN_PCT}"

# ── Step 3: LongBench evaluation ────────────────────────────────────────
echo "[3/3] LongBench: methods=${METHODS}, n=${MAX_SAMPLES}/task"
CVSA_STATIC_BUDGET_TABLE="${TABLE}" "${PYTHON}" eval/longbench_cvsa.py \
  --model "${MODEL}" \
  --methods "${METHODS}" \
  --max-samples "${MAX_SAMPLES}" \
  ${TASKS_FLAG} \
  --output "${OUT}"

echo "Done: $(date)"
echo "Output: ${OUT}"
