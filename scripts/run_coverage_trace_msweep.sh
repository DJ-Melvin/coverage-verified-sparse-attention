#!/bin/bash
# Runtime coverage trace across check intervals M (reuse-window telemetry).
#
# Purpose:
#   Backs the within-reuse-window coverage telemetry across M (design
#   decisions in the experiments section) and the per-step (M=1) failure
#   persistence analysis in the offline per-(layer, head) budget appendix.
#   Records per-step coverage / recovery traces with
#   eval/diagnostics_hybrid_trace.py at a chosen check interval M, using the
#   same (model, task, samples) as the M=4 trace in
#   results/hybrid_coverage_trace_summary.json, so cross-M deltas are
#   directly comparable. Only check_interval varies.
#
# Config: Llama-3.1-8B-Instruct, hotpotqa, 2 samples, 384 decode steps, 8K
# context, 5% budget, tau=0.90, dense layer 0.
#
# Output:
#   results/camera_ready/hybrid_coverage_trace_llama_M<M>.pt
#
# Usage:
#   sbatch scripts/run_coverage_trace_msweep.sh 1
#   sbatch scripts/run_coverage_trace_msweep.sh 2
#   sbatch scripts/run_coverage_trace_msweep.sh 8

#SBATCH --job-name=cvsa-mtrace
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/coverage_trace_M%x_%j.out
#SBATCH --error=logs/coverage_trace_M%x_%j.err

set -euo pipefail

CHECK_INTERVAL="${1:-}"
case "${CHECK_INTERVAL}" in
  1|2|4|8) ;;
  *)
    echo "Usage: sbatch $0 {1|2|4|8}" >&2
    echo "  (M=4 already exists in results/hybrid_coverage_trace_summary.json)" >&2
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
PYTHON=${PYTHON:-python}

# Match the existing trace config (Llama, hotpotqa, 2 samples, 384 steps, 8K ctx,
# 5% budget, tau=0.90, dL0). Only check_interval varies — that's the whole point.
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TASKS=${TASKS:-hotpotqa}
NUM_SAMPLES=${NUM_SAMPLES:-2}
MAX_STEPS=${MAX_STEPS:-384}
MAX_CONTEXT=${MAX_CONTEXT:-8000}
BUDGET_PCT=${BUDGET_PCT:-0.05}
TAU=${TAU:-0.90}
DENSE_LAYERS=${DENSE_LAYERS:-0}
OUTPUT=${OUTPUT:-results/camera_ready/hybrid_coverage_trace_llama_M${CHECK_INTERVAL}.pt}

echo "=== Hybrid coverage trace M=${CHECK_INTERVAL} ==="
echo "MODEL=${MODEL}"
echo "TASKS=${TASKS}, NUM_SAMPLES=${NUM_SAMPLES}, MAX_STEPS=${MAX_STEPS}"
echo "OUTPUT=${OUTPUT}"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! "${PYTHON}" -c "import torch, transformers, datasets; print('torch=' + torch.__version__)" ; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  exit 1
fi

"${PYTHON}" eval/diagnostics_hybrid_trace.py \
    --model "${MODEL}" \
    --tasks "${TASKS}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-steps "${MAX_STEPS}" \
    --max-context "${MAX_CONTEXT}" \
    --budget-pct "${BUDGET_PCT}" \
    --tau "${TAU}" \
    --check-interval "${CHECK_INTERVAL}" \
    --dense-layers "${DENSE_LAYERS}" \
    --output "${OUTPUT}"

echo "Done at $(date)"
echo "Output: ${OUTPUT}"
