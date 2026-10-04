#!/bin/bash
# Long-decode coverage trace (drift over decode depth).
#
# Purpose:
#   Backs the long-decode coverage trace (stability over long decode, and the
#   reuse-step limitation that cites it). Tests whether error accumulates
#   with decode length: selection is refreshed and the certificate recomputed
#   against the current query every M steps, so per-step coverage and
#   recovery rate should show no degradation trend with decode depth if
#   nothing persists beyond one window.
#
#   Uses eval/diagnostics_hybrid_trace.py: the generation loop takes exactly
#   MAX_STEPS greedy steps with NO EOS early-exit, and KV is pre-sized for
#   seq_len + max_steps. Default is 1,024 steps; the 32,768-step trace
#   corresponds to MAX_STEPS=32768 (binned summary:
#   results/camera_ready/longdecode_32k_bins.json).
#
#   Budget note when MAX_STEPS >> context: _scoring_step computes the budget
#   as budget_pct * ALLOCATED pages (block_tables width = seq_len + max_steps),
#   so early decode steps run at an inflated effective budget (~25% at step 0
#   for 8K ctx + 32K alloc), converging to the nominal 5% as decode fills the
#   allocation. Read only the late region where valid ≈ allocated (for the
#   32K-step run: steps 16K–32K). Negligible for MAX_STEPS ≲ context/8 (the
#   1,024-step run stays at ≈5% throughout). Raw .pt traces are gitignored
#   (results/**/*.pt); at 32K steps they exceed 100MB.
#
# Config: deployed CVSA-c4 settings (5% budget, tau=0.90, M=4, dL0), Llama,
# 2 tasks x 2 samples (hotpotqa QA + gov_report summarization), 8K ctx.
#
# Output:
#   results/camera_ready/longdecode_trace_llama_M<CHECK_INTERVAL>.pt
#   Analyze with: python eval/analyze_longdecode_drift.py \
#     --trace results/camera_ready/longdecode_trace_llama_M4.pt
#
# Usage:
#   sbatch scripts/run_longdecode_trace.sh
#   # smaller/faster variant:
#   NUM_SAMPLES=1 sbatch scripts/run_longdecode_trace.sh
#
# Walltime: about 3 minutes for 4 samples x 1,024 steps
# (diagnostics_hybrid_trace.py runs at near-native decode speed), so extending
# to 2,048/4,096 steps or more samples takes minutes, not hours.

#SBATCH --job-name=cvsa-longdecode
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/longdecode_%x_%j.out
#SBATCH --error=logs/longdecode_%x_%j.err

set -euo pipefail

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

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TASKS=${TASKS:-hotpotqa,gov_report}
NUM_SAMPLES=${NUM_SAMPLES:-2}
MAX_STEPS=${MAX_STEPS:-1024}
MAX_CONTEXT=${MAX_CONTEXT:-8000}
BUDGET_PCT=${BUDGET_PCT:-0.05}
TAU=${TAU:-0.90}
CHECK_INTERVAL=${CHECK_INTERVAL:-4}
DENSE_LAYERS=${DENSE_LAYERS:-0}
OUTPUT=${OUTPUT:-results/camera_ready/longdecode_trace_llama_M${CHECK_INTERVAL}.pt}

echo "=== Long-decode drift trace: ${MAX_STEPS} steps, M=${CHECK_INTERVAL} ==="
echo "MODEL=${MODEL}  TASKS=${TASKS}  samples/task=${NUM_SAMPLES}"
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
echo "Next: pull locally, then:"
echo "  python eval/analyze_longdecode_drift.py --trace ${OUTPUT}"
