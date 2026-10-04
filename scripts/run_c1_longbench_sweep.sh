#!/bin/bash
# M=1 (check_interval=1, per-step verifier) LongBench run.
#
# Purpose:
#   Backs the M=1 endpoint of the check-interval (M) sweep (M-sweep appendix,
#   per-model quality table) at the canonical 5% base budget, tau=0.90,
#   AR=20%, dense layer 0: the page selection is refreshed and verified at
#   every decode step.
#
# Methods: dense,cvsa_5_90_c1_ar20_dL0
#   dense is rerun here so the c1 row is scored on the same n=50 samples per
#   task as its reference.
#
# Output:
#   results/camera_ready/c1_longbench_<llama|mistral|qwen>.json
#
# Usage:
#   sbatch scripts/run_c1_longbench_sweep.sh llama
#   sbatch scripts/run_c1_longbench_sweep.sh mistral
#   sbatch scripts/run_c1_longbench_sweep.sh qwen

#SBATCH --job-name=cvsa-c1-longbench
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/c1_longbench_%x_%j.out
#SBATCH --error=logs/c1_longbench_%x_%j.err

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

# Canonical setting: matches dense_layer_ablation / exact_vs_ar_canonical pipelines.
# Methods: dense (reference) + c1 (per-step verifier at 5% base budget, AR20, dL0).
# We deliberately rerun dense here so the same sample basis is used for the c1 row
# (no mixing of dense references with different sample counts across runs).
METHODS=${METHODS:-dense,cvsa_5_90_c1_ar20_dL0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
OUT=${OUT:-results/camera_ready/c1_longbench_${SHORT}.json}

echo "=== c1 LongBench sweep: ${SHORT} (${MODEL}) ==="
echo "Methods: ${METHODS}"
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
  --output "${OUT}"

echo "Done: $(date)"
echo "Output: ${OUT}"
