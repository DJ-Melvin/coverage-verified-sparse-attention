#!/bin/bash
# SnapKV and AdaKV inside the paged-decode pipeline.
#
# Purpose:
#   Backs the single-pipeline rerun of the baselines (bootstrap / pipeline
#   appendix, single paged-decode yardstick table). SnapKV and AdaKV run in
#   the SAME paged-decode pipeline as CVSA, with a dense reference in the same
#   run, so every method shares one dense reference. The same output also
#   gives the paged-decode side of the cross-pipeline (KVPress vs paged)
#   comparison.
#
#   eval/longbench_cvsa.py dispatches the snapkv_10 / adakv_10 method strings
#   to make_baseline_backend (eval/longbench_harness.py make_backend).
#
# Methods: dense,snapkv_10,adakv_10   (10% budget for both baselines)
#
# Output:
#   results/camera_ready/baseline_paged_decode_<llama|mistral|qwen>.json
#
# Usage:
#   # Smoke (1 task, 5 samples, ~30 min):
#   MAX_SAMPLES=5 TASKS=hotpotqa \
#     sbatch scripts/run_baseline_paged_decode.sh llama
#
#   # Full (16 LongBench tasks, 50 samples per task):
#   sbatch scripts/run_baseline_paged_decode.sh llama
#   sbatch scripts/run_baseline_paged_decode.sh mistral
#   sbatch scripts/run_baseline_paged_decode.sh qwen

#SBATCH --job-name=cvsa-baseline-paged
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/baseline_paged_%x_%j.out
#SBATCH --error=logs/baseline_paged_%x_%j.err

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

# Include dense as a reference column so cross-pipeline drift can be computed
# in the same JSON (no separate KVPress dense lookup needed). snapkv_10 and
# adakv_10 are the paged-decode baselines. After a smoke run, check that
# scores[snapkv_10] / scores[adakv_10] are non-null for the tested tasks;
# cross-pipeline drift = scores[snapkv_10] here vs means[hf_snapkv] in
# results/kvpress_baselines_<model>*.json.
METHODS=${METHODS:-dense,snapkv_10,adakv_10}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS_FLAG=""
if [ -n "${TASKS:-}" ]; then
  TASKS_FLAG="--tasks ${TASKS}"
fi
OUT=${OUT:-results/camera_ready/baseline_paged_decode_${SHORT}.json}

echo "=== Paged-decode baselines: ${SHORT} (${MODEL}) ==="
echo "Methods: ${METHODS}"
echo "Samples per task: ${MAX_SAMPLES}"
echo "Tasks override: ${TASKS:-<canonical 16-task>}"
echo "Output: ${OUT}"
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
