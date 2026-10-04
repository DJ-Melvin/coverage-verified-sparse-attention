#!/bin/bash
# Production-trace calibration of the page-level coverage proxy.
#
# Purpose:
#   Backs the production-distribution proxy accept/reject confusion matrix
#   (bootstrap / diagnostics appendix). eval/production_proxy_calibration.py
#   measures the deployed page-level coverage proxy against token-level
#   coverage c* on real LongBench decode traces. Generation uses dense
#   attention, so the trace distribution is not affected by CVSA errors; the
#   coverage computations run as a side channel on the same per-step Q/KV
#   state. It reports proxy accept / token accept / false accept / false
#   reject / agreement by proxy tau, token target tau, budget, task and layer.
#
#   Defaults: TASKS=hotpotqa+qasper+lcc+passage_retrieval_en, BUDGETS=0.05,
#   TAUS=0.90,0.92,0.95,0.97 (TOKEN_TAUS defaults to TAUS), LAYERS=paper.
#
# Output:
#   results/camera_ready/production_proxy_calibration_<llama|mistral|qwen>.json
#
# Usage:
#   # Smoke (1 sample, 16 steps per task):
#   sbatch scripts/run_production_proxy_calibration.sh qwen
#
#   # Fuller run:
#   MAX_SAMPLES=4 MAX_STEPS=64 TASKS=all sbatch scripts/run_production_proxy_calibration.sh llama
#
#   # Calibrate proxy threshold against a fixed token target:
#   TAUS=0.90,0.95,0.97,0.98,0.99 TOKEN_TAUS=0.90 sbatch scripts/run_production_proxy_calibration.sh qwen

#SBATCH --job-name=cvsa-proxy-cal
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/production_proxy_calibration_%x_%j.out
#SBATCH --error=logs/production_proxy_calibration_%x_%j.err

set -euo pipefail

MODEL_KEY=${1:-qwen}

case "${MODEL_KEY}" in
  llama)
    MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
    ;;
  mistral)
    MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
    ;;
  qwen)
    MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}. Use llama, mistral, or qwen." >&2
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

TASKS=${TASKS:-hotpotqa+qasper+lcc+passage_retrieval_en}
MAX_SAMPLES=${MAX_SAMPLES:-1}
MAX_STEPS=${MAX_STEPS:-16}
MAX_LENGTH=${MAX_LENGTH:-31500}
LAYERS=${LAYERS:-paper}
BUDGETS=${BUDGETS:-0.05}
TAUS=${TAUS:-0.90,0.92,0.95,0.97}
TOKEN_TAUS=${TOKEN_TAUS:-${TAUS}}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
OUTPUT=${OUTPUT:-results/camera_ready/production_proxy_calibration_${MODEL_KEY}.json}

echo "Starting production proxy calibration at $(date)"
echo "MODEL_KEY=${MODEL_KEY}"
echo "MODEL=${MODEL}"
echo "TASKS=${TASKS}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LAYERS=${LAYERS}"
echo "TAUS=${TAUS}"
echo "TOKEN_TAUS=${TOKEN_TAUS}"
echo "OUTPUT=${OUTPUT}"
echo "ENV_BIN=${ENV_BIN}"
echo "HF_HOME=${HF_HOME}"
echo "PYTHON=${PYTHON}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! "${PYTHON}" -c "import sys, torch, triton, datasets, transformers; print('python_executable=' + sys.executable); print('torch=' + torch.__version__)" ; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  echo "Expected ENV_BIN=${ENV_BIN} to contain the paper experiment Python." >&2
  echo "Override with: ENV_BIN=/path/to/env/bin sbatch scripts/run_production_proxy_calibration.sh ${MODEL_KEY}" >&2
  exit 1
fi

"${PYTHON}" eval/production_proxy_calibration.py \
  --model "${MODEL}" \
  --tasks "${TASKS}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --max-length "${MAX_LENGTH}" \
  --layers "${LAYERS}" \
  --budgets "${BUDGETS}" \
  --taus "${TAUS}" \
  --token-taus "${TOKEN_TAUS}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --output "${OUTPUT}"

echo "Done at $(date)"
