#!/bin/bash
# Stochastic-decoding probe: dense vs CVSA-c4 under seed-paired nucleus sampling.
#
# Purpose:
#   Backs the stochastic-decoding check (bootstrap / diagnostics appendix).
#   The coverage certificate is computed per step from the current query's
#   scores/LSE before the token is drawn, so it does not depend on the
#   decoding rule. This probe measures the CVSA-c4 vs dense gap under
#   sampling, to be compared with the greedy per-task gap on the same tasks.
#
# Design:
#   - Nucleus sampling at the Llama-3.1 model-card defaults
#     (temperature 0.6, top-p 0.9), seed-paired: sample i of every method
#     uses RNG seed (SAMPLE_SEED + i), so dense and CVSA draw from the same
#     stream and the per-sample Δ is a paired comparison.
#   - 8 tasks = qasper, lcc, hotpotqa, passage_retrieval_en (including the
#     two tasks on which CVSA-c4 is weakest) + the 4 long-form generation
#     tasks where sampling matters most (gov_report, multi_news, qmsum,
#     samsum).
#   - Methods: dense,cvsa_5_90_c4_ar20_dL0 (same config as the greedy runs).
#
# Output:
#   results/camera_ready/sampling_probe_<llama|mistral|qwen>.json
#   Compare against the greedy per-task Δ on the same tasks, using paired
#   per-sample statistics rather than means only.
#
# Usage:
#   # Smoke (1 task, 5 samples):
#   MAX_SAMPLES=5 TASKS=hotpotqa sbatch scripts/run_sampling_probe.sh llama
#   # Full probe:
#   sbatch scripts/run_sampling_probe.sh llama

#SBATCH --job-name=cvsa-samp
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/sampling_%x_%j.out
#SBATCH --error=logs/sampling_%x_%j.err

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

METHODS=${METHODS:-dense,cvsa_5_90_c4_ar20_dL0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
TASKS=${TASKS:-qasper+lcc+hotpotqa+passage_retrieval_en+gov_report+multi_news+qmsum+samsum}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.9}
SAMPLE_SEED=${SAMPLE_SEED:-1234}
OUT=${OUT:-results/camera_ready/sampling_probe_${SHORT}.json}

echo "=== Sampling probe: ${SHORT} (${MODEL}) ==="
echo "Methods: ${METHODS}"
echo "Tasks: ${TASKS}"
echo "Samples per task: ${MAX_SAMPLES}"
echo "Sampling: temp=${TEMPERATURE} top_p=${TOP_P} seed=${SAMPLE_SEED} (paired)"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Pre-flight: the runner must have sampling support (nucleus_sample in
# eval/longbench_harness.py) — fail loudly if it is missing.
if ! grep -q "nucleus_sample" "${ROOT}/eval/longbench_harness.py"; then
  echo "ERROR: ${ROOT}/eval/longbench_harness.py lacks nucleus_sample (sampling support)." >&2
  exit 3
fi

"${PYTHON}" eval/longbench_cvsa.py \
  --model "${MODEL}" \
  --methods "${METHODS}" \
  --max-samples "${MAX_SAMPLES}" \
  --tasks "${TASKS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --sample-seed "${SAMPLE_SEED}" \
  --output "${OUT}"

echo "Done: $(date)"
echo "Output: ${OUT}"
