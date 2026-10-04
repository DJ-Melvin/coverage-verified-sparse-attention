#!/bin/bash
# Reasoning-length generation: dense vs CVSA-c4 on MATH-500.
#
# Purpose:
#   Backs the reasoning-length generation probe (limitations appendix):
#   long chain-of-thought generation with DeepSeek-R1-Distill-Llama-8B on
#   MATH-500, where the generated trace itself is long.
#
# Methods: dense,cvsa_5_90_c4_ar20_dL0
#
# Protocol: sampling temp=0.6 / top-p 0.95 (DeepSeek model-card recommendation;
# greedy degenerates on R1-distill), seed-paired per problem across methods,
# EOS-stopped, max 16,384 new tokens. Grading: last \boxed{} + light
# normalization; the same grader is used for both arms, so the comparison is
# paired. START and N select a contiguous slice of MATH-500 problems.
#
# Budget semantics: the backend's page budget = 5% of ALLOCATED pages
# (prompt + max-new), i.e., a fixed ~52-page budget provisioned for the full
# generation — NOT 5% of the current cache.
#
# Output:
#   results/camera_ready/math_reasoning_r1distill8b.json (checkpointed per
#   problem). Report paired accuracy and the disagreement count; also compare
#   avg_gen_tokens (a large inflation under CVSA indicates degenerate output).
#
# Walltime estimate: avg R1-8B trace on MATH-500 ≈ 3–6k tokens @ ~30ms/step
# => ~2–4 min/problem/method; n=50 × 2 methods ≈ 4–7h. Reserve 24h.
# Model download (~16GB) on first run adds ~10-20 min.
#
# Usage:
#   # Smoke (3 problems, ~20 min incl. model download):
#   N=3 sbatch scripts/run_math_reasoning.sh
#   # Full run (N=50 by default):
#   sbatch scripts/run_math_reasoning.sh

#SBATCH --job-name=cvsa-mathpilot
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mathpilot_%x_%j.out
#SBATCH --error=logs/mathpilot_%x_%j.err

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

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
METHODS=${METHODS:-dense,cvsa_5_90_c4_ar20_dL0}
N=${N:-50}
START=${START:-0}
MAX_NEW=${MAX_NEW:-16384}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
SEED=${SEED:-1234}
OUT=${OUT:-results/camera_ready/math_reasoning_r1distill8b.json}

echo "=== MATH-500 reasoning pilot: dense vs CVSA ==="
echo "MODEL=${MODEL}  METHODS=${METHODS}  START=${START} N=${N}  MAX_NEW=${MAX_NEW}"
echo "temp=${TEMP} top_p=${TOP_P} seed=${SEED} (seed-paired per problem)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if ! "${PYTHON}" -c "import torch, transformers, datasets; print('torch=' + torch.__version__)" ; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  exit 1
fi

"${PYTHON}" eval/math_reasoning_eval.py \
    --model "${MODEL}" \
    --methods "${METHODS}" \
    --n "${N}" \
    --start "${START}" \
    --max-new "${MAX_NEW}" \
    --temperature "${TEMP}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    --output "${OUT}"

echo "Done at $(date)"
echo "Output: ${OUT} (checkpointed per problem — partial results usable)"
