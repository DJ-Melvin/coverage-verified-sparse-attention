#!/bin/bash
# Tactic reimplementation: validation against the Tactic paper's own results.
#
# Purpose:
#   Backs the Tactic reproduction (selector-agnostic appendix, "Reproducing
#   Tactic"). Tactic (arXiv 2502.12216) has no public implementation, so it is
#   reimplemented from the paper. This job checks the reimplementation against
#   the operating points the Tactic paper reports for Llama-3.1-8B-Instruct
#   (its Table 2), on the six LongBench tasks it evaluates, at its target
#   cumulative-mass values, and runs the gated and CVSA rows on the same
#   tasks and samples.
#
#   The compared quantity is the mean delta vs dense, not the absolute score,
#   because the two harnesses differ (samples, template, max_gen, scorer).
#   eval/summarize_tactic.py computes the deltas from the dense arm of the same
#   run and compares them with the published ones; report both absolute scores
#   and deltas.
#
# Methods (one job, same tasks/samples):
#   dense                  same-path reference; all deltas use it
#   tactic_70, tactic_90   Tactic at target cumulative mass 0.70 / 0.90
#   tactic_90_g90          Tactic + the unchanged coverage gate (tau=0.90)
#   cvsa_5_90_c4_ar20_dL0  CVSA-c4
#
# Output:
#   results/camera_ready/tactic_validation_<model>.json
#   Summarize with: python eval/summarize_tactic.py <output.json>
#
# Cost: the Tactic backend is a token-level torch path (gather + masked
# softmax), correctness-first rather than speed-first, and far slower than the
# paged CVSA path. Run the smoke step first and check its timing before the
# full job. tests/test_tactic.py runs first; the job aborts if it fails.
#
# Usage:
#   # 1) smoke + timing (minutes)
#   MAX_SAMPLES=2 TASKS=hotpotqa sbatch scripts/run_tactic_validation.sh llama
#
#   # 2) full validation
#   sbatch scripts/run_tactic_validation.sh llama

#SBATCH --job-name=cvsa-tactic-val
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/tacticval_%x_%j.out
#SBATCH --error=logs/tacticval_%x_%j.err

set -euo pipefail

MODEL_KEY="${1:-llama}"
case "${MODEL_KEY}" in
  llama)   MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct};  SHORT=llama ;;
  mistral) MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}; SHORT=mistral ;;
  qwen)    MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct};           SHORT=qwen ;;
  *) echo "Usage: sbatch $0 {llama|mistral|qwen}" >&2; exit 2 ;;
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

# The six LongBench tasks the Tactic paper reports.
TASKS=${TASKS:-hotpotqa+triviaqa+multifieldqa_en+qasper+narrativeqa+musique}

# Single job doing validation AND comparison, on the six tasks the Tactic paper
# itself reports.
# dense              = same-path reference; all deltas are computed against it
# tactic_70/90       = the paper's two reported settings (validation)
# tactic_90_g90      = Tactic + the UNCHANGED coverage gate (composition)
# cvsa_5_90_c4_ar20_dL0 = CVSA-c4 on the same tasks/samples
METHODS=${METHODS:-dense,tactic_70,tactic_90,tactic_90_g90,cvsa_5_90_c4_ar20_dL0}
MAX_SAMPLES=${MAX_SAMPLES:-50}
OUT=${OUT:-results/camera_ready/tactic_validation_${SHORT}.json}

echo "=== Tactic reproduction — VALIDATION GATE: ${SHORT} ==="
echo "Model:   ${MODEL}"
echo "Tasks:   ${TASKS}"
echo "Methods: ${METHODS}"
echo "Samples: ${MAX_SAMPLES}"
echo "Start:   $(date)"
echo "Node:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

if ! "${PYTHON}" -c "import torch, transformers, datasets; print('torch=' + torch.__version__)"; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  exit 1
fi

# Unit tests must pass before we trust any number from this backend.
echo "--- running Tactic unit tests ---"
"${PYTHON}" -m pytest tests/test_tactic.py -q || {
  echo "Tactic unit tests FAILED — aborting before spending GPU time." >&2
  exit 1
}
echo

START=$(date +%s)
"${PYTHON}" eval/longbench_cvsa.py \
  --model "${MODEL}" \
  --methods "${METHODS}" \
  --max-samples "${MAX_SAMPLES}" \
  --tasks "${TASKS}" \
  --output "${OUT}"
END=$(date +%s)

echo
echo "Done: $(date)"
echo "Wall clock: $(( (END - START) / 60 )) min"
echo "Output: ${OUT}"
