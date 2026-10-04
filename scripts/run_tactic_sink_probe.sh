#!/bin/bash
# Tactic reimplementation: attention-sink / recent-window probe.
#
# Purpose:
#   Backs the attention-sink departure described in the Tactic reproduction
#   (selector-agnostic appendix, "Reproducing Tactic"). The Tactic paper does
#   not mention attention sinks, and the plain reimplementation (no forced
#   sinks) falls short of the published result at the LOW target mass (0.70)
#   while matching it at the high one: at a low target few tokens are
#   selected, so the sink is not picked up incidentally. This job measures
#   whether force-including the first tokens (and, separately, a recent
#   window) accounts for that gap. tactic_70 is re-run in the same job so the
#   comparison is within one run rather than across runs.
#
# Methods (one job, same six Tactic-paper LongBench tasks and samples):
#   dense                same-path reference; deltas are computed against it
#   tactic_70            Tactic at target mass 0.70, no forced sinks
#   tactic_70_sk4        + the first 4 (sink) tokens force-included
#   tactic_70_sk4_rc64   + sinks and a 64-token recent window
#
# Output:
#   results/camera_ready/tactic_sink_probe_<model>.json
#   Summarize with: python eval/summarize_tactic.py <output.json>
#   (token-weighted read share; gated variants are excluded from the
#   comparison with the published deltas).
#
# Usage:
#   MAX_SAMPLES=2 TASKS=qasper sbatch scripts/run_tactic_sink_probe.sh llama   # smoke
#   sbatch scripts/run_tactic_sink_probe.sh llama                              # full (~3.6h)

#SBATCH --job-name=cvsa-tactic-sink
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/tacticsink_%x_%j.out
#SBATCH --error=logs/tacticsink_%x_%j.err

set -euo pipefail

# Record the arguments before dispatching on them, and absorb stray
# whitespace or CR that survives a copy-paste.
echo "argv: argc=$# raw=[${*:-}]"
MODEL_KEY_RAW="${1:-llama}"
MODEL_KEY=$(printf '%s' "${MODEL_KEY_RAW}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
case "${MODEL_KEY}" in
  llama|llama3|llama31)  MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct};  SHORT=llama ;;
  mistral|mistral7b)     MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}; SHORT=mistral ;;
  qwen|qwen2|qwen25)     MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct};           SHORT=qwen ;;
  *) printf 'Unrecognized model key: %q (normalized: %q)\n' "${MODEL_KEY_RAW}" "${MODEL_KEY}" >&2
     echo "Usage: sbatch $0 {llama|mistral|qwen}" >&2
     exit 2 ;;
esac
echo "Resolved model key: ${SHORT}"

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

TASKS=${TASKS:-hotpotqa+triviaqa+multifieldqa_en+qasper+narrativeqa+musique}

# dense           = same-path reference (deltas are computed against it)
# tactic_70       = no forced sinks, re-run for a within-job A/B rather than
#                   a cross-run comparison
# tactic_70_sk4   = + 4 sink tokens force-included
# tactic_70_sk4_rc64 = + sink and a 64-token recent window
METHODS=${METHODS:-dense,tactic_70,tactic_70_sk4,tactic_70_sk4_rc64}
MAX_SAMPLES=${MAX_SAMPLES:-50}
OUT=${OUT:-results/camera_ready/tactic_sink_probe_${SHORT}.json}

echo "=== Tactic A8 probe (attention sink / recent window): ${SHORT} ==="
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
echo
echo "=== 判讀 ==="
echo "不要用肉眼讀上面那張表,跑彙整器(它用 token-weighted 的 read 基準,"
echo "並且會把 gated 變體排除在 fidelity 判定之外):"
echo
echo "    python eval/summarize_tactic.py ${OUT}"
