#!/bin/bash
#SBATCH --job-name=cvsa-recovery-trace
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/recovery_trace_%A_%a.out
#SBATCH --error=logs/recovery_trace_%A_%a.err

# Collect runtime CVSA recovery traces for the appendix recovery heatmap.
#
# This runs only the paper configuration (CVSA-c4, AR20, dL0), because the figure
# needs per-layer/per-step accept masks from rich_traces rather than baseline
# cache-simulation traces. The method key cvsa_ar20_c4_dL0 of
# scripts/collect_page_traces.py also names the PKL that
# eval/plot_recovery_heatmap.py reads.
#
# Outputs:
#   results/page_traces_recovery_mistral/cvsa_ar20_c4_dL0.pkl
#   results/page_traces_recovery_qwen/cvsa_ar20_c4_dL0.pkl

set -euo pipefail

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

TASK="${TASK:-hotpotqa}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_STEPS="${MAX_STEPS:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen2.5-7B-Instruct"
)
SHORT_NAMES=(
  "mistral"
  "qwen"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${#MODELS[@]}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}; expected 0..$((${#MODELS[@]} - 1))" >&2
  exit 2
fi

MODEL="${MODELS[$IDX]}"
SHORT="${SHORT_NAMES[$IDX]}"
OUT_DIR="results/page_traces_recovery_${SHORT}"
OUT_PKL="${OUT_DIR}/cvsa_ar20_c4_dL0.pkl"

echo "============================================================"
echo "CVSA runtime recovery trace collection"
echo "============================================================"
echo "Start:       $(date)"
echo "Host:        $(hostname)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Model:       ${MODEL}"
echo "Task:        ${TASK}"
echo "Samples:     ${NUM_SAMPLES}"
echo "Decode steps:${MAX_STEPS}"
echo "Output:      ${OUT_PKL}"
echo "============================================================"

mkdir -p "${OUT_DIR}"

python scripts/collect_page_traces.py \
  --model "${MODEL}" \
  --block-size "${BLOCK_SIZE}" \
  --methods cvsa_ar20_c4_dL0 \
  --task "${TASK}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --output "${OUT_DIR}/"

python - "${OUT_PKL}" <<'PY'
import pickle
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    payload = pickle.load(f)

samples = payload.get("samples", [])
if not samples:
    raise SystemExit(f"No samples found in {path}")

missing = [i for i, s in enumerate(samples) if "rich_traces" not in s]
if missing:
    raise SystemExit(f"Missing rich_traces for samples: {missing}")

n_steps = [len(s["rich_traces"]) for s in samples]
n_layers = payload.get("num_layers")
n_heads = payload.get("num_kv_heads")
print("Validation:")
print(f"  file={path}")
print(f"  model={payload.get('model')}")
print(f"  samples={len(samples)}")
print(f"  rich_trace_steps={n_steps}")
print(f"  layers={n_layers}, kv_heads={n_heads}")
PY

echo "Done: $(date)"
echo "Output ready: ${OUT_PKL}"
