#!/bin/bash
#SBATCH --job-name=cvsa-tau-ext
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/tau_ext_%x_%j.out
#SBATCH --error=logs/tau_ext_%x_%j.err

# Extended tau sweep on the canonical c4/ar20/dL0/5%-budget config.
#
# Adds tau=0.80 and tau=0.97 to the original 0.85/0.90/0.95 set, so the
# sweep covers a wide enough range to refute "tau cherry-picked" concerns.
# Also extends to Mistral and Qwen (the original tau_ablation.json was
# Llama-only).
#
# Usage:
#   sbatch scripts/run_tau_ablation_extended.sh llama
#   sbatch scripts/run_tau_ablation_extended.sh mistral
#   sbatch scripts/run_tau_ablation_extended.sh qwen
#
# Each invocation runs 7 methods (dense + fixed_5 + 5 tau values) on one
# 7B model, ~14h on a single H100/H200. Submit 3 jobs in parallel.

set -euo pipefail

MODEL_KEY="${1:-}"
case "${MODEL_KEY}" in
    llama)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        SHORT="llama"
        ;;
    mistral)
        MODEL="mistralai/Mistral-7B-Instruct-v0.3"
        SHORT="mistral"
        ;;
    qwen)
        MODEL="Qwen/Qwen2.5-7B-Instruct"
        SHORT="qwen"
        ;;
    *)
        echo "Usage: sbatch $0 {llama|mistral|qwen}" >&2
        exit 1
        ;;
esac

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "=== Extended tau ablation: ${SHORT} (${MODEL}) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

# 7 methods: dense + no-verify (fixed_5) + 5 tau points clustered around 0.90.
# canonical config = cvsa_5_{tau}_c4_ar20_dL0
METHODS="dense,fixed_5,\
cvsa_5_80_c4_ar20_dL0,\
cvsa_5_85_c4_ar20_dL0,\
cvsa_5_90_c4_ar20_dL0,\
cvsa_5_95_c4_ar20_dL0,\
cvsa_5_97_c4_ar20_dL0"

OUT="results/tau_ablation_extended_${SHORT}.json"

python eval/longbench_cvsa.py \
    --model "${MODEL}" \
    --methods "${METHODS}" \
    --max-samples 50 \
    --output "${OUT}"

echo "=== Done: ${SHORT} at $(date) ==="
echo "Output: ${OUT}"
