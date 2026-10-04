#!/bin/bash
#SBATCH --job-name=cvsa-dl-ablate
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/dl_ablate_%x_%j.out
#SBATCH --error=logs/dl_ablate_%x_%j.err

# Canonical dense-layer ablation for the paper's primary setting.
#
# Purpose:
#   Tests whether the layer-0 dense policy is necessary under the exact
#   configuration used in the main paper: 5% base budget, tau=0.90, M=4,
#   AR=20%, LongBench 16-task, n=50.
#
# Usage:
#   sbatch scripts/run_dense_layer_ablation_canonical.sh llama
#   sbatch scripts/run_dense_layer_ablation_canonical.sh mistral
#   sbatch scripts/run_dense_layer_ablation_canonical.sh qwen

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

echo "=== Canonical dense-layer ablation: ${SHORT} (${MODEL}) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

METHODS="dense,fixed_5,cvsa_5_90_c4_ar20,cvsa_5_90_c4_ar20_dL0"
OUT="results/dense_layer_ablation_${SHORT}.json"

python eval/longbench_cvsa.py \
    --model "${MODEL}" \
    --methods "${METHODS}" \
    --max-samples 50 \
    --output "${OUT}"

echo "=== Done: ${SHORT} at $(date) ==="
echo "Output: ${OUT}"
