#!/bin/bash
#SBATCH --job-name=cvsa-exact-ar
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/exact_ar_%x_%j.out
#SBATCH --error=logs/exact_ar_%x_%j.err

# Canonical exact-recovery vs AR20 comparison.
#
# exact row:
#   cvsa_5_90_c4_dL0 = 5% budget, tau=0.90, M=4, dL0, no ar suffix.
#   In eval/longbench_cvsa.py, no ar suffix leaves approx_recovery_pct=None,
#   so rejected heads use exact complement attention + LSE merge.
#
# AR row:
#   cvsa_5_90_c4_ar20_dL0 = the paper's practical primary configuration.
#
# Usage:
#   sbatch scripts/run_exact_vs_ar_canonical.sh mistral
#   sbatch scripts/run_exact_vs_ar_canonical.sh qwen
#   sbatch scripts/run_exact_vs_ar_canonical.sh llama

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

echo "=== Canonical exact-vs-AR comparison: ${SHORT} (${MODEL}) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

METHODS="dense,cvsa_5_90_c4_dL0,cvsa_5_90_c4_ar20_dL0"
OUT="results/exact_vs_ar_canonical_${SHORT}.json"

python eval/longbench_cvsa.py \
    --model "${MODEL}" \
    --methods "${METHODS}" \
    --max-samples 50 \
    --output "${OUT}"

echo "=== Done: ${SHORT} at $(date) ==="
echo "Output: ${OUT}"
