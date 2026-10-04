#!/bin/bash
#SBATCH --job-name=c8-missing
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/c8_missing_16task_%A_%a.out
#SBATCH --error=logs/c8_missing_16task_%A_%a.err

set -euo pipefail

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen2.5-7B-Instruct"
)

TAGS=(
  "mistral"
  "qwen"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
OUT="results/c8_${TAG}_16task.json"

echo "=== Missing c8 LongBench-16 run: ${TAG} ==="
echo "Model: ${MODEL}"
echo "Output: ${OUT}"
echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval/longbench_cvsa.py \
  --model "${MODEL}" \
  --methods dense,cvsa_5_90_c8_ar20_dL0 \
  --max-samples 50 \
  --output "${OUT}"

echo "Done: $(date)"
