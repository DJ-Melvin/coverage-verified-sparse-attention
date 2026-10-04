#!/bin/bash
#SBATCH --job-name=cvsa-thm-scaled
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/thm_val_scaled_%j.out
#SBATCH --error=logs/thm_val_scaled_%j.err

# Scaled Theorem 2 validation: more prompts + more layers per model.
# Purpose: confirm Qwen proxy degradation (ratio ~1.28) is robust, not
# sampling noise. Previous 6x6 run per model -> now 12 prompts x 10 layers.
#
# Layer selection:
#   Llama-8B   (32 L): [0, 1, 4, 8, 12, 16, 20, 24, 28, 31]
#   Mistral-7B (32 L): [0, 1, 4, 8, 12, 16, 20, 24, 28, 31]
#   Qwen-7B    (28 L): [0, 1, 3, 6, 9, 13, 17, 20, 23, 27]

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "=== Theorem 2 validation  scaled (12 prompts x 10 layers per model) ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "--- Llama-3.1-8B (32 layers, picking 10) ---"
python eval/error_bound_validation.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --n-prompts 12 \
    --layers "0,1,4,8,12,16,20,24,28,31" \
    --output results/error_bound_validation_llama_scaled.json

echo ""
echo "--- Mistral-7B-Instruct-v0.3 (32 layers, picking 10) ---"
python eval/error_bound_validation.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --n-prompts 12 \
    --layers "0,1,4,8,12,16,20,24,28,31" \
    --output results/error_bound_validation_mistral_scaled.json

echo ""
echo "--- Qwen2.5-7B-Instruct (28 layers, picking 10) ---"
python eval/error_bound_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n-prompts 12 \
    --layers "0,1,3,6,9,13,17,20,23,27" \
    --output results/error_bound_validation_qwen_scaled.json

echo ""
echo "Done at $(date)"
