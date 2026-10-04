#!/bin/bash
#SBATCH --job-name=cvsa-thm-ctok
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/thm_val_ctok_%j.out
#SBATCH --error=logs/thm_val_ctok_%j.err

# Theorem 2 validation with token-level coverage c* (path C).
#
# Goal: re-run the validation that produced *_scaled.json, but the modified
# error_bound_validation.py now also logs:
#   - coverage_token (per-Q-head c* from full-attention LSE)
#   - theorem_exact_token, theorem_bound_token (using c* instead of page-max c-hat)
#
# Hypothesis: Qwen's 3.3% bound violations under the page-max proxy disappear
# (or shrink) when the bound is computed against true token-level coverage.
#
# Outputs go to *_ctok.json so the existing *_scaled.json files remain untouched.

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "=== Theorem 2 validation  token-level c* (path C) ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "--- Llama-3.1-8B (32 layers, picking 10) ---"
python eval/error_bound_validation.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --n-prompts 12 \
    --layers "0,1,4,8,12,16,20,24,28,31" \
    --output results/error_bound_validation_llama_ctok.json

echo ""
echo "--- Mistral-7B-Instruct-v0.3 (32 layers, picking 10) ---"
python eval/error_bound_validation.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --n-prompts 12 \
    --layers "0,1,4,8,12,16,20,24,28,31" \
    --output results/error_bound_validation_mistral_ctok.json

echo ""
echo "--- Qwen2.5-7B-Instruct (28 layers, picking 10) ---"
python eval/error_bound_validation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n-prompts 12 \
    --layers "0,1,3,6,9,13,17,20,23,27" \
    --output results/error_bound_validation_qwen_ctok.json

echo ""
echo "Done at $(date)"
