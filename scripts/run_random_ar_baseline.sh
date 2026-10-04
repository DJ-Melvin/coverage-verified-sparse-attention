#!/bin/bash
#SBATCH --job-name=cvsa-rand
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/random_ar_%j.out
#SBATCH --error=logs/random_ar_%j.err

# Random-AR counterfactual: at each (layer, step), override the coverage
# decision with Bernoulli(0.20), triggering approximate recovery on a random
# 20% of layer-steps instead of the coverage-flagged layer-steps.

set -e

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

METHOD_RAND="cvsa_5_90_c4_ar20_rand20_dL0"
METHOD_BASELINE="cvsa_5_90_c4_ar20_dL0"
METHOD_FIXED="fixed_5"

echo "=== Random-AR counterfactual (LongBench 16 tasks, 100 samples) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Methods under test: $METHOD_RAND, $METHOD_BASELINE, $METHOD_FIXED"

echo ""
echo "--- Llama-3.1-8B ---"
python eval/longbench_cvsa.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --methods "dense,${METHOD_FIXED},${METHOD_RAND},${METHOD_BASELINE}" \
    --max-samples 100 \
    --output results/random_ar_llama.json

echo ""
echo "--- Mistral-7B-Instruct-v0.3 ---"
python eval/longbench_cvsa.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --methods "dense,${METHOD_FIXED},${METHOD_RAND},${METHOD_BASELINE}" \
    --max-samples 100 \
    --output results/random_ar_mistral.json

echo ""
echo "--- Qwen2.5-7B-Instruct ---"
python eval/longbench_cvsa.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --methods "dense,${METHOD_FIXED},${METHOD_RAND},${METHOD_BASELINE}" \
    --max-samples 100 \
    --output results/random_ar_qwen.json

echo ""
echo "=== Done at $(date) ==="
