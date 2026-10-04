#!/bin/bash
#SBATCH --job-name=cvsa-boot
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/bootstrap_data_%j.out
#SBATCH --error=logs/bootstrap_data_%j.err

# The CVSA method key stays in its historical ssa_* spelling here:
# scripts/compute_paired_ci.py and eval/plot_pareto.py read
# results/bootstrap_*.json under that key.

set -e

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting bootstrap data collection (2 models x 100 samples) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""
echo "========== Mistral v0.3 =========="
python eval/longbench_cvsa.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --methods dense,ssa_5_90_c4_ar20_dL0,snapkv_10,h2o_10 \
    --max-samples 100 \
    --output results/bootstrap_mistral.json

echo ""
echo "========== Qwen 7B =========="
python eval/longbench_cvsa.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --methods dense,ssa_5_90_c4_ar20_dL0,snapkv_10,h2o_10 \
    --max-samples 100 \
    --output results/bootstrap_qwen.json

echo ""
echo "Done at $(date)"
