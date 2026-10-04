#!/bin/bash
#SBATCH --job-name=cvsa-diagnostics
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/diagnostics_%j.out
#SBATCH --error=logs/diagnostics_%j.err

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting CVSA diagnostics at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python eval/diagnostics.py \
    --num-samples 3 \
    --max-decode 128 \
    --max-context 8000

echo "Done at $(date)"
