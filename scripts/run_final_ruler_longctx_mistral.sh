#!/bin/bash
#SBATCH --job-name=rlong-mistral
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/ruler_longctx_mistral_%j.out
#SBATCH --error=logs/ruler_longctx_mistral_%j.err

cd /path/to/cvsa
mkdir -p logs results
export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "RULER mistral Long Context (64K-128K) at $(date)"

python eval/ruler_cvsa.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --methods "dense,fixed_10,ssa_5_90_c4_ar20_dL0,ssa_5_90_c8_ar20_dL0" \
    --context-lens 65536,131072 \
    --n-trials 5 \
    --output results/ruler_longctx_mistral.json

echo "Done at $(date)"
