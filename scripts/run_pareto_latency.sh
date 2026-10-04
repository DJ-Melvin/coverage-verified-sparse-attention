#!/bin/bash
#SBATCH --job-name=pareto-lat
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pareto_latency_%j.out
#SBATCH --error=logs/pareto_latency_%j.err

# Unified latency benchmark: CVSA + all baselines at matched conditions.
# For valid Pareto plot (Figure 2). All methods measured in same session.
#
# Model: Llama-3.1-8B-Instruct
# Methods: dense, fixed_10, CVSA-c4, CVSA-c8, SnapKV, H2O, AdaKV.
# Context: 8K (anchor), 32K, 65K (long-context story)
# Batch: B=8 (all ctx), B=32 (8K+32K only, 65K B=32 OOMs)

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

METHODS="dense,fixed_10,cvsa_5_90_c4_ar20_dL0,cvsa_5_90_c8_ar20_dL0,snapkv_10,h2o_10,adakv_10"

echo "=== Pareto Latency Benchmark ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "Methods: ${METHODS}"
echo ""

# Phase 1: B=8, all context lengths (8K/32K/65K)
echo "===== Phase 1: B=8, ctx=8K/32K/65K ====="
python eval/latency_benchmark.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --methods "${METHODS}" \
    --context-lens 8192,32768,65536 \
    --batch-sizes 8 \
    --num-decode 128 \
    --output results/pareto_latency_b8.json

echo ""
echo "===== Phase 2: B=32, ctx=8K/32K ====="
# Phase 2: B=32, 8K+32K only (65K B=32 needs ~85GB, OOMs on H100)
python eval/latency_benchmark.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --methods "${METHODS}" \
    --context-lens 8192,32768 \
    --batch-sizes 32 \
    --num-decode 128 \
    --output results/pareto_latency_b32.json

echo ""
echo "=== Complete at $(date) ==="
