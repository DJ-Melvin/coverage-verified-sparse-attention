#!/bin/bash
#SBATCH --job-name=cvsa-htrace
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/hybrid_coverage_trace_%j.out
#SBATCH --error=logs/hybrid_coverage_trace_%j.err

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH
export PYTHONPATH=/path/to/cvsa:$PYTHONPATH

MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
TASKS=${TASKS:-hotpotqa}
NUM_SAMPLES=${NUM_SAMPLES:-2}
MAX_STEPS=${MAX_STEPS:-384}
MAX_CONTEXT=${MAX_CONTEXT:-8000}
BUDGET_PCT=${BUDGET_PCT:-0.05}
TAU=${TAU:-0.90}
CHECK_INTERVAL=${CHECK_INTERVAL:-4}
DENSE_LAYERS=${DENSE_LAYERS:-0}
OUTPUT=${OUTPUT:-results/hybrid_coverage_trace_llama.pt}

echo "Starting CVSA hybrid coverage trace at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL=${MODEL}"
echo "TASKS=${TASKS}"
echo "NUM_SAMPLES=${NUM_SAMPLES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "OUTPUT=${OUTPUT}"

python eval/diagnostics_hybrid_trace.py \
    --model "${MODEL}" \
    --tasks "${TASKS}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-steps "${MAX_STEPS}" \
    --max-context "${MAX_CONTEXT}" \
    --budget-pct "${BUDGET_PCT}" \
    --tau "${TAU}" \
    --check-interval "${CHECK_INTERVAL}" \
    --dense-layers "${DENSE_LAYERS}" \
    --output "${OUTPUT}"

echo "Done at $(date)"
