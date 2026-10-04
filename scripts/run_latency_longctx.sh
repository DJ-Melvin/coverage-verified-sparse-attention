#!/bin/bash
#SBATCH --job-name=cvsa-longctx
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/latency_longctx_%j.out
#SBATCH --error=logs/latency_longctx_%j.err

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting long-context latency at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Long-context decode latency (32K, 64K, 128K) for the three 7B models.
# Pass the model key as the first argument: llama | mistral | qwen.
case "${1:-llama}" in
    llama)   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct ;;
    mistral) MODEL=mistralai/Mistral-7B-Instruct-v0.3 ;;
    qwen)    MODEL=Qwen/Qwen2.5-7B-Instruct ;;
    *) echo "Usage: sbatch $0 {llama|mistral|qwen}" >&2; exit 1 ;;
esac
python eval/latency_benchmark.py \
    --model "${MODEL}" \
    --context-lens 32768,65536,131072 \
    --batch-sizes 1,8 \
    --methods dense,fixed_10,cvsa_5_90_c4_ar20_dL0,cvsa_5_90_c8_ar20_dL0 \
    --num-decode 128 \
    --output "results/longctx_lat_${1:-llama}.json"

echo "Done at $(date)"
