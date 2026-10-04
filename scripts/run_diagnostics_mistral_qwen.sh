#!/bin/bash
#SBATCH --job-name=cvsa-diag-2models
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/diag_2models_%j.out
#SBATCH --error=logs/diag_2models_%j.err

# Run eval/diagnostics.py on Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct
# to collect per-layer mean_coverage at 4 budgets (5/10/15/20%)  needed for
# Figure 1 multi-model post-AR coverage panel (paired with existing Llama
# results/diagnostics.pt from run_diagnostics.sh).

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting 2-model diagnostics at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

#  Mistral-7B-Instruct-v0.3
echo ""
echo "=== Mistral-7B-Instruct-v0.3 ==="
python eval/diagnostics.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --output results/diagnostics_mistral.pt \
    --num-samples 3 --max-decode 128 --max-context 8000

#  Qwen2.5-7B-Instruct
echo ""
echo "=== Qwen2.5-7B-Instruct ==="
python eval/diagnostics.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output results/diagnostics_qwen.pt \
    --num-samples 3 --max-decode 128 --max-context 8000

echo ""
echo "Done at $(date)"
echo "Outputs:"
ls -la results/diagnostics_mistral.pt results/diagnostics_qwen.pt
