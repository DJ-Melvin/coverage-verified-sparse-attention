#!/bin/bash
#SBATCH --job-name=cvsa-longbench
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/longbench_%j.out
#SBATCH --error=logs/longbench_%j.err

set -e

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting LongBench CVSA eval at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Full 16-task evaluation
# Methods: dense baseline, fixed 10% sparse, CVSA at tau=0.90 and 0.95.
python eval/longbench_cvsa.py \
    --tasks hotpotqa+triviaqa+qasper+multifieldqa_en+2wikimqa+narrativeqa+musique+gov_report+multi_news+samsum+qmsum+trec+lcc+repobench-p+passage_count+passage_retrieval_en \
    --methods dense,fixed_10,cvsa_10_90,cvsa_10_95 \
    --max-samples 50 \
    --output results/longbench_cvsa.json

echo "Done at $(date)"
