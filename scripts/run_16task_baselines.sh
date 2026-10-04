#!/bin/bash
#SBATCH --job-name=cvsa-16task
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/16task_%j.out
#SBATCH --error=logs/16task_%j.err

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting 16-task baseline comparison at $(date)"

python -c "
import sys, os, torch, json, numpy as np
sys.path.insert(0, '.')
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from eval.longbench_harness import LongBenchE2ERunner
from eval.baselines import make_backend as make_baseline
from cvsa.backend import CVSABackend

model = ModelForDecode('meta-llama/Meta-Llama-3.1-8B-Instruct', dtype=torch.bfloat16, device='cuda')
config = model.config
runner = LongBenchE2ERunner(model, block_size=16)

tasks = [
    'hotpotqa', 'triviaqa', 'qasper', 'multifieldqa_en',
    '2wikimqa', 'narrativeqa', 'musique',
    'gov_report', 'multi_news', 'samsum', 'qmsum',
    'trec', 'lcc', 'repobench-p',
    'passage_count', 'passage_retrieval_en',
]

def make(method):
    if method == 'dense':
        return DenseAttentionBackend(config.num_kv_heads, config.num_q_heads, config.head_dim, 16)
    elif method == 'lazy_10_90':
        return CVSABackend(num_kv_heads=config.num_kv_heads, num_q_heads=config.num_q_heads,
            head_dim=config.head_dim, block_size=16, num_layers=config.num_layers,
            budget_pct=0.10, coverage_threshold=0.90, lazy=True)
    elif method == 'ssa_10_90':
        return CVSABackend(num_kv_heads=config.num_kv_heads, num_q_heads=config.num_q_heads,
            head_dim=config.head_dim, block_size=16, num_layers=config.num_layers,
            budget_pct=0.10, coverage_threshold=0.90, lazy=False)
    elif method == 'fixed_10':
        return CVSABackend(num_kv_heads=config.num_kv_heads, num_q_heads=config.num_q_heads,
            head_dim=config.head_dim, block_size=16, num_layers=config.num_layers,
            budget_pct=0.10, coverage_threshold=0.0, lazy=False)
    else:
        return make_baseline(method, config, 16, config.num_layers)

methods = ['dense', 'fixed_10', 'ssa_10_90', 'lazy_10_90', 'adakv_10', 'snapkv_10', 'h2o_10']

results = {m: {} for m in methods}
for task in tasks:
    print(f'\\nTask: {task}', flush=True)
    for method in methods:
        try:
            backend = make(method)
            score, elapsed, n, _, _ = runner.evaluate_dataset(task, backend, 50)
            results[method][task] = score
            extra = ''
            if hasattr(backend, 'get_stats'):
                s = backend.get_stats()
                if s.get('lazy', False):
                    extra = f' rescore={s[\"rescore_rate\"]:.0f}%'
                elif s.get('total_scoring_steps', 0) > 0:
                    extra = f' accept={s[\"all_accepted_pct\"]:.0f}%'
            print(f'  {method:<14} {score:>6.2f}  ({n} samples, {elapsed:.0f}s){extra}', flush=True)
        except Exception as e:
            print(f'  {method:<14} ERROR: {e}', flush=True)
            results[method][task] = None

# Summary
print('\\n' + '='*100)
print('16-TASK LONGBENCH RESULTS')
print('='*100)
header = f\"{'Method':<14}\"
for t in tasks: header += f' {t[:6]:>7}'
header += f' {\"MEAN\":>7}'
print(header)
print('-'*len(header))
for m in methods:
    vals = [results[m].get(t) for t in tasks]
    valid = [v for v in vals if v is not None]
    row = f'{m:<14}'
    for v in vals:
        row += f' {v:>7.1f}' if v is not None else f' {\"ERR\":>7}'
    row += f' {np.mean(valid):>7.2f}' if valid else ' N/A'
    print(row)

os.makedirs('results', exist_ok=True)
with open('results/16task_baselines.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to results/16task_baselines.json')
"

echo "Done at $(date)"
