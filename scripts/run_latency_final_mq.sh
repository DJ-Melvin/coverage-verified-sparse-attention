#!/bin/bash
#SBATCH --job-name=lat-mq
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/latency_final_mq_%j.out
#SBATCH --error=logs/latency_final_mq_%j.err

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "=== Final Latency: Mistral + Qwen (3 repeats) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -c "
import sys, os, json, time, gc
import numpy as np, torch
sys.path.insert(0, '.')
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
import importlib.util
_spec = importlib.util.spec_from_file_location('unified_latency',
    '/path/to/cvsa/eval/unified_latency.py')
_lat = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_lat)
benchmark_batched = _lat.benchmark_batched
generate_prompt = _lat.generate_prompt
make_backend = _lat.make_backend

MODELS = [
    ('mistral7b', 'mistralai/Mistral-7B-Instruct-v0.2'),
    ('qwen7b', 'Qwen/Qwen2.5-7B-Instruct'),
]

METHODS = ['dense', 'fixed_5', 'lazy_5_90_c2', 'lazy_5_90_c4',
           'snapkv_10', 'h2o_10', 'adakv_10', 'quest_10']
BATCH_SIZES = [1, 8, 32]
CTX_LEN = 8192
NUM_DECODE = 64
NUM_REPEATS = 3

for model_tag, model_name in MODELS:
    print(f'\n{\"=\"*70}')
    print(f'MODEL: {model_name}')
    print(f'{\"=\"*70}')

    model = ModelForDecode(model_name, dtype=torch.bfloat16)
    config = model.config
    print(f'Config: {config.num_layers}L, {config.num_q_heads}Q/{config.num_kv_heads}KV, '
          f'GQA={config.num_q_heads//config.num_kv_heads}:1')

    prompt = generate_prompt(model.tokenizer, CTX_LEN)
    model_results = {}

    for method in METHODS:
        for B in BATCH_SIZES:
            key = f'{method}_B{B}'
            repeat_ms = []
            repeat_tps = []

            for rep in range(NUM_REPEATS):
                try:
                    backend = make_backend(method, config, 16)
                    stats = benchmark_batched(model, prompt, NUM_DECODE, backend, 16, B, warmup=5)
                    repeat_ms.append(stats['mean_ms'])
                    repeat_tps.append(stats['tokens_per_sec'])
                    del backend; gc.collect(); torch.cuda.empty_cache()
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f'  ERROR {method} B={B} rep={rep}: {e}')

            if repeat_ms:
                median_ms = float(np.median(repeat_ms))
                std_ms = float(np.std(repeat_ms))
                median_tps = float(np.median(repeat_tps))
                std_tps = float(np.std(repeat_tps))
                model_results[key] = {
                    'method': method, 'batch_size': B,
                    'median_ms': median_ms, 'std_ms': std_ms,
                    'median_tps': median_tps, 'std_tps': std_tps,
                    'all_ms': repeat_ms, 'all_tps': repeat_tps,
                }
                extra = ''
                if len(repeat_ms) == NUM_REPEATS:
                    extra = f' [{\" / \".join(f\"{x:.1f}\" for x in repeat_ms)}]'
                print(f'  {method:<16} B={B:<2}  {median_ms:>7.1f} +/- {std_ms:>4.1f} ms  '
                      f'{median_tps:>6.0f} +/- {std_tps:>3.0f} tok/s{extra}', flush=True)

    # Summary
    print(f'\n--- {model_tag} SUMMARY (B=32, 8K) ---')
    dk = model_results.get('dense_B32', {})
    dense_ms = dk.get('median_ms', 1)
    print(f'{\"Method\":<16} {\"ms\":>8} {\"tok/s\":>8} {\"Speedup\":>8}')
    for method in METHODS:
        r = model_results.get(f'{method}_B32')
        if not r: continue
        spd = dense_ms / r['median_ms']
        print(f'{method:<16} {r[\"median_ms\"]:>8.1f} {r[\"median_tps\"]:>8.0f} {spd:>7.2f}x')

    with open(f'results/{model_tag}_latency_final.json', 'w') as f:
        json.dump(model_results, f, indent=2)
    print(f'Saved results/{model_tag}_latency_final.json')

    del model; gc.collect(); torch.cuda.empty_cache()

print(f'\nDone at {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
"

echo "=== Complete at $(date) ==="
