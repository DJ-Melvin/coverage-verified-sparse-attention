#!/bin/bash
#SBATCH --job-name=kvp-llama70b
#SBATCH --partition=normal2
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/kvp_llama70b_%j.out
#SBATCH --error=logs/kvp_llama70b_%j.err

# KVPress canonical baselines for Llama-3.1-70B-Instruct on 4xH200.
# 8 tasks  n=30 (matches results/70b_ar_quality.json layout).
# Reference dense = 38.75 (chat-template-fixed; see project_70b_baselines_task).
# MAX_LEN=7500 to match existing paged-decode 70B runs (truncation parity).

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -c "import kvpress" 2>/dev/null || pip install --user kvpress

echo "=== KVPress baselines: Llama-3.1-70B-Instruct (4xH200) ==="
echo "Start: $(date)  Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import kvpress, transformers; print(f'kvpress {kvpress.__version__}, transformers {transformers.__version__}')"

python <<'PY'
import sys, gc, json, time
import numpy as np, torch
from contextlib import nullcontext

sys.path.insert(0, '.')
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from kvpress import SnapKVPress
try:
    from kvpress import AdaKVPress
    HAS_ADAKV = True
except ImportError:
    HAS_ADAKV = False
    print('WARN: AdaKVPress not available in this kvpress version', flush=True)

from eval.longbench_harness import DATASET2PROMPT, DATASET2METRIC
from eval.longbench_harness import DATASET2MAXGEN as _DS2MAXGEN

#  Config (locked; matches 70b_ar_quality.json layout)
MODEL_ID = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
MAX_LEN = 7500                         # match paged-decode 70B truncation
N_SAMPLES = 30
BUDGET_FRAC = 0.10
COMPRESSION = 1.0 - BUDGET_FRAC
PAGED_DENSE_REF = 38.75                # 70b_ar_quality.json dense (chat-fixed)
NO_CHAT_TASKS = {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}
TASKS = [
    'hotpotqa', 'triviaqa', 'qasper', 'multifieldqa_en',
    'samsum', 'trec', 'lcc', 'narrativeqa',
]
OUT_PATH = 'results/kvpress_baselines_llama70b.json'

print(f'Loading {MODEL_ID} sharded across 4xH200...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',
    attn_implementation='sdpa',
)
model.eval()
embed_dev = next(model.model.embed_tokens.parameters()).device
print(f'  embed device: {embed_dev}', flush=True)


def prepare_prompt(task, item):
    prompt = DATASET2PROMPT[task].format(**item)
    tokens = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids[0]
    if len(tokens) > MAX_LEN:
        half = MAX_LEN // 2
        prompt = (tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                  tokenizer.decode(tokens[-half:], skip_special_tokens=True))
    if task not in NO_CHAT_TASKS:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return prompt


def run_task(task, press=None):
    data = load_dataset('THUDM/LongBench', task, split='test', trust_remote_code=True)
    n = min(N_SAMPLES, len(data))
    data = data.select(range(n))
    max_gen = _DS2MAXGEN[task]
    metric_fn = DATASET2METRIC[task]
    sample_scores = []
    for i, item in enumerate(data):
        prompt = prepare_prompt(task, item)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(embed_dev)
        seq_len = input_ids.shape[1]
        ctx = press(model) if press is not None else nullcontext()
        with torch.no_grad(), ctx:
            out = model.generate(
                input_ids, max_new_tokens=max_gen, do_sample=False,
                num_beams=1, pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)
        pred_clean = pred.split('\n')[0].strip()
        answers = item['answers']
        if isinstance(answers, str):
            answers = [answers]
        sample_scores.append(float(metric_fn(pred_clean, answers)))
        del input_ids, out
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f'    [{i+1}/{n}] running_avg={np.mean(sample_scores)*100:.1f}', flush=True)
    return sample_scores


configs = [('hf_dense', None), ('hf_snapkv', SnapKVPress(compression_ratio=COMPRESSION))]
if HAS_ADAKV:
    try:
        configs.append(('hf_adakv', AdaKVPress(SnapKVPress(compression_ratio=COMPRESSION))))
    except TypeError:
        configs.append(('hf_adakv', AdaKVPress(press=SnapKVPress(compression_ratio=COMPRESSION))))

per_sample = {name: {} for name, _ in configs}

for name, press in configs:
    print(f'\n--- [{name}] ---', flush=True)
    t0 = time.time()
    for t in TASKS:
        try:
            scores = run_task(t, press=press)
            per_sample[name][t] = scores
            print(f'  {name:<12} {t:<22} {np.mean(scores)*100:>6.2f}  (n={len(scores)})', flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            per_sample[name][t] = []
            print(f'  {name:<12} {t:<22} ERROR: {e}', flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f'  [{name}] elapsed: {(time.time()-t0)/60:.1f} min', flush=True)
    # Incremental save (long run; protect against late OOM)
    with open(OUT_PATH, 'w') as f:
        json.dump({
            'config': f'Llama-3.1-70B, 8 tasks, n={N_SAMPLES}, budget={BUDGET_FRAC}, max_length={MAX_LEN}, sdpa, KVPress',
            'paged_dense_ref': PAGED_DENSE_REF,
            'per_sample_scores': per_sample,
        }, f, indent=2)

#  Summary
print('\n' + '='*100)
print('MEANS (100 LongBench scale)')
print('='*100)
header = f'{"Method":<12}'
for t in TASKS:
    header += f'{t[:9]:>10}'
header += f'{"MEAN":>8}'
print(header)
print('-' * len(header))

means = {}
for name, _ in configs:
    row = f'{name:<12}'
    per_task_means = []
    for t in TASKS:
        scores = per_sample[name].get(t)
        if scores:
            v = np.mean(scores) * 100
            per_task_means.append(v)
            row += f'{v:>10.2f}'
        else:
            row += f'{"ERR":>10}'
    m = np.mean(per_task_means) if per_task_means else 0.0
    means[name] = m
    row += f'{m:>8.2f}'
    print(row)

dense_diff = means.get('hf_dense', 0.0) - PAGED_DENSE_REF
print('\n' + '='*100)
print('PRE-FLIGHT DENSE MATCH')
print('='*100)
print(f'KVPress dense    : {means.get("hf_dense", 0):.2f}')
print(f'Paged-decode ref : {PAGED_DENSE_REF}')
print(f'Diff             : {dense_diff:+.2f}')
if abs(dense_diff) <= 0.5:
    print(' Within 0.5pt  pipelines aligned')
elif abs(dense_diff) <= 1.0:
    print('~ 0.5-1pt drift  note in 4.1 footnote')
elif abs(dense_diff) <= 2.0:
    print(' 1-2pt drift  disclose explicitly')
else:
    print(' >2pt drift  pipeline divergence; baseline numbers contaminated')

#  Bootstrap CI
print('\n' + '='*100)
print('BOOTSTRAP 95% CI (10K resamples; paired across methods)')
print('='*100)
N_BOOT = 10000
rng = np.random.default_rng(42)

valid_tasks = [t for t in TASKS if all(per_sample[n].get(t) for n, _ in configs)]
arrs_per_method = {
    name: {t: np.array(per_sample[name][t], dtype=np.float64) for t in valid_tasks}
    for name, _ in configs
}
all_dists = {name: np.empty(N_BOOT) for name, _ in configs}
for b in range(N_BOOT):
    idx_per_task = {}
    for t in valid_tasks:
        n = len(arrs_per_method[configs[0][0]][t])
        idx_per_task[t] = rng.integers(0, n, size=n)
    for name, _ in configs:
        task_means = [arrs_per_method[name][t][idx_per_task[t]].mean() * 100
                      for t in valid_tasks]
        all_dists[name][b] = np.mean(task_means)

for name, _ in configs:
    d = all_dists[name]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f'{name:<12} {d.mean():>6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}]  (width={hi-lo:.2f})')

if 'hf_dense' in all_dists:
    print('\n---  vs hf_dense (paired) ---')
    for name, _ in configs:
        if name == 'hf_dense':
            continue
        deltas = all_dists[name] - all_dists['hf_dense']
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        print(f'{name:<12} ={deltas.mean():+6.2f}  95% CI [{lo:+6.2f}, {hi:+6.2f}]')

with open(OUT_PATH, 'w') as f:
    json.dump({
        'config': f'Llama-3.1-70B, 8 tasks, n={N_SAMPLES}, budget={BUDGET_FRAC}, max_length={MAX_LEN}, sdpa, KVPress',
        'paged_dense_ref': PAGED_DENSE_REF,
        'per_sample_scores': per_sample,
        'means': means,
        'dense_match_diff': dense_diff,
    }, f, indent=2)
print(f'\nSaved: {OUT_PATH}')
PY

echo ""
echo "=== Done at $(date) ==="
