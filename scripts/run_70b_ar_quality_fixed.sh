#!/bin/bash
#SBATCH --job-name=cvsa-70q-fix
#SBATCH --partition=normal2
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/70b_ar_quality_fixed_%j.out
#SBATCH --error=logs/70b_ar_quality_fixed_%j.err

# 70B AR quality FIXED rerun:
#   - Multi-EOS break (Llama-3.1's [128001,128008,128009], not just tokenizer.eos_token_id)
#   - First-line truncation for QA/summarization tasks
#   - 7 tasks (drop samsum: paged loop's open-ended-summarization stopping is
#     not equivalent to HF generate; documented limitation per
#     results/70b_samsum_diagnose.json showing HF generate scores 45.14 vs
#     paged 19.33 on same model)
#   - n=30 (matches existing 70b_ar_quality + KVPress 70B baselines)
#   - Per-sample logging for downstream bootstrap CI
#
# 3 configs: dense, ar20_c2_dL0, ar20_c4_dL0. Outputs go to NEW file
# results/70b_ar_quality_fixed.json (does NOT overwrite the prior broken
# results/70b_ar_quality.json so we can compare).

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== 70B AR quality FIXED (7-task, EOS-multi + first-line) ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python <<'PY'
import sys, os, gc, importlib, time, json
import numpy as np, torch

sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from cvsa.runtime.dense import DenseAttentionBackend
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager
from cvsa.backend import CVSABackend
from datasets import load_dataset
from eval.longbench_harness import DATASET2PROMPT, DATASET2MAXGEN, DATASET2METRIC

MODEL = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
OUT_PATH = 'results/70b_ar_quality_fixed.json'

#  7 tasks (samsum dropped per documented framework limitation)
TASKS = ['hotpotqa', 'triviaqa', 'qasper', 'multifieldqa_en',
         'trec', 'lcc', 'narrativeqa']
N_SAMPLES = 30
MAX_CONTEXT = 7500
NO_CHAT_TASKS = {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}
# Tasks where splitting at first newline destroys the answer (multi-line outputs
# legitimately needed by the metric)
NO_SPLIT_TASKS = {'trec', 'lcc', 'repobench-p'}

print(f'Loading {MODEL} (sharded across 4 GPUs)...', flush=True)
hf_config = AutoConfig.from_pretrained(MODEL)
num_layers = hf_config.num_hidden_layers
num_kv_heads = hf_config.num_key_value_heads
num_q_heads = hf_config.num_attention_heads
head_dim = hf_config.hidden_size // num_q_heads
print(f'  layers={num_layers}, q={num_q_heads}, kv={num_kv_heads}', flush=True)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
hf_model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#  EOS multi-ID fix
gc_eos = getattr(hf_model.generation_config, 'eos_token_id', None)
if gc_eos is None:
    eos_ids = [tokenizer.eos_token_id]
elif isinstance(gc_eos, int):
    eos_ids = [gc_eos]
else:
    eos_ids = list(gc_eos)
if tokenizer.eos_token_id not in eos_ids:
    eos_ids.append(tokenizer.eos_token_id)
print(f'EOS token IDs to break on: {eos_ids}', flush=True)

inner_model = hf_model.model
layers = inner_model.layers
lm_head = hf_model.lm_head
layer_devices = [next(layer.parameters()).device for layer in layers]
embed_dev = next(inner_model.embed_tokens.parameters()).device
head_dev = next(lm_head.parameters()).device
kv_device = 'cuda:0'
attn_mod = type(layers[0].self_attn).__module__
apply_rotary = importlib.import_module(attn_mod).apply_rotary_pos_emb
rotary = inner_model.rotary_emb

block_size = 16


def truncate_middle(text, max_len):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_len:
        return text
    half = max_len // 2
    kept = tokens[:half] + tokens[-half:]
    return tokenizer.decode(kept, skip_special_tokens=True)


def generate_70b(prompt, backend, max_new_tokens, task=None):
    prompt = truncate_middle(prompt, MAX_CONTEXT)
    if task in NO_CHAT_TASKS:
        input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=False).to(embed_dev)
    else:
        messages = [{'role': 'user', 'content': prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(formatted, return_tensors='pt', truncation=False).to(embed_dev)
    seq_len = input_ids.shape[1]

    chunk_size = 4096
    past_kv = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        with torch.no_grad():
            outputs = inner_model(input_ids=chunk, past_key_values=past_kv,
                                  use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values

    first_logits = lm_head(outputs.last_hidden_state[:, -1:, :].to(head_dev).squeeze(1))
    cur_token = first_logits.argmax(dim=-1)

    n_pages = (seq_len + block_size - 1) // block_size
    num_blocks = n_pages + max_new_tokens // block_size + 32
    layer_caches, layer_managers = [], []
    for li in range(num_layers):
        cache = PagedKVCache(num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            dtype=torch.bfloat16, device=kv_device)
        mgr = SequenceKVManager(cache, 1, max_context_len=seq_len + max_new_tokens + 32)
        k = past_kv[li][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(torch.bfloat16).to(kv_device)
        v = past_kv[li][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(torch.bfloat16).to(kv_device)
        mgr.prefill_kv(0, k, v)
        layer_caches.append(cache)
        layer_managers.append(mgr)
    del outputs, past_kv; torch.cuda.empty_cache()

    if hasattr(backend, 'reset'):
        backend.reset()
    generated = [cur_token.item()]

    # Check first token immediately for EOS
    if cur_token.item() in eos_ids:
        del layer_caches, layer_managers; gc.collect(); torch.cuda.empty_cache()
        return tokenizer.decode(generated, skip_special_tokens=True)

    for step in range(max_new_tokens - 1):
        pos = seq_len + step
        hidden = inner_model.embed_tokens(cur_token.unsqueeze(0).to(embed_dev))
        pos_ids = torch.full((1, 1), pos, device=hidden.device)
        cos, sin = rotary(hidden, pos_ids)

        for li, layer in enumerate(layers):
            c, m = layer_caches[li], layer_managers[li]
            ld = layer_devices[li]
            hidden = hidden.to(ld)
            cos_l, sin_l = cos.to(ld), sin.to(ld)

            res = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(torch.bfloat16)
            k = layer.self_attn.k_proj(hidden).to(torch.bfloat16)
            v = layer.self_attn.v_proj(hidden).to(torch.bfloat16)
            q = q.view(1, 1, num_q_heads, head_dim).transpose(1, 2)
            k = k.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary(q, k, cos_l, sin_l)
            ks, vs = k.squeeze(2), v.squeeze(2)
            m.append_token_kv(0, ks[0].to(kv_device), vs[0].to(kv_device))

            q_kv = q.squeeze(2).to(kv_device)
            out = backend(q_kv, c.k_cache, c.v_cache, m.block_tables, m.context_lens, li)
            out = out.to(ld).view(1, 1, num_q_heads * head_dim)
            hidden = res + layer.self_attn.o_proj(out)
            res = hidden
            hidden = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = res + hidden
            if hasattr(backend, 'step_done'):
                backend.step_done(li, num_layers)

        logits = lm_head(hidden.to(head_dev).squeeze(1))
        cur_token = logits.argmax(dim=-1).squeeze()
        generated.append(cur_token.item())
        #  EOS multi-ID break
        if cur_token.item() in eos_ids:
            break

    del layer_caches, layer_managers; gc.collect(); torch.cuda.empty_cache()
    return tokenizer.decode(generated, skip_special_tokens=True)


#  Configs (AR only)
configs = [
    ('dense', None),
    ('ar20_c2_dL0', dict(budget_pct=0.05, coverage_threshold=0.90, lazy=True, check_interval=2,
                         approx_recovery_pct=0.20, dense_layers=[0])),
    ('ar20_c4_dL0', dict(budget_pct=0.05, coverage_threshold=0.90, lazy=True, check_interval=4,
                         approx_recovery_pct=0.20, dense_layers=[0])),
]

#  Resume support
def load_or_init():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            return json.load(f)
    return {
        'meta': {
            'model': MODEL, 'tasks': TASKS, 'max_samples': N_SAMPLES,
            'eos_ids': eos_ids,
            'fix_notes': 'multi-EOS break + first-line truncation for non-NO_SPLIT_TASKS; samsum dropped per framework limitation',
        },
        'scores': {n: {} for n, _ in configs},
        'per_sample_scores': {n: {} for n, _ in configs},
        'predictions': {n: {} for n, _ in configs},
    }

state = load_or_init()

def save():
    with open(OUT_PATH + '.tmp', 'w') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(OUT_PATH + '.tmp', OUT_PATH)

t_start = time.time()
for name, kwargs in configs:
    print(f'\n Config: {name} ', flush=True)
    for task in TASKS:
        if task in state['scores'][name]:
            v = state['scores'][name][task]
            print(f'  {task:<22} SKIP (done: {v:.2f})', flush=True)
            continue
        print(f'  Task: {task}', flush=True)
        data = load_dataset('THUDM/LongBench', task, split='test', trust_remote_code=True)
        prompt_fmt = DATASET2PROMPT[task]
        max_gen = DATASET2MAXGEN[task]
        metric_fn = DATASET2METRIC[task]

        sample_scores = []
        task_preds = []
        for i in range(min(N_SAMPLES, len(data))):
            sample = data[i]
            prompt = prompt_fmt.format(**sample)

            if kwargs is None:
                backend = DenseAttentionBackend(num_kv_heads, num_q_heads, head_dim, block_size)
            else:
                backend = CVSABackend(
                    num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
                    head_dim=head_dim, block_size=block_size, num_layers=num_layers,
                    reuse_interval=16, **kwargs)
            try:
                pred_full = generate_70b(prompt, backend, max_gen, task=task)
                # Per-task post-processing
                if task in NO_SPLIT_TASKS:
                    pred_for_metric = pred_full
                else:
                    pred_for_metric = pred_full.split('\n')[0].strip()
                answers = sample['answers']
                if isinstance(answers, str):
                    answers = [answers]
                score = metric_fn(pred_for_metric, answers)
                sample_scores.append(float(score))
                task_preds.append(pred_full)
            except Exception as e:
                print(f'    Sample {i} error: {e}', flush=True)
                import traceback; traceback.print_exc()
                sample_scores.append(0.0)
                task_preds.append(None)
            if (i + 1) % 10 == 0:
                print(f'    [{i+1}/{min(N_SAMPLES, len(data))}] running_avg={np.mean(sample_scores)*100:.1f}', flush=True)

        task_mean = float(np.mean(sample_scores) * 100) if sample_scores else 0.0
        state['scores'][name][task] = task_mean
        state['per_sample_scores'][name][task] = sample_scores
        state['predictions'][name][task] = task_preds
        if task in ('trec', 'lcc'):
            for k_, p in enumerate(task_preds[:2]):
                if p is not None:
                    s = p[:200].replace('\n', '\\n')
                    print(f'      sample[{k_}] preview: {s}', flush=True)
        total_min = (time.time() - t_start) / 60
        print(f'    {task}: {task_mean:.2f}  [t+{total_min:.0f}m]', flush=True)
        save()
    cfg_mean = float(np.mean(list(state['scores'][name].values())))
    print(f'   {name} 7-task MEAN: {cfg_mean:.2f} ', flush=True)

#  Summary
print('\n' + '='*80)
print('70B AR QUALITY FIXED (7-task, samsum dropped) RESULTS')
print('='*80)
dense_mean = float(np.mean(list(state['scores']['dense'].values())))
print(f'{"config":<16} {"7-task mean":>12} {" vs dense":>12}')
for name, _ in configs:
    if name not in state['scores'] or not state['scores'][name]:
        continue
    m = float(np.mean(list(state['scores'][name].values())))
    print(f'{name:<16} {m:>12.2f} {m-dense_mean:>+12.2f}')

# Cross-pipeline check vs KVPress
try:
    kvp = json.load(open('results/kvpress_baselines_llama70b.json'))['per_sample_scores']
    print('\n--- Cross-pipeline check (7-task means) ---')
    for label, key in [('KVPress dense', 'hf_dense'),
                       ('KVPress SnapKV', 'hf_snapkv'),
                       ('KVPress AdaKV', 'hf_adakv')]:
        if key not in kvp:
            continue
        per_task_means_kvp = []
        for t in TASKS:  # SAME 7 tasks
            s = kvp[key].get(t)
            if s and len(s) > 0:
                m = np.mean(s)
                per_task_means_kvp.append(m * (100 if m < 1.5 else 1))
        if per_task_means_kvp:
            kvp_m = float(np.mean(per_task_means_kvp))
            print(f'  {label:<18} {kvp_m:.2f}')
    print(f'\n  Paged dense (this run, 7-task) : {dense_mean:.2f}')
except Exception as e:
    print(f'(KVPress comparison skipped: {e})')

print(f'\nTotal wall-time: {(time.time()-t_start)/60:.1f} min')
print(f'Saved: {OUT_PATH}')
PY

echo ""
echo "=== Done at $(date) ==="
