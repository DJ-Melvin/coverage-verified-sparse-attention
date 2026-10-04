#!/bin/bash
#SBATCH --job-name=diag-v03
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/diagnostics_multimodel_v03_%j.out
#SBATCH --error=logs/diagnostics_multimodel_v03_%j.err

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

echo "Starting multi-model diagnostics (Mistral v0.3) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -c "
import sys, os, torch, json, time, importlib
import numpy as np
sys.path.insert(0, '.')
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager
from cvsa.kernels.page_scores import triton_score_pages
from cvsa.kernels.coverage import triton_check_coverage

def run_diagnostics(model, model_name, n_steps=64):
    config = model.config
    num_q_per_kv = config.num_q_heads // config.num_kv_heads
    block_size = 16

    base = ('The transformer architecture has revolutionized natural language processing. ' * 100)
    prompt = model.tokenizer.decode(model.tokenizer.encode(base)[:8192])

    input_ids = model.tokenizer.encode(prompt, return_tensors='pt', truncation=False).to('cuda')
    seq_len = input_ids.shape[1]
    print(f'\n{\"=\"*60}')
    print(f'Model: {model_name}')
    print(f'Config: layers={config.num_layers}, q={config.num_q_heads}, kv={config.num_kv_heads}')
    print(f'Context: {seq_len} tokens')
    print(f'{\"=\"*60}')

    inner_model = model.model.model
    with torch.no_grad():
        outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values

    n_pages = (seq_len + block_size - 1) // block_size
    layer_caches, layer_managers = [], []
    for li in range(config.num_layers):
        cache = PagedKVCache(num_blocks=n_pages + 64, block_size=block_size,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
            dtype=torch.bfloat16, device='cuda')
        mgr = SequenceKVManager(cache, 1, max_context_len=seq_len + n_steps + 16)
        k = past_kv[li][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(torch.bfloat16)
        v = past_kv[li][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(torch.bfloat16)
        mgr.prefill_kv(0, k, v)
        layer_caches.append(cache)
        layer_managers.append(mgr)
    del outputs, past_kv; torch.cuda.empty_cache()

    layers = model.get_layers()
    attn_mod = type(layers[0].self_attn).__module__
    apply_rotary = importlib.import_module(attn_mod).apply_rotary_pos_emb
    rotary = model.model.model.rotary_emb

    budget_configs = [(0.05, 0.90), (0.10, 0.90)]
    layer_stats = {li: {f'{int(b*100)}_{int(t*100)}': [] for b, t in budget_configs}
                   for li in range(config.num_layers)}

    first_logits = model.lm_head(
        model.model.model(input_ids[:, -1:], use_cache=False, return_dict=True
        ).last_hidden_state.squeeze(1))
    cur_tokens = first_logits.argmax(dim=-1)

    for step in range(n_steps):
        pos = seq_len + step
        hidden = model.embed_tokens(cur_tokens.unsqueeze(1))
        pos_ids = torch.full((1, 1), pos, device='cuda')
        cos, sin = rotary(hidden, pos_ids)

        for li, layer in enumerate(layers):
            c, m = layer_caches[li], layer_managers[li]
            res = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(c.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(c.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(c.k_cache.dtype)
            q = q.view(1, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)
            ks, vs = k.squeeze(2), v.squeeze(2)
            m.append_token_kv(0, ks[0], vs[0])

            q_flat = q.squeeze(2)
            n_valid = ((m.context_lens.float() - 1) / block_size).long() + 1

            page_scores = triton_score_pages(
                q_flat, c.k_cache, m.block_tables, m.context_lens,
                config.num_kv_heads, num_q_per_kv, block_size)

            for budget_pct, tau in budget_configs:
                budget = max(3, int(m.block_tables.shape[1] * budget_pct))
                _, topk_indices = torch.topk(page_scores, k=budget, dim=-1, largest=True, sorted=False)
                page_indices = topk_indices.to(torch.int32)
                n_per_head = torch.full((1, config.num_kv_heads), budget,
                    dtype=torch.int32, device='cuda')
                n_per_head = torch.min(n_per_head,
                    n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32))

                accept_mask, coverage = triton_check_coverage(
                    page_scores, page_indices, n_per_head, tau)

                key = f'{int(budget_pct*100)}_{int(tau*100)}'
                for h in range(config.num_kv_heads):
                    layer_stats[li][key].append(accept_mask[0, h].item())

            backend = DenseAttentionBackend(config.num_kv_heads, config.num_q_heads, config.head_dim, block_size)
            out = backend(q_flat, c.k_cache, c.v_cache, m.block_tables, m.context_lens, li)
            out = out.view(1, 1, config.num_q_heads * config.head_dim)
            hidden = res + layer.self_attn.o_proj(out)
            res = hidden
            hidden = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = res + hidden

        logits = model.lm_head(hidden.squeeze(1))
        cur_tokens = logits.argmax(dim=-1)

        if (step + 1) % 16 == 0:
            print(f'  Step {step+1}/{n_steps}', flush=True)

    del layer_caches, layer_managers; torch.cuda.empty_cache()

    result = {}
    for b_pct, tau in budget_configs:
        key = f'{int(b_pct*100)}_{int(tau*100)}'
        print(f'\n  Budget={b_pct*100:.0f}%, tau={tau}:')
        print(f'  {\"Layer\":>5} {\"Accept%\":>8} {\"Category\":>12}')
        print(f'  {\"-\"*30}')

        layer_rates = []
        for li in range(config.num_layers):
            accepts = layer_stats[li][key]
            rate = np.mean(accepts) * 100
            layer_rates.append(rate)
            cat = 'SKIP' if rate >= 99.0 else ('PROBLEM' if rate < 80 else 'MODERATE')
            if rate < 80 or rate >= 99.0 or li in [0, config.num_layers-1]:
                print(f'  L{li:>3d}  {rate:>7.1f}%  {cat}')

        n_skip = sum(1 for r in layer_rates if r >= 99.0)
        n_problem = sum(1 for r in layer_rates if r < 80)
        n_moderate = config.num_layers - n_skip - n_problem
        print(f'  Summary: {n_skip} skip-able (>99%), {n_moderate} moderate, {n_problem} problem (<80%)')

        result[key] = {
            'per_layer': {str(li): layer_rates[li] for li in range(config.num_layers)},
            'n_skip': n_skip, 'n_problem': n_problem, 'n_moderate': n_moderate,
        }

    return result

#  Run on all models (Mistral v0.3)
all_results = {}

for model_name in [
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'Qwen/Qwen2.5-7B-Instruct',
]:
    print(f'\nLoading {model_name}...')
    t0 = time.time()
    model = ModelForDecode(model_name, dtype=torch.bfloat16, device='cuda')
    print(f'Loaded in {time.time()-t0:.1f}s')
    result = run_diagnostics(model, model_name)
    all_results[model_name] = result
    del model; torch.cuda.empty_cache()

#  Cross-model comparison
print('\n' + '='*70)
print('CROSS-MODEL LAYER-ADAPTIVE COMPARISON (5% & 10% budget, tau=0.90)')
print('='*70)
for key_label, key in [('5%', '5_90'), ('10%', '10_90')]:
    print(f'\n--- Budget={key_label} ---')
    for mname, result in all_results.items():
        short = mname.split('/')[-1]
        r = result[key]
        total = r['n_skip'] + r['n_moderate'] + r['n_problem']
        print(f'{short}: skip={r[\"n_skip\"]}/{total}, moderate={r[\"n_moderate\"]}, problem={r[\"n_problem\"]}')
        problem = [int(li) for li, rate in r['per_layer'].items() if rate < 80]
        if problem:
            print(f'  Problem layers: {problem}')

with open('results/diagnostics_multimodel.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSaved to results/diagnostics_multimodel.json')
"

echo "Done at $(date)"
