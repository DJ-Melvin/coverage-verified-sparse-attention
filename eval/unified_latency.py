#!/usr/bin/env python3
"""Unified latency benchmark: ALL methods in ONE run, same conditions.

No per-layer CUDA sync. Fair comparison for all methods.
Includes both CVSA and the baselines from eval/baselines.py.

Usage:
    python eval/unified_latency.py
"""

import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager
from eval.baselines import make_backend as make_baseline_backend
from cvsa.backend import CVSABackend


def generate_prompt(tokenizer, target_len):
    base = (
        "The transformer architecture has revolutionized natural language "
        "processing since its introduction. Self-attention mechanisms allow "
        "models to attend to all positions in the input sequence simultaneously, "
        "enabling parallel computation and capturing long-range dependencies "
        "effectively. However, the quadratic complexity of self-attention with "
        "respect to sequence length poses significant challenges for processing "
        "long documents. Various approaches have been proposed to address this "
        "limitation, including sparse attention patterns, linear attention "
        "approximations, and hierarchical attention mechanisms. "
    )
    tokens = tokenizer.encode(base)
    n_repeats = (target_len // len(tokens)) + 2
    tokens = tokenizer.encode(base * n_repeats)[:target_len]
    return tokenizer.decode(tokens)


def make_backend(method, config, block_size):
    """Create backend. Handles both CVSA and baseline methods.

    Method format:
        dense                — FlashInfer dense baseline
        fixed_{B}            — Fixed sparse, B% budget, no verification
        cvsa_{B}_{T}         — CVSA, B% budget, tau=T/100 (fixed interval R=16)
        lazy_{B}_{T}         — CVSA with a coverage check at every step (M=1)
        cvsa_/lazy_..._c{M}  — Lazy scheduling with check interval M
        cvsa_/lazy_..._ar{P} — Approximate recovery, expand to P%
        cvsa_/lazy_..._dL{L} — layer L always dense (e.g. dL0, dL0.8.9)
        snapkv_* adakv_* h2o_* quest_*
                             — baselines from eval/baselines.py

    `ssa_` is accepted as an alias of `cvsa_` and builds the identical
    backend. An unrecognized suffix raises ValueError.
    """
    # `ssa_` is the legacy spelling of `cvsa_`; both build the same backend.
    if method.startswith("ssa_"):
        method = "cvsa_" + method[len("ssa_"):]

    if method == "dense":
        return DenseAttentionBackend(
            config.num_kv_heads, config.num_q_heads,
            config.head_dim, block_size)

    if method.startswith("cvsa_") or method.startswith("fixed_") or method.startswith("lazy_"):
        parts = method.split("_")
        if method.startswith("fixed_"):
            if len(parts) != 2:
                raise ValueError(
                    f"Method '{method}': fixed_{{B}} takes no suffixes here")
            budget = int(parts[1]) / 100.0
            tau = 0.0
            lazy = False
        elif method.startswith("lazy_"):
            budget = int(parts[1]) / 100.0
            tau = int(parts[2]) / 100.0
            lazy = True
        else:
            budget = int(parts[1]) / 100.0
            lazy = False
            tau = int(parts[2]) / 100.0
        check_interval = None
        approx_recovery_pct = None
        dense_layers = None
        for p in parts[3:]:
            if p.startswith("c") and p[1:].isdigit():
                check_interval = int(p[1:])
            elif p.startswith("ar") and p[2:].isdigit():
                approx_recovery_pct = int(p[2:]) / 100.0
            elif p.startswith("dL") and p[2:].replace(".", "").isdigit():
                dense_layers = [int(x) for x in p[2:].split(".")]
            else:
                raise ValueError(
                    f"Method '{method}': unrecognized suffix '{p}'")
        if check_interval is not None:
            lazy = True
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=budget,
            coverage_threshold=tau,
            reuse_interval=16,
            lazy=lazy,
            check_interval=check_interval,
            approx_recovery_pct=approx_recovery_pct,
            dense_layers=dense_layers)

    # Baseline from eval/baselines.py
    return make_baseline_backend(method, config, block_size, config.num_layers)


@torch.no_grad()
def benchmark_batched(model, prompt, num_decode, backend, block_size,
                      batch_size, warmup=5):
    """Batched decode benchmark. NO per-layer sync. Pure E2E timing."""
    import importlib
    config = model.config
    device = model.device
    dtype = model.dtype
    B = batch_size

    input_ids = model.tokenizer.encode(
        prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = input_ids.shape[1]

    inner_model = model.model.model
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values
    first_token = model.lm_head(
        outputs.last_hidden_state[:, -1:, :].squeeze(1)).argmax(dim=-1)

    n_pages = (seq_len + block_size - 1) // block_size
    extra = max(16, (num_decode + block_size - 1) // block_size + 4)
    num_blocks = (n_pages + extra + 64) * B

    layer_caches, layer_managers = [], []
    for li in range(config.num_layers):
        cache = PagedKVCache(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
            dtype=dtype, device=str(device))
        mgr = SequenceKVManager(cache, B,
                                max_context_len=seq_len + num_decode + 16)
        k = past_kv[li][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        v = past_kv[li][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        for b in range(B):
            mgr.prefill_kv(b, k, v)
        layer_caches.append(cache)
        layer_managers.append(mgr)

    del outputs, past_kv
    torch.cuda.empty_cache()

    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    backend.reset()
    cur_tokens = first_token.expand(B)

    # Warmup
    for step in range(warmup):
        pos = seq_len + step
        hidden = model.embed_tokens(cur_tokens.unsqueeze(1))
        pos_ids = torch.full((B, 1), pos, device=device)
        cos, sin = rotary_emb(hidden, pos_ids)
        for li, layer in enumerate(layers):
            c, m = layer_caches[li], layer_managers[li]
            res = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(c.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(c.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(c.k_cache.dtype)
            q = q.view(B, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            ks, vs = k.squeeze(2), v.squeeze(2)
            for b in range(B):
                m.append_token_kv(b, ks[b], vs[b])
            out = backend(q.squeeze(2), c.k_cache, c.v_cache,
                          m.block_tables, m.context_lens, li)
            out = out.view(B, 1, config.num_q_heads * config.head_dim)
            hidden = res + layer.self_attn.o_proj(out)
            res = hidden
            hidden = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = res + hidden
            backend.step_done(li, config.num_layers)
        logits = model.lm_head(
            model.final_norm(hidden)[:, 0, :].to(model.model.lm_head.weight.dtype))
        cur_tokens = logits.argmax(dim=-1)

    torch.cuda.synchronize()
    if hasattr(backend, '_accept_counts'):
        backend._accept_counts = 0
        backend._reject_counts = 0
        backend._total_scoring_steps = 0
        backend._per_head_accept_sum = 0
        backend._per_head_total = 0

    # Timed — NO per-layer sync
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    step_ms = []

    for step in range(num_decode):
        pos = seq_len + warmup + step
        torch.cuda.synchronize()
        start_ev.record()

        hidden = model.embed_tokens(cur_tokens.unsqueeze(1))
        pos_ids = torch.full((B, 1), pos, device=device)
        cos, sin = rotary_emb(hidden, pos_ids)
        for li, layer in enumerate(layers):
            c, m = layer_caches[li], layer_managers[li]
            res = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(c.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(c.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(c.k_cache.dtype)
            q = q.view(B, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            ks, vs = k.squeeze(2), v.squeeze(2)
            for b in range(B):
                m.append_token_kv(b, ks[b], vs[b])
            out = backend(q.squeeze(2), c.k_cache, c.v_cache,
                          m.block_tables, m.context_lens, li)
            out = out.view(B, 1, config.num_q_heads * config.head_dim)
            hidden = res + layer.self_attn.o_proj(out)
            res = hidden
            hidden = layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = res + hidden
            backend.step_done(li, config.num_layers)
        logits = model.lm_head(
            model.final_norm(hidden)[:, 0, :].to(model.model.lm_head.weight.dtype))
        cur_tokens = logits.argmax(dim=-1)

        end_ev.record()
        torch.cuda.synchronize()
        step_ms.append(start_ev.elapsed_time(end_ev))

    ms = np.array(step_ms)
    result = {
        "batch_size": B, "context_len": seq_len, "num_decode": num_decode,
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "tokens_per_sec": float(B * 1000.0 / ms.mean()),
    }
    if hasattr(backend, 'get_stats'):
        result["cvsa_stats"] = backend.get_stats()

    del layer_caches, layer_managers
    torch.cuda.empty_cache()
    return result


def main():
    print("Unified Latency Benchmark (all methods, same conditions)")
    print("=" * 70)

    model = ModelForDecode(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    config = model.config
    print(f"  {config.num_q_heads}Q/{config.num_kv_heads}KV, "
          f"D={config.head_dim}, {config.num_layers}L")
    gpu = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu}\n")

    # Methods: CVSA + kernel-fair baselines + baselines with Python overhead
    methods = [
        "dense",
        # Fixed sparse (no verification):
        "fixed_10",
        # CVSA fixed-interval R=16:
        "cvsa_10_90",
        # Lazy CVSA (check every M steps, reuse in between):
        "lazy_10_90_c2",   # check every 2 steps
        "lazy_10_90_c4",   # check every 4 steps
        "lazy_10_90_c8",   # check every 8 steps
        # Lazy CVSA (check every step, for reference):
        "lazy_10_90",
        # Published baselines:
        "snapkv_10",
        "adakv_10",
        "h2o_10",
    ]

    configs = [
        (8192, 1), (8192, 8), (8192, 32),
        (32768, 1), (32768, 8),
    ]

    all_results = {}

    for ctx_len, B in configs:
        kv_gb = 2 * ctx_len * config.num_kv_heads * config.head_dim * 2 * config.num_layers * B / 1e9
        if kv_gb + 16 > 75:
            print(f"\nSkipping ctx={ctx_len} B={B} (est {kv_gb+16:.0f}GB > 75GB)")
            continue

        print(f"\n{'=' * 70}")
        print(f"Context: {ctx_len}, Batch: {B}  (KV: {kv_gb:.1f}GB)")
        print(f"{'=' * 70}")

        prompt = generate_prompt(model.tokenizer, ctx_len)
        num_decode = 64 if B >= 16 else 128

        for method in methods:
            print(f"  {method:<16} (B={B})...", end="", flush=True)
            try:
                backend = make_backend(method, config, 16)
                if B == 1:
                    # Use B=1 path (simpler, from latency_benchmark)
                    result = benchmark_batched(
                        model, prompt, num_decode, backend, 16, 1)
                else:
                    result = benchmark_batched(
                        model, prompt, num_decode, backend, 16, B)

                key = f"{method}@{ctx_len}@B{B}"
                all_results[key] = result

                tps = result['tokens_per_sec']
                extra = ""
                if "cvsa_stats" in result:
                    s = result["cvsa_stats"]
                    if s.get("total_scoring_steps", 0) > 0:
                        extra = f"  acc={s['all_accepted_pct']:.0f}%"
                print(f"  {result['mean_ms']:>7.1f}ms  {tps:>6.0f}tok/s{extra}")

            except Exception as e:
                print(f"  ERROR: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    # Summary tables
    for ctx_len, B in configs:
        tag = f"ctx={ctx_len} B={B}"
        dk = f"dense@{ctx_len}@B{B}"
        if dk not in all_results:
            continue

        dense_ms = all_results[dk]["mean_ms"]
        dense_tps = all_results[dk]["tokens_per_sec"]

        print(f"\n{'─' * 65}")
        print(f"{tag}")
        print(f"{'Method':<16} {'ms/step':>8} {'tok/s':>8} {'E2E spd':>8} {'Tput gain':>10}")
        print(f"{'─' * 65}")

        for method in methods:
            mk = f"{method}@{ctx_len}@B{B}"
            if mk not in all_results:
                continue
            r = all_results[mk]
            spd = dense_ms / r["mean_ms"]
            tput = r["tokens_per_sec"] / dense_tps
            print(f"{method:<16} {r['mean_ms']:>7.1f} {r['tokens_per_sec']:>7.0f} "
                  f"{spd:>7.2f}x {tput:>9.2f}x")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/unified_latency.json", "w") as f:
        json.dump({"results": all_results}, f, indent=2, default=str)
    print(f"\nSaved to results/unified_latency.json")


if __name__ == "__main__":
    main()
