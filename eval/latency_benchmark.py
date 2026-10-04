#!/usr/bin/env python3
"""Latency benchmark for CVSA vs dense vs fixed-sparse.

Measures wall-clock decode latency on Llama-3.1-8B at various context lengths.
Uses real model weights and real text to ensure representative KV patterns.

Reports:
  - Per-token decode latency (ms)
  - Tokens/second throughput
  - Speedup vs dense
  - CVSA breakdown: scoring vs reuse steps, accept/reject ratio

Usage:
    cd /path/to/cvsa
    python eval/latency_benchmark.py \
        --context-lens 4096,8192,16384,32768 \
        --methods dense,fixed_10,cvsa_10_90,cvsa_10_95 \
        --num-decode 256 \
        --output results/latency.json
"""

import argparse
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
from cvsa.backend import CVSABackend


def make_backend(method: str, config, block_size: int):
    """Factory for attention backends.

    Method format:
        dense               — FlashInfer dense baseline
        fixed_{B}           — Fixed sparse, B% budget
        cvsa_{B}_{T}        — CVSA, B% budget, tau=T/100 (fixed interval R=16)
        cvsa_{B}_{T}_c{M}   — Lazy scheduling with check interval M
        cvsa_{B}_{T}_ar{P}  — Approximate recovery, expand to P%
        cvsa_{B}_{T}_dL{L}  — layer L always dense (e.g. dL0, dL0.8.9)
        snapkv_* adakv_* h2o_* quest_*
                            — baselines from eval/baselines.py

    `ssa_` is accepted as an alias of `cvsa_` and builds the identical
    backend. Suffixes can be combined (e.g. cvsa_5_90_c4_ar20_dL0); an
    unrecognized suffix raises ValueError.
    """
    # `ssa_` is the legacy spelling of `cvsa_`; both build the same backend.
    if method.startswith("ssa_"):
        method = "cvsa_" + method[len("ssa_"):]

    if method == "dense":
        return DenseAttentionBackend(
            config.num_kv_heads, config.num_q_heads,
            config.head_dim, block_size)

    if method.startswith("fixed_"):
        parts = method.split("_")
        if len(parts) != 2:
            raise ValueError(
                f"Method '{method}': fixed_{{B}} takes no suffixes here")
        budget_pct = int(parts[1]) / 100.0
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=budget_pct,
            coverage_threshold=0.0,
            reuse_interval=16,
        )

    if method.startswith("cvsa_"):
        parts = method.split("_")
        budget_pct = int(parts[1]) / 100.0
        tau = int(parts[2]) / 100.0
        dense_layers = None
        check_interval = None
        approx_recovery_pct = None
        for p in parts[3:]:
            if p.startswith("dL"):
                dense_layers = [int(x) for x in p[2:].split(".")]
            elif p.startswith("c") and p[1:].isdigit():
                check_interval = int(p[1:])
            elif p.startswith("ar") and p[2:].isdigit():
                approx_recovery_pct = int(p[2:]) / 100.0
            else:
                raise ValueError(
                    f"Method '{method}': unrecognized suffix '{p}'")
        lazy = check_interval is not None
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=budget_pct,
            coverage_threshold=tau,
            reuse_interval=16,
            dense_layers=dense_layers,
            lazy=lazy,
            check_interval=check_interval,
            approx_recovery_pct=approx_recovery_pct,
        )

    # Fall through to the published baselines (snapkv_*, adakv_*, h2o_*,
    # quest_*) in eval/baselines.py.
    try:
        from eval.baselines import make_backend as _make_baseline_backend
    except ImportError as e:
        raise ValueError(f"Unknown method: {method} (baseline import failed: {e})")
    return _make_baseline_backend(method, config, block_size, config.num_layers)


def generate_synthetic_prompt(tokenizer, target_len):
    """Generate a prompt of approximately target_len tokens.

    Uses repeating real text to fill the context, ensuring realistic
    token distribution for RoPE and attention patterns.
    """
    # A paragraph of real English text
    base_text = (
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
    # Repeat to reach target length
    tokens = tokenizer.encode(base_text)
    n_repeats = (target_len // len(tokens)) + 2
    long_text = base_text * n_repeats
    tokens = tokenizer.encode(long_text)[:target_len]
    return tokenizer.decode(tokens)


@torch.no_grad()
def benchmark_decode(model, prompt, num_decode, backend, block_size,
                     warmup=5):
    """Run decode loop and measure per-token latency.

    Returns dict with timing and stats.
    """
    import importlib
    config = model.config
    device = model.device
    dtype = model.dtype

    # Reset peak-memory tracker to capture benchmark-scope allocation only
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    mem_before_bytes = torch.cuda.memory_allocated(device)

    # Tokenize and prefill
    input_ids = model.tokenizer.encode(
        prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = input_ids.shape[1]

    inner_model = model.model.model
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values

    next_token = model.lm_head(
        outputs.last_hidden_state[:, -1:, :].squeeze(1)).argmax(dim=-1)

    # Build paged KV caches
    n_pages = (seq_len + block_size - 1) // block_size
    extra_pages = max(16, (num_decode + block_size - 1) // block_size + 4)
    max_pages = n_pages + extra_pages
    num_blocks = max_pages + 64

    layer_caches = []
    layer_managers = []
    for layer_idx in range(config.num_layers):
        cache = PagedKVCache(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
            dtype=dtype, device=str(device))
        manager = SequenceKVManager(cache, 1,
                                    max_context_len=seq_len + num_decode + 16)
        k = past_kv[layer_idx][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        v = past_kv[layer_idx][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        manager.prefill_kv(0, k, v)
        layer_caches.append(cache)
        layer_managers.append(manager)

    del outputs, past_kv
    torch.cuda.empty_cache()

    # Decode setup
    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    backend.reset()
    cur_token = next_token

    # Warmup
    for step in range(warmup):
        position = seq_len + step
        hidden = model.embed_tokens(cur_token.unsqueeze(0))
        position_ids = torch.tensor([[position]], device=device)
        cos, sin = rotary_emb(hidden, position_ids)

        for layer_idx, layer in enumerate(layers):
            cache = layer_caches[layer_idx]
            manager = layer_managers[layer_idx]
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(cache.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(cache.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(cache.k_cache.dtype)
            q = q.view(1, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            manager.append_token_kv(0, k.squeeze(2)[0], v.squeeze(2)[0])
            attn_output = backend(
                q.squeeze(2), cache.k_cache, cache.v_cache,
                manager.block_tables, manager.context_lens, layer_idx)
            attn_output = attn_output.view(1, 1, config.num_q_heads * config.head_dim)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden = residual + attn_output
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            backend.step_done(layer_idx, config.num_layers)

        lm_dtype = model.model.lm_head.weight.dtype
        logits = model.lm_head(
            model.final_norm(hidden)[:, 0, :].to(lm_dtype))
        cur_token = logits.argmax(dim=-1)

    torch.cuda.synchronize()

    # Reset stats after warmup
    if hasattr(backend, 'reset'):
        # Don't full-reset (loses reuse state), just reset counters
        if hasattr(backend, '_accept_counts'):
            backend._accept_counts = 0
            backend._reject_counts = 0
            backend._total_scoring_steps = 0
            backend._per_head_accept_sum = 0
            backend._per_head_total = 0

    # Timed decode — measure both E2E and attention-only
    torch.cuda.synchronize()
    e2e_start = torch.cuda.Event(enable_timing=True)
    e2e_end = torch.cuda.Event(enable_timing=True)
    attn_start = torch.cuda.Event(enable_timing=True)
    attn_end = torch.cuda.Event(enable_timing=True)

    per_step_ms = []
    per_step_attn_ms = []

    for step in range(num_decode):
        position = seq_len + warmup + step

        torch.cuda.synchronize()
        e2e_start.record()

        hidden = model.embed_tokens(cur_token.unsqueeze(0))
        position_ids = torch.tensor([[position]], device=device)
        cos, sin = rotary_emb(hidden, position_ids)

        step_attn_ms = 0.0
        for layer_idx, layer in enumerate(layers):
            cache = layer_caches[layer_idx]
            manager = layer_managers[layer_idx]
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(cache.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(cache.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(cache.k_cache.dtype)
            q = q.view(1, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            manager.append_token_kv(0, k.squeeze(2)[0], v.squeeze(2)[0])

            # Attention-only timing
            attn_start.record()
            attn_output = backend(
                q.squeeze(2), cache.k_cache, cache.v_cache,
                manager.block_tables, manager.context_lens, layer_idx)
            attn_end.record()
            torch.cuda.synchronize()
            step_attn_ms += attn_start.elapsed_time(attn_end)

            attn_output = attn_output.view(1, 1, config.num_q_heads * config.head_dim)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden = residual + attn_output
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            backend.step_done(layer_idx, config.num_layers)

        lm_dtype = model.model.lm_head.weight.dtype
        logits = model.lm_head(
            model.final_norm(hidden)[:, 0, :].to(lm_dtype))
        cur_token = logits.argmax(dim=-1)

        e2e_end.record()
        torch.cuda.synchronize()
        per_step_ms.append(e2e_start.elapsed_time(e2e_end))
        per_step_attn_ms.append(step_attn_ms)

    # Collect results
    ms_arr = np.array(per_step_ms)
    attn_arr = np.array(per_step_attn_ms)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    result = {
        "context_len": seq_len,
        "num_decode": num_decode,
        # E2E timing
        "mean_ms": float(ms_arr.mean()),
        "median_ms": float(np.median(ms_arr)),
        "p95_ms": float(np.percentile(ms_arr, 95)),
        "p99_ms": float(np.percentile(ms_arr, 99)),
        "std_ms": float(ms_arr.std()),
        "tokens_per_sec": float(1000.0 / ms_arr.mean()),
        # Attention-only timing
        "attn_mean_ms": float(attn_arr.mean()),
        "attn_median_ms": float(np.median(attn_arr)),
        "attn_p95_ms": float(np.percentile(attn_arr, 95)),
        "attn_pct_of_e2e": float(attn_arr.mean() / ms_arr.mean() * 100),
        # Memory (bytes)
        "peak_memory_bytes": int(peak_bytes),
        "peak_memory_gb": float(peak_bytes / 1e9),
        "mem_before_bytes": int(mem_before_bytes),
        "benchmark_delta_gb": float((peak_bytes - mem_before_bytes) / 1e9),
    }

    if hasattr(backend, 'get_stats'):
        result["cvsa_stats"] = backend.get_stats()

    # Cleanup
    del layer_caches, layer_managers
    torch.cuda.empty_cache()

    return result


@torch.no_grad()
def benchmark_decode_batched(model, prompt, num_decode, backend, block_size,
                             batch_size, warmup=5):
    """Run batched decode loop (B>1) and measure per-step latency.

    All B sequences use the same prompt (same prefill KV).
    Returns dict with timing and stats.
    """
    import importlib
    config = model.config
    device = model.device
    dtype = model.dtype
    B = batch_size

    # Reset peak-memory tracker to capture benchmark-scope allocation only
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    mem_before_bytes = torch.cuda.memory_allocated(device)

    # Tokenize and prefill (single sequence, then replicate)
    input_ids = model.tokenizer.encode(
        prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = input_ids.shape[1]

    inner_model = model.model.model
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values

    first_token = model.lm_head(
        outputs.last_hidden_state[:, -1:, :].squeeze(1)).argmax(dim=-1)

    # Build paged KV caches — B sequences share the same cache pool
    n_pages = (seq_len + block_size - 1) // block_size
    extra_pages = max(16, (num_decode + block_size - 1) // block_size + 4)
    max_pages = n_pages + extra_pages
    num_blocks = (max_pages + 64) * B  # enough blocks for all sequences

    layer_caches = []
    layer_managers = []
    for layer_idx in range(config.num_layers):
        cache = PagedKVCache(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
            dtype=dtype, device=str(device))
        manager = SequenceKVManager(cache, B,
                                    max_context_len=seq_len + num_decode + 16)

        k = past_kv[layer_idx][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        v = past_kv[layer_idx][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        # Prefill each sequence with the same KV
        for b in range(B):
            manager.prefill_kv(b, k, v)

        layer_caches.append(cache)
        layer_managers.append(manager)

    del outputs, past_kv
    torch.cuda.empty_cache()

    # Decode setup
    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    backend.reset()
    cur_tokens = first_token.expand(B)  # [B]

    # Warmup
    for step in range(warmup):
        position = seq_len + step
        hidden = model.embed_tokens(cur_tokens.unsqueeze(1))  # [B, 1, hidden]
        position_ids = torch.full((B, 1), position, device=device)
        cos, sin = rotary_emb(hidden, position_ids)

        for layer_idx, layer in enumerate(layers):
            cache = layer_caches[layer_idx]
            manager = layer_managers[layer_idx]
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(cache.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(cache.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(cache.k_cache.dtype)
            q = q.view(B, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_store = k.squeeze(2)  # [B, H_kv, D]
            v_store = v.squeeze(2)
            for b in range(B):
                manager.append_token_kv(b, k_store[b], v_store[b])
            q_for_attn = q.squeeze(2)  # [B, H_q, D]
            attn_output = backend(
                q_for_attn, cache.k_cache, cache.v_cache,
                manager.block_tables, manager.context_lens, layer_idx)
            attn_output = attn_output.view(B, 1, config.num_q_heads * config.head_dim)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden = residual + attn_output
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            backend.step_done(layer_idx, config.num_layers)

        lm_dtype = model.model.lm_head.weight.dtype
        logits = model.lm_head(
            model.final_norm(hidden)[:, 0, :].to(lm_dtype))
        cur_tokens = logits.argmax(dim=-1)  # [B]

    torch.cuda.synchronize()

    # Reset stats after warmup
    if hasattr(backend, '_accept_counts'):
        backend._accept_counts = 0
        backend._reject_counts = 0
        backend._total_scoring_steps = 0
        backend._per_head_accept_sum = 0
        backend._per_head_total = 0

    # Timed decode — E2E only, NO per-layer sync (realistic throughput)
    torch.cuda.synchronize()
    e2e_start = torch.cuda.Event(enable_timing=True)
    e2e_end = torch.cuda.Event(enable_timing=True)

    per_step_ms = []

    for step in range(num_decode):
        position = seq_len + warmup + step

        torch.cuda.synchronize()
        e2e_start.record()

        hidden = model.embed_tokens(cur_tokens.unsqueeze(1))  # [B, 1, hidden]
        position_ids = torch.full((B, 1), position, device=device)
        cos, sin = rotary_emb(hidden, position_ids)

        for layer_idx, layer in enumerate(layers):
            cache = layer_caches[layer_idx]
            manager = layer_managers[layer_idx]
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(cache.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(cache.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(cache.k_cache.dtype)
            q = q.view(B, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_store = k.squeeze(2)
            v_store = v.squeeze(2)
            for b in range(B):
                manager.append_token_kv(b, k_store[b], v_store[b])

            q_for_attn = q.squeeze(2)  # [B, H_q, D]
            attn_output = backend(
                q_for_attn, cache.k_cache, cache.v_cache,
                manager.block_tables, manager.context_lens, layer_idx)

            attn_output = attn_output.view(B, 1, config.num_q_heads * config.head_dim)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden = residual + attn_output
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            backend.step_done(layer_idx, config.num_layers)

        lm_dtype = model.model.lm_head.weight.dtype
        logits = model.lm_head(
            model.final_norm(hidden)[:, 0, :].to(lm_dtype))
        cur_tokens = logits.argmax(dim=-1)

        e2e_end.record()
        torch.cuda.synchronize()
        per_step_ms.append(e2e_start.elapsed_time(e2e_end))

    # Collect results
    ms_arr = np.array(per_step_ms)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    result = {
        "batch_size": B,
        "context_len": seq_len,
        "num_decode": num_decode,
        "mean_ms": float(ms_arr.mean()),
        "median_ms": float(np.median(ms_arr)),
        "p95_ms": float(np.percentile(ms_arr, 95)),
        "p99_ms": float(np.percentile(ms_arr, 99)),
        "std_ms": float(ms_arr.std()),
        "tokens_per_sec": float(B * 1000.0 / ms_arr.mean()),
        # Memory (bytes)
        "peak_memory_bytes": int(peak_bytes),
        "peak_memory_gb": float(peak_bytes / 1e9),
        "mem_before_bytes": int(mem_before_bytes),
        "benchmark_delta_gb": float((peak_bytes - mem_before_bytes) / 1e9),
    }

    if hasattr(backend, 'get_stats'):
        result["cvsa_stats"] = backend.get_stats()

    del layer_caches, layer_managers
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CVSA Latency Benchmark")
    parser.add_argument("--model",
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--context-lens",
                        default="4096,8192,16384",
                        help="Comma-separated context lengths")
    parser.add_argument("--methods",
                        default="dense,fixed_10,cvsa_10_90,cvsa_10_95")
    parser.add_argument("--num-decode", type=int, default=256,
                        help="Number of decode steps to measure")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup decode steps before measurement (eats "
                             "Triton autotune / JIT cost; increase for new "
                             "shapes e.g. 128K B=8)")
    parser.add_argument("--batch-sizes",
                        default="1",
                        help="Comma-separated batch sizes (e.g., 1,4,8)")
    parser.add_argument("--output",
                        default="results/latency.json")
    parser.add_argument("--track-timing", action="store_true",
                        help="Enable per-kernel CUDA-event timing in the CVSA backend")
    args = parser.parse_args()

    print("CVSA Latency Benchmark")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Context lengths: {args.context_lens}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Methods: {args.methods}")
    print(f"  Decode steps: {args.num_decode}")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = ModelForDecode(args.model, dtype=torch.bfloat16)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    config = model.config
    print(f"  {config.num_q_heads}Q/{config.num_kv_heads}KV, "
          f"D={config.head_dim}, {config.num_layers}L")
    print()

    context_lens = [int(x) for x in args.context_lens.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    methods = args.methods.split(",")

    all_results = {}

    for batch_size in batch_sizes:
        for ctx_len in context_lens:
            # Check memory feasibility
            kv_gb = 2 * ctx_len * config.num_kv_heads * config.head_dim * 2 * config.num_layers * batch_size / 1e9
            print(f"\n{'=' * 70}")
            print(f"Context: {ctx_len}, Batch: {batch_size}  "
                  f"(est KV: {kv_gb:.1f} GB)")
            print(f"{'=' * 70}")

            prompt = generate_synthetic_prompt(model.tokenizer, ctx_len)
            actual_len = len(model.tokenizer.encode(prompt))
            n_pages = (actual_len + args.block_size - 1) // args.block_size
            print(f"  Actual tokens: {actual_len}, Pages: {n_pages}")

            for method in methods:
                print(f"\n  Method: {method}  (B={batch_size})", flush=True)
                backend = make_backend(method, config, args.block_size)
                if args.track_timing and hasattr(backend, "enable_timing"):
                    backend.enable_timing(True)

                try:
                    if batch_size == 1:
                        result = benchmark_decode(
                            model, prompt, args.num_decode, backend,
                            args.block_size, warmup=args.warmup)
                    else:
                        result = benchmark_decode_batched(
                            model, prompt, args.num_decode, backend,
                            args.block_size, batch_size, warmup=args.warmup)

                    key = f"{method}@{ctx_len}@B{batch_size}"
                    all_results[key] = result

                    tok_s = result['tokens_per_sec']
                    print(f"    E2E:  {result['mean_ms']:.2f} ms/step  "
                          f"({tok_s:.1f} tok/s)")
                    print(f"    P95: {result['p95_ms']:.2f} ms  "
                          f"P99: {result['p99_ms']:.2f} ms")

                    if "cvsa_stats" in result:
                        stats = result["cvsa_stats"]
                        if stats["total_scoring_steps"] > 0:
                            print(f"    CVSA: accept={stats['all_accepted_pct']:.0f}%  "
                                  f"per_head={stats['per_head_accept_rate']:.0f}%  "
                                  f"scoring_steps={stats['total_scoring_steps']}")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"    ERROR: {e}")

                gc.collect()
                torch.cuda.empty_cache()

    # Summary tables — one per batch size
    for batch_size in batch_sizes:
        bs_tag = f"B{batch_size}"
        print(f"\n\n{'=' * 80}")
        print(f"LATENCY SUMMARY — {bs_tag} (ms/step, throughput in tok/s)")
        print(f"{'=' * 80}")

        header = f"{'Method':<20}"
        for ctx_len in context_lens:
            header += f" {ctx_len:>12}"
        print(header)
        print("-" * len(header))

        for method in methods:
            row = f"{method:<20}"
            for ctx_len in context_lens:
                key = f"{method}@{ctx_len}@{bs_tag}"
                if key in all_results:
                    r = all_results[key]
                    row += f" {r['mean_ms']:>5.1f}/{r['tokens_per_sec']:>5.0f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

        # Speedup vs dense
        if "dense" in methods:
            for mode, label, key_field in [
                ("E2E", "E2E SPEEDUP", "mean_ms"),
            ]:
                print(f"\n{label} vs dense — {bs_tag}:")
                header = f"{'Method':<20}"
                for ctx_len in context_lens:
                    header += f" {ctx_len:>12}"
                print(header)
                print("-" * len(header))

                for method in methods:
                    if method == "dense":
                        continue
                    row = f"{method:<20}"
                    for ctx_len in context_lens:
                        dk = f"dense@{ctx_len}@{bs_tag}"
                        mk = f"{method}@{ctx_len}@{bs_tag}"
                        if (dk in all_results and mk in all_results
                                and key_field in all_results[dk]
                                and key_field in all_results[mk]
                                and all_results[mk][key_field] > 0):
                            speedup = (all_results[dk][key_field] /
                                       all_results[mk][key_field])
                            row += f" {speedup:>11.2f}x"
                        else:
                            row += f" {'N/A':>12}"
                    print(row)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": vars(args),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
