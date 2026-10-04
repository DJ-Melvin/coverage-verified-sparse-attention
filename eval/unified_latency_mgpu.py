"""Multi-GPU (layer-sharded) batched decode-latency benchmark.

Targets the Llama/Mistral B=8, >=65K throughput cells. Those cells do not fit
a single H200 (Llama B=8 @128K KV cache is ~137 GB), so this benchmark shards
the model's LAYERS across the visible GPUs via `device_map="auto"`, places each layer's paged KV cache on that layer's
GPU, and moves activations across per-layer devices in the manual decode loop.
Each layer's attention (and the CVSA backend / Triton kernels) run entirely on
one device, so no kernel needs to be device-parallel.

This is PIPELINE (layer) parallelism, not tensor parallelism — chosen because
the paged-decode + Triton CVSA path assumes single-device tensors per call.

Relationship to the single-GPU path:
  * eval/unified_latency.py  -> single-GPU B=1/B=8 (existing paper cells).
  * this file                -> multi-GPU only for cells that don't fit 1 GPU.
Numbers for cells that fit one GPU should be taken from the single-GPU script;
this benchmark is for the >=65K B=8 Llama/Mistral cells.

*** UNTESTED on multi-GPU hardware. Run the smoke config first
    (--smoke) on the target multi-GPU node and confirm it reproduces a known
    single-GPU cell before trusting the >=65K numbers. ***

Usage (on a >=2 GPU node; 4x H200 recommended for 128K):
  python eval/unified_latency_mgpu.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --context-lens 65536,131072 --batch-sizes 8 \
      --methods dense,cvsa_5_90_c4_ar20_dL0 \
      --num-decode 64 \
      --output results/camera_ready/longctx_mgpu_llama.json

  # Smoke (validate correctness vs a single-GPU cell first):
  python eval/unified_latency_mgpu.py --smoke \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --output results/camera_ready/longctx_mgpu_smoke.json
"""

import argparse
import gc
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root: cvsa.*, eval.*
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager

# Reuse the exact method->backend factory from the single-GPU benchmark so the
# method strings (dense, fixed_*, cvsa_*_c4_ar20_dL0, snapkv_*, ...) match.
from eval.latency_benchmark import make_backend


def _sync_all():
    """Synchronize every visible CUDA device (pipeline spans multiple GPUs)."""
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def generate_synthetic_prompt(tokenizer, target_len):
    base_text = (
        "The transformer architecture has revolutionized natural language "
        "processing since its introduction. Self-attention mechanisms allow "
        "models to attend to all positions in the input sequence simultaneously, "
        "enabling parallel computation and capturing long-range dependencies. "
    )
    tokens = tokenizer.encode(base_text)
    n_repeats = (target_len // max(len(tokens), 1)) + 2
    long_text = base_text * n_repeats
    tokens = tokenizer.encode(long_text)[:target_len]
    return tokenizer.decode(tokens)


@torch.no_grad()
def benchmark_batched_mgpu(model, prompt, num_decode, backend, block_size,
                           batch_size, warmup=3, track_timing=False):
    """Layer-sharded multi-GPU batched decode. Wall-clock timed with a full
    multi-device sync around each step (CUDA-event timing is not valid across
    devices)."""
    config = model.config
    dtype = model.dtype
    B = batch_size

    layers = model.get_layers()
    # Per-layer device (where that layer's weights live under device_map=auto).
    layer_devices = [next(layer.parameters()).device for layer in layers]
    embed_dev = model.model.model.embed_tokens.weight.device
    norm_dev = model.model.model.norm.weight.device
    lmhead_dev = model.model.lm_head.weight.device

    # Prefill via HF's own multi-device forward (device_map=auto handles it).
    input_ids = model.tokenizer.encode(
        prompt, return_tensors="pt", truncation=False).to(embed_dev)
    seq_len = input_ids.shape[1]
    inner_model = model.model.model
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values
    first_hidden = outputs.last_hidden_state[:, -1:, :].squeeze(1).to(lmhead_dev)
    first_token = model.model.lm_head(first_hidden).argmax(dim=-1)

    n_pages = (seq_len + block_size - 1) // block_size
    extra = max(16, (num_decode + block_size - 1) // block_size + 4)
    num_blocks = (n_pages + extra + 64) * B

    # One paged cache per layer, placed on that layer's device.
    layer_caches, layer_managers = [], []
    for li in range(config.num_layers):
        dev = layer_devices[li]
        cache = PagedKVCache(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
            dtype=dtype, device=str(dev))
        mgr = SequenceKVManager(cache, B,
                                max_context_len=seq_len + num_decode + 16)
        # past_kv[li] tensors already live on layer li's device under auto.
        k = past_kv[li][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype).to(dev)
        v = past_kv[li][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype).to(dev)
        for b in range(B):
            mgr.prefill_kv(b, k, v)
        layer_caches.append(cache)
        layer_managers.append(mgr)

    del outputs, past_kv
    gc.collect()
    torch.cuda.empty_cache()

    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    backend.reset()
    # Per-kernel CUDA-event timing (scoring / coverage / sparse_draft / ar_recovery
    # / lse_merge / complement). Enable for the latency-breakdown of the 1.19x
    # cells. Adds small per-kernel event overhead → run this as a dedicated
    # breakdown pass; take the headline speedup from a separate non-timing run.
    if track_timing and hasattr(backend, "enable_timing"):
        backend.enable_timing(True)
    cur_tokens = first_token.expand(B).to(embed_dev)

    def run_step(pos):
        nonlocal cur_tokens
        hidden = model.embed_tokens(cur_tokens.unsqueeze(1))          # embed_dev
        pos_ids = torch.full((B, 1), pos, device=embed_dev)
        cos, sin = rotary_emb(hidden, pos_ids)                        # embed_dev
        for li, layer in enumerate(layers):
            dev = layer_devices[li]
            # Triton kernels (page scoring, coverage, sparse decode) launch on
            # the CURRENT cuda device; under layer sharding this layer's tensors
            # live on `dev`, so make it current or Triton hits
            # cudaErrorInvalidAddressSpace. torch ops are device-correct anyway.
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            hidden = hidden.to(dev, non_blocking=True)
            cos_l = cos.to(dev, non_blocking=True)
            sin_l = sin.to(dev, non_blocking=True)
            c, m = layer_caches[li], layer_managers[li]
            res = hidden
            hidden = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(hidden).to(c.k_cache.dtype)
            k = layer.self_attn.k_proj(hidden).to(c.k_cache.dtype)
            v = layer.self_attn.v_proj(hidden).to(c.k_cache.dtype)
            q = q.view(B, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(B, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos_l, sin_l)
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
        hidden = hidden.to(norm_dev)
        hidden = model.final_norm(hidden)[:, 0, :].to(lmhead_dev)
        logits = model.model.lm_head(hidden.to(model.model.lm_head.weight.dtype))
        cur_tokens = logits.argmax(dim=-1).to(embed_dev)

    # Warmup
    for step in range(warmup):
        run_step(seq_len + step)
    _sync_all()

    # Timed (wall-clock, full multi-device sync per step)
    step_ms = []
    for step in range(num_decode):
        _sync_all()
        t0 = time.perf_counter()
        run_step(seq_len + warmup + step)
        _sync_all()
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    ms = np.array(step_ms)
    result = {
        "batch_size": B, "context_len": seq_len, "num_decode": num_decode,
        "num_gpus": torch.cuda.device_count(),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "tokens_per_sec": float(B * 1000.0 / ms.mean()),
    }
    if hasattr(backend, "get_stats"):
        try:
            result["cvsa_stats"] = backend.get_stats()
        except Exception as e:
            result["cvsa_stats_error"] = str(e)

    del layer_caches, layer_managers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def estimate_kv_gb(config, ctx_len, B):
    return (2 * ctx_len * config.num_kv_heads * config.head_dim * 2
            * config.num_layers * B) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--context-lens", default="65536,131072")
    ap.add_argument("--batch-sizes", default="8")
    ap.add_argument("--methods", default="dense,cvsa_5_90_c4_ar20_dL0")
    ap.add_argument("--num-decode", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timing", action="store_true",
                    help="Enable per-kernel CUDA-event timing (scoring/coverage/"
                         "sparse_draft/ar_recovery/lse_merge). For the #7 latency "
                         "breakdown of the 1.19x cells. Run as a dedicated pass; "
                         "take speedup from a separate non-timing run.")
    ap.add_argument("--output", default="results/camera_ready/longctx_mgpu.json")
    ap.add_argument("--smoke", action="store_true",
                    help="Small config to validate correctness before the "
                         "real >=65K run (ctx=8192, B=8, dense + c4).")
    args = ap.parse_args()

    if args.smoke:
        context_lens = [8192]
        batch_sizes = [8]
        methods = ["dense", "cvsa_5_90_c4_ar20_dL0"]
        num_decode = 16
    else:
        context_lens = [int(x) for x in args.context_lens.split(",")]
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
        methods = [m.strip() for m in args.methods.split(",")]
        num_decode = args.num_decode

    n_gpu = torch.cuda.device_count()
    print("Multi-GPU (layer-sharded) latency benchmark")
    print("=" * 70)
    print(f"Model:   {args.model}")
    print(f"Visible GPUs: {n_gpu}")
    if n_gpu < 2:
        print("WARNING: <2 GPUs visible. This benchmark is only needed for "
              "cells that don't fit one GPU; use eval/unified_latency.py for "
              "single-GPU cells.")
    for d in range(n_gpu):
        name = torch.cuda.get_device_name(d)
        tot = torch.cuda.get_device_properties(d).total_memory / 1e9
        print(f"  cuda:{d}  {name}  {tot:.0f}GB")

    model = ModelForDecode(args.model, dtype=torch.bfloat16, device_map="auto")
    config = model.config
    print(f"  {config.num_q_heads}Q/{config.num_kv_heads}KV, "
          f"D={config.head_dim}, {config.num_layers}L\n")

    total_gpu_gb = sum(torch.cuda.get_device_properties(d).total_memory
                       for d in range(n_gpu)) / 1e9
    weights_gb = 16.0  # ~bf16 7-8B; rough headroom accounting

    all_results = {}
    for ctx_len in context_lens:
        for B in batch_sizes:
            kv_gb = estimate_kv_gb(config, ctx_len, B)
            print(f"\n{'=' * 70}\nContext {ctx_len}  Batch {B}  "
                  f"(est KV {kv_gb:.0f}GB across {n_gpu} GPUs, "
                  f"~{kv_gb/max(n_gpu,1):.0f}GB/GPU)\n{'=' * 70}")
            if kv_gb + weights_gb > total_gpu_gb * 0.92:
                print(f"  SKIP: est {kv_gb + weights_gb:.0f}GB > "
                      f"{total_gpu_gb*0.92:.0f}GB usable. Add more GPUs.")
                continue
            prompt = generate_synthetic_prompt(model.tokenizer, ctx_len)
            for method in methods:
                print(f"  {method:<28} (B={B}, ctx={ctx_len})...",
                      end="", flush=True)
                try:
                    backend = make_backend(method, config, 16)
                    r = benchmark_batched_mgpu(
                        model, prompt, num_decode, backend, 16, B,
                        warmup=args.warmup, track_timing=args.timing)
                    all_results[f"{method}@{ctx_len}@B{B}"] = r
                    print(f"  {r['mean_ms']:.1f}ms  {r['tokens_per_sec']:.0f}tok/s")
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  OOM: {e}")
                    all_results[f"{method}@{ctx_len}@B{B}"] = {"error": "OOM"}
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results[f"{method}@{ctx_len}@B{B}"] = {"error": str(e)}
                gc.collect()
                torch.cuda.empty_cache()

    # Speedup vs dense per (ctx, B)
    for ctx_len in context_lens:
        for B in batch_sizes:
            dk = f"dense@{ctx_len}@B{B}"
            if dk not in all_results or "mean_ms" not in all_results[dk]:
                continue
            dense_ms = all_results[dk]["mean_ms"]
            for method in methods:
                k = f"{method}@{ctx_len}@B{B}"
                if k in all_results and "mean_ms" in all_results[k]:
                    all_results[k]["speedup_vs_dense"] = (
                        dense_ms / all_results[k]["mean_ms"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model": args.model, "num_gpus": n_gpu,
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
