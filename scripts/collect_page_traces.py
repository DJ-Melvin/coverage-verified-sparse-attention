#!/usr/bin/env python3
"""Collect per-step page access traces for the recovery heatmap.

Runs the CVSA paper configuration on LongBench samples and records which KV
pages it accesses at each decode step, per layer, together with the per-head
accept/reject decision of the coverage gate. The traces feed
eval/plot_recovery_heatmap.py.

Methods:
    cvsa_ar20_c4_dL0  — 5% budget, tau=0.90, c4, ar20, layer 0 dense
                        (`vsa_ar20_c4_dL0` is accepted as an alias; output
                        files are named after the string passed)
    dense             — all valid pages (reference)

Usage:
    python scripts/collect_page_traces.py \
        --methods cvsa_ar20_c4_dL0 \
        --task hotpotqa \
        --num-samples 5 \
        --max-steps 128 \
        --output results/page_traces/

Requires GPU.
"""

import argparse
import importlib
import json
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager
from cvsa.runtime.dense import DenseAttentionBackend
from eval.longbench_harness import DATASET2PROMPT
from cvsa.backend import CVSABackend
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Tracing wrapper
# ---------------------------------------------------------------------------

class TracingWrapper:
    """Wraps any attention backend to record page access traces.

    After each decode step (all layers processed), records the set of unique
    page indices accessed by the backend (union across all KV heads), per layer.

    Attributes:
        traces: list of dicts, one per decode step.
            traces[step][layer_idx] = sorted list of accessed page indices.
    """

    def __init__(self, backend, block_size, num_layers, rich=False):
        self.backend = backend
        self.block_size = block_size
        self.num_layers = num_layers
        self.rich = rich  # capture per-head accept/reject metadata
        self.traces = []
        self.rich_traces = [] if rich else None
        self._current_step = {}
        self._current_rich = {} if rich else None

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens, layer_idx):
        n_valid = int(((context_lens[0].float() - 1) / self.block_size).long() + 1)

        output = self.backend(q, k_cache, v_cache, block_tables, context_lens, layer_idx)

        pages = self._extract_pages(layer_idx, n_valid)
        self._current_step[layer_idx] = sorted(pages)

        if self.rich:
            self._current_rich[layer_idx] = self._extract_rich(layer_idx)

        return output

    def _extract_pages(self, layer_idx, n_valid):
        """Extract the set of page indices accessed by the backend at this layer."""
        b = self.backend

        # Dense: all valid pages
        if isinstance(b, DenseAttentionBackend):
            return set(range(n_valid))

        # CVSA: _cached_selection is always current after any step type.
        if isinstance(b, CVSABackend):
            if layer_idx in b._dense_layers:
                return set(range(n_valid))
            if layer_idx in b._cached_selection:
                page_indices, n_per_head = b._cached_selection[layer_idx]
                pages = set()
                for h in range(page_indices.shape[1]):
                    n = int(n_per_head[0, h].item())
                    pages.update(page_indices[0, h, :n].cpu().tolist())
                # On scoring steps with rejects, include expanded/complement
                # pages for rejected heads (ar20 or exact recovery).
                exp = b._last_expanded_selection.get(layer_idx)
                accept = b._last_accept_mask.get(layer_idx)
                if exp is not None and accept is not None:
                    exp_pi, exp_nph = exp
                    for h in range(exp_pi.shape[1]):
                        if not accept[0, h]:  # rejected head
                            n = int(exp_nph[0, h].item())
                            pages.update(exp_pi[0, h, :n].cpu().tolist())
                return pages
            return set(range(n_valid))  # fallback before first scoring

        # Fallback: all pages
        return set(range(n_valid))

    def _extract_rich(self, layer_idx):
        """Extract per-head accept/reject metadata (lightweight)."""
        b = self.backend
        if not isinstance(b, CVSABackend):
            return None
        accept = b._last_accept_mask.get(layer_idx)
        if accept is None:
            return None
        n_kv = accept.shape[1]
        mask = accept[0].cpu().tolist()  # [H_kv] booleans
        has_exp = layer_idx in b._last_expanded_selection and \
                  b._last_expanded_selection[layer_idx] is not None
        return {
            "accept_mask": mask,
            "n_rejected": sum(1 for v in mask if not v),
            "has_expanded": has_exp,
        }

    def step_done(self, layer_idx, num_layers):
        self.backend.step_done(layer_idx, num_layers)
        if layer_idx == num_layers - 1:
            self.traces.append(dict(self._current_step))
            self._current_step = {}
            if self.rich:
                self.rich_traces.append(dict(self._current_rich))
                self._current_rich = {}

    def reset(self):
        self.backend.reset()
        self.traces = []
        self._current_step = {}
        if self.rich:
            self.rich_traces = []
            self._current_rich = {}

    def get_traces(self):
        return self.traces

    def __getattr__(self, name):
        return getattr(self.backend, name)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def make_tracing_backend(method, config, block_size):
    """Create attention backend wrapped with TracingWrapper."""
    nl = config.num_layers

    if method == "dense":
        backend = DenseAttentionBackend(
            config.num_kv_heads, config.num_q_heads,
            config.head_dim, block_size)

    elif method in ("cvsa_ar20_c4_dL0", "vsa_ar20_c4_dL0"):
        # Paper's main config: 5% budget, ar20, c4, layer-0 dense.
        # `vsa_ar20_c4_dL0` is the legacy name of the same configuration.
        backend = CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=nl,
            budget_pct=0.05,
            coverage_threshold=0.90,
            lazy=True, check_interval=4,
            approx_recovery_pct=0.20,
            dense_layers=[0])

    else:
        raise ValueError(f"Unknown method: {method}")

    rich = isinstance(backend, CVSABackend)
    return TracingWrapper(backend, block_size, nl, rich=rich)


# ---------------------------------------------------------------------------
# Simplified generate (no EOS stopping, fixed number of steps)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_trace(model, prompt, max_steps, backend, block_size=16):
    """Run decode loop for exactly max_steps steps, recording page traces.

    Unlike E2EGenerator.generate(), this never stops at EOS because we need a
    fixed number of steps for consistent trace collection.

    Returns (seq_len, n_pages) metadata.
    """
    config = model.config
    tokenizer = model.tokenizer
    device = model.device
    dtype = model.dtype

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                 truncation=False).to(device)
    seq_len = input_ids.shape[1]

    # Prefill
    inner_model = model.model.model  # e.g. LlamaModel
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values

    # First generated token
    normed = outputs.last_hidden_state[:, -1:, :]
    logits = model.lm_head(normed.squeeze(1))
    next_token = logits.argmax(dim=-1)  # [1]

    # Build paged KV caches
    n_pages = (seq_len + block_size - 1) // block_size
    extra_pages = max(16, (max_steps + block_size - 1) // block_size + 4)
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
                                    max_context_len=seq_len + max_steps + 16)

        k = past_kv[layer_idx][0][0, :, :seq_len, :]  # [H_kv, seq_len, D]
        v = past_kv[layer_idx][1][0, :, :seq_len, :]
        k_seq = k.transpose(0, 1).contiguous().to(dtype)  # [seq_len, H_kv, D]
        v_seq = v.transpose(0, 1).contiguous().to(dtype)
        manager.prefill_kv(0, k_seq, v_seq)

        layer_caches.append(cache)
        layer_managers.append(manager)

    del outputs, past_kv
    torch.cuda.empty_cache()

    # Decode setup
    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    cur_token = next_token
    backend.reset()

    # Decode loop: exactly max_steps, no EOS stopping.
    for step in range(max_steps):
        position = seq_len + step
        hidden = model.embed_tokens(cur_token.unsqueeze(0))  # [1, 1, hidden]
        position_ids = torch.tensor([[position]], device=device)
        cos, sin = rotary_emb(hidden, position_ids)

        for layer_idx, layer in enumerate(layers):
            cache = layer_caches[layer_idx]
            manager = layer_managers[layer_idx]

            residual = hidden
            hidden = layer.input_layernorm(hidden)

            q = layer.self_attn.q_proj(hidden)
            k = layer.self_attn.k_proj(hidden)
            v = layer.self_attn.v_proj(hidden)

            cache_dtype = cache.k_cache.dtype
            q, k, v = q.to(cache_dtype), k.to(cache_dtype), v.to(cache_dtype)

            q = q.view(1, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            k_store = k.squeeze(2)  # [1, H_kv, D]
            v_store = v.squeeze(2)
            manager.append_token_kv(0, k_store[0], v_store[0])

            q_for_attn = q.squeeze(2)  # [1, H_q, D]
            attn_output = backend(
                q_for_attn,
                cache.k_cache, cache.v_cache,
                manager.block_tables, manager.context_lens,
                layer_idx)

            attn_output = attn_output.view(1, 1,
                                           config.num_q_heads * config.head_dim)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden = residual + attn_output

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

            backend.step_done(layer_idx, config.num_layers)

        # LM head (greedy)
        hidden = model.final_norm(hidden)
        lm_dtype = model.model.lm_head.weight.dtype
        logits = model.lm_head(hidden[:, 0, :].to(lm_dtype))
        cur_token = logits.argmax(dim=-1)

    # Cleanup
    del layer_caches, layer_managers
    torch.cuda.empty_cache()

    return seq_len, n_pages


# ---------------------------------------------------------------------------
# Prompt preparation (simplified from LongBenchE2ERunner)
# ---------------------------------------------------------------------------

def prepare_prompts(tokenizer, task, data, max_length=31500, model_name=""):
    """Format and truncate LongBench prompts."""
    prompt_format = DATASET2PROMPT[task]
    prompts = []
    use_chat = ("instruct" in model_name.lower() or "chat" in model_name.lower())
    no_chat_tasks = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

    for item in data:
        prompt = prompt_format.format(**item)
        tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokens) > max_length:
            half = max_length // 2
            prompt = (tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokens[-half:], skip_special_tokens=True))
        if use_chat and task not in no_chat_tasks and \
                hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect page access traces for cache simulation")
    parser.add_argument("--model",
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--methods", default="cvsa_ar20_c4_dL0")
    parser.add_argument("--task", default="hotpotqa")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--output", default="results/page_traces/")
    args = parser.parse_args()

    print("Page Trace Collection for Cache Simulation")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Task:        {args.task}")
    print(f"  Methods:     {args.methods}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  Decode steps:{args.max_steps}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = ModelForDecode(args.model, dtype=torch.bfloat16)
    config = model.config
    print(f"  Loaded in {time.time() - t0:.1f}s  "
          f"({config.num_q_heads}Q/{config.num_kv_heads}KV, "
          f"D={config.head_dim}, {config.num_layers}L)")

    # Load dataset
    print(f"Loading {args.task} dataset...")
    data = load_dataset("THUDM/LongBench", args.task, split="test",
                        trust_remote_code=True)
    if args.num_samples < len(data):
        data = data.select(range(args.num_samples))

    prompts = prepare_prompts(model.tokenizer, args.task, data,
                              model_name=config.model_name)
    print(f"  {len(prompts)} samples prepared")

    methods = args.methods.split(",")
    os.makedirs(args.output, exist_ok=True)

    all_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        method_samples = []

        for i, prompt in enumerate(prompts):
            t0 = time.time()
            backend = make_tracing_backend(method, config, args.block_size)

            seq_len, n_pages = generate_trace(
                model, prompt, args.max_steps, backend, args.block_size)

            traces = backend.get_traces()
            n_steps = len(traces)
            elapsed = time.time() - t0

            # Compute stats
            if traces:
                pages_per_step = [
                    sum(len(pages) for pages in step.values()) / config.num_layers
                    for step in traces
                ]
                avg_pages = sum(pages_per_step) / len(pages_per_step)
            else:
                avg_pages = 0

            sample_data = {
                "sample_idx": i,
                "seq_len": seq_len,
                "n_pages": n_pages,
                "n_steps": n_steps,
                "traces": traces,
            }
            if backend.rich and backend.rich_traces:
                sample_data["rich_traces"] = backend.rich_traces
            method_samples.append(sample_data)

            print(f"  Sample {i}: seq={seq_len}, pages={n_pages}, "
                  f"steps={n_steps}, avg_pages/layer={avg_pages:.1f}, "
                  f"time={elapsed:.1f}s")

            # Collect CVSA stats if available.
            inner = backend.backend
            if hasattr(inner, 'get_stats'):
                stats = inner.get_stats()
                if stats.get('total_scoring_steps', 0) > 0:
                    print(f"           accept={stats['all_accepted_pct']:.0f}% "
                          f"per_head={stats['per_head_accept_rate']:.0f}% "
                          f"rescore={stats.get('rescore_rate', 100):.0f}%")

        result = {
            "method": method,
            "model": args.model,
            "block_size": args.block_size,
            "num_layers": config.num_layers,
            "num_kv_heads": config.num_kv_heads,
            "task": args.task,
            "max_steps": args.max_steps,
            "samples": method_samples,
        }
        all_results[method] = result

        # Save per-method
        outpath = os.path.join(args.output, f"{method}.pkl")
        with open(outpath, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved {outpath}")

    # Save combined
    combined_path = os.path.join(args.output, "all_traces.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nAll traces saved {combined_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary: avg pages accessed per step per layer")
    print(f"{'='*60}")
    for method, result in all_results.items():
        all_avg = []
        for s in result["samples"]:
            if s["traces"]:
                avg = sum(
                    sum(len(p) for p in step.values()) / config.num_layers
                    for step in s["traces"]
                ) / len(s["traces"])
                all_avg.append(avg)
        n_pages = result["samples"][0]["n_pages"] if result["samples"] else 0
        mean_avg = sum(all_avg) / len(all_avg) if all_avg else 0
        frac = mean_avg / max(n_pages, 1) * 100
        print(f"  {method:<12}: {mean_avg:>6.1f} / {n_pages} pages "
              f"({frac:.1f}%)")


if __name__ == "__main__":
    main()
