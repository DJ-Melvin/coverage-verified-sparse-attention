#!/usr/bin/env python3
"""Diagnostic-only hybrid coverage trace for CVSA.

This script runs dense decoding for correctness, while a side channel mimics
CVSA's hybrid schedule:

* every M decode steps, refresh the cached top-k page selection;
* on reuse steps, keep the cached selection fixed;
* at every step, measure the cached selection's coverage under current page
  scores.

The output is designed for the per-step coverage trace figure. It is not a
throughput benchmark and it does not alter the generated tokens.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from cvsa.runtime.dense import DenseAttentionBackend  # noqa: E402
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager  # noqa: E402
from cvsa.runtime.model_loader import ModelForDecode  # noqa: E402
from datasets import load_dataset  # noqa: E402
from cvsa.kernels.page_scores import triton_score_pages  # noqa: E402
from cvsa.kernels.coverage import triton_check_coverage  # noqa: E402


FALLBACK_PROMPTS = {
    "hotpotqa": (
        "Answer the question based on the given passages. Only give the "
        "answer.\n\n{context}\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer the question.\n\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "narrativeqa": (
        "Read the story and answer the question.\n\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "qasper": (
        "Read the paper and answer the question.\n\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
}


def _load_prompt_templates() -> Dict[str, str]:
    try:
        from eval.longbench_harness import DATASET2PROMPT  # type: ignore

        out = dict(FALLBACK_PROMPTS)
        out.update(DATASET2PROMPT)
        return out
    except Exception:
        return FALLBACK_PROMPTS


class HybridCoverageTraceBackend:
    """Dense attention plus side-channel cached-selection coverage tracing."""

    def __init__(
        self,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        block_size: int,
        num_layers: int,
        budget_pct: float,
        tau: float,
        check_interval: int,
        dense_layers: Sequence[int] = (),
    ):
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.budget_pct = budget_pct
        self.tau = tau
        self.check_interval = check_interval
        self.dense_layers = set(int(x) for x in dense_layers)

        self._dense = DenseAttentionBackend(
            num_kv_heads, num_q_heads, head_dim, block_size
        )
        self.reset()

    def reset(self):
        self._dense.reset()
        self._step = 0
        self._cached_selection: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.coverage_data: Dict[int, List[torch.Tensor]] = {
            l: [] for l in range(self.num_layers)
        }
        self.accept_data: Dict[int, List[torch.Tensor]] = {
            l: [] for l in range(self.num_layers)
        }
        self.refresh_steps: List[bool] = []
        self.n_pages_by_step: List[int] = []
        self.budget_by_step: List[int] = []

    def _is_refresh_step(self) -> bool:
        return self._step % self.check_interval == 0

    def step_done(self, layer_idx: int, num_layers: int):
        self._dense.step_done(layer_idx, num_layers)
        if layer_idx == num_layers - 1:
            self.refresh_steps.append(self._is_refresh_step())
            self._step += 1

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens, layer_idx):
        output = self._dense(
            q, k_cache, v_cache, block_tables, context_lens, layer_idx
        )

        n_pages = int(block_tables.shape[1])
        if layer_idx == 0:
            budget_for_record = max(3, int(n_pages * self.budget_pct))
            self.n_pages_by_step.append(n_pages)
            self.budget_by_step.append(min(budget_for_record, n_pages))

        if layer_idx in self.dense_layers:
            cov = torch.full(
                (q.shape[0], self.num_kv_heads),
                float("nan"),
                dtype=torch.float32,
                device=q.device,
            )
            acc = torch.ones(
                (q.shape[0], self.num_kv_heads), dtype=torch.bool, device=q.device
            )
            self.coverage_data[layer_idx].append(cov.detach().cpu())
            self.accept_data[layer_idx].append(acc.detach().cpu())
            return output

        if n_pages < 3:
            cov = torch.ones(
                (q.shape[0], self.num_kv_heads),
                dtype=torch.float32,
                device=q.device,
            )
            acc = torch.ones_like(cov, dtype=torch.bool)
            self.coverage_data[layer_idx].append(cov.detach().cpu())
            self.accept_data[layer_idx].append(acc.detach().cpu())
            return output

        page_scores = triton_score_pages(
            q,
            k_cache,
            block_tables,
            context_lens,
            self.num_kv_heads,
            self.num_q_per_kv,
            self.block_size,
        )

        n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
        budget = max(3, int(n_pages * self.budget_pct))
        budget = min(budget, n_pages)

        if self._is_refresh_step() or layer_idx not in self._cached_selection:
            _, topk_indices = torch.topk(
                page_scores, k=budget, dim=-1, largest=True, sorted=False
            )
            page_indices = topk_indices.to(torch.int32)
            n_per_head = torch.full(
                (q.shape[0], self.num_kv_heads),
                budget,
                dtype=torch.int32,
                device=q.device,
            )
            n_per_head = torch.min(
                n_per_head,
                n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32),
            ).clamp(max=budget)
            self._cached_selection[layer_idx] = (page_indices, n_per_head)

        page_indices, n_per_head = self._cached_selection[layer_idx]
        accept_mask, coverage = triton_check_coverage(
            page_scores, page_indices, n_per_head, self.tau
        )

        self.coverage_data[layer_idx].append(coverage.detach().cpu())
        self.accept_data[layer_idx].append(accept_mask.detach().cpu())

        return output

    def tensors(self) -> Dict[str, np.ndarray]:
        layer_covs = []
        layer_accept = []
        n_steps = min(len(v) for v in self.coverage_data.values())
        for layer in range(self.num_layers):
            cov = torch.stack(self.coverage_data[layer][:n_steps])[:, 0, :]
            acc = torch.stack(self.accept_data[layer][:n_steps])[:, 0, :]
            layer_covs.append(cov)
            layer_accept.append(acc)

        coverage_slh = torch.stack(layer_covs, dim=1).to(torch.float16).numpy()
        accept_slh = torch.stack(layer_accept, dim=1).numpy()
        return {
            "coverage_slh": coverage_slh,
            "accept_slh": accept_slh,
            "refresh_steps": np.array(self.refresh_steps[:n_steps], dtype=bool),
            "n_pages_by_step": np.array(self.n_pages_by_step[:n_steps], dtype=np.int32),
            "budget_by_step": np.array(self.budget_by_step[:n_steps], dtype=np.int32),
        }


def format_prompt(item: dict, template: str) -> str:
    try:
        return template.format(**item)
    except KeyError:
        context = item.get("context", "")
        question = item.get("input", item.get("question", ""))
        return f"{context}\n\nQuestion: {question}\nAnswer:"


def load_prompts(model, tasks: Sequence[str], num_samples: int, max_context: int):
    templates = _load_prompt_templates()
    prompts = []
    use_chat = "instruct" in model.config.model_name.lower()
    no_chat_tasks = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

    for task in tasks:
        data = load_dataset(
            "THUDM/LongBench", task, split="test", trust_remote_code=True
        )
        n = min(num_samples, len(data))
        template = templates.get(task, FALLBACK_PROMPTS.get(task, "{context}\n{input}"))
        for i in range(n):
            prompt = format_prompt(data[i], template)
            tokens = model.tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0]
            if len(tokens) > max_context:
                half = max_context // 2
                prompt = (
                    model.tokenizer.decode(tokens[:half], skip_special_tokens=True)
                    + model.tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                )
            if (
                use_chat
                and task not in no_chat_tasks
                and hasattr(model.tokenizer, "apply_chat_template")
            ):
                prompt = model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ctx_len = len(model.tokenizer.encode(prompt))
            prompts.append((task, i, prompt, ctx_len))
    return prompts


@torch.no_grad()
def generate_trace(model, prompt: str, max_steps: int, backend, block_size: int):
    config = model.config
    device = model.device
    dtype = model.dtype

    input_ids = model.tokenizer.encode(
        prompt, return_tensors="pt", truncation=False
    ).to(device)
    seq_len = input_ids.shape[1]

    inner_model = model.model.model
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values

    logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    cur_token = logits.argmax(dim=-1)

    n_pages = (seq_len + block_size - 1) // block_size
    extra_pages = max(16, (max_steps + block_size - 1) // block_size + 4)
    num_blocks = n_pages + extra_pages + 64

    layer_caches = []
    layer_managers = []
    for layer_idx in range(config.num_layers):
        cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=str(device),
        )
        manager = SequenceKVManager(
            cache, 1, max_context_len=seq_len + max_steps + 16
        )
        k = past_kv[layer_idx][0][0, :, :seq_len, :]
        v = past_kv[layer_idx][1][0, :, :seq_len, :]
        manager.prefill_kv(
            0,
            k.transpose(0, 1).contiguous().to(dtype),
            v.transpose(0, 1).contiguous().to(dtype),
        )
        layer_caches.append(cache)
        layer_managers.append(manager)

    del outputs, past_kv
    torch.cuda.empty_cache()

    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    backend.reset()
    for step in range(max_steps):
        position = seq_len + step
        hidden = model.embed_tokens(cur_token.unsqueeze(0))
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
            q = q.to(cache.k_cache.dtype)
            k = k.to(cache.k_cache.dtype)
            v = v.to(cache.k_cache.dtype)

            q = q.view(1, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
            k = k.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            v = v.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            manager.append_token_kv(0, k.squeeze(2)[0], v.squeeze(2)[0])
            q_for_attn = q.squeeze(2)

            attn_output = backend(
                q_for_attn,
                cache.k_cache,
                cache.v_cache,
                manager.block_tables,
                manager.context_lens,
                layer_idx,
            )

            attn_output = attn_output.view(
                1, 1, config.num_q_heads * config.head_dim
            )
            hidden = residual + layer.self_attn.o_proj(attn_output)

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)

            backend.step_done(layer_idx, config.num_layers)

        hidden = model.final_norm(hidden)
        logits = model.lm_head(hidden[:, 0, :].to(model.model.lm_head.weight.dtype))
        cur_token = logits.argmax(dim=-1)

        if (step + 1) % 32 == 0:
            print(f"  decoded {step + 1}/{max_steps} steps", flush=True)

    del layer_caches, layer_managers
    torch.cuda.empty_cache()
    return seq_len, n_pages


def summarize(
    samples: List[Dict], dense_layers: Sequence[int], tau: float, check_interval: int
):
    covs = [s["coverage_slh"].astype(np.float32) for s in samples]
    all_cov = np.concatenate(covs, axis=0)
    valid = np.ones(all_cov.shape[1], dtype=bool)
    valid[list(dense_layers)] = False
    cov_valid = all_cov[:, valid, :]

    out = {
        "mean_coverage": float(np.nanmean(cov_valid)),
        "p10_coverage": float(np.nanpercentile(cov_valid, 10)),
        "recovery_rate": float(np.nanmean(cov_valid < tau)),
        "by_step_mod": {},
    }
    for mod in range(check_interval):
        mod_covs = []
        for s in samples:
            cov = s["coverage_slh"].astype(np.float32)[:, valid, :]
            mod_covs.append(cov[np.arange(cov.shape[0]) % check_interval == mod])
        merged = np.concatenate(mod_covs, axis=0)
        out["by_step_mod"][str(mod)] = {
            "mean_coverage": float(np.nanmean(merged)),
            "recovery_rate": float(np.nanmean(merged < tau)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Collect diagnostic-only CVSA hybrid cached-coverage traces."
    )
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="results/hybrid_coverage_trace_llama.pt")
    parser.add_argument("--tasks", default="hotpotqa")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=384)
    parser.add_argument("--max-context", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--budget-pct", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--check-interval", type=int, default=4)
    parser.add_argument("--dense-layers", default="0")
    args = parser.parse_args()

    dense_layers = tuple(
        int(x) for x in args.dense_layers.split(",") if x.strip() != ""
    )
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    print("Hybrid coverage trace diagnostics")
    print("=" * 70)
    print(f"model:          {args.model}")
    print(f"tasks:          {tasks}")
    print(f"samples/task:   {args.num_samples}")
    print(f"steps/sample:   {args.max_steps}")
    print(f"budget:         {args.budget_pct:.3f}")
    print(f"tau:            {args.tau:.2f}")
    print(f"check interval: {args.check_interval}")
    print(f"dense layers:   {dense_layers}")

    t0 = time.time()
    model = ModelForDecode(args.model, dtype=torch.bfloat16, device="cuda")
    config = model.config
    print(
        f"loaded in {time.time() - t0:.1f}s "
        f"({config.num_layers}L, {config.num_q_heads}Q/{config.num_kv_heads}KV)"
    )

    prompts = load_prompts(model, tasks, args.num_samples, args.max_context)
    print(f"prepared {len(prompts)} prompts")

    samples = []
    for idx, (task, sample_idx, prompt, ctx_len) in enumerate(prompts):
        print(
            f"\n[{idx + 1}/{len(prompts)}] {task}#{sample_idx} "
            f"ctx={ctx_len} tokens"
        )
        backend = HybridCoverageTraceBackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=args.block_size,
            num_layers=config.num_layers,
            budget_pct=args.budget_pct,
            tau=args.tau,
            check_interval=args.check_interval,
            dense_layers=dense_layers,
        )
        seq_len, n_pages = generate_trace(
            model, prompt, args.max_steps, backend, args.block_size
        )
        trace = backend.tensors()
        trace.update(
            {
                "task": task,
                "sample_idx": int(sample_idx),
                "ctx_len": int(ctx_len),
                "seq_len": int(seq_len),
                "prefill_pages": int(n_pages),
            }
        )
        samples.append(trace)

        valid_layers = np.ones(config.num_layers, dtype=bool)
        valid_layers[list(dense_layers)] = False
        cov = trace["coverage_slh"].astype(np.float32)[:, valid_layers, :]
        print(
            f"  mean coverage={np.nanmean(cov):.4f}, "
            f"recovery@tau={np.nanmean(cov < args.tau) * 100:.1f}%"
        )

    summary = summarize(samples, dense_layers, args.tau, args.check_interval)
    print("\nSummary")
    print("=" * 70)
    print(
        f"mean coverage={summary['mean_coverage']:.4f}, "
        f"p10={summary['p10_coverage']:.4f}, "
        f"recovery={summary['recovery_rate'] * 100:.1f}%"
    )
    for mod, row in summary["by_step_mod"].items():
        print(
            f"  step mod 4 = {mod}: mean={row['mean_coverage']:.4f}, "
            f"recovery={row['recovery_rate'] * 100:.1f}%"
        )

    payload = {
        "model": args.model,
        "num_layers": config.num_layers,
        "num_q_heads": config.num_q_heads,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "block_size": args.block_size,
        "budget_pct": args.budget_pct,
        "tau": args.tau,
        "check_interval": args.check_interval,
        "dense_layers": list(dense_layers),
        "samples": samples,
        "summary": summary,
        "args": vars(args),
    }

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
