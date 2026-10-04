#!/usr/bin/env python3
"""Production-trace proxy calibration for CVSA.

This diagnostic measures the deployed page-level coverage proxy against
token-level coverage c* on actual LongBench decode traces. Generation uses
dense attention so the trace distribution is not affected by CVSA errors; the
coverage computations run as a side channel on the same per-step Q/KV state.

Primary output:
  proxy accept / token accept / false accept / false reject / agreement
  by proxy tau, token target tau, budget, task, and layer.

This is a standalone diagnostic and deliberately does not modify the main
evaluation pipeline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DenseAttentionBackend = None
PagedKVCache = None
SequenceKVManager = None
ModelForDecode = None
DATASET2MAXGEN = None
DATASET2PROMPT = None
load_dataset = None
triton_sparse_decode_unnorm = None
triton_score_pages = None


def ensure_runtime_imports() -> None:
    """Import GPU/runtime dependencies after argparse handles --help."""

    global DenseAttentionBackend
    global PagedKVCache
    global SequenceKVManager
    global ModelForDecode
    global DATASET2MAXGEN
    global DATASET2PROMPT
    global load_dataset
    global triton_sparse_decode_unnorm
    global triton_score_pages

    if ModelForDecode is not None:
        return
    from datasets import load_dataset as _load_dataset
    from cvsa.runtime.dense import DenseAttentionBackend as _DenseAttentionBackend
    from cvsa.runtime.kv_cache import PagedKVCache as _PagedKVCache
    from cvsa.runtime.kv_cache import SequenceKVManager as _SequenceKVManager
    from cvsa.runtime.model_loader import ModelForDecode as _ModelForDecode
    from eval.longbench_harness import DATASET2MAXGEN as _DATASET2MAXGEN
    from eval.longbench_harness import DATASET2PROMPT as _DATASET2PROMPT
    from cvsa.kernels.sparse_decode import (
        triton_sparse_decode_unnorm as _triton_sparse_decode_unnorm,
    )
    from cvsa.kernels.page_scores import triton_score_pages as _triton_score_pages

    DenseAttentionBackend = _DenseAttentionBackend
    PagedKVCache = _PagedKVCache
    SequenceKVManager = _SequenceKVManager
    ModelForDecode = _ModelForDecode
    DATASET2MAXGEN = _DATASET2MAXGEN
    DATASET2PROMPT = _DATASET2PROMPT
    load_dataset = _load_dataset
    triton_sparse_decode_unnorm = _triton_sparse_decode_unnorm
    triton_score_pages = _triton_score_pages


ALL_TASKS = [
    "hotpotqa", "triviaqa", "qasper", "multifieldqa_en",
    "2wikimqa", "narrativeqa", "musique",
    "gov_report", "multi_news", "samsum", "qmsum",
    "trec", "lcc", "repobench-p",
    "passage_count", "passage_retrieval_en",
]


PAPER_LAYER_GRID = {
    "llama": [0, 1, 4, 8, 12, 16, 20, 24, 28, 31],
    "mistral": [0, 1, 4, 8, 12, 16, 20, 24, 28, 31],
    "qwen": [0, 1, 3, 6, 9, 13, 17, 20, 23, 27],
}


@dataclass
class Counter:
    n: int = 0
    proxy_accept: int = 0
    token_accept: int = 0
    false_accept: int = 0
    false_reject: int = 0
    agree: int = 0
    proxy_cov_sum: float = 0.0
    token_cov_sum: float = 0.0
    proxy_cov_min: float = 1.0
    token_cov_min: float = 1.0

    def add(
        self,
        proxy_cov: torch.Tensor,
        token_cov: torch.Tensor,
        proxy_tau: float,
        token_tau: float,
    ) -> None:
        proxy_cov = proxy_cov.detach().float().flatten().cpu()
        token_cov = token_cov.detach().float().flatten().cpu()
        if proxy_cov.numel() != token_cov.numel():
            raise ValueError(
                f"proxy/token size mismatch: {proxy_cov.numel()} vs {token_cov.numel()}"
            )

        p = proxy_cov >= proxy_tau
        t = token_cov >= token_tau
        self.n += int(proxy_cov.numel())
        self.proxy_accept += int(p.sum().item())
        self.token_accept += int(t.sum().item())
        self.false_accept += int((p & ~t).sum().item())
        self.false_reject += int((~p & t).sum().item())
        self.agree += int((p == t).sum().item())
        self.proxy_cov_sum += float(proxy_cov.sum().item())
        self.token_cov_sum += float(token_cov.sum().item())
        self.proxy_cov_min = min(self.proxy_cov_min, float(proxy_cov.min().item()))
        self.token_cov_min = min(self.token_cov_min, float(token_cov.min().item()))

    def to_dict(self) -> dict:
        n = max(self.n, 1)
        return {
            "n_qheads": self.n,
            "proxy_accept": self.proxy_accept,
            "token_accept": self.token_accept,
            "false_accept": self.false_accept,
            "false_reject": self.false_reject,
            "proxy_accept_rate": self.proxy_accept / n,
            "token_accept_rate": self.token_accept / n,
            "false_accept_all_rate": self.false_accept / n,
            "false_accept_proxy_rate": (
                self.false_accept / self.proxy_accept if self.proxy_accept else None
            ),
            "false_reject_all_rate": self.false_reject / n,
            "false_reject_token_rate": (
                self.false_reject / self.token_accept if self.token_accept else None
            ),
            "agreement_rate": self.agree / n,
            "mean_proxy_coverage": self.proxy_cov_sum / n,
            "mean_token_coverage": self.token_cov_sum / n,
            "min_proxy_coverage": self.proxy_cov_min if self.n else None,
            "min_token_coverage": self.token_cov_min if self.n else None,
        }


@dataclass
class Accumulator:
    overall: dict = field(default_factory=lambda: defaultdict(Counter))
    by_task: dict = field(default_factory=lambda: defaultdict(Counter))
    by_layer: dict = field(default_factory=lambda: defaultdict(Counter))
    by_task_layer: dict = field(default_factory=lambda: defaultdict(Counter))

    @staticmethod
    def _key(proxy_tau: float, token_tau: float, budget: float) -> str:
        return f"proxy_tau={proxy_tau:.2f}|token_tau={token_tau:.2f}|budget={budget:.2f}"

    def add(
        self,
        *,
        task: str,
        layer: int,
        budget: float,
        proxy_tau: float,
        token_tau: float,
        proxy_cov_q: torch.Tensor,
        token_cov_q: torch.Tensor,
    ) -> None:
        key = self._key(proxy_tau, token_tau, budget)
        self.overall[key].add(proxy_cov_q, token_cov_q, proxy_tau, token_tau)
        self.by_task[(task, key)].add(proxy_cov_q, token_cov_q, proxy_tau, token_tau)
        self.by_layer[(layer, key)].add(proxy_cov_q, token_cov_q, proxy_tau, token_tau)
        self.by_task_layer[(task, layer, key)].add(
            proxy_cov_q, token_cov_q, proxy_tau, token_tau)

    def to_dict(self) -> dict:
        return {
            "overall": {k: v.to_dict() for k, v in sorted(self.overall.items())},
            "by_task": {
                task: {
                    key: self.by_task[(task, key)].to_dict()
                    for (_task, key) in sorted(self.by_task)
                    if _task == task
                }
                for task in sorted({k[0] for k in self.by_task})
            },
            "by_layer": {
                str(layer): {
                    key: self.by_layer[(layer, key)].to_dict()
                    for (_layer, key) in sorted(self.by_layer)
                    if _layer == layer
                }
                for layer in sorted({k[0] for k in self.by_layer})
            },
            "by_task_layer": {
                task: {
                    str(layer): {
                        key: self.by_task_layer[(task, layer, key)].to_dict()
                        for (_task, _layer, key) in sorted(self.by_task_layer)
                        if _task == task and _layer == layer
                    }
                    for layer in sorted(
                        {k[1] for k in self.by_task_layer if k[0] == task}
                    )
                }
                for task in sorted({k[0] for k in self.by_task_layer})
            },
        }


@dataclass
class CellRecorder:
    """Per-cell (q-head-step) raw dump for the FA->error analysis.

    Stores proxy coverage (ĉ), token coverage (c*), and sparse-vs-dense output
    error per cell, tau-INDEPENDENT (tau is applied offline). Offline: FA cells =
    (ĉ >= tau) & (c* < tau); compare their c*/error to true-accept cells.
    """
    proxy: list = field(default_factory=list)
    token: list = field(default_factory=list)
    err: list = field(default_factory=list)
    layer: list = field(default_factory=list)
    budget: list = field(default_factory=list)
    task: list = field(default_factory=list)

    def append(self, proxy_cov, token_cov, err, task, layer, budget):
        n = int(proxy_cov.numel())
        self.proxy.append(proxy_cov.flatten().cpu())
        self.token.append(token_cov.flatten().cpu())
        self.err.append(err.flatten().cpu())
        self.layer.extend([int(layer)] * n)
        self.budget.extend([float(budget)] * n)
        self.task.extend([task] * n)

    def save(self, path: Path) -> int:
        import numpy as np
        if not self.proxy:
            return 0
        proxy = torch.cat(self.proxy).numpy()
        np.savez_compressed(
            path,
            proxy=proxy,
            token=torch.cat(self.token).numpy(),
            err=torch.cat(self.err).numpy(),
            layer=np.array(self.layer, dtype=np.int16),
            budget=np.array(self.budget, dtype=np.float32),
            task=np.array(self.task),
        )
        return int(proxy.shape[0])


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.replace("+", ",").split(",") if x.strip()]


def parse_tasks(raw: str) -> list[str]:
    tasks = [x.strip() for x in raw.replace(",", "+").split("+") if x.strip()]
    if tasks == ["all"]:
        return ALL_TASKS
    unknown = sorted(set(tasks) - set(ALL_TASKS))
    if unknown:
        raise ValueError(f"Unknown LongBench tasks: {unknown}")
    return tasks


def model_key(model_name: str) -> str:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen"
    if "mistral" in lowered:
        return "mistral"
    if "llama" in lowered:
        return "llama"
    return "unknown"


def choose_layers(model_name: str, num_layers: int, raw: str) -> list[int]:
    if raw == "all":
        return list(range(num_layers))
    if raw == "paper":
        key = model_key(model_name)
        if key in PAPER_LAYER_GRID:
            return [li for li in PAPER_LAYER_GRID[key] if li < num_layers]
        return sorted(set([0, 1, num_layers // 4, num_layers // 2, num_layers - 1]))
    return sorted({int(x) for x in raw.replace("+", ",").split(",") if x.strip()})


def prepare_prompts(model: ModelForDecode, task: str, max_samples: int, max_length: int):
    data = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    if max_samples and len(data) > max_samples:
        data = data.select(range(max_samples))

    prompt_format = DATASET2PROMPT[task]
    prompts = []
    model_name_l = model.config.model_name.lower()
    use_chat = ("instruct" in model_name_l or "chat" in model_name_l)
    no_chat_tasks = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

    for item in data:
        prompt = prompt_format.format(**item)
        tokens = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokens) > max_length:
            half = max_length // 2
            prompt = (
                model.tokenizer.decode(tokens[:half], skip_special_tokens=True)
                + model.tokenizer.decode(tokens[-half:], skip_special_tokens=True)
            )
        if use_chat and task not in no_chat_tasks and hasattr(model.tokenizer, "apply_chat_template"):
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompts.append(prompt)
    return prompts


def compute_proxy_and_token_coverage(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_kv_heads: int,
    num_q_per_kv: int,
    block_size: int,
    budget_pct: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return page-proxy and token-level coverage at Q-head granularity."""
    batch, num_q_heads, _ = q.shape
    n_pages = block_tables.shape[1]
    n_valid = ((context_lens.float() - 1) / block_size).long() + 1
    budget = max(3, int(n_pages * budget_pct))
    budget = min(budget, int(n_valid.max().item()))

    page_scores = triton_score_pages(
        q, k_cache, block_tables, context_lens,
        num_kv_heads, num_q_per_kv, block_size,
    )
    topk_vals, topk_indices = torch.topk(
        page_scores, k=budget, dim=-1, largest=True, sorted=False)
    page_indices = topk_indices.to(torch.int32)
    n_per_head = torch.full(
        (batch, num_kv_heads), budget, dtype=torch.int32, device=q.device)
    n_per_head = torch.min(
        n_per_head,
        n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32),
    )

    # Page-level proxy coverage, matching the runtime verifier.
    lse_all_proxy = torch.logsumexp(page_scores, dim=-1)
    idx = torch.arange(budget, device=q.device)
    valid_topk = idx.view(1, 1, -1) < n_per_head.unsqueeze(-1)
    lse_sel_proxy = torch.logsumexp(
        topk_vals.masked_fill(~valid_topk, float("-inf")), dim=-1)
    proxy_cov_kv = torch.exp(lse_sel_proxy - lse_all_proxy).clamp(0, 1)
    proxy_cov_q = proxy_cov_kv.repeat_interleave(num_q_per_kv, dim=1)

    # Token-level c*: selected-set LSE divided by full-context LSE.
    _o_sel, m_sel, l_sel = triton_sparse_decode_unnorm(
        q, k_cache, v_cache, block_tables, page_indices, n_per_head,
        context_lens, num_kv_heads, num_q_per_kv, block_size)
    lse_sel_q = (m_sel + torch.log(l_sel.clamp(min=1e-30))).float()

    full_pages = torch.arange(
        n_pages, device=q.device, dtype=torch.int32
    ).view(1, 1, -1).expand(batch, num_kv_heads, -1).contiguous()
    full_n = n_valid.unsqueeze(1).expand(batch, num_kv_heads).to(torch.int32).contiguous()
    _o_full, m_full, l_full = triton_sparse_decode_unnorm(
        q, k_cache, v_cache, block_tables, full_pages, full_n,
        context_lens, num_kv_heads, num_q_per_kv, block_size)
    lse_all_q = (m_full + torch.log(l_full.clamp(min=1e-30))).float()
    token_cov_q = torch.exp(lse_sel_q - lse_all_q).clamp(0, 1)

    # Per-q-head sparse-vs-dense output error = ||O_sel - O_full||_2 over head_dim
    # (FA->error analysis). O = o/l is the normalized attention output; the per-row
    # max offset cancels in o/l, so the two normalized outputs are directly comparable.
    o_sel_norm = _o_sel.float() / l_sel.clamp(min=1e-30).unsqueeze(-1).float()
    o_full_norm = _o_full.float() / l_full.clamp(min=1e-30).unsqueeze(-1).float()
    err_q = (o_sel_norm - o_full_norm).norm(dim=-1)  # [B, H_q]
    return proxy_cov_q, token_cov_q, err_q


def run_prompt(
    *,
    model: ModelForDecode,
    prompt: str,
    task: str,
    sample_idx: int,
    layers_to_probe: set[int],
    budgets: Iterable[float],
    proxy_taus: Iterable[float],
    token_taus: Iterable[float],
    max_steps: int,
    block_size: int,
    acc: Accumulator,
    sample_stride: int,
    cell_recorder: "CellRecorder | None" = None,
) -> dict:
    config = model.config
    tokenizer = model.tokenizer
    device = model.device
    dtype = model.dtype

    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=False).to(device)
    seq_len = input_ids.shape[1]
    max_new = min(max_steps, DATASET2MAXGEN.get(task, max_steps))

    inner_model = model.model.model
    with torch.no_grad():
        outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values
    logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    cur_token = logits.argmax(dim=-1)

    n_pages = (seq_len + block_size - 1) // block_size
    extra_pages = max(16, (max_new + block_size - 1) // block_size + 4)
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
        manager = SequenceKVManager(cache, 1, max_context_len=seq_len + max_new + 16)
        k = past_kv[layer_idx][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        v = past_kv[layer_idx][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(dtype)
        manager.prefill_kv(0, k, v)
        layer_caches.append(cache)
        layer_managers.append(manager)

    del outputs, past_kv
    torch.cuda.empty_cache()

    dense = DenseAttentionBackend(
        config.num_kv_heads, config.num_q_heads, config.head_dim, block_size)
    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb
    eos_id = tokenizer.eos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    probed_steps = 0
    generated = []
    with torch.no_grad():
        for step in range(max_new):
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
                q_flat = q.squeeze(2)

                should_probe = layer_idx in layers_to_probe and (step % sample_stride == 0)
                if should_probe:
                    for budget in budgets:
                        proxy_cov_q, token_cov_q, err_q = compute_proxy_and_token_coverage(
                            q=q_flat,
                            k_cache=cache.k_cache,
                            v_cache=cache.v_cache,
                            block_tables=manager.block_tables,
                            context_lens=manager.context_lens,
                            num_kv_heads=config.num_kv_heads,
                            num_q_per_kv=config.num_q_per_kv,
                            block_size=block_size,
                            budget_pct=budget,
                        )
                        if cell_recorder is not None:
                            cell_recorder.append(
                                proxy_cov_q, token_cov_q, err_q,
                                task, layer_idx, budget)
                        for proxy_tau in proxy_taus:
                            for token_tau in token_taus:
                                acc.add(
                                    task=task,
                                    layer=layer_idx,
                                    budget=budget,
                                    proxy_tau=proxy_tau,
                                    token_tau=token_tau,
                                    proxy_cov_q=proxy_cov_q,
                                    token_cov_q=token_cov_q,
                                )
                    probed_steps += 1

                attn_output = dense(
                    q_flat,
                    cache.k_cache,
                    cache.v_cache,
                    manager.block_tables,
                    manager.context_lens,
                    layer_idx,
                )
                attn_output = attn_output.view(1, 1, config.num_q_heads * config.head_dim)
                hidden = residual + layer.self_attn.o_proj(attn_output)
                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = residual + layer.mlp(hidden)
                dense.step_done(layer_idx, config.num_layers)

            hidden = model.final_norm(hidden)
            lm_dtype = model.model.lm_head.weight.dtype
            logits = model.lm_head(hidden[:, 0, :].to(lm_dtype))
            cur_token = logits.argmax(dim=-1)
            token_id = int(cur_token.item())
            generated.append(token_id)
            if token_id == eos_id or token_id == eot_id:
                break

    del layer_caches, layer_managers
    torch.cuda.empty_cache()

    return {
        "task": task,
        "sample_idx": sample_idx,
        "prompt_tokens": int(seq_len),
        "decode_steps": len(generated),
        "probed_layer_steps": probed_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", default="hotpotqa+qasper+lcc+passage_retrieval_en")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=31500)
    parser.add_argument("--layers", default="paper",
                        help="'paper', 'all', or comma-separated layer ids")
    parser.add_argument("--budgets", default="0.05")
    parser.add_argument("--taus", default="0.90,0.92,0.95,0.97",
                        help="Proxy acceptance thresholds.")
    parser.add_argument("--token-taus", default=None,
                        help="Token-level c* targets. Defaults to --taus.")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="Probe every N decode steps.")
    parser.add_argument("--output", default="results/camera_ready/production_proxy_calibration.json")
    args = parser.parse_args()

    ensure_runtime_imports()

    tasks = parse_tasks(args.tasks)
    budgets = parse_float_list(args.budgets)
    proxy_taus = parse_float_list(args.taus)
    token_taus = parse_float_list(args.token_taus) if args.token_taus else proxy_taus

    print("Production proxy calibration")
    print("=" * 80)
    print(f"model: {args.model}")
    print(f"tasks: {tasks}")
    print(f"max_samples/task: {args.max_samples}")
    print(f"max_steps: {args.max_steps}")
    print(f"budgets: {budgets}")
    print(f"proxy_taus: {proxy_taus}")
    print(f"token_taus: {token_taus}")

    t0 = time.time()
    model = ModelForDecode(args.model, dtype=torch.bfloat16, device="cuda")
    config = model.config
    layers = choose_layers(args.model, config.num_layers, args.layers)
    layers_to_probe = set(layers)
    print(f"loaded in {time.time() - t0:.1f}s")
    print(f"config: L={config.num_layers}, Q={config.num_q_heads}, KV={config.num_kv_heads}, D={config.head_dim}")
    print(f"layers: {layers}")

    acc = Accumulator()
    cells = CellRecorder()  # per-cell (ĉ, c*, error) dump for the FA->error analysis
    samples = []
    for task in tasks:
        prompts = prepare_prompts(model, task, args.max_samples, args.max_length)
        for i, prompt in enumerate(prompts):
            print(f"\n[{task} {i + 1}/{len(prompts)}] running...", flush=True)
            meta = run_prompt(
                model=model,
                prompt=prompt,
                task=task,
                sample_idx=i,
                layers_to_probe=layers_to_probe,
                budgets=budgets,
                proxy_taus=proxy_taus,
                token_taus=token_taus,
                max_steps=args.max_steps,
                block_size=args.block_size,
                acc=acc,
                sample_stride=args.sample_stride,
                cell_recorder=cells,
            )
            samples.append(meta)
            print(
                f"  prompt_tokens={meta['prompt_tokens']} "
                f"decode_steps={meta['decode_steps']} "
                f"probed_layer_steps={meta['probed_layer_steps']}",
                flush=True,
            )

    out = {
        "config": vars(args),
        "model_config": {
            "num_layers": config.num_layers,
            "num_q_heads": config.num_q_heads,
            "num_kv_heads": config.num_kv_heads,
            "num_q_per_kv": config.num_q_per_kv,
            "head_dim": config.head_dim,
        },
        "layers": layers,
        "samples": samples,
        "summary": acc.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved to {output}")

    # Per-cell (ĉ, c*, error) dump for the FA->error analysis.
    cells_path = output.with_suffix(".cells.npz")
    n_cells = cells.save(cells_path)
    print(f"Saved {n_cells} per-cell (proxy, token, err) records to {cells_path}")

    focus = f"proxy_tau={0.90:.2f}|token_tau={0.90:.2f}|budget={0.05:.2f}"
    if focus in out["summary"]["overall"]:
        print("\nFocus summary proxy_tau=0.90, token_tau=0.90, budget=0.05:")
        print(json.dumps(out["summary"]["overall"][focus], indent=2))


if __name__ == "__main__":
    main()
