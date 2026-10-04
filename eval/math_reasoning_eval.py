#!/usr/bin/env python3
"""Reasoning-decode quality pilot: dense vs CVSA on MATH-500.

Design:
  - Model: DeepSeek-R1-Distill-Llama-8B (Llama backbone -> backend-compatible).
  - Data: HuggingFaceH4/MATH-500 test split (fields: problem, answer).
  - Decoding: sampling (temp 0.6, top-p 0.95, per DeepSeek's model card; greedy
    is KNOWN to degenerate on R1-distill), seed-paired per problem across
    methods, EOS-stopped, capped at --max-new tokens.
  - Grading: last \\boxed{...} (brace-matched), light normalization, exact
    match; same grader both arms so the PAIRED comparison is fair even if the
    grader is imperfect.

⚠️ Budget semantics disclosure: prompts are short (~100-300 tokens) while the
KV allocation covers prompt + max-new. The deployed backend computes its page
budget as budget_pct × ALLOCATED pages, so "5%" here = 5% of the maximum
sequence length (a fixed page budget provisioned for the generation), NOT 5%
of the current cache. Early decode is effectively unconstrained until the
cache outgrows the budget. Report this alongside any number.

Usage (GPU node):
  python eval/math_reasoning_eval.py \
      --methods dense,cvsa_5_90_c4_ar20_dL0 --n 50 --max-new 16384 \
      --output results/camera_ready/math_reasoning_r1distill8b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager  # noqa: E402
from cvsa.runtime.model_loader import ModelForDecode  # noqa: E402
from datasets import load_dataset  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_make_backend():
    """Load make_backend by file path (avoids eval/ package collision)."""
    spec = importlib.util.spec_from_file_location(
        "longbench_cvsa_for_math", os.path.join(_ROOT, "eval", "longbench_cvsa.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_backend


# ─── boxed-answer extraction + normalization ─────────────────────────────
def extract_boxed(text: str):
    """Return the content of the LAST \\boxed{...} with brace matching."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out) if depth == 0 else None


def normalize_answer(ans):
    if ans is None:
        return None
    s = ans.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("~", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    return s


def is_correct(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p is None:
        return False
    if p == g:
        return True
    try:  # numeric fallback (plain floats; 1/2 vs 0.5 NOT handled)
        return abs(float(p) - float(g)) < 1e-6
    except (TypeError, ValueError):
        return False


# ─── sampling ────────────────────────────────────────────────────────────
def sample_token(logits, temperature, top_p, generator):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum - sorted_probs > top_p          # tokens fully past the nucleus
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(sorted_probs, 1, generator=generator)
    return sorted_idx.gather(-1, pick).squeeze(-1)


# ─── generation with a paged backend (mirrors diagnostics_hybrid_trace) ──
@torch.no_grad()
def generate(model, prompt_ids, max_new, backend, block_size, eos_ids,
             temperature, top_p, seed):
    config = model.config
    device = model.device
    dtype = model.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    input_ids = prompt_ids.to(device)
    seq_len = input_ids.shape[1]

    inner_model = model.model.model
    outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values
    logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    cur_token = sample_token(logits, temperature, top_p, gen)

    n_pages = (seq_len + block_size - 1) // block_size
    extra_pages = max(16, (max_new + block_size - 1) // block_size + 4)
    num_blocks = n_pages + extra_pages + 64

    layer_caches, layer_managers = [], []
    for layer_idx in range(config.num_layers):
        cache = PagedKVCache(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
            dtype=dtype, device=str(device))
        manager = SequenceKVManager(cache, 1,
                                    max_context_len=seq_len + max_new + 16)
        k = past_kv[layer_idx][0][0, :, :seq_len, :]
        v = past_kv[layer_idx][1][0, :, :seq_len, :]
        manager.prefill_kv(0, k.transpose(0, 1).contiguous().to(dtype),
                           v.transpose(0, 1).contiguous().to(dtype))
        layer_caches.append(cache)
        layer_managers.append(manager)
    del outputs, past_kv
    torch.cuda.empty_cache()

    layers = model.get_layers()
    attn_module = type(layers[0].self_attn).__module__
    apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
    rotary_emb = model.model.model.rotary_emb

    backend.reset()
    generated = [int(cur_token.item())]
    for step in range(max_new - 1):
        if int(cur_token.item()) in eos_ids:
            break
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
            attn_output = backend(q.squeeze(2), cache.k_cache, cache.v_cache,
                                  manager.block_tables, manager.context_lens,
                                  layer_idx)
            attn_output = attn_output.view(1, 1, config.num_q_heads * config.head_dim)
            hidden = residual + layer.self_attn.o_proj(attn_output)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)
            backend.step_done(layer_idx, config.num_layers)

        hidden = model.final_norm(hidden)
        logits = model.lm_head(hidden[:, 0, :].to(model.model.lm_head.weight.dtype))
        cur_token = sample_token(logits, temperature, top_p, gen)
        generated.append(int(cur_token.item()))

    del layer_caches, layer_managers
    torch.cuda.empty_cache()
    return generated


def main():
    ap = argparse.ArgumentParser(description="MATH-500 reasoning pilot: dense vs CVSA.")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--methods", default="dense,cvsa_5_90_c4_ar20_dL0")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="dataset offset")
    ap.add_argument("--max-new", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    make_backend = _load_make_backend()

    print(f"Loading {args.model}")
    model = ModelForDecode(args.model, dtype=torch.bfloat16, device="cuda")
    config = model.config
    tok = model.tokenizer
    eos_ids = {tok.eos_token_id}
    print(f"  {config.num_layers}L {config.num_q_heads}Q/{config.num_kv_heads}KV; eos={eos_ids}")

    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idxs = list(range(args.start, min(args.start + args.n, len(data))))
    print(f"MATH-500: problems {idxs[0]}..{idxs[-1]} (n={len(idxs)})")
    print(f"decoding: temp={args.temperature} top_p={args.top_p} max_new={args.max_new} (seed-paired)")

    results = {"config": vars(args), "problems": [], "accuracy": {}, "gen_tokens": {}}
    for meth in methods:
        results["accuracy"][meth] = 0
        results["gen_tokens"][meth] = 0

    for j, di in enumerate(idxs):
        row = data[di]
        prompt = tok.apply_chat_template(
            [{"role": "user",
              "content": row["problem"] +
              "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt, return_tensors="pt", truncation=False)
        rec = {"idx": di, "gold": row["answer"], "prompt_tokens": int(prompt_ids.shape[1]),
               "methods": {}}
        for meth in methods:
            backend = make_backend(meth, config, args.block_size)
            t0 = time.time()
            toks = generate(model, prompt_ids, args.max_new, backend,
                            args.block_size, eos_ids,
                            args.temperature, args.top_p, seed=args.seed + di)
            text = tok.decode(toks, skip_special_tokens=True)
            pred = extract_boxed(text)
            ok = is_correct(pred, row["answer"])
            rec["methods"][meth] = {
                "gen_tokens": len(toks), "extracted": pred, "correct": bool(ok),
                "seconds": round(time.time() - t0, 1),
            }
            results["accuracy"][meth] += int(ok)
            results["gen_tokens"][meth] += len(toks)
            print(f"[{j+1}/{len(idxs)}] #{di} {meth:26s} tokens={len(toks):6d} "
                  f"correct={ok}  pred={str(pred)[:40]!r} gold={row['answer'][:40]!r}",
                  flush=True)
        results["problems"].append(rec)
        with open(args.output, "w") as f:   # checkpoint after every problem
            json.dump(results, f, indent=1)

    n = len(idxs)
    print("\n===== PILOT SUMMARY =====")
    for meth in methods:
        acc = results["accuracy"][meth] / n * 100
        avg = results["gen_tokens"][meth] / n
        print(f"  {meth:28s} accuracy={acc:.1f}%  avg_gen_tokens={avg:.0f}")
    paired = sum(1 for r in results["problems"]
                 if len(methods) == 2 and
                 r["methods"][methods[0]]["correct"] != r["methods"][methods[1]]["correct"])
    print(f"  disagreements: {paired}/{n}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
