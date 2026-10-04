#!/usr/bin/env python3
"""Empirical validation of the coverage → error bound theorem.

Theorem: ||O_draft - O_dense||₂ = (1-c) · ||O_draft - O_comp||₂
Bound:   ||O_draft - O_dense||₂ ≤ (1-c) · D_V

This script measures actual errors vs the theoretical bound across
layers, heads, and decode steps using real model KV cache.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvsa.kernels.sparse_decode import triton_sparse_decode_unnorm
from cvsa.kernels.page_scores import triton_score_pages
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager
from eval.longbench_harness import E2EGenerator, DATASET2PROMPT
from cvsa.complement import compute_complement_pages
from cvsa.kernels.coverage import triton_check_coverage


def compute_dense_output(q, k_cache, v_cache, block_tables, context_lens,
                         num_kv_heads, num_q_per_kv, block_size):
    """Compute exact dense attention output using FlashInfer."""
    backend = DenseAttentionBackend(
        num_kv_heads, num_kv_heads * num_q_per_kv,
        q.shape[-1], block_size)
    return backend(q, k_cache, v_cache, block_tables, context_lens, 0)


def measure_errors(q, k_cache, v_cache, block_tables, context_lens,
                   num_kv_heads, num_q_per_kv, block_size, budget_pct, tau):
    """Measure actual error, coverage, and theoretical bound for one step."""
    B, H_q, D = q.shape
    n_pages = block_tables.shape[1]
    n_valid = ((context_lens.float() - 1) / block_size).long() + 1
    budget = max(3, int(n_pages * budget_pct))

    # 1. Score pages and select top-k
    page_scores = triton_score_pages(
        q, k_cache, block_tables, context_lens,
        num_kv_heads, num_q_per_kv, block_size)
    topk_vals, topk_indices = torch.topk(
        page_scores, k=budget, dim=-1, largest=True, sorted=False)
    page_indices = topk_indices.to(torch.int32)
    n_per_head = torch.full((B, num_kv_heads), budget,
                            dtype=torch.int32, device=q.device)
    n_per_head = torch.min(n_per_head,
                           n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32))

    # 2. Draft attention (unnormalized)
    o_d, m_d, l_d = triton_sparse_decode_unnorm(
        q, k_cache, v_cache, block_tables, page_indices, n_per_head,
        context_lens, num_kv_heads, num_q_per_kv, block_size)
    # Normalized draft output
    o_draft = o_d / l_d.unsqueeze(-1).clamp(min=1e-6)  # [B, H_q, D]

    # 2b. Full-attention LSE for token-level coverage c*
    # Pass all valid pages as "selected" -> kernel returns true LSE_all per Q-head.
    # Used to compute c* = exp(LSE_sel - LSE_all), independent of page-max proxy.
    # Kernel expects page_indices as 3D [B, H_kv, max_pages]; block_tables is 2D
    # [B, n_pages], so broadcast across the KV-head axis before calling.
    block_tables_3d = (block_tables.unsqueeze(1)
                       .expand(B, num_kv_heads, -1)
                       .contiguous()
                       .to(torch.int32))
    n_full = n_valid.unsqueeze(1).expand(B, num_kv_heads).contiguous().to(torch.int32)
    _o_full, m_full, l_full = triton_sparse_decode_unnorm(
        q, k_cache, v_cache, block_tables, block_tables_3d, n_full,
        context_lens, num_kv_heads, num_q_per_kv, block_size)
    # LSE_sel per Q-head from the draft kernel (m_d, l_d already computed above);
    # LSE_all per Q-head from full-attention kernel (m_full, l_full).
    LSE_sel_q = (m_d + torch.log(l_d.clamp(min=1e-30))).float()        # [B, H_q]
    LSE_all_q = (m_full + torch.log(l_full.clamp(min=1e-30))).float()  # [B, H_q]
    c_star_q = torch.exp(LSE_sel_q - LSE_all_q).clamp(0, 1)            # [B, H_q]

    # 3. Coverage per KV head (page-max proxy, used in code/kernels)
    accept_mask, coverage = triton_check_coverage(
        page_scores, page_indices, n_per_head, tau)
    # coverage: [B, H_kv]

    # 4. Complement attention
    # Force all heads to compute complement (ignore accept_mask)
    all_reject = torch.zeros_like(accept_mask)
    comp_indices, n_comp = compute_complement_pages(
        page_indices, n_per_head, context_lens, block_size, all_reject)
    o_c, m_c, l_c = triton_sparse_decode_unnorm(
        q, k_cache, v_cache, block_tables, comp_indices, n_comp,
        context_lens, num_kv_heads, num_q_per_kv, block_size)
    o_comp = o_c / l_c.unsqueeze(-1).clamp(min=1e-6)  # [B, H_q, D]

    # 5. Dense attention (ground truth)
    o_dense = compute_dense_output(
        q, k_cache, v_cache, block_tables, context_lens,
        num_kv_heads, num_q_per_kv, block_size).float()  # [B, H_q, D]
    o_draft = o_draft.float()
    o_comp = o_comp.float()

    # 6. Compute D_V (value diameter) per KV head
    # Sample V vectors from selected + complement pages to estimate D_V
    # For efficiency, compute per-head max ||v_i - v_j|| from a sample
    D_V_per_head = torch.zeros(B, num_kv_heads, device=q.device)
    for b in range(B):
        for h in range(num_kv_heads):
            # Gather V vectors from all valid pages for this head
            n_v = n_valid[b].item()
            v_vectors = []
            for p in range(min(n_v, 64)):  # sample up to 64 pages
                phys = block_tables[b, p].item()
                n_tokens = min(block_size, context_lens[b].item() - p * block_size)
                # v_cache: [num_blocks, block_size, H_kv, D]
                v_page = v_cache[phys, :n_tokens, h, :].float()  # [n_tokens, D]
                v_vectors.append(v_page)
            if v_vectors:
                all_v = torch.cat(v_vectors, dim=0)  # [total_tokens, D]
                # Efficient diameter estimate: max pairwise distance
                # Use random sampling for efficiency
                if all_v.shape[0] > 200:
                    idx = torch.randperm(all_v.shape[0])[:200]
                    all_v = all_v[idx]
                dists = torch.cdist(all_v.unsqueeze(0), all_v.unsqueeze(0)).squeeze(0)
                D_V_per_head[b, h] = dists.max().item()

    # 7. Compute per-head errors
    # Expand coverage from KV-head to Q-head: [B, H_kv] → [B, H_q]
    cov_q = coverage.repeat_interleave(num_q_per_kv, dim=1)  # [B, H_q] (page-max proxy)
    D_V_q = D_V_per_head.repeat_interleave(num_q_per_kv, dim=1)  # [B, H_q]

    # Actual error per Q-head: ||O_draft - O_dense||₂
    actual_error = (o_draft - o_dense).norm(dim=-1)  # [B, H_q]
    draft_comp_norm = (o_draft - o_comp).norm(dim=-1)  # [B, H_q]

    # Theorem prediction (page-max proxy): (1 - hat_c) * ||O_draft - O_comp||₂
    theorem_exact = (1 - cov_q) * draft_comp_norm  # [B, H_q]
    # Upper bound (page-max proxy): (1 - hat_c) * D_V
    theorem_bound = (1 - cov_q) * D_V_q  # [B, H_q]

    # Theorem prediction (token-level c*): (1 - c*) * ||O_draft - O_comp||₂
    theorem_exact_token = (1 - c_star_q) * draft_comp_norm  # [B, H_q]
    # Upper bound (token-level c*): (1 - c*) * D_V
    theorem_bound_token = (1 - c_star_q) * D_V_q  # [B, H_q]

    # O_dense norm for relative error
    dense_norm = o_dense.norm(dim=-1).clamp(min=1e-6)  # [B, H_q]

    return {
        "coverage": coverage.cpu(),                          # [B, H_kv] (page-max proxy)
        "coverage_token": c_star_q.cpu(),                    # [B, H_q] (token-level c*)
        "actual_error": actual_error.cpu(),                  # [B, H_q]
        "theorem_exact": theorem_exact.cpu(),                # [B, H_q] (proxy)
        "theorem_bound": theorem_bound.cpu(),                # [B, H_q] (proxy)
        "theorem_exact_token": theorem_exact_token.cpu(),    # [B, H_q] (true c*)
        "theorem_bound_token": theorem_bound_token.cpu(),    # [B, H_q] (true c*)
        "relative_error": (actual_error / dense_norm).cpu(),  # [B, H_q]
        "D_V": D_V_per_head.cpu(),                           # [B, H_kv]
        "accept_mask": accept_mask.cpu(),                     # [B, H_kv]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model id")
    parser.add_argument("--output", default="results/error_bound_validation.json",
                        help="Output JSON path")
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices; default picks evenly-spaced layers")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = ModelForDecode(args.model, dtype=torch.bfloat16, device="cuda")
    config = model.config
    num_q_per_kv = config.num_q_heads // config.num_kv_heads
    print(f"  layers={config.num_layers}, q_heads={config.num_q_heads}, kv_heads={config.num_kv_heads}")

    # Load real prompts
    from datasets import load_dataset
    data = load_dataset("THUDM/LongBench", "hotpotqa", split="test",
                        trust_remote_code=True)
    prompt_format = DATASET2PROMPT["hotpotqa"]

    block_size = 16
    n_prompts = args.n_prompts
    if args.layers:
        n_layers_to_test = [int(x) for x in args.layers.split(",")]
    else:
        # Pick 6 evenly-spaced layers including L0 and last
        L = config.num_layers
        n_layers_to_test = sorted(set([0, 1, L // 8, L // 4, L // 2, L - 1]))[:6]
    print(f"  testing layers: {n_layers_to_test}")

    all_results = {
        "coverage": [], "coverage_token": [],
        "actual_error": [], "theorem_exact": [], "theorem_bound": [],
        "theorem_exact_token": [], "theorem_bound_token": [],
        "relative_error": [], "D_V": [], "layer": [], "accept": [],
    }

    for prompt_idx in range(n_prompts):
        prompt = prompt_format.format(**data[prompt_idx])
        tokens = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # Truncate to ~4K for faster iteration
        if len(tokens) > 4096:
            half = 2048
            prompt = (model.tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                      model.tokenizer.decode(tokens[-half:], skip_special_tokens=True))

        input_ids = model.tokenizer.encode(
            prompt, return_tensors="pt", truncation=False).to("cuda")
        seq_len = input_ids.shape[1]
        print(f"\nPrompt {prompt_idx+1}/{n_prompts}: {seq_len} tokens")

        inner_model = model.model.model
        with torch.no_grad():
            outputs = inner_model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values

        # Test each layer
        for li in n_layers_to_test:
            n_pages = (seq_len + block_size - 1) // block_size
            num_blocks = n_pages + 32
            cache = PagedKVCache(
                num_blocks=num_blocks, block_size=block_size,
                num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
                dtype=torch.bfloat16, device="cuda")
            manager = SequenceKVManager(cache, 1, max_context_len=seq_len + 64)

            k = past_kv[li][0][0, :, :seq_len, :].transpose(0, 1).contiguous().to(torch.bfloat16)
            v = past_kv[li][1][0, :, :seq_len, :].transpose(0, 1).contiguous().to(torch.bfloat16)
            manager.prefill_kv(0, k, v)

            # Get query from next-token prediction
            q_proj = model.get_layers()[li].self_attn.q_proj
            next_hidden = outputs.last_hidden_state[:, -1:, :]
            next_hidden = model.get_layers()[li].input_layernorm(next_hidden)
            q = q_proj(next_hidden).view(1, config.num_q_heads, config.head_dim).to(torch.bfloat16)

            for budget_pct in [0.03, 0.05, 0.10, 0.15]:
                for tau in [0.80, 0.90, 0.95]:
                    result = measure_errors(
                        q, cache.k_cache, cache.v_cache,
                        manager.block_tables, manager.context_lens,
                        config.num_kv_heads, num_q_per_kv,
                        block_size, budget_pct, tau)

                    # Store per-head results
                    for h in range(config.num_q_heads):
                        all_results["coverage"].append(
                            result["coverage"][0, h // num_q_per_kv].item())
                        all_results["coverage_token"].append(
                            result["coverage_token"][0, h].item())
                        all_results["actual_error"].append(
                            result["actual_error"][0, h].item())
                        all_results["theorem_exact"].append(
                            result["theorem_exact"][0, h].item())
                        all_results["theorem_bound"].append(
                            result["theorem_bound"][0, h].item())
                        all_results["theorem_exact_token"].append(
                            result["theorem_exact_token"][0, h].item())
                        all_results["theorem_bound_token"].append(
                            result["theorem_bound_token"][0, h].item())
                        all_results["relative_error"].append(
                            result["relative_error"][0, h].item())
                        all_results["D_V"].append(
                            result["D_V"][0, h // num_q_per_kv].item())
                        all_results["layer"].append(li)
                        all_results["accept"].append(
                            result["accept_mask"][0, h // num_q_per_kv].item())

            del cache, manager
            torch.cuda.empty_cache()

        del outputs, past_kv
        torch.cuda.empty_cache()

    # === Analysis ===
    print("\n" + "=" * 80)
    print("ERROR BOUND VALIDATION")
    print("=" * 80)

    cov = np.array(all_results["coverage"])
    cov_tok = np.array(all_results["coverage_token"])
    actual = np.array(all_results["actual_error"])
    exact = np.array(all_results["theorem_exact"])
    bound = np.array(all_results["theorem_bound"])
    exact_tok = np.array(all_results["theorem_exact_token"])
    bound_tok = np.array(all_results["theorem_bound_token"])
    rel_err = np.array(all_results["relative_error"])
    layers = np.array(all_results["layer"])
    D_V = np.array(all_results["D_V"])

    # Compare proxy vs token-level coverage and bounds
    print(f"\n[Coverage] page-max proxy median = {np.median(cov):.4f}, "
          f"token-level c* median = {np.median(cov_tok):.4f}")
    valid_b_tok = bound_tok > 1e-8
    if valid_b_tok.any():
        viol_proxy = int((actual > bound * 1.01).sum())
        viol_token = int((actual > bound_tok * 1.01).sum())
        print(f"[Bound]    proxy violations = {viol_proxy}/{len(actual)}, "
              f"c* violations = {viol_token}/{len(actual)}")

    # 1. Theorem verification: actual ≈ (1-c)||O_draft - O_comp||
    print("\n1. Theorem Exactness: actual_error vs (1-c)||O_draft - O_comp||")
    print("-" * 60)
    valid = exact > 1e-8
    if valid.any():
        ratio = actual[valid] / exact[valid]
        print(f"   Ratio actual/theorem_exact: mean={ratio.mean():.4f}, "
              f"std={ratio.std():.4f}, max={ratio.max():.4f}")
        print(f"   (Should be ≈1.0 if theorem is correct)")

    # 2. Bound tightness: actual vs (1-c)*D_V
    print("\n2. Bound Tightness: actual_error vs (1-c)·D_V")
    print("-" * 60)
    valid_bound = bound > 1e-8
    if valid_bound.any():
        tightness = actual[valid_bound] / bound[valid_bound]
        print(f"   Ratio actual/(1-c)·D_V: mean={tightness.mean():.4f}, "
              f"std={tightness.std():.4f}, max={tightness.max():.4f}")
        violations = (actual > bound * 1.01).sum()  # allow 1% numerical tolerance
        print(f"   Bound violations: {violations}/{len(actual)} "
              f"({violations/len(actual)*100:.1f}%)")
        print(f"   (Ratio < 1 means bound holds; closer to 1 = tighter)")

    # 3. Coverage vs actual error
    print("\n3. Coverage vs Relative Error")
    print("-" * 60)
    print(f"   {'Coverage Range':<20} {'Mean Rel Error':>15} {'Max Rel Error':>15} {'Count':>8}")
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95),
                   (0.95, 0.99), (0.99, 1.0)]:
        mask = (cov >= lo) & (cov < hi)
        if mask.any():
            print(f"   [{lo:.2f}, {hi:.2f})       "
                  f"{rel_err[mask].mean():>15.6f} {rel_err[mask].max():>15.6f} "
                  f"{mask.sum():>8d}")

    # 4. Per-layer analysis
    print("\n4. Per-Layer Error (budget=10%, τ=0.90)")
    print("-" * 60)
    print(f"   {'Layer':>5} {'Mean Cov':>10} {'Mean Error':>12} {'Mean Rel Err':>12} {'Mean D_V':>10}")
    for li in n_layers_to_test:
        mask = layers == li
        if mask.any():
            print(f"   L{li:>3d}  {cov[mask].mean():>10.4f} "
                  f"{actual[mask].mean():>12.6f} {rel_err[mask].mean():>12.6f} "
                  f"{D_V[mask].mean():>10.2f}")

    # 5. D_V statistics
    print("\n5. Value Diameter D_V Statistics")
    print("-" * 60)
    print(f"   Mean D_V: {D_V.mean():.2f}")
    print(f"   Std  D_V: {D_V.std():.2f}")
    print(f"   Min  D_V: {D_V.min():.2f}")
    print(f"   Max  D_V: {D_V.max():.2f}")

    # 6. Key finding for paper
    print("\n" + "=" * 80)
    print("KEY FINDING FOR PAPER")
    print("=" * 80)
    high_cov = cov >= 0.90
    if high_cov.any():
        print(f"   For coverage ≥ 0.90 ({high_cov.sum()} heads):")
        print(f"     Mean relative error: {rel_err[high_cov].mean():.6f}")
        print(f"     Max  relative error: {rel_err[high_cov].max():.6f}")
        print(f"     Mean bound tightness (actual/bound): "
              f"{(actual[high_cov] / bound[high_cov].clip(1e-8)).mean():.4f}")
    low_cov = cov < 0.90
    if low_cov.any():
        print(f"   For coverage < 0.90 ({low_cov.sum()} heads):")
        print(f"     These heads are REJECTED → exact recovery → error = 0")

    # Save for plotting
    import json
    save_data = {k: [float(x) for x in v] for k, v in all_results.items()}
    save_data["_meta"] = {"model": args.model, "n_prompts": n_prompts,
                          "layers_tested": n_layers_to_test}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(save_data, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
