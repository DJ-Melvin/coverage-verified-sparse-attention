"""Correctness tests for the CVSA Triton kernels.

Compares new fused Triton implementations against original PyTorch implementations.
"""

import os
import sys
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvsa.kernels.coverage import (
    triton_compute_complement,
    triton_check_coverage,
    triton_lse_merge,
)
from cvsa.complement import compute_complement_pages
from cvsa.lse_merge import selective_lse_merge
from cvsa.kernels.sparse_decode import triton_sparse_decode_unnorm


def make_test_data(B=1, H_kv=8, H_q=32, D=128, block_size=16,
                   n_ctx=512, budget_pct=0.10, device="cuda"):
    """Create synthetic test data matching real CVSA shapes."""
    num_q_per_kv = H_q // H_kv
    n_pages = (n_ctx - 1) // block_size + 1
    budget = max(3, int(n_pages * budget_pct))

    # Allocate paged KV cache
    num_blocks = B * n_pages + 16  # extra blocks
    k_cache = torch.randn(num_blocks, block_size, H_kv, D, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, block_size, H_kv, D, device=device, dtype=torch.bfloat16)

    # Block tables: identity mapping (logical page i → physical block i + offset)
    block_tables = torch.zeros(B, n_pages, dtype=torch.int32, device=device)
    for b in range(B):
        block_tables[b] = torch.arange(n_pages, device=device) + b * n_pages

    context_lens = torch.full((B,), n_ctx, dtype=torch.int32, device=device)
    q = torch.randn(B, H_q, D, device=device, dtype=torch.bfloat16)

    # Page scores and top-k selection
    page_scores = torch.randn(B, H_kv, n_pages, device=device, dtype=torch.float32)
    topk_vals, topk_indices = torch.topk(page_scores, k=budget, dim=-1, largest=True, sorted=False)
    page_indices = topk_indices.to(torch.int32)
    n_per_head = torch.full((B, H_kv), budget, dtype=torch.int32, device=device)

    # Coverage → accept_mask (with some heads rejected)
    lse_all = torch.logsumexp(page_scores, dim=-1)
    idx_range = torch.arange(budget, device=device)
    head_mask = idx_range.unsqueeze(0).unsqueeze(0) < n_per_head.unsqueeze(-1)
    masked_vals = topk_vals.masked_fill(~head_mask, float('-inf'))
    lse_sel = torch.logsumexp(masked_vals, dim=-1)
    coverage = (lse_sel - lse_all).exp()
    # Use a high tau to ensure SOME heads are rejected for testing
    tau = 0.999
    accept_mask = coverage >= tau

    return {
        "B": B, "H_kv": H_kv, "H_q": H_q, "D": D,
        "block_size": block_size, "budget": budget,
        "num_q_per_kv": num_q_per_kv, "n_pages": n_pages,
        "q": q, "k_cache": k_cache, "v_cache": v_cache,
        "block_tables": block_tables, "context_lens": context_lens,
        "page_scores": page_scores, "topk_vals": topk_vals,
        "page_indices": page_indices, "n_per_head": n_per_head,
        "accept_mask": accept_mask, "coverage": coverage, "tau": tau,
    }


def test_complement_indices():
    """Test Triton complement index computation vs original sort-based approach."""
    print("=" * 60)
    print("TEST: Complement Indices")
    print("=" * 60)

    for B in [1, 4]:
        for n_ctx in [256, 512, 1024, 8192]:
            d = make_test_data(B=B, n_ctx=n_ctx)

            n_rejected = (~d["accept_mask"]).sum().item()
            if n_rejected == 0:
                d["accept_mask"][0, 0] = False

            # --- Original: sort-based ---
            comp_ref, n_comp_ref = compute_complement_pages(
                d["page_indices"], d["n_per_head"], d["context_lens"],
                d["block_size"], d["accept_mask"])

            # --- New: Triton cumsum-based ---
            comp_new, n_comp_new = triton_compute_complement(
                d["page_indices"], d["n_per_head"], d["context_lens"],
                d["block_size"], d["accept_mask"])

            # Compare n_complement counts
            count_match = (n_comp_ref == n_comp_new).all().item()

            # Compare actual indices (sets should be equal, order may differ)
            all_match = True
            for b in range(B):
                for h in range(d["H_kv"]):
                    nc = n_comp_ref[b, h].item()
                    if nc == 0:
                        continue
                    ref_set = set(comp_ref[b, h, :nc].tolist())
                    new_set = set(comp_new[b, h, :nc].tolist())
                    if ref_set != new_set:
                        all_match = False

            status = "PASS" if (count_match and all_match) else "FAIL"
            print(f"  B={B}, ctx={n_ctx}: count_match={count_match}, "
                  f"sets_match={all_match} [{n_rejected} rejected] [{status}]")

    print()


def test_coverage():
    """Test Triton coverage check vs original PyTorch implementation."""
    print("=" * 60)
    print("TEST: Coverage Check")
    print("=" * 60)

    for B in [1, 4]:
        for n_ctx in [256, 512, 1024]:
            for tau in [0.80, 0.90, 0.95]:
                d = make_test_data(B=B, n_ctx=n_ctx)

                # --- Original PyTorch ---
                page_scores = d["page_scores"]
                page_indices = d["page_indices"]
                n_per_head = d["n_per_head"]
                budget = d["budget"]

                lse_all = torch.logsumexp(page_scores, dim=-1)
                idx_range = torch.arange(budget, device="cuda")
                head_mask = idx_range.unsqueeze(0).unsqueeze(0) < n_per_head.unsqueeze(-1)
                topk_vals = torch.gather(page_scores, 2, page_indices.long())
                masked = topk_vals.masked_fill(~head_mask, float('-inf'))
                lse_sel = torch.logsumexp(masked, dim=-1)
                cov_ref = (lse_sel - lse_all).exp()
                accept_ref = cov_ref >= tau

                # --- New Triton ---
                accept_new, cov_new = triton_check_coverage(
                    page_scores, page_indices, n_per_head, tau)

                # Compare
                cov_diff = (cov_ref - cov_new).abs().max().item()
                accept_match = (accept_ref == accept_new).all().item()

                status = "PASS" if (cov_diff < 1e-4 and accept_match) else "FAIL"
                print(f"  B={B}, ctx={n_ctx}, tau={tau}: "
                      f"cov_diff={cov_diff:.6f}, accept_match={accept_match} [{status}]")

    print()


def test_lse_merge():
    """Test Triton LSE merge vs original PyTorch implementation."""
    print("=" * 60)
    print("TEST: LSE Merge")
    print("=" * 60)

    for B in [1, 4]:
        H_kv, H_q, D = 8, 32, 128
        num_q_per_kv = H_q // H_kv

        # Create synthetic draft/complement attention outputs
        o_d = torch.randn(B, H_q, D, device="cuda", dtype=torch.float32)
        m_d = torch.randn(B, H_q, device="cuda", dtype=torch.float32)
        l_d = torch.rand(B, H_q, device="cuda", dtype=torch.float32) + 0.1

        o_c = torch.randn(B, H_q, D, device="cuda", dtype=torch.float32)
        m_c = torch.randn(B, H_q, device="cuda", dtype=torch.float32)
        l_c = torch.rand(B, H_q, device="cuda", dtype=torch.float32) + 0.1

        # Test with various accept rates
        for accept_rate in [0.0, 0.5, 1.0]:
            if accept_rate == 0.0:
                accept_mask = torch.zeros(B, H_kv, dtype=torch.bool, device="cuda")
            elif accept_rate == 1.0:
                accept_mask = torch.ones(B, H_kv, dtype=torch.bool, device="cuda")
            else:
                accept_mask = torch.rand(B, H_kv, device="cuda") > 0.5

            # --- Original PyTorch ---
            ref = selective_lse_merge(
                o_d, m_d, l_d, o_c, m_c, l_c,
                accept_mask, num_q_per_kv, torch.bfloat16)

            # --- New Triton ---
            new = triton_lse_merge(
                o_d, m_d, l_d, o_c, m_c, l_c,
                accept_mask, H_kv, num_q_per_kv, torch.bfloat16)

            max_diff = (ref.float() - new.float()).abs().max().item()
            mean_diff = (ref.float() - new.float()).abs().mean().item()

            status = "PASS" if max_diff < 0.01 else "FAIL"
            print(f"  B={B}, accept={accept_rate:.0%}: "
                  f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")

    print()


def benchmark_kernels():
    """Quick latency comparison: old vs new."""
    import time

    print("=" * 60)
    print("BENCHMARK: Old vs New (100 iterations, B=1, 8K context)")
    print("=" * 60)

    d = make_test_data(B=1, n_ctx=8192)
    # Force some rejections
    d["accept_mask"][:, :4] = False
    n_warmup = 20
    n_iters = 100

    # --- Complement Index: Old ---
    for _ in range(n_warmup):
        compute_complement_pages(
            d["page_indices"], d["n_per_head"], d["context_lens"],
            d["block_size"], d["accept_mask"])
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        compute_complement_pages(
            d["page_indices"], d["n_per_head"], d["context_lens"],
            d["block_size"], d["accept_mask"])
    torch.cuda.synchronize()
    old_comp = (time.perf_counter() - t0) / n_iters * 1000

    # --- Complement Index: New ---
    for _ in range(n_warmup):
        triton_compute_complement(
            d["page_indices"], d["n_per_head"], d["context_lens"],
            d["block_size"], d["accept_mask"])
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        triton_compute_complement(
            d["page_indices"], d["n_per_head"], d["context_lens"],
            d["block_size"], d["accept_mask"])
    torch.cuda.synchronize()
    new_comp = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  Complement Idx:   old={old_comp:.3f}ms  new={new_comp:.3f}ms  "
          f"speedup={old_comp/new_comp:.2f}x")

    # --- Coverage: Old ---
    page_scores = d["page_scores"]
    page_indices = d["page_indices"]
    n_per_head = d["n_per_head"]
    budget = d["budget"]

    for _ in range(n_warmup):
        lse_all = torch.logsumexp(page_scores, dim=-1)
        idx_range = torch.arange(budget, device="cuda")
        head_mask = idx_range.unsqueeze(0).unsqueeze(0) < n_per_head.unsqueeze(-1)
        topk_vals = torch.gather(page_scores, 2, page_indices.long())
        masked = topk_vals.masked_fill(~head_mask, float('-inf'))
        lse_sel = torch.logsumexp(masked, dim=-1)
        coverage = (lse_sel - lse_all).exp()
        accept = coverage >= 0.90
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        lse_all = torch.logsumexp(page_scores, dim=-1)
        idx_range = torch.arange(budget, device="cuda")
        head_mask = idx_range.unsqueeze(0).unsqueeze(0) < n_per_head.unsqueeze(-1)
        topk_vals = torch.gather(page_scores, 2, page_indices.long())
        masked = topk_vals.masked_fill(~head_mask, float('-inf'))
        lse_sel = torch.logsumexp(masked, dim=-1)
        coverage = (lse_sel - lse_all).exp()
        accept = coverage >= 0.90
    torch.cuda.synchronize()
    old_cov = (time.perf_counter() - t0) / n_iters * 1000

    # --- Coverage: New ---
    for _ in range(n_warmup):
        triton_check_coverage(page_scores, page_indices, n_per_head, 0.90)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        triton_check_coverage(page_scores, page_indices, n_per_head, 0.90)
    torch.cuda.synchronize()
    new_cov = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  Coverage:         old={old_cov:.3f}ms  new={new_cov:.3f}ms  "
          f"speedup={old_cov/new_cov:.2f}x")

    # --- LSE Merge: Old ---
    o_d = torch.randn(1, 32, 128, device="cuda", dtype=torch.float32)
    m_d = torch.randn(1, 32, device="cuda", dtype=torch.float32)
    l_d = torch.rand(1, 32, device="cuda", dtype=torch.float32) + 0.1
    o_c = torch.randn(1, 32, 128, device="cuda", dtype=torch.float32)
    m_c = torch.randn(1, 32, device="cuda", dtype=torch.float32)
    l_c = torch.rand(1, 32, device="cuda", dtype=torch.float32) + 0.1
    am = d["accept_mask"]

    for _ in range(n_warmup):
        selective_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, am, 4, torch.bfloat16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        selective_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, am, 4, torch.bfloat16)
    torch.cuda.synchronize()
    old_merge = (time.perf_counter() - t0) / n_iters * 1000

    # --- LSE Merge: New ---
    for _ in range(n_warmup):
        triton_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, am, 8, 4, torch.bfloat16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        triton_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, am, 8, 4, torch.bfloat16)
    torch.cuda.synchronize()
    new_merge = (time.perf_counter() - t0) / n_iters * 1000

    print(f"  LSE Merge:        old={old_merge:.3f}ms  new={new_merge:.3f}ms  "
          f"speedup={old_merge/new_merge:.2f}x")

    total_old = old_comp + old_cov + old_merge
    total_new = new_comp + new_cov + new_merge
    print(f"  ──────────────────────────────────────")
    print(f"  TOTAL:            old={total_old:.3f}ms  new={total_new:.3f}ms  "
          f"speedup={total_old/total_new:.2f}x")
    print()


if __name__ == "__main__":
    print("CVSA Triton Kernel Correctness Tests")
    print("=" * 60)
    print()

    test_coverage()
    test_lse_merge()
    test_complement_indices()
    benchmark_kernels()
