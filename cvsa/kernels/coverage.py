"""Triton kernels for CVSA-specific operations.

Fused, optimized replacements for PyTorch implementations:
1. Complement indices: replaces sort-based complement with O(n) cumsum compaction
2. Coverage check: fuses logsumexp(all) + logsumexp(sel) + compare
3. LSE merge: fuses all element-wise merge operations

These components are CVSA-specific overhead; optimizing them does not affect baselines.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ═══════════════════════════════════════════════════════════════
# 1. Complement index computation
# ═══════════════════════════════════════════════════════════════
# Replaces: compute_complement_pages() with its expensive sort
# Uses O(n) cumsum-based compaction per (b, h_kv) instead of O(n log n) sort.
# The existing tiled attention kernel is already well-optimized, so we keep it.

@triton.jit
def _complement_idx_kernel(
    SelectedPages,   # [B_H_kv, MAX_SEL]
    NSelected,       # [B_H_kv]
    ContextLens,     # [B]
    AcceptMask,      # [B_H_kv] uint8
    CompIndices,     # [B_H_kv, MAX_COMP] output
    NComplement,     # [B_H_kv] output int32
    block_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    MAX_SEL: tl.constexpr,
    MAX_COMP: tl.constexpr,
    PAGES_BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Compute complement page indices: all valid pages NOT in selected set.

    Grid: (B * H_kv,)
    Each program handles one (batch, kv_head).
    Uses vectorized membership check + cumsum compaction per chunk.
    """
    bh_idx = tl.program_id(0)
    batch_idx = bh_idx // num_kv_heads

    # Early exit for accepted heads
    accepted = tl.load(AcceptMask + bh_idx)
    if accepted != 0:
        tl.store(NComplement + bh_idx, 0)
        return

    ctx_len = tl.load(ContextLens + batch_idx).to(tl.int32)
    n_valid = (ctx_len - 1) // block_size + 1
    n_sel = tl.load(NSelected + bh_idx).to(tl.int32)

    # Load selected page indices: [MAX_SEL]
    sel_offs = tl.arange(0, MAX_SEL)
    sel_pages = tl.load(SelectedPages + bh_idx * MAX_SEL + sel_offs,
                        mask=sel_offs < n_sel, other=-1).to(tl.int32)

    # Process pages in chunks, compact complement indices using cumsum
    write_count = 0

    for c in range(NUM_CHUNKS):
        page_ids = c * PAGES_BLOCK + tl.arange(0, PAGES_BLOCK)
        valid = page_ids < n_valid

        # Vectorized membership check: is each page in selected set?
        is_selected = tl.zeros([PAGES_BLOCK], dtype=tl.int1)
        for i in range(MAX_SEL):
            is_selected = is_selected | ((page_ids == tl.load(
                SelectedPages + bh_idx * MAX_SEL + i).to(tl.int32)) & (i < n_sel))

        complement = valid & ~is_selected
        comp_int = complement.to(tl.int32)

        # Cumsum for compaction: positions within this chunk
        positions = tl.cumsum(comp_int, axis=0) - 1 + write_count
        n_in_chunk = tl.sum(comp_int)

        # Store complement page IDs at their compacted positions
        tl.store(CompIndices + bh_idx * MAX_COMP + positions,
                 page_ids.to(tl.int32),
                 mask=complement & (positions < MAX_COMP) & (positions >= 0))

        write_count += n_in_chunk

    tl.store(NComplement + bh_idx, write_count.to(tl.int32))


def triton_compute_complement(
    page_indices: torch.Tensor,    # [B, H_kv, max_selected] int32
    n_per_head: torch.Tensor,      # [B, H_kv] int32
    context_lens: torch.Tensor,    # [B] int32
    block_size: int,
    accept_mask: torch.Tensor,     # [B, H_kv] bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute complement page indices: all valid pages NOT in selected set.

    Drop-in replacement for compute_complement_pages() using Triton.
    O(n) per head instead of O(n log n) sort.

    Returns: (complement_indices [B,H_kv,max_comp] int32, n_complement [B,H_kv] int32)
    """
    B, H_kv, max_selected = page_indices.shape

    # Pad selected pages to next power of 2
    MAX_SEL = _next_power_of_2(max_selected)
    if MAX_SEL > max_selected:
        pad = MAX_SEL - max_selected
        page_indices = torch.nn.functional.pad(page_indices, (0, pad), value=-1)

    n_valid_max = int(((context_lens.max().item() - 1) // block_size) + 1)
    MAX_COMP = n_valid_max  # upper bound on complement size

    PAGES_BLOCK = min(256, _next_power_of_2(n_valid_max))
    NUM_CHUNKS = (n_valid_max + PAGES_BLOCK - 1) // PAGES_BLOCK

    # Flatten inputs
    sp_flat = page_indices.reshape(B * H_kv, MAX_SEL).contiguous()
    ns_flat = n_per_head.reshape(B * H_kv).contiguous()
    am_flat = accept_mask.to(torch.uint8).reshape(B * H_kv).contiguous()

    # Allocate outputs
    comp_indices = torch.zeros(B * H_kv, MAX_COMP, dtype=torch.int32, device=page_indices.device)
    n_complement = torch.zeros(B * H_kv, dtype=torch.int32, device=page_indices.device)

    grid = (B * H_kv,)

    _complement_idx_kernel[grid](
        sp_flat, ns_flat, context_lens, am_flat,
        comp_indices, n_complement,
        block_size=block_size,
        num_kv_heads=H_kv,
        MAX_SEL=MAX_SEL,
        MAX_COMP=MAX_COMP,
        PAGES_BLOCK=PAGES_BLOCK,
        NUM_CHUNKS=NUM_CHUNKS,
    )

    return (comp_indices.reshape(B, H_kv, MAX_COMP),
            n_complement.reshape(B, H_kv))


# ═══════════════════════════════════════════════════════════════
# 2. Coverage check kernel
# ═══════════════════════════════════════════════════════════════
# Replaces: logsumexp(all) + gather + logsumexp(sel) + exp + compare
# 6-8 PyTorch kernel launches → 1 Triton kernel.

@triton.jit
def _coverage_kernel(
    PageScores,        # [B_H_kv, n_pages]
    SelectedIndices,   # [B_H_kv, MAX_SEL]
    NPerHead,          # [B_H_kv]
    AcceptOut,         # [B_H_kv] uint8 output
    CoverageOut,       # [B_H_kv] float32 output
    tau,               # float32 threshold
    n_pages: tl.constexpr,
    MAX_SEL: tl.constexpr,
    PAGES_BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Coverage check: computes coverage = exp(lse_sel - lse_all) per (b, h_kv).

    Grid: (B * H_kv,)
    """
    bh_idx = tl.program_id(0)
    n_sel = tl.load(NPerHead + bh_idx).to(tl.int32)

    # 1. lse_all = logsumexp over ALL page scores
    max_all = -1e20
    sum_exp_all = 0.0

    for c in range(NUM_CHUNKS):
        offs = c * PAGES_BLOCK + tl.arange(0, PAGES_BLOCK)
        mask = offs < n_pages
        scores = tl.load(PageScores + bh_idx * n_pages + offs,
                         mask=mask, other=-1e20).to(tl.float32)

        chunk_max = tl.max(scores, axis=0)
        m_new = tl.maximum(max_all, chunk_max)
        sum_exp_all = sum_exp_all * tl.exp(max_all - m_new) + \
                      tl.sum(tl.exp(scores - m_new))
        max_all = m_new

    lse_all = max_all + tl.log(sum_exp_all + 1e-30)

    # 2. lse_sel = logsumexp over SELECTED page scores (gather + reduce)
    sel_offs = tl.arange(0, MAX_SEL)
    sel_mask = sel_offs < n_sel
    sel_idx = tl.load(SelectedIndices + bh_idx * MAX_SEL + sel_offs,
                      mask=sel_mask, other=0).to(tl.int32)
    sel_scores = tl.load(PageScores + bh_idx * n_pages + sel_idx,
                         mask=sel_mask, other=-1e20).to(tl.float32)

    max_sel = tl.max(tl.where(sel_mask, sel_scores, -1e20))
    sum_exp_sel = tl.sum(tl.where(sel_mask,
                                   tl.exp(sel_scores - max_sel), 0.0))
    lse_sel = max_sel + tl.log(sum_exp_sel + 1e-30)

    # 3. Coverage and accept/reject
    cov = tl.exp(lse_sel - lse_all)
    accept = cov >= tau

    tl.store(AcceptOut + bh_idx, accept.to(tl.uint8))
    tl.store(CoverageOut + bh_idx, cov)


def triton_check_coverage(
    page_scores: torch.Tensor,       # [B, H_kv, n_pages]
    selected_indices: torch.Tensor,  # [B, H_kv, budget] int32
    n_per_head: torch.Tensor,        # [B, H_kv] int32
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute coverage and accept/reject mask.

    coverage[b,h] = exp(logsumexp(selected_scores) - logsumexp(all_scores))
    accept[b,h] = coverage >= tau

    Returns: (accept_mask [B,H_kv] bool, coverage [B,H_kv] float32)
    """
    B, H_kv, n_pages = page_scores.shape
    budget = selected_indices.shape[2]

    MAX_SEL = _next_power_of_2(budget)
    if MAX_SEL > budget:
        pad = MAX_SEL - budget
        selected_indices = torch.nn.functional.pad(selected_indices, (0, pad), value=0)

    PAGES_BLOCK = min(256, _next_power_of_2(n_pages))
    NUM_CHUNKS = (n_pages + PAGES_BLOCK - 1) // PAGES_BLOCK

    # Flatten
    ps_flat = page_scores.reshape(B * H_kv, n_pages).contiguous()
    si_flat = selected_indices.reshape(B * H_kv, MAX_SEL).contiguous()
    nph_flat = n_per_head.reshape(B * H_kv).contiguous()

    # Outputs
    accept_out = torch.empty(B * H_kv, device=page_scores.device, dtype=torch.uint8)
    coverage_out = torch.empty(B * H_kv, device=page_scores.device, dtype=torch.float32)

    grid = (B * H_kv,)

    _coverage_kernel[grid](
        ps_flat, si_flat, nph_flat,
        accept_out, coverage_out,
        tau,
        n_pages=n_pages,
        MAX_SEL=MAX_SEL,
        PAGES_BLOCK=PAGES_BLOCK,
        NUM_CHUNKS=NUM_CHUNKS,
    )

    accept_mask = accept_out.reshape(B, H_kv).bool()
    coverage = coverage_out.reshape(B, H_kv)
    return accept_mask, coverage


# ═══════════════════════════════════════════════════════════════
# 3. LSE merge kernel
# ═══════════════════════════════════════════════════════════════
# Replaces: selective_lse_merge() — 8+ PyTorch kernel launches → 1 Triton kernel.

@triton.jit
def _lse_merge_kernel(
    O_draft, M_draft, L_draft,
    O_comp, M_comp, L_comp,
    AcceptMask,
    Output,
    num_kv_heads: tl.constexpr,
    num_q_per_kv: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Merge draft and complement attention with per-head accept/reject.

    Grid: (B * H_q,)
    Accepted heads: O = O_draft / L_draft
    Rejected heads: O = LSE_merge(draft, complement)
    """
    bq_idx = tl.program_id(0)
    batch_idx = bq_idx // (num_kv_heads * num_q_per_kv)
    q_in_batch = bq_idx % (num_kv_heads * num_q_per_kv)
    kv_head_idx = q_in_batch // num_q_per_kv
    bkv_idx = batch_idx * num_kv_heads + kv_head_idx

    d_offs = tl.arange(0, HEAD_DIM)

    accepted = tl.load(AcceptMask + bkv_idx)
    m_d = tl.load(M_draft + bq_idx).to(tl.float32)
    l_d = tl.load(L_draft + bq_idx).to(tl.float32)
    o_d = tl.load(O_draft + bq_idx * HEAD_DIM + d_offs,
                  mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

    if accepted != 0:
        # Accepted: simple normalization
        l_safe = tl.maximum(l_d, 1e-6)
        out = o_d / l_safe
    else:
        # Rejected: online softmax merge for exact dense output
        m_c = tl.load(M_comp + bq_idx).to(tl.float32)
        l_c = tl.load(L_comp + bq_idx).to(tl.float32)
        o_c = tl.load(O_comp + bq_idx * HEAD_DIM + d_offs,
                      mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

        m_all = tl.maximum(m_d, m_c)
        scale_d = tl.exp(m_d - m_all)
        scale_c = tl.exp(m_c - m_all)

        l_merged = l_d * scale_d + l_c * scale_c
        o_merged = o_d * scale_d + o_c * scale_c

        l_safe = tl.maximum(l_merged, 1e-6)
        out = o_merged / l_safe

    tl.store(Output + bq_idx * HEAD_DIM + d_offs,
             out.to(Output.dtype.element_ty),
             mask=d_offs < HEAD_DIM)


def triton_lse_merge(
    o_draft: torch.Tensor,     # [B, H_q, D] float32 unnormalized
    m_draft: torch.Tensor,     # [B, H_q] float32
    l_draft: torch.Tensor,     # [B, H_q] float32
    o_comp: torch.Tensor,      # [B, H_q, D] float32 unnormalized
    m_comp: torch.Tensor,      # [B, H_q] float32
    l_comp: torch.Tensor,      # [B, H_q] float32
    accept_mask: torch.Tensor, # [B, H_kv] bool
    num_kv_heads: int,
    num_q_per_kv: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Merge draft and complement attention with per-head accept/reject.

    Returns: [B, H_q, D] in output_dtype.
    """
    B, H_q, D = o_draft.shape

    od_flat = o_draft.reshape(B * H_q, D).contiguous()
    md_flat = m_draft.reshape(B * H_q).contiguous()
    ld_flat = l_draft.reshape(B * H_q).contiguous()
    oc_flat = o_comp.reshape(B * H_q, D).contiguous()
    mc_flat = m_comp.reshape(B * H_q).contiguous()
    lc_flat = l_comp.reshape(B * H_q).contiguous()
    am_flat = accept_mask.to(torch.uint8).reshape(B * num_kv_heads).contiguous()

    output = torch.empty(B * H_q, D, device=o_draft.device, dtype=output_dtype)

    grid = (B * H_q,)
    _lse_merge_kernel[grid](
        od_flat, md_flat, ld_flat,
        oc_flat, mc_flat, lc_flat,
        am_flat, output,
        num_kv_heads=num_kv_heads,
        num_q_per_kv=num_q_per_kv,
        HEAD_DIM=D,
    )

    return output.reshape(B, H_q, D)
