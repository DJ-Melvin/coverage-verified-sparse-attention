"""Triton kernel for fused page scoring: max(Q·K) per page per KV head.

A single fused kernel that reads K directly from the paged cache, one KV
head at a time, instead of gathering all heads into a dense tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _page_score_kernel(
    Q_rep,         # [B, H_kv, D] — representative Q per KV head
    K_cache,       # [num_blocks, BLOCK_SIZE, H_kv, D]
    BlockTables,   # [B, max_pages]
    ContextLens,   # [B]
    PageScores,    # [B, H_kv, max_pages] output
    scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_pages: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Strides
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_btb: tl.constexpr, stride_btp: tl.constexpr,
    stride_psb: tl.constexpr, stride_psh: tl.constexpr, stride_psp: tl.constexpr,
):
    """Compute max(Q·K_token) for each page and KV head.

    Grid: (B * H_kv, max_pages)
    Each program computes one page score for one (batch, kv_head).
    """
    bh_idx = tl.program_id(0)   # batch * H_kv index
    page_idx = tl.program_id(1)  # logical page index

    batch_idx = bh_idx // num_kv_heads
    kv_head = bh_idx % num_kv_heads

    # Check if this page is valid
    ctx_len = tl.load(ContextLens + batch_idx).to(tl.int32)
    page_start_token = page_idx * BLOCK_SIZE
    if page_start_token >= ctx_len:
        tl.store(PageScores + batch_idx * stride_psb + kv_head * stride_psh + page_idx * stride_psp,
                 float("-inf"))
        return

    tokens_in_page = tl.minimum(BLOCK_SIZE, ctx_len - page_start_token)

    # Get physical block
    phys_block = tl.load(BlockTables + batch_idx * stride_btb + page_idx * stride_btp).to(tl.int32)

    # Load Q: [HEAD_DIM]
    d_offs = tl.arange(0, HEAD_DIM)
    q_vec = tl.load(Q_rep + batch_idx * stride_qb + kv_head * stride_qh + d_offs * stride_qd,
                     mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

    # Load K block: [BLOCK_SIZE, HEAD_DIM]
    t_offs = tl.arange(0, BLOCK_SIZE)
    k_base = K_cache + phys_block * stride_kb + kv_head * stride_kh
    k_ptrs = k_base + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    valid_t = t_offs < tokens_in_page
    k_block = tl.load(k_ptrs, mask=valid_t[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

    # Compute Q·K scores: [BLOCK_SIZE]
    scores = tl.sum(k_block * q_vec[None, :], axis=1) * scale
    scores = tl.where(valid_t, scores, float("-inf"))

    # Max score for this page
    max_score = tl.max(scores, axis=0)

    tl.store(PageScores + batch_idx * stride_psb + kv_head * stride_psh + page_idx * stride_psp,
             max_score)


def triton_score_pages(
    q: torch.Tensor,           # [B, H_q, D]
    k_cache: torch.Tensor,     # [num_blocks, block_size, H_kv, D]
    block_tables: torch.Tensor, # [B, max_pages]
    context_lens: torch.Tensor, # [B]
    num_kv_heads: int,
    num_q_per_kv: int,
    block_size: int,
) -> torch.Tensor:
    """Fused Triton page scorer: max(Q·K) per page per KV head.

    Returns: page_scores [B, H_kv, max_pages] float32
    """
    B, H_q, D = q.shape
    max_pages = block_tables.shape[1]

    # GQA: representative Q per KV head (mean over group)
    q_grouped = q.view(B, num_kv_heads, num_q_per_kv, D)
    q_rep = q_grouped.mean(dim=2).contiguous()  # [B, H_kv, D]

    page_scores = torch.full(
        (B, num_kv_heads, max_pages), float("-inf"),
        device=q.device, dtype=torch.float32,
    )

    grid = (B * num_kv_heads, max_pages)

    _page_score_kernel[grid](
        q_rep, k_cache, block_tables, context_lens, page_scores,
        scale=D ** -0.5,
        num_kv_heads=num_kv_heads,
        max_pages=max_pages,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        stride_qb=q_rep.stride(0), stride_qh=q_rep.stride(1), stride_qd=q_rep.stride(2),
        stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_btb=block_tables.stride(0), stride_btp=block_tables.stride(1),
        stride_psb=page_scores.stride(0), stride_psh=page_scores.stride(1),
        stride_psp=page_scores.stride(2),
    )

    return page_scores
