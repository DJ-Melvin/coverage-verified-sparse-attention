"""Triton kernel for sparse paged attention decode.

Two versions:
1. Single-program kernel: one program per (batch, q_head). Simple, good at high BS.
2. Tiled kernel: multiple programs per head for better GPU utilization at low BS.
   Uses PAGES_PER_TILE pages per program with cross-tile log-sum-exp reduction.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_decode_kernel(
    Q,             # [B * H_q, D]
    K_cache,       # [num_blocks, BLOCK_SIZE, H_kv, D]
    V_cache,       # [num_blocks, BLOCK_SIZE, H_kv, D]
    Output,        # [B * H_q, D]
    PageIndices,   # [B * H_kv, max_selected]
    NPerHead,      # [B * H_kv]
    BlockTables,   # [B, max_pages_total]
    ContextLens,   # [B]
    scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_per_kv: tl.constexpr,
    max_selected: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_qbh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_obh: tl.constexpr, stride_od: tl.constexpr,
    stride_pibh: tl.constexpr, stride_pis: tl.constexpr,
    stride_btb: tl.constexpr, stride_btp: tl.constexpr,
):
    """Per-(batch, q_head) sparse attention over selected pages.

    Each program handles one (batch, q_head) and iterates over all selected
    pages for the corresponding KV head using online softmax.

    Key optimization: loads K/V as [BLOCK_SIZE, HEAD_DIM] blocks and computes
    dot products as matrix-vector products using tl.dot / tl.sum.
    """
    bq_idx = tl.program_id(0)  # index into [B * H_q]

    batch_idx = bq_idx // (num_kv_heads * num_q_per_kv)
    q_head_in_batch = bq_idx % (num_kv_heads * num_q_per_kv)
    kv_head_idx = q_head_in_batch // num_q_per_kv
    bkv_idx = batch_idx * num_kv_heads + kv_head_idx

    n_sel = tl.load(NPerHead + bkv_idx).to(tl.int32)
    ctx_len = tl.load(ContextLens + batch_idx).to(tl.int32)

    # Load Q vector: [HEAD_DIM]
    d_offs = tl.arange(0, HEAD_DIM)
    q_vec = tl.load(Q + bq_idx * stride_qbh + d_offs * stride_qd,
                     mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

    # Online softmax accumulators
    m_acc = tl.full([1], float("-inf"), dtype=tl.float32)  # [1] scalar trick
    l_acc = tl.zeros([1], dtype=tl.float32)
    o_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Token offsets within a page (reused across pages)
    t_offs = tl.arange(0, BLOCK_SIZE)

    for p_rank in range(max_selected):
        # Mask for pages beyond selection count
        page_active = p_rank < n_sel

        # Load logical page index and map to physical block
        logical_page = tl.load(PageIndices + bkv_idx * stride_pibh + p_rank * stride_pis).to(tl.int32)
        physical_block = tl.load(BlockTables + batch_idx * stride_btb + logical_page * stride_btp).to(tl.int32)

        # Valid tokens in this page
        page_start = logical_page * BLOCK_SIZE
        tokens_in_page = tl.minimum(BLOCK_SIZE, ctx_len - page_start)
        # Mask: valid token AND page is active
        valid = (t_offs < tokens_in_page) & page_active

        # ── Compute Q·K scores ──
        # Load K: [BLOCK_SIZE, HEAD_DIM] for this page and KV head
        # K_cache layout: [num_blocks, BLOCK_SIZE, H_kv, D]
        # Base pointer for this page and head
        k_base = K_cache + physical_block * stride_kb + kv_head_idx * stride_kh

        # Load K values: [BLOCK_SIZE] for each dimension, accumulate dot product
        # We compute scores[t] = sum_d(q[d] * K[t, d]) using tl.dot
        # But K is strided, so we load per-dimension and accumulate

        # Approach: load K as [BLOCK_SIZE, HEAD_DIM] block
        # k_ptrs[t, d] = k_base + t * stride_kt + d * stride_kd
        k_ptrs = k_base + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=valid[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

        # scores[t] = sum_d(q[d] * K[t, d])
        scores = tl.sum(k_block * q_vec[None, :], axis=1) * scale  # [BLOCK_SIZE]
        scores = tl.where(valid, scores, float("-inf"))

        # ── Online softmax update ──
        m_page = tl.max(scores, axis=0)  # scalar
        m_page_vec = tl.full([1], m_page, dtype=tl.float32)
        m_new = tl.maximum(m_acc, m_page_vec)

        # Rescale previous accumulator
        alpha = tl.exp(m_acc - m_new)  # [1]
        p_vals = tl.exp(scores - m_new)  # [BLOCK_SIZE] (broadcasts m_new scalar)
        p_vals = tl.where(valid, p_vals, 0.0)
        l_page = tl.sum(p_vals, axis=0)

        l_acc = l_acc * alpha + l_page
        o_acc = o_acc * tl.broadcast_to(alpha, [HEAD_DIM])

        # ── Accumulate weighted V ──
        v_base = V_cache + physical_block * stride_vb + kv_head_idx * stride_vh
        v_ptrs = v_base + t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=valid[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

        # weighted_v[d] = sum_t(p_vals[t] * V[t, d])
        weighted_v = tl.sum(p_vals[:, None] * v_block, axis=0)  # [HEAD_DIM]
        o_acc = o_acc + weighted_v

        m_acc = m_new

    # Normalize
    o_acc = o_acc / tl.maximum(tl.broadcast_to(l_acc, [HEAD_DIM]), 1e-6)

    # Store output (match input dtype: bfloat16 or float16)
    tl.store(Output + bq_idx * stride_obh + d_offs * stride_od,
             o_acc.to(Q.dtype.element_ty),
             mask=d_offs < HEAD_DIM)


@triton.jit
def _sparse_decode_tiled_kernel(
    Q, K_cache, V_cache,
    M_scratch, L_scratch, O_scratch,
    PageIndices, NPerHead, BlockTables, ContextLens,
    scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_per_kv: tl.constexpr,
    max_selected: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    stride_qbh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_pibh: tl.constexpr, stride_pis: tl.constexpr,
    stride_btb: tl.constexpr, stride_btp: tl.constexpr,
    num_tiles: tl.constexpr,
):
    """Tiled sparse decode: each program handles PAGES_PER_TILE pages.

    Grid: (B * H_q, num_tiles)
    Writes per-tile (m, l, o) for cross-tile reduction.
    """
    bq_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)

    batch_idx = bq_idx // (num_kv_heads * num_q_per_kv)
    q_head_in_batch = bq_idx % (num_kv_heads * num_q_per_kv)
    kv_head_idx = q_head_in_batch // num_q_per_kv
    bkv_idx = batch_idx * num_kv_heads + kv_head_idx

    n_sel = tl.load(NPerHead + bkv_idx).to(tl.int32)
    ctx_len = tl.load(ContextLens + batch_idx).to(tl.int32)

    tile_start = tile_idx * PAGES_PER_TILE
    # Scratch base offsets (flat: [B*H_q, num_tiles] for m/l, [B*H_q, num_tiles, D] for o)
    scratch_idx = bq_idx * num_tiles + tile_idx

    if tile_start >= n_sel:
        tl.store(M_scratch + scratch_idx, float("-inf"))
        tl.store(L_scratch + scratch_idx, 0.0)
        d_offs = tl.arange(0, HEAD_DIM)
        tl.store(O_scratch + scratch_idx * HEAD_DIM + d_offs,
                 tl.zeros([HEAD_DIM], dtype=tl.float32),
                 mask=d_offs < HEAD_DIM)
        return

    # Load Q vector
    d_offs = tl.arange(0, HEAD_DIM)
    q_vec = tl.load(Q + bq_idx * stride_qbh + d_offs * stride_qd,
                     mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

    t_offs = tl.arange(0, BLOCK_SIZE)
    m_acc = tl.full([1], float("-inf"), dtype=tl.float32)
    l_acc = tl.zeros([1], dtype=tl.float32)
    o_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for p_offset in range(PAGES_PER_TILE):
        p_rank = tile_start + p_offset
        page_active = p_rank < n_sel

        logical_page = tl.load(PageIndices + bkv_idx * stride_pibh + p_rank * stride_pis).to(tl.int32)
        physical_block = tl.load(BlockTables + batch_idx * stride_btb + logical_page * stride_btp).to(tl.int32)

        page_start = logical_page * BLOCK_SIZE
        tokens_in_page = tl.minimum(BLOCK_SIZE, ctx_len - page_start)
        valid = (t_offs < tokens_in_page) & page_active

        k_base = K_cache + physical_block * stride_kb + kv_head_idx * stride_kh
        k_ptrs = k_base + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=valid[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

        scores = tl.sum(k_block * q_vec[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_page = tl.max(scores, axis=0)
        m_page_vec = tl.full([1], m_page, dtype=tl.float32)
        m_new = tl.maximum(m_acc, m_page_vec)

        alpha = tl.exp(m_acc - m_new)
        p_vals = tl.exp(scores - m_new)
        p_vals = tl.where(valid, p_vals, 0.0)
        l_page = tl.sum(p_vals, axis=0)

        l_acc = l_acc * alpha + l_page
        o_acc = o_acc * tl.broadcast_to(alpha, [HEAD_DIM])

        v_base = V_cache + physical_block * stride_vb + kv_head_idx * stride_vh
        v_ptrs = v_base + t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=valid[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

        weighted_v = tl.sum(p_vals[:, None] * v_block, axis=0)
        o_acc = o_acc + weighted_v
        m_acc = m_new

    # Store tile results — extract scalar from [1] vectors via tl.sum
    tl.store(M_scratch + scratch_idx, tl.sum(m_acc, axis=0))
    tl.store(L_scratch + scratch_idx, tl.sum(l_acc, axis=0))
    tl.store(O_scratch + scratch_idx * HEAD_DIM + d_offs, o_acc, mask=d_offs < HEAD_DIM)


@triton.jit
def _reduce_tiles_kernel(
    M_scratch, L_scratch, O_scratch,
    Output,
    num_tiles: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """Reduce per-tile (m, l, o) into final output via log-sum-exp."""
    bq_idx = tl.program_id(0)
    d_offs = tl.arange(0, HEAD_DIM)

    # Find global max across tiles
    global_m = float("-inf")
    for t in range(num_tiles):
        m_t = tl.load(M_scratch + bq_idx * num_tiles + t)
        global_m = tl.maximum(global_m, m_t)

    # Accumulate with rescaling
    global_l = 0.0
    global_o = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for t in range(num_tiles):
        m_t = tl.load(M_scratch + bq_idx * num_tiles + t)
        l_t = tl.load(L_scratch + bq_idx * num_tiles + t)

        alpha = tl.exp(m_t - global_m)
        global_l += l_t * alpha

        o_t = tl.load(O_scratch + (bq_idx * num_tiles + t) * HEAD_DIM + d_offs,
                       mask=d_offs < HEAD_DIM, other=0.0)
        global_o += o_t * alpha

    # Normalize
    global_o = global_o / tl.maximum(global_l, 1e-6)

    # Store output in original dtype
    tl.store(Output + bq_idx * HEAD_DIM + d_offs,
             global_o.to(out_dtype),
             mask=d_offs < HEAD_DIM)


def _next_power_of_2(n: int) -> int:
    """Round up to the next power of 2 (minimum 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def triton_sparse_decode(
    q: torch.Tensor,           # [B, H_q, D]
    k_cache: torch.Tensor,     # [num_blocks, block_size, H_kv, D]
    v_cache: torch.Tensor,     # [num_blocks, block_size, H_kv, D]
    block_tables: torch.Tensor, # [B, max_pages_total]
    page_indices: torch.Tensor, # [B, H_kv, max_selected]
    n_per_head: torch.Tensor,  # [B, H_kv]
    context_lens: torch.Tensor, # [B]
    num_kv_heads: int,
    num_q_per_kv: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Launch Triton sparse decode kernel.

    Automatically selects single-program or tiled kernel based on batch size
    and number of selected pages. Returns: [B, H_q, D].
    """
    B, H_q, D = q.shape
    actual_max_selected = page_indices.shape[2]

    padded_max_selected = _next_power_of_2(actual_max_selected)
    if padded_max_selected > actual_max_selected:
        pad_size = padded_max_selected - actual_max_selected
        pi_padded = torch.nn.functional.pad(page_indices, (0, pad_size), value=0)
    else:
        pi_padded = page_indices

    pi_flat = pi_padded.reshape(B * num_kv_heads, padded_max_selected).contiguous()
    nph_flat = n_per_head.reshape(B * num_kv_heads).contiguous()
    q_flat = q.reshape(B * H_q, D).contiguous()

    # Decide kernel: use tiled when many pages per program would slow down
    # single-program kernel. Tiled splits pages across programs for better
    # GPU utilization and reduced per-program work.
    total_programs = B * H_q
    use_tiled = padded_max_selected > 64

    if use_tiled:
        return _launch_tiled(
            q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables, context_lens,
            B, H_q, D, num_kv_heads, num_q_per_kv, padded_max_selected, block_size,
        )
    else:
        output = torch.zeros(B * H_q, D, device=q.device, dtype=q.dtype)

        _sparse_decode_kernel[(total_programs,)](
            q_flat, k_cache, v_cache, output,
            pi_flat, nph_flat, block_tables, context_lens,
            scale=D ** -0.5,
            num_kv_heads=num_kv_heads,
            num_q_per_kv=num_q_per_kv,
            max_selected=padded_max_selected,
            BLOCK_SIZE=block_size,
            HEAD_DIM=D,
            stride_qbh=q_flat.stride(0), stride_qd=q_flat.stride(1),
            stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
            stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
            stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
            stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
            stride_obh=output.stride(0), stride_od=output.stride(1),
            stride_pibh=pi_flat.stride(0), stride_pis=pi_flat.stride(1),
            stride_btb=block_tables.stride(0), stride_btp=block_tables.stride(1),
        )

        return output.reshape(B, H_q, D)


_tiled_scratch_cache = {}  # cache_key -> (m, l, o, output)


def _launch_tiled(q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables,
                  context_lens, B, H_q, D, num_kv_heads, num_q_per_kv,
                  max_selected, block_size):
    """Launch tiled kernel + reduction for better GPU utilization."""
    device = q_flat.device
    dtype = q_flat.dtype

    # Choose PAGES_PER_TILE to get enough programs
    total_bq = B * H_q
    target_tiles = max(4, (512 + total_bq - 1) // total_bq)
    pages_per_tile = max(1, _next_power_of_2(max_selected // target_tiles))
    pages_per_tile = max(4, min(pages_per_tile, max_selected))
    num_tiles = (max_selected + pages_per_tile - 1) // pages_per_tile
    num_tiles_padded = _next_power_of_2(num_tiles)

    # Reuse cached scratch buffers AND output buffer
    cache_key = (str(device), total_bq, num_tiles_padded, D, str(dtype))
    if cache_key in _tiled_scratch_cache:
        m_scratch, l_scratch, o_scratch, output = _tiled_scratch_cache[cache_key]
    else:
        m_scratch = torch.empty(total_bq * num_tiles_padded,
                               device=device, dtype=torch.float32)
        l_scratch = torch.empty(total_bq * num_tiles_padded,
                               device=device, dtype=torch.float32)
        o_scratch = torch.empty(total_bq * num_tiles_padded * D,
                               device=device, dtype=torch.float32)
        output = torch.empty(total_bq, D, device=device, dtype=dtype)
        _tiled_scratch_cache[cache_key] = (m_scratch, l_scratch, o_scratch, output)

    # Launch tiled kernel
    grid = (total_bq, num_tiles_padded)

    _sparse_decode_tiled_kernel[grid](
        q_flat, k_cache, v_cache,
        m_scratch, l_scratch, o_scratch,
        pi_flat, nph_flat, block_tables, context_lens,
        scale=D ** -0.5,
        num_kv_heads=num_kv_heads,
        num_q_per_kv=num_q_per_kv,
        max_selected=max_selected,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        PAGES_PER_TILE=pages_per_tile,
        stride_qbh=q_flat.stride(0), stride_qd=q_flat.stride(1),
        stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        stride_pibh=pi_flat.stride(0), stride_pis=pi_flat.stride(1),
        stride_btb=block_tables.stride(0), stride_btp=block_tables.stride(1),
        num_tiles=num_tiles_padded,
    )

    # Launch reduction kernel (writes all positions, so empty output is fine)
    out_dtype = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    _reduce_tiles_kernel[(total_bq,)](
        m_scratch, l_scratch, o_scratch,
        output,
        num_tiles=num_tiles_padded,
        HEAD_DIM=D,
        out_dtype=out_dtype,
    )

    return output.reshape(B, H_q, D)


@triton.jit
def _sparse_decode_unnorm_kernel(
    Q, K_cache, V_cache,
    O_out,     # [B * H_q, D] float32 — unnormalized accumulator
    M_out,     # [B * H_q] float32 — running max
    L_out,     # [B * H_q] float32 — running sum-of-exp
    PageIndices, NPerHead, BlockTables, ContextLens,
    scale: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_q_per_kv: tl.constexpr,
    max_selected: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_qbh: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
    stride_pibh: tl.constexpr, stride_pis: tl.constexpr,
    stride_btb: tl.constexpr, stride_btp: tl.constexpr,
):
    """Like _sparse_decode_kernel but stores (o_unnorm, m, l) without normalizing.

    Used by MRSA to merge fine sparse attention with coarse page-summary attention
    via online softmax.
    """
    bq_idx = tl.program_id(0)

    batch_idx = bq_idx // (num_kv_heads * num_q_per_kv)
    q_head_in_batch = bq_idx % (num_kv_heads * num_q_per_kv)
    kv_head_idx = q_head_in_batch // num_q_per_kv
    bkv_idx = batch_idx * num_kv_heads + kv_head_idx

    n_sel = tl.load(NPerHead + bkv_idx).to(tl.int32)
    ctx_len = tl.load(ContextLens + batch_idx).to(tl.int32)

    d_offs = tl.arange(0, HEAD_DIM)
    q_vec = tl.load(Q + bq_idx * stride_qbh + d_offs * stride_qd,
                     mask=d_offs < HEAD_DIM, other=0.0).to(tl.float32)

    m_acc = tl.full([1], float("-inf"), dtype=tl.float32)
    l_acc = tl.zeros([1], dtype=tl.float32)
    o_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    t_offs = tl.arange(0, BLOCK_SIZE)

    for p_rank in range(max_selected):
        page_active = p_rank < n_sel
        logical_page = tl.load(PageIndices + bkv_idx * stride_pibh + p_rank * stride_pis).to(tl.int32)
        physical_block = tl.load(BlockTables + batch_idx * stride_btb + logical_page * stride_btp).to(tl.int32)
        page_start = logical_page * BLOCK_SIZE
        tokens_in_page = tl.minimum(BLOCK_SIZE, ctx_len - page_start)
        valid = (t_offs < tokens_in_page) & page_active

        k_base = K_cache + physical_block * stride_kb + kv_head_idx * stride_kh
        k_ptrs = k_base + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=valid[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
        scores = tl.sum(k_block * q_vec[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_page = tl.max(scores, axis=0)
        m_page_vec = tl.full([1], m_page, dtype=tl.float32)
        m_new = tl.maximum(m_acc, m_page_vec)
        alpha = tl.exp(m_acc - m_new)
        p_vals = tl.exp(scores - m_new)
        p_vals = tl.where(valid, p_vals, 0.0)
        l_page = tl.sum(p_vals, axis=0)
        l_acc = l_acc * alpha + l_page
        o_acc = o_acc * tl.broadcast_to(alpha, [HEAD_DIM])

        v_base = V_cache + physical_block * stride_vb + kv_head_idx * stride_vh
        v_ptrs = v_base + t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=valid[:, None] & (d_offs[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
        weighted_v = tl.sum(p_vals[:, None] * v_block, axis=0)
        o_acc = o_acc + weighted_v
        m_acc = m_new

    # Store UNnormalized output + softmax state (all float32)
    tl.store(O_out + bq_idx * HEAD_DIM + d_offs, o_acc, mask=d_offs < HEAD_DIM)
    tl.store(M_out + bq_idx, tl.sum(m_acc, axis=0))
    tl.store(L_out + bq_idx, tl.sum(l_acc, axis=0))


def _launch_unnorm(q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables,
                   context_lens, B, H_q, D, num_kv_heads, num_q_per_kv,
                   padded_max_selected, block_size):
    """Launch unnorm kernel and return (O_unnorm, M, L) in float32."""
    total_programs = B * H_q

    o_unnorm = torch.zeros(total_programs, D, device=q_flat.device, dtype=torch.float32)
    m_out = torch.full((total_programs,), float('-inf'), device=q_flat.device, dtype=torch.float32)
    l_out = torch.zeros(total_programs, device=q_flat.device, dtype=torch.float32)

    _sparse_decode_unnorm_kernel[(total_programs,)](
        q_flat, k_cache, v_cache,
        o_unnorm, m_out, l_out,
        pi_flat, nph_flat, block_tables, context_lens,
        scale=D ** -0.5,
        num_kv_heads=num_kv_heads,
        num_q_per_kv=num_q_per_kv,
        max_selected=padded_max_selected,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        stride_qbh=q_flat.stride(0), stride_qd=q_flat.stride(1),
        stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        stride_pibh=pi_flat.stride(0), stride_pis=pi_flat.stride(1),
        stride_btb=block_tables.stride(0), stride_btp=block_tables.stride(1),
    )

    return (o_unnorm.reshape(B, H_q, D),
            m_out.reshape(B, H_q),
            l_out.reshape(B, H_q))


def triton_sparse_decode_unnorm(
    q, k_cache, v_cache, block_tables, page_indices, n_per_head,
    context_lens, num_kv_heads, num_q_per_kv, block_size=64,
):
    """Like triton_sparse_decode but returns (O_unnorm, M, L) in float32.

    O_unnorm: [B, H_q, D] — unnormalized weighted-V accumulator
    M: [B, H_q] — running max of attention scores
    L: [B, H_q] — running sum of exp(score - max)

    These allow exact merging with other attention computations via online softmax.
    """
    B, H_q, D = q.shape
    actual_max_selected = page_indices.shape[2]
    padded_max_selected = _next_power_of_2(actual_max_selected)

    if padded_max_selected > actual_max_selected:
        pad_size = padded_max_selected - actual_max_selected
        pi_padded = torch.nn.functional.pad(page_indices, (0, pad_size), value=0)
    else:
        pi_padded = page_indices

    pi_flat = pi_padded.reshape(B * num_kv_heads, padded_max_selected).contiguous()
    nph_flat = n_per_head.reshape(B * num_kv_heads).contiguous()
    q_flat = q.reshape(B * H_q, D).contiguous()

    if padded_max_selected > 64:
        # For large page counts, use tiled kernel + unnorm reduction
        return _launch_tiled_unnorm(
            q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables, context_lens,
            B, H_q, D, num_kv_heads, num_q_per_kv, padded_max_selected, block_size,
        )

    return _launch_unnorm(
        q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables, context_lens,
        B, H_q, D, num_kv_heads, num_q_per_kv, padded_max_selected, block_size,
    )


@triton.jit
def _reduce_tiles_unnorm_kernel(
    M_scratch, L_scratch, O_scratch,
    M_out, L_out, O_out,
    num_tiles: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Reduce per-tile (m, l, o) into merged UNnormalized output."""
    bq_idx = tl.program_id(0)
    d_offs = tl.arange(0, HEAD_DIM)

    global_m = float("-inf")
    for t in range(num_tiles):
        m_t = tl.load(M_scratch + bq_idx * num_tiles + t)
        global_m = tl.maximum(global_m, m_t)

    global_l = 0.0
    global_o = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for t in range(num_tiles):
        m_t = tl.load(M_scratch + bq_idx * num_tiles + t)
        l_t = tl.load(L_scratch + bq_idx * num_tiles + t)
        alpha = tl.exp(m_t - global_m)
        global_l += l_t * alpha
        o_t = tl.load(O_scratch + (bq_idx * num_tiles + t) * HEAD_DIM + d_offs,
                       mask=d_offs < HEAD_DIM, other=0.0)
        global_o += o_t * alpha

    tl.store(O_out + bq_idx * HEAD_DIM + d_offs, global_o, mask=d_offs < HEAD_DIM)
    tl.store(M_out + bq_idx, global_m)
    tl.store(L_out + bq_idx, global_l)


def _launch_tiled_unnorm(q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables,
                         context_lens, B, H_q, D, num_kv_heads, num_q_per_kv,
                         max_selected, block_size):
    """Launch tiled kernel + unnorm reduction returning (O_unnorm, M, L)."""
    device = q_flat.device
    total_bq = B * H_q

    target_tiles = max(4, (512 + total_bq - 1) // total_bq)
    pages_per_tile = max(1, _next_power_of_2(max_selected // target_tiles))
    pages_per_tile = max(4, min(pages_per_tile, max_selected))
    num_tiles = (max_selected + pages_per_tile - 1) // pages_per_tile
    num_tiles_padded = _next_power_of_2(num_tiles)

    m_scratch = torch.empty(total_bq * num_tiles_padded, device=device, dtype=torch.float32)
    l_scratch = torch.empty(total_bq * num_tiles_padded, device=device, dtype=torch.float32)
    o_scratch = torch.empty(total_bq * num_tiles_padded * D, device=device, dtype=torch.float32)

    grid = (total_bq, num_tiles_padded)
    _sparse_decode_tiled_kernel[grid](
        q_flat, k_cache, v_cache,
        m_scratch, l_scratch, o_scratch,
        pi_flat, nph_flat, block_tables, context_lens,
        scale=D ** -0.5,
        num_kv_heads=num_kv_heads,
        num_q_per_kv=num_q_per_kv,
        max_selected=max_selected,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        PAGES_PER_TILE=pages_per_tile,
        stride_qbh=q_flat.stride(0), stride_qd=q_flat.stride(1),
        stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        stride_pibh=pi_flat.stride(0), stride_pis=pi_flat.stride(1),
        stride_btb=block_tables.stride(0), stride_btp=block_tables.stride(1),
        num_tiles=num_tiles_padded,
    )

    o_out = torch.empty(total_bq * D, device=device, dtype=torch.float32)
    m_out = torch.empty(total_bq, device=device, dtype=torch.float32)
    l_out = torch.empty(total_bq, device=device, dtype=torch.float32)

    _reduce_tiles_unnorm_kernel[(total_bq,)](
        m_scratch, l_scratch, o_scratch,
        m_out, l_out, o_out,
        num_tiles=num_tiles_padded,
        HEAD_DIM=D,
    )

    return (o_out.reshape(B, H_q, D),
            m_out.reshape(B, H_q),
            l_out.reshape(B, H_q))


def prepare_reuse_state(page_indices, n_per_head, num_kv_heads, B):
    """Pre-compute flat tensors for reuse steps. Call once at scoring time.

    Returns (pi_flat, nph_flat, padded_max_selected) for triton_sparse_decode_reuse().
    """
    actual_max_selected = page_indices.shape[2]
    padded_max_selected = _next_power_of_2(actual_max_selected)
    if padded_max_selected > actual_max_selected:
        pad_size = padded_max_selected - actual_max_selected
        pi_padded = torch.nn.functional.pad(page_indices, (0, pad_size), value=0)
    else:
        pi_padded = page_indices
    pi_flat = pi_padded.reshape(B * num_kv_heads, padded_max_selected).contiguous()
    nph_flat = n_per_head.reshape(B * num_kv_heads).contiguous()
    return pi_flat, nph_flat, padded_max_selected


def triton_sparse_decode_reuse(
    q, k_cache, v_cache, block_tables, context_lens,
    pi_flat, nph_flat, padded_max_selected,
    num_kv_heads, num_q_per_kv, block_size,
):
    """Fast reuse path: uses pre-computed flat tensors from prepare_reuse_state().

    Skips padding, reshaping, and contiguous() calls that allocate tensors on
    every call. Only q is reshaped (a free view if contiguous).
    """
    B, H_q, D = q.shape
    q_flat = q.reshape(B * H_q, D)  # view if contiguous — no allocation
    total_programs = B * H_q
    use_tiled = padded_max_selected > 64

    if use_tiled:
        return _launch_tiled(
            q_flat, k_cache, v_cache, pi_flat, nph_flat, block_tables, context_lens,
            B, H_q, D, num_kv_heads, num_q_per_kv, padded_max_selected, block_size,
        )
    else:
        output = torch.empty(total_programs, D, device=q.device, dtype=q.dtype)
        _sparse_decode_kernel[(total_programs,)](
            q_flat, k_cache, v_cache, output,
            pi_flat, nph_flat, block_tables, context_lens,
            scale=D ** -0.5,
            num_kv_heads=num_kv_heads,
            num_q_per_kv=num_q_per_kv,
            max_selected=padded_max_selected,
            BLOCK_SIZE=block_size,
            HEAD_DIM=D,
            stride_qbh=q_flat.stride(0), stride_qd=q_flat.stride(1),
            stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
            stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
            stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
            stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
            stride_obh=output.stride(0), stride_od=output.stride(1),
            stride_pibh=pi_flat.stride(0), stride_pis=pi_flat.stride(1),
            stride_btb=block_tables.stride(0), stride_btp=block_tables.stride(1),
        )
        return output.reshape(B, H_q, D)
