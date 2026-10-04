"""Dense paged-decode attention backend.

FlashInfer batch decode over the paged KV cache, with a Triton fallback
(the sparse-decode kernel with every page selected) for GQA ratios that
FlashInfer does not support (e.g. 7:1 in Qwen2.5).
"""

import torch

try:
    import flashinfer
except ImportError:
    flashinfer = None


class DenseAttentionBackend:
    """Dense attention via FlashInfer for E2E decode.

    Falls back to Triton kernel (selecting all pages) for GQA ratios
    not supported by FlashInfer (e.g., 7:1 in Qwen2.5).
    """

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size):
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self._workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                                       device='cuda')
        self._use_triton_fallback = False
        self._triton_dense_state = {}  # cached all-page selection per n_pages

    def reset(self):
        self._triton_dense_state = {}

    def step_done(self, layer_idx, num_layers):
        pass

    def _triton_dense(self, q, k_cache, v_cache, block_tables, context_lens):
        """Dense attention via Triton kernel: select ALL pages."""
        from cvsa.kernels.sparse_decode import (
            prepare_reuse_state, triton_sparse_decode_reuse,
        )
        B = q.shape[0]
        n_pages = block_tables.shape[1]
        n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1

        # Cache the all-pages selection state (only rebuild when n_pages changes)
        cur_max = n_valid.max().item()
        cached = self._triton_dense_state.get(n_pages)
        if cached is None or cached[3] < cur_max:
            all_indices = torch.arange(
                n_pages, device=q.device, dtype=torch.int32
            ).unsqueeze(0).unsqueeze(0).expand(
                B, self.num_kv_heads, -1).contiguous()
            all_nph = n_valid.unsqueeze(1).expand(
                B, self.num_kv_heads).to(torch.int32).contiguous()
            pi_flat, nph_flat, padded = prepare_reuse_state(
                all_indices, all_nph, self.num_kv_heads, B)
            self._triton_dense_state[n_pages] = (
                pi_flat, nph_flat, padded, cur_max)

        pi_flat, nph_flat, padded, _ = self._triton_dense_state[n_pages]
        return triton_sparse_decode_reuse(
            q, k_cache, v_cache, block_tables, context_lens,
            pi_flat, nph_flat, padded,
            self.num_kv_heads, self.num_q_per_kv, self.block_size)

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        """q: [B, H_q, D] -> [B, H_q, D]"""
        B, H_q, D = q.shape

        # Once we know FlashInfer doesn't work, skip straight to Triton
        if self._use_triton_fallback:
            return self._triton_dense(
                q, k_cache, v_cache, block_tables, context_lens)

        # Try FlashInfer first (fastest for supported GQA ratios)
        try:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._workspace, "NHD")

            indptr = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
            all_phys = []
            for b in range(B):
                n_valid = (context_lens[b].item() - 1) // self.block_size + 1
                all_phys.append(block_tables[b, :n_valid].int())
                indptr[b + 1] = indptr[b] + n_valid
            physical = torch.cat(all_phys)

            last_page_len = torch.zeros(B, dtype=torch.int32, device='cuda')
            for b in range(B):
                r = context_lens[b].item() % self.block_size
                last_page_len[b] = r if r > 0 else self.block_size

            q_fi = q.to(k_cache.dtype)
            wrapper.begin_forward(
                indptr, physical, last_page_len,
                num_qo_heads=H_q, num_kv_heads=self.num_kv_heads,
                head_dim=D, page_size=self.block_size, q_data_type=q_fi.dtype,
            )
            output = wrapper.run(q_fi, (k_cache, v_cache))
            wrapper.end_forward()
            return output
        except (RuntimeError, AttributeError, TypeError):
            # FlashInfer doesn't support this GQA ratio — switch to Triton
            self._use_triton_fallback = True
            return self._triton_dense(
                q, k_cache, v_cache, block_tables, context_lens)
