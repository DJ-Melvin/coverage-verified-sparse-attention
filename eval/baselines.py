"""Paged-decode reimplementations of the published baselines.

SnapKV (Li et al., 2024), Ada-KV (Feng et al.), H2O (Zhang et al., 2023) and
Quest (Tang et al., ICML 2024), each implemented at page granularity on the
same paged KV cache and Triton sparse-decode kernels as CVSA, so that only
the page-selection policy differs. Method names are parsed by make_backend().
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ─── Quest baseline: min/max key metadata page scoring (Tang et al., ICML 2024) ───

class QuestBackend:
    """Quest: Query-Aware Sparsity via per-page Key min/max metadata.

    Reimplements Quest's page selection algorithm (Tang et al., ICML 2024)
    within our evaluation framework for fair comparison using identical
    sparse attention kernels and KV cache infrastructure.

    For each page, maintains channel-wise Key min/max across tokens.
    Page score = sum_d max(Q_d * K_min_d, Q_d * K_max_d) — an upper bound
    on the max attention score any token in the page could achieve.

    Following the paper, first `dense_layers` layers use full attention.
    """

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size,
                 budget_pct=0.03, reuse_interval=16, dense_layers=2):
        from cvsa.kernels.sparse_decode import (
            triton_sparse_decode, prepare_reuse_state,
            triton_sparse_decode_reuse,
        )

        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.budget_pct = budget_pct
        self.reuse_interval = reuse_interval
        self.dense_layers = dense_layers

        self._triton_sparse_decode = triton_sparse_decode
        self._prepare_reuse_state = prepare_reuse_state
        self._triton_sparse_decode_reuse = triton_sparse_decode_reuse

        # Dense attention for first layers
        self._dense_backend = None

        self._step = 0
        self._reuse_state = {}
        # Per-layer page metadata: K_min, K_max [num_blocks, H_kv, D]
        self._k_min = {}
        self._k_max = {}
        self._metadata_initialized = {}

    def reset(self):
        self._step = 0
        self._reuse_state = {}
        self._k_min = {}
        self._k_max = {}
        self._metadata_initialized = {}

    @property
    def is_scoring_step(self):
        return self._step % self.reuse_interval == 0

    def step_done(self, layer_idx, num_layers):
        if layer_idx == num_layers - 1:
            self._step += 1

    def _build_page_metadata(self, k_cache, block_tables, context_lens,
                             layer_idx):
        """Compute per-page K_min and K_max from k_cache.

        k_cache: [num_blocks, block_size, H_kv, D]
        """
        if layer_idx in self._metadata_initialized:
            return
        num_blocks = k_cache.shape[0]
        H_kv = self.num_kv_heads
        D = self.head_dim
        device = k_cache.device

        # Initialize with extreme values
        k_min = torch.full((num_blocks, H_kv, D), float('inf'),
                           dtype=torch.float32, device=device)
        k_max = torch.full((num_blocks, H_kv, D), float('-inf'),
                           dtype=torch.float32, device=device)

        B = block_tables.shape[0]
        for b in range(B):
            n_tokens = context_lens[b].item()
            n_pages = (n_tokens + self.block_size - 1) // self.block_size
            for p in range(n_pages):
                phys_page = block_tables[b, p].item()
                # How many valid tokens in this page?
                if p < n_pages - 1:
                    valid = self.block_size
                else:
                    rem = n_tokens % self.block_size
                    valid = rem if rem > 0 else self.block_size
                # k_cache[phys_page, :valid, :, :] -> [valid, H_kv, D]
                page_keys = k_cache[phys_page, :valid, :, :].float()
                k_min[phys_page] = torch.min(k_min[phys_page],
                                              page_keys.min(dim=0).values)
                k_max[phys_page] = torch.max(k_max[phys_page],
                                              page_keys.max(dim=0).values)

        self._k_min[layer_idx] = k_min
        self._k_max[layer_idx] = k_max
        self._metadata_initialized[layer_idx] = True

    def _update_page_metadata(self, k_cache, block_tables, context_lens,
                              layer_idx):
        """Incrementally update metadata for the latest token's page."""
        if layer_idx not in self._metadata_initialized:
            self._build_page_metadata(k_cache, block_tables, context_lens,
                                      layer_idx)
            return

        B = block_tables.shape[0]
        k_min = self._k_min[layer_idx]
        k_max = self._k_max[layer_idx]

        for b in range(B):
            n_tokens = context_lens[b].item()
            page_idx = (n_tokens - 1) // self.block_size
            phys_page = block_tables[b, page_idx].item()
            token_in_page = (n_tokens - 1) % self.block_size

            # Read the new token's key vector
            new_key = k_cache[phys_page, token_in_page, :, :].float()  # [H_kv, D]
            k_min[phys_page] = torch.min(k_min[phys_page], new_key)
            k_max[phys_page] = torch.max(k_max[phys_page], new_key)

    def _score_pages_quest(self, q, block_tables, context_lens, layer_idx):
        """Score pages using Quest's min/max upper-bound estimation.

        q: [B, 1, H_q, D] or [B, H_q, D]
        Returns: [B, H_kv, n_pages] page scores
        """
        B = block_tables.shape[0]
        n_pages = block_tables.shape[1]
        device = q.device

        # Reshape q to [B, H_kv, D] — mean over Q heads per KV group (matches triton_score_pages)
        if q.dim() == 4:
            q = q.squeeze(1)  # [B, H_q, D]
        q_per_kv = q.view(B, self.num_kv_heads, self.num_q_per_kv,
                          self.head_dim)
        q_rep = q_per_kv.mean(dim=2)  # [B, H_kv, D]

        k_min = self._k_min[layer_idx]  # [num_blocks, H_kv, D]
        k_max = self._k_max[layer_idx]

        scores = torch.full((B, self.num_kv_heads, n_pages), float('-inf'),
                            dtype=torch.float32, device=device)

        for b in range(B):
            n_tokens = context_lens[b].item()
            n_valid = (n_tokens + self.block_size - 1) // self.block_size

            # Gather physical pages for this batch
            phys_pages = block_tables[b, :n_valid].long()  # [n_valid]
            page_k_min = k_min[phys_pages]  # [n_valid, H_kv, D]
            page_k_max = k_max[phys_pages]  # [n_valid, H_kv, D]

            # q_rep[b]: [H_kv, D]
            q_b = q_rep[b].float().unsqueeze(0)  # [1, H_kv, D]

            # Quest score: sum_d max(Q_d * K_min_d, Q_d * K_max_d)
            prod_min = q_b * page_k_min  # [n_valid, H_kv, D]
            prod_max = q_b * page_k_max  # [n_valid, H_kv, D]
            upper_bound = torch.max(prod_min, prod_max)  # [n_valid, H_kv, D]
            page_score = upper_bound.sum(dim=-1)  # [n_valid, H_kv]
            page_score = page_score.transpose(0, 1)  # [H_kv, n_valid]

            scores[b, :, :n_valid] = page_score

        return scores

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        # First `dense_layers` layers use full attention (following Quest paper)
        if layer_idx < self.dense_layers:
            if self._dense_backend is None:
                from cvsa.runtime.dense import DenseAttentionBackend
                self._dense_backend = DenseAttentionBackend(
                    self.num_kv_heads, self.num_q_per_kv * self.num_kv_heads,
                    self.head_dim, self.block_size,
                )
            return self._dense_backend(
                q, k_cache, v_cache, block_tables, context_lens, layer_idx)

        # Update page metadata incrementally
        self._update_page_metadata(k_cache, block_tables, context_lens,
                                   layer_idx)

        B = q.shape[0]
        n_pages = block_tables.shape[1]
        budget = max(3, int(n_pages * self.budget_pct))

        if self.is_scoring_step:
            # Quest page scoring
            page_scores = self._score_pages_quest(
                q, block_tables, context_lens, layer_idx)

            topk_vals, topk_indices = torch.topk(
                page_scores, k=budget, dim=-1, largest=True, sorted=False)
            page_indices = topk_indices.to(torch.int32)

            n_per_head = torch.full((B, self.num_kv_heads), budget,
                                     dtype=torch.int32, device=q.device)
            n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
            n_per_head = torch.min(
                n_per_head,
                n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32),
            )

            output = self._triton_sparse_decode(
                q, k_cache, v_cache, block_tables, page_indices, n_per_head,
                context_lens, self.num_kv_heads, self.num_q_per_kv,
                self.block_size,
            )

            pi_flat, nph_flat, padded = self._prepare_reuse_state(
                page_indices, n_per_head, self.num_kv_heads, B)
            self._reuse_state[layer_idx] = (pi_flat, nph_flat, padded)
            return output
        else:
            if layer_idx not in self._reuse_state:
                self._step -= 1
                return self(q, k_cache, v_cache, block_tables, context_lens,
                            layer_idx)
            pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
            return self._triton_sparse_decode_reuse(
                q, k_cache, v_cache, block_tables, context_lens,
                pi_flat, nph_flat, padded,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )


# ─── Ada-KV backend (NeurIPS 2025, per-head adaptive budget via attention entropy) ───

class AdaKVBackend:
    """Ada-KV (NeurIPS 2025): Per-head adaptive budget via attention concentration.

    Faithful implementation of Feng et al.: during calibration (first K decode
    steps), compute DENSE attention weights and measure per-head concentration.
    Heads with diffuse attention (need many tokens for 80% mass) get more budget;
    concentrated heads get less. Budget is fixed after calibration.

    This uses ACTUAL attention weights (not page-score proxies), matching Ada-KV's
    theoretical loss bound approach.
    """

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size,
                 budget_pct=0.03, reuse_interval=16, cal_steps=8, exponent=1.0):
        from cvsa.kernels.sparse_decode import (
            triton_sparse_decode, prepare_reuse_state,
            triton_sparse_decode_reuse,
        )
        from cvsa.kernels.page_scores import triton_score_pages

        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.budget_pct = budget_pct
        self.reuse_interval = reuse_interval
        self.cal_steps = cal_steps
        self.exponent = exponent

        self._triton_sparse_decode = triton_sparse_decode
        self._prepare_reuse_state = prepare_reuse_state
        self._triton_sparse_decode_reuse = triton_sparse_decode_reuse
        self._triton_score_pages = triton_score_pages

        self._step = 0
        self._reuse_state = {}
        self._calibrating = True
        self._conc_samples = {}  # layer_idx -> list of [H_kv] concentration
        self._head_budgets = {}  # layer_idx -> [H_kv] per-head budgets

    def reset(self):
        self._step = 0
        self._reuse_state = {}
        self._calibrating = True
        self._conc_samples = {}
        self._head_budgets = {}

    @property
    def is_scoring_step(self):
        return self._step % self.reuse_interval == 0

    def step_done(self, layer_idx, num_layers):
        if layer_idx == num_layers - 1:
            self._step += 1
            if self._step == self.cal_steps and self._calibrating:
                self._finalize_calibration()

    def _finalize_calibration(self):
        """Compute per-head budgets from collected attention concentration."""
        for layer_idx, conc_list in self._conc_samples.items():
            conc = torch.stack(conc_list, dim=0).mean(dim=0)  # [H_kv]
            mean_conc = conc.mean().clamp(min=1e-8)
            # Diffuse heads (high needed_pages) get MORE budget
            multipliers = (conc / mean_conc) ** self.exponent
            multipliers = multipliers / multipliers.mean()
            multipliers = multipliers.clamp(0.5, 2.5)
            multipliers = multipliers / multipliers.mean()
            self._head_budgets[layer_idx] = multipliers
        self._conc_samples = {}
        self._calibrating = False

    def _measure_concentration(self, q, k_cache, v_cache, block_tables,
                                context_lens, layer_idx):
        """Compute per-KV-head attention concentration via dense attention."""
        B = q.shape[0]
        device = q.device
        scale = self.head_dim ** -0.5
        # Materialize K from paged cache
        ctx_len = context_lens[0].item()
        n_blocks = (ctx_len + self.block_size - 1) // self.block_size
        bt = block_tables[0, :n_blocks]
        k_parts = []
        for bi in range(n_blocks):
            blk = bt[bi].item()
            nt = ctx_len - bi * self.block_size if bi == n_blocks - 1 else self.block_size
            k_parts.append(k_cache[blk, :nt])  # [nt, H_kv, D]
        K = torch.cat(k_parts, dim=0)  # [ctx_len, H_kv, D]
        # Compute per-KV-head attention weights
        # Average Q across GQA group for representative query
        q_kv = q[0].reshape(self.num_kv_heads, self.num_q_per_kv, self.head_dim)
        q_rep = q_kv.mean(dim=1)  # [H_kv, D]
        scores = torch.einsum('hd,thd->ht', q_rep.float(), K.float()) * scale
        weights = torch.softmax(scores, dim=-1)  # [H_kv, ctx_len]
        # Concentration: number of tokens needed to reach 80% mass (per head)
        sorted_w, _ = weights.sort(dim=-1, descending=True)
        cumsum = sorted_w.cumsum(dim=-1)
        # needed_tokens[h] = first index where cumsum >= 0.80
        needed = (cumsum >= 0.80).float().argmax(dim=-1).float() + 1  # [H_kv]
        return needed.cpu()

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        B = q.shape[0]
        n_pages = block_tables.shape[1]
        base_budget = max(3, int(n_pages * self.budget_pct))
        device = q.device

        # During calibration: measure concentration AND do sparse attention
        if self._calibrating:
            conc = self._measure_concentration(
                q, k_cache, v_cache, block_tables, context_lens, layer_idx)
            if layer_idx not in self._conc_samples:
                self._conc_samples[layer_idx] = []
            self._conc_samples[layer_idx].append(conc)

        # Determine per-head budgets
        if layer_idx in self._head_budgets:
            mults = self._head_budgets[layer_idx].to(device)
            budgets = (mults * base_budget).round().clamp(min=3).int()
            max_budget = min(budgets.max().item(), n_pages)
        else:
            budgets = None
            max_budget = base_budget

        if self.is_scoring_step:
            page_scores = self._triton_score_pages(
                q, k_cache, block_tables, context_lens,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )
            actual_k = min(max_budget, page_scores.shape[-1])
            _, topk_indices = torch.topk(
                page_scores, k=actual_k, dim=-1, largest=True, sorted=True)
            page_indices = topk_indices.to(torch.int32)

            if budgets is not None:
                n_per_head = budgets.unsqueeze(0).expand(B, -1).contiguous()
                n_per_head = n_per_head.clamp(max=actual_k).to(torch.int32)
            else:
                n_per_head = torch.full((B, self.num_kv_heads), actual_k,
                                         dtype=torch.int32, device=device)
            n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
            n_per_head = torch.min(
                n_per_head,
                n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32),
            )

            output = self._triton_sparse_decode(
                q, k_cache, v_cache, block_tables, page_indices, n_per_head,
                context_lens, self.num_kv_heads, self.num_q_per_kv,
                self.block_size,
            )

            pi_flat, nph_flat, padded = self._prepare_reuse_state(
                page_indices, n_per_head, self.num_kv_heads, B)
            self._reuse_state[layer_idx] = (pi_flat, nph_flat, padded)
            return output
        else:
            if layer_idx not in self._reuse_state:
                self._step -= 1
                return self(q, k_cache, v_cache, block_tables, context_lens,
                            layer_idx)
            pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
            return self._triton_sparse_decode_reuse(
                q, k_cache, v_cache, block_tables, context_lens,
                pi_flat, nph_flat, padded,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )


# ─── SnapKV-style backend (score once at prefill, never re-score) ───

class SnapKVBackend:
    """SnapKV-style sparse attention: one-shot page selection, no adaptation.

    Scores pages ONCE, before the first decode step, then reuses that
    selection for the entire decode. This emulates SnapKV's prefill-based
    KV compression at page granularity.

    SnapKV (Li et al., 2024) selects important KV positions during prefill
    using an observation window. set_prefill_scores() implements this from
    the last `obs_window` prompt queries; if no prefill scores were set for
    a layer, the first decode token's Q is used instead.

    Same budget and same sparse-decode kernel as the other page-level
    methods; only the selection policy differs.
    """

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size,
                 budget_pct=0.03, pool_kernel=1, obs_window=32,
                 obs_agg='mean', page_agg='sum'):
        from cvsa.kernels.sparse_decode import (
            triton_sparse_decode, prepare_reuse_state,
            triton_sparse_decode_reuse,
        )
        from cvsa.kernels.page_scores import triton_score_pages

        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.block_size = block_size
        self.budget_pct = budget_pct
        # Paper-faithful SnapKV is implemented in set_prefill_scores: last
        # `obs_window` prompt Qs -> per-token softmax importance -> token-level
        # avg_pool1d(k=7) -> force-keep obs window -> aggregate to pages.
        # `pool_kernel` below is a separate PAGE-level max-pool applied to
        # scores at decode time. Default 1 (disabled) because Fix A
        # showed k=7 at page granularity over-smooths 16x
        # (7 pages * 16 tok/page = 112-token span) and destroys retrieval.
        self.pool_kernel = pool_kernel
        self.obs_window = obs_window
        # Aggregation knobs (ablation). Fix B default (mean+sum)
        # underperformed at page-level granularity; max variants preserve
        # locality peaks that the diffuse mean+sum pipeline washes out.
        assert obs_agg in ('mean', 'max'), obs_agg
        assert page_agg in ('sum', 'max'), page_agg
        self.obs_agg = obs_agg
        self.page_agg = page_agg

        self._triton_sparse_decode = triton_sparse_decode
        self._prepare_reuse_state = prepare_reuse_state
        self._triton_sparse_decode_reuse = triton_sparse_decode_reuse
        self._triton_score_pages = triton_score_pages

        self._step = 0
        self._reuse_state = {}
        self._scored = False  # True after first scoring
        self._prefill_page_scores = {}  # layer_idx -> [1, H_kv, n_prefill_pages]

    def reset(self):
        self._step = 0
        self._reuse_state = {}
        self._scored = False
        # IMPORTANT: do NOT clear _prefill_page_scores here. The harness calls
        # set_prefill_scores(...) for every layer BEFORE reset(), and clearing
        # here would erase the prefill signal right before the decode loop
        # starts. The dict is naturally overwritten on the next generate().

    def set_prefill_scores(self, layer_idx, k_seq, q_window, block_size):
        """SnapKV paper-faithful prefill scoring (Fix B).

        The eval harness auto-invokes this once per layer before decode if the
        backend exposes `obs_window` + `set_prefill_scores`. It passes the
        post-RoPE Q vectors from the last W prompt tokens (GQA-averaged per
        KV head) and the full prefill K cache for this layer.

        Args:
            layer_idx: transformer layer index.
            k_seq:     [seq_len, H_kv, D] post-RoPE K from prefill.
            q_window:  [W, H_kv, D]       post-RoPE obs-window Q.
            block_size: page size in tokens (16).
        """
        S, H_kv, D = k_seq.shape
        W = q_window.shape[0]
        if W == 0:
            return  # no-op sentinel
        scale = D ** -0.5

        # [H_kv, W, D] @ [H_kv, D, S] -> [H_kv, W, S]
        q_t = q_window.float().permute(1, 0, 2)
        k_t = k_seq.float().permute(1, 2, 0)
        scores = torch.bmm(q_t, k_t) * scale
        weights = torch.softmax(scores, dim=-1)            # [H_kv, W, S]

        # Aggregate across obs queries -> per-token importance [H_kv, S].
        # obs_agg='mean' is paper-faithful; 'max' preserves per-query peaks
        # that mean would dilute across diverse prompt positions.
        if self.obs_agg == 'max':
            importance = weights.max(dim=1).values
        else:
            importance = weights.mean(dim=1)

        # Token-level avg_pool1d(k=7) — this is what the SnapKV paper does.
        if S > 7:
            importance = torch.nn.functional.avg_pool1d(
                importance.unsqueeze(0), kernel_size=7, stride=1, padding=3,
            ).squeeze(0)

        # Obs-window tokens always kept.
        importance[:, -W:] = importance.max() + 1.0

        # Aggregate tokens -> pages. page_agg='sum' is the Fix B default;
        # 'max' picks each page's strongest token as the page representative,
        # avoiding dilution when only a few tokens per page carry the signal.
        n_pages = (S + block_size - 1) // block_size
        pad_len = n_pages * block_size - S
        if pad_len > 0:
            pad_val = float("-inf") if self.page_agg == 'max' else 0.0
            importance = torch.nn.functional.pad(importance, (0, pad_len), value=pad_val)
        viewed = importance.view(H_kv, n_pages, block_size)
        if self.page_agg == 'max':
            page_scores = viewed.max(dim=-1).values  # [H_kv, n_pages]
        else:
            page_scores = viewed.sum(dim=-1)          # [H_kv, n_pages]

        self._prefill_page_scores[layer_idx] = page_scores.unsqueeze(0).contiguous().float()

    @property
    def is_scoring_step(self):
        # Only score on the very first step
        return not self._scored

    def step_done(self, layer_idx, num_layers):
        if layer_idx == num_layers - 1:
            if not self._scored:
                self._scored = True  # Lock after first full pass
            self._step += 1

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        B = q.shape[0]
        n_pages = block_tables.shape[1]
        budget = max(3, int(n_pages * self.budget_pct))

        if self.is_scoring_step:
            # Fix B path: use paper-faithful prefill scores when available.
            # Fallback is the legacy decode-Q top-k scoring (used only if
            # set_prefill_scores was never called for this layer, e.g. when
            # obs_window=0 or the harness is running without the hook).
            if layer_idx in self._prefill_page_scores:
                prefill_scores = self._prefill_page_scores[layer_idx]  # [1, H_kv, n_prefill]
                n_pages = block_tables.shape[1]
                n_prefill = prefill_scores.shape[-1]
                if n_pages >= n_prefill:
                    page_scores = prefill_scores.new_full(
                        (1, prefill_scores.shape[1], n_pages), float("-inf"),
                    )
                    page_scores[:, :, :n_prefill] = prefill_scores
                else:
                    page_scores = prefill_scores[:, :, :n_pages].clone()

                # Force-include pages appended since prefill — they hold fresh
                # decode tokens and must be attended to avoid causal gaps.
                n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
                max_valid = int(n_valid.max().item())
                if max_valid > n_prefill:
                    page_scores[:, :, n_prefill:max_valid] = float("inf")

                page_scores = page_scores.expand(B, -1, -1).contiguous()
            else:
                # Legacy: score with the current (first) decode-step Q.
                page_scores = self._triton_score_pages(
                    q, k_cache, block_tables, context_lens,
                    self.num_kv_heads, self.num_q_per_kv, self.block_size,
                )  # [B, H_kv, n_pages]

            # Optional PAGE-level max-pool (ablation knob). Default
            # pool_kernel=1 disables this because page-level k=7 was Fix A
            # and regressed Llama-3.1-8B 16-task mean by 12.7pt.
            if self.pool_kernel > 1 and page_scores.shape[-1] >= self.pool_kernel:
                B_, H_, N_ = page_scores.shape
                pad = self.pool_kernel // 2
                page_scores = torch.nn.functional.max_pool1d(
                    page_scores.reshape(B_ * H_, 1, N_),
                    kernel_size=self.pool_kernel,
                    stride=1,
                    padding=pad,
                ).reshape(B_, H_, N_)

            topk_vals, topk_indices = torch.topk(
                page_scores, k=budget, dim=-1, largest=True, sorted=False)
            page_indices = topk_indices.to(torch.int32)

            n_per_head = torch.full((B, self.num_kv_heads), budget,
                                     dtype=torch.int32, device=q.device)
            n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
            n_per_head = torch.min(
                n_per_head,
                n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32),
            )

            output = self._triton_sparse_decode(
                q, k_cache, v_cache, block_tables, page_indices, n_per_head,
                context_lens, self.num_kv_heads, self.num_q_per_kv,
                self.block_size,
            )

            # Save permanently — will never re-score
            pi_flat, nph_flat, padded = self._prepare_reuse_state(
                page_indices, n_per_head, self.num_kv_heads, B)
            self._reuse_state[layer_idx] = (pi_flat, nph_flat, padded)

            return output
        else:
            if layer_idx not in self._reuse_state:
                # Force scoring for this layer
                old_scored = self._scored
                self._scored = False
                result = self(q, k_cache, v_cache, block_tables, context_lens,
                              layer_idx)
                self._scored = old_scored
                return result
            pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
            return self._triton_sparse_decode_reuse(
                q, k_cache, v_cache, block_tables, context_lens,
                pi_flat, nph_flat, padded,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )


# ─── H2O sparse attention backend ───

class H2OAttentionBackend:
    """H2O: Heavy Hitter Oracle (recent window + heavy hitters).

    Splits budget into recent pages (sliding window) and heavy hitters
    (highest Q·K scores from non-recent pages). Each head gets the same
    total budget but different heavy hitter selections.
    """

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size,
                 budget_pct=0.03, reuse_interval=16, recent_ratio=0.5):
        from cvsa.kernels.sparse_decode import (
            triton_sparse_decode, prepare_reuse_state,
            triton_sparse_decode_reuse,
        )
        from cvsa.kernels.page_scores import triton_score_pages

        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.block_size = block_size
        self.budget_pct = budget_pct
        self.reuse_interval = reuse_interval
        self.recent_ratio = recent_ratio

        self._triton_sparse_decode = triton_sparse_decode
        self._prepare_reuse_state = prepare_reuse_state
        self._triton_sparse_decode_reuse = triton_sparse_decode_reuse
        self._triton_score_pages = triton_score_pages

        self._step = 0
        self._reuse_state = {}

    def reset(self):
        self._step = 0
        self._reuse_state = {}

    @property
    def is_scoring_step(self):
        return self._step % self.reuse_interval == 0

    def step_done(self, layer_idx, num_layers):
        if layer_idx == num_layers - 1:
            self._step += 1

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        B = q.shape[0]
        n_pages = block_tables.shape[1]
        budget = max(3, int(n_pages * self.budget_pct))

        if self.is_scoring_step:
            n_recent = max(1, int(budget * self.recent_ratio))
            n_heavy = budget - n_recent

            page_scores = self._triton_score_pages(
                q, k_cache, block_tables, context_lens,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )

            n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1

            page_indices = torch.zeros(B, self.num_kv_heads, budget,
                                       device=q.device, dtype=torch.int32)
            n_per_head = torch.full((B, self.num_kv_heads), budget,
                                     dtype=torch.int32, device=q.device)

            for b in range(B):
                nv = int(n_valid[b].item())
                actual_recent = min(n_recent, nv)
                actual_heavy = min(n_heavy, max(0, nv - actual_recent))
                actual_total = actual_recent + actual_heavy

                recent_start = max(0, nv - actual_recent)
                recent_pages = torch.arange(recent_start, nv,
                                            device=q.device, dtype=torch.int32)

                for h in range(self.num_kv_heads):
                    scores_h = page_scores[b, h].clone()
                    scores_h[recent_start:nv] = float("-inf")
                    scores_h[nv:] = float("-inf")

                    if actual_heavy > 0:
                        _, heavy_idx = scores_h.topk(actual_heavy)
                        selected = torch.cat([heavy_idx.to(torch.int32), recent_pages])
                    else:
                        selected = recent_pages

                    n = min(len(selected), budget)
                    page_indices[b, h, :n] = selected[:n]
                    n_per_head[b, h] = n

            output = self._triton_sparse_decode(
                q, k_cache, v_cache, block_tables, page_indices, n_per_head,
                context_lens, self.num_kv_heads, self.num_q_per_kv,
                self.block_size,
            )

            pi_flat, nph_flat, padded = self._prepare_reuse_state(
                page_indices, n_per_head, self.num_kv_heads, B)
            self._reuse_state[layer_idx] = (pi_flat, nph_flat, padded)
            return output
        else:
            if layer_idx not in self._reuse_state:
                self._step -= 1
                return self(q, k_cache, v_cache, block_tables, context_lens,
                            layer_idx)
            pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
            return self._triton_sparse_decode_reuse(
                q, k_cache, v_cache, block_tables, context_lens,
                pi_flat, nph_flat, padded,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )


# ─── Method-name dispatch ───

def make_backend(method_name, config, block_size, num_layers):
    """Create a baseline attention backend from a method name.

    B is the per-head page budget in percent of the context's pages
    (at least 3 pages).

      snapkv_{B}[_k{K}][_w{W}][_obsmax][_pagemax]
          SnapKV: pages scored once from the last W prompt queries
          (default W=32); _k{K} adds a page-level max-pool of width K
          (default 1 = off); _obsmax / _pagemax switch the query / page
          aggregation from mean / sum to max.
      adakv_{B}[_e{E}][_c{C}]
          Ada-KV: per-head budgets from attention concentration measured
          over the first C decode steps (default 8), exponent E (default 1).
      h2o_{B}
          H2O: recent pages plus heavy-hitter pages, re-scored every 16 steps.
      quest_{B}[_r_{R}]
          Quest: min/max key metadata page scoring, re-scored every R steps
          (default 16); the first two layers use dense attention.

    Any other name raises ValueError.
    """
    if method_name.startswith("snapkv_"):
        # snapkv_10                  -> Fix B default: mean+sum (paper-faithful)
        # snapkv_10_obsmax           -> max-over-W obs queries + sum-over-page
        # snapkv_10_pagemax          -> mean-over-W + max-over-page
        # snapkv_10_obsmax_pagemax   -> max + max
        # snapkv_10_k3               -> ablation: page-level max-pool kernel=3
        # snapkv_10_w16              -> ablation: shrink obs window to 16
        # snapkv_10_w1_k1            -> W=1 reproducibility control (not true
        #                               pre-Fix-A; routes through set_prefill_scores)
        parts = method_name.split("_")
        pct = int(parts[1]) / 100.0
        pool_kernel = 1
        obs_window = 32
        obs_agg = 'mean'
        page_agg = 'sum'
        for p in parts[2:]:
            if p.startswith("k") and p[1:].isdigit():
                pool_kernel = int(p[1:])
            elif p.startswith("w") and p[1:].isdigit():
                obs_window = int(p[1:])
            elif p == 'obsmax':
                obs_agg = 'max'
            elif p == 'pagemax':
                page_agg = 'max'
        return SnapKVBackend(
            config.num_kv_heads, config.num_q_heads, config.head_dim,
            block_size, budget_pct=pct,
            pool_kernel=pool_kernel, obs_window=obs_window,
            obs_agg=obs_agg, page_agg=page_agg,
        )
    elif method_name.startswith("adakv_"):
        # adakv_3 = Ada-KV at 3% avg budget (attention-concentration-based)
        # adakv_3_e2 = Ada-KV with exponent=2.0
        parts = method_name.split("_")
        pct = int(parts[1]) / 100.0
        exponent = 1.0
        cal_steps = 8
        for p in parts[2:]:
            if p.startswith("e") and p[1:].isdigit():
                exponent = float(p[1:])
            elif p.startswith("c") and p[1:].isdigit():
                cal_steps = int(p[1:])
        return AdaKVBackend(
            config.num_kv_heads, config.num_q_heads, config.head_dim,
            block_size, budget_pct=pct, reuse_interval=16,
            cal_steps=cal_steps, exponent=exponent,
        )
    elif method_name.startswith("h2o_"):
        pct = int(method_name.split("_")[1]) / 100.0
        return H2OAttentionBackend(
            config.num_kv_heads, config.num_q_heads, config.head_dim,
            block_size, budget_pct=pct, reuse_interval=16,
        )
    elif method_name.startswith("quest_"):
        # quest_3 = 3% budget reuse/16, quest_3_r1 = 3% budget reuse/1
        parts = method_name.split("_")
        pct = int(parts[1]) / 100.0
        reuse = int(parts[3]) if len(parts) > 3 and parts[2] == "r" else 16
        return QuestBackend(
            config.num_kv_heads, config.num_q_heads, config.head_dim,
            block_size, budget_pct=pct, reuse_interval=reuse,
            dense_layers=2,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")
