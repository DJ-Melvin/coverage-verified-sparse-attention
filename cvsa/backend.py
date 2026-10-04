"""Coverage-Verified Sparse Attention (CVSA) backend.

Sparse attention as draft, coverage as acceptance criterion,
and recovery on low-coverage heads.

Two scheduling modes:
  Fixed-interval (default): Score every R steps, reuse selection in between.
  Lazy (lazy=True): Check coverage every M steps using cached selection.
    Re-score only when coverage drops below tau.

Usage:
    # Fixed-interval (R=16)
    backend = CVSABackend(
        num_kv_heads=8, num_q_heads=32, head_dim=128, block_size=16,
        num_layers=32, budget_pct=0.10, coverage_threshold=0.90)

    # Lazy scoring (coverage-driven re-scoring)
    backend = CVSABackend(
        ..., lazy=True, coverage_threshold=0.90)

    # Paper configuration (c4, 20% approximate recovery, layer 0 dense)
    backend = CVSABackend(
        ..., budget_pct=0.05, lazy=True, check_interval=4,
        approx_recovery_pct=0.20, dense_layers=[0])
"""

from typing import Dict, List, Optional, Set, Tuple

import torch

from cvsa.kernels.sparse_decode import (
    triton_sparse_decode_unnorm,
    prepare_reuse_state,
    triton_sparse_decode_reuse,
)
from cvsa.kernels.page_scores import triton_score_pages
from cvsa.runtime.dense import DenseAttentionBackend
from cvsa.kernels.coverage import (
    triton_compute_complement,
    triton_check_coverage,
    triton_lse_merge,
)


class CVSABackend:
    """CVSA backend: draft with sparse attention, verify with coverage.

    Scoring step (every R steps):
      1. Score all pages, select top-k per head (DRAFT)
      2. Compute unnormalized sparse attention → (O_d, M_d, L_d)
      3. Check coverage per head: accept if coverage ≥ τ
      4. For rejected heads: compute complement attention, LSE merge → exact dense
    Reuse step (R-1 out of R steps):
      - Reuse cached page selection, zero scoring overhead
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        block_size: int,
        num_layers: int = 32,
        budget_pct: float = 0.10,
        reuse_interval: int = 16,
        coverage_threshold: float = 0.90,
        dense_layers: Optional[List[int]] = None,
        lazy: bool = False,
        check_interval: Optional[int] = None,
        coverage_interval: Optional[int] = None,
        approx_recovery_pct: Optional[float] = None,
        adaptive_selection: bool = False,
        adaptive_floor_ratio: float = 0.5,
        random_reject_prob: Optional[float] = None,
        static_budget_table: Optional[Dict[int, List[float]]] = None,
        selector: str = "exact",
        top_p: Optional[float] = None,
        light_normalizer: bool = False,
    ):
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.budget_pct = budget_pct
        self.reuse_interval = reuse_interval
        self.tau = coverage_threshold

        # Per-layer dense policy: these layers skip CVSA, use FlashInfer dense
        self._dense_layers: Set[int] = set(dense_layers or [])
        self._dense_backend = None
        if self._dense_layers:
            self._dense_backend = DenseAttentionBackend(
                num_kv_heads, num_q_heads, head_dim, block_size)

        # Approximate recovery: on reject, expand to larger sparse set instead
        # of full complement. None = exact recovery (default), 0.20 = expand to 20%.
        self.approx_recovery_pct = approx_recovery_pct

        # Lazy scoring: check coverage periodically, re-score on demand
        # lazy=True, check_interval=None: check every step (original lazy)
        # lazy=True, check_interval=M: check every M steps (hybrid)
        # coverage_interval (optional): coverage-gate cadence in steps.
        #   Defaults to check_interval, so scoring and coverage are jointly
        #   amortized (the deployed CVSA-c4/c8 behavior).
        #   Set < check_interval to decouple: e.g. check_interval=4,
        #   coverage_interval=1 runs page scoring every 4 steps while the
        #   coverage gate evaluates the cached selection at every step
        #   (paper-described variant). The cached page_scores from the most
        #   recent scoring step are reused as the cached all-context proxy
        #   between refreshes; see _coverage_only_step.
        self.lazy = lazy
        self.check_interval = check_interval or 1  # default: every step
        self.coverage_interval = coverage_interval or self.check_interval

        # Adaptive selection: AdaKV-style global competition across heads.
        # Instead of uniform top-k per head, heads compete for H*budget total
        # page slots. Heads with higher scores win more slots.
        self.adaptive_selection = adaptive_selection
        self.adaptive_floor_ratio = adaptive_floor_ratio

        # Random-reject override: if set, overrides coverage-based accept_mask
        # with a Bernoulli(random_reject_prob) per-layer decision. Used for
        # counterfactual "random-20% AR" baseline proving coverage signal
        # is targeted, not arbitrary. `random_reject_prob` is the per-(layer, step)
        # probability of triggering AR (so setting to the same rate as observed
        # AR rate gives matched compute).
        self.random_reject_prob = random_reject_prob

        # Static per-(layer, head) budget policy (DuoAttention-style control):
        # layer_idx -> [H_kv] budget fractions derived from OFFLINE
        # coverage statistics (eval/diagnostics_head.py atlas). When set, the
        # scoring step uses these fixed per-head budgets instead of the uniform
        # budget_pct, and there is no query-dependent reallocation. Combine with
        # coverage_threshold=0.0 for a no-verification static policy at matched
        # effective compute (see eval/build_static_budget_table.py).
        self.static_budget_table: Optional[Dict[int, torch.Tensor]] = None
        if static_budget_table is not None:
            self.static_budget_table = {
                int(layer): torch.as_tensor(fracs, dtype=torch.float32)
                for layer, fracs in static_budget_table.items()
            }

        # Page selector: "exact" ranks pages by triton_score_pages (default);
        # "quest" ranks by Quest-style per-page key min/max upper bounds
        # (Tang et al., ICML 2024) — a lightweight-indexer selector used to
        # validate selector-agnosticism. The coverage gate always verifies the
        # REALIZED exact mass of whatever selection the selector proposed.
        if selector not in ("exact", "quest"):
            raise ValueError(f"Unknown selector: {selector!r}")
        self.selector = selector

        # Fully lightweight mode: the coverage
        # gate's all-context normalizer comes from the SAME Quest min/max
        # sketch scores that drive selection, so no full-key scan runs
        # anywhere in the pipeline. The gate then checks the selection
        # against what the indexer can see: it still catches flat-score /
        # insufficient-budget steps and routes recovery there, but indexer
        # estimation bias itself is no longer observable. The delta to the
        # exact-normalizer selq configuration measures the value of the
        # exact normalizer.
        if light_normalizer and selector != "quest":
            raise ValueError(
                "light_normalizer=True requires selector='quest' "
                "(the sketch supplies both selection and normalizer)")
        self.light_normalizer = light_normalizer

        # Twilight-style top-p budgeting ("twip"): per-head budgets
        # chosen as the smallest page set whose cumulative softmax mass over
        # the SELECTOR's (estimated) page scores reaches top_p. With
        # coverage_threshold=0 this reproduces the pre-hoc adaptive-budget
        # principle (Twilight, top-p attention) inside our pipeline; with the
        # gate on, it measures verification's value ON TOP of that principle.
        # No budget cap (Twilight has none); budgets clamp only to n_valid.
        if top_p is not None and not (0.0 < top_p < 1.0):
            raise ValueError(f"top_p must be in (0, 1), got {top_p}")
        self.top_p = top_p

        # Tracing: expose per-layer accept/reject + expanded selection for
        # page trace collection (collect_page_traces.py).  Cleared per layer
        # at the start of each __call__ so TracingWrapper always sees the
        # state from the *current* call only.
        self._last_accept_mask: Dict[int, Optional[torch.Tensor]] = {}
        self._last_expanded_selection: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Effective budget instrumentation: total pages attended / max possible.
        # Lets us report "effective budget" alongside nominal budget for fair
        # comparison with fixed-sparse baselines. Uses GPU-tensor accumulators
        # to avoid .item() sync on the reuse hot path — synced once in get_stats().
        # Lazy-initialized on first call (need device info).
        self._attn_work: Optional[torch.Tensor] = None  # pages attended (sum over heads)
        self._attn_work_max: Optional[torch.Tensor] = None  # dense equivalent

        self._step = 0
        self._reuse_state: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        # For lazy mode: cache page selection for coverage checking
        self._cached_selection: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # page_indices [B, H_kv, budget], n_per_head [B, H_kv]
        # For coverage_interval < check_interval (paper-described variant):
        # cache the page_scores tensor from the most recent scoring step so
        # the coverage gate can run on cached scores between scoring refreshes.
        # Empty when check_interval == coverage_interval (deployed default).
        self._cached_page_scores: Dict[int, torch.Tensor] = {}

        # Diagnostics (lightweight, always on)
        self._accept_counts = 0  # steps where ALL heads have good coverage
        self._reject_counts = 0  # steps where coverage triggered re-scoring
        self._total_scoring_steps = 0  # full scoring steps executed
        self._total_coverage_checks = 0  # lazy: coverage checks performed
        self._per_head_accept_sum = 0
        self._per_head_total = 0
        self._dense_layer_steps = 0

        # Per-kernel timing (opt-in). Aggregated across all layers/steps;
        # divide by _kernel_counts at read time for per-call averages.
        self._track_timing = False
        self._kernel_times_ms: Dict[str, float] = {
            "scoring": 0.0, "coverage": 0.0, "sparse_draft": 0.0,
            "ar_recovery": 0.0, "lse_merge": 0.0, "complement": 0.0,
        }
        self._kernel_counts: Dict[str, int] = {k: 0 for k in self._kernel_times_ms}
        self._pending_events: list = []  # [(name, start_event, end_event)]

    def enable_timing(self, track: bool = True):
        """Turn per-kernel CUDA-event timing on/off. Adds negligible
        overhead (a couple microseconds per kernel) when on."""
        self._track_timing = track

    def _time_kernel(self, name: str, fn, *args, **kwargs):
        """Wrap a kernel call to record CUDA events. No-op if timing off."""
        if not self._track_timing:
            return fn(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        self._pending_events.append((name, start, end))
        return result

    def _flush_timing(self):
        """Synchronize and accumulate all pending CUDA event pairs."""
        if not self._track_timing or not self._pending_events:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending_events:
            self._kernel_times_ms[name] += start.elapsed_time(end)
            self._kernel_counts[name] += 1
        self._pending_events.clear()

    def reset(self):
        self._step = 0
        self._reuse_state = {}
        self._cached_selection = {}
        self._cached_page_scores = {}
        self._last_accept_mask = {}
        self._last_expanded_selection = {}
        if self._attn_work is not None:
            self._attn_work.zero_()
            self._attn_work_max.zero_()
        self._accept_counts = 0
        self._reject_counts = 0
        self._total_scoring_steps = 0
        self._total_coverage_checks = 0
        self._per_head_accept_sum = 0
        self._per_head_total = 0
        self._dense_layer_steps = 0
        for k in self._kernel_times_ms:
            self._kernel_times_ms[k] = 0.0
            self._kernel_counts[k] = 0
        self._pending_events.clear()

    @property
    def is_scoring_step(self):
        return self._step % self.reuse_interval == 0

    def step_done(self, layer_idx: int, num_layers: int):
        if layer_idx == num_layers - 1:
            self._step += 1

    def get_stats(self) -> dict:
        total_checks = max(self._total_coverage_checks, self._total_scoring_steps, 1)
        stats = {
            "total_scoring_steps": self._total_scoring_steps,
            "total_coverage_checks": self._total_coverage_checks,
            "all_accepted_pct": self._accept_counts / max(total_checks, 1) * 100,
            "some_rejected_pct": self._reject_counts / max(total_checks, 1) * 100,
            "per_head_accept_rate": self._per_head_accept_sum / max(self._per_head_total, 1) * 100,
            "total_decode_steps": self._step,
            "lazy": self.lazy,
            "check_interval": self.check_interval if self.lazy else None,
            "rescore_rate": self._total_scoring_steps / max(self._total_coverage_checks, 1) * 100 if self.lazy else 100.0,
        }
        if self._attn_work is not None:
            pages_attended = int(self._attn_work.item())
            pages_max = int(self._attn_work_max.item())
            if pages_max > 0:
                stats["effective_budget_pct"] = pages_attended / pages_max * 100
                stats["total_pages_attended"] = pages_attended
                stats["total_pages_max"] = pages_max
        if self._track_timing:
            self._flush_timing()
            n_decode = max(self._step, 1)
            stats["kernel_times_ms_total"] = dict(self._kernel_times_ms)
            stats["kernel_counts"] = dict(self._kernel_counts)
            stats["kernel_ms_per_decode_step"] = {
                k: self._kernel_times_ms[k] / n_decode
                for k in self._kernel_times_ms
            }
        return stats

    def __call__(
        self,
        q: torch.Tensor,            # [B, H_q, D]
        k_cache: torch.Tensor,      # [num_blocks, block_size, H_kv, D]
        v_cache: torch.Tensor,      # [num_blocks, block_size, H_kv, D]
        block_tables: torch.Tensor,  # [B, max_pages]
        context_lens: torch.Tensor,  # [B]
        layer_idx: int,
    ) -> torch.Tensor:
        # Clear per-layer tracing state so TracingWrapper sees current call only
        self._last_accept_mask.pop(layer_idx, None)
        self._last_expanded_selection.pop(layer_idx, None)

        # Per-layer dense policy: skip CVSA for cornerstone layers
        if layer_idx in self._dense_layers:
            self._dense_layer_steps += 1
            # Dense attends all valid pages on every head → full work
            n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
            self._record_work(context_lens, n_valid.sum() * self.num_kv_heads)
            return self._dense_backend(
                q, k_cache, v_cache, block_tables, context_lens, layer_idx)

        if self.lazy:
            # Hybrid: scoring runs every check_interval steps; coverage runs
            # every coverage_interval steps (using cached page_scores between
            # scoring refreshes). When coverage_interval == check_interval
            # (default), the _coverage_only_step branch is unreachable and
            # behavior is identical to the deployed implementation.
            is_scoring_step = (self._step % self.check_interval == 0)
            is_coverage_step = (self._step % self.coverage_interval == 0)
            has_cache = layer_idx in self._reuse_state
            has_scores = layer_idx in self._cached_page_scores
            if is_scoring_step:
                return self._lazy_step(
                    q, k_cache, v_cache, block_tables, context_lens, layer_idx)
            elif is_coverage_step and has_cache and has_scores:
                # Paper-described variant: coverage on cached scores, no
                # fresh scoring kernel. Only reachable when
                # coverage_interval < check_interval.
                return self._coverage_only_step(
                    q, k_cache, v_cache, block_tables, context_lens, layer_idx)
            elif has_cache:
                return self._reuse_step(
                    q, k_cache, v_cache, block_tables, context_lens, layer_idx)
            else:
                # No cached state yet → force lazy step
                return self._lazy_step(
                    q, k_cache, v_cache, block_tables, context_lens, layer_idx)
        elif self.is_scoring_step:
            return self._scoring_step(
                q, k_cache, v_cache, block_tables, context_lens, layer_idx)
        else:
            return self._reuse_step(
                q, k_cache, v_cache, block_tables, context_lens, layer_idx)

    def _record_work(self, context_lens: torch.Tensor,
                     pages_attended: torch.Tensor):
        """Accumulate effective-budget counters (on-device, no sync).

        pages_attended: scalar tensor (int) — pages actually visited this layer
            call, summed over batch × KV heads.
        Max is computed from context_lens: for each batch element, n_valid
            pages × num_kv_heads (what dense would attend).
        """
        if self._attn_work is None:
            self._attn_work = torch.zeros(
                (), dtype=torch.long, device=context_lens.device)
            self._attn_work_max = torch.zeros(
                (), dtype=torch.long, device=context_lens.device)
        n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
        dense_max = n_valid.sum() * self.num_kv_heads
        # Device-safe accumulation: under layer-sharded multi-GPU decode the
        # per-layer tensors live on different devices than this global counter
        # (created once on the first layer's device). Cast the increments to the
        # counter's device. No-op on single-GPU (same device) — existing runs
        # are unaffected.
        self._attn_work += pages_attended.to(torch.long).to(self._attn_work.device)
        self._attn_work_max += dense_max.to(torch.long).to(self._attn_work_max.device)

    def _lazy_step(self, q, k_cache, v_cache, block_tables, context_lens,
                   layer_idx):
        """Lazy scoring: check coverage every step, re-score only when needed.

        1. Score pages (cheap, 0.06ms/layer)
        2. Check coverage of CACHED selection
        3. Coverage OK → reuse (fast path)
        4. Coverage low → full re-scoring + verification
        """
        B = q.shape[0]
        n_pages = block_tables.shape[1]

        # Always score pages (needed for coverage check). With
        # light_normalizer the sketch replaces the exact scan here too, so
        # no full-key read happens anywhere on this path.
        if self.light_normalizer:
            page_scores = self._time_kernel(
                "scoring", self._quest_page_scores,
                q, k_cache, block_tables, context_lens,
            )  # [B, H_kv, max_pages]
        else:
            page_scores = self._time_kernel(
                "scoring",
                triton_score_pages,
                q, k_cache, block_tables, context_lens,
                self.num_kv_heads, self.num_q_per_kv, self.block_size,
            )  # [B, H_kv, max_pages]

        self._total_coverage_checks += 1
        had_cache = layer_idx in self._cached_selection

        # Check coverage of cached selection (if available)
        if had_cache:
            page_indices, n_per_head = self._cached_selection[layer_idx]

            # Coverage computation — fused Triton kernel
            accept_mask, coverage = self._time_kernel(
                "coverage",
                triton_check_coverage,
                page_scores, page_indices, n_per_head, self.tau)

            # Counterfactual baseline: replace coverage decision with a
            # random per-(layer, step) Bernoulli draw at matched rate.
            # All heads at this layer accept or all reject together, to
            # match AR's per-layer batched compute pattern.
            if self.random_reject_prob is not None:
                if torch.rand(1, device=accept_mask.device).item() < self.random_reject_prob:
                    accept_mask = torch.zeros_like(accept_mask)
                else:
                    accept_mask = torch.ones_like(accept_mask)

            self._per_head_accept_sum += accept_mask.sum().item()
            self._per_head_total += accept_mask.numel()

            if accept_mask.all():
                # FAST PATH: cached selection still good → reuse
                self._accept_counts += 1
                self._last_accept_mask[layer_idx] = accept_mask
                pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
                # Effective budget: reuse attends the cached selection
                cached_nph = self._cached_selection[layer_idx][1]
                self._record_work(context_lens, cached_nph.sum())
                return self._time_kernel(
                    "sparse_draft",
                    triton_sparse_decode_reuse,
                    q, k_cache, v_cache, block_tables, context_lens,
                    pi_flat, nph_flat, padded,
                    self.num_kv_heads, self.num_q_per_kv, self.block_size)

        # SLOW PATH: no cache or coverage dropped → full scoring step
        self._reject_counts += 1
        return self._scoring_step(
            q, k_cache, v_cache, block_tables, context_lens,
            layer_idx, page_scores=page_scores)

    def _coverage_only_step(self, q, k_cache, v_cache, block_tables,
                            context_lens, layer_idx):
        """Coverage gate using cached page_scores; no fresh scoring kernel.

        Paper-described variant (coverage_interval < check_interval).
        Reuses the page_scores tensor cached at the most recent scoring
        step (`_cached_page_scores[layer_idx]`) as the all-context proxy
        for the coverage check. The expensive scoring kernel is NOT called.

        On accept-all: reuse cached selection.
        On reject:     trigger a forced scoring step (matches `_lazy_step`
                       slow-path behavior, including the
                       random_reject_prob counterfactual override).
        """
        page_scores = self._cached_page_scores[layer_idx]
        page_indices, n_per_head = self._cached_selection[layer_idx]

        accept_mask, coverage = self._time_kernel(
            "coverage",
            triton_check_coverage,
            page_scores, page_indices, n_per_head, self.tau)

        # Counterfactual: same override semantics as _lazy_step.
        if self.random_reject_prob is not None:
            if torch.rand(1, device=accept_mask.device).item() < self.random_reject_prob:
                accept_mask = torch.zeros_like(accept_mask)
            else:
                accept_mask = torch.ones_like(accept_mask)

        self._per_head_accept_sum += accept_mask.sum().item()
        self._per_head_total += accept_mask.numel()
        self._total_coverage_checks += 1

        if accept_mask.all():
            # FAST PATH: cached selection still good → reuse
            self._accept_counts += 1
            self._last_accept_mask[layer_idx] = accept_mask
            pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
            cached_nph = self._cached_selection[layer_idx][1]
            self._record_work(context_lens, cached_nph.sum())
            return self._time_kernel(
                "sparse_draft",
                triton_sparse_decode_reuse,
                q, k_cache, v_cache, block_tables, context_lens,
                pi_flat, nph_flat, padded,
                self.num_kv_heads, self.num_q_per_kv, self.block_size)

        # SLOW PATH: coverage dropped → forced scoring step. We do not
        # reuse the cached page_scores here because they are stale w.r.t.
        # the current query; the scoring step will compute fresh scores.
        self._reject_counts += 1
        return self._scoring_step(
            q, k_cache, v_cache, block_tables, context_lens,
            layer_idx, page_scores=None)

    def _adaptive_select(self, page_scores, base_budget, n_valid, device):
        """AdaKV-style global competition across heads for page budget.

        Instead of each head getting `base_budget` pages, all heads compete
        for H * base_budget total slots. Heads whose top page scores are
        higher win more slots. A floor ratio blends toward uniform to prevent
        extreme allocations.

        Returns: (page_indices, n_per_head, max_budget)
        """
        B, H, P = page_scores.shape
        total_budget = H * base_budget
        floor_ratio = self.adaptive_floor_ratio

        # Sort each head's page scores descending
        sorted_scores, sorted_indices = page_scores.sort(dim=-1, descending=True)

        # Flatten across heads → [B, H*P], global top-k
        flat_scores = sorted_scores.reshape(B, H * P)
        _, global_top = torch.topk(flat_scores, k=min(total_budget, H * P), dim=-1)

        # Map global indices back to head index
        head_of = global_top // P  # [B, total_budget]

        # Count per-head wins
        raw_budget = torch.zeros(B, H, device=device, dtype=torch.float32)
        raw_budget.scatter_add_(-1, head_of, torch.ones_like(head_of, dtype=torch.float32))

        # Floor blending: shrink toward uniform
        floor_cap = base_budget * floor_ratio
        final_budget = torch.round(raw_budget * (1 - floor_ratio) + floor_cap).to(torch.int32)

        # Clamp to valid range
        n_valid_exp = n_valid.unsqueeze(1).expand(B, H).to(torch.int32)
        final_budget = final_budget.clamp(min=1)
        final_budget = torch.min(final_budget, n_valid_exp)

        max_budget = int(final_budget.max().item())

        # Per-head top-k with variable budget: take top max_budget per head,
        # downstream kernels read only the first final_budget[b, h] entries.
        # sorted=True is REQUIRED: budgets differ per head (same contract as
        # _static_select/_topp_select), so truncation must keep the TOP-scored
        # pages — unsorted topk truncated per head attends an arbitrary subset.
        topk_k = min(max_budget, P)
        _, topk_indices = torch.topk(page_scores, k=topk_k, dim=-1, largest=True, sorted=True)
        page_indices = topk_indices.to(torch.int32)

        return page_indices, final_budget, max_budget

    def _quest_page_scores(self, q, k_cache, block_tables, context_lens):
        """Quest-style page ranking from per-page key min/max metadata.

        score[b, h, p] = sum_d max(q_d * kmin_d, q_d * kmax_d) — the Quest
        upper bound on the page's max dot product (Tang et al., ICML 2024).
        q is mean-pooled over the query heads of each KV group, matching the
        pooling convention of triton_score_pages. Tokens beyond context_len
        (garbage slots in the last partial page) are masked out of the
        min/max, and fully-invalid pages score -inf.

        Returns [B, H_kv, n_pages] float32 ranking scores. Default mode uses
        these for SELECTION only, with the coverage gate verifying realized
        exact mass. With light_normalizer=True they also supply the gate's
        all-context normalizer (fully lightweight mode).
        """
        if q.dim() == 4:
            q = q.squeeze(1)
        B, n_pages = block_tables.shape
        device = q.device

        # Gather paged keys: [B, P, block, H_kv, D]
        k_pages = k_cache[block_tables.long().clamp(min=0)]

        # Validity mask for tokens past context_len in the last page
        tok_pos = (
            torch.arange(n_pages, device=device).view(1, -1, 1) * self.block_size
            + torch.arange(self.block_size, device=device).view(1, 1, -1)
        )  # [1, P, block]
        valid = (tok_pos < context_lens.view(-1, 1, 1)).unsqueeze(-1).unsqueeze(-1)

        pos_inf = torch.full_like(k_pages, float("inf"))
        neg_inf = torch.full_like(k_pages, float("-inf"))
        k_min = torch.where(valid, k_pages, pos_inf).amin(dim=2)  # [B, P, H_kv, D]
        k_max = torch.where(valid, k_pages, neg_inf).amax(dim=2)

        q_rep = q.view(B, self.num_kv_heads, self.num_q_per_kv,
                       self.head_dim).mean(dim=2)                 # [B, H_kv, D]
        q_b = q_rep.unsqueeze(1).float()                          # [B, 1, H_kv, D]
        upper = torch.maximum(q_b * k_min.float(), q_b * k_max.float())
        # 1/sqrt(d) scaling, matching triton_score_pages (scale=D**-0.5).
        # Ranking (selq/twip selection) is invariant to this factor; the
        # light_normalizer gate and _topp_select's softmax mass depend on it.
        scale = self.head_dim ** -0.5
        scores = (upper.sum(dim=-1) * scale) \
            .transpose(1, 2).contiguous()                         # [B, H_kv, P]

        page_valid = valid.squeeze(-1).squeeze(-1).any(dim=2)     # [B, P]
        scores = scores.masked_fill(~page_valid.unsqueeze(1), float("-inf"))
        return scores

    def _static_select(self, sel_scores, layer_idx, n_valid, device):
        """Per-head top-k with OFFLINE static budgets (matched-compute control).

        Budgets come from static_budget_table[layer_idx] ([H_kv] fractions
        derived from offline coverage statistics). No query-dependent
        reallocation, no recovery — the static counterpart to CVSA's dynamic
        verify-and-recover routing.

        Returns: (page_indices, n_per_head, max_budget)
        """
        B, H, P = sel_scores.shape
        fracs = self.static_budget_table[layer_idx]
        if fracs.device != sel_scores.device:
            fracs = fracs.to(sel_scores.device)
            self.static_budget_table[layer_idx] = fracs

        per_head = torch.clamp((fracs * P).round().long(), min=3, max=P)  # [H]
        n_per_head = torch.min(
            per_head.view(1, H).expand(B, H),
            n_valid.view(B, 1).expand(B, H),
        ).to(torch.int32)
        max_budget = int(per_head.max().item())

        # sorted=True so truncating head h to its n_per_head[b, h] keeps the
        # TOP-scored pages (budgets differ per head, unlike the uniform path).
        _, topk_indices = torch.topk(
            sel_scores, k=min(max_budget, P), dim=-1, largest=True, sorted=True)
        page_indices = topk_indices.to(torch.int32)
        return page_indices, n_per_head, max_budget

    def _topp_select(self, sel_scores, n_valid, device):
        """Per-head top-p budget selection on the selector's page scores.

        Each head takes the smallest prefix of its score-sorted pages whose
        cumulative softmax mass reaches self.top_p (the pre-hoc adaptive-
        budget principle of Twilight / top-p attention, at page granularity).
        Invalid pages score -inf and thus carry zero mass.

        Returns: (page_indices, n_per_head, max_budget); page_indices are
        sorted descending by score so the first n_per_head[b, h] entries are
        exactly head h's top-p set.

        The softmax uses the attention temperature 1/sqrt(d): top-p mass is
        defined on the attention distribution softmax(q.k/sqrt(d)). Both page
        scorers (triton_score_pages and _quest_page_scores) already apply the
        1/sqrt(d) factor, so the scores are used here without rescaling.
        """
        B, H, P = sel_scores.shape
        scaled = sel_scores.float()
        probs = torch.softmax(scaled, dim=-1)                      # [B, H, P]
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        # Smallest k with cum[k-1] >= top_p  (== count of strictly-below + 1)
        need = (cum < self.top_p).sum(dim=-1) + 1                  # [B, H]
        need = need.clamp(min=3, max=P)
        n_per_head = torch.min(
            need, n_valid.view(B, 1).expand(B, H)).to(torch.int32)
        max_budget = int(n_per_head.max().item())
        page_indices = sorted_idx[:, :, :max_budget].to(torch.int32)
        return page_indices, n_per_head, max_budget

    def _scoring_step(self, q, k_cache, v_cache, block_tables, context_lens,
                      layer_idx, page_scores=None):
        B = q.shape[0]
        n_pages = block_tables.shape[1]
        n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1  # [B]

        # Budget
        budget = max(3, int(n_pages * self.budget_pct))

        # 1. Score all pages (skip if pre-computed by lazy step). With
        # light_normalizer the sketch scores stand in for the exact scan:
        # they drive selection, the coverage gate, and recovery expansion,
        # so the whole path is free of full-key scans.
        if page_scores is None:
            if self.light_normalizer:
                page_scores = self._time_kernel(
                    "scoring", self._quest_page_scores,
                    q, k_cache, block_tables, context_lens,
                )  # [B, H_kv, max_pages]
            else:
                page_scores = self._time_kernel(
                    "scoring",
                    triton_score_pages,
                    q, k_cache, block_tables, context_lens,
                    self.num_kv_heads, self.num_q_per_kv, self.block_size,
                )  # [B, H_kv, max_pages]

        # Cache page_scores for the coverage_interval < check_interval
        # variant (_coverage_only_step). No-op for the deployed configuration
        # because _coverage_only_step never fires when the two intervals match.
        self._cached_page_scores[layer_idx] = page_scores

        # 2. Top-k selection (uniform, adaptive, static-table, or alt selector)
        # sel_scores drives page RANKING only; page_scores always drives the
        # coverage gate. Default: page_scores is the exact scan, so the gate
        # verifies realized coverage of whatever the selector proposed. With
        # light_normalizer: page_scores IS the sketch, so selection and gate
        # share one estimated view and no second scoring pass is needed.
        sel_scores = page_scores
        if self.selector == "quest" and not self.light_normalizer:
            sel_scores = self._time_kernel(
                "scoring", self._quest_page_scores,
                q, k_cache, block_tables, context_lens)

        if self.top_p is not None:
            page_indices, n_per_head, budget = self._topp_select(
                sel_scores, n_valid, q.device)
        elif self.static_budget_table is not None:
            page_indices, n_per_head, budget = self._static_select(
                sel_scores, layer_idx, n_valid, q.device)
        elif self.adaptive_selection:
            page_indices, n_per_head, budget = self._adaptive_select(
                sel_scores, budget, n_valid, q.device)
        else:
            topk_vals, topk_indices = torch.topk(
                sel_scores, k=budget, dim=-1, largest=True, sorted=False)
            page_indices = topk_indices.to(torch.int32)

            n_per_head = torch.full(
                (B, self.num_kv_heads), budget,
                dtype=torch.int32, device=q.device)
            n_per_head = torch.min(
                n_per_head,
                n_valid.unsqueeze(1).expand_as(n_per_head).to(torch.int32))
            n_per_head = n_per_head.clamp(max=budget)

        # 3. Draft attention (unnormalized)
        o_d, m_d, l_d = self._time_kernel(
            "sparse_draft",
            triton_sparse_decode_unnorm,
            q, k_cache, v_cache, block_tables, page_indices, n_per_head,
            context_lens, self.num_kv_heads, self.num_q_per_kv,
            self.block_size)

        # 4. Coverage (per KV-head) — fused Triton kernel
        accept_mask, coverage = self._time_kernel(
            "coverage",
            triton_check_coverage,
            page_scores, page_indices, n_per_head, self.tau)

        # Counterfactual: override coverage with random per-layer decision.
        # Only applied when random_reject_prob is set, for ablation purposes.
        if self.random_reject_prob is not None:
            if torch.rand(1, device=accept_mask.device).item() < self.random_reject_prob:
                accept_mask = torch.zeros_like(accept_mask)
            else:
                accept_mask = torch.ones_like(accept_mask)

        self._last_accept_mask[layer_idx] = accept_mask

        # 5. Branch. Track extra pages attended beyond draft for eff-budget.
        recovery_pages: Optional[torch.Tensor] = None
        if accept_mask.all():
            output = o_d / l_d.unsqueeze(-1).clamp(min=1e-6)
            output = output.to(q.dtype)
            if not self.lazy:
                self._accept_counts += 1
        elif self.approx_recovery_pct is not None:
            # 5-approx. Approximate recovery: expanded sparse for rejected heads
            exp_budget = max(budget + 1, int(n_pages * self.approx_recovery_pct))
            exp_budget = min(exp_budget, n_pages)

            # Expansion ranking follows the SELECTOR (a lightweight-selector
            # deployment has no exact scores to expand with), so recovery
            # quality is honestly limited by the same selector under test.
            _, exp_topk_indices = torch.topk(
                sel_scores, k=exp_budget, dim=-1, largest=True, sorted=False)
            exp_page_indices = exp_topk_indices.to(torch.int32)
            exp_n_per_head = torch.full(
                (B, self.num_kv_heads), exp_budget,
                dtype=torch.int32, device=q.device)
            exp_n_per_head = torch.min(
                exp_n_per_head,
                n_valid.unsqueeze(1).expand_as(exp_n_per_head).to(torch.int32))

            o_exp, m_exp, l_exp = self._time_kernel(
                "ar_recovery",
                triton_sparse_decode_unnorm,
                q, k_cache, v_cache, block_tables,
                exp_page_indices, exp_n_per_head,
                context_lens, self.num_kv_heads, self.num_q_per_kv,
                self.block_size)
            recovery_pages = exp_n_per_head.sum()

            # Per-head select: accepted → draft, rejected → expanded
            o_draft_norm = o_d / l_d.unsqueeze(-1).clamp(min=1e-6)
            o_exp_norm = o_exp / l_exp.unsqueeze(-1).clamp(min=1e-6)
            accept_q = accept_mask.repeat_interleave(
                self.num_q_per_kv, dim=1)  # [B, H_q]
            output = torch.where(
                accept_q.unsqueeze(-1), o_draft_norm, o_exp_norm)
            output = output.to(q.dtype)
            self._last_expanded_selection[layer_idx] = (exp_page_indices, exp_n_per_head)
            if not self.lazy:
                self._reject_counts += 1
        else:
            # 5-exact. Exact recovery via complement + LSE merge
            # 5a. Complement indices — Triton O(n) kernel (replaces sort)
            comp_indices, n_comp = self._time_kernel(
                "complement",
                triton_compute_complement,
                page_indices, n_per_head, context_lens, self.block_size,
                accept_mask)

            # 5b. Complement attention (existing tiled kernel)
            o_c, m_c, l_c = self._time_kernel(
                "ar_recovery",
                triton_sparse_decode_unnorm,
                q, k_cache, v_cache, block_tables, comp_indices, n_comp,
                context_lens, self.num_kv_heads, self.num_q_per_kv,
                self.block_size)
            recovery_pages = n_comp.sum()

            # 5c. LSE merge — fused Triton kernel
            output = self._time_kernel(
                "lse_merge",
                triton_lse_merge,
                o_d, m_d, l_d, o_c, m_c, l_c,
                accept_mask, self.num_kv_heads, self.num_q_per_kv, q.dtype)
            self._last_expanded_selection[layer_idx] = (comp_indices, n_comp)
            if not self.lazy:
                self._reject_counts += 1

        # 6. Effective budget: draft always runs; recovery runs when rejected.
        pages_attended = n_per_head.sum()
        if recovery_pages is not None:
            pages_attended = pages_attended + recovery_pages
        self._record_work(context_lens, pages_attended)

        # 7. Cache reuse state + selection for lazy coverage check
        pi_flat, nph_flat, padded = prepare_reuse_state(
            page_indices, n_per_head, self.num_kv_heads, B)
        self._reuse_state[layer_idx] = (pi_flat, nph_flat, padded)
        self._cached_selection[layer_idx] = (page_indices, n_per_head)

        # 8. Update diagnostics
        self._total_scoring_steps += 1
        if not self.lazy:
            self._per_head_accept_sum += accept_mask.sum().item()
            self._per_head_total += accept_mask.numel()

        return output

    def _reuse_step(self, q, k_cache, v_cache, block_tables, context_lens,
                    layer_idx):
        if layer_idx not in self._reuse_state:
            # Fallback: force a scoring step
            self._step -= 1
            return self(q, k_cache, v_cache, block_tables, context_lens,
                        layer_idx)

        pi_flat, nph_flat, padded = self._reuse_state[layer_idx]
        # Effective budget: reuse attends the cached selection
        cached_nph = self._cached_selection[layer_idx][1]
        self._record_work(context_lens, cached_nph.sum())
        return triton_sparse_decode_reuse(
            q, k_cache, v_cache, block_tables, context_lens,
            pi_flat, nph_flat, padded,
            self.num_kv_heads, self.num_q_per_kv, self.block_size)
