"""Token-level reproduction of Tactic (Zhu et al., arXiv 2502.12216).

Tactic selects tokens until their cumulative attention score reaches a target
fraction P of the total, estimating that cumulative score cheaply via
clustering-based sorting plus distribution fitting.  There is no public
reference implementation, so this is reimplemented from the paper text; every
point the paper leaves underspecified is tagged (A1, A2, ...) in the comments
below, at the place where we fix the choice.

The reproduction runs k-means over keys at prefill, ranks clusters by
query-centroid dot product, uses exact scores for the leading segment and an
``a/x + b`` fit for the remainder, and accumulates to P.

Selection is token-level, as in the original.  Attention is computed by masking
the full score vector rather than by a sparse kernel: this evaluation path
targets quality and read-traffic accounting, not wall-clock, so correctness is
preferred over speed.

Read traffic is counted on the *union* of the tokens selected by the query heads
sharing a KV head, since those heads read one shared K/V (A10).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

_NEG_INF = float("-inf")
_EULER_GAMMA = 0.5772156649015329


class TacticBackend:
    """Tactic selection behind the same call signature as the CVSA backend."""

    def __init__(
        self,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        block_size: int,
        num_layers: int,
        target_p: float = 0.80,
        cluster_size: int = 32,        # paper: average cluster size
        kmeans_iters: int = 10,        # paper: max 10 iterations
        exact_frac: float = 0.02,      # paper "~1-2%"; A6: we take the upper end
        fit_pos: Tuple[float, float] = (0.10, 0.60),  # paper sample positions
        fit_window: float = 0.01,      # A5: averaging window at each fit position
        refresh_interval: int = 2048,  # paper: re-cluster every 2048 steps
        tau: Optional[float] = None,   # coverage gate on top of Tactic's selection
        recovery_mult: float = 4.0,    # recovery budget as a multiple of the draft
        sink_tokens: int = 0,          # A8 probe: force-include leading tokens
        recent_tokens: int = 0,        # A8 probe: force-include trailing tokens
        seed: int = 1234,
    ):
        if not 0.0 < target_p < 1.0:
            raise ValueError(f"target_p must be in (0, 1), got {target_p}")
        if tau is not None and not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")

        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_layers = num_layers

        self.target_p = target_p
        self.cluster_size = cluster_size
        self.kmeans_iters = kmeans_iters
        self.exact_frac = exact_frac
        self.fit_pos = fit_pos
        self.fit_window = fit_window
        self.refresh_interval = refresh_interval
        self.tau = tau
        self.recovery_mult = recovery_mult
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.seed = seed

        # (layer, batch) -> (centroids per kv head, assignment per kv head, n)
        self._clusters: Dict[Tuple[int, int], Tuple[List, List, int]] = {}
        self._steps: Dict[Tuple[int, int], int] = {}

        self.reset_stats()

    # ------------------------------------------------------- runner protocol
    def reset(self) -> None:
        """Per-sequence reset, called by the runner before each generation.

        Clearing the cluster cache is a correctness requirement, not just
        interface compliance: centroids are fitted to one sequence's keys and
        must never carry over into the next sample.

        Stats are cleared here too, matching ``CVSABackend.reset``,
        so the read-traffic figures of both backends are measured on the same
        basis and remain comparable.
        """
        self._clusters = {}
        self._steps = {}
        self.reset_stats()

    def step_done(self, layer_idx: int, num_layers: int) -> None:
        """Called after each layer; the decode-step counter is per layer here.

        Cluster staleness is tracked per (layer, batch) inside ``__call__``,
        so this is a no-op kept for runner compatibility.
        """
        return None

    # ---------------------------------------------------------------- stats
    def reset_stats(self) -> None:
        self._tokens_read = 0     # union over each GQA group
        self._tokens_max = 0
        # Tactic selects individual tokens. A paged serving stack reads whole
        # pages, so the traffic it would actually realize is set by how many
        # pages the selection touches, not by how many tokens it names. Count
        # both and let the comparison use whichever the deployment target has.
        self._pages_touched = 0
        self._pages_max = 0
        self._fit_calls = 0
        self._fit_neg_a = 0       # ordering violation: score rises with rank
        self._fit_neg_b = 0       # steep curve: truncated at the zero crossing
        self._fit_singular = 0    # identical sample ranks
        self._gate_checks = 0
        self._gate_rejects = 0
        self._tokens_recovered = 0

    def get_stats(self) -> Dict[str, float]:
        calls = max(self._fit_calls, 1)
        stats = {
            "effective_budget_pct": 100.0 * self._tokens_read / max(self._tokens_max, 1),
            "total_tokens_read": self._tokens_read,
            "total_tokens_max": self._tokens_max,
            # what a paged stack would have to fetch for the same selection
            "pages_touched_pct": 100.0 * self._pages_touched / max(self._pages_max, 1),
            "total_pages_touched": self._pages_touched,
            "total_pages_max": self._pages_max,
            # split so an ordering problem is never confused with a steep curve
            "fit_neg_a_pct": 100.0 * self._fit_neg_a / calls,
            "fit_neg_b_pct": 100.0 * self._fit_neg_b / calls,
            "fit_singular_pct": 100.0 * self._fit_singular / calls,
            "target_p": self.target_p,
            "tau": self.tau,
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
        }
        if self.tau is not None:
            stats.update({
                "gate_reject_pct": 100.0 * self._gate_rejects / max(self._gate_checks, 1),
                "tokens_recovered": self._tokens_recovered,
            })
        return stats

    # ------------------------------------------------------------ internals
    @staticmethod
    def _harmonic(n: float) -> float:
        """H_n, used to sum the fitted a/x tail in closed form."""
        if n < 1:
            return 0.0
        return math.log(n) + _EULER_GAMMA + 1.0 / (2.0 * n)

    def _materialize(self, cache: torch.Tensor, block_tables: torch.Tensor) -> torch.Tensor:
        """Paged cache -> contiguous [B, n_tok, H_kv, D]."""
        gathered = cache[block_tables.long().clamp(min=0)]   # [B, P, bs, H, D]
        b, p, bs, h, d = gathered.shape
        return gathered.reshape(b, p * bs, h, d)

    def _kmeans(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """k-means over key vectors for one (batch, kv_head).

        Euclidean distance on unnormalised keys (A1, A2).
        keys: [n, D] -> (centroids [k, D], assignment [n]).
        """
        n = keys.shape[0]
        k = max(1, -(-n // self.cluster_size))          # ceil(n / cluster_size)
        g = torch.Generator().manual_seed(self.seed)
        init = torch.randperm(n, generator=g)[:k].to(keys.device)
        kf = keys.float()
        centroids = kf[init].clone()
        assign = torch.zeros(n, dtype=torch.long, device=keys.device)
        ones = torch.ones(n, device=keys.device, dtype=kf.dtype)
        for _ in range(self.kmeans_iters):
            assign = torch.cdist(kf, centroids, p=2).argmin(dim=1)
            new = torch.zeros_like(centroids)
            cnt = torch.zeros(centroids.shape[0], device=keys.device, dtype=kf.dtype)
            new.index_add_(0, assign, kf)
            cnt.index_add_(0, assign, ones)
            nonempty = cnt > 0
            new[nonempty] = new[nonempty] / cnt[nonempty].unsqueeze(1)
            new[~nonempty] = centroids[~nonempty]         # keep empty centroids put
            if torch.allclose(new, centroids):
                centroids = new
                break
            centroids = new
        return centroids, assign

    def _fit_tail(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """Solve y = a/x + b through two (rank, value) points (paper eq.).

        A negative ``b`` is normal rather than a failure: for a steep curve
        ``a/x1`` slightly exceeds ``y1``.  It is handled where the tail is
        summed, by truncating the fitted curve at its zero crossing (attention
        scores cannot be negative), so the fit is used rather than discarded.
        A negative ``a`` is a genuine signal: it means the score at the 10%
        position is below the one at 60%, i.e. the cluster ordering is not
        descending.  The two are counted separately so they can be told apart.
        """
        self._fit_calls += 1
        inv1, inv2 = 1.0 / max(x1, 1.0), 1.0 / max(x2, 1.0)
        denom = inv1 - inv2
        if abs(denom) < 1e-12:
            self._fit_singular += 1
            return 0.0, max((y1 + y2) / 2.0, 0.0)
        a = (y1 - y2) / denom
        b = y1 - a * inv1
        if a < 0.0:
            self._fit_neg_a += 1          # ordering violation (A1/A4)
            a = 0.0
            b = max(b, 0.0)
        elif b < 0.0:
            self._fit_neg_b += 1          # steep curve; truncated when summed
        return a, b

    def _tail_sum(self, a: float, b: float, m: int, k: int) -> float:
        """Sum of ``max(a/x + b, 0)`` over ranks ``x`` in ``(m, k]``.

        Truncating at the zero crossing keeps the estimate non-negative and
        keeps the partial sums monotone in ``k``, which the cut-point search
        below relies on.
        """
        if k <= m or (a <= 0.0 and b <= 0.0):
            return 0.0
        hi = k
        if b < 0.0:
            hi = min(k, int(math.floor(-a / b)))
        if hi <= m:
            return 0.0
        return a * (self._harmonic(hi) - self._harmonic(m)) + b * (hi - m)

    def _force_include(self, chosen: torch.Tensor, n: int) -> torch.Tensor:
        """Union in the always-kept leading/trailing tokens (A8 probe).

        The paper does not say whether attention sinks or a recent window are
        force-included.  The default is not to add them, since inventing extra
        machinery for a competitor is its own kind of unfaithfulness; these
        knobs exist to test whether that omission explains the gap we see at
        low target mass, where few tokens are selected and missing the sink
        would be most damaging.
        """
        if self.sink_tokens <= 0 and self.recent_tokens <= 0:
            return chosen
        parts = [chosen]
        if self.sink_tokens > 0:
            parts.append(torch.arange(min(self.sink_tokens, n), device=chosen.device))
        if self.recent_tokens > 0:
            lo = max(0, n - self.recent_tokens)
            parts.append(torch.arange(lo, n, device=chosen.device))
        return torch.unique(torch.cat(parts))

    def _select_tactic(
        self, exact: torch.Tensor, order: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Estimate cumulative mass along the cluster order and cut at P.

        Uses exact scores for the leading segment and the fitted a/x + b curve
        for the remainder, and normalises by the *estimated* total (A7).
        """
        m = min(max(1, int(self.exact_frac * n)), n)
        head = order[:m]
        exact_head = exact[head]
        s_exact = float(exact_head.sum())

        if m >= n:
            return head

        # two sample points on the remaining ranks (paper: 10% and 60%)
        w = max(1, int(self.fit_window * n / 2))
        pts = []
        for frac in self.fit_pos:
            c = min(max(int(frac * n), m + 1), n - 1)
            lo, hi = max(m, c - w), min(n, c + w + 1)
            rank = (lo + hi - 1) / 2.0 + 1.0        # 1-indexed rank
            pts.append((rank, float(exact[order[lo:hi]].mean())))
        a, b = self._fit_tail(pts[0][0], pts[0][1], pts[1][0], pts[1][1])

        s_hat = s_exact + self._tail_sum(a, b, m, n)
        target = self.target_p * s_hat

        csum_head = torch.cumsum(exact_head, dim=0)
        hit = (csum_head >= target).nonzero()
        if hit.numel() > 0:
            return order[: int(hit[0].item()) + 1]

        need = target - s_exact
        lo, hi = m, n
        while lo < hi:                                # closed-form tail walk
            mid = (lo + hi) // 2
            if self._tail_sum(a, b, m, mid) >= need:
                hi = mid
            else:
                lo = mid + 1
        return order[: min(max(lo, m), n)]

    def _apply_gate(self, exact, order, chosen, n):
        """Coverage gate on top of Tactic's selection (the composition claim).

        Reads the realised coverage of whatever Tactic selected and, when it
        falls below tau, expands along Tactic's own ordering until the target is
        met or the recovery budget is exhausted.  Recovery therefore follows the
        selector under test, so its quality is limited by that selector rather
        than by an oracle ranking.

        This path uses exact token-level coverage, which is the idealised form
        of the page-level proxy the deployed system uses; it tests the principle
        of verification rather than our particular proxy.
        """
        self._gate_checks += 1
        total = exact.sum()
        cov = exact[chosen].sum() / total.clamp(min=1e-9)
        if float(cov) >= self.tau:
            return chosen

        self._gate_rejects += 1
        k0 = int(chosen.numel())
        budget = min(n, int(self.recovery_mult * k0))
        csum = torch.cumsum(exact[order], dim=0)
        need = self.tau * float(total)
        hit = (csum >= need).nonzero()
        k_new = int(hit[0].item()) + 1 if hit.numel() > 0 else n
        k_new = min(max(k_new, k0), budget)
        # Union rather than replace: the gate may only ADD. Returning
        # order[:k_new] outright would silently drop any force-included tokens
        # that fall outside the leading prefix of the selector's ordering.
        out = torch.unique(torch.cat([chosen, order[:k_new]]))
        self._tokens_recovered += max(0, int(out.numel()) - k0)
        return out

    # ------------------------------------------------------------------ call
    def __call__(
        self,
        q: torch.Tensor,             # [B, H_q, D]
        k_cache: torch.Tensor,       # [num_blocks, block_size, H_kv, D]
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,  # [B, max_pages]
        context_lens: torch.Tensor,  # [B]
        layer_idx: int,
    ) -> torch.Tensor:
        if q.dim() == 4:
            q = q.squeeze(1)
        B, H_q, D = q.shape
        G = self.num_q_per_kv
        scale = D ** -0.5

        K = self._materialize(k_cache, block_tables)
        V = self._materialize(v_cache, block_tables)
        n_tok = K.shape[1]
        out = torch.zeros_like(q)

        for b in range(B):
            n = max(1, min(int(context_lens[b].item()), n_tok))
            Kb, Vb = K[b, :n], V[b, :n]
            key = (layer_idx, b)

            stale = self._steps.get(key, 0) >= self.refresh_interval
            if key not in self._clusters or stale:
                cents, assigns = [], []
                for h in range(self.num_kv_heads):
                    c, a = self._kmeans(Kb[:, h, :])
                    cents.append(c)
                    assigns.append(a)
                self._clusters[key] = (cents, assigns, n)
                self._steps[key] = 0
            cents, assigns, _ = self._clusters[key]

            for h in range(self.num_kv_heads):
                Kh = Kb[:, h, :].float()
                Vh = Vb[:, h, :].float()

                asg = assigns[h]
                if asg.shape[0] < n:      # tokens appended since clustering (A3)
                    extra = torch.cdist(Kh[asg.shape[0]:], cents[h], p=2).argmin(dim=1)
                    asg = torch.cat([asg, extra])
                elif asg.shape[0] > n:
                    asg = asg[:n]

                union = torch.zeros(n, dtype=torch.bool, device=q.device)
                for j in range(G):
                    hq = h * G + j
                    qv = q[b, hq].float()
                    logits = (Kh @ qv) * scale
                    exact = torch.exp(logits - logits.max())   # ratio-invariant

                    cscore = cents[h] @ qv
                    crank = torch.empty_like(cscore)
                    crank[torch.argsort(cscore, descending=True)] = torch.arange(
                        cscore.shape[0], device=cscore.device, dtype=cscore.dtype)
                    order = torch.argsort(crank[asg], stable=True)   # A4
                    chosen = self._select_tactic(exact, order, n)

                    chosen = self._force_include(chosen, n)
                    if self.tau is not None:
                        chosen = self._apply_gate(exact, order, chosen, n)

                    masked = torch.full_like(logits, _NEG_INF)
                    masked[chosen] = logits[chosen]
                    out[b, hq] = (torch.softmax(masked, dim=0).unsqueeze(-1) * Vh).sum(0).to(q.dtype)
                    union[chosen] = True

                # one shared K/V read per KV head -> count the union (A10)
                self._tokens_read += int(union.sum().item())
                self._tokens_max += n

                # same union, charged at page granularity
                bs = self.block_size
                n_pages = (n + bs - 1) // bs
                pad = n_pages * bs - n
                u = (union if pad == 0
                     else torch.cat([union, union.new_zeros(pad)]))
                self._pages_touched += int(u.view(n_pages, bs).any(dim=1).sum().item())
                self._pages_max += n_pages

            self._steps[key] = self._steps.get(key, 0) + 1

        return out
