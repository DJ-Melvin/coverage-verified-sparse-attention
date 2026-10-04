"""CPU unit tests for the Tactic reproduction (cvsa/tactic.py).

These check the properties the spec pins down, so that a failure here means the
reimplementation deviates from the paper rather than merely scoring differently:

  1. selecting every token reproduces dense attention
  2. k-means produces a valid partition
  3. the a/x + b fit is exact on points drawn from that family
  4. traffic is counted on the GQA union, not per query head
  5. the cluster order is a permutation and respects within-cluster position order
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvsa.tactic import TacticBackend  # noqa: E402


def _cache(n_blocks, block_size, h_kv, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n_blocks, block_size, h_kv, d, generator=g)
    v = torch.randn(n_blocks, block_size, h_kv, d, generator=g)
    return k, v


def _mk(h_kv=2, h_q=4, d=8, block=4, n_blocks=16, **kw):
    return TacticBackend(num_kv_heads=h_kv, num_q_heads=h_q, head_dim=d,
                         block_size=block, num_layers=1, **kw)


def test_full_selection_matches_dense():
    """With every token in the exact leading segment, output is full attention."""
    torch.manual_seed(1)
    h_kv, h_q, d, block, nb = 2, 4, 8, 4, 8
    k, v = _cache(nb, block, h_kv, d)
    q = torch.randn(1, h_q, d)
    bt = torch.arange(nb).unsqueeze(0)
    cl = torch.tensor([nb * block])

    # exact_frac=1.0 puts all n tokens in the exact head, so _select_tactic
    # returns every token and the masked softmax equals dense attention.
    be = _mk(h_kv, h_q, d, block, nb, target_p=0.9, exact_frac=1.0)
    got = be(q, k, v, bt, cl, layer_idx=0)

    n = nb * block
    K = k.reshape(n, h_kv, d)
    V = v.reshape(n, h_kv, d)
    ref = torch.zeros_like(q)
    for hq in range(h_q):
        h = hq // (h_q // h_kv)
        logits = (K[:, h, :] @ q[0, hq]) * (d ** -0.5)
        ref[0, hq] = torch.softmax(logits, 0) @ V[:, h, :]
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


def test_kmeans_is_a_valid_partition():
    torch.manual_seed(2)
    be = _mk(cluster_size=8)
    keys = torch.randn(64, 8)
    cents, assign = be._kmeans(keys)
    assert cents.shape == (8, 8), cents.shape          # ceil(64/8)
    assert assign.shape == (64,)
    assert int(assign.min()) >= 0 and int(assign.max()) < cents.shape[0]
    assert torch.isfinite(cents).all()


def test_fit_recovers_known_curve():
    """Points drawn from y = a/x + b must recover (a, b) exactly."""
    be = _mk()
    a_true, b_true = 12.0, 0.5
    x1, x2 = 10.0, 60.0
    y1, y2 = a_true / x1 + b_true, a_true / x2 + b_true
    a, b = be._fit_tail(x1, y1, x2, y2)
    assert abs(a - a_true) < 1e-6 and abs(b - b_true) < 1e-6


def test_fit_handles_singular_input():
    """Identical sample ranks are unsolvable and counted separately."""
    be = _mk()
    a, b = be._fit_tail(10.0, 1.0, 10.0, 5.0)   # identical x -> singular
    assert a >= 0.0 and b >= 0.0
    assert be._fit_singular == 1
    assert be._fit_neg_a == 0 and be._fit_neg_b == 0


def test_traffic_counted_on_gqa_union_not_per_head():
    """Read must be the union over the group, so it can never exceed n per KV head."""
    torch.manual_seed(3)
    h_kv, h_q, d, block, nb = 2, 8, 8, 4, 8      # 4 query heads per KV head
    k, v = _cache(nb, block, h_kv, d)
    q = torch.randn(1, h_q, d)
    bt = torch.arange(nb).unsqueeze(0)
    n = nb * block
    cl = torch.tensor([n])

    be = _mk(h_kv, h_q, d, block, nb, target_p=0.9)
    be(q, k, v, bt, cl, layer_idx=0)
    st = be.get_stats()
    assert st["total_tokens_max"] == n * h_kv, st
    assert st["total_tokens_read"] <= n * h_kv, "union exceeded the cache"
    assert st["effective_budget_pct"] <= 100.0


def test_pages_touched_bounds_token_read():
    """A token selection costs whole pages on a paged stack.

    Pages touched can never be fewer than the pages the chosen tokens would
    fill if they were packed, nor more than every page in the cache, and it is
    charged on the same GQA union as the token count.
    """
    torch.manual_seed(7)
    h_kv, h_q, d, block, nb = 2, 4, 8, 4, 16
    k, v = _cache(nb, block, h_kv, d)
    q = torch.randn(1, h_q, d)
    bt = torch.arange(nb).unsqueeze(0)
    n = nb * block
    cl = torch.tensor([n])

    be = _mk(h_kv, h_q, d, block, nb, target_p=0.7)
    be(q, k, v, bt, cl, layer_idx=0)
    st = be.get_stats()

    assert st["total_pages_max"] == nb * h_kv, st
    packed = -(-st["total_tokens_read"] // block)          # ceil
    assert packed <= st["total_pages_touched"] <= nb * h_kv, st
    assert 0.0 <= st["pages_touched_pct"] <= 100.0


def test_scattered_selection_touches_more_pages_than_it_reads_tokens():
    """The whole point of the counter: scatter makes page traffic exceed token traffic."""
    torch.manual_seed(8)
    h_kv, h_q, d, block, nb = 1, 1, 8, 8, 12
    k, v = _cache(nb, block, h_kv, d)
    q = torch.randn(1, h_q, d)
    bt = torch.arange(nb).unsqueeze(0)
    n = nb * block
    cl = torch.tensor([n])

    be = _mk(h_kv, h_q, d, block, nb, target_p=0.5)
    be(q, k, v, bt, cl, layer_idx=0)
    st = be.get_stats()

    # k-means groups by key similarity, not by position, so the selected tokens
    # land in scattered positions and the page share is the larger number.
    assert st["pages_touched_pct"] >= st["effective_budget_pct"] - 1e-6, st


def test_cluster_order_is_permutation_and_stable():
    """Order must cover every token once and keep position order inside a cluster."""
    torch.manual_seed(4)
    be = _mk(cluster_size=4)
    n = 32
    keys = torch.randn(n, 8)
    cents, assign = be._kmeans(keys)
    qv = torch.randn(8)
    cscore = cents @ qv
    crank = torch.empty_like(cscore)
    crank[torch.argsort(cscore, descending=True)] = torch.arange(
        cscore.shape[0], dtype=cscore.dtype)
    order = torch.argsort(crank[assign], stable=True)

    assert sorted(order.tolist()) == list(range(n)), "not a permutation"
    for c in range(cents.shape[0]):
        members = [i for i in order.tolist() if int(assign[i]) == c]
        assert members == sorted(members), f"cluster {c} lost position order"


def test_tactic_selects_fewer_tokens_at_lower_target():
    """Lower P must not read more than higher P."""
    torch.manual_seed(5)
    h_kv, h_q, d, block, nb = 1, 2, 8, 8, 16
    k, v = _cache(nb, block, h_kv, d)
    q = torch.randn(1, h_q, d)
    bt = torch.arange(nb).unsqueeze(0)
    cl = torch.tensor([nb * block])

    reads = []
    for p in (0.5, 0.9):
        be = _mk(h_kv, h_q, d, block, nb, target_p=p, cluster_size=8)
        be(q, k, v, bt, cl, layer_idx=0)
        reads.append(be.get_stats()["total_tokens_read"])
    assert reads[0] <= reads[1], f"P=0.5 read {reads[0]} > P=0.9 read {reads[1]}"


def test_runner_protocol_methods_exist():
    """The eval runner calls these unconditionally; missing them aborts the job."""
    be = _mk()
    for name in ("reset", "step_done", "get_stats"):
        assert callable(getattr(be, name, None)), f"missing {name}()"
    be.step_done(0, 1)   # must not raise


def test_reset_clears_cluster_cache():
    """Clusters are fitted per sequence and must not leak into the next sample."""
    torch.manual_seed(6)
    h_kv, h_q, d, block, nb = 1, 2, 8, 8, 8
    k, v = _cache(nb, block, h_kv, d)
    q = torch.randn(1, h_q, d)
    bt = torch.arange(nb).unsqueeze(0)
    cl = torch.tensor([nb * block])

    be = _mk(h_kv, h_q, d, block, nb, target_p=0.8, cluster_size=8)
    be(q, k, v, bt, cl, layer_idx=0)
    assert be._clusters, "clustering did not run"
    assert be.get_stats()["total_tokens_max"] > 0

    be.reset()
    assert be._clusters == {}, "cluster cache survived reset -> would leak"
    assert be._steps == {}
    assert be.get_stats()["total_tokens_max"] == 0, "stats survived reset"


def test_second_sequence_reclusters_after_reset():
    """A different sequence must get its own centroids, not the previous ones."""
    torch.manual_seed(7)
    h_kv, h_q, d, block, nb = 1, 1, 8, 8, 8
    bt = torch.arange(nb).unsqueeze(0)
    cl = torch.tensor([nb * block])
    q = torch.randn(1, h_q, d)

    be = _mk(h_kv, h_q, d, block, nb, target_p=0.8, cluster_size=8)
    k1, v1 = _cache(nb, block, h_kv, d, seed=11)
    be(q, k1, v1, bt, cl, layer_idx=0)
    cent1 = be._clusters[(0, 0)][0][0].clone()

    be.reset()
    k2, v2 = _cache(nb, block, h_kv, d, seed=22)
    be(q, k2, v2, bt, cl, layer_idx=0)
    cent2 = be._clusters[(0, 0)][0][0]

    assert not torch.allclose(cent1, cent2), "centroids reused across sequences"


def test_negative_b_is_kept_not_clamped():
    """b<0 is a steep curve, not a failure: keep the fit, truncate the tail."""
    be = _mk()
    # steep points -> a>0 but b<0
    a, b = be._fit_tail(x1=800.0, y1=1.0, x2=4800.0, y2=0.10)
    assert a > 0.0, a
    assert b < 0.0, "expected a steep fit with negative intercept"
    assert be._fit_neg_a == 0, "must not be counted as an ordering violation"
    assert be._fit_neg_b == 1


def test_negative_a_is_flagged_as_ordering_violation():
    """Score rising with rank means the cluster order is not descending."""
    be = _mk()
    a, b = be._fit_tail(x1=800.0, y1=0.1, x2=4800.0, y2=1.0)   # increasing
    assert a == 0.0 and b >= 0.0
    assert be._fit_neg_a == 1 and be._fit_neg_b == 0


def test_tail_sum_truncates_at_zero_crossing():
    """max(a/x+b,0) must be summed, and partial sums must be monotone in k."""
    be = _mk()
    a, b = 100.0, -1.0          # zero crossing at x=100
    assert be._tail_sum(a, b, 10, 50) > 0
    full = be._tail_sum(a, b, 10, 1000)
    at_cross = be._tail_sum(a, b, 10, 100)
    assert abs(full - at_cross) < 1e-6, "tail grew past the zero crossing"
    prev = 0.0
    for k in range(11, 200, 7):
        cur = be._tail_sum(a, b, 10, k)
        assert cur >= prev - 1e-9, f"non-monotone at k={k}"
        prev = cur


def test_gate_raises_coverage_above_tau():
    """With the gate on, realised coverage of the selection must reach tau."""
    torch.manual_seed(8)
    be = _mk(target_p=0.5, tau=0.95, cluster_size=8)
    n = 200
    exact = torch.rand(n) + 1e-3
    order = torch.argsort(exact, descending=True)
    chosen = order[:5]                      # deliberately far below tau
    out = be._apply_gate(exact, order, chosen, n)
    cov = exact[out].sum() / exact.sum()
    assert out.numel() >= chosen.numel()
    assert float(cov) >= 0.95 - 1e-6 or out.numel() == int(5 * be.recovery_mult)
    assert be._gate_rejects == 1


def test_gate_leaves_high_coverage_selection_untouched():
    be = _mk(target_p=0.9, tau=0.5)
    exact = torch.ones(100)
    order = torch.arange(100)
    chosen = order[:80]                     # coverage 0.8 > tau 0.5
    out = be._apply_gate(exact, order, chosen, 100)
    assert torch.equal(out, chosen)
    assert be._gate_rejects == 0


def test_gate_respects_recovery_budget():
    """Recovery must not exceed recovery_mult x the draft size."""
    be = _mk(target_p=0.3, tau=0.999, recovery_mult=2.0)
    n = 100
    exact = torch.ones(n)
    order = torch.arange(n)
    chosen = order[:10]
    out = be._apply_gate(exact, order, chosen, n)
    assert out.numel() <= 20, f"exceeded budget: {out.numel()}"


def test_force_include_is_off_by_default():
    """Default must stay faithful to the paper: nothing is force-included."""
    be = _mk()
    chosen = torch.tensor([5, 6, 7])
    assert torch.equal(be._force_include(chosen, 100), chosen)


def test_force_include_adds_sink_and_recent_without_duplicates():
    be = _mk(sink_tokens=4, recent_tokens=3)
    chosen = torch.tensor([2, 50])
    out = be._force_include(chosen, 100)
    assert set(out.tolist()) == {0, 1, 2, 3, 50, 97, 98, 99}
    assert out.numel() == len(set(out.tolist())), "duplicates leaked in"


def test_force_include_clamps_to_sequence_length():
    be = _mk(sink_tokens=10, recent_tokens=10)
    out = be._force_include(torch.tensor([1]), 5)
    assert int(out.max()) < 5 and int(out.min()) >= 0


def test_gate_never_drops_force_included_tokens():
    """The gate may only add; forced tokens outside the prefix must survive."""
    torch.manual_seed(9)
    be = _mk(target_p=0.5, tau=0.99, sink_tokens=2, recovery_mult=100.0)
    n = 50
    exact = torch.rand(n) + 1e-3
    order = torch.argsort(exact, descending=True)
    chosen = be._force_include(order[:3], n)
    out = be._apply_gate(exact, order, chosen, n)
    for t in chosen.tolist():
        assert t in out.tolist(), f"gate dropped force-included token {t}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
