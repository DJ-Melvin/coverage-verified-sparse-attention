#!/usr/bin/env python3
"""Unit tests for the effective-budget instrumentation.

Tests cover the accounting logic that does not require running the actual
attention kernels:
  - _record_work accumulation + lazy init
  - reset() clears effective-budget tensors
  - get_stats() exposes effective_budget_pct
  - make_backend parses a CVSA method string (and its ssa_ alias)

Usage:
    cd /path/to/cvsa
    python tests/test_effective_budget.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cvsa.backend import CVSABackend


def _make_backend(**kw):
    defaults = dict(
        num_kv_heads=4, num_q_heads=32, head_dim=128, block_size=16,
        num_layers=32, budget_pct=0.05, coverage_threshold=0.90,
    )
    defaults.update(kw)
    return CVSABackend(**defaults)


def test_record_work_lazy_init_and_accumulate():
    """_record_work creates tensors on first call and accumulates correctly."""
    b = _make_backend()
    assert b._attn_work is None
    assert b._attn_work_max is None

    device = "cpu"  # test with CPU; real runs are on CUDA
    context_lens = torch.tensor([64, 128], dtype=torch.int32, device=device)
    # n_valid = ceil(cl / 16) computed as ((cl-1)/16) + 1 = [4, 8]
    # dense_max = (4+8) * num_kv_heads(4) = 48
    pages_attended = torch.tensor(10, dtype=torch.long, device=device)
    b._record_work(context_lens, pages_attended)

    assert b._attn_work is not None
    assert b._attn_work_max is not None
    assert int(b._attn_work.item()) == 10
    assert int(b._attn_work_max.item()) == 48

    # Second call accumulates
    b._record_work(context_lens, torch.tensor(5, dtype=torch.long, device=device))
    assert int(b._attn_work.item()) == 15
    assert int(b._attn_work_max.item()) == 96

    print("  PASS: _record_work lazy init + accumulate")


def test_get_stats_effective_budget():
    """get_stats() reports effective_budget_pct once work is recorded."""
    b = _make_backend()

    # Before any work: effective_budget_pct not present
    stats = b.get_stats()
    assert "effective_budget_pct" not in stats

    # After work: present and correct
    ctx = torch.tensor([64], dtype=torch.int32)  # n_valid=4, max=4*4=16
    b._record_work(ctx, torch.tensor(2, dtype=torch.long))  # attended=2
    stats = b.get_stats()
    assert "effective_budget_pct" in stats
    assert abs(stats["effective_budget_pct"] - (2 / 16 * 100)) < 1e-6
    assert stats["total_pages_attended"] == 2
    assert stats["total_pages_max"] == 16

    print("  PASS: get_stats effective_budget_pct")


def test_reset_clears_effective_budget():
    """reset() zeros the effective-budget tensors if already initialized."""
    b = _make_backend()
    ctx = torch.tensor([64], dtype=torch.int32)
    b._record_work(ctx, torch.tensor(5, dtype=torch.long))
    assert int(b._attn_work.item()) == 5

    b.reset()
    assert int(b._attn_work.item()) == 0
    assert int(b._attn_work_max.item()) == 0

    # Stats no longer report effective_budget_pct when work_max==0
    stats = b.get_stats()
    assert "effective_budget_pct" not in stats

    print("  PASS: reset clears effective budget")


def test_make_backend_parses_cvsa_method():
    """eval/longbench_cvsa.make_backend parses cvsa_5_90_c4_ar20.

    The legacy ssa_ prefix must build an identical backend, and an
    unrecognized suffix must raise rather than be silently dropped. (The
    paper configuration adds _dL0, whose dense layer needs CUDA to build.)

    Skipped gracefully if the eval module's heavy deps (transformers, etc.)
    aren't importable in this environment — the parse logic is exercised
    indirectly by any actual eval run.
    """
    try:
        from eval.longbench_cvsa import make_backend
    except ImportError as e:
        print(f"  SKIP: make_backend parse test (deps missing: {e})")
        return

    class Cfg:
        num_kv_heads = 4
        num_q_heads = 32
        head_dim = 128
        num_layers = 32

    b = make_backend("cvsa_5_90_c4_ar20", Cfg(), 16)
    assert b.budget_pct == 0.05
    assert b.tau == 0.90
    assert b.lazy
    assert b.check_interval == 4
    assert b.coverage_interval == 4
    assert b.approx_recovery_pct == 0.2
    assert b._dense_layers == set()

    alias = make_backend("ssa_5_90_c4_ar20", Cfg(), 16)
    assert vars(alias) == vars(b), "ssa_ alias must build the same backend"

    raised = False
    try:
        make_backend("cvsa_5_90_c4_ar20_bogus7", Cfg(), 16)
    except ValueError:
        raised = True
    assert raised, "unrecognized suffix should raise ValueError"

    print("  PASS: make_backend parses cvsa_5_90_c4_ar20")


if __name__ == "__main__":
    print("Testing effective-budget instrumentation...")
    test_record_work_lazy_init_and_accumulate()
    test_get_stats_effective_budget()
    test_reset_clears_effective_budget()
    test_make_backend_parses_cvsa_method()
    print("\nAll tests passed!")
