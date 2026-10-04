"""Tests for LSE merge: verify merged output matches dense attention."""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvsa.lse_merge import selective_lse_merge


def _dense_attention(q, k, v):
    """q: [B,H,D], k: [B,H,N,D], v: [B,H,N,D] → [B,H,D]"""
    D = q.shape[-1]
    scores = torch.einsum("bhd,bhnd->bhn", q, k) / (D ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhn,bhnd->bhd", weights, v)


def _unnorm_attention(q, k, v):
    """q: [B,H,D], k: [B,H,N,D], v: [B,H,N,D] → (O[B,H,D], M[B,H], L[B,H])"""
    D = q.shape[-1]
    scores = torch.einsum("bhd,bhnd->bhn", q, k) / (D ** 0.5)
    M = scores.max(dim=-1).values
    exp_s = (scores - M.unsqueeze(-1)).exp()
    L = exp_s.sum(dim=-1)
    O = torch.einsum("bhn,bhnd->bhd", exp_s, v)
    return O, M, L


def test_merge_equals_dense():
    """Draft (subset) + complement = dense (all), via LSE merge."""
    torch.manual_seed(42)
    B, H_q, D, N = 1, 4, 64, 20
    H_kv = 2
    G = H_q // H_kv

    q = torch.randn(B, H_q, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H_q, N, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H_q, N, D, device="cuda", dtype=torch.float32)

    o_dense = _dense_attention(q, k, v)

    o_d, m_d, l_d = _unnorm_attention(q, k[:, :, :5], v[:, :, :5])
    o_c, m_c, l_c = _unnorm_attention(q, k[:, :, 5:], v[:, :, 5:])

    # All rejected → merged should = dense
    accept_mask = torch.zeros(B, H_kv, dtype=torch.bool, device="cuda")
    merged = selective_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, accept_mask, G, torch.float32)

    error = (merged - o_dense).abs().max().item()
    assert error < 1e-4, f"Merge error too large: {error}"
    print(f"  test_merge_equals_dense PASSED (max error: {error:.2e})")


def test_accepted_heads_use_draft():
    """Accepted heads return normalized draft; rejected heads match dense."""
    torch.manual_seed(42)
    B, H_kv, D, N = 1, 2, 64, 20
    G = 2
    H_q = H_kv * G

    # GQA: K/V at KV-head granularity, expanded for Q heads
    k_kv = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32)
    v_kv = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32)
    k = k_kv.repeat_interleave(G, dim=1)
    v = v_kv.repeat_interleave(G, dim=1)
    q = torch.randn(B, H_q, D, device="cuda", dtype=torch.float32)

    o_d, m_d, l_d = _unnorm_attention(q, k[:, :, :5], v[:, :, :5])
    o_c, m_c, l_c = _unnorm_attention(q, k[:, :, 5:], v[:, :, 5:])

    # Accept KV-head 0 (Q-heads 0,1), reject KV-head 1 (Q-heads 2,3)
    accept_mask = torch.tensor([[True, False]], device="cuda")
    merged = selective_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, accept_mask, G, torch.float32)

    # Accepted Q-heads should match draft-only
    draft_norm = o_d / l_d.clamp(min=1e-6).unsqueeze(-1)
    err_acc = (merged[:, :G] - draft_norm[:, :G]).abs().max().item()
    assert err_acc < 1e-6, f"Accepted heads error: {err_acc}"

    # Rejected Q-heads should match dense
    o_dense = _dense_attention(q, k, v)
    err_rej = (merged[:, G:] - o_dense[:, G:]).abs().max().item()
    assert err_rej < 1e-4, f"Rejected heads error: {err_rej}"

    print(f"  test_accepted_heads_use_draft PASSED (acc={err_acc:.2e}, rej={err_rej:.2e})")


def test_all_accepted():
    """When all heads accepted, result = normalized draft."""
    torch.manual_seed(42)
    B, H_q, D = 1, 4, 64
    H_kv, G = 2, 2

    o_d = torch.randn(B, H_q, D, device="cuda", dtype=torch.float32)
    m_d = torch.randn(B, H_q, device="cuda", dtype=torch.float32)
    l_d = torch.rand(B, H_q, device="cuda", dtype=torch.float32) + 0.1

    o_c = torch.zeros_like(o_d)
    m_c = torch.full_like(m_d, float('-inf'))
    l_c = torch.zeros_like(l_d)

    accept_mask = torch.ones(B, H_kv, dtype=torch.bool, device="cuda")
    merged = selective_lse_merge(o_d, m_d, l_d, o_c, m_c, l_c, accept_mask, G, torch.float32)

    expected = o_d / l_d.unsqueeze(-1)
    error = (merged - expected).abs().max().item()
    assert error < 1e-6, f"All-accepted error: {error}"
    print(f"  test_all_accepted PASSED (error: {error:.2e})")


if __name__ == "__main__":
    print("Running LSE merge tests...")
    test_merge_equals_dense()
    test_accepted_heads_use_draft()
    test_all_accepted()
    print("All LSE merge tests PASSED!")
