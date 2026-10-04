"""Tests for complement page set computation."""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvsa.complement import compute_complement_pages


def test_basic():
    """Selected + complement = all valid pages, no overlap."""
    B, H_kv = 1, 2
    block_size = 16
    context_lens = torch.tensor([160], device="cuda")  # 10 pages
    n_pages = 10

    # Select pages [0, 2, 4] for head 0; [1, 3] for head 1
    page_indices = torch.zeros(B, H_kv, 3, dtype=torch.int32, device="cuda")
    page_indices[0, 0] = torch.tensor([0, 2, 4])
    page_indices[0, 1] = torch.tensor([1, 3, 0])  # padding at idx 2
    n_per_head = torch.tensor([[3, 2]], dtype=torch.int32, device="cuda")

    comp_idx, n_comp = compute_complement_pages(
        page_indices, n_per_head, context_lens, block_size)

    # Head 0: complement = {1,3,5,6,7,8,9} → 7 pages
    assert n_comp[0, 0].item() == 7, f"Expected 7, got {n_comp[0,0].item()}"
    # Head 1: complement = {0,2,4,5,6,7,8,9} → 8 pages
    assert n_comp[0, 1].item() == 8, f"Expected 8, got {n_comp[0,1].item()}"

    # Verify no overlap: selected ∩ complement = ∅
    for h in range(H_kv):
        n_sel = n_per_head[0, h].item()
        selected = set(page_indices[0, h, :n_sel].tolist())
        n_c = n_comp[0, h].item()
        complement = set(comp_idx[0, h, :n_c].tolist())
        assert len(selected & complement) == 0, f"Head {h}: overlap found"
        assert len(selected | complement) == n_pages, f"Head {h}: union != all"

    print("  test_basic PASSED")


def test_accept_mask():
    """Accepted heads get n_complement=0."""
    B, H_kv = 1, 4
    block_size = 16
    context_lens = torch.tensor([80], device="cuda")  # 5 pages

    page_indices = torch.zeros(B, H_kv, 2, dtype=torch.int32, device="cuda")
    page_indices[0, :, 0] = torch.tensor([0, 1, 2, 3])
    page_indices[0, :, 1] = torch.tensor([1, 2, 3, 4])
    n_per_head = torch.full((B, H_kv), 2, dtype=torch.int32, device="cuda")

    # Accept heads 0 and 2, reject heads 1 and 3
    accept_mask = torch.tensor([[True, False, True, False]], device="cuda")

    comp_idx, n_comp = compute_complement_pages(
        page_indices, n_per_head, context_lens, block_size, accept_mask)

    assert n_comp[0, 0].item() == 0, "Accepted head 0 should have 0 complement"
    assert n_comp[0, 1].item() == 3, "Rejected head 1 should have 3 complement"
    assert n_comp[0, 2].item() == 0, "Accepted head 2 should have 0 complement"
    assert n_comp[0, 3].item() == 3, "Rejected head 3 should have 3 complement"

    print("  test_accept_mask PASSED")


def test_full_budget():
    """When all pages selected, complement is empty."""
    B, H_kv = 1, 1
    block_size = 16
    context_lens = torch.tensor([48], device="cuda")  # 3 pages

    page_indices = torch.tensor([[[0, 1, 2]]], dtype=torch.int32, device="cuda")
    n_per_head = torch.tensor([[3]], dtype=torch.int32, device="cuda")

    comp_idx, n_comp = compute_complement_pages(
        page_indices, n_per_head, context_lens, block_size)

    assert n_comp[0, 0].item() == 0, f"Expected 0, got {n_comp[0,0].item()}"
    print("  test_full_budget PASSED")


def test_batch():
    """Multiple sequences in a batch."""
    B, H_kv = 2, 1
    block_size = 16
    context_lens = torch.tensor([80, 48], device="cuda")  # 5 and 3 pages

    page_indices = torch.zeros(B, H_kv, 2, dtype=torch.int32, device="cuda")
    page_indices[0, 0] = torch.tensor([0, 1])
    page_indices[1, 0] = torch.tensor([0, 1])
    n_per_head = torch.tensor([[2], [2]], dtype=torch.int32, device="cuda")

    comp_idx, n_comp = compute_complement_pages(
        page_indices, n_per_head, context_lens, block_size)

    assert n_comp[0, 0].item() == 3, f"Seq 0: expected 3, got {n_comp[0,0].item()}"
    assert n_comp[1, 0].item() == 1, f"Seq 1: expected 1, got {n_comp[1,0].item()}"
    print("  test_batch PASSED")


if __name__ == "__main__":
    print("Running complement tests...")
    test_basic()
    test_accept_mask()
    test_full_budget()
    test_batch()
    print("All complement tests PASSED!")
