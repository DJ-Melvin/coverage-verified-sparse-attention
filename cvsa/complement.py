"""Complement page set computation for CVSA verification.

Given selected pages (draft), computes the complement set (all valid pages
NOT selected) for use in the verification kernel.
"""

import torch
from typing import Tuple


def compute_complement_pages(
    page_indices: torch.Tensor,   # [B, H_kv, max_selected] int32
    n_per_head: torch.Tensor,     # [B, H_kv] int32
    context_lens: torch.Tensor,   # [B] int32
    block_size: int,
    accept_mask: torch.Tensor = None,  # [B, H_kv] bool — skip complement for accepted heads
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute complement page set: all_valid - selected.

    For accepted heads (accept_mask=True), sets n_complement=0 so the
    attention kernel will early-exit at zero cost.

    Returns:
        complement_indices: [B, H_kv, max_complement] int32
        n_complement: [B, H_kv] int32
    """
    B, H_kv, max_selected = page_indices.shape
    device = page_indices.device

    # Number of valid pages per sequence
    n_valid = ((context_lens.float() - 1) / block_size).long() + 1  # [B]
    max_pages = int(n_valid.max().item())

    # Build selected mask: [B, H_kv, max_pages]
    selected_mask = torch.zeros(B, H_kv, max_pages, dtype=torch.bool, device=device)

    # Mask for valid selection indices (within n_per_head)
    idx_range = torch.arange(max_selected, device=device)
    sel_valid = idx_range.unsqueeze(0).unsqueeze(0) < n_per_head.unsqueeze(-1)  # [B, H_kv, max_sel]

    # Clamp indices to valid range before scatter
    safe_indices = page_indices.long().clamp(0, max_pages - 1)
    selected_mask.scatter_(2, safe_indices, sel_valid)

    # Valid page mask: [B, max_pages] → [B, H_kv, max_pages]
    page_range = torch.arange(max_pages, device=device)
    valid_mask = page_range.unsqueeze(0) < n_valid.unsqueeze(1)  # [B, max_pages]
    valid_mask_3d = valid_mask.unsqueeze(1).expand(B, H_kv, max_pages)

    # Complement = valid AND NOT selected
    complement_mask = valid_mask_3d & ~selected_mask

    # Count complement pages per head
    n_complement = complement_mask.sum(dim=-1).to(torch.int32)  # [B, H_kv]

    # Zero out complement for accepted heads
    if accept_mask is not None:
        n_complement = torch.where(accept_mask, torch.zeros_like(n_complement), n_complement)

    max_complement = int(n_complement.max().item())
    if max_complement == 0:
        dummy = torch.zeros(B, H_kv, 1, dtype=torch.int32, device=device)
        return dummy, n_complement

    # Pack complement indices: sort descending so True (complement) comes first
    sorted_vals, sorted_idx = complement_mask.float().sort(dim=-1, descending=True)
    complement_indices = sorted_idx[:, :, :max_complement].to(torch.int32)

    return complement_indices, n_complement
