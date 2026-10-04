"""Log-sum-exp merge for CVSA verification.

Merges draft (sparse) and complement attention outputs per-head.
For accepted heads: just normalize draft.
For rejected heads: online softmax merge → numerically exact dense equivalent.
"""

import torch


def selective_lse_merge(
    o_draft: torch.Tensor,     # [B, H_q, D] float32 unnormalized
    m_draft: torch.Tensor,     # [B, H_q] float32
    l_draft: torch.Tensor,     # [B, H_q] float32
    o_comp: torch.Tensor,      # [B, H_q, D] float32 unnormalized
    m_comp: torch.Tensor,      # [B, H_q] float32
    l_comp: torch.Tensor,      # [B, H_q] float32
    accept_mask: torch.Tensor, # [B, H_kv] bool
    num_q_per_kv: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Merge draft and complement attention with per-head accept/reject.

    Accepted heads: O = O_draft / L_draft (sparse result)
    Rejected heads: O = LSE_merge(draft, complement) (exact dense equivalent)

    Returns: [B, H_q, D] in output_dtype
    """
    B, H_q, D = o_draft.shape

    # Expand accept_mask from KV-head to Q-head granularity
    # [B, H_kv] → [B, H_q] via repeat_interleave
    accept_q = accept_mask.repeat_interleave(num_q_per_kv, dim=1)  # [B, H_q]

    # Accepted heads: simple normalization
    l_safe = l_draft.clamp(min=1e-6)
    o_accepted = o_draft / l_safe.unsqueeze(-1)  # [B, H_q, D]

    # Rejected heads: online softmax merge (numerically stable)
    m_all = torch.maximum(m_draft, m_comp)                     # [B, H_q]
    scale_d = (m_draft - m_all).exp()                          # [B, H_q]
    scale_c = (m_comp - m_all).exp()                           # [B, H_q]

    l_merged = l_draft * scale_d + l_comp * scale_c            # [B, H_q]
    o_merged = (o_draft * scale_d.unsqueeze(-1) +
                o_comp * scale_c.unsqueeze(-1))                # [B, H_q, D]

    l_merged_safe = l_merged.clamp(min=1e-6)
    o_rejected = o_merged / l_merged_safe.unsqueeze(-1)        # [B, H_q, D]

    # Select per head
    output = torch.where(accept_q.unsqueeze(-1), o_accepted, o_rejected)

    return output.to(output_dtype)
