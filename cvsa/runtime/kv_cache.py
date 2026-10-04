"""Paged KV cache manager for standalone decode benchmark.

Allocates a pool of physical blocks and manages page tables per sequence.
Compatible with both FlashInfer and our Triton kernels.

Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
This matches vLLM's paged attention and our Triton kernel expectations.
"""

import torch
from dataclasses import dataclass


@dataclass
class CacheConfig:
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device


class PagedKVCache:
    """Simple paged KV cache with block allocation."""

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int,
                 head_dim: int, dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda"):
        self.config = CacheConfig(
            num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            dtype=dtype, device=torch.device(device),
        )
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Allocate cache pool
        self.k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        # Block allocation tracking
        self._next_free = 0
        self._total_blocks = num_blocks

    def allocate_blocks(self, n: int) -> torch.Tensor:
        """Allocate n contiguous physical blocks. Returns block IDs (int32)."""
        assert self._next_free + n <= self._total_blocks, \
            f"Out of blocks: need {n}, have {self._total_blocks - self._next_free}"
        block_ids = torch.arange(
            self._next_free, self._next_free + n,
            dtype=torch.int32, device=self.config.device,
        )
        self._next_free += n
        return block_ids

    def write_kv(self, block_id: int, token_offset: int,
                 k: torch.Tensor, v: torch.Tensor):
        """Write a single token's K/V into the cache.

        k, v: [num_kv_heads, head_dim]
        """
        self.k_cache[block_id, token_offset] = k
        self.v_cache[block_id, token_offset] = v

    def write_kv_block(self, block_id: int, k_block: torch.Tensor,
                       v_block: torch.Tensor, n_tokens: int):
        """Write a block of tokens' K/V into the cache.

        k_block, v_block: [n_tokens, num_kv_heads, head_dim]
        """
        self.k_cache[block_id, :n_tokens] = k_block[:n_tokens]
        self.v_cache[block_id, :n_tokens] = v_block[:n_tokens]

    def reset(self):
        """Free all blocks (does not zero memory for speed)."""
        self._next_free = 0


class SequenceKVManager:
    """Manages KV cache for a batch of sequences during prefill + decode."""

    def __init__(self, cache: PagedKVCache, batch_size: int, max_context_len: int):
        self.cache = cache
        self.batch_size = batch_size
        self.block_size = cache.block_size
        self.max_pages = (max_context_len + cache.block_size - 1) // cache.block_size

        # Block table: [B, max_pages] maps logical page -> physical block
        self.block_tables = torch.zeros(
            batch_size, self.max_pages,
            dtype=torch.int32, device=cache.config.device,
        )

        # Context lengths: [B]
        self.context_lens = torch.zeros(
            batch_size, dtype=torch.int32, device=cache.config.device,
        )

        # Track allocated pages per sequence
        self._pages_allocated = [0] * batch_size

    def prefill_kv(self, seq_idx: int, k_all: torch.Tensor, v_all: torch.Tensor):
        """Store prefill KV for one sequence.

        k_all, v_all: [seq_len, num_kv_heads, head_dim]
        """
        seq_len = k_all.shape[0]
        n_pages = (seq_len + self.block_size - 1) // self.block_size

        block_ids = self.cache.allocate_blocks(n_pages)
        self.block_tables[seq_idx, :n_pages] = block_ids

        for p in range(n_pages):
            start = p * self.block_size
            end = min(start + self.block_size, seq_len)
            n_tokens = end - start
            self.cache.write_kv_block(
                block_ids[p].item(),
                k_all[start:end], v_all[start:end], n_tokens,
            )

        self.context_lens[seq_idx] = seq_len
        self._pages_allocated[seq_idx] = n_pages

    def append_token_kv(self, seq_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append one decode token's KV to the cache.

        k, v: [num_kv_heads, head_dim]
        """
        ctx_len = self.context_lens[seq_idx].item()
        page_idx = ctx_len // self.block_size
        token_offset = ctx_len % self.block_size

        # Allocate new page if needed
        if token_offset == 0 and page_idx >= self._pages_allocated[seq_idx]:
            new_block = self.cache.allocate_blocks(1)
            self.block_tables[seq_idx, page_idx] = new_block[0]
            self._pages_allocated[seq_idx] += 1

        block_id = self.block_tables[seq_idx, page_idx].item()
        self.cache.write_kv(block_id, token_offset, k, v)
        self.context_lens[seq_idx] += 1
