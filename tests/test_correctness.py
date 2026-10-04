#!/usr/bin/env python3
"""Correctness test: CVSA output vs dense on real Llama-3.1-8B.

Uses a LongBench prompt (long context) to ensure meaningful coverage dynamics.
Tests tau=1.0 (always verify → should match dense exactly).

Usage:
    cd /path/to/cvsa
    python tests/test_correctness.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from eval.longbench_harness import E2EGenerator, DATASET2PROMPT
from cvsa.backend import CVSABackend


def test_with_long_context():
    """CVSA correctness on a real long-context prompt."""
    print("Loading model...")
    model = ModelForDecode(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype=torch.bfloat16, device="cuda")
    config = model.config
    generator = E2EGenerator(model, block_size=16)

    # Use a LongBench prompt for long context
    from datasets import load_dataset
    data = load_dataset("THUDM/LongBench", "hotpotqa", split="test",
                        trust_remote_code=True)
    prompt_format = DATASET2PROMPT["hotpotqa"]
    prompt = prompt_format.format(**data[0])

    # Truncate to manageable size
    tokens = model.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    max_len = 4000
    if len(tokens) > max_len:
        half = max_len // 2
        prompt = (model.tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                  model.tokenizer.decode(tokens[-half:], skip_special_tokens=True))

    ctx_len = len(model.tokenizer.encode(prompt))
    n_pages = (ctx_len + 15) // 16
    max_new = 20
    print(f"Context: {ctx_len} tokens, {n_pages} pages, generating {max_new} tokens")

    # 1. Dense baseline
    print("\n--- Dense ---")
    dense_backend = DenseAttentionBackend(
        config.num_kv_heads, config.num_q_heads, config.head_dim, 16)
    dense_text = generator.generate(prompt, max_new, dense_backend)
    print(f"  Output: {dense_text[:80]!r}...")

    # 2. CVSA with tau=1.0 (always verify → should produce exact dense)
    print("\n--- CVSA tau=1.0 (always verify) ---")
    cvsa_verify = CVSABackend(
        num_kv_heads=config.num_kv_heads,
        num_q_heads=config.num_q_heads,
        head_dim=config.head_dim,
        block_size=16,
        num_layers=config.num_layers,
        budget_pct=0.10,
        coverage_threshold=1.0,  # coverage < 1.0 always → always reject → always verify
    )
    cvsa_verify_text = generator.generate(prompt, max_new, cvsa_verify)
    stats = cvsa_verify.get_stats()
    print(f"  Output: {cvsa_verify_text[:80]!r}...")
    print(f"  Stats: all_accepted={stats['all_accepted_pct']:.0f}%, "
          f"per_head_accept={stats['per_head_accept_rate']:.1f}%, "
          f"scoring={stats['total_scoring_steps']}")

    # Check match
    dense_toks = model.tokenizer.encode(dense_text)
    verify_toks = model.tokenizer.encode(cvsa_verify_text)
    match = sum(1 for a, b in zip(dense_toks, verify_toks) if a == b)
    total = max(len(dense_toks), len(verify_toks))
    print(f"  Token match: {match}/{total}")
    if dense_text == cvsa_verify_text:
        print("  ✓ EXACT MATCH with dense")
    else:
        first_diff = next((i for i, (a, b) in enumerate(zip(dense_toks, verify_toks)) if a != b), -1)
        print(f"  ✗ First divergence at token {first_diff}")

    # 3. CVSA with tau=0.90 (normal operation)
    print("\n--- CVSA tau=0.90 (normal) ---")
    cvsa_normal = CVSABackend(
        num_kv_heads=config.num_kv_heads,
        num_q_heads=config.num_q_heads,
        head_dim=config.head_dim,
        block_size=16,
        num_layers=config.num_layers,
        budget_pct=0.10,
        coverage_threshold=0.90,
    )
    cvsa_text = generator.generate(prompt, max_new, cvsa_normal)
    stats = cvsa_normal.get_stats()
    print(f"  Output: {cvsa_text[:80]!r}...")
    print(f"  Stats: all_accepted={stats['all_accepted_pct']:.0f}%, "
          f"per_head_accept={stats['per_head_accept_rate']:.1f}%, "
          f"scoring={stats['total_scoring_steps']}")

    # 4. CVSA with tau=0.50 (aggressive)
    print("\n--- CVSA tau=0.50 (aggressive, mostly accept) ---")
    cvsa_aggressive = CVSABackend(
        num_kv_heads=config.num_kv_heads,
        num_q_heads=config.num_q_heads,
        head_dim=config.head_dim,
        block_size=16,
        num_layers=config.num_layers,
        budget_pct=0.10,
        coverage_threshold=0.50,
    )
    cvsa_agg_text = generator.generate(prompt, max_new, cvsa_aggressive)
    stats_agg = cvsa_aggressive.get_stats()
    print(f"  Output: {cvsa_agg_text[:80]!r}...")
    print(f"  Stats: all_accepted={stats_agg['all_accepted_pct']:.0f}%, "
          f"per_head_accept={stats_agg['per_head_accept_rate']:.1f}%")

    # 5. CVSA with tau=0.95 (conservative)
    print("\n--- CVSA tau=0.95 (conservative) ---")
    cvsa_conservative = CVSABackend(
        num_kv_heads=config.num_kv_heads,
        num_q_heads=config.num_q_heads,
        head_dim=config.head_dim,
        block_size=16,
        num_layers=config.num_layers,
        budget_pct=0.10,
        coverage_threshold=0.95,
    )
    cvsa_cons_text = generator.generate(prompt, max_new, cvsa_conservative)
    stats_cons = cvsa_conservative.get_stats()
    print(f"  Output: {cvsa_cons_text[:80]!r}...")
    print(f"  Stats: all_accepted={stats_cons['all_accepted_pct']:.0f}%, "
          f"per_head_accept={stats_cons['per_head_accept_rate']:.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    test_with_long_context()
