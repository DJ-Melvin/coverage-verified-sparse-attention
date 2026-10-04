"""Long-context coverage diagnostics for approximate recovery analysis.

Collects per-head coverage at multiple budget levels (5%–30%) on long contexts
(32K, 64K, 128K). The key new analysis: **conditional coverage** — for heads
rejected at 5% budget, what is their coverage at larger budgets?

This answers: "If we expand from 5% to X% on reject, how many heads are rescued?"

Usage:
    cd /path/to/cvsa
    python eval/diagnostics_longctx.py \
        --context-len 32768 \
        --num-samples 3 \
        --max-decode 64 \
        --output results/diagnostics_longctx_32k.pt
"""

import sys
import os
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvsa.kernels.page_scores import triton_score_pages
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from eval.longbench_harness import E2EGenerator, DATASET2PROMPT


BUDGETS = (0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TAU_VALUES = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97)


class LongCtxCoverageCollector:
    """Collects per-head coverage at multiple budgets. Stores raw per-head
    coverage tensors so we can do conditional analysis offline."""

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size,
                 num_layers, budgets=BUDGETS):
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.budgets = budgets

        self._dense = DenseAttentionBackend(
            num_kv_heads, num_q_heads, head_dim, block_size)

        # Raw coverage: [layer][step] = {budget: coverage [B, H_kv]}
        self.coverage_raw: Dict[int, List[Dict[float, torch.Tensor]]] = {
            l: [] for l in range(num_layers)
        }
        self._step = 0

    def reset(self):
        self._dense.reset()
        self.coverage_raw = {l: [] for l in range(self.num_layers)}
        self._step = 0

    def step_done(self, layer_idx, num_layers):
        self._dense.step_done(layer_idx, num_layers)
        if layer_idx == num_layers - 1:
            self._step += 1

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        output = self._dense(q, k_cache, v_cache, block_tables, context_lens,
                             layer_idx)

        n_pages = block_tables.shape[1]
        if n_pages < 3:
            return output

        page_scores = triton_score_pages(
            q, k_cache, block_tables, context_lens,
            self.num_kv_heads, self.num_q_per_kv, self.block_size,
        )  # [B, H_kv, max_pages]

        lse_all = torch.logsumexp(page_scores, dim=-1)  # [B, H_kv]

        B = q.shape[0]
        n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1

        step_cov = {}
        for budget_pct in self.budgets:
            k = max(3, int(n_pages * budget_pct))
            k = min(k, n_pages)

            topk_vals, _ = torch.topk(page_scores, k=k, dim=-1,
                                      largest=True, sorted=False)

            idx_range = torch.arange(k, device=q.device)
            head_mask = idx_range.unsqueeze(0).unsqueeze(0) < \
                n_valid.unsqueeze(1).unsqueeze(2).clamp(max=k)
            masked = topk_vals.masked_fill(~head_mask, float('-inf'))

            lse_sel = torch.logsumexp(masked, dim=-1)
            coverage = (lse_sel - lse_all).exp()  # [B, H_kv]
            step_cov[budget_pct] = coverage.detach().cpu()

        self.coverage_raw[layer_idx].append(step_cov)
        return output


def analyze_conditional_coverage(collector, base_budget=0.05, tau=0.90):
    """For heads rejected at base_budget, what's their coverage at larger budgets?

    Returns:
        conditional: {expand_budget: {
            'rescue_rate': fraction of rejected heads now accepted,
            'mean_coverage': mean coverage of originally-rejected heads,
            'per_layer_rescue': [layer] rescue rates,
        }}
        Also returns unconditional stats for comparison.
    """
    num_layers = collector.num_layers
    budgets = collector.budgets

    # Stack all coverage data: [steps, layers, H_kv] per budget
    cov_by_budget = {}
    for budget in budgets:
        layers = []
        for layer in range(num_layers):
            data = collector.coverage_raw[layer]
            if not data:
                continue
            layer_cov = torch.stack([d[budget] for d in data])  # [steps, B, H_kv]
            layer_cov = layer_cov[:, 0, :]  # [steps, H_kv] (B=1)
            layers.append(layer_cov)
        if layers:
            cov_by_budget[budget] = torch.stack(layers, dim=1)  # [steps, layers, H_kv]

    if base_budget not in cov_by_budget:
        return {}, {}

    base_cov = cov_by_budget[base_budget]  # [steps, layers, H_kv]
    rejected_mask = base_cov < tau  # True = rejected at base budget

    n_rejected = rejected_mask.sum().item()
    n_total = rejected_mask.numel()

    # Unconditional acceptance rates
    unconditional = {}
    for budget in budgets:
        if budget not in cov_by_budget:
            continue
        cov = cov_by_budget[budget]
        accepted = (cov >= tau).float()
        per_layer = accepted.mean(dim=(0, 2))  # [layers]
        unconditional[budget] = {
            'acceptance_rate': accepted.mean().item(),
            'per_layer_acceptance': per_layer.tolist(),
            'mean_coverage': cov.mean().item(),
        }

    # Conditional: among rejected heads at base_budget, what happens at larger budgets?
    conditional = {}
    for budget in budgets:
        if budget <= base_budget or budget not in cov_by_budget:
            continue

        cov = cov_by_budget[budget]
        # Coverage of originally-rejected heads at this expanded budget
        rejected_cov = cov[rejected_mask]  # [N_rejected]

        if len(rejected_cov) == 0:
            continue

        rescued = (rejected_cov >= tau).float()
        rescue_rate = rescued.mean().item()
        mean_cov = rejected_cov.mean().item()

        # Per-layer rescue rate
        per_layer_rescue = []
        for layer in range(num_layers):
            layer_mask = rejected_mask[:, layer, :]  # [steps, H_kv]
            layer_rejected_cov = cov[:, layer, :][layer_mask]
            if len(layer_rejected_cov) == 0:
                per_layer_rescue.append(1.0)  # no rejections = all rescued
            else:
                per_layer_rescue.append(
                    (layer_rejected_cov >= tau).float().mean().item())

        # Coverage distribution of rejected heads
        cov_percentiles = torch.quantile(
            rejected_cov.float(),
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        ).tolist()

        conditional[budget] = {
            'rescue_rate': rescue_rate,
            'mean_coverage': mean_cov,
            'per_layer_rescue': per_layer_rescue,
            'coverage_percentiles': {
                'p10': cov_percentiles[0],
                'p25': cov_percentiles[1],
                'p50': cov_percentiles[2],
                'p75': cov_percentiles[3],
                'p90': cov_percentiles[4],
            },
        }

    return conditional, unconditional


def print_analysis(conditional, unconditional, base_budget, tau, ctx_len):
    """Print formatted results."""
    print(f"\n{'=' * 80}")
    print(f"COVERAGE ANALYSIS — Context {ctx_len//1024}K, base_budget={base_budget*100:.0f}%, tau={tau}")
    print(f"{'=' * 80}")

    # Unconditional
    print(f"\n--- Unconditional Acceptance Rates (tau={tau}) ---")
    print(f"{'Budget':>8s}  {'Accept':>8s}  {'Reject':>8s}  {'Mean Cov':>10s}")
    for budget in sorted(unconditional.keys()):
        u = unconditional[budget]
        print(f"{budget*100:>7.0f}%  {u['acceptance_rate']*100:>7.1f}%  "
              f"{(1-u['acceptance_rate'])*100:>7.1f}%  {u['mean_coverage']:>9.4f}")

    # Conditional
    if not conditional:
        print("\nNo rejections at base budget — nothing to analyze.")
        return

    base_reject_rate = 1 - unconditional.get(base_budget, {}).get('acceptance_rate', 1)
    print(f"\n--- Conditional: Heads rejected at {base_budget*100:.0f}% "
          f"({base_reject_rate*100:.1f}% of all heads) ---")
    print(f"{'Expand to':>10s}  {'Rescue':>8s}  {'Still rej':>10s}  "
          f"{'Mean cov':>10s}  {'p50 cov':>8s}  {'p90 cov':>8s}")
    for budget in sorted(conditional.keys()):
        c = conditional[budget]
        still_rej = (1 - c['rescue_rate']) * base_reject_rate
        pct = c['coverage_percentiles']
        print(f"{budget*100:>9.0f}%  {c['rescue_rate']*100:>7.1f}%  "
              f"{still_rej*100:>9.2f}%  "
              f"{c['mean_coverage']:>9.4f}  {pct['p50']:>7.4f}  {pct['p90']:>7.4f}")

    # Per-layer view: worst layers
    print(f"\n--- Per-layer rescue rates (worst 5 layers) ---")
    print(f"{'Layer':>6s}", end="")
    for budget in sorted(conditional.keys()):
        print(f"  {'→'+str(int(budget*100))+'%':>7s}", end="")
    print()
    # Find worst layers at the smallest expand budget
    first_expand = min(conditional.keys())
    worst_layers = sorted(
        range(len(conditional[first_expand]['per_layer_rescue'])),
        key=lambda l: conditional[first_expand]['per_layer_rescue'][l]
    )[:8]
    for layer in worst_layers:
        print(f"  L{layer:<3d}", end="")
        for budget in sorted(conditional.keys()):
            r = conditional[budget]['per_layer_rescue'][layer]
            print(f"  {r*100:>6.1f}%", end="")
        print()


def get_long_prompts(tokenizer, target_len, num_samples):
    """Get prompts at target context length.

    For <=64K: use real LongBench narrativeqa (longest natural prompts).
    For 128K: concatenate multiple long prompts.
    """
    from datasets import load_dataset

    prompts = []

    # Collect long prompts from narrativeqa + gov_report
    candidates = []
    for ds_name in ["narrativeqa", "gov_report", "musique", "hotpotqa"]:
        try:
            data = load_dataset("THUDM/LongBench", ds_name, split="test",
                                trust_remote_code=True)
            fmt = DATASET2PROMPT[ds_name]
            for i in range(min(50, len(data))):
                prompt = fmt.format(**data[i])
                tlen = len(tokenizer.encode(prompt))
                candidates.append((ds_name, i, prompt, tlen))
        except Exception as e:
            print(f"  Warning: skipping {ds_name}: {e}")

    candidates.sort(key=lambda x: -x[3])

    if target_len <= 65536:
        # Use prompts closest to target length (slightly over)
        for ds_name, i, prompt, tlen in candidates:
            if tlen < target_len * 0.7:
                continue
            # Truncate to target
            tokens = tokenizer.encode(prompt)
            if len(tokens) > target_len:
                half = target_len // 2
                prompt = (tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                          tokenizer.decode(tokens[-half:], skip_special_tokens=True))
            tlen = len(tokenizer.encode(prompt))
            if tlen > target_len * 0.5:
                prompts.append((f"{ds_name}#{i}", prompt, tlen))
            if len(prompts) >= num_samples:
                break

        # If not enough, use shorter prompts padded to target
        if len(prompts) < num_samples:
            for ds_name, i, prompt, tlen in candidates:
                if len(prompts) >= num_samples:
                    break
                tokens = tokenizer.encode(prompt)
                if len(tokens) > target_len:
                    half = target_len // 2
                    prompt = (tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                              tokenizer.decode(tokens[-half:], skip_special_tokens=True))
                tlen = len(tokenizer.encode(prompt))
                if tlen > target_len * 0.3:
                    prompts.append((f"{ds_name}#{i}", prompt, tlen))
    else:
        # 128K: concatenate multiple long prompts
        for s in range(num_samples):
            combined = ""
            combined_len = 0
            idx = s  # start from different offset for diversity
            while combined_len < target_len and idx < len(candidates):
                _, _, prompt, tlen = candidates[idx]
                combined += "\n\n---\n\n" + prompt
                combined_len = len(tokenizer.encode(combined))
                idx += 1

            # Truncate to target
            tokens = tokenizer.encode(combined)
            if len(tokens) > target_len:
                half = target_len // 2
                combined = (
                    tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                    tokenizer.decode(tokens[-half:], skip_special_tokens=True))
            tlen = len(tokenizer.encode(combined))
            prompts.append((f"concat#{s}", combined, tlen))

    return prompts


def run(args):
    print(f"Loading model...")
    model = ModelForDecode(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype=torch.bfloat16, device="cuda")
    config = model.config
    generator = E2EGenerator(model, block_size=16)

    ctx_len = args.context_len
    print(f"\nTarget context: {ctx_len//1024}K, samples={args.num_samples}, "
          f"decode={args.max_decode}")

    # Get prompts
    prompts = get_long_prompts(model.tokenizer, ctx_len, args.num_samples)
    print(f"Got {len(prompts)} prompts")
    for name, _, tlen in prompts:
        print(f"  {name}: {tlen} tokens ({tlen/1024:.1f}K)")

    # Collect coverage
    all_conditional = []
    all_unconditional = []

    for idx, (name, prompt, tlen) in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] {name} ({tlen} tokens, "
              f"{(tlen+15)//16} pages)")

        collector = LongCtxCoverageCollector(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=16,
            num_layers=config.num_layers,
            budgets=BUDGETS,
        )

        _ = generator.generate(prompt, args.max_decode, collector)

        # Analyze
        for base_b in [0.03, 0.05]:
            for tau in [0.85, 0.90, 0.95]:
                cond, uncond = analyze_conditional_coverage(
                    collector, base_budget=base_b, tau=tau)
                print_analysis(cond, uncond, base_b, tau, ctx_len)

                all_conditional.append({
                    'sample': name, 'ctx_len': tlen,
                    'base_budget': base_b, 'tau': tau,
                    'conditional': cond, 'unconditional': uncond,
                })

        # Also save raw per-budget acceptance for the main tau=0.90
        _, uncond_main = analyze_conditional_coverage(
            collector, base_budget=0.05, tau=0.90)
        all_unconditional.append({
            'sample': name, 'ctx_len': tlen,
            'unconditional': uncond_main,
        })

        # Free memory
        del collector
        torch.cuda.empty_cache()

    # Aggregate conditional results across samples (for main config: base=5%, tau=0.90)
    print(f"\n\n{'#' * 80}")
    print(f"AGGREGATED RESULTS — {ctx_len//1024}K context")
    print(f"{'#' * 80}")

    main_conds = [r for r in all_conditional
                  if r['base_budget'] == 0.05 and r['tau'] == 0.90]
    if main_conds:
        # Average rescue rates
        expand_budgets = set()
        for r in main_conds:
            expand_budgets.update(r['conditional'].keys())

        print(f"\n--- Avg Conditional Rescue (base=5%, tau=0.90, n={len(main_conds)}) ---")
        print(f"{'Expand':>8s}  {'Rescue':>8s}  {'Mean cov':>10s}  {'p50':>8s}  {'p90':>8s}")
        for budget in sorted(expand_budgets):
            rescues = [r['conditional'][budget]['rescue_rate']
                       for r in main_conds if budget in r['conditional']]
            covs = [r['conditional'][budget]['mean_coverage']
                    for r in main_conds if budget in r['conditional']]
            p50s = [r['conditional'][budget]['coverage_percentiles']['p50']
                    for r in main_conds if budget in r['conditional']]
            p90s = [r['conditional'][budget]['coverage_percentiles']['p90']
                    for r in main_conds if budget in r['conditional']]
            print(f"{budget*100:>7.0f}%  {np.mean(rescues)*100:>7.1f}%  "
                  f"{np.mean(covs):>9.4f}  {np.mean(p50s):>7.4f}  {np.mean(p90s):>7.4f}")

    # Save
    out = {
        'context_len': ctx_len,
        'budgets': list(BUDGETS),
        'tau_values': list(TAU_VALUES),
        'all_conditional': all_conditional,
        'all_unconditional': all_unconditional,
        'args': vars(args),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Long-context coverage diagnostics for approximate recovery")
    parser.add_argument("--context-len", type=int, default=32768,
                        help="Target context length in tokens")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of prompts to evaluate")
    parser.add_argument("--max-decode", type=int, default=64,
                        help="Decode steps per sample")
    parser.add_argument("--output", type=str,
                        default="results/diagnostics_longctx.pt")
    args = parser.parse_args()
    run(args)
