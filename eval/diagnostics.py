"""CVSA diagnostics: coverage distribution, acceptance rates, per-layer patterns.

Runs real model inference with dense attention, collects page scores as a
side channel, then analyzes coverage offline across budget and tau sweeps.

This determines the optimal tau and validates the CVSA cost model.

Usage:
    cd /path/to/cvsa
    python eval/diagnostics.py [--num-samples 3] [--max-decode 128] [--max-context 8000]
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


class CoverageCollectorBackend:
    """Wraps dense attention, collects per-layer coverage at each decode step.

    Uses dense attention for output (so generation is unaffected), but also
    computes page scores and coverage at each step for analysis.
    """

    def __init__(self, num_kv_heads, num_q_heads, head_dim, block_size,
                 num_layers, budgets=(0.05, 0.10, 0.15, 0.20)):
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.budgets = budgets

        self._dense = DenseAttentionBackend(
            num_kv_heads, num_q_heads, head_dim, block_size)

        # coverage_data[layer][step] = {budget: coverage_tensor [B, H_kv]}
        self.coverage_data: Dict[int, List[Dict[float, torch.Tensor]]] = {
            l: [] for l in range(num_layers)
        }
        self._step = 0

    def reset(self):
        self._dense.reset()
        self.coverage_data = {l: [] for l in range(self.num_layers)}
        self._step = 0

    def step_done(self, layer_idx, num_layers):
        self._dense.step_done(layer_idx, num_layers)
        if layer_idx == num_layers - 1:
            self._step += 1

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens,
                 layer_idx):
        # Dense output for correctness
        output = self._dense(q, k_cache, v_cache, block_tables, context_lens,
                             layer_idx)

        # Side channel: compute page scores and coverage
        n_pages = block_tables.shape[1]
        if n_pages < 3:
            return output

        page_scores = triton_score_pages(
            q, k_cache, block_tables, context_lens,
            self.num_kv_heads, self.num_q_per_kv, self.block_size,
        )  # [B, H_kv, max_pages]

        lse_all = torch.logsumexp(page_scores, dim=-1)  # [B, H_kv]

        step_cov = {}
        for budget_pct in self.budgets:
            k = max(3, int(n_pages * budget_pct))
            k = min(k, n_pages)

            topk_vals, _ = torch.topk(page_scores, k=k, dim=-1,
                                      largest=True, sorted=False)

            # Mask invalid pages
            B = q.shape[0]
            n_valid = ((context_lens.float() - 1) / self.block_size).long() + 1
            idx_range = torch.arange(k, device=q.device)
            head_mask = idx_range.unsqueeze(0).unsqueeze(0) < \
                n_valid.unsqueeze(1).unsqueeze(2).clamp(max=k)
            masked = topk_vals.masked_fill(~head_mask, float('-inf'))

            lse_sel = torch.logsumexp(masked, dim=-1)  # [B, H_kv]
            coverage = (lse_sel - lse_all).exp()  # [B, H_kv]
            step_cov[budget_pct] = coverage.detach().cpu()

        self.coverage_data[layer_idx].append(step_cov)
        return output


def analyze_coverage(collector: CoverageCollectorBackend,
                     budgets: Tuple[float, ...],
                     tau_values: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)):
    """Analyze collected coverage data.

    Returns dict with:
      - per_layer_mean_coverage[budget][layer] = mean coverage
      - acceptance_rate[budget][tau] = fraction of (step, layer, head) accepted
      - per_layer_acceptance[budget][tau][layer] = acceptance rate for that layer
      - per_layer_reject_rank[budget][tau] = layers sorted by rejection rate (most rejecting first)
      - coverage_std[budget][layer] = std of coverage (temporal stability)
    """
    num_layers = collector.num_layers
    num_kv_heads = collector.num_kv_heads
    results = {}

    for budget in budgets:
        # Collect all coverage values: [num_steps, num_layers, H_kv]
        all_cov = []
        for layer in range(num_layers):
            layer_data = collector.coverage_data[layer]
            if not layer_data:
                continue
            layer_covs = torch.stack([d[budget] for d in layer_data])  # [steps, B, H_kv]
            # Squeeze batch dim (B=1 typical)
            layer_covs = layer_covs[:, 0, :]  # [steps, H_kv]
            all_cov.append(layer_covs)

        if not all_cov:
            continue

        all_cov = torch.stack(all_cov, dim=1)  # [steps, layers, H_kv]
        n_steps = all_cov.shape[0]

        # Mean coverage per layer
        mean_cov = all_cov.mean(dim=(0, 2))  # [layers]

        # Std coverage per layer (temporal stability)
        std_cov = all_cov.std(dim=0).mean(dim=1)  # [layers]

        # Acceptance rates at various tau
        accept_rates = {}
        per_layer_accept = {}
        for tau in tau_values:
            accepted = (all_cov >= tau).float()

            # Overall acceptance rate
            accept_rates[tau] = accepted.mean().item()

            # Per-layer acceptance rate
            per_layer = accepted.mean(dim=(0, 2))  # [layers]
            per_layer_accept[tau] = per_layer.tolist()

        results[budget] = {
            "mean_coverage": mean_cov.tolist(),
            "std_coverage": std_cov.tolist(),
            "acceptance_rate": accept_rates,
            "per_layer_acceptance": per_layer_accept,
            "n_steps": n_steps,
        }

    return results


def print_results(results, tau_values=(0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)):
    """Print formatted analysis results."""

    print("\n" + "=" * 80)
    print("CVSA COVERAGE & ACCEPTANCE RATE ANALYSIS")
    print("=" * 80)

    for budget, data in sorted(results.items()):
        print(f"\n{'─' * 70}")
        print(f"Budget = {budget*100:.0f}%  ({data['n_steps']} decode steps)")
        print(f"{'─' * 70}")

        # Mean coverage per layer
        mean_cov = data["mean_coverage"]
        print(f"\nMean coverage per layer (min/mean/max):")
        cov_arr = np.array(mean_cov)
        print(f"  min={cov_arr.min():.4f} (L{cov_arr.argmin()})  "
              f"mean={cov_arr.mean():.4f}  "
              f"max={cov_arr.max():.4f} (L{cov_arr.argmax()})")

        # Std per layer
        std_cov = data["std_coverage"]
        std_arr = np.array(std_cov)
        print(f"\nCoverage std per layer (temporal stability):")
        print(f"  min_std={std_arr.min():.4f} (L{std_arr.argmin()})  "
              f"mean_std={std_arr.mean():.4f}  "
              f"max_std={std_arr.max():.4f} (L{std_arr.argmax()})")

        # Acceptance rate table
        print(f"\nAcceptance rate by tau:")
        print(f"  {'tau':>6s}  {'all-head':>10s}  {'per-head':>10s}")
        for tau in tau_values:
            rate = data["acceptance_rate"][tau]
            # Per-head rate
            per_layer = data["per_layer_acceptance"][tau]
            per_head_rate = np.mean(per_layer)
            print(f"  {tau:6.2f}  {per_head_rate*100:9.1f}%  {rate*100:9.1f}%")

        # Bottom-5 layers (most rejecting) at tau=0.90
        tau_focus = 0.90
        if tau_focus in data["per_layer_acceptance"]:
            per_layer = np.array(data["per_layer_acceptance"][tau_focus])
            worst = np.argsort(per_layer)[:5]
            print(f"\nMost-rejecting layers at tau={tau_focus}:")
            for idx in worst:
                print(f"  L{idx:2d}: accept={per_layer[idx]*100:.1f}%  "
                      f"mean_cov={mean_cov[idx]:.4f}  std={std_cov[idx]:.4f}")

            best = np.argsort(per_layer)[-5:][::-1]
            print(f"\nMost-accepting layers at tau={tau_focus}:")
            for idx in best:
                print(f"  L{idx:2d}: accept={per_layer[idx]*100:.1f}%  "
                      f"mean_cov={mean_cov[idx]:.4f}  std={std_cov[idx]:.4f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: Per-head acceptance rate by budget x tau")
    print(f"{'=' * 80}")
    budgets = sorted(results.keys())
    header = f"{'budget':>8s}"
    for tau in tau_values:
        header += f"  {tau:>6.2f}"
    print(header)
    for budget in budgets:
        row = f"{budget*100:>7.0f}%"
        for tau in tau_values:
            rate = results[budget]["acceptance_rate"][tau]
            row += f"  {rate*100:5.1f}%"
        print(row)


def run_diagnostics(args):
    """Main diagnostics routine."""
    print(f"Loading model: {args.model}")
    model = ModelForDecode(
        args.model,
        dtype=torch.bfloat16, device="cuda")
    config = model.config
    generator = E2EGenerator(model, block_size=16)

    budgets = (0.05, 0.10, 0.15, 0.20)

    # Load LongBench prompts
    from datasets import load_dataset
    datasets_to_use = ["hotpotqa", "multifieldqa_en", "narrativeqa", "qasper"]
    prompts = []
    for ds_name in datasets_to_use:
        try:
            data = load_dataset("THUDM/LongBench", ds_name, split="test",
                                trust_remote_code=True)
            fmt = DATASET2PROMPT[ds_name]
            for i in range(min(args.num_samples, len(data))):
                prompt = fmt.format(**data[i])
                # Truncate
                tokens = model.tokenizer(prompt, truncation=False,
                                         return_tensors="pt").input_ids[0]
                if len(tokens) > args.max_context:
                    half = args.max_context // 2
                    prompt = (
                        model.tokenizer.decode(tokens[:half],
                                               skip_special_tokens=True) +
                        model.tokenizer.decode(tokens[-half:],
                                               skip_special_tokens=True))
                ctx_len = len(model.tokenizer.encode(prompt))
                if ctx_len > 100:  # skip very short prompts
                    prompts.append((ds_name, i, prompt, ctx_len))
        except Exception as e:
            print(f"  Warning: skipping {ds_name}: {e}")

    print(f"Collected {len(prompts)} prompts")

    # Aggregate results across prompts
    all_results = {}

    for idx, (ds_name, sample_idx, prompt, ctx_len) in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] {ds_name}#{sample_idx} "
              f"({ctx_len} tokens, {(ctx_len+15)//16} pages)")

        collector = CoverageCollectorBackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=16,
            num_layers=config.num_layers,
            budgets=budgets,
        )

        # Generate with collection
        _ = generator.generate(prompt, args.max_decode, collector)

        # Analyze this sample
        sample_results = analyze_coverage(collector, budgets)

        # Merge into aggregate
        for budget, data in sample_results.items():
            if budget not in all_results:
                all_results[budget] = {
                    "mean_coverages": [],
                    "std_coverages": [],
                    "acceptance_rates": {tau: [] for tau in
                                         (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)},
                    "per_layer_acceptances": {tau: [] for tau in
                                              (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)},
                    "n_steps_total": 0,
                }
            all_results[budget]["mean_coverages"].append(data["mean_coverage"])
            all_results[budget]["std_coverages"].append(data["std_coverage"])
            all_results[budget]["n_steps_total"] += data["n_steps"]
            for tau in data["acceptance_rate"]:
                all_results[budget]["acceptance_rates"][tau].append(
                    data["acceptance_rate"][tau])
                all_results[budget]["per_layer_acceptances"][tau].append(
                    data["per_layer_acceptance"][tau])

        # Print per-sample summary
        for budget in sorted(sample_results.keys()):
            rate_90 = sample_results[budget]["acceptance_rate"].get(0.90, 0)
            mean_cov = np.mean(sample_results[budget]["mean_coverage"])
            print(f"  budget={budget*100:.0f}%: mean_cov={mean_cov:.4f}, "
                  f"accept@0.90={rate_90*100:.1f}%")

    # Aggregate and print
    print("\n\n" + "#" * 80)
    print("AGGREGATED RESULTS")
    print("#" * 80)

    agg_results = {}
    tau_values = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)
    for budget, data in all_results.items():
        mean_cov_all = np.mean(data["mean_coverages"], axis=0).tolist()
        std_cov_all = np.mean(data["std_coverages"], axis=0).tolist()
        accept = {tau: np.mean(data["acceptance_rates"][tau])
                  for tau in tau_values}
        per_layer = {tau: np.mean(data["per_layer_acceptances"][tau], axis=0).tolist()
                     for tau in tau_values}
        agg_results[budget] = {
            "mean_coverage": mean_cov_all,
            "std_coverage": std_cov_all,
            "acceptance_rate": accept,
            "per_layer_acceptance": per_layer,
            "n_steps": data["n_steps_total"],
        }

    print_results(agg_results, tau_values)

    # Save raw data
    if os.path.isabs(args.output):
        out_path = args.output
    else:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.output,
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "agg_results": agg_results,
        "all_results": all_results,
        "args": vars(args),
    }, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVSA coverage diagnostics")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default="results/diagnostics.pt",
                        help="Output .pt path (relative to repo root or absolute)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Samples per dataset")
    parser.add_argument("--max-decode", type=int, default=128,
                        help="Max decode steps per sample")
    parser.add_argument("--max-context", type=int, default=8000,
                        help="Max context length (tokens)")
    args = parser.parse_args()
    run_diagnostics(args)
