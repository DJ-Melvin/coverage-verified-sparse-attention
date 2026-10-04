"""Per-(layer, head) coverage / recovery-rate diagnostics.

Companion to ``eval/diagnostics.py`` that preserves the head dimension so we
can build a "Recovery Atlas" heatmap (Figure C1).  The point of saving this
data separately is that we want to look at WHICH heads in WHICH layers
trigger recovery most often, aggregated across many decode steps and
prompts.  ``eval/diagnostics.py`` collapses head info into a per-layer mean
which is too coarse for the atlas figure.

Reuses :class:`CoverageCollectorBackend` from ``eval.diagnostics`` so the
side-channel coverage computation stays canonical.

Output layout::

    {
      "model": str,
      "num_layers": int,
      "num_kv_heads": int,
      "tau_values": [float, ...],
      "budgets": [float, ...],
      "n_total_steps": int,           # summed across all prompts
      "n_prompts": int,
      "args": {...},
      "atlas": {
          budget: {
              "mean_coverage_lh":   [L, H],     # mean coverage per cell
              "std_coverage_lh":    [L, H],     # temporal std per cell
              "min_coverage_lh":    [L, H],     # worst-case min coverage
              "recovery_rate_lh": {
                  tau: [L, H],                  # P(coverage < tau) per cell
              },
              "n_steps_per_prompt": [int, ...], # per-prompt step counts
          }
      },
    }

Usage (on a GPU node)::

    cd /path/to/cvsa
    python eval/diagnostics_head.py \\
        --num-samples 5 \\
        --max-decode 128 \\
        --max-context 8000 \\
        --output results/diagnostics_head.pt

Or via SLURM::

    sbatch scripts/run_diagnostics_head.sh
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.diagnostics import CoverageCollectorBackend  # noqa: E402

from cvsa.runtime.model_loader import ModelForDecode  # noqa: E402
from eval.longbench_harness import E2EGenerator, DATASET2PROMPT  # noqa: E402


# ─── Per-head analyzer (preserves the head dimension) ────────────────────
def analyze_coverage_per_head(
    collector: CoverageCollectorBackend,
    budgets: Tuple[float, ...],
    tau_values: Tuple[float, ...],
    save_step_trace: bool = False,
) -> Dict[float, Dict]:
    """Return per-(layer, head) coverage statistics, keyed by budget.

    Each value is a dict with ``mean_coverage_lh``, ``std_coverage_lh``,
    ``min_coverage_lh`` (each ``[L, H]`` numpy array), plus
    ``recovery_rate_lh`` mapping ``tau -> [L, H]`` reject-fractions, and
    ``n_steps`` for that prompt.

    If ``save_step_trace`` is True, also includes ``trace_slh`` with the
    raw ``[S, L, H]`` coverage tensor for use in temporal-trace figures.
    """
    num_layers = collector.num_layers
    out: Dict[float, Dict] = {}

    for budget in budgets:
        layer_covs: List[torch.Tensor] = []
        n_steps_min = None
        for layer in range(num_layers):
            data = collector.coverage_data[layer]
            if not data:
                continue
            stacked = torch.stack([d[budget] for d in data])  # [steps, B, H_kv]
            stacked = stacked[:, 0, :]                         # [steps, H_kv]
            layer_covs.append(stacked)
            n_steps_min = stacked.shape[0] if n_steps_min is None \
                else min(n_steps_min, stacked.shape[0])

        if not layer_covs or n_steps_min is None:
            continue

        # Trim to a common step count and stack into [steps, L, H]
        layer_covs = [c[:n_steps_min] for c in layer_covs]
        all_cov = torch.stack(layer_covs, dim=1)               # [S, L, H]

        mean_lh = all_cov.mean(dim=0)                          # [L, H]
        std_lh = all_cov.std(dim=0)                            # [L, H]
        min_lh = all_cov.min(dim=0).values                     # [L, H]

        recovery_rate_lh: Dict[float, np.ndarray] = {}
        for tau in tau_values:
            # recovery = coverage < tau, i.e. NOT accepted
            rejected = (all_cov < tau).float()
            recovery_rate_lh[float(tau)] = rejected.mean(dim=0).cpu().numpy()

        entry = {
            "mean_coverage_lh": mean_lh.cpu().numpy(),
            "std_coverage_lh": std_lh.cpu().numpy(),
            "min_coverage_lh": min_lh.cpu().numpy(),
            "recovery_rate_lh": recovery_rate_lh,
            "n_steps": int(n_steps_min),
        }
        if save_step_trace:
            # Cast to float16 to keep file size manageable for long contexts
            # (full-fidelity values live in mean/std/min summaries above).
            entry["trace_slh"] = all_cov.to(torch.float16).cpu().numpy()
        out[float(budget)] = entry

    return out


# ─── Cross-prompt aggregation ────────────────────────────────────────────
def aggregate_atlas(
    per_prompt: List[Dict[float, Dict]],
    budgets: Tuple[float, ...],
    tau_values: Tuple[float, ...],
) -> Dict[float, Dict]:
    """Step-weighted average of [L, H] arrays across prompts."""
    atlas: Dict[float, Dict] = {}
    for budget in budgets:
        budget = float(budget)
        rows = [p[budget] for p in per_prompt if budget in p]
        if not rows:
            continue
        weights = np.array([r["n_steps"] for r in rows], dtype=np.float64)
        w_sum = weights.sum()

        def _wmean(key: str) -> np.ndarray:
            arrs = np.stack([r[key] for r in rows])              # [P, L, H]
            return (arrs * weights[:, None, None]).sum(axis=0) / w_sum

        mean_lh = _wmean("mean_coverage_lh")
        std_lh = _wmean("std_coverage_lh")
        min_lh = np.stack([r["min_coverage_lh"] for r in rows]).min(axis=0)

        recovery: Dict[float, np.ndarray] = {}
        for tau in tau_values:
            arrs = np.stack([r["recovery_rate_lh"][float(tau)] for r in rows])
            recovery[float(tau)] = (arrs * weights[:, None, None]).sum(axis=0) / w_sum

        atlas[budget] = {
            "mean_coverage_lh": mean_lh,
            "std_coverage_lh": std_lh,
            "min_coverage_lh": min_lh,
            "recovery_rate_lh": recovery,
            "n_steps_per_prompt": [int(r["n_steps"]) for r in rows],
        }
    return atlas


# ─── Prompt loading (mirrors eval/diagnostics.py) ─────────────────────────
def load_prompts(model: ModelForDecode, args) -> List[Tuple[str, int, str, int]]:
    from datasets import load_dataset

    datasets_to_use = ["hotpotqa", "multifieldqa_en", "narrativeqa", "qasper"]
    prompts: List[Tuple[str, int, str, int]] = []
    for ds_name in datasets_to_use:
        try:
            data = load_dataset(
                "THUDM/LongBench", ds_name, split="test",
                trust_remote_code=True,
            )
            fmt = DATASET2PROMPT[ds_name]
            for i in range(min(args.num_samples, len(data))):
                prompt = fmt.format(**data[i])
                tokens = model.tokenizer(
                    prompt, truncation=False, return_tensors="pt"
                ).input_ids[0]
                if len(tokens) > args.max_context:
                    half = args.max_context // 2
                    prompt = (
                        model.tokenizer.decode(tokens[:half], skip_special_tokens=True)
                        + model.tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                    )
                ctx_len = len(model.tokenizer.encode(prompt))
                if ctx_len > 100:
                    prompts.append((ds_name, i, prompt, ctx_len))
        except Exception as e:
            print(f"  Warning: skipping {ds_name}: {e}")
    return prompts


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-(layer, head) coverage diagnostics for the Recovery Atlas figure."
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=str,
                        default="results/diagnostics_head.pt")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Prompts per LongBench task to collect.")
    parser.add_argument("--max-decode", type=int, default=128,
                        help="Decode steps per prompt.")
    parser.add_argument("--max-context", type=int, default=8000,
                        help="Truncate prompts longer than this many tokens.")
    parser.add_argument("--budgets", type=str, default="0.05,0.10,0.15,0.20",
                        help="Comma-separated page budgets to evaluate.")
    parser.add_argument("--tau-values", type=str,
                        default="0.80,0.85,0.90,0.93,0.95,0.97,0.99",
                        help="Comma-separated tau values for recovery-rate maps.")
    parser.add_argument(
        "--save-step-trace", type=int, default=0, metavar="K",
        help="If >0, also save the raw [steps, L, H] coverage trace for "
             "the first K prompts (used by the per-step coverage trace "
             "figure). Adds K * S * L * H * 2 bytes per budget.",
    )
    args = parser.parse_args()

    budgets = tuple(float(b) for b in args.budgets.split(","))
    tau_values = tuple(float(t) for t in args.tau_values.split(","))

    print(f"Loading model: {args.model}")
    model = ModelForDecode(args.model, dtype=torch.bfloat16, device="cuda")
    config = model.config
    generator = E2EGenerator(model, block_size=16)

    prompts = load_prompts(model, args)
    print(f"Collected {len(prompts)} prompts")

    per_prompt_atlases: List[Dict[float, Dict]] = []

    for idx, (ds_name, sample_idx, prompt, ctx_len) in enumerate(prompts):
        print(
            f"\n[{idx+1}/{len(prompts)}] {ds_name}#{sample_idx} "
            f"({ctx_len} tokens, {(ctx_len+15)//16} pages)"
        )

        collector = CoverageCollectorBackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=16,
            num_layers=config.num_layers,
            budgets=budgets,
        )

        _ = generator.generate(prompt, args.max_decode, collector)

        save_trace = idx < args.save_step_trace
        sample_atlas = analyze_coverage_per_head(
            collector, budgets, tau_values, save_step_trace=save_trace,
        )
        per_prompt_atlases.append(sample_atlas)

        # Quick per-prompt summary at canonical (B=5%, tau=0.9)
        b_focus = 0.05 if 0.05 in {float(b) for b in budgets} else float(budgets[0])
        if b_focus in sample_atlas:
            mean_cov = sample_atlas[b_focus]["mean_coverage_lh"].mean()
            rec_90 = sample_atlas[b_focus]["recovery_rate_lh"][0.90].mean()
            print(
                f"  budget={b_focus*100:.0f}% mean_cov={mean_cov:.4f} "
                f"recovery@τ=0.90 mean={rec_90*100:.1f}%"
            )

    atlas = aggregate_atlas(per_prompt_atlases, budgets, tau_values)

    # ── Top-level summary ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RECOVERY ATLAS SUMMARY  (per-(layer, head) view, aggregated)")
    print("=" * 80)
    for budget, data in sorted(atlas.items()):
        recv = data["recovery_rate_lh"][0.90]      # [L, H]
        L, H = recv.shape
        flat = recv.reshape(-1)
        n_total_cells = flat.size
        n_hot = int((flat >= 0.50).sum())
        n_cold = int((flat <= 0.05).sum())
        worst = np.argsort(-flat)[:5]
        worst_str = ", ".join(
            f"L{int(i)//H}H{int(i)%H}={flat[i]*100:.0f}%" for i in worst
        )
        print(
            f"\n  budget={budget*100:.0f}%  "
            f"hot cells (rec≥50%): {n_hot}/{n_total_cells}  "
            f"cold cells (rec≤5%): {n_cold}/{n_total_cells}"
        )
        print(f"    top-5 hottest cells: {worst_str}")

    # ── Collect per-prompt traces (only for prompts asked to save them) ─
    step_traces: List[Dict] = []
    if args.save_step_trace > 0:
        for idx, p_atlas in enumerate(per_prompt_atlases[: args.save_step_trace]):
            ds_name, sample_idx, _, ctx_len = prompts[idx]
            traces_per_budget = {
                b: data["trace_slh"]
                for b, data in p_atlas.items()
                if "trace_slh" in data
            }
            if traces_per_budget:
                step_traces.append({
                    "task": ds_name,
                    "sample_idx": int(sample_idx),
                    "ctx_len": int(ctx_len),
                    "traces": traces_per_budget,    # {budget: [S, L, H] fp16}
                })

    # ── Save ─────────────────────────────────────────────────────────
    payload = {
        "model": args.model,
        "num_layers": config.num_layers,
        "num_kv_heads": config.num_kv_heads,
        "tau_values": list(tau_values),
        "budgets": list(budgets),
        "n_prompts": len(per_prompt_atlases),
        "n_total_steps": sum(
            sum(b["n_steps_per_prompt"]) for b in atlas.values()
        ) // max(len(atlas), 1),
        "atlas": atlas,
        "step_traces": step_traces,
        "args": vars(args),
    }

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            out_path,
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
