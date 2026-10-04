#!/usr/bin/env python3
"""Build a static per-(layer, head) budget table from offline coverage stats.

Matched-compute control: compares CVSA's DYNAMIC per-step recovery with a
STATIC policy that assigns larger budgets to components identified through
offline coverage statistics (DuoAttention-style). This script builds that
static policy from the Recovery Atlas
(eval/diagnostics_head.py output): each (layer, head) cell gets the budget
CVSA itself would spend there in EXPECTATION,

    b[l, h] = base + recovery_rate[l, h] * (ar - base)

where recovery_rate[l, h] = P(coverage < tau | base budget) measured offline.
This is the most favorable static policy derivable from coverage statistics:
same selector, same offline signal CVSA uses, only the per-step routing is
frozen. Optionally rescales to hit a target mean budget so the comparison is
at matched effective compute (--target-mean-pct).

Usage (on the GPU node, after the calibration atlas exists):
    python eval/build_static_budget_table.py \
        --atlas results/camera_ready/atlas_calib_llama.pt \
        --output results/camera_ready/static_budget_llama.json \
        --tau 0.90 --calib-budget 0.05 --base 0.05 --ar 0.20 \
        --dense-layers 0 --target-mean-pct 8.8

Consumed by eval/longbench_cvsa.py `staticpol_*` methods via the env var
CVSA_STATIC_BUDGET_TABLE.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def _lookup_float_key(d: dict, want: float, what: str):
    for key in d:
        if abs(float(key) - want) < 1e-6:
            return d[key]
    raise KeyError(f"{what}={want} not in atlas (has: {sorted(d.keys())})")


def _rescale_to_target(b: np.ndarray, calib_mask: np.ndarray, base: float,
                       ar: float, target_mean: float) -> np.ndarray:
    """Multiplicatively scale budgets (bisection) so the mean over non-dense
    layers hits target_mean after clamping to [base, ar]."""
    lo, hi = 0.01, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        scaled = np.clip(b * mid, base, ar)
        mean = scaled[calib_mask].mean()
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return np.clip(b * ((lo + hi) / 2), base, ar)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static per-(layer, head) budget table from Recovery Atlas.")
    parser.add_argument("--atlas", required=True,
                        help="diagnostics_head.py output (.pt)")
    parser.add_argument("--output", required=True, help="table JSON path")
    parser.add_argument("--calib-budget", type=float, default=0.05,
                        help="Atlas budget slice to calibrate from.")
    parser.add_argument("--tau", type=float, default=0.90,
                        help="Operating threshold for the recovery-rate map.")
    parser.add_argument("--base", type=float, default=0.05,
                        help="Base budget fraction (CVSA base).")
    parser.add_argument("--ar", type=float, default=0.20,
                        help="Recovery budget fraction (CVSA AR).")
    parser.add_argument("--dense-layers", type=str, default="0",
                        help="Comma-separated always-dense layers (excluded "
                             "from mean; runtime handles them dense).")
    parser.add_argument("--target-mean-pct", type=float, default=None,
                        help="If set, rescale so the mean budget over "
                             "non-dense layers equals this percent "
                             "(matched effective compute).")
    args = parser.parse_args()

    payload = torch.load(args.atlas, map_location="cpu", weights_only=False)
    atlas = payload["atlas"]
    entry = _lookup_float_key(atlas, args.calib_budget, "calib-budget")
    recovery = np.asarray(
        _lookup_float_key(entry["recovery_rate_lh"], args.tau, "tau"),
        dtype=np.float64)  # [L, H]
    num_layers, num_heads = recovery.shape

    dense_layers = ([int(x) for x in args.dense_layers.split(",")]
                    if args.dense_layers else [])
    non_dense = np.ones(num_layers, dtype=bool)
    for layer in dense_layers:
        non_dense[layer] = False
    calib_mask = np.repeat(non_dense[:, None], num_heads, axis=1)

    # Expectation policy: the budget CVSA spends at this cell on average.
    budgets = args.base + recovery * (args.ar - args.base)  # [L, H]

    raw_mean = budgets[calib_mask].mean()
    if args.target_mean_pct is not None:
        target = args.target_mean_pct / 100.0
        budgets = _rescale_to_target(budgets, calib_mask, args.base,
                                     args.ar, target)

    final_mean = budgets[calib_mask].mean()
    per_layer_mean = budgets.mean(axis=1)

    table = {
        "model": payload.get("model"),
        "source_atlas": args.atlas,
        "calib_budget": args.calib_budget,
        "tau": args.tau,
        "base_budget": args.base,
        "ar": args.ar,
        "dense_layers": dense_layers,
        "n_calib_prompts": payload.get("n_prompts"),
        "raw_mean_frac": float(raw_mean),
        "final_mean_frac_nondense": float(final_mean),
        "target_mean_pct": args.target_mean_pct,
        "layers": {str(l): [float(b) for b in budgets[l]]
                   for l in range(num_layers)},
    }
    with open(args.output, "w") as f:
        json.dump(table, f, indent=1)

    print(f"Static budget table -> {args.output}")
    print(f"  model: {table['model']}   layers x heads: "
          f"{num_layers} x {num_heads}")
    print(f"  recovery-rate source: budget={args.calib_budget}, "
          f"tau={args.tau}, {table['n_calib_prompts']} calib prompts")
    print(f"  expectation-policy mean (non-dense): {raw_mean * 100:.2f}%"
          + (f" -> rescaled to {final_mean * 100:.2f}%"
             if args.target_mean_pct is not None else ""))
    hot = np.dstack(np.unravel_index(
        np.argsort(-budgets, axis=None)[:5], budgets.shape))[0]
    hot_str = ", ".join(
        f"L{l}H{h}={budgets[l, h] * 100:.0f}%" for l, h in hot)
    print(f"  top-5 budget cells: {hot_str}")
    print(f"  per-layer mean range: {per_layer_mean.min() * 100:.1f}% - "
          f"{per_layer_mean.max() * 100:.1f}%")


if __name__ == "__main__":
    main()
