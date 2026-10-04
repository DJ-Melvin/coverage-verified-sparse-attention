#!/usr/bin/env python3
"""Long-decode drift analysis for the per-step coverage trace.

Consumes the .pt from scripts/run_longdecode_trace.sh (samples[].coverage_slh
= per-decode-step coverage [S, L, H]) and answers one question: does coverage
or recovery rate DEGRADE as decode depth grows? Reports per-bin means, a
least-squares slope per 1,000 steps, and an early-vs-late-half comparison,
formatted as a markdown table.

Pure numpy; run locally on the saved .pt. No GPU.

Usage:
  python eval/analyze_longdecode_drift.py \
      --trace results/camera_ready/longdecode_trace_llama_M4.pt [--bin 128]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coverage/recovery drift vs decode depth.")
    parser.add_argument("--trace", required=True,
                        help=".pt from run_longdecode_trace.sh")
    parser.add_argument("--bin", type=int, default=128,
                        help="Steps per bin for the table.")
    parser.add_argument("--tau", type=float, default=0.90)
    args = parser.parse_args()

    payload = torch.load(args.trace, map_location="cpu", weights_only=False)
    dense_layers = set(payload.get("dense_layers", (0,)))
    num_layers = payload["num_layers"]
    valid = np.array([layer not in dense_layers for layer in range(num_layers)])

    samples = payload["samples"]
    print(f"trace: {args.trace}")
    print(f"model: {payload.get('model')}   samples: {len(samples)}   "
          f"tau={args.tau}   non-dense layers: {int(valid.sum())}/{num_layers}")

    # Per-step aggregates, averaged across samples (truncate to common length)
    n_steps = min(s["coverage_slh"].shape[0] for s in samples)
    per_step_cov = np.zeros(n_steps)
    per_step_rec = np.zeros(n_steps)
    for s in samples:
        cov = s["coverage_slh"].astype(np.float32)[:n_steps][:, valid, :]
        per_step_cov += np.nanmean(cov, axis=(1, 2))
        per_step_rec += np.nanmean(cov < args.tau, axis=(1, 2))
    per_step_cov /= len(samples)
    per_step_rec /= len(samples)

    # Binned table (markdown-ready)
    print(f"\n| decode steps | mean coverage | recovery rate |")
    print("|---|---|---|")
    for start in range(0, n_steps, args.bin):
        end = min(start + args.bin, n_steps)
        print(f"| {start + 1}-{end} | "
              f"{per_step_cov[start:end].mean():.4f} | "
              f"{per_step_rec[start:end].mean() * 100:.1f}% |")

    # Trend: least-squares slope per 1,000 steps
    x = np.arange(n_steps, dtype=np.float64)
    cov_slope = np.polyfit(x, per_step_cov, 1)[0] * 1000
    rec_slope = np.polyfit(x, per_step_rec, 1)[0] * 1000

    half = n_steps // 2
    cov_early, cov_late = per_step_cov[:half].mean(), per_step_cov[half:].mean()
    rec_early, rec_late = per_step_rec[:half].mean(), per_step_rec[half:].mean()

    print(f"\ntrend over {n_steps} steps:")
    print(f"  coverage slope : {cov_slope:+.4f} per 1,000 steps "
          f"(early half {cov_early:.4f} -> late half {cov_late:.4f}, "
          f"delta {cov_late - cov_early:+.4f})")
    print(f"  recovery slope : {rec_slope * 100:+.2f} pp per 1,000 steps "
          f"(early {rec_early * 100:.1f}% -> late {rec_late * 100:.1f}%, "
          f"delta {(rec_late - rec_early) * 100:+.1f} pp)")

    print("\nReading guide:")
    print("  - Flat coverage + flat recovery across 1,024 steps => no error")
    print("    compounding with decode depth; the M-step refresh resets any")
    print("    within-window drift, exactly as the mechanism argument claims.")
    print("  - A negative coverage trend or rising recovery trend would mean")
    print("    depth-dependent drift: quantify the")
    print("    magnitude against the M-sweep quality gradient.")
    print("  - Per-sample heterogeneity: rerun with --bin 256 and check the")
    print("    per-task splits if the aggregate trend is ambiguous.")


if __name__ == "__main__":
    main()
