#!/usr/bin/env python3
"""Plot bandwidth roofline: f_kv vs speedup for CVSA.

Uses existing data (verify_ar_b32.json + architecture specs).
Generates figure for paper showing decode is bandwidth-bound
and CVSA speedup comes from KV bandwidth reduction.

Usage:
    python eval/plot_bandwidth_roofline.py
    # If profile_ar_bandwidth.json exists, overlays per-component data
"""

import json
import os
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import apply_paper_style, COLORS, FIGSIZES

apply_paper_style()

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRS = [
    REPO_ROOT / "paper" / "figures",
]

# ─── Architecture specs ───
MODELS = {
    "Llama-8B": {
        "layers": 32, "q_heads": 32, "kv_heads": 8,
        "head_dim": 128, "hidden": 4096, "inter": 14336,
    },
    "Mistral-7B": {
        "layers": 32, "q_heads": 32, "kv_heads": 8,
        "head_dim": 128, "hidden": 4096, "inter": 14336,
    },
    "Qwen-7B": {
        "layers": 28, "q_heads": 28, "kv_heads": 4,
        "head_dim": 128, "hidden": 3584, "inter": 18944,
    },
}

# ─── Measured latency — canonical same-session sources ───
# Format: {model: {batch: (dense_ms, cvsa_c4_ms)}}
LATENCY = {
    "Llama-8B":   {32: (128.9, 72.5)},  # ar20_lat_llama_b32.json
    "Mistral-7B": {32: (127.7, 72.0)},  # ar20_lat_mistral_b32.json
    "Qwen-7B":    {32: (162.5, 65.0)},  # ar20_lat_qwen_b32.json
}

# Cross-batch speedup (canonical same-session sources)
# B=1 from latfill_8k_*.json, B=8 from pareto_latency_b8 / pareto_qwen_65k /
# latfill_8k_mistral (no same-session baselines for Mistral B=8), B=32 from ar20_lat_*_b32
CROSS_BATCH_SPEEDUP = {
    "Llama-8B":   {1: 1.09, 8: 1.44, 32: 1.78},
    "Mistral-7B": {1: 1.09, 8: 1.44, 32: 1.77},
    "Qwen-7B":    {1: 1.12, 8: 1.87, 32: 2.50},
}

EFF_BUDGET = 0.12  # CVSA-c4 average effective budget
BLOCK_SIZE = 16
DTYPE_BYTES = 2


def compute_model_weights_bytes(m):
    """Total model weight bytes (bf16)."""
    per_layer = (
        m["hidden"] * m["q_heads"] * m["head_dim"] * DTYPE_BYTES  # Q
        + m["hidden"] * m["kv_heads"] * m["head_dim"] * DTYPE_BYTES  # K
        + m["hidden"] * m["kv_heads"] * m["head_dim"] * DTYPE_BYTES  # V
        + m["q_heads"] * m["head_dim"] * m["hidden"] * DTYPE_BYTES  # O
        + 3 * m["hidden"] * m["inter"] * DTYPE_BYTES  # MLP (gate+up+down)
    )
    return per_layer * m["layers"]


def compute_kv_bytes_per_seq(m, ctx=8192):
    """KV cache bytes per sequence (K + V)."""
    n_pages = ctx // BLOCK_SIZE
    page_bytes = BLOCK_SIZE * m["head_dim"] * DTYPE_BYTES
    return 2 * m["kv_heads"] * n_pages * page_bytes * m["layers"]


def theoretical_speedup(f_kv, eff_budget):
    """Predicted speedup from pure bandwidth reduction."""
    return 1.0 / (1.0 - f_kv + f_kv * eff_budget)


def main():
    for fig_dir in FIGURE_DIRS:
        fig_dir.mkdir(parents=True, exist_ok=True)

    # ─── Figure 1: f_kv vs Speedup (main roofline) ───
    fig, ax = plt.subplots(1, 1, figsize=(5.35, 2.45))

    colors = {
        "Llama-8B":   COLORS['llama'],
        "Mistral-7B": COLORS['mistral'],
        "Qwen-7B":    COLORS['qwen'],
    }
    markers = {1: "s", 8: "D", 32: "o"}
    model_order = ["Llama-8B", "Mistral-7B", "Qwen-7B"]

    # Theoretical line
    f_range = np.linspace(0, 0.82, 250)
    theo_line = [theoretical_speedup(f, EFF_BUDGET) for f in f_range]
    ax.plot(
        f_range * 100,
        theo_line,
        color="#A9A9A9",
        linestyle=(0, (4, 2)),
        alpha=0.95,
        linewidth=1.35,
        zorder=1,
    )
    ax.text(
        56.5,
        theoretical_speedup(0.565, EFF_BUDGET) + 0.20,
        "bandwidth-only\nroofline",
        color="#777777",
        fontsize=8.0,
        ha="center",
        va="bottom",
    )

    # Data points
    for model_name in model_order:
        batches = CROSS_BATCH_SPEEDUP[model_name]
        m = MODELS[model_name]
        w_bytes = compute_model_weights_bytes(m)
        kv_per_seq = compute_kv_bytes_per_seq(m)
        color = colors[model_name]
        xs, ys = [], []

        for B, measured in batches.items():
            kv_total = kv_per_seq * B
            f_kv = kv_total / (kv_total + w_bytes)
            xs.append(f_kv * 100)
            ys.append(measured)

        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.45, zorder=3)

        for B, measured in batches.items():
            kv_total = kv_per_seq * B
            f_kv = kv_total / (kv_total + w_bytes)
            ax.scatter(
                f_kv * 100,
                measured,
                c=color,
                marker=markers[B],
                s=58,
                zorder=5,
                edgecolors="white",
                linewidth=0.55,
            )

    model_handles = [
        Line2D([0], [0], color=colors[m], marker="o", linestyle="-",
               linewidth=1.0, markersize=5.5, markeredgecolor="white",
               markeredgewidth=0.5, label=m)
        for m in model_order
    ]
    batch_handles = [
        Line2D([0], [0], color="#777777", marker=markers[B], linestyle="",
               markersize=5.2, label=f"B={B}")
        for B in [1, 8, 32]
    ]

    ax.set_xlabel(r"$f_{\mathrm{kv}}$ = KV bandwidth fraction (%)")
    ax.set_ylabel("Decode speedup vs dense")
    ax.set_xlim(0, 80)
    ax.set_ylim(0.95, 2.72)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5])
    ax.axhline(y=1.0, color=COLORS['iso'], alpha=0.4, linewidth=0.5,
               linestyle=':')
    ax.grid(axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg1 = ax.legend(
        handles=model_handles,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.04),
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.0,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=batch_handles,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(1.01, 0.02),
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.65,
        borderpad=0.0,
    )

    fig.tight_layout()
    for fig_dir in FIGURE_DIRS:
        fig.savefig(fig_dir / "bandwidth_roofline.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "bandwidth_roofline.png", dpi=220, bbox_inches="tight")
    print("Saved bandwidth_roofline.{pdf,png}")

    # ─── Figure 2: Per-step byte breakdown (if profile data exists) ───
    profile_path = "results/profile_ar_bandwidth.json"
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
        print(f"Loaded profile data from {profile_path}")
        # TODO: plot per-component timing breakdown bar chart
    else:
        print(f"No profile data at {profile_path} — run eval/profile_ar_bandwidth.py first")
        print("Generating analytical breakdown instead...")

    # ─── Figure 3: Analytical byte-per-token comparison ───
    fig2, ax2 = plt.subplots(1, 1, figsize=FIGSIZES['double'])

    batch_range = [1, 2, 4, 8, 16, 32, 64]
    m = MODELS["Llama-8B"]
    w_bytes = compute_model_weights_bytes(m)
    kv_per_seq = compute_kv_bytes_per_seq(m)

    dense_bytes = [(w_bytes + kv_per_seq * B) / B / 1e9 for B in batch_range]
    cvsa_bytes = [(w_bytes + kv_per_seq * B * EFF_BUDGET) / B / 1e9 for B in batch_range]
    weight_only = [w_bytes / B / 1e9 for B in batch_range]

    ax2.plot(batch_range, dense_bytes, "o-", color=COLORS['dense'],
             label="Dense", linewidth=1.4, markersize=5)
    ax2.plot(batch_range, cvsa_bytes, "s-", color=COLORS['cvsa'],
             label="CVSA-c4 (eff. budget 12%)", linewidth=1.4, markersize=5)
    ax2.plot(batch_range, weight_only, "--", color=COLORS['iso'], alpha=0.7,
             label="Weights only (lower bound)", linewidth=1)

    # Fill the savings region
    ax2.fill_between(batch_range, cvsa_bytes, dense_bytes,
                     alpha=0.15, color=COLORS['cvsa'], label="Bandwidth saved")

    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("HBM bytes per token (GB)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batch_range)
    ax2.set_xticklabels([str(b) for b in batch_range])
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    fig2.tight_layout()
    for fig_dir in FIGURE_DIRS:
        fig2.savefig(fig_dir / "bytes_per_token.pdf", bbox_inches="tight")
        fig2.savefig(fig_dir / "bytes_per_token.png", dpi=220, bbox_inches="tight")
    print("Saved bytes_per_token.{pdf,png}")

    # ─── Print summary table ───
    print(f"\n{'=' * 75}")
    print("ROOFLINE SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Model':<12} {'B':>3} {'f_kv':>7} {'Predicted':>10} {'Measured':>10} {'Eff':>6}")
    print("-" * 55)

    for model_name in ["Llama-8B", "Mistral-7B", "Qwen-7B"]:
        m = MODELS[model_name]
        w_bytes = compute_model_weights_bytes(m)
        kv_per_seq = compute_kv_bytes_per_seq(m)
        for B in [1, 8, 32]:
            measured = CROSS_BATCH_SPEEDUP[model_name].get(B)
            if measured is None:
                continue
            kv_total = kv_per_seq * B
            f_kv = kv_total / (kv_total + w_bytes)
            pred = theoretical_speedup(f_kv, EFF_BUDGET)
            eff = measured / pred * 100
            print(f"{model_name:<12} {B:>3} {f_kv:>6.1%} {pred:>9.2f}x {measured:>9.2f}x {eff:>5.0f}%")


if __name__ == "__main__":
    main()
