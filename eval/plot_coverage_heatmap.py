#!/usr/bin/env python3
"""Coverage landscape heatmap for the paper.

The public repository should be able to rebuild this figure from compact JSON
summaries. If ``results/coverage_heatmap_summary.json`` is present, the script
uses it directly. Otherwise it falls back to raw diagnostics under ``results/``
or ``artifact/supplement/results/`` and writes the compact summary.

Outputs:
  paper/figures/coverage_heatmap.pdf
  paper/figures/coverage_heatmap.png
  results/coverage_heatmap_summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import apply_paper_style, FIGSIZES  # noqa: E402

apply_paper_style()

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

SUMMARY_PATH = REPO / "results/coverage_heatmap_summary.json"
MULTIMODEL_PATH = REPO / "results/diagnostics_multimodel.json"

MODEL_KEYS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]
MODEL_SHORT = ["Llama-3.1-8B", "Mistral-7B", "Qwen2.5-7B"]


def _raw_path(filename: str) -> Path:
    candidates = [
        REPO / "results" / filename,
        REPO / "artifact/supplement/results" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {filename}. Provide {SUMMARY_PATH.relative_to(REPO)} or place "
        f"the raw file under results/ or artifact/supplement/results/."
    )


def _load_or_build_summary() -> dict:
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    import torch

    with MULTIMODEL_PATH.open("r", encoding="utf-8") as f:
        diag = json.load(f)
    diag_pt = torch.load(_raw_path("diagnostics.pt"), map_location="cpu", weights_only=False)

    llama_budget_sweep = {
        "budgets": [float(b) for b in sorted(diag_pt["agg_results"].keys())],
        "per_layer_acceptance_tau_0_90": {
            f"{float(b):.2f}": [
                float(x) for x in diag_pt["agg_results"][b]["per_layer_acceptance"][0.9]
            ]
            for b in sorted(diag_pt["agg_results"].keys())
        },
    }
    summary = {
        "diagnostics_multimodel": diag,
        "llama_budget_sweep": llama_budget_sweep,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Saved {SUMMARY_PATH.relative_to(REPO)}")
    return summary


def _coverage_matrix(diag: dict, budget_key: str) -> np.ndarray:
    max_layers = max(
        len(diag[mk][budget_key]["per_layer"]) for mk in MODEL_KEYS if mk in diag
    )
    matrix = np.full((len(MODEL_KEYS), max_layers), np.nan)
    for i, mk in enumerate(MODEL_KEYS):
        if mk not in diag:
            continue
        for layer, value in diag[mk][budget_key]["per_layer"].items():
            matrix[i, int(layer)] = float(value)
    return matrix


def main() -> None:
    summary = _load_or_build_summary()
    diag = summary["diagnostics_multimodel"]
    budget_sweep = summary["llama_budget_sweep"]

    fig = plt.figure(figsize=FIGSIZES["wide_3p"], constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.1], wspace=0.08)

    cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#d73027", "#fee090", "#1a9850"], N=256
    )
    cmap.set_bad("white")

    # Panel (a): layer x model verifier acceptance at 5% budget.
    ax = fig.add_subplot(gs[0])
    cov_matrix = _coverage_matrix(diag, "5_90")
    im = ax.imshow(
        cov_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(MODEL_SHORT)))
    ax.set_yticklabels(MODEL_SHORT, fontsize=10)
    xticks = list(range(0, cov_matrix.shape[1], 4))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlabel("Layer index")

    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            v = cov_matrix[i, j]
            if not np.isfinite(v):
                continue
            if v < 50:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
            elif v < 70:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6.5, color="black")

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Acceptance rate (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Panel (b): Llama acceptance rate vs budget by layer group.
    ax2 = fig.add_subplot(gs[1])
    budgets = [float(b) for b in budget_sweep["budgets"]]
    acc_by_budget = {
        float(k): np.asarray(v, dtype=np.float64)
        for k, v in budget_sweep["per_layer_acceptance_tau_0_90"].items()
    }
    acc_5pct = acc_by_budget[0.05]
    problem = np.where(acc_5pct < 0.5)[0]
    moderate = np.where((acc_5pct >= 0.5) & (acc_5pct < 0.9))[0]
    good = np.where(acc_5pct >= 0.9)[0]

    groups = [
        ("Problem layers", problem, "#d73027", "s"),
        ("Moderate layers", moderate, "#fc8d59", "D"),
        ("High-coverage layers", good, "#1a9850", "o"),
    ]
    for label, indices, color, marker in groups:
        if len(indices) == 0:
            continue
        means = [float(acc_by_budget[b][indices].mean()) for b in budgets]
        ax2.plot([b * 100 for b in budgets], means, f"{marker}-",
                 color=color, linewidth=2, markersize=7,
                 label=f"{label} (n={len(indices)})")

    overall = [float(acc_by_budget[b].mean()) for b in budgets]
    ax2.plot([b * 100 for b in budgets], overall, "k--",
             linewidth=1.5, alpha=0.6, label="Overall mean")
    ax2.axhline(y=0.9, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.text(21, 0.91, r"$\tau$=0.90", fontsize=8, color="0.4")
    ax2.set_xlabel("Budget (%)")
    ax2.set_ylabel("Acceptance rate at $\\tau$=0.90")
    ax2.set_xlim(3, 22)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([5, 10, 15, 20])
    ax2.legend(loc="lower right", framealpha=0.9, edgecolor="0.8")
    ax2.grid(True)

    # Panel (c): coverage distribution at 5% vs 10%.
    ax3 = fig.add_subplot(gs[2])
    for budget_key, alpha, label_suffix in [
        ("5_90", 0.85, "5% budget"),
        ("10_90", 0.65, "10% budget"),
    ]:
        values = []
        for mk in MODEL_KEYS:
            if mk not in diag:
                continue
            pl = diag[mk][budget_key]["per_layer"]
            values.extend(float(pl[k]) for k in sorted(pl.keys(), key=int))
        values = np.asarray(values)
        ax3.hist(values, bins=np.arange(0, 105, 5), alpha=alpha,
                 edgecolor="white", linewidth=0.5,
                 label=f"{label_suffix} (mean={values.mean():.0f}%)")

    ax3.axvline(x=90, color="black", linestyle="-", linewidth=1.2)
    covs_5 = []
    for mk in MODEL_KEYS:
        if mk not in diag:
            continue
        pl = diag[mk]["5_90"]["per_layer"]
        covs_5.extend(float(pl[k]) for k in sorted(pl.keys(), key=int))
    covs_5 = np.asarray(covs_5)
    n_above = int((covs_5 >= 90).sum())
    n_total = int(len(covs_5))
    ax3.text(
        0.74, 0.54,
        f"At 5%: {n_above}/{n_total} $\\geq$ 90%\n{n_total - n_above} need recovery",
        transform=ax3.transAxes, fontsize=9.0, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f0f0f0",
                  edgecolor="0.75", linewidth=0.8),
    )
    ax3.set_xlabel("Per-layer mean coverage (%)")
    ax3.set_ylabel("Number of layers")
    ax3.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
               framealpha=0.92, edgecolor="0.8", fontsize=8.8,
               borderpad=0.45, labelspacing=0.32, handlelength=1.4)
    ax3.grid(True, axis="y")
    ax3.text(0.88, 0.42, r"$\tau$=0.90", fontsize=8, color="black",
             rotation=90, ha="center", va="center", transform=ax3.transAxes)

    (REPO / "paper/figures").mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = REPO / f"paper/figures/coverage_heatmap.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out.relative_to(REPO)}")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("COVERAGE LANDSCAPE -- NUMERIC SUMMARY")
    print("=" * 70)
    print("\nPanel (a): Per-layer verifier acceptance at 5% budget, tau=0.90")
    for i, mk in enumerate(MODEL_KEYS):
        if mk not in diag:
            continue
        pl = diag[mk]["5_90"]["per_layer"]
        vals = np.asarray([float(pl[k]) for k in sorted(pl.keys(), key=int)])
        d = diag[mk]["5_90"]
        print(f"  {MODEL_SHORT[i]:16s}  layers={len(vals)}  mean={vals.mean():.1f}%  "
              f"min={vals.min():.1f}% (L{vals.argmin()})  "
              f"skip={d['n_skip']}  problem={d['n_problem']}  moderate={d['n_moderate']}")

    print(f"\nPanel (b): Llama-8B layer group acceptance at tau=0.90")
    for label, indices, _, _ in groups:
        print(f"  {label} (n={len(indices)}): layers={indices.tolist()}")
        for b in budgets:
            print(f"    budget={b * 100:.0f}%: "
                  f"mean_accept={acc_by_budget[b][indices].mean() * 100:.1f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
