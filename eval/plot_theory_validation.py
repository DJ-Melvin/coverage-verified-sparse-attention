#!/usr/bin/env python3
"""Theory validation figure for the CVSA paper appendix.

Single-row, 3 panels (one per 7B model: Llama / Mistral / Qwen).
Each panel shows Theorem 2's equality under the token-level coverage c*:
  predicted = (1 - c*) * ||O_draft - O_comp||  vs  actual = ||O_draft - O_dense||
on the y=x line, color-coded by layer.

Data source: results/theory_validation_summary.json, if available; otherwise
results/error_bound_validation_{model}_ctok.json produced by
run_theorem_validation_ctok.sh.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import apply_paper_style, COLORS, FIGSIZES, clean_spines

apply_paper_style()
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODELS = [
    ("Llama-3.1-8B  (4:1 GQA)", "results/error_bound_validation_llama_ctok.json"),
    ("Mistral-7B  (4:1 GQA)",   "results/error_bound_validation_mistral_ctok.json"),
    ("Qwen2.5-7B  (7:1 GQA)",   "results/error_bound_validation_qwen_ctok.json"),
]
SUMMARY_PATH = "results/theory_validation_summary.json"

# Categorical layer palette (qualitative, distinct hues across models)
LAYER_PALETTE = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                 '#ff7f00', '#a65628', '#1b9e77', '#ffd92f',
                 '#7570b3', '#666666']


def load_panel_data(json_path, title, cached):
    if cached is not None and title in cached.get("panels", {}):
        entry = cached["panels"][title]
        return (
            np.asarray(entry["actual_error"], dtype=np.float64),
            np.asarray(entry["predicted_error"], dtype=np.float64),
            np.asarray(entry["layer"], dtype=int),
            int(entry["n_valid_cells"]),
        )

    with open(json_path) as f:
        d = json.load(f)
    actual = np.array(d["actual_error"])
    exact_t = np.array(d["theorem_exact_token"])
    layers = np.array(d["layer"]).astype(int)

    mask = (actual > 1e-6) & (exact_t > 1e-6)
    actual_f, exact_f, layers_f = actual[mask], exact_t[mask], layers[mask]
    n_valid = len(actual_f)
    return actual_f, exact_f, layers_f, n_valid


def plot_panel(ax, json_path, title, cached=None):
    actual_f, exact_f, layers_f, n_valid = load_panel_data(json_path, title, cached)

    # Map sorted unique layers to fixed palette (so every model's L0 is the same hue)
    uniq_layers = sorted(set(int(l) for l in layers_f))
    color_map = {li: LAYER_PALETTE[i % len(LAYER_PALETTE)]
                 for i, li in enumerate(uniq_layers)}

    for li in uniq_layers:
        m = layers_f == li
        ax.scatter(exact_f[m], actual_f[m], s=5, alpha=0.30,
                   c=color_map[li], label=f'L{li}',
                   rasterized=True, zorder=3)

    vmin = min(actual_f.min(), exact_f.min()) * 0.5
    vmax = max(actual_f.max(), exact_f.max()) * 2
    ax.plot([vmin, vmax], [vmin, vmax], color='0.15',
            linewidth=1.0, zorder=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Predicted: $(1{-}c^\star)\,\|O_\mathrm{draft}-O_\mathrm{comp}\|_2$',
                  fontsize=8.5)
    ax.set_ylabel(r'Actual: $\|O_\mathrm{draft}-O_\mathrm{dense}\|_2$',
                  fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='major', linewidth=0.4, alpha=0.45,
            color=COLORS['grid'])
    clean_spines(ax)

    # Annotate median ratio (should be 1.0000 under c*)
    ratio = actual_f / exact_f
    med = np.median(ratio)
    ax.text(0.96, 0.05,
            f'median ratio = {med:.4f}',
            transform=ax.transAxes,
            fontsize=8.5, ha='right', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='0.6', linewidth=0.6))

    summary = {
        "actual_error": [float(f"{v:.6g}") for v in actual_f],
        "predicted_error": [float(f"{v:.6g}") for v in exact_f],
        "layer": layers_f.astype(int).tolist(),
        "median_ratio": float(med),
        "n_valid_cells": int(n_valid),
    }
    return uniq_layers, color_map, summary


def main():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['wide_3p'])
    cached = None
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)

    last_layers, last_cmap = None, None
    panel_summaries = {}
    for ax, (title, path) in zip(axes, MODELS):
        last_layers, last_cmap, summary = plot_panel(ax, path, title, cached)
        panel_summaries[title] = summary

    # Shared legend across all 3 panels (per-layer marker swatches)
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=last_cmap[li],
                      markeredgecolor='white', markeredgewidth=0.8,
                      markersize=7, label=f'L{li}')
               for li in last_layers]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.04), ncol=len(last_layers),
               framealpha=0.95, edgecolor='0.85', fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("paper/figures", exist_ok=True)
    for fmt in ['pdf', 'png']:
        fig.savefig(f"paper/figures/theory_validation.{fmt}",
                    dpi=300, bbox_inches='tight')
        print(f"Saved paper/figures/theory_validation.{fmt}")
    plt.close()

    if cached is None:
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump({"panels": panel_summaries}, f, indent=2)
            f.write("\n")
        print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
