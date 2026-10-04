#!/usr/bin/env python3
"""Compact budget-fairness plot for the main paper.

This is a smaller counterpart to ``plot_budget_fairness.py``.  It keeps the
core comparison only: paged baselines swept from 10% to 20% budget
and the CVSA-c4 operating point at its measured effective budget.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import COLORS, apply_paper_style, clean_spines  # noqa: E402


BASELINES = {
    "Llama": {
        "SnapKV": {10: 37.26, 15: 37.24, 20: 37.20},
        "H2O": {10: 38.94, 15: 39.40, 20: 39.80},
        "AdaKV": {10: 39.59, 15: 40.09, 20: 39.44},
        "Quest": {10: 37.82, 15: 38.75, 20: 38.45},
    },
    "Mistral": {
        "SnapKV": {10: 34.55, 15: 34.57, 20: 34.13},
        "H2O": {10: 38.19, 15: 37.87, 20: 37.91},
        "AdaKV": {10: 37.82, 15: 37.80, 20: 37.69},
        "Quest": {10: 37.17, 15: 37.64, 20: 38.20},
    },
    "Qwen": {
        "SnapKV": {10: 32.70, 15: 32.32, 20: 32.48},
        "H2O": {10: 32.88, 15: 33.28, 20: 33.40},
        "AdaKV": {10: 33.63, 15: 33.09, 20: 33.33},
        "Quest": {10: 36.03, 15: 36.16, 20: 36.90},
    },
}

CVSA_C4 = {
    "Llama": (12.3, 41.87),
    "Mistral": (11.8, 41.71),
    "Qwen": (12.0, 38.54),
}

DENSE = {
    "Llama": 42.57,
    "Mistral": 42.73,
    "Qwen": 40.18,
}

BASELINE_STYLES = {
    "SnapKV": ("#6d6d6d", "v"),
    "H2O": ("#8a8a8a", "^"),
    "AdaKV": ("#a1a1a1", "s"),
    "Quest": ("#b8b8b8", "D"),
}

# The sidecar is small enough that the shared greys above blur together, yet
# every baseline must stay identifiable.
# It keeps the grey-for-baselines convention of the other figures but spreads
# the lightness and outlines the markers so each curve can be told apart.
SIDECAR_STYLES = {
    "SnapKV": ("#3a3a3a", "v"),
    "H2O": ("#6a6a6a", "^"),
    "AdaKV": ("#979797", "s"),
    "Quest": ("#c0c0c0", "D"),
}


def _panel_limits(model: str) -> tuple[float, float]:
    values = [DENSE[model], CVSA_C4[model][1]]
    for curve in BASELINES[model].values():
        values.extend(curve.values())
    pad = 0.9
    return min(values) - pad, max(values) + pad


def _plot_panels(fig, axes, micro: bool = False) -> None:
    model_names = ["Llama", "Mistral", "Qwen"]
    for idx, (ax, model) in enumerate(zip(axes, model_names)):
        dense_q = DENSE[model]
        cvsa_x, cvsa_y = CVSA_C4[model]

        ax.axhline(
            dense_q,
            color=COLORS["dense"],
            linestyle=":",
            linewidth=0.85 if micro else 1.0,
            alpha=0.55,
            zorder=1,
        )

        best_at_20 = -1e9
        for name, curve in BASELINES[model].items():
            xs = sorted(curve.keys())
            ys = [curve[x] for x in xs]
            color, marker = BASELINE_STYLES[name]
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                markersize=2.7 if micro else 3.4,
                linewidth=0.85 if micro else 1.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                alpha=0.88,
                zorder=3,
            )
            best_at_20 = max(best_at_20, curve[20])

        ax.scatter(
            [cvsa_x],
            [cvsa_y],
            marker="*",
            s=74 if micro else 105,
            c=COLORS["cvsa"],
            edgecolors="white",
            linewidth=0.7,
            zorder=6,
        )

        gap = cvsa_y - best_at_20
        ax.plot(
            [20.0, cvsa_x],
            [best_at_20, cvsa_y],
            color="0.25",
            linewidth=0.65 if micro else 0.8,
            alpha=0.72,
            zorder=2,
        )
        ax.text(
            16.2,
            (best_at_20 + cvsa_y) / 2,
            f"+{gap:.1f}",
            fontsize=5.8 if micro else 6.8,
            color="0.15",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="0.78",
                linewidth=0.35,
                boxstyle="round,pad=0.16",
                alpha=0.95,
            ),
            zorder=7,
        )

        ax.set_title(model, fontsize=7.2 if micro else 8.3, pad=1.5)
        ax.set_xlim(8.7, 21.2)
        ax.set_ylim(*_panel_limits(model))
        ax.set_xticks([10, 15, 20])
        ax.tick_params(axis="both", labelsize=5.9 if micro else 7.0, pad=1.5)
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)
        clean_spines(ax)

        if idx == 0:
            ax.set_ylabel("LongBench", fontsize=6.5 if micro else 7.8, labelpad=2)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        if not micro or idx == 1:
            ax.set_xlabel("Budget (%)", fontsize=6.5 if micro else 7.8, labelpad=1.5)

        if not micro:
            ax.text(
                cvsa_x + 0.25,
                cvsa_y,
                "CVSA-c4",
                fontsize=7.0,
                color=COLORS["cvsa"],
                fontweight="bold",
                va="center",
            )

    if not micro:
        handles = [
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle="-",
                markersize=3.8,
                linewidth=1.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=name,
            )
            for name, (color, marker) in BASELINE_STYLES.items()
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                color="white",
                marker="*",
                markersize=8.2,
                markerfacecolor=COLORS["cvsa"],
                markeredgecolor="white",
                label="CVSA-c4",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS["dense"],
                linestyle=":",
                linewidth=1.0,
                label="Dense",
            )
        )
        fig.legend(
            handles=handles,
            ncol=6,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            fontsize=7.0,
            handlelength=1.25,
            columnspacing=0.75,
        )


def save_variant(name: str, figsize: tuple[float, float], micro: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _plot_panels(fig, axes, micro=micro)
    if micro:
        fig.subplots_adjust(left=0.105, right=0.995, top=0.88, bottom=0.24, wspace=0.12)
    else:
        fig.subplots_adjust(left=0.055, right=0.995, top=0.78, bottom=0.23, wspace=0.16)

    os.makedirs("paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"paper/figures/{name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved {out}")
    plt.close(fig)


def save_sidecar_variant(outdir: str = "paper/figures") -> None:
    """Narrow vertical stack intended for placement beside Table 2."""
    # 1.62in tall keeps the three panels the size they were at 1.50in with a
    # one-row legend; the extra height holds the two-row per-baseline legend.
    fig, axes = plt.subplots(3, 1, figsize=(2.62, 1.62), sharex=True)
    model_names = ["Llama", "Mistral", "Qwen"]

    for idx, (ax, model) in enumerate(zip(axes, model_names)):
        dense_q = DENSE[model]
        cvsa_x, cvsa_y = CVSA_C4[model]

        ax.axhline(
            dense_q,
            color=COLORS["dense"],
            linestyle=":",
            linewidth=0.75,
            alpha=0.52,
            zorder=1,
        )

        best_at_20 = -1e9
        for name, curve in BASELINES[model].items():
            xs = sorted(curve.keys())
            ys = [curve[x] for x in xs]
            color, marker = SIDECAR_STYLES[name]
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                markersize=2.8,
                linewidth=0.68,
                markerfacecolor=color,
                markeredgecolor="0.2",
                markeredgewidth=0.25,
                zorder=3,
            )
            best_at_20 = max(best_at_20, curve[20])

        ax.scatter(
            [cvsa_x],
            [cvsa_y],
            marker="*",
            s=55,
            c=COLORS["cvsa"],
            edgecolors="white",
            linewidth=0.6,
            zorder=6,
        )

        gap = cvsa_y - best_at_20
        ax.plot(
            [20.0, cvsa_x],
            [best_at_20, cvsa_y],
            color="0.25",
            linewidth=0.52,
            alpha=0.70,
            zorder=2,
        )
        ax.text(
            16.2,
            (best_at_20 + cvsa_y) / 2,
            f"+{gap:.1f}",
            fontsize=5.2,
            color="0.15",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="0.78",
                linewidth=0.3,
                boxstyle="round,pad=0.13",
                alpha=0.95,
            ),
            zorder=7,
        )

        # Upper right, above the dense line: every curve starts at the left
        # edge, so a label there sits on the 10% markers once they are large
        # enough to read.
        ax.text(
            0.985,
            0.985,
            model,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            fontweight="bold",
            color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.82),
        )
        ax.set_xlim(8.7, 21.2)
        # Headroom above the dense line holds the model label, the one region
        # that no curve, marker, gap line or gap box crosses in any panel.
        y0, y1 = _panel_limits(model)
        y1 += 1.2
        ax.set_ylim(y0, y1)
        yticks = [round(y0 + 0.25), round(y1 - 0.25)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks], fontsize=5.3)
        ax.grid(axis="y", linewidth=0.24, alpha=0.28)
        ax.tick_params(axis="x", labelsize=5.2, pad=0.6, length=1.6, width=0.42)
        ax.tick_params(axis="y", labelsize=4.8, pad=0.6, length=1.6, width=0.42)
        clean_spines(ax)

    axes[-1].set_xticks([10, 15, 20])
    axes[-1].set_xlabel("Budget (%)", fontsize=5.4, labelpad=0.4)
    axes[1].set_ylabel("Quality", fontsize=5.4, labelpad=2.0)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=0.8,
            markerfacecolor=color,
            markeredgecolor="0.2",
            markeredgewidth=0.25,
            label=name,
        )
        for name, (color, marker) in SIDECAR_STYLES.items()
    ] + [
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markerfacecolor=COLORS["cvsa"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.0,
            label="CVSA-c4",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["dense"],
            linestyle=":",
            linewidth=0.9,
            label="Dense",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=3,
        frameon=False,
        fontsize=5.6,
        handlelength=1.4,
        handletextpad=0.32,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.13, right=0.995, top=0.785, bottom=0.15, hspace=0.25)

    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{outdir}/budget_fairness_sidecar.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.006)
        print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar-only", action="store_true",
                    help="render only the sidecar figure used in the main text")
    ap.add_argument("--outdir", default="paper/figures",
                    help="directory for the sidecar output (default keeps the "
                         "original location)")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    apply_paper_style()
    if not args.sidecar_only:
        save_variant("budget_fairness_compact", (6.65, 1.55), micro=False)
        save_variant("budget_fairness_micro", (3.55, 1.38), micro=True)
    save_sidecar_variant(args.outdir)


if __name__ == "__main__":
    main()
