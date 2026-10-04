#!/usr/bin/env python3
"""Plot the runtime recovery heatmap used in the appendix.

Each panel is a true decode-time trace from ``collect_page_traces.py`` with
``rich_traces`` enabled.  A cell is the fraction of KV heads in that
layer/step bin that failed the CVSA coverage check and entered recovery,
averaged over HotpotQA samples and grouped into M=4 decode-step bins.

Inputs:
  results/recovery_heatmap_summary.json, if it already contains compact
    binned matrices; otherwise the raw trace PKLs below are used to regenerate
    that summary.
  results/page_traces_ar20/cvsa_ar20_c4_dL0.pkl
  results/page_traces_recovery_mistral/cvsa_ar20_c4_dL0.pkl
    or results/page_traces_recovery_mistral/all_traces.pkl
  results/page_traces_recovery_qwen/cvsa_ar20_c4_dL0.pkl
    or results/page_traces_recovery_qwen/all_traces.pkl
  The PKLs (and the entry inside them) are named by the collect_page_traces.py
  method key; traces collected before the rename use ``vsa_ar20_c4_dL0``.

Outputs:
  paper/figures/recovery_heatmap.{pdf,png}
  results/recovery_heatmap_summary.json
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from _paper_style import apply_paper_style, clean_spines, COLORS  # noqa: E402

apply_paper_style()


REPO = Path(__file__).resolve().parents[1]
OUT_PDF = REPO / "paper/figures/recovery_heatmap.pdf"
OUT_PNG = REPO / "paper/figures/recovery_heatmap.png"
OUT_SUMMARY = REPO / "results/recovery_heatmap_summary.json"

BIN_STEPS = 4
RECOVERY_VMAX = 70.0
# Historical method key: names the trace PKLs and the entry inside them.
METHOD_KEY = "cvsa_ar20_c4_dL0"
LEGACY_METHOD_KEY = "vsa_ar20_c4_dL0"  # name used by earlier trace files

TRACE_SOURCES = [
    (
        "Llama-3.1-8B",
        "results/page_traces_ar20/cvsa_ar20_c4_dL0.pkl",
        COLORS["llama"],
    ),
    (
        "Mistral-7B",
        "results/page_traces_recovery_mistral/cvsa_ar20_c4_dL0.pkl",
        COLORS["mistral"],
    ),
    (
        "Qwen2.5-7B",
        "results/page_traces_recovery_qwen/cvsa_ar20_c4_dL0.pkl",
        COLORS["qwen"],
    ),
]


def _recovery_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "cvsa_recovery",
        [
            (0.00, "#0B8F62"),  # accepted / no recovery
            (0.22, "#BFE7B1"),
            (0.45, "#FFF4B8"),
            (0.70, "#F59E0B"),
            (1.00, "#B91C1C"),  # frequent recovery
        ],
        N=256,
    )
    cmap.set_bad("#D9D9D9")
    return cmap


def _safe_nanmean(arr: np.ndarray, axis=None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _load_runtime_trace(path: Path) -> tuple[np.ndarray, dict]:
    """Return [layers, step_bins] recovery fraction in percent."""
    source_path = path
    if not source_path.exists():
        source_path = path.with_name(f"{LEGACY_METHOD_KEY}.pkl")
    if not source_path.exists():
        fallback = path.with_name("all_traces.pkl")
        if fallback.exists():
            source_path = fallback
        else:
            raise FileNotFoundError(
                f"Missing {path.relative_to(REPO)} and fallback "
                f"{fallback.relative_to(REPO)}"
            )

    with source_path.open("rb") as f:
        payload = pickle.load(f)
    if "num_layers" not in payload:
        key = next((k for k in (METHOD_KEY, LEGACY_METHOD_KEY) if k in payload), None)
        if key is None:
            raise KeyError(
                f"{source_path.relative_to(REPO)} does not contain {METHOD_KEY}; "
                f"available keys: {list(payload.keys())[:10]}"
            )
        payload = payload[key]

    num_layers = int(payload["num_layers"])
    num_heads = int(payload["num_kv_heads"])
    samples = payload["samples"]
    max_steps = max(int(s["n_steps"]) for s in samples)

    per_sample = np.full((len(samples), num_layers, max_steps), np.nan, dtype=np.float32)
    for sample_idx, sample in enumerate(samples):
        rich_traces = sample.get("rich_traces", [])
        if not rich_traces:
            raise ValueError(f"{path} sample {sample_idx} has no rich_traces")
        for step_idx, layer_trace in enumerate(rich_traces):
            for layer_idx in range(num_layers):
                entry = layer_trace.get(layer_idx)
                if entry is None:
                    continue
                per_sample[sample_idx, layer_idx, step_idx] = (
                    float(entry["n_rejected"]) / float(num_heads)
                )

    mean_layer_step = _safe_nanmean(per_sample, axis=0)
    n_bins = int(np.ceil(max_steps / BIN_STEPS))
    binned = np.full((num_layers, n_bins), np.nan, dtype=np.float32)
    for b in range(n_bins):
        lo, hi = b * BIN_STEPS, min((b + 1) * BIN_STEPS, max_steps)
        binned[:, b] = _safe_nanmean(mean_layer_step[:, lo:hi], axis=1)

    layer_mean = _safe_nanmean(mean_layer_step, axis=1)
    finite = np.isfinite(mean_layer_step)
    non_dense_values = mean_layer_step[finite]
    hot_layers = np.argsort(np.nan_to_num(layer_mean, nan=-1.0))[::-1][:6]

    matrix_percent = binned * 100.0
    summary = {
        "source": str(source_path.relative_to(REPO)),
        "model": payload.get("model", "unknown"),
        "method": payload.get("method", "unknown"),
        "task": payload.get("task", "unknown"),
        "num_samples": len(samples),
        "num_layers": num_layers,
        "num_kv_heads": num_heads,
        "max_steps": max_steps,
        "bin_steps": BIN_STEPS,
        "binned_recovery_percent": np.round(matrix_percent, 3).tolist(),
        "mean_recovery_fraction_non_dense": float(np.nanmean(non_dense_values)),
        "p90_recovery_fraction_non_dense": float(np.nanpercentile(non_dense_values, 90)),
        "p95_recovery_fraction_non_dense": float(np.nanpercentile(non_dense_values, 95)),
        "max_recovery_fraction_non_dense": float(np.nanmax(non_dense_values)),
        "hottest_layers": [
            {
                "layer": int(l),
                "mean_recovery_fraction": float(layer_mean[l]),
            }
            for l in hot_layers
            if np.isfinite(layer_mean[l])
        ],
    }
    return matrix_percent, summary


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _load_cached_summary() -> list[tuple[str, str, np.ndarray, dict]] | None:
    if not OUT_SUMMARY.exists():
        return None
    with OUT_SUMMARY.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    runtime = payload.get("runtime", {})
    panels = []
    color_by_label = {
        "Llama-3.1-8B": COLORS["llama"],
        "Mistral-7B": COLORS["mistral"],
        "Qwen2.5-7B": COLORS["qwen"],
    }
    for label, _, default_color in TRACE_SOURCES:
        summary = runtime.get(label)
        if not summary or "binned_recovery_percent" not in summary:
            return None
        matrix = np.array(summary["binned_recovery_percent"], dtype=np.float32)
        panels.append((label, color_by_label.get(label, default_color), matrix, summary))
    return panels


def main() -> None:
    cmap = _recovery_cmap()
    panels = _load_cached_summary()
    summary_by_model = {}
    if panels is None:
        panels = []
        for label, rel_path, color in TRACE_SOURCES:
            matrix, summary = _load_runtime_trace(REPO / rel_path)
            panels.append((label, color, matrix, summary))
            summary_by_model[label] = summary
    else:
        summary_by_model = {label: summary for label, _, _, summary in panels}

    fig = plt.figure(figsize=(7.1, 4.95))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1.0, 1.0, 0.88],
        hspace=0.23,
        wspace=0.06,
    )

    axes = []
    im = None
    for row, (label, color, matrix, summary) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        axes.append(ax)
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=RECOVERY_VMAX,
        )

        max_step = summary["max_steps"]
        xticks = [0, matrix.shape[1] // 4, matrix.shape[1] // 2,
                  3 * matrix.shape[1] // 4, matrix.shape[1] - 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels(["0", str(max_step // 4), str(max_step // 2),
                            str(3 * max_step // 4), str(max_step)])
        if row < len(panels) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f"Decode step, grouped by {BIN_STEPS}")

        ymax = matrix.shape[0] - 1
        yticks = [0, max(1, ymax // 4), max(1, ymax // 2), max(1, 3 * ymax // 4), ymax]
        ax.set_yticks(yticks)
        ax.set_ylabel("Layer")
        ax.axhline(0.5, color="white", linewidth=0.7, alpha=0.92)
        ax.tick_params(length=2.2)

        hot = ", ".join(f"L{x['layer']}" for x in summary["hottest_layers"][:4])
        ax.text(
            0.006,
            0.94,
            label,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color=color,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="#CFD4DA",
                linewidth=0.45,
                alpha=0.94,
            ),
        )
        ax.text(
            0.995,
            0.94,
            (
                f"mean {_format_pct(summary['mean_recovery_fraction_non_dense'])}; "
                f"P90 {_format_pct(summary['p90_recovery_fraction_non_dense'])}; "
                f"hot {hot}"
            ),
            transform=ax.transAxes,
            fontsize=7.5,
            color="#222222",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="#CFD4DA",
                linewidth=0.45,
                alpha=0.94,
            ),
        )
        ax.text(
            0.006,
            0.018,
            "dense L0",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="bottom",
            color="#5F6368",
        )
        clean_spines(ax)

    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Recovered heads (%)", fontsize=8.7)
    cbar.set_ticks([0, 15, 30, 45, 60, 70])
    cbar.set_ticklabels(["0", "15", "30", "45", "60", ">=70"])
    cbar.ax.tick_params(labelsize=8.0, length=2.2)

    dense_patch = Patch(facecolor="#D9D9D9", edgecolor="#AAAAAA", label="Layer 0 dense")
    fig.legend(
        handles=[dense_patch],
        loc="lower center",
        bbox_to_anchor=(0.48, -0.006),
        frameon=False,
        fontsize=8.1,
        handlelength=1.4,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "runtime": summary_by_model,
        "figure": {
            "pdf": str(OUT_PDF.relative_to(REPO)),
            "png": str(OUT_PNG.relative_to(REPO)),
            "vmax_percent": RECOVERY_VMAX,
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {OUT_PDF.relative_to(REPO)}")
    print(f"Saved {OUT_PNG.relative_to(REPO)}")
    print(f"Saved {OUT_SUMMARY.relative_to(REPO)}")
    for label, data in summary_by_model.items():
        print(
            f"{label}: mean={data['mean_recovery_fraction_non_dense'] * 100:.1f}% "
            f"p90={data['p90_recovery_fraction_non_dense'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
