#!/usr/bin/env python3
"""Plot the diagnostic-only CVSA hybrid coverage trace.

The script is summary-first for public release. If
``results/hybrid_coverage_trace_summary.json`` exists, it rebuilds the figure
without loading the raw ``.pt`` trace. Otherwise it loads the raw trace from
``results/`` or ``artifact/supplement/results/`` and writes a compact summary
containing only the plotted time window.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from eval._paper_style import apply_paper_style
except Exception:
    apply_paper_style = None


REPO = Path(__file__).resolve().parents[1]
RAW_INPUT = "hybrid_coverage_trace_llama.pt"
SUMMARY = REPO / "results/hybrid_coverage_trace_summary.json"
OUT_PDF = REPO / "paper/figures/hybrid_coverage_trace.pdf"
OUT_PNG = REPO / "paper/figures/hybrid_coverage_trace.png"
TAU_DEFAULT = 0.90
MAX_PLOT_STEPS = 192


def _raw_path() -> Path:
    candidates = [
        REPO / "results" / RAW_INPUT,
        REPO / "artifact/supplement/results" / RAW_INPUT,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {RAW_INPUT}. Provide {SUMMARY.relative_to(REPO)} or place the "
        "raw trace under results/ or artifact/supplement/results/."
    )


def _head_scores(cov: np.ndarray, refresh: np.ndarray, dense_layers: set[int], tau: float):
    """Rank heads by sawtooth drift after refresh and threshold crossings."""
    check_interval = int(np.median(np.diff(np.flatnonzero(refresh)))) if refresh.sum() > 1 else 4
    refresh_steps = np.flatnonzero(refresh)
    valid = []
    for layer in range(cov.shape[1]):
        if layer in dense_layers:
            continue
        for head in range(cov.shape[2]):
            series = cov[:, layer, head]
            if np.isnan(series).all():
                continue
            drift_values = []
            crossings = 0
            for step in refresh_steps:
                if step + check_interval - 1 >= len(series):
                    continue
                segment = series[step: step + check_interval]
                if np.isnan(segment).any():
                    continue
                drift_values.append(float(segment[0] - segment[-1]))
                if segment[0] >= tau and np.nanmin(segment[1:]) < tau:
                    crossings += 1
            below = np.mean(series < tau)
            amp = np.nanpercentile(series, 95) - np.nanpercentile(series, 5)
            drift = max(float(np.mean(drift_values)), 0.0) if drift_values else 0.0
            balance = min(float(below), 1.0 - float(below))
            score = 0.55 * drift + 0.25 * float(amp) + 0.20 * balance + 0.015 * crossings
            valid.append((score, drift, amp, below, crossings, layer, head))
    valid.sort(reverse=True)
    return valid


def _choose_sample_and_heads(samples: list[dict], dense_layers: set[int], tau: float, k: int = 3):
    best = None
    for sample_idx, sample in enumerate(samples):
        cov = sample["coverage_slh"].astype(np.float32)
        refresh = sample["refresh_steps"].astype(bool)
        scores = _head_scores(cov, refresh, dense_layers, tau)
        if not scores:
            continue
        sample_score = float(np.mean([row[0] for row in scores[: min(8, len(scores))]]))
        if best is None or sample_score > best[0]:
            best = (sample_score, sample_idx, scores)
    if best is None:
        raise ValueError("No valid coverage heads found")

    chosen = []
    used_layers = set()
    for row in best[2]:
        layer, head = int(row[-2]), int(row[-1])
        if layer in used_layers:
            continue
        chosen.append((layer, head))
        used_layers.add(layer)
        if len(chosen) == k:
            break
    return best[1], chosen


def _load_or_build_summary() -> dict:
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))

    import torch

    payload = torch.load(_raw_path(), map_location="cpu", weights_only=False)
    dense_layers = set(int(x) for x in payload.get("dense_layers", []))
    tau = float(payload.get("tau", TAU_DEFAULT))
    check_interval = int(payload.get("check_interval", 4))
    sample_idx, heads = _choose_sample_and_heads(payload["samples"], dense_layers, tau, k=3)
    sample = payload["samples"][sample_idx]
    cov = sample["coverage_slh"].astype(np.float32)
    refresh = sample["refresh_steps"].astype(bool)

    max_steps = min(cov.shape[0], MAX_PLOT_STEPS)
    cov = cov[:max_steps]
    refresh = refresh[:max_steps]
    reject_frac = np.nanmean(cov < tau, axis=2).T

    head_series = []
    for layer, head in heads:
        head_series.append({
            "layer": int(layer),
            "head": int(head),
            "coverage": np.round(cov[:, layer, head], 5).tolist(),
        })

    summary = {
        "source": str(_raw_path().relative_to(REPO)),
        "sample_idx": int(sample_idx),
        "tau": tau,
        "check_interval": check_interval,
        "dense_layers": sorted(int(x) for x in dense_layers),
        "max_steps": int(max_steps),
        "refresh_steps": np.flatnonzero(refresh).astype(int).tolist(),
        "head_series": head_series,
        "reject_frac_by_layer_step": np.round(reject_frac, 5).tolist(),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {SUMMARY.relative_to(REPO)}")
    return summary


def main():
    if apply_paper_style is not None:
        apply_paper_style()

    data = _load_or_build_summary()
    tau = float(data.get("tau", TAU_DEFAULT))
    check_interval = int(data.get("check_interval", 4))
    sample_idx = int(data["sample_idx"])
    dense_layers = set(int(x) for x in data.get("dense_layers", []))
    head_series = data["head_series"]
    reject_frac = np.asarray(data["reject_frac_by_layer_step"], dtype=np.float32)
    max_steps = int(data["max_steps"])
    refresh = np.zeros(max_steps, dtype=bool)
    refresh[np.asarray(data["refresh_steps"], dtype=int)] = True
    x = np.arange(max_steps)

    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.72, 0.95], hspace=0.16,
        left=0.074, right=0.965, top=0.82, bottom=0.205,
    )

    ax = fig.add_subplot(gs[0, 0])
    for step in x[refresh]:
        ax.axvline(step, color="#c7c7c7", lw=0.42, alpha=0.55, zorder=0)

    colors = ["#2b6cb0", "#d95f02", "#1b9e77"]
    for color, series in zip(colors, head_series):
        layer = int(series["layer"])
        head = int(series["head"])
        cov = np.asarray(series["coverage"], dtype=np.float32)
        ax.plot(x, cov, lw=1.25, color=color, label=f"Layer {layer}, Head {head}")
        ax.scatter(
            x[refresh], cov[refresh], s=7, color=color,
            edgecolors="white", linewidths=0.25, zorder=4,
        )

    ax.axhline(tau, color="#444444", ls="--", lw=0.9, label=rf"$\tau={tau:.2f}$")
    ax.set_xlim(0, max_steps - 1)
    ax.set_ylim(0.65, 1.01)
    ax.set_ylabel("Coverage")
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(
        ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.02),
        frameon=False, handlelength=1.0, columnspacing=0.6,
        borderaxespad=0.0, fontsize=7.4,
    )
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    ax.text(
        0.995, 1.025,
        f"sample {sample_idx}; dots/gray ticks = scoring steps",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.4, color="#555555",
    )

    ax_h = fig.add_subplot(gs[1, 0], sharex=ax)
    masked = np.ma.masked_invalid(reject_frac)
    im = ax_h.imshow(
        masked, aspect="auto", interpolation="nearest", origin="lower",
        cmap="YlOrRd", vmin=0.0, vmax=1.0,
        extent=(-0.5, max_steps - 0.5, -0.5, reject_frac.shape[0] - 0.5),
    )
    for layer in dense_layers:
        ax_h.axhspan(layer - 0.5, layer + 0.5, color="#d9d9d9", zorder=3)
    for step in x[refresh]:
        ax_h.axvline(step, color="#6b6b6b", lw=0.35, alpha=0.35, zorder=4)

    ax_h.set_xlabel(f"Decode step (vertical ticks: scoring every M={check_interval})")
    ax_h.set_ylabel("Layer")
    ax_h.set_yticks([0, reject_frac.shape[0] // 2, reject_frac.shape[0] - 1])
    cbar = fig.colorbar(im, ax=ax_h, pad=0.01, fraction=0.035)
    cbar.set_label("Rejected heads")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=220)
    print(f"Saved {OUT_PDF.relative_to(REPO)}")
    print(f"Saved {OUT_PNG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
