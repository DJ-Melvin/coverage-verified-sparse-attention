"""LongBench per-task panel grid (16 panels, absolute scores).

One panel per LongBench task, 4 bars per panel (Dense, CVSA-c4,
AdaKV, SnapKV) showing the 3-model average. Three coloured dots
overlay each bar to show the individual per-model values, so the
spread is visible without separate panels per model.

Why absolute scores instead of delta-vs-dense:
    Some tasks score ~95 (triviaqa) and others ~3 (passage_count).
    A delta of -1.5 has very different importance in those two
    cases. Absolute scores let each panel auto-scale and the Dense
    bar serves as a natural reference - readers see directly that
    "CVSA red bar reaches the Dense grey bar; baseline bars stop
    short", which is the headline message.

Output:
  paper/figures/longbench_per_task.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _paper_style import apply_paper_style, COLORS, clean_spines  # noqa: E402

apply_paper_style()


REPO = Path(__file__).resolve().parents[1]
OUT_PDF = REPO / "paper/figures/longbench_per_task.pdf"
OUT_PNG = REPO / "paper/figures/longbench_per_task.png"


TASK_ORDER_RAW = [
    "hotpotqa", "triviaqa", "qasper", "multifieldqa_en",
    "2wikimqa", "narrativeqa", "musique",
    "gov_report", "multi_news", "samsum", "qmsum",
    "trec", "lcc", "repobench-p",
    "passage_count", "passage_retrieval_en",
]

TASK_LABELS = {
    "hotpotqa": "HotpotQA",
    "triviaqa": "TriviaQA",
    "qasper": "Qasper",
    "multifieldqa_en": "MultiFieldQA",
    "2wikimqa": "2WikiMQA",
    "narrativeqa": "NarrativeQA",
    "musique": "MuSiQue",
    "gov_report": "GovReport",
    "multi_news": "MultiNews",
    "samsum": "SAMSum",
    "qmsum": "QMSum",
    "trec": "TREC",
    "lcc": "LCC",
    "repobench-p": "RepoBench-P",
    "passage_count": "PassageCount",
    "passage_retrieval_en": "PassageRetr.",
}


MODELS = [
    {
        "label":     "Llama-3.1-8B",
        "color":     COLORS["llama"],
        "cvsa_file": "results/burst_full.json",
        "cvsa_key":  "ar20_c4_dL0",
        "dense_file":"results/16task_baselines.json",
        "dense_key": "dense",
        "kvpress":   "results/kvpress_baselines_llama8b.json",
    },
    {
        "label":     "Mistral-7B-v0.3",
        "color":     COLORS["mistral"],
        "cvsa_file": "results/ar_mistral7b_v03.json",
        "cvsa_key":  "ar20_c4",
        "dense_file":"results/ar_mistral7b_v03.json",
        "dense_key": "dense",
        "kvpress":   "results/kvpress_baselines_mistral7b.json",
    },
    {
        "label":     "Qwen2.5-7B",
        "color":     COLORS["qwen"],
        "cvsa_file": "results/burst_qwen.json",
        "cvsa_key":  "ar20_c4_dL0",
        "dense_file":"results/ar_qwen7b.json",
        "dense_key": "dense",
        "kvpress":   "results/kvpress_baselines_qwen7b.json",
    },
]


METHODS = ["Dense", "CVSA-c4", "AdaKV", "SnapKV"]
METHOD_COLORS = {
    "Dense":   "#3F3F46",  # near-black (canonical reference)
    "CVSA-c4": "#D62728",  # paper red
    "AdaKV":   "#6B7280",  # mid grey
    "SnapKV":  "#C8CDD3",  # light grey
}
METHOD_EDGES = {
    "Dense":   "#1F1F23",
    "CVSA-c4": "#7F1D1D",
    "AdaKV":   "#374151",
    "SnapKV":  "#6B7280",
}


# ─── Loaders ─────────────────────────────────────────────────────────────
def _qual(d: dict) -> dict:
    return d["quality"] if "quality" in d else d


def _score(value) -> float:
    if isinstance(value, dict):
        if "score" in value:
            return float(value["score"])
        nums = [float(x) for x in value.values() if isinstance(x, (int, float))]
        if nums:
            return float(np.mean(nums))
    if isinstance(value, list):
        return float(np.mean([float(x) for x in value]))
    return float(value)


def load_paged(path: str, key: str) -> dict:
    payload = json.load((REPO / path).open(encoding="utf-8"))
    table = _qual(payload)[key]
    return {t: _score(table[t]) for t in table}


def load_kvpress(path: str) -> tuple[dict, dict]:
    payload = json.load((REPO / path).open(encoding="utf-8"))
    ps = payload["per_sample_scores"]

    def _means(tag: str) -> dict:
        return {t: float(np.mean(ps[tag][t])) * 100.0 for t in ps[tag]}

    return _means("hf_snapkv"), _means("hf_adakv")


# Per-(model, method, task) absolute scores
scores: dict[str, dict[str, dict[str, float]]] = {}
for m in MODELS:
    paged_dense = load_paged(m["dense_file"], m["dense_key"])
    paged_cvsa  = load_paged(m["cvsa_file"], m["cvsa_key"])
    kv_snap, kv_ada = load_kvpress(m["kvpress"])
    scores[m["label"]] = {
        "Dense":   paged_dense,
        "CVSA-c4": paged_cvsa,
        "AdaKV":   kv_ada,
        "SnapKV":  kv_snap,
    }


# ─── Sort tasks by 3-model mean of CVSA gain over best baseline ──────────
def _gain(t: str) -> float:
    g = []
    for m in MODELS:
        ml = m["label"]
        cvsa = scores[ml]["CVSA-c4"][t]
        best_base = max(scores[ml]["AdaKV"][t], scores[ml]["SnapKV"][t])
        g.append(cvsa - best_base)
    return float(np.mean(g))


TASK_ORDER = sorted(TASK_ORDER_RAW, key=_gain, reverse=True)


# ─── Figure: 4×4 grid (compact for main paper) ───────────────────────────
fig, axes = plt.subplots(
    2, 8, figsize=(7.55, 2.15),
    gridspec_kw={"wspace": 0.20, "hspace": 0.34},
)
axes = axes.flatten()

bar_xs = np.arange(len(METHODS)) * 0.58
bar_w = 0.42

for i, task in enumerate(TASK_ORDER):
    ax = axes[i]

    # 3-model averages per method
    avgs = []
    per_model_vals = []
    for method in METHODS:
        per_m = [scores[m["label"]][method][task] for m in MODELS]
        per_model_vals.append(per_m)
        avgs.append(float(np.mean(per_m)))

    # Bars
    bar_colors = [METHOD_COLORS[m] for m in METHODS]
    bar_edges  = [METHOD_EDGES[m]  for m in METHODS]
    ax.bar(
        bar_xs, avgs, width=bar_w,
        color=bar_colors, edgecolor=bar_edges, linewidth=0.5,
        zorder=2, alpha=0.95,
    )

    # Auto-fit y range with enough headroom for the value labels.
    all_vals = [v for vals in per_model_vals for v in vals] + avgs
    lo, hi = min(all_vals), max(all_vals)
    raw_span = max(hi - lo, 1e-3)
    span = max(raw_span, 5.0)         # minimum 5-pt visual span
    extra = (span - raw_span) / 2.0
    pad_lo = span * 0.06 + extra
    pad_hi = span * 0.40 + extra
    y_lo = max(0.0, lo - pad_lo)
    y_hi = hi + pad_hi
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(bar_xs[0] - 0.31, bar_xs[-1] + 0.31)

    # Bar value labels. The extra y-headroom above prevents labels from
    # touching panel borders, while the wider bar spacing avoids collisions.
    label_offset = (y_hi - y_lo) * 0.030
    label_ceiling = y_hi - (y_hi - y_lo) * 0.030
    label_gap = (y_hi - y_lo) * 0.055
    placed_label_ys: list[float] = []
    for j, v in enumerate(avgs):
        label_y = min(v + label_offset, label_ceiling)
        if placed_label_ys and abs(label_y - placed_label_ys[-1]) < label_gap:
            label_y = min(label_y + label_gap, label_ceiling)
        ax.text(
            bar_xs[j], label_y,
            f"{v:.1f}",
            ha="center", va="bottom", fontsize=5.8,
            color=METHOD_EDGES[METHODS[j]], fontweight="bold",
            zorder=5,
        )
        placed_label_ys.append(label_y)

    # Title (task name) — minimal padding so panels can pack tightly
    ax.set_title(TASK_LABELS[task], fontsize=6.3, color="#1f1f1f",
                 fontweight="bold", pad=0.3)

    # Drop x-tick labels per panel (legend at bottom carries method id)
    ax.set_xticks(bar_xs)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.tick_params(axis="y", labelsize=4.7, length=1.0, pad=0.2)
    ax.yaxis.grid(True, linestyle=":", color="#D5D5D5",
                  linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    clean_spines(ax)


# ─── Shared legend (4 methods, single row at bottom) ─────────────────────
from matplotlib.patches import Patch

method_handles = [
    Patch(facecolor=METHOD_COLORS[m], edgecolor=METHOD_EDGES[m],
          linewidth=0.5, label=m)
    for m in METHODS
]
fig.legend(
    handles=method_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.006),
    ncol=4, frameon=False, fontsize=7.2, handlelength=1.0,
    columnspacing=1.0,
)

# Reserve a small bottom strip for the legend so it sits right under
# the bottom row of panels instead of floating in cropped whitespace.
fig.subplots_adjust(
    left=0.035, right=0.998, bottom=0.20, top=0.91,
    wspace=0.20, hspace=0.34,
)

out_paths = [OUT_PDF, OUT_PNG]

for out in out_paths:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".png":
        plt.savefig(out, dpi=220, bbox_inches="tight")
    else:
        plt.savefig(out, bbox_inches="tight")

# Console summary
for out in out_paths:
    print("Saved:", out)
