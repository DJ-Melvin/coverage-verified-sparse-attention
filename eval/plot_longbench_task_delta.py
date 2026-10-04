#!/usr/bin/env python3
"""Compact LongBench per-task delta figure for the main paper.

Shows CVSA-c4 minus dense for each LongBench task and each 7B model.
Negative values indicate residual quality loss relative to dense attention.

Outputs:
  paper/figures/longbench_task_delta.{pdf,png}
  results/longbench_task_delta_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from _paper_style import apply_paper_style, COLORS  # noqa: E402

apply_paper_style()

REPO = Path(__file__).resolve().parents[1]
OUT_PDF = REPO / "paper/figures/longbench_task_delta.pdf"
OUT_PNG = REPO / "paper/figures/longbench_task_delta.png"
OUT_JSON = REPO / "results/longbench_task_delta_summary.json"

TASK_ORDER = [
    "hotpotqa", "triviaqa", "qasper", "multifieldqa_en",
    "2wikimqa", "narrativeqa", "musique",
    "gov_report", "multi_news", "samsum", "qmsum",
    "trec", "lcc", "repobench-p",
    "passage_count", "passage_retrieval_en",
]

TASK_LABELS = {
    "hotpotqa": "Hotpot",
    "triviaqa": "Trivia",
    "qasper": "Qasper",
    "multifieldqa_en": "MultiFld",
    "2wikimqa": "2Wiki",
    "narrativeqa": "Narr.",
    "musique": "MuSiQue",
    "gov_report": "GovRep",
    "multi_news": "MultiNews",
    "samsum": "SAMSum",
    "qmsum": "QMSum",
    "trec": "TREC",
    "lcc": "LCC",
    "repobench-p": "Repo",
    "passage_count": "Count",
    "passage_retrieval_en": "Retr.",
}

SOURCES = [
    (
        "Llama-8B",
        "results/16task_baselines.json", "dense",
        "results/burst_full.json", "ar20_c4_dL0",
        COLORS["llama"],
    ),
    (
        "Mistral-7B",
        "results/ar_mistral7b_v03.json", "dense",
        "results/ar_mistral7b_v03.json", "ar20_c4",
        COLORS["mistral"],
    ),
    (
        "Qwen-7B",
        "results/ar_qwen7b.json", "dense",
        "results/burst_qwen.json", "ar20_c4_dL0",
        COLORS["qwen"],
    ),
]


def _method_table(payload: dict, key: str) -> dict:
    return payload["quality"][key] if "quality" in payload else payload[key]


def _score(value) -> float:
    if isinstance(value, dict):
        if "score" in value:
            return float(value["score"])
        numeric = [float(x) for x in value.values() if isinstance(x, (int, float))]
        if numeric:
            return float(np.mean(numeric))
    if isinstance(value, list):
        return float(np.mean([float(x) for x in value]))
    return float(value)


def load_deltas() -> tuple[np.ndarray, list[dict]]:
    rows = []
    summaries = []
    for model, dense_path, dense_key, cvsa_path, cvsa_key, _color in SOURCES:
        dense_payload = json.load(open(REPO / dense_path, encoding="utf-8"))
        cvsa_payload = json.load(open(REPO / cvsa_path, encoding="utf-8"))
        dense = _method_table(dense_payload, dense_key)
        cvsa = _method_table(cvsa_payload, cvsa_key)

        deltas = []
        per_task = {}
        for task in TASK_ORDER:
            d = _score(dense[task])
            c = _score(cvsa[task])
            delta = c - d
            deltas.append(delta)
            per_task[task] = {
                "dense": d,
                "cvsa_c4": c,
                "delta": delta,
            }
        rows.append(deltas)
        sorted_tasks = sorted(per_task.items(), key=lambda kv: kv[1]["delta"])
        summaries.append({
            "model": model,
            "mean_delta": float(np.mean(deltas)),
            "worst_tasks": [
                {"task": t, "delta": float(v["delta"])}
                for t, v in sorted_tasks[:5]
            ],
            "best_tasks": [
                {"task": t, "delta": float(v["delta"])}
                for t, v in sorted_tasks[-5:][::-1]
            ],
            "per_task": per_task,
        })
    return np.asarray(rows, dtype=np.float32), summaries


def main() -> None:
    deltas, summaries = load_deltas()

    sort_idx = np.argsort(deltas.mean(axis=0))
    sorted_deltas = deltas[:, sort_idx]
    sorted_tasks = [TASK_ORDER[i] for i in sort_idx]

    cmap = LinearSegmentedColormap.from_list(
        "delta_red_green",
        ["#B91C1C", "#E87979", "#FFF7D6", "#8FD19E", "#0B8F62"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=4.0)

    fig = plt.figure(figsize=(7.08, 1.72))
    ax = fig.add_axes([0.095, 0.34, 0.80, 0.50])
    cax = fig.add_axes([0.915, 0.34, 0.017, 0.50])

    im = ax.imshow(sorted_deltas, aspect="auto", cmap=cmap, norm=norm)

    ax.set_yticks(np.arange(len(SOURCES)))
    ax.set_yticklabels([s[0] for s in SOURCES], fontsize=8.7)
    for tick, source in zip(ax.get_yticklabels(), SOURCES):
        tick.set_color(source[5])
        tick.set_fontweight("bold")
        tick.set_fontsize(8.1)

    ax.set_xticks(np.arange(len(sorted_tasks)))
    ax.set_xticklabels(
        [TASK_LABELS[t] for t in sorted_tasks],
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=6.75,
    )
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(-0.5, len(sorted_tasks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SOURCES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(sorted_deltas.shape[0]):
        for j in range(sorted_deltas.shape[1]):
            v = sorted_deltas[i, j]
            color = "white" if abs(v) >= 3.0 else "#222222"
            ax.text(
                j, i, f"{v:+.1f}",
                ha="center", va="center",
                fontsize=6.45,
                color=color,
                fontweight="bold" if abs(v) >= 2.5 else "normal",
            )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\Delta$ score", fontsize=8.0)
    cbar.set_ticks([-5, -2.5, 0, 2.5, 4])
    cbar.ax.tick_params(labelsize=7.5, length=2)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    overall = {
        "task_order": sorted_tasks,
        "mean_delta_by_task": {
            task: float(deltas[:, TASK_ORDER.index(task)].mean())
            for task in TASK_ORDER
        },
        "models": summaries,
        "figure": {
            "pdf": str(OUT_PDF.relative_to(REPO)),
            "png": str(OUT_PNG.relative_to(REPO)),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print(f"Saved {OUT_PDF.relative_to(REPO)}")
    print(f"Saved {OUT_PNG.relative_to(REPO)}")
    print(f"Saved {OUT_JSON.relative_to(REPO)}")
    print("Worst mean-delta tasks:")
    for task in sorted_tasks[:5]:
        print(f"  {task:22s} {deltas[:, TASK_ORDER.index(task)].mean():+.2f}")


if __name__ == "__main__":
    main()
