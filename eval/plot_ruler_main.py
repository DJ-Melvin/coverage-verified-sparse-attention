#!/usr/bin/env python3
"""RULER main figure — 3 panels on the differentiating tasks only.

Panels: MV-NIAH, MQ-NIAH, CWE.
(S-NIAH, MK-NIAH, VT omitted — all methods saturate at >=99% on the
short-context regime; full 6-subtask, per-model line grid lives in
the appendix.)

Lines per panel:
  - Dense (black star)
  - CVSA-c4 (red)
  - CVSA-c8 (light blue)
  - Fixed-10% (grey plus)
  - Eviction baselines (SnapKV / H2O / Ada-KV) collapsed into a grey
    min-max band labeled "Eviction baselines (10%) range"
All values averaged across 3 7B models (Llama, Mistral, Qwen).
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import apply_paper_style, COLORS, FIGSIZES

apply_paper_style()


MODELS = ['Llama-3.1-8B', 'Mistral-7B', 'Qwen-7B']
MODEL_SOURCES = {
    'Llama-3.1-8B': (
        'results/ruler_final_llama.json',
        'results/ruler_longctx_llama.json',
        'results/ruler_longctx_eviction_llama.json',
    ),
    'Mistral-7B': (
        'results/ruler_final_mistral.json',
        'results/ruler_longctx_mistral.json',
        'results/ruler_longctx_eviction_mistral.json',
    ),
    'Qwen-7B': (
        'results/ruler_final_qwen.json',
        'results/ruler_longctx_qwen.json',
        'results/ruler_longctx_eviction_qwen.json',
    ),
}

CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]
CONTEXT_LABELS = ['4K', '8K', '16K', '32K', '64K', '128K']

# Only the 3 differentiating tasks in the main figure — the other 3
# (S-NIAH / MK-NIAH / VT) saturate at short contexts and live in the
# 6-subtask appendix line grid.
FOCUS_TASKS = [
    ('mv_niah', '(A) MV-NIAH'),
    ('mq_niah', '(B) MQ-NIAH'),
    ('cwe',     '(C) CWE'),
]

# Method → (display label, json key, color, marker, linestyle)
# json keys are the method keys stored in results/ruler_*.json (historical ssa_* prefix).
PRIMARY_METHODS = [
    ('Dense',    'dense',                    COLORS['dense'],     '*', '-'),
    ('CVSA-c4',  'ssa_5_90_c4_ar20_dL0',    COLORS['cvsa'],       'o', '-'),
    ('CVSA-c8',  'ssa_5_90_c8_ar20_dL0',    COLORS['cvsa_light'], '^', '-'),
    ('Fixed-10%', 'fixed_10',                COLORS['fixed'],     'P', '-'),
]

# Baselines collapsed into a range band
BAND_METHODS = [
    ('SnapKV', 'snapkv_10'),
    ('H2O',    'h2o_10'),
    ('Ada-KV', 'adakv_10'),
]
BAND_COLOR = COLORS['snapkv']  # mid grey


def load_all():
    data = {}
    for model, paths in MODEL_SOURCES.items():
        data[model] = {}
        for path in paths:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                d = json.load(f)
            for method_key, mres in d.get('results', {}).items():
                if method_key not in data[model]:
                    data[model][method_key] = {}
                for task, tres in mres.items():
                    if task not in data[model][method_key]:
                        data[model][method_key][task] = {}
                    for ctx_str, acc in tres.items():
                        data[model][method_key][task][int(ctx_str)] = acc
    return data


def mean_over_models(data, method_key, task, ctx):
    vals = []
    for model in MODELS:
        v = data[model].get(method_key, {}).get(task, {}).get(ctx)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def plot_panel(ax, data, task, title, ylabel=False):
    # ── Baseline band (min/max of SnapKV, H2O, Ada-KV, averaged over 3 models) ──
    xs = []
    band_lo, band_hi = [], []
    for ctx in CONTEXTS:
        vals = []
        for _, mkey in BAND_METHODS:
            v = mean_over_models(data, mkey, task, ctx)
            if v is not None:
                vals.append(v)
        if len(vals) >= 2:  # need at least 2 baselines for a band
            xs.append(ctx)
            band_lo.append(min(vals))
            band_hi.append(max(vals))
    if xs:
        ax.fill_between(xs, band_lo, band_hi, color=BAND_COLOR, alpha=0.18,
                        zorder=2, linewidth=0)
        # Upper + lower outline to make the band edge visible
        ax.plot(xs, band_lo, color=BAND_COLOR, alpha=0.60, linewidth=1.1,
                linestyle=(0, (3, 2)), zorder=3)
        ax.plot(xs, band_hi, color=BAND_COLOR, alpha=0.60, linewidth=1.1,
                linestyle=(0, (3, 2)), zorder=3)

    # ── Primary method lines ──
    for label, mkey, color, marker, ls in PRIMARY_METHODS:
        pts = [(ctx, mean_over_models(data, mkey, task, ctx)) for ctx in CONTEXTS]
        pts = [(c, v) for c, v in pts if v is not None]
        if not pts:
            continue
        xs_p, ys_p = zip(*pts)
        is_cvsa = label.startswith('CVSA')
        lw = 2.1 if is_cvsa or label == 'Dense' else 1.35
        ms = 7.2 if is_cvsa or label == 'Dense' else 5.2
        alpha = 1.0 if is_cvsa or label == 'Dense' else 0.75
        zo = 6 if is_cvsa else (5 if label == 'Dense' else 4)
        ax.plot(xs_p, ys_p, marker=marker, color=color, linewidth=lw,
                markersize=ms, markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.7, linestyle=ls, alpha=alpha, zorder=zo)

    ax.set_xscale('log', base=2)
    ax.set_xticks(CONTEXTS)
    ax.set_xticklabels(CONTEXT_LABELS)
    ax.set_xlabel('Context')
    if ylabel:
        ax.set_ylabel('Accuracy (%)')
    ax.set_title(title, fontweight='bold', fontsize=10.5)
    ax.grid(True, alpha=0.32)
    ax.set_ylim(0, 110)  # include low 128K MV-NIAH points after long-context reruns

def plot():
    # Keep the native canvas close to the NeurIPS text width. The previous
    # wide canvas was heavily downscaled in LaTeX, making labels tiny.
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.65))
    data = load_all()
    for ax, (task, title) in zip(axes, FOCUS_TASKS):
        plot_panel(ax, data, task, title, ylabel=(ax is axes[0]))

    # Shared legend — primary methods + the baseline band
    handles = []
    for label, _, color, marker, ls in PRIMARY_METHODS:
        handles.append(Line2D([0], [0], marker=marker, color='w',
                              markerfacecolor=color, markeredgecolor='white',
                              markersize=7.2, linewidth=2.0, linestyle=ls,
                              label=label))
    # Band handle
    handles.append(Patch(facecolor=BAND_COLOR, alpha=0.25, edgecolor=BAND_COLOR,
                         linewidth=1.0, linestyle=(0, (3, 2)),
                         label='Eviction baselines (SnapKV / H2O / Ada-KV)'))

    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=len(handles),
               framealpha=0.94, edgecolor='0.85', fontsize=8.3,
               columnspacing=0.85, handletextpad=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.91], w_pad=0.7)
    os.makedirs("paper/figures", exist_ok=True)
    for fmt in ['pdf', 'png']:
        fig.savefig(f'paper/figures/ruler_main.{fmt}', dpi=300, bbox_inches='tight')
        print(f'Saved paper/figures/ruler_main.{fmt}')
    plt.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plot()
