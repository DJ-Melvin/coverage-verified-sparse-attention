#!/usr/bin/env python3
"""Figure: Budget scaling within a fixed implementation.

Three panels (Llama / Mistral / Qwen). Per panel:
  - Four baseline curves at nominal budget 10 / 15 / 20 % (within paged-decode impl)
  - CVSA-c2 marker at its effective budget (~16%)
  - CVSA-c4 marker at its effective budget (~12%)
  - Dense horizontal reference

Key message: within a single implementation, increased budget does not lift
baselines toward dense -- the gain is structural to CVSA's verification, not
purely a function of compute spent.

Implementation note: this figure intentionally uses our paged-decode reimpl
for ALL three budget points to keep the budget-scaling comparison monotonic
within one framework. Canonical SnapKV / Ada-KV numbers (NVIDIA KVPress) at
the 10% point are reported separately in Table 1; mixing implementations
within a single curve would conflate two distinct effects.

Data source: paged-decode 16-task runs (results/16task_baselines.json,
qwen7b_16task.json, ar_mistral7b_v03.json) + Table 2 effective budgets.
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import apply_paper_style, COLORS, FIGSIZES

apply_paper_style()

# ─── Data from Table 4 ────────────────────────────────────────────
# baselines[model][method] = {budget_pct: quality}
BASELINES = {
    'Llama-3.1-8B': {
        'SnapKV': {10: 37.26, 15: 37.24, 20: 37.20},
        'H2O':    {10: 38.94, 15: 39.40, 20: 39.80},
        'AdaKV':  {10: 39.59, 15: 40.09, 20: 39.44},
        'Quest':  {10: 37.82, 15: 38.75, 20: 38.45},
    },
    'Mistral-7B': {
        'SnapKV': {10: 34.55, 15: 34.57, 20: 34.13},
        'H2O':    {10: 38.19, 15: 37.87, 20: 37.91},
        'AdaKV':  {10: 37.82, 15: 37.80, 20: 37.69},
        'Quest':  {10: 37.17, 15: 37.64, 20: 38.20},
    },
    'Qwen2.5-7B': {
        'SnapKV': {10: 32.70, 15: 32.32, 20: 32.48},
        'H2O':    {10: 32.88, 15: 33.28, 20: 33.40},
        'AdaKV':  {10: 33.63, 15: 33.09, 20: 33.33},
        'Quest':  {10: 36.03, 15: 36.16, 20: 36.90},
    },
}

# CVSA effective budgets from Table 2a note (per-model averages from latency bench)
CVSA = {
    'Llama-3.1-8B': {'c2': (16.5, 42.49), 'c4': (12.3, 41.87)},
    'Mistral-7B':   {'c2': (15.8, 42.53), 'c4': (11.8, 41.71)},
    'Qwen2.5-7B':   {'c2': (15.9, 39.19), 'c4': (12.0, 38.54)},
}

DENSE = {
    'Llama-3.1-8B': 42.57,
    'Mistral-7B':   42.73,
    'Qwen2.5-7B':   40.18,
}

BASELINE_COLORS = {
    'SnapKV': COLORS['snapkv'],
    'H2O':    COLORS['h2o'],
    'AdaKV':  COLORS['adakv'],
    'Quest':  COLORS['quest'],
}
BASELINE_MARKERS = {
    'SnapKV': 'v',
    'H2O':    '^',
    'AdaKV':  's',
    'Quest':  'D',
}
CVSA_COLOR = COLORS['cvsa']
DENSE_COLOR = COLORS['dense']


def plot():
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZES['wide_3p'], sharey=False)

    for ax, model in zip(axes, ['Llama-3.1-8B', 'Mistral-7B', 'Qwen2.5-7B']):
        dense_q = DENSE[model]

        # Dense reference line
        ax.axhline(y=dense_q, color=DENSE_COLOR, linestyle=':',
                   linewidth=1.1, alpha=0.55, zorder=2)
        ax.text(20.5, dense_q, ' Dense', fontsize=8.5,
                color=DENSE_COLOR, va='center', fontweight='bold')

        # Baseline curves
        for bname, bdata in BASELINES[model].items():
            xs = sorted(bdata.keys())
            ys = [bdata[x] for x in xs]
            color = BASELINE_COLORS[bname]
            marker = BASELINE_MARKERS[bname]
            ax.plot(xs, ys, '-', color=color, linewidth=1.5,
                    marker=marker, markersize=7,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=0.9, alpha=0.85, zorder=4,
                    label=bname)

        # CVSA-c2 and c4 markers (at effective budget)
        for cname, (eff_b, q) in CVSA[model].items():
            is_c2 = cname == 'c2'
            ax.scatter(eff_b, q, marker='*', s=240 if is_c2 else 170,
                       c=CVSA_COLOR, edgecolors='white', linewidth=1.3,
                       zorder=7, label=f'CVSA-{cname} (ours)' if model == 'Llama-3.1-8B' else None)
            # Annotate
            offset = (7, -4) if is_c2 else (7, -4)
            ax.annotate(f'CVSA-{cname}', (eff_b, q),
                        textcoords='offset points', xytext=offset,
                        fontsize=8.5, color=CVSA_COLOR, fontweight='bold')

        # Connect CVSA points with faint line to emphasize they're the same family
        c2_x, c2_y = CVSA[model]['c2']
        c4_x, c4_y = CVSA[model]['c4']
        ax.plot([c2_x, c4_x], [c2_y, c4_y], ':', color=CVSA_COLOR,
                linewidth=1.2, alpha=0.45, zorder=3)

        # Diagonal gap arrow: best baseline @20% -> CVSA-c4 (primary config).
        # Arrow points UP-LEFT to convey CVSA's dual-axis dominance: lower
        # effective budget AND higher quality. This reads more naturally than
        # a fixed-x vertical arrow when the gap magnitude varies across models.
        best_q_20 = max(BASELINES[model][b][20] for b in BASELINES[model])
        gap = c4_y - best_q_20
        ax.annotate(
            '',
            xy=(c4_x + 0.25, c4_y - 0.10),       # arrowhead near CVSA-c4
            xytext=(20.0 - 0.25, best_q_20 + 0.10),  # tail at best baseline @20%
            arrowprops=dict(arrowstyle='->', color='0.3',
                            lw=1.2, shrinkA=0, shrinkB=0),
            zorder=6,
        )
        # Label sits at the midpoint of the diagonal, with a white box so it
        # reads cleanly across baseline curves.
        mid_x = (c4_x + 20.0) / 2
        mid_y = (c4_y + best_q_20) / 2
        ax.text(mid_x, mid_y, f'+{gap:.1f} pt',
                fontsize=9, color='0.15',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.28',
                          facecolor='white', edgecolor='0.55',
                          linewidth=0.6, alpha=0.95),
                zorder=8)

        ax.set_xlabel('Budget (%)')
        ax.set_ylabel('Quality (16-task LongBench)')
        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.set_xlim(8, 22.5)
        all_q = [dense_q] + [CVSA[model]['c2'][1], CVSA[model]['c4'][1]]
        for bdata in BASELINES[model].values():
            all_q.extend(bdata.values())
        ax.set_ylim(min(all_q) - 1.5, max(all_q) + 1.5)
        ax.grid(True)
        ax.set_xticks([10, 15, 20])

    # Shared legend on top
    handles = []
    for bname in ['SnapKV', 'H2O', 'AdaKV', 'Quest']:
        handles.append(Line2D(
            [0], [0], marker=BASELINE_MARKERS[bname], color=BASELINE_COLORS[bname],
            linestyle='-', markerfacecolor=BASELINE_COLORS[bname],
            markeredgecolor='white', markeredgewidth=0.9, markersize=7,
            linewidth=1.5, label=f'{bname} (10/15/20%)'
        ))
    handles.append(Line2D([0], [0], marker='*', color='w',
                          markerfacecolor=CVSA_COLOR, markeredgecolor='white',
                          markeredgewidth=1.2, markersize=13,
                          label='CVSA-c2 / c4 (ours, at effective budget)'))
    handles.append(Line2D([0], [0], color=DENSE_COLOR, linestyle=':',
                          linewidth=1.2, label='Dense'))

    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.04), ncol=6, framealpha=0.92,
               edgecolor='0.8', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("paper/figures", exist_ok=True)
    for fmt in ['pdf', 'png']:
        fig.savefig(f'paper/figures/budget_fairness.{fmt}', dpi=300, bbox_inches='tight')
        print(f'Saved paper/figures/budget_fairness.{fmt}')
    plt.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plot()
