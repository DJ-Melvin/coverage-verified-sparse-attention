#!/usr/bin/env python3
"""Generate Pareto frontier and tradeoff figures for the CVSA paper.

Paper-ready style: no titles, serif font, clean layout.
Titles are handled by LaTeX caption.

Uses AR (approximate recovery) data. Three models: Llama-8B, Qwen-7B, Mistral-7B (v0.3).

Quality points and 95% CIs for {dense, CVSA-c4, CVSA-c8, Fixed-10%} are derived from
per-sample LongBench scores via bootstrap (n=10000) when available; other points use
published summary means. Latencies from results/verify_ar_b32.json (5-rep median, std<0.2ms).
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_style import apply_paper_style, COLORS, FIGSIZES

apply_paper_style()

# Local shorthand onto the shared palette
C = {
    'cvsa':      COLORS['cvsa'],        # red (CVSA-c4 primary)
    'cvsa_c8':   COLORS['cvsa_light'],  # light blue (CVSA-c8 secondary)
    'cvsa_line': COLORS['iso'],         # neutral grey for the c2→c4→c8 connector
    'baseline':  COLORS['snapkv'],
    'dense':     COLORS['dense'],
    'fixed':     COLORS['fixed'],
    'iso':       COLORS['iso'],
}


def _cvsa_color(cname: str) -> str:
    """Per-config CVSA color: c2 deep red, c4 crimson (primary), c8 light blue."""
    return {
        'c2': COLORS['cvsa_accent'],
        'c4': COLORS['cvsa'],
        'c8': COLORS['cvsa_light'],
    }.get(cname, COLORS['cvsa'])


def _cvsa_marker(cname: str) -> str:
    """Per-config CVSA marker: c2=up-tri, c4=circle (primary), c8=down-tri.
    Down-tri (not square) so c8 stays visually distinct from Fixed-sparse 's'.
    """
    return {'c2': '^', 'c4': 'o', 'c8': 'v'}.get(cname, 'o')


# ─── Bootstrap helpers ─────────────────────────────────────────────
def bootstrap_ci(per_sample_by_task, n_boot=10000, seed=42):
    """Hierarchical bootstrap over samples within tasks.

    Returns (mean, ci_lo, ci_hi) — mean is the point estimate of the
    task-averaged metric; CI is 95% equal-tailed.
    """
    rng = np.random.default_rng(seed)
    tasks = list(per_sample_by_task.keys())
    # Point estimate: mean over tasks of per-task sample mean
    point = float(np.mean([np.mean(per_sample_by_task[t]) for t in tasks]))
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        task_avgs = np.empty(len(tasks))
        for i, t in enumerate(tasks):
            arr = np.asarray(per_sample_by_task[t], dtype=np.float64)
            idx = rng.integers(0, len(arr), size=len(arr))
            task_avgs[i] = arr[idx].mean()
        boot_means[b] = task_avgs.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_bootstrap_all():
    """Compute bootstrap CIs for all (model, method) pairs with per-sample data.

    Cached to results/pareto_bootstrap_cache.json to keep re-runs fast.
    """
    cache_path = "results/pareto_bootstrap_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # (model display name, per-sample json path, method key -> pareto label)
    # Method keys are the historical ssa_* keys stored in the result files.
    SOURCES = [
        ("Llama-3.1-8B", "results/longctx_quality_llama.json", {
            "dense": "dense",
            "ssa_5_90_c4_ar20_dL0": "c4",
            "ssa_5_90_c8_ar20_dL0": "c8",
            "fixed_10": "Fixed-10%",
        }),
        ("Qwen2.5-7B", "results/bootstrap_qwen.json", {
            "dense": "dense",
            "ssa_5_90_c4_ar20_dL0": "c4",
        }),
        ("Mistral-7B", "results/bootstrap_mistral.json", {
            "dense": "dense",
            "ssa_5_90_c4_ar20_dL0": "c4",
        }),
    ]
    out = {}
    for model, path, method_map in SOURCES:
        with open(path) as f:
            d = json.load(f)
        pss = d["per_sample_scores"]
        out[model] = {}
        for src_key, label in method_map.items():
            if src_key not in pss:
                continue
            sample_map = pss[src_key]
            # Skip if any null
            if any(v is None for v in sample_map.values()):
                continue
            mean, lo, hi = bootstrap_ci(sample_map)
            out[model][label] = {
                "mean": round(mean, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            }
    with open(cache_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ─── Data (B=32, 8K context, H100 80GB) ──────────────────────────
# tok/s = 32 * 1000 / ms; latencies from results/verify_ar_b32.json (5-rep median).
# Quality: point estimates overridden by bootstrap means where per-sample data exists.

# Baseline quality sources (KVPress canonical):
#   SnapKV / Ada-KV: NVIDIA KVPress (kvpress_baselines_*.json), token-level paper-faithful.
#   H2O / Quest:     our paged-decode reimpl (16task_baselines.json, qwen7b_16task.json,
#                    ar_mistral7b_v03.json) -- both within published Δ-vs-dense ranges.
# Latencies for ALL baselines are from our paged-decode pipeline; KVPress runs in HF SDPA
# and does not produce comparable latency, so Pareto positions combine canonical quality
# (KVPress) with achievable latency (paged) -- footnoted in §4.2.
MODELS = {
    'Llama-3.1-8B': {
        'dense':    {'q': 42.57, 'ms': 127.99},
        'c2':       {'q': 42.46, 'ms': 97.85},   # AR20-c2 dL0
        'c4':       {'q': 41.87, 'ms': 72.70},   # AR20-c4 dL0 (recommended)
        'c8':       {'q': 41.18, 'ms': 72.67},   # AR20-c8 dL0
        'SnapKV':   {'q': 40.64, 'ms': 75.4},    # KVPress canonical (was paged 37.26)
        'H2O':      {'q': 38.94, 'ms': 112.8},   # paged reimpl (in published range)
        'Ada-KV':   {'q': 41.02, 'ms': 90.0},    # KVPress canonical (was paged 39.59)
        'Quest':    {'q': 37.82, 'ms': 139.8},   # paged reimpl
        'Fixed-10%': {'q': 39.36, 'ms': 78.5},
    },
    'Qwen2.5-7B': {
        'dense':    {'q': 40.18, 'ms': 162.40},
        'c2':       {'q': 38.39, 'ms': 77.84},
        'c4':       {'q': 38.54, 'ms': 66.54},
        'c8':       {'q': 37.59, 'ms': 66.36},
        'SnapKV':   {'q': 36.85, 'ms': 64.1},    # KVPress canonical (was paged 32.70)
        'H2O':      {'q': 32.88, 'ms': 80.1},    # paged reimpl
        'Ada-KV':   {'q': 37.01, 'ms': 75.8},    # KVPress canonical (was paged 33.63)
        'Quest':    {'q': 36.03, 'ms': 125.5},   # paged reimpl
        'Fixed-10%': {'q': 33.68, 'ms': 67.5},
    },
    'Mistral-7B': {
        'dense':    {'q': 42.73, 'ms': 127.90},
        'c2':       {'q': 42.53, 'ms': 97.19},
        'c4':       {'q': 41.71, 'ms': 73.17},
        'c8':       {'q': 40.59, 'ms': 73.13},
        'SnapKV':   {'q': 38.19, 'ms': 73.5},    # KVPress canonical (was paged 34.55)
        'H2O':      {'q': 38.19, 'ms': 108.9},   # paged reimpl
        'Ada-KV':   {'q': 38.32, 'ms': 87.1},    # KVPress canonical (was paged 37.82)
        'Quest':    {'q': 37.17, 'ms': 141.0},   # paged reimpl
        'Fixed-10%': {'q': None, 'ms': 77.8},
    },
}

def ms_to_tps(ms, batch=32):
    return batch * 1000.0 / ms


def apply_bootstrap_overrides(models, ci_data):
    """Use bootstrap point estimates for dots; attach ci_lo/ci_hi for error bars."""
    for model, methods in ci_data.items():
        for label, stats in methods.items():
            if model in models and label in models[model]:
                models[model][label]['q'] = stats['mean']
                models[model][label]['ci_lo'] = stats['ci_lo']
                models[model][label]['ci_hi'] = stats['ci_hi']


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Pareto frontier (main paper) — 3 panels
# ═══════════════════════════════════════════════════════════════════

def plot_pareto_main():
    ci_data = load_bootstrap_all()
    apply_bootstrap_overrides(MODELS, ci_data)

    # Top-strip shared legend identifies every marker; inline labels only
    # used for Dense (anchor) and the CVSA-c4 callout (headline result).
    # figsize tuned for high space utilization at \textwidth (~5.5") display;
    # gridspec wspace bumped so each panel's y-axis label has clear horizontal
    # room from the previous panel's right edge.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.4), sharey=False,
                             gridspec_kw={'wspace': 0.32})

    for ax, (model_name, data) in zip(axes, MODELS.items()):
        dense_q = data['dense']['q']
        dense_t = ms_to_tps(data['dense']['ms'])

        # --- Baselines (no CIs in our data — single-run summaries) ---
        baseline_names = ['SnapKV', 'H2O', 'Ada-KV', 'Quest']
        # Place every baseline label CENTERED directly below its marker.
        # Centering removes left/right ambiguity (which marker does this
        # label belong to?) and keeps the label as close as possible to
        # its anchor point. Baseline markers are all at distinct (x, y)
        # so center-below labels do not collide with each other across
        # any of the three panels.
        BL_DEFAULT = (0, -16, 'center')
        BL_OVERRIDE = {
            # Llama: SnapKV at upper-left of marker so it doesn't stack with
            # Fixed-10% (which sits ~17 data units to its left). With SnapKV
            # vacated upward, Fixed-10% returns to default center-below —
            # well-separated vertically from SnapKV's new upper position.
            ('Llama-3.1-8B', 'SnapKV'): (5, 8, 'right'),
            # Qwen Ada-KV (422) and SnapKV (499) sit close enough that their
            # tagged bboxes collide at center-below. Shift Ada-KV to the
            # marker's lower-left so its right edge clears SnapKV's box.
            ('Qwen2.5-7B', 'Ada-KV'): (-10, -16, 'center'),
        }
        def bl_place(name):
            return BL_OVERRIDE.get((model_name, name), BL_DEFAULT)
        # Per-baseline grey shades + small inline labels with bracketed
        # reference numbers. Numbering follows the unsrtnat numeric
        # bibliography order (FlashAttention=1, vLLM=2 appear first in the
        # intro, so H2O / SnapKV / Ada-KV / Quest get [3]-[6]).
        # Unified single grey for all baselines (Ada-KV's tone) — per-method
        # shading was too subtle to read at print size and added noise.
        baseline_colors = {
            'SnapKV': COLORS['adakv'],
            'H2O':    COLORS['adakv'],
            'Ada-KV': COLORS['adakv'],
            'Quest':  COLORS['adakv'],
        }
        baseline_refs = {
            'H2O': 3, 'SnapKV': 4, 'Ada-KV': 5, 'Quest': 6,
        }
        for name in baseline_names:
            d = data[name]
            q, t = d['q'], ms_to_tps(d['ms'])
            bc = baseline_colors[name]
            dx, dy, ha = bl_place(name)
            ax.scatter(t, q, c=bc, s=85, zorder=5,
                       marker='D', edgecolors='white', linewidth=0.7,
                       alpha=0.95)
            ax.annotate(f"{name} [{baseline_refs[name]}]",
                        (t, q), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.5, ha=ha,
                        color=bc, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.95))

        # --- Fixed sparse (point only — CI dropped for cross-panel consistency:
        # only Llama has per-sample bootstrap data for Fixed-10%, so showing
        # a CI on Llama alone would be visually inconsistent across panels.
        # Dense + CVSA-c* keep CIs since those are the focal comparisons.)
        if 'Fixed-10%' in data and data['Fixed-10%']['q'] is not None:
            d = data['Fixed-10%']
            q, t = d['q'], ms_to_tps(d['ms'])
            ax.scatter(t, q, c=C['fixed'], s=55, zorder=5,
                       marker='s', edgecolors='white', linewidth=0.7)
            dx, dy, ha = bl_place('Fixed-10%')
            ax.annotate("Fixed-10%", (t, q), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8.5, ha=ha,
                        color=C['fixed'], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.95))

        # --- Dense (with CI) ---
        if 'ci_lo' in data['dense']:
            d = data['dense']
            yerr = [[dense_q - d['ci_lo']], [d['ci_hi'] - dense_q]]
            ax.errorbar(dense_t, dense_q, yerr=yerr, fmt='none',
                        ecolor=C['dense'], elinewidth=1.2, capsize=3,
                        capthick=1.0, alpha=0.7, zorder=4)
        ax.scatter(dense_t, dense_q, c=C['dense'], s=110, zorder=6,
                   marker='*', edgecolors='white', linewidth=0.7)
        # Dense shifted further left in Llama/Mistral so it doesn't merge
        # with the CVSA-c4 callout (the C prefix made the callout wider).
        # Qwen has more room and -25 would cross the y-axis tick labels,
        # so it keeps the original -8 offset.
        dense_dx = -8 if model_name == 'Qwen2.5-7B' else -25
        ax.annotate("Dense", (dense_t, dense_q), textcoords="offset points",
                    xytext=(dense_dx, 10), fontsize=14, color=C['dense'],
                    fontweight='bold')

        # --- CVSA curve ---
        cvsa_configs = []
        for cname in ['c2', 'c4', 'c8']:
            if cname in data:
                d = data[cname]
                cvsa_configs.append((cname, d['q'], ms_to_tps(d['ms']),
                                     d.get('ci_lo'), d.get('ci_hi')))
        cvsa_configs.sort(key=lambda x: x[2])

        quals_s = [x[1] for x in cvsa_configs]
        tputs_s = [x[2] for x in cvsa_configs]
        ax.plot(tputs_s, quals_s, c=C['cvsa_line'], linewidth=2, alpha=0.45, zorder=3)

        # Callout extends UP-LEFT from the c4 marker (ha='right' below) so the
        # text doesn't overflow the panel into the next panel's y-axis.
        # Per-model dy: Qwen has the biggest Dense–c4 gap (~1.8 pts), so the
        # default dy lands the callout BELOW the dotted dense reference line;
        # bump dy=38 to lift the whole callout above the line. Mistral c2 sits
        # at y=42.53 — only ~5 pdf points above the default callout — so the
        # red callout merges with the c2 triangle; bump dy=32 to clear it.
        c4_callout = {
            'Qwen2.5-7B':  (-12, 38),
            'Mistral-7B':  (-12, 32),
        }.get(model_name, (-12, 22))

        for cname, q, t, lo, hi in cvsa_configs:
            is_rec = cname == 'c4'
            cfg_color = _cvsa_color(cname)
            cfg_marker = _cvsa_marker(cname)
            if lo is not None:
                yerr = [[q - lo], [hi - q]]
                ax.errorbar(t, q, yerr=yerr, fmt='none', ecolor=cfg_color,
                            elinewidth=1.3, capsize=3, capthick=1.1,
                            alpha=0.75, zorder=6)
            # Halo behind the recommended c4 marker for extra visual pop
            if is_rec:
                ax.scatter(t, q, c=cfg_color, s=380, zorder=6.5,
                           marker=cfg_marker, alpha=0.20, linewidth=0)
            ax.scatter(t, q, c=cfg_color,
                       s=220 if is_rec else 90, zorder=7,
                       marker=cfg_marker, edgecolors='white',
                       linewidth=1.6 if is_rec else 1.1)
            if is_rec:
                speedup = t / dense_t
                ax.annotate(f"CVSA-c4 ({speedup:.2f}×)",
                            (t, q), textcoords="offset points",
                            xytext=c4_callout, ha='right',
                            fontsize=14, color=COLORS['cvsa'], fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=COLORS['cvsa'],
                                            lw=1.5, shrinkA=2, shrinkB=12))

        # --- Dense reference dotted line (Dense label is already inline) ---
        ax.axhline(y=dense_q, color=C['dense'],
                   linestyle=':', alpha=0.40, linewidth=0.8)

        ax.set_xlabel("Throughput (tok/s, B=32, 8K)", fontsize=14.5)
        ax.set_ylabel("Quality (16-task LongBench)", fontsize=14.5)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True)

        ax.set_title(model_name, fontsize=16, fontweight='bold', pad=10)

        # Axis limits — adaptive per model, with extra headroom
        all_t = [ms_to_tps(data[k]['ms']) for k in data]
        all_q = [data[k]['q'] for k in data if data[k]['q'] is not None]
        # Extra left padding so Quest's centered-below label (Quest is the
        # leftmost data point in Llama/Mistral) stays clear of the y-axis.
        ax.set_xlim(min(all_t) * 0.88, max(all_t) * 1.10)
        ax.set_ylim(min(all_q) - 1.8, max(all_q) + 2.4)

    # Compact top-strip legend (6 entries, one row) — identifies every marker
    # at a glance while keeping the strip thin so panels claim most of the
    # vertical space.
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=C['dense'],
               markersize=12, label='Dense'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['cvsa_accent'],
               markersize=9, label='CVSA-c2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['cvsa'],
               markeredgecolor='white', markeredgewidth=1.0,
               markersize=11, label='CVSA-c4 (recommended)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['cvsa_light'],
               markersize=9, label='CVSA-c8'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['adakv'],
               markersize=8, label='Baselines (10%)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C['fixed'],
               markersize=8, label='Fixed sparse (10%)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 1.10), ncol=6, framealpha=0.92,
               edgecolor='0.85', fontsize=12.5,
               handletextpad=0.4, columnspacing=1.5)

    # Reserve top ~14% for the legend strip + breathing room above panel
    # titles; panels claim the remaining ~86%.
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    os.makedirs("paper/figures", exist_ok=True)
    for fmt in ['pdf', 'png']:
        fig.savefig(f"paper/figures/pareto_frontier.{fmt}", dpi=300, bbox_inches='tight')
        print(f"Saved paper/figures/pareto_frontier.{fmt}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Tradeoff curve (main paper) — Llama-8B, quality vs throughput
# ═══════════════════════════════════════════════════════════════════

def plot_tradeoff_curve():
    """Quality--speed tradeoff as check interval M varies (Llama-8B)."""
    ci_data = load_bootstrap_all()
    llama_ci = ci_data.get('Llama-3.1-8B', {})

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZES['double'])

    def q_of(label, default):
        return llama_ci.get(label, {}).get('mean', default)
    def ci_of(label):
        d = llama_ci.get(label, {})
        return (d.get('ci_lo'), d.get('ci_hi'))

    dense_q = q_of('dense', 42.57)
    dense_ms = 127.99
    dense_lo, dense_hi = ci_of('dense')

    configs = [
        ("c2",  q_of('c2', 42.46), 97.85, *ci_of('c2')),  # no CI
        ("c4",  q_of('c4', 41.87), 72.70, *ci_of('c4')),
        ("c8",  q_of('c8', 41.18), 72.67, *ci_of('c8')),
    ]
    names = [c[0] for c in configs]
    quals = [c[1] for c in configs]
    tputs = [ms_to_tps(c[2]) for c in configs]

    ax.plot(tputs, quals, '-', color=C['cvsa'], linewidth=2, alpha=0.55, zorder=3)

    for n, q, ms, lo, hi in configs:
        t = ms_to_tps(ms)
        if lo is not None:
            yerr = [[q - lo], [hi - q]]
            ax.errorbar(t, q, yerr=yerr, fmt='none', ecolor=C['cvsa'],
                        elinewidth=1.3, capsize=3, capthick=1.1,
                        alpha=0.75, zorder=4)

    for n, q, t in zip(names, quals, tputs):
        is_rec = n == 'c4'
        ax.scatter(t, q, c=C['cvsa'], s=110 if is_rec else 80, zorder=5,
                   marker='o', edgecolors='white', linewidth=1.2)

    offsets = [(10, 10), (-75, -18), (10, 6)]
    for i, (n, q, t) in enumerate(zip(names, quals, tputs)):
        is_rec = n == 'c4'
        sp = t / ms_to_tps(dense_ms)
        if is_rec:
            ax.annotate(f"CVSA-{n} ({sp:.2f}×)\n(recommended)", (t, q),
                        textcoords="offset points", xytext=offsets[i],
                        fontsize=9, fontweight='bold', color=C['cvsa'],
                        arrowprops=dict(arrowstyle='->', color=C['cvsa'],
                                        lw=1, shrinkA=2, shrinkB=14))
        else:
            ax.annotate(f"{n} ({sp:.2f}×)", (t, q), textcoords="offset points",
                        xytext=offsets[i], fontsize=8.5, fontweight='bold',
                        color=C['cvsa'])

    dense_t = ms_to_tps(dense_ms)
    if dense_lo is not None:
        yerr = [[dense_q - dense_lo], [dense_hi - dense_q]]
        ax.errorbar(dense_t, dense_q, yerr=yerr, fmt='none',
                    ecolor=C['dense'], elinewidth=1.2, capsize=3,
                    capthick=1.0, alpha=0.7, zorder=4)
    ax.scatter(dense_t, dense_q, c=C['dense'], s=90, marker='*',
               edgecolors='white', linewidth=0.7, zorder=6)
    ax.annotate("Dense", (dense_t, dense_q), textcoords="offset points",
                xytext=(-8, 10), fontsize=9, color=C['dense'], fontweight='bold')

    ax.axhline(y=dense_q, color=C['dense'], linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(y=dense_q - 1, color=C['iso'], linestyle='--',
               alpha=0.45, linewidth=0.7, label='Dense − 1 pt iso-quality')

    ax.set_xlabel("Throughput (tok/s, B=32, 8K)")
    ax.set_ylabel("Quality (16-task LongBench)")
    ax.set_xlim(dense_t * 0.9, max(tputs) * 1.08)
    lo_y = min([c[1] for c in configs] + [v for v in [dense_lo] if v is not None])
    hi_y = max([dense_q] + [v for v in [dense_hi] if v is not None])
    ax.set_ylim(lo_y - 0.6, hi_y + 0.8)
    ax.grid(True)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(f"paper/figures/tradeoff_curve.{fmt}", dpi=300, bbox_inches='tight')
        print(f"Saved paper/figures/tradeoff_curve.{fmt}")
    plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Generating paper figures...\n")
    plot_pareto_main()
    # plot_tradeoff_curve()  # Deprecated — redundant with pareto_frontier c2/c4/c8 points; keep function for ad-hoc use
    print("\nDone.")
