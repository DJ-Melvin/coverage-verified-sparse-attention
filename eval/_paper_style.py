"""Shared paper-level plotting style for CVSA figures.

Import and use:
    from _paper_style import apply_paper_style, COLORS, MARKERS, LINESTYLES, FIGSIZES
    apply_paper_style()

This module centralizes:
  - Matplotlib rcParams (Helvetica/Arial sans-serif, sizes tuned for NeurIPS body text)
  - Semantic color palette (red CVSA accent + greyscale baselines, minimalist flat)
  - Marker / linestyle conventions (CVSA=o, Dense=*, baseline=D, Fixed=s)
  - Figure-size constants aligned with NeurIPS single/double-column widths

Goal: every figure in the paper looks like it came from the same authors.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ─── Palette (Red/light-blue CVSA accent; everything else greyscale) ─────
# Design: CVSA-c4 = soft red (primary emphasis); CVSA-c8 = light blue;
# Dense/baselines/Fixed = grey scale (dark→light). Overall lighter feel —
# fewer "punchy" colors, red is the only warm note.
COLORS = {
    # Methods — the only non-grey in the palette
    'cvsa':      '#D62728',   # bright red (CVSA-c4, primary emphasis); was MIT crimson #A31F34
    'cvsa_light':'#7AAED0',   # light blue (CVSA-c8, secondary)
    'cvsa_accent':'#A02020',  # deeper red for heavy emphasis (paired with brighter cvsa)

    # Reference + structural baseline: grey
    'dense':     '#333333',   # dark grey — reference (distinguished by * marker)
    'fixed':     '#888888',   # mid grey — same family as baselines

    # Eviction baselines: light grey gradient (not too dark)
    'snapkv':    '#707070',
    'h2o':       '#858585',
    'adakv':     '#9C9C9C',
    'quest':     '#B2B2B2',   # lightest

    # Per-model (for multi-model scaling plots) — single-hue warm + cool
    'llama':     '#7AAED0',   # light blue
    'mistral':   '#888888',   # grey
    'qwen':      '#C44E52',   # red

    # Semantics / decoration (lighter than before)
    'iso':       '#BBBBBB',
    'grid':      '#EDEDED',   # very light grid
    'annot':     '#444444',
}


# ─── Marker conventions ──────────────────────────────────────────────────
MARKERS = {
    'dense':     '*',
    'cvsa':      'o',
    'cvsa_c2':   's',   # only used when distinguishing c2/c4/c8 in one panel
    'cvsa_c4':   'o',
    'cvsa_c8':   '^',
    'baseline':  'D',
    'fixed':     's',

    # Per-baseline marker (so color + marker both identify method)
    'snapkv':    'v',
    'h2o':       '^',
    'adakv':     's',
    'quest':     'D',
}

MARKERSIZES = {
    'dense':     140,
    'cvsa':      95,
    'cvsa_c4':   120,   # emphasized (recommended config)
    'cvsa_other':65,
    'baseline':  55,
    'fixed':     50,
}


# ─── Linestyles ──────────────────────────────────────────────────────────
LINESTYLES = {
    'main':      '-',    # primary method
    'reference': ':',    # dense reference (horizontal line)
    'iso':       '--',   # iso-quality thresholds
    'baseline':  '-',    # baseline curves (same solid, distinguish by color)
    'pending':   (0, (3, 3)),  # incomplete / future data
}


# ─── Figure sizes (inches, aligned with NeurIPS column widths) ───────────
# NeurIPS 2026 template: text width ≈ 5.5 in (single column sty), wide figures span.
FIGSIZES = {
    'single':    (3.3, 2.4),     # 1-column small plot
    'single_tall':(3.3, 3.5),    # 1-column taller
    'double':    (6.75, 3.0),    # full-width 1-panel
    'double_tall':(6.75, 4.0),   # full-width, slightly taller
    'wide_2p':   (11.0, 3.8),    # 2 panels side-by-side
    'wide_3p':   (13.5, 3.8),    # 3 panels side-by-side
    'wide_4p':   (14.5, 3.6),    # 4 panels side-by-side (rare)
    'grid_3x2':  (11.0, 7.0),    # 3×2 grid
    'grid_6x3':  (13.5, 8.5),    # 6×3 grid (appendix per-task × model)
}


# ─── rcParams ────────────────────────────────────────────────────────────
RC_PARAMS = {
    # Fonts — sans-serif (Arial first; Helvetica only on macOS/Linux) for cleaner figure aesthetic
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
    'mathtext.fontset': 'dejavusans',

    # Sizes — tuned for NeurIPS 10pt body
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'legend.title_fontsize': 9,

    # Lines & markers (tightened further — lighter feel)
    'lines.linewidth': 1.2,
    'lines.markersize': 4.5,
    'lines.markeredgewidth': 0.6,

    # Axes
    'axes.linewidth': 0.6,
    'axes.edgecolor': '0.35',
    'axes.labelcolor': '0.15',
    'axes.titlecolor': '0.15',

    # Ticks
    'xtick.color': '0.35',
    'ytick.color': '0.35',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,

    # Grid (much lighter — subtle guide, doesn't compete with data)
    'axes.grid': False,       # turn on per-axes, not globally
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.35,
    'grid.linewidth': 0.4,

    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '0.85',
    'legend.fancybox': False,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.3,

    # PDF / PS — embed Type 42 fonts to avoid "Type 3" submission warnings
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # Savefig
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,

    # Misc
    'figure.autolayout': False,  # we'll call tight_layout manually
}


# ─── Helpers ─────────────────────────────────────────────────────────────
def apply_paper_style() -> None:
    """Apply the shared paper-level matplotlib style. Call at the top of each plot script."""
    mpl.rcParams.update(RC_PARAMS)


def method_color(method: str) -> str:
    """Look up the semantic color for a method name. Case-insensitive, tolerant of '-' / '_'."""
    key = method.lower().replace('-', '').replace('_', '').replace(' ', '')
    alias = {
        'cvsa': 'cvsa', 'cvsac2': 'cvsa_light', 'cvsac4': 'cvsa', 'cvsac8': 'cvsa_light',
        'dense': 'dense',
        'fixed': 'fixed', 'fixed10': 'fixed', 'fixed5': 'fixed', 'fixedsparse': 'fixed',
        'snapkv': 'snapkv', 'snapkv10': 'snapkv',
        'h2o': 'h2o', 'h2o10': 'h2o',
        'adakv': 'adakv', 'adakv10': 'adakv', 'adakvm': 'adakv',
        'quest': 'quest', 'quest10': 'quest',
        'llama': 'llama', 'llama318b': 'llama', 'llama3170b': 'llama',
        'mistral': 'mistral', 'mistral7b': 'mistral',
        'qwen': 'qwen', 'qwen257b': 'qwen',
    }
    palette_key = alias.get(key, key)
    return COLORS.get(palette_key, '#666666')


def method_marker(method: str) -> str:
    """Look up the semantic marker for a method name."""
    key = method.lower().replace('-', '').replace('_', '').replace(' ', '')
    alias = {
        'dense': 'dense',
        'cvsa': 'cvsa', 'cvsac2': 'cvsa_c2', 'cvsac4': 'cvsa_c4', 'cvsac8': 'cvsa_c8',
        'fixed': 'fixed', 'fixed10': 'fixed', 'fixed5': 'fixed',
        'snapkv': 'snapkv', 'h2o': 'h2o', 'adakv': 'adakv', 'quest': 'quest',
    }
    marker_key = alias.get(key, 'baseline')
    return MARKERS.get(marker_key, 'D')


def legend_handle(method: str, label: str | None = None, **kwargs):
    """Build a Line2D legend handle for a method using its semantic color + marker."""
    color = method_color(method)
    marker = method_marker(method)
    return Line2D(
        [0], [0],
        marker=marker, color='w',
        markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.0,
        markersize=kwargs.pop('markersize', 7),
        linestyle=kwargs.pop('linestyle', '-'),
        linewidth=kwargs.pop('linewidth', 1.4),
        label=label or method,
        **kwargs,
    )


def add_iso_quality_line(ax, y_value: float, delta: float = -1.0, label: str = 'Dense −1 pt') -> None:
    """Add a horizontal dashed line at (y_value + delta) as an iso-quality reference."""
    ax.axhline(y=y_value + delta, color=COLORS['iso'],
               linestyle=LINESTYLES['iso'], alpha=0.55, linewidth=0.7)


def clean_spines(ax) -> None:
    """Remove top/right spines for a cleaner paper look (keep left/bottom)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
