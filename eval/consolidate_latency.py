#!/usr/bin/env python3
"""Consolidate all latency results into unified tables.

Scans results/*.json for latency data in various key formats and produces:
  - results/latency_consolidated.json  — canonical (model, method, ctx, B) -> metrics
  - results/latency_all.csv            — flat CSV for inspection
  - results/tables/latency_main.tex    — B=32 main paper table
  - results/tables/latency_longctx.tex — long-context B=1 and B=8 table
  - results/tables/latency_70b.tex     — 70B table

Handles the following key formats:
  - {method}@{ctx}@B{B}           (latency_benchmark.py output, longctx files)
  - {method}_ctx{N}_B{B}          (70b_ar20_latency.json, 70b_baselines.json)
  - {method}_B{B}                 (llama8b_latency_final.json etc., implicit ctx=8192)

Run locally (no GPU needed) once all SLURM jobs are done:
    python eval/consolidate_latency.py
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict


# ─── File-to-model mapping ────────────────────────────────────────
MODEL_OF_FILE = [
    # (filename substring, model tag) — first match wins
    ("llama8b", "Llama-8B"),
    ("ar20_lat_llama", "Llama-8B"),
    ("longctx_lat_llama", "Llama-8B"),
    ("longctx_baselines_llama", "Llama-8B"),
    ("bigbatch_16k_b32_llama", "Llama-8B"),
    ("bigbatch_32k_b16_llama", "Llama-8B"),
    ("qwen7b", "Qwen-7B"),
    ("ar20_lat_qwen", "Qwen-7B"),
    ("longctx_lat_qwen", "Qwen-7B"),
    ("longctx_baselines_qwen", "Qwen-7B"),
    ("bigbatch_16k_b32_qwen", "Qwen-7B"),
    ("bigbatch_32k_b16_qwen", "Qwen-7B"),
    ("mistral7b_latency_final", "Mistral-7B"),
    ("ar_mistral7b_v03", "Mistral-7B"),
    ("ar20_lat_mistral", "Mistral-7B"),
    ("longctx_lat_mistral", "Mistral-7B"),
    ("longctx_baselines_mistral", "Mistral-7B"),
    ("bigbatch_16k_b32_mistral", "Mistral-7B"),
    ("bigbatch_32k_b16_mistral", "Mistral-7B"),
    ("70b", "Llama-70B"),
]

# Files to skip (pre-AR / obsolete / non-latency)
SKIP_SUBSTRINGS = [
    "niah", "ruler", "quality", "diagnostics", "baselines_15pct",
    "error_bars", "ablation", "16task", "burst_full", "burst_qwen",
    "hybrid", "pareto", "approx_recovery_quality", "approx_recovery_ruler",
    "page_traces", "cache_sim", "temporal", "tiered", "final_70b",
    "longbench", "final_ruler",
]


# ─── Method normalization ─────────────────────────────────────────
# Raw keys are the method strings stored in the latency result files
# (historical ssa_* prefix); values are the CVSA display labels.
METHOD_LABEL = {
    "dense": "Dense",
    "fixed_5": "Fixed-5%",
    "fixed_10": "Fixed-10%",
    "ssa_5_90_c2_ar20_dL0": "CVSA-c2",
    "ssa_5_90_c4_ar20_dL0": "CVSA-c4",
    "ssa_5_90_c8_ar20_dL0": "CVSA-c8",
    # Current spelling written by the launchers.
    "cvsa_5_90_c2_ar20_dL0": "CVSA-c2",
    "cvsa_5_90_c4_ar20_dL0": "CVSA-c4",
    "cvsa_5_90_c8_ar20_dL0": "CVSA-c8",
    # Aliases used by ar_mistral7b_v03.json — same config as above (dL0 included via base_kw)
    "ar20_c2": "CVSA-c2",
    "ar20_c4": "CVSA-c4",
    "ar20_c8": "CVSA-c8",
    "snapkv_10": "SnapKV",
    "h2o_10": "H2O",
    "adakv_10": "Ada-KV",
    "quest_10": "Quest",
}

# Display order for paper tables
METHOD_ORDER = [
    "Dense", "Fixed-10%",
    "CVSA-c2", "CVSA-c4", "CVSA-c8",
    "SnapKV", "H2O", "Ada-KV", "Quest",
]


# ─── Key parsing ──────────────────────────────────────────────────
RE_AT_FORMAT = re.compile(r"^(.+)@(\d+)@B(\d+)$")
RE_CTX_FORMAT = re.compile(r"^(.+)_ctx(\d+)_B(\d+)$")
RE_B_FORMAT = re.compile(r"^(.+)_B(\d+)$")


def parse_key(key, default_ctx=8192):
    """Parse a result key into (method, ctx, batch). Returns None if unrecognized."""
    m = RE_AT_FORMAT.match(key)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = RE_CTX_FORMAT.match(key)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = RE_B_FORMAT.match(key)
    if m:
        return m.group(1), default_ctx, int(m.group(2))
    return None


def extract_ms(val):
    """Extract latency ms from a result dict. Handles multiple key names."""
    if not isinstance(val, dict):
        return None
    for k in ("median_ms", "mean_ms"):
        if k in val and isinstance(val[k], (int, float)):
            return float(val[k])
    return None


def extract_tps(val):
    for k in ("median_tps", "tokens_per_sec"):
        if k in val and isinstance(val[k], (int, float)):
            return float(val[k])
    return None


def file_to_model(filename):
    for substr, tag in MODEL_OF_FILE:
        if substr in filename:
            return tag
    return None


# ─── Main ─────────────────────────────────────────────────────────
# Files nested by model tag at top level (e.g. {"llama": {...}, "qwen": {...}})
NESTED_BY_MODEL_TAG = {
    "llama": "Llama-8B",
    "qwen": "Qwen-7B",
    "mistral": "Mistral-7B",
}


def scan_results(results_dir="results"):
    """Return dict keyed by (model, method_label, ctx, B) -> metrics."""
    consolidated = {}
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".json"):
            continue
        if any(s in filename for s in SKIP_SUBSTRINGS):
            continue
        if "latency" not in filename and "lat_" not in filename \
                and "bigbatch" not in filename and "70b_ar20" not in filename \
                and "70b_baselines" not in filename and "ar20_lat" not in filename \
                and "ar_mistral7b_v03" not in filename and "burst_lat" not in filename \
                and "verify_ar" not in filename and "longctx_baselines" not in filename:
            continue

        path = os.path.join(results_dir, filename)
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"[skip] {filename}: {e}", file=sys.stderr)
            continue

        # Nested-by-model-tag format (e.g. verify_ar_b32.json)
        if isinstance(data, dict) and set(data.keys()) & set(NESTED_BY_MODEL_TAG.keys()):
            for tag, model in NESTED_BY_MODEL_TAG.items():
                sub = data.get(tag)
                if not isinstance(sub, dict):
                    continue
                _ingest_dict(sub, model, filename, consolidated)
            continue

        model = file_to_model(filename)
        if model is None:
            print(f"[warn] {filename}: no model mapping", file=sys.stderr)
            continue

        # Find the dict of results
        candidates = []
        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], dict):
                candidates.append(data["results"])
            if "latency" in data and isinstance(data["latency"], dict):
                candidates.append(data["latency"])
            # Also treat top-level as results dict if it has result-shaped values
            candidates.append(data)

        for results_dict in candidates:
            _ingest_dict(results_dict, model, filename, consolidated)
    return consolidated


def _ingest_dict(results_dict, model, filename, consolidated):
    for key, val in results_dict.items():
        parsed = parse_key(key)
        if parsed is None:
            continue
        method_raw, ctx, B = parsed
        method_label = METHOD_LABEL.get(method_raw)
        if method_label is None:
            continue
        ms = extract_ms(val)
        if ms is None:
            continue
        tps = extract_tps(val) or (B * 1000.0 / ms)
        key3 = (model, method_label, ctx, B)
        entry = {
            "median_ms": ms,
            "median_tps": tps,
            "source": filename,
        }
        prev = consolidated.get(key3)
        if prev is None:
            consolidated[key3] = entry
        else:
            if _prefer(filename, prev["source"]):
                consolidated[key3] = entry


def _prefer(new_file, old_file):
    """Return True if new_file should override old_file."""
    # Preference order (higher = better)
    def score(f):
        s = 0
        if "verify_ar_b32" in f: s += 12      # 5-repeat canonical for AR at 8K×B=32
        if "bigbatch" in f: s += 10
        if "longctx_baselines" in f: s += 10
        if "70b_baselines" in f: s += 10
        if "ar20_lat" in f: s += 8
        if "longctx_lat" in f: s += 7
        if "70b_ar20" in f: s += 7
        if "ar_mistral7b_v03" in f: s += 6
        if "_latency_final" in f: s += 5
        if "latency_final" in f: s += 4
        return s
    return score(new_file) > score(old_file)


# ─── Output writers ───────────────────────────────────────────────
def write_csv(consolidated, path):
    # Compute speedup relative to dense for the same (model, ctx, B)
    dense = {(m, c, b): v["median_ms"]
             for (m, meth, c, b), v in consolidated.items() if meth == "Dense"}
    rows = []
    for (model, method, ctx, B), v in sorted(consolidated.items()):
        d_ms = dense.get((model, ctx, B))
        sp = (d_ms / v["median_ms"]) if d_ms else None
        rows.append([
            model, method, ctx, B,
            f"{v['median_ms']:.2f}",
            f"{v['median_tps']:.1f}",
            f"{sp:.2f}" if sp else "",
            v["source"],
        ])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "ctx", "batch", "median_ms",
                    "median_tps", "speedup_vs_dense", "source"])
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def _make_pivot(consolidated, model, cells, methods=None):
    """Build rows for a model at given (ctx, B) cells.

    Returns dict: method -> list of (ms, speedup) aligned with cells.
    """
    if methods is None:
        methods = METHOD_ORDER
    rows = {}
    for method in methods:
        row = []
        for (ctx, B) in cells:
            v = consolidated.get((model, method, ctx, B))
            if v is None:
                row.append((None, None))
            else:
                d = consolidated.get((model, "Dense", ctx, B))
                sp = (d["median_ms"] / v["median_ms"]) if d else None
                row.append((v["median_ms"], sp))
        rows[method] = row
    return rows


def _fmt_cell(ms, sp):
    if ms is None:
        return "—"
    if sp is None:
        return f"{ms:.1f}"
    return f"{ms:.1f} ({sp:.2f}$\\times$)"


def _tex(label):
    """Escape `%` for LaTeX (unescaped `%` is a comment)."""
    return label.replace("%", "\\%")


def write_latex_main(consolidated, path):
    """B=32 main table: 3 × 7B models at 8K/16K × B=32."""
    cells = [(8192, 32), (16384, 32)]
    models = ["Llama-8B", "Mistral-7B", "Qwen-7B"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Main latency table (B=32, 8K + 16K)\n")
        f.write("\\begin{tabular}{l" + "cc" * len(models) + "}\n")
        f.write("\\toprule\n")
        f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in models) + " \\\\\n")
        f.write("Method & " + " & ".join("8K & 16K" for _ in models) + " \\\\\n")
        f.write("\\midrule\n")
        for method in METHOD_ORDER:
            cells_row = []
            for model in models:
                rows = _make_pivot(consolidated, model, cells, [method])
                for (ms, sp) in rows[method]:
                    cells_row.append(_fmt_cell(ms, sp))
            if all(c == "—" for c in cells_row):
                continue
            f.write(f"{_tex(method)} & " + " & ".join(cells_row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Wrote {path}")


def write_latex_longctx(consolidated, path):
    """Long-context table at B=1 and B=8 for 3 × 7B models."""
    ctxs = [8192, 32768, 65536, 131072]
    batches = [1, 8]
    models = ["Llama-8B", "Mistral-7B", "Qwen-7B"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Long-context latency (B=1, B=8)\n")
        for B in batches:
            cells = [(c, B) for c in ctxs]
            f.write(f"\n% --- B={B} ---\n")
            f.write("\\begin{tabular}{l" + "c" * len(ctxs) * len(models) + "}\n")
            f.write("\\toprule\n")
            f.write(" & " + " & ".join(
                f"\\multicolumn{{{len(ctxs)}}}{{c}}{{{m}}}" for m in models) + " \\\\\n")
            f.write("Method & " + " & ".join(
                " & ".join(f"{c//1024}K" for c in ctxs) for _ in models) + " \\\\\n")
            f.write("\\midrule\n")
            for method in METHOD_ORDER:
                cells_row = []
                for model in models:
                    rows = _make_pivot(consolidated, model, cells, [method])
                    for (ms, sp) in rows[method]:
                        cells_row.append(_fmt_cell(ms, sp))
                if all(c == "—" for c in cells_row):
                    continue
                f.write(f"{_tex(method)} & " + " & ".join(cells_row) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
    print(f"Wrote {path}")


def write_latex_70b(consolidated, path):
    """70B latency table."""
    cells = [(8192, 1), (8192, 4), (8192, 8),
             (32768, 1), (32768, 4), (65536, 1)]
    model = "Llama-70B"
    with open(path, "w", encoding="utf-8") as f:
        f.write("% 70B latency table\n")
        f.write("\\begin{tabular}{l" + "c" * len(cells) + "}\n")
        f.write("\\toprule\n")
        f.write("Method & " + " & ".join(
            f"{c//1024}K B={b}" for (c, b) in cells) + " \\\\\n")
        f.write("\\midrule\n")
        rows = _make_pivot(consolidated, model, cells)
        for method in METHOD_ORDER:
            row = rows[method]
            if all(ms is None for (ms, _) in row):
                continue
            f.write(f"{_tex(method)} & " + " & ".join(
                _fmt_cell(ms, sp) for (ms, sp) in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Wrote {path}")


def print_summary(consolidated):
    """Print a coverage matrix for quick inspection."""
    models = sorted(set(k[0] for k in consolidated))
    for model in models:
        print(f"\n{'=' * 60}")
        print(f"{model}")
        print(f"{'=' * 60}")
        # Build (ctx, B) -> set of methods
        cells = defaultdict(set)
        for (m, meth, c, b), _ in consolidated.items():
            if m == model:
                cells[(c, b)].add(meth)
        # Sort and print
        for (c, b) in sorted(cells.keys()):
            methods_here = sorted(cells[(c, b)], key=lambda x: METHOD_ORDER.index(x)
                                  if x in METHOD_ORDER else 99)
            print(f"  ctx={c:>6} B={b:<3}: {', '.join(methods_here)}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    os.makedirs("results/tables", exist_ok=True)

    print("Scanning results/ …")
    consolidated = scan_results("results")
    print(f"Found {len(consolidated)} (model, method, ctx, B) cells.\n")

    print_summary(consolidated)

    # Save canonical json
    as_dict = {
        f"{m}|{meth}|{c}|{b}": v
        for (m, meth, c, b), v in consolidated.items()
    }
    with open("results/latency_consolidated.json", "w") as f:
        json.dump(as_dict, f, indent=2)
    print("\nWrote results/latency_consolidated.json")

    write_csv(consolidated, "results/latency_all.csv")
    write_latex_main(consolidated, "results/tables/latency_main.tex")
    write_latex_longctx(consolidated, "results/tables/latency_longctx.tex")
    write_latex_70b(consolidated, "results/tables/latency_70b.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()
