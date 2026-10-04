#!/usr/bin/env python
"""Summarize a Tactic reproduction run written by eval/longbench_cvsa.py.

Covers both the pre-registered validation gate and the A8 sink/recent probe.

Read share is aggregated as sum(tokens_read) / sum(tokens_max) over tasks,
the token-weighted basis that matches the pages-weighted convention the paper
quotes. The mean of the per-task percentages is a different quantity and is
deliberately not reported, so the two never get mixed up.

Quality is the unweighted mean over tasks, which is the LongBench convention,
and deltas are taken against the dense arm of the SAME run so that nothing is
compared across harnesses.

Usage:
    python eval/summarize_tactic.py results/camera_ready/tactic_sink_probe_llama.json
"""
from __future__ import annotations

import argparse
import json
import sys

# Tactic (arXiv 2502.12216) Table 2, Llama-3.1-8B: mean delta vs Full.
PAPER_DELTA = {70: -1.37, 90: -0.34}
PASS_BAND = 1.0    # |ours - paper| <= this  -> reproduction accepted
PARTIAL_BAND = 2.5  # within this            -> partial, must be disclosed


def is_gated(method: str) -> bool:
    """True if our coverage gate is composed onto this method (_g<tau>)."""
    return any(p.startswith("g") and p[1:].isdigit()
               for p in method.split("_")[1:])


def target_p_of(method: str, st: dict) -> int | None:
    """Percent target mass for a tactic method, from stats then from the name."""
    if st:
        for s in st.values():
            if s.get("target_p") is not None:
                return int(round(100 * float(s["target_p"])))
    parts = method.split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="+",
                    help="one run, or several disjoint task shards of one sweep")
    ap.add_argument("--baseline", default="dense",
                    help="method whose scores deltas are taken against")
    args = ap.parse_args()

    # Several files merge task-wise. This is only sound when each task's
    # methods, including the baseline, ran inside the same job, which is what
    # sharding a sweep by task produces. Sharding by METHOD instead would put
    # a method and its reference in different jobs and is not supported here.
    scores: dict = {}
    stats: dict = {}
    cfg: dict = {}
    seen: dict = {}
    for path in args.json_path:
        with open(path) as fh:
            blob = json.load(fh)
        cfg = cfg or blob.get("config", {})
        for m, per_task in blob["scores"].items():
            for t, v in per_task.items():
                if v is None:
                    continue
                if t in seen.setdefault(m, {}) and seen[m][t] != path:
                    print(f"refusing to merge: {m}/{t} appears in "
                          f"{seen[m][t]} and {path}")
                    return 2
                seen[m][t] = path
                scores.setdefault(m, {})[t] = v
        # "cvsa_stats" in current output; "ssa_stats" in files written
        # before the rename.
        blob_stats = blob.get("cvsa_stats", blob.get("ssa_stats", {}))
        for m, per_task in blob_stats.items():
            for t, s in per_task.items():
                stats.setdefault(m, {})[t] = s
    if len(args.json_path) > 1:
        n_tasks = len({t for per in scores.values() for t in per})
        print(f"merged {len(args.json_path)} shards -> {n_tasks} tasks\n")

    if args.baseline not in scores:
        print(f"baseline {args.baseline!r} not in this run: {list(scores)}")
        return 2

    print(f"model      : {cfg.get('model', '?')}")
    print(f"samples    : {cfg.get('max_samples', '?')} per task")
    all_tasks = sorted({t for per in scores.values() for t in per})
    print(f"tasks      : {'+'.join(all_tasks) if all_tasks else cfg.get('tasks', '?')}")
    print()

    base = scores[args.baseline]
    rows = []
    for method in scores:
        common = [t for t in base
                  if base.get(t) is not None and scores[method].get(t) is not None]
        if not common:
            continue
        q = sum(scores[method][t] for t in common) / len(common)
        qb = sum(base[t] for t in common) / len(common)
        st = stats.get(method, {})
        # Tactic selects tokens, CVSA selects pages; each backend logs its own
        # granularity. Both ratios are the same quantity, the fraction of the
        # KV cache actually read, so they belong in one column. The granularity
        # is recorded per row rather than folded away.
        read = tot = 0.0
        pages = pages_tot = 0.0
        unit = None
        for t in common:
            s = st.get(t, {})
            if s.get("total_tokens_max"):
                read += s.get("total_tokens_read", 0.0)
                tot += s["total_tokens_max"]
                unit = "tok"
            elif s.get("total_pages_max"):
                read += s.get("total_pages_attended", 0.0)
                tot += s["total_pages_max"]
                unit = "page"
            # Page share is what a paged serving stack would actually fetch.
            # A token selector names tokens but pays for the pages holding
            # them, so the two columns come apart exactly by how scattered
            # the selection is.
            if s.get("total_pages_max"):
                pages += (s.get("total_pages_touched")
                          if s.get("total_pages_touched") is not None
                          else s.get("total_pages_attended", 0.0))
                pages_tot += s["total_pages_max"]
        rows.append({
            "method": method,
            "n_tasks": len(common),
            "q": q,
            "d": q - qb,
            "read": 100.0 * read / tot if tot else None,
            "pages": 100.0 * pages / pages_tot if pages_tot else None,
            "unit": unit,
            "st": st,
        })

    order = {args.baseline: 0}
    rows.sort(key=lambda r: (order.get(r["method"], 1), r["method"]))

    print(f"{'method':<26} {'tasks':>5} {'quality':>8} {'delta':>8} "
          f"{'read%':>8} {'unit':>5} {'pages%':>8}")
    print("-" * 74)
    for r in rows:
        read = f"{r['read']:8.2f}" if r["read"] is not None else f"{'--':>8}"
        pg = f"{r['pages']:8.2f}" if r["pages"] is not None else f"{'--':>8}"
        delta = "     ---" if r["method"] == args.baseline else f"{r['d']:+8.2f}"
        print(f"{r['method']:<26} {r['n_tasks']:5d} {r['q']:8.2f} {delta} "
              f"{read} {(r['unit'] or ''):>5} {pg}")
    print()
    print("read%  is each method's own granularity; pages% is the share of pages")
    print("a paged serving stack must fetch for that same selection, which is the")
    print("column to compare when the deployment target reads whole pages.")

    # ---- verdict against the pre-registered criterion -------------------
    # Only an unmodified reproduction may be scored against the published
    # numbers. A variant carrying our coverage gate is a composition result:
    # matching the paper there would say nothing about fidelity. Older run
    # files may also hold an exact-score upper-bound arm ("tacticoracle_*"),
    # which is not a reproduction either.
    tac = [r for r in rows if r["method"].startswith("tactic")
           and not r["method"].startswith("tacticoracle")
           and not is_gated(r["method"])]
    if tac:
        print()
        print("verdict vs Tactic Table 2 (pre-registered, delta-based)")
        print("-" * 60)
        for r in tac:
            p = target_p_of(r["method"], r["st"])
            ref = PAPER_DELTA.get(p)
            if ref is None:
                print(f"{r['method']:<26} no published delta for P={p}")
                continue
            gap = abs(r["d"] - ref)
            mark = ("PASS" if gap <= PASS_BAND
                    else "PARTIAL" if gap <= PARTIAL_BAND else "FAIL")
            print(f"{r['method']:<26} ours {r['d']:+6.2f}  paper {ref:+6.2f}  "
                  f"|gap| {gap:4.2f}  {mark}")
        print()
        print("PASS    -> usable for comparison claims in the paper")
        print("PARTIAL -> usable only with the gap disclosed alongside")
        print("FAIL    -> must not back any comparison claim")

    # ---- composition: our gate on top of a Tactic selection --------------
    gated = [r for r in rows if r["method"].startswith("tactic")
             and is_gated(r["method"])]
    if gated:
        print()
        print("composition (our gate on a Tactic selection) -- NOT a fidelity test")
        print("-" * 66)
        for r in gated:
            stem = "_".join(p for p in r["method"].split("_")
                            if not (p.startswith("g") and p[1:].isdigit()))
            base_r = next((x for x in rows if x["method"] == stem), None)
            if base_r is None:
                print(f"{r['method']:<26} {r['d']:+6.2f}  (ungated arm not in this run)")
                continue
            dr = (r["read"] - base_r["read"]
                  if r["read"] is not None and base_r["read"] is not None else None)
            drs = f"{dr:+6.2f} pt read" if dr is not None else "read n/a"
            print(f"{r['method']:<26} {r['d'] - base_r['d']:+6.2f} quality vs "
                  f"{stem}   {drs}")

    # ---- A8: did force-inclusion buy quality, and at what traffic? -------
    probe = [r for r in rows if ("_sk" in r["method"] or "_rc" in r["method"])
             and not is_gated(r["method"])]
    if probe:
        ref_row = next((r for r in rows
                        if r["method"].startswith("tactic")
                        and not is_gated(r["method"])
                        and "_sk" not in r["method"] and "_rc" not in r["method"]), None)
        if ref_row is not None:
            print()
            print(f"A8 probe, against {ref_row['method']}")
            print("-" * 60)
            for r in probe:
                dq = r["d"] - ref_row["d"]
                dr = (r["read"] - ref_row["read"]
                      if r["read"] is not None and ref_row["read"] is not None else None)
                drs = f"{dr:+6.2f} pt read" if dr is not None else "read n/a"
                print(f"{r['method']:<26} {dq:+6.2f} quality   {drs}")
            print()
            print("Quality and traffic are reported together on purpose: a gain")
            print("bought by reading more is not the same finding as a gain from")
            print("a better selection, and only the second one closes A8.")

    # ---- fit diagnostics -------------------------------------------------
    diag = [r for r in rows if r["st"]]
    if diag:
        print()
        print("tail-fit diagnostics (task-mean of per-task percentages)")
        print("-" * 60)
        print(f"{'method':<26} {'neg_a':>7} {'neg_b':>7} {'singular':>9} {'gate_rej':>9}")
        for r in diag:
            vals = list(r["st"].values())
            if not vals:
                continue

            def m(key):
                xs = [v[key] for v in vals if v.get(key) is not None]
                return sum(xs) / len(xs) if xs else None

            def f(x):
                return f"{x:7.2f}" if x is not None else f"{'--':>7}"

            gr = m("gate_reject_pct")
            print(f"{r['method']:<26} {f(m('fit_neg_a_pct'))} {f(m('fit_neg_b_pct'))} "
                  f"{f(m('fit_singular_pct')):>9} "
                  f"{(f'{gr:9.2f}' if gr is not None else f'{chr(45)*2:>9}')}")
        print()
        print("neg_a is the load-bearing one: a negative slope means the fitted")
        print("tail contradicts the observed ordering. neg_b is ordinary for a")
        print("steep curve and is not a defect.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
