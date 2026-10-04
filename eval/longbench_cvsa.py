#!/usr/bin/env python3
"""LongBench quality evaluation for CVSA.

Compares CVSA at various budget/tau settings against dense and fixed-sparse
baselines on all 16 LongBench tasks, using the evaluation runner in
eval/longbench_harness.py (LongBenchE2ERunner, E2EGenerator).

Usage:
    cd /path/to/cvsa
    python eval/longbench_cvsa.py \
        --tasks hotpotqa+triviaqa+qasper+multifieldqa_en \
        --methods dense,fixed_10,cvsa_10_90,cvsa_10_95 \
        --max-samples 50 \
        --output results/longbench_cvsa.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvsa.runtime.model_loader import ModelForDecode
from cvsa.runtime.dense import DenseAttentionBackend
from eval.longbench_harness import LongBenchE2ERunner
from eval.baselines import make_backend as make_baseline_backend
from cvsa.backend import CVSABackend


# All 16 LongBench tasks
ALL_TASKS = [
    "hotpotqa", "triviaqa", "qasper", "multifieldqa_en",
    "2wikimqa", "narrativeqa", "musique",
    "gov_report", "multi_news", "samsum", "qmsum",
    "trec", "lcc", "repobench-p",
    "passage_count", "passage_retrieval_en",
]


def make_backend(method: str, config, block_size: int):
    """Factory for attention backends.

    Method format:
        dense                — FlashInfer dense baseline
        fixed_{B}[_adasel]   — Fixed sparse, B% budget, no verification
                               (adasel: AdaKV-style cross-head allocation)
        cvsa_{B}_{T}         — CVSA, B% budget, tau=T/100 (fixed interval R=16)
        cvsa_{B}_{T}_c{M}    — Lazy scheduling with check interval M
        cvsa_{B}_{T}_cv{N}   — Coverage gate every N steps on cached page
                               scores (with c{M}, N < M)
        cvsa_{B}_{T}_ar{P}   — Approximate recovery, expand to P%
                               (without ar: exact recovery via complement +
                               LSE merge)
        cvsa_{B}_{T}_dL{L}   — layer L always dense (e.g. dL0, dL0.1.2.3)
        cvsa_{B}_{T}_adasel  — AdaKV-style cross-head page allocation
        cvsa_{B}_{T}_rand{P} — counterfactual: random per-(layer, step)
                               reject with probability P% instead of the gate
        cvsa_{B}_{T}_selq    — Quest min/max lightweight selector for page
                               RANKING; coverage gate + recovery unchanged
        cvsa_{B}_{T}_selq_ln — fully lightweight: Quest sketch drives selection
                               AND the gate's normalizer (no full-key scan
                               anywhere)
        cvsa_{B}_{T}_tp{P}   — Twilight-style top-p selector WITH the coverage
                               gate ("Twilight+CVSA" composition)
        staticpol[_dL{L}]    — Static offline per-(layer, head) budgets, no
                               verification (table JSON path via env var
                               CVSA_STATIC_BUDGET_TABLE)
        twip_{P}[_dL{L}]     — Twilight-style pre-hoc top-p selector (per-head
                               budgets = top-p mass on Quest-estimated weights),
                               NO verification (e.g. twip_85_dL0)
        tactic_{P}           — Tactic reproduction (token-level), target mass
                               P%; optional _g{T} (coverage gate at tau=T/100),
                               _sk{N} / _rc{N} (force-include N sink / recent
                               tokens). See cvsa/tactic.py.
        snapkv_* adakv_* h2o_* quest_*
                             — baselines from eval/baselines.py

    `ssa_` is accepted as an alias of `cvsa_` and builds the identical
    backend. Suffixes can be combined (e.g. cvsa_5_90_c4_ar20_dL0); an
    unrecognized cvsa_ suffix raises ValueError.

    Examples:
        dense, fixed_5, cvsa_10_90, cvsa_5_90_c4_ar20_dL0, twip_85_dL0
    """
    # `ssa_` is the legacy spelling of `cvsa_`; both build the same backend.
    if method.startswith("ssa_"):
        method = "cvsa_" + method[len("ssa_"):]

    if method == "dense":
        return DenseAttentionBackend(
            config.num_kv_heads, config.num_q_heads,
            config.head_dim, block_size)

    if method.startswith("fixed_"):
        parts = method.split("_")
        budget_pct = int(parts[1]) / 100.0
        adaptive = "adasel" in parts
        # CVSA with tau=0 → always accept → pure sparse, no verification
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=budget_pct,
            coverage_threshold=0.0,  # always accept = fixed sparse
            reuse_interval=16,
            adaptive_selection=adaptive,
        )

    if method.startswith("staticpol"):
        # Static offline per-(layer, head) budget policy (matched-compute
        # comparison): budgets fixed from OFFLINE coverage statistics, no
        # verification, no recovery. The DuoAttention-style control that
        # isolates dynamic routing from static budget shape.
        # Table JSON from eval/build_static_budget_table.py, path via
        # CVSA_STATIC_BUDGET_TABLE. Supported suffix: dL{L} (e.g.
        # staticpol_dL0). Selection ranking = same exact scorer as CVSA.
        table_path = os.environ.get("CVSA_STATIC_BUDGET_TABLE", "")
        if not table_path:
            raise ValueError(
                "staticpol_* requires CVSA_STATIC_BUDGET_TABLE=<table.json> "
                "(built by eval/build_static_budget_table.py)")
        with open(table_path) as f:
            table = json.load(f)
        dense_layers = None
        for p in method.split("_")[1:]:
            if p.startswith("dL"):
                dense_layers = [int(x) for x in p[2:].split(".")]
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=float(table.get("base_budget", 0.05)),
            coverage_threshold=0.0,  # always accept = no verification
            reuse_interval=16,       # same convention as fixed_* controls
            dense_layers=dense_layers,
            static_budget_table={int(l): v for l, v in table["layers"].items()},
        )

    if method.startswith("twip_"):
        # Twilight-style pre-hoc top-p selector, NO verification (the
        # adaptive-budget baseline): per-head adaptive budgets = smallest page set
        # reaching top-p cumulative mass on ESTIMATED (Quest min/max) weights.
        # twip_{P}[_dL{L}], e.g. twip_85_dL0. Gated counterpart:
        # cvsa_{B}_{T}_c{M}_ar{P}_dL{L}_tp{P}.
        parts = method.split("_")
        top_p = int(parts[1]) / 100.0
        dense_layers = None
        for p in parts[2:]:
            if p.startswith("dL"):
                dense_layers = [int(x) for x in p[2:].split(".")]
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=0.05,          # unused: top-p decides budgets
            coverage_threshold=0.0,   # always accept = pure pre-hoc top-p
            reuse_interval=16,        # same convention as fixed_* controls
            dense_layers=dense_layers,
            selector="quest",         # estimated weights, per Twilight
            top_p=top_p,
        )

    if method.startswith("cvsa_"):
        parts = method.split("_")
        budget_pct = int(parts[1]) / 100.0
        tau = int(parts[2]) / 100.0
        dense_layers = None
        check_interval = None
        coverage_interval = None
        approx_recovery_pct = None
        adaptive_selection = False
        random_reject_prob = None
        selector = "exact"
        top_p = None
        light_normalizer = False
        for p in parts[3:]:
            if p.startswith("dL"):
                dense_layers = [int(x) for x in p[2:].split(".")]
            elif p.startswith("cv") and p[2:].isdigit():
                # cv{N} sets coverage_interval independently from
                # check_interval. Used by the paper-described variant
                # (e.g. c4_cv1 => scoring every 4, coverage every step).
                # Must be matched before the generic c{N} branch below.
                coverage_interval = int(p[2:])
            elif p.startswith("c") and p[1:].isdigit():
                check_interval = int(p[1:])
            elif p.startswith("ar") and p[2:].isdigit():
                approx_recovery_pct = int(p[2:]) / 100.0
            elif p == "adasel":
                adaptive_selection = True
            elif p == "selq":
                # Quest min/max lightweight-indexer selector; coverage gate
                # unchanged (selector-agnostic experiment).
                selector = "quest"
            elif p == "ln":
                # Fully lightweight mode: the gate's
                # all-context normalizer is the SAME Quest sketch estimate
                # that drives selection, so no full-key scan runs anywhere.
                # Implies selector="quest".
                light_normalizer = True
                selector = "quest"
            elif p.startswith("tp") and p[2:].isdigit():
                # Twilight-style top-p budgeting on estimated weights; the
                # gate verifies realized coverage of the top-p selection
                # ("Twilight+CVSA" composition).
                top_p = int(p[2:]) / 100.0
                selector = "quest"
            elif p.startswith("rand") and p[4:].isdigit():
                # Counterfactual: override coverage with random reject at
                # matched rate (e.g., "rand20" = Bernoulli(0.20) per layer-step).
                # Used to prove coverage signal is targeted, not arbitrary.
                random_reject_prob = int(p[4:]) / 100.0
            else:
                raise ValueError(
                    f"Method '{method}': unrecognized suffix '{p}'")
        lazy = check_interval is not None
        return CVSABackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            budget_pct=budget_pct,
            coverage_threshold=tau,
            reuse_interval=16,
            dense_layers=dense_layers,
            lazy=lazy,
            check_interval=check_interval,
            coverage_interval=coverage_interval,
            approx_recovery_pct=approx_recovery_pct,
            adaptive_selection=adaptive_selection,
            random_reject_prob=random_reject_prob,
            selector=selector,
            top_p=top_p,
            light_normalizer=light_normalizer,
        )

    # Tactic reproduction.
    #   tactic_{P}        — faithful token-level reproduction at target mass P%
    # Implementation and underspecified-point choices: cvsa/tactic.py
    #   tactic_{P}_g{T}   — Tactic selection with our coverage gate at tau=T/100
    if method.startswith("tactic_"):
        from cvsa.tactic import TacticBackend
        parts = method.split("_")
        target_p = int(parts[1]) / 100.0
        tau = None
        sink_tokens = 0
        recent_tokens = 0
        for p in parts[2:]:
            if p.startswith("g") and p[1:].isdigit():
                tau = int(p[1:]) / 100.0
            elif p.startswith("sk") and p[2:].isdigit():
                sink_tokens = int(p[2:])      # A8 sink probe
            elif p.startswith("rc") and p[2:].isdigit():
                recent_tokens = int(p[2:])    # A8 recent-window probe
        return TacticBackend(
            num_kv_heads=config.num_kv_heads,
            num_q_heads=config.num_q_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_layers=config.num_layers,
            target_p=target_p,
            tau=tau,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
        )

    # Baseline methods from eval/baselines.py
    # (paged-decode reimplementations of SnapKV, AdaKV, H2O, Quest).
    _BASELINE_PREFIXES = ("snapkv_", "adakv_", "h2o_", "quest_")
    if any(method.startswith(p) for p in _BASELINE_PREFIXES):
        return make_baseline_backend(
            method, config, block_size, config.num_layers)

    raise ValueError(f"Unknown method: {method}")


def load_resume(args):
    """Read a compatible run before loading weights or touching its output."""
    if not os.path.exists(args.output):
        return None
    with open(args.output) as fh:
        previous = json.load(fh)
    config = previous.get("config") if isinstance(previous, dict) else None
    if not isinstance(config, dict):
        raise ValueError(
            f"Cannot resume {args.output}: missing run configuration. "
            "Use a different --output file.")

    # Older greedy runs predate the sampling flags; these were their defaults.
    legacy_defaults = {"temperature": 0.0, "top_p": 1.0, "sample_seed": 1234}
    differences = []
    for key in ("model", "block_size", "max_samples", "tasks", "methods",
                "temperature", "top_p", "sample_seed"):
        saved = config.get(key, legacy_defaults.get(key))
        requested = getattr(args, key)
        if saved != requested:
            differences.append(f"{key}: saved={saved!r}, requested={requested!r}")
    if differences:
        raise ValueError(
            f"Cannot resume {args.output} with different settings:\n  "
            + "\n  ".join(differences)
            + "\nUse a different --output file; the existing results were not changed.")
    return previous


def main():
    parser = argparse.ArgumentParser(
        description="LongBench quality evaluation for CVSA")
    parser.add_argument("--model",
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tasks",
                        default="+".join(ALL_TASKS))
    parser.add_argument("--methods",
                        default="dense,fixed_10,cvsa_10_90,cvsa_10_95")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output",
                        default="results/longbench_cvsa.json")
    # Stochastic-decoding probe: temperature<=0 (default) keeps the exact
    # greedy path; >0 enables seed-paired nucleus sampling in the runner.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=1234)
    args = parser.parse_args()

    try:
        prev = load_resume(args)
    except (ValueError, OSError) as e:
        parser.error(str(e))

    print("CVSA LongBench Quality Evaluation")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Methods: {args.methods}")
    print(f"  Max samples/task: {args.max_samples}")
    if args.temperature > 0:
        print(f"  Sampling: temp={args.temperature} top_p={args.top_p} "
              f"seed={args.sample_seed} (seed-paired across methods)")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = ModelForDecode(args.model, dtype=torch.bfloat16)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    config = model.config
    print(f"  {config.num_q_heads}Q/{config.num_kv_heads}KV, "
          f"D={config.head_dim}, {config.num_layers}L")
    print()

    runner = LongBenchE2ERunner(model, block_size=args.block_size)
    tasks = args.tasks.split("+")
    methods = args.methods.split(",")

    results = {m: {} for m in methods}
    all_predictions = {m: {} for m in methods}
    per_sample_scores = {m: {} for m in methods}
    cvsa_stats = {m: {} for m in methods}  # backend get_stats() per cell
    errors = {m: {} for m in methods}

    def completed_cells():
        return sum(results[m].get(t) is not None for m in methods for t in tasks)

    expected_cells = len(methods) * len(tasks)

    def save():
        """Write the output file. Called after every cell, not just at the end.

        A long sweep that only writes on completion loses everything to a
        queue time limit, which is how one 24-hour Tactic run was lost. Each
        cell is independent, so each one is worth keeping on its own.
        """
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "config": vars(args),
            "scores": results,
            "per_sample_scores": per_sample_scores,
            "predictions": all_predictions,
            "cvsa_stats": {m: dict(s) for m, s in cvsa_stats.items() if s},
            "status": "complete" if completed_cells() == expected_cells else "incomplete",
            "completed_cells": completed_cells(),
            "expected_cells": expected_cells,
            "errors": {m: dict(e) for m, e in errors.items() if e},
        }
        tmp = args.output + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, args.output)   # never leave a half-written file

    # Resume: a cell already present in the output file is not recomputed, so
    # a sweep too long for one allocation can be continued by resubmitting the
    # same command. Cells are keyed by (method, task) and carry their own
    # stats, so a resumed file is identical to one produced in a single run.
    # Files written before the rename store the stats under "ssa_stats".
    if prev is not None:
        if "cvsa_stats" not in prev and "ssa_stats" in prev:
            prev["cvsa_stats"] = prev["ssa_stats"]
        restored = 0
        for m in methods:
            for t in prev.get("scores", {}).get(m, {}):
                if t not in tasks or prev["scores"][m][t] is None:
                    continue
                results[m][t] = prev["scores"][m][t]
                for dst, key in ((all_predictions, "predictions"),
                                 (per_sample_scores, "per_sample_scores"),
                                 (cvsa_stats, "cvsa_stats")):
                    val = prev.get(key, {}).get(m, {}).get(t)
                    if val is not None:
                        dst[m][t] = val
                restored += 1
        print(f"Resuming from {args.output}: {restored} cells already done.\n")

    for task in tasks:
        print(f"\n{'─' * 70}")
        print(f"Task: {task}")
        print(f"{'─' * 70}")

        for method in methods:
            if results[method].get(task) is not None:
                print(f"  Method: {method}  (already done, skipping)", flush=True)
                continue
            print(f"  Method: {method}", flush=True)
            try:
                backend = make_backend(method, config, args.block_size)

                # Pass sampling kwargs only when sampling is requested, so
                # the greedy path calls the runner exactly as before
                # sampling support was added.
                sampling_kwargs = {}
                if args.temperature > 0:
                    sampling_kwargs = dict(temperature=args.temperature,
                                           top_p=args.top_p,
                                           sample_seed=args.sample_seed)
                score, elapsed, n, preds, sample_scores = \
                    runner.evaluate_dataset(task, backend, args.max_samples,
                                            **sampling_kwargs)

                results[method][task] = score
                all_predictions[method][task] = preds
                per_sample_scores[method][task] = sample_scores

                # Collect backend stats if available
                if hasattr(backend, 'get_stats'):
                    stats = backend.get_stats()
                    cvsa_stats[method][task] = stats
                    # Verification backends report accept rates; selection-only
                    # backends (e.g. the Tactic reproduction) report read share.
                    if stats.get('total_scoring_steps', 0) > 0:
                        extra = (f"  accept={stats['all_accepted_pct']:.0f}% "
                                 f"per_head={stats['per_head_accept_rate']:.0f}%")
                    elif 'effective_budget_pct' in stats:
                        extra = f"  read={stats['effective_budget_pct']:.1f}%"
                    else:
                        extra = ""
                else:
                    extra = ""

                print(f"  => {method:<16} {score:>6.2f}  "
                      f"({n} samples, {elapsed:.0f}s){extra}", flush=True)
                save()

            except Exception as e:
                import traceback
                traceback.print_exc()
                results[method][task] = None
                for cell_data in (all_predictions, per_sample_scores, cvsa_stats):
                    cell_data[method].pop(task, None)
                errors[method][task] = f"{type(e).__name__}: {e}"
                print(f"  => {method:<16} ERROR: {e}")
                save()

    # Summary table
    print(f"\n{'=' * 80}")
    print("LONGBENCH RESULTS — CVSA")
    print(f"{'=' * 80}")

    header = f"{'Task':<22}"
    for method in methods:
        header += f" {method:>12}"
    print(header)
    print("-" * len(header))

    for task in tasks:
        row = f"{task:<22}"
        for method in methods:
            val = results[method].get(task)
            if val is not None:
                row += f" {val:>12.2f}"
            else:
                row += f" {'ERROR':>12}"
        print(row)

    # Show completeness before aggregate scores. A partial task set is not
    # the requested benchmark, so do not label its average as the full mean.
    print("-" * len(header))
    row = f"{'COMPLETED TASKS':<22}"
    for method in methods:
        count = sum(results[method].get(t) is not None for t in tasks)
        row += f" {str(count) + '/' + str(len(tasks)):>12}"
    print(row)

    row = f"{'MEAN':<22}"
    for method in methods:
        vals = [results[method][t] for t in tasks
                if results[method].get(t) is not None]
        if vals and len(vals) == len(tasks):
            row += f" {np.mean(vals):>12.2f}"
        else:
            row += f" {'INCOMPLETE':>12}"
    print(row)

    # Delta from dense
    if "dense" in methods:
        row = f"{'Δ vs dense':<22}"
        for method in methods:
            if method == "dense":
                row += f" {'—':>12}"
            else:
                dense_vals = [results["dense"][t] for t in tasks
                              if results["dense"].get(t) is not None
                              and results[method].get(t) is not None]
                method_vals = [results[method][t] for t in tasks
                               if results["dense"].get(t) is not None
                               and results[method].get(t) is not None]
                if dense_vals and len(dense_vals) == len(tasks):
                    delta = np.mean(method_vals) - np.mean(dense_vals)
                    row += f" {delta:>+12.2f}"
                else:
                    row += f" {'INCOMPLETE':>12}"
        print(row)

    # CVSA acceptance stats summary
    gated_methods = [m for m in methods if m.startswith(("cvsa_", "ssa_"))]
    if gated_methods:
        print(f"\n{'=' * 80}")
        print("CVSA ACCEPTANCE STATISTICS")
        print(f"{'=' * 80}")
        for method in gated_methods:
            if method in cvsa_stats and cvsa_stats[method]:
                all_accept = [s["all_accepted_pct"]
                              for s in cvsa_stats[method].values()
                              if s.get("total_scoring_steps", 0) > 0]
                per_head = [s["per_head_accept_rate"]
                            for s in cvsa_stats[method].values()
                            if s.get("total_scoring_steps", 0) > 0]
                if all_accept:
                    print(f"  {method:<16}: all_accept={np.mean(all_accept):.1f}%  "
                          f"per_head={np.mean(per_head):.1f}%")

    # Save (already written after every cell; this is the final consistent copy)
    save()
    print(f"\nResults saved to {args.output}")
    if completed_cells() != expected_cells:
        print(f"INCOMPLETE: {completed_cells()}/{expected_cells} method/task cells completed. "
              "Rerun the same command to retry failed cells.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
