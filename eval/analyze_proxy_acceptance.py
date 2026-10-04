"""Analyze page-proxy accept/reject decisions against token-level coverage.

The error_bound_validation_*_ctok.json files flatten results in the loop order
used by eval/error_bound_validation.py:

  prompt, layer, budget in [3,5,10,15]%, tau in [0.80,0.90,0.95], q-head.

This script reconstructs those fields and reports proxy false-accept and
false-reject rates relative to token-level coverage c*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BUDGETS = [0.03, 0.05, 0.10, 0.15]
TAUS = [0.80, 0.90, 0.95]
Q_HEADS = {"llama": 32, "mistral": 32, "qwen": 28}


def model_key(path: Path) -> str:
    name = path.name
    for key in Q_HEADS:
        if key in name:
            return key
    raise ValueError(f"Cannot infer model key from {path}")


def reconstruct_rows(path: Path):
    key = model_key(path)
    q_heads = Q_HEADS[key]
    data = json.loads(path.read_text())
    layers = data["_meta"]["layers_tested"]
    n_prompts = data["_meta"]["n_prompts"]
    expected = n_prompts * len(layers) * len(BUDGETS) * len(TAUS) * q_heads
    if expected != len(data["coverage"]):
        raise ValueError(f"{path}: expected {expected} rows, found {len(data['coverage'])}")

    rows = []
    idx = 0
    for prompt in range(n_prompts):
        for layer in layers:
            for budget in BUDGETS:
                for tau in TAUS:
                    for head in range(q_heads):
                        rows.append(
                            {
                                "prompt": prompt,
                                "layer": int(layer),
                                "budget": budget,
                                "tau": tau,
                                "head": head,
                                "coverage_proxy": float(data["coverage"][idx]),
                                "coverage_token": float(data["coverage_token"][idx]),
                                "proxy_accept": bool(data["accept"][idx]),
                                "actual_error": float(data["actual_error"][idx]),
                                "relative_error": float(data["relative_error"][idx]),
                            }
                        )
                        idx += 1
    return key, rows


def summarize(rows, tau: float, budget: float | None = None):
    subset = [r for r in rows if r["tau"] == tau and (budget is None or r["budget"] == budget)]
    n = len(subset)
    proxy_accept = [r["proxy_accept"] for r in subset]
    token_accept = [r["coverage_token"] >= tau for r in subset]
    p = sum(proxy_accept)
    t = sum(token_accept)
    false_accept = sum(a and not b for a, b in zip(proxy_accept, token_accept))
    false_reject = sum((not a) and b for a, b in zip(proxy_accept, token_accept))
    agree = sum(a == b for a, b in zip(proxy_accept, token_accept))
    return {
        "n": n,
        "proxy_accept": p,
        "token_accept": t,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "proxy_accept_rate": p / n if n else 0.0,
        "token_accept_rate": t / n if n else 0.0,
        "false_accept_all_rate": false_accept / n if n else 0.0,
        "false_accept_proxy_rate": false_accept / p if p else None,
        "false_reject_all_rate": false_reject / n if n else 0.0,
        "false_reject_token_rate": false_reject / t if t else None,
        "agreement_rate": agree / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/error_bound_validation_llama_ctok.json",
            "results/error_bound_validation_mistral_ctok.json",
            "results/error_bound_validation_qwen_ctok.json",
        ],
    )
    parser.add_argument("--output", default="results/proxy_acceptance_analysis.json")
    args = parser.parse_args()

    out = {}
    for raw in args.inputs:
        key, rows = reconstruct_rows(Path(raw))
        out[key] = {
            "tau_0.90_all_budgets": summarize(rows, 0.90),
            "tau_0.90_budget_5pct": summarize(rows, 0.90, 0.05),
            "tau_sweep_all_budgets": {str(t): summarize(rows, t) for t in TAUS},
        }

        # Qwen has the only material proxy drift; layer breakdown helps locate it.
        if key == "qwen":
            layer_summary = {}
            for layer in sorted({r["layer"] for r in rows}):
                layer_rows = [r for r in rows if r["layer"] == layer]
                layer_summary[str(layer)] = summarize(layer_rows, 0.90, 0.05)
            out[key]["tau_0.90_budget_5pct_by_layer"] = layer_summary

    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
