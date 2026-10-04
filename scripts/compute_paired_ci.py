"""Compute paired bootstrap CIs between CVSA-c4 and KVPress baselines.

Both pipelines use deterministic prefix sampling, so the first 50 samples per
task are paired across methods. CVSA per-sample scores are stored on a 0-100
scale; KVPress scores are stored on a 0-1 scale and are converted to 0-100.

Output: results/cvsa_vs_kvpress_paired_ci.json
"""

import json
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODELS = [
    (
        "Llama-3.1-8B",
        "results/longctx_quality_llama.json",
        "results/kvpress_baselines_llama8b.json",
    ),
    (
        "Mistral-7B-v0.3",
        "results/bootstrap_mistral.json",
        "results/kvpress_baselines_mistral7b.json",
    ),
    (
        "Qwen2.5-7B",
        "results/bootstrap_qwen.json",
        "results/kvpress_baselines_qwen7b.json",
    ),
]

# Method key as stored in the per-sample result files (historical ssa_* prefix).
CVSA_KEY = "ssa_5_90_c4_ar20_dL0"
N_BOOT = 10_000
N_PAIR = 50
OUT_PATH = "results/cvsa_vs_kvpress_paired_ci.json"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


results = {}

for model, cvsa_fp, kvp_fp in MODELS:
    cvsa = load_json(cvsa_fp)["per_sample_scores"]
    kvp = load_json(kvp_fp)["per_sample_scores"]

    common = []
    for task in cvsa.get(CVSA_KEY, {}):
        if (
            task in kvp.get("hf_snapkv", {})
            and kvp["hf_snapkv"][task]
            and task in kvp.get("hf_adakv", {})
            and kvp["hf_adakv"][task]
            and len(cvsa[CVSA_KEY][task]) >= N_PAIR
            and len(kvp["hf_snapkv"][task]) >= N_PAIR
            and len(kvp["hf_adakv"][task]) >= N_PAIR
        ):
            common.append(task)

    if not common:
        results[model] = {"error": "no eligible tasks"}
        continue

    rng = np.random.default_rng(42)
    methods = {
        "CVSA-c4": {t: np.array(cvsa[CVSA_KEY][t][:N_PAIR], float) for t in common},
        "SnapKV": {t: np.array(kvp["hf_snapkv"][t][:N_PAIR], float) * 100 for t in common},
        "AdaKV": {t: np.array(kvp["hf_adakv"][t][:N_PAIR], float) * 100 for t in common},
    }
    dists = {m: np.empty(N_BOOT) for m in methods}
    for b in range(N_BOOT):
        idx = {t: rng.integers(0, N_PAIR, size=N_PAIR) for t in common}
        for method, arrays in methods.items():
            task_means = [arrays[t][idx[t]].mean() for t in common]
            dists[method][b] = np.mean(task_means)

    model_out = {
        "n_common_tasks": len(common),
        "n_samples_per_task": N_PAIR,
        "n_bootstrap_resamples": N_BOOT,
        "common_tasks": common,
        "methods": {},
        "pairwise_delta": {},
    }
    for method in methods:
        lo, hi = np.percentile(dists[method], [2.5, 97.5])
        model_out["methods"][method] = {
            "mean": float(dists[method].mean()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
    for method in ["SnapKV", "AdaKV"]:
        delta = dists["CVSA-c4"] - dists[method]
        lo, hi = np.percentile(delta, [2.5, 97.5])
        model_out["pairwise_delta"][f"CVSA_minus_{method}"] = {
            "mean": float(delta.mean()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "significant_5pct": bool((lo > 0) or (hi < 0)),
        }
    results[model] = model_out

    print(f"\n=== {model} ===")
    print(f"  ({len(common)} tasks, paired n={N_PAIR})")
    for method in methods:
        value = model_out["methods"][method]
        print(f"  {method:<10} {value['mean']:>6.2f}  [{value['ci_lo']:.2f}, {value['ci_hi']:.2f}]")
    for method in ["SnapKV", "AdaKV"]:
        delta = model_out["pairwise_delta"][f"CVSA_minus_{method}"]
        sig = "significant" if delta["significant_5pct"] else "overlaps 0"
        print(
            f"    delta(CVSA-{method}) = {delta['mean']:+.2f}  "
            f"[{delta['ci_lo']:+.2f}, {delta['ci_hi']:+.2f}]  {sig}"
        )

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "description": (
                "Paired bootstrap 95% CI for CVSA-c4 vs KVPress SnapKV/AdaKV. "
                "Both runs use deterministic prefix sampling so the first n=50 "
                "samples per task are paired across methods."
            ),
            "methodology": {
                "n_samples_paired": N_PAIR,
                "n_bootstrap_resamples": N_BOOT,
                "resample_strategy": "within-task index resampling, shared across methods (paired)",
                "random_seed": 42,
                "cvsa_source_files": {m[0]: m[1] for m in MODELS},
                "kvpress_source_files": {m[0]: m[2] for m in MODELS},
                "cvsa_method_key": CVSA_KEY,
                "kvpress_method_keys": ["hf_snapkv", "hf_adakv"],
                "scale_normalization": "CVSA stored as 0-100; KVPress stored as 0-1",
            },
            "results": results,
        },
        f,
        indent=2,
    )
print(f"\nSaved: {OUT_PATH}")
