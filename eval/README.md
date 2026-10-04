# Evaluation, analysis and plotting

Run everything from the repository root.

## Evaluation entry points (GPU)

- `longbench_cvsa.py`: LongBench quality for dense, CVSA, fixed-budget and
  baseline methods (method grammar in its `make_backend` docstring).
- `longbench_harness.py`: the LongBench harness (prompts, metrics, paged
  decode runner).
- `baselines.py`: paged SnapKV, AdaKV, H2O and Quest.
- `ruler_cvsa.py`: RULER retrieval tasks.
- `latency_benchmark.py`, `unified_latency.py`, `unified_latency_mgpu.py`:
  decode latency and throughput (single GPU, and layer-sharded across GPUs
  for long contexts; `--timing` adds the per-kernel breakdown).
- `diagnostics.py`, `diagnostics_head.py`, `diagnostics_longctx.py`,
  `diagnostics_hybrid_trace.py`: coverage landscape, per-head recovery and
  per-step coverage traces behind the diagnostic figures.
- `error_bound_validation.py`, `production_proxy_calibration.py`: token-level
  validation of the error equality and of the page-level coverage proxy.
- `math_reasoning_eval.py`: MATH-500 with DeepSeek-R1-Distill-Llama-8B.

## Analysis

- `analyze_fa_error.py`: per-cell error of proxy false accepts.
- `analyze_longdecode_drift.py`: per-bin coverage and recovery of a long
  decode trace.
- `summarize_tactic.py`: merges and summarizes the Tactic runs.
- `analyze_proxy_acceptance.py`, `consolidate_latency.py`,
  `build_ruler_table.py`, `build_static_budget_table.py`: table builders.

## Figures

`plot_*.py` read `results/` and write to `paper/figures/`;
`bash scripts/make_figures.sh` runs all of them. `_paper_style.py` holds the
shared style.
