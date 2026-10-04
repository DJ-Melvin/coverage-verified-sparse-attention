# Experiment launchers

Slurm launchers for the experiments in the paper. Each is a template: set
`#SBATCH --account`, the partition, and, if needed, `ENV_BIN` (a Python
environment's `bin/`) and `HF_HOME` before submitting. Older launchers use
literal `cd /path/to/cvsa` and `export PATH=/path/to/env/bin:$PATH` lines;
replace both with your repository and environment paths. Submit from the
repository root and create `logs/` **before** `sbatch`, since Slurm opens its
log files before the script's own `mkdir` runs. Most take the model as
the first argument (`llama`, `mistral`, `qwen`). The LongBench launchers
added for the camera-ready version also accept `MAX_SAMPLES` and `TASKS`
overrides for a smoke test:

```bash
mkdir -p logs
MAX_SAMPLES=5 TASKS=hotpotqa OUT=results/smoke_fixed_llama.json \
  sbatch scripts/run_fixed_budget_noverify.sh llama
```

[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) lists which launcher backs which
table or figure.

Use a separate output file for smoke tests. `eval/longbench_cvsa.py` resumes
only when the model, block size, sample count, tasks, methods and sampling
settings match. A failed cell is saved and retried by the same command;
incomplete runs return a nonzero exit code.

## Submission experiments

| Purpose | Script |
| --- | --- |
| LongBench, CVSA and baselines | `run_longbench.sh`, `run_16task_baselines.sh`, `run_bootstrap_data.sh`, `run_c8_missing_16task.sh` |
| Threshold sweep, random recovery | `run_tau_ablation_extended.sh`, `run_random_ar_baseline.sh` |
| Dense-layer and recovery ablations | `run_dense_layer_ablation_canonical.sh`, `run_exact_vs_ar_canonical.sh` |
| KVPress baselines | `run_kvpress_baselines_{llama8b,mistral7b,qwen7b,llama70b}.sh` |
| 70B quality | `run_70b_ar_quality_fixed.sh` |
| RULER | `run_final_ruler_{llama,mistral,qwen}.sh`, `run_final_ruler_longctx_*.sh`, `run_final_ruler_longctx_eviction_baselines.sh` |
| Latency and throughput | `run_pareto_latency.sh`, `run_latency_final.sh`, `run_latency_final_mq.sh`, `run_latency_longctx.sh` |
| Coverage diagnostics and traces | `run_diagnostics.sh`, `run_diagnostics_mistral_qwen.sh`, `run_diagnostics_multimodel_v03.sh`, `run_hybrid_coverage_trace.sh`, `run_recovery_traces_multimodel.sh` |
| Theorem and proxy validation | `run_theorem_validation_scaled.sh`, `run_theorem_validation_ctok.sh` |

## Camera-ready experiments

| Purpose | Script |
| --- | --- |
| Matched effective compute | `run_fixed_budget_noverify.sh`, `run_static_policy_ablation.sh` |
| Selector-agnostic gate, fully lightweight mode | `run_selector_agnostic.sh`, `run_lightnorm.sh` |
| Top-p selector with and without the gate | `run_twip_composition.sh` |
| Tactic reproduction | `run_tactic_validation.sh`, `run_tactic_sink_probe.sh` |
| Twilight realized-coverage audit | `run_twilight_longbench.sh` (needs `TWILIGHT_ROOT`) |
| Single paged pipeline for all baselines | `run_baseline_paged_decode.sh` |
| Proxy calibration on the production distribution | `run_production_proxy_calibration.sh` |
| Long-context throughput and kernel breakdown | `run_longctx_mgpu.sh` |
| M = 1 LongBench row | `run_c1_longbench_sweep.sh` |
| Coverage traces across M; 32K-step decode | `run_coverage_trace_msweep.sh`, `run_longdecode_trace.sh` |
| Stochastic decoding | `run_sampling_probe.sh` |
| MATH-500 reasoning | `run_math_reasoning.sh` |

## Other

- `make_figures.sh`: redraws every paper figure on CPU from `results/`.
- `collect_page_traces.py`: per-step trace collection used by
  `run_recovery_traces_multimodel.sh`.
- `compute_paired_ci.py`: paired bootstrap intervals against the KVPress
  baselines.
