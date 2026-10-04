#!/bin/bash
# Redraw every figure in the paper from the cached results in results/.
#
# CPU only; needs numpy, matplotlib and (for the coverage-motivation figure) a
# CPU build of torch. Figures are written to paper/figures/.
#
# Usage (from the repository root):
#   bash scripts/make_figures.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
for script in \
    plot_pareto \
    plot_coverage_motivation \
    plot_ruler_main \
    plot_longbench_per_task \
    plot_hybrid_coverage_trace \
    plot_budget_fairness_compact \
    plot_budget_fairness \
    plot_theory_validation \
    plot_coverage_heatmap \
    plot_recovery_heatmap \
    plot_bandwidth_roofline; do
  echo "=== ${script}"
  "${PYTHON}" "eval/${script}.py"
done
echo "Figures written to paper/figures/"
