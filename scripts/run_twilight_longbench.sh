#!/bin/bash
# Twilight realized-coverage audit on LongBench, using the official Twilight code.
#
# Purpose:
#   Backs the Twilight realized-coverage audit (selector-agnostic appendix,
#   "Auditing a pre-hoc adaptive budget at its source"). Runs the released
#   Twilight implementation (https://github.com/tsinghua-ideal/Twilight) at
#   its default configuration -- Quest selector + hierarchical top-p pruner
#   (p=0.85) + 4-bit min/max estimator, ALGO_CONFIG=configs/config_quest_twi.json
#   -- on Llama-3.1-8B-Instruct (the same checkpoint as the CVSA runs), and
#   scores it with Twilight's own LongBench pred.py / eval.py. Twilight's
#   per-task logs record the exact-weight attention mass captured by its
#   selection and its achieved average budget (get_total_avg_budget).
#
#   Default TASKS is a four-task subset (qasper lcc hotpotqa
#   passage_retrieval_en) at MAX_SAMPLES=100 per task; pass TASKS
#   (space-separated LongBench task names) to run other tasks. The paper's
#   audit uses all 16 English tasks at MAX_SAMPLES=100.
#
# Setup (once):
#   git clone --recursive https://github.com/tsinghua-ideal/Twilight
#   git -C Twilight checkout ce6c050d63907840cb49df64200aad901dd7ea76
#   git -C Twilight submodule update   # flash-topk-attention @ d8803b2
#   git -C Twilight apply "$PWD/scripts/twilight_pred.patch"
#   Build Twilight's conda env per its README (default name `twi`, python
#   3.10; override with CONDA_ENV). The CVSA Python environment is not used.
#   scripts/twilight_pred.patch changes only Twilight's harness
#   (benchmark/LongBench/pred.py), not the method:
#     - a MAX_SAMPLES cap on the per-task sample loop (checked below), so each
#       task uses its first MAX_SAMPLES samples;
#     - sets generated_content in a decoding branch that left it unset
#       (crashed samsum and narrativeqa);
#     - num_logits_to_keep=1 on the prefill pass, to avoid OOM on the longest
#       LongBench inputs.
#   Dense reference arm: copy scripts/twilight_config_dense.json to
#   ${TWILIGHT_ROOT}/benchmark/configs/config_dense.json and run with
#   ALGO_CONFIG=configs/config_dense.json.
#
# Environment:
#   TWILIGHT_ROOT  (required) path to the Twilight checkout
#   MODELPATH      local path to the Llama-3.1-8B-Instruct weights
#   CUDA_HOME      CUDA toolkit for Twilight's JIT-compiled top-p kernel
#   GCC_BIN        bin dir of a GCC>=9 host compiler for that kernel
#
# Output:
#   ${TWILIGHT_ROOT}/benchmark/results_longbench/<MODEL>/, copied to
#   results/camera_ready/twilight_longbench_llama/ (override with CVSA_RESULTS).
#
# Usage:
#   TWILIGHT_ROOT=/path/to/Twilight \
#   MODELPATH=/path/to/Meta-Llama-3.1-8B-Instruct \
#   MAX_SAMPLES=100 \
#     sbatch scripts/run_twilight_longbench.sh

#SBATCH --job-name=twilight-longbench
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/twilight_longbench_%j.out
#SBATCH --error=logs/twilight_longbench_%j.err

set -euo pipefail

# ---- User-configurable (MUST verify) --------------------------------------
# Required: path to a checkout of https://github.com/tsinghua-ideal/Twilight.
TWILIGHT_ROOT=${TWILIGHT_ROOT:-}
if [ -z "${TWILIGHT_ROOT}" ]; then
  echo "ERROR: TWILIGHT_ROOT is not set." >&2
  echo "Clone Twilight and point TWILIGHT_ROOT at the checkout:" >&2
  echo "  git clone --recursive https://github.com/tsinghua-ideal/Twilight" >&2
  echo "  TWILIGHT_ROOT=/path/to/Twilight sbatch $0" >&2
  exit 2
fi
MODEL=${MODEL:-Meta-Llama-3.1-8B-Instruct}
# Required: local path to the model weights.
MODELPATH=${MODELPATH:-}
if [ -z "${MODELPATH}" ]; then
  echo "ERROR: MODELPATH is not set (local path to Meta-Llama-3.1-8B-Instruct)." >&2
  exit 2
fi
CONDA_ENV=${CONDA_ENV:-twi}
# Twilight method = Quest selector + hierarchical top-p pruner (NOT plain quest).
ALGO_CONFIG=${ALGO_CONFIG:-configs/config_quest_twi.json}
# Default 4-task subset: the two tasks on which CVSA is weakest (qasper, lcc)
# plus two representative QA/retrieval tasks (hotpotqa, passage_retrieval_en).
TASKS=${TASKS:-"qasper lcc hotpotqa passage_retrieval_en"}
# 100 samples/task (not 50) for tighter paired confidence intervals.
MAX_SAMPLES=${MAX_SAMPLES:-100}
# Where the Twilight outputs are copied after the run.
CVSA_RESULTS=${CVSA_RESULTS:-$(pwd)/results/camera_ready/twilight_longbench_llama}
# ---------------------------------------------------------------------------

export MAX_SAMPLES                                  # consumed by patched pred.py
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Activate the Twilight conda env (independent of the CVSA environment).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Twilight JIT-compiles its top-p CUDA kernel at runtime (flashinfer/torch
# cpp_extension.load) → needs a CUDA toolkit (nvcc + CUDA_HOME) matching
# torch's CUDA build (tested with CUDA 12.4 / torch cu124). Set CUDA_HOME to
# your toolkit path; `module` may be unavailable in a non-interactive batch shell.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# torch's CUDA-extension JIT needs GCC>=9 as host compiler. Point GCC_BIN at a
# GCC>=9 bin dir (tested with gcc 11); its libstdc++ goes on LD_LIBRARY_PATH so
# the compiled kernel finds the matching GLIBCXX at runtime.
if [ -n "${GCC_BIN:-}" ]; then
  export PATH="${GCC_BIN}:${PATH}"
  export CC="${GCC_BIN}/gcc"
  export CXX="${GCC_BIN}/g++"
  export LD_LIBRARY_PATH="${GCC_BIN}/../lib64:${LD_LIBRARY_PATH:-}"
fi
# Compile the JIT kernel only for sm_90 (H100/H200) → far faster than all-archs.
# Override TORCH_CUDA_ARCH_LIST for other GPUs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# HF cache (same convention as the CVSA launchers).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

cd "${TWILIGHT_ROOT}/benchmark"

TS=$(date +%m%d%H%M)
OUTPUT_DIR="results_longbench/${MODEL}"
DATASET_PATH="HF"
CONFIG_PATH="LongBench/config/"

echo "=== Twilight LongBench head-to-head ==="
echo "Twilight root : ${TWILIGHT_ROOT}"
echo "Model         : ${MODEL} (${MODELPATH})"
echo "Algo config   : ${ALGO_CONFIG}  (Quest + top-p 0.85 pruner + 4-bit estimator)"
echo "Tasks         : ${TASKS}"
echo "Max samples   : ${MAX_SAMPLES} per task"
echo "Env           : ${CONDA_ENV}"
echo "Start         : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Pre-flight: confirm the sample-limit patch (scripts/twilight_pred.patch) is present.
if ! grep -q "MAX_SAMPLES" LongBench/pred.py; then
  echo "ERROR: LongBench/pred.py is NOT patched for MAX_SAMPLES." >&2
  echo "Apply scripts/twilight_pred.patch to the Twilight checkout first," >&2
  echo "otherwise Twilight runs the FULL test split and the comparison basis" >&2
  echo "will not match CVSA's n=50." >&2
  exit 3
fi

# Predict per task (mirrors Twilight's LongBench driver, restricted to TASKS).
for task in ${TASKS}; do
  echo ""
  echo "--- pred: ${task} ---"
  python -u LongBench/pred.py \
    --model "${MODEL}" \
    --model_path "${MODELPATH}" \
    --task "${task}" \
    --algo-config-path "${ALGO_CONFIG}" \
    --e 0 \
    --t "${TS}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --config-path "${CONFIG_PATH}"
done

echo ""
echo "--- eval (scores all completed tasks for timestamp ${TS}) ---"
python -u LongBench/eval.py \
  --model "${MODEL}" \
  --e 0 \
  --t "${TS}" \
  --output-dir "${OUTPUT_DIR}"

# Copy Twilight outputs into the CVSA results tree.
mkdir -p "${CVSA_RESULTS}"
cp -r "${OUTPUT_DIR}" "${CVSA_RESULTS}/" 2>/dev/null || true

echo ""
echo "Done: $(date)"
echo "Twilight results: ${TWILIGHT_ROOT}/benchmark/${OUTPUT_DIR}"
echo "Copied to       : ${CVSA_RESULTS}"
