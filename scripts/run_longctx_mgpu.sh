#!/bin/bash
# Multi-GPU (layer-sharded) long-context decode throughput.
#
# Purpose:
#   Relates to the long-context throughput appendix: measures dense vs CVSA-c4
#   decode throughput for Llama/Mistral at B=8 and >=65K context, which the
#   single-GPU throughput harness does not cover. With TIMING=1 it also
#   records a per-kernel latency breakdown.
#
#   Uses eval/unified_latency_mgpu.py: device_map="auto" shards layers across
#   the visible GPUs, each layer's paged KV cache lives on its layer's GPU.
#   4 GPUs recommended (128K B=8 Llama KV ~137GB; needs the aggregate memory).
#
# Methods: dense,cvsa_5_90_c4_ar20_dL0   CONTEXT_LENS=65536,131072   BATCH_SIZES=8
#
# Output:
#   results/camera_ready/longctx_mgpu_<llama|mistral>.json        (speedup_vs_dense per cell)
#   results/camera_ready/longctx_mgpu_smoke_<llama|mistral>.json  (SMOKE=1)
#
# Run the smoke step FIRST and confirm it reproduces a known single-GPU cell
# before trusting the >=65K numbers (see the note at the top of
# eval/unified_latency_mgpu.py).
#
# Usage:
#   # 1) Smoke on the target multi-GPU node (ctx=8K B=8; ~15 min):
#   SMOKE=1 sbatch scripts/run_longctx_mgpu.sh llama
#   #    -> compare results/camera_ready/longctx_mgpu_smoke_llama.json dense@8192@B8
#   #       tok/s against the single-GPU unified_latency.py 8K B=8 dense cell.
#
#   # 2) Long-context cells (65K + 128K, B=8):
#   sbatch scripts/run_longctx_mgpu.sh llama
#   sbatch scripts/run_longctx_mgpu.sh mistral

# NOTE: measure throughput on the same GPU type as the single-GPU long-context
# cells (H200) for comparability. At 141GB/GPU, B=8 @65K (~85GB) fits one
# H200 and @128K (~153GB) needs 2; gpu:4 gives headroom. Set --partition to an
# H200 partition on your cluster.
#SBATCH --job-name=cvsa-longctx-mgpu
#SBATCH --partition=normal2
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/longctx_mgpu_%x_%j.out
#SBATCH --error=logs/longctx_mgpu_%x_%j.err

set -euo pipefail

MODEL_KEY="${1:-}"
case "${MODEL_KEY}" in
  llama)
    MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
    SHORT=llama
    ;;
  mistral)
    MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
    SHORT=mistral
    ;;
  *)
    echo "Usage: sbatch $0 {llama|mistral}   (Qwen B=8 already fits 1 GPU)" >&2
    exit 2
    ;;
esac

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
mkdir -p logs results/camera_ready

# Optional: ENV_BIN=/path/to/env/bin selects a specific Python environment.
ENV_BIN=${ENV_BIN:-}
if [ -n "${ENV_BIN:-}" ]; then export PATH="${ENV_BIN}:$PATH"; fi

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# The repository root (cvsa/ and eval/) is on PYTHONPATH.
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
PYTHON=${PYTHON:-python}

METHODS=${METHODS:-dense,cvsa_5_90_c4_ar20_dL0}
CONTEXT_LENS=${CONTEXT_LENS:-65536,131072}
BATCH_SIZES=${BATCH_SIZES:-8}
NUM_DECODE=${NUM_DECODE:-64}
# TIMING=1 → per-kernel latency breakdown. Run it as a dedicated pass; take
# speedup from a non-timing run.
TIMING_FLAG=""
[ "${TIMING:-0}" = "1" ] && TIMING_FLAG="--timing"

echo "=== Multi-GPU long-context throughput: ${SHORT} (${MODEL}) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "Start: $(date)"

if ! "${PYTHON}" -c "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())"; then
  echo "Missing runtime dependency in PYTHON=${PYTHON}." >&2
  exit 1
fi

if [ "${SMOKE:-0}" = "1" ]; then
  echo ">>> SMOKE run (ctx=8K, B=8, dense + c4) — validate before real cells."
  "${PYTHON}" eval/unified_latency_mgpu.py \
    --model "${MODEL}" \
    --smoke \
    --output "results/camera_ready/longctx_mgpu_smoke_${SHORT}.json"
  echo ""
  echo "Smoke done. Compare dense@8192@B8 tok/s vs the single-GPU"
  echo "unified_latency.py 8K B=8 dense cell before running the real job."
  exit 0
fi

echo ">>> REAL run: ${MODEL}  ctx=${CONTEXT_LENS}  B=${BATCH_SIZES}"
"${PYTHON}" eval/unified_latency_mgpu.py \
  --model "${MODEL}" \
  --context-lens "${CONTEXT_LENS}" \
  --batch-sizes "${BATCH_SIZES}" \
  --methods "${METHODS}" \
  --num-decode "${NUM_DECODE}" \
  ${TIMING_FLAG} \
  --output "${OUT:-results/camera_ready/longctx_mgpu_${SHORT}.json}"

echo "Done: $(date)"
echo "Output: results/camera_ready/longctx_mgpu_${SHORT}.json"
