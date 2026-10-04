#!/bin/bash
#SBATCH --job-name=rlong-evict
#SBATCH --partition=normal
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ruler_longctx_eviction_%A_%a.out
#SBATCH --error=logs/ruler_longctx_eviction_%A_%a.err

# Fill the missing eviction-baseline points for Figure 4.
#
# Existing long-context RULER jobs write Dense / Fixed-10% / CVSA-c4 / CVSA-c8.
# at 64K and 128K. This job adds SnapKV / H2O / Ada-KV at the same contexts,
# then merges those methods into results/ruler_longctx_{llama,mistral,qwen}.json
# so eval/plot_ruler_main.py can consume the completed long-context data.

set -euo pipefail

cd /path/to/cvsa
mkdir -p logs results

export HF_HOME=$HOME/.cache/huggingface
export PATH=/path/to/env/bin:$PATH

MODEL_IDS=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen2.5-7B-Instruct"
)
MODEL_KEYS=("llama" "mistral" "qwen")
MODEL_LABELS=("Llama-3.1-8B" "Mistral-7B-v0.3" "Qwen2.5-7B")

IDX="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODEL_IDS[$IDX]}"
KEY="${MODEL_KEYS[$IDX]}"
LABEL="${MODEL_LABELS[$IDX]}"

METHODS="${METHODS:-snapkv_10,h2o_10,adakv_10}"
TASKS="${TASKS:-mv_niah,mq_niah,cwe}"
CONTEXT_LENS="${CONTEXT_LENS:-65536,131072}"
N_TRIALS="${N_TRIALS:-5}"

TMP_OUTPUT="results/ruler_longctx_eviction_${KEY}.json"
TARGET_OUTPUT="results/ruler_longctx_${KEY}.json"

echo "RULER long-context eviction baselines for ${LABEL}"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Methods: ${METHODS}"
echo "Tasks: ${TASKS}"
echo "Contexts: ${CONTEXT_LENS}"
echo "Trials: ${N_TRIALS}"

python eval/ruler_cvsa.py \
  --model "${MODEL}" \
  --methods "${METHODS}" \
  --tasks "${TASKS}" \
  --context-lens "${CONTEXT_LENS}" \
  --n-trials "${N_TRIALS}" \
  --output "${TMP_OUTPUT}"

python - "${TARGET_OUTPUT}" "${TMP_OUTPUT}" <<'PY'
import json
import os
import sys

target_path, source_path = sys.argv[1], sys.argv[2]

with open(source_path) as f:
    source = json.load(f)

if os.path.exists(target_path):
    with open(target_path) as f:
        target = json.load(f)
else:
    target = {
        "config": source.get("config", {}).copy(),
        "results": {},
    }

target.setdefault("config", {})
target.setdefault("results", {})

existing_methods = target["config"].get("methods", [])
if not isinstance(existing_methods, list):
    existing_methods = []
for method in source.get("config", {}).get("methods", []):
    if method not in existing_methods:
        existing_methods.append(method)
target["config"]["methods"] = existing_methods

existing_tasks = target["config"].get("tasks", [])
if not isinstance(existing_tasks, list):
    existing_tasks = []
for task in source.get("config", {}).get("tasks", []):
    if task not in existing_tasks:
        existing_tasks.append(task)
target["config"]["tasks"] = existing_tasks

existing_contexts = target["config"].get("context_lens", [])
if not isinstance(existing_contexts, list):
    existing_contexts = []
for ctx in source.get("config", {}).get("context_lens", []):
    if ctx not in existing_contexts:
        existing_contexts.append(ctx)
target["config"]["context_lens"] = sorted(existing_contexts)
target["config"]["longctx_eviction_trials"] = source.get("config", {}).get("n_trials")

merged_cells = 0
for method, method_results in source.get("results", {}).items():
    target["results"].setdefault(method, {})
    for task, task_results in method_results.items():
        target["results"][method].setdefault(task, {})
        for ctx, value in task_results.items():
            target["results"][method][task][ctx] = value
            merged_cells += 1

with open(target_path, "w") as f:
    json.dump(target, f, indent=2)
    f.write("\n")

print(f"Merged {merged_cells} cells from {source_path} into {target_path}")
PY

echo "Done: $(date)"
