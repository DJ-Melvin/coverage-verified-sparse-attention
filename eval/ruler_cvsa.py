#!/usr/bin/env python3
"""RULER benchmark evaluation for CVSA and baselines.

Implements the 6 synthetic long-context tasks from RULER (Hsieh et al., 2024):
  1. S-NIAH: Single Needle-in-a-Haystack retrieval
  2. MK-NIAH: Multi-Key NIAH (multiple needles, query one key)
  3. MV-NIAH: Multi-Value NIAH (same key, different values at depths)
  4. MQ-NIAH: Multi-Query NIAH (answer multiple questions)
  5. VT: Variable Tracking (chain of assignments)
  6. CWE: Common Words Extraction (find most frequent words)

Evaluates across context lengths: 4K, 8K, 16K, 32K.
Supports CVSA (hybrid/lazy/fixed), dense, and baselines (snapkv, h2o, adakv).

Usage:
    cd /path/to/cvsa
    python eval/ruler_cvsa.py \
        --methods dense,lazy_5_90_c2,snapkv_10,h2o_10 \
        --context-lens 4096,8192,16384,32768 \
        --n-trials 10 \
        --output results/ruler_llama8b.json
"""

import os
import sys
import json
import time
import random
import argparse
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import importlib.util

from cvsa.runtime.model_loader import ModelForDecode
from eval.longbench_harness import E2EGenerator

# Import make_backend from unified_latency via file path to avoid eval/ namespace conflict
_ul_spec = importlib.util.spec_from_file_location(
    "unified_latency",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_latency.py"))
_ul_mod = importlib.util.module_from_spec(_ul_spec)
_ul_spec.loader.exec_module(_ul_mod)
make_backend = _ul_mod.make_backend


# ─── Filler text for haystack construction ───

FILLER_PARAGRAPHS = [
    "The development of modern computing can be traced back to the work of "
    "pioneers like Alan Turing and John von Neumann. Their theoretical "
    "frameworks laid the groundwork for the digital revolution that would "
    "transform every aspect of human society in the decades that followed.",

    "Photosynthesis is the process by which green plants convert sunlight "
    "into chemical energy. This remarkable biological mechanism sustains "
    "nearly all life on Earth by producing oxygen and organic compounds "
    "from carbon dioxide and water.",

    "The Renaissance period, spanning roughly from the 14th to the 17th "
    "century, marked a cultural rebirth in Europe. Artists like Leonardo "
    "da Vinci and Michelangelo pushed the boundaries of artistic expression "
    "while scientists began to challenge long-held assumptions about nature.",

    "Ocean currents play a critical role in regulating global climate. "
    "The Gulf Stream, for example, transports warm water from the tropics "
    "to the North Atlantic, significantly warming the climate of Western "
    "Europe compared to other regions at similar latitudes.",

    "The human genome contains approximately three billion base pairs of "
    "DNA organized into 23 pairs of chromosomes. Understanding this genetic "
    "blueprint has opened new frontiers in medicine, enabling personalized "
    "treatments and early detection of hereditary diseases.",

    "Ancient civilizations developed sophisticated astronomical knowledge "
    "long before the invention of the telescope. The Maya, Egyptians, and "
    "Babylonians each created detailed calendars and predicted celestial "
    "events with remarkable accuracy using only naked-eye observations.",

    "The theory of plate tectonics explains how Earth's outer shell is "
    "divided into several plates that glide over the mantle. The interactions "
    "between these plates produce earthquakes, volcanic activity, and the "
    "formation of mountain ranges over geological timescales.",

    "Quantum mechanics describes the behavior of matter and energy at the "
    "smallest scales. At the quantum level, particles can exist in multiple "
    "states simultaneously, a phenomenon known as superposition, which "
    "challenges our everyday intuition about the nature of reality.",

    "The Industrial Revolution began in Britain in the late 18th century "
    "and fundamentally changed how goods were produced. Mechanization of "
    "textile manufacturing led to factories, urbanization, and eventually "
    "transformed economic and social structures across the globe.",

    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "covering approximately 5.5 million square kilometers. It is home to "
    "an estimated 10 percent of all species on Earth and plays a crucial "
    "role in absorbing carbon dioxide from the atmosphere.",

    "Neural networks are computational models inspired by the structure "
    "of biological brains. They consist of layers of interconnected nodes "
    "that process information through weighted connections, enabling "
    "machines to learn patterns from data without explicit programming.",

    "The periodic table organizes chemical elements by their atomic number "
    "and electron configuration. This arrangement reveals patterns in "
    "chemical properties, allowing scientists to predict the behavior of "
    "elements and their compounds with considerable accuracy.",
]

NEEDLE_KEYS = [
    "alpha", "bravo", "charlie", "delta", "echo",
    "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar",
    "papa", "quebec", "romeo", "sierra", "tango",
]

CITY_NAMES = [
    "Tokyo", "London", "Paris", "Berlin", "Sydney",
    "Cairo", "Mumbai", "Toronto", "Seoul", "Lima",
    "Oslo", "Dubai", "Rome", "Athens", "Dublin",
    "Zurich", "Vienna", "Prague", "Warsaw", "Helsinki",
]

COLOR_NAMES = [
    "crimson", "azure", "emerald", "amber", "violet",
    "scarlet", "indigo", "coral", "ivory", "slate",
    "maroon", "teal", "olive", "bronze", "silver",
    "copper", "jade", "pearl", "onyx", "ruby",
]


# ═══════════════════════════════════════════════════════════
# Haystack Construction
# ═══════════════════════════════════════════════════════════

def build_filler(tokenizer, target_tokens, seed=42):
    """Build filler text of approximately target_tokens length."""
    rng = random.Random(seed)
    paragraphs = list(FILLER_PARAGRAPHS)
    filler_chunks = []
    while True:
        rng.shuffle(paragraphs)
        for p in paragraphs:
            filler_chunks.append(p)
        text = "\n\n".join(filler_chunks)
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if n > target_tokens * 1.3:
            break

    text = "\n\n".join(filler_chunks)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def insert_at_depth(filler_text, insert_text, depth):
    """Insert text at a fractional depth into filler text."""
    pos = int(len(filler_text) * max(0.0, min(1.0, depth)))
    if 0.01 < depth < 0.99:
        left = filler_text.rfind("\n\n", 0, pos)
        right = filler_text.find("\n\n", pos)
        if left > 0 and (right < 0 or pos - left < right - pos):
            pos = left + 2
        elif right > 0:
            pos = right + 2
    return filler_text[:pos] + "\n" + insert_text + "\n" + filler_text[pos:]


# ═══════════════════════════════════════════════════════════
# Task 1: Single NIAH (S-NIAH)
# ═══════════════════════════════════════════════════════════

def build_s_niah(tokenizer, target_tokens, trial, seed=42):
    rng = random.Random(seed + trial)
    key = NEEDLE_KEYS[trial % len(NEEDLE_KEYS)]
    value = rng.randint(1000, 9999)
    depth = rng.uniform(0.1, 0.9)

    needle = f"The special code for project {key} is {value}."
    filler = build_filler(tokenizer, target_tokens - 80, seed=seed + trial)
    haystack = insert_at_depth(filler, needle, depth)

    prompt = (
        f"Read the following text and answer the question.\n\n"
        f"{haystack}\n\n"
        f"What is the special code for project {key}? "
        f"Answer with just the number.\nAnswer:"
    )
    return prompt, str(value)


# ═══════════════════════════════════════════════════════════
# Task 2: Multi-Key NIAH (MK-NIAH)
# ═══════════════════════════════════════════════════════════

def build_mk_niah(tokenizer, target_tokens, trial, n_keys=5, seed=42):
    rng = random.Random(seed + trial)
    keys = rng.sample(NEEDLE_KEYS, n_keys)
    values = [rng.randint(1000, 9999) for _ in range(n_keys)]
    depths = sorted(rng.uniform(0.05, 0.95) for _ in range(n_keys))
    query_idx = rng.randint(0, n_keys - 1)

    filler = build_filler(tokenizer, target_tokens - 80 * n_keys, seed=seed + trial)
    haystack = filler
    for i in sorted(range(n_keys), key=lambda x: depths[x], reverse=True):
        needle = f"The access code for project {keys[i]} is {values[i]}."
        haystack = insert_at_depth(haystack, needle, depths[i])

    prompt = (
        f"Read the following text and answer the question.\n\n"
        f"{haystack}\n\n"
        f"What is the access code for project {keys[query_idx]}? "
        f"Answer with just the number.\nAnswer:"
    )
    return prompt, str(values[query_idx])


# ═══════════════════════════════════════════════════════════
# Task 3: Multi-Value NIAH (MV-NIAH)
# ═══════════════════════════════════════════════════════════

def build_mv_niah(tokenizer, target_tokens, trial, n_values=5, seed=42):
    rng = random.Random(seed + trial)
    key = NEEDLE_KEYS[trial % len(NEEDLE_KEYS)]
    values = [rng.randint(1000, 9999) for _ in range(n_values)]
    depths = sorted(rng.uniform(0.05, 0.95) for _ in range(n_values))

    filler = build_filler(tokenizer, target_tokens - 80 * n_values, seed=seed + trial)
    haystack = filler
    for i in sorted(range(n_values), key=lambda x: depths[x], reverse=True):
        needle = f"Update: the code for project {key} has been changed to {values[i]}."
        haystack = insert_at_depth(haystack, needle, depths[i])

    prompt = (
        f"Read the following text and answer the question.\n\n"
        f"{haystack}\n\n"
        f"What is the most recent code for project {key}? "
        f"The code may have been updated multiple times. "
        f"Give only the latest value.\nAnswer:"
    )
    return prompt, str(values[-1])


# ═══════════════════════════════════════════════════════════
# Task 4: Multi-Query NIAH (MQ-NIAH)
# ═══════════════════════════════════════════════════════════

def build_mq_niah(tokenizer, target_tokens, trial, n_queries=3, seed=42):
    rng = random.Random(seed + trial)
    keys = rng.sample(NEEDLE_KEYS, n_queries)
    cities = rng.sample(CITY_NAMES, n_queries)
    depths = sorted(rng.uniform(0.05, 0.95) for _ in range(n_queries))

    filler = build_filler(tokenizer, target_tokens - 80 * n_queries, seed=seed + trial)
    haystack = filler
    for i in sorted(range(n_queries), key=lambda x: depths[x], reverse=True):
        needle = f"Agent {keys[i]} is stationed in {cities[i]}."
        haystack = insert_at_depth(haystack, needle, depths[i])

    questions = ", ".join(keys)
    prompt = (
        f"Read the following text and answer the questions.\n\n"
        f"{haystack}\n\n"
        f"Where are the following agents stationed: {questions}? "
        f"List each agent and their city, separated by commas.\nAnswer:"
    )
    expected = {k: c for k, c in zip(keys, cities)}
    return prompt, expected


def score_mq_niah(generated, expected):
    gen_lower = generated.lower()
    correct = 0
    for key, city in expected.items():
        if city.lower() in gen_lower:
            correct += 1
    return correct / len(expected)


# ═══════════════════════════════════════════════════════════
# Task 5: Variable Tracking (VT)
# ═══════════════════════════════════════════════════════════

def build_vt(tokenizer, target_tokens, trial, n_vars=5, n_steps=7, seed=42):
    rng = random.Random(seed + trial)

    var_names = rng.sample(list(string.ascii_uppercase), n_vars)
    var_values = {v: rng.randint(10, 99) for v in var_names}

    assignments = []
    for _ in range(n_steps):
        target_var = rng.choice(var_names)
        action = rng.choice(["set", "copy"])
        if action == "set":
            new_val = rng.randint(10, 99)
            assignments.append(f"Variable {target_var} is now set to {new_val}.")
            var_values[target_var] = new_val
        else:
            source_var = rng.choice([v for v in var_names if v != target_var])
            assignments.append(
                f"Variable {target_var} is now set to the value of variable {source_var}."
            )
            var_values[target_var] = var_values[source_var]

    filler = build_filler(
        tokenizer, target_tokens - 60 * len(assignments), seed=seed + trial
    )
    haystack = filler
    step_depths = [(i + 1) / (len(assignments) + 1) for i in range(len(assignments))]
    for i in sorted(range(len(assignments)), key=lambda x: step_depths[x], reverse=True):
        haystack = insert_at_depth(haystack, assignments[i], step_depths[i])

    query_var = rng.choice(var_names)
    prompt = (
        f"Read the following text which contains variable assignments "
        f"scattered throughout.\n\n"
        f"{haystack}\n\n"
        f"Based on all the variable assignments above, what is the final "
        f"value of variable {query_var}? Give only the number.\nAnswer:"
    )
    return prompt, str(var_values[query_var])


# ═══════════════════════════════════════════════════════════
# Task 6: Common Words Extraction (CWE)
# ═══════════════════════════════════════════════════════════

def build_cwe(tokenizer, target_tokens, trial, n_target=3, freq=15, seed=42):
    rng = random.Random(seed + trial)

    target_words = rng.sample(COLOR_NAMES, n_target)

    filler = build_filler(tokenizer, target_tokens - 20 * freq * n_target, seed=seed + trial)

    sentences = []
    for word in target_words:
        for _ in range(freq):
            sentences.append(f"The color {word} appeared prominently in the scene.")

    rng.shuffle(sentences)
    haystack = filler
    for i, sent in enumerate(sentences):
        depth = (i + 0.5) / len(sentences)
        haystack = insert_at_depth(haystack, sent, depth)

    prompt = (
        f"Read the following text carefully.\n\n"
        f"{haystack}\n\n"
        f"Which {n_target} colors are mentioned most frequently in the text above? "
        f"List them separated by commas.\nAnswer:"
    )
    return prompt, set(target_words)


def score_cwe(generated, expected_set):
    gen_lower = generated.lower()
    found = sum(1 for w in expected_set if w.lower() in gen_lower)
    return found / len(expected_set)


# ═══════════════════════════════════════════════════════════
# Task Registry
# ═══════════════════════════════════════════════════════════

TASKS = {
    "s_niah": {
        "name": "Single NIAH",
        "build": build_s_niah,
        "score": lambda gen, exp: 1.0 if exp.lower() in gen.lower() else 0.0,
        "max_gen": 32,
    },
    "mk_niah": {
        "name": "Multi-Key NIAH",
        "build": build_mk_niah,
        "score": lambda gen, exp: 1.0 if exp.lower() in gen.lower() else 0.0,
        "max_gen": 32,
    },
    "mv_niah": {
        "name": "Multi-Value NIAH",
        "build": build_mv_niah,
        "score": lambda gen, exp: 1.0 if exp.lower() in gen.lower() else 0.0,
        "max_gen": 32,
    },
    "mq_niah": {
        "name": "Multi-Query NIAH",
        "build": build_mq_niah,
        "score": score_mq_niah,
        "max_gen": 128,
    },
    "vt": {
        "name": "Variable Tracking",
        "build": build_vt,
        "score": lambda gen, exp: 1.0 if exp.lower() in gen.lower() else 0.0,
        "max_gen": 32,
    },
    "cwe": {
        "name": "Common Words Extraction",
        "build": build_cwe,
        "score": score_cwe,
        "max_gen": 64,
    },
}


# ═══════════════════════════════════════════════════════════
# Main Evaluation Loop
# ═══════════════════════════════════════════════════════════

def run_ruler(args):
    print("RULER Benchmark Evaluation (CVSA)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Tasks: {args.tasks}", flush=True)
    print(f"  Methods: {args.methods}", flush=True)
    print(f"  Context lengths: {args.context_lens}", flush=True)
    print(f"  Trials per config: {args.n_trials}", flush=True)
    print(flush=True)

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model = ModelForDecode(args.model, dtype=torch.bfloat16)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    config = model.config
    print(f"  {config.num_q_heads}Q/{config.num_kv_heads}KV, D={config.head_dim}, "
          f"{config.num_layers}L", flush=True)
    print(flush=True)

    generator = E2EGenerator(model, block_size=args.block_size)
    context_lens = [int(x) for x in args.context_lens.split(",")]
    methods = args.methods.split(",")
    task_names = args.tasks.split(",")

    for t in task_names:
        if t not in TASKS:
            raise ValueError(f"Unknown task: {t}. Available: {list(TASKS.keys())}")

    # results[method][task][ctx_len] = accuracy (0-100)
    results = {m: {t: {} for t in task_names} for m in methods}

    for task_name in task_names:
        task_info = TASKS[task_name]
        print(f"\n{'━'*60}", flush=True)
        print(f"Task: {task_info['name']} ({task_name})", flush=True)
        print(f"{'━'*60}", flush=True)

        for ctx_len in context_lens:
            print(f"\n  Context: {ctx_len // 1024}K tokens", flush=True)

            for method in methods:
                scores = []
                t0 = time.time()

                for trial in range(args.n_trials):
                    backend = make_backend(method, config, args.block_size)

                    prompt, expected = task_info["build"](
                        model.tokenizer, ctx_len, trial,
                        seed=42 + trial,
                    )

                    pred = generator.generate(
                        prompt, task_info["max_gen"], backend
                    )
                    score = task_info["score"](pred, expected)
                    scores.append(score)

                    del backend
                    torch.cuda.empty_cache()

                acc = np.mean(scores) * 100
                elapsed = time.time() - t0
                results[method][task_name][str(ctx_len)] = acc
                print(f"    {method:<20} {acc:>5.1f}%  "
                      f"({args.n_trials} trials, {elapsed:.0f}s)", flush=True)

    # ─── Summary Tables ───
    print(f"\n{'='*80}", flush=True)
    print("RULER RESULTS (% accuracy)", flush=True)
    print(f"{'='*80}", flush=True)

    for task_name in task_names:
        task_info = TASKS[task_name]
        print(f"\n{task_info['name']}:", flush=True)

        header = f"{'Context':<10}"
        for method in methods:
            header += f" {method:>14}"
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for ctx_len in context_lens:
            row = f"{ctx_len//1024:>6}K   "
            for method in methods:
                val = results[method][task_name].get(str(ctx_len), 0)
                row += f" {val:>14.1f}"
            print(row, flush=True)

    # Overall summary
    print(f"\nOverall RULER Score:", flush=True)
    for method in methods:
        all_vals = []
        for task_name in task_names:
            for ctx_len in context_lens:
                v = results[method][task_name].get(str(ctx_len), 0)
                all_vals.append(v)
        print(f"  {method:<20} {np.mean(all_vals):>5.1f}%", flush=True)

    # Per-task mean
    print(f"\nPer-Task Mean:", flush=True)
    header = f"{'Task':<16}"
    for method in methods:
        header += f" {method:>14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for task_name in task_names:
        row = f"{task_name:<16}"
        for method in methods:
            vals = [results[method][task_name].get(str(c), 0)
                    for c in context_lens]
            row += f" {np.mean(vals):>14.1f}"
        print(row, flush=True)

    # Save
    output = {
        "config": {
            "model": args.model,
            "methods": methods,
            "tasks": task_names,
            "context_lens": context_lens,
            "n_trials": args.n_trials,
            "block_size": args.block_size,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="RULER benchmark evaluation for CVSA")
    parser.add_argument("--model",
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--methods",
                        default="dense,lazy_5_90_c2,fixed_5,snapkv_10,h2o_10")
    parser.add_argument("--tasks",
                        default="s_niah,mk_niah,mv_niah,mq_niah,vt,cwe")
    parser.add_argument("--context-lens", default="4096,8192,16384,32768")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--output", default="results/ruler_llama8b.json")
    args = parser.parse_args()

    run_ruler(args)


if __name__ == "__main__":
    main()
