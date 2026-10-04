"""LongBench evaluation infrastructure shared by the entry points.

Prompts, generation lengths and metrics follow the LongBench configuration
(classification and code-similarity scoring follow Ada-KV). E2EGenerator
prefills with the HuggingFace model, copies the KV cache into the paged
cache and decodes token by token through a pluggable attention backend
(dense, CVSA, or a baseline from eval/baselines.py); LongBenchE2ERunner
handles data loading, middle truncation, chat templates and scoring.

Entry point: eval/longbench_cvsa.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datasets import load_dataset

from cvsa.runtime.kv_cache import PagedKVCache, SequenceKVManager


# ─── LongBench configs ───

DATASET2PROMPT = {
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
}

DATASET2MAXGEN = {
    "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32,
    "2wikimqa": 32, "gov_report": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "lcc": 64, "repobench-p": 64,
    "narrativeqa": 128, "musique": 32, "qmsum": 512,
    "passage_count": 32, "passage_retrieval_en": 32,
}


# ─── LongBench metrics ───

def normalize_answer(s):
    import re, string
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truths):
    return max(f1_score(prediction, gt) for gt in ground_truths)


def rouge_score(prediction, ground_truths):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for gt in ground_truths:
            try:
                s = scorer.score(prediction, gt)
                scores.append(s["rougeL"].fmeasure)
            except Exception:
                scores.append(0.0)
        return max(scores) if scores else 0.0
    except ImportError:
        return qa_f1_score(prediction, ground_truths)


def classification_score(prediction, ground_truths):
    # Match Ada-KV: prefix match (prediction starts with ground truth)
    pred = prediction.strip().lower()
    return 1.0 if any(pred.startswith(gt.strip().lower())
                      for gt in ground_truths) else 0.0


def code_sim_score(prediction, ground_truths):
    # Match Ada-KV: fuzzywuzzy token-level ratio
    try:
        from fuzzywuzzy import fuzz
        return max(fuzz.ratio(prediction, gt) / 100.0
                   for gt in ground_truths) if ground_truths else 0.0
    except ImportError:
        from difflib import SequenceMatcher
        return max(SequenceMatcher(None, prediction, gt).ratio()
                   for gt in ground_truths) if ground_truths else 0.0


def count_score(prediction, ground_truths):
    """Count accuracy: check if predicted number matches ground truth."""
    import re
    numbers = re.findall(r"\d+", prediction)
    gt = ground_truths[0] if isinstance(ground_truths, list) else ground_truths
    right = sum(1 for n in numbers if str(n) == str(gt))
    return float(right / len(numbers)) if numbers else 0.0


def retrieval_score(prediction, ground_truths):
    """Paragraph retrieval: check if predicted paragraph ID matches."""
    import re
    gt = ground_truths[0] if isinstance(ground_truths, list) else ground_truths
    matches = re.findall(r'\d+', str(gt))
    gt_id = matches[0] if matches else str(gt)
    numbers = re.findall(r"\d+", prediction)
    right = sum(1 for n in numbers if str(n) == str(gt_id))
    return float(right / len(numbers)) if numbers else 0.0


DATASET2METRIC = {
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "triviaqa": qa_f1_score,
    "narrativeqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "samsum": rouge_score,
    "qmsum": rouge_score,
    "trec": classification_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
}


# ─── E2E Generator ───

class E2EGenerator:
    """Generate text using standalone E2E decode with real model weights."""

    def __init__(self, model, block_size=16):
        self.model = model
        self.config = model.config
        self.block_size = block_size
        self.device = model.device
        self.dtype = model.dtype

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, attn_backend,
                 temperature=0.0, top_p=1.0, sample_seed=None):
        """Generate text for a single prompt.

        Args:
            prompt: input text string
            max_new_tokens: max tokens to generate
            attn_backend: attention backend (dense or sparse)
            temperature: <=0 keeps the exact greedy path (default);
                >0 enables nucleus sampling (stochastic-decoding probe)
            top_p: nucleus mass when sampling
            sample_seed: per-sample RNG seed so paired methods draw from
                the same stream

        Returns:
            generated_text: string (excluding prompt)
        """
        config = self.config
        tokenizer = self.model.tokenizer

        rng = None
        if temperature > 0:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(0 if sample_seed is None else sample_seed)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt",
                                      truncation=False).to(self.device)
        seq_len = input_ids.shape[1]

        # Prefill (B=1)
        inner_model = self.model.model.model  # LlamaModel

        # Capture real Q from the observation window if the backend scores
        # pages at prefill (set_prefill_scores, e.g. SnapKV)
        _prefill_q_hooks = []
        _prefill_q_data = {}  # layer_idx -> [W, H_kv, D]
        if hasattr(attn_backend, 'set_prefill_scores'):
            obs_W = min(getattr(attn_backend, 'obs_window', 32), seq_len)
            H_kv = config.num_kv_heads
            H_q = config.num_q_heads
            GQA = H_q // H_kv
            D_head = config.head_dim

            def _make_q_hook(layer_idx):
                def _hook(module, args, kwargs):
                    hidden = args[0] if len(args) > 0 else kwargs.get('hidden_states')
                    if hidden is not None:
                        with torch.no_grad():
                            q = module.q_proj(hidden)  # [1, seq_len, H_q*D]
                            q = q.view(1, -1, H_q, D_head).squeeze(0).float()
                            # Last W tokens, grouped by KV head
                            q_win = q[-obs_W:].view(obs_W, H_kv, GQA, D_head).mean(dim=2)
                            _prefill_q_data[layer_idx] = q_win.detach()
                return _hook

            for li, layer in enumerate(inner_model.layers):
                h = layer.self_attn.register_forward_pre_hook(
                    _make_q_hook(li), with_kwargs=True)
                _prefill_q_hooks.append(h)

        outputs = inner_model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values

        # Remove hooks
        for h in _prefill_q_hooks:
            h.remove()

        # Apply RoPE to captured Q vectors so they match post-RoPE K in cache.
        # The hook captures pre-RoPE Q (q_proj output), but K in cache is
        # post-RoPE. For correct Q·K scores we need post-RoPE Q.
        if _prefill_q_data:
            _obs_W = min(getattr(attn_backend, 'obs_window', 32), seq_len)
            _pos_ids = torch.arange(
                seq_len - _obs_W, seq_len, device=self.device).unsqueeze(0)
            _dummy = torch.zeros(1, _obs_W, config.head_dim,
                                 device=self.device)
            _rope_cos, _rope_sin = self.model.model.model.rotary_emb(
                _dummy, _pos_ids)                         # [1, W, D]
            _rope_cos = _rope_cos.squeeze(0).unsqueeze(1)  # [W, 1, D]
            _rope_sin = _rope_sin.squeeze(0).unsqueeze(1)  # [W, 1, D]
            for _li in _prefill_q_data:
                _qw = _prefill_q_data[_li]                 # [W, H_kv, D]
                _qw_f = _qw.float()
                _half = _qw_f.shape[-1] // 2
                _rot = torch.cat([-_qw_f[..., _half:], _qw_f[..., :_half]],
                                 dim=-1)
                _prefill_q_data[_li] = (
                    _qw_f * _rope_cos.float() + _rot * _rope_sin.float()
                ).to(_qw.dtype)

        # Get first generated token
        normed = outputs.last_hidden_state[:, -1:, :]
        logits = self.model.lm_head(normed.squeeze(1))
        next_token = nucleus_sample(logits, temperature, top_p, rng)  # [1]

        # Build paged KV caches
        n_pages = (seq_len + self.block_size - 1) // self.block_size
        extra_pages = max(16, (max_new_tokens + self.block_size - 1) // self.block_size + 4)
        max_pages = n_pages + extra_pages
        num_blocks = max_pages + 64

        layer_caches = []
        layer_managers = []
        for layer_idx in range(config.num_layers):
            cache = PagedKVCache(
                num_blocks=num_blocks, block_size=self.block_size,
                num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
                dtype=self.dtype, device=str(self.device),
            )
            manager = SequenceKVManager(cache, 1,
                                         max_context_len=seq_len + max_new_tokens + 16)

            # Copy KV into paged cache (cast to cache dtype for quantized models)
            k = past_kv[layer_idx][0][0, :, :seq_len, :]  # [H_kv, seq_len, D]
            v = past_kv[layer_idx][1][0, :, :seq_len, :]
            k_seq = k.transpose(0, 1).contiguous().to(self.dtype)  # [seq_len, H_kv, D]
            v_seq = v.transpose(0, 1).contiguous().to(self.dtype)
            manager.prefill_kv(0, k_seq, v_seq)

            # Compute prefill page scores from the real observation-window Q
            if hasattr(attn_backend, 'set_prefill_scores') and layer_idx in _prefill_q_data:
                q_window = _prefill_q_data[layer_idx]  # [W, H_kv, D] real Q vectors
                attn_backend.set_prefill_scores(layer_idx, k_seq, q_window,
                                                 self.block_size)

            layer_caches.append(cache)
            layer_managers.append(manager)

        del outputs, past_kv
        torch.cuda.empty_cache()

        # Decode loop — import model-specific apply_rotary_pos_emb
        import importlib
        layers = self.model.get_layers()
        attn_module = type(layers[0].self_attn).__module__
        apply_rotary_pos_emb = importlib.import_module(attn_module).apply_rotary_pos_emb
        rotary_emb = self.model.model.model.rotary_emb

        generated_ids = [next_token.item()]
        cur_token = next_token
        attn_backend.reset()

        eos_id = tokenizer.eos_token_id
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        for step in range(max_new_tokens - 1):
            position = seq_len + step
            B = 1

            # Embed
            hidden = self.model.embed_tokens(cur_token.unsqueeze(0))  # [1, 1, hidden]

            # RoPE
            position_ids = torch.tensor([[position]], device=self.device)
            cos, sin = rotary_emb(hidden, position_ids)

            for layer_idx, layer in enumerate(layers):
                cache = layer_caches[layer_idx]
                manager = layer_managers[layer_idx]

                residual = hidden
                hidden = layer.input_layernorm(hidden)

                q = layer.self_attn.q_proj(hidden)
                k = layer.self_attn.k_proj(hidden)
                v = layer.self_attn.v_proj(hidden)

                # Cast to cache dtype (4-bit quantized models may output float16)
                cache_dtype = cache.k_cache.dtype
                q = q.to(cache_dtype)
                k = k.to(cache_dtype)
                v = v.to(cache_dtype)

                q = q.view(1, 1, config.num_q_heads, config.head_dim).transpose(1, 2)
                k = k.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
                v = v.view(1, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)

                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                k_store = k.squeeze(2)
                v_store = v.squeeze(2)
                manager.append_token_kv(0, k_store[0], v_store[0])

                q_for_attn = q.squeeze(2)
                attn_output = attn_backend(
                    q_for_attn,
                    cache.k_cache, cache.v_cache,
                    manager.block_tables, manager.context_lens,
                    layer_idx,
                )

                attn_output = attn_output.view(1, 1, config.num_q_heads * config.head_dim)
                attn_output = layer.self_attn.o_proj(attn_output)
                hidden = residual + attn_output

                residual = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = residual + hidden

                attn_backend.step_done(layer_idx, config.num_layers)

            # LM head — cast to match lm_head weight dtype (4-bit models have mixed dtypes)
            hidden = self.model.final_norm(hidden)
            lm_dtype = self.model.model.lm_head.weight.dtype
            logits = self.model.lm_head(hidden[:, 0, :].to(lm_dtype))
            cur_token = nucleus_sample(logits, temperature, top_p, rng)

            token_id = cur_token.item()
            generated_ids.append(token_id)

            # Stop on EOS or EOT
            if token_id == eos_id or token_id == eot_id:
                break

        # Cleanup
        del layer_caches, layer_managers
        torch.cuda.empty_cache()

        # Decode tokens to text
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = text.replace("<|eot_id|>", "").strip()
        return text


def nucleus_sample(logits, temperature, top_p, generator):
    """Temperature + top-p token draw; temperature<=0 is exact greedy.

    Ported from eval/math_reasoning_eval.py::sample_token so the
    LongBench sampling probe and the reasoning pilot share one sampling
    definition. `generator` must live on the logits device.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum - sorted_probs > top_p          # tokens fully past the nucleus
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(sorted_probs, 1, generator=generator)
    return sorted_idx.gather(-1, pick).squeeze(-1)


# ─── Runner ───

class LongBenchE2ERunner:
    def __init__(self, model, block_size=16):
        self.model = model
        self.config = model.config
        self.generator = E2EGenerator(model, block_size)
        self.tokenizer = model.tokenizer
        self.max_length = 31500  # leave room for generation
        self.block_size = block_size

    def prepare_prompts(self, dataset_name, data):
        prompt_format = DATASET2PROMPT[dataset_name]
        prompts = []
        model_name = self.config.model_name.lower()
        use_chat = ("instruct" in model_name or "chat" in model_name)
        # Following Ada-KV: chat template hurts few-shot/code tasks
        no_chat_tasks = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
        for item in data:
            prompt = prompt_format.format(**item)
            # Step 1: Truncate raw content FIRST (middle truncation)
            tokens = self.tokenizer(prompt, truncation=False,
                                     return_tensors="pt").input_ids[0]
            if len(tokens) > self.max_length:
                half = self.max_length // 2
                prompt = (self.tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                          self.tokenizer.decode(tokens[-half:], skip_special_tokens=True))
            # Step 2: Apply chat template AFTER truncation, only for suitable tasks
            if use_chat and dataset_name not in no_chat_tasks and \
               hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            prompts.append(prompt)
        return prompts

    def evaluate_dataset(self, dataset_name, attn_backend, max_samples=None,
                         temperature=0.0, top_p=1.0, sample_seed=None):
        data = load_dataset("THUDM/LongBench", dataset_name, split="test",
                            trust_remote_code=True)
        if max_samples and len(data) > max_samples:
            data = data.select(range(max_samples))

        prompts = self.prepare_prompts(dataset_name, data)
        max_gen = DATASET2MAXGEN[dataset_name]
        metric_fn = DATASET2METRIC[dataset_name]

        scores = []
        predictions = []
        t0 = time.time()

        for i, (prompt, item) in enumerate(zip(prompts, data)):
            # Seed per sample index so paired methods share RNG streams
            seed_i = None if sample_seed is None else sample_seed + i
            pred = self.generator.generate(prompt, max_gen, attn_backend,
                                           temperature=temperature,
                                           top_p=top_p, sample_seed=seed_i)
            # Take first line for classification tasks
            pred_clean = pred.split("\n")[0].strip()
            predictions.append(pred_clean)

            answers = item["answers"]
            if isinstance(answers, str):
                answers = [answers]
            score = metric_fn(pred_clean, answers)
            scores.append(score)

            if (i + 1) % 5 == 0 or i == 0:
                print(f"    [{i+1}/{len(prompts)}] "
                      f"running_avg={np.mean(scores)*100:.1f}", flush=True)

        elapsed = time.time() - t0
        mean_score = np.mean(scores) * 100
        per_sample = [s * 100 for s in scores]
        return mean_score, elapsed, len(predictions), predictions, per_sample
