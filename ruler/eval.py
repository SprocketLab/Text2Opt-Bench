#!/usr/bin/env python3
"""
RULER-Style Synthetic Long-Context Evaluation (Hardened)

Generates controlled-length tasks that REQUIRE processing the full input.
Difficulty scales directly with input length — guaranteed to expose
effective context limits that vary by model size.

Task types (inspired by NVIDIA RULER, hardened with distractors):
  1. niah_single     — Single needle retrieval with distractor needles (similar keys)
  2. niah_multikey   — Multi-key retrieval (MK-NIAH) with 3x distractor keys
  3. niah_multivalue — Multi-value NIAH: one key, many values, recall all
  4. (removed — multiquery is redundant with multikey)
  5. vt              — Variable tracking with decoy variables; complexity scales
  6. aggregation     — Count/sum items by category scattered across context

Two-phase workflow for consistency across models:
  Phase 1 — Generate shared samples (once, per-task files):
    python ruler/eval.py \\
        --generate-samples \\
        --samples-dir ruler/samples/ \\
        --tokenizer-model Qwen/Qwen2.5-7B-Instruct \\
        --tasks niah_single,niah_multikey,niah_multivalue,aggregation \\
        --lengths 1024,2048,4096,8192,16384,32000 \\
        --samples-per-length 200

    Creates: samples/niah_single.json, samples/niah_multikey.json, ...

  Phase 2 — Run each model against shared samples:
    python ruler/eval.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --samples-dir ruler/samples/ \\
        --output results/ruler/qwen-7B.json

  Legacy (single-model, generates on the fly):
    python ruler/eval.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --output results/ruler/qwen-7B.json
"""

import argparse
import json
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasets import load_dataset as hf_load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# =============================================================================
# Tokenizer and filler text loading
# =============================================================================
def load_tokenizer(model_name: str):
    """Load the Qwen2.5 HuggingFace tokenizer."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library is required. Install with: pip install transformers")
        sys.exit(1)
    print(f"  Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _count_tokens(tokenizer, text: str) -> int:
    """Count tokens using the HuggingFace tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_essay_paragraphs(min_length: int = 100) -> List[str]:
    """Load Paul Graham essays from HuggingFace."""
    if not HF_DATASETS_AVAILABLE:
        raise RuntimeError("datasets library is required. Install with: pip install datasets")
    ds = hf_load_dataset("sgoel9/paul_graham_essays", split="train")
    paragraphs = []
    for row in ds:
        text = row.get("text", "")
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > min_length:
                paragraphs.append(para)
    print(f"  Loaded {len(paragraphs)} paragraphs from Paul Graham essays")
    return paragraphs


# =============================================================================
# Haystack text generation
# =============================================================================
def _generate_haystack_text(
    tokenizer, target_tokens: int, seed: int, paragraphs: List[str]
) -> str:
    """Generate filler text of approximately target_tokens length."""
    rng = random.Random(seed)
    pool = list(paragraphs)
    text_parts = []
    current_tokens = 0

    while current_tokens < target_tokens:
        rng.shuffle(pool)
        for p in pool:
            text_parts.append(p)
            current_tokens += _count_tokens(tokenizer, p)
            if current_tokens >= target_tokens:
                break

    full_text = "\n\n".join(text_parts)
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        full_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return full_text


def _split_text_at_positions(text: str, positions: List[float]) -> List[str]:
    """Split text at fractional positions (0.0 to 1.0) into segments."""
    sorted_pos = sorted(set(positions))
    char_positions = [int(p * len(text)) for p in sorted_pos]
    char_positions = [0] + char_positions + [len(text)]
    segments = []
    for i in range(len(char_positions) - 1):
        segments.append(text[char_positions[i]:char_positions[i + 1]])
    return segments


# =============================================================================
# Task helpers
# =============================================================================
def _random_value(rng: random.Random, length: int = 6) -> str:
    """Generate a random alphanumeric value."""
    return ''.join(rng.choices(string.ascii_lowercase + string.digits, k=length))


def _random_uuid(rng: random.Random) -> str:
    """Generate a random UUID-like string."""
    parts = [_random_value(rng, n) for n in [8, 4, 4, 4, 12]]
    return '-'.join(parts)


def _make_confusable_name(rng: random.Random, base: str) -> str:
    """Create a name that differs from base by 1 character."""
    chars = list(base)
    pos = rng.randint(0, len(chars) - 1)
    chars[pos] = rng.choice(string.ascii_lowercase + string.digits)
    return ''.join(chars)


# =============================================================================
# Task 1: NIAH Single (with distractors)
# =============================================================================
def generate_niah_single(
    tokenizer, target_tokens: int, sample_idx: int, paragraphs: List[str],
) -> Dict[str, Any]:
    """
    Single needle-in-a-haystack with distractor needles.
    Distractors have confusable key names (1-char difference).
    Difficulty: num_distractors scales with context length.
    """
    rng = random.Random(target_tokens * 1000 + sample_idx)

    num_distractors = max(3, target_tokens // 1024)

    base_name = _random_value(rng, 5)
    key = f"special_item_{base_name}"
    value = _random_uuid(rng)

    distractors = []
    for _ in range(num_distractors):
        dk_name = f"special_item_{_make_confusable_name(rng, base_name)}"
        dk_value = _random_uuid(rng)
        distractors.append((dk_name, dk_value))

    real_needle = f"\nThe {key} is {value}.\n"
    distractor_needles = [f"\nThe {dk} is {dv}.\n" for dk, dv in distractors]
    all_needles = [real_needle] + distractor_needles

    total_needle_tokens = sum(_count_tokens(tokenizer, n) for n in all_needles)
    question = f"What is the {key}? Answer with just the value, nothing else."
    question_tokens = _count_tokens(tokenizer, question)

    haystack_target = target_tokens - total_needle_tokens - question_tokens - 20
    haystack = _generate_haystack_text(
        tokenizer, haystack_target, seed=target_tokens * 1000 + sample_idx, paragraphs=paragraphs
    )

    n_total = len(all_needles)
    positions = sorted([rng.uniform(0.05, 0.95) for _ in range(n_total)])
    rng.shuffle(all_needles)
    segments = _split_text_at_positions(haystack, positions)

    context_parts = []
    for i, seg in enumerate(segments):
        context_parts.append(seg)
        if i < len(all_needles):
            context_parts.append(all_needles[i])
    context = "".join(context_parts)

    prompt = f"{context}\n\n{question}"
    prompt_tokens = _count_tokens(tokenizer, prompt)

    return {
        "problem_id": f"niah_single_{target_tokens}_{sample_idx}",
        "task": "niah_single",
        "prompt": prompt,
        "answer": value,
        "prompt_tokens": prompt_tokens,
        "num_distractors": num_distractors,
        "target_length": target_tokens,
    }


# =============================================================================
# Task 2: NIAH Multi-Key (MK-NIAH)
# =============================================================================
def generate_niah_multikey(
    tokenizer, target_tokens: int, sample_idx: int, paragraphs: List[str],
) -> Dict[str, Any]:
    """
    Multi-key NIAH: scatter N real key-value pairs + 3N distractors.
    Ask for all real values in one question.
    Keys and distractors scale with context length.
    """
    rng = random.Random(target_tokens * 2000 + sample_idx)

    num_keys = max(2, target_tokens // 2048)
    num_distractors = num_keys * 3

    keys_values = []
    for i in range(num_keys):
        k = f"item_{chr(65 + i)}_{_random_value(rng, 4)}"
        v = _random_uuid(rng)
        keys_values.append((k, v))

    distractors = []
    for i in range(num_distractors):
        dk = f"item_{chr(65 + (i % num_keys))}_{_random_value(rng, 4)}"
        dv = _random_uuid(rng)
        distractors.append((dk, dv))

    all_needles = (
        [(k, v, True) for k, v in keys_values] +
        [(k, v, False) for k, v in distractors]
    )
    rng.shuffle(all_needles)

    needle_strs = [f"\nThe {k} is {v}.\n" for k, v, _ in all_needles]
    total_needle_tokens = sum(_count_tokens(tokenizer, n) for n in needle_strs)

    key_list = ", ".join(k for k, v in keys_values)
    question = (
        f"List the values of the following specific items in order, separated by commas: {key_list}\n"
        f"Answer with just the values separated by commas, nothing else."
    )
    question_tokens = _count_tokens(tokenizer, question)

    haystack_target = target_tokens - total_needle_tokens - question_tokens - 30
    haystack = _generate_haystack_text(
        tokenizer, haystack_target, seed=target_tokens * 2000 + sample_idx, paragraphs=paragraphs
    )

    n_total = len(all_needles)
    positions = sorted([rng.uniform(0.05, 0.95) for _ in range(n_total)])
    segments = _split_text_at_positions(haystack, positions)

    context_parts = []
    for i, seg in enumerate(segments):
        context_parts.append(seg)
        if i < len(needle_strs):
            context_parts.append(needle_strs[i])
    context = "".join(context_parts)

    prompt = f"{context}\n\n{question}"
    prompt_tokens = _count_tokens(tokenizer, prompt)
    answer = ", ".join(v for _, v in keys_values)

    return {
        "problem_id": f"niah_multikey_{target_tokens}_{sample_idx}",
        "task": "niah_multikey",
        "prompt": prompt,
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "num_keys": num_keys,
        "num_distractors": num_distractors,
        "target_length": target_tokens,
    }


# =============================================================================
# Task 3: NIAH Multi-Value (MV-NIAH)
# =============================================================================
def generate_niah_multivalue(
    tokenizer, target_tokens: int, sample_idx: int, paragraphs: List[str],
) -> Dict[str, Any]:
    """
    Multi-value NIAH: ONE key appears N times with N different values.
    Model must recall ALL values for that key.
    Distractor keys with similar names also present.
    N and distractors scale with context length.
    """
    rng = random.Random(target_tokens * 5000 + sample_idx)

    num_values = max(2, target_tokens // 2048)
    num_distractor_keys = max(2, target_tokens // 4096 + 1)

    base_name = _random_value(rng, 5)
    key = f"record_{base_name}"

    values = [_random_uuid(rng) for _ in range(num_values)]

    # Distractor keys: similar names, each with one value
    distractor_needles = []
    for _ in range(num_distractor_keys):
        dk_name = f"record_{_make_confusable_name(rng, base_name)}"
        dk_val = _random_uuid(rng)
        distractor_needles.append(f"\nThe {dk_name} has entry {dk_val}.\n")

    # Real needles: same key, different values
    real_needles = [f"\nThe {key} has entry {v}.\n" for v in values]

    all_needles = real_needles + distractor_needles
    rng.shuffle(all_needles)

    total_needle_tokens = sum(_count_tokens(tokenizer, n) for n in all_needles)
    question = (
        f"List ALL entries for {key}, separated by commas. "
        f"Answer with just the values separated by commas, nothing else."
    )
    question_tokens = _count_tokens(tokenizer, question)

    haystack_target = target_tokens - total_needle_tokens - question_tokens - 30
    haystack = _generate_haystack_text(
        tokenizer, haystack_target, seed=target_tokens * 5000 + sample_idx, paragraphs=paragraphs
    )

    n_total = len(all_needles)
    positions = sorted([rng.uniform(0.05, 0.95) for _ in range(n_total)])
    segments = _split_text_at_positions(haystack, positions)

    context_parts = []
    for i, seg in enumerate(segments):
        context_parts.append(seg)
        if i < len(all_needles):
            context_parts.append(all_needles[i])
    context = "".join(context_parts)

    prompt = f"{context}\n\n{question}"
    prompt_tokens = _count_tokens(tokenizer, prompt)
    answer = ", ".join(sorted(values))

    return {
        "problem_id": f"niah_multivalue_{target_tokens}_{sample_idx}",
        "task": "niah_multivalue",
        "prompt": prompt,
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "num_values": num_values,
        "num_distractor_keys": num_distractor_keys,
        "target_length": target_tokens,
    }


# =============================================================================
# Task 5: Aggregation
# =============================================================================
def generate_aggregation(
    tokenizer, target_tokens: int, sample_idx: int, paragraphs: List[str],
) -> Dict[str, Any]:
    """
    Aggregation: scatter tagged items throughout the context.
    Ask the model to count or sum items matching a specific category.
    Forces scanning the ENTIRE context. Difficulty scales with length.
    """
    rng = random.Random(target_tokens * 4000 + sample_idx)

    num_categories = max(3, target_tokens // 4096 + 2)
    items_per_category = max(3, target_tokens // 2048)
    total_items = num_categories * items_per_category

    categories = [f"group_{chr(65 + i)}" for i in range(num_categories)]
    target_category = rng.choice(categories)

    items = []
    for cat in categories:
        for _ in range(items_per_category):
            val = rng.randint(10, 99)
            items.append((cat, val))
    rng.shuffle(items)

    item_strs = [f"\nRecord: {cat} has value {val}.\n" for cat, val in items]
    total_item_tokens = sum(_count_tokens(tokenizer, s) for s in item_strs)

    target_items = [(cat, val) for cat, val in items if cat == target_category]
    if rng.random() < 0.5:
        question = f"How many records belong to {target_category}? Answer with just the number, nothing else."
        answer = str(len(target_items))
        q_type = "count"
    else:
        question = f"What is the sum of all values in {target_category}? Answer with just the number, nothing else."
        answer = str(sum(v for _, v in target_items))
        q_type = "sum"

    question_tokens = _count_tokens(tokenizer, question)

    haystack_target = target_tokens - total_item_tokens - question_tokens - 30
    haystack = _generate_haystack_text(
        tokenizer, haystack_target, seed=target_tokens * 4000 + sample_idx, paragraphs=paragraphs
    )

    positions = sorted([rng.uniform(0.02, 0.98) for _ in range(total_items)])
    segments = _split_text_at_positions(haystack, positions)

    context_parts = []
    for i, seg in enumerate(segments):
        context_parts.append(seg)
        if i < len(item_strs):
            context_parts.append(item_strs[i])
    context = "".join(context_parts)

    prompt = f"{context}\n\n{question}"
    prompt_tokens = _count_tokens(tokenizer, prompt)

    return {
        "problem_id": f"aggregation_{target_tokens}_{sample_idx}",
        "task": "aggregation",
        "prompt": prompt,
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "num_categories": num_categories,
        "total_items": total_items,
        "question_type": q_type,
        "target_length": target_tokens,
    }


# =============================================================================
# Task registry
# =============================================================================
TASK_GENERATORS = {
    "niah_single": generate_niah_single,
    "niah_multikey": generate_niah_multikey,
    "niah_multivalue": generate_niah_multivalue,
    "aggregation": generate_aggregation,
}

ALL_TASKS = list(TASK_GENERATORS.keys())


# =============================================================================
# Evaluators
# =============================================================================
def evaluate_niah_single(prediction: str, answer: str) -> Dict[str, Any]:
    """Check if the predicted value matches (case-insensitive, stripped)."""
    pred = prediction.strip().lower()
    gold = answer.strip().lower()
    uuids = re.findall(r'[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}', pred)
    exact = (gold in pred) or (gold in [u.lower() for u in uuids])
    return {"score": 1.0 if exact else 0.0, "metric": "exact_match"}


def evaluate_niah_multikey(prediction: str, answer: str) -> Dict[str, Any]:
    """All values must be present (strict exact match)."""
    pred = prediction.strip().lower()
    expected_values = [v.strip().lower() for v in answer.split(",")]
    found = sum(1 for v in expected_values if v in pred)
    return {"score": 1.0 if found == len(expected_values) else 0.0, "metric": "exact_match"}


def evaluate_niah_multivalue(prediction: str, answer: str) -> Dict[str, Any]:
    """All values must be present (strict exact match)."""
    pred = prediction.strip().lower()
    expected_values = [v.strip().lower() for v in answer.split(",")]
    found = sum(1 for v in expected_values if v in pred)
    return {"score": 1.0 if found == len(expected_values) else 0.0, "metric": "exact_match"}



def evaluate_aggregation(prediction: str, answer: str) -> Dict[str, Any]:
    """Check if the model produced the correct count/sum."""
    pred = prediction.strip()
    numbers = re.findall(r'\b\d+\b', pred)
    if numbers and numbers[0] == answer:
        return {"score": 1.0, "metric": "exact_match"}
    if answer in pred:
        return {"score": 1.0, "metric": "exact_match"}
    return {"score": 0.0, "metric": "exact_match"}


TASK_EVALUATORS = {
    "niah_single": evaluate_niah_single,
    "niah_multikey": evaluate_niah_multikey,
    "niah_multivalue": evaluate_niah_multivalue,
    "aggregation": evaluate_aggregation,
}


# =============================================================================
# Sample generation
# =============================================================================
def generate_all_samples(
    tasks: List[str],
    lengths: List[int],
    samples_per_length: int,
    tokenizer,
    paragraphs: List[str],
) -> List[Dict[str, Any]]:
    """Generate all synthetic samples. No max_model_len check here."""
    samples = []

    for task in tasks:
        gen_fn = TASK_GENERATORS.get(task)
        if gen_fn is None:
            print(f"  WARNING: Unknown task '{task}', skipping")
            continue

        for length in lengths:
            for i in range(samples_per_length):
                sample = gen_fn(tokenizer, length, i, paragraphs)
                samples.append(sample)

        task_count = len([s for s in samples if s['task'] == task])
        print(f"  {task}: {task_count} samples")

    print(f"  Total: {len(samples)} samples")
    return samples


def save_samples(samples: List[Dict], config: Dict, output_dir: str):
    """Save samples to per-task JSON files inside a directory.

    Creates one file per task: ``<output_dir>/<task>.json``
    Each file has: {"config": {...}, "samples": [...]}
    """
    out = _resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group samples by task
    by_task: Dict[str, List[Dict]] = {}
    for s in samples:
        by_task.setdefault(s["task"], []).append(s)

    for task, task_samples in sorted(by_task.items()):
        task_config = dict(config, task=task)
        task_path = out / f"{task}.json"
        with open(task_path, "w") as f:
            json.dump({"config": task_config, "samples": task_samples}, f)
        print(f"  {task}: {len(task_samples)} samples -> {task_path}")

    print(f"\nSaved {len(samples)} samples across {len(by_task)} files in {out}/")


def load_samples(path: str) -> tuple:
    """Load samples from a directory of per-task JSON files.

    Each ``<task>.json`` has {"config": {...}, "samples": [...]}.
    Returns (merged_config, all_samples).
    """
    resolved = _resolve_path(path)

    if resolved.is_dir():
        task_files = sorted(resolved.glob("*.json"))
        if not task_files:
            raise FileNotFoundError(f"No .json files found in {resolved}")

        all_samples = []
        config = {}
        tasks_loaded = []
        for tf in task_files:
            with open(tf) as f:
                data = json.load(f)
            tc = data.get("config", {})
            ts = data.get("samples", [])
            task_name = tc.get("task", tf.stem)
            tasks_loaded.append(task_name)
            all_samples.extend(ts)
            # Merge config (keep shared keys from first file, accumulate tasks)
            if not config:
                config = {k: v for k, v in tc.items() if k != "task"}
            print(f"  {task_name}: {len(ts)} samples from {tf.name}")

        config["tasks"] = tasks_loaded
        print(f"  Total: {len(all_samples)} samples from {len(task_files)} files")
        print(f"  Config: tasks={config.get('tasks')}, lengths={config.get('lengths')}, "
              f"samples_per_length={config.get('samples_per_length')}")
        return config, all_samples
    else:
        # Backward compat: single file
        with open(resolved) as f:
            data = json.load(f)
        config = data.get("config", {})
        samples = data.get("samples", [])
        print(f"  Loaded {len(samples)} samples from {path}")
        print(f"  Config: tasks={config.get('tasks')}, lengths={config.get('lengths')}, "
              f"samples_per_length={config.get('samples_per_length')}")
        return config, samples


# =============================================================================
# Inference
# =============================================================================
def run_generate_samples(args):
    """Generate samples and save to file, then exit."""
    tasks = [t.strip() for t in args.tasks.split(",")]
    lengths = [int(l.strip()) for l in args.lengths.split(",")]

    print(f"\n{'='*60}")
    print(f"RULER Sample Generation")
    print(f"{'='*60}")

    tokenizer = load_tokenizer(args.tokenizer_model)
    paragraphs = load_essay_paragraphs()

    print(f"\nGenerating shared samples...")
    print(f"  Tasks: {tasks}")
    print(f"  Lengths: {lengths}")
    print(f"  Samples per length: {args.samples_per_length}")
    print(f"  Tokenizer: {args.tokenizer_model}")

    samples = generate_all_samples(tasks, lengths, args.samples_per_length, tokenizer, paragraphs)

    config = {
        "tasks": tasks,
        "lengths": lengths,
        "samples_per_length": args.samples_per_length,
        "tokenizer_model": args.tokenizer_model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    save_samples(samples, config, args.samples_dir)


def run_evaluation(args) -> List[Dict[str, Any]]:
    """Run model inference on samples and evaluate."""
    if not VLLM_AVAILABLE:
        print("Error: vLLM is not installed.")
        sys.exit(1)

    # Load samples: from file or generate on the fly
    if args.samples_dir:
        print(f"\nLoading pre-generated samples from {args.samples_dir}...")
        config, samples = load_samples(args.samples_dir)
    else:
        # Backward compat: generate on the fly
        tasks = [t.strip() for t in args.tasks.split(",")]
        lengths = [int(l.strip()) for l in args.lengths.split(",")]
        tokenizer = load_tokenizer(args.tokenizer_model)
        paragraphs = load_essay_paragraphs()
        print(f"\nGenerating samples on the fly...")
        samples = generate_all_samples(tasks, lengths, args.samples_per_length, tokenizer, paragraphs)

    if not samples:
        print("No samples to evaluate.")
        return []

    # Derive max_model_len
    max_model_len = args.max_model_len
    if max_model_len is None:
        try:
            cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            max_model_len = getattr(cfg, "max_position_embeddings", None)
            if max_model_len:
                print(f"  Derived max_model_len={max_model_len} from model config")
            else:
                max_model_len = 32768
        except Exception:
            max_model_len = 32768

    # Filter samples that exceed the model's context window
    valid_indices = []
    skipped = 0
    for i, s in enumerate(samples):
        if max_model_len and s["prompt_tokens"] + args.max_tokens > max_model_len:
            skipped += 1
            continue
        valid_indices.append(i)

    if skipped > 0:
        print(f"  Skipped {skipped} samples exceeding max_model_len={max_model_len}")
    print(f"  Running inference on {len(valid_indices)} of {len(samples)} samples")

    prompts = [samples[i]["prompt"] for i in valid_indices]

    # Initialize vLLM
    print(f"\nInitializing vLLM: {args.model}")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  max_model_len={max_model_len}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=args.enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    print(f"\nGenerating {len(prompts)} responses...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/sec)")

    # Build raw results (include ALL samples, mark skipped ones)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_results = []
    output_idx = 0
    for i, sample in enumerate(samples):
        if i in set(valid_indices):
            generated = outputs[output_idx].outputs[0].text
            output_idx += 1
        else:
            generated = ""  # skipped sample

        raw_results.append({
            "problem_id": sample["problem_id"],
            "meta": {
                "prompt_tokens": sample["prompt_tokens"],
                "problem_type": "ruler",
                "task": sample["task"],
                "target_length": sample["target_length"],
            },
            "evaluation_result": None,
            "model": args.model,
            "generated_output": generated,
            "answer": sample["answer"],
        })

    raw_path = output_path.with_suffix(".raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Raw outputs saved to {raw_path}")

    return _evaluate_and_save(raw_results, output_path)


def _evaluate_and_save(raw_results, output_path):
    """Evaluate and save results."""
    from collections import defaultdict
    task_length_scores = defaultdict(lambda: defaultdict(list))

    for entry in raw_results:
        task = entry["meta"]["task"]
        target_len = entry["meta"]["target_length"]
        prediction = entry["generated_output"]
        answer = entry["answer"]

        if not prediction:
            continue  # skip samples that weren't run

        eval_fn = TASK_EVALUATORS.get(task, evaluate_niah_single)
        ev = eval_fn(prediction, answer)
        entry["evaluation_result"] = ev
        task_length_scores[task][target_len].append(ev["score"])

    # Summary
    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")

    for task in sorted(task_length_scores.keys()):
        print(f"\n  {task}:")
        for length in sorted(task_length_scores[task].keys()):
            scores = task_length_scores[task][length]
            mean = sum(scores) / len(scores) * 100
            print(f"    {length:6d} tokens: {mean:5.1f}% (n={len(scores)})")

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return raw_results


def run_validate_only(args):
    """Re-evaluate existing raw results."""
    raw_path = _resolve_path(args.validate_only)
    with open(raw_path) as f:
        raw_results = json.load(f)
    print(f"Loaded {len(raw_results)} entries from {raw_path}")
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _evaluate_and_save(raw_results, output_path)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RULER-style synthetic long-context evaluation (hardened)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--tokenizer-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for tokenizer (all Qwen2.5 models share the same tokenizer)",
    )
    parser.add_argument(
        "--tasks", type=str,
        default="niah_single,niah_multikey,niah_multivalue,vt,aggregation",
        help="Comma-separated task names",
    )
    parser.add_argument(
        "--lengths", type=str, default="1024,2048,4096,8192,16384,32000",
        help="Comma-separated target token lengths",
    )
    parser.add_argument("--samples-per-length", type=int, default=200)
    parser.add_argument(
        "--output", type=str,
        default="results/ruler/eval_results.json",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graph capture (fixes NCCL issues with >2 PCIe GPUs)")
    parser.add_argument("--validate-only", type=str, default=None)
    parser.add_argument(
        "--generate-samples", action="store_true",
        help="Generate samples and save to --samples-dir, then exit (no model inference)",
    )
    parser.add_argument(
        "--samples-dir", type=str, default=None,
        help="Directory for per-task sample files. With --generate-samples: output dir. With --model: input dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_only:
        run_validate_only(args)
    elif args.generate_samples:
        if not args.samples_dir:
            print("Error: --generate-samples requires --samples-dir")
            sys.exit(1)
        run_generate_samples(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
