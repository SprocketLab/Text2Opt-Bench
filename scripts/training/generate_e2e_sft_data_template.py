#!/usr/bin/env python3
"""
Generate end-to-end SFT training data for template categories (Transportation, JSSP).

Reads template problem JSONs and produces (instruction, output) pairs where:
  - instruction = LLM_description (natural language problem)
  - output = gold_solution (complete Gurobi solver code)

This is the baseline comparison: single model generates full code from text.
Same size filtering as generate_binding_sft_data_template.py.
"""

import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent.parent
LLAMA_DATA = Path("data/training")

SYSTEM_PROMPT = (
    "You are an expert operations research engineer. "
    "Read the optimization problem description and write a complete Python script "
    "using Gurobi to solve it. Return ONLY a fenced ```python ... ``` block."
)

INSTRUCTION_PREFIX = (
    "Write a complete Gurobi Python script to solve the following optimization problem.\n\n"
)

# ── Size cutoffs (same as binding data) ──────────────────────────────
SIZE_FILTERS = {
    "transportation": lambda d: (
        d["instance_data"]["num_sources"] <= 10
        and d["instance_data"]["num_destinations"] <= 10
    ),
    "transportation_capped": lambda d: (
        d["instance_data"]["num_sources"] * d["instance_data"]["num_destinations"] <= 52
    ),
    "jssp": lambda d: d["instance_data"]["n_jobs"] <= 5,
}


def generate_dataset(
    problem_dir: Path,
    category: str,
    apply_size_filter: bool = False,
) -> list[dict]:
    """Generate LLaMA-Factory format dataset from a directory of problem JSONs."""
    dataset = []
    skipped_size = 0
    skipped_desc = 0
    skipped_gold = 0
    size_filter = SIZE_FILTERS[category]

    problem_files = sorted(problem_dir.glob("*.json"))
    for pf in problem_files:
        with open(pf) as f:
            problem = json.load(f)

        llm_desc = problem.get("LLM_description", "").strip()
        if not llm_desc:
            skipped_desc += 1
            continue

        gold = problem.get("gold_solution", "").strip()
        if not gold:
            skipped_gold += 1
            continue

        if apply_size_filter and not size_filter(problem):
            skipped_size += 1
            continue

        dataset.append({
            "instruction": INSTRUCTION_PREFIX + llm_desc,
            "input": "",
            "output": gold,
        })

    if skipped_size:
        print(f"  Filtered out {skipped_size} instances (above size cutoff)")
    if skipped_desc:
        print(f"  Skipped {skipped_desc} instances (no LLM_description)")
    if skipped_gold:
        print(f"  Skipped {skipped_gold} instances (no gold_solution)")

    return dataset


# ── Category directory configs ────────────────────────────────────────
CATEGORY_CONFIGS = {
    "transportation": {
        "train_dir": BENCH_ROOT / "synthetic_dataset" / "Template_train" / "transportation" / "small",
        "eval_dir": BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "transportation" / "small",
    },
    "jssp": {
        "train_dir": BENCH_ROOT / "synthetic_dataset" / "Template_train" / "jssp" / "small",
        "eval_dir": BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "jssp" / "small",
    },
    "transportation_prose": {
        "train_dir": BENCH_ROOT / "synthetic_dataset" / "Template_train" / "transportation" / "small_prose",
        "eval_dir": BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "transportation" / "small_prose",
        "base_category": "transportation",
    },
    "jssp_prose": {
        "train_dir": BENCH_ROOT / "synthetic_dataset" / "Template_train" / "jssp" / "small_prose",
        "eval_dir": BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "jssp" / "small_prose",
        "base_category": "jssp",
    },
    "transportation_capped": {
        "train_dir": BENCH_ROOT / "synthetic_dataset" / "Template_train" / "transportation" / "small_capped",
        "eval_dir": BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "transportation" / "small",
        "base_category": "transportation",
    },
}


def main():
    categories = sys.argv[1:] if len(sys.argv) > 1 else list(CATEGORY_CONFIGS.keys())

    for category in categories:
        if category not in CATEGORY_CONFIGS:
            print(f"Unknown category: {category}", file=sys.stderr)
            continue

        config = CATEGORY_CONFIGS[category]
        base_category = config.get("base_category", category)
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        # Training data (size-filtered)
        train_dir = config["train_dir"]
        if train_dir.exists():
            print(f"\nTraining data (size-filtered) from {train_dir}")
            train_data = generate_dataset(train_dir, base_category, apply_size_filter=True)
            print(f"  {len(train_data)} training examples")

            out_path = LLAMA_DATA / f"e2e_train_{category}.json"
            with open(out_path, "w") as f:
                json.dump(train_data, f, indent=2)
            print(f"  -> {out_path}")
        else:
            print(f"  Train dir not found: {train_dir}", file=sys.stderr)

        # Eval data (no size filter)
        eval_dir = config["eval_dir"]
        if eval_dir.exists():
            print(f"\nEval data (full range) from {eval_dir}")
            eval_data = generate_dataset(eval_dir, base_category, apply_size_filter=False)
            print(f"  {len(eval_data)} eval examples")

            out_path = LLAMA_DATA / f"e2e_eval_{category}.json"
            with open(out_path, "w") as f:
                json.dump(eval_data, f, indent=2)
            print(f"  -> {out_path}")
        else:
            print(f"  Eval dir not found: {eval_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
