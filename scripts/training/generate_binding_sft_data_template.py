#!/usr/bin/env python3
"""
Generate binding-only SFT training data for template categories (Transportation, JSSP).

Reads template problem JSONs and produces (instruction, output) pairs where:
  - instruction = LLM_description (natural language problem)
  - output = instance_data JSON (domain-specific parameters)

This trains a model to extract structured instance data from prose,
NOT to formulate or solve the optimization problem.

Size filtering mirrors the RA experiment's train-on-easy / eval-on-full design:
  - Transportation: train on instances with num_sources <= 10 AND num_destinations <= 10
  - JSSP: train on instances with n_jobs <= 5
"""

import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent.parent
LLAMA_DATA = Path("data/training")

SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Read the optimization problem description and extract all instance "
    "parameters into structured JSON. "
    "Return ONLY a fenced ```json ... ``` block."
)

INSTRUCTION_PREFIX = (
    "Extract all instance parameters from the following problem description "
    "into structured JSON.\n\n"
)

# ── Size cutoffs (train on small, eval on full range) ────────────────
SIZE_FILTERS = {
    "transportation": lambda d: (
        d["instance_data"]["num_sources"] <= 10
        and d["instance_data"]["num_destinations"] <= 10
    ),
    # capped: train only on grids where sources×dests ≤ 52 (matching JSSP in-dist scale)
    "transportation_capped": lambda d: (
        d["instance_data"]["num_sources"] * d["instance_data"]["num_destinations"] <= 52
    ),
    "jssp": lambda d: d["instance_data"]["n_jobs"] <= 5,
}

# ── Which instance_data keys to extract per category ─────────────────
INSTANCE_DATA_KEYS = {
    "transportation": [
        "num_sources", "num_destinations", "integer_flows",
        "supplies", "demands", "cost_matrix",
    ],
    "jssp": [
        "n_jobs", "n_machines", "processing_times", "machine_assignments",
    ],
}


def extract_binding_target(problem: dict, category: str) -> dict:
    """Extract the instance_data fields relevant to this category."""
    instance_data = problem["instance_data"]
    keys = INSTANCE_DATA_KEYS[category]
    return {k: instance_data[k] for k in keys}


def format_binding_output(target: dict, compact: bool = False) -> str:
    """Wrap the target JSON in a fenced code block."""
    if compact:
        return "```json\n" + json.dumps(target, separators=(',', ':')) + "\n```"
    return "```json\n" + json.dumps(target, indent=2) + "\n```"


def generate_dataset(
    problem_dir: Path,
    category: str,
    apply_size_filter: bool = False,
    compact_json: bool = False,
) -> list[dict]:
    """Generate LLaMA-Factory format dataset from a directory of problem JSONs."""
    dataset = []
    skipped_size = 0
    skipped_desc = 0
    size_filter = SIZE_FILTERS[category]

    problem_files = sorted(problem_dir.glob("*.json"))
    for pf in problem_files:
        with open(pf) as f:
            problem = json.load(f)

        llm_desc = problem.get("LLM_description", "").strip()
        if not llm_desc:
            skipped_desc += 1
            continue

        if apply_size_filter and not size_filter(problem):
            skipped_size += 1
            continue

        target = extract_binding_target(problem, category)
        output = format_binding_output(target, compact=compact_json)

        dataset.append({
            "instruction": INSTRUCTION_PREFIX + llm_desc,
            "input": "",
            "output": output,
        })

    if skipped_size:
        print(f"  Filtered out {skipped_size} instances (above size cutoff)")
    if skipped_desc:
        print(f"  Skipped {skipped_desc} instances (no LLM_description)")

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
    # Capped transportation: train on vars ≤ 52 with compact JSON, eval on full original 50-problem set
    "transportation_capped": {
        "train_dir": BENCH_ROOT / "synthetic_dataset" / "Template_train" / "transportation" / "small_capped",
        "eval_dir": BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "transportation" / "small",
        "base_category": "transportation",
        "compact_json": True,
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

        compact = config.get("compact_json", False)

        # Training data (size-filtered)
        train_dir = config["train_dir"]
        if train_dir.exists():
            print(f"\nTraining data (size-filtered) from {train_dir}")
            train_data = generate_dataset(train_dir, base_category, apply_size_filter=True, compact_json=compact)
            print(f"  {len(train_data)} training examples")

            out_path = LLAMA_DATA / f"binding_train_{category}.json"
            with open(out_path, "w") as f:
                json.dump(train_data, f, indent=2)
            print(f"  -> {out_path}")
        else:
            print(f"  Train dir not found: {train_dir}", file=sys.stderr)

        # Eval data (no size filter — includes both in-dist and OOD)
        eval_dir = config["eval_dir"]
        if eval_dir.exists():
            print(f"\nEval data (full range) from {eval_dir}")
            eval_data = generate_dataset(eval_dir, base_category, apply_size_filter=False, compact_json=compact)
            print(f"  {len(eval_data)} eval examples")

            out_path = LLAMA_DATA / f"binding_eval_{category}.json"
            with open(out_path, "w") as f:
                json.dump(eval_data, f, indent=2)
            print(f"  -> {out_path}")
        else:
            print(f"  Eval dir not found: {eval_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
