#!/usr/bin/env python3
"""
Evaluate a trained binding model on template categories (Transportation, JSSP) using vLLM.

Measures:
  1. JSON parse rate (did the model output valid JSON?)
  2. Field-level accuracy per category
  3. Full-problem accuracy (all fields correct)

Transportation fields: num_sources, num_destinations, integer_flows, supplies, demands, cost_matrix
JSSP fields: n_jobs, n_machines, processing_times, machine_assignments
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

BENCH_ROOT = Path(__file__).resolve().parent.parent.parent

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

# Relative tolerance for numeric comparison
RTOL = 1e-4
ATOL = 1e-6

# ── Size cutoffs for in-dist / OOD split ─────────────────────────────
SIZE_BOUNDARY = {
    "transportation": lambda d: (
        d["instance_data"]["num_sources"] <= 10
        and d["instance_data"]["num_destinations"] <= 10
    ),
    "jssp": lambda d: d["instance_data"]["n_jobs"] <= 5,
}

# ── Which fields to evaluate per category ────────────────────────────
INSTANCE_DATA_KEYS = {
    "transportation": [
        "num_sources", "num_destinations", "integer_flows",
        "supplies", "demands", "cost_matrix",
    ],
    "jssp": [
        "n_jobs", "n_machines", "processing_times", "machine_assignments",
    ],
}


def parse_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from a fenced ```json ... ``` block."""
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def values_match(pred, gold) -> bool:
    """Recursively compare values with tolerance for floats."""
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        if gold == pred:
            return True
        if abs(gold) > ATOL:
            return abs(pred - gold) / abs(gold) < RTOL
        return abs(pred - gold) < ATOL
    if isinstance(gold, bool) and isinstance(pred, bool):
        return pred == gold
    if isinstance(gold, list) and isinstance(pred, list):
        if len(pred) != len(gold):
            return False
        return all(values_match(p, g) for p, g in zip(pred, gold))
    # Exact match for other types (str, bool, etc.)
    return pred == gold


def evaluate_extraction(pred: dict, gold_instance_data: dict, category: str) -> dict:
    """Compare predicted instance_data extraction against ground truth."""
    keys = INSTANCE_DATA_KEYS[category]
    result = {
        "field_results": {},
        "full_correct": True,
        "errors": [],
    }

    for key in keys:
        gold_val = gold_instance_data[key]
        pred_val = pred.get(key)

        if pred_val is None:
            result["field_results"][key] = False
            result["full_correct"] = False
            result["errors"].append(f"{key}: missing")
        elif values_match(pred_val, gold_val):
            result["field_results"][key] = True
        else:
            result["field_results"][key] = False
            result["full_correct"] = False
            # Compact error: show shapes/lengths for arrays
            if isinstance(gold_val, list):
                pred_shape = _shape_str(pred_val)
                gold_shape = _shape_str(gold_val)
                result["errors"].append(
                    f"{key}: shape pred={pred_shape} vs gold={gold_shape}"
                    if pred_shape != gold_shape
                    else f"{key}: values mismatch"
                )
            else:
                result["errors"].append(f"{key}: pred={pred_val} vs gold={gold_val}")

    return result


def _shape_str(val) -> str:
    """Describe the shape of a nested list."""
    if not isinstance(val, list):
        return "scalar"
    if not val:
        return "[]"
    if isinstance(val[0], list):
        return f"[{len(val)}×{len(val[0])}]"
    return f"[{len(val)}]"


def main():
    parser = argparse.ArgumentParser(description="Evaluate binding model on template categories")
    parser.add_argument("--category", required=True, choices=["transportation", "jssp"])
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained binding model",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Override eval directory (defaults to Template/basic_template/<category>/small)",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    category = args.category

    # Default eval directory
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = (
            BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template"
            / category / "small"
        )

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = BENCH_ROOT / "results" / f"binding_eval_{category}.json"

    from vllm import LLM, SamplingParams

    problem_files = sorted(eval_dir.glob("*.json"))
    if args.max_problems > 0:
        problem_files = problem_files[:args.max_problems]

    print(f"Category: {category}")
    print(f"Evaluating {len(problem_files)} problems from {eval_dir}")

    # Load all problems and build prompts
    problems = []
    prompts = []
    for pf in problem_files:
        with open(pf) as f:
            problem = json.load(f)
        llm_desc = problem.get("LLM_description", "").strip()
        if not llm_desc:
            continue
        problems.append((pf.stem, problem))
        prompts.append(INSTRUCTION_PREFIX + llm_desc)

    print(f"Loaded {len(prompts)} problems with descriptions")

    # Initialize vLLM
    print(f"Loading vLLM model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Build chat-formatted prompts
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)

    print(f"Generating {len(chat_prompts)} responses...")
    outputs = llm.generate(chat_prompts, sampling_params)
    print("Generation complete.")

    # Evaluate
    results = []
    is_in_dist = SIZE_BOUNDARY[category]
    keys = INSTANCE_DATA_KEYS[category]

    counts = {"total": 0, "parse_ok": 0, "full_correct": 0}
    counts_in = {"total": 0, "parse_ok": 0, "full_correct": 0}
    counts_ood = {"total": 0, "parse_ok": 0, "full_correct": 0}
    field_counts = {k: 0 for k in keys}
    field_counts_in = {k: 0 for k in keys}
    field_counts_ood = {k: 0 for k in keys}

    for i, output in enumerate(outputs):
        problem_id, problem = problems[i]
        response = output.outputs[0].text
        gold = problem["instance_data"]
        in_dist = is_in_dist(problem)
        bucket = counts_in if in_dist else counts_ood

        counts["total"] += 1
        bucket["total"] += 1

        entry = {
            "problem_id": problem_id,
            "in_distribution": in_dist,
            "num_vars": problem["meta"]["num_vars"],
            "num_constraints": problem["meta"]["num_constraints"],
        }

        pred = parse_json_from_response(response)
        if pred is None:
            entry["parse_ok"] = False
            entry["eval"] = None
            entry["errors"] = ["Failed to parse JSON"]
            print(f"  [{i+1}/{len(outputs)}] {problem_id}: PARSE FAIL")
        else:
            counts["parse_ok"] += 1
            bucket["parse_ok"] += 1
            entry["parse_ok"] = True

            eval_result = evaluate_extraction(pred, gold, category)
            entry["eval"] = eval_result

            if eval_result["full_correct"]:
                counts["full_correct"] += 1
                bucket["full_correct"] += 1

            for key in keys:
                if eval_result["field_results"].get(key, False):
                    field_counts[key] += 1
                    (field_counts_in if in_dist else field_counts_ood)[key] += 1

            status = "PASS" if eval_result["full_correct"] else "FAIL"
            dist_tag = "IN" if in_dist else "OOD"
            err_summary = f" ({len(eval_result['errors'])} errors)" if eval_result["errors"] else ""
            print(f"  [{i+1}/{len(outputs)}] {problem_id} [{dist_tag}]: {status}{err_summary}")

        results.append(entry)

    # Summary
    n = counts["total"]
    n_in = counts_in["total"]
    n_ood = counts_ood["total"]

    def _pct(v, t):
        return f"{v}/{t} ({v/t*100:.1f}%)" if t > 0 else "N/A"

    print(f"\n{'='*70}")
    print(f"BINDING EVALUATION: {category.upper()} ({n} problems: {n_in} in-dist, {n_ood} OOD)")
    print(f"{'='*70}")
    print(f"  {'Metric':25s} {'Overall':>15s} {'In-dist':>15s} {'OOD':>15s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'parse_ok':25s} {_pct(counts['parse_ok'], n):>15s} {_pct(counts_in['parse_ok'], n_in):>15s} {_pct(counts_ood['parse_ok'], n_ood):>15s}")
    print(f"  {'full_correct':25s} {_pct(counts['full_correct'], n):>15s} {_pct(counts_in['full_correct'], n_in):>15s} {_pct(counts_ood['full_correct'], n_ood):>15s}")
    for key in keys:
        print(f"  {key:25s} {_pct(field_counts[key], n):>15s} {_pct(field_counts_in[key], n_in):>15s} {_pct(field_counts_ood[key], n_ood):>15s}")
    print(f"{'='*70}")

    # Save
    summary = {
        "category": category,
        "counts": counts,
        "counts_in_dist": counts_in,
        "counts_ood": counts_ood,
        "field_counts": field_counts,
        "field_counts_in_dist": field_counts_in,
        "field_counts_ood": field_counts_ood,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
