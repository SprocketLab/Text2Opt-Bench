#!/usr/bin/env python3
"""
Evaluate an end-to-end SFT model on template categories.

The model generates complete Gurobi code directly from the problem description.
No two-phase pipeline — single model does binding + formulation + coding.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main.evaluation.problem_evaluator import GeneratedSolutionEvaluator

SYSTEM_PROMPT = (
    "You are an expert operations research engineer. "
    "Read the optimization problem description and write a complete Python script "
    "using Gurobi to solve it. Return ONLY a fenced ```python ... ``` block."
)

INSTRUCTION_PREFIX = (
    "Write a complete Gurobi Python script to solve the following optimization problem.\n\n"
)

SIZE_BOUNDARY = {
    "transportation": lambda d: (
        d["instance_data"]["num_sources"] <= 10
        and d["instance_data"]["num_destinations"] <= 10
    ),
    "jssp": lambda d: d["instance_data"]["n_jobs"] <= 5,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate e2e SFT model on template categories")
    parser.add_argument("--category", required=True, choices=["transportation", "jssp"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    category = args.category

    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = (
            PROJECT_ROOT / "synthetic_dataset" / "Template" / "basic_template"
            / category / "small"
        )

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = PROJECT_ROOT / "results" / f"e2e_eval_{category}.json"

    from vllm import LLM, SamplingParams

    # Load problems
    problem_files = sorted(eval_dir.glob("*.json"))
    if args.max_problems > 0:
        problem_files = problem_files[:args.max_problems]

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

    is_in_dist = SIZE_BOUNDARY[category]
    n_in = sum(1 for _, p in problems if is_in_dist(p))
    n_ood = len(problems) - n_in

    print(f"Category: {category}")
    print(f"Loaded {len(problems)} problems ({n_in} in-dist, {n_ood} OOD)")

    # Initialize vLLM
    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    chat_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)
    print(f"Generating {len(chat_prompts)} responses...")
    outputs = llm.generate(chat_prompts, sampling_params)
    print("Generation complete.")

    # Evaluate
    evaluator = GeneratedSolutionEvaluator()
    results = []
    counts = {"total": 0, "exec_ok": 0, "optimal": 0, "correct": 0}
    counts_in = {"total": 0, "exec_ok": 0, "optimal": 0, "correct": 0}
    counts_ood = {"total": 0, "exec_ok": 0, "optimal": 0, "correct": 0}

    for i, output in enumerate(outputs):
        problem_id, problem_data = problems[i]
        code = output.outputs[0].text
        in_dist = is_in_dist(problem_data)
        bucket = counts_in if in_dist else counts_ood

        counts["total"] += 1
        bucket["total"] += 1

        eval_result = evaluator.evaluate_generated_code(problem_data, code, problem_id)

        entry = {
            "problem_id": problem_id,
            "in_distribution": in_dist,
            "num_vars": problem_data["meta"]["num_vars"],
            "num_constraints": problem_data["meta"]["num_constraints"],
            "execution_succeeded": eval_result.execution_succeeded,
            "status_optimal": eval_result.status_optimal,
            "objective_matches": eval_result.objective_matches,
            "objective_error": eval_result.objective_error,
            "reference_optimum": eval_result.reference_optimum,
        }
        if eval_result.generated_solution.objective_value is not None:
            entry["predicted_optimum"] = eval_result.generated_solution.objective_value

        if eval_result.execution_succeeded:
            counts["exec_ok"] += 1
            bucket["exec_ok"] += 1
        if eval_result.status_optimal:
            counts["optimal"] += 1
            bucket["optimal"] += 1
        if eval_result.objective_matches:
            counts["correct"] += 1
            bucket["correct"] += 1

        status = "PASS" if eval_result.objective_matches else "FAIL"
        dist_tag = "IN" if in_dist else "OOD"
        obj_info = ""
        if eval_result.objective_error is not None:
            obj_info = f" (err={eval_result.objective_error:.4f})"
        elif not eval_result.execution_succeeded:
            stderr = eval_result.generated_solution.stderr[-80:] if eval_result.generated_solution.stderr else ""
            obj_info = f" (exec fail: {stderr})"

        print(f"  [{i+1}/{len(outputs)}] {problem_id} [{dist_tag}]: {status}{obj_info}")
        entry["result"] = status.lower()
        results.append(entry)

    # Summary
    def _pct(v, t):
        return f"{v}/{t} ({v/t*100:.1f}%)" if t > 0 else "N/A"

    n = counts["total"]
    print(f"\n{'='*70}")
    print(f"E2E SFT EVALUATION: {category.upper()}")
    print(f"  Model: {args.model_path}")
    print(f"{'='*70}")
    print(f"  {'Metric':20s} {'Overall':>18s} {'In-dist':>18s} {'OOD':>18s}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    for key in ["exec_ok", "optimal", "correct"]:
        print(f"  {key:20s} {_pct(counts[key], n):>18s} {_pct(counts_in[key], counts_in['total']):>18s} {_pct(counts_ood[key], counts_ood['total']):>18s}")
    print(f"{'='*70}")

    # Save
    summary = {
        "category": category,
        "counts": counts,
        "counts_in_dist": counts_in,
        "counts_ood": counts_ood,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"category": category, "model_path": args.model_path, "eval_dir": str(eval_dir)},
            "summary": summary,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
