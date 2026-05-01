#!/usr/bin/env python3
"""
Two-phase evaluation for template categories (Transportation, JSSP).

Phase 1: Binding model extracts instance_data from problem description (vLLM).
Phase 2: Either (a) deterministic template code or (b) an untrained LLM generates solver code.
Eval:    Execute generated code, compare objective to ground truth.

Usage:
    # Ground-truth upper bound + template code:
    python eval_two_phase_template.py --category transportation --ground-truth --phase2 template

    # Trained binder + template code:
    python eval_two_phase_template.py --category transportation --binding-model <path> --phase2 template

    # Trained binder + untrained LLM:
    python eval_two_phase_template.py --category jssp --binding-model <path> --phase2 model
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main.evaluation.problem_evaluator import GeneratedSolutionEvaluator
from scripts.training.template_solvers import build_template_code

# ── Binding prompts (instance_data extraction) ───────────────────────
BINDING_SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Read the optimization problem description and extract all instance "
    "parameters into structured JSON. "
    "Return ONLY a fenced ```json ... ``` block."
)

BINDING_INSTRUCTION_PREFIX = (
    "Extract all instance parameters from the following problem description "
    "into structured JSON.\n\n"
)

# ── Phase 2 LLM prompt (for --phase2 model) ──────────────────────────
SOLVER_SYSTEM_PROMPT = (
    "You are an optimization coding agent. "
    "Return only executable Python code in a single fenced ```python ... ``` block."
)

INSTANCE_DATA_KEYS = {
    "transportation": [
        "num_sources", "num_destinations", "integer_flows",
        "supplies", "demands", "cost_matrix",
    ],
    "jssp": [
        "n_jobs", "n_machines", "processing_times", "machine_assignments",
    ],
}

SOLVER_SCHEMA_DESCRIPTION = {
    "transportation": textwrap.dedent("""\
        The JSON file contains a transportation problem with these fields:
          num_sources       : int — number of supply sources
          num_destinations  : int — number of demand destinations
          integer_flows     : bool — whether flow variables must be integers
          supplies          : list[int] — available supply at each source
          demands           : list[int] — required demand at each destination
          cost_matrix       : list[list[float]] — cost_matrix[source][destination]

        Model: min Σ cost[i][j] * flow[i][j]
        s.t.  Σ_j flow[i][j] <= supplies[i]  for each source i
              Σ_i flow[i][j] >= demands[j]    for each destination j
              flow[i][j] >= 0"""),
    "jssp": textwrap.dedent("""\
        The JSON file contains a job shop scheduling problem with these fields:
          n_jobs              : int — number of jobs
          n_machines          : int — number of machines
          processing_times    : list[list[int]] — processing_times[job][operation]
          machine_assignments : list[list[int]] — machine_assignments[job][operation] (0-based machine index)

        Model: min Cmax (makespan)
        Variables: Start[j,o] (continuous start time), Cmax (continuous), Y binary for disjunctive pairs
        Constraints:
          - Precedence: Start[j,o+1] >= Start[j,o] + processing_times[j][o]
          - Disjunctive (big-M): for each pair on same machine, either A before B or B before A
          - Makespan: Cmax >= Start[j, last_op] + processing_times[j][last_op]"""),
}

# ── Size boundaries for in-dist / OOD ────────────────────────────────
SIZE_BOUNDARY = {
    "transportation": lambda d: (
        d["instance_data"]["num_sources"] <= 10
        and d["instance_data"]["num_destinations"] <= 10
    ),
    "jssp": lambda d: d["instance_data"]["n_jobs"] <= 5,
}


def parse_json_from_response(text: str) -> Optional[dict]:
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


def parse_code_from_response(text: str) -> Optional[str]:
    """Extract Python code from a fenced block."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fenced block, try the whole response
    if "def solve_problem" in text:
        return text.strip()
    return None


def build_solve_prompt(extracted_path: Path, category: str) -> str:
    """Build the Phase 2 LLM prompt for solver code generation."""
    schema = SOLVER_SCHEMA_DESCRIPTION[category]
    return textwrap.dedent(f"""\
        Write a complete Python script using Gurobi to solve the optimization problem
        described by the data in the JSON file at:

        EXTRACTED_DATA_PATH = r"{extracted_path}"

        {schema}

        Requirements:
          1. Return exactly one fenced ```python ... ``` block.
          2. ONLY import `json` and `from gurobipy import *`. No other imports.
          3. Define `solve_problem()` with no required arguments.
          4. Inside `solve_problem()`, open EXTRACTED_DATA_PATH with json.load().
          5. Build the complete Gurobi model from the loaded data.
          6. Set m.setParam("TimeLimit", 300).
          7. Call m.optimize() and return m.
          8. Do NOT hardcode any numeric data — read everything from the JSON file.""")


def main():
    parser = argparse.ArgumentParser(description="Two-phase eval for template categories")
    parser.add_argument("--category", required=True, choices=["transportation", "jssp"])
    parser.add_argument("--binding-model", default=None,
        help="Path to trained binding model (required unless --ground-truth)")
    parser.add_argument("--ground-truth", action="store_true",
        help="Use ground-truth instance_data (upper bound experiment)")
    parser.add_argument("--phase2", choices=["template", "model"], default="template")
    parser.add_argument("--phase2-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    category = args.category

    # Defaults
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = PROJECT_ROOT / "synthetic_dataset" / "Template" / "basic_template" / category / "small"

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        phase1_tag = "gt" if args.ground_truth else "binder"
        out_path = PROJECT_ROOT / "results" / f"two_phase_{category}_{phase1_tag}_{args.phase2}.json"

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = PROJECT_ROOT / "synthetic_dataset" / f"_binding_cache_{category}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load problems
    problem_files = sorted(eval_dir.glob("*.json"))
    if args.max_problems > 0:
        problem_files = problem_files[:args.max_problems]

    problems = []
    for pf in problem_files:
        with open(pf) as f:
            problem = json.load(f)
        llm_desc = problem.get("LLM_description", "").strip()
        if not llm_desc:
            continue
        problems.append((pf.stem, problem, llm_desc))

    is_in_dist = SIZE_BOUNDARY[category]
    print(f"Category: {category}")
    print(f"Loaded {len(problems)} problems from {eval_dir}")

    n_in = sum(1 for _, p, _ in problems if is_in_dist(p))
    n_ood = len(problems) - n_in
    print(f"  In-distribution: {n_in}, OOD: {n_ood}")

    # ── Phase 1: Binding ──────────────────────────────────────────────
    if args.ground_truth:
        print("\n=== PHASE 1: Ground-truth instance_data ===")
        extracted_paths = []
        binding_responses = []
        binding_parse_ok = 0
        for problem_id, problem_data, _ in problems:
            gt = {k: problem_data["instance_data"][k] for k in INSTANCE_DATA_KEYS[category]}
            binding_responses.append(json.dumps(gt, indent=2))
            binding_parse_ok += 1
            ext_path = cache_dir / f"{problem_id}_extracted.json"
            with open(ext_path, "w") as f:
                json.dump(gt, f, indent=2)
            extracted_paths.append(ext_path)
        print(f"Ground-truth binding for {binding_parse_ok}/{len(problems)} problems.")
        import torch
    else:
        if args.binding_model is None:
            print("ERROR: --binding-model required when not using --ground-truth", file=sys.stderr)
            sys.exit(1)

        from vllm import LLM, SamplingParams

        print(f"\n=== PHASE 1: Binding ({args.binding_model}) ===")
        binding_llm = LLM(
            model=args.binding_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=True,
        )
        tokenizer = binding_llm.get_tokenizer()

        binding_prompts = []
        for _, _, llm_desc in problems:
            messages = [
                {"role": "system", "content": BINDING_SYSTEM_PROMPT},
                {"role": "user", "content": BINDING_INSTRUCTION_PREFIX + llm_desc},
            ]
            binding_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        binding_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)
        print(f"Generating {len(binding_prompts)} binding extractions...")
        binding_outputs = binding_llm.generate(binding_prompts, binding_params)
        print("Binding complete.")

        extracted_paths = []
        binding_responses = []
        binding_parse_ok = 0
        for i, output in enumerate(binding_outputs):
            problem_id = problems[i][0]
            response = output.outputs[0].text
            binding_responses.append(response)
            parsed = parse_json_from_response(response)

            if parsed is None:
                extracted_paths.append(None)
                print(f"  {problem_id}: PARSE FAIL")
                continue

            binding_parse_ok += 1
            ext_path = cache_dir / f"{problem_id}_extracted.json"
            with open(ext_path, "w") as f:
                json.dump(parsed, f, indent=2)
            extracted_paths.append(ext_path)

        print(f"Binding parse rate: {binding_parse_ok}/{len(problems)} ({binding_parse_ok/len(problems)*100:.1f}%)")

        del binding_llm
        import torch
        torch.cuda.empty_cache()

    # ── Phase 2: Solve ────────────────────────────────────────────────
    print(f"\n=== PHASE 2: Solve (mode={args.phase2}) ===")

    if args.phase2 == "template":
        solver_codes = []
        for ext_path in extracted_paths:
            if ext_path is None:
                solver_codes.append(None)
            else:
                code = build_template_code(str(ext_path), category)
                # Wrap in fenced block for evaluator compatibility
                solver_codes.append(f"```python\n{code}\n```")
        print(f"Built {sum(1 for c in solver_codes if c is not None)} template solver codes.")

    elif args.phase2 == "model":
        from vllm import LLM, SamplingParams

        print(f"Loading Phase 2 model: {args.phase2_model}")
        solver_llm = LLM(
            model=args.phase2_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=True,
        )
        solver_tokenizer = solver_llm.get_tokenizer()

        solve_indices = []
        solve_prompts = []
        for i, ext_path in enumerate(extracted_paths):
            if ext_path is None:
                continue
            solve_indices.append(i)
            messages = [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": build_solve_prompt(ext_path, category)},
            ]
            solve_prompts.append(
                solver_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        solve_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)
        print(f"Generating {len(solve_prompts)} solver codes...")
        solve_outputs = solver_llm.generate(solve_prompts, solve_params)
        print("Solver generation complete.")

        solver_codes = [None] * len(problems)
        for j, output in enumerate(solve_outputs):
            solver_codes[solve_indices[j]] = output.outputs[0].text

        del solver_llm
        torch.cuda.empty_cache()

    # ── Phase 3: Evaluate ─────────────────────────────────────────────
    print(f"\n=== PHASE 3: Evaluate (Gurobi execution) ===")
    evaluator = GeneratedSolutionEvaluator()

    results = []
    counts = {"total": 0, "binding_ok": 0, "exec_ok": 0, "optimal": 0, "correct": 0}
    counts_in = {"total": 0, "binding_ok": 0, "exec_ok": 0, "optimal": 0, "correct": 0}
    counts_ood = {"total": 0, "binding_ok": 0, "exec_ok": 0, "optimal": 0, "correct": 0}

    for i, (problem_id, problem_data, _) in enumerate(problems):
        code = solver_codes[i]
        in_dist = is_in_dist(problem_data)
        bucket = counts_in if in_dist else counts_ood

        counts["total"] += 1
        bucket["total"] += 1

        entry = {
            "problem_id": problem_id,
            "in_distribution": in_dist,
            "num_vars": problem_data["meta"]["num_vars"],
            "num_constraints": problem_data["meta"]["num_constraints"],
        }

        if code is None:
            entry["result"] = "binding_fail"
            dist_tag = "IN" if in_dist else "OOD"
            print(f"  [{i+1}/{len(problems)}] {problem_id} [{dist_tag}]: SKIP (binding failed)")
            results.append(entry)
            continue

        counts["binding_ok"] += 1
        bucket["binding_ok"] += 1

        eval_result = evaluator.evaluate_generated_code(problem_data, code, problem_id)

        entry["execution_succeeded"] = eval_result.execution_succeeded
        entry["status_optimal"] = eval_result.status_optimal
        entry["objective_matches"] = eval_result.objective_matches
        entry["objective_error"] = eval_result.objective_error
        entry["reference_optimum"] = eval_result.reference_optimum
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

        print(f"  [{i+1}/{len(problems)}] {problem_id} [{dist_tag}]: {status}{obj_info}")
        entry["result"] = status.lower()
        results.append(entry)

    # ── Summary ───────────────────────────────────────────────────────
    def _pct(v, t):
        return f"{v}/{t} ({v/t*100:.1f}%)" if t > 0 else "N/A"

    n = counts["total"]
    print(f"\n{'='*70}")
    print(f"TWO-PHASE EVALUATION: {category.upper()}")
    print(f"  Phase 1: {'ground-truth' if args.ground_truth else args.binding_model}")
    print(f"  Phase 2: {args.phase2}" + (f" ({args.phase2_model})" if args.phase2 == "model" else ""))
    print(f"{'='*70}")
    print(f"  {'Metric':20s} {'Overall':>18s} {'In-dist':>18s} {'OOD':>18s}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    for key in ["binding_ok", "exec_ok", "optimal", "correct"]:
        print(f"  {key:20s} {_pct(counts[key], n):>18s} {_pct(counts_in[key], counts_in['total']):>18s} {_pct(counts_ood[key], counts_ood['total']):>18s}")
    print(f"{'='*70}")

    # Save
    summary = {
        "category": category,
        "counts": counts,
        "counts_in_dist": counts_in,
        "counts_ood": counts_ood,
    }
    config = {
        "category": category,
        "binding_model": "ground-truth" if args.ground_truth else args.binding_model,
        "phase2": args.phase2,
        "phase2_model": args.phase2_model if args.phase2 == "model" else None,
        "eval_dir": str(eval_dir),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": config, "summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
