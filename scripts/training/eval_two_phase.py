#!/usr/bin/env python3
"""
Two-phase evaluation: trained binding model (Phase 1) → solver (Phase 2) → gurobi eval.

Phase 1: Binding model extracts structured JSON from problem description (vLLM).
Phase 2: Either (a) deterministic template code or (b) an LLM generates solver code from the JSON.
Eval:    Execute generated code, compare objective to ground truth.

Usage:
    # Template-only (no Phase 2 LLM, isolates binding accuracy):
    python eval_two_phase.py --phase2 template

    # Two-model pipeline (binding model + untrained Qwen for Phase 2):
    python eval_two_phase.py --phase2 model --phase2-model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main.evaluation.problem_evaluator import GeneratedSolutionEvaluator


BINDING_SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Read the optimization problem description and extract all decision variables, "
    "constraints, and objective function parameters into structured JSON. "
    "Return ONLY a fenced ```json ... ``` block."
)

BINDING_INSTRUCTION_PREFIX = (
    "Extract all optimization parameters from the following problem description "
    "into structured JSON.\n\n"
)

SOLVER_SYSTEM_PROMPT = (
    "You are an optimization coding agent. "
    "Return only executable Python code in a single fenced ```python ... ``` block."
)


def build_ground_truth(problem: dict) -> dict:
    """Build the ground-truth extraction JSON from a problem file."""
    meta = problem.get("meta", {})
    variables_raw = problem.get("variables", {})
    constraints_raw = problem.get("constraints", {})

    variables = []
    for var_name, var_data in variables_raw.items():
        bounds = var_data.get("range", [0.0, None])
        lb = float(bounds[0]) if bounds[0] is not None else 0.0
        ub = float(bounds[1]) if len(bounds) > 1 and bounds[1] is not None else None
        variables.append({
            "name": var_name,
            "type": str(var_data.get("type", "Continuous")),
            "lb": lb,
            "ub": ub,
            "objective_coefficient": float(var_data.get("objective_linear_coefficient", 0.0)),
        })

    constraint_coefficients: dict[str, dict[str, float]] = {}
    for var_name, var_data in variables_raw.items():
        for constr_name, coeff in (var_data.get("resource_costs") or {}).items():
            if constr_name not in constraint_coefficients:
                constraint_coefficients[constr_name] = {}
            constraint_coefficients[constr_name][var_name] = float(coeff)

    constraints = []
    for constr_name, constr_data in constraints_raw.items():
        constraints.append({
            "name": constr_name,
            "sense": str(constr_data.get("sense", "==")),
            "rhs": float(constr_data.get("rhs", 0.0)),
            "coefficients": constraint_coefficients.get(constr_name, {}),
        })

    return {
        "goal": str(meta.get("goal", "MINIMIZE")).upper(),
        "variables": variables,
        "constraints": constraints,
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


def build_template_code(extracted_path: Path) -> str:
    """Build deterministic solver code from the template (no LLM needed)."""
    return textwrap.dedent(f'''\
        ```python
        import json
        from gurobipy import *

        EXTRACTED_DATA_PATH = r"{extracted_path}"

        def solve_problem():
            with open(EXTRACTED_DATA_PATH, "r") as f:
                d = json.load(f)

            m = Model("problem")
            m.setParam("TimeLimit", 100)

            type_map = {{"Continuous": GRB.CONTINUOUS, "Integer": GRB.INTEGER, "Binary": GRB.BINARY}}

            vars_map = {{}}
            for v in d["variables"]:
                ub = v["ub"] if v["ub"] is not None else GRB.INFINITY
                vars_map[v["name"]] = m.addVar(
                    lb=v["lb"], ub=ub,
                    vtype=type_map.get(v["type"], GRB.CONTINUOUS),
                    name=v["name"]
                )
            m.update()

            obj = quicksum(v["objective_coefficient"] * vars_map[v["name"]] for v in d["variables"])
            sense = GRB.MINIMIZE if d["goal"] == "MINIMIZE" else GRB.MAXIMIZE
            m.setObjective(obj, sense)

            for c in d["constraints"]:
                expr = quicksum(coeff * vars_map[var] for var, coeff in c["coefficients"].items())
                if c["sense"] == "<=":
                    m.addConstr(expr <= c["rhs"], name=c.get("name", ""))
                elif c["sense"] == ">=":
                    m.addConstr(expr >= c["rhs"], name=c.get("name", ""))
                else:
                    m.addConstr(expr == c["rhs"], name=c.get("name", ""))

            m.optimize()
            return m
        ```''')


def build_solve_prompt(extracted_path: Path) -> str:
    """Build the Phase 2 prompt for an LLM solver."""
    schema_note = textwrap.dedent("""\
        Extracted data file schema (JSON):
          goal        : str — "MINIMIZE" or "MAXIMIZE"
          variables   : list of objects, each with:
                          name, type (Continuous/Integer/Binary), lb, ub, objective_coefficient
          constraints : list of objects, each with:
                          name, sense (<=/>=/==), rhs, coefficients (dict: var_name → number)""")

    requirements = textwrap.dedent("""\
        Requirements:
          1. Return exactly one fenced ```python ... ``` block.
          2. ONLY import `json` and `from gurobipy import *`. No other imports. No try/except.
          3. gurobipy IS available — do NOT create mock or fallback implementations.
          4. Define `solve_problem()` with no required arguments.
          5. Inside `solve_problem()`, open and parse EXTRACTED_DATA_PATH with json.load().
          6. Build Gurobi variables from the "variables" list (name, type, lb, ub).
             Use GRB.INFINITY when ub is null.
          7. Build the objective from each variable's objective_coefficient and the goal.
          8. Build all constraints from the "constraints" list using coefficients, sense, rhs.
          9. Set `m.setParam("TimeLimit", 100)`.
         10. Call `m.optimize()` and return `m`.
         11. Do NOT hardcode any numeric coefficients, bounds, or RHS values.""")

    template = build_template_code(extracted_path)

    return textwrap.dedent(f"""\
        Write Python code that solves an optimization model by loading locally extracted data.

        EXTRACTED_DATA_PATH = r"{extracted_path}"

        {schema_note}

        {requirements}

        Use this template as your starting point:
        {template}""")


def main():
    parser = argparse.ArgumentParser(description="Two-phase binding → solve evaluation")
    parser.add_argument("--binding-model",
        default="models/binding")
    parser.add_argument("--ground-truth", action="store_true",
        help="Skip Phase 1 and use ground-truth binding (upper bound experiment)")
    parser.add_argument("--phase2", choices=["template", "model"], default="template",
        help="Phase 2 mode: 'template' (deterministic code) or 'model' (LLM generates code)")
    parser.add_argument("--phase2-model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model for Phase 2 when --phase2=model")
    parser.add_argument("--eval-dir",
        default=str(PROJECT_ROOT / "synthetic_dataset/Unstructured/resource_allocation/eval"))
    parser.add_argument("--output-path",
        default=str(PROJECT_ROOT / "results/two_phase_eval_results.json"))
    parser.add_argument("--cache-dir",
        default=str(PROJECT_ROOT / "synthetic_dataset/_binding_cache"))
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    eval_dir = Path(args.eval_dir)
    cache_dir = Path(args.cache_dir)
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

    print(f"Loaded {len(problems)} problems from {eval_dir}")

    # ── Phase 1: Binding ──────────────────────────────────────────────
    if args.ground_truth:
        print("\n=== PHASE 1: Using ground-truth binding (oracle upper bound) ===")
        extracted_paths = []
        binding_responses = []
        binding_parse_ok = 0
        for problem_id, problem_data, _ in problems:
            gt = build_ground_truth(problem_data)
            binding_responses.append(json.dumps(gt, indent=2))
            binding_parse_ok += 1
            out_path = cache_dir / f"{problem_id}_extracted.json"
            with open(out_path, "w") as f:
                json.dump(gt, f, indent=2)
            extracted_paths.append(out_path)
        print(f"Ground-truth binding for {binding_parse_ok}/{len(problems)} problems.")
        import torch
    else:
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

        # Parse and save extractions
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
            out_path = cache_dir / f"{problem_id}_extracted.json"
            with open(out_path, "w") as f:
                json.dump(parsed, f, indent=2)
            extracted_paths.append(out_path)

        print(f"Binding parse rate: {binding_parse_ok}/{len(problems)} ({binding_parse_ok/len(problems)*100:.1f}%)")

        # Free binding model GPU memory
        del binding_llm
        import torch
        torch.cuda.empty_cache()

    # ── Phase 2: Solve ────────────────────────────────────────────────
    print(f"\n=== PHASE 2: Solve (mode={args.phase2}) ===")

    if args.phase2 == "template":
        # Deterministic template — no LLM needed
        solver_codes = []
        for i, ext_path in enumerate(extracted_paths):
            if ext_path is None:
                solver_codes.append(None)
            else:
                solver_codes.append(build_template_code(ext_path))
        print(f"Built {sum(1 for c in solver_codes if c is not None)} template solver codes.")

    elif args.phase2 == "model":
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

        # Build Phase 2 prompts only for problems with successful binding
        solve_indices = []
        solve_prompts = []
        for i, ext_path in enumerate(extracted_paths):
            if ext_path is None:
                continue
            solve_indices.append(i)
            messages = [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": build_solve_prompt(ext_path)},
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
            idx = solve_indices[j]
            solver_codes[idx] = output.outputs[0].text

        del solver_llm
        torch.cuda.empty_cache()

    # ── Phase 3: Evaluate ─────────────────────────────────────────────
    print(f"\n=== PHASE 3: Evaluate (gurobi execution) ===")
    evaluator = GeneratedSolutionEvaluator()

    results = []
    counts = {
        "total": len(problems),
        "binding_parse_ok": binding_parse_ok,
        "execution_succeeded": 0,
        "status_optimal": 0,
        "objective_matches": 0,
    }

    for i, (problem_id, problem_data, _) in enumerate(problems):
        code = solver_codes[i]
        entry = {
            "problem_id": problem_id,
            "num_vars": problem_data["meta"]["num_vars"],
            "num_constraints": problem_data["meta"]["num_constraints"],
        }

        entry["binding_response"] = binding_responses[i]
        entry["solver_code"] = code

        if code is None:
            entry["result"] = "binding_fail"
            print(f"  [{i+1}/{len(problems)}] {problem_id}: SKIP (binding failed)")
            results.append(entry)
            continue

        eval_result = evaluator.evaluate_generated_code(problem_data, code, problem_id)

        entry["execution_succeeded"] = eval_result.execution_succeeded
        entry["status_optimal"] = eval_result.status_optimal
        entry["objective_matches"] = eval_result.objective_matches
        entry["objective_error"] = eval_result.objective_error
        entry["reference_optimum"] = eval_result.reference_optimum
        if eval_result.generated_solution.objective_value is not None:
            entry["predicted_optimum"] = eval_result.generated_solution.objective_value

        if eval_result.execution_succeeded:
            counts["execution_succeeded"] += 1
        if eval_result.status_optimal:
            counts["status_optimal"] += 1
        if eval_result.objective_matches:
            counts["objective_matches"] += 1

        status = "PASS" if eval_result.objective_matches else "FAIL"
        obj_info = ""
        if eval_result.objective_error is not None:
            obj_info = f" (err={eval_result.objective_error:.4f})"
        elif not eval_result.execution_succeeded:
            stderr = eval_result.generated_solution.stderr[-100:] if eval_result.generated_solution.stderr else ""
            obj_info = f" (exec fail: {stderr})"

        print(f"  [{i+1}/{len(problems)}] {problem_id} (vars={problem_data['meta']['num_vars']}): {status}{obj_info}")

        entry["result"] = status.lower()
        results.append(entry)

    # ── Summary ───────────────────────────────────────────────────────
    n = counts["total"]
    print(f"\n{'='*60}")
    print(f"TWO-PHASE EVALUATION RESULTS ({n} problems)")
    print(f"  Phase 1: {args.binding_model}")
    print(f"  Phase 2: {args.phase2}" + (f" ({args.phase2_model})" if args.phase2 == "model" else ""))
    print(f"{'='*60}")
    for key, val in counts.items():
        if key == "total":
            print(f"  {'total':30s}: {val}")
        else:
            pct = val / n * 100 if n > 0 else 0
            print(f"  {key:30s}: {val:4d}/{n} ({pct:5.1f}%)")
    print(f"{'='*60}")

    # Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "binding_model": args.binding_model,
                "phase2": args.phase2,
                "phase2_model": args.phase2_model if args.phase2 == "model" else None,
                "eval_dir": args.eval_dir,
            },
            "summary": counts,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
