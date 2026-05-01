#!/usr/bin/env python3
"""
Evaluate a trained binding model on the eval_2_11 split using vLLM.

Measures:
  1. JSON parse rate (did the model output valid JSON?)
  2. Structural accuracy (correct # vars, # constraints, goal)
  3. Coefficient-level accuracy (exact match within tolerance)
  4. Full-problem accuracy (all fields correct)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Read the optimization problem description and extract all decision variables, "
    "constraints, and objective function parameters into structured JSON. "
    "Return ONLY a fenced ```json ... ``` block."
)

INSTRUCTION_PREFIX = (
    "Extract all optimization parameters from the following problem description "
    "into structured JSON.\n\n"
)

# Relative tolerance for coefficient comparison
COEFF_RTOL = 1e-4
COEFF_ATOL = 1e-6


def parse_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from a fenced ```json ... ``` block."""
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Fallback: try raw JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def coefficients_match(a: float, b: float) -> bool:
    if a == b:
        return True
    if abs(b) > COEFF_ATOL:
        return abs(a - b) / abs(b) < COEFF_RTOL
    return abs(a - b) < COEFF_ATOL


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


def _constraint_errors(pc: dict, gc: dict) -> list[str]:
    """Return list of error strings comparing a predicted constraint to a gold one."""
    errors = []
    if pc.get("sense", "") != gc["sense"]:
        errors.append(f"sense: pred={pc.get('sense')} vs gold={gc['sense']}")
    prhs = float(pc.get("rhs", 0.0))
    grhs = float(gc["rhs"])
    if not coefficients_match(prhs, grhs):
        errors.append(f"rhs: pred={prhs} vs gold={grhs}")
    pred_coeffs = pc.get("coefficients", {})
    gold_coeffs = gc["coefficients"]
    if set(pred_coeffs.keys()) != set(gold_coeffs.keys()):
        errors.append(f"coeff_keys: pred={set(pred_coeffs.keys())} vs gold={set(gold_coeffs.keys())}")
    else:
        for var_name in gold_coeffs:
            pval = float(pred_coeffs.get(var_name, 0.0))
            gval = float(gold_coeffs[var_name])
            if not coefficients_match(pval, gval):
                errors.append(f"coeffs[{var_name}]: pred={pval} vs gold={gval}")
    return errors


def _find_best_constraint_match(
    gc: dict, pred_constrs: list[dict], used: set
) -> Optional[tuple[int, list[str]]]:
    """Find the predicted constraint that best matches a gold constraint.
    Returns (pred_index, errors) or None if no candidate."""
    best_idx = None
    best_errors: list[str] = []
    best_score = -1  # higher is better

    for pi, pc in enumerate(pred_constrs):
        if pi in used:
            continue
        errs = _constraint_errors(pc, gc)
        # Score: fewer errors is better; 0 errors = perfect match
        score = -len(errs)
        if best_idx is None or score > best_score:
            best_idx = pi
            best_errors = errs
            best_score = score
            if score == 0:
                break  # perfect match, stop searching

    if best_idx is None:
        return None
    return (best_idx, best_errors)


def evaluate_extraction(pred: dict, gold: dict) -> dict:
    """Compare predicted extraction against ground truth."""
    result = {
        "goal_correct": False,
        "num_vars_correct": False,
        "num_constraints_correct": False,
        "variables_correct": False,
        "constraints_correct": False,
        "full_correct": False,
        "errors": [],
    }

    result["goal_correct"] = pred.get("goal", "").upper() == gold["goal"]
    if not result["goal_correct"]:
        result["errors"].append(f"goal: pred={pred.get('goal')} vs gold={gold['goal']}")

    pred_vars = pred.get("variables", [])
    gold_vars = gold["variables"]
    pred_constrs = pred.get("constraints", [])
    gold_constrs = gold["constraints"]

    result["num_vars_correct"] = len(pred_vars) == len(gold_vars)
    result["num_constraints_correct"] = len(pred_constrs) == len(gold_constrs)

    if not result["num_vars_correct"]:
        result["errors"].append(f"num_vars: pred={len(pred_vars)} vs gold={len(gold_vars)}")
    if not result["num_constraints_correct"]:
        result["errors"].append(f"num_constrs: pred={len(pred_constrs)} vs gold={len(gold_constrs)}")

    # Variable-level comparison
    vars_all_ok = result["num_vars_correct"]
    if vars_all_ok:
        for i, (pv, gv) in enumerate(zip(pred_vars, gold_vars)):
            for field in ["lb", "objective_coefficient"]:
                pval = float(pv.get(field, 0.0))
                gval = float(gv.get(field, 0.0))
                if not coefficients_match(pval, gval):
                    vars_all_ok = False
                    result["errors"].append(f"var[{i}].{field}: pred={pval} vs gold={gval}")
            pub = pv.get("ub")
            gub = gv.get("ub")
            if pub is None and gub is None:
                pass
            elif pub is not None and gub is not None:
                if not coefficients_match(float(pub), float(gub)):
                    vars_all_ok = False
                    result["errors"].append(f"var[{i}].ub: pred={pub} vs gold={gub}")
            else:
                vars_all_ok = False
                result["errors"].append(f"var[{i}].ub: pred={pub} vs gold={gub}")
    result["variables_correct"] = vars_all_ok

    # Constraint-level comparison (order-independent: match by content)
    constrs_all_ok = result["num_constraints_correct"]
    if constrs_all_ok:
        used_pred = set()
        for gi, gc in enumerate(gold_constrs):
            best_match = _find_best_constraint_match(gc, pred_constrs, used_pred)
            if best_match is None:
                constrs_all_ok = False
                result["errors"].append(f"gold_constr[{gi}] ({gc.get('name','')}): no matching pred constraint")
                continue
            pi, match_errors = best_match
            used_pred.add(pi)
            if match_errors:
                constrs_all_ok = False
                for err in match_errors:
                    result["errors"].append(f"constr[g{gi}->p{pi}].{err}")
    result["constraints_correct"] = constrs_all_ok

    result["full_correct"] = all([
        result["goal_correct"],
        result["variables_correct"],
        result["constraints_correct"],
    ])

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate binding model with vLLM")
    parser.add_argument(
        "--model-path",
        default="models/binding",
    )
    parser.add_argument(
        "--eval-dir",
        default="synthetic_dataset/Unstructured/resource_allocation/eval",
    )
    parser.add_argument(
        "--output-path",
        default="results/binding_eval_results.json",
    )
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    eval_dir = Path(args.eval_dir)
    problem_files = sorted(eval_dir.glob("*.json"))
    if args.max_problems > 0:
        problem_files = problem_files[:args.max_problems]

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

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4096,
    )

    # Batch generate all at once
    print(f"Generating {len(chat_prompts)} responses...")
    outputs = llm.generate(chat_prompts, sampling_params)
    print("Generation complete.")

    # Evaluate
    results = []
    counts = {
        "total": 0,
        "parse_ok": 0,
        "goal_correct": 0,
        "num_vars_correct": 0,
        "num_constraints_correct": 0,
        "variables_correct": 0,
        "constraints_correct": 0,
        "full_correct": 0,
    }

    for i, output in enumerate(outputs):
        problem_id, problem = problems[i]
        response = output.outputs[0].text
        gold = build_ground_truth(problem)

        counts["total"] += 1
        entry = {
            "problem_id": problem_id,
            "num_vars": problem["meta"]["num_vars"],
            "num_constraints": problem["meta"]["num_constraints"],
            "model_response": response,
        }

        pred = parse_json_from_response(response)
        if pred is None:
            entry["parse_ok"] = False
            entry["eval"] = None
            entry["errors"] = ["Failed to parse JSON from response"]
            print(f"  [{i+1}/{len(outputs)}] {problem_id}: PARSE FAIL")
        else:
            counts["parse_ok"] += 1
            entry["parse_ok"] = True
            eval_result = evaluate_extraction(pred, gold)
            entry["eval"] = eval_result

            for key in ["goal_correct", "num_vars_correct", "num_constraints_correct",
                        "variables_correct", "constraints_correct", "full_correct"]:
                if eval_result[key]:
                    counts[key] += 1

            status = "PASS" if eval_result["full_correct"] else "FAIL"
            err_summary = f" ({len(eval_result['errors'])} errors)" if eval_result["errors"] else ""
            print(f"  [{i+1}/{len(outputs)}] {problem_id} (vars={problem['meta']['num_vars']}): {status}{err_summary}")

        results.append(entry)

    # Summary
    n = counts["total"]
    print(f"\n{'='*60}")
    print(f"BINDING EVALUATION RESULTS ({n} problems)")
    print(f"{'='*60}")
    for key, val in counts.items():
        if key == "total":
            continue
        pct = val / n * 100 if n > 0 else 0
        print(f"  {key:25s}: {val:4d}/{n} ({pct:5.1f}%)")
    print(f"{'='*60}")

    # Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": counts, "results": results}, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
