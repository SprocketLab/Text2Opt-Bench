#!/usr/bin/env python3
"""
Generate binding-only SFT training data for LLaMA-Factory.

Reads resource_allocation problems and produces (instruction, output) pairs where:
  - instruction = LLM_description (natural language problem)
  - output = structured JSON with goal, variables, constraints

This trains a model to do ONLY binding (text -> structured JSON),
not modeling or code generation.
"""

import json
import sys
from pathlib import Path


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


def problem_to_binding_target(problem: dict) -> dict:
    """
    Convert a resource_allocation problem's ground-truth variables/constraints
    into the extraction JSON schema (matching agentic_offload.py's format).
    """
    meta = problem.get("meta", {})
    variables_raw = problem.get("variables", {})
    constraints_raw = problem.get("constraints", {})

    # Build variables list
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

    # Build constraints list
    # Need to invert the variable->constraint mapping to constraint->variable
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


def format_binding_output(target: dict) -> str:
    """Wrap the target JSON in a fenced code block (matching extraction prompt format)."""
    return "```json\n" + json.dumps(target, indent=2) + "\n```"


def generate_dataset(problem_dir: Path) -> list[dict]:
    """Generate LLaMA-Factory format dataset from a directory of problem JSONs."""
    dataset = []

    problem_files = sorted(problem_dir.glob("*.json"))
    for pf in problem_files:
        with open(pf) as f:
            problem = json.load(f)

        llm_desc = problem.get("LLM_description", "").strip()
        if not llm_desc:
            print(f"  Skipping {pf.name}: no LLM_description", file=sys.stderr)
            continue

        target = problem_to_binding_target(problem)
        output = format_binding_output(target)

        dataset.append({
            "instruction": INSTRUCTION_PREFIX + llm_desc,
            "input": "",
            "output": output,
        })

    return dataset


def main():
    bench_root = Path(__file__).resolve().parent.parent.parent
    ra_dir = bench_root / "synthetic_dataset" / "Unstructured" / "resource_allocation"

    # Generate train and eval sets
    for split in ["train_2_11", "eval_2_11"]:
        split_dir = ra_dir / split
        if not split_dir.exists():
            print(f"Skipping {split}: directory not found", file=sys.stderr)
            continue

        dataset = generate_dataset(split_dir)
        print(f"{split}: {len(dataset)} examples")

        # Write to LLaMA-Factory data directory
        out_path = Path("data/training") / f"binding_{split}.json"
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"  -> {out_path}")

    # Also write a combined version for convenience
    train_path = Path("data/training/binding_train_2_11.json")
    if train_path.exists():
        print(f"\nTrain data ready at: {train_path}")


if __name__ == "__main__":
    main()
