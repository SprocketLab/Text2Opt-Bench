#!/usr/bin/env python3
"""
Isomorphism check for passing solutions on resource allocation.

For each passing solution, extract the full Gurobi model structure
(objective, bounds, constraint matrix, RHS, senses) and compare
against the gold solution under a canonical (permutation-invariant) ordering.

This validates that objective-match evaluation does not produce
"correct for the wrong reasons" false positives.
"""

import json
import subprocess
import tempfile
import os
import re
from pathlib import Path
from collections import defaultdict

GRPO_PYTHON = "python3"
RESULTS_DIR = Path("results/resource_allocation")
EVAL_DIR = RESULTS_DIR / "eval"


def strip_markdown_fences(code: str) -> str:
    """Extract the first ```python ... ``` code block."""
    code = code.strip()
    match = re.search(r'```python\s*\n(.*?)(?:\n```)', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    code = re.sub(r'^```python\s*\n', '', code)
    code = re.sub(r'^```\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)
    return code.strip()


def get_canonical_form(code: str, timeout: int = 30):
    """
    Run code through Gurobi and extract a canonical model representation.

    Canonical ordering:
      1. Sort columns (variables) by (obj_coeff, lb, ub)
      2. Reorder constraint matrix columns accordingly
      3. Sort rows (constraints) by (sense, rhs, coefficient_vector)

    Returns a dict with canonical obj, bounds, rhs, senses, A matrix,
    or None on failure.
    """
    code = strip_markdown_fences(code)
    wrapper = code + """

import json, numpy as np, math

def normalize_val(v, decimals=4):
    \"\"\"Round and kill -0.0.\"\"\"
    return round(v, decimals) + 0.0

try:
    m = solve_problem()
    m.update()

    vrs = m.getVars()
    constrs = m.getConstrs()
    n_vars = len(vrs)
    n_constrs = len(constrs)

    obj = [normalize_val(v.Obj) for v in vrs]
    vtypes = [v.VType for v in vrs]

    # Normalize bounds: integer vars get ceil(lb)/floor(ub)
    bounds = []
    for v in vrs:
        lb = round(v.LB, 4) + 0.0
        ub = round(v.UB, 4) + 0.0
        if v.VType in ('I', 'B'):
            lb = float(math.ceil(lb))
            ub = float(math.floor(ub))
        bounds.append((lb, ub))

    rhs = [normalize_val(c.RHS) for c in constrs]
    senses = [c.Sense for c in constrs]
    A = m.getA().toarray()
    A = np.round(A, 4)
    A[np.abs(A) < 1e-6] = 0.0
    A = A + 0.0

    # ── Normalize constraints ──
    for i in range(n_constrs):
        # Convert >= to <= by negating
        if senses[i] == '>':
            A[i, :] = -A[i, :]
            rhs[i] = normalize_val(-rhs[i])
            senses[i] = '<'

        # Note: row scaling (dividing by first nonzero) was tested but
        # introduces rounding artifacts that increase false non-iso rate.

    # ── Step 1: canonical column order by (obj, lb, ub, vtype) ──
    col_key = [(obj[j], bounds[j][0], bounds[j][1], vtypes[j]) for j in range(n_vars)]
    col_perm = sorted(range(n_vars), key=lambda j: col_key[j])

    obj_sorted = [obj[j] for j in col_perm]
    bounds_sorted = [bounds[j] for j in col_perm]
    vtypes_sorted = [vtypes[j] for j in col_perm]
    A_col_sorted = A[:, col_perm]
    # Re-round after reorder
    A_col_sorted = np.round(A_col_sorted, 4) + 0.0

    # ── Step 2: canonical row order by (sense, rhs, row_values) ──
    row_key = []
    for i in range(n_constrs):
        row_tuple = tuple(normalize_val(v) for v in A_col_sorted[i].tolist())
        row_key.append(('<', normalize_val(rhs[i])) + row_tuple)
    row_perm = sorted(range(n_constrs), key=lambda i: row_key[i])

    rhs_sorted = [rhs[i] for i in row_perm]
    senses_sorted = [senses[i] for i in row_perm]
    A_canonical = A_col_sorted[row_perm, :].tolist()

    data = {
        "obj": obj_sorted,
        "bounds": bounds_sorted,
        "rhs": rhs_sorted,
        "senses": senses_sorted,
        "A": A_canonical,
    }
    print("__MODEL__" + json.dumps(data))
except Exception as e:
    print(f"__ERROR__:{e}")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp = f.name
    try:
        result = subprocess.run(
            [GRPO_PYTHON, tmp],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GRB_LICENSE_FILE": os.environ.get("GRB_LICENSE_FILE", "")}
        )
        for line in (result.stdout + result.stderr).split("\n"):
            if line.startswith("__MODEL__"):
                return json.loads(line[9:])
        return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(tmp)


def main():
    # ── Load previous results to only re-check non-iso cases ──────────────
    prev_path = RESULTS_DIR / "isomorphism_detail.json"
    prev_detail = {}
    if prev_path.exists():
        prev_detail = json.load(open(prev_path))
        print(f"Loaded {sum(len(v) for v in prev_detail.values())} previous results from {prev_path}")

    # Cache gold canonical forms (shared across models)
    gold_cache = {}

    model_files = sorted(EVAL_DIR.glob("*.json"))
    all_results = {}
    all_detail = {}  # model -> {pid: "iso" | "non_iso" | "error"}

    for model_path in model_files:
        model_name = model_path.stem
        data = json.load(open(model_path))
        passed = [r for r in data if r["evaluation_result"].get("objective_matches", False)]

        prev_model = prev_detail.get(model_name, {})
        model_detail = {}

        checked = 0
        isomorphic = 0
        non_iso = 0
        errors = 0

        for r in passed:
            pid = r["problem_id"]

            # Skip if previously confirmed isomorphic (no need to re-check)
            if prev_model.get(pid) == "iso":
                model_detail[pid] = "iso"
                checked += 1
                isomorphic += 1
                continue

            # Need to (re-)check: either new, previously non-iso, or error
            if pid not in gold_cache:
                gold_code = r.get("gold_solution", "")
                gold_cache[pid] = get_canonical_form(gold_code) if gold_code else None

            gold = gold_cache[pid]
            if gold is None:
                errors += 1
                model_detail[pid] = "error"
                continue

            gen = get_canonical_form(r.get("generated_code", ""))
            if gen is None:
                errors += 1
                model_detail[pid] = "error"
                continue

            checked += 1
            if gold == gen:
                isomorphic += 1
                model_detail[pid] = "iso"
            else:
                non_iso += 1
                model_detail[pid] = "non_iso"
                if non_iso <= 3:
                    print(f"  NON-ISO {model_name}/{pid}")

        all_detail[model_name] = model_detail

        pct = isomorphic / checked * 100 if checked > 0 else 0
        print(f"{model_name:>25}: {len(passed)} passed, {checked} checked, "
              f"{isomorphic} iso ({pct:.1f}%), {non_iso} non-iso, {errors} errors")
        all_results[model_name] = {
            "passed": len(passed),
            "checked": checked,
            "isomorphic": isomorphic,
            "non_isomorphic": non_iso,
            "errors": errors,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Model':>25} {'Passed':>8} {'Checked':>8} {'Iso':>6} {'Iso%':>7} {'Non-Iso':>8}")
    print(f"{'-'*80}")
    total_checked = 0
    total_iso = 0
    for model_name, r in sorted(all_results.items()):
        pct = r["isomorphic"] / r["checked"] * 100 if r["checked"] > 0 else 0
        print(f"{model_name:>25} {r['passed']:>8} {r['checked']:>8} "
              f"{r['isomorphic']:>6} {pct:>6.1f}% {r['non_isomorphic']:>8}")
        total_checked += r["checked"]
        total_iso += r["isomorphic"]
    total_non = total_checked - total_iso
    print(f"{'TOTAL':>25} {'':>8} {total_checked:>8} {total_iso:>6} "
          f"{total_iso/total_checked*100:>6.1f}% {total_non:>8}")

    # Save both summary and detail
    out_path = RESULTS_DIR / "isomorphism_analysis.json"
    json.dump(all_results, open(out_path, "w"), indent=2)
    json.dump(all_detail, open(prev_path, "w"), indent=2)
    print(f"\nSaved summary to {out_path}")
    print(f"Saved detail to {prev_path}")


if __name__ == "__main__":
    main()
