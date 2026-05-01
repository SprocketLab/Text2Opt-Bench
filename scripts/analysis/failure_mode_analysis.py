#!/usr/bin/env python3
"""
Failure mode analysis for resource allocation problems.

For each failed solution, extract #vars and #constraints from the generated code
by running it through Gurobi, and compare against gold solution metadata.

Classification:
  - Execution error: code doesn't run at all
  - Modeling error: code runs but wrong #vars or #constraints
  - Binding error: correct structure (#vars, #constraints) but wrong objective
"""

import json
import os
import sys
import subprocess
import tempfile
import re
from pathlib import Path
from collections import defaultdict

GRPO_PYTHON = "python3"
RESULTS_DIR = Path("results/resource_allocation")
EVAL_DIR = RESULTS_DIR / "eval"
PASS1_DIR = RESULTS_DIR / "pass1"

# Models in eval/ (already subset to 248 problems)
EVAL_MODELS = {
    "claude-opus-4-6": EVAL_DIR / "claude-opus-4-6.json",
    "claude-sonnet-4-6": EVAL_DIR / "claude-sonnet-4-6.json",
    "DeepSeek-R1": EVAL_DIR / "DeepSeek-R1.json",
    "DeepSeek-V3.2": EVAL_DIR / "DeepSeek-V3.2.json",
    "Llama-3.3-70B": EVAL_DIR / "Llama-3.3-70B-Instruct.json",
    "o4-mini": EVAL_DIR / "o4-mini.json",
}

# Models in pass1/ (full 1012, need to filter to eval subset)
PASS1_MODELS = {
    "gpt-5": PASS1_DIR / "gpt-5-resource-allocation.json",
    "gpt-5-nano": PASS1_DIR / "gpt-5-nano-resource-allocation.json",
    "Qwen2.5-7B": PASS1_DIR / "resource_allocation_Qwen2.5-7B-Instruct.json",
}


def get_eval_problem_ids():
    """Get the 248 problem IDs from an eval file."""
    ref = json.load(open(next(EVAL_DIR.iterdir())))
    return set(r["problem_id"] for r in ref)


def strip_markdown_fences(code: str) -> str:
    """Extract the first ```python ... ``` code block, handling:
    - Prose/reasoning before the code block (Sonnet)
    - Degenerate repetition with ``` ```python separators (Qwen)
    - Leading/trailing fences
    """
    code = code.strip()
    # Try to extract the first fenced code block
    match = re.search(r'```python\s*\n(.*?)(?:\n```)', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip leading/trailing fences
    code = re.sub(r'^```python\s*\n', '', code)
    code = re.sub(r'^```\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)
    return code.strip()


def extract_model_stats(generated_code: str, timeout: int = 30):
    """Run generated code in subprocess, extract NumVars and NumConstrs."""
    generated_code = strip_markdown_fences(generated_code)
    # Wrap the code to print model stats after optimize
    wrapper = generated_code + "\n\n"
    wrapper += """
try:
    m = solve_problem()
    print(f"__STATS__:{m.NumVars}:{m.NumConstrs}:{m.ObjVal if m.Status == 2 else 'NA'}:{m.Status}")
except Exception as e:
    print(f"__ERROR__:{e}")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [GRPO_PYTHON, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GRB_LICENSE_FILE": os.environ.get("GRB_LICENSE_FILE", "")}
        )
        output = result.stdout + result.stderr
        # Parse stats
        for line in output.split("\n"):
            if line.startswith("__STATS__:"):
                parts = line.split(":")
                return {
                    "num_vars": int(parts[1]),
                    "num_constraints": int(parts[2]),
                    "obj_val": float(parts[3]) if parts[3] != "NA" else None,
                    "status": int(parts[4]),
                    "error": None,
                }
            elif line.startswith("__ERROR__:"):
                return {"error": line[10:], "num_vars": None, "num_constraints": None, "obj_val": None, "status": None}
        return {"error": f"no stats output; stderr={result.stderr[:200]}", "num_vars": None, "num_constraints": None, "obj_val": None, "status": None}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "num_vars": None, "num_constraints": None, "obj_val": None, "status": None}
    finally:
        os.unlink(tmp_path)


def classify_failure(gold_stats, gen_stats, eval_result):
    """Classify a failure into error type.

    gold_stats: extracted from running the gold solution code (not metadata).
    """
    if not eval_result.get("execution_succeeded", False):
        return "execution_error"

    if gen_stats.get("error"):
        return "execution_error"

    gold_vars = gold_stats["num_vars"]
    gold_constrs = gold_stats["num_constraints"]
    gen_vars = gen_stats.get("num_vars")
    gen_constrs = gen_stats.get("num_constraints")

    if gen_vars is None:
        return "execution_error"

    vars_match = (gen_vars == gold_vars)
    constrs_match = (gen_constrs == gold_constrs)

    if vars_match and constrs_match:
        return "binding_error"  # right structure, wrong values
    else:
        return "modeling_error"  # wrong structure


# Cache for gold solution stats keyed by problem_id
_gold_stats_cache = {}


def get_gold_stats(problem_id, gold_code):
    """Extract stats from gold solution code, with caching."""
    if problem_id not in _gold_stats_cache:
        _gold_stats_cache[problem_id] = extract_model_stats(gold_code)
    return _gold_stats_cache[problem_id]


def analyze_model(model_name, results, eval_ids):
    """Analyze failures for one model."""
    # Filter to eval subset if needed
    if eval_ids:
        results = [r for r in results if r["problem_id"] in eval_ids]

    total = len(results)
    passed = 0
    failures = defaultdict(int)
    failure_details = []
    gold_meta_mismatches = 0

    for r in results:
        eval_res = r["evaluation_result"]
        succeeded = eval_res.get("objective_matches", False)

        if succeeded:
            passed += 1
            continue

        # For failures, extract stats from generated code
        gen_code = r.get("generated_code", "")
        if not gen_code:
            failures["no_code"] += 1
            continue

        # Extract gold stats from actual gold solution code
        gold_code = r.get("gold_solution", "")
        if not gold_code:
            failures["no_gold"] += 1
            continue
        gold_stats = get_gold_stats(r["problem_id"], gold_code)
        if gold_stats.get("error"):
            failures["gold_error"] += 1
            continue

        # Check if metadata matches gold code
        if (gold_stats["num_vars"] != r["meta"]["num_vars"] or
                gold_stats["num_constraints"] != r["meta"]["num_constraints"]):
            gold_meta_mismatches += 1

        gen_stats = extract_model_stats(gen_code)
        error_type = classify_failure(gold_stats, gen_stats, eval_res)
        failures[error_type] += 1

        # Infer category from problem_id
        pid = r["problem_id"]
        category = r["meta"].get("problem_type", "unknown")
        if category == "unknown":
            # Try to infer from problem_id prefix
            category = pid.rsplit("_", 1)[0] if "_" in pid else "unknown"

        failure_details.append({
            "problem_id": pid,
            "category": category,
            "error_type": error_type,
            "gold_vars": gold_stats["num_vars"],
            "gold_constrs": gold_stats["num_constraints"],
            "meta_vars": r["meta"]["num_vars"],
            "meta_constrs": r["meta"]["num_constraints"],
            "gen_vars": gen_stats.get("num_vars"),
            "gen_constrs": gen_stats.get("num_constraints"),
            "gen_obj": gen_stats.get("obj_val"),
            "gold_obj": r["gurobi_result"]["theoretical_optimum"],
            "gen_error": gen_stats.get("error"),
        })

    if gold_meta_mismatches > 0:
        print(f"  WARNING: {gold_meta_mismatches} problems where gold code stats != metadata")

    return {
        "model": model_name,
        "total": total,
        "passed": passed,
        "pass_rate": f"{passed/total*100:.1f}%",
        "failures": dict(failures),
        "failure_details": failure_details,
    }


def print_summary(all_results):
    """Print summary table."""
    print("\n" + "=" * 100)
    print(f"{'Model':<20} {'Total':>6} {'Pass':>6} {'Rate':>7} {'Exec Err':>10} {'Model Err':>10} {'Bind Err':>10}")
    print("-" * 100)
    for r in all_results:
        f = r["failures"]
        print(f"{r['model']:<20} {r['total']:>6} {r['passed']:>6} {r['pass_rate']:>7} "
              f"{f.get('execution_error', 0):>10} {f.get('modeling_error', 0):>10} {f.get('binding_error', 0):>10}")
    print("=" * 100)

    # Print percentages of failure types
    print(f"\n{'Model':<20} {'Exec%':>8} {'Model%':>8} {'Bind%':>8} {'Total Fail':>10}")
    print("-" * 60)
    for r in all_results:
        f = r["failures"]
        total_fail = r["total"] - r["passed"]
        if total_fail == 0:
            print(f"{r['model']:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {0:>10}")
            continue
        exec_pct = f.get("execution_error", 0) / total_fail * 100
        model_pct = f.get("modeling_error", 0) / total_fail * 100
        bind_pct = f.get("binding_error", 0) / total_fail * 100
        print(f"{r['model']:<20} {exec_pct:>7.1f}% {model_pct:>7.1f}% {bind_pct:>7.1f}% {total_fail:>10}")
    print()


def main():
    eval_ids = get_eval_problem_ids()
    print(f"Eval subset: {len(eval_ids)} problems")

    # First, copy pass1 results to eval/ filtered to eval subset
    for model_name, path in PASS1_MODELS.items():
        out_path = EVAL_DIR / f"{model_name}.json"
        if not out_path.exists():
            print(f"Filtering {model_name} from pass1 ({path.name}) to eval subset...")
            full_data = json.load(open(path))
            filtered = [r for r in full_data if r["problem_id"] in eval_ids]
            print(f"  {len(full_data)} -> {len(filtered)} problems")
            json.dump(filtered, open(out_path, "w"), indent=2)
            print(f"  Saved to {out_path}")

    # Now analyze all models
    all_models = {}
    for model_name, path in {**EVAL_MODELS, **{k: EVAL_DIR / f"{k}.json" for k in PASS1_MODELS}}.items():
        all_models[model_name] = path

    all_results = []
    for model_name, path in sorted(all_models.items()):
        print(f"\nAnalyzing {model_name}...")
        data = json.load(open(path))
        result = analyze_model(model_name, data, eval_ids if path.parent != EVAL_DIR else None)
        all_results.append(result)
        f = result["failures"]
        print(f"  Pass: {result['passed']}/{result['total']} ({result['pass_rate']})")
        print(f"  Failures: exec={f.get('execution_error',0)}, model={f.get('modeling_error',0)}, bind={f.get('binding_error',0)}")

    # Sort by pass rate descending
    all_results.sort(key=lambda r: r["passed"] / r["total"], reverse=True)
    print_summary(all_results)

    # Save detailed results
    out_path = RESULTS_DIR / "failure_mode_analysis.json"
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"Detailed results saved to {out_path}")

    # ── Template problems (550) ──────────────────────────────────────────
    # Skipped: BIND already provides the binding vs modeling diagnostic for
    # template problems. The var/constraint counting approach is less
    # informative here (see Appendix discussion).
    return


    TEMPLATE_DIR = Path("results/template_eval")
    TEMPLATE_MODELS = [
        "claude-opus-4-6", "claude-sonnet-4-6", "DeepSeek-R1", "DeepSeek-V3.2",
        "gpt-5", "gpt-5-nano", "Llama-3.3-70B-Instruct", "o4-mini", "Qwen2.5-7B",
    ]

    template_results = []
    for model_name in TEMPLATE_MODELS:
        model_path = TEMPLATE_DIR / model_name / "default.json"
        if not model_path.exists():
            # Try without -Instruct
            model_path = TEMPLATE_DIR / model_name.replace("-Instruct", "") / "default.json"
        if not model_path.exists():
            print(f"  Skipping {model_name}: {model_path} not found")
            continue

        print(f"\nAnalyzing {model_name} (template)...")
        data = json.load(open(model_path))
        result = analyze_model(model_name, data, eval_ids=None)
        template_results.append(result)
        f = result["failures"]
        print(f"  Pass: {result['passed']}/{result['total']} ({result['pass_rate']})")
        print(f"  Failures: exec={f.get('execution_error',0)}, model={f.get('modeling_error',0)}, bind={f.get('binding_error',0)}")

    template_results.sort(key=lambda r: r["passed"] / r["total"], reverse=True)
    print_summary(template_results)

    # Per-category breakdown across all models
    print("\n" + "=" * 120)
    print("PER-CATEGORY FAILURE BREAKDOWN (all models aggregated)")
    print("=" * 120)
    cat_stats = defaultdict(lambda: defaultdict(int))
    for r in template_results:
        for d in r["failure_details"]:
            cat = d["category"]
            cat_stats[cat][d["error_type"]] += 1
            cat_stats[cat]["total_fail"] += 1

    print(f"{'Category':<30} {'Total Fail':>10} {'Exec Err':>10} {'Model Err':>10} {'Bind Err':>10} {'Bind%':>8}")
    print("-" * 120)
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        tf = s["total_fail"]
        bind_pct = s.get("binding_error", 0) / tf * 100 if tf > 0 else 0
        print(f"{cat:<30} {tf:>10} {s.get('execution_error',0):>10} {s.get('modeling_error',0):>10} "
              f"{s.get('binding_error',0):>10} {bind_pct:>7.1f}%")

    # Per-model per-category breakdown
    print("\n" + "=" * 120)
    print("PER-MODEL PER-CATEGORY FAILURE BREAKDOWN")
    print("=" * 120)
    for r in template_results:
        model_cat = defaultdict(lambda: defaultdict(int))
        for d in r["failure_details"]:
            cat = d["category"]
            model_cat[cat][d["error_type"]] += 1
            model_cat[cat]["total_fail"] += 1

        if not model_cat:
            continue
        print(f"\n--- {r['model']} (Pass: {r['pass_rate']}) ---")
        print(f"  {'Category':<28} {'Fail':>5} {'Exec':>5} {'Model':>6} {'Bind':>5} {'Bind%':>7}")
        for cat in sorted(model_cat.keys()):
            s = model_cat[cat]
            tf = s["total_fail"]
            bind_pct = s.get("binding_error", 0) / tf * 100 if tf > 0 else 0
            print(f"  {cat:<28} {tf:>5} {s.get('execution_error',0):>5} {s.get('modeling_error',0):>6} "
                  f"{s.get('binding_error',0):>5} {bind_pct:>6.1f}%")

    # Save template results
    out_path = TEMPLATE_DIR / "failure_mode_analysis.json"
    json.dump(template_results, open(out_path, "w"), indent=2)
    print(f"\nTemplate results saved to {out_path}")


if __name__ == "__main__":
    main()
