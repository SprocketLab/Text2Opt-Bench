#!/usr/bin/env python3
"""
Run best-of-K sampling, reusing an existing Pass@1 result as sample 0.

This saves API cost by only generating K-1 new samples per problem.
The baseline result (from template_eval/{model}/default.json) is converted
to sample_idx=0, then K-1 fresh samples are generated.
"""
import argparse
import asyncio
import json
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main.utils import (
    get_async_openai_client,
    resolve_chat_deployment,
    load_json,
    save_json,
)
from main.evaluation.problem_evaluator import GeneratedSolutionEvaluator
from main.evaluation.run_eval import PROMPT_TEMPLATE


def log(msg):
    print(msg, flush=True)


def eval_result_to_dict(er):
    gen_sol = getattr(er, "generated_solution", None)
    gen_obj = gen_sol.objective_value if gen_sol else None
    gen_status = gen_sol.status if gen_sol else None
    d = {
        "execution_succeeded": er.execution_succeeded,
        "status_optimal": er.status_optimal,
        "objective_matches": er.objective_matches,
        "objective_error": getattr(er, "objective_error", None),
        "error_message": getattr(er, "error_message", None),
        "generated_objective": gen_obj,
        "gold_objective": getattr(er, "reference_optimum", None),
        "generated_status": gen_status,
        "execution_time": getattr(er, "execution_time", None),
    }
    if gen_sol:
        d["stderr"] = (gen_sol.stderr or "")[-1600:]  # tail to keep file size sane
        d["num_vars"] = getattr(gen_sol, "num_vars", None)
        d["num_constrs"] = getattr(gen_sol, "num_constrs", None)
    return d


def is_success(er_dict):
    return (er_dict.get("execution_succeeded")
            and er_dict.get("status_optimal")
            and er_dict.get("objective_matches"))


def baseline_to_sample(baseline_record):
    """Convert a template_eval record to a best-of-k sample (sample_idx=0)."""
    er = baseline_record.get("evaluation_result", {})
    gen_exec = baseline_record.get("generated_execution", {})
    succeeded = (er.get("execution_succeeded")
                 and er.get("status_optimal")
                 and er.get("objective_matches"))
    return {
        "sample_idx": 0,
        "succeeded": succeeded,
        "execution_succeeded": er.get("execution_succeeded"),
        "status_optimal": er.get("status_optimal"),
        "objective_matches": er.get("objective_matches"),
        "error_message": None,
        "generated_objective": gen_exec.get("objective_value"),
        "gold_objective": er.get("reference_optimum"),
        "generated_status": gen_exec.get("status"),
        "prompt_tokens": 0,   # not available from baseline run
        "completion_tokens": 0,  # not available from baseline run
        "generated_code": baseline_record.get("generated_code", ""),
        "from_baseline": True,
    }


async def run_best_of_k_with_baseline(
    dataset_paths: list[str],
    model: str,
    k: int,
    output_path: str,
    baseline_path: str,
    max_concurrent: int = 300,
    temperature: float = None,
):
    client = get_async_openai_client(model)
    deployment_name = resolve_chat_deployment(model)
    evaluator = GeneratedSolutionEvaluator()

    # Load baseline results (existing Pass@1)
    baseline_data = json.load(open(baseline_path))
    baseline_by_id = {r["problem_id"]: r for r in baseline_data}
    log(f"Loaded {len(baseline_by_id)} baseline results from {baseline_path}")

    # Load all problems
    all_problems = []
    for dp in dataset_paths:
        p = Path(dp)
        if p.is_file() and p.suffix == ".json":
            all_problems.append((str(p), load_json(str(p))))
        elif p.is_dir():
            for f in sorted(glob.glob(str(p / "**/*.json"), recursive=True)):
                all_problems.append((f, load_json(f)))

    log(f"Loaded {len(all_problems)} problems | K={k} (1 baseline + {k-1} new)")

    results_file = Path(output_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if results_file.exists():
        for r in json.load(open(results_file)):
            existing[r["problem_id"]] = r
        log(f"Resumed: {len(existing)} existing results from {results_file}")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    solved_count = 0
    skipped_count = 0
    total_count = len(all_problems)
    save_lock = asyncio.Lock()
    all_results = dict(existing)

    async def save_results():
        async with save_lock:
            save_json(str(results_file), list(all_results.values()))

    async def generate_one_sample(messages, problem_data, problem_id, sample_idx):
        """Generate one independent sample and evaluate it."""
        async with semaphore:
            await asyncio.sleep(0.05)
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    create_kwargs = dict(
                        model=deployment_name,
                        messages=messages,
                        max_completion_tokens=15000,
                    )
                    if temperature is not None:
                        create_kwargs["temperature"] = temperature
                    response = await client.chat.completions.create(**create_kwargs)
                    response_text = response.choices[0].message.content
                    usage = response.usage
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt + 1, 60)
                        log(f"  API Error sample {sample_idx} for {problem_id} (retry {attempt+1}/{max_retries}): {e}")
                        await asyncio.sleep(wait)
                    else:
                        log(f"  API Error sample {sample_idx} for {problem_id} (all retries failed): {e}")
                        return {
                            "sample_idx": sample_idx,
                            "succeeded": False,
                            "error": f"API error: {e}",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        }

            eval_result = await asyncio.to_thread(
                evaluator.evaluate_generated_code,
                problem_data, response_text, problem_id=problem_id
            )
            er_dict = eval_result_to_dict(eval_result)
            success = is_success(er_dict)

            return {
                "sample_idx": sample_idx,
                "succeeded": success,
                **er_dict,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "generated_code": response_text,
            }

    async def process_one(filepath, problem_data):
        nonlocal completed_count, solved_count, skipped_count
        problem_id = os.path.basename(filepath)
        category = _detect_category(filepath)

        # Check existing
        prev = existing.get(problem_id)
        if prev and prev.get("num_samples", 0) >= k:
            skipped_count += 1
            all_results[problem_id] = prev
            return prev

        # Get baseline sample
        baseline_rec = baseline_by_id.get(problem_id)
        if baseline_rec is None:
            log(f"WARNING: No baseline for {problem_id}, generating all {k} samples")
            sample_0 = None
            new_k = k
        else:
            sample_0 = baseline_to_sample(baseline_rec)
            new_k = k - 1

        # Build prompt
        llm_description = problem_data.get("LLM_description", "")
        prompt = PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)
        messages = [{"role": "user", "content": prompt}]

        # Generate new_k samples in parallel
        sample_tasks = [
            generate_one_sample(messages, problem_data, problem_id, i + (1 if sample_0 else 0))
            for i in range(new_k)
        ]
        new_samples = await asyncio.gather(*sample_tasks)

        # Combine baseline + new samples
        if sample_0:
            samples = [sample_0] + list(new_samples)
        else:
            samples = list(new_samples)

        num_success = sum(1 for s in samples if s.get("succeeded"))
        any_success = num_success > 0
        total_prompt_tokens = sum(s.get("prompt_tokens", 0) for s in samples)
        total_completion_tokens = sum(s.get("completion_tokens", 0) for s in samples)
        total_tokens = total_prompt_tokens + total_completion_tokens

        result = {
            "problem_id": problem_id,
            "category": category,
            "num_samples": k,
            "num_success": num_success,
            "any_success": any_success,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "samples": samples,
        }

        all_results[problem_id] = result
        completed_count += 1
        if any_success:
            solved_count += 1
        status = "OK" if any_success else "FAIL"
        log(f"[{completed_count + skipped_count}/{total_count}] "
            f"{status} {num_success}/{k} | {category}/{problem_id} | "
            f"solved={solved_count}/{completed_count}")

        if completed_count % 10 == 0:
            await save_results()

        return result

    tasks = [process_one(fp, pd) for fp, pd in all_problems]
    await asyncio.gather(*tasks)

    await save_results()
    log(f"\nDone! Saved {len(all_results)} results to {results_file}")

    # Print summary
    all_vals = list(all_results.values())
    by_category = defaultdict(list)
    for r in all_vals:
        by_category[r.get("category", "unknown")].append(r)

    log(f"\n{'='*60}")
    log(f"BEST-OF-{k} RESULTS (with baseline)")
    log(f"{'='*60}")

    all_pass1 = []
    for r in all_vals:
        n_succ = r.get("num_success", 0)
        n_samp = r.get("num_samples", k)
        all_pass1.append(n_succ / n_samp)

    pass_at_1 = sum(all_pass1) / len(all_pass1) * 100 if all_pass1 else 0
    pass_at_k = sum(1 for r in all_vals if r.get("any_success")) / len(all_vals) * 100 if all_vals else 0
    avg_tokens = sum(r.get("total_tokens", 0) for r in all_vals) / len(all_vals) if all_vals else 0

    log(f"\n--- Overall (n={len(all_vals)}) ---")
    log(f"  pass@1:  {pass_at_1:.1f}%")
    log(f"  pass@{k}:  {pass_at_k:.1f}%")
    log(f"  avg tokens/problem: {avg_tokens:.0f}")

    for cat in sorted(by_category.keys()):
        cat_results = by_category[cat]
        cat_pass1 = [r.get("num_success", 0) / r.get("num_samples", k) for r in cat_results]
        p1 = sum(cat_pass1) / len(cat_pass1) * 100
        pk = sum(1 for r in cat_results if r.get("any_success")) / len(cat_results) * 100
        avg_t = sum(r.get("total_tokens", 0) for r in cat_results) / len(cat_results)
        log(f"\n--- {cat} (n={len(cat_results)}) ---")
        log(f"  pass@1:  {p1:.1f}%")
        log(f"  pass@{k}:  {pk:.1f}%")
        log(f"  avg tokens/problem: {avg_t:.0f}")


def _detect_category(filepath):
    """Detect problem category from file path."""
    path_lower = filepath.lower()
    for cat in [
        "transportation", "disaster_response", "facility_location",
        "power_transmission", "queuing_staffing", "jssp", "vrptw", "rcpsp",
        "stochastic_transportation", "multiobjective_transportation",
        "modified_facility_location",
    ]:
        if cat in path_lower:
            return cat
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run best-of-K with existing Pass@1 as sample 0"
    )
    parser.add_argument("--dataset-paths", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--baseline-results", required=True,
                        help="Path to existing default.json (Pass@1 results)")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-concurrent", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. If not set, uses API default. Set to 1.0 for vLLM to get diverse samples.")
    args = parser.parse_args()

    asyncio.run(run_best_of_k_with_baseline(
        dataset_paths=args.dataset_paths,
        model=args.model,
        k=args.k,
        output_path=args.output_path,
        baseline_path=args.baseline_results,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
    ))
