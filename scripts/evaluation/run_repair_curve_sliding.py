#!/usr/bin/env python3
"""
Run direct-solve + evaluator repair loop with SLIDING WINDOW context.
Window size = 1: only keep the initial prompt + last round's (response, feedback).

This reduces context growth from O(rounds) to O(1) per round.

Supports both direct-solve (resource_allocation) and agentic (template) modes.
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
from main.evaluation.agentic_offload import (
    AGENTIC_SYSTEM_PROMPT,
    persist_agentic_data,
    build_agentic_prompt,
    build_repair_prompt,
    build_direct_solve_repair_prompt,
)
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
        d["stderr"] = (gen_sol.stderr or "")[-1600:]
        d["num_vars"] = getattr(gen_sol, "num_vars", None)
        d["num_constrs"] = getattr(gen_sol, "num_constrs", None)
    return d


def is_success(er_dict):
    return (er_dict.get("execution_succeeded")
            and er_dict.get("status_optimal")
            and er_dict.get("objective_matches"))


async def run_repair_curve_sliding(
    dataset_paths: list[str],
    model: str,
    max_rounds: int,
    output_path: str,
    max_concurrent: int = 300,
    mode: str = "direct",  # "direct" or "agentic"
    window_size: int = 1,
):
    client = get_async_openai_client(model)
    deployment_name = resolve_chat_deployment(model)
    evaluator = GeneratedSolutionEvaluator()

    # Load all problems
    all_problems = []
    for dp in dataset_paths:
        p = Path(dp)
        if p.is_file() and p.suffix == ".json":
            all_problems.append((str(p), load_json(str(p))))
        elif p.is_dir():
            for f in sorted(glob.glob(str(p / "**/*.json"), recursive=True)):
                all_problems.append((f, load_json(f)))

    log(f"Loaded {len(all_problems)} problems | mode={mode} | window={window_size}")

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

    async def process_one(filepath, problem_data):
        nonlocal completed_count, solved_count, skipped_count
        problem_id = os.path.basename(filepath)
        category = _detect_category(filepath)

        prev = existing.get(problem_id)
        if prev and prev.get("solved_at_round") is not None:
            prev["category"] = category
            skipped_count += 1
            all_results[problem_id] = prev
            return prev
        if prev and prev.get("total_rounds_run", 0) >= max_rounds:
            prev["category"] = category
            skipped_count += 1
            all_results[problem_id] = prev
            return prev

        # Build initial prompt based on mode
        data_path = None
        if mode == "agentic":
            data_path, _mode = persist_agentic_data(problem_data, problem_id)
            agentic_prompt = build_agentic_prompt(problem_data, data_path)
            initial_messages = [
                {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                {"role": "user", "content": agentic_prompt},
            ]
        else:
            llm_description = problem_data.get("LLM_description", "")
            prompt = PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)
            initial_messages = [{"role": "user", "content": prompt}]

        # We store ALL messages for logging, but only send a WINDOW to the API
        all_messages = list(initial_messages)
        per_round = {}
        solved_at = None
        total_rounds = 0

        async with semaphore:
            for round_idx in range(1, max_rounds + 1):
                total_rounds = round_idx
                await asyncio.sleep(0.05)

                # BUILD SLIDING WINDOW: initial prompt + last `window_size` rounds
                # Each round = (assistant response, user feedback) pair
                # For round 1, just send initial_messages
                if round_idx == 1:
                    api_messages = list(initial_messages)
                else:
                    # initial_messages + last window_size rounds of (response, feedback)
                    # all_messages layout:
                    #   [*initial, resp_1, fb_1, resp_2, fb_2, ..., resp_n, fb_n]
                    # Each round after initial adds 2 messages (resp + feedback)
                    n_initial = len(initial_messages)
                    # Total round pairs stored so far
                    n_pairs = (len(all_messages) - n_initial) // 2
                    # Keep last window_size pairs
                    keep_pairs = min(window_size, n_pairs)
                    start_pair = n_pairs - keep_pairs
                    start_idx = n_initial + start_pair * 2
                    api_messages = list(initial_messages) + all_messages[start_idx:]

                # Retry indefinitely — round n must complete before n+1
                response_text = None
                attempt = 0
                while response_text is None:
                    try:
                        response = await client.chat.completions.create(
                            model=deployment_name,
                            messages=api_messages,
                            max_completion_tokens=15000,
                        )
                        response_text = response.choices[0].message.content
                        usage = response.usage
                    except Exception as e:
                        attempt += 1
                        wait = min(2 ** attempt + 1, 120)
                        log(f"  API Error round {round_idx} for {problem_id} (retry {attempt}): {e}")
                        await asyncio.sleep(wait)

                eval_result = evaluator.evaluate_generated_code(
                    problem_data, response_text, problem_id=problem_id
                )
                er_dict = eval_result_to_dict(eval_result)
                success = is_success(er_dict)
                per_round[str(round_idx)] = {
                    "succeeded": success,
                    **er_dict,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "generated_code": response_text,
                }

                if success:
                    solved_at = round_idx
                    all_messages.append({"role": "assistant", "content": response_text})
                    break

                if round_idx < max_rounds:
                    if mode == "agentic":
                        repair = build_repair_prompt(eval_result, data_path=data_path, problem_data=problem_data)
                    else:
                        repair = build_direct_solve_repair_prompt(eval_result, problem_data)
                    all_messages.append({"role": "assistant", "content": response_text})
                    all_messages.append({"role": "user", "content": repair})

        result = {
            "problem_id": problem_id,
            "category": category,
            "solved_at_round": solved_at,
            "total_rounds_run": total_rounds,
            "per_round": per_round,
            "messages": all_messages,
            "window_size": window_size,
        }

        all_results[problem_id] = result
        completed_count += 1
        if solved_at is not None:
            solved_count += 1
        status = "OK" if solved_at else "FAIL"
        rounds_info = f"r{solved_at}" if solved_at else f"r{total_rounds}"
        log(f"[{completed_count + skipped_count}/{total_count}] "
            f"{status} {rounds_info} | {category}/{problem_id} | "
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
    log(f"ACCURACY CURVE — Sliding Window (w={window_size})")
    log(f"{'='*60}")

    log(f"\n--- Overall (n={len(all_vals)}) ---")
    header = f"{'Round':<8}" + "".join(f"{rd:<8}" for rd in range(1, max_rounds + 1))
    log(header)
    row = f"{'Acc%':<8}"
    for rd in range(1, max_rounds + 1):
        solved = sum(1 for r in all_vals if r.get("solved_at_round") is not None and r["solved_at_round"] <= rd)
        pct = solved / len(all_vals) * 100 if all_vals else 0
        row += f"{pct:<8.1f}"
    log(row)

    for cat in sorted(by_category.keys()):
        cat_results = by_category[cat]
        log(f"\n--- {cat} (n={len(cat_results)}) ---")
        header = f"{'Round':<8}" + "".join(f"{rd:<8}" for rd in range(1, max_rounds + 1))
        log(header)
        row = f"{'Acc%':<8}"
        for rd in range(1, max_rounds + 1):
            solved = sum(1 for r in cat_results if r.get("solved_at_round") is not None and r["solved_at_round"] <= rd)
            pct = solved / len(cat_results) * 100 if cat_results else 0
            row += f"{pct:<8.1f}"
        log(row)


def _detect_category(filepath):
    """Detect problem category from file path."""
    path_lower = filepath.lower()
    for cat in [
        "stochastic_transportation", "multiobjective_transportation",
        "modified_facility_location", "transportation",
        "disaster_response", "facility_location",
        "power_transmission", "queuing_staffing",
        "jssp", "vrptw", "rcpsp",
    ]:
        if cat in path_lower:
            return cat
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run repair curve with sliding window context"
    )
    parser.add_argument("--mode", choices=["direct", "agentic"], required=True)
    parser.add_argument(
        "--dataset-paths", nargs="+", required=True,
        help="Paths to dataset directories or files",
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=1,
                        help="Number of previous rounds to keep in context (default: 1)")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-concurrent", type=int, default=300)
    args = parser.parse_args()

    asyncio.run(run_repair_curve_sliding(
        dataset_paths=args.dataset_paths,
        model=args.model,
        max_rounds=args.max_rounds,
        output_path=args.output_path,
        max_concurrent=args.max_concurrent,
        mode=args.mode,
        window_size=args.window_size,
    ))
