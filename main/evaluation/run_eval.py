#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
import time
import textwrap
from pathlib import Path

# Default max output tokens per API call. Override via --max-completion-tokens CLI arg.
MAX_COMPLETION_TOKENS = 15000

# Ensure we can import from the same directory
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from main.utils import (
    get_async_openai_client,
    resolve_chat_deployment,
    load_json,
    save_json,
)

from main.evaluation.problem_evaluator import (
    CodeSolutionEvaluator,
    GeneratedSolutionEvaluator,
    ExecutionResult,
    EvaluationResult,
)

from main.evaluation.agentic_offload import (
    AGENTIC_SYSTEM_PROMPT,
    DEFAULT_ARRAY_CACHE_DIR,
    persist_agentic_data,
    persist_problem_array,
    build_agentic_prompt,
    build_repair_prompt,
    # Direct-solve + repair
    build_direct_solve_repair_prompt,
    # Self-review
    build_self_review_prompt,
    # Structured JSON extraction
    build_structured_extraction_prompt,
    # Chunked extract-then-solve
    _EXTRACTION_SYSTEM_PROMPT,
    build_variable_extraction_prompt,
    chunk_description,
    build_chunk_contribution_prompt,
    merge_chunked_extractions,
    build_extraction_prompt,
    build_solve_from_extracted_prompt,
    build_extraction_repair_prompt,
    parse_extraction_json,
    parse_solver_code,
)

# Claude adapter is now in main.utils (_ClaudeOpenAIAdapter)
from main.utils import _ClaudeOpenAIAdapter


# Lazy-import vLLM: only loaded when actually needed (run_vllm_evaluation).
# Importing eagerly can hang when CUDA init stalls on shared machines.
VLLM_AVAILABLE = None  # None = not yet checked; True/False after first check
LLM = None
SamplingParams = None

def _ensure_vllm():
    global VLLM_AVAILABLE, LLM, SamplingParams
    if VLLM_AVAILABLE is not None:
        return VLLM_AVAILABLE
    try:
        from vllm import LLM as _LLM, SamplingParams as _SP
        LLM = _LLM
        SamplingParams = _SP
        VLLM_AVAILABLE = True
    except ImportError:
        VLLM_AVAILABLE = False
    return VLLM_AVAILABLE

PROMPT_TEMPLATE = textwrap.dedent(
    """
    Solve the following optimization problem using Gurobi.

    {LLM_DESCRIPTION}

    **Your Task:**
    Analyze the problem description above and write a complete Python function `solve_problem()` using `gurobipy` to solve this optimization problem.

    **Problem Analysis Steps:**
    1. **Identify Decision Variables**: Determine what variables need to be defined based on the problem structure. Variable names may not be explicitly given - you may need to infer appropriate names from the problem context.
    2. **Identify Constraints**: Extract all constraints from the problem description, including their types (≤, ≥, =) and right-hand side values.
    3. **Identify Objective Function**: Determine the optimization goal (minimize or maximize) and the objective coefficients from the problem description.
    4. **Determine Variable Types**: Identify whether variables should be Continuous or Integer or Binary based on the problem description.

    **Code Requirements:**
    1. The solution MUST be a valid Python code block wrapped in ```python ... ```.
    2. The function `solve_problem` must return the Gurobi model object `m` after calling `m.optimize()`.
    3. Define all necessary variables with appropriate names, types (Continuous/Integer/Binary), and bounds as inferred from the problem.
    4. Implement all constraints as described in the problem, do not omit any of them.
    5. Set the objective function correctly (minimize or maximize) with the correct coefficients.
    6. Set the time limit to 100 seconds using `m.setParam("TimeLimit", 100)` before calling `m.optimize()`.
    7. Do not test the function or print solutions; just provide the function definition.
    8. Do not import any libraries other than `gurobipy`.
    9. Ensure the code is self-contained and clear, with comments explaining the logic and variable meanings.

    **Template:**
    ```python
    from gurobipy import *

    def solve_problem():
        m = Model("Optimization_Problem")
        m.setParam("TimeLimit", 100)  # Set time limit to 100 seconds

        # Define decision variables based on the problem structure
        # Use meaningful variable names that reflect the problem context
        # ...

        # Set objective function
        # ...

        # Add constraints
        # ...

        m.optimize()
        return m
    ```
    """
).strip()


COT_PROMPT_TEMPLATE = textwrap.dedent(
    """
    Solve the following optimization problem using Gurobi.

    {LLM_DESCRIPTION}

    **IMPORTANT: Before writing ANY code, you MUST complete the formulation checklist below.**

    ## Step 1: Formulation Checklist

    Fill out this checklist by carefully reading the problem description:

    **Objective:**
    - Direction: [MINIMIZE or MAXIMIZE]

    **Decision Variables** (list ALL of them):
    | # | Variable Name | Type | Lower Bound | Upper Bound | Objective Coefficient |
    |---|---|---|---|---|---|
    | 1 | ... | Continuous/Integer/Binary | ... | ... | ... |

    **Constraints** (list ALL of them — do NOT include variable bounds here):
    | # | Constraint Description | LHS (coefficients × variables) | Sense | RHS |
    |---|---|---|---|---|
    | 1 | ... | coeff1 * var1 + coeff2 * var2 + ... | <=/>=/== | ... |

    Double-check:
    - Did you include EVERY variable mentioned in the problem?
    - Did you include EVERY constraint? Count them against the problem description.
    - Are the coefficients exact (not rounded)?
    - Is each variable's type correct (integer vs continuous)?

    ## Step 2: Write the Code

    Now translate your formulation into a complete `solve_problem()` function.

    **Code Requirements:**
    1. The solution MUST be a valid Python code block wrapped in ```python ... ```.
    2. The function `solve_problem` must return the Gurobi model object `m` after calling `m.optimize()`.
    3. Do not import any libraries other than `gurobipy`.
    4. Set `m.setParam("TimeLimit", 100)` before `m.optimize()`.
    5. Do not test the function or print solutions; just provide the function definition.
    """
).strip()


def build_prompt(data: dict, args=None) -> str | None:
    """
    Build the complete prompt from problem data and apply inference wrapper.
    
    Args:
        data: The problem data dictionary containing 'LLM_description'.
        args: Optional arguments object containing 'inference_option'.
        
    Returns:
        The fully constructed prompt ready for inference, or None if LLM_description is missing.
    """
    llm_description = data.get("LLM_description")
    if not llm_description:
        return None
    
    # Get inference option from args
    inference_option = getattr(args, "inference_option", "default") if args else "default"
    
    # Select prompt template
    if inference_option == "cot":
        prompt = COT_PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)
    else:
        prompt = PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)

    # Apply inference wrapper if needed
    if inference_option in ("agentic-offload", "self-review", "structured-json"):
        # Signal to caller that agentic path should be used instead.
        # Array path is built lazily inside process_single_problem.
        return None

    return prompt


def write_results(results, output_path):
    """Helper to write results to file safely"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(output_path, results)
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def _filter_result_data(data: dict, save_variables: bool = False) -> dict:
    """
    Filter result data to only include objective values by default.
    
    Args:
        data: The problem data dictionary
        save_variables: If True, include variable values; if False, only include objective
        
    Returns:
        Filtered data dictionary
    """
    filtered = data.copy()
    
    # Filter gurobi_result - only keep theoretical_optimum
    if "gurobi_result" in filtered:
        gurobi_result = filtered["gurobi_result"].copy()
        if not save_variables and "optimal_values" in gurobi_result:
            del gurobi_result["optimal_values"]
        # Keep only theoretical_optimum and solver_status
        filtered["gurobi_result"] = {
            "solver_status": gurobi_result.get("solver_status"),
            "theoretical_optimum": gurobi_result.get("theoretical_optimum"),
        }
        if save_variables and "optimal_values" in gurobi_result:
            filtered["gurobi_result"]["optimal_values"] = gurobi_result["optimal_values"]
    
    return filtered


async def _api_call_with_retry(client, max_retries=10, **kwargs):
    """Call client.chat.completions.create with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt + 1, 120)
                print(f"  ⚠️ API retry {attempt+1}/{max_retries}: {e}")
                await asyncio.sleep(wait)
            else:
                raise


async def _run_direct_solve_openai(
    client,
    model: str,
    problem_data: dict,
    evaluator,
    problem_id: str,
    max_rounds: int,
    deployment_name: str,
) -> tuple:
    """
    Direct solve + repair loop for problems without instance_data.

    Round 1: Standard 1-shot prompt (full description → Gurobi code).
    Repair:  Error feedback → LLM fixes the code.

    Same as 1-shot baseline for round 1, but adds repair rounds on failure.
    Returns (last_eval_result, rounds_used, token_usage_dict, last_response).
    """
    llm_description = problem_data.get("LLM_description", "")
    prompt = PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)

    messages = [
        {"role": "user", "content": prompt},
    ]

    last_eval = None
    last_response = ""
    rounds_used = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for round_idx in range(1, max_rounds + 1):
        rounds_used = round_idx
        await asyncio.sleep(0.1)
        try:
            response = await _api_call_with_retry(
                client, model=deployment_name,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            last_response = response.choices[0].message.content
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"❌ API Error (direct solve round {round_idx}) for {problem_id}: {e}")
            break

        last_eval = await evaluator.evaluate_generated_code_async(
            problem_data, last_response, problem_id=problem_id
        )

        if (last_eval.execution_succeeded
                and last_eval.status_optimal
                and last_eval.objective_matches):
            break

        if round_idx < max_rounds:
            repair = build_direct_solve_repair_prompt(last_eval, problem_data)
            messages.append({"role": "assistant", "content": last_response})
            messages.append({"role": "user", "content": repair})

    token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
    return last_eval, rounds_used, token_usage, last_response


async def _run_self_review_openai(
    client,
    model: str,
    problem_data: dict,
    evaluator,
    problem_id: str,
    max_rounds: int,
    deployment_name: str,
) -> tuple:
    """
    Self-review agentic approach (no evaluator feedback).

    Turn 1: Standard 1-shot prompt → LLM writes code.
    Turn 2: Self-review → LLM verifies its code against the description, fixes errors.
    Turn 3+: (optional) evaluator-based repair if self-review still fails.

    Returns (last_eval_result, rounds_used, token_usage_dict, last_response).
    """
    llm_description = problem_data.get("LLM_description", "")
    prompt = PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)
    total_prompt_tokens = 0
    total_completion_tokens = 0

    messages = [
        {"role": "user", "content": prompt},
    ]

    # Turn 1: Generate initial code
    await asyncio.sleep(0.1)
    try:
        response = await _api_call_with_retry(client, 
            model=deployment_name,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        initial_code = response.choices[0].message.content
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        print(f"❌ API Error (self-review turn 1) for {problem_id}: {e}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    # Turn 2: Self-review (no evaluator — LLM checks its own code)
    review_prompt = build_self_review_prompt(initial_code)
    messages.append({"role": "assistant", "content": initial_code})
    messages.append({"role": "user", "content": review_prompt})

    await asyncio.sleep(0.1)
    try:
        response = await _api_call_with_retry(client, 
            model=deployment_name,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        reviewed_code = response.choices[0].message.content
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        print(f"❌ API Error (self-review turn 2) for {problem_id}: {e}")
        reviewed_code = initial_code

    messages.append({"role": "assistant", "content": reviewed_code})

    # Evaluate the self-reviewed code
    last_eval = await evaluator.evaluate_generated_code_async(
        problem_data, reviewed_code, problem_id=problem_id
    )
    rounds_used = 2
    last_response = reviewed_code

    if (last_eval.execution_succeeded
            and last_eval.status_optimal
            and last_eval.objective_matches):
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return last_eval, rounds_used, token_usage, last_response

    # Turn 3+: evaluator-based repair if self-review wasn't enough
    for round_idx in range(3, max_rounds + 1):
        rounds_used = round_idx
        repair = build_direct_solve_repair_prompt(last_eval)
        messages.append({"role": "user", "content": repair})

        await asyncio.sleep(0.1)
        try:
            response = await _api_call_with_retry(client, 
                model=deployment_name,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            last_response = response.choices[0].message.content
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"❌ API Error (self-review repair {round_idx}) for {problem_id}: {e}")
            break

        messages.append({"role": "assistant", "content": last_response})
        last_eval = await evaluator.evaluate_generated_code_async(
            problem_data, last_response, problem_id=problem_id
        )

        if (last_eval.execution_succeeded
                and last_eval.status_optimal
                and last_eval.objective_matches):
            break

    token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
    return last_eval, rounds_used, token_usage, last_response


async def _run_structured_json_openai(
    client,
    model: str,
    problem_data: dict,
    extracted_path,
    evaluator,
    problem_id: str,
    max_rounds: int,
    deployment_name: str,
) -> tuple:
    """
    Structured JSON extraction → solve.

    Turn 1: Extract the full formulation into a single JSON (improved prompt).
    Turn 2: Write solver code from the extracted JSON file.
    Repair: On failure, re-extract + re-solve.

    Returns (last_eval_result, rounds_used, token_usage_dict, last_code).
    """
    import json as json_mod

    llm_description = problem_data.get("LLM_description", "")
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Turn 1: Extract formulation into JSON (with JSON mode for guaranteed valid output)
    extract_prompt = build_structured_extraction_prompt(problem_data)

    await asyncio.sleep(0.1)
    try:
        response = await _api_call_with_retry(client, 
            model=deployment_name,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user",   "content": extract_prompt},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
        )
        extract_response = response.choices[0].message.content
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        print(f"❌ API Error (structured extraction) for {problem_id}: {e}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    extracted = parse_extraction_json(extract_response)
    if extracted is None:
        print(f"⚠️  Failed to parse structured extraction for {problem_id}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    with open(extracted_path, "w", encoding="utf-8") as f:
        json_mod.dump(extracted, f, indent=2)

    # Turn 2: Solve from extracted JSON
    solve_prompt = build_solve_from_extracted_prompt(extracted_path)

    await asyncio.sleep(0.1)
    try:
        response = await _api_call_with_retry(client, 
            model=deployment_name,
            messages=[
                {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
                {"role": "user",   "content": solve_prompt},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        last_code = response.choices[0].message.content
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        print(f"❌ API Error (structured solve) for {problem_id}: {e}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    last_eval = await evaluator.evaluate_generated_code_async(
        problem_data, last_code, problem_id=problem_id
    )
    rounds_used = 1

    if (last_eval.execution_succeeded
            and last_eval.status_optimal
            and last_eval.objective_matches):
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return last_eval, rounds_used, token_usage, last_code

    # Repair rounds: re-extract + re-solve
    repair_messages = [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user",   "content": extract_prompt},
        {"role": "assistant", "content": extract_response},
        {"role": "user",   "content": solve_prompt},
        {"role": "assistant", "content": last_code},
    ]

    for round_idx in range(2, max_rounds + 1):
        rounds_used = round_idx
        repair = build_extraction_repair_prompt(
            last_eval, extracted_path, llm_description
        )
        repair_messages.append({"role": "user", "content": repair})

        await asyncio.sleep(0.1)
        try:
            response = await _api_call_with_retry(client, 
                model=deployment_name,
                messages=repair_messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            repair_response = response.choices[0].message.content
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"❌ API Error (structured repair {round_idx}) for {problem_id}: {e}")
            break

        new_extracted = parse_extraction_json(repair_response)
        new_code = parse_solver_code(repair_response)

        if new_extracted:
            with open(extracted_path, "w", encoding="utf-8") as f:
                json_mod.dump(new_extracted, f, indent=2)

        if new_code:
            last_code = new_code
        else:
            last_code = repair_response

        repair_messages.append({"role": "assistant", "content": repair_response})
        last_eval = await evaluator.evaluate_generated_code_async(
            problem_data, last_code, problem_id=problem_id
        )

        if (last_eval.execution_succeeded
                and last_eval.status_optimal
                and last_eval.objective_matches):
            break

    token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
    return last_eval, rounds_used, token_usage, last_code


async def _run_two_phase_openai(
    client,
    model: str,
    problem_data: dict,
    extracted_path,
    evaluator,
    problem_id: str,
    max_rounds: int,
    deployment_name: str,
) -> tuple:
    """
    Chunked extract-then-solve for problems without instance_data.

    Phase 1a: Extract variable list (goal, names, types, bounds, obj coefficients).
    Phase 1b: Chunk the description → extract constraints per chunk (parallel).
    Phase 2:  Write solver code from the extracted JSON file.
    Repair:   On failure, re-send description + error → LLM re-extracts + re-solves.

    Returns (last_eval_result, rounds_used, token_usage_dict, last_code).
    """
    import json as json_mod

    llm_description = problem_data.get("LLM_description", "")
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # ---- Phase 1a: Extract variables ----
    var_prompt = build_variable_extraction_prompt(problem_data)

    await asyncio.sleep(0.1)
    try:
        response = await _api_call_with_retry(client, 
            model=deployment_name,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user",   "content": var_prompt},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        var_response = response.choices[0].message.content
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        print(f"❌ API Error (var extraction) for {problem_id}: {e}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    var_data = parse_extraction_json(var_response)
    if var_data is None or "variables" not in var_data:
        print(f"⚠️  Failed to parse variable extraction for {problem_id}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    goal = var_data.get("goal", "MINIMIZE")
    variables = var_data["variables"]
    variable_names = [v["name"] for v in variables]

    # ---- Phase 1b: Chunk description → extract contributions + limits per chunk ----
    chunks = chunk_description(llm_description, max_chars=1000)

    async def _extract_chunk(chunk_text: str) -> dict:
        """Call API for one chunk, return {contributions: [...], limits: [...], prompt_tokens, completion_tokens}."""
        prompt = build_chunk_contribution_prompt(chunk_text, variable_names)
        await asyncio.sleep(0.1)
        try:
            resp = await _api_call_with_retry(client, 
                model=deployment_name,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=8000,
            )
            parsed = parse_extraction_json(resp.choices[0].message.content)
            chunk_usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            if parsed:
                return {
                    "contributions": parsed.get("contributions", []),
                    "limits": parsed.get("limits", []),
                    **chunk_usage,
                }
            return {"contributions": [], "limits": [], **chunk_usage}
        except Exception as e:
            print(f"⚠️  Chunk extraction error for {problem_id}: {e}")
        return {"contributions": [], "limits": [], "prompt_tokens": 0, "completion_tokens": 0}

    # Run all chunk extractions in parallel
    chunk_results = await asyncio.gather(*[_extract_chunk(c) for c in chunks])
    all_contributions = []
    all_limits = []
    for result in chunk_results:
        all_contributions.extend(result["contributions"])
        all_limits.extend(result["limits"])
        total_prompt_tokens += result.get("prompt_tokens", 0)
        total_completion_tokens += result.get("completion_tokens", 0)

    # Merge: match contributions to limits by resource name → build constraints
    extracted = merge_chunked_extractions(goal, variables, all_contributions, all_limits)
    with open(extracted_path, "w", encoding="utf-8") as f:
        json_mod.dump(extracted, f, indent=2)

    # ---- Phase 2: Solve from extracted data ----
    solve_prompt = build_solve_from_extracted_prompt(extracted_path)

    solve_messages = [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user",   "content": solve_prompt},
    ]

    last_eval = None
    last_code = ""
    rounds_used = 1  # Phase 1 + Phase 2 together count as round 1

    await asyncio.sleep(0.1)
    try:
        response = await _api_call_with_retry(client, 
            model=deployment_name,
            messages=solve_messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        last_code = response.choices[0].message.content
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
    except Exception as e:
        print(f"❌ API Error (solve phase) for {problem_id}: {e}")
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return None, 1, token_usage, ""

    last_eval = await evaluator.evaluate_generated_code_async(
        problem_data, last_code, problem_id=problem_id
    )

    if (last_eval.execution_succeeded
            and last_eval.status_optimal
            and last_eval.objective_matches):
        token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
        return last_eval, rounds_used, token_usage, last_code

    # ---- Repair rounds: single-shot re-extract + re-solve ----
    # For repair, use single-shot extraction (full description) since we now
    # know what went wrong and the model can focus on fixing specific issues.
    extraction_prompt = build_extraction_prompt(problem_data)
    repair_messages = [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user",   "content": extraction_prompt},
        {"role": "assistant", "content": json_mod.dumps(extracted, indent=2)},
        {"role": "user",   "content": solve_prompt},
        {"role": "assistant", "content": last_code},
    ]

    for round_idx in range(2, max_rounds + 1):
        rounds_used = round_idx

        repair = build_extraction_repair_prompt(
            last_eval, extracted_path, llm_description
        )
        repair_messages.append({"role": "user", "content": repair})

        await asyncio.sleep(0.1)
        try:
            response = await _api_call_with_retry(client, 
                model=deployment_name,
                messages=repair_messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            repair_response = response.choices[0].message.content
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"❌ API Error (repair round {round_idx}) for {problem_id}: {e}")
            break

        # Try to parse updated extraction + solver code from repair response
        new_extracted = parse_extraction_json(repair_response)
        new_code = parse_solver_code(repair_response)

        if new_extracted:
            with open(extracted_path, "w", encoding="utf-8") as f:
                json_mod.dump(new_extracted, f, indent=2)

        if new_code:
            last_code = new_code
        else:
            # If no python block found, treat entire response as code
            last_code = repair_response

        repair_messages.append({"role": "assistant", "content": repair_response})

        last_eval = await evaluator.evaluate_generated_code_async(
            problem_data, last_code, problem_id=problem_id
        )

        if (last_eval.execution_succeeded
                and last_eval.status_optimal
                and last_eval.objective_matches):
            break

    token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
    return last_eval, rounds_used, token_usage, last_code


async def _run_agentic_openai(
    client,
    model: str,
    problem_data: dict,
    array_path,
    evaluator,
    problem_id: str,
    max_rounds: int,
    deployment_name: str,
    offload_mode: str = "instance_data",
) -> tuple:
    """
    Run the agentic-offload solve loop for a single problem via OpenAI API.
    Dispatches to two-phase pipeline for problems without instance_data.
    Returns (last_eval_result, rounds_used, token_usage_dict, last_response).
    """
    if offload_mode == "self_review":
        return await _run_self_review_openai(
            client=client,
            model=model,
            problem_data=problem_data,
            evaluator=evaluator,
            problem_id=problem_id,
            max_rounds=max_rounds,
            deployment_name=deployment_name,
        )

    if offload_mode == "structured_json":
        return await _run_structured_json_openai(
            client=client,
            model=model,
            problem_data=problem_data,
            extracted_path=array_path,
            evaluator=evaluator,
            problem_id=problem_id,
            max_rounds=max_rounds,
            deployment_name=deployment_name,
        )

    if offload_mode == "direct_solve":
        return await _run_direct_solve_openai(
            client=client,
            model=model,
            problem_data=problem_data,
            evaluator=evaluator,
            problem_id=problem_id,
            max_rounds=max_rounds,
            deployment_name=deployment_name,
        )

    if offload_mode == "two_phase":
        return await _run_two_phase_openai(
            client=client,
            model=model,
            problem_data=problem_data,
            extracted_path=array_path,
            evaluator=evaluator,
            problem_id=problem_id,
            max_rounds=max_rounds,
            deployment_name=deployment_name,
        )

    agentic_prompt = build_agentic_prompt(problem_data, array_path)
    total_prompt_tokens = 0
    total_completion_tokens = 0

    messages = [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user",   "content": agentic_prompt},
    ]

    last_eval = None
    last_response = ""
    rounds_used = 0

    for round_idx in range(1, max_rounds + 1):
        rounds_used = round_idx
        await asyncio.sleep(0.1)  # rate-limit buffer
        try:
            response = await _api_call_with_retry(
                client, model=deployment_name,
                messages=messages,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            last_response = response.choices[0].message.content
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
        except Exception as e:
            print(f"❌ API Error (agentic round {round_idx}) for {problem_id}: {e}")
            break

        last_eval = await evaluator.evaluate_generated_code_async(
            problem_data, last_response, problem_id=problem_id
        )

        if (last_eval.execution_succeeded
                and last_eval.status_optimal
                and last_eval.objective_matches):
            break

        if round_idx < max_rounds:
            repair = build_repair_prompt(last_eval, data_path=array_path, problem_data=problem_data)
            messages.append({"role": "assistant", "content": last_response})
            messages.append({"role": "user",      "content": repair})

    token_usage = {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens}
    return last_eval, rounds_used, token_usage, last_response


async def process_single_problem(client, model, file_path, evaluator, semaphore, results_list, save_lock, output_path, existing_lookup=None, args=None):
    async with semaphore:
        try:
            try:
                data = load_json(file_path)
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                return None
            
            # Check existing results
            pid = os.path.basename(file_path)
            if existing_lookup and pid in existing_lookup:
                existing = existing_lookup[pid]
                # Check if metadata matches
                # We compare a subset or specific fields to ensure it's the same problem
                # data["meta"] usually contains "problem_type", "num_vars", etc.
                if existing.get("meta") == data.get("meta"):
                    print(f"⏭️  Skipping {pid} (already processed)")
                    return existing

            inference_option = getattr(args, "inference_option", "default") if args else "default"
            save_variables = getattr(args, "save_variables", False) if args else False

            print(f"🔄 Processing {os.path.basename(file_path)}...")

            # ----------------------------------------------------------------
            # Agentic-offload path (includes self-review, structured-json)
            # ----------------------------------------------------------------
            if inference_option in ("agentic-offload", "self-review", "structured-json"):
                max_rounds = getattr(args, "agentic_max_rounds", 3) if args else 3
                agentic_cache_dir = getattr(args, "agentic_cache_dir", DEFAULT_ARRAY_CACHE_DIR) if args else DEFAULT_ARRAY_CACHE_DIR

                # Determine offload mode based on inference option
                if inference_option == "self-review":
                    data_path = None
                    offload_mode = "self_review"
                elif inference_option == "structured-json":
                    cache_dir = Path(agentic_cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(os.path.basename(file_path)).stem
                    data_path = (cache_dir / f"{stem}_extracted.json").resolve()
                    offload_mode = "structured_json"
                else:
                    try:
                        data_path, offload_mode = persist_agentic_data(
                            data, os.path.basename(file_path), cache_dir=agentic_cache_dir
                        )
                    except Exception as e:
                        print(f"❌ Data serialization error for {file_path}: {e}")
                        return None

                deployment_name = resolve_chat_deployment(model)
                try:
                    eval_result, rounds_used, token_usage, content = await _run_agentic_openai(
                        client=client,
                        model=model,
                        problem_data=data,
                        array_path=data_path,
                        evaluator=evaluator,
                        problem_id=os.path.basename(file_path),
                        max_rounds=max_rounds,
                        deployment_name=deployment_name,
                        offload_mode=offload_mode,
                    )
                except Exception as e:
                    print(f"❌ Agentic loop error for {file_path}: {e}")
                    import traceback; traceback.print_exc()
                    return None

                if eval_result is None:
                    print(f"⚠️  No evaluation produced for {file_path}")
                    return None

                filtered_data = _filter_result_data(data, save_variables=save_variables)
                generated_execution = {
                    "objective_value": eval_result.generated_solution.objective_value,
                    "status": eval_result.generated_solution.status,
                    "stderr": eval_result.generated_solution.stderr,
                }
                if save_variables:
                    generated_execution["variable_values"] = eval_result.generated_solution.variable_values

                result = {
                    "problem_id": os.path.basename(file_path),
                    "meta": data.get("meta", {}),
                    "gurobi_result": filtered_data.get("gurobi_result", {}),
                    "model": model,
                    "LLM_description": data.get("LLM_description"),
                    "gold_solution": data.get("gold_solution", ""),
                    "generated_code": content,
                    "evaluation_result": {
                        "execution_succeeded": eval_result.execution_succeeded,
                        "status_optimal": eval_result.status_optimal,
                        "objective_error": eval_result.objective_error,
                        "objective_matches": eval_result.objective_matches,
                        "execution_time": eval_result.execution_time,
                        "generated_status": eval_result.generated_solution.status,
                        "reference_optimum": eval_result.reference_optimum,
                        "agentic_rounds_used": rounds_used,
                    },
                    "generated_execution": generated_execution,
                    "agentic_info": {
                        "data_path": str(data_path),
                        "offload_mode": offload_mode,
                        "rounds_used": rounds_used,
                        "max_rounds": max_rounds,
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0),
                    },
                }

            # ----------------------------------------------------------------
            # Standard 1-pass path (default)
            # ----------------------------------------------------------------
            else:
                prompt = build_prompt(data, args)
                if prompt is None:
                    print(f"⚠️  Skipping {file_path}: no prompt available.")
                    return None

                # Call OpenAI
                try:
                    await asyncio.sleep(0.1)
                    deployment_name = resolve_chat_deployment(model)
                    response = await _api_call_with_retry(
                        client, model=deployment_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                    )
                    content = response.choices[0].message.content
                    usage = response.usage
                except Exception as e:
                    print(f"❌ API Error for {file_path}: {e}")
                    return None

                # Evaluate
                try:
                    eval_result = await evaluator.evaluate_generated_code_async(
                        data,
                        content,
                        problem_id=os.path.basename(file_path)
                    )
                except Exception as e:
                    print(f"❌ Evaluation Error for {file_path}: {e}")
                    import traceback; traceback.print_exc()
                    return None

                filtered_data = _filter_result_data(data, save_variables=save_variables)
                generated_execution = {
                    "objective_value": eval_result.generated_solution.objective_value,
                    "status": eval_result.generated_solution.status,
                    "stderr": eval_result.generated_solution.stderr,
                }
                if save_variables:
                    generated_execution["variable_values"] = eval_result.generated_solution.variable_values

                result = {
                    "problem_id": os.path.basename(file_path),
                    "meta": data.get("meta", {}),
                    "gurobi_result": filtered_data.get("gurobi_result", {}),
                    "model": model,
                    "LLM_description": data.get("LLM_description"),
                    "gold_solution": data.get("gold_solution", ""),
                    "generated_code": content,
                    "evaluation_result": {
                        "execution_succeeded": eval_result.execution_succeeded,
                        "status_optimal": eval_result.status_optimal,
                        "objective_error": eval_result.objective_error,
                        "objective_matches": eval_result.objective_matches,
                        "execution_time": eval_result.execution_time,
                        "generated_status": eval_result.generated_solution.status,
                        "reference_optimum": eval_result.reference_optimum,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    },
                    "generated_execution": generated_execution,
                }

            # Save result incrementally
            async with save_lock:
                results_list.append(result)
                write_results(results_list, output_path)

            icon = "✅" if (eval_result.objective_matches and eval_result.status_optimal) else "❌"
            print(f"   {icon} Processed {os.path.basename(file_path)}")
            return result
            
        except Exception as e:
            print(f"❌ CRITICAL UNHANDLED ERROR processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

def run_vllm_evaluation(args, files_to_process, evaluator):
    if not _ensure_vllm():
        print("❌ Error: vLLM is not installed. Install with: pip install vllm")
        return []

    print(f"🔧 Initializing vLLM with model: {args.model}")
    print(f"   Using tensor_parallel_size={args.tensor_parallel_size} GPUs")
    try:
        # Build LLM config with user-specified parameters
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": getattr(args, 'max_model_len', 16384),
            "trust_remote_code": True,
            "dtype": "auto"
        }
        
        # Add max_num_seqs if specified (helps with KV cache management)
        if hasattr(args, 'max_num_seqs') and args.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = args.max_num_seqs
            print(f"   Using max_num_seqs={args.max_num_seqs}")
        
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize vLLM: {e}")
        return []

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=getattr(args, 'max_model_len', 16384),
    )

    print(f"🚀 Starting evaluation on {len(files_to_process)} files using vLLM")
    
    # Load all data first
    file_data_prompt_tuples = []  # Store (file_path, data, prompt) together
    prompts = []
    
    is_agentic = getattr(args, "inference_option", "default") == "agentic-offload"
    agentic_cache_dir = getattr(args, "agentic_cache_dir", DEFAULT_ARRAY_CACHE_DIR)

    print("Loading problems...")
    for file_path in files_to_process:
        try:
            data = load_json(file_path)

            if is_agentic:
                # Serialize data (instance_data or arrays); build compact prompt.
                data_path, _ = persist_agentic_data(
                    data, os.path.basename(file_path), cache_dir=agentic_cache_dir
                )
                prompt = build_agentic_prompt(data, data_path)
                file_data_prompt_tuples.append((file_path, data, prompt))
                prompts.append(prompt)
            else:
                prompt = build_prompt(data, args)
                if prompt is not None:
                    file_data_prompt_tuples.append((file_path, data, prompt))
                    prompts.append(prompt)
                else:
                    print(f"⚠️  Skipping {file_path}: 'LLM_description' field missing.")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

    if not prompts:
        print("No prompts to process.")
        return []

    print(f"🔄 Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        return []
    
    elapsed = time.time() - start_time
    print(f"✅ Generation completed in {elapsed:.1f}s")

    results = []
    print("Evaluating solutions...")
    
    # Batch save interval to reduce I/O overhead
    SAVE_INTERVAL = 50
    total_outputs = len(outputs)
    
    for i, output in enumerate(outputs):
        file_path, data, current_prompt = file_data_prompt_tuples[i]
        generated_text = output.outputs[0].text

        # Real token counts from vLLM
        vllm_prompt_tokens = len(output.prompt_token_ids)
        vllm_completion_tokens = len(output.outputs[0].token_ids)

        # Evaluate
        eval_result = evaluator.evaluate_generated_code(
            data,
            generated_text,
            problem_id=os.path.basename(file_path)
        )

        # Get save_variables flag from args
        save_variables = getattr(args, "save_variables", False)

        # Filter gurobi_result to only include objective
        filtered_data = _filter_result_data(data, save_variables=save_variables)

        # Construct result dictionary
        generated_execution = {
            "objective_value": eval_result.generated_solution.objective_value,
            "status": eval_result.generated_solution.status,
            "stdout": eval_result.generated_solution.stdout,  # Add stdout for debugging
            "stderr": eval_result.generated_solution.stderr,
        }
        if save_variables:
            generated_execution["variable_values"] = eval_result.generated_solution.variable_values

        result = {
            "problem_id": os.path.basename(file_path),
            "meta": data.get("meta", {}),
            "gurobi_result": filtered_data.get("gurobi_result", {}),
            "model": args.model,
            "LLM_description": data.get("LLM_description"),  # Save original problem description
            "gold_solution": data.get("gold_solution", ""),
            "generated_code": generated_text,
            "evaluation_result": {
                "execution_succeeded": eval_result.execution_succeeded,
                "status_optimal": eval_result.status_optimal,
                "objective_error": eval_result.objective_error,
                "objective_matches": eval_result.objective_matches,
                "execution_time": eval_result.execution_time,
                "generated_status": eval_result.generated_solution.status,
                "reference_optimum": eval_result.reference_optimum,
                "prompt_tokens": vllm_prompt_tokens,
                "completion_tokens": vllm_completion_tokens,
            },
            "generated_execution": generated_execution,
        }
        # Attach agentic metadata for vLLM single-pass agentic runs
        if is_agentic:
            result["agentic_info"] = {
                "array_path": str(file_data_prompt_tuples[i][2]),  # prompt doubles as path indicator
                "rounds_used": 1,
                "max_rounds": 1,
                "prompt_tokens": vllm_prompt_tokens,
                "completion_tokens": vllm_completion_tokens,
                "total_tokens": vllm_prompt_tokens + vllm_completion_tokens,
            }
        results.append(result)
        
        # Save in batches to reduce I/O overhead (every SAVE_INTERVAL or at the end)
        if (i + 1) % SAVE_INTERVAL == 0 or (i + 1) == total_outputs:
            write_results(results, args.results_file)
            print(f"   📝 Saved {len(results)}/{total_outputs} results...")
       
        
        icon = "✅" if (eval_result.status_optimal and eval_result.objective_matches) else "❌"
        # Optional: Print progress
        # print(f"   {icon} Evaluated {os.path.basename(file_path)}")

    return results

async def run_openai_evaluation(args, files_to_process, evaluator):
    from main.utils import is_claude_model, _get_claude_client
    if is_claude_model(args.model):
        # Use a Claude-to-OpenAI adapter
        client = _ClaudeOpenAIAdapter(args.model)
    else:
        client = get_async_openai_client(args.model)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # For incremental saving
    results_list = []
    
    # 1. Load existing results to skip processed files
    existing_lookup = {}
    if os.path.exists(args.results_file):
        try:
            loaded = load_json(args.results_file)
            if isinstance(loaded, list):
                results_list = loaded
                # Create lookup for fast skipping
                for r in results_list:
                    pid = r.get("problem_id")
                    if pid:
                        existing_lookup[pid] = r
                print(f"Loaded {len(results_list)} existing results.")
        except Exception as e:
            print(f"⚠️  Could not load existing results from {args.results_file}: {e}")

    save_lock = asyncio.Lock()

    print(f"🚀 Starting OpenAI evaluation on {len(files_to_process)} files using {args.model}")
    
    tasks = []
    for file_path in files_to_process:
        # Check if already processed
        pid = os.path.basename(file_path)
        if pid in existing_lookup:
            # We already have a result for this ID.
            # Optional: verify metadata match if needed, but for now we skip based on ID.
            # To be robust as requested, we could verify prompt or meta. 
            # But loading the file here to check meta defeats the purpose of skipping.
            # We assume filenames are unique per run.
            # If the user wants strict meta check, they should clear results.
            # But let's respect the "if problem id and metadata match" request loosely:
            # We assume if ID matches, it is the same problem.
            # If we really want to check meta, we'd do it inside process_single_problem.
            # But we can also just skip adding the task entirely if we trust ID.
            # Given the request, let's pass the lookup to process_single_problem 
            # and let it decide after loading the file (safest).
            pass

        task = process_single_problem(
            client,
            args.model,
            file_path,
            evaluator,
            semaphore,
            results_list,
            save_lock,
            args.results_file,
            existing_lookup=existing_lookup,
            args=args,
        )
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    return results_list

async def run_evaluation(args):
    evaluator = GeneratedSolutionEvaluator()
    
    files_to_process = []
    path = Path(args.dataset_path)
    if path.is_file() and path.suffix == ".json":
        # Single file mode
        files_to_process.append(path)
    elif path.is_dir():
        # Directory mode: recursive search for .json files
        files_to_process.extend(list(path.rglob("*.json")))
    else:
        print(f"❌ Error: Path {args.dataset_path} not found or not a JSON file/directory.")
        return

    # Deterministic ordering
    files_to_process = sorted(files_to_process)

    # Detect if model is API-served or local vLLM
    from main.utils import is_api_model, is_claude_model
    if is_api_model(args.model):
        results = await run_openai_evaluation(args, files_to_process, evaluator)
    else:
        results = run_vllm_evaluation(args, files_to_process, evaluator)
        # Clean up lock file if it was created (though we used asyncio.Lock, 
        # but if we used file-based locking we would delete it here. 
        # Since we use asyncio.Lock memory object, no file cleanup needed for locking mechanism itself.
        # But if the user meant "lock file" as in the partial results file, we KEEP it.)

    if not results:
        print("No results to save.")
        return

    # Results are already saved incrementally, but we can print the final summary
    print(f"\n💾 Saved {len(results)} results to {args.results_file}")
    print(
        "\nTo analyze results, run: "
        f"python main/evaluation/analyze_results.py --results-path {args.results_file}"
    )

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on optimization problems.")
    parser.add_argument("--dataset-path", default="synthetic_dataset/", help="Path to a JSON file or directory of JSON files.")
    parser.add_argument("--model", default="gpt-5-nano", help="Model name (OpenAI) or path (vLLM)")
    parser.add_argument("--max-concurrent", type=int, default=1200, help="Max concurrent API requests (OpenAI only).")
    parser.add_argument("--max-completion-tokens", type=int, default=15000, help="Max output tokens per API call. Default: 15000. Use 50000 for large problems.")
    parser.add_argument("--output-path", default="results/model_comparison", help="Base path to save evaluation results (folder will be suffixed with model name).")
    parser.add_argument(
        "--inference-option",
        type=str,
        default="default",
        choices=["default", "agentic-offload",
                 "cot", "self-review", "structured-json"],
        help=(
            "Inference option: 'default' (no examples), "
            "'agentic-offload' (array offload — numeric data stored in local file, no embeddings in prompt), "
            "'cot' (chain-of-thought — formulation checklist before code), "
            "'self-review' (agentic: solve → self-review → optional repair), "
            "'structured-json' (agentic: extract JSON formulation → solve from JSON)."
        ),
    )
    parser.add_argument(
        "--agentic-max-rounds",
        type=int,
        default=3,
        help="Max repair rounds for agentic-offload mode (OpenAI path only). Default: 3.",
    )
    parser.add_argument(
        "--agentic-cache-dir",
        type=str,
        default=str(DEFAULT_ARRAY_CACHE_DIR),
        help="Directory to cache serialized problem array files for agentic-offload mode.",
    )
    parser.add_argument("--file_name", default=None, help="Process only this file (e.g. trans_001.json). If omitted, process all files in --dataset-path.")
    parser.add_argument(
        "--save-variables",
        action="store_true",
        default=False,
        help="If set, save variable values in results. By default, only objective values are saved.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (vLLM only). Default: 1. Use 2-8 for multi-GPU acceleration.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization ratio for vLLM (0.0-1.0). Default: 0.85. Increase to 0.9-0.95 if you have KV cache issues.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to process in parallel (vLLM only). Lower values reduce KV cache usage. Default: auto.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model context length for vLLM (default: 16384).",
    )
    args = parser.parse_args()

    # Set global max completion tokens from CLI
    global MAX_COMPLETION_TOKENS
    MAX_COMPLETION_TOKENS = args.max_completion_tokens

    # Normalise hyphenated arg names to underscore attributes
    args.agentic_max_rounds = args.agentic_max_rounds
    args.agentic_cache_dir = Path(args.agentic_cache_dir)

    # Determine output file path
    output_path = Path(args.output_path)
    if output_path.suffix == '.json':
        # Treat as direct file path
        args.results_file = output_path
        args.output_dir = output_path.parent
    else:
        # Treat as directory, generate filename
        args.output_dir = output_path
        if args.file_name is None:
            suffix = args.model.split("/")[-1]
            option = args.inference_option
            if option != "default":
                suffix = f"{suffix}_{option}"
            args.results_file = args.output_dir / f"{args.dataset_path.split('/')[-1]}_{suffix}.json"
        else:
            args.results_file = args.output_dir / f"{args.file_name}.json"
    args.results_file.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_evaluation(args))

if __name__ == "__main__":
    main()
