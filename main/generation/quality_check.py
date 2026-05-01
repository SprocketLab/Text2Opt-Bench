import json
import sys
import os
import asyncio
from typing import Optional, Dict, Any
from tqdm import tqdm

from main.utils import get_async_openai_client, resolve_chat_deployment, load_json, save_json

# Shared async OpenAI client
aclient = get_async_openai_client()


QUALITY_SYSTEM_PROMPT = """You are an expert in Operations Research (OR) and mathematical optimization dataset curation.
Your task is to evaluate the quality of the natural-language problem description stored under `LLM_description` (the runtime prompt) intended to specify an optimization model (Linear Programming, Mixed-Integer Programming, etc.).

Each JSON object describes a single optimization problem and may contain:
- variables, constraints, meta (type LP/QP, goal, etc.)
- gold_solution: a reference Gurobi implementation
- LLM_description: the natural-language problem statement given to the model (replaces the legacy `prompt` field)
- instance_data: the specific data for the problem type (e.g. transportation problem, resource allocation problem, etc.)

    Input: A JSON object containing the natural language problem description (`LLM_description`) and reference `gold_solution` (Gurobi code).
Output: A JSON evaluation object assessing Inducibility, Ambiguity, Redundancy, and Realism.
"""


def _build_quality_user_prompt(problem_json: Dict[str, Any]) -> str:
    """Format a user prompt describing the sample to be evaluated.

    Only includes LLM_description, gold_solution, and meta fields.
    Excludes instance_data, variables, and constraints to test whether
    the problem can be induced directly from the description and gold code.
    """
    # Only keep LLM_description, gold_solution, and meta
    fields_to_keep = ("LLM_description", "gold_solution", "meta")
    filtered_json = {k: v for k, v in problem_json.items() if k in fields_to_keep}
    full_json_str = json.dumps(filtered_json, indent=2)

    return f"""You are evaluating a synthetic optimization problem.
The goal is to determine if the natural-language description in `LLM_description` is a high-quality specification for the "gold_solution" (Gurobi code).

Note: The runtime system wraps `LLM_description` in a standard template when sending to the model (instructions, code block requirements, etc.). Assume such a template exists; focus your evaluation on whether the `LLM_description` text itself fully specifies the optimization problem.

### Full JSON Problem
{full_json_str}

### Evaluation Task
Evaluate the `LLM_description` (natural-language problem text) based on the following metrics:

1. **Prompt Inducibility (0-100)**:
   - Score 100 if a human expert could mechanically reconstruct the exact variables, constraints, and objective from the description without guessing.
   - Score 0 if the description is vague, wrong, or completely missing critical values.
   - Focus on technical precision: are bounds, units, and logical relationships clear?

2. **Ambiguity (Boolean)**:
   - true: The description has missing information, contradictions, or multiple valid mathematical interpretations.
   - false: The description is strictly well-posed.

3. **Redundancy (0-100)**:
   - Score 0 if the text is concise and efficient.
   - Score 100 if the text is highly repetitive, verbose, or filled with irrelevant fluff.
   - A moderate score (e.g., 20-40) is acceptable if the extra text adds necessary context/flavor.

4. **Realism (0-100)**:
   - Score 100 if the scenario feels like a genuine real-world OR application (e.g., supply chain, finance, scheduling) with plausible constraints.
   - Score 0 if it feels like a generated abstract math problem (e.g., "Item 1", "Item 2") or has nonsensical logic.

Return ONLY a JSON object with these fields:
{{
  "prompt_inducibility": <int>,
  "prompt_inducibility_comment": "<string, max 200 chars>",
  "ambiguity": <bool>,
  "redundancy": <int>,
  "redundancy_comment": "<string, max 200 chars>",
  "realism": <int>,
  "realism_comment": "<string, max 200 chars>"
}}
"""


async def evaluate_sample_async(problem_json: Dict[str, Any], model: str = "gpt-5") -> Optional[Dict[str, Any]]:
    """Call the LLM once to evaluate a single JSON problem."""
    user_content = _build_quality_user_prompt(problem_json)
    try:
        extra_body = {"reasoning_effort": "high"} if model in ("gpt-5.1", "gpt-5") else None
        deployment_name = resolve_chat_deployment(model)
        completion = await aclient.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            **({"extra_body": extra_body} if extra_body is not None else {}),
        )
        raw = completion.choices[0].message.content
        # Best-effort JSON parse; caller can store raw text if this fails
        try:
            return json.loads(raw)
        except Exception:
            return {"raw_response": raw}
    except Exception as e:
        print(f"Error during quality evaluation: {e}")
        return None


async def _process_file_async(path: str, semaphore: Optional[asyncio.Semaphore] = None, model: str = "gpt-5") -> None:
    """Evaluate a single JSON file and attach quality metadata under meta.quality."""

    async def _inner():
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return

        # Skip files that don't have a description (nothing to evaluate)
        description = data.get("LLM_description")
        if not description:
            print(f"Skipping {path}: 'LLM_description' field missing or empty.")
            return

        result = await evaluate_sample_async(data, model=model)
        if not result:
            print(f"Skipping write for {path}: quality eval failed.")
            return

        if "meta" not in data:
            data["meta"] = {}
        data["meta"]["quality"] = result

        try:
            save_json(path, data)
        except Exception as e:
            print(f"Error writing {path}: {e}")

    if semaphore:
        async with semaphore:
            await asyncio.sleep(0.1)
            await _inner()
    else:
        await _inner()


async def run_quality_check(root_dir: str, max_concurrent: int = 1000, model: str = "gpt-5") -> None:
    """Entry point used by run_pipeline.

    Walks `root_dir` recursively, finds all `.json` files, and evaluates
    their quality concurrently with a semaphore, similar to
    `generate_prompts_concurrently`.
    """
    print(f"\nStarting quality check for JSON files in '{root_dir}' using {max_concurrent} concurrent tasks...")

    json_files = []
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".json"):
                json_files.append(os.path.join(dirpath, fn))

    json_files.sort()
    if not json_files:
        print("No JSON files found for quality check.")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    bar = tqdm(total=len(json_files), desc="Quality", unit="file")

    async def _run_with_progress(path):
        try:
            await _process_file_async(path, semaphore=semaphore, model=model)
        finally:
            bar.update(1)

    for p in json_files:
        tasks.append(asyncio.create_task(_run_with_progress(p)))
        await asyncio.sleep(0.05)

    await asyncio.gather(*tasks)
    bar.close()
    print("Quality check completed.")
