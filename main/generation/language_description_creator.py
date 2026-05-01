"""
Language description creator for optimization problems.

This module handles the orchestration of LLM-based prompt generation:
- Async/concurrent API calls
- File I/O and progress tracking
- Rate limiting and retries

The actual system prompts are defined in each generator class (SYSTEM_PROMPT attribute).
"""

import sys
import os
import asyncio
import glob
from tqdm import tqdm
from main.utils import (
    get_async_openai_client,
    resolve_chat_deployment,
    count_tokens,
    load_json,
    save_json,
)

# Import generators to access their SYSTEM_PROMPT class attributes
from main.generation.unstructured.resource_allocation_lp_generator import ResourceAllocationGenerator
from main.generation.unstructured.bipartite_matching_generator import BipartiteMatchingGenerator
from main.generation.basic_template.disaster_response_generator import DisasterResponseGenerator
from main.generation.unstructured.block_diagonal_lp_generator import BlockDiagonalLPGenerator
from main.generation.induce_constraint.facility_location_generator import FacilityLocationGenerator
from main.generation.induce_constraint.power_transmission_generator import PowerTransmissionGenerator
from main.generation.basic_template.vrptw_generator import VRPTWGenerator
from main.generation.basic_template.jssp_generator import JSSPGenerator
from main.generation.unstructured.capital_budgeting_generator import CapitalBudgetingGenerator
from main.generation.induce_constraint.queuing_staffing_generator import QueuingStaffingGenerator
from main.generation.industrial.modified_facility_location_generator import ModifiedFacilityLocationGenerator
from main.generation.industrial.stochastic_transportation_generator import StochasticTransportationGenerator
from main.generation.industrial.multiobjective_transportation_generator import MultiObjectiveTransportationGenerator
from main.generation.basic_template.rcpsp_generator import RCPSPGenerator
from main.generation.base_problem import BaseProblem

# Initialize shared async OpenAI client
aclient = get_async_openai_client()

# Registry mapping problem types to generator classes
PROBLEM_TYPE_REGISTRY = {
    "resource allocation": ResourceAllocationGenerator,
    "transportation problem": BipartiteMatchingGenerator,
    "disaster response logistics": DisasterResponseGenerator,
    "block diagonal resource allocation": BlockDiagonalLPGenerator,
    "facility location": FacilityLocationGenerator,
    "power transmission": PowerTransmissionGenerator,
    "vehicle routing": VRPTWGenerator,
    "job shop scheduling": JSSPGenerator,
    "capital budgeting": CapitalBudgetingGenerator,
    "queuing staffing": QueuingStaffingGenerator,
    "modified facility location": ModifiedFacilityLocationGenerator,
    "stochastic transportation": StochasticTransportationGenerator,
    "multi-objective transportation": MultiObjectiveTransportationGenerator,
    "resource-constrained project scheduling": RCPSPGenerator,
}


def _get_generator_class(problem_data: dict) -> type:
    """
    Get the generator class for a given problem data.
    
    Args:
        problem_data: The problem data dictionary.
        
    Returns:
        The generator class for the problem type.
    """
    problem_type = (
        problem_data.get("meta", {}).get("problem_type", "") if isinstance(problem_data, dict) else ""
    )
    key = problem_type.lower()
    # Try exact match first
    if key in PROBLEM_TYPE_REGISTRY:
        return PROBLEM_TYPE_REGISTRY[key]
    # Try replacing underscores with spaces
    key_normalized = key.replace("_", " ")
    if key_normalized in PROBLEM_TYPE_REGISTRY:
        return PROBLEM_TYPE_REGISTRY[key_normalized]
    # Fall back to base class
    return BaseProblem


async def generate_description_async(problem_data, model="gpt-5", max_retries=3, return_template=False):
    """
    Generate a natural language description for the optimization problem.

    Uses the generator class's build_prompt_messages() to get the full message structure,
    allowing each problem type to fully customize its prompt format.

    For reasoning-capable models like gpt-5, we request higher reasoning effort
    using the `reasoning_effort` parameter.

    After receiving the LLM response, calls the generator's finish_description() method
    to post-process the description (e.g., fill in template placeholders).

    Args:
        return_template: If True, return (finished_description, raw_template) tuple.
                         raw_template is the unfilled description with {PLACEHOLDER} tokens.
    """
    # Get the generator class and build messages
    generator_class = _get_generator_class(problem_data)
    messages = generator_class.build_prompt_messages(problem_data)

    for attempt in range(max_retries):
        try:
            extra_body = {"reasoning_effort": "high"} if model in ("gpt-5.1", "gpt-5") else None
            deployment_name = resolve_chat_deployment(model)

            completion = await aclient.chat.completions.create(
                model=deployment_name,
                messages=messages,
                **({"extra_body": extra_body} if extra_body is not None else {}),
            )

            # Robust check for content vs string
            if hasattr(completion, 'choices'):
                raw_description = completion.choices[0].message.content
            else:
                # Fallback if the client returns a raw string or unexpected object
                raw_description = str(completion)

            # Post-process the description using the generator's finish_description method
            finished_description = generator_class.finish_description(raw_description, problem_data)
            if return_template:
                return finished_description, raw_description
            return finished_description

        except Exception as e:
            # On any API error, wait and retry
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                await asyncio.sleep(delay)
            else:
                print(f"Error generating description after {max_retries} attempts: {e}")
                return None


async def create_prompt_async(problem_data, model="gpt-5"):
    """Generate description only; formatting is done at evaluation time."""
    llm_description = await generate_description_async(problem_data, model=model)
    if not llm_description:
        # Propagate failure to caller
        return None

    return llm_description


async def process_file_async(file_path, overwrite=False, semaphore=None, model="gpt-5"):
    if semaphore:
        async with semaphore:
            await asyncio.sleep(0.1)  # Rate limit inside semaphore
            return await _process_file_core(file_path, overwrite, model=model)
    else:
        return await _process_file_core(file_path, overwrite, model=model)


async def _process_file_core(file_path, overwrite=False, model="gpt-5"):
    try:
        data = load_json(file_path)

        if not overwrite and data.get("LLM_description"):
            print(f"Skipping {file_path}: LLM_description already exists.")
            return

        # Request both the filled description and the raw placeholder template
        result = await generate_description_async(data, model=model, return_template=True)
        if not result:
            # Do NOT write an error string into the file; leave it unchanged
            print(f"Skipping write for {file_path}: failed to generate prompt.")
            return

        llm_description, llm_description_template = result
        if not llm_description:
            print(f"Skipping write for {file_path}: finish_description returned None.")
            return

        # Store filled description and the raw placeholder template at top level
        data["LLM_description"] = llm_description
        data["LLM_description_template"] = llm_description_template

        # Calculate and store token counts
        if "meta" not in data:
            data["meta"] = {}
        full_tokens = count_tokens(llm_description, model=model)
        data["meta"]["prompt_tokens_with_data"] = full_tokens

        # For template problems with instance_data, also compute tokens
        # for the context-only portion (before Annex/Appendix data section)
        if data.get("instance_data"):
            context_text = llm_description
            for marker in ["Annex", "Appendix", "---\n", "===\n"]:
                idx = llm_description.find(marker)
                if idx > 0:
                    context_text = llm_description[:idx]
                    break
            data["meta"]["prompt_tokens_without_data"] = count_tokens(context_text, model=model)
        else:
            data["meta"]["prompt_tokens_without_data"] = full_tokens

        save_json(file_path, data)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


async def _add_template_to_existing_file(file_path, semaphore=None, model="gpt-5", force=False):
    """
    For files that already have LLM_description but lack LLM_description_template,
    re-run the LLM to get the raw template (with placeholders) and save it alongside.
    Does NOT overwrite or regenerate LLM_description.

    Args:
        force: If True, regenerate even if LLM_description_template already exists.
    """
    async def _core():
        try:
            data = load_json(file_path)

            if data.get("LLM_description_template") and not force:
                return  # already has template

            if not data.get("LLM_description"):
                return  # no filled description either — skip

            generator_class = _get_generator_class(data)
            messages = generator_class.build_prompt_messages(data)

            extra_body = {"reasoning_effort": "high"} if model in ("gpt-5.1", "gpt-5") else None
            deployment_name = resolve_chat_deployment(model)

            completion = await aclient.chat.completions.create(
                model=deployment_name,
                messages=messages,
                **({"extra_body": extra_body} if extra_body is not None else {}),
            )
            raw_description = completion.choices[0].message.content

            # Validate that it contains expected placeholders
            inst = data.get("instance_data", {})
            has_any_placeholder = any(
                "{" + k.upper() + "}" in raw_description for k in inst.keys()
            )
            if not has_any_placeholder:
                print(f"⚠️  No placeholders found in template for {file_path}, skipping.")
                return

            data["LLM_description_template"] = raw_description
            save_json(file_path, data)
            print(f"✅ Added LLM_description_template to {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    if semaphore:
        async with semaphore:
            await asyncio.sleep(0.1)
            await _core()
    else:
        await _core()


async def add_templates_to_existing_dataset(
    dataset_dir: str,
    max_concurrent: int = 10,
    model: str = "gpt-5",
    force: bool = False,
):
    """
    Migrate an existing Template dataset to add LLM_description_template fields.
    Safe to re-run — skips files that already have the template (unless force=True).

    Args:
        force: If True, regenerate templates even for files that already have them.
    """
    import glob as glob_mod
    files = sorted(glob_mod.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True))
    # Only process template problems (those with instance_data)
    template_files = []
    for f in files:
        try:
            d = load_json(f)
            needs_template = d.get("instance_data") and (force or not d.get("LLM_description_template"))
            if needs_template:
                template_files.append(f)
        except Exception:
            pass

    if not template_files:
        print("No files need template migration.")
        return

    action = "Regenerating" if force else "Migrating"
    print(f"{action} {len(template_files)} files to add LLM_description_template...")
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.create_task(_add_template_to_existing_file(f, semaphore=semaphore, model=model, force=force))
        for f in template_files
    ]
    await asyncio.gather(*tasks)
    print("Migration complete.")


def rebuild_descriptions_from_templates(dataset_dir: str):
    """
    Re-derive LLM_description from LLM_description_template for all template files.

    This ensures both fields come from the same base text:
      LLM_description = finish_description(LLM_description_template, problem_data)

    No API calls needed — purely local string replacement.
    """
    import glob as glob_mod
    files = sorted(glob_mod.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True))

    updated = 0
    skipped = 0
    errors = 0

    for f in files:
        try:
            data = load_json(f)
            template = data.get("LLM_description_template", "").strip()
            if not template or not data.get("instance_data"):
                skipped += 1
                continue

            generator_class = _get_generator_class(data)
            filled = generator_class.finish_description(template, data)
            if not filled:
                print(f"  Skipping {f}: finish_description returned None")
                errors += 1
                continue

            data["LLM_description"] = filled
            save_json(f, data)
            updated += 1
        except Exception as e:
            print(f"  Error on {f}: {e}")
            errors += 1

    print(f"Done: {updated} updated, {skipped} skipped, {errors} errors")


async def generate_prompts_concurrently(
    output_dir="synthetic_dataset",
    max_concurrent=10,
    overwrite=False,
    model="gpt-5",
):
    """
    Generate prompts for all JSON files in output_dir using asyncio and Semaphores.
    `model` controls which model is used for prompt creation.
    """
    print(
        f"\nStarting prompt creation for files in '{output_dir}' using {max_concurrent} concurrent tasks (model={model})..."
    )
    
    # Collect all JSON files
    files_to_process = glob.glob(os.path.join(output_dir, "*.json"))
    files_to_process.sort() 
    
    if not files_to_process:
        print("No JSON files found to process.")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    bar = tqdm(total=len(files_to_process), desc="Prompts", unit="file")

    async def _run_with_progress(coro):
        try:
            return await coro
        finally:
            if bar:
                bar.update(1)
    
    for f in files_to_process:
        task = asyncio.create_task(
            _run_with_progress(
                process_file_async(f, overwrite=overwrite, semaphore=semaphore, model=model)
            )
        )
        tasks.append(task)
        # Stagger the start of each task to respect rate limits (e.g. 10 req/s)
        await asyncio.sleep(0.1)
    
    await asyncio.gather(*tasks)
    bar.close()
    print("Prompt creation completed.")
