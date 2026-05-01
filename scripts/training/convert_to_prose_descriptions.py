#!/usr/bin/env python3
"""
Convert tabular/structured annex descriptions to prose-embedded format.

For each problem JSON, the LLM-written memo body is kept unchanged.
Only the data annex section is replaced with prose that embeds the numbers
in natural language with randomized ordering — making the extraction task
semantically harder (analogous to resource allocation).

Transportation:
  - Cost matrix table → per-source sentences with randomized destination order
  - Demand list → sentence with randomized destination order
  - Seed: hash(problem_id) for reproducibility

JSSP:
  - Separate processing_times + machine_assignments lists → combined per-job sentences
  - Job listing order is randomized (operations within a job stay in sequence)
  - Seed: hash(problem_id) for reproducibility

Outputs saved to parallel *_prose/ directories:
  synthetic_dataset/Template_train/transportation/small_prose/
  synthetic_dataset/Template/basic_template/transportation/small_prose/
  synthetic_dataset/Template_train/jssp/small_prose/
  synthetic_dataset/Template/basic_template/jssp/small_prose/
"""

import json
import random
import re
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Transportation prose builder ────────────────────────────────────────────

def build_transportation_prose_annex(inst: dict, problem_id: str) -> str:
    """Build a prose annex for a transportation problem with randomized ordering."""
    rng = random.Random(hash(problem_id))

    supplies = inst["supplies"]
    demands = inst["demands"]
    cost_matrix = inst["cost_matrix"]
    num_sources = inst["num_sources"]
    num_dests = inst["num_destinations"]
    integer_flows = inst.get("integer_flows", False)

    # Randomize destination order for demand listing
    dest_order_demands = list(range(num_dests))
    rng.shuffle(dest_order_demands)

    demand_parts = [
        f"Store D{j} needs {demands[j]} units"
        for j in dest_order_demands
    ]
    demand_line = "Destination requirements: " + "; ".join(demand_parts) + "."

    # Build per-source sentences with randomized destination order
    cost_lines = ["Shipping costs by source:"]
    for i in range(num_sources):
        dest_order = list(range(num_dests))
        rng.shuffle(dest_order)
        cost_parts = [
            f"D{j} at ${cost_matrix[i][j]:.2f}/unit"
            for j in dest_order
        ]
        shipment_note = " (integer quantities)" if integer_flows else ""
        cost_lines.append(
            f"  Warehouse S{i} (supply: {supplies[i]} units{shipment_note}): "
            + ", ".join(cost_parts)
        )

    return "Annex \u2013 Shipping Data:\n" + demand_line + "\n\n" + "\n".join(cost_lines)


def _find_annex(desc: str, keywords: list[str]) -> int | None:
    """Find the last occurrence of an annex section using any dash variant (case-insensitive)."""
    # Match "Annex <dash> <keyword>" where dash is en/em dash or hyphen
    for keyword in keywords:
        pattern = rf'Annex\s*[-\u2013\u2014]\s*{re.escape(keyword)}'
        matches = list(re.finditer(pattern, desc, re.IGNORECASE))
        if matches:
            return matches[-1].start()
    # Fallback: last occurrence of bare "Annex" (case-insensitive)
    matches = list(re.finditer(r'Annex', desc, re.IGNORECASE))
    if matches:
        return matches[-1].start()
    return None


def convert_transportation(problem: dict, problem_id: str) -> dict:
    """Replace tabular annex with prose in a transportation problem JSON."""
    desc = problem.get("LLM_description", "")
    inst = problem["instance_data"]

    annex_marker = _find_annex(desc, ["Shipping Data", "Data"])

    if annex_marker is None:
        print(f"  WARNING: No annex found in {problem_id}, appending prose")
        annex_marker = len(desc)
        memo_body = desc.strip() + "\n\n"
    else:
        memo_body = desc[:annex_marker]

    prose_annex = build_transportation_prose_annex(inst, problem_id)
    new_desc = memo_body + prose_annex

    result = dict(problem)
    result["LLM_description"] = new_desc
    return result


# ── JSSP prose builder ──────────────────────────────────────────────────────

def build_jssp_prose_annex(inst: dict, problem_id: str) -> str:
    """Build a prose annex for a JSSP problem with randomized job order."""
    rng = random.Random(hash(problem_id))

    n_jobs = inst["n_jobs"]
    n_machines = inst["n_machines"]
    processing_times = inst["processing_times"]
    machine_assignments = inst["machine_assignments"]

    # Randomize job listing order
    job_order = list(range(n_jobs))
    rng.shuffle(job_order)

    lines = [
        f"- Number of jobs: {n_jobs}",
        f"- Number of machines (workstations): {n_machines}",
        "- Job-by-job production schedule (operations must be performed in the listed sequence):",
    ]

    for j in job_order:
        ops = []
        for op in range(n_machines):
            machine = machine_assignments[j][op]
            time = processing_times[j][op]
            ops.append(f"Op{op + 1} on Machine {machine} for {time} time units")
        lines.append(f"  Job {j + 1}: " + " \u2192 ".join(ops))

    return "Annex \u2013 Production Data\n" + "\n".join(lines)


def convert_jssp(problem: dict, problem_id: str) -> dict:
    """Replace structured lists with prose in a JSSP problem JSON."""
    desc = problem.get("LLM_description", "")
    inst = problem["instance_data"]

    annex_marker = _find_annex(desc, ["Production Data", "Data"])

    # Fallback: many JSSP descriptions have no "Annex" header — data starts with "- Number of jobs"
    if annex_marker is None:
        for marker in ["- Number of jobs", "Number of jobs"]:
            idx = desc.rfind(marker)
            if idx != -1:
                annex_marker = idx
                break

    if annex_marker is None:
        print(f"  WARNING: No annex found in {problem_id}, appending prose")
        memo_body = desc.strip() + "\n\n"
    else:
        memo_body = desc[:annex_marker]

    prose_annex = build_jssp_prose_annex(inst, problem_id)
    new_desc = memo_body + prose_annex

    result = dict(problem)
    result["LLM_description"] = new_desc
    return result


# ── Directory processing ─────────────────────────────────────────────────────

CONVERTERS = {
    "transportation": convert_transportation,
    "jssp": convert_jssp,
}

DIRECTORIES = {
    "transportation": [
        (
            BENCH_ROOT / "synthetic_dataset" / "Template_train" / "transportation" / "small",
            BENCH_ROOT / "synthetic_dataset" / "Template_train" / "transportation" / "small_prose",
        ),
        (
            BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "transportation" / "small",
            BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "transportation" / "small_prose",
        ),
    ],
    "jssp": [
        (
            BENCH_ROOT / "synthetic_dataset" / "Template_train" / "jssp" / "small",
            BENCH_ROOT / "synthetic_dataset" / "Template_train" / "jssp" / "small_prose",
        ),
        (
            BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "jssp" / "small",
            BENCH_ROOT / "synthetic_dataset" / "Template" / "basic_template" / "jssp" / "small_prose",
        ),
    ],
}


def process_directory(src_dir: Path, dst_dir: Path, category: str):
    converter = CONVERTERS[category]
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.json"))
    print(f"  {src_dir.name} -> {dst_dir.name}: {len(files)} files")

    ok = 0
    warn = 0
    for pf in files:
        with open(pf) as f:
            problem = json.load(f)

        problem_id = pf.stem
        try:
            converted = converter(problem, problem_id)
            ok += 1
        except Exception as e:
            print(f"    ERROR {problem_id}: {e}")
            warn += 1
            continue

        with open(dst_dir / pf.name, "w") as f:
            json.dump(converted, f, indent=2)

    print(f"    Done: {ok} converted, {warn} errors")


def main():
    categories = sys.argv[1:] if len(sys.argv) > 1 else list(DIRECTORIES.keys())

    for category in categories:
        if category not in DIRECTORIES:
            print(f"Unknown category: {category}", file=sys.stderr)
            continue

        print(f"\n{'=' * 60}")
        print(f"Converting: {category}")
        print(f"{'=' * 60}")

        for src_dir, dst_dir in DIRECTORIES[category]:
            if not src_dir.exists():
                print(f"  Skipping (not found): {src_dir}")
                continue
            process_directory(src_dir, dst_dir, category)


if __name__ == "__main__":
    main()
