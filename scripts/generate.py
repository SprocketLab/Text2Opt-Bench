#!/usr/bin/env python3
"""
Generate Text2Opt-Bench problems.

The pipeline runs in two phases per problem:

  1. World state — instantiate the category's generator with sampled size
     parameters, build & solve the underlying Gurobi model, retry on
     infeasibility/timeout, and serialize the structured JSON (variables,
     constraints, instance_data, gurobi_result, gold_solution).

  2. Natural-language description — call an LLM (default gpt-5) to fill the
     category's template prompt and merge the prose back into the JSON.
     Skip with --no-description if you only want the structured data.

Usage:
    # 50 transportation problems with sampled sizes (default ranges)
    python scripts/generate.py transportation -n 50 \\
        --output-dir synthetic_dataset/Template/basic_template/transportation/small

    # JSSP with explicit fixed sizes
    python scripts/generate.py jssp -n 20 --params n_jobs=5 n_machines=4 \\
        --output-dir my_jssp/

    # Stress instances (size near the upper bound)
    python scripts/generate.py queuing_staffing -n 10 --size large \\
        --output-dir synthetic_dataset/Template_large/queuing_staffing/large

    # Skip the description LLM call (e.g., for quick smoke test)
    python scripts/generate.py jssp -n 5 --no-description

    # List supported categories
    python scripts/generate.py --list

Model defaults: gpt-5 for descriptions. Override with --model. The runtime
reads OPENAI_API_KEY from the environment or api_keys/keys.json.

Requires: pip install -r requirements.txt
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main.utils import save_json, load_json
from main.generation.language_description_creator import generate_description_async

from main.generation.basic_template.disaster_response_generator import DisasterResponseGenerator
from main.generation.basic_template.jssp_generator import JSSPGenerator
from main.generation.basic_template.vrptw_generator import VRPTWGenerator
from main.generation.basic_template.rcpsp_generator import RCPSPGenerator
from main.generation.induce_constraint.facility_location_generator import FacilityLocationGenerator
from main.generation.induce_constraint.power_transmission_generator import PowerTransmissionGenerator
from main.generation.induce_constraint.queuing_staffing_generator import QueuingStaffingGenerator
from main.generation.industrial.modified_facility_location_generator import ModifiedFacilityLocationGenerator
from main.generation.industrial.multiobjective_transportation_generator import MultiObjectiveTransportationGenerator
from main.generation.industrial.stochastic_transportation_generator import StochasticTransportationGenerator
from main.generation.unstructured.bipartite_matching_generator import BipartiteMatchingGenerator
from main.generation.unstructured.resource_allocation_lp_generator import ResourceAllocationGenerator


# ─────────────────────────────────────────────────────────────────────────
# Per-category configuration
# Each entry: generator_class, default size ranges {param: (min, max)}, prefix.
# Size ranges sample uniformly per problem; pass --params to override.
# ─────────────────────────────────────────────────────────────────────────
CATEGORIES = {
    "transportation": {
        "class": BipartiteMatchingGenerator,
        "ranges": {
            "small": {"num_sources": (3, 26), "num_dests": (3, 26)},
            "large": {"num_sources": (26, 88), "num_dests": (26, 88)},
        },
        "prefix": "trans",
    },
    "jssp": {
        "class": JSSPGenerator,
        "ranges": {"small": {"n_jobs": (3, 14), "n_machines": (3, 5)}},
        "prefix": "jssp",
    },
    "vrptw": {
        "class": VRPTWGenerator,
        "ranges": {"small": {"n_customers": (5, 20), "n_vehicles": (3, 6)}},
        "prefix": "vrptw",
    },
    "rcpsp": {
        "class": RCPSPGenerator,
        "ranges": {"small": {"n_activities": (6, 25)}},
        "fixed": {"n_resources": 2, "n_modes": 2},
        "prefix": "rcpsp",
    },
    "disaster_response": {
        "class": DisasterResponseGenerator,
        "ranges": {"small": {
            "n_depots": (2, 6), "n_units": (3, 6),
            "n_supplies": (2, 4), "n_days": (1, 4),
        }},
        "prefix": "dr",
    },
    "facility_location": {
        "class": FacilityLocationGenerator,
        "ranges": {"small": {"n_facilities": (3, 20), "n_customers": (5, 50)}},
        "prefix": "fl",
    },
    "power_transmission": {
        "class": PowerTransmissionGenerator,
        "ranges": {"small": {"n_nodes": (4, 18)}},
        "prefix": "pt",
    },
    "queuing_staffing": {
        "class": QueuingStaffingGenerator,
        "ranges": {
            "small": {"n_stations": (3, 55), "n_staff_types": (3, 55)},
            "large": {"n_stations": (55, 90), "n_staff_types": (55, 90)},
        },
        "prefix": "qs",
    },
    "modified_facility_location": {
        "class": ModifiedFacilityLocationGenerator,
        "ranges": {"small": {"n_facilities": (3, 15), "n_customers": (5, 30)}},
        "prefix": "mfl",
    },
    "multiobjective_transportation": {
        "class": MultiObjectiveTransportationGenerator,
        "ranges": {
            "small": {"num_sources": (4, 16), "num_dests": (5, 28)},
            "large": {"num_sources": (25, 40), "num_dests": (40, 65)},
        },
        "prefix": "mot",
    },
    "stochastic_transportation": {
        "class": StochasticTransportationGenerator,
        "ranges": {
            "small": {"num_sources": (3, 8), "num_dests": (4, 14)},
            "large": {"num_sources": (8, 25), "num_dests": (14, 40)},
        },
        "fixed": {"num_scenarios": 15},
        "prefix": "st",
    },
    "resource_allocation": {
        "class": ResourceAllocationGenerator,
        "ranges": {"small": {"n_vars": (2, 20), "n_constrs": (2, 10)}},
        "prefix": "ra",
    },
}


TIME_LIMIT = 300.0
MAX_RETRIES = 8


def sample_params(cfg: dict, size: str, rng: np.random.RandomState) -> dict:
    """Sample size params uniformly from the configured range for `size`."""
    ranges = cfg["ranges"].get(size)
    if ranges is None:
        sys.exit(f"Category does not support --size {size!r} (only: {list(cfg['ranges'])}).")
    params = {k: int(rng.randint(lo, hi + 1)) for k, (lo, hi) in ranges.items()}
    params.update(cfg.get("fixed", {}))
    return params


def parse_param_overrides(items: list[str]) -> dict:
    """Parse `key=value` overrides into a dict. Values are int by default."""
    out = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"--params expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def generate_one(cfg: dict, params: dict, seed: int) -> dict | None:
    """Build, solve, and serialize one problem. Returns None if all retries fail."""
    for attempt in range(MAX_RETRIES):
        np.random.seed(seed + attempt * 10000)
        try:
            gen = cfg["class"](**params)
            gen.generate(time_limit=TIME_LIMIT)
            if gen.solver_status != 2:  # not OPTIMAL
                continue
            data = gen.to_json_dict()
            gold = gen.generate_gurobi_code_reference(data)
            data["gold_solution"] = gold if isinstance(gold, str) else gold.get("solution_code", "")
            return data
        except (TimeoutError, RuntimeError):
            continue
        except Exception as e:
            print(f"    attempt {attempt+1}: unexpected error: {e}")
            continue
    return None


async def add_descriptions(problems: list[tuple[Path, dict]], model: str, max_concurrent: int) -> None:
    """Generate LLM descriptions in parallel for problems missing one."""
    sem = asyncio.Semaphore(max_concurrent)

    async def one(path: Path, data: dict) -> None:
        if data.get("LLM_description"):
            return
        async with sem:
            for attempt in range(5):
                try:
                    result = await generate_description_async(data, model=model, return_template=True)
                    if result is None:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    filled, template = result
                    data["LLM_description"] = filled
                    data["LLM_description_template"] = template
                    save_json(str(path), data)
                    print(f"  description OK: {path.name}")
                    return
                except Exception as e:
                    print(f"  description error ({path.name}, attempt {attempt+1}): {e}")
                    await asyncio.sleep(2 ** attempt)
            print(f"  description FAILED: {path.name}")

    await asyncio.gather(*(one(p, d) for p, d in problems))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("category", nargs="?", help=f"One of: {', '.join(CATEGORIES)}")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of problems (default: 10)")
    parser.add_argument("--size", choices=["small", "large"], default="small",
                        help="Size range to sample from (default: small)")
    parser.add_argument("--params", nargs="+", default=[],
                        help="Fixed param overrides, e.g. --params n_jobs=5 n_machines=4")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: synthetic_dataset/_generated/<category>/)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--model", default="gpt-5", help="LLM model for descriptions (default: gpt-5)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Concurrent description requests (default: 10)")
    parser.add_argument("--no-description", action="store_true",
                        help="Skip the LLM description step (structured data only)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate problems whose JSON already exists")
    parser.add_argument("--list", action="store_true", help="List supported categories and exit")
    args = parser.parse_args()

    if args.list:
        print("Supported categories:")
        for name, cfg in CATEGORIES.items():
            sizes = ", ".join(cfg["ranges"].keys())
            print(f"  {name:30s}  sizes: {sizes}")
        return

    if not args.category or args.category not in CATEGORIES:
        parser.print_help()
        sys.exit(f"\nUnknown or missing category. Use --list to see supported categories.")

    cfg = CATEGORIES[args.category]
    overrides = parse_param_overrides(args.params)

    out_dir = args.output_dir or (ROOT / "synthetic_dataset" / "_generated" / args.category)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: structured problems ───────────────────────────────────
    print(f"Generating {args.num} '{args.category}' problems → {out_dir}")
    rng = np.random.RandomState(args.seed)
    written = []
    for idx in range(1, args.num + 1):
        path = out_dir / f"{cfg['prefix']}_{idx:03d}.json"
        if path.exists() and not args.overwrite:
            data = load_json(str(path))
            if data.get("gurobi_result", {}).get("solver_status") == 2:
                print(f"  [{idx:3d}] {path.name} exists, skipping")
                written.append((path, data))
                continue
        params = sample_params(cfg, args.size, rng)
        params.update(overrides)
        print(f"  [{idx:3d}] params={params}")
        data = generate_one(cfg, params, seed=args.seed + idx * 1000)
        if data is None:
            print(f"  [{idx:3d}] FAILED after {MAX_RETRIES} retries")
            continue
        save_json(str(path), data)
        obj = data.get("gurobi_result", {}).get("objective_value", "?")
        print(f"  [{idx:3d}] OK obj={obj}")
        written.append((path, data))

    # ── Phase 2: natural-language descriptions ─────────────────────────
    if args.no_description:
        print(f"Done — {len(written)} problems (structured only).")
        return

    print(f"Generating descriptions for {len(written)} problems with model={args.model}…")
    asyncio.run(add_descriptions(written, args.model, args.max_concurrent))
    print(f"Done — wrote {len(written)} problems to {out_dir}.")


if __name__ == "__main__":
    main()
