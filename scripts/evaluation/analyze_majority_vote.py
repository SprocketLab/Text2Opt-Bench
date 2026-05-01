#!/usr/bin/env python3
"""
Analyze best-of-K results with multiple selection strategies:
  1. Oracle (best-of-K): any correct = pass (needs ground truth)
  2. Majority vote: pick the most common objective value, check if correct
  3. pass@1: average single-sample success rate

Also computes token costs for comparison with repair loops.
"""
import json
import os
from collections import defaultdict, Counter
import math

BASE = str(Path(__file__).resolve().parent.parent.parent)
RESOURCE_PATH = os.path.join(BASE, "results/repair_curve/best_of_5_gpt5nano_resource.json")
TEMPLATE_PATH = os.path.join(BASE, "results/repair_curve/best_of_5_gpt5nano_template.json")

# Tolerance for matching objective values (same as evaluator: 1e-4 relative)
REL_TOL = 1e-4


def objectives_match(a, b):
    """Check if two objective values match within tolerance."""
    if a is None or b is None:
        return False
    if a == b == 0:
        return True
    return abs(a - b) / max(abs(a), abs(b), 1e-10) < REL_TOL


def majority_vote_objective(samples):
    """Pick the most common objective value from samples.

    Groups by approximate equality (within REL_TOL), then picks the largest group.
    Returns the representative value of the majority group, or None if no valid values.
    """
    # Collect valid objective values (from samples that executed and got OPTIMAL)
    valid = []
    for s in samples:
        obj = s.get("generated_objective")
        if obj is not None and s.get("execution_succeeded") and s.get("status_optimal"):
            valid.append(obj)

    if not valid:
        return None

    # Group by approximate equality
    groups = []  # list of (representative_value, count)
    for val in valid:
        matched = False
        for i, (rep, cnt) in enumerate(groups):
            if objectives_match(val, rep):
                groups[i] = (rep, cnt + 1)
                matched = True
                break
        if not matched:
            groups.append((val, 1))

    # Pick the group with highest count (break ties by picking the value closest to any)
    groups.sort(key=lambda x: -x[1])
    return groups[0][0]


def analyze_dataset(path, name):
    data = json.load(open(path))
    print(f"\n{'='*70}")
    print(f"  {name} (n={len(data)}, K={data[0]['num_samples']})")
    print(f"{'='*70}")

    by_cat = defaultdict(list)
    for r in data:
        by_cat[r.get("category", "unknown")].append(r)

    # Compute per-problem results for each strategy
    for group_name, group_data in [("Overall", data)] + list(sorted(by_cat.items())):
        n = len(group_data)
        oracle_correct = 0
        majority_correct = 0
        pass1_sum = 0.0

        for r in group_data:
            samples = r["samples"]
            k = r["num_samples"]
            gold = r["samples"][0].get("gold_objective")  # same for all samples

            # pass@1
            n_succ = sum(1 for s in samples if s.get("succeeded"))
            pass1_sum += n_succ / k

            # Oracle (best-of-K)
            if any(s.get("succeeded") for s in samples):
                oracle_correct += 1

            # Majority vote
            mv_obj = majority_vote_objective(samples)
            if mv_obj is not None and gold is not None and objectives_match(mv_obj, gold):
                majority_correct += 1

        pass1 = pass1_sum / n * 100
        oracle = oracle_correct / n * 100
        majority = majority_correct / n * 100

        if group_name == "Overall":
            avg_tok = sum(r["total_tokens"] for r in group_data) / n
            print(f"\n  --- {group_name} (n={n}) --- avg tokens: {avg_tok:.0f}")
        else:
            print(f"\n  --- {group_name} (n={n}) ---")
        print(f"    pass@1:        {pass1:6.1f}%")
        print(f"    Majority Vote: {majority:6.1f}%")
        print(f"    Oracle (any):  {oracle:6.1f}%")


if __name__ == "__main__":
    if os.path.exists(RESOURCE_PATH):
        analyze_dataset(RESOURCE_PATH, "Resource Allocation (Direct)")
    else:
        print(f"Waiting for {RESOURCE_PATH}...")

    if os.path.exists(TEMPLATE_PATH):
        analyze_dataset(TEMPLATE_PATH, "Template (Agentic)")
    else:
        print(f"Waiting for {TEMPLATE_PATH}...")
