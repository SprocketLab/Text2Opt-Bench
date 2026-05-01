#!/usr/bin/env python3
"""
Generate accuracy heatmaps for two-phase evaluation results.
Adds a red curriculum boundary line showing max num_vars seen during training.

Usage:
    python scripts/training/plot_two_phase_heatmap.py \
        --results-path results/two_phase_template_eval.json \
        --title-suffix "(1.5B Binder → Template)" \
        --curriculum-boundary 11
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_two_phase_results(path: str) -> pd.DataFrame:
    """Load results from two-phase eval JSON (has 'results' list with flat keys)."""
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", data) if isinstance(data, dict) else data
    rows = []
    for r in results:
        rows.append({
            "problem_id": r.get("problem_id"),
            "num_vars": r.get("num_vars"),
            "num_constraints": r.get("num_constraints"),
            "objective_matches": r.get("objective_matches", False),
            "execution_succeeded": r.get("execution_succeeded", False),
            "status_optimal": r.get("status_optimal", False),
        })
    df = pd.DataFrame(rows)
    df["success"] = (
        df["execution_succeeded"].fillna(False)
        & df["status_optimal"].fillna(False)
        & df["objective_matches"].fillna(False)
    )
    return df


def plot_heatmap(
    df: pd.DataFrame,
    output_path: str,
    title: str,
    boundary_vars: Optional[int] = None,
    dpi: int = 300,
) -> None:
    pivot = (
        df.pivot_table(
            index="num_vars",
            columns="num_constraints",
            values="success",
            aggfunc="mean",
        )
        * 100.0
    )

    overall_acc = df["success"].mean() * 100.0 if not df.empty else 0.0

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"{title} - Overall: {overall_acc:.2f}%")
    plt.ylabel("num_vars")
    plt.xlabel("num_constraints")

    if boundary_vars is not None:
        y_labels = sorted(pivot.index.tolist())
        if boundary_vars in y_labels:
            idx = y_labels.index(boundary_vars)
            ax.hlines(idx + 1, *ax.get_xlim(), colors="red", linewidth=4)
        elif y_labels and boundary_vars < min(y_labels):
            ax.hlines(0, *ax.get_xlim(), colors="red", linewidth=4)
        elif y_labels and boundary_vars < max(y_labels):
            import bisect
            idx = bisect.bisect_right(y_labels, boundary_vars)
            ax.hlines(idx, *ax.get_xlim(), colors="red", linewidth=4)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy heatmap for two-phase results")
    parser.add_argument("--results-path", required=True, help="Path to two-phase eval JSON")
    parser.add_argument("--title-suffix", default="", help="Suffix for plot title")
    parser.add_argument("--curriculum-boundary", type=int, default=11,
                        help="Max num_vars in training data (red line on heatmap)")
    parser.add_argument("--output-path", default=None,
                        help="Output PNG path (default: next to results file)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = load_two_phase_results(args.results_path)

    results_path = Path(args.results_path)
    output_path = args.output_path or str(
        results_path.parent / f"{results_path.stem}_accuracy_heatmap.png"
    )

    title = f"Accuracy (%) by num_vars / num_constraints {args.title_suffix}".strip()
    plot_heatmap(df, output_path, title, boundary_vars=args.curriculum_boundary, dpi=args.dpi)

    # Print summary
    overall = df["success"].mean() * 100
    n = len(df)
    passes = df["success"].sum()
    print(f"Overall: {passes}/{n} ({overall:.1f}%)")

    if args.curriculum_boundary:
        in_dist = df[df["num_vars"] <= args.curriculum_boundary]
        ood = df[df["num_vars"] > args.curriculum_boundary]
        if not in_dist.empty:
            print(f"In-distribution (vars <= {args.curriculum_boundary}): "
                  f"{in_dist['success'].sum()}/{len(in_dist)} ({in_dist['success'].mean()*100:.1f}%)")
        if not ood.empty:
            print(f"Out-of-distribution (vars > {args.curriculum_boundary}): "
                  f"{ood['success'].sum()}/{len(ood)} ({ood['success'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
