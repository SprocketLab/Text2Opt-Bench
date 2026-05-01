#!/usr/bin/env python3
"""
Analysis utility for evaluation results produced by run_eval.py.

Outputs:
- Text summaries of accuracy per category (goal, problem_type, optimization_type).
- Accuracy tables by num_vars / num_constraints.
- Optional heatmaps saved to disk if matplotlib/seaborn are available.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required for analyze_results.py") from e

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_results(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path_obj, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected results JSON to be a list of entries.")
    return data


def to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        meta = r.get("meta", {}) or {}
        ev = r.get("evaluation_result", {}) or {}
        rows.append(
            {
                "problem_id": r.get("problem_id"),
                "goal": meta.get("goal"),
                "optimization_type": meta.get("optimization_type"),
                "problem_type": meta.get("problem_type"),
                "num_vars": meta.get("num_vars"),
                "num_constraints": meta.get("num_constraints"),
                "prompt_tokens": (meta.get("prompt_tokens_with_data")
                                  or meta.get("prompt_tokens")),
                "execution_succeeded": ev.get("execution_succeeded"),
                "status_optimal": ev.get("status_optimal"),
                "objective_matches": ev.get("objective_matches"),
            }
        )
    df = pd.DataFrame(rows)
    # Define success metric: execution succeeded AND status matched AND objective matched
    df["success"] = (
        df["execution_succeeded"].fillna(False)
        & df["status_optimal"].fillna(False)
        & df["objective_matches"].fillna(False)
    )
    return df


def summarize_accuracy(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols)["success"]
    summary = grouped.agg(["mean", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "accuracy", "count": "samples"})
    return summary


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Overall Accuracy ===")
    overall = df["success"].mean() if len(df) else float("nan")
    print(f"Overall accuracy: {overall:.3f} over {len(df)} samples")

    for col in ["goal", "problem_type", "optimization_type"]:
        print(f"\n=== Accuracy by {col} ===")
        summary = summarize_accuracy(df, [col])
        print(summary.to_string(index=False))

    print("\n=== Accuracy by num_vars / num_constraints ===")
    summary_vc = summarize_accuracy(df, ["num_vars", "num_constraints"])
    print(summary_vc.to_string(index=False))


def analyze_results(
    results_path: str,
    heatmap_path: str = "results/accuracy_heatmap.png",
    summary_csv: str = "results/summary.csv",
    flat_csv: str = "results/evaluations_flat.csv",
):
    results = load_results(results_path)
    df = to_dataframe(results)

    print_summary(df)
    plot_heatmap(df, heatmap_path, title="Accuracy (%) by num_vars / num_constraints")

    # Save flattened dataframe
    flat_path = Path(flat_csv)
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flat_path, index=False)
    print(f"Saved flat evaluations to {flat_path}")

    # Save summary aggregations
    summaries = []
    for cols in [["goal"], ["problem_type"], ["optimization_type"], ["num_vars", "num_constraints"]]:
        summary = summarize_accuracy(df, cols)
        summary["group_by"] = "|".join(cols)
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summaries to {summary_path}")


def plot_heatmap(df: pd.DataFrame, output_path: str, title: str) -> None:
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available; skipping heatmap.")
        return

    # 1. Overall heatmap (existing logic)
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
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"{title} - Overall: {overall_acc:.2f}%")
    plt.ylabel("num_vars")
    plt.xlabel("num_constraints")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved overall heatmap to {out_path}")

    # 2. Per-problem-type heatmaps
    problem_types = df["problem_type"].unique()
    # If there's only one type (or none), we might skip or just do it anyway.
    # Usually "resource allocation" or similar strings.
    
    base_dir = out_path.parent
    base_name = out_path.stem  # e.g. "accuracy_heatmap"
    ext = out_path.suffix      # e.g. ".png"

    for p_type in problem_types:
        if not p_type: 
            continue
        
        sub_df = df[df["problem_type"] == p_type]
        if sub_df.empty:
            continue

        pivot_sub = (
            sub_df.pivot_table(
                index="num_vars",
                columns="num_constraints",
                values="success",
                aggfunc="mean",
            )
            * 100.0
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_sub, annot=True, fmt=".1f", cmap="viridis")
        safe_name = str(p_type).replace(" ", "_").replace("/", "_")
        
        type_acc = sub_df["success"].mean() * 100.0 if not sub_df.empty else 0.0
        
        plt.title(f"{title} - {p_type} - Overall: {type_acc:.2f}%")
        plt.ylabel("num_vars")
        plt.xlabel("num_constraints")
        
        sub_out_path = base_dir / f"{base_name}_{safe_name}{ext}"
        plt.tight_layout()
        plt.savefig(sub_out_path)
        plt.close()
        print(f"Saved heatmap for '{p_type}' to {sub_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results.")
    parser.add_argument(
        "--results-path",
        required=True,
        help="Path to results JSON produced by run_eval.py",
    )
    parser.add_argument(
        "--heatmap-path",
        default=None,
        help="Where to save the vars/constraints heatmap (if plotting libs available). Defaults to same folder as results.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Where to save aggregated summaries as CSV. Defaults to same folder as results.",
    )
    parser.add_argument(
        "--flat-csv",
        default=None,
        help="Where to save the flattened per-sample dataframe. Defaults to same folder as results.",
    )
    args = parser.parse_args()

    # Determine output directory from input results path
    results_path_obj = Path(args.results_path)
    output_dir = results_path_obj.parent
    
    result_file_name = results_path_obj.stem

    # Set defaults relative to output_dir if not provided
    heatmap_path = args.heatmap_path if args.heatmap_path else str(output_dir / f"{result_file_name}_accuracy_heatmap.png")
    summary_csv = args.summary_csv if args.summary_csv else str(output_dir / f"{result_file_name}_summary.csv")
    flat_csv = args.flat_csv if args.flat_csv else str(output_dir / f"{result_file_name}_evaluations_flat.csv")

    analyze_results(
        results_path=args.results_path,
        heatmap_path=heatmap_path,
        summary_csv=summary_csv,
        flat_csv=flat_csv,
    )


if __name__ == "__main__":
    main()

