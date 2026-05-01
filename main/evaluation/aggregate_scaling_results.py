#!/usr/bin/env python3
"""
Aggregate pipeline_results.json files from scaling-up runs and produce combined
heatmaps plus summary CSVs.

Typical usage:
  python -m main.evaluation.aggregate_scaling_results \
    --base-dir results/scaling_up/gpt-5 \
    --output-json results/scaling_up/gpt-5/pipeline_results_aggregated.json \
    --heatmap results/scaling_up/gpt-5/accuracy_heatmap_aggregated.png
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from main.evaluation.analyze_results import (
    to_dataframe,
    plot_heatmap,
    summarize_accuracy,
)

# Optional plotting for heatmaps
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def collect_results(base_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    """Load and concatenate all results matching the pattern under base_dir."""
    results: List[Dict[str, Any]] = []
    files = sorted(base_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' under {base_dir}")

    for fpath in files:
        with fpath.open("r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {fpath}, got {type(data)}")
            # Annotate source folder for traceability
            for entry in data:
                entry.setdefault("meta", {})
                entry["meta"]["source_dir"] = str(fpath.parent)
            results.extend(data)

    return results


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_summary_and_flat(df: pd.DataFrame, summary_csv: Path, flat_csv: Path) -> None:
    flat_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flat_csv, index=False)

    summaries = []
    for cols in [["goal"], ["problem_type"], ["optimization_type"], ["num_vars", "num_constraints"]]:
        summary = summarize_accuracy(df, cols)
        summary["group_by"] = "|".join(cols)
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)


def plot_accuracy_and_tokens_by_num_vars(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Plot accuracy (color) with annotation showing accuracy + avg tokens per num_vars."""
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available; skipping num_vars accuracy/token heatmap.")
        return

    if df.empty:
        print("No data available; skipping num_vars accuracy/token heatmap.")
        return

    sub = df.copy()
    sub["success"] = sub["success"].fillna(False)

    grouped = (
        sub.groupby("num_vars")
        .agg(
            accuracy=("success", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            samples=("success", "count"),
        )
        .reset_index()
    )

    if grouped.empty:
        print("Grouped data empty; skipping num_vars accuracy/token heatmap.")
        return

    grouped["accuracy_pct"] = grouped["accuracy"] * 100.0
    pivot = grouped.pivot_table(index="num_vars", values="accuracy_pct")

    # Build annotation strings aligned with pivot index
    annot_lookup = {
        row["num_vars"]: f"{row['accuracy_pct']:.1f}%\n{row['avg_prompt_tokens']:.0f} tok"
        if pd.notna(row["avg_prompt_tokens"])
        else f"{row['accuracy_pct']:.1f}%\nNA tok"
        for _, row in grouped.iterrows()
    }
    annot = [[annot_lookup[idx]] for idx in pivot.index]

    plt.figure(figsize=(5, 8))
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
    )
    plt.title(title)
    plt.ylabel("num_vars")
    plt.xlabel("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved num_vars accuracy/token heatmap to {output_path}")


def run(
    base_dir: Path,
    pattern: str,
    output_json: Path,
    heatmap_path: Path,
    num_vars_heatmap_path: Path,
    summary_csv: Path,
    flat_csv: Path,
):
    results = collect_results(base_dir, pattern)
    save_json(output_json, results)
    print(f"Collected {len(results)} entries from {base_dir} into {output_json}")

    df = to_dataframe(results)
    # Full 2D heatmap (vars x constraints) for reference
    plot_heatmap(df, str(heatmap_path), title="Aggregated Accuracy (%) by num_vars / num_constraints")
    # 1D num_vars heatmap with accuracy color + token annotation
    plot_accuracy_and_tokens_by_num_vars(
        df,
        num_vars_heatmap_path,
        title="Accuracy (%) and Avg Prompt Tokens by num_vars",
    )
    save_summary_and_flat(df, summary_csv, flat_csv)
    print(f"Wrote summary to {summary_csv} and flat data to {flat_csv}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate scaling-up results and plot a combined heatmap.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing per-level subfolders with pipeline_results.json",
    )
    parser.add_argument(
        "--pattern",
        default="pipeline_results.json",
        help="Filename pattern to search for under base-dir (default: pipeline_results.json)",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write aggregated JSON list.",
    )
    parser.add_argument(
        "--heatmap",
        required=True,
        help="Path to write aggregated heatmap PNG.",
    )
    parser.add_argument(
        "--num-vars-heatmap",
        required=False,
        help="Path to write accuracy (color) with avg prompt tokens (annotation) by num_vars.",
    )
    parser.add_argument(
        "--summary-csv",
        required=True,
        help="Path to write aggregated summary CSV.",
    )
    parser.add_argument(
        "--flat-csv",
        required=True,
        help="Path to write flattened per-sample CSV.",
    )
    args = parser.parse_args()

    run(
        base_dir=Path(args.base_dir),
        pattern=args.pattern,
        output_json=Path(args.output_json),
        heatmap_path=Path(args.heatmap),
        num_vars_heatmap_path=Path(args.num_vars_heatmap) if args.num_vars_heatmap else Path(args.heatmap).with_name("accuracy_tokens_by_num_vars.png"),
        summary_csv=Path(args.summary_csv),
        flat_csv=Path(args.flat_csv),
    )


if __name__ == "__main__":
    main()

