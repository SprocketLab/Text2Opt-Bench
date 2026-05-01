#!/usr/bin/env python3
"""
RULER Analysis: Score vs Context Length across model sizes.

Generates:
  1. Per-task score vs token length (one line per model)
  2. Combined heatmap: model size x context length
  3. Summary bar chart

Usage:
    # Edit MODEL_RESULTS and OUTPUT_DIR below to point at your result files, then:
    python ruler/analysis.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_RESULTS: Dict[str, Union[str, List[str]]] = {
    "Qwen-0.5B":  "results/ruler/qwen-0.5B.json",
    "Qwen-1.5B":  "results/ruler/qwen-1.5B.json",
    "Qwen-3B":    "results/ruler/qwen-3B.json",
    "Qwen-7B":    "results/ruler/qwen-7B.json",
    "Qwen-14B":   "results/ruler/qwen-14B.json",
    "Qwen-32B":   "results/ruler/qwen-32B.json",
}

OUTPUT_DIR = "results/ruler"

# Only include binding-relevant tasks (retrieval + aggregation), ordered for plotting
BINDING_TASKS_ORDER = ["niah_single", "niah_multikey", "niah_multivalue", "aggregation"]
BINDING_TASKS = set(BINDING_TASKS_ORDER)
# =============================================================================

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    raise SystemExit("pandas and numpy are required") from e

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def load_results(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    with open(path_obj) as f:
        return json.load(f)


def extract_data(results: List[Dict[str, Any]], task_filter=None) -> pd.DataFrame:
    rows = []
    for r in results:
        meta = r.get("meta", {}) or {}
        ev = r.get("evaluation_result", {}) or {}
        if "score" not in ev:
            continue
        task = meta.get("task", "unknown")
        if task_filter and task not in task_filter:
            continue
        rows.append({
            "problem_id": r.get("problem_id"),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "target_length": meta.get("target_length", 0),
            "task": task,
            "score": ev["score"],
        })
    return pd.DataFrame(rows)


def plot_analysis(model_data: Dict[str, pd.DataFrame], output_dir: str, suffix: str = ""):
    if not PLOTTING_AVAILABLE:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })
    n_models = len(model_data)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_models - 1, 1)) for i in range(n_models)]

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use predefined task order (niah first row, aggregation + average second row)
    all_tasks = [t for t in BINDING_TASKS_ORDER if any(t in df["task"].unique() for df in model_data.values())]

    n_tasks = len(all_tasks)
    if n_tasks == 0:
        return

    # Pretty display names for tasks
    TASK_DISPLAY = {
        "niah_single": "NIAH Single",
        "niah_multikey": "NIAH Multi-Key",
        "niah_multivalue": "NIAH Multi-Value",
        "aggregation": "Aggregation",
    }

    # ===== Figure 1: Per-task line plots, single compact row =====
    fig, axes = plt.subplots(1, n_tasks, figsize=(7.0, 2.2),
                             sharey=True, squeeze=False)
    axes = axes[0]  # flatten to 1-D

    for task_idx, task in enumerate(all_tasks):
        ax = axes[task_idx]

        for model_idx, (model_name, df) in enumerate(model_data.items()):
            task_df = df[df["task"] == task]
            if task_df.empty:
                continue

            # Aggregate by target_length
            agg = task_df.groupby("target_length")["score"].agg(["mean", "std", "count"]).reset_index()
            agg = agg.sort_values("target_length")

            color = colors[model_idx % len(colors)]
            ax.plot(
                agg["target_length"], agg["mean"] * 100,
                marker='o', markersize=3, linewidth=1.5,
                label=model_name, color=color,
            )
            # Error bars (stderr)
            if len(agg) > 0 and agg["count"].iloc[0] > 1:
                stderr = agg["std"] / np.sqrt(agg["count"]) * 100
                ax.fill_between(
                    agg["target_length"],
                    (agg["mean"] * 100 - stderr).clip(0),
                    (agg["mean"] * 100 + stderr).clip(upper=100),
                    alpha=0.15, color=color,
                )

        ax.set_xlabel("Context Length (tokens)")
        ax.set_title(TASK_DISPLAY.get(task, task), fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    # Only leftmost panel gets the y-axis label
    axes[0].set_ylabel("Accuracy (%)")

    # Shared legend and title — tight vertical stacking
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(model_data),
               fontsize=7, bbox_to_anchor=(0.5, 1.06), frameon=True)

    fig.suptitle("RULER Binding Tasks: Accuracy vs Context Length",
                 fontsize=11, fontweight='bold', y=1.12)
    plt.savefig(out_dir / f"Ruler_analysis{suffix}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_dir / f"Ruler_analysis{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-task plot to {out_dir / f'Ruler_analysis{suffix}.pdf'}")

    # ===== Figure 2: Combined score vs length (all tasks averaged) =====
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    for model_idx, (model_name, df) in enumerate(model_data.items()):
        agg = df.groupby("target_length")["score"].agg(["mean", "std", "count"]).reset_index()
        agg = agg.sort_values("target_length")
        color = colors[model_idx % len(colors)]

        ax2.plot(
            agg["target_length"], agg["mean"] * 100,
            marker='o', markersize=8, linewidth=3,
            label=model_name, color=color,
        )
        if agg["count"].iloc[0] > 1:
            stderr = agg["std"] / np.sqrt(agg["count"]) * 100
            ax2.fill_between(
                agg["target_length"],
                (agg["mean"] * 100 - stderr).clip(0),
                (agg["mean"] * 100 + stderr).clip(upper=100),
                alpha=0.15, color=color,
            )

    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("Average Score (%)")
    ax2.set_title("RULER Binding Tasks: Average Score vs Context Length")
    ax2.set_xscale('log', base=2)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"token_accuracy_curve{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {out_dir / f'token_accuracy_curve{suffix}.png'}")

    # ===== Figure 3: Heatmap (model x length) =====
    if len(model_data) > 1:
        models = list(model_data.keys())
        first_df = next(iter(model_data.values()))
        lengths = sorted(first_df["target_length"].unique())

        heatmap = np.zeros((len(models), len(lengths)))
        for i, model_name in enumerate(models):
            df = model_data[model_name]
            for j, length in enumerate(lengths):
                scores = df[df["target_length"] == length]["score"]
                heatmap[i, j] = scores.mean() * 100 if len(scores) > 0 else 0

        fig3, ax3 = plt.subplots(1, 1, figsize=(max(8, len(lengths) * 1.5), max(4, len(models) * 0.8)))
        im = ax3.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

        ax3.set_xticks(range(len(lengths)))
        ax3.set_xticklabels([f"{l//1024}K" if l >= 1024 else str(l) for l in lengths])
        ax3.set_yticks(range(len(models)))
        ax3.set_yticklabels(models)
        ax3.set_xlabel("Context Length", fontsize=12)
        ax3.set_title("RULER Binding Tasks: Score Heatmap (Model x Length)", fontsize=13)

        # Add value labels
        for i in range(len(models)):
            for j in range(len(lengths)):
                ax3.text(j, i, f"{heatmap[i, j]:.0f}", ha="center", va="center",
                         fontsize=10, fontweight='bold',
                         color="white" if heatmap[i, j] < 40 else "black")

        fig3.colorbar(im, label="Score (%)")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {out_dir / f'heatmap{suffix}.png'}")


def main():
    print("=" * 60)
    print("RULER Token-Accuracy Analysis")
    print("=" * 60)

    model_data = {}
    for model_name, paths in MODEL_RESULTS.items():
        if isinstance(paths, str):
            paths = [paths]
        try:
            all_results = []
            for p in paths:
                all_results.extend(load_results(p))
            df = extract_data(all_results, task_filter=BINDING_TASKS)
            model_data[model_name] = df
            print(f"\n{model_name}: {len(df)} samples")

            # Per-task, per-length summary
            for task in sorted(df["task"].unique()):
                tdf = df[df["task"] == task]
                print(f"  {task}:")
                for length in sorted(tdf["target_length"].unique()):
                    ldf = tdf[tdf["target_length"] == length]
                    print(f"    {length:6d}: {ldf['score'].mean()*100:5.1f}% (n={len(ldf)})")
        except Exception as e:
            print(f"  Error: {e}")

    if model_data:
        plot_analysis(model_data, OUTPUT_DIR)

    # Also generate all-tasks plots (not for paper, for reference)
    print("\n--- Generating all-tasks plots (reference) ---")
    all_tasks_data = {}
    for model_name, paths in MODEL_RESULTS.items():
        if isinstance(paths, str):
            paths = [paths]
        try:
            all_results = []
            for p in paths:
                all_results.extend(load_results(p))
            df = extract_data(all_results, task_filter=None)
            all_tasks_data[model_name] = df
        except Exception:
            pass
    if all_tasks_data:
        # Temporarily override task order to include all
        global BINDING_TASKS_ORDER
        saved_order = BINDING_TASKS_ORDER
        all_task_names = set()
        for df in all_tasks_data.values():
            all_task_names.update(df["task"].unique())
        BINDING_TASKS_ORDER = sorted(all_task_names)
        plot_analysis(all_tasks_data, OUTPUT_DIR, suffix="_all_tasks")
        BINDING_TASKS_ORDER = saved_order


if __name__ == "__main__":
    main()
