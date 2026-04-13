import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # registers scienceplots styles
import argparse
from matplotlib.lines import Line2D
from plot_config import (
    DATASET_DISPLAY_LABELS,
    DATASET_ORDER,
    FIGSIZE,
    METHOD_LEGEND_ORDER,
    METHOD_PALETTE,
    METHODS_COMPARISON,
    METHOD_RENAME,
    METRIC_LABELS,
    METRIC_LIST,
    MODEL_CHOICES,
    apply_publication_style,
    filter_outliers_iqr,
    load_model_results,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="deep_ensemble",
    choices=MODEL_CHOICES,
    help="Type of model to use",
)
args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    considered_methods = METHODS_COMPARISON.copy()
    data = load_model_results(args.model_name)
    # Rename the methods accordingly
    data = data[data["method"].isin(considered_methods)]
    data["method"] = data["method"].replace(METHOD_RENAME)

    data.sort_values(by=["dataset", "method"], inplace=True)
    # Rename CONSIDERED METHODS accordingly
    considered_methods = [METHOD_RENAME.get(m, m) for m in considered_methods]

    apply_publication_style()
    hue_order = [m for m in METHOD_LEGEND_ORDER if m in data["method"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
    axes = axes.flatten()

    for i, metric in enumerate(METRIC_LIST):
        plot_data = data[["dataset", "method", metric]]
        plot_data = plot_data[plot_data["dataset"].isin(DATASET_ORDER)]
        plot_data = filter_outliers_iqr(plot_data, metric)

        ax = axes[i]
        for method in hue_order:
            method_data = (
                plot_data[plot_data["method"] == method]
                .set_index("dataset")
                .reindex(DATASET_ORDER)
            )
            ax.plot(
                DATASET_ORDER,
                method_data[metric],
                color=METHOD_PALETTE[method],
                marker="o",
                linestyle="--",
                linewidth=2.0,
                markersize=5.5,
                alpha=0.95,
            )

        ax.set_xlabel("dataset", fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=22, fontweight="bold")
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_xticklabels(DATASET_DISPLAY_LABELS, rotation=40)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    legend_handles = [
        Line2D([0], [0], color=METHOD_PALETTE[m], marker="o", linestyle="--", linewidth=2.0, markersize=5.5, label=m)
        for m in hue_order
    ]
    fig.legend(
        handles=legend_handles,
        labels=hue_order,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(hue_order),
        fontsize=22,
    )

    for idx in range(len(METRIC_LIST), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(
        f"metrics_boxplots_{args.model_name}_individual_dataset.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    fig.savefig(
        f"metrics_boxplots_{args.model_name}_individual_dataset.png",
        bbox_inches="tight",
        pad_inches=0.03,
        dpi=600,
    )

    # Create Latex table
    table_data = []
    for method in considered_methods:
        row = [method]
        method_data = data[data["method"] == method]
        for metric in METRIC_LIST:
            mean_val = method_data[metric].mean()
            std_val = method_data[metric].std()
            row.append(f"{mean_val:.4f} ± {std_val:.4f}")
        table_data.append(row)
    columns = ["Method"] + METRIC_LIST
    latex_table = pd.DataFrame(table_data, columns=columns)
    print("Latex Table:\n", latex_table.to_latex(index=False))
