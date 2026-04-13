import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # registers scienceplots styles
import argparse
from plot_config import (
    FIGSIZE,
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
    considered_methods = [METHOD_RENAME.get(m, m) for m in considered_methods]

    apply_publication_style()
    hue_order = [m for m in considered_methods if m in data["method"].unique()]
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
    axes = axes.flatten()

    for i, metric in enumerate(METRIC_LIST):
        plot_data = data[["dataset", "method", metric]].copy()
        plot_data = filter_outliers_iqr(plot_data, metric)

        series_per_method = []
        labels = []
        for method in hue_order:
            values = plot_data.loc[plot_data["method"] == method, metric].dropna().values
            if len(values) > 0:
                series_per_method.append(values)
                labels.append(method)

        ax = axes[i]
        box = ax.boxplot(
            series_per_method,
            tick_labels=labels,
            patch_artist=True,
            widths=0.65,
            showfliers=False,
            medianprops={"color": "#1a1a1a", "linewidth": 1.4},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
            boxprops={"linewidth": 1.1},
        )
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(METHOD_PALETTE[label])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#2b2b2b")

        ax.set_xlabel("method", fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=22, fontweight="bold")
        ax.tick_params(axis="x", rotation=35, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    for idx in range(len(METRIC_LIST), len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    fig.savefig(
        f"metrics_boxplots_{args.model_name}.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    fig.savefig(
        f"metrics_boxplots_{args.model_name}.png",
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
