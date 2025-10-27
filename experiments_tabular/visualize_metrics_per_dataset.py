import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="deep_ensemble",
    choices=[
        "deep_ensemble",
        "dare_ensemble",
        "adversarial_ensemble",
    ],
    help="Type of model to use",
)
args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    CONSIDERED_METHODS = [
        "combined",
        "combined_uncertainty_distance",
        "CLUE",
        "FACE",
        "DICE",
        "GS",
    ]
    RENAME_DICT = {
        "combined": "uncertainty",
        "combined_uncertainty_distance": "uncertainty+distance",
    }
    data_files = [
        "bank",
        "churn",
        "compas",
        "diabetes",
        "fico",
        "home",
        "titanic",
        "breast_cancer",
        "boston_housing",
    ]
    metric_list = [
        "instability",
        "invalidity",
        "dissimilarity",
        "dissparsity",
        "implausability",
        "discriminative_power",
    ]
    d1 = pd.read_csv(f"best_sweep_results_{args.model_name}.csv")
    d2 = pd.read_csv(
        f"best_sweep_results_{args.model_name}_uncertainty_plus_distance_test.csv"
    )
    d3 = pd.read_csv(f"best_sweep_results_{args.model_name}_CLUE.csv")
    d4 = pd.read_csv(f"best_sweep_results_{args.model_name}_FACE.csv")
    data = pd.concat([d1, d2, d3, d4], ignore_index=True)
    # Rename the methods accordingly
    data = data[data["method"].isin(CONSIDERED_METHODS)]
    data["method"] = data["method"].replace(RENAME_DICT)

    data.sort_values(by=["dataset", "method"], inplace=True)
    # Rename CONSIDERED METHODS accordingly
    CONSIDERED_METHODS = [RENAME_DICT.get(m, m) for m in CONSIDERED_METHODS]

    print("Loaded Data: ", data)
    # Plotting boxplots

    sns.set(style="whitegrid")
    hue_order = [m for m in CONSIDERED_METHODS if m in data["method"].unique()]
    plt.figure(figsize=(20, 15))

    palette = {
        "uncertainty": "#D55E00",  # vivid orange to separate from recourse baselines
        "uncertainty+distance": "#009E73",  # contrasting green for distance-aware variant
        "CLUE": "#4C72B0",
        "FACE": "#6BAED6",
        "DICE": "#9ECAE1",
        "GS": "#C6DBEF",
    }

    for i, metric in enumerate(metric_list):
        plot_data = data[["dataset", "method", metric]]
        plot_data = plot_data[plot_data["dataset"].isin(data_files)]
        print("Plotting metric", metric, "with data", plot_data)

        Q1 = plot_data[metric].quantile(0.25)
        Q3 = plot_data[metric].quantile(0.75)
        IQR = Q3 - Q1
        filter = (plot_data[metric] >= Q1 - 1.5 * IQR) & (
            plot_data[metric] <= Q3 + 1.5 * IQR
        )
        plot_data = plot_data.loc[filter]

        plt.subplot(3, 3, i + 1)
        ax = sns.lineplot(
            data=plot_data,
            x="dataset",
            y=metric,
            hue="method",
            hue_order=hue_order,
            alpha=1.0,
            palette=palette,
            marker="o",
            linestyle="--",
        )
        # Increase marker size using matplotlib after plotting
        for line in ax.lines:
            line.set_markersize(12)
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig(
        f"metrics_boxplots_{args.model_name}_individual_dataset.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )

    # Create Latex table
    table_data = []
    for method in CONSIDERED_METHODS:
        row = [method]
        method_data = data[data["method"] == method]
        for metric in metric_list:
            mean_val = method_data[metric].mean()
            std_val = method_data[metric].std()
            row.append(f"{mean_val:.4f} ± {std_val:.4f}")
        table_data.append(row)
    columns = ["Method"] + metric_list
    latex_table = pd.DataFrame(table_data, columns=columns)
    print("Latex Table:\n", latex_table.to_latex(index=False))
