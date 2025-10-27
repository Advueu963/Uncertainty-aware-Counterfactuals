import pandas as pd
import numpy as np
import torch
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
        "validity",
        "connected_ball",
        "robust",
        "feasability",
        "discriminative",
        "plausable",
        "similarity",
        "combined",
        "combined_uncertainty_distance",
        "CLUE",
        "FACE",
        "DICE",
        "GS",
    ]
    RENAME_DICT_METHODS = {
        "combined": "uncertainty",
        "combined_uncertainty_distance": "uncertainty+distance",
        "connected_ball": "connected",
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
        f"best_sweep_results_{args.model_name}_uncertainty_plus_distance.csv"
    )
    d3 = pd.read_csv(f"best_sweep_results_{args.model_name}_CLUE.csv")
    d4 = pd.read_csv(f"best_sweep_results_{args.model_name}_FACE.csv")
    data = pd.concat([d1, d2, d3, d4], ignore_index=True)
    # Rename the methods accordingly
    data = data[data["method"].isin(CONSIDERED_METHODS)]
    data["method"] = data["method"].replace(RENAME_DICT_METHODS)

    data.sort_values(by=["dataset", "method"], inplace=True)
    # Rename CONSIDERED METHODS accordingly
    CONSIDERED_METHODS = [RENAME_DICT_METHODS.get(m, m) for m in CONSIDERED_METHODS]

    def generate_latex_table(df, metrics, methods):
        latex = []
        latex.append("% Medal colors")
        latex.append("\\definecolor{gold}{RGB}{255,215,0}")
        latex.append("\\definecolor{silver}{RGB}{192,192,192}")
        latex.append("\\definecolor{bronze}{RGB}{205,127,50}")
        latex.append("\\begin{table}[ht]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{l l %s}" % (" ".join(["c" for _ in metrics])))
        latex.append("\\hline")
        header = ["Dataset", "Method"] + metrics
        latex.append(" & ".join(header) + " \\\\ ")
        latex.append("\\hline")
        datasets = df["dataset"].unique()
        for dataset in datasets:
            df_dataset = df[df["dataset"] == dataset]
            methods_in_data = [m for m in methods if m in df_dataset["method"].values]
            n_methods = len(methods_in_data)
            # Find medal values for each metric
            medal_ranks = {}
            for metric in metrics:
                vals = df_dataset[["method", metric]].dropna()
                if vals.empty:
                    medal_ranks[metric] = {}
                    continue
                if metric == "discriminative_power":
                    sorted_vals = vals.sort_values(by=metric, ascending=False)
                else:
                    sorted_vals = vals.sort_values(by=metric, ascending=True)
                unique_vals = sorted_vals[metric].unique()
                medal_ranks[metric] = {}
                for idx, v in enumerate(unique_vals[:3]):
                    if idx == 0:
                        medal_ranks[metric][v] = "gold"
                    elif idx == 1:
                        medal_ranks[metric][v] = "silver"
                    elif idx == 2:
                        medal_ranks[metric][v] = "bronze"
            for i, method in enumerate(methods_in_data):
                row = []
                if i == 0:
                    row.append(f"\\multirow{{{n_methods}}}{{*}}{{{dataset}}}")
                else:
                    row.append("")
                row.append(method)
                df_row = df_dataset[df_dataset["method"] == method]
                for metric in metrics:
                    value = (
                        df_row[metric].values[0]
                        if not df_row.empty and metric in df_row
                        else "-"
                    )
                    if value == "-":
                        row.append("-")
                    else:
                        try:
                            value_float = float(value)
                        except Exception:
                            value_float = None
                        formatted = (
                            f"{value_float:.3f}"
                            if value_float is not None
                            else str(value)
                        )
                        color = None
                        if (
                            value_float is not None
                            and value_float in medal_ranks[metric]
                        ):
                            color = medal_ranks[metric][value_float]
                        if color:
                            formatted = f"\\textcolor{{{color}}}{{{formatted}}}"
                        row.append(formatted)
                latex.append(" & ".join(row) + " \\\\ ")
            latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\caption{Results by dataset, method, and metrics}")
        latex.append("\\label{tab:results}")
        latex.append("\\end{table}")
        return "\n".join(latex)

    latex_table = generate_latex_table(data, metric_list, CONSIDERED_METHODS)
    with open(f"latex_table_{args.model_name}.tex", "w") as f:
        f.write(latex_table)
    print("LaTeX table written to latex_table.tex")
