from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_CHOICES = ["deep_ensemble", "dare_ensemble", "adversarial_ensemble"]

METHODS_COMPARISON = [
    "combined",
    "combined_uncertainty_distance",
    "CLUE",
    "FACE",
    "DICE",
    "GS",
]

METHOD_RENAME = {
    "combined": "uncertainty",
    "combined_uncertainty_distance": "uncertainty+d",
}

PROPERTY_METHODS = [
    "validity",
    "connected_ball",
    "robust",
    "feasability",
    "discriminative",
    "plausable",
    "similarity",
    "combined",
    "combined_uncertainty_distance",
]

PROPERTY_RENAME = {
    **METHOD_RENAME,
    "connected_ball": "connected",
}

METRIC_LIST = [
    "invalidity",
    "dissimilarity",
    "dissparsity",
    "implausability",
    "discriminative_power",
    "instability",
]

METRIC_LABELS = {
    "invalidity": "$m_{val}$",
    "dissimilarity": "$m_{sim}$",
    "dissparsity": "$m_{spar}$",
    "implausability": "$m_{plau}$",
    "discriminative_power": "$m_{dis}$",
    "instability": "$m_{sta}$",
}

DATASET_ORDER = [
    "bank",
    "breast_cancer",
    "churn",
    "compas",
    "diabetes",
    "fico",
    "home",
    "boston_housing",
    "titanic",
]

DATASET_DISPLAY_LABELS = [
    "bank",
    "cancer",
    "churn",
    "compas",
    "diabetes",
    "fico",
    "home",
    "housing",
    "titanic",
]

METHOD_LEGEND_ORDER = ["uncertainty", "uncertainty+d", "CLUE", "DICE", "FACE", "GS"]

PROPERTY_ORDER = [
    "uncertainty",
    "uncertainty+d",
    "validity",
    "connected",
    "robust",
    "feasability",
    "discriminative",
    "plausable",
    "similarity",
]

PROPERTY_DISPLAY_LABELS = {
    "validity": "Validity",
    "connected": "Connectedness",
    "robust": "Robustness",
    "feasability": "Feasibility",
    "discriminative": "Discriminativeness",
    "plausable": "Plausibility",
    "similarity": "Similarity",
    "uncertainty": "Uncertainty",
    "uncertainty+d": "Uncertainty+d",
}

FIGSIZE = (15, 9)

METHOD_PALETTE = {
    "uncertainty": "#D55E00",
    "uncertainty+d": "#009E73",
    "CLUE": "#0072B2",
    "FACE": "#56B4E9",
    "DICE": "#CC79A7",
    "GS": "#E69F00",
}

PROPERTY_PALETTE = {
    "validity": "#D55E00",
    "connected": "#E69F00",
    "robust": "#F0E442",
    "feasability": "#009E73",
    "discriminative": "#56B4E9",
    "plausable": "#0072B2",
    "similarity": "#CC79A7",
    "uncertainty": "#D55E00",
    "uncertainty+d": "#009E73",
}


def apply_publication_style():
    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_results_csv(path_options):
    for path in path_options:
        if Path(path).exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"None of these files exist: {path_options}")


def load_model_results(model_name):
    d1 = pd.read_csv(f"best_sweep_results_{model_name}.csv")
    d2 = read_results_csv(
        [
            f"best_sweep_results_{model_name}_uncertainty_plus_distance.csv",
            f"best_sweep_results_{model_name}_uncertainty_plus_distance_test.csv",
        ]
    )
    d3 = pd.read_csv(f"best_sweep_results_{model_name}_CLUE.csv")
    d4 = pd.read_csv(f"best_sweep_results_{model_name}_FACE.csv")
    return pd.concat([d1, d2, d3, d4], ignore_index=True)


def filter_outliers_iqr(plot_data, metric):
    q1 = plot_data[metric].quantile(0.25)
    q3 = plot_data[metric].quantile(0.75)
    iqr = q3 - q1
    keep_mask = (plot_data[metric] >= q1 - 1.5 * iqr) & (
        plot_data[metric] <= q3 + 1.5 * iqr
    )
    return plot_data.loc[keep_mask]
