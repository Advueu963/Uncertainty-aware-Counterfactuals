import os
import numpy as np
import pandas as pd
import argparse
from uncertainty_cfs.tabular_util import (
    get_categorical_features_all,
    get_dataset,
    get_tabular_dataset,
)
import torch
from uncertainty_cfs.architectures import MLP
from probly.representation import Ensemble
from probly.calibration import Temperature
from uncertainty_cfs.property_procedures.utils import (
    predict_probs,
)

from uncertainty_cfs.property_procedures.utils import (
    instability_metric,
    invalidity,
    dissimilarity,
    dissparsity,
    discriminative_power,
    implausability,
)

DATA_FOLDER = os.environ.get("SCRATCH_DSS")

DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
PROPERTY_NAMES = [
    "validity",
    "connected_ball",
    "robust",
    "feasability",
    "discriminative",
    "plausable",
    "similarity",
    "combined",
]
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
N_EPOCHS = 50


def load_property_data(model_name, dataset_name, property_name, sweep_kwargs):
    data = np.load(
        os.path.join(
            os.path.join(DATA_FOLDER, "property_tabular"),
            "CFs_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_test.npy".format(
                model_name, dataset_name, *sweep_kwargs
            ),
        ),
        allow_pickle=True,
    ).item()
    return data[property_name]


def load_carla_data(dataset_name, method_name):
    dataset_original = get_tabular_dataset(dataset_name, return_dataframe=True)
    df = dataset_original["df"]
    df.drop(dataset_original["class_name"], axis=1, inplace=True)
    original_df_columns = df.columns.tolist()

    MODEL_NAME = args.model_name
    try:
        path = os.path.join(
            DATA_FOLDER,
            f"CFs_{MODEL_NAME}_{method_name}_all_datasets.csv",
        )
        print(f"Loading CARLA data from {path}")
        data = pd.read_csv(path)
    except FileNotFoundError as e:
        print(e)
        return None

    data = data[data.dataset == dataset_name]
    data = data[original_df_columns + ["construction_time", "method", "point_index"]]
    section = len(data) // 2
    return {
        "counter_factual": data.iloc[:section]
        .drop(columns=["construction_time", "method", "point_index"])
        .to_numpy(),
        "counter_factual_closest": data.iloc[section:]
        .drop(columns=["construction_time", "method", "point_index"])
        .to_numpy(),
    }


def get_metrics(
    X_points,
    X_points_close,
    y_points,
    y_points_close,
    counter_factual,
    counter_factual_closest,
    X_test,
    y_test,
    model,
    orginal_X,
    categorical_features_all,
):
    # Remove all counterfactuals which are NaN
    mask = ~counter_factual.isnan().any(dim=1)
    counter_factual = counter_factual[mask]
    counter_factual_closest = counter_factual_closest[mask]
    X_points = X_points[mask]
    X_points_close = X_points_close[mask]
    y_points = y_points[mask]
    y_points_close = y_points_close[mask]
    print(
        "Calculation on",
        counter_factual.shape[0],
        "counterfactuals after removing NaNs".format(),
    )

    # Compute only the metrics for counterfactual which have the desired class
    mask = (
        predict_probs(model, counter_factual).mean(dim=1).argmax(dim=1) == DESIRED_CLASS
    )
    counter_factual = counter_factual[mask]
    counter_factual_closest = counter_factual_closest[mask]
    X_points = X_points[mask]
    X_points_close = X_points_close[mask]
    y_points = y_points[mask]
    y_points_close = y_points_close[mask]
    print(
        "Calculation on",
        counter_factual.shape[0],
        f"counterfactuals which are {100 * mask.sum().item() / mask.shape[0] if mask.shape[0] > 0 else 0}% of the original {mask.shape[0]} points".format(),
    )

    if counter_factual.shape[0] == 0:
        print(
            "No counterfactuals found for desired class, returning NaNs for all metrics"
        )
        return {
            "instability": torch.Tensor([np.nan]),
            "invalidity": torch.Tensor([np.nan]),
            "dissimilarity": torch.Tensor([np.nan]),
            "dissparsity": torch.Tensor([np.nan]),
            "implausability": torch.Tensor([np.nan]),
            "discriminative_power": torch.Tensor([np.nan]),
            "proportion_counterfactuals": 0.0,
        }

    instability_values = instability_metric(
        point_of_interest=X_points,
        point_to_compare=X_points_close,
        cf_point_of_interest=counter_factual,
        cf_point_to_compare=counter_factual_closest,
        X=orginal_X,
        categorical_features=categorical_features_all,
    )
    invalidity_values = invalidity(
        counter_factual,
        model=model,
        probability_function=predict_probs,
        desired_class=DESIRED_CLASS,
    )
    dissimilarity_values = dissimilarity(
        point_of_interest=X_points,
        cf_point_of_interest=counter_factual,
        X=orginal_X,
        categorical_features=categorical_features_all,
    )
    dissparsity_values = dissparsity(
        point_of_interest=X_points,
        cf_point_of_interest=counter_factual,
    )

    implausability_values = implausability(
        cf_point_of_interest=counter_factual,
        X=orginal_X,
        categorical_features=categorical_features_all,
    )

    proportion_counterfactuals = counter_factual.shape[0] / X_points.shape[0]

    discriminative_values = []
    for i in range(counter_factual.shape[0]):
        mask = y_test == y_points[i]  # y_points will never be DESIRED_CLASS
        X_equal_poi = torch.Tensor(X_test[mask])
        X_diff_poi = torch.Tensor(X_test[y_test == DESIRED_CLASS])
        if counter_factual[i].isnan().any():
            discriminative_values.append(np.nan)
            continue
        discriminative_values.append(
            discriminative_power(
                point_of_interest=X_points[i : i + 1],
                cf_point_of_interest=counter_factual[i : i + 1],
                class_poi=y_points[i : i + 1],
                class_cf=torch.ones_like(y_points[i : i + 1]) * DESIRED_CLASS,
                X_equal_poi=X_equal_poi,
                X_diff_poi=X_diff_poi,
            )
        )

    discriminative_values = torch.Tensor(discriminative_values)
    return {
        "instability": instability_values,
        "invalidity": invalidity_values,
        "dissimilarity": dissimilarity_values,
        "dissparsity": dissparsity_values,
        "implausability": implausability_values,
        "discriminative_power": discriminative_values,
        "proportion_counterfactuals": proportion_counterfactuals,
    }


def get_point_of_interest_and_closest(X_test, y_test, num_points=100):
    rng = np.random.default_rng(42)

    points_of_interest = rng.choice(
        len(X_test[y_test != DESIRED_CLASS]),
        size=min(100, len(X_test[y_test != DESIRED_CLASS])),
        replace=False,
    )
    print("POI INDICES: ", points_of_interest)
    points_closest_to_interest = []
    for i in points_of_interest:
        # extract a point closest to this point, which is of the other class and not the same point
        distances = np.linalg.norm(
            X_test[y_test != DESIRED_CLASS]
            - (X_test[y_test != DESIRED_CLASS][i]).reshape(1, -1),
            axis=1,
        )
        mask = distances == 0
        distances[mask] = np.inf
        idx = distances.argmin()
        points_closest_to_interest.append(idx)
    points_closest_to_interest = np.array(points_closest_to_interest)
    X_points = torch.Tensor(X_test[points_of_interest])
    y_points = torch.Tensor(y_test[points_of_interest])
    X_points_close = torch.Tensor(X_test[points_closest_to_interest])
    y_points_close = torch.Tensor(y_test[points_closest_to_interest])
    return X_points, X_points_close, y_points, y_points_close


def get_dataset_sweep(sweep_kwargs):
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
    # Restructure to collect rows for each method
    metric_results = []

    MODEL_NAME = args.model_name

    for dataset_name in data_files:
        X_train, X_test, y_train, y_test = get_dataset(dataset_name)
        orginal_X = torch.Tensor(np.concatenate([X_train, X_test], axis=0))
        categorical_features_all = get_categorical_features_all(dataset_name)
        (
            X_points,
            X_points_close,
            y_points,
            y_points_close,
        ) = get_point_of_interest_and_closest(X_test, y_test)

        architecture = MLP(
            input_dim=X_train.shape[1],
            output_dim=len(np.unique(y_train)),
            hidden_dims=[100, 100],
            batch_norm=False,
        )
        ### Setup Model ###
        if (
            MODEL_NAME == "deep_ensemble"
            or MODEL_NAME == "dare_ensemble"
            or MODEL_NAME == "adversarial_ensemble"
        ):
            model = Ensemble(architecture, n_members=20)
        else:
            raise ValueError(f"Model {MODEL_NAME} not recognized!")
        # Evaluate the ensemble model
        model = Temperature(model)
        model.load_state_dict(
            torch.load(f"models/model={MODEL_NAME}_dataset={dataset_name}.pth")
        )
        model.compile()
        model.eval()

        for property_name in PROPERTY_NAMES:
            saved_cfs = load_property_data(
                MODEL_NAME, dataset_name, property_name, sweep_kwargs
            )
            print(f"Dataset: {dataset_name}, Property: {property_name}")
            counter_factual = torch.Tensor(saved_cfs["counter_factual"])
            counter_factual_closest = torch.Tensor(saved_cfs["counter_factual_closest"])
            metrics = get_metrics(
                X_points,
                X_points_close,
                y_points,
                y_points_close,
                counter_factual,
                counter_factual_closest,
                X_test,
                y_test,
                model,
                orginal_X,
                categorical_features_all,
            )

            # Create a row for this method/dataset combination
            row = {
                "dataset": dataset_name,
                "method": property_name,
                "instability": metrics["instability"].mean().item(),
                "invalidity": metrics["invalidity"].mean().item(),
                "dissimilarity": metrics["dissimilarity"].mean().item(),
                "dissparsity": metrics["dissparsity"].mean().item(),
                "implausability": metrics["implausability"].mean().item(),
                "discriminative_power": metrics["discriminative_power"].mean().item(),
                "proportion_counterfactuals": metrics["proportion_counterfactuals"],
            }
            metric_results.append(row)

        for method_name in ["DICE", "GS"]:
            carla_data = load_carla_data(dataset_name, method_name)
            if carla_data is None:
                print(f"Dataset: {dataset_name}, Method: {method_name} not found")
                continue
            counter_factual = torch.Tensor(carla_data["counter_factual"])
            counter_factual_closest = torch.Tensor(
                carla_data["counter_factual_closest"]
            )
            print(f"Dataset: {dataset_name}, Method: {method_name}")
            metrics = get_metrics(
                X_points,
                X_points_close,
                y_points,
                y_points_close,
                counter_factual,
                counter_factual_closest,
                X_test,
                y_test,
                model,
                orginal_X,
                categorical_features_all,
            )

            # Create a row for this method/dataset combination
            row = {
                "dataset": dataset_name,
                "method": method_name,
                "instability": metrics["instability"].mean().item(),
                "invalidity": metrics["invalidity"].mean().item(),
                "dissimilarity": metrics["dissimilarity"].mean().item(),
                "dissparsity": metrics["dissparsity"].mean().item(),
                "implausability": metrics["implausability"].mean().item(),
                "discriminative_power": metrics["discriminative_power"].mean().item(),
                "proportion_counterfactuals": metrics["proportion_counterfactuals"],
            }
            metric_results.append(row)

    # Create DataFrame with proper structure
    data = pd.DataFrame(metric_results)
    return data


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    MODEL_NAME = args.model_name
    with open("sweep_configurations.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines]

    lowest_score = float("inf")
    best_kwargs = None
    best_data = None
    for sweep_kwargs in lines:
        print("Sweep kwargs: ", sweep_kwargs)
        data = get_dataset_sweep(sweep_kwargs)

        score_data = data.copy()
        # Make all metrics lower is better
        score_data["discriminative_power"] = 1 - score_data["discriminative_power"]
        score_data["proportion_counterfactuals"] = (
            1 - score_data["proportion_counterfactuals"]
        )
        # Normalize the metrics to [0,1] for each metric
        for metric in [
            "instability",
            "invalidity",
            "dissimilarity",
            "dissparsity",
            "implausability",
            "discriminative_power",
            "proportion_counterfactuals",
        ]:
            min_val = score_data[metric].min()
            max_val = score_data[metric].max()
            if max_val - min_val > 0:
                score_data[metric] = (score_data[metric] - min_val) / (
                    max_val - min_val
                )
            else:
                score_data[metric] = 0.0
        score_data["score"] = score_data[
            [
                "instability",
                "invalidity",
                "dissimilarity",
                "dissparsity",
                "implausability",
                "discriminative_power",
                "proportion_counterfactuals",
            ]
        ].mean(axis=1)

        mask = score_data["method"].isin(PROPERTY_NAMES)
        data_score = score_data[mask]["score"].mean()
        print(f"Mean score of proposed methods: {data_score} (lower is better)")
        if data_score < lowest_score:
            lowest_score = data_score
            best_kwargs = sweep_kwargs
            best_data = data

    print("Best sweep kwargs: ", best_kwargs)
    print("Best score: ", lowest_score)
    print("Best data: ", best_data)
    best_data.to_csv(f"best_sweep_results_{MODEL_NAME}.csv", index=False)
    print("Saved best results to best_sweep_results.csv")
