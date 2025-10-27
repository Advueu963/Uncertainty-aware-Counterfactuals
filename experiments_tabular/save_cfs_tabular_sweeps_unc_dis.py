import os
import time
import argparse
import numpy as np
import torch
from probly.representation import Ensemble, Dropout, Bayesian
from probly.calibration import Temperature

from uncertainty_cfs.architectures import MLP
from uncertainty_cfs.property_procedures import (
    combined_loss_function_distance,
    counter_factual_optimization_routine,
)
from uncertainty_cfs.property_procedures.utils import (
    predict_probs,
    aleatoric_uncertainty_entropy,
    epistemic_uncertainty_entropy,
)

from uncertainty_cfs.tabular_util import (
    get_dataset,
    get_categorical_feature_lists,
    get_immutable_feature_idx,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="deep_ensemble",
    choices=[
        "deep_ensemble",
        "dare_ensemble",
        "adversarial_ensemble",
        "bayesian",
        "dropout",
    ],
    help="Type of model to use",
)
args = parser.parse_args()

MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_EPOCHS = 50
SAVE_FOLDER = os.path.join(os.environ.get("SCRATCH_DSS"), "property_tabular")
PROPERTY_LOADERS = [
    ("combined", combined_loss_function_distance),
]


if not os.path.exists(SAVE_FOLDER):
    raise ValueError(f"Save folder {SAVE_FOLDER} does not exist!")


def save_cfs_tabular(
    points,
    y_labels,
    point_to_explain,
    point_close_to_explain,
    dataset_name,
    categorical_features_lists=[],
    immutable_features_lists=[],
    configs={},
):
    # Load hyperparameters
    DESIRED_CLASS = configs.get("desired_class", 1)
    MAX_STEPS = configs.get("max_steps", 5000)
    PATIENCE = configs.get("patience", 5000)
    N_POINTS = configs.get("n_points", 20)
    OPTIMIZER_LR = configs.get("optimizer_lr", 0.1)
    DELTA = configs.get("delta", 1.0)
    DESIRED_VALIDITY = configs.get("desired_validity", 0.8)
    LAMBDA_1 = configs.get("lambda_1", 1.0)
    LAMBDA_2 = configs.get("lambda_2", 1.0)
    PROB_WEIGHT = configs.get("prob_weight", 1.0)
    MODEL_NAME = configs.get("model_name", "ensemble")

    architecture = MLP(
        input_dim=points.shape[1],
        output_dim=len(np.unique(y_labels)),
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
    elif MODEL_NAME == "bayesian":
        model = Bayesian(architecture)
    elif MODEL_NAME == "dropout":
        model = Dropout(architecture, p=0.2)
    # Evaluate the ensemble model
    model = Temperature(model)
    model.load_state_dict(
        torch.load(f"models/model={MODEL_NAME}_dataset={dataset_name}.pth")
    )
    model.compile()
    model.eval()
    # Print accuracy
    with torch.no_grad():
        probs = predict_probs(model, torch.Tensor(points))
        y_pred = probs.mean(dim=1).argmax(dim=1)
        accuracy = (y_pred == y_labels).float().mean().item()
        print(f"Model: {MODEL_NAME}, Dataset: {dataset_name}, Accuracy: {accuracy:.4f}")

    # Print some results
    print(f"Dataset: {dataset_name}")

    res = {
        "point_of_interest": point_to_explain,
        "point_closest_to_interest": point_close_to_explain,
        "dataset_name": dataset_name,
    }

    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        a = time.perf_counter()
        counter_factual, counter_factual_steps = counter_factual_optimization_routine(
            point_to_explain=point_to_explain,
            model=model,
            probability_function=predict_probs,
            desired_class=DESIRED_CLASS,
            loss_function=property_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_entropy,
            epistemic_uncertainty_function=epistemic_uncertainty_entropy,
            MAX_STEPS=MAX_STEPS,
            delta=DELTA,
            n_points=N_POINTS,
            lr=OPTIMIZER_LR,
            DESIRED_VALIDITY=DESIRED_VALIDITY,
            p_weight=PROB_WEIGHT,
            lambda_1=LAMBDA_1,
            lambda_2=LAMBDA_2,
            patience=PATIENCE,
            optimization_method="adam",
            categorical_features_lists=categorical_features_lists,
            immutable_features_lists=immutable_features_lists,
        )
        b = time.perf_counter()
        print(f"Time taken for {property_name}: {b - a:.4f} seconds")
        (
            counter_factual_closest,
            counter_factual_steps_closest,
        ) = counter_factual_optimization_routine(
            point_to_explain=point_close_to_explain,
            model=model,
            probability_function=predict_probs,
            desired_class=DESIRED_CLASS,
            loss_function=property_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_entropy,
            epistemic_uncertainty_function=epistemic_uncertainty_entropy,
            MAX_STEPS=MAX_STEPS,
            delta=DELTA,
            n_points=N_POINTS,
            lr=OPTIMIZER_LR,
            DESIRED_VALIDITY=DESIRED_VALIDITY,
            p_weight=PROB_WEIGHT,
            lambda_1=LAMBDA_1,
            lambda_2=LAMBDA_2,
            patience=PATIENCE,
            optimization_method="adam",
            categorical_features_lists=categorical_features_lists,
            immutable_features_lists=immutable_features_lists,
        )
        res[property_name] = {
            "counter_factual": counter_factual,
            "counter_factual_closest": counter_factual_closest,
            "counter_factual_steps": counter_factual_steps,
            "counter_factual_steps_closest": counter_factual_steps_closest,
            "time": b - a,
            "steps": counter_factual_steps.shape[0],
        }
    np.save(
        os.path.join(
            SAVE_FOLDER,
            f"CFs_{MODEL_NAME}_{dataset_name}_{DESIRED_VALIDITY}_{DELTA}_{OPTIMIZER_LR}_{LAMBDA_1}_{LAMBDA_2}_{PROB_WEIGHT}_{N_POINTS}_uncertainty+distance_test.npy",
        ),
        res,
        allow_pickle=True,
    )


def get_point_of_interest_and_closest(X_test, y_test, rng):
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
    return points_of_interest, points_closest_to_interest


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)
    number = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    print(f"Running with SLURM_ARRAY_TASK_ID={number}")
    # Load hyperparameters from file
    with open("sweep_configurations.txt", "r") as f:
        lines = f.readlines()

    params = lines[number].strip().split(",")
    print("Loaded parameters: ", params)
    configs = {
        "desired_validity": float(params[0]),
        "delta": float(params[1]),
        "optimizer_lr": float(params[2]),
        "lambda_1": float(params[3]),
        "lambda_2": float(params[4]),
        "prob_weight": float(params[5]),
        "n_points": int(params[6]),
        "ensemble_member_count": ENSEMBLE_MEMBER_COUNT,
        "n_epochs": N_EPOCHS,
        "desired_class": DESIRED_CLASS,
        "max_steps": MAX_STEPS,
        "patience": PATIENCE,
        "model_name": args.model_name,
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
    for dataset_name in data_files:
        _, X_test, _, y_test = get_dataset(dataset_name)
        categorical_features_lists = get_categorical_feature_lists(dataset_name)
        immutable_features_lists = get_immutable_feature_idx(dataset_name)
        (
            points_of_interest,
            points_closest_to_interest,
        ) = get_point_of_interest_and_closest(X_test, y_test, rng)

        print("Points of interest indices", len(points_of_interest))
        X_points = torch.Tensor(X_test[points_of_interest])
        X_points_close = torch.Tensor(X_test[points_closest_to_interest])
        print("Points closest to interest indices", len(points_closest_to_interest))
        print("Categorical features lists: ", categorical_features_lists)
        print("Immutable features lists: ", immutable_features_lists)
        save_cfs_tabular(
            X_test,
            y_test,
            X_points,
            X_points_close,
            dataset_name,
            categorical_features_lists=categorical_features_lists,
            immutable_features_lists=immutable_features_lists,
            configs=configs,
        )
