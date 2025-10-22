import os
import time

import numpy as np
import torch
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from property_procedures import (
    validity_loss_function,
    connected_loss_function,
    robust_loss_function,
    feasable_loss_function,
    discriminative_loss_function,
    plausable_loss_function,
    similarity_loss_function,
    counter_factual_optimization_routine,
    combined_loss_function,
)
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
)

from training.train_ensemble_tabular import get_dataset, get_categorical_feature_lists, get_immutable_feature_idx
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_folder', type=str, default=os.path.join(os.environ.get("SCRATCH_DSS"),"property_tabular"))
parser.add_argument('--desired_validity', type=float, default=0.8)
parser.add_argument('--delta', type=float, default=1.0)
parser.add_argument('--optimizer_lr', type=float, default=0.1)
parser.add_argument('--n_points', type=int, default=50)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--prob_weight', type=float, default=1.0)
args = parser.parse_args()



DESIRED_VALIDITY = args.desired_validity
DELTA = args.delta
OPTIMIZER_LR = args.optimizer_lr
PROB_WEIGHT = args.prob_weight
LAMBDA_1 = args.lambda_1
LAMBDA_2 = args.lambda_2
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = args.n_points
N_EPOCHS = 50
SAVE_FOLDER = args.save_folder

PROPERTY_LOADERS = [
    (
        "validity",
        validity_loss_function,
    ),
    (
        "connected_ball",
        connected_loss_function,
    ),
    (
        "robust",
        robust_loss_function,
    ),
    (
        "feasability",
        feasable_loss_function,
    ),
    (
        "discriminative",
        discriminative_loss_function,
    ),
    (
        "plausable",
        plausable_loss_function,
    ),
    (
        "similarity",
        similarity_loss_function,
    ),
    ("combined", combined_loss_function),
]


if not os.path.exists(SAVE_FOLDER):
    raise ValueError(f"Save folder {SAVE_FOLDER} does not exist!")


def save_cfs_tabular(points, y_labels, 
                     point_to_explain,
                     point_close_to_explain,
                     dataset_name,
                     categorical_features_lists=[],
                     immutable_features_lists=[]
                     ):
    base_ensemble = [
        MLP_Classifier(
            input_shape=points.shape[1],
            n_classes=len(np.unique(y_labels)),
            n_layers=3,
            num_neurons=100,
            dropout_prob=0,
            batch_norm=True,
            random_state=i,
        )
        for i in range(ENSEMBLE_MEMBER_COUNT)
    ]
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")
    ensemble_model.compile(backend="inductor")
    ensemble_model.eval()

    # Evaluate the ensemble model
    y_probs_ensemble, _ = ensemble_model.predict(points, raw_output=True)
    y_target_ensemble = y_probs_ensemble.mean(dim=1).argmax(dim=1)

    # Print some results
    print(f"Dataset: {dataset_name}")
    print(f"Points shape: {points.shape}")
    print(f"Labels shape: {y_labels.shape}")
    print(f"Probabilities shape: {y_probs_ensemble.shape}")
    print(
        f"Accuracy: {((y_target_ensemble == y_labels).sum() / len(y_labels)).item():.4f}"
    )
    print("-" * 40)

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
            model=ensemble_model,
            probability_function=ensemble_probs,
            desired_class=DESIRED_CLASS,
            loss_function=property_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_ensemble,
            epistemic_uncertainty_function=epistemic_uncertainty_ensemble,
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
            immutable_features_lists=immutable_features_lists
        )
        b = time.perf_counter()
        print(f"Time taken for {property_name}: {b-a:.4f} seconds")
        counter_factual_closest, counter_factual_steps_closest = counter_factual_optimization_routine(
            point_to_explain=point_close_to_explain,
            model=ensemble_model,
            probability_function=ensemble_probs,
            desired_class=DESIRED_CLASS,
            loss_function=property_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_ensemble,
            epistemic_uncertainty_function=epistemic_uncertainty_ensemble,
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
            immutable_features_lists=immutable_features_lists
        )
        res[property_name] = {
            "counter_factual": counter_factual,
            "counter_factual_closest": counter_factual_closest,
            "counter_factual_steps": counter_factual_steps,
            "counter_factual_steps_closest": counter_factual_steps_closest,
            "time" : b-a,
            "steps": counter_factual_steps.shape[0],
            
        }
    np.save(
        os.path.join(
            SAVE_FOLDER,
            f"CFs_{dataset_name}_{DESIRED_VALIDITY}_{DELTA}_{OPTIMIZER_LR}_{LAMBDA_1}_{LAMBDA_2}_{PROB_WEIGHT}_{N_POINTS}.npy",
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
            X_test[y_test != DESIRED_CLASS] - (X_test[y_test != DESIRED_CLASS][i]).reshape(1, -1), axis=1
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

    data_files = [
        "adult",
        "bank",
        "churn",
        "compas",
        "diabetes",
        "fico",
        "german",
        "home",
        "titanic",
        "breast_cancer",
        "boston_housing",
    ]
    for dataset_name in data_files:
        _, X_test, _, y_test = get_dataset(dataset_name)
        categorical_features_lists = get_categorical_feature_lists(dataset_name)
        immutable_features_lists = get_immutable_feature_idx(dataset_name)
        points_of_interest, points_closest_to_interest = get_point_of_interest_and_closest(X_test, y_test, rng)
            
        print("Points of interest indices", len(points_of_interest))
        X_points = torch.Tensor(X_test[points_of_interest])
        X_points_close = torch.Tensor(X_test[points_closest_to_interest])
        print("Points closest to interest indices", len(points_closest_to_interest))
        print("Categorical features lists: ", categorical_features_lists)
        print("Immutable features lists: ", immutable_features_lists)
        save_cfs_tabular(X_test, y_test, X_points, X_points_close, dataset_name, categorical_features_lists=categorical_features_lists, immutable_features_lists=immutable_features_lists)
