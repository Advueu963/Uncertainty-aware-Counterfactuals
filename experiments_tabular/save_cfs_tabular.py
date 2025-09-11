import os

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

from training.train_ensemble_tabular import get_dataset

DESIRED_VALIDITY = 0.999
DELTA = 0.2
OPTIMIZER_LR = 0.01
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = 50
N_EPOCHS = 50
SAVE_FOLDER = "data"

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
    os.makedirs(SAVE_FOLDER)


def save_cfs_tabular(points, y_labels, point_to_explain, dataset_name):
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
        "dataset_name": dataset_name,
    }

    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
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
        )
        res[property_name] = {
            "counter_factual": counter_factual,
            "counter_factual_steps": counter_factual_steps,
        }
    np.save(
        os.path.join(
            SAVE_FOLDER,
            f"CFs_{dataset_name}.npy",
        ),
        res,
        allow_pickle=True,
    )


if __name__ == "__main__":
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
        points_of_interest = np.random.choice(
            len(X_test[y_test != DESIRED_CLASS]),
            size=min(100, len(X_test[y_test != DESIRED_CLASS])),
            replace=False,
        )
        print("Points of interest indices", len(points_of_interest))
        X_points = torch.Tensor(X_test[points_of_interest])
        save_cfs_tabular(X_test, y_test, X_points, dataset_name)
