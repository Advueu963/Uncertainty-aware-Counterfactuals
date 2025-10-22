import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from probly.representation import Ensemble, Dropout, Bayesian
from probly.calibration import Temperature

from uncertainty_cfs.architectures import MLP
from uncertainty_cfs.property_procedures import (
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
from uncertainty_cfs.property_procedures.utils import (
    predict_probs,
    aleatoric_uncertainty_entropy,
    epistemic_uncertainty_entropy,
)
from uncertainty_cfs.data import (
    load_l_dataset,
    load_ring_dataset,
    load_bubbles,
    load_bubbles_noisy,
    load_one_moon,
    load_two_moon,
    load_infinity_dataset,
)
from uncertainty_cfs.property_procedures.utils import (
    visualize_decision_boundry,
    visualize_au_eu_tu,
    visualze_property
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="deep_ensemble", choices=["deep_ensemble","dare_ensemble","adversarial_ensemble", "bayesian", "dropout"], help="Type of model to use")
args = parser.parse_args()


DESIRED_VALIDITY = 0.999
DELTA = 0.2
OPTIMIZER_LR = 0.01
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
DESIRED_CLASS = 1
N_POINTS = 50
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_EPOCHS = 50
SAVE_FOLDER = os.path.join(os.environ.get("SCRATCH_DSS"),"property_tabular")
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

if __name__ == "__main__":
    
    DATASET_LOADERS = [
        (
            "bubbles",
            load_bubbles,
            {"n_samples": 1000},
            torch.tensor(
                np.array([3, 3]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
        (
            "l_dataset",
            load_l_dataset,
            {"n_samples": 333},
            torch.tensor(
                np.array([0, 10]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
        (
            "one_moon",
            load_one_moon,
            {"n_samples": 1000},
            torch.tensor(
                np.array([-1, 0]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
        (
            "ring_dataset",
            load_ring_dataset,
            {"n_samples": 1000, "inner_radius": 1.0, "outer_radius": 2.0, "noise": 0.1},
            torch.tensor(
                np.array([-1, -2]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
        (
            "bubbles_noisy",
            load_bubbles_noisy,
            {"n_samples": 1000},
            torch.tensor(
                np.array([2.5, 2.5]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
        (
            "two_moon",
            load_two_moon,
            {"n_samples": 1000},
            torch.tensor(
                np.array([0, 1]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
        (
            "infinity_dataset",
            load_infinity_dataset,
            {"n_samples": 1000},
            torch.tensor(
                np.array([2, 0]).reshape(-1, 2), dtype=torch.float, requires_grad=True
            ),
        ),
    ]
    MODEL_NAME = args.model_name

    for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
        fig, axes = plt.subplots(nrows=1, ncols=len(PROPERTY_LOADERS) + 4, figsize=(80, 5))
        print(f"Training ensemble on {dataset_name}...")
        points, y_labels, y_probs = loader(**kwargs)

        
        architecture = MLP(
                input_dim=points.shape[1],
                output_dim=len(np.unique(y_labels)),
                hidden_dims=[64, 64],
                batch_norm=False,
        )
        ### Setup Model ###
        if MODEL_NAME == "deep_ensemble" or MODEL_NAME == "dare_ensemble" or MODEL_NAME == "adversarial_ensemble":
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
        print(f"Points shape: {points.shape}")
        print(f"Labels shape: {y_labels.shape}")
        print(f"Probabilities shape: {probs.shape}")
        print(
            f"Accuracy: {((y_pred == y_labels).sum() / len(y_labels)).item():.4f}"
        )
        print("-" * 40)

        for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
            print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

            # Run the property procedure
            counter_factual, counter_factual_steps = counter_factual_optimization_routine(
                point_to_explain=point_of_interest,
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
            )

            visualze_property(
                property_name,
                axes[j],
                model=model,
                points=points,
                y_labels=y_labels,
                point_of_interest=point_of_interest,
                counter_factual=counter_factual,
                counter_factual_steps=counter_factual_steps,
                p_weight=PROB_WEIGHT,
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
                dataset_name=dataset_name,
                delta=DELTA,
                n_points=N_POINTS,
            )


        plt.tight_layout()
        plt.savefig(f"{SAVE_FOLDER}/{dataset_name}_{MODEL_NAME}_counterfactuals.png", dpi=200)

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

        print(f"Visualizing AU, EU, TU for dataset: {dataset_name}")
        visualize_au_eu_tu(fig, model, points, y_labels, axes[:-1])

        print(f"Visualize decision boundary for dataset: {dataset_name}")
        visualize_decision_boundry(
            axes[-1],
            points,
            model,
            y_labels,
            dataset_name,
            probability_function=predict_probs,
        )
        plt.savefig(f"{SAVE_FOLDER}/{dataset_name}_{MODEL_NAME}_uncertainty.png", dpi=200)
        # plt.show()
