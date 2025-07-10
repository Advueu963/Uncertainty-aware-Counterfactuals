import os

import torch
import numpy as np
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from data import (
    load_bubbles_multiclass,
    load_l_dataset_multiclass,
    load_four_moon,
    load_ring_dataset_multiclass,
)
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
import matplotlib.pyplot as plt
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
    visualize_au_eu_tu,
    visualize_decision_boundry,
    visualze_property,
)

DESIRED_VALIDITY = 0.999
DELTA = 0.2
OPTIMIZER_LR = 0.01
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 2
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = 50
N_EPOCHS = 50
SAVE_FOLDER = "strips_all_datasets_muliclass"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
base_ensemble = [
    MLP_Classifier(
        input_shape=2,
        n_classes=4,
        n_layers=2,
        num_neurons=64,
        dropout_prob=0,
        batch_norm=False,
    )
    for _ in range(ENSEMBLE_MEMBER_COUNT)
]


DATASET_LOADERS = [
    (
        "bubbles_multiclass",
        load_bubbles_multiclass,
        {"n_samples": 1000},
        torch.tensor(
            np.array([3, 3]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "l_dataset_multiclass",
        load_l_dataset_multiclass,
        {"n_samples": 333},
        torch.tensor(
            np.array([0, 10]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "four_moon",
        load_four_moon,
        {"n_samples": 1000},
        torch.tensor(
            np.array([-1, 0]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "ring_dataset_multiclass",
        load_ring_dataset_multiclass,
        {"n_samples": 1000, "noise": 0.1},
        torch.tensor(
            np.array([-1, -2]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
]
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


for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
    fig, axes = plt.subplots(nrows=1, ncols=len(PROPERTY_LOADERS), figsize=(80, 10))
    print(f"Training ensemble on {dataset_name}...")
    points, y_labels, y_probs = loader(**kwargs)

    # Train the ensemble model
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"../models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")
    ensemble_model.compile()

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

    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        counter_factual, counter_factual_steps = counter_factual_optimization_routine(
            point_to_explain=point_of_interest,
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
            optimization_method="sgd",
        )

        visualze_property(
            property_name,
            axes[j],
            model=ensemble_model,
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
            DESIRED_CLASS=DESIRED_CLASS,
            multi_class=True,
        )

    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/{dataset_name}_counterfactuals.pdf", dpi=50)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    print(f"Visualizing AU, EU, TU for dataset: {dataset_name}")
    visualize_au_eu_tu(
        fig, ensemble_model, points, y_labels, axes[:-1], multi_class=True
    )

    print(f"Visualize decision boundary for dataset: {dataset_name}")
    visualize_decision_boundry(
        axes[-1],
        points,
        ensemble_model,
        y_labels,
        dataset_name,
        probability_function=ensemble_probs,
        multi_class=True,
    )
    plt.savefig(f"{SAVE_FOLDER}/{dataset_name}_uncertainty.pdf", dpi=50)
    # plt.show()
