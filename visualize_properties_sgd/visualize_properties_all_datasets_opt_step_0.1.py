import torch
import numpy as np
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from data import (
    load_l_dataset,
    load_ring_dataset,
    load_bubbles,
    load_bubbles_noisy,
    load_one_moon,
    load_two_moon,
    load_infinity_dataset,
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
)
import matplotlib.pyplot as plt
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
    visualize_au_eu_tu,
    visualize_eu_std_points,
    visualize_decision_boundry,
    visualze_property,
    visualize_other_cf_methods,
)

DESIRED_VALIDITY = 0.999
DELTA = 0.2
OPTIMIZER_LR = 0.1
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = 50
N_EPOCHS = 50
base_ensemble = [
    MLP_Classifier(
        input_shape=2,
        n_classes=2,
        n_layers=2,
        num_neurons=64,
        dropout_prob=0,
        batch_norm=False,
    )
    for _ in range(ENSEMBLE_MEMBER_COUNT)
]


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
    # ("combined", combined_loss_function),
]

fig, axes = plt.subplots(
    len(DATASET_LOADERS), len(PROPERTY_LOADERS) + 3 + 2 + 1 + 4, figsize=(90, 45)
)

for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
    print(f"Training ensemble on {dataset_name}...")
    points, y_labels, y_probs = loader(**kwargs)

    # Train the ensemble model
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"../models/Ensemble_{dataset_name.capitalize()}_50/")

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
            axes[i, j],
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
        )
    print(f"Visualizing AU, EU, TU for dataset: {dataset_name}")
    visualize_au_eu_tu(fig, ensemble_model, points, y_labels, axes[i, -10:-7])

    # Visualize the MAX EU and AU globally
    visualize_eu_std_points(
        fig,
        ensemble_model,
        points,
        y_labels,
        axes[i, -7:-5],
        delta=DELTA,
        n_points=N_POINTS,
    )

    print(f"Visualize other CF Methods for dataset: {dataset_name}")
    # Visualize other CF Methods
    visualize_other_cf_methods(
        point_of_interest,
        points,
        y_labels,
        axes[i, -5:-1],
        dataset_name,
        n_models=ENSEMBLE_MEMBER_COUNT,
        n_epochs=50,
        noisy=False,
    )

    print(f"Visualize decision boundary for dataset: {dataset_name}")
    visualize_decision_boundry(
        axes[i, -1],
        points,
        ensemble_model,
        y_labels,
        dataset_name,
        probability_function=ensemble_probs,
    )


plt.savefig("visualize_properties_all_datasets_opt_step_size_0.1.pdf")
# plt.show()
