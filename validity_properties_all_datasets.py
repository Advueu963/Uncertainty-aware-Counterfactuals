import pandas as pd
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
    similarity_loss_function,
    validity_loss_function,
    robust_loss_function,
    feasable_loss_function,
    counter_factual_optimization_routine,
    combined_loss_function,
    connected_loss_function,
    discriminative_loss_function,
    plausable_loss_function,
)
import matplotlib.pyplot as plt
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
    invalidity,
    counter_factual_baseline,
)

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
    ("combined", combined_loss_function),
]


def visualize_other_cf_methods(
    point_of_interest,
    points,
    y_labels,
    points_poi,
    points_desired_poi,
    axes,
    dataset_name,
):
    (
        counterfactuals_clue,
        counterfactuals_dice,
        counterfactuals_face,
        counterfactuals_growing_sphere,
    ) = counter_factual_baseline(
        dataset_name,
        point_of_interest,
        n_models=ENSEMBLE_MEMBER_COUNT,
        n_epochs=N_EPOCHS,
        noisy=False,
    )
    # Compute the pairwise L2 distance between the counterfactuals
    if counterfactuals_growing_sphere is not None:
        invalidity_gs = invalidity(
            torch.from_numpy(counterfactuals_growing_sphere.values).float(),
            model=ensemble_model,
            probability_function=ensemble_probs,
            desired_class=DESIRED_CLASS,
        )
    else:
        invalidity_gs = torch.tensor([-1], dtype=torch.float32)
        print(
            "No counterfactuals found through Growing Spheres. Skipping distance calculation."
        )

    if counterfactuals_clue is not None:
        invalidity_clue = invalidity(
            torch.from_numpy(counterfactuals_clue.values).float(),
            model=ensemble_model,
            probability_function=ensemble_probs,
            desired_class=DESIRED_CLASS,
        )
    else:
        invalidity_clue = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through CLUE. Skipping distance calculation.")

    if counterfactuals_dice is not None:
        invalidity_dice = invalidity(
            torch.from_numpy(counterfactuals_dice.values).float(),
            model=ensemble_model,
            probability_function=ensemble_probs,
            desired_class=DESIRED_CLASS,
        )
    else:
        invalidity_dice = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through DICE. Skipping distance calculation.")

    if counterfactuals_face is not None:
        invalidity_face = invalidity(
            torch.from_numpy(counterfactuals_face.values),
            model=ensemble_model,
            probability_function=ensemble_probs,
            desired_class=DESIRED_CLASS,
        )
    else:
        invalidity_face = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through Face. Skipping distance calculation.")

    return (
        invalidity_gs,
        invalidity_clue,
        invalidity_dice,
        invalidity_face,
    )


fig, axes = plt.subplots(
    len(DATASET_LOADERS), len(PROPERTY_LOADERS) + 3 + 2 + 1 + 4, figsize=(80, 40)
)

invalidity_cfs_mean = []
invalidity_cfs_std = []
for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
    print(f"Training ensemble on {dataset_name}...")
    points, y_labels, y_probs = loader(**kwargs)

    # select points closest to point_of_interest with same label
    point_same_label = points[y_labels == 0]
    distances = torch.norm(point_same_label - point_of_interest.reshape(1, -1), dim=1)
    closest_indices = torch.argsort(distances)
    points_poi = point_same_label[closest_indices][:5]

    point_desired_label = points[y_labels == DESIRED_CLASS]
    distances = torch.norm(
        point_desired_label - point_of_interest.reshape(1, -1), dim=1
    )
    closest_indices = torch.argsort(distances)
    points_desired_poi = point_desired_label[closest_indices][:5]

    # Train the ensemble model
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")

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

    invalidity_cfs_mean.append([])
    invalidity_cfs_std.append([])
    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        cf_poi = []
        for _ in range(5):
            (
                counter_factual,
                counter_factual_steps,
            ) = counter_factual_optimization_routine(
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
            cf_poi.append(counter_factual)

        invalidity_cf_property = [
            invalidity(
                cf_poi[i].detach().numpy(),
                model=ensemble_model,
                probability_function=ensemble_probs,
                desired_class=DESIRED_CLASS,
            )
            for i in range(len(cf_poi))
        ]
        invalidity_cfs_mean[-1].append(np.mean(invalidity_cf_property).item())
        invalidity_cfs_std[-1].append(np.std(invalidity_cf_property).item())

    # Visualize other CF Methods
    invalidity_baselines_mean = []
    for _ in range(5):
        (
            invalidity_gs,
            invalidity_clue,
            invalidity_dice,
            invalidity_face,
        ) = visualize_other_cf_methods(
            point_of_interest,
            points,
            y_labels,
            points_poi,
            points_desired_poi,
            axes[i, -5:-1],
            dataset_name,
        )
        invalidity_baselines_mean.append(
            [
                invalidity_gs.mean().item(),
                invalidity_clue.mean().item(),
                invalidity_dice.mean().item(),
                invalidity_face.mean().item(),
            ]
        )

    invalidity_cfs_mean[-1] += [*np.mean(invalidity_baselines_mean, axis=0)]
    invalidity_cfs_std[-1] += [*np.std(invalidity_baselines_mean, axis=0)]
    print(invalidity_cfs_mean[-1])

invalidity_data = pd.DataFrame(
    np.array(
        [
            [
                f"{values} +/- {std}"
                for values, std in zip(invalidity_cfs_mean[i], invalidity_cfs_std[i])
            ]
            for i in range(len(invalidity_cfs_mean))
        ]
    ),
    index=[f"{dataset_name}" for dataset_name, _, _, _ in DATASET_LOADERS],
    columns=[property_name for property_name, _ in PROPERTY_LOADERS]
    + ["GS", "CLUE", "DICE", "FACE"],
)
print(invalidity_data)
# Save the L2 stability data to a CSV file
invalidity_data.to_csv("validity_data.csv")
# plt.show()
