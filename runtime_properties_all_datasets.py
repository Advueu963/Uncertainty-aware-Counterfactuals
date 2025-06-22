from time import perf_counter

import pandas as pd
import torch
import numpy as np
from carla.recourse_methods import GrowingSpheres, Clue, Dice, Face
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
)
import matplotlib.pyplot as plt
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
)
from synthetic_to_carla import Synthetic_CARLA, MyOwnModel

DESIRED_VALIDITY = 0.999
DELTA = 0.5
OPTIMIZER_LR = 0.1
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
MAX_STEPS = 1000
PATIENCE = 50
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = 50
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
            np.array([3.9, 3.9]).reshape(-1, 2), dtype=torch.float, requires_grad=True
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
    # (
    #     "connected_ball",
    #     connected_loss_function,
    # ),
    (
        "robust",
        robust_loss_function,
    ),
    (
        "feasability",
        feasable_loss_function,
    ),
    # (
    #     "discriminative",
    #     discriminative_loss_function,
    # ),
    # (
    #     "plausable",
    #     plausable_loss_function,
    # ),
    (
        "similarity",
        similarity_loss_function,
    ),
    ("combined", combined_loss_function),
]


def visualize_other_cf_methods(point_of_interest, points, y_labels, axes, dataset_name):
    dataset = Synthetic_CARLA(dataset_name)

    model = MyOwnModel(dataset)

    # load artificial neural networke from catalog
    # model = MLModelCatalog(dataset, "ann", backend="pytorch")

    # load a recourse model and pass black box model
    gs = GrowingSpheres(model)
    clue = Clue(dataset, model)
    dice = Dice(model)
    fa = Face(model, {"mode": "knn", "fraction": 0.2})

    point_of_interest_df = pd.DataFrame(
        {
            "x0": [p for p in [*point_of_interest[:, 0].detach().numpy()]],
            "x1": [p for p in [*point_of_interest[:, 1].detach().numpy()]],
            "label": [0] * (1),
        }
    )

    # generate counterfactual examples
    print("Factuals:")
    print(point_of_interest_df)
    a = perf_counter()
    _ = gs.get_counterfactuals(point_of_interest_df)
    b = perf_counter()
    time_growing_sphere = b - a

    a = perf_counter()
    _ = dice.get_counterfactuals(point_of_interest_df)
    b = perf_counter()
    time_dice = b - a

    # generate counterfactual examples using CLUE
    try:
        a = perf_counter()
        _ = clue.get_counterfactuals(point_of_interest_df)
        b = perf_counter()
        time_clue = b - a
    except ValueError as e:
        print(f"Error generating counterfactuals with CLUE: {e}")
        time_clue = np.inf

    try:
        a = perf_counter()
        _ = fa.get_counterfactuals(point_of_interest_df)
        b = perf_counter()
        time_face = b - a
    except ValueError as e:
        print(f"Error generating counterfactuals with Face: {e}")
        time_face = np.inf

    return time_growing_sphere, time_clue, time_dice, time_face


fig, axes = plt.subplots(
    len(DATASET_LOADERS), len(PROPERTY_LOADERS) + 3 + 2 + 1 + 4, figsize=(80, 40)
)

l2_distance_stable = []
l2_distance_stable_std = []
for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
    print(f"Training ensemble on {dataset_name}...")
    points, y_labels, y_probs = loader(**kwargs)

    # Train the ensemble model
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"models/Ensemble_{dataset_name.capitalize()}/")

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

    l2_distance_stable.append([])
    l2_distance_stable_std.append([])
    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        cf_time = []
        for _ in range(5):
            a = perf_counter()
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
            )
            b = perf_counter()
            cf_time.append(b - a)

        l2_distance_stable[-1].append(np.mean(cf_time).item())
        l2_distance_stable_std[-1].append(np.std(cf_time).item())

    # Visualize other CF Methods
    l2_distances_mean = []
    for _ in range(5):
        time_gs, time_clue, time_dice, time_face = visualize_other_cf_methods(
            point_of_interest, points, y_labels, axes[i, -5:-1], dataset_name
        )
        l2_distances_mean.append([time_gs, time_clue, time_dice, time_face])

    l2_distance_stable[-1] += [*np.mean(l2_distances_mean, axis=0)]
    l2_distance_stable_std[-1] += [*np.std(l2_distances_mean, axis=0)]
    print(l2_distance_stable[-1])

l2_stability_data = pd.DataFrame(
    np.array(
        [
            [
                f"{values} +/- {std}"
                for values, std in zip(l2_distance_stable[i], l2_distance_stable_std[i])
            ]
            for i in range(len(l2_distance_stable))
        ]
    ),
    index=[f"{dataset_name}" for dataset_name, _, _, _ in DATASET_LOADERS],
    columns=[property_name for property_name, _ in PROPERTY_LOADERS]
    + ["GS", "CLUE", "DICE", "FACE"],
)
print(l2_stability_data)
# Save the L2 stability data to a CSV file
l2_stability_data.to_csv("time_data.csv")
# plt.show()
