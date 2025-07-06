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
    dissimilarity,
)
from synthetic_to_carla import Synthetic_CARLA, MyOwnModel

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
N_ITERATIONS = 5
OPTIMIZATION_METHOD = "sgd"  # "adam" or "sgd"
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
    counterfactuals_growing_sphere = gs.get_counterfactuals(point_of_interest_df)

    counterfactuals_dice = dice.get_counterfactuals(point_of_interest_df)

    # generate counterfactual examples using CLUE
    try:
        counterfactuals_clue = clue.get_counterfactuals(point_of_interest_df)
        print(counterfactuals_clue)
    except ValueError as e:
        print(f"Error generating counterfactuals with CLUE: {e}")
        counterfactuals_clue = None

    try:
        counterfactuals_face = fa.get_counterfactuals(point_of_interest_df)
        print(counterfactuals_face)
    except ValueError as e:
        print(f"Error generating counterfactuals with Face: {e}")
        counterfactuals_face = None

    # Compute the pairwise L2 distance between the counterfactuals
    if counterfactuals_growing_sphere is not None:
        l2_distance_growing_sphere = dissimilarity(
            point_of_interest,
            torch.from_numpy(counterfactuals_growing_sphere.values[0]),
            points,
        )
    else:
        l2_distance_growing_sphere = torch.tensor([-1], dtype=torch.float32)
        print(
            "No counterfactuals found through Growing Spheres. Skipping distance calculation."
        )

    if counterfactuals_clue is not None:
        l2_distance_clue = dissimilarity(
            point_of_interest, torch.from_numpy(counterfactuals_clue.values[0]), points
        )
    else:
        l2_distance_clue = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through CLUE. Skipping distance calculation.")

    if counterfactuals_dice is not None:
        l2_distance_dice = dissimilarity(
            point_of_interest, torch.from_numpy(counterfactuals_dice.values[0]), points
        )
    else:
        l2_distance_dice = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through DICE. Skipping distance calculation.")

    if counterfactuals_face is not None:
        l2_distance_face = dissimilarity(
            point_of_interest, torch.from_numpy(counterfactuals_face.values[0]), points
        )
    else:
        l2_distance_face = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through Face. Skipping distance calculation.")

    for ax, cf, title in zip(
        axes.flatten(),
        [
            counterfactuals_growing_sphere,
            counterfactuals_clue,
            counterfactuals_dice,
            counterfactuals_face,
        ],
        ["Growing Spheres", "CLUE", "DICE", "Face"],
    ):
        if cf is None:
            print(f"No counterfactuals found through {title}. Skipping visualization.")
            continue
        # visualize the data itself
        ax.scatter(
            points[y_labels == 0, 0],
            points[y_labels == 0, 1],
            marker="s",
            s=2,
            alpha=0.3,
            color="blue",
            label="Class 0",
        )
        ax.scatter(
            points[y_labels == 1, 0],
            points[y_labels == 1, 1],
            marker="o",
            s=2,
            alpha=0.3,
            color="red",
            label="Class 1",
        )
        ax.scatter(
            point_of_interest[:, 0].detach(),
            point_of_interest[:, 1].detach(),
            marker="s",
            s=50,
            alpha=0.6,
            color="indigo",
            label="Point of Interest",
        )
        ax.scatter(
            cf.values[0:1, 0],
            cf.values[0:1, 1],
            marker="s",
            s=50,
            color="aqua",
            label=f"CF_{title}",
        )
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title(f"{dataset_name} - {title}")

    return (
        l2_distance_growing_sphere,
        l2_distance_clue,
        l2_distance_dice,
        l2_distance_face,
    )


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

    l2_distance_stable.append([])
    l2_distance_stable_std.append([])
    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        cf_poi = []
        for _ in range(N_ITERATIONS):
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
                optimization_method=OPTIMIZATION_METHOD,
            )
            cf_poi.append(counter_factual)

        l2_distance = [
            dissimilarity(point_of_interest, cf_poi[i], points).detach().numpy()
            for i in range(len(cf_poi))
        ]
        l2_distance_stable[-1].append(np.mean(l2_distance).item())
        l2_distance_stable_std[-1].append(np.std(l2_distance).item())

    # Visualize other CF Methods
    l2_distances_mean = []
    for _ in range(N_ITERATIONS):
        (
            l2_distance_growing_sphere,
            l2_distance_clue,
            l2_distance_dice,
            l2_distance_face,
        ) = visualize_other_cf_methods(
            point_of_interest, points, y_labels, axes[i, -5:-1], dataset_name
        )
        l2_distances_mean.append(
            [
                l2_distance_growing_sphere.mean().item(),
                l2_distance_clue.mean().item(),
                l2_distance_dice.mean().item(),
                l2_distance_face.mean().item(),
            ]
        )

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
l2_stability_data.to_csv(f"similarity_{OPTIMIZATION_METHOD}.csv")
# plt.show()
