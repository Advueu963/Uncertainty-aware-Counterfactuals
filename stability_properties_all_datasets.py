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
)
import matplotlib.pyplot as plt
import matplotlib as mpl
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
    total_uncertainty_ensemble,
    sample_delta_ball,
    instability_metric,
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


def visualize_au_eu_tu(points, y_labels, axes):
    # Visualze AU, EU, TU
    grid_points = np.linspace(points.min() - 1, points.max() + 1, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T
    # Get the labels of the grid_points
    y_probs_ensemble = ensemble_probs(
        ensemble_model, torch.tensor(grid_points, dtype=torch.float32)
    )
    total_uncertainty = total_uncertainty_ensemble(y_probs_ensemble)
    aleatoric_uncertainty = aleatoric_uncertainty_ensemble(y_probs_ensemble)
    epistemic_uncertainty = epistemic_uncertainty_ensemble(y_probs_ensemble)
    # Shared value range
    all_vals = torch.concatenate(
        [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()
    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis
    plots = []
    for ax, values, title in zip(
        axes,
        [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty],
        ["TU", "AU", "EU"],
    ):
        # Scatter data
        ax.scatter(
            points[y_labels == 0, 0],
            points[y_labels == 0, 1],
            marker="s",
            s=2,
            alpha=0.5,
            color="blue",
            label="Class 0",
        )
        ax.scatter(
            points[y_labels == 1, 0],
            points[y_labels == 1, 1],
            marker=".",
            s=2,
            alpha=0.5,
            color="red",
            label="Class 1",
        )

        # Unified contourf
        contour = ax.contourf(
            xx,
            yy,
            values.detach().numpy().reshape(xx.shape),
            levels=levels,
            cmap=cmap,
            norm=norm,
            alpha=0.5,
        )
        plots.append(contour)

        ax.set_title(title)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
    # Single shared colorbar using one of the contour handles
    fig.colorbar(plots[0], ax=axes, shrink=0.8, label="Uncertainty Measure Value")
    print("_" * 40)


def visualize_path(
    ax,
    points,
    y_labels,
    point_of_interest,
    counter_factual,
    counter_factual_steps,
    dataset_name,
    property_name,
):
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
        alpha=1,
        color="indigo",
        label="Point of Interest",
    )
    ax.scatter(
        counter_factual[:, 0],
        counter_factual[:, 1],
        marker="s",
        s=50,
        color="aqua",
        label="CF",
    )
    ax.scatter(
        counter_factual_steps[:, 0],
        counter_factual_steps[:, 1],
        marker=".",
        s=3,
        alpha=0.5,
        color="lime",
        label="CF Steps",
    )
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("CF Path")
    ax.set_title(f"{dataset_name} - {property_name}")


def visualze_path_with_underlying(
    model,
    ax,
    points,
    y_labels,
    point_of_interest,
    counter_factual,
    counter_factual_steps,
    lambda_1,
    lambda_2,
    dataset_name,
    property_name,
    delta=1,
    n_points=10,
):
    # Create gird of points
    grid_points = np.linspace(points.min() - 5, points.max() + 5, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)
    probs_ensemble = ensemble_probs(model, tensor_points)
    probs = probs_ensemble.mean(dim=1)

    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(model, delta_ball.view(-1, 2))
    eu_delta_ball = epistemic_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )
    au_delta_ball = aleatoric_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )

    max_eu_ball = torch.max(eu_delta_ball, dim=0)[0]
    max_au_ball = torch.max(au_delta_ball, dim=0)[0]

    combination = (
        PROB_WEIGHT * probs[:, DESIRED_CLASS]
        + lambda_1 * max_eu_ball
        + lambda_2 * max_au_ball
    ).detach()
    combination = combination - combination.min()  # Normalize to start from 0
    combination = combination / combination.max()

    # Shared value range
    all_vals = torch.concatenate([combination], dim=0)
    vmin = all_vals.min()
    vmax = all_vals.max()

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

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
        alpha=1,
        color="indigo",
        label="Point of Interest",
    )
    ax.scatter(
        counter_factual[:, 0],
        counter_factual[:, 1],
        marker="s",
        s=50,
        color="aqua",
        label="CF",
    )
    ax.scatter(
        counter_factual_steps[:, 0],
        counter_factual_steps[:, 1],
        marker=".",
        s=3,
        alpha=0.5,
        color="lime",
        label="CF Steps",
    )
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title(f"{dataset_name} - {property_name}")

    # Unified contourf
    ax.contourf(
        xx,
        yy,
        combination.numpy().reshape(xx.shape),
        levels=levels,
        cmap=cmap,
        norm=norm,
        alpha=0.1,
    )


# visualize how std of eu looks like aroung the points
def visualize_eu_std_points(model, points, y_labels, axes, delta=1, n_points=10):
    """
    Visualize a given point and its AU, EU, and TU.
    """
    # Create gird of points
    grid_points = np.linspace(points.min() - 5, points.max() + 5, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)

    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(model, delta_ball.view(-1, 2))
    eu_delta_ball = epistemic_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )
    au_delta_ball = aleatoric_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )

    au_delta_ball = torch.max(au_delta_ball, dim=0)[0]
    max_eu_ball = torch.max(eu_delta_ball, dim=0)[0]

    # Shared value range
    all_vals = torch.concatenate([au_delta_ball, max_eu_ball], dim=0)
    vmin = all_vals.min()
    vmax = all_vals.max()

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    plots = []
    for ax, data, title in zip(
        axes.flatten(), [au_delta_ball, max_eu_ball], ["Hyper_MAX_AU", "Hyper_MAX_EU"]
    ):
        # Scatter data
        ax.scatter(
            points[y_labels == 0, 0],
            points[y_labels == 0, 1],
            marker="s",
            s=10,
            alpha=0.5,
            color="blue",
            label="Class 0",
        )
        ax.scatter(
            points[y_labels == 1, 0],
            points[y_labels == 1, 1],
            marker="x",
            s=10,
            alpha=0.5,
            color="red",
            label="Class 1",
        )

        # Unified contourf
        contour = ax.contourf(
            xx,
            yy,
            data.detach().numpy().reshape(xx.shape),
            levels=levels,
            cmap=cmap,
            norm=norm,
            alpha=0.5,
        )
        plots.append(contour)

        ax.set_title(title)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")

    # Single shared colorbar using one of the contour handles
    fig.colorbar(plots[0], ax=axes, shrink=0.8, label="Uncertainty Measure Value")


def visualize_decision_boundry(
    points, model, y_labels, probability_function=ensemble_probs
):
    # Visualize decision boundary
    xx, yy = np.meshgrid(
        np.linspace(points[:, 0].min() - 1, points[:, 0].max() + 1, 100),
        np.linspace(points[:, 1].min() - 1, points[:, 1].max() + 1, 100),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_probs_ensemble = probability_function(
        model, torch.tensor(grid_points, dtype=torch.float32)
    )
    grid_probs = grid_probs_ensemble.mean(dim=1)
    grid_labels = grid_probs.argmax(dim=1).numpy()
    grid_labels = grid_labels.reshape(xx.shape)
    axes[i, -1].contourf(xx, yy, grid_labels, alpha=0.5, cmap="coolwarm")
    axes[i, -1].set_title(f"Decision Boundary - {dataset_name}")
    axes[i, -1].set_xlabel("X1")
    axes[i, -1].set_ylabel("X2")
    axes[i, -1].scatter(
        points[y_labels == 0, 0],
        points[y_labels == 0, 1],
        marker="s",
        s=2,
        alpha=0.5,
        color="blue",
        label="Class 0",
    )
    axes[i, -1].scatter(
        points[y_labels == 1, 0],
        points[y_labels == 1, 1],
        marker="o",
        s=2,
        alpha=0.5,
        color="red",
        label="Class 1",
    )
    axes[i, -1].legend()
    axes[i, -1].set_xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
    axes[i, -1].set_ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)
    print("_" * 40)


def visualize_other_cf_methods(
    point_of_interest, point_cloest_to_interest, points, y_labels, axes, dataset_name
):
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
            "x0": [
                p
                for p in [
                    *point_of_interest[:, 0].detach().numpy(),
                    *point_cloest_to_interest[:, 0].detach().numpy(),
                ]
            ],
            "x1": [
                p
                for p in [
                    *point_of_interest[:, 1].detach().numpy(),
                    *point_cloest_to_interest[:, 1].detach().numpy(),
                ]
            ],
            "label": [0] * (2),  # Assuming the point of interest is labeled as 1
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
        l2_distance_growing_sphere = instability_metric(
            point_of_interest,
            point_cloest_to_interest,
            torch.from_numpy(counterfactuals_growing_sphere.values[0]),
            torch.from_numpy(counterfactuals_growing_sphere.values[-1]),
            points,
        )
    else:
        l2_distance_growing_sphere = torch.tensor([-1], dtype=torch.float32)
        print(
            "No counterfactuals found through Growing Spheres. Skipping distance calculation."
        )

    if counterfactuals_clue is not None:
        l2_distance_clue = instability_metric(
            point_of_interest,
            point_cloest_to_interest,
            torch.from_numpy(counterfactuals_clue.values[0]),
            torch.from_numpy(counterfactuals_clue.values[-1]),
            points,
        )
    else:
        l2_distance_clue = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through CLUE. Skipping distance calculation.")

    if counterfactuals_dice is not None:
        l2_distance_dice = instability_metric(
            point_of_interest,
            point_cloest_to_interest,
            torch.from_numpy(counterfactuals_dice.values[0]),
            torch.from_numpy(counterfactuals_dice.values[-1]),
            points,
        )
    else:
        l2_distance_dice = torch.tensor([-1], dtype=torch.float32)
        print("No counterfactuals found through DICE. Skipping distance calculation.")

    if counterfactuals_face is not None:
        l2_distance_face = instability_metric(
            point_of_interest,
            point_cloest_to_interest,
            torch.from_numpy(counterfactuals_face.values[0]),
            torch.from_numpy(counterfactuals_face.values[-1]),
            points,
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

    point_closest_to_interest = points[y_labels == 0][
        torch.linalg.vector_norm(
            points[y_labels == 0] - point_of_interest, dim=1, ord=2
        ).argmin()
    ]
    point_closest_to_interest = point_closest_to_interest.view(-1, 2)

    l2_distance_stable.append([])
    l2_distance_stable_std.append([])
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
            )
            cf_poi.append(counter_factual)

        cf_poi_close = []
        for _ in range(5):
            cf_close, cf_steps_close = counter_factual_optimization_routine(
                point_to_explain=point_closest_to_interest,
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
            cf_poi_close.append(cf_close)

        l2_distance = [
            instability_metric(
                point_of_interest,
                point_closest_to_interest,
                cf_poi[i],
                cf_poi_close[i],
                points,
            )
            .detach()
            .numpy()
            for i in range(len(cf_poi))
        ]
        l2_distance_stable[-1].append(np.mean(l2_distance).item())
        l2_distance_stable_std[-1].append(np.std(l2_distance).item())

    # Visualize other CF Methods
    l2_distances_mean = []
    for _ in range(5):
        (
            l2_distance_growing_sphere,
            l2_distance_clue,
            l2_distance_dice,
            l2_distance_face,
        ) = visualize_other_cf_methods(
            point_of_interest,
            point_closest_to_interest,
            points,
            y_labels,
            axes[i, -5:-1],
            dataset_name,
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
l2_stability_data.to_csv("stability_data.csv")
