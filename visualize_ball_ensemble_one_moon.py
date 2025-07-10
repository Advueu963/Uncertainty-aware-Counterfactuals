import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from data import load_one_moon
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from property_procedures.utils import (
    sample_delta_ball,
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
    visualize_data_points,
)

DESIRED_VALIDITY = 0.999
DELTA = 0.5
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


def visualize_eu_std_points(
    fig,
    model,
    points,
    y_labels,
    axes,
    delta=1,
    n_points=10,
    noisy=False,
    multi_class=False,
):
    """
    Visualize a given point and its AU, EU, and TU.
    """
    # Create gird of points
    grid_points = np.linspace(points.min() - 5, points.max() + 5, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)

    if noisy:
        tensor_points = torch.hstack(
            [tensor_points, torch.zeros((tensor_points.shape[0], 8))]
        )
    epistemic_uncertainty = epistemic_uncertainty_ensemble(
        ensemble_probs(model, tensor_points)
    )
    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(
        model, delta_ball.view(-1, *tensor_points.shape[1:])
    )
    eu_delta_ball = epistemic_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )
    _ = aleatoric_uncertainty_ensemble(probs_ensemble_delta_ball).view(n_points, -1)

    std_eu_ball = torch.std(eu_delta_ball, dim=0)
    max_eu_ball = torch.max(eu_delta_ball, dim=0)[0]
    _ = torch.min(eu_delta_ball, dim=0)[0]
    mean_eu_ball = torch.mean(eu_delta_ball, dim=0)

    # Shared value range
    all_vals = torch.concatenate(
        [std_eu_ball, epistemic_uncertainty, max_eu_ball, mean_eu_ball], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()
    print(vmin, vmax)

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    plots = []
    for ax, data, title in zip(
        axes.flatten(),
        [std_eu_ball, epistemic_uncertainty, max_eu_ball, mean_eu_ball],
        ["Std EU", "EU", "Max EU", "Mean EU"],
    ):
        # Scatter data
        visualize_data_points(ax, points, y_labels, multi_class=multi_class)

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


def visualize_au_std_points(
    fig,
    model,
    points,
    y_labels,
    axes,
    delta=1,
    n_points=10,
    noisy=False,
    multi_class=False,
):
    """
    Visualize a given point and its AU, EU, and TU.
    """
    # Create gird of points
    grid_points = np.linspace(points.min() - 5, points.max() + 5, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)

    if noisy:
        tensor_points = torch.hstack(
            [tensor_points, torch.zeros((tensor_points.shape[0], 8))]
        )
    aleatoric_uncertainty = aleatoric_uncertainty_ensemble(
        ensemble_probs(model, tensor_points)
    )
    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(
        model, delta_ball.view(-1, *tensor_points.shape[1:])
    )
    au_delta_ball = aleatoric_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )

    std_au_ball = torch.std(au_delta_ball, dim=0)
    max_au_ball = torch.max(au_delta_ball, dim=0)[0]
    mean_au_ball = torch.mean(au_delta_ball, dim=0)

    # Shared value range
    all_vals = torch.concatenate(
        [std_au_ball, aleatoric_uncertainty, max_au_ball, mean_au_ball], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()
    print(vmin, vmax)

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    plots = []
    for ax, data, title in zip(
        axes.flatten(),
        [std_au_ball, aleatoric_uncertainty, max_au_ball, mean_au_ball],
        ["Std AU", "AU", "Max AU", "Mean AU"],
    ):
        # Scatter data
        visualize_data_points(ax, points, y_labels, multi_class=multi_class)

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


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
points, y_labels, y_probs = load_one_moon()

# Train the ensemble model
ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
ensemble_model.load(f"models/Ensemble_One_moon_{N_EPOCHS}/")
ensemble_model.compile()

visualize_eu_std_points(
    fig,
    ensemble_model,
    points,
    y_labels,
    axes,
    delta=DELTA,
    n_points=N_POINTS,
    noisy=False,
    multi_class=False,
)

plt.savefig("ensemble_one_moon_eu_agg.pdf", dpi=100)
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

visualize_au_std_points(
    fig,
    ensemble_model,
    points,
    y_labels,
    axes,
    delta=DELTA,
    n_points=N_POINTS,
    noisy=False,
    multi_class=False,
)
plt.savefig("ensemble_one_moon_au_agg.pdf", dpi=100)
plt.show()
