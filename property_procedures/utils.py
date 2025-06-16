import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier


def sample_delta_ball(reference_point, delta, n_points):
    """
    :param point:
    :param delta:
    :param norm:
    :return:
    """
    delta_samples = []
    for _ in range(n_points):
        sample = np.random.randn(*reference_point.shape)
        sample /= np.linalg.norm(sample, ord=2, axis=-1, keepdims=True)
        delta_sample = sample * delta
        delta_samples.append([delta_sample])
    samples = np.concatenate(delta_samples, axis=0)
    delta_samples = (reference_point + samples).astype(np.float32)
    return torch.tensor(
        delta_samples.reshape(-1, *reference_point.shape[1:])
    ).requires_grad_(True)


def sample_line(x1, x2, num_samples=10):
    """
    Returns `num_samples` points along the line from `x1` to `x2`.
    Supports gradients with respect to x1 and x2.

    Args:
        x1 (torch.Tensor): Starting point (shape: [D])
        x2 (torch.Tensor): Ending point (shape: [D])
        num_samples (int): Number of points to sample along the line

    Returns:
        torch.Tensor: Points along the line (shape: [num_samples, D])
    """
    # Ensure inputs are tensors with gradient tracking
    x1 = x1.requires_grad_()
    x2 = x2.requires_grad_()

    # Create interpolation coefficients t in [0, 1]
    t = torch.linspace(0, 1, num_samples, device=x1.device).unsqueeze(
        1
    )  # shape: [num_samples, 1]

    # Linear interpolation
    points = (1 - t) * x1 + t * x2  # shape: [num_samples, D]
    return points


def ensemble_probs(ensemble_model, points):
    probs, _ = ensemble_model.predict(points, raw_output=True)
    return probs.clamp(
        1e-8, 1 - 1e-8
    )  # Ensure probabilities are in a valid range [1e-8, 1 - 1e-8]


def total_uncertainty_ensemble(y_probs_ensemble):
    y_probs_mean = y_probs_ensemble.mean(dim=1)
    return y_probs_mean.mul(y_probs_mean.log2()).sum(dim=-1).mul(-1)


def aleatoric_uncertainty_ensemble(y_probs_ensemble):
    return (
        y_probs_ensemble.mul((y_probs_ensemble + 1e-8).log2())
        .sum(dim=-1)
        .mean(dim=1)
        .mul(-1)
    )


def epistemic_uncertainty_ensemble(y_probs_ensemble):
    return (
        total_uncertainty_ensemble(y_probs_ensemble)
        - aleatoric_uncertainty_ensemble(y_probs_ensemble)
    ).clip(0, 1)


# Look at a given point and AU,EU,TU
def visualize_au_eu_tu_points(ensemble_model, points, y_labels):
    """
    Visualize a given point and its AU, EU, and TU.
    """
    # Create gird of points
    grid_points = np.linspace(points.min(), points.max(), 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T
    # Get the labels of the grid_points
    y_probs_ensemble = ensemble_probs(ensemble_model, torch.tensor(grid_points))
    print(y_probs_ensemble.shape)
    total_uncertainty = total_uncertainty_ensemble(y_probs_ensemble)
    aleatoric_uncertainty = aleatoric_uncertainty_ensemble(y_probs_ensemble)
    epistemic_uncertainty = epistemic_uncertainty_ensemble(y_probs_ensemble)

    # Shared value range
    all_vals = torch.concatenate(
        [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()
    print(vmin, vmax)

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

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
    fig.colorbar(
        plots[0],
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Uncertainty Measure Value",
    )

    plt.show()


# visualize how std of eu looks like aroung the points
def visualize_eu_std_points(ensemble_model, points, y_labels, delta=1, n_points=10):
    """
    Visualize a given point and its AU, EU, and TU.
    """
    # Create gird of points
    grid_points = np.linspace(points.min(), points.max(), 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points)
    probs_ensemble = ensemble_probs(ensemble_model, points)
    epistemic_uncertainty = epistemic_uncertainty_ensemble(probs_ensemble)

    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(delta_ball.view(-1, 2))
    eu_delta_ball = epistemic_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )

    std_eu_ball = torch.std(eu_delta_ball, dim=0)
    max_eu_ball = torch.max(eu_delta_ball, dim=0)[0]
    min_eu_ball = torch.min(eu_delta_ball, dim=0)[0]

    # Shared value range
    all_vals = torch.concatenate(
        [std_eu_ball, epistemic_uncertainty, max_eu_ball, min_eu_ball], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()
    print(vmin, vmax)

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

    plots = []
    for ax, data, title in zip(
        axes.flatten(),
        [epistemic_uncertainty, std_eu_ball, max_eu_ball, min_eu_ball],
        ["EU", "STD_EU", "MAX_EU", "MIN_EU"],
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
    fig.colorbar(
        plots[0],
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Uncertainty Measure Value",
    )

    plt.show()


# visualize how std of eu looks like aroung the points
def visualize_au_std_points(ensemble_model, points, y_labels, delta=1, n_points=10):
    """
    Visualize a given point and its AU, EU, and TU.
    """
    # Create gird of points
    grid_points = np.linspace(points.min(), points.max(), 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points)
    probs_ensemble = ensemble_probs(ensemble_model, tensor_points)
    aleatoric_uncertainty = aleatoric_uncertainty_ensemble(probs_ensemble)

    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(delta_ball.view(-1, 2))
    au_delta_ball = aleatoric_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )

    std_au_ball = torch.std(au_delta_ball, dim=0)
    max_au_ball = torch.max(au_delta_ball, dim=0)[0]
    min_au_ball = torch.min(au_delta_ball, dim=0)[0]

    # Shared value range
    all_vals = torch.concatenate(
        [std_au_ball, aleatoric_uncertainty, max_au_ball, min_au_ball], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()
    print(vmin, vmax)

    # Define shared levels and normalization
    levels = torch.linspace(vmin, vmax, 15)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

    plots = []
    for ax, data, title in zip(
        axes.flatten(),
        [aleatoric_uncertainty, std_au_ball, max_au_ball, min_au_ball],
        ["AU", "STD_AU", "MAX_AU", "MIN_AU"],
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
    fig.colorbar(
        plots[0],
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Uncertainty Measure Value",
    )

    plt.show()


###########################################################################
############# Evaluation metrics for counterfactuals ######################
###########################################################################


def distance_func(point_of_interest, cf_point_of_interest, X):
    abs_distances = torch.abs(point_of_interest - cf_point_of_interest)
    data_median = torch.median(X, axis=0).values
    median_absolute_deviation = torch.median(torch.abs(X - data_median), axis=0).values
    distance = (
        abs_distances / (median_absolute_deviation + 1e-8)
    ).sum()  # Avoid division by zero
    return distance


def instability_metric(
    point_of_interest, point_to_compare, cf_point_of_interest, cf_point_to_compare, X
):
    """
    Calculate the dissimilarity metric between two points and their counterfactuals.
    """
    # Calculate the Euclidean distance between the points and their counterfactuals
    distance_original = distance_func(point_of_interest, point_to_compare, X)
    distance_cf = distance_func(cf_point_of_interest, cf_point_to_compare, X)

    # Return the dissimilarity metric
    return 1 / (1 + distance_original) + distance_cf


def discriminative_power(
    point_of_interest,
    cf_point_of_interest,
    class_poi,
    class_cf,
    X_equal_poi,
    X_diff_poi,
):
    """
    Calculate the discriminative power of a point of interest and its counterfactual.
    """
    k_nn = KNeighborsClassifier(n_neighbors=1)
    X_train = np.vstack([point_of_interest, cf_point_of_interest])
    y_train = np.array([class_poi, class_cf])
    k_nn.fit(X_train, y_train)

    # Accuracy on the X_diff_poi and X_equal_poi
    X_diff_poi = np.array(X_diff_poi)
    X_equal_poi = np.array(X_equal_poi)
    y_diff_poi = k_nn.predict(X_diff_poi)
    y_equal_poi = k_nn.predict(X_equal_poi)
    accuracy = np.mean(np.hstack([y_diff_poi == class_cf, y_equal_poi == class_poi]))
    # Return the discriminative power
    return accuracy


def dissimilarity(point_of_interest, cf_point_of_interest, X):
    # Calculate the Euclidean distance between the points and their counterfactuals
    distance_original = distance_func(point_of_interest, cf_point_of_interest, X)

    # Return the dissimilarity metric
    return distance_original


def dissparsity(point_of_interest, cf_point_of_interest):
    return np.mean(point_of_interest != cf_point_of_interest)


def implausability(cf_point_of_interest, X):
    """
    Calculate the implausibility of a counterfactual point.
    """
    # Calculate the distance from the counterfactual point to the training data
    distances = torch.tensor(
        [distance_func(X[i], cf_point_of_interest, X) for i in range(X.shape[0])]
    )
    # Return the implausibility metric
    return torch.min(distances)
