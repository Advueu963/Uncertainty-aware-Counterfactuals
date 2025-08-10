import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier

from carla.recourse_methods import GrowingSpheres, Clue, Dice, Face
from synthetic_to_carla import Synthetic_CARLA, MyOwnModel


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
    probs = (probs + 1e-8) / (1 + 1e-8)  # Avoid log(0) issues and zero gradients
    return probs


def total_uncertainty_ensemble(y_probs_ensemble):
    y_probs_mean = y_probs_ensemble.mean(dim=1)
    return (
        y_probs_mean.mul(y_probs_mean.log2())
        .sum(dim=-1)
        .mul(-1)
        .div(np.log2(y_probs_ensemble.shape[1]))
    )


def aleatoric_uncertainty_ensemble(y_probs_ensemble):
    return (
        y_probs_ensemble.mul((y_probs_ensemble + 1e-8).log2())
        .sum(dim=-1)
        .mean(dim=1)
        .mul(-1)
        .div(np.log2(y_probs_ensemble.shape[1]))
    )


def epistemic_uncertainty_ensemble(y_probs_ensemble):
    return (
        total_uncertainty_ensemble(y_probs_ensemble)
        - aleatoric_uncertainty_ensemble(y_probs_ensemble)
    ).clip(0, 1)


def counter_factual_baseline(
    dataset_name, point_of_interest, n_models, n_epochs, noisy
):
    dataset = Synthetic_CARLA(dataset_name, noisy)
    model = MyOwnModel(dataset, n_models=n_models, n_epochs=n_epochs, noisy=noisy)
    # load artificial neural networke from catalog
    # model = MLModelCatalog(dataset, "ann", backend="pytorch")
    # load a recourse model and pass black box model
    gs = GrowingSpheres(model)
    clue = Clue(dataset, model)
    dice = Dice(model)
    fa = Face(model, {"mode": "knn", "fraction": 0.2})
    if noisy:
        point_of_interest_df = pd.DataFrame(
            {
                "x0": point_of_interest[:, 0].detach().numpy(),
                "x1": point_of_interest[:, 1].detach().numpy(),
                **{
                    f"noise_{i}": point_of_interest[:, i + 2].detach().numpy()
                    for i in range(8)
                },
                "label": 1,
            }
        )
    else:
        point_of_interest_df = pd.DataFrame(
            {
                "x0": point_of_interest[:, 0].detach().numpy(),
                "x1": point_of_interest[:, 1].detach().numpy(),
                "label": 1,
            }
        )
    # generate counterfactual examples
    print("Factuals:")
    print(point_of_interest_df)
    for _ in range(5):
        try:
            counterfactuals_growing_sphere = gs.get_counterfactuals(
                point_of_interest_df
            )
            break
        except ValueError as e:
            print(f"Error generating counterfactuals with Growing Spheres: {e}")
            counterfactuals_growing_sphere = None
    for _ in range(5):
        try:
            counterfactuals_dice = dice.get_counterfactuals(point_of_interest_df)
            break
        except ValueError as e:
            print(f"Error generating counterfactuals with DICE: {e}")
            counterfactuals_dice = None
    # generate counterfactual examples using CLUE
    for _ in range(5):
        try:
            counterfactuals_clue = clue.get_counterfactuals(point_of_interest_df)
            print(counterfactuals_clue)
            break
        except ValueError as e:
            print(f"Error generating counterfactuals with CLUE: {e}")
            counterfactuals_clue = None
            # If an error occurs, you might want to handle it or retry
            # For example, you could log the error or adjust parameters
    for _ in range(5):
        try:
            counterfactuals_face = fa.get_counterfactuals(point_of_interest_df)
            print(counterfactuals_face)
            break
        except ValueError as e:
            print(f"Error generating counterfactuals with Face: {e}")
            # If an error occurs, you might want to handle it or retry
            # For example, you could log the error or adjust parameters
            counterfactuals_face = None
    return (
        counterfactuals_clue,
        counterfactuals_dice,
        counterfactuals_face,
        counterfactuals_growing_sphere,
    )


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


def invalidity(cf_point_of_interest, model, probability_function, desired_class=1):
    """
    Calculate the invalidity of a counterfactual point.
    """
    # Get the probabilities of the counterfactual point
    probs_ensemble = probability_function(model, cf_point_of_interest)
    probs = probs_ensemble.mean(dim=1)

    # Return the invalidity metric
    return (1 - probs[0, desired_class]).detach().numpy()


##################################################################
########################## Visualization #########################
##################################################################


def visualize_other_cf_methods(
    point_of_interest,
    points,
    y_labels,
    axes,
    dataset_name,
    n_models,
    n_epochs=50,
    noisy=False,
):
    (
        counterfactuals_clue,
        counterfactuals_dice,
        counterfactuals_face,
        counterfactuals_growing_sphere,
    ) = counter_factual_baseline(
        dataset_name, point_of_interest, n_models, n_epochs, noisy
    )

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
            cf.values[:, 0],
            cf.values[:, 1],
            marker="s",
            s=50,
            color="aqua",
            label=f"CF_{title}",
        )
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title(f"{dataset_name} - {title}")


def visualze_property(
    property_name,
    ax,
    model,
    points,
    y_labels,
    point_of_interest,
    counter_factual,
    counter_factual_steps,
    p_weight,
    lambda_1,
    lambda_2,
    dataset_name,
    delta=1,
    n_points=10,
    DESIRED_CLASS=1,
    noisy=False,
    multi_class=False,
):
    match property_name:
        case "validity":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                lambda_1,
                lambda_2,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "connected_ball":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                -lambda_1,
                0,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "robust":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                -lambda_1,
                -lambda_2,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "feasability":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                -lambda_1,
                0,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "discriminative":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                0,
                -lambda_2,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "discriminative2":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                0,
                -lambda_2,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "plausable":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                -lambda_1,
                0,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "similarity":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                0,
                lambda_2,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )
        case "combined":
            visualze_path_with_underlying(
                model,
                ax,
                points,
                y_labels,
                point_of_interest,
                counter_factual,
                counter_factual_steps,
                p_weight,
                0,
                0,
                dataset_name,
                property_name,
                delta=delta,
                n_points=n_points,
                DESIRED_CLASS=DESIRED_CLASS,
                noisy=noisy,
                multi_class=multi_class,
            )


def visualize_au_eu_tu(
    fig, ensemble_model, points, y_labels, axes, noisy=False, multi_class=False
):
    # Visualze AU, EU, TU
    grid_points = np.linspace(points.min() - 5, points.max() + 5, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T
    # Get the labels of the grid_points
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)
    if noisy:
        tensor_points = torch.hstack(
            [tensor_points, torch.zeros((tensor_points.shape[0], 8))]
        )
    y_probs_ensemble = ensemble_probs(ensemble_model, tensor_points)
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
        visualize_data_points(ax, points, y_labels, multi_class=multi_class)

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


def visualize_decision_boundry(
    ax,
    points,
    model,
    y_labels,
    dataset_name,
    probability_function=ensemble_probs,
    noisy=False,
    multi_class=False,
):
    # Visualize decision boundary
    xx, yy = np.meshgrid(
        np.linspace(points[:, 0].min() - 2, points[:, 0].max() + 2, 100),
        np.linspace(points[:, 1].min() - 2, points[:, 1].max() + 2, 100),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)
    if noisy:
        tensor_points = torch.hstack(
            [tensor_points, torch.zeros((tensor_points.shape[0], 8))]
        )
    grid_probs_ensemble = probability_function(model, tensor_points)
    grid_probs = grid_probs_ensemble.mean(dim=1)
    grid_labels = grid_probs.argmax(dim=1).numpy()
    grid_labels = grid_labels.reshape(xx.shape)
    ax.contourf(xx, yy, grid_labels, alpha=0.5, cmap="coolwarm")
    ax.set_title(f"Decision Boundary - {dataset_name}")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    visualize_data_points(ax, points, y_labels, multi_class=multi_class)
    ax.legend()
    ax.set_xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
    ax.set_ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)
    print("_" * 40)


def visualze_path_with_underlying(
    model,
    ax,
    points,
    y_labels,
    point_of_interest,
    counter_factual,
    counter_factual_steps,
    p_weight,
    lambda_1,
    lambda_2,
    dataset_name,
    property_name,
    delta=1,
    n_points=10,
    DESIRED_CLASS=1,
    noisy=False,
    multi_class=False,
):
    # Create gird of points
    grid_points = np.linspace(points.min() - 2, points.max() + 2, 100)
    xx, yy = np.meshgrid(grid_points, grid_points)
    grid_points = np.array([xx.flatten(), yy.flatten()]).T

    # Extract loss value
    tensor_points = torch.tensor(grid_points, dtype=torch.float32)
    if noisy:
        tensor_points = torch.hstack(
            [tensor_points, torch.zeros((tensor_points.shape[0], 8))]
        )
    probs_ensemble = ensemble_probs(model, tensor_points)
    probs = probs_ensemble.mean(dim=1)

    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(
        model, delta_ball.view(-1, *tensor_points.shape[1:])
    )
    eu_delta_ball = epistemic_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )
    au_delta_ball = aleatoric_uncertainty_ensemble(probs_ensemble_delta_ball).view(
        n_points, -1
    )

    max_eu_ball = torch.mean(eu_delta_ball, dim=0)
    max_au_ball = torch.mean(au_delta_ball, dim=0)

    combination = (
        p_weight * probs[:, DESIRED_CLASS]
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
    visualize_data_points(ax, points, y_labels, multi_class=multi_class)
    visualize_cf_path(ax, counter_factual, counter_factual_steps, point_of_interest)
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


def visualize_cf_path(ax, counter_factual, counter_factual_steps, point_of_interest):
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


def visualize_data_points(ax, points, y_labels, multi_class=False):
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
    if multi_class:
        ax.scatter(
            points[y_labels == 2, 0],
            points[y_labels == 2, 1],
            marker="s",
            s=2,
            alpha=0.3,
            color="green",
            label="Class 2",
        )
        ax.scatter(
            points[y_labels == 3, 0],
            points[y_labels == 3, 1],
            marker="s",
            s=2,
            alpha=0.3,
            color="orange",
            label="Class 3",
        )


# visualize how std of eu looks like aroung the points
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

    # sample_ball
    delta_ball = sample_delta_ball(tensor_points.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = ensemble_probs(
        model, delta_ball.view(-1, *tensor_points.shape[1:])
    )
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
