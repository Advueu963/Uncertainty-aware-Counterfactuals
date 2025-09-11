import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier

from carla.recourse_methods import GrowingSpheres, Clue, Dice, Face
from synthetic_to_carla import Synthetic_CARLA, MyOwnModel


def sample_delta_ball(reference_point, delta, n_points):
    """Samples points from a delta-ball around a reference point.

    Args:
        reference_point (np.ndarray): The center point of the delta-ball.
        delta (float): The radius of the delta-ball.
        n_points (int): The number of points to sample.

    Returns:
        torch.Tensor: A tensor containing the sampled points.
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
        delta_samples.reshape(reference_point.shape[0], -1, *reference_point.shape[1:])
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
    """Compute Ensemble probabilities for given points using `epiuc` package.

    Args:
        ensemble_model (epiuc.classification.EnsembleClassifier): The ensemble model to use for predictions.
        points (torch.Tensor): The input points for which to compute probabilities.

    Returns:
        torch.Tensor: The computed probabilities for the input points.
    """
    probs, _ = ensemble_model.predict(points, raw_output=True)
    probs = (probs + 1e-8) / (1 + 1e-8)  # Avoid log(0) issues and zero gradients
    return probs


def total_uncertainty_ensemble(y_probs_ensemble):
    """Compute the total uncertainty for an ensemble of models.

    Args:
        y_probs_ensemble (torch.Tensor): The predicted probabilities from the ensemble.

    Returns:
        torch.Tensor: The total uncertainty for the ensemble.
    """
    y_probs_mean = y_probs_ensemble.mean(dim=1)
    return (
        y_probs_mean.mul(y_probs_mean.log2())
        .sum(dim=-1)
        .mul(-1)
        .div(np.log2(y_probs_ensemble.shape[1]))
    )


def aleatoric_uncertainty_ensemble(y_probs_ensemble):
    """Compute the aleatoric uncertainty for an ensemble of models.

    Args:
        y_probs_ensemble (torch.Tensor): The predicted probabilities from the ensemble.

    Returns:
        torch.Tensor: The aleatoric uncertainty for the ensemble.
    """
    return (
        y_probs_ensemble.mul((y_probs_ensemble + 1e-8).log2())
        .sum(dim=-1)
        .mean(dim=1)
        .mul(-1)
        .div(np.log2(y_probs_ensemble.shape[1]))
    )


def epistemic_uncertainty_ensemble(y_probs_ensemble):
    """Compute the epistemic uncertainty for an ensemble of models.

    Args:
        y_probs_ensemble (torch.Tensor): The predicted probabilities from the ensemble.

    Returns:
        torch.Tensor: The epistemic uncertainty for the ensemble.
    """
    return (
        total_uncertainty_ensemble(y_probs_ensemble)
        - aleatoric_uncertainty_ensemble(y_probs_ensemble)
    ).clip(0, 1)


def counter_factual_baseline(
    dataset_name, point_of_interest, n_models, n_epochs, noisy
):
    """Counterfactual baseline generation.

    Args:
        dataset_name (str): The name of the dataset.
        point_of_interest (torch.Tensor): The point of interest for counterfactual generation.
        n_models (int): The number of models in the ensemble.
        n_epochs (int): The number of training epochs.
        noisy (bool): Whether the dataset is noisy.

    Returns:
        tuple: A tuple containing the generated counterfactuals.
    """

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
    """Compute the distance between a point of interest and its counterfactual.


    Args:
        point_of_interest (torch.Tensor): The original point of interest.
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.
        X (torch.Tensor): The dataset.

    Returns:
        torch.Tensor: The computed distance.
    """
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
    Args:
        point_of_interest (torch.Tensor): The original point of interest.
        point_to_compare (torch.Tensor): The point to compare against.
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.
        cf_point_to_compare (torch.Tensor): The counterfactual point to compare against.
        X (torch.Tensor): The dataset.
    Returns:
        float: The dissimilarity metric.
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
    Args:
        point_of_interest (torch.Tensor): The original point of interest.
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.
        class_poi (int): The class of the point of interest.
        class_cf (int): The class of the counterfactual point of interest.
        X_equal_poi (np.ndarray): List of points that are equal to the point of interest.
        X_diff_poi (np.ndarray): List of points that are different from the point of interest.
    Returns:
        float: The discriminative power metric
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
    """Calculate the dissimilarity between a point of interest and its counterfactual.

    Args:
        point_of_interest (torch.Tensor): The original point of interest.
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.
        X (torch.Tensor): The dataset.

    Returns:
        float: The dissimilarity metric.
    """

    # Calculate the Euclidean distance between the points and their counterfactuals
    distance_original = distance_func(point_of_interest, cf_point_of_interest, X)

    # Return the dissimilarity metric
    return distance_original


def dissparsity(point_of_interest, cf_point_of_interest):
    """Calculate the dissparsity between a point of interest and its counterfactual.

    Args:
        point_of_interest (torch.Tensor): The original point of interest.
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.

    Returns:
        float: The dissparsity metric.
    """

    return np.mean(point_of_interest != cf_point_of_interest)


def implausability(cf_point_of_interest, X):
    """
    Calculate the implausibility of a counterfactual point.

    Args:
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.
        X (torch.Tensor): The training data.
    Returns:
        float: The implausibility metric.
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
    Args:
        cf_point_of_interest (torch.Tensor): The counterfactual point of interest.
        model (torch.nn.Module): The model used for prediction.
        probability_function (callable): The function to compute probabilities.
        desired_class (int): The class for which to compute the invalidity.
    Returns:
        float: The invalidity metric.
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
    # Enhanced uncertainty visualization with improved aesthetics
    grid_resolution = 150  # Higher resolution for smoother contours
    grid_points = np.linspace(points.min() - 5, points.max() + 5, grid_resolution)
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

    # Shared value range for consistent comparison
    all_vals = torch.concatenate(
        [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty], dim=0
    )
    vmin = all_vals.min()
    vmax = all_vals.max()

    # Enhanced color scheme - using a professional colormap
    cmap = mpl.cm.plasma  # Professional colormap
    levels = 20  # More levels for smoother gradients
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plots = []
    uncertainty_data = [total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty]
    titles = [
        "Total Uncertainty (TU)",
        "Aleatoric Uncertainty (AU)",
        "Epistemic Uncertainty (EU)",
    ]

    for ax, values, title in zip(axes, uncertainty_data, titles):
        # Enhanced data point visualization
        visualize_data_points(ax, points, y_labels, multi_class=multi_class)

        # Enhanced contour plot with more levels
        contour = ax.contourf(
            xx,
            yy,
            values.detach().numpy().reshape(xx.shape),
            levels=levels,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )

        # Add contour lines for better definition
        ax.contour(
            xx,
            yy,
            values.detach().numpy().reshape(xx.shape),
            levels=10,
            colors="white",
            alpha=0.3,
            linewidths=0.5,
        )

        plots.append(contour)

        # Enhanced styling
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Feature 1", fontsize=11, fontweight="medium")
        ax.set_ylabel("Feature 2", fontsize=11, fontweight="medium")

        # Enhanced grid and background
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
        ax.set_facecolor("#fafafa")

        # Add subtle border styling
        for spine in ax.spines.values():
            spine.set_color("gray")
            spine.set_linewidth(0.8)

    # Enhanced colorbar with better positioning and styling
    cbar = fig.colorbar(plots[0], ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label("Uncertainty Value", fontsize=11, fontweight="medium")
    cbar.ax.tick_params(labelsize=9)

    # Set overall figure title
    fig.suptitle(
        "Uncertainty Analysis: Total, Aleatoric, and Epistemic Components",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )


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
    # Enhanced decision boundary visualization
    grid_resolution = 200  # Higher resolution for smoother boundaries
    xx, yy = np.meshgrid(
        np.linspace(points[:, 0].min() - 2, points[:, 0].max() + 2, grid_resolution),
        np.linspace(points[:, 1].min() - 2, points[:, 1].max() + 2, grid_resolution),
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

    # Enhanced color scheme for decision boundaries
    if multi_class:
        # Professional color scheme for multiclass
        colors = ["#E3F2FD", "#FFF3E0", "#E8F5E8", "#FCE4EC"]  # Light versions
        cmap = mpl.colors.ListedColormap(colors[: len(np.unique(grid_labels))])
    else:
        # Professional blue-red scheme for binary classification
        cmap = mpl.colors.ListedColormap(["#E3F2FD", "#FFEBEE"])

    # Plot decision boundary with enhanced styling
    _ = ax.contourf(
        xx, yy, grid_labels, alpha=0.3, cmap=cmap, levels=len(np.unique(grid_labels))
    )

    # Add decision boundary lines
    ax.contour(
        xx, yy, grid_labels, colors="gray", alpha=0.6, linewidths=1.5, linestyles="--"
    )

    # Enhanced title and labels
    if dataset_name:
        ax.set_title(
            f"Decision Boundary: {dataset_name}", fontsize=12, fontweight="bold", pad=10
        )
    else:
        ax.set_title("Decision Boundary", fontsize=12, fontweight="bold", pad=10)

    ax.set_xlabel("Feature 1", fontsize=11, fontweight="medium")
    ax.set_ylabel("Feature 2", fontsize=11, fontweight="medium")

    # Visualize data points with enhanced styling
    visualize_data_points(ax, points, y_labels, multi_class=multi_class)

    # Enhanced legend styling
    legend = ax.legend(
        loc="best", frameon=True, fancybox=True, shadow=True, fontsize=9, framealpha=0.9
    )
    if legend:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("gray")
        legend.get_frame().set_linewidth(0.5)

    # Enhanced grid and styling
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_facecolor("#fafafa")

    # Set axis limits with proper padding
    ax.set_xlim(points[:, 0].min() - 1, points[:, 0].max() + 1)
    ax.set_ylim(points[:, 1].min() - 1, points[:, 1].max() + 1)

    # Add subtle border styling
    for spine in ax.spines.values():
        spine.set_color("gray")
        spine.set_linewidth(0.8)


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
    # Enhanced color scheme for counterfactual visualization
    colors = {
        "poi": "#2E3440",  # Dark gray for point of interest
        "cf": "#5E81AC",  # Professional blue for counterfactual
        "path": "#A3BE8C",  # Soft green for path
    }

    # # Plot connecting line between original point and counterfactual
    # if len(counter_factual_steps) > 1:
    #     ax.plot(
    #         [point_of_interest[0, 0].detach(), counter_factual[0, 0]],
    #         [point_of_interest[0, 1].detach(), counter_factual[0, 1]],
    #         color=colors["line"],
    #         alpha=0.6,
    #         linewidth=2,
    #         linestyle="--",
    #         zorder=2,
    #         label="Optimization Path",
    #     )

    # Plot optimization steps with enhanced styling
    if len(counter_factual_steps) > 0:
        ax.scatter(
            counter_factual_steps[:, 0],
            counter_factual_steps[:, 1],
            marker=".",
            s=15,
            alpha=0.6,
            c=colors["path"],
            edgecolors="white",
            linewidth=0.3,
            label="Optimization Steps",
            zorder=3,
        )

    # Plot point of interest with enhanced styling
    ax.scatter(
        point_of_interest[:, 0].detach(),
        point_of_interest[:, 1].detach(),
        marker="*",
        s=120,
        alpha=0.9,
        c=colors["poi"],
        edgecolors="white",
        linewidth=1.5,
        label="Point of Interest",
        zorder=5,
    )

    # Plot counterfactual with enhanced styling
    ax.scatter(
        counter_factual[:, 0],
        counter_factual[:, 1],
        marker="D",
        s=80,
        alpha=0.9,
        c=colors["cf"],
        edgecolors="white",
        linewidth=1.2,
        label="Counterfactual",
        zorder=5,
    )


def visualize_data_points(ax, points, y_labels, multi_class=False):
    # Enhanced color scheme and styling for data points
    colors = {
        "class0": "#1f77b4",  # Professional blue
        "class1": "#ff7f0e",  # Professional orange
        "class2": "#2ca02c",  # Professional green
        "class3": "#d62728",  # Professional red
    }

    # Enhanced styling for binary classification
    ax.scatter(
        points[y_labels == 0, 0],
        points[y_labels == 0, 1],
        marker="o",
        s=25,
        alpha=0.7,
        c=colors["class0"],
        edgecolors="white",
        linewidth=0.5,
        label="Class 0",
        zorder=3,
    )
    ax.scatter(
        points[y_labels == 1, 0],
        points[y_labels == 1, 1],
        marker="s",
        s=25,
        alpha=0.7,
        c=colors["class1"],
        edgecolors="white",
        linewidth=0.5,
        label="Class 1",
        zorder=3,
    )

    # Enhanced styling for multiclass
    if multi_class:
        ax.scatter(
            points[y_labels == 2, 0],
            points[y_labels == 2, 1],
            marker="^",
            s=25,
            alpha=0.7,
            c=colors["class2"],
            edgecolors="white",
            linewidth=0.5,
            label="Class 2",
            zorder=3,
        )
        ax.scatter(
            points[y_labels == 3, 0],
            points[y_labels == 3, 1],
            marker="D",
            s=25,
            alpha=0.7,
            c=colors["class3"],
            edgecolors="white",
            linewidth=0.5,
            label="Class 3",
            zorder=3,
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
