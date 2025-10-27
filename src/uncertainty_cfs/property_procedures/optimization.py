import torch
import numpy as np
from torch.optim import Adam, SGD


def counter_factual_optimization_routine(
    point_to_explain,
    model,
    loss_function,
    probability_function,
    aleatoric_uncertainty_function,
    epistemic_uncertainty_function,
    desired_class,
    MAX_STEPS=1000,
    DESIRED_VALIDITY=0.8,
    lr=0.01,
    p_weight=1,
    lambda_1=0.1,
    lambda_2=0.7,
    patience=10,
    delta=0.1,
    n_points=10,
    optimization_method="adam",
    categorical_features_lists=[],
    immutable_features_lists=[],
    **kwargs,
):
    """Counterfactual optimization routine.

    Args:
        point_to_explain (torch.Tensor): The point to explain.
        model (torch.nn.Module): The model to optimize.
        loss_function (callable): The loss function to use.
        probability_function (callable): The function to compute probabilities.
        aleatoric_uncertainty_function (callable): The function to compute aleatoric uncertainty.
        epistemic_uncertainty_function (callable): The function to compute epistemic uncertainty.
        desired_class (int): The class to achieve.
        MAX_STEPS (int, optional): The maximum number of optimization steps. Defaults to 1000.
        DESIRED_VALIDITY (float, optional): The desired validity of the counterfactual. Defaults to 0.8.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        p_weight (int, optional): The weight for the probability loss. Defaults to 1.
        lambda_1 (float, optional): The weight for the first regularization term. Defaults to 0.1.
        lambda_2 (float, optional): The weight for the second regularization term. Defaults to 0.7.
        patience (int, optional): The number of steps to wait for improvement before stopping. Defaults to 10.
        delta (float, optional): The perturbation size for generating counterfactuals. Defaults to 0.1.
        n_points (int, optional): The number of points to sample for uncertainty estimation. Defaults to 10.
        optimization_method (str, optional): The optimization method to use. Defaults to "adam".
        categorical_features_lists (list, optional): List of lists containing indices of categorical features. A list consists of all columns that belong to one categorical feature. Defaults to [].

    Raises:
        ValueError: If the optimization method is not supported.

    Returns:
        torch.Tensor: The optimized counterfactual.
    """
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    # Initialize the optimizer
    if optimization_method == "adam":
        optimizer = Adam(
            [counter_factual], lr=lr, betas=(0.99, 0.999)
        )  # Changing the first beta makes it more responsive to changes
    elif optimization_method == "sgd":
        optimizer = SGD([counter_factual], lr=lr)
    else:
        raise ValueError("Unsupported optimization method. Use 'adam' or 'sgd'.")

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    target_probs = probs[:, desired_class].min()
    # Optimization Iteration

    T = 0
    patience_counter = 0
    prior_loss = 0
    start_cf_construction = False
    while (
        T < MAX_STEPS
        and target_probs < DESIRED_VALIDITY
        and patience_counter < patience
    ):
        optimizer.zero_grad()

        loss, grad = loss_function(
            point_to_explain=point_to_explain,
            counter_factual=counter_factual,
            model=model,
            probability_function=probability_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_function,
            epistemic_uncertainty_function=epistemic_uncertainty_function,
            desired_class=desired_class,
            p_weight=p_weight,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            delta=delta,
            n_points=n_points,
            start_cf_construction=start_cf_construction,
            **kwargs,
        )

        # Remove the gradients for points that have reached the desired validity already
        proportion_reaching_desired_validity = (
            torch.sum(probs[:, desired_class] >= DESIRED_VALIDITY).item()
            / probs.shape[0]
        )
        # if proportion_reaching_desired_validity > 0:
        #     print("Proportion of points reaching desired validity: ", proportion_reaching_desired_validity)
        #     print("PROBS REACHING DESIRED VALIDITY: ", probs[probs[:,desired_class] >= DESIRED_VALIDITY, desired_class])
        #     print("GRADS    REACHING DESIRED VALIDITY: ", grad[probs[:,desired_class] >= DESIRED_VALIDITY, :])
        # print("PROBS REACHING DESIRED VALIDITY: ", proportion_reaching_desired_validity)
        # print("GRAD BEFORE IMMUTABILITY: ", grad)
        grad[:, immutable_features_lists] = (
            0  # Enforce immutability constraints in the gradients
        )

        counter_factual.grad = grad
        # print("GRAD AFTER IMMUTABILITY: ", counter_factual.grad)
        # print("GRAD: ", counter_factual.grad)
        optimizer.step()

        # raise NotImplementedError("This should not be reached anymore.")
        # Enforce categorical constraints by setting the maximum value in each categorical feature to 1 and the rest to 0
        for cat_list in categorical_features_lists:
            cat_list = np.array(cat_list)
            with torch.no_grad():
                # Get the indices of the maximum value in the categorical feature
                max_index = torch.argmax(counter_factual[:, cat_list], dim=1)

                new_vals = torch.nn.functional.one_hot(
                    max_index, num_classes=len(cat_list)
                ).float()
                counter_factual[:, cat_list] = new_vals

        probs_ensemble = probability_function(model, counter_factual.detach().clone())
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[:, desired_class].min()

        # Check for early stopping
        if abs(prior_loss - loss.mean().item()) < 0.01:  # pyright: ignore[reportUnknownMemberType]
            patience_counter += 1
        else:
            patience_counter = 0

        prior_loss = loss.mean().item()
        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
        print(
            "TARGET PROBS",
            target_probs.item(),
            "BEST Prob",
            probs[:, desired_class].max(),
            "LOSS",
            loss.mean().item(),
            "FINISHING RATIO",
            proportion_reaching_desired_validity,
        )
    # Print out reason for stopping
    if T >= MAX_STEPS:
        print("Reached maximum number of steps.")
    elif target_probs >= DESIRED_VALIDITY:
        print("Desired validity reached.")
    elif patience_counter >= patience:
        print("Patience limit reached.")
    return counter_factual.detach(), counter_factual_steps


def counter_factual_optimization_routine_schut(
    point_to_explain,
    model,
    loss_function,
    probability_function,
    aleatoric_uncertainty_function,
    epistemic_uncertainty_function,
    desired_class,
    MAX_STEPS=1000,
    DESIRED_VALIDITY=0.8,
    lr=0.01,
    p_weight=1,
    lambda_1=0.1,
    lambda_2=0.7,
    patience=10,
    delta=0.1,
    n_points=10,
    categorical_features_lists=[],
    immutable_features_lists=[],
    **kwargs,
):
    """Counterfactual optimization routine.

    Args:
        point_to_explain (torch.Tensor): The point to explain.
        model (torch.nn.Module): The model to optimize.
        loss_function (callable): The loss function to use.
        probability_function (callable): The function to compute probabilities.
        aleatoric_uncertainty_function (callable): The function to compute aleatoric uncertainty.
        epistemic_uncertainty_function (callable): The function to compute epistemic uncertainty.
        desired_class (int): The class to achieve.
        MAX_STEPS (int, optional): The maximum number of optimization steps. Defaults to 1000.
        DESIRED_VALIDITY (float, optional): The desired validity of the counterfactual. Defaults to 0.8.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        p_weight (int, optional): The weight for the probability loss. Defaults to 1.
        lambda_1 (float, optional): The weight for the first regularization term. Defaults to 0.1.
        lambda_2 (float, optional): The weight for the second regularization term. Defaults to 0.7.
        patience (int, optional): The number of steps to wait for improvement before stopping. Defaults to 10.
        delta (float, optional): The perturbation size for generating counterfactuals. Defaults to 0.1.
        n_points (int, optional): The number of points to sample for uncertainty estimation. Defaults to 10.
        optimization_method (str, optional): The optimization method to use. Defaults to "adam".
        categorical_features_lists (list, optional): List of lists containing indices of categorical features. A list consists of all columns that belong to one categorical feature. Defaults to [].

    Raises:
        ValueError: If the optimization method is not supported.

    Returns:
        torch.Tensor: The optimized counterfactual.
    """
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    # Initialize the optimizer
    optimizer = SGD([counter_factual], lr=lr)

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    target_probs = probs[:, desired_class].min()
    # Optimization Iteration

    T = 0
    patience_counter = 0
    prior_loss = 0
    start_cf_construction = False
    updated_features = torch.zeros_like(point_to_explain).view(
        point_to_explain.shape[0], -1
    )  # To track which features have been updated
    while (
        T < MAX_STEPS
        and target_probs < DESIRED_VALIDITY
        and patience_counter < patience
    ):
        optimizer.zero_grad()

        loss, grad = loss_function(
            point_to_explain=point_to_explain,
            counter_factual=counter_factual,
            model=model,
            probability_function=probability_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_function,
            epistemic_uncertainty_function=epistemic_uncertainty_function,
            desired_class=desired_class,
            p_weight=p_weight,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            delta=delta,
            n_points=n_points,
            start_cf_construction=start_cf_construction,
            **kwargs,
        )

        # Get indices of highest gradient
        flatten_grads = grad.view(point_to_explain.shape[0], -1)
        flatten_grads[updated_features > 5] = (
            0  # Ignore features that have been updated more than 5 times
        )
        idx_max_grad = torch.argmax(
            torch.abs(flatten_grads), dim=1
        )  # Assuming grad is of shape (1, C, H, W) for images
        # Zero out all gradients except the one with the highest absolute value
        updated_grad = torch.zeros_like(grad).view(point_to_explain.shape[0], -1)
        updated_grad[:, idx_max_grad] = torch.sign(flatten_grads[:, idx_max_grad])
        grad = updated_grad.view_as(grad)

        counter_factual.grad = grad
        optimizer.step()

        updated_features[:, idx_max_grad] += 1  # Mark this feature as updated

        # TODO: Make the valid range adjustable for different datasets
        with (
            torch.no_grad()
        ):  # This deactivates gradient tracking for the operations within this block
            counter_factual.clamp_(
                -0.5, 3
            )  # Assuming the valid range is [-0.5, 3] which is for MNIST dataset

        # Extract Target probabilities
        probs_ensemble = probability_function(model, counter_factual.detach().clone())
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[:, desired_class].min()

        # Check for early stopping
        if abs(prior_loss - loss.mean().item()) < 0.01:  # pyright: ignore[reportUnknownMemberType]
            patience_counter += 1
        else:
            patience_counter = 0

        prior_loss = loss.mean().item()
        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
        # print("TARGET PROBS", target_probs.item(), "LOSS", loss.mean().item())
    # Print out reason for stopping
    if T >= MAX_STEPS:
        print("Reached maximum number of steps.")
    elif target_probs >= DESIRED_VALIDITY:
        print("Desired validity reached.")
    elif patience_counter >= patience:
        print("Patience limit reached.")
    return counter_factual.detach(), counter_factual_steps
