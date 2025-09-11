import torch
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

    Raises:
        ValueError: If the optimization method is not supported.

    Returns:
        torch.Tensor: The optimized counterfactual.
    """
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    # Initialize the optimizer
    if optimization_method == "adam":
        optimizer = Adam([counter_factual], lr=lr)
    elif optimization_method == "sgd":
        optimizer = SGD([counter_factual], lr=lr)
    else:
        raise ValueError("Unsupported optimization method. Use 'adam' or 'sgd'.")

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    target_probs = probs[0, desired_class]
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

        counter_factual.grad = grad

        optimizer.step()
        # print(counter_factual,au, t)

        # Extract Target probabilities
        probs_ensemble = probability_function(model, counter_factual.detach().clone())
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[0, desired_class]

        # Check for early stopping
        if abs(prior_loss - loss.mean().item()) < 0.01:
            patience_counter += 1
        else:
            patience_counter = 0

        prior_loss = loss.mean().item()
        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
    # Print out reason for stopping
    if T >= MAX_STEPS:
        print("Reached maximum number of steps.")
    elif target_probs >= DESIRED_VALIDITY:
        print("Desired validity reached.")
    elif patience_counter >= patience:
        print("Patience limit reached.")
    return counter_factual.detach(), counter_factual_steps
