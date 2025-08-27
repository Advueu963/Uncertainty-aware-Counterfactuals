def sparse_loss_function(
    point_to_explain,
    counter_factual,
    model,
    probability_function,
    aleatoric_uncertainty_function,
    epistemic_uncertainty_function,
    desired_class,
    p_weight=1,
    lambda_1=0.1,
    lambda_2=0.7,
    delta=0.1,
    n_points=10,
    start_cf_construction=False,
):
    """Sparsity loss function for counterfactual explanations.

    Args:
        point_to_explain (torch.Tensor): The factual point to explain
        counter_factual (torch.Tensor): The counterfactual point to optimize
        model (torch.nn.Module): The model to use for predictions
        probability_function (Callable): Function to compute probabilities
        aleatoric_uncertainty_function (Callable): Function to compute aleatoric uncertainty
        epistemic_uncertainty_function (Callable): Function to compute epistemic uncertainty
        desired_class (int): The class to which the counterfactual should belong
        p_weight (int, optional): Weight for the probability term. Defaults to 1.
        lambda_1 (float, optional): Weight for the epistemic uncertainty term. Defaults to 0.1.
        lambda_2 (float, optional): Weight for the aleatoric uncertainty term. Defaults to 0.7.
        delta (float, optional): Radius for the delta-ball sampling. Defaults to 0.1.
        n_points (int, optional): Number of points to sample in the delta-ball. Defaults to 10.
        start_cf_construction (bool, optional): Whether to start counterfactual construction. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The loss value and the gradient of the counterfactual.
    """
    probs_ensemble_cf = probability_function(model, counter_factual)

    au_cf = aleatoric_uncertainty_function(probs_ensemble_cf)

    #
    # # Extract loss value
    loss = -(p_weight * au_cf)
    loss.backward()

    # pick the grad of the point with the largest difference
    grad = counter_factual.grad

    return loss, grad
