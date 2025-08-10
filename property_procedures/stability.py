from property_procedures.utils import sample_delta_ball


def stability_loss_function(
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
    """Stability loss function for counterfactual explanations.

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
    cf_probs_ensemble = probability_function(model, counter_factual)
    cf_probs = cf_probs_ensemble.mean(dim=1)
    target_prob = cf_probs[0, desired_class]
    _ = aleatoric_uncertainty_function(
        cf_probs_ensemble
    ) + epistemic_uncertainty_function(cf_probs_ensemble)

    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = probability_function(model, delta_ball)
    probs_delta_ball = probs_ensemble_delta_ball.mean(dim=1)
    target_prob_delta_ball = probs_delta_ball[:, desired_class].mean()
    au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)
    eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)
    if start_cf_construction:
        if target_prob < 0.5:
            loss = -(au_delta_ball + eu_delta_ball).mean()
        else:
            loss = -(target_prob_delta_ball).mean()
    else:
        loss = (au_delta_ball + eu_delta_ball).mean()

    loss.backward()

    grad = delta_ball.grad.mean(dim=0, keepdim=True)

    return loss, grad
