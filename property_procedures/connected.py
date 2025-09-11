from property_procedures.utils import sample_delta_ball, sample_line
import torch


def connected_loss_function(
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
    **kwargs,
):
    """Undirected-Connectedness loss function for counterfactual explanations.

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
    # Extract loss value
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)
    # print( "CF SHape", counter_factual.shape)

    # # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    # print("Delta ball shape", delta_ball.shape)
    # delta_ball = get_knn_points(counter_factual, k=n_points)
    probs_ensemble_delta_ball = probability_function(
        model, delta_ball.reshape(-1, *counter_factual.shape[1:])
    )
    probs_delta_ball = probs_ensemble_delta_ball.mean(dim=1)
    eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball).reshape(
        counter_factual.shape[0], -1
    )

    # print("Probs delta ball shape", probs_delta_ball.shape)
    # print("EU delta ball shape", eu_delta_ball.shape)
    # print("Probs shape", probs.shape)

    target_probs = probs[:, desired_class]
    _ = probs_delta_ball[:, desired_class]

    loss = p_weight * target_probs.log2()
    loss = loss - torch.where(
        target_probs > 0, lambda_2 * (eu_delta_ball.mean(dim=1).log2()), 0
    )
    loss = loss.mul(-1)

    # print("Loss", loss.shape)

    loss.sum().backward()
    # print("Delta ball grad shape", delta_ball.grad.shape)
    # print("CF grad shape", counter_factual.grad.shape)
    # print("target shape ", (counter_factual.grad + delta_ball.grad.sum(dim=1)).shape)
    grad = torch.where(
        (target_probs < 0.51).unsqueeze(1),
        counter_factual.grad + delta_ball.grad.sum(dim=1),
        counter_factual.grad,
    )

    return loss, grad


def connected_loss_function_reference_point(
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
    **kwargs,
):
    """Directed Connectedness loss function for counterfactual explanations.

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
    reference_point = kwargs["reference_point"]
    connecting_line = sample_line(
        reference_point, counter_factual, num_samples=n_points
    )

    # Extract loss value
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    # # sample_ball
    probs_ensemble_delta_ball = probability_function(model, connecting_line)
    au_line = aleatoric_uncertainty_function(probs_ensemble_delta_ball)
    eu_line = epistemic_uncertainty_function(probs_ensemble_delta_ball)

    target_probs = probs[0, desired_class]

    loss = p_weight * target_probs.log2()
    if target_probs > 0.51:
        loss = (
            loss - lambda_1 * (eu_line.mean().log2()) - lambda_2 * au_line.mean().log2()
        )

    loss = loss.mul(-1)

    loss.backward()
    grad = counter_factual.grad
    return loss, grad
