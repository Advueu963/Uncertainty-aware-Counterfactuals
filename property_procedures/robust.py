from property_procedures.utils import sample_delta_ball


def robust_loss_function(
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
    # Extract loss value
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = probability_function(model, delta_ball)
    _ = probs_ensemble_delta_ball.mean(dim=1)[:, desired_class]
    eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)
    au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)

    target_probs = probs[0, desired_class]

    # Calculate the loss
    loss = p_weight * target_probs.log2()

    if target_probs > 0.51:
        loss = loss - lambda_1 * (eu_delta_ball + au_delta_ball).mean().log2()
    loss = loss.mul(-1)
    loss.backward()

    if delta_ball.grad is None:
        grad = counter_factual.grad
    else:
        grad = counter_factual.grad + delta_ball.grad.sum(dim=0, keepdim=True)

    return loss, grad
