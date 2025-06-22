from property_procedures.utils import sample_delta_ball


def feasable_loss_function(
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
    # Extract loss value
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    # total uncertainty of the point itself

    # # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    # delta_ball = get_knn_points(counter_factual, k=n_points)
    probs_ensemble_delta_ball = probability_function(model, delta_ball)
    eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)

    target_probs = probs[0, desired_class]

    # loss = -target_probs + t*tu_point
    loss = (p_weight * target_probs) - lambda_1 * (eu_delta_ball.mean())
    loss = loss.mul(-1)

    # loss = (1-lam)*(-target_probs_delta_ball.mean()) + lam*tu_delta_ball.std()

    loss.backward()

    return loss, counter_factual.grad + delta_ball.grad.sum(dim=0, keepdim=True)
