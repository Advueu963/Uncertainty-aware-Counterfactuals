from property_procedures.utils import sample_delta_ball, sample_line


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
    # Extract loss value
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    # # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    # delta_ball = get_knn_points(counter_factual, k=n_points)
    probs_ensemble_delta_ball = probability_function(model, delta_ball)
    probs_delta_ball = probs_ensemble_delta_ball.mean(dim=1)
    eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)

    target_probs = probs[0, desired_class]
    _ = probs_delta_ball[:, desired_class]

    loss = p_weight * target_probs.log2()
    if target_probs > 0.51:
        loss = loss - lambda_1 * (eu_delta_ball.mean().log2())

    loss = loss.mul(-1)

    loss.backward()
    if target_probs < 0.51:
        grad = counter_factual.grad
    else:
        grad = counter_factual.grad + delta_ball.grad.sum(dim=0)
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
