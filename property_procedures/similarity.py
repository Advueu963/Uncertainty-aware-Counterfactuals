from property_procedures.utils import sample_line, sample_delta_ball


def similarity_loss_function(
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

    reference_point = point_to_explain.clone().detach().requires_grad_(True)

    # sample ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    probs_ensemble_delta_ball = probability_function(model, delta_ball)
    au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)

    # sample_ball
    sampled_line_points = sample_line(
        counter_factual, reference_point, num_samples=n_points
    )
    probs_ensemble_line_points = probability_function(model, sampled_line_points)
    _ = aleatoric_uncertainty_function(probs_ensemble_line_points)

    target_probs = probs[0, desired_class]
    # Calculate the loss
    loss = p_weight * (-target_probs) - lambda_2 * au_delta_ball.max()

    loss.backward()

    return loss, counter_factual.grad + delta_ball.grad.sum(dim=0, keepdim=True)
