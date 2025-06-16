from property_procedures.utils import sample_delta_ball


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
):
    # Extract loss value
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    # # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    # delta_ball = get_knn_points(counter_factual, k=n_points)
    probs_ensemble_delta_ball = probability_function(model, delta_ball)
    probs_delta_ball = probs_ensemble_delta_ball.mean(dim=1)
    au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)
    eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)

    target_probs = probs[0, desired_class]
    target_probs_delta_ball = probs_delta_ball[:, desired_class]

    # loss = -target_probs + t*tu_point
    if target_probs < 0.5:
        loss = -(eu_delta_ball + au_delta_ball).mean()
    else:
        loss = (
            -(p_weight * target_probs_delta_ball.mean())
            + lambda_1 * (eu_delta_ball.mean())
            + lambda_2 * (au_delta_ball.mean())
        )

    loss.backward()
    return loss, delta_ball.grad.mean(dim=0).view(-1, *counter_factual.shape[1:])
