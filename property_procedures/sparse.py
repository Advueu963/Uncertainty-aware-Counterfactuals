from property_procedures.utils import sample_delta_ball


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
    probs_ensemble_cf = probability_function(model, counter_factual)
    target_probs_cf = probs_ensemble_cf.mean(dim=1)[0, desired_class]

    delt_ball_cf = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    #
    # # Extract loss value
    probs_ensemble_delta_ball = probability_function(model, delt_ball_cf)
    au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)

    differences = au_delta_ball.std(dim=0)

    if target_probs_cf < 0.5:
        # negative loss to increase the uncertainty
        loss = -(differences)
    else:
        # positive loss to then decrease the uncertainty
        loss = differences
    loss.backward()

    # pick the grad of the point with the largest difference
    grad = delt_ball_cf.grad.mean(dim=0).view(1, *counter_factual.shape[1:])

    return loss, grad
