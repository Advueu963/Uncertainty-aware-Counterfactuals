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

    au_cf = aleatoric_uncertainty_function(probs_ensemble_cf)

    #
    # # Extract loss value
    loss = -(p_weight * au_cf)
    loss.backward()

    # pick the grad of the point with the largest difference
    grad = counter_factual.grad

    return loss, grad
