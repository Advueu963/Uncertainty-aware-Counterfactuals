def validity_loss_function(
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
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)
    target_probs = probs[0, desired_class]

    # if target_probs < 0.5:
    #     loss = -p_weight * (
    #         lambda_2 * aleatoric_uncertainty_function(probs_ensemble)
    #         + lambda_1 * epistemic_uncertainty_function(probs_ensemble)
    #     ).log2()
    # else:
    #     loss = -(p_weight * target_probs.log2())

    loss = -p_weight * target_probs.log2()

    loss.backward()

    return loss, counter_factual.grad
