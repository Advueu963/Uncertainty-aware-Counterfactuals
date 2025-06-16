import torch

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


def sparse_procedure_iteratively(
    point_to_explain,
    probability_function,
    desired_class,
    MAX_STEPS=1000,
    delta=1,
    n_points=10,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble = probability_function(counter_factual)
    probs = probs_ensemble.mean(dim=0)

    target_probs = probs[0, desired_class]

    # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(), delta, n_points)
    # delta_ball = get_knn_points(point_to_explain, k=n_points)
    probs_ensemble_delta_ball = probability_function(delta_ball)

    T = 0
    bernoulli_prob_importance = torch.zeros(counter_factual.shape)
    while T < MAX_STEPS:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble = probability_function(counter_factual)
        probs = probs_ensemble.mean(dim=0)
        # eu = epistemic_uncertainty_ensemble(probs_ensemble)

        # sample_ball
        delta_ball = sample_delta_ball(
            counter_factual.detach().numpy(), delta, n_points
        )
        # delta_ball = get_knn_points(counter_factual, k=n_points)
        probs_ensemble_delta_ball = probability_function(delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)

        target_probs = probs[0, desired_class]
        target_probs_delta_ball = probs_delta_ball[:, desired_class]

        # Calculate the loss
        loss = -target_probs
        loss2 = -(target_probs_delta_ball.mean())

        # Backpropagate
        loss.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        most_magnitude_feature_change = delta_ball.grad.abs().mean(dim=0).argmax()
        bernoulli_prob_importance[0, most_magnitude_feature_change] += 1

        bernoulli_variable = torch.bernoulli(
            bernoulli_prob_importance.div(bernoulli_prob_importance.sum())
        )

        counter_factual.grad = bernoulli_variable * counter_factual.grad
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1

    return counter_factual.detach(), counter_factual_steps


def sparse_procedure_baseline(
    point_to_explain,
    probability_function,
    desired_class,
    MAX_STEPS=1000,
    delta=1,
    n_points=10,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble = probability_function(counter_factual)
    probs = probs_ensemble.mean(dim=0)

    target_probs = probs[0, desired_class]

    T = 0
    while T < MAX_STEPS:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble = probability_function(counter_factual)
        probs = probs_ensemble.mean(dim=0)

        target_probs = probs[0, desired_class]

        # Calculate the loss
        loss = -target_probs + 0.01 * torch.linalg.norm(
            counter_factual - point_to_explain.detach(), ord=1
        )

        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1

    return counter_factual.detach(), counter_factual_steps
