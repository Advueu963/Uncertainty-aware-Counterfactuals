import torch
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


def connected_ball_procedure(
    model,
    point_to_explain,
    probability_function,
    aleatoric_uncertainty_function,
    epistemic_uncertainty_function,
    desired_class,
    MAX_STEPS=1000,
    DESIRED_VALIDITY=0.8,
    delta=1,
    n_points=10,
    p_weight=1,
    lr=0.01,
    lambda_1=0.1,
    lambda_2=0.7,
    patience=10,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=lr)

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)
    target_probs = probs[0, desired_class]

    T = 0
    patience_counter = 0
    prior_loss = 0
    while T < MAX_STEPS and target_probs < DESIRED_VALIDITY:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=1)

        # total uncertainty of the point itself

        # # sample_ball
        delta_ball = sample_delta_ball(
            counter_factual.detach().numpy(), delta, n_points
        )
        # delta_ball = get_knn_points(counter_factual, k=n_points)
        probs_ensemble_delta_ball = probability_function(model, delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=1)
        au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)
        eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)

        target_probs = probs[0, desired_class]
        target_probs_delta_ball = probs_delta_ball[:, desired_class]

        # loss = -target_probs + t*tu_point
        if target_probs < 0.5:
            loss = -(eu_delta_ball + au_delta_ball).max()
        else:
            loss = (
                -(p_weight * target_probs_delta_ball.mean())
                + lambda_1 * (eu_delta_ball.max())
                - lambda_2 * (au_delta_ball.max())
            )
        # loss = (1-lam)*(-target_probs_delta_ball.mean()) + lam*tu_delta_ball.std()

        # Backpropagate
        loss.backward()
        counter_factual.grad = delta_ball.grad.mean(dim=0).view(
            -1, *counter_factual.shape[1:]
        )
        optimizer.step()

        # Check for early stopping
        if prior_loss - loss.item() < 0.01:
            patience_counter += 1
        else:
            patience_counter = 0
        prior_loss = loss.item()

        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
    return counter_factual.detach(), counter_factual_steps


def connectedness_procedure(
    model,
    point_to_explain,
    reference_point,
    probability_function,
    aleatoric_uncertainty_function,
    epistemic_uncertainty_function,
    desired_class,
    MAX_STEPS=1000,
    n_points=10,
    lambda_1=0.1,
    lambda_2=0.7,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=0)

    # Optimization Iteration

    T = 0
    while T < MAX_STEPS:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=0)

        # sample_ball
        sampled_line_points = sample_line(
            counter_factual, reference_point, num_samples=n_points
        )
        probs_ensemble_line_points = probability_function(model, sampled_line_points)
        eu_line = epistemic_uncertainty_function(probs_ensemble_line_points)
        au_line = aleatoric_uncertainty_function(probs_ensemble_line_points)

        target_probs = probs[0, desired_class]

        # Calculate the loss
        loss = (
            (-target_probs) + lambda_1 * (eu_line.mean()) + lambda_2 * (au_line.std())
        )
        # loss = (-target_probs).add(t*au)
        # Backpropagate
        loss.backward()
        optimizer.step()
        # print(counter_factual, au_line.std() , t, T)

        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
    return counter_factual.detach(), counter_factual_steps
