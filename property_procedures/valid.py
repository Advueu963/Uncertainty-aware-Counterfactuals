import torch

from property_procedures.utils import total_uncertainty_ensemble


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
):
    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)
    target_probs = probs[0, desired_class]

    if target_probs < 0.5:
        loss = -p_weight * (
            lambda_2 * aleatoric_uncertainty_function(probs_ensemble)
            + lambda_1 * epistemic_uncertainty_function(probs_ensemble)
        )
    else:
        loss = -(p_weight * target_probs)

    loss.backward()

    return loss, counter_factual.grad


def validity_procedure(
    model,
    point_to_explain,
    probability_function,
    desired_class,
    MAX_STEPS=1000,
    DESIRED_VALIDITY=0.8,
    p_weight=1,
    lr=0.01,
    lambda_1=0.1,
    lambda_2=0.7,
    patience=10,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=lr)

    # Get the probs in the start
    probs = probability_function(model, counter_factual).mean(dim=1)
    target_probs = probs[0, desired_class]
    # Optimization Iteration
    T = 0
    patience_counter = 0
    prior_loss = 0
    while T < MAX_STEPS and target_probs < DESIRED_VALIDITY:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[0, desired_class]
        # Calculate the loss

        if target_probs < 0.5:
            loss = -p_weight * total_uncertainty_ensemble(probs_ensemble)
        else:
            loss = -(p_weight * target_probs).log2()

        # if target_probs >= 0.5:
        #     loss = -(p_weight * target_probs)
        # else:
        #     loss = -(p_weight * target_probs) - total_uncertainty_ensemble(probs_ensemble)
        # print(target_probs) probs[:,probs.argmax(dim=1)].log2()
        # Backpropagate
        loss.backward()
        # print(counter_factual)
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
