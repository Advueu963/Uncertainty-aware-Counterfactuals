import torch
from torch.optim import Adam


def counter_factual_optimization_routine(
    point_to_explain,
    model,
    loss_function,
    probability_function,
    aleatoric_uncertainty_function,
    epistemic_uncertainty_function,
    desired_class,
    MAX_STEPS=1000,
    DESIRED_VALIDITY=0.8,
    lr=0.01,
    p_weight=1,
    lambda_1=0.1,
    lambda_2=0.7,
    patience=10,
    delta=0.1,
    n_points=10,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = Adam([counter_factual], lr=lr)

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    target_probs = probs[0, desired_class]
    # Optimization Iteration

    T = 0
    patience_counter = 0
    prior_loss = 0
    while (
        T < MAX_STEPS
        and target_probs < DESIRED_VALIDITY
        and patience_counter < patience
    ):
        optimizer.zero_grad()

        loss, grad = loss_function(
            point_to_explain=point_to_explain,
            counter_factual=counter_factual,
            model=model,
            probability_function=probability_function,
            aleatoric_uncertainty_function=aleatoric_uncertainty_function,
            epistemic_uncertainty_function=epistemic_uncertainty_function,
            desired_class=desired_class,
            p_weight=p_weight,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            delta=delta,
            n_points=n_points,
        )

        counter_factual.grad = grad
        optimizer.step()
        # print(counter_factual,au, t)

        # Extract Target probabilities
        probs_ensemble = probability_function(model, counter_factual.detach().clone())
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[0, desired_class]

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
