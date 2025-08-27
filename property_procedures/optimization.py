import torch
from torch.optim import Adam, SGD


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
    optimization_method="adam",
    **kwargs,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    # Initialize the optimizer
    if optimization_method == "adam":
        optimizer = Adam([counter_factual], lr=lr)
    elif optimization_method == "sgd":
        optimizer = SGD([counter_factual], lr=lr)
    else:
        raise ValueError("Unsupported optimization method. Use 'adam' or 'sgd'.")

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    target_probs = probs[0, desired_class]
    # Optimization Iteration

    T = 0
    patience_counter = 0
    prior_loss = 0
    start_cf_construction = False
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
            start_cf_construction=start_cf_construction,
            **kwargs,
        )

        counter_factual.grad = grad

        optimizer.step()
        # print(counter_factual,au, t)

        # Extract Target probabilities
        probs_ensemble = probability_function(model, counter_factual.detach().clone())
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[0, desired_class]

        # Check for early stopping
        if abs(prior_loss - loss.item()) < 0.01:
            patience_counter += 1
        else:
            patience_counter = 0

        prior_loss = loss.item()
        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
    # Print out reason for stopping
    if T >= MAX_STEPS:
        print("Reached maximum number of steps.")
    elif target_probs >= DESIRED_VALIDITY:
        print("Desired validity reached.")
    elif patience_counter >= patience:
        print("Patience limit reached.")
    return counter_factual.detach(), counter_factual_steps


def counter_factual_optimization_routine_schut(
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
    optimization_method="adam",
    **kwargs,
):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    # Initialize the optimizer
    if optimization_method == "adam":
        optimizer = Adam([counter_factual], lr=lr)
    elif optimization_method == "sgd":
        optimizer = SGD([counter_factual], lr=lr)
    else:
        raise ValueError("Unsupported optimization method. Use 'adam' or 'sgd'.")

    probs_ensemble = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=1)

    target_probs = probs[0, desired_class]
    # Optimization Iteration

    T = 0
    patience_counter = 0
    prior_loss = 0
    start_cf_construction = False
    counter_feature_changes = torch.zeros_like(counter_factual.flatten())
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
            start_cf_construction=start_cf_construction,
            **kwargs,
        )

        viewed_grad = grad.flatten()
        viewed_grad = viewed_grad[counter_feature_changes < 5]
        most_imporant_grad = viewed_grad.abs().argmax()
        applied_grad = torch.zeros_like(counter_factual)
        applied_grad = applied_grad.flatten()
        applied_grad[(grad.flatten() == viewed_grad[most_imporant_grad])] = 1.0 if viewed_grad[most_imporant_grad] > 0 else -1.0
        applied_grad = applied_grad.view_as(counter_factual)  # Reshape to match counter_factual
        
        counter_factual.grad = applied_grad
        counter_feature_changes[(grad.flatten() == viewed_grad[most_imporant_grad])] += 1

        optimizer.step()
        # print(counter_factual,au, t)

        # Extract Target probabilities
        probs_ensemble = probability_function(model, counter_factual.detach().clone())
        probs = probs_ensemble.mean(dim=1)
        target_probs = probs[0, desired_class]

        # Check for early stopping
        if abs(prior_loss - loss.item()) < 0.01:
            patience_counter += 1
        else:
            patience_counter = 0

        prior_loss = loss.item()
        # Save the intermediate steps
        counter_factual_steps = torch.cat(
            (counter_factual_steps, counter_factual.detach())
        )
        T += 1
    # Print out reason for stopping
    if T >= MAX_STEPS:
        print("Reached maximum number of steps.")
    elif target_probs >= DESIRED_VALIDITY:
        print("Desired validity reached.")
    elif patience_counter >= patience:
        print("Patience limit reached.")
    return counter_factual.detach(), counter_factual_steps
