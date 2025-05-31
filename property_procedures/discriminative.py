import torch

def discriminiative_procedure(model,
                              point_to_explain,
                              probability_function,
                              aleatoric_uncertainty_function,
                              desired_class,
                              MAX_STEPS=1000,
                              DESIRED_VALIDITY=0.8):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble  = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=0)

    target_probs = probs[0,desired_class]
    # Optimization Iteration

    t = 0
    T = 0
    while T < MAX_STEPS and target_probs < DESIRED_VALIDITY:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble  = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=0)
        au = aleatoric_uncertainty_function(probs_ensemble)


        target_probs = probs[0,desired_class]

        # Calculate the loss
        lam= 1 if target_probs > 0.7 else 0.001
        loss = (1-lam)*(-target_probs) + lam*au

        # Backpropagate
        loss.backward()
        optimizer.step()
        #print(counter_factual,au, t)

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps