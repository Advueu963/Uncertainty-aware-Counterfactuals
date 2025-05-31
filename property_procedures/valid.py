import torch


def validity_procedure(model, point_to_explain, probability_function, desired_class, MAX_STEPS=1000, DESIRED_VALIDITY=0.8):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    # Get the probs in the start
    probs = probability_function(model, counter_factual).mean(dim=0)
    target_probs = probs[0,desired_class]
    # Optimization Iteration
    T = 0
    while T < MAX_STEPS and target_probs < DESIRED_VALIDITY:
        optimizer.zero_grad()

        # Extract loss value
        probs = probability_function(model, counter_factual).mean(dim=0)
        target_probs = probs[0,desired_class]
        # Calculate the loss
        loss = -target_probs
        #print(target_probs)
        # Backpropagate
        loss.backward()
        #print(counter_factual)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps
