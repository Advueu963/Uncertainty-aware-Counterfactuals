import torch
from property_procedures.utils import sample_delta_ball

def robust_procedure(model,
                    point_to_explain, probability_function,
                        epistemic_uncertainty_function,
                        aleatoric_uncertainty_function,
                     desired_class, MAX_STEPS=1000, DESIRED_VALIDITY=0.8,
                     delta=0.2, n_points=10):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble  = probability_function(model,counter_factual)
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
        eu = epistemic_uncertainty_function(probs_ensemble)


        # sample_ball
        delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
        probs_ensemble_delta_ball = probability_function(model, delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)
        eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)
        au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)


        target_probs = probs[0,desired_class]
        target_probs_delta_ball = probs_delta_ball[:,desired_class]



        # Calculate the loss
        lam1 = 0.6
        lam2 = 0.8 if target_probs > 0.5 else 0
        loss = -(target_probs_delta_ball.mean()) + lam1*(eu_delta_ball.max()) + lam2*(au_delta_ball.max())

        # Backpropagate
        loss.backward()
        counter_factual.grad = delta_ball.grad.mean(dim=0).view(-1,2)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps
