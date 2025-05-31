import torch
from property_procedures.utils import sample_delta_ball

def feasability_procedure(model,
                        point_to_explain,
                          probability_function,
                          epistemic_uncertainty_function,
                          desired_class, MAX_STEPS=1000, DESIRED_VALIDITY=0.8, delta=1, n_points=10):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
    #delta_ball = get_knn_points(point_to_explain, k=n_points)
    probs_ensemble_delta_ball = probability_function(model,delta_ball)

    probs = probability_function(model,counter_factual)
    probs_ensemble = probs.mean(dim=0)
    target_probs = probs_ensemble[0,desired_class]

    t = 0
    T = 0
    while T < MAX_STEPS and target_probs < DESIRED_VALIDITY:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble  = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=0)

        # total uncertainty of the point itself


        # # sample_ball
        delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
        #delta_ball = get_knn_points(counter_factual, k=n_points)
        probs_ensemble_delta_ball = probability_function(model, delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)
        eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)

        target_probs = probs[0,desired_class]
        target_probs_delta_ball = probs_delta_ball[:,desired_class]

        #loss = -target_probs + t*tu_point
        loss = (-target_probs_delta_ball.mean()) + 0.5*eu_delta_ball.max() #- lam2*(au_delta_ball.max())
        #loss = (1-lam)*(-target_probs_delta_ball.mean()) + lam*tu_delta_ball.std()

        # Backpropagate
        loss.backward()
        counter_factual.grad = delta_ball.grad.mean(dim=0).view(-1,2)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps