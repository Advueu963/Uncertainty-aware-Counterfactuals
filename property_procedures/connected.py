import torch
from property_procedures.utils import sample_delta_ball, sample_line


def connected_ball_procedure(model, point_to_explain, probability_function,
                             aleatoric_uncertainty_function, epistemic_uncertainty_function,desired_class,
                             MAX_STEPS=1000,
                             DESIRED_VALIDITY=0.8,
                             delta=1, n_points=10):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble  = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=0)
    target_probs = probs[0,desired_class]

    t = 0
    T = 0
    while T < MAX_STEPS and target_probs < DESIRED_VALIDITY:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble  = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=0)
        eu = epistemic_uncertainty_function(probs_ensemble)

        # total uncertainty of the point itself


        # # sample_ball
        delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
        #delta_ball = get_knn_points(counter_factual, k=n_points)
        probs_ensemble_delta_ball = probability_function(model, delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)
        au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)
        eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)

        target_probs = probs[0,desired_class]
        target_probs_delta_ball = probs_delta_ball[:,desired_class]

        #loss = -target_probs + t*tu_point
        lam1 = 0.1
        lam2 = 0.7
        loss = -(target_probs_delta_ball.mean()) + lam2*(eu_delta_ball.max()) - lam1*(au_delta_ball.max())
        #loss = (1-lam)*(-target_probs_delta_ball.mean()) + lam*tu_delta_ball.std()

        # Backpropagate
        loss.backward()
        counter_factual.grad = delta_ball.grad.mean(dim=0).view(-1,2)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps

def connectedness_procedure(model, point_to_explain,reference_point, probability_function, aleatoric_uncertainty_function, epistemic_uncertainty_function,desired_class, MAX_STEPS=1000, n_points=10):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble  = probability_function(model, counter_factual)
    probs = probs_ensemble.mean(dim=0)

    # Optimization Iteration

    t = 0
    T = 0
    while T < MAX_STEPS:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble  = probability_function(model, counter_factual)
        probs = probs_ensemble.mean(dim=0)




        # sample_ball
        sampled_line_points = sample_line(counter_factual, reference_point, num_samples=n_points)
        probs_ensemble_line_points = probability_function(model, sampled_line_points)
        probs_delta_ball = probs_ensemble_line_points.mean(dim=0)
        eu_line = epistemic_uncertainty_function(probs_ensemble_line_points)
        au_line = aleatoric_uncertainty_function(probs_ensemble_line_points)


        target_probs = probs[0,desired_class]

        # Calculate the loss
        lam1 = 0.7
        lam2 = 1 if target_probs > 0.5 else 0
        loss = (-target_probs) +  lam1*(eu_line.mean()) + lam2*(au_line.std())
        #loss = (-target_probs).add(t*au)
        # Backpropagate
        loss.backward()
        optimizer.step()
        #print(counter_factual, au_line.std() , t, T)

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps
