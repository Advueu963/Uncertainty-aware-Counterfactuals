import  torch
from property_procedures.utils import sample_delta_ball

def stable_procedure_baseline(point_to_explain,
                              probability_function,
                              desired_class,
                              MAX_STEPS=1000, delta=1, n_points=10):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble  = probability_function(counter_factual)
    probs = probs_ensemble.mean(dim=0)

    target_probs = probs[0,desired_class]
    # Optimization Iteration

    t = 0
    T = 0
    while T < MAX_STEPS:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble  = probability_function(counter_factual)
        probs = probs_ensemble.mean(dim=0)


        # sample_ball
        delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
        probs_ensemble_delta_ball = probability_function(delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)


        target_probs = probs[0,desired_class]
        target_probs_delta_ball = probs_delta_ball[:,desired_class]

        loss = (-target_probs_delta_ball.mean())

        # Backpropagate
        loss.backward()
        counter_factual.grad = delta_ball.grad.mean(dim=0,keepdims=True)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps

def stable_procedure_tu_start_ball(point_to_explain, probability_function,
                                      aleatoric_uncertainty_function,
                                    epistemic_uncertainty_function,
                                   total_uncertainty_function,
                                   desired_class, MAX_STEPS=1000, delta=1, n_points=10):
    counter_factual = point_to_explain.detach().clone().requires_grad_(True)
    counter_factual_steps = counter_factual.detach().clone()
    optimizer = torch.optim.Adam([counter_factual], lr=0.01)

    probs_ensemble  = probability_function(counter_factual)
    probs = probs_ensemble.mean(dim=0)

    target_probs = probs[0,desired_class]


    # sample_ball
    delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
    #delta_ball = get_knn_points(point_to_explain, k=n_points)
    probs_ensemble_delta_ball = probability_function(delta_ball)
    probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)
    tu_delta_ball_start = total_uncertainty_function(probs_ensemble_delta_ball).detach()
    au_delta_ball_start = epistemic_uncertainty_function(probs_ensemble_delta_ball).detach()
    eu_delta_ball_start = aleatoric_uncertainty_function(probs_ensemble_delta_ball).detach()
    # Optimization Iteratio

    t = 0
    T = 0
    while T < MAX_STEPS:
        optimizer.zero_grad()

        # Extract loss value
        probs_ensemble  = probability_function(counter_factual)
        probs = probs_ensemble.mean(dim=0)
        au = total_uncertainty_function(probs_ensemble)
        #eu = epistemic_uncertainty_ensemble(probs_ensemble)


        # sample_ball
        delta_ball = sample_delta_ball(counter_factual.detach().numpy(),delta,n_points)
        #delta_ball = get_knn_points(counter_factual, k=n_points)
        probs_ensemble_delta_ball = probability_function(delta_ball)
        probs_delta_ball = probs_ensemble_delta_ball.mean(dim=0)
        tu_delta_ball = total_uncertainty_function(probs_ensemble_delta_ball)
        au_delta_ball = aleatoric_uncertainty_function(probs_ensemble_delta_ball)
        eu_delta_ball = epistemic_uncertainty_function(probs_ensemble_delta_ball)


        target_probs = probs[0,desired_class]
        target_probs_delta_ball = probs_delta_ball[:,desired_class]

        # Calculate the loss
        lam = 0.1
        loss =  -target_probs_delta_ball.mean() + lam*( (tu_delta_ball.mean() - tu_delta_ball_start.mean())**2 + (tu_delta_ball.std() -tu_delta_ball_start.std())**2 )
        #loss =  -((1-lam)*target_probs_delta_ball.mean()) + lam*( (au_delta_ball.mean() - au_delta_ball_start.mean())**2 + (au_delta_ball.std() -au_delta_ball_start.std())**2 )
        #loss =  -((1-lam)*target_probs_delta_ball.mean()) + lam*( (eu_delta_ball.mean() - eu_delta_ball_start.mean())**2 + (eu_delta_ball.std() -eu_delta_ball_start.std())**2 )




        # Backpropagate
        loss.backward()
        #lam = 0.9
        counter_factual.grad = delta_ball.grad.mean(dim=0,keepdims=True)
        optimizer.step()

        # Save the intermediate steps
        counter_factual_steps = torch.cat((counter_factual_steps, counter_factual.detach()))
        T += 1
    return counter_factual.detach(), counter_factual_steps
