import numpy as np
import os 
import subprocess

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    RANGE_VALIDITY = np.linspace(0.5, 0.9, 10)
    RANGE_DELTA = np.linspace(0.1, 2, 10)
    RANGE_LR = np.linspace(0.01,0.2,10)
    RANGE_LAMBDA_1 = np.linspace(1,10,10)
    RANGE_LAMBDA_2 = np.linspace(1,10,10)
    RANGE_P_WEIGHT = np.linspace(1,10,10)
    RANGE_N_POINTS = np.linspace(1,20,10).astype(int)
    
    
    N_SWEEPS = 50
    combinations = []
    for _ in range(N_SWEEPS):
        combinations.append((
            rng.choice(RANGE_VALIDITY),
            rng.choice(RANGE_DELTA),
            rng.choice(RANGE_LR),
            rng.choice(RANGE_LAMBDA_1),
            rng.choice(RANGE_LAMBDA_2),
            rng.choice(RANGE_P_WEIGHT),
            rng.choice(RANGE_N_POINTS),
        ))
    print("Generated combinations: ", combinations)
    with open("sweep_configurations.txt","w") as f:
        for comb in combinations:
            f.write(",".join([str(c) for c in comb])+"\n")
    print("Saved to sweep_configurations.txt")
    
    
    
    #### CARLA CLUE HYPERPARAMETERS ####
    RANGE_WIDTH = [5,10,20]
    RANGE_DEPTH = [2,3,4]
    RANGE_LATENT_DIM = [5,10,20]
    RANGE_BATCH_SIZE = [32,64,128]
    RANGE_EPOCHS = [10,20,50]
    RANGE_LR_CARLA = [0.001,0.01,0.1]
    RANGE_EARLY_STOP = [5,10,20]
    rng = np.random.default_rng(42)
    combinations = []
    for _ in range(N_SWEEPS):
        combinations.append((
            rng.choice(RANGE_WIDTH),
            rng.choice(RANGE_DEPTH),
            rng.choice(RANGE_LATENT_DIM),
            rng.choice(RANGE_BATCH_SIZE),
            rng.choice(RANGE_EPOCHS),
            rng.choice(RANGE_LR_CARLA),
            rng.choice(RANGE_EARLY_STOP),
        ))
    print("Generated CARLA combinations: ", combinations)
    with open("sweep_configurations_clue.txt","w") as f:
        for comb in combinations:
            f.write(",".join([str(c) for c in comb])+"\n")
    print("Saved to sweep_configurations_clue.txt")
    
    #### CARLA FACE HYPERPARAMETERS ####
    RANGE_MODE = ["knn","epsilon"]
    RANGE_FRACTION = np.linspace(0.05,0.5,20)
    RANGE_RADIUS = np.linspace(0.1,2,20)
    rng = np.random.default_rng(42)
    combinations = []   
    for _ in range(N_SWEEPS):
        combinations.append((
            rng.choice(RANGE_MODE),
            rng.choice(RANGE_FRACTION),
            rng.choice(RANGE_RADIUS),
        ))
    print("Generated FACE combinations: ", combinations)
    with open("sweep_configurations_face.txt","w") as f:
        for comb in combinations:
            f.write(",".join([str(c) for c in comb])+"\n")
    print("Saved to sweep_configurations_face.txt")     
    