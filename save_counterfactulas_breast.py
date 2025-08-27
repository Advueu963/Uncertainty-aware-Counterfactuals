import torch
import numpy as np
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from data import (
    load_breast_dataset,
    load_breast_dataset_poi,
)
from joblib import Parallel, delayed
from property_procedures import (
    validity_loss_function,
    connected_loss_function,
    robust_loss_function,
    feasable_loss_function,
    discriminative_loss_function,
    plausable_loss_function,
    similarity_loss_function,
    counter_factual_optimization_routine,
    combined_loss_function,
)
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
    counter_factual_baseline,
)
import os

DESIRED_VALIDITY = 0.8
DELTA = 0.2
OPTIMIZER_LR = 0.1
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = 50
N_EPOCHS = 50
N_ITERATIONS = 5  # Number of iterations for each property evaluation
OPTIMIZATION_METHOD = "sgd"

breast_cancer_poi = load_breast_dataset_poi()




DATASET_LOADERS = [
    ("breast_cancer", load_breast_dataset, {"n_samples": "all"}, point)
    for point in breast_cancer_poi
]
PROPERTY_LOADERS = [
    (
        "validity",
        validity_loss_function,
    ),
    (
        "connected_ball",
        connected_loss_function,
    ),
    (
        "robust",
        robust_loss_function,
    ),
    (
        "feasability",
        feasable_loss_function,
    ),
    (
        "discriminative",
        discriminative_loss_function,
    ),
    (
        "plausable",
        plausable_loss_function,
    ),
    (
        "similarity",
        similarity_loss_function,
    ),
    ("combined", combined_loss_function),
]


def main_routine(i, dataset_name, loader, kwargs, point_of_interest):
    print("Starting main routine for point of interest:", point_of_interest)
    print(f"Training ensemble on {dataset_name}...")
    points, y_labels, y_probs = loader(**kwargs)

    # Train the ensemble model
    base_ensemble = [
        MLP_Classifier(
            input_shape=30,
            n_classes=2,
            n_layers=2,
            num_neurons=64,
            dropout_prob=0,
            batch_norm=False,
        )
        for _ in range(ENSEMBLE_MEMBER_COUNT)
    ]
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")

    # Evaluate the ensemble model
    y_probs_ensemble, _ = ensemble_model.predict(points, raw_output=True)
    y_target_ensemble = y_probs_ensemble.mean(dim=1).argmax(dim=1)

    # Print some results
    print(f"Dataset: {dataset_name}")
    print(f"Points shape: {points.shape}")
    print(f"Labels shape: {y_labels.shape}")
    print(f"Probabilities shape: {y_probs_ensemble.shape}")
    print(
        f"Accuracy: {((y_target_ensemble == y_labels).sum() / len(y_labels)).item():.4f}"
    )
    print("-" * 40)

    point_closest_to_interest = points[y_labels == 0][
        torch.linalg.vector_norm(
            points[y_labels == 0] - point_of_interest, dim=1, ord=2
        ).argmin()
    ]
    point_closest_to_interest = point_closest_to_interest.view(1, -1)
    property_to_cfs = {}
    point_of_interest = point_of_interest.view(1, -1)
    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        cf_poi = []
        for _ in range(N_ITERATIONS):
            (
                counter_factual,
                counter_factual_steps,
            ) = counter_factual_optimization_routine(
                point_to_explain=point_of_interest,
                model=ensemble_model,
                probability_function=ensemble_probs,
                desired_class=DESIRED_CLASS,
                loss_function=property_function,
                aleatoric_uncertainty_function=aleatoric_uncertainty_ensemble,
                epistemic_uncertainty_function=epistemic_uncertainty_ensemble,
                MAX_STEPS=MAX_STEPS,
                delta=DELTA,
                n_points=N_POINTS,
                lr=OPTIMIZER_LR,
                DESIRED_VALIDITY=DESIRED_VALIDITY,
                p_weight=PROB_WEIGHT,
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
                patience=PATIENCE,
                optimization_method=OPTIMIZATION_METHOD,
            )
            cf_poi.append(counter_factual)
        cf_poi = torch.stack(cf_poi, dim=0).numpy()
        # shape: (N_ITERATIONS, 2)
        property_to_cfs[property_name] = cf_poi

        cf_poi_close = []
        for _ in range(N_ITERATIONS):
            cf_close, cf_steps_close = counter_factual_optimization_routine(
                point_to_explain=point_closest_to_interest,
                model=ensemble_model,
                probability_function=ensemble_probs,
                desired_class=DESIRED_CLASS,
                loss_function=property_function,
                aleatoric_uncertainty_function=aleatoric_uncertainty_ensemble,
                epistemic_uncertainty_function=epistemic_uncertainty_ensemble,
                MAX_STEPS=MAX_STEPS,
                delta=DELTA,
                n_points=N_POINTS,
                lr=OPTIMIZER_LR,
                DESIRED_VALIDITY=DESIRED_VALIDITY,
                p_weight=PROB_WEIGHT,
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
                patience=PATIENCE,
                optimization_method=OPTIMIZATION_METHOD,
            )
            cf_poi_close.append(cf_close)
        cf_poi_close = torch.stack(cf_poi_close, dim=0).numpy()
        # shape: (N_ITERATIONS, 2)
        property_to_cfs[f"{property_name}_close"] = cf_poi_close

    # Visualize other CF Methods
    cfs_baseline = []
    for _ in range(N_ITERATIONS):
        cf_gs, cf_clue, cf_dice, cf_face = counter_factual_baseline(
            dataset_name=dataset_name,
            point_of_interest=point_of_interest,
            n_models=20,
            n_epochs=N_EPOCHS,
            noisy=False,
        )
        if cf_gs is None:
            cf_gs = np.zeros_like(point_of_interest.detach())
        if cf_clue is None:
            cf_clue = np.zeros_like(point_of_interest.detach())
        if cf_dice is None:
            cf_dice = np.zeros_like(point_of_interest.detach())
        if cf_face is None:
            cf_face = np.zeros_like(point_of_interest.detach())
        cfs_baseline.append([cf_gs, cf_clue, cf_dice, cf_face])
    # shape: (N_ITERATIONS, 4, 2, 1)
    cfs_baseline = np.array(cfs_baseline)
    property_to_cfs["baselines"] = cfs_baseline

    #
    cfs_baseline = []
    for _ in range(N_ITERATIONS):
        cf_gs, cf_clue, cf_dice, cf_face = counter_factual_baseline(
            dataset_name=dataset_name,
            point_of_interest=point_closest_to_interest,
            n_models=20,
            n_epochs=N_EPOCHS,
            noisy=False,
        )
        if cf_gs is None:
            cf_gs = np.zeros_like(point_closest_to_interest.detach())
        if cf_clue is None:
            cf_clue = np.zeros_like(point_closest_to_interest.detach())
        if cf_dice is None:
            cf_dice = np.zeros_like(point_closest_to_interest.detach())
        if cf_face is None:
            cf_face = np.zeros_like(point_closest_to_interest.detach())
        cfs_baseline.append([cf_gs, cf_clue, cf_dice, cf_face])
    cfs_baseline = np.array(cfs_baseline)
    # shape: (N_ITERATIONS, 4, 2, 1)
    property_to_cfs["baselines_close"] = cfs_baseline

    np.savez(
        os.path.join(f"saved_cfs/counterfactuals_{dataset_name}_{N_EPOCHS}_{MAX_STEPS}/", f"counter_factuals_{dataset_name}_{i}.npz"),
        **property_to_cfs,
    )

if __name__ == "__main__":
    for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
        SAVE_FOLDER = f"saved_cfs/counterfactuals_{dataset_name}_{N_EPOCHS}_{MAX_STEPS}/"
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
    
    Parallel(n_jobs=-1)(
            delayed(main_routine) (i,dataset_name, loader, kwargs, point_of_interest)

            for  i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS)
            if i not in [8 , 9,  11, 13, 20]
        )
    