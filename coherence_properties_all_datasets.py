import pandas as pd
import torch
import numpy as np
from carla.recourse_methods import GrowingSpheres, Clue, Dice, Face
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from data import (
    load_l_dataset,
    load_ring_dataset,
    load_bubbles,
    load_bubbles_noisy,
    load_one_moon,
    load_two_moon,
    load_infinity_dataset,
)
from property_procedures import (
    similarity_loss_function,
    validity_loss_function,
    robust_loss_function,
    feasable_loss_function,
    counter_factual_optimization_routine,
    combined_loss_function,
    connected_loss_function,
    discriminative_loss_function,
    plausable_loss_function,
)
import matplotlib.pyplot as plt
from property_procedures.utils import (
    ensemble_probs,
    epistemic_uncertainty_ensemble,
    aleatoric_uncertainty_ensemble,
)
from synthetic_to_carla import Synthetic_CARLA, MyOwnModel

DESIRED_VALIDITY = 0.999
DELTA = 0.2
OPTIMIZER_LR = 0.01
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
MAX_STEPS = 5000
PATIENCE = MAX_STEPS
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
N_POINTS = 50
N_EPOCHS = 50
base_ensemble = [
    MLP_Classifier(
        input_shape=2,
        n_classes=2,
        n_layers=2,
        num_neurons=64,
        dropout_prob=0,
        batch_norm=False,
    )
    for _ in range(ENSEMBLE_MEMBER_COUNT)
]


DATASET_LOADERS = [
    (
        "bubbles",
        load_bubbles,
        {"n_samples": 1000},
        torch.tensor(
            np.array([3, 3]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "l_dataset",
        load_l_dataset,
        {"n_samples": 333},
        torch.tensor(
            np.array([0, 10]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "one_moon",
        load_one_moon,
        {"n_samples": 1000},
        torch.tensor(
            np.array([-1, 0]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "ring_dataset",
        load_ring_dataset,
        {"n_samples": 1000, "inner_radius": 1.0, "outer_radius": 2.0, "noise": 0.1},
        torch.tensor(
            np.array([-1, -2]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "bubbles_noisy",
        load_bubbles_noisy,
        {"n_samples": 1000},
        torch.tensor(
            np.array([2.5, 2.5]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "two_moon",
        load_two_moon,
        {"n_samples": 1000},
        torch.tensor(
            np.array([0, 1]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
    (
        "infinity_dataset",
        load_infinity_dataset,
        {"n_samples": 1000},
        torch.tensor(
            np.array([2, 0]).reshape(-1, 2), dtype=torch.float, requires_grad=True
        ),
    ),
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


def visualize_other_cf_methods(point_of_interest, points, y_labels, axes, dataset_name):
    dataset = Synthetic_CARLA(dataset_name)

    model = MyOwnModel(dataset, n_models=ENSEMBLE_MEMBER_COUNT, n_epochs=N_EPOCHS)

    # load artificial neural networke from catalog
    # model = MLModelCatalog(dataset, "ann", backend="pytorch")

    # load a recourse model and pass black box model
    gs = GrowingSpheres(model)
    clue = Clue(dataset, model)
    dice = Dice(model)
    fa = Face(model, {"mode": "knn", "fraction": 0.2})

    point_of_interest_df = pd.DataFrame(
        {
            "x0": [p for p in [*point_of_interest[:, 0].detach().numpy()]],
            "x1": [p for p in [*point_of_interest[:, 1].detach().numpy()]],
            "label": [0] * (1),
        }
    )

    # generate counterfactual examples
    print("Factuals:")
    print(point_of_interest_df)
    counterfactuals_growing_sphere = gs.get_counterfactuals(point_of_interest_df)

    counterfactuals_dice = dice.get_counterfactuals(point_of_interest_df)

    # generate counterfactual examples using CLUE
    try:
        counterfactuals_clue = clue.get_counterfactuals(point_of_interest_df)
        print(counterfactuals_clue)
    except ValueError as e:
        print(f"Error generating counterfactuals with CLUE: {e}")
        counterfactuals_clue = None

    try:
        counterfactuals_face = fa.get_counterfactuals(point_of_interest_df)
        print(counterfactuals_face)
    except ValueError as e:
        print(f"Error generating counterfactuals with Face: {e}")
        counterfactuals_face = None

    return (
        counterfactuals_growing_sphere,
        counterfactuals_dice,
        counterfactuals_clue,
        counterfactuals_face,
    )


fig, axes = plt.subplots(
    len(DATASET_LOADERS), len(PROPERTY_LOADERS) + 3 + 2 + 1 + 4, figsize=(80, 40)
)

coherence = []
coherence_std = []
for i, (dataset_name, loader, kwargs, point_of_interest) in enumerate(DATASET_LOADERS):
    print(f"Training ensemble on {dataset_name}...")
    points, y_labels, y_probs = loader(**kwargs)

    # Train the ensemble model
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

    # Train some different models

    random_Forest = RandomForestClassifier()
    random_Forest.fit(points.detach().numpy(), y_labels.detach().numpy())

    logistic_regression = LogisticRegression()
    logistic_regression.fit(points.detach().numpy(), y_labels.detach().numpy())

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(points.detach().numpy(), y_labels.detach().numpy())

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(points.detach().numpy(), y_labels.detach().numpy())

    # Print some results
    print(
        f"Random Forest Accuracy: {random_Forest.score(points.detach().numpy(), y_labels.detach().numpy()):.4f}"
    )
    print(
        f"Logistic Regression Accuracy: {logistic_regression.score(points.detach().numpy(), y_labels.detach().numpy()):.4f}"
    )
    print(
        f"KNN Accuracy: {knn.score(points.detach().numpy(), y_labels.detach().numpy()):.4f}"
    )
    print(
        f"Decision Tree Accuracy: {decision_tree.score(points.detach().numpy(), y_labels.detach().numpy()):.4f}"
    )
    print("-" * 40)

    models = [decision_tree, random_Forest, logistic_regression, knn]

    coherence.append([])
    coherence_std.append([])

    # Visualize other CF Methods

    cf_gs, cf_clue, cf_dice, cf_face = visualize_other_cf_methods(
        point_of_interest, points, y_labels, axes[i, -5:-1], dataset_name
    )

    for j, (property_name, property_function) in enumerate(PROPERTY_LOADERS):
        print(f"Evaluating property: {property_name} on dataset: {dataset_name}")

        # Run the property procedure
        counter_factual, counter_factual_steps = counter_factual_optimization_routine(
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
            optimization_method="sgd",
        )

        # loop through the different models
        coherence_property = []
        for model in models:
            # Calculate the coherence
            prediction = model.predict(counter_factual.detach().numpy())
            if isinstance(prediction, tuple):
                prediction = prediction[0].argmax(dim=-1)
            coherence_property.append(prediction == DESIRED_CLASS)
        coherence[-1].append(coherence_property)

    for cf_baseline in [cf_gs, cf_clue, cf_dice, cf_face]:
        coherence_baseline = []
        for model in models:
            # Calculate the coherence
            prediction = model.predict(point_of_interest.detach().numpy())
            if isinstance(prediction, tuple):
                prediction = prediction[0].argmax(dim=-1)
            coherence_baseline.append(prediction == DESIRED_CLASS)
        coherence[-1].append(coherence_baseline)


coherence = np.array(coherence)
coherence = coherence.mean(axis=(-2, -1))
dataframe = pd.DataFrame(
    coherence,
    columns=[
        *[property_name for property_name, _ in PROPERTY_LOADERS],
        "GS",
        "CLUE",
        "DICE",
        "FACE",
    ],
    index=[dataset_name for dataset_name, _, _, _ in DATASET_LOADERS],
)
dataframe.to_csv("coherence.csv")
