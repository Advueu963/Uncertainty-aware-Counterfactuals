from data import load_l_dataset
import numpy as np
import matplotlib.pyplot as plt

from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from property_procedures.face_uncertainty_2 import graph_search
from property_procedures.face_method_default import graph_search as graph_search_default

X, y, y_probs = load_l_dataset()


def weight_function(x1, x2):
    return np.linalg.norm(x1 - x2)


point_of_interest = np.array([0, 10])

ENSEMBLE_MEMBER_COUNT = 20
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
ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
ensemble_model.load("models/Ensemble_L_dataset/")


indices = np.arange(len(X))
candidates = graph_search(
    np.vstack([point_of_interest, X]), 0, [], ensemble_model, lambda x, y: 1, frac=0.99
)
candidates2 = graph_search_default(
    np.vstack([point_of_interest, X]), 0, [], ensemble_model, frac=0.2
)
print("Candidates found:", candidates)
print("Candidates found (default):", candidates2)
# visualize the candidates
# What is distance between point of interest and candidates?


plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", alpha=0.5, label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="orange", alpha=0.5, label="Class 1")
plt.scatter(
    candidates[0],
    candidates[1],
    color="red",
    marker="x",
    label="Candidate_FACE_UNCERTAINTY",
)
plt.scatter(
    candidates2[0], candidates2[1], color="red", marker="x", label="Candidate_FACE"
)
plt.scatter(
    point_of_interest[0],
    point_of_interest[1],
    color="red",
    marker="o",
    label="Point of Interest",
)
plt.title("Candidate Counterfactuals")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
