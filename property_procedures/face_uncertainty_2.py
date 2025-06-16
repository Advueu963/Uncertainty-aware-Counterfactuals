# import helpers

import numpy as np

# import Dijkstra's shortest path algorithm
from scipy.sparse import csgraph, csr_matrix

# import graph building methods
import torch
from property_procedures.utils import sample_line


def graph_search(
    data,
    index,
    keys_immutable,
    model,
    weight_function,
    p_norm=2,
    frac=1,
):
    # This one implements the FACE method from
    # Rafael Poyiadzi et al (2020), "FACE: Feasible and Actionable Counterfactual Explanations",
    # Conference on AI, Ethics & Accountability (AIES, 2020)
    """
    :param data: df
    :param n_neighbors: int > 0; number of neighbors when constructing knn graph
    :param step: float > 0; step_size for growing spheres
    :param mode: str; either 'knn' or 'epsilon'
    :param model: classification model (either tf keras, pytorch or sklearn)
    :param p_norm: float=>1; denotes the norm (classical: 1 or 2)
    :param frac: float 0 < number =< 1; fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    :param keys_immutable: list; list of input names that may not be searched over
    :param radius: float > 0; parameter for epsilon density graph
    :return: candidate_counterfactual_star: np array (min. cost counterfactual explanation)
    """
    # Choose a subset of data for computational efficiency
    data = choose_random_subset(data, frac, index)

    # ADD CONSTRAINTS by immutable inputs into adjacency matrix
    # if element in adjacency matrix 0, then it cannot be reached
    # this ensures that paths only take same sex / same race / ... etc. routes
    if len(keys_immutable) == 0:
        immutable_constraint_matrix1 = np.ones((data.shape[0], data.shape[0]))
        immutable_constraint_matrix2 = np.ones((data.shape[0], data.shape[0]))
    else:
        for i in range(len(keys_immutable)):
            (
                immutable_constraint_matrix1,
                immutable_constraint_matrix2,
            ) = build_constraints(data, i, keys_immutable)

    # POSITIVE PREDICTIONS
    y_predicted, _ = model.predict(data)
    y_predicted = np.argmax(y_predicted.detach().numpy(), axis=1)
    y_positive_indeces = np.where(y_predicted == 1)

    # obtain candidate targets (CT); two conditions need to be met:
    # (1) CT needs to be predicted positively & (2) CT needs to have certain "density"
    # for knn I interpret 'certain density' as sufficient number of neighbours
    neighbor_candidates = find_counterfactuals(
        data,
        model,
        immutable_constraint_matrix1,
        immutable_constraint_matrix2,
        index,
        y_positive_indeces,
    )

    candidate_counterfactual_star = np.array(neighbor_candidates)

    # STEP 4 -- COMPUTE DISTANCES between x^F and candidate x^CF; else return NaN
    if candidate_counterfactual_star.size == 0:
        candidate_counterfactual_star = np.empty(
            data.shape[1],
        )
        candidate_counterfactual_star[:] = np.nan

        return candidate_counterfactual_star

    if p_norm == 1:
        c_dist = np.abs((data[index] - candidate_counterfactual_star)).sum(axis=1)
    elif p_norm == 2:
        c_dist = np.square((data[index] - candidate_counterfactual_star)).sum(axis=1)
    else:
        raise ValueError("Distance not defined yet. Choose p_norm to be 1 or 2")

    min_index = np.argmin(c_dist)
    candidate_counterfactual_star = candidate_counterfactual_star[min_index]

    return candidate_counterfactual_star


def choose_random_subset(data, frac, index):
    """
    Choose a subset of data for computational efficiency

    Parameters
    ----------
    data : pd.DataFrame
    frac: float 0 < number =< 1
        fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    index: int

    Returns
    -------
    pd.DataFrame
    """
    number_samples = int(np.rint(frac * data.shape[0]))
    list_to_choose = (
        np.arange(0, index).tolist() + np.arange(index + 1, data.shape[0]).tolist()
    )
    chosen_indeces = np.random.choice(
        list_to_choose,
        replace=False,
        size=number_samples,
    )
    chosen_indeces = [
        index
    ] + chosen_indeces.tolist()  # make sure sample under consideration included
    data = data[chosen_indeces]
    return data


def build_constraints(data, i, keys_immutable, epsilon=0.5):
    """

    Parameters
    ----------
    data: pd.DataFrame
    i : int
        Position of immutable key
    keys_immutable: list[str]
        Immutable feature
    epsilon: int

    Returns
    -------
    np.ndarray, np.ndarray
    """
    immutable_constraint_matrix = np.outer(
        data[keys_immutable[i]].values + epsilon,
        data[keys_immutable[i]].values + epsilon,
    )
    immutable_constraint_matrix1 = immutable_constraint_matrix / ((1 + epsilon) ** 2)
    immutable_constraint_matrix1 = ((immutable_constraint_matrix1 == 1) * 1).astype(
        float
    )
    immutable_constraint_matrix2 = immutable_constraint_matrix / (epsilon**2)
    immutable_constraint_matrix2 = ((immutable_constraint_matrix2 == 1) * 1).astype(
        float
    )
    return immutable_constraint_matrix1, immutable_constraint_matrix2


def find_counterfactuals(
    data,
    weight_function,
    immutable_constraint_matrix1,
    immutable_constraint_matrix2,
    index,
    y_positive_indeces,
):
    """
    Steps 1 to 3 of the FACE algorithm

    Parameters
    ----------

    Returns
    -------
    list
    """
    # STEP 1 -- BUILD NETWORK GRAPH
    graph = build_graph(
        data,
        weight_function,
        immutable_constraint_matrix1,
        immutable_constraint_matrix2,
    )
    # STEP 2 -- APPLY SHORTEST PATH ALGORITHM  ## indeces=index (corresponds to x^F)
    distances, min_distance = shortest_path(graph, index)
    distances_y_positive = distances[y_positive_indeces]
    min_distance_y_positive = distances_y_positive.min()

    # STEP 3 -- FIND COUNTERFACTUALS
    # minimum distance candidate counterfactuals
    min_distance_indeces = np.array([0])
    min_distance_indeces = np.c_[
        min_distance_indeces, np.array(np.where(distances == min_distance_y_positive))
    ]
    min_distance_indeces = np.delete(min_distance_indeces, 0)
    indeces_counterfactuals = np.intersect1d(
        np.array(y_positive_indeces), np.array(min_distance_indeces)
    )

    candidate_counterfactuals_star = []
    for i in range(indeces_counterfactuals.shape[0]):
        candidate_counterfactuals_star.append(data[indeces_counterfactuals[i]])

    return candidate_counterfactuals_star


def shortest_path(graph, index):
    """
    Uses dijkstras shortest path

    Parameters
    ----------
    graph: CSR matrix
    index: int

    Returns
    -------
    np.ndarray, float
    """
    distances = csgraph.dijkstra(
        csgraph=graph, directed=False, indices=index, return_predecessors=False
    )
    distances[index] = np.inf  # avoid min. distance to be x^F itself
    min_distance = distances.min()
    return distances, min_distance


def build_graph(
    data, model, immutable_constraint_matrix1, immutable_constraint_matrix2
):
    """

    Parameters
    ----------
    immutable_constraint_matrix1: np.ndarray
    immutable_constraint_matrix2: np.ndarray
    is_knn: bool
    n: int

    Returns
    -------
    CSR matrix
    """
    adjacency_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                line = sample_line(
                    torch.tensor(data[i]).float(),
                    torch.tensor(data[j]).float(),
                    num_samples=10,
                )
                _, eu = model.predict(line)

                _, eu = model.predict(np.vstack([data[i], data[j]]))
                adjacency_matrix[i, j] = 1 if eu.detach().max() < 0.5 else 0
    adjacency_matrix = np.multiply(
        adjacency_matrix,
        immutable_constraint_matrix1,
        immutable_constraint_matrix2,
    )  # element wise multiplication
    graph = csr_matrix(adjacency_matrix)
    return graph
