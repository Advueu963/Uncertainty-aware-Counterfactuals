import os
import numpy as np
import pandas as pd
from training.train_ensemble_tabular import get_categorical_features_all, get_dataset
import torch
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier

DATA_FOLDER = os.environ.get("SCRATCH_DSS")

from property_procedures.utils import (
    instability_metric,
    invalidity,
    dissimilarity,
    dissparsity,
    discriminative_power,
    implausability,
    ensemble_probs
)
DESIRED_CLASS = 1
ENSEMBLE_MEMBER_COUNT = 20
PROPERTY_NAMES = [
        "validity",
        "connected_ball",
        "robust",
        "feasability",
        "discriminative",
        "plausable",
        "similarity",
        "combined",
]
N_EPOCHS = 50

def load_property_data(dataset_name, property_name):
    data = np.load(
        os.path.join(
            DATA_FOLDER,
            f"CFs_{dataset_name}.npy",
        ),
        allow_pickle=True,
    ).item()
    return data[property_name]

def load_carla_data(dataset_name,method_name):
    try:
        path = os.path.join(
            DATA_FOLDER,
            f"CFs_{method_name}_{dataset_name}_closest.csv",
        )
        print(f"Loading CARLA data from {path}")
        data = pd.read_csv(
            path
        )
    except FileNotFoundError as e:
        print(e)
        return None
    section = len(data) // 2
    return {
        "counter_factual": data.iloc[:section].drop(columns=["construction_time","method","point_index"]).to_numpy(),
        "counter_factual_closest": data.iloc[section:].drop(columns=["construction_time","method","point_index"]).to_numpy(),
    }
    
def get_metrics(X_points, X_points_close, y_points, y_points_close, counter_factual, counter_factual_closest, X_test, y_test, ensemble_model, orginal_X, categorical_features_all):
    # Remove all counterfactuals which are NaN
    mask = ~counter_factual.isnan().any(dim=1)
    counter_factual = counter_factual[mask]
    counter_factual_closest = counter_factual_closest[mask]
    X_points = X_points[mask]
    X_points_close = X_points_close[mask]
    y_points = y_points[mask]
    y_points_close = y_points_close[mask]
    print("Calculation on", counter_factual.shape[0], "counterfactuals after removing NaNs".format())
    
    
    
    # Compute only the metrics for counterfactual which have the desired class
    mask = (ensemble_probs(ensemble_model, counter_factual).mean(dim=1).argmax(dim=1) == DESIRED_CLASS)
    counter_factual = counter_factual[mask]
    counter_factual_closest = counter_factual_closest[mask]
    X_points = X_points[mask]
    X_points_close = X_points_close[mask]
    y_points = y_points[mask]
    y_points_close = y_points_close[mask]
    print("Calculation on", counter_factual.shape[0], f"counterfactuals which are {100*mask.sum().item()/mask.shape[0] if mask.shape[0]>0 else 0}% of the original {mask.shape[0]} points".format())
    
    
    if counter_factual.shape[0] == 0:
        print("No counterfactuals found for desired class, returning NaNs for all metrics")
        return {
            "instability": torch.Tensor([np.nan]),
            "invalidity": torch.Tensor([np.nan]),
            "dissimilarity": torch.Tensor([np.nan]),
            "dissparsity": torch.Tensor([np.nan]),
            "implausability": torch.Tensor([np.nan]),
            "discriminative_power": torch.Tensor([np.nan]),
        }
    
    
    instability_values = instability_metric(
                point_of_interest=X_points,
                point_to_compare=X_points_close,
                cf_point_of_interest=counter_factual,
                cf_point_to_compare=counter_factual_closest,
                X=orginal_X,
                categorical_features=categorical_features_all,
            )
    invalidity_values = invalidity(
                counter_factual,
                model=ensemble_model,
                probability_function=ensemble_probs,
                desired_class=DESIRED_CLASS,
            )
    dissimilarity_values = dissimilarity(
                point_of_interest=X_points,
                cf_point_of_interest=counter_factual,
                X=orginal_X,
                categorical_features=categorical_features_all,
            )
    dissparsity_values = dissparsity(
                point_of_interest=X_points,
                cf_point_of_interest=counter_factual,
            )
            
    implausability_values = implausability(
                cf_point_of_interest=counter_factual,
                X=orginal_X,
                categorical_features=categorical_features_all,
            )
            
    discriminative_values = []
    for i in range(counter_factual.shape[0]):
        mask = y_test == y_points[i] # y_points will never be DESIRED_CLASS
        X_equal_poi = torch.Tensor(X_test[mask])
        X_diff_poi = torch.Tensor(X_test[y_test == DESIRED_CLASS])
        if counter_factual[i].isnan().any():
            discriminative_values.append(np.nan)
            continue
        discriminative_values.append(
        discriminative_power(
            point_of_interest=X_points[i:i+1],
            cf_point_of_interest=counter_factual[i:i+1],
            class_poi=y_points[i:i+1],
            class_cf=torch.ones_like(y_points[i:i+1])*DESIRED_CLASS,
            X_equal_poi=X_equal_poi,
            X_diff_poi=X_diff_poi,
        )
        )

    discriminative_values = torch.Tensor(discriminative_values)
    return {
        "instability": instability_values,
        "invalidity": invalidity_values,
        "dissimilarity": dissimilarity_values,
        "dissparsity": dissparsity_values,
        "implausability": implausability_values,
        "discriminative_power": discriminative_values,
    }

def load_epistemic_model(input_shape, output_shape, dataset_name):
    base_ensemble = [
        MLP_Classifier(
            input_shape=input_shape,
            n_classes=output_shape,
            n_layers=3,
            num_neurons=100,
            dropout_prob=0,
            batch_norm=True,
            random_state=i,
        )
        for i in range(ENSEMBLE_MEMBER_COUNT)
    ]
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    ensemble_model.load(f"models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")
    ensemble_model.compile(backend="inductor")
    ensemble_model.eval()
    return ensemble_model

def get_point_of_interest_and_closest(X_test, y_test, num_points=100):
    points_of_interest = np.random.choice(
            len(X_test[y_test != DESIRED_CLASS]),
            size=min(100, len(X_test[y_test != DESIRED_CLASS])),
            replace=False,
    )
    points_closest_to_interest = []
    for i in points_of_interest:
        # extract a point closest to this point
        idx = np.linalg.norm(
                X_test[y_test != DESIRED_CLASS] - X_test[i], axis=1
        ).argmin()
        points_closest_to_interest.append(idx)
    points_closest_to_interest = np.array(points_closest_to_interest)
    X_points = torch.Tensor(X_test[points_of_interest])
    y_points = torch.Tensor(y_test[points_of_interest])
    X_points_close = torch.Tensor(X_test[points_closest_to_interest])
    y_points_close = torch.Tensor(y_test[points_closest_to_interest])
    return X_points, X_points_close, y_points, y_points_close

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    data_files = [
        "adult",
        "bank",
        "churn",
        "compas",
        "diabetes",
        "fico",
        "german",
        "home",
        "titanic",
        "breast_cancer",
        "boston_housing",
    ]
    # Restructure to collect rows for each method
    metric_results = []
    
    for dataset_name in data_files:
        X_train, X_test, y_train, y_test = get_dataset(dataset_name)
        orginal_X = torch.Tensor(np.concatenate([X_train, X_test], axis=0))
        categorical_features_all = get_categorical_features_all(dataset_name)
        X_points, X_points_close, y_points, y_points_close = get_point_of_interest_and_closest(X_test, y_test)

        ensemble_model = load_epistemic_model(X_points.shape[1], len(np.unique(y_test)), dataset_name)

        for property_name in PROPERTY_NAMES:
            saved_cfs = load_property_data(dataset_name, property_name)
            print(f"Dataset: {dataset_name}, Property: {property_name}")
            counter_factual = torch.Tensor(saved_cfs["counter_factual"])
            counter_factual_closest = torch.Tensor(saved_cfs["counter_factual_closest"])
            metrics = get_metrics(X_points, X_points_close, y_points, y_points_close, counter_factual, counter_factual_closest, X_test, y_test, ensemble_model, orginal_X, categorical_features_all)
            
            # Create a row for this method/dataset combination
            row = {
                "dataset": dataset_name,
                "method": property_name,
                "instability": metrics["instability"].mean().item(),
                "invalidity": metrics["invalidity"].mean().item(),
                "dissimilarity": metrics["dissimilarity"].mean().item(),
                "dissparsity": metrics["dissparsity"].mean().item(),
                "implausability": metrics["implausability"].mean().item(),
                "discriminative_power": metrics["discriminative_power"].mean().item(),
            }
            metric_results.append(row)
        
        for method_name in ["CLUE","DICE","FACE","GS"]:
            carla_data = load_carla_data(dataset_name, method_name)
            if carla_data is None:
                print(f"Dataset: {dataset_name}, Method: {method_name} not found")
                continue
            counter_factual = torch.Tensor(carla_data["counter_factual"])
            counter_factual_closest = torch.Tensor(carla_data["counter_factual_closest"])
            print(f"Dataset: {dataset_name}, Method: {method_name}")
            metrics = get_metrics(X_points, X_points_close, y_points, y_points_close, counter_factual, counter_factual_closest, X_test, y_test, ensemble_model, orginal_X, categorical_features_all)
            
            # Create a row for this method/dataset combination
            row = {
                "dataset": dataset_name,
                "method": method_name,
                "instability": metrics["instability"].mean().item(),
                "invalidity": metrics["invalidity"].mean().item(),
                "dissimilarity": metrics["dissimilarity"].mean().item(),
                "dissparsity": metrics["dissparsity"].mean().item(),
                "implausability": metrics["implausability"].mean().item(),
                "discriminative_power": metrics["discriminative_power"].mean().item(),
            }
            metric_results.append(row)
            
    # Create DataFrame with proper structure
    data = pd.DataFrame(metric_results)
    print(data)
    data.to_csv("computed_metrics.csv", index=False)
