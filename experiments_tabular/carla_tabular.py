import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
from carla.recourse_methods import GrowingSpheres, Clue, Dice, Face
from uncertainty_cfs.architectures import MLP
from probly.representation import Ensemble, Dropout, Bayesian
from probly.calibration import Temperature

from uncertainty_cfs.tabular_util import get_tabular_dataset
from uncertainty_cfs.synthetic_to_carla import Tabular_CARLA, Tabular_Carla_Model
DESIRED_CLASS = 1
SAVE_FOLDER = os.environ.get("SCRATCH_DSS")

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="GS", choices=["GS", "CLUE", "DICE", "FACE"], help="Type of recourse method to use")
parser.add_argument("--model_name", type=str, default="deep_ensemble", choices=["deep_ensemble","dare_ensemble","adversarial_ensemble", "bayesian", "dropout"], help="Type of model to use")
args = parser.parse_args()


def get_point_of_interest_and_closest(X_test, y_test, rng):
    points_of_interest = rng.choice(
        len(X_test[y_test != DESIRED_CLASS]),
        size=min(100, len(X_test[y_test != DESIRED_CLASS])),
        replace=False,
    )
    #print("POI INDICES: ", points_of_interest)
    points_closest_to_interest = []
    for i in points_of_interest:
        # extract a point closest to this point, which is of the other class and not the same point
        distances = np.linalg.norm(
            X_test[y_test != DESIRED_CLASS] - (X_test[y_test != DESIRED_CLASS][i]).reshape(1, -1), axis=1
        )
        mask = distances == 0
        distances[mask] = np.inf
        idx = distances.argmin()
        points_closest_to_interest.append(idx)
    points_closest_to_interest = np.array(points_closest_to_interest)
    return points_of_interest, points_closest_to_interest


if __name__ == '__main__':
    #### Setup ####
    MODEL_NAME = args.model_name
    rng = np.random.default_rng(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data_files = [
        "bank",
        "churn",
        "compas",
        "diabetes",
        "fico",
        "home",
        "titanic",
        "breast_cancer",
        "boston_housing",
    ]
    hyper_params = {
        "GS": {},
        "CLUE": {},
        "DICE": {},
        "FACE": {},
    }
    

    
    method_name = args.method
    # Load the dataset and select points to explain
    
    erg = None
    for dataset_name in data_files:
        
        
        
        #### Load CLUE HYPERPARAMS ####
        with open('sweep_configurations_clue.txt', 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            params = line.strip().split(',')
            hyper_params["CLUE"][i] = {
                "data_name": dataset_name,
                "width": int(params[0]),
                "depth": int(params[1]),
                "latent_dim": int(params[2]),
                "batch_size": int(params[3]),
                "epochs": int(params[4]),
                "lr": float(params[5]),
                "early_stop": int(params[6]),
            }
        print("Loaded CLUE hyperparameters: ", hyper_params["CLUE"])
        #### Load FACE HYPERPARAMS ####
        with open('sweep_configurations_face.txt', 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            params = line.strip().split(',')
            hyper_params["FACE"][i] = {
                "mode": params[0],
                "fraction": float(params[1]),
                "radius": float(params[2]),
            }
        print("Loaded FACE hyperparameters: ", hyper_params["FACE"])
        hyper_params_method = hyper_params[method_name]



        print(f"Starting experiment on dataset: {dataset_name} with method {method_name}")
        data = get_tabular_dataset(dataset_name, return_dataframe=True, random_state=42)
        original_df = data["df"]
        X_test, y_test = data["X_test"].astype(np.float32), data["y_test"]
        # Select points not of the desired class
        points_of_interest, points_closest_to_interest = get_point_of_interest_and_closest(X_test, y_test, rng)
        print("Points of interest indices: ", points_of_interest)
        print("Points closest to interest indices: ", points_closest_to_interest)
        
        points = X_test[points_of_interest]
        y_labels = y_test[points_of_interest]
        
        print("MLP INPUT DIM: ", points.shape[1])
        architecture = MLP(
                input_dim=points.shape[1],
                output_dim=len(np.unique(y_labels)),
                hidden_dims=[100, 100],
                batch_norm=False,
        )
        ### Setup Model ###
        if MODEL_NAME == "deep_ensemble" or MODEL_NAME == "dare_ensemble" or MODEL_NAME == "adversarial_ensemble":
            model = Ensemble(architecture, n_members=20)
        elif MODEL_NAME == "bayesian":
            model = Bayesian(architecture)
        elif MODEL_NAME == "dropout":
            model = Dropout(architecture, p=0.2)
        # Evaluate the ensemble model
        model = Temperature(model)
        model.load_state_dict(
                torch.load(f"models/model={MODEL_NAME}_dataset={dataset_name}.pth")
        )
        model.compile()
        model.eval()
        
        # Load the dataset and model for CARLA methods  
        dataset = Tabular_CARLA(dataset_name)
        model = Tabular_Carla_Model(dataset,
            model,
            output_shape=len(np.unique(y_labels)),
            input_order=dataset.features,
        )
        
        
        # Gather points of interest in dataframe format for CARLA
        df_points = original_df.iloc[data["idx_test"][y_test != DESIRED_CLASS][points_of_interest]].astype(np.float32)
        df_points_closest = original_df.iloc[data["idx_test"][y_test != DESIRED_CLASS][points_closest_to_interest]].astype(np.float32)
        
        print("POI POINTS: ", df_points,)
        print("FEATURE INPUT ORDER: ", model.feature_input_order)
        print("CATEGORICAL FEATURES: ", len(dataset.categorical_features))
        print("CONTINUOUS FEATURES: ", len(dataset.continuous_features))
        print("INPUT DIMENSIONS MODEL: ", len(model.feature_input_order))


        # Initialize CARLA methods
        match method_name:    
            case "GS":
                method = GrowingSpheres(model)  
            case "CLUE":
                choosen_hyperparam_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
                print("Chosen hyperparameter index for CLUE: ", choosen_hyperparam_index)
                HYPERPARAMS = hyper_params_method[choosen_hyperparam_index]
                method = Clue(dataset, model, HYPERPARAMS)
            case "DICE":
                method = Dice(model)
            case "FACE":
                choosen_hyperparam_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
                print("Chosen hyperparameter index for FACE: ", choosen_hyperparam_index)
                HYPERPARAMS = hyper_params_method[choosen_hyperparam_index]
                #
                method = Face(model, HYPERPARAMS)
            case _:
                raise ValueError("Method not recognized")

        
        # Run experiments
        print("=" * 40)
        print(f"Point to explain: {df_points}")  
        print(df_points, df_points.columns)
        cfs_poi,cfs_times = method.get_counterfactuals(df_points)
        # Only compute for the unique df_points_closets
        
        
        
        if method_name == "FACE":
            df_points_closest_unique = df_points_closest.drop_duplicates()
            cfs_closest,cfs_times_closest = method.get_counterfactuals(df_points_closest_unique)
            for idx in df_points_closest_unique.index:
                for _ in range(1,len(df_points_closest.index[df_points_closest.index == idx])):
                    cfs_closest = pd.concat([cfs_closest, cfs_closest.iloc[[df_points_closest_unique.index.get_loc(idx)]]], ignore_index=True)
                    cfs_times_closest.append(cfs_times_closest[df_points_closest_unique.index.get_loc(idx)])
        else:
            cfs_closest,cfs_times_closest = method.get_counterfactuals(df_points_closest)

        
        
        
        # Save results
        print("CFs POI: ", cfs_poi)
        print("CFS Times POI: ", cfs_times)
        print("CFs Closest: ", cfs_closest)
        print("CFS Times Closest: ", cfs_times_closest)
        cfs_poi["construction_time"] = cfs_times
        cfs_poi["method"] = method_name
        cfs_poi["point_index"] = df_points.index
        cfs_poi["dataset"] = dataset_name
        
        cfs_closest["construction_time"] = cfs_times_closest
        cfs_closest["method"] = method_name
        cfs_closest["point_index"] = df_points.index
        cfs_closest["dataset"] = dataset_name

        if erg is None:
            erg = pd.concat([cfs_poi, cfs_closest], ignore_index=True)
        else:
            erg = pd.concat([erg, pd.concat([cfs_poi, cfs_closest], ignore_index=True)], ignore_index=True)

    if method_name == "CLUE" or method_name == "FACE":
        choosen_hyperparam_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        method_name = method_name + f"_{choosen_hyperparam_index}"
    print("Final results: ", erg)
    print("Saving results to: ", os.path.join(SAVE_FOLDER, f"CFs_{MODEL_NAME}_{method_name}_all_datasets.csv"))
    erg.to_csv(os.path.join(SAVE_FOLDER, f"CFs_{MODEL_NAME}_{method_name}_all_datasets.csv"), index=False)