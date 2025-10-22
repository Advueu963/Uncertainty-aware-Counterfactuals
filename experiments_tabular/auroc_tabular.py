import argparse
import numpy as np

from probly.quantification.classification import mutual_information
from probly.representation import Ensemble
from probly.calibration import Temperature
from probly.tasks import out_of_distribution_detection
import torch

from uncertainty_cfs.architectures import MLP
from uncertainty_cfs.property_procedures.utils import (
    predict_probs,
    aleatoric_uncertainty_entropy,
    epistemic_uncertainty_entropy,
)

from uncertainty_cfs.tabular_util import (
    get_dataset,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="deep_ensemble", choices=["deep_ensemble","dare_ensemble","adversarial_ensemble"], help="Type of model to use")
args = parser.parse_args()

if __name__ == "__main__":
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
    for dataset_name in data_files:
        X_train, X_test, y_train, y_test = get_dataset(dataset_name)
        _, X_ood, _, y_ood = get_dataset(dataset_name, severity=2)
        print(f"Dataset: {dataset_name}, y_train classes: {np.unique(y_train)}")
        print(X_ood.shape, X_test.shape)
        assert len(np.unique(y_train)) == len(np.unique(y_ood))
        
        architecture = MLP(
            input_dim=X_train.shape[1],
            output_dim=len(np.unique(y_train)),
            hidden_dims=[100, 100],
            batch_norm=False,
    )
        ### Setup Model ###
        if args.model_name == "deep_ensemble" or args.model_name == "dare_ensemble" or args.model_name == "adversarial_ensemble":
            model = Ensemble(architecture, n_members=20)
        else:
            raise NotImplementedError(f"Model {args.model_name} not implemented in this script.")
        # Evaluate the ensemble model
        model = Temperature(model)
        model.load_state_dict(
                torch.load(f"models/model={args.model_name}_dataset={dataset_name}.pth")
        )
        model.compile()
        model.eval()
        # Get epistemic uncertainty scores
        ood_probs = model.predict_representation(
            torch.tensor(X_ood, dtype=torch.float32)
        )
        test_probs = model.predict_representation(
            torch.tensor(X_test, dtype=torch.float32)
        )
        ood_epistemic_scores = epistemic_uncertainty_entropy(
            ood_probs
        ).detach().numpy()
        test_epistemic_scores = epistemic_uncertainty_entropy(
            test_probs
        ).detach().numpy()
        # Compute AUROC for OOD detection
        auroc = out_of_distribution_detection(
            in_distribution=test_epistemic_scores,
            out_distribution=ood_epistemic_scores,
        )
        print(f"AUROC for {dataset_name}: {auroc:.4f}")
        print("-" * 40)
