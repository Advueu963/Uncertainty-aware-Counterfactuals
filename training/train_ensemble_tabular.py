from property_procedures.tabular_util import get_tabular_dataset
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
import numpy as np
import torch

ENSEMBLE_MEMBER_COUNT = 20
N_EPOCHS = 50
BATCH_SIZE = 128
torch._functorch.config.donated_buffer = False


def get_dataset(name, severity=-1):
    dataset = get_tabular_dataset(name, severity=severity)
    X_train, X_test, y_train, y_test = (
        dataset["X_train"].astype(np.float32),
        dataset["X_test"].astype(np.float32),
        dataset["y_train"],
        dataset["y_test"],
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
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
    for dataset_name in data_files:
        X_train, X_test, y_train, y_test = get_dataset(dataset_name)
        base_ensemble = [
            MLP_Classifier(
                input_shape=X_train.shape[1],
                n_classes=len(np.unique(y_train)),
                n_layers=3,
                num_neurons=100,
                dropout_prob=0,
                batch_norm=True,
                random_state=i,
            )
            for i in range(ENSEMBLE_MEMBER_COUNT)
        ]
        ensemble_model = Ensemble_Classifier(
            base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT
        )
        print(f"Training ensemble on {dataset_name}...")
        # Train the ensemble model
        ensemble_model.fit(X_train, y_train, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
        # Evaluate the ensemble model
        y_probs_ensemble, _ = ensemble_model.predict(X_test, raw_output=True)
        y_target_ensemble = y_probs_ensemble.mean(dim=1).argmax(dim=1)
        # Print some results
        print(
            f"Accuracy: {((y_target_ensemble == y_test).sum() / len(y_test)).item():.4f}"
        )
        print("-" * 40)
        # Save the model if needed
        ensemble_model.save(f"models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")
