import numpy as np
from sklearn.metrics import roc_auc_score

from epiuc.utils.general import to_numpy
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier

from training.train_ensemble_tabular import get_dataset

ENSEMBLE_MEMBER_COUNT = 20
N_EPOCHS = 50
BATCH_SIZE = 128


def get_auroc_acc(indistribution_data, ood_data, model):
    _, unc_clean = model.predict(indistribution_data)
    prob_dirty, unc_dirty = model.predict(ood_data)

    prob_dirty = to_numpy(prob_dirty.detach().cpu())
    unc_clean = to_numpy(unc_clean.detach().cpu())
    unc_dirty = to_numpy(unc_dirty.detach().cpu())
    print("Dirty Unc: ", unc_dirty.max(), unc_dirty.min())
    print("Clean Unc: ", unc_clean.max(), unc_clean.min())

    uncertainty_scores = np.hstack((unc_clean.flatten(), unc_dirty.flatten()))

    true_predictions = np.hstack(
        (
            np.zeros(len(indistribution_data)),
            np.ones(len(ood_data)),
        )
    )
    # compute AUROC

    rocs_aucs = roc_auc_score(true_predictions, uncertainty_scores)

    return rocs_aucs


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
        _, X_ood, _, y_ood = get_dataset(dataset_name, severity=2)
        print(f"Dataset: {dataset_name}, y_train classes: {np.unique(y_train)}")
        print(X_ood.shape, X_test.shape)
        assert len(np.unique(y_train)) == len(np.unique(y_ood))
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
        ensemble_model.load(f"models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")
        ensemble_model.eval()
        auroc = get_auroc_acc(X_test, X_ood, ensemble_model)
        print(f"AUROC for {dataset_name}: {auroc:.4f}")
        print("-" * 40)
