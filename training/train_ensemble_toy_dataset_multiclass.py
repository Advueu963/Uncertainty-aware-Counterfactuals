from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from data import (
    load_bubbles_multiclass,
    load_l_dataset_multiclass,
    load_four_moon,
    load_ring_dataset_multiclass,
)

ENSEMBLE_MEMBER_COUNT = 20
N_EPOCHS = 100
BATCH_SIZE = 128
base_ensemble = [
    MLP_Classifier(
        input_shape=2,
        n_classes=4,
        n_layers=2,
        num_neurons=64,
        dropout_prob=0,
        batch_norm=False,
    )
    for _ in range(ENSEMBLE_MEMBER_COUNT)
]
ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)

DATASET_LOADERS = [
    ("bubbles_multiclass", load_bubbles_multiclass, {"n_samples": 1000}),
    ("l_dataset_multiclass", load_l_dataset_multiclass, {"n_samples": 333}),
    ("four_moon", load_four_moon, {"n_samples": 1000}),
    (
        "ring_dataset_multiclass",
        load_ring_dataset_multiclass,
        {"n_samples": 1000, "noise": 0.1},
    ),
]

for dataset_name, loader, kwargs in DATASET_LOADERS:
    print(f"Training ensemble on {dataset_name}...")
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    points, y_labels, y_probs = loader(**kwargs)

    # Train the ensemble model
    ensemble_model.fit(points, y_labels, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)

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
    # Save the model if needed
    ensemble_model.save(f"../models/Ensemble_{dataset_name.capitalize()}_{N_EPOCHS}/")
