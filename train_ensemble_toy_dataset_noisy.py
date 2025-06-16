from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from data import load_noisy_datasets

ENSEMBLE_MEMBER_COUNT = 20
base_ensemble = [
    MLP_Classifier(
        input_shape=10,
        n_classes=2,
        n_layers=2,
        num_neurons=64,
        dropout_prob=0,
        batch_norm=False,
    )
    for _ in range(ENSEMBLE_MEMBER_COUNT)
]
ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)

DATASET_LOADERS = [
    ("bubbles", load_noisy_datasets, {"n_samples": 1000}),
    ("l_dataset", load_noisy_datasets, {"n_samples": 333}),
    ("one_moon", load_noisy_datasets, {"n_samples": 1000}),
    (
        "ring_dataset",
        load_noisy_datasets,
        {"n_samples": 1000, "inner_radius": 1.0, "outer_radius": 2.0, "noise": 0.1},
    ),
    ("bubbles_noisy", load_noisy_datasets, {"n_samples": 1000}),
    ("two_moon", load_noisy_datasets, {"n_samples": 1000}),
    ("infinity_dataset", load_noisy_datasets, {"n_samples": 1000}),
]

for dataset_name, loader, kwargs in DATASET_LOADERS:
    print(f"Training ensemble on {dataset_name}...")
    ensemble_model = Ensemble_Classifier(base_ensemble, n_models=ENSEMBLE_MEMBER_COUNT)
    points, y_labels, y_probs = loader(dataset_name, **kwargs)

    # Train the ensemble model
    ensemble_model.fit(points, y_labels, n_epochs=50, batch_size=64)

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
    ensemble_model.save(f"models/Ensemble_{dataset_name.capitalize()}_extended/")
