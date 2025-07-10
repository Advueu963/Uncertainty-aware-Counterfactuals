import numpy as np

from carla import Data
from data import (
    load_datasets,
)
import pandas as pd
from carla import MLModel
from epiuc.uncertainty.classification import MLP_Classifier
from epiuc.uncertainty.wrapper import Ensemble_Classifier


# Custom data set implementations need to inherit from the Data interface
class Synthetic_CARLA(Data):
    def __init__(self, dataset_name, noisy=False, **kwargs):
        # Load the dataset based on the name
        X, y, y_probs = load_datasets(dataset_name, noisy=noisy, **kwargs)

        if noisy:
            data_set = pd.DataFrame(
                {
                    "x0": X[:, 0],
                    "x1": X[:, 1],
                    **{f"noise_{i}": X[:, i + 2] for i in range(8)},
                    "label": y,
                }
            )
        else:
            data_set = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "label": y})
        self._dataset = data_set
        self._dataset_train = data_set.sample(frac=0.8, random_state=42)
        self._dataset_test = data_set.drop(self._dataset_train.index)
        self.name = dataset_name
        self.noisy = noisy

    # List of all categorical features
    @property
    def categorical(self):
        return []

    # List of all continuous features
    @property
    def continuous(self):
        if self.noisy:
            return ["x0", "x1"] + [f"noise_{i}" for i in range(8)]
        else:
            return ["x0", "x1"]

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return []

    # Feature name of the target column
    @property
    def target(self):
        return "label"

    # The full dataset
    @property
    def df(self):
        return self._dataset

    # The training split of the dataset
    @property
    def df_train(self):
        return self._dataset_train

    # The test split of the dataset
    @property
    def df_test(self):
        return self._dataset_test

    # Data transformation, for example normalization of continuous features
    # and encoding of categorical features
    def transform(self, df):
        # Example transformation: normalize continuous features
        transformed_df = df.copy()
        for col in self.continuous:
            transformed_df[col] = (
                transformed_df[col] - transformed_df[col].mean()
            ) / transformed_df[col].std()
        # Example encoding of categorical features
        for col in self.categorical:
            transformed_df[col] = transformed_df[col].astype("category").cat.codes
        # Ensure immutables are not transformed
        for col in self.immutables:
            if col in transformed_df.columns:
                transformed_df[col] = df[col]
        # Ensure target column is not transformed
        if self.target in transformed_df.columns:
            transformed_df[self.target] = df[self.target]
        # Return the transformed DataFrame
        return transformed_df

    # Inverts transform operation
    def inverse_transform(self, df):
        # Example inverse transformation: denormalize continuous features
        original_df = df.copy()
        for col in self.continuous:
            mean = self._dataset[col].mean()
            std = self._dataset[col].std()
            original_df[col] = df[col] * std + mean
        # Example decoding of categorical features
        for col in self.categorical:
            original_df[col] = (
                original_df[col]
                .astype("category")
                .cat.rename_categories(
                    self._dataset[col].astype("category").cat.categories
                )
            )
        # Ensure immutables are not transformed
        for col in self.immutables:
            if col in original_df.columns:
                original_df[col] = df[col]
        # Ensure target column is not transformed
        if self.target in original_df.columns:
            original_df[self.target] = df[self.target]
        # Return the original DataFrame
        return original_df


class MyOwnModel(MLModel):
    def __init__(
        self,
        data: Synthetic_CARLA,
        n_models=10,
        input_shape=2,
        n_epochs=50,
        noisy=False,
    ):
        super().__init__(data)
        # The constructor can be used to load or build an
        self.noisy = noisy
        base_ensemble = [
            MLP_Classifier(
                input_shape=input_shape + (8 if noisy else 0),
                n_classes=2,
                n_layers=2,
                num_neurons=64,
                dropout_prob=0,
                batch_norm=False,
            )
            for _ in range(n_models)
        ]
        ensemble_model = Ensemble_Classifier(base_ensemble, n_models=n_models)
        if noisy:
            ensemble_model.load(
                f"models/Ensemble_{data.name.capitalize()}_extended_{n_epochs}/"
            )
        else:
            ensemble_model.load(f"models/Ensemble_{data.name.capitalize()}_{n_epochs}/")

        self._mymodel = ensemble_model

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        if self.noisy:
            return ["x0", "x1"] + [f"noise_{i}" for i in range(8)]
        else:
            return ["x0", "x1"]

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        y_prob, _ = self._mymodel.predict(x)
        # Convert probabilities to class labels
        y_pred = y_prob.argmax(axis=1)
        return y_pred.detach().numpy()

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        # Ensure x is a tensor if needed
        if isinstance(x, pd.DataFrame):
            if "label" in x.columns:
                x = x.drop(columns=["label"])
            x = x.values.astype(np.float32)
        y_prob, _ = self._mymodel.predict(x)
        return y_prob.detach().numpy()
