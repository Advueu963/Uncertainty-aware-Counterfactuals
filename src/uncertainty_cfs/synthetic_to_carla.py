import numpy as np

from carla import Data
from .data import (
    load_datasets,
)
import pandas as pd
from carla import MLModel
import torch.nn as nn
from .tabular_util import get_tabular_dataset


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


class Tabular_CARLA(Data):
    def __init__(self, dataset_name, **kwargs):
        data = get_tabular_dataset(
            dataset_name, return_dataframe=True, random_state=42, **kwargs
        )
        self.features = data["feature_names"]
        self._categorical_features_changeable = list(
            np.array(data["feature_names"])[data["categorical_features"]]
        )  # CARLA enforces list format. Used numpy array for slice indexing
        self._continuous_features_changeable = list(
            np.array(data["feature_names"])[data["continuous_features"]]
        )  # CARLA enforces list format. Used numpy array for slice indexing
        self.immutable_features = [
            f
            for f in self.features
            if f
            not in self._categorical_features_changeable
            + self._continuous_features_changeable
        ]
        self.categorical_features = list(
            np.array(data["feature_names"])[data["categorical_features_all"]]
        )  # CARLA enforces list format. Used numpy array for slice indexing
        self.continuous_features = list(
            np.array(data["feature_names"])[data["continuous_features_all"]]
        )  # CARLA enforces list format. Used numpy array for slice indexing
        self._dataset = data["df"].astype(np.float32)
        self._dataset_train = self._dataset.iloc[data["idx_train"]]
        self._dataset_test = self._dataset.iloc[data["idx_test"]]
        self._target = data["class_name"]
        self.name = dataset_name

    # Feature name of the target column
    @property
    def target(self):
        return self._target

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

    # List of all categorical features
    @property
    def categorical(self):
        return self.categorical_features

    # List of all continuous features
    @property
    def continuous(self):
        return self.continuous_features

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return self.immutable_features


class Tabular_Carla_Model(MLModel):
    def __init__(
        self,
        data: Tabular_CARLA,
        model: nn.Module,
        output_shape: int,
        input_order: list = None,
    ):
        super().__init__(data)

        self.output_shape = output_shape
        self.input_order = input_order if input_order is not None else data.features
        self._mymodel = model
        self.input_order = input_order

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.input_order

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
        y_pred = self._mymodel.predict(x)
        return y_pred.detach().numpy()

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        # Check if x contains nan, if so return nan array
        if isinstance(x, pd.DataFrame):
            if x.isnull().values.any():
                return np.full((x.shape[0], self.output_shape), np.nan)
        elif isinstance(x, np.ndarray):
            if np.isnan(x).any():
                return np.full((x.shape[0], self.output_shape), np.nan)
        # Ensure x is a tensor if needed
        if isinstance(x, pd.DataFrame):
            if self._data.target in x.columns:
                x = x.drop(columns=[self._data.target])
            x = x.values
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        y_prob = self._mymodel.predict_proba(x)
        return y_prob.detach().numpy()


class MyOwnModel(MLModel):
    def __init__(
        self,
        data: Synthetic_CARLA,
        model: nn.Module,
        noisy: bool = False,
    ):
        super().__init__(data)
        # The constructor can be used to load or build an
        self.noisy = noisy
        self._mymodel = model

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
