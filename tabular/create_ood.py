import pandas as pd
import numpy as np
import os


def create_ood_datasets(dataset, severity=1):
    random_number_generator = np.random.default_rng(seed=42)
    ood_dataset = dataset.copy()
    numerical_columns = dataset.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        # Introduce random noise to the numerical columns
        noise = random_number_generator.normal(
            loc=0, scale=severity, size=ood_dataset[col].shape
        )
        ood_dataset[col] += noise
    categorical_columns = dataset.select_dtypes(include=["object", "category"]).columns
    # TODO: Might also be interesting to replace the category with another category
    for col in categorical_columns:
        # Randomly shuffle the categories to create OOD samples
        random_number_generator.shuffle(ood_dataset[col].values)

    return ood_dataset


if __name__ == "__main__":
    severity = 2
    os.makedirs(f"severity_{severity}", exist_ok=True)
    data_files = [
        "adult.csv",
        "bank.csv",
        "churn.csv",
        "compas-scores-two-years.csv",
        "diabetes.csv",
        "fico.csv",
        "german_credit.csv",
        "home.csv",
        "titanic.csv",
    ]
    for data_file in data_files:
        dataset = pd.read_csv(data_file)
        ood_dataset = create_ood_datasets(dataset, severity=severity)
        ood_dataset.to_csv(f"severity_{severity}/ood_{data_file[:-4]}.csv", index=False)
