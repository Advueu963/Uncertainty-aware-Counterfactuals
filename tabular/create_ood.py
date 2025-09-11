import pandas as pd
import numpy as np
import os

datasets_to_class_column = {
    "titanic": "Survived",
    "compas-scores-two-years": "class",
    "german_credit": "default",
    "fico": "RiskPerformance",
    "churn": "churn",
    "adult": "class",
    "home": "in_sf",
    "bank": "give_credit",
    "ctg": "CLASS",
    "diabetes": "Outcome",
    "boston_housing_clas": "HousingClas",
    "breast_cancer": "diagnosis",
}


def create_ood_datasets(dataset, target_col, severity=1):
    random_number_generator = np.random.default_rng(seed=42)
    ood_dataset = dataset.copy()
    numerical_columns = dataset.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if col == target_col:
            continue
        # Introduce random noise to the numerical columns
        noise = random_number_generator.normal(
            loc=0, scale=severity, size=ood_dataset[col].shape
        )
        ood_dataset[col] += noise
    categorical_columns = dataset.select_dtypes(include=["object", "category"]).columns
    # TODO: Might also be interesting to replace the category with another category
    for col in categorical_columns:
        if col == target_col:
            continue
        # Randomly shuffle the categories to create OOD samples
        random_number_generator.shuffle(ood_dataset[col].values)

    return ood_dataset


if __name__ == "__main__":
    import sys

    severity = int(sys.argv[1]) if len(sys.argv) > 1 else 1
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
        "breast_cancer.csv",
        "boston_housing_clas.csv",
    ]
    for data_file in data_files:
        print(f"Processing {data_file} with severity {severity}")
        target_col = datasets_to_class_column.get(data_file[:-4], None)
        dataset = pd.read_csv(data_file)
        #        dataset.drop("Unnamed: 0", axis=1, inplace=True, errors='ignore')
        ood_dataset = create_ood_datasets(dataset, target_col, severity=severity)
        ood_dataset.to_csv(f"severity_{severity}/ood_{data_file[:-4]}.csv", index=False)
