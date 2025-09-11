import pandas as pd

data = pd.read_csv("HousingData.csv")
median_medv = data["MEDV"].median()
print(f"Median MEDV: {median_medv}")
data["MEDV"] = (data["MEDV"] > median_medv).astype(
    int
)  # this value was choosen according to Schut et al 2021 https://github.com/oscarkey/explanations-by-minimizing-uncertainty/blob/master/uces/tabular.py
data = data.rename(columns={"MEDV": "HousingClas"})
data.to_csv("boston_housing_clas.csv", index=False)
print("Converted HousingReg to HousingClas and saved to HousingClas.csv")
