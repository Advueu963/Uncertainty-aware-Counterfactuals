## Orginal file from: https://github.com/riccotti/Scamander
## Modifications: Removed unused dataset loading functions. Added some more helper functions.


import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


datasets = {
    "adult": ("adult.csv", "tab"),
    "bank": ("bank.csv", "tab"),
    "churn": ("churn.csv", "tab"),
    "compas": ("compas-scores-two-years.csv", "tab"),
    "fico": ("fico.csv", "tab"),
    "diabetes": ("diabetes.csv", "tab"),
    "german": ("german_credit.csv", "tab"),
    "titanic": ("titanic.csv", "tab"),
    "home": ("home.csv", "tab"),
    "boston_housing": ("boston_housing_clas.csv", "tab"),
    "breast_cancer": ("breast_cancer.csv", "tab"),
}
datasets_to_class_column = {
    "titantic": "Survived",
    "compas": "class",
    "german": "default",
    "fico": "RiskPerformance",
    "churn": "churn",
    "adult": "class",
    "home": "in_sf",
    "bank": "give_credit",
    "ctg": "CLASS",
    "diabetes": "Outcome",
    "boston_housing": "HousingClas",
    "breast_cancer": "diagnosis",
}


def remove_missing_values(df):
    for column_name, nbr_missing in df.isna().sum().to_dict().items():
        if nbr_missing > 0:
            if column_name in df._get_numeric_data().columns:
                mean = df[column_name].mean()
                df[column_name] = df[column_name].fillna(mean)
            else:
                mode = df[column_name].mode().values[0]
                df[column_name] = df[column_name].fillna(mode)
    return df


def one_hot_encoding(df, class_name):
    """
    Return one hot encoding of the dataframe passed
    :param df: input dataframe
    :param class_name: target class name
    :return: one hot encoded dataframe
    """

    dfX = pd.get_dummies(
        df[[c for c in df.columns if c not in [class_name]]], prefix_sep="="
    )
    class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
    dfY = df[class_name].map(class_name_map)
    df = pd.concat([dfX, dfY], axis=1)  # , join_axes=[dfX.index])
    feature_names = list(dfX.columns)
    class_values = sorted(class_name_map)

    return df, feature_names, class_values


def get_categorical_features_lists(feature_names):
    i = 0
    categorical_features_names = list()
    categorical_features_lists = list()
    while i < len(feature_names) - 1:
        if "=" not in feature_names[i]:
            i += 1
            continue
        fn0, val0 = feature_names[i].split("=")[:2]
        values = [val0]
        indexes = [i]
        for j in range(i + 1, len(feature_names)):
            fn1, val1 = feature_names[j].split("=")[:2]
            i = j
            if fn1 == fn0:
                values.append(val1)
                indexes.append(j)
                # i = j
            else:
                # i = j
                break
            # i = j
        categorical_features_names.append([fn0, values])
        categorical_features_lists.append(indexes)
        # i = j
    return categorical_features_lists


def get_titanic_dataset(filename):
    class_name = "Survived"
    df = pd.read_csv(filename, sep=",", skipinitialspace=True)
    df["Family"] = df["SibSp"] + df["Parch"]
    df.drop(
        ["PassengerId", "Name", "Cabin", "Ticket", "SibSp", "Parch"],
        axis=1,
        inplace=True,
    )
    df["Age"] = df["Age"].fillna(np.mean(df["Age"]))
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df, class_name


def get_compas_dataset(filename):
    class_name = "class"
    df = pd.read_csv(filename, sep=",", skipinitialspace=True)
    columns = [
        "age",  # 'age_cat',
        "sex",
        "race",
        "priors_count",
        "days_b_screening_arrest",
        "c_jail_in",
        "c_jail_out",
        "c_charge_degree",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
        "decile_score",
        "score_text",
    ]

    df = df[columns]
    df["days_b_screening_arrest"] = np.abs(df["days_b_screening_arrest"])
    df["c_jail_out"] = pd.to_datetime(df["c_jail_out"])
    df["c_jail_in"] = pd.to_datetime(df["c_jail_in"])
    df["length_of_stay"] = (df["c_jail_out"] - df["c_jail_in"]).dt.days
    df["length_of_stay"] = np.abs(df["length_of_stay"])
    df["length_of_stay"] = df["length_of_stay"].fillna(
        df["length_of_stay"].value_counts().index[0]
    )
    df["days_b_screening_arrest"] = df["days_b_screening_arrest"].fillna(
        df["days_b_screening_arrest"].value_counts().index[0]
    )
    df["length_of_stay"] = df["length_of_stay"].astype(int)
    df["days_b_screening_arrest"] = df["days_b_screening_arrest"].astype(int)
    df["class"] = df["score_text"]
    df.drop(
        ["c_jail_in", "c_jail_out", "decile_score", "score_text"], axis=1, inplace=True
    )

    return df, class_name


def get_adult_dataset(filename):
    class_name = "class"
    df = pd.read_csv(
        filename, sep=",", skipinitialspace=True, na_values="?", keep_default_na=True
    )
    df.drop(["fnlwgt", "education-num"], axis=1, inplace=True)
    return df, class_name


def get_home_dataset(filename):
    class_name = "in_sf"
    df = pd.read_csv(
        filename, sep=",", skipinitialspace=True, na_values="?", keep_default_na=True
    )
    return df, class_name


def get_german_dataset(filename):
    class_name = "default"
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = [c.replace("=", "") for c in df.columns]
    return df, class_name


def get_fico_dataset(filename):
    class_name = "RiskPerformance"
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True)
    return df, class_name


def get_churn_dataset(filename):
    class_name = "churn"
    df = pd.read_csv(
        filename, skipinitialspace=True, na_values="?", keep_default_na=True
    )
    columns2remove = ["phone number"]
    df.drop(columns2remove, inplace=True, axis=1)
    return df, class_name


def get_bank_dataset(filename):
    class_name = "give_credit"
    df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True, index_col=0)
    return df, class_name


def get_ctg_dataset(filename):
    class_name = "CLASS"
    df = pd.read_csv(filename, sep=",", skipinitialspace=True)
    df.drop(["FileName", "Date", "SegFile", "NSP"], axis=1, inplace=True)
    return df, class_name


def get_diabetes_dataset(filename):
    class_name = "Outcome"
    df = pd.read_csv(filename, sep=",", skipinitialspace=True)
    return df, class_name


def get_boston_housing(filename):
    class_name = "HousingClas"
    df = pd.read_csv(filename, sep=",", skipinitialspace=True)
    return df, class_name


def get_breast_cancer_dataset(filename):
    class_name = "diagnosis"
    df = pd.read_csv(filename, sep=",", skipinitialspace=True)
    df.drop(["id"], axis=1, inplace=True)
    return df, class_name


dataset_read_function_map = {
    "adult": get_adult_dataset,
    "bank": get_bank_dataset,
    "churn": get_churn_dataset,
    "compas": get_compas_dataset,
    "diabetes": get_diabetes_dataset,
    "german_credit": get_german_dataset,
    "fico": get_fico_dataset,
    "german": get_german_dataset,
    "home": get_home_dataset,
    "titanic": get_titanic_dataset,
    "breast_cancer": get_breast_cancer_dataset,
    "boston_housing": get_boston_housing,
}


def get_tabular_dataset(
    name,
    path="./tabular/",
    normalize=None,
    test_size=0.3,
    random_state=None,
    return_dataframe=False,
    encode="onehot",
    severity=-1,
):
    get_dataset_fn = dataset_read_function_map[name]

    if 1 <= severity <= 5:
        filename = path + f"severity_{int(severity)}/" + "ood_" + datasets[name][0]
    else:
        filename = path + datasets[name][0]
    data_type = datasets[name][1]

    df, class_name = get_dataset_fn(filename)

    

    df = remove_missing_values(df)
    features = [c for c in df.columns if c not in [class_name]]
    continuous_features_names = list(df[features]._get_numeric_data().columns)
    categorical_features_names = [
        c for c in df.columns if c not in continuous_features_names and c != class_name
    ]
    n_rows = len(df)
    n_cols = len(features)
    n_cont_cols = len(continuous_features_names)
    n_cate_cols = n_cols - n_cont_cols
    n_classes = len(df[class_name].unique())
    


    if encode == "onehot":
        df, feature_names, class_values = one_hot_encoding(df, class_name)
        categorical_features_lists_all = get_categorical_features_lists(feature_names)
        variable_features_names = dataset_variable_features_names[name]
        variable_features = [
            i for i, f in enumerate(feature_names) if f in variable_features_names
        ]
    elif encode in ["none", None]:
        feature_names = continuous_features_names + categorical_features_names
        df = df[feature_names + [class_name]]
        class_name_map = {v: k for k, v in enumerate(sorted(df[class_name].unique()))}
        class_values = sorted(class_name_map)

        categorical_features_lists_all = [
            [cidx]
            for cidx in np.arange(len(continuous_features_names), len(feature_names), 1)
        ]
        variable_features_names = dataset_variable_features_names_noencode[name]
        variable_features = [
            i for i, f in enumerate(feature_names) if f in variable_features_names
        ]
    else:
        raise Exception("Unknown encoding %s" % encode)

    continuous_features = list()
    continuous_features_all = list()
    for i, f in enumerate(feature_names):
        if f in continuous_features_names:
            continuous_features_all.append(i)
            if f in variable_features_names:
                continuous_features.append(i)

    categorical_features = list()
    categorical_features_lists = list()
    categorical_features_all = list()
    for index in categorical_features_lists_all:
        categorical_features_all.extend(index)
        if index[0] in variable_features:
            categorical_features.extend(index)
            categorical_features_lists.append(index)

    n_cols1h = len(feature_names)
    n_var = len(variable_features)
    n_cont_var = len(continuous_features)
    n_cate_var = len(categorical_features_lists)
    n_cate_var1h = n_var - n_cont_var

    if return_dataframe:
        dfo = df.copy()
    else:
        dfo = None
    
    X = df[feature_names].values
    y = df[class_name].values
    indices = np.arange(len(X))

    if normalize == "minmax":
        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)
        # scaler = ColumnTransformer(transformers=[(normalize, StandardScaler(), continuous_features_all)],
        #                            remainder='passthrough')
        # X = scaler.fit_transform(X)
        scaler = MinMaxScaler()
        X[:, continuous_features_all] = scaler.fit_transform(
            X[:, continuous_features_all]
        )
    elif normalize == "standard":
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        # scaler = ColumnTransformer(transformers=[(normalize, StandardScaler(), continuous_features_all)],
        #                            remainder='passthrough')
        # X = scaler.fit_transform(X)
        scaler = StandardScaler()
        X[:, continuous_features_all] = scaler.fit_transform(
            X[:, continuous_features_all]
        )
    else:
        scaler = None

    if encode in ["none", None]:
        import warnings
        warnings.warn("Using None Encoder causes issues with returning the dataframe.")
        encoder = OrdinalEncoder()
        encoder_y = LabelEncoder()
        if len(categorical_features_all):
            X_cat = encoder.fit_transform(X[:, categorical_features_all])
            X_con = X[:, continuous_features_all]
            X = np.hstack([X_con, X_cat]).astype(np.float64)
        y = encoder_y.fit_transform(y)
    else:
        encoder = None
        encoder_y = None

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=test_size, random_state=random_state, stratify=y
    )

    data = {
        "name": name,
        "data_type": data_type,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "idx_train": indices_train,
        "idx_test": indices_test,
        "class_name": class_name,
        "class_values": class_values,
        "feature_names": feature_names,
        "continuous_features_names": continuous_features_names,
        "categorical_features_names": categorical_features_names,
        "n_classes": n_classes,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_cont_cols": n_cont_cols,
        "n_cate_cols": n_cate_cols,
        "n_cols1h": n_cols1h,
        "n_var": n_var,
        "n_cont_var": n_cont_var,
        "n_cate_var": n_cate_var,
        "n_cate_var1h": n_cate_var1h,
        "variable_features": variable_features,
        "variable_features_names": variable_features_names,
        "continuous_features": continuous_features,
        "continuous_features_all": continuous_features_all,
        "categorical_features": categorical_features,
        "categorical_features_all": categorical_features_all,
        "categorical_features_lists": categorical_features_lists,
        "categorical_features_lists_all": categorical_features_lists_all,
        "scaler": scaler,
        "df": dfo,
        "encoder": encoder,
        "encoder_y": encoder_y,
    }

    return data


dataset_variable_features_names = {
    "adult": [
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass=Federal-gov",
        "workclass=Local-gov",
        "workclass=Never-worked",
        "workclass=Private",
        "workclass=Self-emp-inc",
        "workclass=Self-emp-not-inc",
        "workclass=State-gov",
        "workclass=Without-pay",
        "occupation=Adm-clerical",
        "occupation=Armed-Forces",
        "occupation=Craft-repair",
        "occupation=Exec-managerial",
        "occupation=Farming-fishing",
        "occupation=Handlers-cleaners",
        "occupation=Machine-op-inspct",
        "occupation=Other-service",
        "occupation=Priv-house-serv",
        "occupation=Prof-specialty",
        "occupation=Protective-serv",
        "occupation=Sales",
        "occupation=Tech-support",
        "occupation=Transport-moving",
    ],
    "bank": [
        "revolving",
        "nbr_30_59_days_past_due_not_worse",
        "debt_ratio",
        "monthly_income",
        "nbr_open_credits_and_loans",
        "nbr_90_days_late",
        "nbr_real_estate_loans_or_lines",
        "nbr_60_89_days_past_due_not_worse",
        "dependents",
    ],
    "churn": [
        "account length",
        "number vmail messages",
        "total day minutes",
        "total day calls",
        "total day charge",
        "total eve minutes",
        "total eve calls",
        "total eve charge",
        "total night minutes",
        "total night calls",
        "total night charge",
        "total intl minutes",
        "total intl calls",
        "total intl charge",
        "customer service calls",
        "international plan=no",
        "international plan=yes",
        "voice mail plan=no",
        "voice mail plan=yes",
    ],
    "compas": [
        "priors_count",
        "days_b_screening_arrest",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
        "length_of_stay",
        "c_charge_degree=F",
        "c_charge_degree=M",
    ],
    "diabetes": [
        # 'Pregnancies',
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        # 'DiabetesPedigreeFunction',
        # 'Age',
    ],
    "fico": [
        # 'ExternalRiskEstimate',
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MaxDelqEver",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "NetFractionRevolvingBurden",
        "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        "PercentTradesWBalance",
    ],
    "german": [
        "duration_in_month",
        "credit_amount",
        "installment_as_income_perc",
        "present_res_since",
        # 'age',
        "credits_this_bank",
        # 'people_under_maintenance',
        "account_check_status=0 <= ... < 200 DM",
        "account_check_status=< 0 DM",
        "account_check_status=>= 200 DM / salary assignments for at least 1 year",
        "account_check_status=no checking account",
        # 'credit_history=all credits at this bank paid back duly',
        # 'credit_history=critical account/ other credits existing (not at this bank)',
        # 'credit_history=delay in paying off in the past',
        # 'credit_history=existing credits paid back duly till now',
        # 'credit_history=no credits taken/ all credits paid back duly',
        # 'purpose=(vacation - does not exist?)',
        # 'purpose=business',
        # 'purpose=car (new)',
        # 'purpose=car (used)',
        # 'purpose=domestic appliances',
        # 'purpose=education',
        # 'purpose=furniture/equipment',
        # 'purpose=radio/television',
        # 'purpose=repairs',
        # 'purpose=retraining',
        "savings=.. >= 1000 DM ",
        "savings=... < 100 DM",
        "savings=100 <= ... < 500 DM",
        "savings=500 <= ... < 1000 DM ",
        "savings=unknown/ no savings account",
        "present_emp_since=.. >= 7 years",
        "present_emp_since=... < 1 year ",
        "present_emp_since=1 <= ... < 4 years",
        "present_emp_since=4 <= ... < 7 years",
        "present_emp_since=unemployed",
        # 'personal_status_sex=female : divorced/separated/married',
        # 'personal_status_sex=male : divorced/separated',
        # 'personal_status_sex=male : married/widowed',
        # 'personal_status_sex=male : single',
        "other_debtors=co-applicant",
        "other_debtors=guarantor",
        "other_debtors=none",
        "property=if not A121 : building society savings agreement/ life insurance",
        "property=if not A121/A122 : car or other, not in attribute 6",
        "property=real estate",
        "property=unknown / no property",
        "other_installment_plans=bank",
        "other_installment_plans=none",
        "other_installment_plans=stores",
        # 'housing=for free',
        # 'housing=own',
        # 'housing=rent',
        "job=management/ self-employed/ highly qualified employee/ officer",
        "job=skilled employee / official",
        "job=unemployed/ unskilled - non-resident",
        "job=unskilled - resident",
        "telephone=none",
        "telephone=yes, registered under the customers name ",
        # 'foreign_worker=no',
        # 'foreign_worker=yes',
    ],
    "home": [
        "beds",
        "bath",
        "price",
        # 'year_built',
        # 'sqft',
        "price_per_sqft",
        # 'elevation',
    ],
    "titanic": [
        "Pclass",
        # 'Age',
        # 'SibSp',
        # 'Parch',
        "Family",
        "Fare",
        # 'Sex=female',
        # 'Sex=male',
        "Embarked=C",
        "Embarked=Q",
        "Embarked=S",
    ],
    "breast_cancer": [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ],
    "boston_housing": [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ],
}

dataset_variable_features_names_noencode = {
    "adult": [
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass",
        "occupation",
    ],
    "bank": [
        "revolving",
        "nbr_30_59_days_past_due_not_worse",
        "debt_ratio",
        "monthly_income",
        "nbr_open_credits_and_loans",
        "nbr_90_days_late",
        "nbr_real_estate_loans_or_lines",
        "nbr_60_89_days_past_due_not_worse",
        "dependents",
    ],
    "churn": [
        "account length",
        "number vmail messages",
        "total day minutes",
        "total day calls",
        "total day charge",
        "total eve minutes",
        "total eve calls",
        "total eve charge",
        "total night minutes",
        "total night calls",
        "total night charge",
        "total intl minutes",
        "total intl calls",
        "total intl charge",
        "customer service calls",
        "international plan=no",
        "international plan=yes",
        "voice mail plan=no",
        "voice mail plan=yes",
    ],
    "compas": [
        "priors_count",
        "days_b_screening_arrest",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
        "length_of_stay",
        "c_charge_degree=F",
        "c_charge_degree=M",
    ],
    "diabetes": [
        # 'Pregnancies',
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        # 'DiabetesPedigreeFunction',
        # 'Age',
    ],
    "fico": [
        # 'ExternalRiskEstimate',
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MaxDelqEver",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "NetFractionRevolvingBurden",
        "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        "PercentTradesWBalance",
    ],
    "german": [
        "duration_in_month",
        "credit_amount",
        "installment_as_income_perc",
        "present_res_since",
        # 'age',
        "credits_this_bank",
        # 'people_under_maintenance',
        "account_check_status",
        # 'credit_history',
        # 'purpose',
        "savings",
        "present_emp_since",
        # 'personal_status_sex',
        "other_debtors",
        "property",
        "other_installment_plans",
        # 'housing',
        "job",
        "telephone",
        # 'foreign_worker',
    ],
    "home": [
        "beds",
        "bath",
        "price",
        # 'year_built',
        # 'sqft',
        "price_per_sqft",
        # 'elevation',
    ],
    "titanic": [
        "Pclass",
        # 'Age',
        # 'SibSp',
        # 'Parch',
        "Family",
        "Fare",
        # 'Sex',
        "Embarked",
    ],
    "breast_cancer": [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ],
    "boston_housing": [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ],
}


# def highlight_diff(data, color='red', x=None):
#     attr = 'color: %s' % color
#     is_diff = np.abs(df_x.values - data.values) != 0
#     return pd.DataFrame(np.where(is_diff, attr, ''),
#                         index=data.index, columns=data.columns)


def highlight_diff_first(data, color="red"):
    attr = "color: %s" % color
    is_diff = data.values[:, 0].reshape(-1, 1) != data.values
    return pd.DataFrame(
        np.where(is_diff, attr, ""), index=data.index, columns=data.columns
    )


def show_diff(
    x,
    cf_list,
    features_names,
    cont_features_names,
    cate_features_names,
    continuous_features_idx,
    categorical_features_lists_idx,
    color="red",
    scaler=None,
    y=None,
    y_proba=None,
    variable_features_names=None,
):
    df_x = pd.DataFrame(x.reshape(1, -1), columns=features_names)
    df_cf = pd.DataFrame(cf_list, columns=features_names)
    df_a = pd.concat([df_x, df_cf]).reset_index(drop=True)

    cols2remove = list()
    columns = np.array(df_a.columns)
    for cfn, indexes in zip(cate_features_names, categorical_features_lists_idx):
        values = df_a.iloc[:, indexes].idxmax(1).tolist()
        values = np.array([v.replace("%s=" % cfn, "") for v in values])
        df_a[cfn] = values
        cols2remove.extend(columns[indexes])
    df_a.drop(cols2remove, axis=1, inplace=True)
    if scaler:
        data = df_a.values
        data[:, continuous_features_idx] = scaler.inverse_transform(
            data[:, continuous_features_idx].astype(np.float64)
        )
        df_a = pd.DataFrame(data=data, columns=df_a.columns)

    feat_names = cont_features_names + cate_features_names
    if variable_features_names:
        variable_features_names_real = set(
            [c.split("=")[0] for c in variable_features_names]
        )
        feat_names = [c for c in feat_names if c in variable_features_names_real]

    if y is not None:
        df_a["class"] = y
        feat_names += ["class"]

    if y_proba is not None:
        df_a["prob"] = y_proba
        feat_names += ["prob"]

    df_a_T = df_a[feat_names].T
    df_a_T.columns = ["x"] + ["cf_%i" % i for i in range(len(cf_list))]
    return df_a_T.style.apply(highlight_diff_first, color=color, axis=None)


# da rivedere
# def show_diff1h(x, cf_list, variable_features, variable_features_names, continuous_features_all,
#                 color='red', scaler=None):
#     df_x = pd.DataFrame(x.reshape(1,-1)[:, variable_features], columns=variable_features_names)
#     df_cf = pd.DataFrame(cf_list[:, variable_features], columns=variable_features_names)
#     df_a = pd.concat([df_x, df_cf]).reset_index(drop=True)
#     if scaler:
#         data = df_a.values
#         data[:,continuous_features_all] = scaler.inverse_transform(data[:,continuous_features_all].astype(np.float64))
#         df_a = pd.DataFrame(data=data, columns=df_a.columns)
#     df_a_T = df_a.T
#     df_a_T.columns = ['x'] + ['cf_%i' % i for i in range(len(cf_list))]
#     return df_a_T.style.apply(highlight_diff_first, color=color, axis=None)


def get_dict_record(
    x, features_names, continuous_features_all, categorical_features_lists_all, scaler
):
    x_dict = {}

    values = scaler.inverse_transform(
        x[continuous_features_all].reshape(1, -1).astype(np.float64)
    )[0]
    for a, v in zip(continuous_features_all, values):
        x_dict[features_names[a]] = v

    for indexes in categorical_features_lists_all:
        val = np.argmax(x[indexes])
        a = features_names[indexes[val]].split("=")[0]
        v = features_names[indexes[val]].split("=")[1]
        x_dict[a] = v

    return x_dict



def get_dataset(name, severity=-1):
    dataset = get_tabular_dataset(name, severity=severity, random_state=42)
    X_train, X_test, y_train, y_test = (
        dataset["X_train"].astype(np.float32),
        dataset["X_test"].astype(np.float32),
        dataset["y_train"],
        dataset["y_test"],
    )
    return X_train, X_test, y_train, y_test

def get_categorical_feature_lists(name, severity=-1):
    dataset = get_tabular_dataset(name, severity=severity, random_state=42)
    return dataset["categorical_features_lists"]

def get_categorical_features_all(name, severity=-1):
    dataset = get_tabular_dataset(name, severity=severity, random_state=42)
    return dataset["categorical_features_all"]

def get_numerical_feature_idx(name, severity=-1):
    dataset = get_tabular_dataset(name, severity=severity, random_state=42)
    return dataset["continuous_features"]

def get_immutable_feature_idx(name, severity=-1):
    dataset = get_tabular_dataset(name, severity=severity, random_state=42)
    all_features = list(range(dataset["X_train"].shape[1]))
    immutable_features = [
        idx for idx in all_features if idx not in dataset["variable_features"]
    ]
    return immutable_features
