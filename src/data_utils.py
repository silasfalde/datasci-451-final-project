import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import CONFIG


def load_joined_data():
    """Load and join features and labels."""
    features = pd.read_csv(CONFIG["paths"]["raw_features"])
    labels = pd.read_csv(CONFIG["paths"]["raw_labels"])

    assert len(features) == len(labels), "Feature and label counts mismatch"
    assert features[CONFIG["id_col"]].is_unique, "Duplicate IDs in features"
    assert labels[CONFIG["id_col"]].is_unique, "Duplicate IDs in labels"

    joined = features.merge(labels, on=CONFIG["id_col"])
    assert len(joined) == len(features), "Join lost rows"

    return joined


def standardize_missing(df):
    """Convert empty strings to NaN and factor categorical columns."""
    df = df.copy()
    df = df.replace("", np.nan)

    # Convert character columns to categorical
    for col in df.select_dtypes(include=["object"]).columns:
        if col != CONFIG["id_col"]:
            df[col] = df[col].astype("category")

    return df


def add_strata_key(df, outcomes):
    """Add stratification key combining both outcomes."""
    df = df.copy()
    df["_strata_key"] = (
        df[outcomes[0]].astype(str) + "_" + df[outcomes[1]].astype(str)
    )
    return df


def make_folds(df, n_folds=3, seed=451):
    """Create stratified k-fold split on combined outcome labels."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for train_idx, test_idx in skf.split(df, df["_strata_key"]):
        folds.append({"train": train_idx, "test": test_idx})

    return folds


def preprocess_for_baseline(train_df, test_df, outcome):
    """Prepare data for logistic/RF models with imputation and encoding."""
    train = train_df.copy()
    test = test_df.copy()

    # Numerical imputation (median)
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in [CONFIG["id_col"], outcome, "_strata_key"]:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

    # Categorical imputation (mode)
    cat_cols = train.select_dtypes(include=["category", "object"]).columns
    for col in cat_cols:
        if col not in [CONFIG["id_col"], "_strata_key"]:
            mode_val = train[col].mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
                train[col] = train[col].fillna(mode_val)
                test[col] = test[col].fillna(mode_val)

    # One-hot encoding for categorical
    cat_cols = train.select_dtypes(include=["category"]).columns
    if len(cat_cols) > 0:
        train = pd.get_dummies(train, columns=cat_cols, drop_first=False)
        test = pd.get_dummies(test, columns=cat_cols, drop_first=False)

        # Align columns
        missing_cols = set(train.columns) - set(test.columns)
        for col in missing_cols:
            test[col] = 0

        test = test[train.columns]

    return train, test


def preprocess_for_bayesian(train_df, test_df, outcome):
    """Prepare key predictors for Bayesian model."""
    key_predictors = [
        "doctor_recc_h1n1",
        "doctor_recc_seasonal",
        "h1n1_concern",
        "opinion_h1n1_vacc_effective",
        "opinion_h1n1_risk",
        "opinion_seas_vacc_effective",
        "opinion_seas_risk",
        "health_worker",
        "health_insurance",
        "age_group",
        "education",
        "race",
        "sex",
        CONFIG["group_col"],
    ]

    cols_to_keep = [CONFIG["id_col"], outcome] + key_predictors
    cols_to_keep = [c for c in cols_to_keep if c in train_df.columns]

    train = train_df[cols_to_keep].copy()
    test = test_df[cols_to_keep].copy()

    # Fill missing
    for col in train.columns:
        if train[col].dtype == "object" or train[col].dtype.name == "category":
            mode_val = train[col].mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
                train[col] = train[col].fillna(mode_val)
                test[col] = test[col].fillna(mode_val)
        else:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

    return train, test


def get_X_y(df, outcome, id_cols_to_drop=None):
    """Extract features and target."""
    if id_cols_to_drop is None:
        id_cols_to_drop = [CONFIG["id_col"], "_strata_key"]

    cols_to_drop = [c for c in id_cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop + [outcome])
    y = df[outcome]

    return X, y
