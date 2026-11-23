"""Shared code for the Bank Marketing project.

The notebook and the Streamlit app both import this file, so the data is
always cleaned and prepared in the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET = "y"

# 'duration' is only known AFTER the call, so it leaks the answer (see UCI notes).
LEAKY_COLUMNS = ["duration"]
# These two columns have correlation > 0.9 with euribor3m, so they repeat the same information.
CORRELATED_COLUMNS = ["emp.var.rate", "nr.employed"]
# In these columns 'unknown' means a missing value.
UNKNOWN_AS_MISSING = ["job", "marital", "education", "housing", "loan"]

NUMERIC_FEATURES = [
    "age", "campaign", "previous", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "total_contacts", "was_contacted_before", "days_since_last",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "poutcome", "age_group",
]
# Columns with long right tails. Values above the 99th percentile are capped.
CAP_COLUMNS = ["campaign", "previous", "total_contacts"]


def load_raw(path=DATA_PATH):
    """Read the original UCI file (it uses ';' as separator)."""
    return pd.read_csv(path, sep=";")


def clean(df):
    """Cleaning steps that do not learn anything from the data."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop(columns=LEAKY_COLUMNS + CORRELATED_COLUMNS, errors="ignore")
    for col in UNKNOWN_AS_MISSING:
        df[col] = df[col].replace("unknown", np.nan)
    # Only 18 clients are 'illiterate'. Such a small group is not reliable, so we join it with basic.4y.
    df["education"] = df["education"].replace("illiterate", "basic.4y")
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "yes").astype(int)
    return df.reset_index(drop=True)


def add_features(df):
    """Feature engineering. It also fixes the pdays = 999 problem."""
    df = df.copy()
    # pdays = 999 is a code for 'never contacted', not a real number of days.
    # But 4,110 rows have previous > 0 and pdays = 999 at the same time.
    # 'previous' is more reliable, so we use it to say if the client was contacted before.
    df["was_contacted_before"] = (df["previous"] > 0).astype(int)
    days = df["pdays"].where(df["pdays"] != 999)
    # Clients with no real value stay NaN here. The imputer in the pipeline fills them with -1.
    df["days_since_last"] = days
    df = df.drop(columns=["pdays"])
    df["total_contacts"] = df["campaign"] + df["previous"]
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 50, 65, 120],
        labels=["<=25", "26-35", "36-50", "51-65", "65+"],
    ).astype(str)
    return df


def prepare(df):
    """Full fixed preparation: clean + new features."""
    return add_features(clean(df))


def split_xy(df):
    return df.drop(columns=[TARGET]), df[TARGET]


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Caps very large values at a percentile that is learned on the training data only."""

    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.quantile, axis=0)
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(scale=True, cap_quantile=0.99):
    """ColumnTransformer used inside every model pipeline.

    It is fitted on the training data only, so there is no data leakage.
    """
    other_numeric = [c for c in NUMERIC_FEATURES if c not in CAP_COLUMNS]
    scaler = StandardScaler() if scale else "passthrough"
    capped = Pipeline([
        ("cap", QuantileCapper(cap_quantile) if cap_quantile < 1 else "passthrough"),
        ("scale", scaler),
    ])
    numeric = Pipeline([
        # days_since_last is empty for most clients; we fill it with -1 and the flag tells the model why.
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("capped", capped, CAP_COLUMNS),
        ("num", numeric, other_numeric),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_splits(df, test_size=0.15, val_size=0.15):
    """Stratified 70/15/15 split. The same settings are used in the notebook."""
    from sklearn.model_selection import train_test_split

    X, y = split_xy(df)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp,
        random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test
