# src/features.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is already 0/1, so pass it through as numeric-ish
PASSTHROUGH = ["SeniorCitizen"]


def build_preprocessor(X) -> ColumnTransformer:
    categorical = [c for c in X.columns if c not in NUMERIC + PASSTHROUGH]
    numeric_pipe = Pipeline([
        # TotalCharges NaN only occurs when tenure == 0 (never billed), so 0 is defensible.
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
        ("pass", "passthrough", PASSTHROUGH),
    ])