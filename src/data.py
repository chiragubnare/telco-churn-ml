import pandas as pd

TARGET = "Churn"
DROP_COLS = ["customerID"]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applied identically at train and inference time. Raw rows in, model-ready columns out."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    # Blank strings -> NaN; JSON string "29.85" -> float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def load_training_data(path: str):
    df = pd.read_csv(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = clean_features(df.drop(columns=[TARGET]))
    return X, y