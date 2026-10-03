import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.data import clean_features

MODEL_PATH = "models/churn_pipeline.joblib"


@lru_cache(maxsize=1)
def load_bundle(path: str = MODEL_PATH) -> dict:
    # joblib/pickle executes code on load: only load files you produced yourself.
    return joblib.load(path)


def predict_one(payload: dict) -> dict:
    bundle = load_bundle()
    X = clean_features(pd.DataFrame([payload]))
    proba = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    thr = float(bundle["threshold"])
    return {
        "churn_prediction": int(proba >= thr),
        "churn_probability": round(proba, 4),
        "threshold": round(thr, 4),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="Path to a JSON file with one customer")
    args = ap.parse_args()
    with open(args.json_path) as f:
        print(json.dumps(predict_one(json.load(f)), indent=2))