import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import load_training_data
from src.features import build_preprocessor

SEED = 42
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.joblib"


def evaluate(name, pipe, X, y, threshold=0.5):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
    }


def main():
    X, y = load_training_data(DATA_PATH)

    # 1. Holdout split FIRST. The test set is touched exactly once, at the end.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    # 2. Baseline: logistic regression, class-weighted
    baseline = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    cv_res = cross_validate(baseline, X_train, y_train, cv=cv,
                            scoring=["roc_auc", "average_precision"])
    print(f"LR  CV  ROC-AUC={cv_res['test_roc_auc'].mean():.3f}  "
          f"PR-AUC={cv_res['test_average_precision'].mean():.3f}")
    baseline.fit(X_train, y_train)

    # 3. Primary model: XGBoost, tuned with randomized search, optimizing PR-AUC
    xgb = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", XGBClassifier(
            scale_pos_weight=neg / pos,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    param_dist = {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(2, 6),            # shallow: small, noisy dataset
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__min_child_weight": randint(1, 10),
        "clf__reg_lambda": loguniform(0.1, 10),
    }
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=30, scoring="average_precision",
        cv=cv, random_state=SEED, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    print(f"XGB CV  PR-AUC={search.best_score_:.3f}  params={search.best_params_}")
    best = search.best_estimator_

    # 4. Choose the decision threshold on out-of-fold TRAIN predictions, not on test
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    threshold = float(thr[np.argmax(f1s)])
    print(f"Threshold (max OOF F1): {threshold:.3f}")

    # 5. One-time holdout evaluation
    results = [
        evaluate("LogReg baseline @0.5", baseline, X_test, y_test),
        evaluate("XGBoost @0.5", best, X_test, y_test),
        evaluate("XGBoost @tuned thr", best, X_test, y_test, threshold),
    ]
    print(f"\n{'model':<24}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>7}{'Prec':>7}{'Rec':>7}")
    for r in results:
        print(f"{r['model']:<24}{r['roc_auc']:>9.3f}{r['pr_auc']:>9.3f}"
              f"{r['f1']:>7.3f}{r['precision']:>7.3f}{r['recall']:>7.3f}")
    print(f"\nMajority-class accuracy (predict 'no churn'): {1 - y_test.mean():.3f}")

    # 6. Save pipeline + threshold together so inference can't drift from training
    joblib.dump({"pipeline": best, "threshold": threshold}, MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()