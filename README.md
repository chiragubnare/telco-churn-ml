# Telco Customer Churn: Prediction Pipeline

Predicts customer churn on the Kaggle Telco dataset. A scikit-learn `Pipeline`
(preprocessing + model) is saved together with a decision threshold and served
through a CLI and a FastAPI endpoint.

## Project structure

```
telco-churn-ml/
├── data/WA_Fn-UseC_-Telco-Customer-Churn.csv   # download from Kaggle (not redistributed)
├── src/
│   ├── data.py        # loading + cleaning, shared by training and inference
│   ├── features.py    # ColumnTransformer (impute, scale, one-hot)
│   ├── train.py       # split, CV, tuning, evaluation, model export
│   ├── predict.py     # CLI / function inference
│   └── api.py         # FastAPI app
├── tests/test_api.py
├── models/churn_pipeline.joblib
├── sample.json
├── pytest.ini
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9-3.12.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`.

## Run

All commands run from the repository root.

```bash
python -m src.train                  # trains, evaluates, writes models/churn_pipeline.joblib
python -m src.predict sample.json    # CLI inference on one customer
uvicorn src.api:app --port 8000      # API: POST /predict, GET /health, docs at /docs
pytest -q                            # tests
```

A trained model is already included in `models/`, so inference works without retraining.

Example request (`sample.json` is the payload from the brief) and response:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8783,
  "threshold": 0.5456
}
```

`churn_probability` is a risk score, not a calibrated probability (see Limitations).
The API returns the same values as the CLI for the same input.

## Data findings (EDA)

- <rows> rows, <cols> columns. Target `Churn`: <x>% positive, so imbalance is moderate.
- `TotalCharges` is parsed as a string. It has 11 blank values, all with
  `tenure == 0` (customers who were never billed). These are converted to NaN and
  imputed with 0, since a never-billed customer has paid nothing.
- `customerID` is an identifier and is dropped.
- Strongest signals: <churn rate for month-to-month vs. one/two-year contracts>,
  <tenure pattern>, <internet service type>.

## Approach

- **Split first.** 80/20 stratified split before any fitting. The test set is used once, for final evaluation.
- **No leakage.** Imputation, scaling and one-hot encoding are inside the `Pipeline`, so they are re-fit on each CV training fold only.
- **Imbalance.** Handled with class weighting (`class_weight="balanced"` for logistic regression, `scale_pos_weight` for XGBoost) instead of SMOTE, because the imbalance is moderate and weighting adds no leakage risk.
- **Models.** Logistic regression baseline and a tuned XGBoost model.
- **Tuning.** 30-iteration randomized search, 5-fold stratified CV, optimizing PR-AUC (average precision).
- **Threshold.** Chosen by maximizing F1 on out-of-fold training predictions (never on the test set). The value is stored in the artifact, so the API's label matches the evaluation.
- **Serving.** Raw JSON is cleaned by the same `clean_features` function used in training, then passed to the saved pipeline.

## Results

Cross-validation (5-fold, training split only):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.846 | 0.660 |
| XGBoost (tuned) | <> | <> |

Holdout test set (n = <n_test>, used once):

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Majority class ("no churn") | 0.500 | <base rate> | 0.000 | n/a | 0.000 |
| Logistic Regression @0.5 | <> | <> | <> | <> | <> |
| XGBoost @0.5 | <> | <> | <> | <> | <> |
| XGBoost @0.5456 (tuned threshold) | <> | <> | <> | <> | <> |

<One or two sentences, written from your numbers: which model performs better and
by how much; whether the gap is larger than the fold-to-fold spread; and which
model is saved in the artifact and why. If the models are within noise, say so.>

**Why accuracy is insufficient:** a model that predicts "no churn" for every
customer scores about <1 - base rate> accuracy while identifying zero churners.
The costs are also asymmetric: a missed churner loses recurring revenue, while a
false alarm costs a retention offer. PR-AUC, precision and recall measure
performance on the minority class, which accuracy hides. ROC-AUC is reported as a
secondary ranking metric.

## Limitations

- **Cleaning is outside the artifact.** `clean_features` is called at inference, not stored inside the saved pipeline. Training and serving share the same function, but the coupling is manual.
- **Categorical values are not validated.** Unseen or misspelled categories are encoded as all zeros and give a degraded prediction without an error.
- **Probabilities are not calibrated.** Class weighting inflates predicted probabilities, so `churn_probability` is a ranking score, not a literal likelihood.
- **Threshold is slightly optimistic.** Hyperparameters and the threshold were selected on the same training data. The F1 curve is also flat near its peak, so the exact value is noisy.
- **No temporal validation.** The data is a single snapshot without dates, so a random split cannot test drift.
- **Small holdout.** About <n_pos> positive cases in the holdout, so metrics carry meaningful uncertainty.

## If I had 2 more days

1. **Calibrated probabilities and a cost-based threshold.** Wrap the model in
   `CalibratedClassifierCV`, check a reliability curve and Brier score, then pick
   the threshold by maximizing expected value under an explicit cost matrix
   (retention offer cost vs. lost customer lifetime value). This fixes the
   inflated probabilities caused by class weighting, and replaces max-F1, which
   weights both error types equally even though their business costs differ.
2. **Self-contained, reproducible deployment.** Move `clean_features` into the
   pipeline with `FunctionTransformer` so the artifact accepts raw rows, add
   enumerated-category validation that logs unseen values, and add a Dockerfile
   and CI running `pytest`. This removes the train/serve coupling and the silent
   failure on bad categories listed above.
3. **Statistically sound model selection.** Use repeated stratified (or nested)
   CV and bootstrap confidence intervals on the holdout, and compare models with
   paired fold differences. With roughly <n_pos> positive holdout cases, the
   difference between logistic regression and XGBoost may be within noise, and
   this tests whether the chosen model is actually better.
