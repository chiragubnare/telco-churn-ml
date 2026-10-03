from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

SAMPLE = {
    "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": "29.85",
}


def test_sample_payload():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_string_and_float_totalcharges_match():
    a = client.post("/predict", json=SAMPLE).json()
    b = client.post("/predict", json={**SAMPLE, "TotalCharges": 29.85}).json()
    assert a == b


def test_blank_totalcharges_handled():
    r = client.post("/predict", json={**SAMPLE, "tenure": 0, "TotalCharges": " "})
    assert r.status_code == 200


def test_missing_field_rejected():
    bad = {k: v for k, v in SAMPLE.items() if k != "Contract"}
    assert client.post("/predict", json=bad).status_code == 422