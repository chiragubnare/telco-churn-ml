from typing import Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_bundle, predict_one

load_bundle()  # fail fast at startup if the artifact is missing
app = FastAPI(title="Churn Inference API")


class Customer(BaseModel):
    customerID: Optional[str] = None  # accepted, then dropped by clean_features
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Union[str, float, None] = None  # sample sends a string; blank is legal


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    # Sync def: FastAPI runs it in a threadpool, so sklearn doesn't block the event loop.
    return predict_one(customer.model_dump())