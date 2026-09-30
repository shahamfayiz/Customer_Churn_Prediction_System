from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Logistic Regression"
)


model = joblib.load("../models/logistic_regression_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
feature_columns = joblib.load("../models/feature_columns.pkl")
total_charges_median = joblib.load("../models/total_charges_median.pkl")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data["TotalCharges"] = data["TotalCharges"].fillna(
        total_charges_median
    )

    binary_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for column in binary_columns:
        data[column] = data[column].map({
            "Yes": 1,
            "No": 0
        })

    data = pd.get_dummies(
        data,
        drop_first=True,
        dtype=int
    )

    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(
        data_scaled
    )[0][1]

    if prediction == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 4)
    }