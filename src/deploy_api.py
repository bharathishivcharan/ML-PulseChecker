
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("models/churn_model.pkl")
FEATURES = ["total_events", "total_sessions", "avg_events_per_session", "revenue_per_session", "active_days"]

# FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0")

class UserFeatures(BaseModel):
    total_events: int
    total_sessions: int
    avg_events_per_session: float
    revenue_per_session: float
    active_days: int

@app.get("/")
def root():
    return {"message": "Welcome to ML-PulseChecker Churn API"}

@app.post("/predict")
def predict_churn(user: UserFeatures):
    data = pd.DataFrame([user.dict()])
    proba = model.predict_proba(data[FEATURES])[:,1][0]
    return {"churn_probability": round(float(proba), 4)}