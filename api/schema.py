from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from fastapi.responses import JSONResponse
from typing import Annotated, Literal, Optional
import joblib
import json
import os

# -----------------------------
# App & Model Setup
# -----------------------------
app = FastAPI(title="Credit Risk API")
 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
 
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)




 
#After saving the model, the next pahase is making pydantic model 
class CreditApplication(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "checking_account_status": "no account",
                "credit_history": "existing paid",
                "savings_account": "< 100 DM",
                "employment_since": "1–4 years",
                "purpose": "car (used)",
                "personal_status_sex": "male: single",
                "other_debtors": "none",
                "property": "real estate",
                "other_installment_plans": "none",
                "housing": "own",
                "job": "skilled employee",
                "telephone": "no phone",
                "duration_months": 24,
                "credit_amount": 3000.0,
                "installment_rate_pct": 2,
                "residence_since": 3,
                "age": 35,
                "existing_credits_count": 1,
                "dependent_count": 1,
            }
        }
    )

    # Categorical fields — just Field with description, no example
    checking_account_status: Annotated[str, Field(description="Status of checking account")]
    credit_history: Annotated[str, Field(description="Credit history at this bank")]
    savings_account: Annotated[str, Field(description="Savings account balance")]
    employment_since: Annotated[str, Field(description="Years in current employment")]
    purpose: Annotated[str, Field(description="Purpose of the loan")]
    personal_status_sex: Annotated[str, Field(description="Personal status and sex")]
    other_debtors: Annotated[str, Field(description="Other debtors or guarantors")]
    property: Annotated[str, Field(description="Most valuable property owned")]
    other_installment_plans: Annotated[str, Field(description="Other active installment plans")]
    housing: Annotated[str, Field(description="Housing situation")]
    job: Annotated[str, Field(description="Job category")]
    telephone: Annotated[str, Field(description="Telephone registered under applicant name?")]

    # Numeric fields — keep ge/le/gt constraints, drop example
    duration_months: Annotated[int, Field(gt=0, description="Loan duration in months")]
    credit_amount: Annotated[float, Field(gt=0, description="Loan amount in Deutsche Marks")]
    installment_rate_pct: Annotated[int, Field(ge=1, le=4, description="Installment rate % of disposable income")]
    residence_since: Annotated[int, Field(ge=1, le=4, description="Years at current residence")]
    age: Annotated[int, Field(ge=18, le=100, description="Applicant age in years")]
    existing_credits_count: Annotated[int, Field(ge=1, description="Number of existing credits at this bank")]
    dependent_count: Annotated[int, Field(ge=1, le=2, description="Number of dependants")]
 
 
class RiskPrediction(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "risk_score": 0.23,
                "risk_label": "low_risk",
                "confidence_band": "low risk (uncertain)"
            }
        }
    )

    risk_score: Annotated[float, Field(ge=0, le=1, description="Probability of default")]
    risk_label: Annotated[str, Field(description="Decision at threshold 0.5")]
    confidence_band: Annotated[str, Field(description="Human-readable confidence band")]
 
# -----------------------------
# Helpers
# -----------------------------
CONFIDENCE_BANDS = [
    (0.20, "low risk (confident)"),
    (0.35, "low risk (uncertain)"),
    (0.65, "borderline"),
    (0.80, "high risk (uncertain)"),
    (1.01, "high risk (confident)"),
]
 
def get_confidence(prob: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if prob < threshold:
            return label
    return "high risk (confident)"
 

def save_data(data):
    with open ('CreditApplication.json', 'w') as f:
        json.dump(data, f)
# -----------------------------
# This is for human readable 
@app.get("/") 
def home():
    return {"message": "Credit Risk API running"}
#this is for machine readable
@app.get ('/health')
def health_check():
    return {
        "status" : "ok"
        
    }
 
@app.post("/predict", response_model=RiskPrediction)
def predict(data: CreditApplication):
    input_dict = data.model_dump()
    X = preprocessor.transform([input_dict])
    prob = float(model.predict_proba(X)[0][1])
    return RiskPrediction(
        risk_score=round(prob, 3),
        risk_label="high_risk" if prob >= 0.5 else "low_risk",
        confidence=get_confidence(prob),
    )