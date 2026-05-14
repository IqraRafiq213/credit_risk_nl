import streamlit as st
import joblib
import json
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Credit Risk Predictor")
st.caption("Fill in the applicant details below and click **Predict** to get a risk assessment.")

# -----------------------------
# Load model & preprocessor
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_artifacts()

# -----------------------------
# Helpers
# -----------------------------
CONFIDENCE_BANDS = [
    (0.20, "✅ Low risk (confident)"),
    (0.35, "🟡 Low risk (uncertain)"),
    (0.65, "🟠 Borderline"),
    (0.80, "🔴 High risk (uncertain)"),
    (1.01, "🔴 High risk (confident)"),
]

def get_confidence(prob: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if prob < threshold:
            return label
    return "🔴 High risk (confident)"

def save_application(data: dict) -> None:
    with open("CreditApplication.json", "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Form
# -----------------------------
with st.form("credit_form"):

    st.subheader("📋 Account & Credit History")
    col1, col2 = st.columns(2)
    with col1:
        checking_account_status = st.selectbox(
            "Checking account status",
            ["< 0 DM", "0–200 DM", "> 200 DM", "no account"],
        )
        credit_history = st.selectbox(
            "Credit history",
            ["no credits taken", "all paid at bank", "existing paid", "delay in past", "critical account"],
        )
    with col2:
        savings_account = st.selectbox(
            "Savings account",
            ["< 100 DM", "100–500 DM", "500–1000 DM", "> 1000 DM", "no savings"],
        )
        existing_credits_count = st.number_input(
            "Existing credits at this bank", min_value=1, max_value=10, value=1
        )

    st.divider()
    st.subheader("💼 Employment & Personal")
    col3, col4 = st.columns(2)
    with col3:
        employment_since = st.selectbox(
            "Employment since",
            ["unemployed", "< 1 year", "1–4 years", "4–7 years", "> 7 years"],
        )
        personal_status_sex = st.selectbox(
            "Personal status & sex",
            ["male: divorced", "female: divorced/married", "male: single", "male: married"],
        )
        job = st.selectbox(
            "Job category",
            ["unskilled non-resident", "unskilled resident", "skilled employee", "highly qualified"],
        )
    with col4:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        dependent_count = st.selectbox("Number of dependants", [1, 2])
        telephone = st.selectbox(
            "Telephone (optional)",
            ["no phone", "yes, registered"],
        )

    st.divider()
    st.subheader("💰 Loan Details")
    col5, col6 = st.columns(2)
    with col5:
        credit_amount = st.number_input(
            "Credit amount (DM)", min_value=1.0, value=3000.0, step=100.0
        )
        duration_months = st.number_input(
            "Duration (months)", min_value=1, value=24
        )
        installment_rate_pct = st.slider(
            "Installment rate (% of income)", min_value=1, max_value=4, value=2
        )
    with col6:
        purpose = st.selectbox(
            "Purpose",
            ["car (new)", "car (used)", "furniture", "radio/TV", "domestic",
             "repairs", "education", "vacation", "retraining", "business", "other"],
        )
        other_installment_plans = st.selectbox(
            "Other installment plans", ["none", "bank", "stores"]
        )

    st.divider()
    st.subheader("🏠 Property & Housing")
    col7, col8 = st.columns(2)
    with col7:
        property_ = st.selectbox(
            "Property owned",
            ["real estate", "savings/insurance", "car/other", "no property"],
        )
        housing = st.selectbox("Housing", ["own", "rent", "free"])
    with col8:
        other_debtors = st.selectbox(
            "Other debtors / guarantors", ["none", "co-applicant", "guarantor"]
        )
        residence_since = st.slider(
            "Years at current residence", min_value=1, max_value=4, value=3
        )

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_dict = {
        "checking_account_status": checking_account_status,
        "credit_history": credit_history,
        "savings_account": savings_account,
        "employment_since": employment_since,
        "purpose": purpose,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "property": property_,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "job": job,
        "telephone": telephone,
        "duration_months": int(duration_months),
        "credit_amount": float(credit_amount),
        "installment_rate_pct": int(installment_rate_pct),
        "residence_since": int(residence_since),
        "age": int(age),
        "existing_credits_count": int(existing_credits_count),
        "dependent_count": int(dependent_count),
    }

    save_application(input_dict)

    X = preprocessor.transform([input_dict])
    prob = float(model.predict_proba(X)[0][1])
    risk_label = "high_risk" if prob >= 0.5 else "low_risk"
    confidence = get_confidence(prob)

    st.divider()
    st.subheader("📊 Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Risk Score", f"{prob:.3f}")
    col_b.metric("Risk Label", risk_label.replace("_", " ").title())
    col_c.metric("Confidence", confidence)

    if risk_label == "low_risk":
        st.success("✅ This applicant is assessed as **low risk**.")
    else:
        st.error("🚨 This applicant is assessed as **high risk**.")

    with st.expander("View submitted data"):
        st.json(input_dict)