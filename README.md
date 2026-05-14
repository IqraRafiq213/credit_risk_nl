# 🏦 Credit Risk Predictor (German Credit Dataset)

A machine learning project that predicts credit risk using the German Credit dataset. The project covers the full pipeline from data exploration and model training to a deployable Streamlit app and a FastAPI backend.

---

## 📁 Project Structure

```
credit_risk_nl/
├── api/                  # FastAPI backend (main.py)
├── app/                  # Streamlit frontend (app.py)
├── models/               # Saved model & preprocessor (.pkl files)
├── notebooks/            # Jupyter notebooks for EDA & training
├── results/              # Evaluation outputs (metrics, plots)
├── src/                  # Reusable source code (preprocessing, training)
├── tests/                # Unit tests
├── requirement.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/IqraRafiq213/credit_risk_nl.git
cd credit_risk_nl
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirement.txt
```

---

## 🚀 Running the App

### Streamlit (standalone — no API needed)
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501`. Fill in the applicant form and click **Predict Risk**.

### FastAPI backend
```bash
uvicorn api.main:app --reload
```
Opens at `http://127.0.0.1:8000`. Visit `/docs` for the interactive Swagger UI.

### Docker (Streamlit)
```bash
docker build -t credit-risk-app .
docker run -p 8501:8501 credit-risk-app
```

---

## 🧠 Model

The model is trained on the [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) and predicts whether a loan applicant is **low risk** or **high risk**.

The pipeline uses a `ColumnTransformer` preprocessor (`preprocessor.pkl`) followed by a trained classifier (`best_model.pkl`).

**Input features:**

| Feature | Type | Description |
|---|---|---|
| `checking_account_status` | Categorical | Balance in checking account |
| `credit_history` | Categorical | Past repayment behaviour |
| `savings_account` | Categorical | Savings balance |
| `employment_since` | Categorical | Duration in current job |
| `purpose` | Categorical | Reason for the loan |
| `personal_status_sex` | Categorical | Marital status and sex |
| `other_debtors` | Categorical | Co-applicant or guarantor |
| `property` | Categorical | Most valuable asset owned |
| `other_installment_plans` | Categorical | Other active repayments |
| `housing` | Categorical | Rent / own / free |
| `job` | Categorical | Skill level of employment |
| `telephone` | Categorical (optional) | Registered phone |
| `duration_months` | Numeric | Loan term |
| `credit_amount` | Numeric | Loan amount (DM) |
| `installment_rate_pct` | Numeric | % of disposable income |
| `residence_since` | Numeric | Years at current address |
| `age` | Numeric | Applicant age |
| `existing_credits_count` | Numeric | Credits at this bank |
| `dependent_count` | Numeric | Number of dependants |

**Output:**

| Field | Description |
|---|---|
| `risk_score` | Probability of default (0–1) |
| `risk_label` | `low_risk` or `high_risk` at threshold 0.5 |
| `confidence_band` | Human-readable band: *low risk (confident)*, *low risk (uncertain)*, *borderline*, *high risk (uncertain)*, *high risk (confident)* |

---

## 🧪 Quick Test

Run a single prediction directly without starting any server:

```bash
python api/main.py
```

This executes the `if __name__ == "__main__"` block with a sample applicant and prints the result to the terminal.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
fastapi
uvicorn
pydantic
streamlit
joblib
```

---

## 📓 Notebooks

The `notebooks/` folder contains step-by-step Jupyter notebooks covering:
- Exploratory data analysis (EDA)
- Feature engineering and preprocessing
- Model selection and hyperparameter tuning
- Evaluation (ROC-AUC, confusion matrix, classification report)

To run:
```bash
jupyter notebook notebooks/
```
