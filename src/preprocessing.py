# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import joblib
import os

# %%
# Load German Credit dataset
ds = fetch_ucirepo(id=144)
X, y = ds.data.features, ds.data.targets

# %%
#mapping labels from UCI code decoder
label_maps = {'checking_account_status': {'A11': '< 0 DM', 'A12': '0–200 DM','A13': '> 200 DM', 'A14': 'no account' },
    'credit_history': {'A30': 'no credits taken', 'A31': 'all paid at bank','A32': 'existing paid',    'A33': 'delay in past','A34': 'critical account' },
    'purpose': {'A40': 'car (new)',    'A41': 'car (used)',  'A42': 'furniture','A43': 'radio/TV',    'A44': 'domestic',    'A45': 'repairs','A46': 'education',   'A47': 'vacation',    'A48': 'retraining',
        'A49': 'business',    'A410': 'other'},
    'savings_account': { 'A61': '< 100 DM',  'A62': '100–500 DM', 'A63': '500–1000 DM','A64': '> 1000 DM', 'A65': 'no savings'},
    'employment_since': { 'A71': 'unemployed', 'A72': '< 1 year',  'A73': '1–4 years','A74': '4–7 years',  'A75': '> 7 years' },
    'personal_status_sex': { 'A91': 'male: divorced',          'A92': 'female: divorced/married','A93': 'male: single','A94': 'male: married'},
    'other_debtors_guarantors': {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'},
    'property': {'A121': 'real estate',       'A122': 'savings/insurance','A123': 'car/other','A124': 'no property'},
    'other_installment_plans': {'A141': 'bank', 'A142': 'stores', 'A143': 'none'},
    'housing':{'A151': 'rent', 'A152': 'own',  'A153': 'free'},
    'job': {'A171': 'unskilled non-resident', 'A172': 'unskilled resident','A173': 'skilled employee','A174': 'highly qualified' },
    'telephone': {'A191': 'no phone', 'A192': 'yes, registered'},
    'foreign_worker':   {'A201': 'yes', 'A202': 'no'},
}

# %%
#ordinal mapping 
ordinal_maps = {'checking_account_status': ['no account', '> 200 DM', '0–200 DM', '< 0 DM'],  # ordered: safest → riskiest (confirmed in EDA)

    'credit_history': [ 'critical account', 'delay in past', 'existing paid','all paid at bank', 'no credits taken'],  # ordered: lowest → highest default rate (EDA finding)

    'savings_account': ['> 1000 DM', '500–1000 DM', 'no savings','100–500 DM', '< 100 DM'],  # ordered: safest → riskiest

    'employment_since': [ '4–7 years', '> 7 years', '1–4 years','unemployed', '< 1 year'],  # ordered by default rate from EDA
}

# %%
#nominal column 
nominal_cols = [
    'purpose','personal_status_sex','other_debtors_guarantors',
    'property','other_installment_plans','housing','job','telephone'
]

# %% [markdown]
# excluded foreign workers due to ethical and legal reasons. 

# %%
exclude_cols = ['foreign_worker']

# %%
numeric_cols = ['duration_months','credit_amount','installment_rate_pct','residence_since','age','existing_credits_count','dependants_count']

# %% [markdown]
#  Drop candidates identified in EDA (near-zero Mann-Whitney significance)

# %%
drop_candidate = ['residence_since', 'dependants_count', 'telephone', 'job']

# %% [markdown]
# The core function is to decode values 

# %%
def decode(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col, v in label_maps.items():
        if col in X.columns:
            X[col] = X[col].mapped(v).fillna(X[col])
    return X

# %%
#pipeline to drop column 
def dropcolumn(X:pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = exclude_cols.copy()
    if drop_candidate:
        cols_to_drop += drop_candidate
    existing = [c for c in cols_to_drop if c in X.columns]
    return X.drop(columns=existing)

# %%
def build_encoders(X: pd.DataFrame):
    ordinal_cols = [c for c in ordinal_maps if c in X.columns]
    ordinal_enc  = OrdinalEncoder(handle_unknown='use_encoded_value')     
    ordinal_enc.fit(X[ordinal_cols])
    nominal = [c for c in nominal_cols if c in X.columns]
    nominal_enc  = OrdinalEncoder(handle_unknown='use_encoded_value')
    nominal_enc.fit(X[nominal])
       # Numeric scaler
    num_cols = [c for c in numeric_cols if c in X.columns]
    scaler   = StandardScaler()
    scaler.fit(X[num_cols])

    return ordinal_enc, nominal_enc, scaler

# %%
def apply_encoders( X: pd.DataFrame, ordinal_enc: OrdinalEncoder, nominal_enc: OrdinalEncoder,scaler: StandardScaler) -> pd.DataFrame:
    """Apply fitted encoders to a dataframe. Safe to use on train and test."""
    X = X.copy()

    ordinal_cols = [c for c in ordinal_maps  if c in X.columns]
    nominal_cols = [c for c in nominal_cols  if c in X.columns]
    num_cols     = [c for c in numeric_cols  if c in X.columns]

    X[ordinal_cols] = ordinal_enc.transform(X[ordinal_cols])
    X[nominal_cols] = nominal_enc.transform(X[nominal_cols])
    X[num_cols]     = scaler.transform(X[num_cols])

    return X


# %%
def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    drop_low_signal: bool = False,
    random_state: int = 42,
    save_artifacts: bool = True,
):
    """
    Full preprocessing pipeline:
    1) Decode UCI codes
    2) Drop excluded
    3) Stratified train/test split
    4) Fit encoders on train only (no leakage)
    5) Apply to both splits
    6) Optionally save artifacts to models/

    Returns:
        X_train, X_test, y_train, y_test
    """

    X = decode(X)
    X = dropcolumn(X, drop_low_signal=drop_low_signal)

    y_binary = (y == 2).astype(int)  # 1 = bad credit

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=test_size,
        stratify=y_binary,
        random_state=random_state,
    )

    # Fit encoders on train only
    ordinal_enc, nominal_enc, scaler = build_encoders(X_train)

    # Transform
    X_train = apply_encoders(X_train, ordinal_enc, nominal_enc, scaler)
    X_test = apply_encoders(X_test, ordinal_enc, nominal_enc, scaler)

    # Save artifacts
    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(ordinal_enc, "models/ordinal_enc.pkl")
        joblib.dump(nominal_enc, "models/nominal_enc.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        print("Artifacts saved to models/")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Class balance (train): {y_train.mean():.1%} bad credit")
    print(f"Class balance (test):  {y_test.mean():.1%} bad credit")

    return X_train, X_test, y_train, y_test

# %% [markdown]
# loads saved artifacts (encoder and scalars) from the list so the app can re use them
# 
# 

# %%
def load_artifacts():
    """ load artifacts and scalar. Used by FastAPI at startup"""
    ordinal_enc = joblib.load('models/ordinal_enc.pkl')
    nominal_enc = joblib.load('models/nominal_enc.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return ordinal_enc, nominal_enc, scaler

# %%
def preprocess_single(input_dict: dict) -> pd.DataFrame:
    """
    Preprocess a single applicant dict from the API.
    Decodes, drops excluded cols, applies saved encoders.
    Returns a single-row DataFrame ready for model.predict_proba().
    """
    X = pd.DataFrame([input_dict])
    X = decode(X)
    X = dropcolumn(X, drop_low_signal=False)
    ordinal_enc, nominal_enc, scaler = load_artifacts()
    return apply_encoders(X, ordinal_enc, nominal_enc, scaler)


