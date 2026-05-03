# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
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


nominal_cols = [
    'purpose','personal_status_sex','other_debtors_guarantors',
    'property','other_installment_plans','housing','job','telephone'
]

exclude_cols = ['foreign_worker']

drop_candidate = ['residence_since', 'dependants_count', 'telephone', 'job']

numeric_cols = [
    'duration_months','credit_amount','installment_rate_pct',
    'residence_since','age','existing_credits_count','dependants_count'
]

# %% [markdown]
#  Drop candidates identified in EDA (near-zero Mann-Whitney significance)


# %% [markdown]
# The core function is to decode values 

# %%


# %%
def decode(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col, mapping in label_maps.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(X[col])
    return X


def drop_columns(X: pd.DataFrame) -> pd.DataFrame:
    cols = exclude_cols + drop_candidate
    cols = [c for c in cols if c in X.columns]
    return X.drop(columns=cols)
print("Columns after drop:", X.columns.tolist())

def build_preprocessor(X: pd.DataFrame):
    ordinal_cols = [c for c in ordinal_maps if c in X.columns]
    nominal = [c for c in nominal_cols if c in X.columns]
    numeric = [c for c in numeric_cols if c in X.columns]

    ordinal_enc = OrdinalEncoder(
        categories=[ordinal_maps[c] for c in ordinal_cols],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    nominal_enc = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    print("Ordinal:", ordinal_cols)
    print("Nominal:", nominal)
    print("Numeric:", numeric)
    scaler = StandardScaler()

    preprocessor = ColumnTransformer([
        ("ord", ordinal_enc, ordinal_cols),
        ("nom", nominal_enc, nominal),
        ("num", scaler, numeric),
    ])

    return preprocessor


# %% MAIN PIPELINE


    ...
def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
):
    print("Input X shape:", X.shape)          # ADD THIS
    print("Input X columns:", X.columns.tolist())  # ADD THIS
    X = decode(X)
    print("After decode:", X.shape)            # ADD THIS
    X = drop_columns(X)
    print("After drop:", X.shape)              # ADD THIS
    # 1. Decode & clean
    X = decode(X)
    X = drop_columns(X)

    # 2. Target
    y_binary = (y == 2).astype(int)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary,
        test_size=test_size,
        stratify=y_binary,
        random_state=random_state,
    )

    # 4. Preprocessor
    preprocessor = build_preprocessor(X_train)

    # 5. Fit + transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # 6. Convert to DataFrame
    feature_names = preprocessor.get_feature_names_out()

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # 7. Save
    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        print("✅ Preprocessor saved")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train bad rate: {y_train.mean():.1%}")
    print(f"Test bad rate:  {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# %% API HELPER

def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")


def preprocess_single(input_dict: dict) -> pd.DataFrame:
    X = pd.DataFrame([input_dict])
    X = decode(X)
    X = drop_columns(X)

    preprocessor = load_preprocessor()
    X = preprocessor.transform(X)

    return pd.DataFrame(X)