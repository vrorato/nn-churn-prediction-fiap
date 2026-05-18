import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from nn_churn_prediction_fiap.features.pipeline import ALL_FEATURES
from nn_churn_prediction_fiap.models.baseline import make_logreg


@pytest.fixture(scope="session")
def sample_raw_df() -> pd.DataFrame:
    """50 rows of synthetic data matching raw_schema."""
    rng = np.random.default_rng(42)
    n = 50
    tenures = rng.integers(1, 72, size=n).tolist()
    monthly = rng.uniform(20.0, 100.0, size=n).round(2).tolist()
    total = [str(round(t * m, 2)) for t, m in zip(tenures, monthly, strict=False)]
    churn = ["Yes"] * 13 + ["No"] * 37

    rows = []
    for i in range(n):
        rows.append(
            {
                "customerID": f"{i:04d}-TEST",
                "gender": "Male" if i % 2 == 0 else "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes" if i % 3 == 0 else "No",
                "Dependents": "No",
                "tenure": int(tenures[i]),
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": float(monthly[i]),
                "TotalCharges": total[i],
                "Churn": churn[i],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def sample_clean_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Same data after cleaning: TotalCharges as float, Churn as int."""
    df = sample_raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def trained_model(sample_clean_df: pd.DataFrame) -> Pipeline:
    """LogReg pipeline trained on synthetic data (no MLflow, fast)."""
    X = sample_clean_df[ALL_FEATURES]
    y = sample_clean_df["Churn"]
    model = make_logreg(random_state=42)
    model.fit(X, y)
    return model
