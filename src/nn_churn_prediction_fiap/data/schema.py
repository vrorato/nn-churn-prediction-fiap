import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

raw_schema = DataFrameSchema(
    columns={
        "customerID": Column(str, nullable=False),
        "gender": Column(str, pa.Check.isin(["Male", "Female"]), nullable=False),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Partner": Column(str, pa.Check.isin(["Yes", "No"]), nullable=False),
        "Dependents": Column(str, pa.Check.isin(["Yes", "No"]), nullable=False),
        "tenure": Column(int, pa.Check.ge(0), nullable=False),
        "PhoneService": Column(str, pa.Check.isin(["Yes", "No"]), nullable=False),
        "MultipleLines": Column(
            str,
            pa.Check.isin(["Yes", "No", "No phone service"]),
            nullable=False,
        ),
        "InternetService": Column(
            str,
            pa.Check.isin(["DSL", "Fiber optic", "No"]),
            nullable=False,
        ),
        "OnlineSecurity": Column(
            str,
            pa.Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
        ),
        "OnlineBackup": Column(
            str,
            pa.Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
        ),
        "DeviceProtection": Column(
            str,
            pa.Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
        ),
        "TechSupport": Column(
            str,
            pa.Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
        ),
        "StreamingTV": Column(
            str,
            pa.Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
        ),
        "StreamingMovies": Column(
            str,
            pa.Check.isin(["Yes", "No", "No internet service"]),
            nullable=False,
        ),
        "Contract": Column(
            str,
            pa.Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
        ),
        "PaperlessBilling": Column(str, pa.Check.isin(["Yes", "No"]), nullable=False),
        "PaymentMethod": Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
            nullable=False,
        ),
        "MonthlyCharges": Column(float, pa.Check.ge(0), nullable=False),
        # TotalCharges is object dtype in the raw CSV — whitespace for tenure=0 rows
        "TotalCharges": Column(str, nullable=False),
        "Churn": Column(str, pa.Check.isin(["Yes", "No"]), nullable=False),
    },
    strict=False,  # allow extra columns (e.g. future additions) without failing
    coerce=False,
)
