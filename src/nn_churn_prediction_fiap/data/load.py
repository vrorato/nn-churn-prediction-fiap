import pandas as pd

from nn_churn_prediction_fiap.config import RAW_DATA_PATH, TARGET_COL
from nn_churn_prediction_fiap.data.schema import raw_schema
from nn_churn_prediction_fiap.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV, validate schema, clean known issues, return model-ready df."""
    df = pd.read_csv(path, dtype={"TotalCharges": str})

    raw_schema.validate(df)
    logger.info(
        "schema validation passed",
        extra={"rows": len(df), "cols": len(df.columns)},
    )

    df = _clean(df)
    logger.info("data loaded and cleaned", extra={"rows": len(df)})
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is whitespace for tenure=0 rows; drop those ~11 rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    n_dropped = df["TotalCharges"].isna().sum()
    if n_dropped > 0:
        logger.info(
            "dropping rows with unparseable TotalCharges",
            extra={"n": int(n_dropped)},
        )
        df = df.dropna(subset=["TotalCharges"])

    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    df = df.reset_index(drop=True)
    return df
