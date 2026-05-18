import pandas as pd
import pandera
import pytest

from nn_churn_prediction_fiap.data.schema import raw_schema


def test_valid_data_passes(sample_raw_df: pd.DataFrame) -> None:
    raw_schema.validate(sample_raw_df)


def test_missing_column_raises(sample_raw_df: pd.DataFrame) -> None:
    df = sample_raw_df.drop(columns=["tenure"])
    with pytest.raises(pandera.errors.SchemaError):
        raw_schema.validate(df)


def test_invalid_categorical_raises(sample_raw_df: pd.DataFrame) -> None:
    df = sample_raw_df.copy()
    df.loc[0, "gender"] = "Other"
    with pytest.raises(pandera.errors.SchemaError):
        raw_schema.validate(df)


def test_negative_tenure_raises(sample_raw_df: pd.DataFrame) -> None:
    df = sample_raw_df.copy()
    df.loc[0, "tenure"] = -1
    with pytest.raises(pandera.errors.SchemaError):
        raw_schema.validate(df)
