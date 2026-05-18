import subprocess
import sys

import pytest

from nn_churn_prediction_fiap.config import RAW_DATA_PATH


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw data not available")
def test_train_logreg_smoke() -> None:
    """Full training pipeline runs without error on a 500-row subsample."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nn_churn_prediction_fiap.cli.train",
            "--model",
            "logreg",
            "--max-rows",
            "500",
            "--max-epochs",
            "2",
            "--n-folds",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
