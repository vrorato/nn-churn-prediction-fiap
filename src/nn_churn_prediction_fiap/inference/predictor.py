import os

import mlflow.sklearn

from nn_churn_prediction_fiap.config import MLFLOW_TRACKING_URI
from nn_churn_prediction_fiap.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_URI_TEMPLATE = "models:/churn-mlp@{alias}"


def load_model(alias: str = "Staging"):
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        uri = f"file://{model_path}"
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        uri = _MODEL_URI_TEMPLATE.format(alias=alias)
    logger.info("loading_model", extra={"uri": uri})
    model = mlflow.sklearn.load_model(uri)
    logger.info("model_loaded", extra={"uri": uri, "type": type(model).__name__})
    return model
