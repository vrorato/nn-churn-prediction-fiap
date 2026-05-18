import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from nn_churn_prediction_fiap.api.dependencies import get_predictor
from nn_churn_prediction_fiap.api.middleware import LatencyLoggingMiddleware
from nn_churn_prediction_fiap.api.schemas import CustomerFeatures, PredictionResponse
from nn_churn_prediction_fiap.config import OPERATING_THRESHOLD
from nn_churn_prediction_fiap.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")
app.add_middleware(LatencyLoggingMiddleware)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures, model=Depends(get_predictor)):  # noqa: B008
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(df)[:, 1][0])

    logger.info(
        "prediction",
        extra={
            "churn_probability": proba,
            "churn_flag": proba >= OPERATING_THRESHOLD,
        },
    )

    return PredictionResponse(
        churn_probability=proba,
        churn_flag=proba >= OPERATING_THRESHOLD,
        threshold=OPERATING_THRESHOLD,
        model_version="v1",
    )
