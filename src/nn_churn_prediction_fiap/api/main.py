import subprocess
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from nn_churn_prediction_fiap.api.dependencies import get_predictor, set_predictor
from nn_churn_prediction_fiap.api.middleware import LatencyLoggingMiddleware
from nn_churn_prediction_fiap.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
)
from nn_churn_prediction_fiap.config import OPERATING_THRESHOLD
from nn_churn_prediction_fiap.inference.predictor import load_model
from nn_churn_prediction_fiap.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_ALIAS = "Staging"


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_predictor(load_model(alias=_MODEL_ALIAS))
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(LatencyLoggingMiddleware)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_alias": _MODEL_ALIAS,
        "git_commit": _git_commit(),
    }


def _predict_one(features: CustomerFeatures, model) -> PredictionResponse:
    df = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(df)[:, 1][0])
    return PredictionResponse(
        churn_probability=proba,
        churn_flag=proba >= OPERATING_THRESHOLD,
        threshold=OPERATING_THRESHOLD,
        model_version=_MODEL_ALIAS,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: CustomerFeatures,
    request: Request,
    model=Depends(get_predictor),  # noqa: B008
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = _predict_one(features, model)
    request_id = request.headers.get("X-Request-ID", "n/a")
    logger.info(
        "prediction",
        extra={
            "request_id": request_id,
            "churn_probability": result.churn_probability,
            "churn_flag": result.churn_flag,
        },
    )
    return result


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    items: list[CustomerFeatures],
    request: Request,
    model=Depends(get_predictor),  # noqa: B008
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not items:
        raise HTTPException(status_code=422, detail="Batch must not be empty")

    results = [_predict_one(f, model) for f in items]
    request_id = request.headers.get("X-Request-ID", "n/a")
    logger.info(
        "batch_prediction",
        extra={"request_id": request_id, "batch_size": len(results)},
    )
    return results
