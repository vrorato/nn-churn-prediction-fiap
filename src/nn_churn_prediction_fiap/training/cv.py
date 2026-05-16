import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from nn_churn_prediction_fiap.config import (
    DATASET_SHA256,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    RANDOM_STATE,
    TARGET_COL,
)
from nn_churn_prediction_fiap.training.evaluate import compute_metrics
from nn_churn_prediction_fiap.utils.logging import get_logger

logger = get_logger(__name__)


def run_cv(
    model_factory: callable,
    df: pd.DataFrame,
    model_name: str,
    model_family: str = "baseline",
    extra_params: dict | None = None,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """
    Run stratified K-fold CV for a given model factory.

    Returns mean metrics across folds. Logs everything to MLflow:
    - parent run: mean/std metrics, model params, tags
    - nested child run per fold: per-fold metrics
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X = df.drop(columns=[TARGET_COL, "customerID"], errors="ignore")
    y = df[TARGET_COL].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, float]] = []

    with mlflow.start_run(run_name=model_name) as parent_run:
        mlflow.set_tags(
            {
                "model_name": model_name,
                "model_family": model_family,
                "dataset_sha256": DATASET_SHA256,
                "n_folds": n_folds,
                "random_state": random_state,
            }
        )
        if extra_params:
            mlflow.log_params(extra_params)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_factory(random_state=random_state)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = compute_metrics(y_val, y_proba)
            fold_metrics.append(metrics)

            with mlflow.start_run(
                run_name=f"{model_name}_fold{fold_idx}",
                nested=True,
                tags={"fold": fold_idx, "parent_run_id": parent_run.info.run_id},
            ):
                mlflow.log_metrics({f"fold_{k}": v for k, v in metrics.items()})

            logger.info(
                "fold complete",
                extra={"model": model_name, "fold": fold_idx, **metrics},
            )

        mean_metrics = _aggregate(fold_metrics, parent_run)

    logger.info("cv complete", extra={"model": model_name, **mean_metrics})
    return mean_metrics


def _aggregate(
    fold_metrics: list[dict[str, float]], run: "mlflow.ActiveRun"
) -> dict[str, float]:
    keys = fold_metrics[0].keys()
    mean_metrics: dict[str, float] = {}
    for k in keys:
        values = np.array([m[k] for m in fold_metrics])
        mean_val = float(values.mean())
        std_val = float(values.std())
        mlflow.log_metric(f"mean_{k}", mean_val)
        mlflow.log_metric(f"std_{k}", std_val)
        mean_metrics[f"mean_{k}"] = mean_val
        mean_metrics[f"std_{k}"] = std_val
    return mean_metrics
