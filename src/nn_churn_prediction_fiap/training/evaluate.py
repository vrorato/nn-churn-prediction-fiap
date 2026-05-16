import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from nn_churn_prediction_fiap.config import (
    OPERATING_THRESHOLD,
    RECALL_AT_PRECISION_THRESHOLD,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = OPERATING_THRESHOLD,
    recall_at_precision: float = RECALL_AT_PRECISION_THRESHOLD,
) -> dict[str, float]:
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    recall_at_p = _recall_at_fixed_precision(y_true, y_proba, recall_at_precision)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        f"recall_at_p{int(recall_at_precision * 100)}": float(recall_at_p),
    }


def _recall_at_fixed_precision(
    y_true: np.ndarray, y_proba: np.ndarray, min_precision: float
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns arrays in decreasing threshold order;
    # find all thresholds where precision >= min_precision and take max recall
    mask = precision >= min_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())
