import copy

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from nn_churn_prediction_fiap.utils.logging import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    pos_weight: float = 2.7,
    max_epochs: int = 150,
    patience: int = 10,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Train `model` with early stopping on validation PR-AUC.
    Logs per-epoch metrics to the active MLflow run (if any).
    Returns the model restored to its best-validation-PR-AUC weights.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_pr_auc = -1.0
    best_state: dict = {}
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_pr_auc = _val_pr_auc(model, val_loader, device)

        scheduler.step(val_pr_auc)

        if mlflow.active_run():
            mlflow.log_metrics(
                {"epoch_train_loss": train_loss, "epoch_val_pr_auc": val_pr_auc},
                step=epoch,
            )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    "early stopping",
                    extra={"epoch": epoch, "best_val_pr_auc": best_pr_auc},
                )
                break

    model.load_state_dict(best_state)
    return model


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        n += len(X_batch)
    return total_loss / n


def _val_pr_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            all_logits.append(model(X_batch.to(device)).cpu())
            all_labels.append(y_batch)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    proba = torch.sigmoid(torch.tensor(logits)).numpy()
    return float(average_precision_score(labels, proba))
