import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nn_churn_prediction_fiap.features.pipeline import make_preprocessor
from nn_churn_prediction_fiap.training.torch_loop import fit_model, get_device
from nn_churn_prediction_fiap.utils.seed import set_seed


class MLPModule(nn.Module):
    """
    Configurable MLP: Linear → BatchNorm1d → ReLU → Dropout, repeated per hidden layer.
    Final layer outputs a single logit (use BCEWithLogitsLoss, not sigmoid here).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around MLPModule + fit_model.
    Receives already-preprocessed numpy X from the Pipeline's ColumnTransformer.
    Does an internal stratified split for early-stopping validation.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        lr: float = 1e-3,
        pos_weight: float = 2.7,
        max_epochs: int = 150,
        patience: int = 10,
        batch_size: int = 256,
        val_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        set_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=self.val_fraction,
            stratify=y,
            random_state=self.random_state,
        )

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])

        module = MLPModule(
            input_dim=self.n_features_in_,
            hidden_sizes=tuple(self.hidden_sizes),
            dropout=self.dropout,
        )

        self.module_ = fit_model(
            model=module,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            lr=self.lr,
            pos_weight=self.pos_weight,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            device=get_device(),
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        device = get_device()
        self.module_.eval()
        self.module_.to(device)
        with torch.no_grad():
            logits = self.module_(torch.tensor(X, device=device))
            proba_pos = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_mlp(
    random_state: int = 42,
    hidden_sizes: tuple[int, ...] = (64, 32),
    dropout: float = 0.3,
    lr: float = 1e-3,
    pos_weight: float = 2.7,
    max_epochs: int = 150,
    patience: int = 10,
    batch_size: int = 256,
) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                TorchMLPClassifier(
                    hidden_sizes=hidden_sizes,
                    dropout=dropout,
                    lr=lr,
                    pos_weight=pos_weight,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    random_state=random_state,
                ),
            ),
        ]
    )


MLP_CONFIGS: dict[str, dict] = {
    "mlp-small": {
        "hidden_sizes": (64, 32),
        "dropout": 0.3,
        "lr": 1e-3,
        "pos_weight": 2.7,
    },
    "mlp-medium": {
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.3,
        "lr": 1e-3,
        "pos_weight": 2.7,
    },
    "mlp-high-dropout": {
        "hidden_sizes": (128, 64),
        "dropout": 0.5,
        "lr": 5e-4,
        "pos_weight": 2.7,
    },
    "mlp-weighted": {
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.3,
        "lr": 1e-3,
        "pos_weight": 3.5,
    },
}
