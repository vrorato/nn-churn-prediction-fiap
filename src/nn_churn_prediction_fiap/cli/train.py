import argparse

import mlflow

from nn_churn_prediction_fiap.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_FOLDS,
    RANDOM_STATE,
)
from nn_churn_prediction_fiap.data.load import load_raw
from nn_churn_prediction_fiap.models.baseline import BASELINES
from nn_churn_prediction_fiap.models.mlp import MLP_CONFIGS, make_mlp
from nn_churn_prediction_fiap.training.cv import run_cv
from nn_churn_prediction_fiap.utils.logging import get_logger
from nn_churn_prediction_fiap.utils.seed import set_seed

logger = get_logger(__name__)

_ALL_MODELS = list(BASELINES.keys()) + list(MLP_CONFIGS.keys())


def _mlp_factory(config: dict):
    def factory(random_state: int = RANDOM_STATE):
        return make_mlp(random_state=random_state, **config)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a churn model with CV.")
    parser.add_argument("--model", default="logreg", choices=_ALL_MODELS)
    parser.add_argument("--max-rows", type=int, default=None, dest="max_rows")
    parser.add_argument("--max-epochs", type=int, default=None, dest="max_epochs")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS, dest="n_folds")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_raw()
    if args.max_rows:
        df = df.sample(
            n=min(args.max_rows, len(df)), random_state=RANDOM_STATE
        ).reset_index(drop=True)
        logger.info("subsampled data", extra={"n_rows": len(df)})

    if args.model in BASELINES:
        factory = BASELINES[args.model]
        family = "baseline"
        extra_params = None
    else:
        config = {**MLP_CONFIGS[args.model]}
        if args.max_epochs is not None:
            config["max_epochs"] = args.max_epochs
        factory = _mlp_factory(config)
        family = "mlp"
        extra_params = config

    run_cv(
        model_factory=factory,
        df=df,
        model_name=args.model,
        model_family=family,
        extra_params=extra_params,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
