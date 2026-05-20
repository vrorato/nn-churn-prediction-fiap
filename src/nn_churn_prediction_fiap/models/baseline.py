from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nn_churn_prediction_fiap.features.pipeline import ALL_FEATURES, make_preprocessor

__all__ = ["ALL_FEATURES", "BASELINES", "make_dummy", "make_gbm", "make_logreg"]


def make_dummy(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            ("clf", DummyClassifier(strategy="stratified", random_state=random_state)),
        ]
    )


def make_logreg(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_gbm(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=random_state,
                ),
            ),
        ]
    )


BASELINES: dict[str, callable] = {
    "dummy": make_dummy,
    "logreg": make_logreg,
    "gbm": make_gbm,
}
