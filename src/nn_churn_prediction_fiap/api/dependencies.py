from typing import Any

_predictor: Any = None


def get_predictor() -> Any:
    return _predictor


def set_predictor(model: Any) -> None:
    global _predictor
    _predictor = model
