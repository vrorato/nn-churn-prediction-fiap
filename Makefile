ROOT_DIR := $(shell pwd)
MLFLOW_URI := sqlite:///$(ROOT_DIR)/mlflow.db

.PHONY: setup lint format test train serve mlflow

setup:
	uv sync --all-extras --group dev

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

train:
	uv run python -m nn_churn_prediction_fiap.cli.train

mlflow:
	uv run mlflow ui --backend-store-uri $(MLFLOW_URI) --port 5000

serve:
	uv run uvicorn nn_churn_prediction_fiap.api.main:app --host 0.0.0.0 --port 8000 --reload
