SHELL := /bin/bash
ROOT_DIR := $(shell pwd)
MLFLOW_URI := sqlite:///$(ROOT_DIR)/mlflow.db

# Resolve uv: use from PATH if already installed, otherwise fall back to the
# default install location so subsequent targets work even after a fresh install.
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: install setup lint format test train serve mlflow docker

# ── install ──────────────────────────────────────────────────────────────────
# First-time setup after cloning. Installs uv if missing, then syncs deps.
# Linux / macOS / Git-Bash / WSL:
#   make install
# Windows (PowerShell, no make): run the two commands printed by `make install`
#   manually or use WSL.
install:
ifeq ($(OS),Windows_NT)
	@echo "Detected Windows. Installing uv via PowerShell..."
	powershell -ExecutionPolicy ByPass -Command \
	  "if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { irm https://astral.sh/uv/install.ps1 | iex }"
	uv sync --all-extras --group dev
	uv run pre-commit install
else
	@command -v uv >/dev/null 2>&1 \
	  || (echo "uv not found — installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	$(UV) sync --all-extras --group dev
	$(UV) run pre-commit install
endif

# ── dev helpers ───────────────────────────────────────────────────────────────
setup:
	$(UV) sync --all-extras --group dev

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests

test:
	$(UV) run pytest

# ── model ─────────────────────────────────────────────────────────────────────
train:
	$(UV) run python -m nn_churn_prediction_fiap.cli.train

mlflow:
	$(UV) run mlflow ui --backend-store-uri $(MLFLOW_URI) --port 5000

# ── serving ───────────────────────────────────────────────────────────────────
serve:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(UV) run uvicorn nn_churn_prediction_fiap.api.main:app \
	  --host 0.0.0.0 --port 8000 --reload

docker:
	docker compose up --build
