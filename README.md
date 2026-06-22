# Telco Customer Churn Prediction

End-to-end ML system that predicts which telecom customers are likely to churn, so a retention team can intervene before they leave. Built with PyTorch MLP + sklearn baselines, tracked with MLflow, and served via FastAPI.

---

## Architecture

```
  data/raw/ (sha256-versioned CSV)
        │
        ▼
  src/data/      ← pandera schema validation + cleaning
  src/features/  ← ColumnTransformer (scaler, OHE, TenureBucket)
        │
        ▼
  src/models/    ← MLP (PyTorch) + baselines (LogReg, GBM, Dummy)
  src/training/  ← stratified 5-fold CV, early stopping, MLflow logging
        │
        ▼
  MLflow Model Registry  ← models:/churn-mlp@Staging
        │
        ▼
  src/api/  (FastAPI)
    GET  /health
    POST /predict
    POST /predict/batch
```

---

## Prerequisites

- **Python 3.12** (managed automatically by `uv`)
- **make**
  - Linux / macOS: already available
  - Windows: use [WSL](https://learn.microsoft.com/windows/wsl/) (recommended) or install `make` via [Chocolatey](https://chocolatey.org/) (`choco install make`) / [Scoop](https://scoop.sh/) (`scoop install make`)

## Quickstart

```bash
# 1. Clone and install everything (uv + all dependencies + pre-commit hooks)
git clone <repo-url>
cd nn-churn-prediction-fiap
make install

# 2. Place the raw CSV at data/raw/raw.csv
#    Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 3. Train all models (baselines + MLP sweep, logs to MLflow)
make train

# 4. Start the API
make serve

# 5. (Optional) Open MLflow UI
make mlflow
```

> **Windows (PowerShell, no make):** run these two commands instead of `make install`:
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> uv sync --all-extras --group dev
> ```

---

## Example requests

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'

# Response:
# {
#   "churn_probability": 0.856,
#   "churn_flag": true,
#   "threshold": 0.35,
#   "model_version": "Staging"
# }
```

Interactive docs available at `http://localhost:8000/docs` when the server is running locally.

---

## Live API (Render)

The API is deployed on Render's free tier:

**https://churn-api-latest-vfru.onrender.com/docs**

> The free tier spins down after 15 minutes of inactivity. The first request after idle will take **20–30 seconds** to cold-start — just wait and retry.

---

## Model results (5-fold CV, seed=42)

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| DummyClassifier | 0.264 ± 0.007 | 0.492 | 0.255 |
| LogisticRegression | 0.658 ± 0.012 | 0.845 | 0.625 |
| GradientBoostingClassifier | **0.663 ± 0.016** | 0.844 | 0.586 |
| **MLP small (served)** | 0.651 ± 0.015 | 0.842 | 0.618 |

GBM edges out the MLP on this small tabular dataset — this is expected and reported honestly. See [`docs/model_comparison.md`](docs/model_comparison.md) for the full table.

---

## Run with Docker

```bash
# Build and start API + MLflow
make docker

# API: http://localhost:8000
# MLflow UI: http://localhost:5000
```

---

## Run tests

```bash
make test
```

8 tests covering schema validation, smoke training, and API endpoints.

---

## Documentation

| Doc | Description |
|---|---|
| [`docs/model_card.md`](docs/model_card.md) | Intended use, metrics, bias analysis, limitations |
| [`docs/model_comparison.md`](docs/model_comparison.md) | Full results table — all models, all metrics |
| [`docs/architecture.md`](docs/architecture.md) | Batch vs real-time decision + justification |
| [`docs/monitoring.md`](docs/monitoring.md) | Drift metrics, alert thresholds, response playbook |

---

## Project structure

```
src/nn_churn_prediction_fiap/
├── data/          # schema + loader
├── features/      # ColumnTransformer + TenureBucket
├── models/        # MLP (PyTorch) + baselines
├── training/      # CV runner, training loop, metrics
├── inference/     # MLflow model loader
├── api/           # FastAPI app, schemas, middleware
├── cli/           # train entry point
└── utils/         # logger, seed
```
