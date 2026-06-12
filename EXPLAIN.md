# How This Codebase Works

End-to-end walkthrough from raw data to live predictions.

---

## Big Picture

```
data/raw/raw.csv
      │
      ▼
data/load.py          ← ingest + validate + clean
      │
      ▼
features/pipeline.py  ← encode + scale
      │
      ├── models/baseline.py   ← Dummy / LogReg / GBM
      └── models/mlp.py        ← PyTorch MLP (sklearn-wrapped)
                │
                ▼
        training/torch_loop.py ← Adam, early stop, gradient clip
                │
                ▼
        training/cv.py         ← StratifiedKFold + MLflow logging
                │
                ▼
        mlflow.db              ← model registry
                │
                ▼
        inference/predictor.py ← load model by alias
                │
                ▼
        api/main.py            ← FastAPI /predict endpoint
```

---

## 1. Central Config — `src/.../config.py`

Everything that changes between environments lives here as constants:

| Constant | Value | Purpose |
|---|---|---|
| `RAW_DATA_PATH` | `data/raw/raw.csv` | Where the CSV lives |
| `TARGET_COL` | `"Churn"` | Label column name |
| `RANDOM_STATE` | `42` | Seeds every random step |
| `N_FOLDS` | `5` | Stratified K-fold splits |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Local SQLite experiment store |
| `OPERATING_THRESHOLD` | `0.35` | Probability cutoff for the binary flag |
| `RECALL_AT_PRECISION_THRESHOLD` | `0.70` | Business KPI: recall when precision ≥ 70% |
| `DATASET_SHA256` | `88be4b…` | Fingerprint attached to every MLflow run |

The threshold is 0.35 (not the default 0.5) because churn is costly to miss — lowering it trades precision for recall.

---

## 2. Data Ingestion — `src/.../data/`

### `data/schema.py`

Defines `raw_schema` using **Pandera**. Every column has its type, allowed values, and nullability declared. Examples:

- `gender`: must be `"Male"` or `"Female"`, never null
- `TotalCharges`: declared as `str` because the raw CSV stores whitespace for customers with `tenure = 0`
- `Churn`: must be `"Yes"` or `"No"` — converted to int later

`strict=False` means extra future columns won't crash validation.

### `data/load.py`

`load_raw()` does three things in order:

1. **Read** — `pd.read_csv(..., dtype={"TotalCharges": str})` forces the column to stay as text so whitespace rows don't silently become `NaN` before validation.
2. **Validate** — `raw_schema.validate(df)` raises an error if any cell violates the schema. Runs before any transformation so you catch data drift early.
3. **Clean** (`_clean`) — strips whitespace and coerces `TotalCharges` to float, drops the ~11 rows where it cannot be parsed (tenure=0 rows with no total), then converts `Churn` `"Yes"/"No"` → `1/0`.

After this function, the DataFrame is clean and ready for scikit-learn.

---

## 3. Feature Engineering — `src/.../features/`

### `features/pipeline.py`

`make_preprocessor()` returns a **sklearn `ColumnTransformer`** with two branches:

| Branch | Columns | Transform |
|---|---|---|
| `"num"` | `tenure`, `MonthlyCharges`, `TotalCharges` | `StandardScaler` (zero mean, unit variance) |
| `"cat"` | all binary + categorical columns | `OneHotEncoder(handle_unknown="ignore")` |

`remainder="drop"` discards `customerID` and any column not explicitly listed, so no ID leaks into the model.

`handle_unknown="ignore"` means a new category at inference time becomes an all-zero vector instead of raising an error.

### `features/transformers.py`

`TenureBucket` is a custom sklearn-compatible transformer (implements `fit`/`transform`/`get_feature_names_out`). It bins `tenure` (months) into four ordinal buckets: `0-12`, `13-24`, `25-48`, `49+`. Designed to be plugged into a pipeline step if bucket-level features are needed experimentally.

---

## 4. Models — `src/.../models/`

Both files follow the same pattern: return a **`Pipeline([("pre", preprocessor), ("clf", classifier)])`** so training and inference always apply the same preprocessing automatically.

### `models/baseline.py`

Three baselines:

| Name | Classifier | Why it exists |
|---|---|---|
| `dummy` | `DummyClassifier(strategy="stratified")` | Lower bound — random guessing proportional to class balance |
| `logreg` | `LogisticRegression(class_weight="balanced")` | Strong linear baseline; `balanced` compensates for ~73/27 class split |
| `gbm` | `GradientBoostingClassifier(n_estimators=200, lr=0.05, depth=4)` | Tree-based upper bound for classical ML |

### `models/mlp.py`

Two classes, one factory:

**`MLPModule(nn.Module)`** — the PyTorch network architecture:
- Stacks `Linear → BatchNorm1d → ReLU → Dropout` per hidden layer
- Final layer outputs a **single logit** (no sigmoid here — that's applied by the loss function)
- `BatchNorm1d` stabilises training when features have different scales; `Dropout` prevents overfitting on the ~7000-row dataset

**`TorchMLPClassifier(BaseEstimator, ClassifierMixin)`** — makes the network look like any sklearn estimator:
- `fit()`: splits off 15% as an internal validation set for early stopping, builds `MLPModule`, calls `fit_model()`
- `predict_proba()`: sets `module_.eval()`, runs a forward pass on CPU or GPU, applies sigmoid to convert logits to probabilities
- `predict()`: threshold at 0.5 (the CV threshold; the API uses `OPERATING_THRESHOLD=0.35`)

**`make_mlp()`** — assembles the full Pipeline.

**`MLP_CONFIGS`** — named hyperparameter presets (`mlp-small`, `mlp-medium`, `mlp-high-dropout`, `mlp-weighted`) selectable from the CLI.

---

## 5. Training Loop — `src/.../training/torch_loop.py`

`fit_model()` is the low-level PyTorch training function. It receives numpy arrays and a `nn.Module`.

Key decisions:

| Decision | Reason |
|---|---|
| `BCEWithLogitsLoss(pos_weight=2.7)` | Class imbalance: ~27% churn. `pos_weight` penalises false negatives more heavily |
| `Adam` optimizer | Default choice; works well on small tabular datasets |
| `ReduceLROnPlateau(mode="max", factor=0.5, patience=5)` | Halves LR when val PR-AUC stops improving for 5 epochs |
| Early stopping on **PR-AUC** (`patience=10`) | PR-AUC is the primary business metric; stops before overfitting |
| Gradient clipping (`max_norm=1.0`) | Prevents exploding gradients in deep layers |
| `copy.deepcopy(model.state_dict())` on best epoch | Restores best weights after early stopping, not last epoch |

If an MLflow run is active, every epoch's `train_loss` and `val_pr_auc` are logged as step metrics.

---

## 6. Cross-Validation + Experiment Tracking — `src/.../training/cv.py` + `evaluate.py`

### `cv.py` — `run_cv()`

Runs **Stratified 5-fold CV** so each fold has the same churn ratio. Structure in MLflow:

```
parent run: "logreg"          ← mean/std metrics, tags, params
  ├── nested run: "logreg_fold0"  ← per-fold metrics
  ├── nested run: "logreg_fold1"
  ...
```

Tags attached to every parent run include `dataset_sha256` so you can always trace which dataset version produced a run.

### `evaluate.py` — `compute_metrics()`

Four metrics per fold:

| Metric | Description |
|---|---|
| `pr_auc` | Area under Precision-Recall curve — primary KPI |
| `roc_auc` | Area under ROC curve |
| `f1` | F1 at `OPERATING_THRESHOLD=0.35` |
| `recall_at_p70` | Max recall achievable while keeping precision ≥ 70% |

`recall_at_p70` is a business constraint: "catch as many churners as you can, but don't spam more than 30% false positives".

---

## 7. CLI Entry Point — `src/.../cli/train.py`

Invoked as `python -m nn_churn_prediction_fiap.cli.train` or `make train`.

```
--model     logreg | gbm | dummy | mlp-small | mlp-medium | …
--max-rows  subsample for quick smoke tests
--max-epochs  override MLP epochs (useful in CI)
--n-folds   override fold count
```

The script wires everything together: seeds → load data → pick factory → `run_cv()`.

---

## 8. Inference — `src/.../inference/predictor.py`

`load_model(alias="Staging")` fetches the model from the **MLflow Model Registry**:

```
uri = "models:/churn-mlp@Staging"
model = mlflow.sklearn.load_model(uri)
```

The loaded object is the full sklearn `Pipeline` (preprocessor + classifier). It accepts a raw `pd.DataFrame` with the same column names as the training data — no manual preprocessing needed at inference time.

The alias-based loading (`@Staging`, `@Production`) means you can promote a new model version in the registry without changing any application code.

---

## 9. API — `src/.../api/`

Built with **FastAPI** and served by **Uvicorn** (`make serve`).

### `api/schemas.py`

Two Pydantic models:

- **`CustomerFeatures`** — the request body. Each field mirrors the raw CSV schema. Categorical columns use `Literal[...]` types so FastAPI validates allowed values before the model is even called.
- **`PredictionResponse`** — `churn_probability` (float), `churn_flag` (bool), `threshold` (echoed back), `model_version` (alias string).

### `api/dependencies.py`

A module-level `_predictor` variable with `get_predictor()` / `set_predictor()` accessors. FastAPI's `Depends(get_predictor)` injects the loaded model into route handlers without recreating it per request.

### `api/middleware.py` — `LatencyLoggingMiddleware`

Wraps every HTTP request. Generates a short `request_id` (UUID prefix), records `time.perf_counter()` before and after, then emits a structured JSON log line with `method`, `path`, `status_code`, and `latency_ms`.

### `api/main.py` — Routes

**Startup** (`lifespan`): calls `load_model("Staging")` once and stores it via `set_predictor()`. The app won't start serving until the model is loaded.

| Route | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "model_alias": "Staging", "git_commit": "abc1234"}` |
| `/predict` | POST | Single-customer prediction |
| `/predict/batch` | POST | List of customers, returns list of responses |

**Prediction flow** (both routes):
1. Pydantic validates the request body
2. `pd.DataFrame([features.model_dump()])` converts the Pydantic model to a one-row DataFrame
3. `model.predict_proba(df)[:, 1]` runs the full sklearn Pipeline (preprocess → neural net)
4. Compare probability to `OPERATING_THRESHOLD=0.35` for the binary flag
5. Log `request_id`, `churn_probability`, `churn_flag` as JSON

---

## 10. Utilities — `src/.../utils/`

### `utils/logging.py`

All loggers emit **structured JSON** (`_JsonFormatter`). Every `logger.info("event", extra={"key": val})` becomes a single JSON line with `ts`, `level`, `logger`, `message`, plus any extras. This makes logs grep-able and easy to ingest into observability tools.

### `utils/seed.py`

`set_seed(42)` seeds `os.environ["PYTHONHASHSEED"]`, `random`, `numpy`, and `torch` (including CUDA) in one call. Called at the top of every training script for full reproducibility.

---

## 11. Infrastructure

| File | Purpose |
|---|---|
| `Makefile` | Shortcuts: `make train`, `make serve`, `make test`, `make mlflow`, `make docker` |
| `pyproject.toml` | Dependencies via `uv`; `api` is an optional group so the base install stays slim |
| `Dockerfile` + `docker-compose.yml` | Containerise both the API and MLflow UI |
| `mlflow.db` | SQLite file that stores all experiment runs and the model registry locally |
| `.pre-commit-config.yaml` | Runs `ruff` on every commit |

---

## Data Flow Summary

```
raw.csv
  → load_raw()           validates schema, cleans TotalCharges, encodes Churn as 0/1
  → ColumnTransformer    scales numerics, OHE categoricals, drops customerID
  → TorchMLPClassifier   internal 85/15 split, trains MLPModule with BCEWithLogitsLoss
  → fit_model()          Adam + ReduceLROnPlateau, early stop on val PR-AUC, best weights restored
  → run_cv()             5-fold loop, logs to MLflow (parent + nested child runs)
  → mlflow.db            model registered as "churn-mlp@Staging"
  → load_model()         fetches full Pipeline from registry
  → FastAPI /predict     Pydantic validates → DataFrame → pipeline.predict_proba → threshold → JSON
```
