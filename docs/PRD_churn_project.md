# PRD — Telco Customer Churn Prediction

**Owner:** [you] · **Duration:** 5 weeks (~10h/week) · **Status:** Draft v1.0

---

## 1. Problem & context

A telecommunications operator is losing customers at an accelerated rate. Reactive retention (calling customers *after* they cancel) is too late and too expensive. The business needs a **predictive model that flags customers at risk of churn early enough for the retention team to intervene**.

This project delivers an end-to-end ML system: from data exploration to a served API, with all reproducibility, testing, and documentation practices a real production team would expect.

## 2. Stakeholders

| Stakeholder | Interest | Success looks like |
|---|---|---|
| Retention team | Daily list of high-risk customers, ranked | They work the top-N list and save customers |
| Finance / Board | ROI on retention spend | Avoided churn cost > model + campaign cost |
| Data / ML team | Maintainable, monitorable model | Reproducible, tested, documented |
| Customer | Relevant offers, not spam | High precision at the top of the ranking |

## 3. Business metrics & SLOs

**Business metrics**
- **Avoided churn cost** = `TP × avg_customer_LTV − (TP + FP) × retention_offer_cost`
- **Precision@K** at the top decile (the retention team has limited capacity)
- **Lift over random** in the top decile

**Technical metrics** (in priority order)
1. **PR-AUC** — primary, because the positive class (churn) is the minority and the cost of imbalance matters
2. **ROC-AUC** — secondary, for ranking quality independent of threshold
3. **F1 at chosen threshold** — operational metric for the deployed cutoff
4. **Recall @ fixed precision** (e.g., recall when precision ≥ 0.7) — operational realism

**SLOs for the served model**
- API latency p95 < 200ms for single prediction
- Availability ≥ 99% (single-instance scope; HA is out of scope)
- Drift alert if PSI on top features > 0.2 weekly
- Performance alert if rolling PR-AUC on labeled feedback drops > 5% vs validation

## 4. Scope

**In scope**
- Tabular model on Telco Customer Churn (IBM, Kaggle)
- MLP in PyTorch as the central deliverable
- sklearn baselines (DummyClassifier, LogisticRegression, plus a tree baseline for fairness)
- MLflow tracking for every experiment
- FastAPI service with `/predict`, `/health`
- Tests, linting, structured logging, model card, monitoring plan
- README + 5-min STAR video

**Out of scope (for v1)**
- Streaming / real-time event ingestion
- Multi-model A/B in production
- Retraining automation (we'll *document* the plan, not build it)
- Cloud deploy is optional, not required

## 5. Data

- **Source:** [Telco Customer Churn — IBM (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,043 rows, ~21 columns
- **Target:** `Churn` (Yes/No), ~26% positive class
- **Known issue:** `TotalCharges` has whitespace/empty values for tenure=0 — handle in EDA
- **Versioning:** hash the raw CSV, log the hash as an MLflow tag (`dataset_sha256`) on every run

## 6. Mandatory non-functional requirements

- Fixed seeds everywhere (`numpy`, `random`, `torch`, sklearn `random_state`)
- Stratified K-fold (5 folds) for model selection
- Model Card at `docs/model_card.md` (Google/HuggingFace format)
- ≥3 automated tests: **smoke** (training pipeline runs end-to-end on a sample), **schema** (pandera on inputs), **API** (TestClient hits `/predict` and `/health`)
- Structured JSON logging — `print()` is banned, enforced by ruff `T201`
- `ruff check` exits 0; `ruff format` applied
- Layout: `src/`, `data/{raw,interim,processed}/`, `models/`, `tests/`, `notebooks/`, `docs/`
- `pyproject.toml` + `Makefile` (`make lint test run`)

## 7. Architecture (target)

```
                  ┌────────────────────────────────────────┐
                  │   data/raw  (versioned by sha256 tag)  │
                  └───────────────────┬────────────────────┘
                                      ▼
              ┌─────────────────────────────────────────────┐
              │ src/data:  loaders + pandera schemas        │
              │ src/features: sklearn ColumnTransformer +   │
              │               custom transformers           │
              └───────────────────┬─────────────────────────┘
                                  ▼
        ┌────────────────────────────────────────────────────┐
        │ src/models:                                        │
        │   - baseline.py  (Dummy, LogReg, RF/GBM)           │
        │   - mlp.py       (PyTorch MLP + training loop)     │
        │ src/training: CV runner, early stopping, MLflow    │
        └───────────────────┬────────────────────────────────┘
                            ▼
          ┌─────────────────────────────────────────────────┐
          │ MLflow tracking server (local file backend)     │
          │  - params, metrics, dataset hash, model artifact│
          └───────────────────┬─────────────────────────────┘
                              ▼
            ┌──────────────────────────────────────────┐
            │ src/api (FastAPI):                       │
            │  - /health, /predict (Pydantic schemas)  │
            │  - structured JSON logs + latency mw     │
            │  - loads pinned model from MLflow URI    │
            └──────────────────────────────────────────┘
```

## 7b. Folder structure

```
telco-churn/
├── .github/
│   └── workflows/
│       └── ci.yml                    # ruff + pytest on push/PR (optional but cheap)
├── .pre-commit-config.yaml           # ruff format + check on commit
├── .gitignore                        # excludes data/raw/, mlruns/, *.pkl, .venv/
├── .python-version                   # pins Python (e.g., 3.11)
├── pyproject.toml                    # deps, ruff config, pytest config, build metadata
├── Makefile                          # setup | lint | format | test | train | serve | docker
├── Dockerfile                        # multi-stage: builder → slim runtime
├── docker-compose.yml                # API + MLflow for local end-to-end
├── README.md                         # 30-sec pitch, quickstart, architecture diagram
│
├── data/
│   ├── raw/                          # original CSV, gitignored; sha256 logged to MLflow
│   ├── interim/                      # post-cleaning, pre-feature-engineering (gitignored)
│   ├── processed/                    # ready-for-model splits (gitignored)
│   └── README.md                     # dataset source, license, sha256 of raw file
│
├── notebooks/
│   ├── 01_eda.ipynb                  # exploration + data readiness verdict
│   ├── 02_baseline_results.ipynb     # reads MLflow runs, renders comparison plots
│   ├── 03_mlp_experiments.ipynb      # threshold sweeps, cost trade-off curves
│   └── 04_bias_analysis.ipynb        # slice metrics for the model card
│   # Rule: notebooks import from src/, contain NO function defs, only calls + plots
│
├── src/
│   └── telco_churn/                  # importable package (matches pyproject name)
│       ├── __init__.py
│       │
│       ├── config.py                 # paths, constants, threshold defaults — single source
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logging.py            # get_logger() returning JSON formatter
│       │   ├── seed.py               # set_seed() for numpy/random/torch/PYTHONHASHSEED
│       │   └── mlflow_helpers.py     # tag helpers: dataset_sha256, git_commit
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py             # pandera DataFrameSchema for raw input
│       │   ├── load.py               # load + validate against schema
│       │   └── split.py              # stratified splits, returns indices for reproducibility
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── transformers.py       # custom sklearn transformers (TenureBucket, etc.)
│       │   └── pipeline.py           # ColumnTransformer factory (numeric/categorical paths)
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline.py           # Dummy, LogReg, GBM — all as sklearn Pipelines
│       │   └── mlp.py                # nn.Module + sklearn-compatible wrapper
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── cv.py                 # stratified K-fold runner, logs to MLflow
│       │   ├── torch_loop.py         # training loop: early stopping, scheduler, clipping
│       │   └── evaluate.py           # metrics: PR-AUC, ROC-AUC, F1, recall@precision
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predictor.py          # loads model from MLflow URI, exposes .predict_proba
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py               # FastAPI app: /health, /predict
│       │   ├── schemas.py            # Pydantic models: CustomerFeatures, PredictionResponse
│       │   ├── middleware.py         # latency logging, request_id injection
│       │   └── dependencies.py       # model loader as FastAPI dependency
│       │
│       └── cli/
│           ├── __init__.py
│           ├── train.py              # `python -m telco_churn.cli.train`
│           └── predict.py            # batch scoring entry point
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # shared fixtures: sample df, trained mini-model
│   ├── test_schema.py                # pandera rejects bad input (required test #2)
│   ├── test_smoke.py                 # train --max-rows=500 --max-epochs=2 (required test #1)
│   ├── test_api.py                   # TestClient: /health, /predict happy + 422 (required #3)
│   ├── test_features.py              # custom transformers behave correctly
│   └── test_training.py              # seed determinism: same seed → same metrics
│
├── models/
│   └── .gitkeep                      # local artifacts; production models live in MLflow
│
├── mlruns/                           # MLflow file backend; gitignored
│
├── scripts/
│   ├── download_data.sh              # idempotent: fetch Kaggle CSV, verify sha256
│   └── run_full_pipeline.sh          # data → baselines → mlp → register (for the video demo)
│
└── docs/
    ├── PRD.md                        # this document
    ├── model_card.md                 # HuggingFace-style: intended use, metrics, bias, limits
    ├── model_comparison.md           # results table: MLP vs baselines, mean±std across folds
    ├── architecture.md               # batch vs real-time decision + justification
    ├── monitoring.md                 # metrics, alerts, response playbook
    └── images/
        └── architecture.png          # the diagram from section 7, exported
```

**Conventions worth defending in a review:**

- **`src/telco_churn/` not just `src/`.** The extra level lets the package be `pip install -e .` and imported as `from telco_churn.data import load` from anywhere — including notebooks and tests — without `sys.path` hacks.
- **`data/raw/` is gitignored.** Reproducibility comes from the sha256 + the download script, not from committing CSVs. The model card cites the hash.
- **Notebooks are thin.** All logic lives in `src/`. A notebook that defines functions is a notebook that will silently diverge from the package. Enforce this in code review (or in your own discipline).
- **`models/` exists but is mostly empty.** Real model artifacts live in MLflow. The folder is there for one-off exports (ONNX, the registered model pulled for Docker builds).
- **`inference/` is separate from `api/`.** The API is one consumer. A batch scoring job is another. Both call `inference/predictor.py`. Don't couple business logic to FastAPI.
- **`cli/` over loose scripts.** Entry points via `python -m telco_churn.cli.train` get tested, type-checked, and packaged. Loose scripts in `scripts/` are for one-shot ops (data download, full demo run).

## 8. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MLP doesn't beat tree baseline | High | Medium | That's fine — *report it honestly* in the model card. Pick whichever wins; the project's value is the rigor, not MLP supremacy. |
| Class imbalance hurts loss | Medium | Medium | `pos_weight` in `BCEWithLogitsLoss`; PR-AUC as the selection metric |
| Data leakage via target encoding / scaler fit on full set | Medium | High | Fit transformers **inside CV folds** via sklearn `Pipeline` |
| Scope creep on the API/deploy | Medium | Medium | Cloud deploy is optional; cut it before extending sprints |
| Over-engineering tests | Low | Low | Cap at 3 mandatory + extras only if free |

---

# Sprint plan

Five one-week sprints. Each ends with a **demo-able artifact** and a **definition of done** you can check yourself against. Friday is review/buffer day.

## Sprint 0 — Foundations (Days 1–2 of week 1, ~3h)

Get the boring stuff done once so it never gets in your way.

**Tasks**
- `git init`, create the directory skeleton, push to GitHub
- `pyproject.toml` with deps grouped (`core`, `dev`, `api`)
- `ruff` config in pyproject (lines, target-version, select rules incl. `T201` to ban `print`)
- `Makefile` targets: `setup`, `lint`, `format`, `test`, `train`, `serve`
- `.pre-commit-config.yaml` running ruff
- Logger module: `src/utils/logging.py` returning a JSON-formatted logger
- Seed util: `src/utils/seed.py` that sets numpy/random/torch/PYTHONHASHSEED in one call
- Download dataset to `data/raw/`, add to `.gitignore`, commit the sha256 in a `data/README.md`
- Create an empty MLflow experiment, confirm `mlflow ui` runs

**DoD:** `make lint test` passes on an empty test file. `python -c "from src.utils.logging import get_logger; get_logger('x').info('hi')"` outputs JSON.

---

## Sprint 1 — EDA & baselines (Week 1, days 3-5, ~7h)

**Goal:** understand the data and establish the floor that the MLP must beat.

**Tasks**
- `notebooks/01_eda.ipynb`: volume, dtypes, missingness (esp. `TotalCharges`), target balance, univariate distributions by churn, correlation, leakage check (anything that looks too predictive — e.g., `Contract` type)
- A short **"data readiness"** section at the end of the notebook: green/yellow/red on volume, label quality, drift expectations
- `src/data/schema.py`: pandera `DataFrameSchema` for the raw input
- `src/data/load.py`: loader that validates against the schema
- `src/models/baseline.py`: function returning a sklearn `Pipeline` for each baseline (Dummy stratified, LogReg with scaling, plus RandomForest or GradientBoosting — having one tree baseline now saves time in Sprint 2)
- `src/training/cv.py`: stratified 5-fold runner that logs to MLflow (params, all 4 metrics, dataset hash tag, fold index)
- Run all baselines, capture results

**MLflow conventions** (lock these now, you'll thank yourself):
- One **experiment** = the project. One **run** = one model config. Nested runs for folds.
- Tags: `dataset_sha256`, `git_commit`, `model_family` (baseline|mlp), `stage` (sprint1|sprint2|...)
- Metrics logged: `pr_auc`, `roc_auc`, `f1`, `recall_at_p70` — both per-fold and mean/std

**DoD:** EDA notebook tells a story (not just charts). MLflow UI shows ≥3 baseline runs with all 4 metrics. You can state out loud: *"The model needs to beat PR-AUC of X to be worth deploying."*

---

## Sprint 2 — MLP in PyTorch (Week 2, ~10h)

**Goal:** Build the MLP rigorously and compare honestly.

**Tasks**
- `src/models/mlp.py`: `nn.Module` MLP. Don't overthink architecture — 2-3 hidden layers, ReLU, dropout, BatchNorm or LayerNorm. Sigmoid output via `BCEWithLogitsLoss` (numerically stable, supports `pos_weight`).
- `src/training/torch_loop.py`: training loop with
  - `DataLoader` from a `TensorDataset` (after sklearn pipeline `.transform`)
  - Early stopping on validation PR-AUC (patience ~10)
  - LR scheduler (cosine or plateau)
  - Gradient clipping
  - Per-epoch MLflow metric logging
- Hyperparameter sweep: pick **3-5 configs** by hand or with Optuna (small budget — you have hours, not days). Vary: hidden sizes, dropout, lr, `pos_weight`.
- Run all configs through the same stratified CV runner as baselines — **same folds, same seed**. This is the only fair comparison.
- **Cost trade-off analysis**: pick a plausible `LTV` and `offer_cost`, sweep the decision threshold, plot expected profit vs threshold for each model. The chosen operating threshold goes into the model card.
- **Comparison table** in `docs/model_comparison.md` with mean±std for each metric across all models. Bold the winner per metric.

**Honest expectations:** on Telco Churn (small, tabular, ~7k rows), a well-tuned gradient boosting model often matches or beats an MLP. **That's a finding, not a failure.** Document it. The MLP is still the central deliverable per the brief — register it, ship it via API — but the comparison must be honest.

**DoD:** MLflow has the MLP runs alongside baselines, all comparable. `docs/model_comparison.md` exists. You've picked the model that will be served and registered it in MLflow Model Registry with a `Staging` alias.

---

## Sprint 3 — Refactor, pipeline, tests (Week 3, ~10h)

**Goal:** Turn the notebook-grade code into a package.

**Tasks**
- **Refactor pass.** Move anything still living in notebooks into `src/`. Notebooks become *thin* — they import from `src` and visualize. Rule of thumb: a notebook should have no function definitions, only calls and plots.
- **Reproducible pipeline:** a single `sklearn.Pipeline` (or `ColumnTransformer` + estimator) that goes raw → prediction. For the MLP, wrap it in a custom estimator that exposes `.fit/.predict/.predict_proba` so it composes the same way (or use `skorch` if you don't want to write the wrapper — your call).
- **Custom transformers** for any feature engineering (e.g., a `TenureBucket` transformer) inheriting `BaseEstimator, TransformerMixin`.
- **Tests in `tests/`:**
  1. `test_schema.py` — pandera schema rejects bad input (missing column, wrong dtype, out-of-range value)
  2. `test_smoke.py` — `train --max-rows=500 --max-epochs=2` runs end-to-end without error and produces an MLflow run
  3. `test_api.py` — FastAPI `TestClient`: `/health` returns 200; `/predict` with a valid Pydantic body returns a probability in [0,1]; an invalid body returns 422
- **Hit ≥80% coverage** on `src/data` and `src/api` (the parts that matter for correctness). Don't chase coverage on training loops — diminishing returns.
- `pyproject.toml` complete; `make lint` and `make test` green.

**DoD:** A new contributor could clone, run `make setup test`, and have everything pass. No `print()` anywhere — `ruff check` enforces it.

---

## Sprint 4 — API & serving (Week 4, ~10h)

**Goal:** Ship a usable service.

**Tasks**
- `src/api/main.py`: FastAPI app
  - `GET /health` → `{"status": "ok", "model_version": "...", "git_commit": "..."}`
  - `POST /predict` → Pydantic `CustomerFeatures` in, `{"churn_probability": float, "churn_flag": bool, "threshold": float, "model_version": str}` out
  - Optional `POST /predict/batch` for lists
- **Latency middleware**: log p50/p95 to structured logs per request; expose simple counters
- **Model loading**: load from MLflow Model Registry by URI on startup (e.g., `models:/churn-mlp/Production`). Don't hardcode paths.
- **Structured logging** in every handler: `logger.info("prediction", extra={"request_id": ..., "latency_ms": ..., "prob": ...})`
- `Dockerfile` (multi-stage: builder + slim runtime). Healthcheck pointed at `/health`.
- `docker-compose.yml` running API + MLflow if you want local end-to-end
- Manual smoke: `curl` examples in README

**Pydantic tips for this dataset**
- Use `Literal[...]` for the categorical fields (`Contract`, `PaymentMethod`, etc.) so invalid values are rejected at the boundary, not at the model
- `Field(ge=0)` on numerics like `tenure`, `MonthlyCharges`
- Return informative 422 errors — FastAPI does this automatically if you let it

**DoD:** `docker compose up` → API responds in <200ms p95 on local. `pytest tests/test_api.py` passes. Logs are JSON.

---

## Sprint 5 — Documentation, model card, video (Week 5, ~10h)

**Goal:** Make it presentable and finishable.

**Tasks**
- **Model card** at `docs/model_card.md` (use the HuggingFace template as a starting point):
  - Intended use & out-of-scope use
  - Training data, evaluation data, metrics (with confidence intervals)
  - **Bias analysis**: slice metrics by `gender`, `SeniorCitizen`, `Partner`, contract type. Report disparities honestly.
  - Limitations: small dataset, single point in time, no concept drift studied, US-style telecom only
  - Failure modes: new tenure values, missing fields, distribution shift
- **Deployment architecture decision** at `docs/architecture.md`: batch (nightly scoring → CRM list) vs real-time API. Pick one, *justify it with the business context* (retention team workflow is batch-natural, but real-time is more flexible). One page max.
- **Monitoring plan** at `docs/monitoring.md`:
  - Metrics: input drift (PSI per feature), prediction drift (proba distribution), performance on labeled feedback (rolling PR-AUC)
  - Alerts: thresholds, who gets paged
  - Playbook: "if drift alert fires, do X; if performance drops, do Y"
- **README** rewrite: 30-second pitch, architecture diagram (the one from this PRD), quickstart (`make setup train serve`), example `curl`, link to model card
- **STAR video (5 min)**:
  - **S**ituation: telco losing customers, reactive retention failing
  - **T**ask: predictive churn model + production-grade delivery
  - **A**ction: walk through architecture, show MLflow comparison, show API call live
  - **R**esult: metrics achieved, what you'd do next (retraining, online A/B)
  - Record in one take if possible; perfect is the enemy of done
- **(Optional) Cloud deploy**: only if you have buffer. Cloud Run / Azure Container Apps / Fly.io are the lowest-friction options for a Dockerized FastAPI.

**DoD:** Repo is presentable to a stranger. Model card is honest, not marketing. Video exists. You'd be comfortable putting this in a portfolio.

---

## Cross-cutting checklist (review weekly)

- [ ] Seeds set in every entry point (`train.py`, `predict.py`, tests)
- [ ] No `print()` anywhere — only the structured logger
- [ ] Every MLflow run has `dataset_sha256` and `git_commit` tags
- [ ] CV splits are stratified and use the **same** `random_state` across all model families
- [ ] Transformers are fit inside the pipeline, never on the full dataset before splitting
- [ ] `ruff check` exits 0 before every commit (pre-commit hook handles this)
- [ ] Tests pass: `make test`

## What "good" looks like at the end

- A reviewer can clone, run two commands, see the model train and the API respond
- MLflow UI tells the story of model selection at a glance
- The model card admits limitations a less rigorous project would hide
- The code passes lint, tests, and reads like it was written by someone who's done this before — because by the end of Sprint 5, you will have.
