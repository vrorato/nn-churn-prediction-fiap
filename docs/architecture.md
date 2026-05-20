# Deployment Architecture Decision

## Decision: Real-time API (with batch-friendly interface)

The system is deployed as a **synchronous REST API** (`POST /predict`, `POST /predict/batch`) rather than a nightly batch job.

---

## Options considered

### Option A — Nightly batch scoring

A scheduled job loads the full customer table, scores all customers, writes a ranked list to a CRM or database, and the retention team works that list the next morning.

**Pros:**
- Simpler infrastructure (no persistent service, no uptime SLO)
- Efficient for large volumes — one model load, vectorised inference
- Natural fit for a "work the list tomorrow" retention workflow

**Cons:**
- 24-hour lag — a customer who triggers a cancellation event at 11 PM is not flagged until the next day's list
- Harder to integrate with real-time channels (live chat, support calls)
- Requires a scheduled job runner (Airflow, cron, etc.) in addition to the model

### Option B — Real-time API (chosen)

A persistent FastAPI service loads the model once at startup and serves predictions on demand, with sub-200 ms p95 latency.

**Pros:**
- Can be called from any CRM, support tool, or chatbot in real time
- Single entry point for both single predictions and micro-batches (`/predict/batch`)
- The `/predict/batch` endpoint makes nightly batch scoring possible too — the caller simply sends all customers in one request
- Easier to test, version, and monitor (HTTP 200/422/503 tells you immediately if something is wrong)

**Cons:**
- Requires an always-on service and a p95 < 200 ms SLO
- Model must be loaded at startup — cold-start time is ~5 s (MLflow load from SQLite)

---

## Justification

The retention team workflow is naturally batch (they work a daily list), but the API approach is chosen because:

1. **Flexibility without extra cost.** Batch scoring is a subset of the API — a nightly cron job can `POST /predict/batch` with all customers and write the result to a database. No second deployment needed.
2. **Testability.** HTTP endpoints have clear contracts (schemas, status codes, latency SLOs). A batch script is harder to test and monitor.
3. **Scope.** This is a single-instance academic deployment. An always-on FastAPI service on a small VM or container is cheaper than the operational overhead of a separate job scheduler.

If the customer base grew to millions of rows where network overhead becomes the bottleneck, the right move would be to add a dedicated batch-scoring CLI (`cli/predict.py`) that runs inference in-process — not to replace the API.

---

## Architecture diagram

```
  Caller (CRM / nightly cron / analyst)
            │
            │  HTTP POST /predict
            │       or
            │  HTTP POST /predict/batch
            ▼
  ┌─────────────────────────────┐
  │  FastAPI service            │
  │  - Pydantic validation      │
  │  - LatencyLoggingMiddleware │
  │  - GET /health              │
  └────────────┬────────────────┘
               │  model.predict_proba(df)
               ▼
  ┌─────────────────────────────┐
  │  sklearn Pipeline           │
  │  - ColumnTransformer        │
  │    (scaler + OHE)           │
  │  - TorchMLPClassifier       │
  │    (wraps nn.Module)        │
  └────────────┬────────────────┘
               │  loaded once at startup
               ▼
  ┌─────────────────────────────┐
  │  MLflow Model Registry      │
  │  models:/churn-mlp@Staging  │
  │  (SQLite backend)           │
  └─────────────────────────────┘
```

---

## SLOs

| SLO | Target | Measurement |
|---|---|---|
| Latency p95 (single prediction) | < 200 ms | LatencyLoggingMiddleware per-request log |
| Availability | ≥ 99% (single-instance scope) | External health-check ping on `/health` |
| Cold-start time | < 15 s | Docker healthcheck start-period |

HA, auto-scaling, and multi-instance load balancing are out of scope for v1.
