# Monitoring Plan

## Overview

Three layers of monitoring cover the full failure surface of the served model:

1. **Input drift** — has the customer data distribution shifted?
2. **Prediction drift** — has the score distribution shifted (even without labels)?
3. **Performance drift** — has accuracy dropped on labeled feedback?

---

## Metrics

### 1. Input drift — Population Stability Index (PSI)

Compute PSI weekly on the top predictive features against the training distribution baseline.

**Features to monitor (in priority order):**

| Feature | Why it matters |
|---|---|
| `Contract` | Largest driver of churn rate; a shift (e.g. more month-to-month) changes base rates |
| `tenure` | Distribution shift here changes what the TenureBucket transformer produces |
| `MonthlyCharges` | Pricing changes can silently invalidate the scaler |
| `InternetService` | Product mix changes affect feature co-occurrence patterns |
| `TotalCharges` | Derived from tenure × charges; drifts compound |

**PSI thresholds:**

| PSI | Interpretation | Action |
|---|---|---|
| < 0.1 | No significant shift | No action |
| 0.1 – 0.2 | Minor shift | Log warning, investigate manually |
| > 0.2 | Significant shift — **alert** | Page on-call, freeze model promotion, investigate retraining |

**Implementation:** Compute PSI by binning the training distribution into 10 equal-frequency bins, then measuring how the current week's distribution falls across those same bins.

### 2. Prediction drift — score distribution

Monitor the distribution of `churn_probability` scores across all predictions in a rolling 7-day window.

**Metrics:**
- Mean score (should stay near ~0.35–0.45 at this threshold)
- % of predictions flagged as `churn_flag = True` (expected ~30–40% at threshold 0.35)
- KS statistic vs training score distribution

**Alert:** Score mean shifts by > 0.10 or flag rate changes by > 15 percentage points week-over-week.

This catches model degradation even before labels arrive — a sudden drop or spike in predicted churn is a red flag regardless of ground truth.

### 3. Performance drift — rolling PR-AUC on labeled feedback

When churn outcomes become known (typically 30–60 days after prediction), join predictions to actuals and compute rolling PR-AUC.

**Alert:** Rolling PR-AUC (30-day window) drops > 5% relative to validation PR-AUC (0.651).

**Minimum label volume:** Do not compute rolling PR-AUC on fewer than 200 labeled examples — the metric is unreliable below that.

---

## Alerts

| Alert | Trigger | Severity | Who gets paged |
|---|---|---|---|
| Input drift | PSI > 0.2 on any monitored feature | High | ML on-call |
| Score distribution shift | Mean score or flag rate shifts > threshold | Medium | ML on-call |
| Performance drop | Rolling PR-AUC drops > 5% vs validation | High | ML lead + retention team lead |
| API errors | HTTP 5xx rate > 1% over 5 min | Critical | SRE / ML on-call |
| API latency | p95 > 200 ms over 5 min | Medium | SRE |
| Model not loaded | `/health` returns status ≠ "ok" | Critical | SRE |

---

## Response playbook

### Input drift fires (PSI > 0.2)

1. Identify which feature drifted and by how much.
2. Check if it is a data pipeline issue (upstream schema change, ETL bug) — fix first.
3. If the data is correct, the real-world distribution has shifted. Retrain with recent data.
4. Do not promote the retrained model to production without running CV and comparing to current model metrics.
5. If retraining is not yet ready, consider falling back to the LogisticRegression baseline (more robust to distribution shift on this dataset).

### Performance drop fires (rolling PR-AUC drops > 5%)

1. Check if the drop correlates with a known data issue (holiday season, product change).
2. Inspect prediction distribution — is the model over-predicting or under-predicting?
3. Retrain on recent labeled data. Use the same CV pipeline (`make train`) to validate before promoting.
4. If the drop is concentrated in a specific segment (e.g. two-year contracts), consider a segment-specific model or rule override.

### API errors / latency spike

1. Check container health (`docker compose ps`, `/health` endpoint).
2. Check MLflow connectivity — the model loads from SQLite on startup; if the DB is locked, restart the service.
3. If latency is the issue, profile the request: the sklearn pipeline's `ColumnTransformer` is the likely bottleneck for large batches.

---

## What is NOT monitored (and why)

- **Individual prediction fairness in real-time:** Slice metrics (gender, contract type) are computed offline in the model card. Real-time per-request fairness scoring adds latency for no operational benefit at this scale.
- **Feature importance drift:** Shapley values per request are expensive. Monitor input distributions instead — that is the root cause.
- **Retraining automation:** Retraining is triggered manually after an alert. Automated retraining without human review is out of scope for v1.
