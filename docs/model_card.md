# Model Card — Telco Customer Churn MLP

## Model details

| Field | Value |
|---|---|
| Model type | MLP (64→32, ReLU, Dropout 0.3, BatchNorm) wrapped in sklearn Pipeline |
| Framework | PyTorch 2.x + scikit-learn |
| Task | Binary classification — predict customer churn (Yes/No) |
| Input | 18 tabular features (demographics, services, billing) |
| Output | `churn_probability` ∈ [0, 1]; `churn_flag` = `churn_probability ≥ 0.35` |
| Operating threshold | 0.35 (cost-optimal; see threshold section below) |
| Registry | MLflow Model Registry — `churn-mlp`, alias `Staging` |
| Training date | May 2026 |
| Dataset sha256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |

---

## Intended use

**Primary use case:** Flag customers at risk of churn so a retention team can intervene proactively (e.g., targeted offers, account reviews).

**Appropriate users:** Retention operations teams, CRM systems, data/ML teams building on top of this score.

**Out-of-scope uses:**
- Decisions that affect credit, employment, housing, or legal rights — this model has not been audited for those contexts.
- Real-time fraud detection or any latency-critical (< 10 ms) path.
- Customer segments far outside the Telco Churn (IBM) distribution (e.g., enterprise accounts, non-US markets).
- Retraining-free production use beyond 3–6 months without drift monitoring.

---

## Training data

- **Source:** [Telco Customer Churn — IBM (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,032 rows after dropping 11 rows with unparseable `TotalCharges` (tenure = 0, whitespace values)
- **Features:** 18 columns — 3 numeric (`tenure`, `MonthlyCharges`, `TotalCharges`), 15 categorical (services, contract, payment method, demographics)
- **Target:** `Churn` (binary); ~26.6% positive class (moderate imbalance)
- **Split strategy:** Stratified 5-fold cross-validation (`random_state=42`) — no separate held-out test set (dataset is small)
- **Preprocessing:** `StandardScaler` on numerics; `OneHotEncoder` on categoricals; `TenureBucket` custom transformer — all fit *inside* CV folds to prevent leakage

---

## Evaluation

All metrics are mean ± std across 5 stratified CV folds. Primary metric is **PR-AUC** (handles class imbalance).

### CV results (5-fold, seed=42)

| Model | PR-AUC | ROC-AUC | F1 | Recall @ Prec≥0.70 |
|---|---|---|---|---|
| DummyClassifier | 0.264 ± 0.007 | 0.492 ± 0.020 | 0.255 ± 0.029 | 0.000 |
| LogisticRegression | 0.658 ± 0.012 | **0.845 ± 0.003** | **0.625 ± 0.011** | 0.424 ± 0.056 |
| GradientBoostingClassifier | **0.663 ± 0.016** | 0.844 ± 0.004 | 0.586 ± 0.016 | **0.428 ± 0.066** |
| **MLP small (64→32)** ← served | 0.651 ± 0.015 | 0.842 ± 0.005 | 0.618 ± 0.015 | 0.410 ± 0.062 |
| MLP medium (128→64→32) | 0.645 ± 0.023 | 0.839 ± 0.007 | 0.617 ± 0.016 | 0.420 ± 0.054 |
| MLP high-dropout (128→64, p=0.5) | 0.646 ± 0.021 | 0.840 ± 0.006 | 0.616 ± 0.015 | 0.408 ± 0.060 |
| MLP weighted (pos_weight=3.5) | 0.644 ± 0.023 | 0.839 ± 0.006 | 0.603 ± 0.018 | 0.418 ± 0.059 |

**Honest note:** GBM outperforms all MLPs on this dataset. This is expected for small tabular data (~7k rows). The MLP is served per the project brief; if raw performance were the only goal, GBM would be the pragmatic choice.

### Full-dataset metrics at threshold 0.35 (served model)

| Metric | Value |
|---|---|
| PR-AUC | 0.675 |
| ROC-AUC | 0.859 |
| F1 | 0.608 |
| Precision | 0.455 |
| Recall | 0.917 |

The low precision / high recall profile is intentional — at threshold 0.35, the model catches 91.7% of churners at the cost of also flagging ~55% false positives. Given the cost model (LTV $1,000, offer cost $100), this is the profit-maximising operating point.

### Operating threshold

Cost model (illustrative telecom assumptions):
- `avg_customer_LTV` = $1,000
- `retention_offer_cost` = $100
- `profit = TP × $1,000 − (TP + FP) × $100`

Threshold sweep shows profit is maximised at **0.35**. Default 0.5 is too conservative, leaving ~15% of true churners undetected for no cost benefit.

---

## Bias analysis

Metrics computed on the full cleaned dataset (n = 7,032) using the served model at threshold 0.35.

| Slice | N | Churn % | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| **Overall** | 7,032 | 26.6% | 0.675 | 0.859 | 0.608 | 0.455 | 0.917 |
| gender = Male | 3,549 | 26.2% | 0.671 | 0.856 | 0.598 | 0.445 | 0.908 |
| gender = Female | 3,483 | 27.0% | 0.679 | 0.861 | 0.618 | 0.464 | 0.925 |
| SeniorCitizen = Yes | 1,142 | 41.7% | 0.737 | 0.814 | 0.690 | 0.539 | 0.960 |
| SeniorCitizen = No | 5,890 | 23.7% | 0.651 | 0.861 | 0.583 | 0.430 | 0.902 |
| Partner = Yes | 3,393 | 19.7% | 0.613 | 0.870 | 0.570 | 0.426 | 0.864 |
| Partner = No | 3,639 | 33.0% | 0.705 | 0.839 | 0.629 | 0.471 | 0.946 |
| Contract = Month-to-month | 3,875 | 42.7% | 0.706 | 0.772 | 0.640 | 0.476 | 0.976 |
| Contract = One year | 1,472 | 11.3% | 0.277 | 0.777 | 0.359 | 0.262 | 0.572 |
| Contract = Two year | 1,685 | 2.8% | 0.140 | 0.794 | 0.097 | 0.214 | 0.062 |

### Interpretation

**Gender:** Minimal disparity (PR-AUC gap 0.008). No evidence of gender-based performance inequality.

**SeniorCitizen:** Seniors have higher PR-AUC (0.737 vs 0.651). This reflects that seniors have a much higher churn rate (41.7% vs 23.7%), making positives easier to detect — not a model advantage, but a base-rate effect.

**Partner:** Non-partnered customers (PR-AUC 0.705) are easier to score than partnered ones (0.613). Partnered customers have lower churn rates (19.7%), reducing positive signal.

**Contract type — critical limitation:** The model performs very poorly on one-year (PR-AUC 0.277) and two-year contracts (0.140). Two-year customers almost never churn (2.8%), so the model sees near-zero positive examples and produces near-random scores for that segment. **The model should not be used as the sole retention signal for locked-in customers.** A contract-conditional strategy (e.g., only score month-to-month customers) would be more reliable in practice.

---

## Limitations

1. **Small dataset:** 7,032 rows. Performance estimates carry uncertainty (see ± std in CV table). Conclusions about low-churn subgroups (two-year contracts, n ≈ 47 positives) are particularly unreliable.
2. **Single point in time:** No temporal splits — the model cannot distinguish whether patterns are stable over time or reflect a one-off snapshot.
3. **No concept drift studied:** Customer behaviour, tariff structures, and competitor landscape change. The model should be retrained when input drift (PSI > 0.2) or performance drift (rolling PR-AUC drops > 5%) is detected.
4. **US-style telecom only:** The IBM dataset reflects a North American provider. Transfer to other markets is untested.
5. **Threshold is fixed:** The 0.35 threshold was chosen under an illustrative cost model. Real deployment requires validation against actual LTV and campaign cost data.

## Failure modes

- **New tenure values at boundary (tenure = 0):** These rows had unparseable `TotalCharges` in the source data and were dropped during training. If the API receives a row with `tenure = 0`, the pipeline will process it, but the model has seen very few such cases.
- **Distribution shift in `MonthlyCharges`:** If a new pricing tier significantly changes the charge distribution, feature scaling will be off and predictions will degrade silently.
- **Missing fields:** The API returns HTTP 422 for any missing required field — this is a hard boundary check, not a graceful fallback.
- **Two-year contract customers:** As noted in the bias section, the model is effectively unreliable for this segment.
