# Model Comparison

All models evaluated with stratified 5-fold cross-validation (`random_state=42`, `n_folds=5`).  
Primary metric: **PR-AUC** (handles class imbalance; ~26.6% positive class).  
Dataset: Telco Customer Churn, 7,032 rows after cleaning.  
MLflow experiment: `nn-churn-prediction-fiap` — all runs traceable by `dataset_sha256`.

---

## Results table

| Model | PR-AUC (mean±std) | ROC-AUC (mean±std) | F1 (mean±std) | Recall @ Precision≥0.70 |
|---|---|---|---|---|
| DummyClassifier (stratified) | 0.264 ± 0.007 | 0.492 ± 0.020 | 0.255 ± 0.029 | 0.000 |
| LogisticRegression | **0.658 ± 0.012** | **0.845 ± 0.003** | **0.625 ± 0.011** | 0.424 ± 0.056 |
| GradientBoostingClassifier | **0.663 ± 0.016** | 0.844 ± 0.004 | 0.586 ± 0.016 | **0.428 ± 0.066** |
| MLP small (64→32) | 0.651 ± 0.015 | 0.842 ± 0.005 | 0.618 ± 0.015 | 0.410 ± 0.062 |
| MLP medium (128→64→32) | 0.645 ± 0.023 | 0.839 ± 0.007 | 0.617 ± 0.016 | 0.420 ± 0.054 |
| MLP high-dropout (128→64, p=0.5) | 0.646 ± 0.021 | 0.840 ± 0.006 | 0.616 ± 0.015 | 0.408 ± 0.060 |
| MLP weighted (pos_weight=3.5) | 0.644 ± 0.023 | 0.839 ± 0.006 | 0.603 ± 0.018 | 0.418 ± 0.059 |

Bold = winner per metric.

---

## Honest finding

**GBM wins on PR-AUC (0.663), followed closely by LogReg (0.658). All MLP configs fall slightly below both.**

This is a well-known result on small, structured tabular data (~7k rows, 21 features): gradient boosting methods tend to match or outperform neural networks. The MLP is nonetheless the central deliverable per the project brief and is served via the API — but the comparison must be reported honestly.

Key observations:
- The gap between GBM and the best MLP is ~0.012 PR-AUC — within each other's ± ranges, so not dramatic.
- LogReg achieves the best F1 because `class_weight="balanced"` gives it calibrated recall-precision trade-off at the default 0.5 threshold.
- All models substantially beat the dummy baseline (PR-AUC 0.264 → 0.663, a **2.5× lift**).
- MLP configs show higher variance across folds (std up to 0.023 vs GBM's 0.016), suggesting less stability on this dataset size.

---

## Model selected for serving

**`mlp-small`** (hidden sizes 64→32, dropout 0.3, lr 1e-3, pos_weight 2.7)

Rationale:
- Best PR-AUC among MLP configs and lowest variance.
- Smallest architecture → fastest inference, lowest memory footprint.
- PR-AUC 0.651 is within noise of GBM — acceptable given the project's goal of demonstrating an end-to-end MLP pipeline.
- If production PR-AUC is the only concern, **GBM (0.663) would be the pragmatic choice**.

Registered in MLflow Model Registry as `churn-mlp` with alias `Staging`.

---

## Operating threshold

Cost model (illustrative telecom assumptions):
- avg_customer_LTV = $1,000
- retention_offer_cost = $100
- Avoided churn value = `TP × $1,000 − (TP + FP) × $100`

Threshold sweep on held-out data shows optimal expected profit at **threshold ≈ 0.35**, where recall is higher without excessive false positives. The default 0.5 threshold is conservative; lowering to 0.35 recovers ~15% more true churners at an acceptable precision cost.

**Operating threshold: 0.35** — logged in config and used by the API's `churn_flag` field.

---

## What would improve the MLP

1. More data — GBM's edge shrinks as n grows; MLPs scale better.
2. Feature engineering (e.g. tenure buckets, charge-per-month ratio) could help both.
3. Larger hyperparameter search with Optuna (budget constraint here: 4 configs by hand).
4. Longer training with warmup LR schedule instead of ReduceLROnPlateau.
