# STAR Video Script — Telco Churn Prediction

**Total time:** 5 minutes  
**Format:** Screen recording + voiceover. Keep the terminal and browser visible.  
**Tip:** Do one take. Stumbles are fine — confidence matters more than perfection.

---

## [S] Situation — 0:00 to 0:45

> *Show: nothing yet, or a simple slide with the word "Problem"*

"A telecom company is losing customers at an accelerated rate — and finding out about it too late.

The current approach is reactive: a customer cancels, support calls them, tries to offer a discount. By then, the decision is made. The customer is gone.

The problem isn't effort — it's timing. The retention team has capacity to intervene, but no way to know *who* to call *before* they leave.

This project builds a system that flags at-risk customers early enough for the team to act."

---

## [T] Task — 0:45 to 1:15

> *Show: project folder structure in the terminal or IDE*

"My task was to deliver a complete, production-grade ML system — not just a notebook with a model, but something a real team could pick up and run.

That means: data validation, reproducible training, honest model comparison, a tested API, and documentation that admits the model's limitations.

The dataset is the IBM Telco Customer Churn dataset — about 7,000 customers, 26% positive class, 18 features."

---

## [A] Action — 1:15 to 3:45

*This is the longest section. Move through it at a steady pace — don't rush.*

### Architecture — 1:15 to 1:45

> *Show: README.md open in the browser or IDE, pointing at the architecture diagram*

"The system is structured in layers. Raw data goes through a pandera schema validator, then a sklearn ColumnTransformer — standard scaler on numerics, one-hot encoding on categoricals, and a custom TenureBucket transformer. That preprocessed data feeds into either the baselines or the MLP.

Everything is tracked in MLflow — every fold, every metric, every model config. And the final model is served via a FastAPI REST API."

### Training & comparison — 1:45 to 2:30

> *Show: MLflow UI at localhost:5000, click into the experiment, show the runs table*

"I trained four model families: a dummy baseline, logistic regression, gradient boosting, and four MLP configurations — all evaluated with stratified 5-fold cross-validation using the same folds and the same seed.

Here in MLflow you can see every run, with PR-AUC as the primary metric because the positive class is the minority.

The honest result: GBM wins at PR-AUC 0.663. The best MLP gets 0.651 — close, but not ahead.

This is expected on small tabular data, and it's documented honestly in the model card. The MLP is still the central deliverable and the one being served — but I'm not hiding the comparison."

### API demo — 2:30 to 3:30

> *Show: terminal, run `make serve`, then run the curl command*

"The model is registered in MLflow Model Registry and loaded at API startup. Let me show you."

```
make serve
```

> *Wait for the startup log line `model_loaded` to appear, then run:*

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

"Health check returns status ok, the model alias — Staging — and the git commit, so we know exactly what's deployed.

Now a prediction:"

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
    "gender": "Female", "Partner": "Yes", "Dependents": "No",
    "PhoneService": "Yes", "PaperlessBilling": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check"
  }' | python3 -m json.tool
```

"86% churn probability, flagged as high-risk. Operating threshold is 0.35 — chosen by sweeping a cost model where each saved customer is worth $1,000 and each retention offer costs $100.

If I send an invalid payload, I get a 422 immediately — the API validates every field at the boundary before the model ever sees it."

### Tests — 3:30 to 3:45

> *Show: terminal, run `make test`*

```
make test
```

"Eight tests, all passing. Schema validation, end-to-end smoke training, and API tests — health, valid prediction, and invalid input rejection."

---

## [R] Result — 3:45 to 5:00

> *Show: docs/model_card.md open, pointing at the bias table*

"The served model achieves PR-AUC 0.651 — a 2.5× lift over the dummy baseline.

At the operating threshold of 0.35, it catches 91.7% of churners. The trade-off is precision at 45.5% — meaning roughly half of flagged customers won't actually churn. Given the cost model, that's still profitable.

The bias analysis in the model card shows near-parity across gender. But there's a critical limitation I want to highlight: the model barely works for customers on two-year contracts. They almost never churn — only 2.8% — so the model has almost no signal for that group. That's documented, and a contract-conditional scoring strategy would be the right fix in production.

What I'd do next:

First, **retraining pipeline** — when labeled feedback comes in 30 to 60 days after predictions, roll the PR-AUC forward and retrain if it drops more than 5%. The monitoring plan in the docs has the full playbook.

Second, **online A/B test** — run GBM and MLP in parallel on disjoint customer segments and measure actual churn rates, not just CV metrics. That's the only way to know which model saves more customers in the real world.

The repo is fully reproducible. Clone it, run `make setup train serve`, and the full system comes up. Thanks."

---

## Timing guide

| Section | Duration | Cumulative |
|---|---|---|
| Situation | 45 s | 0:45 |
| Task | 30 s | 1:15 |
| Architecture | 30 s | 1:45 |
| Training & MLflow | 45 s | 2:30 |
| API demo | 60 s | 3:30 |
| Tests | 15 s | 3:45 |
| Results & next steps | 75 s | 5:00 |

## Recording tips

- Run `make serve` *before* you start recording so the model is already loaded.
- Have the curl command in a script or history — don't type it live.
- Keep MLflow UI open in a browser tab, pre-navigated to the experiment runs table.
- If you go over time, cut the tests section — it's the least important visually.
