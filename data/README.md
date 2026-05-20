# Data

## Source

**Telco Customer Churn — IBM dataset**
- Origin: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- License: IBM (public, educational use)
- ~7,043 rows, 21 columns
- Target: `Churn` (Yes/No), ~26% positive class

## Raw file

| Field | Value |
|---|---|
| Path | `data/raw/raw.csv` |
| sha256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |

The raw CSV is gitignored. To reproduce, download from Kaggle and verify the hash:

```bash
sha256sum data/raw/raw.csv
```

Every MLflow run logs this hash as the `dataset_sha256` tag so any experiment can be traced back to this exact file.

## Folder structure

| Folder | Contents | Gitignored |
|---|---|---|
| `data/raw/` | Original CSV, never modified | Yes |
| `data/interim/` | Post-cleaning, pre-feature-engineering | Yes |
| `data/processed/` | Train/val/test splits ready for modeling | Yes |
