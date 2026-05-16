from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "raw.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42
N_FOLDS = 5
TARGET_COL = "Churn"
DATASET_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "nn-churn-prediction-fiap"

RECALL_AT_PRECISION_THRESHOLD = 0.70
OPERATING_THRESHOLD = 0.5
