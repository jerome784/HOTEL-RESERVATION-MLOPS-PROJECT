import os

# Project root directory (…/MLOPS-Hotel-Reservation)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------- CONFIG -------------------- #
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# -------------------- ARTIFACTS -------------------- #
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Raw data (optional, if you use it)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw_data.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

# Processed artifacts (used by preprocessing + training)
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

# Model artifacts
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")
