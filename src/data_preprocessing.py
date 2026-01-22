import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml


# ---------------- PATH DEFINITIONS ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")

TRAIN_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DIR, "test.csv")

PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

os.makedirs(PROCESSED_DIR, exist_ok=True)

logger = get_logger(__name__)


# ---------------- DATA PREPROCESSING CLASS ---------------- #
class DataPreprocessing:

    def __init__(self, config):
        if not config or "data_processing" not in config:
            raise ValueError("Invalid config: 'data_processing' section missing")

        self.config = config["data_processing"]

        self.categorical_cols = self.config["categorical_columns"]
        self.numerical_cols = self.config["numerical_columns"]
        self.skew_threshold = self.config["skewness_threshold"]
        self.no_of_features = self.config["no_of_features"]

        self.label_encoders = {}
        self.selected_features = None

        logger.info("Data Preprocessing initialized")

    # -------------------------------------------------- #
    def load_data(self):
        try:
            logger.info("Loading train and test data")

            train_df = pd.read_csv(TRAIN_PATH)
            test_df = pd.read_csv(TEST_PATH)

            return train_df, test_df

        except Exception as e:
            raise CustomException("Failed to load raw data", e)

    # -------------------------------------------------- #
    def preprocess(self, df, is_train=True):
        try:
            logger.info("Starting preprocessing")

            df.columns = df.columns.str.lower()
            df.drop(columns=["booking_id"], errors="ignore", inplace=True)
            df.drop_duplicates(inplace=True)

            # ---------- LABEL ENCODING ---------- #
            for col in self.categorical_cols:
                if col not in df.columns:
                    logger.warning(f"{col} not found, skipping")
                    continue

                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    df[col] = self.label_encoders[col].transform(df[col])

            # ---------- SKEWNESS HANDLING ---------- #
            valid_num_cols = [c for c in self.numerical_cols if c in df.columns]
            skewness = df[valid_num_cols].skew()

            for col in skewness[skewness > self.skew_threshold].index:
                df[col] = np.log1p(df[col])

            return df

        except Exception as e:
            raise CustomException("Preprocessing failed", e)

    # -------------------------------------------------- #
    def balance_data(self, df):
        try:
            logger.info("Applying SMOTE")

            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_res, columns=X.columns)
            balanced_df["booking_status"] = y_res

            return balanced_df

        except Exception as e:
            raise CustomException("SMOTE failed", e)

    # -------------------------------------------------- #
    def select_features(self, df):
        try:
            logger.info("Selecting features")

            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            self.selected_features = (
                importance_df.head(self.no_of_features)["feature"].tolist()
            )

            logger.info(f"Selected features: {self.selected_features}")

            return df[self.selected_features + ["booking_status"]]

        except Exception as e:
            raise CustomException("Feature selection failed", e)

    # -------------------------------------------------- #
    def apply_selected_features(self, df):
        try:
            return df[self.selected_features + ["booking_status"]]
        except Exception as e:
            raise CustomException("Applying selected features failed", e)

    # -------------------------------------------------- #
    def save_data(self, train_df, test_df):
        try:
            train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
            test_df.to_csv(PROCESSED_TEST_PATH, index=False)

            logger.info("Processed data saved successfully")

        except Exception as e:
            raise CustomException("Saving processed data failed", e)

    # -------------------------------------------------- #
    def run(self):
        try:
            logger.info("Starting data preprocessing pipeline")

            train_df, test_df = self.load_data()

            train_df = self.preprocess(train_df, is_train=True)
            test_df = self.preprocess(test_df, is_train=False)

            train_df = self.balance_data(train_df)

            train_df = self.select_features(train_df)
            test_df = self.apply_selected_features(test_df)

            self.save_data(train_df, test_df)

            logger.info("Data preprocessing completed successfully")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

        finally:
            logger.info("Data preprocessing execution finished")


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    preprocessing = DataPreprocessing(read_yaml(CONFIG_PATH))
    preprocessing.run()
