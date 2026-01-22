import os
import joblib

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml, load_data

# Prefer explicit imports over *
from config.model_params import LIGHTGBM_PARAMS, RANDOM_SEARCH_PARAMS


# ---------------- PATH DEFINITIONS ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

os.makedirs(MODEL_DIR, exist_ok=True)

logger = get_logger(__name__)
logger.info("model_training.py loaded (module import executed)")


# ---------------- MODEL TRAINING CLASS ---------------- #
class ModelTraining:
    def __init__(self, config: dict, train_path: str, test_path: str, model_output_path: str):
        self.config = config or {}
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

        logger.info("ModelTraining initialized")
        logger.info(f"Train path: {self.train_path}")
        logger.info(f"Test path: {self.test_path}")
        logger.info(f"Model output path: {self.model_output_path}")

    # -------------------------------------------------- #
    def load_and_split_data(self):
        try:
            logger.info(f"Loading training data from: {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading testing data from: {self.test_path}")
            test_df = load_data(self.test_path)

            if "booking_status" not in train_df.columns or "booking_status" not in test_df.columns:
                raise ValueError("booking_status column is missing in processed train/test data.")

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data split completed for Model Training")
            logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
            logger.info(f"y_train labels: {sorted(y_train.unique().tolist())}")
            logger.info(f"y_test labels:  {sorted(y_test.unique().tolist())}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.exception("Error while loading/splitting data")
            raise CustomException("Failed to load and split data", e)

    # -------------------------------------------------- #
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing LightGBM model")

            lgbm_model = lgb.LGBMClassifier(
                random_state=self.random_search_params["random_state"]
            )

            logger.info("Starting hyperparameter tuning (RandomizedSearchCV)")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"],
            )

            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed")
            logger.info(f"Best parameters: {random_search.best_params_}")

            return random_search.best_estimator_, random_search.best_params_

        except Exception as e:
            logger.exception("Error while training model")
            raise CustomException("Failed to train LightGBM model", e)

    # -------------------------------------------------- #
    def evaluate_model(self, model, X_test, y_test):
        """
        IMPORTANT FIX:
        If y_test is string labels (e.g., 'Canceled', 'Not_Canceled'),
        sklearn needs pos_label specified for precision/recall/f1.
        """
        try:
            logger.info("Evaluating model")

            y_pred = model.predict(X_test)

            # Choose your positive class
            # For hotel cancellation, usually "Canceled" is treated as positive.
            pos_label = "Canceled" if "Canceled" in set(y_test.unique()) else None

            accuracy = accuracy_score(y_test, y_pred)

            if pos_label is None:
                # Fallback: if labels are numeric 0/1, use default binary metrics behavior.
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                logger.info("Using default sklearn pos_label behavior (likely numeric labels).")
            else:
                precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                logger.info(f"Using pos_label='{pos_label}' for precision/recall/f1.")

            logger.info(f"Accuracy:  {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall:    {recall}")
            logger.info(f"F1 Score:  {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        except Exception as e:
            logger.exception("Error while evaluating model")
            raise CustomException("Failed to evaluate model", e)

    # -------------------------------------------------- #
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving trained model")
            joblib.dump(model, self.model_output_path)

            if not os.path.exists(self.model_output_path):
                raise IOError(f"Model not found after saving at: {self.model_output_path}")

            logger.info(f"Model saved at: {self.model_output_path}")
            logger.info(f"Saved model size (bytes): {os.path.getsize(self.model_output_path)}")

        except Exception as e:
            logger.exception("Error while saving model")
            raise CustomException("Failed to save model", e)

    # -------------------------------------------------- #
    def run(self):
        try:
            logger.info("Starting Model Training pipeline")

            with mlflow.start_run():
                logger.info("MLflow run started")

                # Log datasets (as artifacts)
                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path="datasets")
                else:
                    logger.warning(f"Train dataset not found for mlflow.log_artifact: {self.train_path}")

                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path="datasets")
                else:
                    logger.warning(f"Test dataset not found for mlflow.log_artifact: {self.test_path}")

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                best_model, best_params = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_model, X_test, y_test)

                self.save_model(best_model)

                # Log model file as artifact
                mlflow.log_artifact(self.model_output_path, artifact_path="model")

                # Log best params from search (cleaner than dumping all get_params())
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)

                # Optional: also log sklearn model in MLflow model registry format
                # mlflow.sklearn.log_model(best_model, artifact_path="sklearn-model")

                logger.info("Model Training completed successfully")
                logger.info(f"Final Metrics: {metrics}")

        except Exception as e:
            logger.exception("Error in model training pipeline")
            raise CustomException("Failed during model training pipeline", e)

        finally:
            logger.info("Model training execution finished")


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    trainer = ModelTraining(
        config=config,
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
    )
    trainer.run()
