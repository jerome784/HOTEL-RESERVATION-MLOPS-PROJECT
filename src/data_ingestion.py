import os
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml

# Define paths - use absolute paths based on script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "artifacts", "raw")
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        if not config or "data_ingestion" not in config:
            raise ValueError("Invalid config: 'data_ingestion' section not found")
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        # AWS S3 client (uses aws configure credentials)
        self.s3_client = boto3.client("s3")

        logger.info(
            f"Data Ingestion started with bucket={self.bucket_name}, file={self.file_name}"
        )

    def download_csv_from_s3(self):
        try:
            logger.info("Downloading CSV file from S3")

            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=self.file_name,
                Filename=RAW_FILE_PATH
            )

            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while downloading CSV from S3")
            raise CustomException("Failed to download CSV file from S3", e)

    def split_data(self):
        try:
            logger.info("Starting data splitting process")

            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data,
                test_size=1 - self.train_test_ratio,
                random_state=36 

            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException(
                "Failed to split data into training and test sets", e
            )

    def run(self):
        try:
            logger.info("Starting data ingestion process")

            self.download_csv_from_s3()
            self.split_data()

            logger.info("Data ingestion completed successfully")

        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            raise

        finally:
            logger.info("Data ingestion execution finished")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
