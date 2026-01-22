import os
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as yaml_file:
            content = yaml_file.read()
            logger.info(f"File content length: {len(content)} chars")
            config = yaml.safe_load(content)
            if config is None:
                logger.warning(f"YAML file loaded as None, returning empty dict")
                config = {}
            logger.info("Successfully read the yaml file")
            return config
        

    except Exception as e:
        logger.error(f"Error occurred while reading yaml file: {str(e)}")
        raise CustomException("Failed to read YAML file", e)
    

def load_data(file_path):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape: {data.shape}")
        logger.info(f"Columns in loaded data: {list(data.columns)}")  # Log the column names
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise CustomException("Failed to load data", e)