from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import (
    CONFIG_PATH,
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    MODEL_OUTPUT_PATH,
)



if __name__=="__main__":
    
    ### 1.data ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    ## 2. Data processing
    preprocessing = DataPreprocessing(read_yaml(CONFIG_PATH))
    preprocessing.run()

    ## 3 Model Training
    config = read_yaml(CONFIG_PATH)
    trainer = ModelTraining(
        config=config,
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
    )
    trainer.run()
    