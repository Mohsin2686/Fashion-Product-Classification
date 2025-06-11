from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.dataset_preparation import DataPreparation
from cnnClassifier import logger



STAGE_NAME = "Data Preparation"

class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_prepare_dataset_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        
        # Prepare and load the dataset
        #data_preparation.prepare_dataset()
        train_ds,val_ds,test_ds = data_preparation.generate_datasets()
        return train_ds, val_ds, test_ds
        

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e