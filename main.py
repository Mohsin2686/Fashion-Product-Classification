from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_prepare_dataset import DataPreparationPipeline
from cnnClassifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_05_model_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion stage"


# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e




# STAGE_NAME = "Prepare base model"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    prepare_base_model = PrepareBaseModelTrainingPipeline()
#    prepare_base_model.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e


STAGE_NAME = "Data Preparation "
train_ds, val_ds, test_ds = None, None, None
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = DataPreparationPipeline()
   train_ds,val_ds,test_ds = model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# STAGE_NAME = "Training"
# try: 
#    logger.info(f"*******************")
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    model_trainer = ModelTrainingPipeline(train_ds, val_ds)
#    model_trainer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e



STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main(test_ds)
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e