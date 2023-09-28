
from cropPrediction import logger
from cropPrediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cropPrediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from cropPrediction.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from cropPrediction.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from cropPrediction.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from pathlib import Path


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training = ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model Evaluation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_evaluation = ModelEvaluationPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
        logger.exception(e)
        raise e
