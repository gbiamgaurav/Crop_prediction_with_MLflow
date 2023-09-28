
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from urllib.parse import urlparse
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path 
from cropPrediction.entity.config_entity import ModelEvaluationConfig
from cropPrediction.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        acc = accuracy_score(actual, pred)
        p_score = precision_score(actual, pred, average="weighted")
        return acc, p_score
    

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predictions = model.predict(test_x)

            (acc, p_score) = self.eval_metrics(test_y, predictions)
            
            # Saving metrics as local
            scores = {"Accuracy Score": acc, "Precision Score": p_score}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("Accuracy Score", acc)
            mlflow.log_metric("Precision Score", p_score)
            


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestModel")
            else:
                mlflow.sklearn.log_model(model, "model")
    


    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        (acc, p_score) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {"Accuracy Score": acc, "Precision Score": p_score}
        save_json(path=Path(self.config.metric_file_name), data=scores)