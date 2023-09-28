import os
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from cropPrediction import logger
from pathlib import Path 
from cropPrediction.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None 
        self.transformed_df = None

    def get_data_transformation(self):
        try:
            # Load the dataset
            df = pd.read_csv(self.config.data_path)

            # Divide the dataset into independent and dependent features
            X = df.drop(columns=["label"], axis=1)
            y = df["label"]

            logger.info("Dividing the dataset into independent and dependent features completed")

            # Create an instance of LabelEncoder
            label_encoder = LabelEncoder()

            # Fit the label encoder to your categorical labels (y) and transform them
            y_encoded = label_encoder.fit_transform(y)

            logger.info("Encoding Target variable completed")

            numerical_features = X.select_dtypes(exclude="object").columns
            
            # Define the pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])

            # Define the Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                ]
            )

            self.preprocessor = preprocessor

            # Transform the whole data using the preprocessor
            X_transformed = preprocessor.fit_transform(X)

            # Get the updated column names after ordinal encoding
            column_names = numerical_features.to_list()

            # Combine X_transformed and y back into one Dataframe
            self.transformed_df = pd.DataFrame(X_transformed, columns=column_names)
            self.transformed_df["label"] = y_encoded

            logger.info("Data preprocessing completed")

        except Exception as e:
            raise e

    def save_preprocessor(self):
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved to {self.config.preprocessor_path}")
        else:
            logger.warning("Preprocessor is not available. Please call get_data_transformation to create it.") 

    def train_test_split(self, test_size=0.2, random_state=None):
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not available. Please call get_data_transformation.")

        # Split the data into train and test sets
        train, test = train_test_split(self.transformed_df, test_size=test_size, random_state=random_state)

        # Save the encoded train and test sets in the form of CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split the data into train and test sets.")
        logger.info(f"Shape of train data: {train.shape}")
        logger.info(f"Shape of test data: {test.shape}")