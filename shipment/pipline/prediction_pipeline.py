import os
import sys
import glob
from typing import Optional, Union, Dict, List
import numpy as np
import pandas as pd
from pandas import DataFrame

from shipment.constants import (
    MODEL_PUSHER_BUCKET_NAME,
    S3_MODEL_KEY_PATH,
    MODEL_FILE_NAME,
    SCHEMA_FILE_PATH,
)
from shipment.configuration.s3_operations import S3Operation
from shipment.entity.config_entity import PredictionPipelineConfig
from shipment.exception import shippingException
from shipment.logger import logging
from shipment.utils.main_utils import MainUtils


class ShipmentData:
    """
    ShipmentData class is responsible for mapping user / form input data to a Pandas DataFrame
    with column names that match the schema expected by the trained preprocessor and model.
    """

    def __init__(
        self,
        artist_reputation: float,
        height: float,
        width: float,
        weight: float,
        material: str,
        price_of_sculpture: float,
        base_shipping_price: float,
        international: str,
        express_shipment: str,
        installation_included: str,
        transport: str,
        fragile: str,
        customer_information: str,
        remote_location: str,
        customer_id: Optional[str] = None,
        artist_name: Optional[str] = None,
        customer_location: Optional[str] = None,
        scheduled_date: Optional[str] = None,
        delivery_date: Optional[str] = None,
    ):
        """
        :param artist_reputation: Reputation score of the artist
        :param height: Height of the sculpture
        :param width: Width of the sculpture
        :param weight: Weight of the sculpture
        :param material: Material of the sculpture
        :param price_of_sculpture: Price of the sculpture
        :param base_shipping_price: Base shipping price
        :param international: 'Yes' or 'No'
        :param express_shipment: 'Yes' or 'No'
        :param installation_included: 'Yes' or 'No'
        :param transport: Mode of transport (e.g. 'Airways', 'Roadways', 'Waterways')
        :param fragile: 'Yes' or 'No'
        :param customer_information: Customer class/info (e.g. 'Working Class', 'Wealthy')
        :param remote_location: 'Yes' or 'No'
        """
        try:
            self.artist_reputation = float(artist_reputation)
            self.height = float(height)
            self.width = float(width)
            self.weight = float(weight)
            self.material = str(material)
            self.price_of_sculpture = float(price_of_sculpture)
            self.base_shipping_price = float(base_shipping_price)
            self.international = str(international)
            self.express_shipment = str(express_shipment)
            self.installation_included = str(installation_included)
            self.transport = str(transport)
            self.fragile = str(fragile)
            self.customer_information = str(customer_information)
            self.remote_location = str(remote_location)

            self.customer_id = customer_id
            self.artist_name = artist_name
            self.customer_location = customer_location
            self.scheduled_date = scheduled_date
            self.delivery_date = delivery_date

        except Exception as e:
            raise shippingException(e, sys) from e

    def get_shipment_data_as_dict(self) -> Dict[str, List[Union[float, str]]]:
        """
        Method Name :   get_shipment_data_as_dict
        Description :   Returns the input data as a dictionary matching the schema columns.
        Output      :   Dictionary with column names as keys and single-element lists as values.
        """
        logging.info("Entered get_shipment_data_as_dict method of ShipmentData class")
        try:
            custom_data_input_dict = {
                "Artist Reputation": [self.artist_reputation],
                "Height": [self.height],
                "Width": [self.width],
                "Weight": [self.weight],
                "Material": [self.material],
                "Price Of Sculpture": [self.price_of_sculpture],
                "Base Shipping Price": [self.base_shipping_price],
                "International": [self.international],
                "Express Shipment": [self.express_shipment],
                "Installation Included": [self.installation_included],
                "Transport": [self.transport],
                "Fragile": [self.fragile],
                "Customer Information": [self.customer_information],
                "Remote Location": [self.remote_location],
            }
            logging.info("Created shipment input data dictionary successfully")
            return custom_data_input_dict

        except Exception as e:
            raise shippingException(e, sys) from e

    def get_shipment_input_data_frame(self) -> DataFrame:
        """
        Method Name :   get_shipment_input_data_frame
        Description :   Converts the input dictionary into a pandas DataFrame.
        Output      :   DataFrame containing a single row of prediction features.
        """
        logging.info("Entered get_shipment_input_data_frame method of ShipmentData class")
        try:
            shipment_dict = self.get_shipment_data_as_dict()
            df = pd.DataFrame(shipment_dict)
            logging.info("Converted shipment dictionary to DataFrame")
            return df

        except Exception as e:
            raise shippingException(e, sys) from e

    def get_data_as_data_frame(self) -> DataFrame:
        """
        Alias method for get_shipment_input_data_frame.
        """
        return self.get_shipment_input_data_frame()

    def get_data_as_dict(self) -> Dict[str, List[Union[float, str]]]:
        """
        Alias method for get_shipment_data_as_dict.
        """
        return self.get_shipment_data_as_dict()


# Alias for CustomData commonly used across ML pipelines
CustomData = ShipmentData


class PredictionPipeline:
    """
    PredictionPipeline class is responsible for loading the trained model
    (from AWS S3 or local artifacts) and generating price predictions on input data.
    """

    def __init__(
        self,
        prediction_pipeline_config: Optional[PredictionPipelineConfig] = None,
        s3_op: Optional[S3Operation] = None,
        model: Optional[object] = None,
    ):
        """
        :param prediction_pipeline_config: Configuration for prediction pipeline
        :param s3_op: S3Operation object for downloading model from AWS S3
        :param model: Pre-loaded model object (optional)
        """
        try:
            self.prediction_pipeline_config = (
                prediction_pipeline_config
                if prediction_pipeline_config is not None
                else PredictionPipelineConfig()
            )
            self.s3_op = s3_op if s3_op is not None else S3Operation()
            self.utils = (
                getattr(self.prediction_pipeline_config, "UTILS", None)
                if getattr(self.prediction_pipeline_config, "UTILS", None) is not None
                else MainUtils()
            )
            self.model = model

        except Exception as e:
            raise shippingException(e, sys) from e

    def get_latest_local_model_path(self) -> Optional[str]:
        """
        Method Name :   get_latest_local_model_path
        Description :   Searches for the latest trained model file in local artifact directories.
        Output      :   File path string if found, else None.
        """
        logging.info("Entered get_latest_local_model_path method of PredictionPipeline class")
        try:
            # 1. Check direct config path
            if (
                self.prediction_pipeline_config.MODEL_FILE_PATH
                and os.path.exists(self.prediction_pipeline_config.MODEL_FILE_PATH)
            ):
                logging.info(
                    f"Found local model at configured path: {self.prediction_pipeline_config.MODEL_FILE_PATH}"
                )
                return self.prediction_pipeline_config.MODEL_FILE_PATH

            # 2. Search in artifacts directory patterns
            search_patterns = [
                os.path.join(os.getcwd(), "artifacts", "*", "ModelTrainerArtifacts", MODEL_FILE_NAME),
                os.path.join(os.getcwd(), "artifacts", "ModelTrainerArtifacts", MODEL_FILE_NAME),
                os.path.join(os.getcwd(), "artifacts", "**", MODEL_FILE_NAME),
                os.path.join(os.getcwd(), MODEL_FILE_NAME),
            ]

            matching_files = []
            for pattern in search_patterns:
                files = glob.glob(pattern, recursive=True)
                matching_files.extend(files)

            if matching_files:
                # Pick the latest modified model file
                latest_file = max(matching_files, key=os.path.getmtime)
                logging.info(f"Found latest local model file at: {latest_file}")
                return latest_file

            logging.info("No local model file found in search patterns.")
            return None

        except Exception as e:
            logging.warning(f"Error while searching for local model: {e}")
            return None

    def get_model_from_s3(self) -> Optional[object]:
        """
        Method Name :   get_model_from_s3
        Description :   Fetches and loads the trained model object directly from AWS S3 bucket.
        Output      :   Model object if available, else None.
        """
        logging.info("Entered get_model_from_s3 method of PredictionPipeline class")
        try:
            bucket_name = self.prediction_pipeline_config.BUCKET_NAME
            s3_model_key = self.prediction_pipeline_config.S3_MODEL_KEY_PATH

            if self.s3_op.is_model_present(bucket_name=bucket_name, s3_model_key=s3_model_key):
                logging.info(f"Loading model from S3 bucket '{bucket_name}' key '{s3_model_key}'")
                model = self.s3_op.load_model(model_name=s3_model_key, bucket_name=bucket_name)
                logging.info("Successfully loaded model from S3")
                return model

            logging.info(f"Model key '{s3_model_key}' not found in S3 bucket '{bucket_name}'")
            return None

        except Exception as e:
            logging.warning(f"Failed to load model from S3 (falling back to local): {e}")
            return None

    def get_model(self) -> object:
        """
        Method Name :   get_model
        Description :   Retrieves the trained model object from cache, S3, or local storage.
        Output      :   Model object (e.g. CostModel).
        """
        logging.info("Entered get_model method of PredictionPipeline class")
        try:
            if self.model is not None:
                return self.model

            # Try loading from S3 first
            s3_model = self.get_model_from_s3()
            if s3_model is not None:
                self.model = s3_model
                return self.model

            # Fallback to local artifacts
            local_model_path = self.get_latest_local_model_path()
            if local_model_path is not None and os.path.exists(local_model_path):
                logging.info(f"Loading model from local path: {local_model_path}")
                self.model = self.utils.load_object(local_model_path)
                return self.model

            raise Exception(
                f"No trained model could be found in AWS S3 bucket '{self.prediction_pipeline_config.BUCKET_NAME}' "
                f"or in local artifact directories. Please ensure model training/pusher has completed."
            )

        except Exception as e:
            raise shippingException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Method Name :   predict
        Description :   Runs inference on the provided DataFrame and returns predicted shipping prices.
        Output      :   numpy array of predictions.
        """
        logging.info("Entered predict method of PredictionPipeline class")
        try:
            model = self.get_model()

            # If the dataframe contains target column, drop it
            if "Cost" in dataframe.columns:
                dataframe = dataframe.drop(columns=["Cost"], axis=1)

            logging.info(f"Predicting on dataframe with shape: {dataframe.shape}")
            predictions = model.predict(dataframe)
            logging.info("Prediction completed successfully")
            logging.info("Exited predict method of PredictionPipeline class")

            return predictions

        except Exception as e:
            raise shippingException(e, sys) from e

    def predict_batch(
        self,
        input_csv_path: str,
        output_csv_path: Optional[str] = None,
    ) -> DataFrame:
        """
        Method Name :   predict_batch
        Description :   Reads an input CSV, runs batch predictions, and appends a 'Predicted_Cost' column.
        Output      :   DataFrame with predictions.
        """
        logging.info("Entered predict_batch method of PredictionPipeline class")
        try:
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input file not found at: {input_csv_path}")

            df = pd.read_csv(input_csv_path)
            logging.info(f"Loaded input CSV with shape: {df.shape}")

            predictions = self.predict(df)
            df["Predicted_Cost"] = predictions

            if output_csv_path is not None:
                os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
                df.to_csv(output_csv_path, index=False)
                logging.info(f"Saved batch predictions to: {output_csv_path}")

            logging.info("Exited predict_batch method of PredictionPipeline class")
            return df

        except Exception as e:
            raise shippingException(e, sys) from e


# Aliases for convenience and backward compatibility
PredictPipeline = PredictionPipeline
SingleValuePredictionPipeline = PredictionPipeline
