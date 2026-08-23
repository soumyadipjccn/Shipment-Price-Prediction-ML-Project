import os
import sys
import pandas as pd
from typing import Optional
from shipment.logger import logging
from shipment.exception import shippingException
from shipment.entity.config_entity import ModelEvaluationConfig
from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
)
from shipment.constants import MODEL_CONFIG_FILE


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifacts,
        data_transformation_artifact: DataTransformationArtifacts,
        model_trainer_artifact: ModelTrainerArtifacts,
    ):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param data_ingestion_artifact: Artifact from data ingestion stage
        :param data_transformation_artifact: Artifact from data transformation stage
        :param model_trainer_artifact: Artifact from model trainer stage
        """
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def get_best_model(self) -> Optional[object]:
        """
        Method Name :   get_best_model

        Description :   This method fetches the existing best/production model if available. 
        
        Output      :   Best model object or None 
        """
        logging.info("Entered get_best_model method of ModelEvaluation class")
        try:
            # If production/previous model exists in target directory or storage, load it
            # Otherwise return None
            logging.info("Exited get_best_model method of ModelEvaluation class")
            return None
        except Exception as e:
            raise shippingException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Method Name :   initiate_model_evaluation

        Description :   This method initiates model evaluation by testing the trained model on test data
                        and comparing its performance against previous models or base score.
        
        Output      :   Model evaluation artifacts 
        """
        logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
        try:
            # Create Model evaluation artifacts directory
            os.makedirs(
                self.model_evaluation_config.MODEL_EVALUATION_ARTIFACTS_DIR,
                exist_ok=True,
            )
            logging.info(
                f"Created artifacts directory for {os.path.basename(self.model_evaluation_config.MODEL_EVALUATION_ARTIFACTS_DIR)}"
            )

            # Loading test data
            test_data_path = self.data_ingestion_artifact.test_data_file_path
            test_df = pd.read_csv(test_data_path)
            logging.info(f"Loaded test dataset from {test_data_path}")

            # Extracting target column
            target_column_name = self.model_evaluation_config.SCHEMA_CONFIG[
                "target_column"
            ]
            x_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            logging.info("Extracted test features and target column")

            # Loading the trained model
            trained_model_path = (
                self.model_trainer_artifact.trained_model_file_path
            )
            trained_model = self.model_evaluation_config.UTILS.load_object(
                trained_model_path
            )
            logging.info(f"Loaded trained model from {trained_model_path}")

            # Predicting with trained model and calculating score
            y_trained_pred = trained_model.predict(x_test)
            trained_model_score = (
                self.model_evaluation_config.UTILS.get_model_score(
                    y_test, y_trained_pred
                )
            )
            logging.info(f"Trained model score: {trained_model_score}")

            # Fetching existing best/production model
            best_model = self.get_best_model()
            best_model_score = None
            is_model_accepted = False

            if best_model is not None:
                y_best_pred = best_model.predict(x_test)
                best_model_score = (
                    self.model_evaluation_config.UTILS.get_model_score(
                        y_test, y_best_pred
                    )
                )
                logging.info(f"Best existing model score: {best_model_score}")

                improved_score = trained_model_score - best_model_score
                if improved_score >= self.model_evaluation_config.CHANGED_THRESHOLD:
                    is_model_accepted = True
                    logging.info(
                        f"Trained model is accepted as it improved score by {improved_score} (threshold: {self.model_evaluation_config.CHANGED_THRESHOLD})"
                    )
                else:
                    is_model_accepted = False
                    logging.info(
                        f"Trained model is rejected. Improved score {improved_score} is below threshold {self.model_evaluation_config.CHANGED_THRESHOLD}"
                    )
            else:
                logging.info("No previous model found for comparison")
                base_model_score = 0.0
                try:
                    model_config = self.model_evaluation_config.UTILS.read_yaml_file(
                        filename=MODEL_CONFIG_FILE
                    )
                    base_model_score = float(model_config.get("base_model_score", 0.0))
                except Exception:
                    pass

                if trained_model_score >= base_model_score:
                    is_model_accepted = True
                    logging.info(
                        f"Trained model accepted with score {trained_model_score} (base score: {base_model_score})"
                    )
                else:
                    is_model_accepted = False
                    logging.info(
                        f"Trained model rejected with score {trained_model_score} (below base score: {base_model_score})"
                    )

            # Preparing evaluation report
            eval_report = {
                "trained_model_score": float(trained_model_score),
                "best_model_score": (
                    float(best_model_score) if best_model_score is not None else None
                ),
                "is_model_accepted": is_model_accepted,
                "difference": float(
                    trained_model_score
                    - (best_model_score if best_model_score is not None else 0.0)
                ),
            }

            # Saving evaluation report
            report_file_path = (
                self.model_evaluation_config.EVALUATION_REPORT_FILE_PATH
            )
            self.model_evaluation_config.UTILS.write_json_to_yaml_file(
                eval_report, report_file_path
            )
            logging.info(f"Saved evaluation report to {report_file_path}")

            model_evaluation_artifact = ModelEvaluationArtifacts(
                is_model_accepted=is_model_accepted,
                trained_model_score=float(trained_model_score),
                best_model_score=(
                    float(best_model_score) if best_model_score is not None else 0.0
                ),
                evaluation_report_file_path=report_file_path,
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info(
                "Exited initiate_model_evaluation method of ModelEvaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise shippingException(e, sys) from e
