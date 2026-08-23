import os
import sys
from shipment.logger import logging
from shipment.exception import shippingException
from shipment.entity.config_entity import ModelPusherConfig
from shipment.entity.artifacts_entity import (
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts,
)
from shipment.configuration.s3_operations import S3Operation


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifact: ModelTrainerArtifacts,
        model_evaluation_artifact: ModelEvaluationArtifacts,
        s3_op: S3Operation = None,
    ):
        """
        :param model_pusher_config: Configuration for model pusher
        :param model_trainer_artifact: Artifact from model trainer stage
        :param model_evaluation_artifact: Artifact from model evaluation stage
        :param s3_op: S3Operation object for interacting with AWS S3
        """
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        self.s3_op = s3_op if s3_op is not None else S3Operation()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        Method Name :   initiate_model_pusher

        Description :   This method pushes the trained model to AWS S3 bucket if accepted by evaluation.
        
        Output      :   Model pusher artifacts
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            os.makedirs(
                self.model_pusher_config.MODEL_PUSHER_ARTIFACTS_DIR,
                exist_ok=True,
            )
            logging.info(
                f"Created artifacts directory for {os.path.basename(self.model_pusher_config.MODEL_PUSHER_ARTIFACTS_DIR)}"
            )

            # Check if model was accepted in evaluation
            is_model_accepted = self.model_evaluation_artifact.is_model_accepted
            if not is_model_accepted:
                logging.info(
                    "Trained model was not accepted in model evaluation stage. Skipping push to AWS S3."
                )
                model_pusher_artifact = ModelPusherArtifacts(
                    bucket_name=self.model_pusher_config.BUCKET_NAME,
                    s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
                    is_model_pushed=False,
                )
                return model_pusher_artifact

            logging.info("Trained model accepted. Uploading model to AWS S3 bucket...")
            trained_model_path = self.model_trainer_artifact.trained_model_file_path

            # Upload model to S3
            self.s3_op.upload_file(
                from_filename=trained_model_path,
                to_filename=self.model_pusher_config.S3_MODEL_KEY_PATH,
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                remove=False,
            )
            logging.info(
                f"Uploaded trained model from {trained_model_path} to S3 bucket '{self.model_pusher_config.BUCKET_NAME}' at key '{self.model_pusher_config.S3_MODEL_KEY_PATH}'"
            )

            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
                is_model_pushed=True,
            )
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise shippingException(e, sys) from e
