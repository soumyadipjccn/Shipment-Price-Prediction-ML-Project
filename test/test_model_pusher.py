import os
import tempfile
from unittest.mock import MagicMock

from shipment.components.model_pusher import ModelPusher
from shipment.configuration.s3_operations import S3Operation
from shipment.entity.config_entity import ModelPusherConfig
from shipment.entity.artifacts_entity import (
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts,
)


def test_model_pusher_accepted_model():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_file = os.path.join(temp_dir, "shipping_price_model.pkl")
        with open(model_file, "w") as f:
            f.write("mock_model_content")

        pusher_artifacts_dir = os.path.join(temp_dir, "pusher_artifacts")

        config = ModelPusherConfig(
            BUCKET_NAME="test-bucket",
            S3_MODEL_KEY_PATH="model/shipping_price_model.pkl",
            MODEL_PUSHER_ARTIFACTS_DIR=pusher_artifacts_dir,
        )

        trainer_artifacts = ModelTrainerArtifacts(
            trained_model_file_path=model_file
        )

        evaluation_artifacts = ModelEvaluationArtifacts(
            is_model_accepted=True,
            trained_model_score=0.95,
            best_model_score=0.85,
            evaluation_report_file_path="report.yaml"
        )

        mock_s3 = MagicMock(spec=S3Operation)

        model_pusher = ModelPusher(
            model_pusher_config=config,
            model_trainer_artifact=trainer_artifacts,
            model_evaluation_artifact=evaluation_artifacts,
            s3_op=mock_s3,
        )

        pusher_artifacts = model_pusher.initiate_model_pusher()

        assert isinstance(pusher_artifacts, ModelPusherArtifacts)
        assert pusher_artifacts.is_model_pushed is True
        assert pusher_artifacts.bucket_name == "test-bucket"
        assert pusher_artifacts.s3_model_path == "model/shipping_price_model.pkl"
        mock_s3.upload_file.assert_called_once_with(
            from_filename=model_file,
            to_filename="model/shipping_price_model.pkl",
            bucket_name="test-bucket",
            remove=False,
        )
        assert os.path.exists(pusher_artifacts_dir)
        print("Model pusher accepted model test passed successfully!")


def test_model_pusher_rejected_model():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_file = os.path.join(temp_dir, "shipping_price_model.pkl")
        with open(model_file, "w") as f:
            f.write("mock_model_content")

        pusher_artifacts_dir = os.path.join(temp_dir, "pusher_artifacts")

        config = ModelPusherConfig(
            BUCKET_NAME="test-bucket",
            S3_MODEL_KEY_PATH="model/shipping_price_model.pkl",
            MODEL_PUSHER_ARTIFACTS_DIR=pusher_artifacts_dir,
        )

        trainer_artifacts = ModelTrainerArtifacts(
            trained_model_file_path=model_file
        )

        evaluation_artifacts = ModelEvaluationArtifacts(
            is_model_accepted=False,
            trained_model_score=0.50,
            best_model_score=0.85,
            evaluation_report_file_path="report.yaml"
        )

        mock_s3 = MagicMock(spec=S3Operation)

        model_pusher = ModelPusher(
            model_pusher_config=config,
            model_trainer_artifact=trainer_artifacts,
            model_evaluation_artifact=evaluation_artifacts,
            s3_op=mock_s3,
        )

        pusher_artifacts = model_pusher.initiate_model_pusher()

        assert isinstance(pusher_artifacts, ModelPusherArtifacts)
        assert pusher_artifacts.is_model_pushed is False
        mock_s3.upload_file.assert_not_called()
        print("Model pusher rejected model test passed successfully!")


if __name__ == "__main__":
    test_model_pusher_accepted_model()
    test_model_pusher_rejected_model()
