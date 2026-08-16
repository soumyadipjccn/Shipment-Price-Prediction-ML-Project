import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from shipment.components.model_evaluation import ModelEvaluation
from shipment.components.model_trainer import CostModel
from shipment.entity.config_entity import ModelEvaluationConfig
from shipment.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
)
from shipment.utils.main_utils import MainUtils


def test_model_evaluation():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
            "target": [3.0, 6.0, 9.0, 12.0, 15.0]
        })
        test_data_path = os.path.join(temp_dir, "test.csv")
        test_df.to_csv(test_data_path, index=False)

        # Train a mock dummy model
        X_train = test_df[["feature1", "feature2"]]
        y_train = test_df["target"]
        
        # Preprocessor dummy class with transform method
        class DummyPreprocessor:
            def transform(self, X):
                return X
                
        raw_model = LinearRegression()
        raw_model.fit(X_train, y_train)
        cost_model = CostModel(preprocessing_object=DummyPreprocessor(), trained_model_object=raw_model)

        utils = MainUtils()
        model_path = os.path.join(temp_dir, "model.pkl")
        utils.save_object(model_path, cost_model)

        # Configure evaluation config and artifacts
        eval_artifacts_dir = os.path.join(temp_dir, "eval_artifacts")
        report_file_path = os.path.join(eval_artifacts_dir, "evaluation_report.yaml")

        config = ModelEvaluationConfig(
            MODEL_EVALUATION_ARTIFACTS_DIR=eval_artifacts_dir,
            EVALUATION_REPORT_FILE_PATH=report_file_path,
            CHANGED_THRESHOLD=0.01,
            SCHEMA_CONFIG={"target_column": "target"},
            UTILS=utils
        )

        ingestion_artifacts = DataIngestionArtifacts(
            train_data_file_path="",
            test_data_file_path=test_data_path
        )
        transformation_artifacts = DataTransformationArtifacts(
            transformed_object_file_path="",
            transformed_train_file_path="",
            transformed_test_file_path=""
        )
        trainer_artifacts = ModelTrainerArtifacts(
            trained_model_file_path=model_path
        )

        model_eval = ModelEvaluation(
            model_evaluation_config=config,
            data_ingestion_artifact=ingestion_artifacts,
            data_transformation_artifact=transformation_artifacts,
            model_trainer_artifact=trainer_artifacts
        )

        evaluation_artifacts = model_eval.initiate_model_evaluation()

        assert evaluation_artifacts.is_model_accepted is True
        assert evaluation_artifacts.trained_model_score >= 0.99
        assert os.path.exists(report_file_path)
        print("Model evaluation unit test passed successfully!")


if __name__ == "__main__":
    test_model_evaluation()
