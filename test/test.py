import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.linear_model import LinearRegression

from shipment.pipline.prediction_pipeline import (
    ShipmentData,
    CustomData,
    PredictionPipeline,
    PredictPipeline,
    SingleValuePredictionPipeline,
)
from shipment.entity.config_entity import PredictionPipelineConfig
from shipment.components.model_trainer import CostModel
from shipment.configuration.s3_operations import S3Operation
from shipment.utils.main_utils import MainUtils


class DummyPreprocessor:
    def transform(self, X):
        # Return numeric values only for dummy model
        numeric_cols = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
        return X[numeric_cols].values


def test_shipment_data_dataframe_creation():
    data = ShipmentData(
        artist_reputation=0.75,
        height=15.0,
        width=10.0,
        weight=250.0,
        material="Bronze",
        price_of_sculpture=5000.0,
        base_shipping_price=120.0,
        international="Yes",
        express_shipment="No",
        installation_included="Yes",
        transport="Airways",
        fragile="No",
        customer_information="Wealthy",
        remote_location="No",
    )

    df = data.get_shipment_input_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["Artist Reputation"].iloc[0] == 0.75
    assert df["Material"].iloc[0] == "Bronze"
    assert df["Price Of Sculpture"].iloc[0] == 5000.0

    d = data.get_shipment_data_as_dict()
    assert isinstance(d, dict)
    assert "Artist Reputation" in d
    assert d["Transport"] == ["Airways"]

    # Test alias CustomData
    custom_data = CustomData(
        artist_reputation=0.5,
        height=10.0,
        width=5.0,
        weight=100.0,
        material="Clay",
        price_of_sculpture=2000.0,
        base_shipping_price=50.0,
        international="No",
        express_shipment="Yes",
        installation_included="No",
        transport="Roadways",
        fragile="Yes",
        customer_information="Working Class",
        remote_location="Yes",
    )
    df_custom = custom_data.get_data_as_data_frame()
    assert isinstance(df_custom, pd.DataFrame)
    print("ShipmentData and CustomData tests passed successfully!")


def test_prediction_pipeline_with_direct_model():
    # Train dummy model
    numeric_cols = ["Artist Reputation", "Height", "Width", "Weight", "Price Of Sculpture", "Base Shipping Price"]
    X_train = pd.DataFrame({
        "Artist Reputation": [0.5, 0.8],
        "Height": [10.0, 20.0],
        "Width": [5.0, 10.0],
        "Weight": [100.0, 200.0],
        "Price Of Sculpture": [1000.0, 2000.0],
        "Base Shipping Price": [50.0, 100.0],
    })
    y_train = pd.Series([200.0, 400.0])

    raw_model = LinearRegression()
    raw_model.fit(X_train.values, y_train)

    cost_model = CostModel(preprocessing_object=DummyPreprocessor(), trained_model_object=raw_model)

    pipeline = PredictionPipeline(model=cost_model)

    data = ShipmentData(
        artist_reputation=0.5,
        height=10.0,
        width=5.0,
        weight=100.0,
        material="Bronze",
        price_of_sculpture=1000.0,
        base_shipping_price=50.0,
        international="No",
        express_shipment="No",
        installation_included="No",
        transport="Roadways",
        fragile="No",
        customer_information="Working Class",
        remote_location="No",
    )

    df = data.get_shipment_input_data_frame()
    preds = pipeline.predict(df)
    assert len(preds) == 1
    assert np.isclose(preds[0], 200.0, atol=1e-2)
    print("PredictionPipeline direct model test passed successfully!")


def test_prediction_pipeline_s3_loading():
    mock_s3 = MagicMock(spec=S3Operation)
    mock_s3.is_model_present.return_value = True

    raw_model = LinearRegression()
    X_dummy = np.array([[0.5, 10.0, 5.0, 100.0, 1000.0, 50.0]])
    y_dummy = np.array([300.0])
    raw_model.fit(X_dummy, y_dummy)
    cost_model = CostModel(preprocessing_object=DummyPreprocessor(), trained_model_object=raw_model)

    mock_s3.load_model.return_value = cost_model

    config = PredictionPipelineConfig(
        BUCKET_NAME="test-bucket",
        S3_MODEL_KEY_PATH="model/shipping_price_model.pkl",
    )

    pipeline = PredictionPipeline(prediction_pipeline_config=config, s3_op=mock_s3)

    data = ShipmentData(
        artist_reputation=0.5,
        height=10.0,
        width=5.0,
        weight=100.0,
        material="Bronze",
        price_of_sculpture=1000.0,
        base_shipping_price=50.0,
        international="No",
        express_shipment="No",
        installation_included="No",
        transport="Roadways",
        fragile="No",
        customer_information="Working Class",
        remote_location="No",
    )

    preds = pipeline.predict(data.get_shipment_input_data_frame())
    assert len(preds) == 1
    assert np.isclose(preds[0], 300.0, atol=1e-2)
    mock_s3.is_model_present.assert_called_once()
    mock_s3.load_model.assert_called_once()
    print("PredictionPipeline S3 loading test passed successfully!")


def test_prediction_pipeline_local_loading_and_batch():
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "shipping_price_model.pkl")

        raw_model = LinearRegression()
        X_dummy = np.array([[0.5, 10.0, 5.0, 100.0, 1000.0, 50.0]])
        y_dummy = np.array([500.0])
        raw_model.fit(X_dummy, y_dummy)
        cost_model = CostModel(preprocessing_object=DummyPreprocessor(), trained_model_object=raw_model)

        utils = MainUtils()
        utils.save_object(model_path, cost_model)

        config = PredictionPipelineConfig(
            MODEL_FILE_PATH=model_path,
        )

        mock_s3 = MagicMock(spec=S3Operation)
        mock_s3.is_model_present.return_value = False

        pipeline = PredictionPipeline(prediction_pipeline_config=config, s3_op=mock_s3)

        # Batch prediction test
        input_csv = os.path.join(temp_dir, "batch_input.csv")
        output_csv = os.path.join(temp_dir, "batch_output.csv")

        batch_df = pd.DataFrame({
            "Artist Reputation": [0.5, 0.5],
            "Height": [10.0, 10.0],
            "Width": [5.0, 5.0],
            "Weight": [100.0, 100.0],
            "Material": ["Bronze", "Clay"],
            "Price Of Sculpture": [1000.0, 1000.0],
            "Base Shipping Price": [50.0, 50.0],
            "International": ["No", "Yes"],
            "Express Shipment": ["No", "Yes"],
            "Installation Included": ["No", "Yes"],
            "Transport": ["Roadways", "Airways"],
            "Fragile": ["No", "Yes"],
            "Customer Information": ["Working Class", "Wealthy"],
            "Remote Location": ["No", "Yes"],
        })
        batch_df.to_csv(input_csv, index=False)

        result_df = pipeline.predict_batch(input_csv_path=input_csv, output_csv_path=output_csv)
        assert "Predicted_Cost" in result_df.columns
        assert len(result_df) == 2
        assert os.path.exists(output_csv)
        print("PredictionPipeline local loading and batch prediction tests passed successfully!")


if __name__ == "__main__":
    test_shipment_data_dataframe_creation()
    test_prediction_pipeline_with_direct_model()
    test_prediction_pipeline_s3_loading()
    test_prediction_pipeline_local_loading_and_batch()
