import os
from os import environ
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts", TIMESTAMP)


MODEL_CONFIG_FILE = "config/model.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"

DB_URL = os.getenv("MONGO_DB_URL", "")
TEST_SIZE = 0.2


# Database related constants
DB_NAME = "ShipmentDB"
COLLECTION_NAME = "shipment_collection"

# Data Ingestion related constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_TRAIN_DIR = "Train"
DATA_INGESTION_TEST_DIR = "Test"
DATA_INGESTION_TRAIN_FILE_NAME = "train.csv"
DATA_INGESTION_TEST_FILE_NAME = "test.csv"


# Data Validation related constants
DATA_VALIDATION_ARTIFACT_DIR = "DataValidationArtifacts"
DATA_DRIFT_FILE_NAME = "DataDriftReport.yaml"



DATA_TRANSFORMATION_ARTIFCATS_DIR = "DataTransformationArtifacts"
TRANSFORMED_TRAIN_DATA_DIR = "TransformedTrain"
TRANSFORMED_TEST_DATA_DIR = "TransformedTest"
TRANSFORMED_TRAIN_DATA_FILE_NAME = "transformed_train_data.npz"
TRANSFORMED_TEST_DATA_FILE_NAME = "transformed_test_data.npz"
PREPROCESSOR_OBJECT_FILE_NAME = "shipping_preprocessor.pkl"


MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
MODEL_FILE_NAME = "shipping_price_model.pkl"
MODEL_SAVE_FORMAT = ".pkl"


# Model Evaluation related constants
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
MODEL_EVALUATION_REPORT_FILE_NAME = "model_evaluation_report.yaml"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02


# AWS & Model Pusher related constants
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"
AWS_REGION_NAME = "us-east-1"
MODEL_BUCKET_NAME = "shipment-price-model-bucket"
MODEL_PUSHER_BUCKET_NAME = "shipment-price-model-bucket"
MODEL_PUSHER_ARTIFACTS_DIR = "ModelPusherArtifacts"
MODEL_PUSHER_S3_KEY = "model"
S3_MODEL_KEY_PATH = "model/shipping_price_model.pkl"


