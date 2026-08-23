import os
import sys
import boto3
import dill
from shipment.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    AWS_REGION_NAME,
)
from shipment.exception import shippingException
from shipment.logger import logging


class S3Operation:
    def __init__(self):
        """
        Initializes the S3 client and resource using environment variables or AWS credentials.
        """
        try:
            self.access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY, None)
            self.secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY, None)
            self.region_name = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", AWS_REGION_NAME))

            session_kwargs = {}
            if self.access_key_id and self.secret_access_key:
                session_kwargs["aws_access_key_id"] = self.access_key_id
                session_kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.region_name:
                session_kwargs["region_name"] = self.region_name

            self.session = boto3.Session(**session_kwargs)
            self.s3_client = self.session.client("s3")
            self.s3_resource = self.session.resource("s3")

        except Exception as e:
            raise shippingException(e, sys) from e

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = False,
    ) -> None:
        """
        Method Name :   upload_file
        Description :   This method uploads a file to S3 bucket.
        """
        logging.info("Entered upload_file method of S3Operation class")
        try:
            logging.info(
                f"Uploading {from_filename} to bucket: {bucket_name} as {to_filename}"
            )
            self.s3_client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(
                f"Uploaded {from_filename} to bucket: {bucket_name} as {to_filename}"
            )

            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file: {from_filename}")

            logging.info("Exited upload_file method of S3Operation class")

        except Exception as e:
            raise shippingException(e, sys) from e

    def download_file(
        self, bucket_name: str, bucket_file_name: str, dest_file_name: str
    ) -> None:
        """
        Method Name :   download_file
        Description :   This method downloads a file from S3 bucket.
        """
        logging.info("Entered download_file method of S3Operation class")
        try:
            dest_dir = os.path.dirname(dest_file_name)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

            logging.info(
                f"Downloading {bucket_file_name} from bucket: {bucket_name} to {dest_file_name}"
            )
            self.s3_client.download_file(bucket_name, bucket_file_name, dest_file_name)
            logging.info(
                f"Downloaded {bucket_file_name} from bucket: {bucket_name} to {dest_file_name}"
            )
            logging.info("Exited download_file method of S3Operation class")

        except Exception as e:
            raise shippingException(e, sys) from e

    def is_model_present(self, bucket_name: str, s3_model_key: str) -> bool:
        """
        Method Name :   is_model_present
        Description :   This method checks if a model file is present in S3 bucket.
        """
        logging.info("Entered is_model_present method of S3Operation class")
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_model_key)
            logging.info(f"Model {s3_model_key} is present in bucket {bucket_name}")
            return True
        except Exception:
            logging.info(f"Model {s3_model_key} is not present in bucket {bucket_name}")
            return False

    def load_model(
        self, model_name: str, bucket_name: str, model_dir: str = None
    ) -> object:
        """
        Method Name :   load_model
        Description :   This method loads a serialized model object directly from S3 bucket.
        """
        logging.info("Entered load_model method of S3Operation class")
        try:
            logging.info(f"Loading model {model_name} from S3 bucket {bucket_name}")
            response = self.s3_client.get_object(Bucket=bucket_name, Key=model_name)
            model_bytes = response["Body"].read()
            model_object = dill.loads(model_bytes)
            logging.info(f"Loaded model {model_name} from S3 bucket {bucket_name}")
            return model_object

        except Exception as e:
            raise shippingException(e, sys) from e

    def sync_folder_to_s3(self, folder: str, aws_bucket_url: str) -> None:
        """
        Method Name :   sync_folder_to_s3
        Description :   Syncs a local folder to S3 bucket using AWS CLI.
        """
        logging.info("Entered sync_folder_to_s3 method of S3Operation class")
        try:
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            os.system(command)
            logging.info("Exited sync_folder_to_s3 method of S3Operation class")
        except Exception as e:
            raise shippingException(e, sys) from e

    def sync_folder_from_s3(self, folder: str, aws_bucket_url: str) -> None:
        """
        Method Name :   sync_folder_from_s3
        Description :   Syncs an S3 bucket folder to local directory using AWS CLI.
        """
        logging.info("Entered sync_folder_from_s3 method of S3Operation class")
        try:
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            os.system(command)
            logging.info("Exited sync_folder_from_s3 method of S3Operation class")
        except Exception as e:
            raise shippingException(e, sys) from e
