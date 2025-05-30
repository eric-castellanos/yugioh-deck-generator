import os
import logging
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def upload_to_s3(
    bucket: str,
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    s3_client=None,
):
    """
    Uploads in-memory bytes (e.g., CSV or Parquet) to S3.

    Parameters:
    - bucket: Target S3 bucket name.
    - key: Full S3 key (path + filename).
    - data: File content as bytes (e.g., BytesIO.getvalue()).
    - content_type: MIME type (default: application/octet-stream).
    - s3_client: Optional boto3 S3 client.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    try:
        logging.info(f"Uploading to s3://{bucket}/{key}")
        s3_client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        logging.info(f"Successfully uploaded to s3://{bucket}/{key}")
    except (BotoCoreError, ClientError) as e:
        logging.exception(f"Failed to upload to s3://{bucket}/{key}")
        raise