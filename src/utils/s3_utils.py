import os
import logging
from datetime import datetime
import io
from typing import Optional, Dict, Any, Union

import boto3
import s3fs
from botocore.exceptions import BotoCoreError, ClientError
import polars as pl
import pandas as pd


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

def read_parquet_from_s3(bucket: str, key: str) -> pl.DataFrame:
    """
    Reads a Parquet file from S3 and returns it as a Polars DataFrame.

    Parameters:
    - bucket: Name of the S3 bucket
    - key: Key (path) to the Parquet file in S3

    Returns:
    - Polars DataFrame containing the file's contents
    """
    try:
        logging.info(f"Reading Parquet from s3://{bucket}/{key}")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj["Body"].read()))

    except (BotoCoreError, ClientError) as e:
        logging.exception(f"Failed to read s3://{bucket}/{key}")
        raise

def read_csv_from_s3(
    bucket: str, 
    key: str, 
    pandas_kwargs: Optional[Dict[str, Any]] = None,
    use_polars: bool = False
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Reads a CSV file from S3 and returns it as a DataFrame.

    Parameters:
    -----------
    bucket : str
        Name of the S3 bucket
    key : str
        Key (path) to the CSV file in S3
    pandas_kwargs : dict, optional
        Additional keyword arguments to pass to pandas.read_csv
    use_polars : bool, default=False
        If True, returns a Polars DataFrame, otherwise returns Pandas

    Returns:
    --------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame containing the file's contents
    """
    try:
        logging.info(f"Reading CSV from s3://{bucket}/{key}")
        s3_client = boto3.client("s3")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        
        if use_polars:
            return pl.read_csv(io.BytesIO(data))
        else:
            kwargs = pandas_kwargs or {}
            return pd.read_csv(io.BytesIO(data), **kwargs)

    except (BotoCoreError, ClientError) as e:
        logging.exception(f"Failed to read CSV s3://{bucket}/{key}")
        raise