import logging
import sys
from datetime import datetime
import os

import s3fs
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import matplotlib.font_manager as fm
import polars as pl
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def read_raw(path : str):
    fs = s3fs.S3FileSystem(anon=False)

    try:

        logging.info("Reading raw data from S3")

        # Load the Parquet file from S3
        with fs.open(path, "rb") as f:
            df = pl.read_parquet(f)

        return df
    
    except FileNotFoundError:
        logging.exception(f"Error: The file '{path}' was not found.")

def generate_eda_report(df):
    """Generate an EDA report and save it to a local file."""
    today_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"yugioh_eda_report_{today_str}.html"
    local_path = f"/tmp/{filename}"

    try:
        logging.info(f"Generating EDA report at {local_path}")
        df = df.to_pandas()

        # ✅ Use minimal=True to disable heavy/optional visualizations
        report = ProfileReport(
            df,
            title="Yu-Gi-Oh Card EDA Report"
        )

        report.to_file(local_path)
        logging.info("EDA report generation complete")
        return local_path
    except Exception as e:
        logging.exception("Failed to generate EDA report")
        raise

def upload_eda_report_to_s3(local_path: str, bucket: str, prefix: str):
    """Uploads the EDA report from local_path to S3."""
    filename = os.path.basename(local_path)
    month_str = datetime.today().strftime('%Y-%m')
    s3_key = f"{prefix}/{month_str}/{filename}"

    try:
        logging.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, s3_key)
        logging.info(f"Successfully uploaded EDA report to s3://{bucket}/{s3_key}")
    except (BotoCoreError, ClientError) as e:
        logging.exception(f"Failed to upload {local_path} to S3 bucket '{bucket}' with key '{s3_key}'")
        raise

if __name__ == "__main__":

    # Example raw S3 file path
    raw_s3_path = "s3://yugioh-data/raw/2025-05/yugioh_raw_2025-05-22.parquet"
    df = read_raw(raw_s3_path)

    # Generate and upload report
    local_path = generate_eda_report(df)
    upload_eda_report_to_s3(local_path, bucket="yugioh-data", prefix="reports")

