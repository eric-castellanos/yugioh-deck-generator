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

from src.utils.s3_utils import read_parquet_from_s3, upload_to_s3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def generate_eda_report(df):
    """Generate an EDA report and save it to a local file."""
    today_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"yugioh_deck_scoring_eda_report_{today_str}.html"
    local_path = f"/tmp/{filename}"

    try:
        logging.info(f"Generating EDA report at {local_path}")
        df = df.to_pandas()

        # ✅ Use minimal=True to disable heavy/optional visualizations
        report = ProfileReport(
            df,
            title="Yu-Gi-Oh Deck Scoring Card EDA Report"
        )

        report.to_file(local_path)
        logging.info("EDA report generation complete")
        return local_path, filename
    except Exception as e:
        logging.exception("Failed to generate EDA report")
        raise


def upload_eda_report_to_s3(local_path: str, filename: str, bucket: str = "yugioh-data"):
    """Upload the generated EDA report to S3."""
    try:
        # Read the HTML file as bytes
        with open(local_path, 'rb') as f:
            report_data = f.read()
        
        # Generate S3 key with current date
        month_str = datetime.today().strftime('%Y-%m')
        s3_key = f"reports/deck_scoring/{month_str}/{filename}"
        
        logging.info(f"Uploading EDA report to s3://{bucket}/{s3_key}")
        
        # Upload to S3 with proper content type for HTML
        upload_to_s3(
            bucket=bucket,
            key=s3_key,
            data=report_data,
            content_type="text/html"
        )
        
        logging.info(f"Successfully uploaded EDA report to s3://{bucket}/{s3_key}")
        
        # Clean up local file
        if os.path.exists(local_path):
            os.remove(local_path)
            logging.info(f"Cleaned up local file: {local_path}")
            
        return f"s3://{bucket}/{s3_key}"
        
    except Exception as e:
        logging.exception("Failed to upload EDA report to S3")
        raise

if __name__ == "__main__":
    try:
        # Load data from S3
        logging.info("Loading data from S3...")
        df = read_parquet_from_s3(
            bucket="yugioh-data", 
            key="processed/feature_engineered/deck_scoring/2025-08/feature_engineered.parquet"
        )
        logging.info(f"Loaded DataFrame with {df.height} rows and {df.width} columns")

        # Generate EDA report
        local_path, filename = generate_eda_report(df)
        
        # Upload report to S3
        s3_path = upload_eda_report_to_s3(local_path, filename, bucket="yugioh-data")
        
        logging.info(f"EDA workflow completed successfully. Report available at: {s3_path}")
        
    except Exception as e:
        logging.exception("EDA workflow failed")
        sys.exit(1)