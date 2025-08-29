import logging
import sys
from datetime import datetime
import os

import s3fs
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import matplotlib.font_manager as fm
import polars as pl
import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.s3_utils import read_parquet_from_s3, read_csv_from_s3, upload_to_s3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def generate_raw_eda_report(df):
    """Generate an EDA report for raw data and save it to a local file."""
    today_str = datetime.today().strftime('%Y-%m-%d')
    filename = f"yugioh_raw_data_eda_report_{today_str}.html"
    local_path = f"/tmp/{filename}"

    try:
        logging.info(f"Generating raw data EDA report at {local_path}")
        
        # Convert to pandas if it's a Polars DataFrame
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Generate report with minimal configuration for raw data analysis
        report = ProfileReport(
            df,
            title="Yu-Gi-Oh Raw Data EDA Report",
            minimal=True,  # Use minimal mode for faster processing of large raw datasets
            explorative=False  # Disable explorative analysis for raw data
        )

        report.to_file(local_path)
        logging.info("Raw data EDA report generation complete")
        return local_path, filename
    except Exception as e:
        logging.exception("Failed to generate raw data EDA report")
        raise


def upload_raw_eda_report_to_s3(local_path: str, filename: str, bucket: str = "yugioh-data"):
    """Upload the generated raw data EDA report to S3."""
    try:
        # Read the HTML file as bytes
        with open(local_path, 'rb') as f:
            report_data = f.read()
        
        # Generate S3 key with current date for raw data reports
        month_str = datetime.today().strftime('%Y-%m')
        s3_key = f"reports/raw_data/{month_str}/{filename}"
        
        logging.info(f"Uploading raw data EDA report to s3://{bucket}/{s3_key}")
        
        # Upload to S3 with proper content type for HTML
        upload_to_s3(
            bucket=bucket,
            key=s3_key,
            data=report_data,
            content_type="text/html"
        )
        
        logging.info(f"Successfully uploaded raw data EDA report to s3://{bucket}/{s3_key}")
        
        # Clean up local file
        if os.path.exists(local_path):
            os.remove(local_path)
            logging.info(f"Cleaned up local file: {local_path}")
            
        return f"s3://{bucket}/{s3_key}"
        
    except Exception as e:
        logging.exception("Failed to upload raw data EDA report to S3")
        raise


def load_raw_data_from_s3(bucket: str = "yugioh-data"):
    """Load raw data from S3. You can modify this function to load your specific raw data."""
    try:
        # Example: Load raw card data - adjust the key path as needed
        logging.info("Loading raw card data from S3...")
        
        # Try loading from different potential raw data locations
        possible_keys = [
            "raw_data/cards/cards.csv",
            "raw_data/cards.csv", 
            "data/raw/cards.csv",
            "cards.csv"
        ]
        
        df = None
        for key in possible_keys:
            try:
                logging.info(f"Attempting to load from s3://{bucket}/{key}")
                df = read_csv_from_s3(bucket=bucket, key=key, use_polars=True)
                logging.info(f"Successfully loaded data from s3://{bucket}/{key}")
                break
            except Exception as e:
                logging.warning(f"Failed to load from s3://{bucket}/{key}: {e}")
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find raw data in any of the expected S3 locations")
            
        return df
        
    except Exception as e:
        logging.exception("Failed to load raw data from S3")
        raise


if __name__ == "__main__":
    try:
        # Load raw data from S3
        logging.info("Loading raw data from S3...")
        df = load_raw_data_from_s3(bucket="yugioh-data")
        
        logging.info(f"Loaded raw DataFrame with {df.height} rows and {df.width} columns")
        
        # Display basic info about the raw data
        logging.info("Raw data columns:")
        for col in df.columns:
            logging.info(f"  - {col}: {df[col].dtype}")

        # Generate EDA report for raw data
        local_path, filename = generate_raw_eda_report(df)
        
        # Upload report to S3
        s3_path = upload_raw_eda_report_to_s3(local_path, filename, bucket="yugioh-data")
        
        logging.info(f"Raw data EDA workflow completed successfully. Report available at: {s3_path}")
        
    except Exception as e:
        logging.exception("Raw data EDA workflow failed")
        sys.exit(1)
