"""
Create three different feature-engineered datasets for K-means clustering comparison:
1. TF-IDF features only
2. Word embedding features only  
3. Combined TF-IDF + word embedding features
"""

import io
import logging
from datetime import datetime
from typing import Literal

import boto3
import polars as pl

from src.utils.s3_utils import upload_to_s3
from feature_engineering import (
    read_from_s3, 
    encode_categoricals, 
    normalize_numeric_columns, 
    drop_columns,
    cats_to_dummies_eager,
    add_tfidf_description_features,
    add_word_embedding_features,
    add_combined_text_features
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_feature_dataset(feature_type: Literal["tfidf", "embeddings", "combined"]) -> str:
    """
    Create a feature-engineered dataset with specified feature type.
    
    Parameters:
    - feature_type: Type of text features to use
    
    Returns:
    - S3 key where the dataset was uploaded
    """
    logging.info(f"Creating {feature_type} feature dataset...")
    
    # Load base data
    lf = read_from_s3("yugioh-data", "raw/2025-05/yugioh_raw_2025-05-22.parquet")
    
    # Apply base preprocessing
    lf = encode_categoricals(lf, ["type", "attribute", "archetype"])
    lf = normalize_numeric_columns(lf, ["atk", "def", "level"])
    lf = drop_columns(lf)
    
    # Collect to DataFrame for text feature processing
    df = lf.collect()
    
    # Add text features based on type
    if feature_type == "tfidf":
        df = add_tfidf_description_features(df, max_features=300, n_components=50)
        logging.info("Added TF-IDF features")
        
    elif feature_type == "embeddings":
        df = add_word_embedding_features(df, model_name="all-MiniLM-L6-v2", n_components=50)
        logging.info("Added word embedding features")
        
    elif feature_type == "combined":
        df = add_combined_text_features(
            df, 
            tfidf_max_features=300, 
            tfidf_components=25,
            embedding_model="all-MiniLM-L6-v2", 
            embedding_components=25
        )
        logging.info("Added combined TF-IDF + embedding features")
    
    # Convert categorical to dummies
    df = cats_to_dummies_eager(df)
    
    # Log dataset info
    logging.info(f"Final dataset shape: {df.shape}")
    logging.info(f"Feature columns: {[col for col in df.columns if col not in ['id', 'name', 'archetype']]}")
    
    # Write to S3
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    
    month_str = datetime.today().strftime('%Y-%m')
    s3_key = f"processed/feature_engineered/{month_str}/feature_engineered_{feature_type}.parquet"
    
    upload_to_s3(
        bucket="yugioh-data",
        key=s3_key,
        data=buffer.getvalue(),
        content_type="application/octet-stream"
    )
    
    logging.info(f"Uploaded {feature_type} dataset to s3://yugioh-data/{s3_key}")
    return s3_key

def main():
    """Create all three feature datasets."""
    datasets = {}
    
    for feature_type in ["tfidf", "embeddings", "combined"]:
        try:
            s3_key = create_feature_dataset(feature_type)
            datasets[feature_type] = s3_key
            logging.info(f"✅ Created {feature_type} dataset: {s3_key}")
        except Exception as e:
            logging.error(f"❌ Failed to create {feature_type} dataset: {e}")
            raise
    
    logging.info("🎉 All feature datasets created successfully!")
    logging.info("Dataset locations:")
    for feature_type, s3_key in datasets.items():
        logging.info(f"  {feature_type:>10}: s3://yugioh-data/{s3_key}")

if __name__ == "__main__":
    main()
