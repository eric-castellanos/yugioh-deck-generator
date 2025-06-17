from typing import List, Union, Literal
import io
import logging
from datetime import datetime

import boto3
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.utils.s3_utils import upload_to_s3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_from_s3(bucket: str, key: str) -> pl.DataFrame:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)

    lf = pl.scan_parquet(response['Body'])

    return lf

def encode_categoricals(lf: pl.LazyFrame, cat_cols: List[str], fill_value: str = "None") -> pl.LazyFrame:
    """
    Fill nulls in specified categorical columns and one-hot encode them.

    Parameters:
    - df: Polars DataFrame
    - cat_cols: List of column names to treat as categorical
    - fill_value: Value to use for nulls (default "None")

    Returns:
    - Transformed Polars DataFrame with one-hot encoded columns
    """

    ## may have to add to_dummies at some point in the training code
    lf = lf.with_columns([
        pl.col(col).fill_null("Unknown").cast(pl.Categorical) for col in ["type", "attribute", "archetype"]
    ])

    return lf

def normalize_numeric_columns(
    lf: pl.LazyFrame,
    numeric_cols: List[str],
    fill_value: float = 0.0
) -> pl.LazyFrame:
    """
    Fill nulls, cast to float, and z-score normalize numeric columns in a Polars DataFrame.

    Parameters:
    - df: Polars DataFrame
    - numeric_cols: List of numeric column names to normalize
    - fill_value: Value to fill nulls with (default 0.0)
    - add_suffix: Suffix for normalized columns (default '_norm')

    Returns:
    - Polars DataFrame with normalized columns added
    """

    # Fill and cast first
    lf = lf.with_columns([
        pl.col(col).fill_null(fill_value).cast(pl.Float32) for col in numeric_cols
    ])

    # Collect temporarily to compute stats
    df_temp = lf.select(numeric_cols).collect()
    stats = {
        col: {
            "mean": df_temp[col].mean(),
            "std": df_temp[col].std()
        } for col in numeric_cols
    }

    # Inject computed constants into lazy frame
    lf = lf.with_columns([
        ((pl.col(col) - stats[col]["mean"]) / stats[col]["std"]).alias(f"{col}_norm")
        for col in numeric_cols
    ])

    return lf

def drop_columns(
    df: Union[pl.LazyFrame, pl.DataFrame]
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Drops all price-related columns from the provided DataFrame or LazyFrame.

    Parameters:
    - df: A Polars DataFrame or LazyFrame

    Returns:
    - A Polars DataFrame or LazyFrame with price columns removed
    """
    cols = [
        "tcgplayer_price",
        "ebay_price",
        "amazon_price",
        "coolstuffinc_price",
        "image_url",
        "image_url_small",
        "image_url_cropped"
    ]
    return df.drop(cols)

def cats_to_dummies_eager(df : pl.DataFrame) -> pl.DataFrame:
    """
    Converts all categorical columns in the given Polars DataFrame into one-hot encoded columns.

    Parameters:
    - df: Polars DataFrame with categorical columns

    Returns:
    - Polars DataFrame with one-hot encoded columns replacing the original categoricals
    """
    # Identify all categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype == pl.Categorical]

    # Split: keep 'archetype', drop others
    cat_cols_to_drop = [col for col in cat_cols if col != "archetype"]

    # One-hot encode all categorical columns (including 'archetype')
    df_dummies = df.select(cat_cols).to_dummies()

    # Drop all categorical cols except 'archetype'
    df_base = df.drop(cat_cols_to_drop)

    # Combine
    return df_base.hstack(df_dummies)

# may be worth trying out different values of max_features and n_components to view perforomance differences
def add_tfidf_description_features(df: pl.DataFrame, 
                                   desc_col: str = "desc", 
                                   max_features: int = 300, 
                                   n_components: int = 50) -> pl.DataFrame:
    """
    Adds TF-IDF + TruncatedSVD features from the `desc` column to the input Polars DataFrame.

    Parameters:
    - df: Polars DataFrame containing card data.
    - desc_col: Column name containing the card descriptions.
    - max_features: Max vocabulary size for TF-IDF.
    - n_components: Number of dimensions for SVD.

    Returns:
    - Polars DataFrame with appended description features.
    """
    # Fill nulls in description and convert to list of strings
    descriptions = df.select(pl.col(desc_col).fill_null("")).to_series().to_list()

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=n_components)
    desc_features = svd.fit_transform(tfidf_matrix)

    # Convert to Polars DataFrame
    desc_df = pl.DataFrame(desc_features, schema=[f"desc_feat_{i}" for i in range(n_components)])

    # Append to original dataframe
    df = df.hstack(desc_df)

    return df


def add_word_embedding_features(df: pl.DataFrame, 
                              desc_col: str = "desc", 
                              model_name: str = "all-MiniLM-L6-v2",
                              n_components: int = 50) -> pl.DataFrame:
    """
    Adds word embedding features from the `desc` column using sentence transformers.

    Parameters:
    - df: Polars DataFrame containing card data.
    - desc_col: Column name containing the card descriptions.
    - model_name: Sentence transformer model name.
    - n_components: Number of dimensions after optional dimensionality reduction.

    Returns:
    - Polars DataFrame with appended word embedding features.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    # Fill nulls in description and convert to list of strings
    descriptions = df.select(pl.col(desc_col).fill_null("")).to_series().to_list()

    # Load pre-trained sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embeddings = model.encode(descriptions)
    
    # Optional dimensionality reduction if embeddings are too large
    if embeddings.shape[1] > n_components:
        svd = TruncatedSVD(n_components=n_components)
        embeddings = svd.fit_transform(embeddings)
    
    # Convert to Polars DataFrame
    embed_df = pl.DataFrame(embeddings, schema=[f"embed_feat_{i}" for i in range(embeddings.shape[1])])

    # Append to original dataframe
    df = df.hstack(embed_df)

    return df


def add_combined_text_features(df: pl.DataFrame, 
                             desc_col: str = "desc",
                             tfidf_max_features: int = 300,
                             tfidf_components: int = 25,
                             embedding_model: str = "all-MiniLM-L6-v2", 
                             embedding_components: int = 25) -> pl.DataFrame:
    """
    Adds both TF-IDF and word embedding features from the `desc` column.

    Parameters:
    - df: Polars DataFrame containing card data.
    - desc_col: Column name containing the card descriptions.
    - tfidf_max_features: Max vocabulary size for TF-IDF.
    - tfidf_components: Number of dimensions for TF-IDF SVD.
    - embedding_model: Sentence transformer model name.
    - embedding_components: Number of dimensions for embedding reduction.

    Returns:
    - Polars DataFrame with appended TF-IDF and embedding features.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    # Fill nulls in description and convert to list of strings
    descriptions = df.select(pl.col(desc_col).fill_null("")).to_series().to_list()

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    svd_tfidf = TruncatedSVD(n_components=tfidf_components)
    tfidf_features = svd_tfidf.fit_transform(tfidf_matrix)
    
    # Word embedding features
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(descriptions)
    
    if embeddings.shape[1] > embedding_components:
        svd_embed = TruncatedSVD(n_components=embedding_components)
        embeddings = svd_embed.fit_transform(embeddings)
    
    # Combine features
    combined_features = np.hstack([tfidf_features, embeddings])
    
    # Convert to Polars DataFrame
    feature_names = [f"tfidf_feat_{i}" for i in range(tfidf_components)] + \
                   [f"embed_feat_{i}" for i in range(embeddings.shape[1])]
    
    combined_df = pl.DataFrame(combined_features, schema=feature_names)

    # Append to original dataframe
    df = df.hstack(combined_df)

    return df


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


def create_all_feature_datasets():
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
    
    return datasets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create feature-engineered datasets")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "all"],
        default="single",
        help="Create single TF-IDF dataset (legacy) or all three datasets"
    )
    
    args = parser.parse_args()
    
    if args.mode == "all":
        # Create all three feature datasets
        create_all_feature_datasets()
    else:
        # Legacy behavior - create single TF-IDF dataset
        lf = read_from_s3("yugioh-data", "raw/2025-05/yugioh_raw_2025-05-22.parquet")

        lf = encode_categoricals(lf, ["type", "attribute", "archetype"])
        lf = normalize_numeric_columns(
            lf,
            ["atk", "def", "level"]
        )
        lf = drop_columns(lf)

        df = lf.collect()
        df = add_tfidf_description_features(df)
        df = cats_to_dummies_eager(df)

        # Write Polars DataFrame to in-memory Parquet buffer
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        # Define key and upload
        month_str = datetime.today().strftime('%Y-%m')
        s3_key = f"processed/feature_engineered/{month_str}/feature_engineered.parquet"

        upload_to_s3(
            bucket="yugioh-data",
            key=s3_key,
            data=buffer.getvalue(),
            content_type="application/octet-stream"
        )