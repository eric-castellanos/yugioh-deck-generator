from typing import Optional
from datetime import datetime
import io

import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

from src.utils.s3_utils import read_parquet_from_s3, upload_to_s3

def load_card_data(card_data_path : str) -> pl.DataFrame:
    """
    Load Yu-Gi-Oh! card metadata (id, name, description, effect) from a data source.
    Args:
        card_data_path (s3 path or local path): Path to raw card data.
    Returns:
        pl.DataFrame: DataFrame containing card metadata.
    """
    embed_cols = ['id', 'name', 'desc']

    if "s3://" in card_data_path:
        bucket, key = card_data_path.replace("s3://", "").split("/", 1)
        card_df = read_parquet_from_s3(bucket=bucket, key=key, columns=embed_cols)

    else:
        card_df = pl.read_parquet(card_data_path, columns=embed_cols)

    return card_df


def compute_embeddings(df: pl.DataFrame, model_name : str = "all-MiniLM-L6-v2", n_components : int = 50) -> pl.DataFrame:
    """
    Compute embeddings for each card using a pretrained model.
    Args:
        df (pl.DataFrame): DataFrame with card metadata.
        model_name (str): Name of pretrained model to compute embeddings for card descriptions/effects.
        n_components: Number of dimensions after optional dimensionality reduction.
    Returns:
        pl.DataFrame: DataFrame with card embeddings added.
    """
    descriptions = df.select(pl.col('desc').fill_null("")).to_series().to_list()

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

    # Concatenate embeddings with original df
    df_out = df.with_columns(embed_df)

    return df_out


def save_embeddings_local(df: pl.DataFrame, path: str) -> None:
    """
    Save card embeddings DataFrame to a local Parquet file.
    Args:
        df (pl.DataFrame): DataFrame with card embeddings.
        path (str): Local file path to save Parquet file.
    """
    year, month = datetime.now().year, datetime.now().month
    df.write_parquet(f"{path}/card_embeddings_{year}_{month}.parquet")

def save_embeddings_s3(df: pl.DataFrame, bucket: str = "yugioh-data", key: str = None) -> None:
    """
    Save card embeddings DataFrame to an S3 bucket as a Parquet file.
    Args:
        df (pl.DataFrame): DataFrame with card embeddings.
        bucket (str): S3 bucket name.
        key (str): S3 object key (path).
    """
    year, month = datetime.now().year, datetime.now().month
    if key is None:
        key = f"processed/card_embeddings/card_embeddings_{year}_{month}.parquet"
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    upload_to_s3(bucket=bucket, key=key, data=buf.getvalue())


def main(card_data_path : str) -> None:
    """
    Main function to orchestrate card embedding computation and saving.
    """
    card_df = load_card_data(card_data_path)
    card_with_embed_df = compute_embeddings(card_df)
    save_embeddings_local(card_with_embed_df, path="data/card_embeddings")
    save_embeddings_s3(card_with_embed_df)


if __name__ == "__main__":
    card_data_path = "s3://yugioh-data/raw/2025-06/yugioh_raw_2025-06-01.parquet"
    main(card_data_path)