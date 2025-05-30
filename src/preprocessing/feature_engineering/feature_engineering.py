from typing import List, Union
import tempfile
import atexit
import os

import boto3
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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
    cat_cols = [col for col in df.columns if df[col].dtype == pl.Categorical]

    if not cat_cols:
        return df

    # One-hot encode and combine with non-categorical columns
    df_non_cat = df.drop(cat_cols)
    df_dummies = df.select(cat_cols).to_dummies()

    return df_non_cat.hstack(df_dummies)

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

if __name__ == "__main__":
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
    print(df.head(5))
    print(df.columns)
