import io
import logging

import boto3
import polars as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

from src.utils.s3_utils import read_parquet_from_s3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parameters
S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/2025-05/feature_engineered.parquet"
N_CLUSTERS = 10
PCA_COMPONENTS = 50
RANDOM_SEED = 42


def main():
    logging.info("Loading feature-engineered dataset from S3")
    df = read_parquet_from_s3(S3_BUCKET, S3_KEY)


    meta_cols = ["id", "name", "archetype"]
    meta_df = df.select(meta_cols)
    feature_df = df.drop(meta_cols)

    logging.info(f"Data loaded. Original feature shape: {feature_df.shape}")

    # Drop non-numeric columns (e.g., strings like 'desc')
    feature_df = feature_df.select([
        col for col in feature_df.columns
        if feature_df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ])

    X = feature_df.to_numpy()

    logging.info(f"Applying PCA to reduce to {PCA_COMPONENTS} components")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X)
    logging.info(f"PCA complete. Reduced feature shape: {X_reduced.shape}")

    logging.info(f"Running KMeans with {N_CLUSTERS} clusters")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(X_reduced)

    meta_df = meta_df.with_columns(pl.Series(name="cluster", values=labels))

    if "archetype" in meta_df.columns:
        score = adjusted_rand_score(meta_df["archetype"].to_list(), labels)
        logging.info(f"Adjusted Rand Index (vs. archetype): {score:.4f}")

    silhouette = silhouette_score(X_reduced, labels)
    logging.info(f"Silhouette Score: {silhouette:.4f}")

    # visualization
    logging.info("Plotting 2D PCA projection of the reduced space")
    X_vis = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_reduced)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, label="Cluster")
    plt.title("KMeans Clusters (2D PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logging.info("Example cards per cluster:")
    for cluster_id in range(N_CLUSTERS):
        logging.info(f"Cluster {cluster_id}:")
        cluster_cards = meta_df.filter(pl.col("cluster") == cluster_id).head(5)
        for row in cluster_cards.iter_rows():
            logging.info(f" - {row[1]} (Archetype: {row[2]})")


if __name__ == "__main__":
    main()