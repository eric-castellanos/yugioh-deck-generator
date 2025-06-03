import io
import logging
import json

import boto3
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from kneed import KneeLocator

from src.utils.s3_utils import read_parquet_from_s3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parameters
S3_BUCKET = "yugioh-data"
S3_KEY = "processed/feature_engineered/2025-05/feature_engineered.parquet"
N_CLUSTERS = 5 # (based on elbow plot) 
PCA_COMPONENTS = 2
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

    # Plot explained variance (scree plot)
    explained_var = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_var) + 1), explained_var.cumsum(), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot (PCA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("src/viz/clustering/scree_plot.png")
    plt.close()

    logging.info(f"Total explained variance by {PCA_COMPONENTS} components: {explained_var.sum():.4f}")

    logging.info("Projecting to 2D using t-SNE...")
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED).fit_transform(X_reduced)

    # t-SNE plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=10)

    # Manually create legend based on unique cluster labels
    unique_labels = np.unique(labels)
    legend_patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=f"Cluster {i}") for i in unique_labels]
    plt.legend(handles=legend_patches, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE Projection of KMeans Clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("src/viz/clustering/tsne_clusters.png")
    plt.close()
    
    inertias = []
    k_values = range(2, 20)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_reduced)  # use PCA-reduced features
        inertias.append(kmeans.inertia_)
        print(f"K: {k} -> WCSS: {kmeans.inertia_}")

    plt.figure(figsize=(8, 4))
    plt.plot(k_values, inertias, marker='o')
    plt.title("Elbow Plot: KMeans Inertia vs. Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("src/viz/clustering/elbow_plot.png")

    logging.info("Example cards per cluster:")
    for cluster_id in range(N_CLUSTERS):
        logging.info(f"Cluster {cluster_id}:")
        cluster_cards = meta_df.filter(pl.col("cluster") == cluster_id).head(5)
        for row in cluster_cards.iter_rows():
            logging.info(f" - {row[1]} (Archetype: {row[2]})")

    metrics = {
        "n_clusters": N_CLUSTERS,
        "pca_components": PCA_COMPONENTS,
        "explained_variance_ratio": explained_var.sum(),
        "silhouette_score": float(silhouette),
        "adjusted_rand_index": float(score) if "archetype" in meta_df.columns else None
    }

    with open("src/model/metrics/clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()