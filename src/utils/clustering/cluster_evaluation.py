import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas

# Get mean vector for each cluster (excluding noise if using HDBSCAN)
for cluster_id in np.unique(labels):
    if cluster_id == -1:  # Skip noise
        continue
    cluster_vectors = X[labels == cluster_id]
    mean_vector = cluster_vectors.mean(axis=0)

    # Top TF-IDF words or embedding dimensions
    top_features = np.argsort(mean_vector)[::-1][:10]
    print(f"Cluster {cluster_id} Top Features:", top_features)

centroid = X[labels == cluster_id].mean(axis=0).reshape(1, -1)
sims = cosine_similarity(X[labels == cluster_id], centroid).flatten()
top_idxs = sims.argsort()[::-1][:5]  # Top 5 cards closest to centroid