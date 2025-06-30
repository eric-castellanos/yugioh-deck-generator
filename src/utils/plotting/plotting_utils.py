from typing import Union

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np

from src.utils.mlflow.mlflow_utils import log_artifact

def plot_scree_curve(
    model: Union[TruncatedSVD, PCA],
    output_path: str = None,
    show: bool = True,
    title: str = "Scree Plot"
) -> np.ndarray:
    """
    Plots the cumulative explained variance ratio from a fitted PCA or TruncatedSVD model.

    Args:
        model (Union[TruncatedSVD, PCA]): A fitted dimensionality reduction model.
        output_path (str, optional): If provided, saves the plot to this path.
        show (bool): Whether to display the plot using plt.show().
        title (str): Title of the plot.

    Returns:
        np.ndarray: Cumulative explained variance values.
    """
    if not hasattr(model, "explained_variance_ratio_"):
        raise ValueError("Model must be fitted and expose explained_variance_ratio_.")

    cumulative_variance = np.cumsum(model.explained_variance_ratio_)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o'
    )
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # Uncomment this if you're using MLflow and want to log
        log_artifact(output_path, artifact_path="visualizations")
    if show:
        plt.show()

    return cumulative_variance