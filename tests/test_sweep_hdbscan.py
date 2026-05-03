from __future__ import annotations

import numpy as np

from yugioh_deck_generator.clustering.sweep_hdbscan import compute_cluster_metrics, is_good_run


def test_compute_cluster_metrics_basic() -> None:
    labels = np.asarray([0, 0, 1, 1, 1, -1, -1], dtype=int)
    metrics = compute_cluster_metrics(labels)
    assert int(metrics["num_clusters"]) == 2
    assert int(metrics["num_noise_cards"]) == 2
    assert float(metrics["noise_ratio"]) > 0.0
    assert float(metrics["largest_cluster_ratio"]) > 0.0
    assert "top_10_cluster_sizes_json" in metrics


def test_is_good_run_logic() -> None:
    good_metrics = {
        "num_clusters": 12,
        "noise_ratio": 0.3,
        "largest_cluster_ratio": 0.2,
    }
    bad_metrics = {
        "num_clusters": 5,
        "noise_ratio": 0.7,
        "largest_cluster_ratio": 0.5,
    }
    assert is_good_run(good_metrics, 0.5, 10, 0.4) is True
    assert is_good_run(bad_metrics, 0.5, 10, 0.4) is False
