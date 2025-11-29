from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
    cluster_persistence: np.ndarray | None = None,
) -> Dict[str, float | Dict[str, float]]:
    """
    Compute clustering quality metrics while handling noisy points gracefully.
    """
    results: Dict[str, float | Dict[str, float]] = {}
    mask = labels != -1
    clustered_labels = labels[mask]
    clustered_embeddings = embeddings[mask]

    if len(set(clustered_labels)) > 1:
        results["silhouette"] = float(silhouette_score(clustered_embeddings, clustered_labels))
        results["davies_bouldin"] = float(davies_bouldin_score(clustered_embeddings, clustered_labels))
    else:
        results["silhouette"] = float("nan")
        results["davies_bouldin"] = float("nan")

    results["noise_ratio"] = float((labels == -1).mean())

    density: Dict[str, float] = {}
    if probabilities is not None and probabilities.shape[0] == labels.shape[0]:
        for label in set(labels):
            if label == -1:
                continue
            mask_label = labels == label
            density[str(label)] = float(probabilities[mask_label].mean())
    results["cluster_density"] = density
    results["avg_cluster_density"] = float(np.mean(list(density.values()))) if density else float("nan")

    if cluster_persistence is not None and cluster_persistence.size:
        persistence_mapping = {str(idx): float(value) for idx, value in enumerate(cluster_persistence)}
        results["cluster_persistence"] = persistence_mapping

    logger.info("Evaluation metrics: %s", results)
    return results
