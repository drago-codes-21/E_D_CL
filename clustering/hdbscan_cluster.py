from __future__ import annotations

import logging
from typing import Tuple

import hdbscan
import numpy as np

logger = logging.getLogger(__name__)


def cluster_embeddings(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Run HDBSCAN clustering on reduced embeddings.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    logger.info(
        "Running HDBSCAN with min_cluster_size=%d, min_samples=%s",
        min_cluster_size,
        str(min_samples),
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    noise_ratio = float((labels == -1).mean())
    logger.info("HDBSCAN produced %d clusters, noise ratio %.2f", len(set(labels)) - (1 if -1 in labels else 0), noise_ratio)
    return labels, clusterer
