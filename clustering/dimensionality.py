from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import umap

logger = logging.getLogger(__name__)


def reduce_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int | None = 42,
) -> Tuple[np.ndarray, umap.UMAP]:
    """
    Reduce embeddings with UMAP.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    logger.info("Running UMAP with n_neighbors=%d, min_dist=%.2f", n_neighbors, min_dist)
    reduced = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction complete: shape %s", reduced.shape)
    return reduced, reducer
