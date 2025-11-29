from __future__ import annotations

import itertools
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from clustering.dimensionality import reduce_embeddings
from clustering.hdbscan_cluster import cluster_embeddings
from evaluation.metrics import evaluate_clustering
from pipeline.settings import ClusteringConfig, StabilityConfig

logger = logging.getLogger(__name__)


def run_parameter_sweep(
    embeddings: np.ndarray,
    base_labels: Sequence[int],
    stability_cfg: StabilityConfig,
    clustering_cfg: ClusteringConfig,
    umap_metric: str,
    n_components: int,
) -> pd.DataFrame:
    """
    Evaluate stability across UMAP/HDBSCAN parameter combinations.
    """
    if not stability_cfg.enabled:
        return pd.DataFrame()

    if not (stability_cfg.umap_n_neighbors and stability_cfg.umap_min_dist and stability_cfg.hdbscan_min_cluster_size):
        logger.warning("Stability sweep enabled but parameter grids are empty; skipping.")
        return pd.DataFrame()

    sample_size = min(stability_cfg.sample_size, len(embeddings))
    rng = np.random.default_rng(stability_cfg.random_state)
    sample_idx = rng.choice(len(embeddings), size=sample_size, replace=False)
    sample_embeddings = embeddings[sample_idx]
    base_sample_labels = np.array(base_labels)[sample_idx]

    min_samples_grid = (
        stability_cfg.hdbscan_min_samples
        if stability_cfg.hdbscan_min_samples
        else ([clustering_cfg.min_samples] if clustering_cfg.min_samples is not None else [None])
    )

    combos = list(
        itertools.product(
            stability_cfg.umap_n_neighbors,
            stability_cfg.umap_min_dist,
            stability_cfg.hdbscan_min_cluster_size,
            min_samples_grid,
        ),
    )

    records = []
    for n_neighbors, min_dist, min_cluster_size, min_samples in combos:
        reduced, _ = reduce_embeddings(
            sample_embeddings,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=umap_metric,
            random_state=stability_cfg.random_state,
        )
        labels, clusterer = cluster_embeddings(
            reduced,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            cluster_selection_method=clustering_cfg.cluster_selection_method,
        )
        metrics = evaluate_clustering(reduced, labels, probabilities=getattr(clusterer, "probabilities_", None))
        ari = (
            float(adjusted_rand_score(base_sample_labels, labels))
            if len(set(labels)) > 1 and len(set(base_sample_labels)) > 1
            else float("nan")
        )
        records.append(
            {
                "umap_n_neighbors": n_neighbors,
                "umap_min_dist": min_dist,
                "hdbscan_min_cluster_size": min_cluster_size,
                "hdbscan_min_samples": min_samples,
                "silhouette": metrics.get("silhouette"),
                "noise_ratio": metrics.get("noise_ratio"),
                "avg_cluster_density": metrics.get("avg_cluster_density"),
                "ari_vs_base": ari,
            },
        )
    return pd.DataFrame(records)
