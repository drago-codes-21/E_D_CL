from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def extract_cluster_keywords(
    texts: List[str],
    labels: np.ndarray,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    top_k: int = 8,
) -> Dict[int, List[str]]:
    """
    Extract top TF-IDF keywords for each cluster label.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_array = np.array(vectorizer.get_feature_names_out())

    clusters: Dict[int, List[str]] = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        if mask.sum() == 0:
            continue
        cluster_scores = tfidf_matrix[mask].mean(axis=0).A1
        top_indices = cluster_scores.argsort()[::-1][:top_k]
        keywords = feature_array[top_indices].tolist()
        clusters[int(label)] = keywords
        logger.debug("Cluster %d keywords: %s", label, keywords)

    logger.info("Extracted keywords for %d clusters", len(clusters))
    return clusters
