from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_cluster_sizes(labels: np.ndarray, output_path: Path, figsize: tuple[int, int] = (8, 6), dpi: int = 200) -> None:
    """
    Plot bar chart of cluster sizes excluding noise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(labels[labels != -1])
    labels_sorted, sizes = zip(*sorted(counts.items(), key=lambda x: x[1], reverse=True)) if counts else ([], [])
    plt.figure(figsize=figsize, dpi=dpi)
    sns.barplot(x=list(labels_sorted), y=list(sizes), palette="crest")
    plt.xlabel("Cluster")
    plt.ylabel("Size")
    plt.title("Cluster Sizes")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved cluster size bar plot to %s", output_path)
