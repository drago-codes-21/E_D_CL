from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_umap_scatter(
    reduced: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 200,
    title: str = "UMAP Scatter by Cluster",
    random_state: Optional[int] = 42,
) -> None:
    """
    Plot UMAP-reduced embeddings colored by cluster label.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    palette = sns.color_palette("hls", len(set(labels)) + 1)
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=labels,
        palette=palette,
        s=40,
        linewidth=0,
        legend="full",
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved UMAP scatter plot to %s", output_path)
