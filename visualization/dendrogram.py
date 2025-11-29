from __future__ import annotations

import logging
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_hdbscan_tree(clusterer: hdbscan.HDBSCAN, output_path: Path, figsize: tuple[int, int] = (10, 8), dpi: int = 200) -> None:
    """
    Plot the HDBSCAN condensed tree for visual inspection.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(clusterer, "condensed_tree_"):
        logger.warning("Clusterer has no condensed_tree_; skipping dendrogram plot.")
        return
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette="deep", axis=ax)
    ax.set_title("HDBSCAN Condensed Tree")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved HDBSCAN condensed tree to %s", output_path)
