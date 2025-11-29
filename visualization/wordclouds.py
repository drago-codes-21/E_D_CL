from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from wordcloud import WordCloud

logger = logging.getLogger(__name__)


def generate_wordclouds(cluster_keywords: Dict[int, List[str]], output_dir: Path, dpi: int = 200) -> None:
    """
    Generate word cloud PNGs for each cluster based on its keywords.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, keywords in cluster_keywords.items():
        if not keywords:
            continue
        text = " ".join(keywords)
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(8, 4), dpi=dpi)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Cluster {cluster_id} Keywords")
        out_path = output_dir / f"wordcloud_cluster_{cluster_id}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved wordcloud for cluster %d to %s", cluster_id, out_path)
