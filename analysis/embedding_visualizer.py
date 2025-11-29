from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone embedding visualizer (2D/3D) for clustering outputs.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to pipeline config.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Max rows to plot (random sample).")
    parser.add_argument("--output-dir", type=str, default="output/plots/insights", help="Directory for png exports.")
    parser.add_argument("--random-state", type=int, default=42, help="Sampling/projection seed.")
    return parser.parse_args()


def load_embeddings(path: Path, row_count: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    try:
        emb = np.load(path, mmap_mode=None, allow_pickle=False)
        return np.asarray(emb)
    except Exception:
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        total_items = path.stat().st_size // itemsize
        if row_count == 0 or total_items % row_count != 0:
            raise ValueError("Unable to infer embedding dimension from memmap file.")
        dim = total_items // row_count
        memmap = np.memmap(path, dtype=dtype, mode="r", shape=(row_count, dim))
        return np.array(memmap)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    return pd.read_csv(path, usecols=["cluster"])


def sample_rows(embeddings: np.ndarray, clusters: np.ndarray, sample_size: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    total = len(clusters)
    if sample_size >= total:
        return embeddings, clusters
    rng = np.random.default_rng(random_state)
    idx = rng.choice(total, size=sample_size, replace=False)
    return embeddings[idx], clusters[idx]


def reduce_pca(data: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(data)


def _color_map(num_clusters: int) -> plt.cm.ScalarMappable:
    if num_clusters <= 10:
        cmap = plt.get_cmap("tab10")
    elif num_clusters <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max(num_clusters - 1, 1))
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_2d(coords: np.ndarray, clusters: np.ndarray, output_path: Path) -> None:
    unique = np.unique(clusters)
    mapper = _color_map(len(unique))

    plt.figure(figsize=(10, 8))
    if -1 in unique:
        noise_mask = clusters == -1
        plt.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            s=12,
            alpha=0.4,
            color="#7f7f7f",
            label="Noise (-1)",
        )
    for cluster in unique:
        if cluster == -1:
            continue
        mask = clusters == cluster
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.6,
            color=mapper.to_rgba(cluster),
            label=f"Cluster {cluster}",
        )
    plt.title("Embedding PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if len(unique) <= 15:
        plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_3d(coords: np.ndarray, clusters: np.ndarray, output_path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    unique = np.unique(clusters)
    mapper = _color_map(len(unique))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if -1 in unique:
        noise_mask = clusters == -1
        ax.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            coords[noise_mask, 2],
            s=12,
            alpha=0.3,
            color="#7f7f7f",
            label="Noise (-1)",
        )
    for cluster in unique:
        if cluster == -1:
            continue
        mask = clusters == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            s=12,
            alpha=0.5,
            color=mapper.to_rgba(cluster),
            label=f"Cluster {cluster}",
        )
    ax.set_title("Embedding PCA (3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if len(unique) <= 10:
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.config))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("embedding_viz")

    results = load_results(settings.paths.results_csv)
    embeddings = load_embeddings(settings.paths.embeddings_npy, len(results))
    if len(embeddings) != len(results):
        raise ValueError("Embeddings and results row counts do not match.")

    clusters = results["cluster"].fillna(-1).to_numpy()
    embeddings_sampled, clusters_sampled = sample_rows(embeddings, clusters, args.sample_size, args.random_state)
    logger.info("Sampled %d/%d rows for visualization", len(clusters_sampled), len(clusters))

    coords_2d = reduce_pca(embeddings_sampled, 2, args.random_state)
    coords_3d = reduce_pca(embeddings_sampled, 3, args.random_state)

    out_dir = Path(args.output_dir)
    plot_2d(coords_2d, clusters_sampled, out_dir / "embedding_pca_2d.png")
    logger.info("Saved 2D PCA scatter to %s", out_dir / "embedding_pca_2d.png")
    plot_3d(coords_3d, clusters_sampled, out_dir / "embedding_pca_3d.png")
    logger.info("Saved 3D PCA scatter to %s", out_dir / "embedding_pca_3d.png")


if __name__ == "__main__":
    main()
