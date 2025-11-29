from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.settings import PipelineSettings, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rough training-time estimator for the clustering pipeline.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file.")
    parser.add_argument("--rows", type=int, default=None, help="Override dataset row count (skip CSV scan).")
    parser.add_argument("--input-csv", type=str, default=None, help="Optional CSV path override.")
    return parser.parse_args()


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    count = -1  # subtract header
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for count, _ in enumerate(f, start=0):
            pass
    return max(0, count)


def estimate_embeddings(rows: int, batch_size: int) -> Tuple[float, float]:
    # baseline: ~45 emails/s at batch 32 on CPU. scale linearly with batch but cap.
    base_throughput = 45.0
    throughput = base_throughput * (batch_size / 32)
    throughput = max(10.0, min(throughput, 220.0))
    seconds = rows / throughput
    return seconds, throughput


def estimate_umap(rows: int, neighbors: int, components: int) -> float:
    # heuristically O(rows * neighbors * components) with small constant
    coeff = 1.8e-5  # seconds per (row*neighbor) for 2 components
    seconds = coeff * rows * neighbors * max(components / 2, 1)
    return seconds


def estimate_hdbscan(rows: int, min_cluster_size: int, min_samples: int) -> float:
    # cost grows super-linearly; approximate with rows * log(rows)
    coeff = 8e-5  # tuned for CPU baseline
    complexity = rows * max(math.log2(rows + 1), 1)
    penalty = math.sqrt(max(min_cluster_size, 1) / 50) * (min_samples / max(min_cluster_size, 1))
    return coeff * complexity * max(penalty, 0.6)


def estimate_keywords(rows: int, max_features: int) -> float:
    coeff = 2.5e-5
    return coeff * rows * math.log(max_features + 1, 2)


def summarize(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def analyze(settings: PipelineSettings, rows: int) -> Dict[str, float]:
    estimates: Dict[str, float] = {}
    emb_seconds, throughput = estimate_embeddings(rows, settings.embeddings.batch_size)
    estimates["embeddings"] = emb_seconds
    estimates["embeddings_throughput"] = throughput

    umap_seconds = estimate_umap(
        rows,
        settings.dimensionality.n_neighbors,
        settings.dimensionality.n_components,
    )
    estimates["umap"] = umap_seconds

    hdbscan_seconds = estimate_hdbscan(
        rows,
        settings.clustering.min_cluster_size,
        settings.clustering.min_samples or settings.clustering.min_cluster_size,
    )
    estimates["hdbscan"] = hdbscan_seconds

    keywords_seconds = estimate_keywords(
        rows,
        settings.keywords.max_features,
    )
    estimates["keywords"] = keywords_seconds

    # Parameter sweep: count combos and estimate per combo using sample_size
    sweep_cfg = settings.evaluation.stability
    combos = (
        len(sweep_cfg.umap_n_neighbors)
        * len(sweep_cfg.umap_min_dist)
        * len(sweep_cfg.hdbscan_min_cluster_size)
        * len(sweep_cfg.hdbscan_min_samples)
    )
    if sweep_cfg.enabled and combos > 0:
        sweep_rows = min(sweep_cfg.sample_size, rows)
        sweep_umap = estimate_umap(sweep_rows, sum(sweep_cfg.umap_n_neighbors) / len(sweep_cfg.umap_n_neighbors), settings.dimensionality.n_components)
        sweep_hdb = estimate_hdbscan(sweep_rows, sum(sweep_cfg.hdbscan_min_cluster_size) / len(sweep_cfg.hdbscan_min_cluster_size), sum(sweep_cfg.hdbscan_min_samples) / len(sweep_cfg.hdbscan_min_samples))
        estimates["parameter_sweep"] = combos * (sweep_umap + sweep_hdb)
        estimates["parameter_sweep_runs"] = combos
    else:
        estimates["parameter_sweep"] = 0.0
        estimates["parameter_sweep_runs"] = 0

    estimates["total"] = sum(
        estimates[name]
        for name in ["embeddings", "umap", "hdbscan", "keywords", "parameter_sweep"]
    )
    return estimates


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.config))

    rows = args.rows
    if rows is None:
        csv_path = Path(args.input_csv) if args.input_csv else settings.paths.input_csv
        rows = count_rows(csv_path)

    estimates = analyze(settings, rows)
    print("=== Estimated Pipeline Runtime ===")
    print(f"Rows: {rows:,}")
    print(f"Embeddings:  {summarize(estimates['embeddings'])}  (~{estimates['embeddings_throughput']:.0f} emails/sec)")
    print(f"UMAP:        {summarize(estimates['umap'])}")
    print(f"HDBSCAN:     {summarize(estimates['hdbscan'])}")
    print(f"Keywords:    {summarize(estimates['keywords'])}")
    if estimates["parameter_sweep_runs"] > 0:
        print(
            f"Parameter sweep ({int(estimates['parameter_sweep_runs'])} runs): "
            f"{summarize(estimates['parameter_sweep'])}"
        )
    print(f"Total (heuristic): {summarize(estimates['total'])}")
    print("\nNotes: estimates assume CPU execution; adjust manually if running on GPU or slower hardware.")


if __name__ == "__main__":
    main()
