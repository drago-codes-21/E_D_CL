from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from statistics import median
from typing import Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.settings import PipelineSettings, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clustering outputs and recommend tuning actions.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the pipeline config.")
    parser.add_argument("--log-file", type=str, default="logs/app.log", help="Path to the pipeline log file.")
    return parser.parse_args()


def load_latest_metrics(log_path: Path) -> Dict[str, float]:
    if not log_path.exists():
        return {}
    lines = log_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if "Pipeline complete. Metrics:" in line:
            _, metrics_str = line.split("Metrics:", maxsplit=1)
            metrics_str = metrics_str.strip()
            try:
                return json.loads(metrics_str)
            except json.JSONDecodeError:
                continue
    return {}


def analyze_results(results_path: Path) -> Dict[str, object]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    cluster_counts: Counter[int] = Counter()
    mailbox_clusters: defaultdict[str, set[int]] = defaultdict(set)
    total_rows = 0

    for chunk in pd.read_csv(results_path, usecols=["cluster", "mailbox"], chunksize=50000):
        total_rows += len(chunk)
        clusters = chunk["cluster"].fillna(-1).astype(int)
        cluster_counts.update(clusters.tolist())

        for mailbox, cluster in zip(chunk["mailbox"], clusters):
            if cluster >= 0 and isinstance(mailbox, str) and mailbox:
                mailbox_clusters[mailbox].add(int(cluster))

    noise = cluster_counts.pop(-1, 0)
    cluster_items = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
    sizes = [count for _, count in cluster_items]
    cluster_count = len(cluster_items)
    avg_size = (sum(sizes) / cluster_count) if cluster_count else 0.0
    med_size = median(sizes) if cluster_count else 0.0

    mailbox_spread = sorted(
        ((mailbox, len(clusters)) for mailbox, clusters in mailbox_clusters.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "total_rows": total_rows,
        "cluster_count": cluster_count,
        "noise": noise,
        "noise_ratio": noise / total_rows if total_rows else 0.0,
        "avg_cluster_size": avg_size,
        "median_cluster_size": med_size,
        "top_clusters": cluster_items[:10],
        "tail_clusters": cluster_items[-5:],
        "mailbox_spread": mailbox_spread,
        "mailboxes_multi_cluster": sum(1 for _, span in mailbox_spread if span > 1),
    }


def read_cluster_summary(summary_path: Path, top_n: int = 5) -> List[Dict[str, object]]:
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data = sorted(data, key=lambda item: item.get("size", 0), reverse=True)
    return data[:top_n]


def analyze_parameter_sweep(sweep_path: Path, top_n: int = 3) -> List[Dict[str, object]]:
    if not sweep_path.exists() or sweep_path.stat().st_size == 0:
        return []
    df = pd.read_csv(sweep_path)
    if df.empty:
        return []
    best = df.sort_values("silhouette", ascending=False).head(top_n)
    return best.to_dict(orient="records")


def generate_recommendations(
    stats: Dict[str, object],
    settings: PipelineSettings,
    metrics: Dict[str, float],
) -> List[str]:
    recs: List[str] = []
    cluster_count = stats.get("cluster_count", 0)
    total_rows = stats.get("total_rows", 0)
    avg_size = stats.get("avg_cluster_size", 0) or 0
    noise_ratio = metrics.get("noise_ratio", stats.get("noise_ratio", 0))

    target_range = list(settings.rationalization.target_cluster_range or [])
    target_low, target_high = (target_range + [0, 0])[:2]
    current_min = settings.clustering.min_cluster_size
    current_samples = settings.clustering.min_samples or current_min

    if target_high and cluster_count > target_high:
        desired_clusters = (target_low + target_high) / 2 if target_low else target_high
        desired_avg = total_rows / desired_clusters if desired_clusters else avg_size
        suggested_min = max(current_min + 25, int(desired_avg * 0.8))
        suggested_samples = max(current_samples, int(suggested_min * 0.5))
        recs.append(
            f"{cluster_count} clusters exceeds target range ({target_low}-{target_high}). "
            f"Consider raising HDBSCAN min_cluster_size from {current_min} to ~{suggested_min} "
            f"and min_samples to ~{suggested_samples}."
        )
    elif target_low and 0 < cluster_count < target_low:
        desired_avg = total_rows / target_low if target_low else avg_size
        suggested_min = max(5, int(desired_avg * 0.7))
        recs.append(
            f"{cluster_count} clusters is below the desired lower bound ({target_low}). "
            f"Lower HDBSCAN min_cluster_size to ~{suggested_min} or decrease min_samples "
            f"to encourage finer segmentation."
        )

    if noise_ratio and noise_ratio > 0.1:
        recs.append(
            f"Noise ratio {noise_ratio:.2%} is high. "
            "Tighten preprocessing or increase UMAP n_neighbors/min_dist to smooth the manifold."
        )

    mailbox_spread: List[Tuple[str, int]] = stats.get("mailbox_spread", [])
    if mailbox_spread:
        heavy_mailboxes = [mb for mb, span in mailbox_spread if span > (target_high or 10)]
        if heavy_mailboxes:
            recs.append(
                f"{len(heavy_mailboxes)} mailboxes appear in more than {target_high or 10} clusters. "
                "Review their routing rules or force-assign them to strategic clusters."
            )

    if not recs:
        recs.append("Cluster counts and noise are within the desired envelope. Continue monitoring.")
    return recs


def pretty_print_summary(
    stats: Dict[str, object],
    metrics: Dict[str, float],
    summaries: List[Dict[str, object]],
    sweep: List[Dict[str, object]],
    recs: List[str],
) -> None:
    print("=== Pipeline Diagnostics ===")
    print(f"Total emails processed: {stats.get('total_rows', 0):,}")
    print(f"Clusters discovered: {stats.get('cluster_count', 0):,} (noise: {stats.get('noise', 0):,})")
    print(f"Avg / median cluster size: {stats.get('avg_cluster_size', 0):.1f} / {stats.get('median_cluster_size', 0):.1f}")
    if metrics:
        print(
            f"Silhouette: {metrics.get('silhouette', float('nan')):.3f}, "
            f"Davies-Bouldin: {metrics.get('davies_bouldin', float('nan')):.3f}, "
            f"Noise ratio: {metrics.get('noise_ratio', stats.get('noise_ratio', 0)):.2%}"
        )

    print("\nTop clusters:")
    for cluster_id, size in stats.get("top_clusters", []):
        print(f"  Cluster {cluster_id}: {size:,} emails")

    if summaries:
        print("\nRepresentative cluster summaries:")
        for item in summaries:
            keywords = ", ".join(item.get("keywords", [])[:6])
            print(f"  Cluster {item.get('cluster')}: size {item.get('size', 0):,} | keywords: {keywords}")

    if sweep:
        print("\nBest parameter sweep combinations:")
        for row in sweep:
            print(
                f"  UMAP(n_neighbors={row['umap_n_neighbors']}, min_dist={row['umap_min_dist']}) + "
                f"HDBSCAN(min_cluster_size={row['hdbscan_min_cluster_size']}, min_samples={row['hdbscan_min_samples']}) "
                f"=> silhouette {row['silhouette']:.3f}, noise {row['noise_ratio']:.2f}"
            )

    print("\nRecommendations:")
    for rec in recs:
        print(f" - {rec}")


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.config))

    metrics = load_latest_metrics(Path(args.log_file))
    stats = analyze_results(settings.paths.results_csv)
    summaries = read_cluster_summary(settings.paths.cluster_summary_json)
    sweep = analyze_parameter_sweep(settings.paths.parameter_sweep_csv)
    recs = generate_recommendations(stats, settings, metrics)

    pretty_print_summary(stats, metrics, summaries, sweep, recs)


if __name__ == "__main__":
    main()
