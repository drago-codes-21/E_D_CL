from __future__ import annotations

import argparse
import json
import logging
import logging.config
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from clustering.dimensionality import reduce_embeddings
from clustering.hdbscan_cluster import cluster_embeddings
from clustering.keyword_extractor import extract_cluster_keywords
from embeddings.embedder import SentenceEmbedder
from evaluation.metrics import evaluate_clustering
from evaluation.reports import save_cluster_names, save_cluster_summary, save_metrics, save_sample_emails
from evaluation.stability import run_parameter_sweep
from labeling.summarizer import summarize_clusters
from mapping.mailbox_mapper import build_mailbox_cluster_mapping, save_mailbox_mapping
from pipeline.settings import PipelineSettings, load_settings
from preprocessing.loader import load_emails, save_processed
from preprocessing.normalizer import combine_and_normalize
from reporting.cluster_summary import build_cluster_summary, collect_sample_emails
from utils.reproducibility import set_global_seed
from visualization.bar_charts import plot_cluster_sizes
from visualization.dendrogram import plot_hdbscan_tree
from visualization.scatter_plot import plot_umap_scatter
from visualization.wordclouds import generate_wordclouds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Email clustering pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    return parser.parse_args()


def setup_logging(logging_conf: Path) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.config.fileConfig(logging_conf, disable_existing_loggers=False)


def _handle_empty_dataset(settings: PipelineSettings, df: pd.DataFrame) -> None:
    results_csv = settings.paths.results_csv
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    save_cluster_names({}, settings.paths.cluster_names_json)
    save_metrics(
        {"silhouette": float("nan"), "davies_bouldin": float("nan"), "noise_ratio": float("nan")},
        settings.paths.plots_dir / "metrics.json",
    )
    save_cluster_summary([], settings.paths.cluster_summary_json)
    save_sample_emails({}, settings.paths.sample_emails_json)
    save_mailbox_mapping(pd.DataFrame(), settings.paths.mailbox_mapping_csv)


def enrich_results(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: Sequence[int],
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["embedding"] = embeddings.tolist()
    enriched["cluster"] = labels
    if probabilities is not None:
        enriched["cluster_probability"] = probabilities
    return enriched


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.config))
    setup_logging(settings.logging.config_file)
    set_global_seed(settings.reproducibility.seed)
    logger = logging.getLogger("email_clustering")

    df_raw = load_emails(
        settings.paths.input_csv,
        settings.ingestion,
        settings.preprocessing.text_column_subject,
        settings.preprocessing.text_column_body,
    )
    df_norm = combine_and_normalize(
        df_raw,
        settings.preprocessing,
        settings.preprocessing.extra_boilerplate_phrases,
        settings.ingestion.attachment_delimiter,
    )
    save_processed(df_norm, settings.paths.cleaned_csv)

    texts = df_norm["clean_text"].tolist()
    if not texts:
        logger.warning("No rows remaining after preprocessing; exiting.")
        _handle_empty_dataset(settings, df_norm)
        return

    embedder = SentenceEmbedder(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
        batch_size=settings.embeddings.batch_size,
        normalize=settings.embeddings.normalize,
        model_path=settings.embeddings.model_path,
    )
    embeddings = embedder.generate_embeddings(texts, settings.paths.embeddings_npy, use_memmap=settings.embeddings.use_memmap)

    reduced, _ = reduce_embeddings(
        embeddings,
        n_neighbors=settings.dimensionality.n_neighbors,
        min_dist=settings.dimensionality.min_dist,
        n_components=settings.dimensionality.n_components,
        metric=settings.dimensionality.metric,
        random_state=settings.reproducibility.seed,
    )
    labels, clusterer = cluster_embeddings(
        reduced,
        min_cluster_size=settings.clustering.min_cluster_size,
        min_samples=settings.clustering.min_samples,
        cluster_selection_epsilon=settings.clustering.cluster_selection_epsilon,
        cluster_selection_method=settings.clustering.cluster_selection_method,
    )

    cluster_keywords = extract_cluster_keywords(
        texts,
        labels,
        max_features=settings.keywords.max_features,
        ngram_range=tuple(settings.keywords.ngram_range),
        top_k=settings.keywords.top_k,
    )
    save_cluster_names(cluster_keywords, settings.paths.cluster_names_json)

    probabilities = getattr(clusterer, "probabilities_", None)
    persistence = getattr(clusterer, "cluster_persistence_", None)
    metrics = evaluate_clustering(reduced, labels, probabilities=probabilities, cluster_persistence=persistence)
    save_metrics(metrics, settings.paths.plots_dir / "metrics.json")

    results_df = enrich_results(df_norm, embeddings, labels, probabilities)
    settings.paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(settings.paths.results_csv, index=False)

    cluster_summary = build_cluster_summary(
        df_norm,
        labels,
        cluster_keywords,
        metrics,
        settings.preprocessing.mailbox_column,
    )
    save_cluster_summary(cluster_summary, settings.paths.cluster_summary_json)

    sample_emails = collect_sample_emails(
        df_norm,
        labels,
        settings.preprocessing.text_column_subject,
        settings.preprocessing.text_column_body,
        settings.preprocessing.mailbox_column,
        settings.preprocessing.sender_column,
        settings.preprocessing.timestamp_column,
        settings.labeling.sample_emails_per_cluster,
    )
    save_sample_emails(sample_emails, settings.paths.sample_emails_json)

    if settings.labeling.llm_summary.enabled:
        subject_samples = {cluster_id: [item["subject"] for item in items] for cluster_id, items in sample_emails.items()}
        llm_summaries = summarize_clusters(
            cluster_keywords,
            subject_samples,
            settings.labeling.llm_summary.prompt_template,
        )
        save_cluster_summary(
            [{**entry, "llm_summary": llm_summaries.get(entry["cluster"])} for entry in cluster_summary],
            settings.paths.cluster_summary_json,
        )

    mailbox_mapping = build_mailbox_cluster_mapping(df_norm, labels, settings.preprocessing.mailbox_column)
    save_mailbox_mapping(mailbox_mapping, settings.paths.mailbox_mapping_csv)

    sweep_df = run_parameter_sweep(
        embeddings,
        labels,
        settings.evaluation.stability,
        settings.clustering,
        settings.dimensionality.metric,
        settings.dimensionality.n_components,
    )
    if not sweep_df.empty:
        settings.paths.parameter_sweep_csv.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(settings.paths.parameter_sweep_csv, index=False)

    plots_dir = settings.paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_umap_scatter(
        reduced,
        labels,
        plots_dir / "umap_scatter.png",
        figsize=tuple(settings.visualization.figsize),
        dpi=settings.visualization.dpi,
        random_state=settings.visualization.random_state,
    )
    plot_cluster_sizes(labels, plots_dir / "cluster_sizes.png", figsize=(8, 6), dpi=settings.visualization.dpi)
    # generate_wordclouds(cluster_keywords, plots_dir)
    plot_hdbscan_tree(clusterer, plots_dir / "hdbscan_tree.png", figsize=tuple(settings.visualization.figsize), dpi=settings.visualization.dpi)

    logger.info("Pipeline complete. Metrics: %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
