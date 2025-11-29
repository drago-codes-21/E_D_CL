from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    input_csv: Path
    cleaned_csv: Path
    embeddings_npy: Path
    results_csv: Path
    cluster_names_json: Path
    cluster_summary_json: Path
    sample_emails_json: Path
    mailbox_mapping_csv: Path
    parameter_sweep_csv: Path
    plots_dir: Path


@dataclass
class LoggingConfig:
    config_file: Path


@dataclass
class ReproducibilityConfig:
    seed: int = 42


@dataclass
class IngestionConfig:
    required_columns: List[str]
    timestamp_column: str
    timestamp_format: Optional[str]
    timezone: str
    lookback_days: int
    sender_column: str
    mailbox_column: str
    attachments_column: str
    attachment_delimiter: str
    drop_duplicates: bool


@dataclass
class PreprocessingConfig:
    text_column_subject: str
    text_column_body: str
    sender_column: str
    timestamp_column: str
    mailbox_column: str
    attachments_column: str
    include_sender: bool
    include_timestamp: bool
    include_attachments: bool
    attachment_join_token: str
    min_chars: int
    lowercase: bool
    punctuation_spacing: bool
    extra_boilerplate_phrases: List[str]


@dataclass
class EmbeddingConfig:
    model_name: str
    batch_size: int
    device: str
    normalize: bool
    model_path: Optional[Path]
    use_memmap: bool


@dataclass
class DimensionalityConfig:
    n_neighbors: int
    min_dist: float
    n_components: int
    metric: str


@dataclass
class ClusteringConfig:
    min_cluster_size: int
    min_samples: Optional[int]
    cluster_selection_epsilon: float
    cluster_selection_method: str


@dataclass
class KeywordConfig:
    max_features: int
    ngram_range: List[int]
    top_k: int


@dataclass
class LabelingLLMConfig:
    enabled: bool
    prompt_template: str


@dataclass
class LabelingConfig:
    sample_emails_per_cluster: int
    llm_summary: LabelingLLMConfig


@dataclass
class StabilityConfig:
    enabled: bool
    sample_size: int
    random_state: int
    umap_n_neighbors: List[int]
    umap_min_dist: List[float]
    hdbscan_min_cluster_size: List[int]
    hdbscan_min_samples: List[int]


@dataclass
class EvaluationConfig:
    stability: StabilityConfig


@dataclass
class VisualizationConfig:
    random_state: int
    figsize: List[int]
    dpi: int


@dataclass
class RationalizationConfig:
    mailbox_count: int
    target_cluster_range: List[int]


@dataclass
class PipelineSettings:
    paths: PathsConfig
    logging: LoggingConfig
    reproducibility: ReproducibilityConfig
    ingestion: IngestionConfig
    preprocessing: PreprocessingConfig
    embeddings: EmbeddingConfig
    dimensionality: DimensionalityConfig
    clustering: ClusteringConfig
    keywords: KeywordConfig
    labeling: LabelingConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    rationalization: RationalizationConfig


def _to_path(value: Any) -> Path:
    return Path(value).expanduser()


def load_settings(config_path: Path) -> PipelineSettings:
    raw_cfg: Dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths = PathsConfig(
        input_csv=_to_path(raw_cfg["paths"]["input_csv"]),
        cleaned_csv=_to_path(raw_cfg["paths"]["cleaned_csv"]),
        embeddings_npy=_to_path(raw_cfg["paths"]["embeddings_npy"]),
        results_csv=_to_path(raw_cfg["paths"]["results_csv"]),
        cluster_names_json=_to_path(raw_cfg["paths"]["cluster_names_json"]),
        cluster_summary_json=_to_path(raw_cfg["paths"]["cluster_summary_json"]),
        sample_emails_json=_to_path(raw_cfg["paths"]["sample_emails_json"]),
        mailbox_mapping_csv=_to_path(raw_cfg["paths"]["mailbox_mapping_csv"]),
        parameter_sweep_csv=_to_path(raw_cfg["paths"]["parameter_sweep_csv"]),
        plots_dir=_to_path(raw_cfg["paths"]["plots_dir"]),
    )

    logging_cfg = LoggingConfig(config_file=_to_path(raw_cfg["logging"]["config_file"]))
    reproducibility = ReproducibilityConfig(seed=int(raw_cfg.get("reproducibility", {}).get("seed", 42)))

    ingestion_cfg = raw_cfg["ingestion"]
    ingestion = IngestionConfig(
        required_columns=list(ingestion_cfg.get("required_columns", [])),
        timestamp_column=ingestion_cfg["timestamp_column"],
        timestamp_format=ingestion_cfg.get("timestamp_format"),
        timezone=ingestion_cfg.get("timezone", "UTC"),
        lookback_days=int(ingestion_cfg.get("lookback_days", 180)),
        sender_column=ingestion_cfg["sender_column"],
        mailbox_column=ingestion_cfg["mailbox_column"],
        attachments_column=ingestion_cfg["attachments_column"],
        attachment_delimiter=ingestion_cfg.get("attachment_delimiter", ";"),
        drop_duplicates=bool(ingestion_cfg.get("drop_duplicates", True)),
    )

    prep_cfg = raw_cfg["preprocessing"]
    preprocessing = PreprocessingConfig(
        text_column_subject=prep_cfg["text_column_subject"],
        text_column_body=prep_cfg["text_column_body"],
        sender_column=prep_cfg["sender_column"],
        timestamp_column=prep_cfg["timestamp_column"],
        mailbox_column=prep_cfg["mailbox_column"],
        attachments_column=prep_cfg["attachments_column"],
        include_sender=bool(prep_cfg.get("include_sender", True)),
        include_timestamp=bool(prep_cfg.get("include_timestamp", True)),
        include_attachments=bool(prep_cfg.get("include_attachments", True)),
        attachment_join_token=prep_cfg.get("attachment_join_token", " "),
        min_chars=int(prep_cfg.get("min_chars", 40)),
        lowercase=bool(prep_cfg.get("lowercase", True)),
        punctuation_spacing=bool(prep_cfg.get("punctuation_spacing", True)),
        extra_boilerplate_phrases=list(prep_cfg.get("extra_boilerplate_phrases", [])),
    )

    embed_cfg = raw_cfg["embeddings"]
    embeddings = EmbeddingConfig(
        model_name=embed_cfg["model_name"],
        batch_size=int(embed_cfg.get("batch_size", 32)),
        device=embed_cfg.get("device", "auto"),
        normalize=bool(embed_cfg.get("normalize", True)),
        model_path=_to_path(embed_cfg["model_path"]) if embed_cfg.get("model_path") else None,
        use_memmap=bool(embed_cfg.get("use_memmap", False)),
    )

    dim_cfg = raw_cfg["dimensionality_reduction"]
    dimensionality = DimensionalityConfig(
        n_neighbors=int(dim_cfg["n_neighbors"]),
        min_dist=float(dim_cfg["min_dist"]),
        n_components=int(dim_cfg["n_components"]),
        metric=dim_cfg["metric"],
    )

    cluster_cfg = raw_cfg["clustering"]
    clustering = ClusteringConfig(
        min_cluster_size=int(cluster_cfg["min_cluster_size"]),
        min_samples=cluster_cfg.get("min_samples"),
        cluster_selection_epsilon=float(cluster_cfg.get("cluster_selection_epsilon", 0.0)),
        cluster_selection_method=cluster_cfg.get("cluster_selection_method", "eom"),
    )

    keyword_cfg = raw_cfg["keywords"]
    keywords = KeywordConfig(
        max_features=int(keyword_cfg["max_features"]),
        ngram_range=list(keyword_cfg["ngram_range"]),
        top_k=int(keyword_cfg["top_k"]),
    )

    labeling_cfg = raw_cfg.get("labeling", {})
    llm_cfg = labeling_cfg.get("llm_summary", {})
    labeling = LabelingConfig(
        sample_emails_per_cluster=int(labeling_cfg.get("sample_emails_per_cluster", 3)),
        llm_summary=LabelingLLMConfig(
            enabled=bool(llm_cfg.get("enabled", False)),
            prompt_template=llm_cfg.get("prompt_template", "Summarize cluster keywords: {keywords}"),
        ),
    )

    eval_cfg = raw_cfg.get("evaluation", {}).get("stability", {})
    stability = StabilityConfig(
        enabled=bool(eval_cfg.get("enabled", False)),
        sample_size=int(eval_cfg.get("sample_size", 2000)),
        random_state=int(eval_cfg.get("random_state", reproducibility.seed)),
        umap_n_neighbors=list(eval_cfg.get("umap", {}).get("n_neighbors", [])),
        umap_min_dist=list(eval_cfg.get("umap", {}).get("min_dist", [])),
        hdbscan_min_cluster_size=list(eval_cfg.get("hdbscan", {}).get("min_cluster_size", [])),
        hdbscan_min_samples=list(eval_cfg.get("hdbscan", {}).get("min_samples", [])),
    )
    evaluation = EvaluationConfig(stability=stability)

    viz_cfg = raw_cfg["visualization"]
    visualization = VisualizationConfig(
        random_state=int(viz_cfg.get("random_state", reproducibility.seed)),
        figsize=list(viz_cfg.get("figsize", [10, 8])),
        dpi=int(viz_cfg.get("dpi", 200)),
    )

    rational_cfg = raw_cfg.get("rationalization", {})
    rationalization = RationalizationConfig(
        mailbox_count=int(rational_cfg.get("mailbox_count", 0)),
        target_cluster_range=list(rational_cfg.get("target_cluster_range", [])),
    )

    return PipelineSettings(
        paths=paths,
        logging=logging_cfg,
        reproducibility=reproducibility,
        ingestion=ingestion,
        preprocessing=preprocessing,
        embeddings=embeddings,
        dimensionality=dimensionality,
        clustering=clustering,
        keywords=keywords,
        labeling=labeling,
        evaluation=evaluation,
        visualization=visualization,
        rationalization=rationalization,
    )
