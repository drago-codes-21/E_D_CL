from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    logger.info("Saved %s", path)


def save_cluster_names(cluster_keywords: Dict[int, List[str]], path: Path) -> None:
    _write_json(cluster_keywords, path)


def save_metrics(metrics: Dict[str, object], path: Path) -> None:
    _write_json(metrics, path)


def save_cluster_summary(summary: Sequence[Dict[str, object]], path: Path) -> None:
    _write_json(list(summary), path)


def save_sample_emails(samples: Dict[int, List[Dict[str, str]]], path: Path) -> None:
    _write_json(samples, path)
