from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pytz

from pipeline.settings import IngestionConfig

logger = logging.getLogger(__name__)


def _ensure_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")


def _filter_last_n_days(df: pd.DataFrame, column: str, lookback_days: int) -> pd.DataFrame:
    if column not in df.columns:
        return df
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    mask = df[column] >= cutoff
    filtered = df[mask].reset_index(drop=True)
    logger.info("Filtered emails to lookback window: %d -> %d rows", len(df), len(filtered))
    return filtered


def load_emails(csv_path: Path, ingestion_cfg: IngestionConfig, subject_col: str, body_col: str) -> pd.DataFrame:
    """
    Load raw emails, validate required columns, parse timestamps, and filter by lookback window.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    _ensure_required_columns(df, ingestion_cfg.required_columns)

    timestamp_col = ingestion_cfg.timestamp_column
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        format=ingestion_cfg.timestamp_format,
        utc=True,
    )
    df = df.dropna(subset=[timestamp_col]).reset_index(drop=True)
    df = _filter_last_n_days(df, timestamp_col, ingestion_cfg.lookback_days)

    if ingestion_cfg.drop_duplicates:
        df = df.drop_duplicates(subset=[subject_col, body_col, timestamp_col])

    fill_columns: Iterable[str] = {subject_col, body_col, ingestion_cfg.attachments_column, ingestion_cfg.sender_column, ingestion_cfg.mailbox_column}
    for col in fill_columns:
        if col in df.columns:
            df[col] = df[col].fillna("")

    local_tz = pytz.timezone(ingestion_cfg.timezone)
    df[timestamp_col] = df[timestamp_col].dt.tz_convert(local_tz)

    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def save_processed(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save processed DataFrame to CSV, creating parent directories if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)
