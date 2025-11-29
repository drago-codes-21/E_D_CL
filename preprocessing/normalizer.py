from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from pipeline.settings import PreprocessingConfig
from preprocessing.cleaner import normalize_punctuation, remove_noise

logger = logging.getLogger(__name__)


def _format_attachments(raw_value: str, delimiter: str, join_token: str) -> str:
    if not raw_value:
        return ""
    attachments = [part.strip() for part in raw_value.split(delimiter) if part.strip()]
    if not attachments:
        return ""
    return f"attachments: {join_token.join(attachments)}"


def _format_sender(sender: str) -> str:
    return f"sender: {sender.strip()}" if sender else ""


def _format_timestamp(timestamp: str) -> str:
    return f"timestamp: {timestamp}" if timestamp else ""


def combine_and_normalize(df: pd.DataFrame, config: PreprocessingConfig, extra_boilerplate: Iterable[str], attachment_delimiter: str) -> pd.DataFrame:
    """
    Combine and normalize the configured text fields into a single clean_text column.
    """
    df = df.copy()
    df["subject_clean"] = df[config.text_column_subject].apply(lambda txt: remove_noise(str(txt), extra_boilerplate))
    df["body_clean"] = df[config.text_column_body].apply(lambda txt: remove_noise(str(txt), extra_boilerplate))

    sender_meta_series = (
        df[config.sender_column].astype(str).apply(_format_sender)
        if config.include_sender and config.sender_column in df.columns
        else pd.Series([""] * len(df))
    )
    timestamp_meta_series = (
        df[config.timestamp_column].astype(str).apply(_format_timestamp)
        if config.include_timestamp and config.timestamp_column in df.columns
        else pd.Series([""] * len(df))
    )
    attachment_meta_series = (
        df[config.attachments_column]
        .astype(str)
        .apply(lambda val: _format_attachments(val, attachment_delimiter, config.attachment_join_token))
        if config.include_attachments and config.attachments_column in df.columns
        else pd.Series([""] * len(df))
    )

    metadata = [
        " ".join(filter(None, parts))
        for parts in zip(
            sender_meta_series.tolist(),
            timestamp_meta_series.tolist(),
            attachment_meta_series.tolist(),
        )
    ]

    combined: list[str] = []
    for subject, body, meta in zip(df["subject_clean"], df["body_clean"], metadata):
        blocks = [subject.strip(), body.strip(), meta.strip()]
        text = " ".join(filter(None, blocks))
        if config.lowercase:
            text = text.lower()
        if config.punctuation_spacing:
            text = normalize_punctuation(text)
        combined.append(text)

    df["clean_text"] = combined
    df = df[df["clean_text"].str.len() >= config.min_chars].reset_index(drop=True)
    logger.info("Normalized text; kept %d rows", len(df))
    return df
