from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def build_mailbox_cluster_mapping(df: pd.DataFrame, labels: Sequence[int], mailbox_col: str) -> pd.DataFrame:
    df = df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]
    if df.empty:
        return pd.DataFrame(columns=["mailbox", "cluster", "count", "ratio"])
    counts = (
        df.groupby([mailbox_col, "cluster"])
        .size()
        .reset_index(name="count")
        .sort_values(["count"], ascending=False)
    )
    totals = counts.groupby(mailbox_col)["count"].transform("sum")
    counts["ratio"] = counts["count"] / totals
    counts = counts.rename(columns={mailbox_col: "mailbox"})
    return counts


def save_mailbox_mapping(mapping: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(path, index=False)
    logger.info("Saved mailbox mapping to %s", path)
