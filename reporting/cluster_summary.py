from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import pandas as pd


def build_cluster_summary(
    df: pd.DataFrame,
    labels: Sequence[int],
    keywords: Dict[int, List[str]],
    metrics: Dict[str, object],
    mailbox_column: str,
) -> List[Dict[str, object]]:
    cluster_density = {int(k): float(v) for k, v in metrics.get("cluster_density", {}).items()}
    summary: List[Dict[str, object]] = []
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        subset = df_with_labels[df_with_labels["cluster"] == cluster_id]
        mailboxes = Counter(subset[mailbox_column]).most_common(5)
        summary.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(subset)),
                "keywords": keywords.get(int(cluster_id), []),
                "top_mailboxes": mailboxes,
                "density": cluster_density.get(int(cluster_id)),
            },
        )
    return summary


def collect_sample_emails(
    df: pd.DataFrame,
    labels: Sequence[int],
    subject_col: str,
    body_col: str,
    mailbox_col: str,
    sender_col: str,
    timestamp_col: str,
    per_cluster: int,
) -> Dict[int, List[Dict[str, str]]]:
    df = df.copy()
    df["cluster"] = labels
    samples: Dict[int, List[Dict[str, str]]] = {}
    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            continue
        subset = df[df["cluster"] == cluster_id].head(per_cluster)
        samples[int(cluster_id)] = [
            {
                "subject": str(row[subject_col]),
                "body_snippet": str(row[body_col])[:400],
                "mailbox": str(row[mailbox_col]),
                "sender": str(row[sender_col]),
                "timestamp": str(row[timestamp_col]),
            }
            for _, row in subset.iterrows()
        ]
    return samples
