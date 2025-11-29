from __future__ import annotations

from typing import Dict, List, Sequence


def summarize_clusters(
    cluster_keywords: Dict[int, List[str]],
    sample_subjects: Dict[int, Sequence[str]],
    prompt_template: str,
) -> Dict[int, str]:
    """
    Lightweight placeholder summarizer that can be swapped with an actual LLM call.
    """
    summaries: Dict[int, str] = {}
    for cluster_id, keywords in cluster_keywords.items():
        keyword_text = ", ".join(keywords)
        subject_text = "; ".join(sample_subjects.get(cluster_id, [])[:2])
        summaries[cluster_id] = prompt_template.format(keywords=keyword_text, subjects=subject_text)
    return summaries
