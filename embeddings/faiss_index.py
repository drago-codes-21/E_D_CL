from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an inner-product FAISS index for similarity search.
    Assumes embeddings are normalized.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index with %d vectors of dim %d", embeddings.shape[0], dim)
    return index


def search_index(index: faiss.IndexFlatIP, query_vecs: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index. Returns scores and indices.
    """
    scores, ids = index.search(query_vecs, top_k)
    return scores, ids


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def load_index(path: Path) -> faiss.IndexFlatIP:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(str(path))
    logger.info("Loaded FAISS index from %s", path)
    return index
