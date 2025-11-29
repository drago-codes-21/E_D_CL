from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from .embedder import SentenceEmbedder

logger = logging.getLogger(__name__)


def benchmark_model(
    texts: List[str],
    model_name: str,
    device: str = "auto",
    batch_size: int = 32,
    model_path: Path | None = None,
) -> float:
    """
    Quick timing benchmark for embedding generation.
    Returns seconds elapsed.
    """
    embedder = SentenceEmbedder(model_name=model_name, device=device, batch_size=batch_size, model_path=model_path)
    start = time.time()
    _ = embedder.encode(texts)
    elapsed = time.time() - start
    logger.info("Benchmark for %s: %.2fs on %d texts", model_name, elapsed, len(texts))
    return elapsed
