from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """
    Wrapper around SentenceTransformer for robust batch embedding generation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        model_path: Path | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.model_path = model_path
        self.model: SentenceTransformer | None = None

    def _resolve_device(self) -> str:
        if self.device.lower() != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            self.load()
        assert self.model is not None
        return self.model

    def load(self) -> None:
        if self.model_path:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Local model path not found: {self.model_path}")
            logger.info("Loading embedding model from local path: %s", self.model_path)
            self.model = SentenceTransformer(str(self.model_path), device=self._resolve_device())
        else:
            logger.info("Loading embedding model by name: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name, device=self._resolve_device())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        logger.info("Encoding %d texts in batches of %d", len(texts), self.batch_size)
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return embeddings

    def encode_to_memmap(self, texts: Sequence[str], path: Path) -> np.memmap:
        model = self._ensure_model()
        path.parent.mkdir(parents=True, exist_ok=True)
        dim = model.get_sentence_embedding_dimension()
        mmap = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(texts), dim))
        logger.info("Encoding %d texts to memmap %s with dim %d", len(texts), path, dim)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = model.encode(
                batch,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ).astype(np.float32)
            mmap[start : start + len(batch)] = batch_embeddings
        mmap.flush()
        return mmap

    def generate_embeddings(self, texts: Sequence[str], path: Path, use_memmap: bool = False) -> np.ndarray:
        if use_memmap:
            embeddings = self.encode_to_memmap(texts, path)
        else:
            embeddings = self.encode(texts)
            self.save_embeddings(embeddings, path)
        return embeddings

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        logger.info("Saved embeddings to %s", path)

    @staticmethod
    def load_embeddings(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        logger.info("Loading embeddings from %s", path)
        return np.load(path)
