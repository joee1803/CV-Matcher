from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .preprocess import clean_text


class EmbeddingCache:
    """Persistent SHA256-keyed embedding cache backed by .npy files."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _path_for_hash(self, hash_key: str) -> Path:
        return self.cache_dir / f"{hash_key}.npy"

    def get_hash(self, text: str) -> str:
        return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        key = self.get_hash(text)
        path = self._path_for_hash(key)
        if not path.exists():
            self._misses += 1
            return None
        self._hits += 1
        return np.load(path, allow_pickle=False)

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self.get_hash(text)
        path = self._path_for_hash(key)
        np.save(path, np.asarray(embedding, dtype=np.float32))

    def encode_batch(self, texts: list[str], embedder) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is None:
                miss_indices.append(i)
                miss_texts.append(text)
            else:
                results[i] = cached

        if miss_texts:
            fresh = embedder.encode(miss_texts)
            for offset, idx in enumerate(miss_indices):
                emb = np.asarray(fresh[offset], dtype=np.float32)
                self.set(texts[idx], emb)
                results[idx] = emb

        return np.vstack([np.asarray(item, dtype=np.float32) for item in results if item is not None])

    def clear(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("*.npy"):
            path.unlink(missing_ok=True)
            removed += 1
        self._hits = 0
        self._misses = 0
        return removed

    def stats(self) -> dict[str, float | int]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "requests": total,
            "hit_rate": hit_rate,
        }
