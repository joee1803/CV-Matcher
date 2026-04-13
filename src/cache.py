"""Persistent embedding cache used to speed up repeat runs."""

from __future__ import annotations

import hashlib
import json
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
        """Map a text hash to its cached `.npy` embedding path."""
        return self.cache_dir / f"{hash_key}.npy"

    def get_hash(self, text: str) -> str:
        """Hash normalized text so identical content shares cache entries."""
        return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Return a cached embedding if it exists on disk."""
        key = self.get_hash(text)
        path = self._path_for_hash(key)
        if not path.exists():
            self._misses += 1
            return None
        self._hits += 1
        return np.load(path, allow_pickle=False)

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Persist one embedding to disk."""
        key = self.get_hash(text)
        path = self._path_for_hash(key)
        np.save(path, np.asarray(embedding, dtype=np.float32))

    def encode_batch(self, texts: list[str], embedder) -> np.ndarray:
        """Encode a batch by mixing cache hits with freshly generated embeddings."""
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
        """Delete cached embedding files and reset runtime counters."""
        removed = 0
        for path in self.cache_dir.glob("*.npy"):
            path.unlink(missing_ok=True)
            removed += 1
        self._hits = 0
        self._misses = 0
        return removed

    def stats(self) -> dict[str, float | int]:
        """Return runtime cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "requests": total,
            "hit_rate": hit_rate,
        }


class ExplanationCache:
    """Persistent JSON cache for match explanations."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "explanations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _payload_hash(self, job_id: str, candidate_id: str, similarity: float, cv_text: str, job_text: str) -> str:
        normalized = "\n".join(
            [
                job_id.strip().lower(),
                candidate_id.strip().lower(),
                f"{float(similarity):.6f}",
                clean_text(cv_text),
                clean_text(job_text),
            ]
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _path_for_key(self, hash_key: str) -> Path:
        return self.cache_dir / f"{hash_key}.json"

    def get(self, job_id: str, candidate_id: str, similarity: float, cv_text: str, job_text: str) -> str | None:
        hash_key = self._payload_hash(job_id, candidate_id, similarity, cv_text, job_text)
        path = self._path_for_key(hash_key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._misses += 1
            return None
        explanation = str(payload.get("explanation", "")).strip()
        if not explanation:
            self._misses += 1
            return None
        self._hits += 1
        return explanation

    def set(self, job_id: str, candidate_id: str, similarity: float, cv_text: str, job_text: str, explanation: str) -> None:
        hash_key = self._payload_hash(job_id, candidate_id, similarity, cv_text, job_text)
        path = self._path_for_key(hash_key)
        path.write_text(json.dumps({"explanation": explanation}, ensure_ascii=True), encoding="utf-8")

    def clear(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("*.json"):
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


