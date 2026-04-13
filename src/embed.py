"""Thin wrapper around SentenceTransformer embedding generation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from sentence_transformers import SentenceTransformer

from .preprocess import clean_text


def _clear_broken_loopback_proxy() -> None:
    # Some local environments export a dead loopback proxy that breaks
    # Hugging Face downloads. Clear only that known-invalid value.
    invalid = {"http://127.0.0.1:9", "https://127.0.0.1:9"}
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        value = os.environ.get(key, "").strip().lower()
        if value in invalid:
            os.environ.pop(key, None)


def _configure_local_model_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "cn6000-matcher" / "model_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_root / "sentence_transformers"))


class Embedder:
    """Load the embedding model and expose normalized batch encoding."""
    def __init__(self, model_name: str, allow_download: bool = False):
        # Disable telemetry noise and keep startup deterministic.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        _clear_broken_loopback_proxy()
        _configure_local_model_cache()
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception as exc:
            if allow_download:
                self.model = SentenceTransformer(model_name)
                return
            raise RuntimeError(
                "Model is not cached locally. Run once with --allow-model-download "
                "or pre-download the model, then rerun for fast local-only startup."
            ) from exc

    def encode(self, texts: list[str], batch_size: int = 64):
        """Encode a list of `texts` into normalized embeddings.

        - We clean texts with `clean_text` before encoding.
        - `batch_size` helps control CPU/GPU memory usage for large sets.
        - `normalize_embeddings=True` ensures embeddings are unit length so
          inner product equals cosine similarity and works with FAISS IndexFlatIP.
        """
        cleaned = [clean_text(t) for t in texts]
        return self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )


