from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer

from .preprocess import clean_text


class Embedder:
    def __init__(self, model_name: str, allow_download: bool = False):
        # Disable telemetry noise and keep startup deterministic.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
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
