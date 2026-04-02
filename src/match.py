"""Vector similarity and ranking helpers."""

from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the matrix of inner-products between `a` and `b`.

    When embeddings are L2-normalized this is equivalent to cosine similarity.
    """
    return a @ b.T


def rank_candidates(sim_scores: np.ndarray, candidate_ids: list[str], top_k: int) -> list[tuple[str, float]]:
    """Return the best candidate ids and scores for one job vector."""
    
    k = min(top_k, len(candidate_ids))
    if k == 0:
        return []
    # Use argpartition for large arrays to avoid full sort cost; then sort the
    # small top-k slice to return results in descending order.
    if len(candidate_ids) > 500 and k < len(candidate_ids):
        part = np.argpartition(-sim_scores, k - 1)[:k]
        ordered = part[np.argsort(-sim_scores[part])]
        return [(candidate_ids[i], float(sim_scores[i])) for i in ordered]
    else:
        idx = np.argsort(-sim_scores)[:k]
        return [(candidate_ids[i], float(sim_scores[i])) for i in idx]



