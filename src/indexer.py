from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None


# Persist candidate embeddings for index rebuild/reuse between runs.
def _save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))


def _load_embeddings(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


# Build or load a nearest-neighbor index (FAISS preferred, Annoy fallback).
def build_or_load_index(embeddings: np.ndarray, outputs_dir: Path, prefer_faiss: bool = True, n_trees: int = 10):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    emb_path = outputs_dir / "embeddings.npy"
    faiss_path = outputs_dir / "faiss.index"
    annoy_path = outputs_dir / "annoy.idx"

    # Save embeddings for reuse
    _save_embeddings(emb_path, embeddings)

    if prefer_faiss and faiss is not None:
        # build or load FAISS index
        dim = embeddings.shape[1]
        if faiss_path.exists():
            idx = faiss.read_index(str(faiss_path))
        else:
            idx = faiss.IndexFlatIP(dim)
            idx.add(embeddings)
            faiss.write_index(idx, str(faiss_path))
        return "faiss", idx, embeddings

    # Fallback to Annoy if available
    if AnnoyIndex is not None:
        dim = embeddings.shape[1]
        aidx = AnnoyIndex(dim, metric="angular")
        if annoy_path.exists():
            aidx.load(str(annoy_path))
        else:
            for i, v in enumerate(embeddings):
                aidx.add_item(i, v.tolist())
            aidx.build(n_trees)
            aidx.save(str(annoy_path))
        return "annoy", aidx, embeddings

    # Neither available — return None so caller can fallback
    return None, None, embeddings


# Query the selected index backend and return indices with similarities.
def query_index(index_type: str, index_obj, queries: np.ndarray, top_k: int, embeddings: np.ndarray):
    queries = np.asarray(queries, dtype=np.float32)
    if index_type == "faiss":
        sims, idxs = index_obj.search(queries, top_k)
        return idxs, sims
    if index_type == "annoy":
        # Annoy returns indices and distances (angular). We'll compute exact cosine similarities
        all_idxs = []
        all_sims = []
        # ensure embeddings are float32 numpy
        cand_emb = np.asarray(embeddings, dtype=np.float32)
        for q in queries:
            idxs, dists = index_obj.get_nns_by_vector(q.tolist(), top_k, include_distances=True)
            if len(idxs) == 0:
                all_idxs.append(np.array([], dtype=np.int64))
                all_sims.append(np.array([], dtype=np.float32))
                continue
            idxs_arr = np.array(idxs, dtype=np.int64)
            # compute exact cosine similarity between q and these candidate embeddings
            # embeddings were normalized by the embedder so inner product == cosine
            sims = (q @ cand_emb[idxs_arr].T).astype(np.float32)
            all_idxs.append(idxs_arr)
            all_sims.append(sims)
        # convert lists to arrays padded to top_k
        max_k = max((len(a) for a in all_idxs), default=0)
        idxs_out = np.full((len(queries), max_k), -1, dtype=np.int64)
        sims_out = np.zeros((len(queries), max_k), dtype=np.float32)
        for i, (idx_arr, sim_arr) in enumerate(zip(all_idxs, all_sims)):
            k = len(idx_arr)
            if k > 0:
                idxs_out[i, :k] = idx_arr
                sims_out[i, :k] = sim_arr
        return idxs_out, sims_out

    raise RuntimeError(f"Unknown index type: {index_type}")
