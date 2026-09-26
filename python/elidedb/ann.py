"""ANN tiers over the embeddings table. Three mechanisms, one contract:

- exact       one matmul; correct by definition; right up to ~10^6 vectors
- IVF         learned cells (HDBSCAN centroids, embeddings.py) — prune cells,
              scan survivors exactly; noise always scanned
- HNSW        graph ANN (hnswlib) for sub-ms search past the matmul crossover
- IVF-PQ      product quantization: vectors compressed ~48x into subspace
              codes; asymmetric-distance scan + EXACT rerank of the top pool
              (the SCANN/Faiss recipe: approximate to shortlist, never to
              answer)

Artifacts are version-suffixed sidecars under tables/embeddings/_index/,
rebuilt like any derived state, and recorded in the table log. Search picks
the best available tier automatically; every path supports HYBRID
retrieval — time-range and stream predicates pushed into candidate
selection so vector search composes with the store's core dimension, time.
"""
from __future__ import annotations

import json

import numpy as np


def _run_dir(store):
    d = store.dir / "tables" / "embeddings" / "_index"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _emb(store):
    from .embeddings import _vec_table
    return _vec_table(store)


def build_hnsw(store, M: int = 16, ef_construction: int = 200) -> dict:
    """Hierarchical navigable small world graph over unit vectors
    (inner-product space == cosine). O(log n) expected search hops."""
    import hnswlib
    t, vecs = _emb(store)
    v = store.table("embeddings").state().version
    ix = hnswlib.Index(space="ip", dim=vecs.shape[1])
    ix.init_index(max_elements=len(vecs), M=M,
                  ef_construction=ef_construction, random_seed=7)
    ix.add_items(vecs, np.arange(len(vecs)))
    path = _run_dir(store) / f"hnsw.v{v}.bin"
    ix.save_index(str(path))
    (path.with_suffix(".json")).write_text(json.dumps(
        {"dim": int(vecs.shape[1]), "n": len(vecs), "M": M}))
    # An ANN index is a derived sidecar keyed to the embeddings DATA version,
    # not a new data version — building it must NOT advance the log (doing so
    # would invalidate the artifact we just named .v{v} against the bumped
    # current version). The .v{v} filename IS the binding.
    return {"n": len(vecs), "bytes": path.stat().st_size, "version": v}


def load_hnsw(store):
    # Look for the artifact BEFORE importing. hnswlib is an optional
    # accelerator; when it is absent the planner must fall back to the exact
    # scan, not raise ModuleNotFoundError out of the middle of a query and
    # take down search entirely.
    cands = sorted(_run_dir(store).glob("hnsw.v*.bin"), reverse=True)
    if not cands:
        return None
    try:
        import hnswlib
    except ImportError:
        return None
    v = store.table("embeddings").state().version
    for cand in cands:
        if int(cand.stem.split(".v")[-1]) != v:
            continue  # stale: embeddings changed since this was built
        meta = json.loads(cand.with_suffix(".json").read_text())
        ix = hnswlib.Index(space="ip", dim=meta["dim"])
        ix.load_index(str(cand), max_elements=meta["n"])
        return ix
    return None


def build_ivfpq(store, nlist: int | None = None, m: int = 8,
                nbits: int = 8) -> dict:
    """Coarse k-means cells + per-subspace codebooks. A d=1152 float32
    vector becomes `m` uint8 codes (m=8 → 576x… realistically 4608B → 8B =
    576x raw, ~48x vs the parquet-compressed vectors). Scans read codes, not
    vectors; the exact rerank reads only the shortlist's true vectors."""
    from sklearn.cluster import KMeans
    t, vecs = _emb(store)
    n, d = vecs.shape
    v = store.table("embeddings").state().version
    nlist = nlist or max(1, int(np.sqrt(n)))
    coarse = KMeans(n_clusters=min(nlist, n), n_init=4,
                    random_state=0).fit(vecs)
    resid = vecs - coarse.cluster_centers_[coarse.labels_]
    assert d % m == 0, f"dim {d} not divisible by m={m}"
    sub = d // m
    codebooks = np.zeros((m, 2 ** nbits, sub), np.float32)
    codes = np.zeros((n, m), np.uint8)
    for j in range(m):
        block = resid[:, j * sub:(j + 1) * sub]
        km = KMeans(n_clusters=min(2 ** nbits, n), n_init=2,
                    random_state=j).fit(block)
        k = km.cluster_centers_.shape[0]
        codebooks[j, :k] = km.cluster_centers_
        codes[:, j] = km.labels_.astype(np.uint8)
    path = _run_dir(store) / f"ivfpq.v{v}.npz"
    np.savez_compressed(path, centers=coarse.cluster_centers_,
                        labels=coarse.labels_.astype(np.int32),
                        codebooks=codebooks, codes=codes)
    # derived sidecar keyed to the embeddings data version; no log bump (see
    # build_hnsw) — the .v{v} filename binds it to the current vectors.
    return {"n": n, "nlist": int(nlist), "m": m,
            "bytes": path.stat().st_size,
            "code_bytes_per_vec": m, "version": v}


def load_ivfpq(store):
    v = store.table("embeddings").state().version
    for cand in sorted(_run_dir(store).glob("ivfpq.v*.npz"), reverse=True):
        if int(cand.stem.split(".v")[-1]) == v:
            return np.load(cand)
    return None


def search_ivfpq(store, q: np.ndarray, k: int, nprobe: int = 8,
                 rerank: int = 4, mask: np.ndarray | None = None):
    """ADC scan: distance ≈ coarse-center dot + sum of per-subspace code
    dots (table lookups, no vector reads), then exact rerank of the top
    `rerank*k` shortlist. Approximation shortlists; it never answers."""
    art = load_ivfpq(store)
    if art is None:
        return None
    t, vecs = _emb(store)
    centers, labels = art["centers"], art["labels"]
    codebooks, codes = art["codebooks"], art["codes"]
    m, ksub, sub = codebooks.shape
    probe = np.argsort(centers @ q)[::-1][:nprobe]
    cand = np.isin(labels, probe)
    if mask is not None:
        cand &= mask
    idx = np.where(cand)[0]
    if len(idx) == 0:
        return [], {"scanned": 0, "total": len(vecs)}
    # lookup tables: q-subvector · every codeword, per subspace
    lut = np.stack([codebooks[j] @ q[j * sub:(j + 1) * sub]
                    for j in range(m)])                      # [m, ksub]
    approx = centers[labels[idx]] @ q + \
        lut[np.arange(m)[None, :], codes[idx]].sum(axis=1)
    short = idx[np.argsort(approx)[::-1][:max(k * rerank, k)]]
    exact = vecs[short] @ q
    order = np.argsort(exact)[::-1][:k]
    return ([(int(short[i]), float(exact[i])) for i in order],
            {"scanned": int(len(idx)), "total": len(vecs),
             "code_bytes": int(len(idx) * m),
             "reranked": int(len(short))})
