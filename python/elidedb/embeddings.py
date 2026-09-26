"""Semantic layer over Parquet: embeddings and centroids are tables like
everything else — versioned by the log, readable by DuckDB, no sidecar
binary formats. Vectors are fixed_size_list<float32>[d] columns.

Retrieval is two-stage (learned-cell IVF): rank cluster centroids, scan the
top nprobe cells exactly in full space; HDBSCAN noise is always scanned so
the prune can cost recall nothing. All ranking is one numpy matmul — exact,
simple, and fast far past 100k windows."""
from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_MODEL_CACHE = {}
DEFAULT_MODEL = "mlx-community/siglip-so400m-patch14-384"

# Ingest cost is ALL model, not storage: measured on an M-series machine,
# byte-range decode runs at 2.7 ms/frame while the 384px tower runs at
# 90.3 ms/frame — 97% of ingest is the encoder. So the encoder is a choice,
# not a constant.
#
#   quality  siglip-so400m-patch14-384   90.3 ms/frame   1152-d   (default)
#   fast     siglip-so400m-patch14-224   27.7 ms/frame   1152-d   3.3x faster
#
# `fast` is the SAME model and the same output space, fed 224px instead of
# 384px, so it is 256 patches per image instead of 729. Vectors from the two
# are NOT interchangeable — an index must be built and queried with one of
# them, which is why the model id is recorded in the table's metadata and the
# query path reads it back.
MODELS = {
    "quality": "mlx-community/siglip-so400m-patch14-384",
    "fast": "mlx-community/siglip-so400m-patch14-224",
}


def resolve_model(name):
    """Accept a preset name ('fast'/'quality') or a raw HF model id."""
    return MODELS.get(name, name) if name else DEFAULT_MODEL


# PORTABILITY: the mlx build of siglip-so400m is a straight conversion
# of the google checkpoint, so the SAME weights run through
# transformers on any machine and land in the same embedding space.
# Backend is auto-detected (mlx where it imports, torch elsewhere) and
# can be forced with ELIDEDB_TEXT_BACKEND=torch for parity testing.
_HF_EQUIV = {
    "mlx-community/siglip-so400m-patch14-384":
        "google/siglip-so400m-patch14-384",
    "mlx-community/siglip-so400m-patch14-224":
        "google/siglip-so400m-patch14-224",
}


def _backend():
    if "backend" not in _MODEL_CACHE:
        import os
        forced = os.environ.get("ELIDEDB_TEXT_BACKEND", "").strip()
        if forced:
            _MODEL_CACHE["backend"] = forced
        else:
            try:
                import mlx_embeddings  # noqa: F401
                _MODEL_CACHE["backend"] = "mlx"
            except ImportError:
                _MODEL_CACHE["backend"] = "torch"
    return _MODEL_CACHE["backend"]


def _load_model(model_id):
    if model_id not in _MODEL_CACHE:
        from mlx_embeddings.utils import load
        _MODEL_CACHE[model_id] = load(model_id)
    return _MODEL_CACHE[model_id]


def _load_torch(model_id):
    key = ("torch", model_id)
    if key not in _MODEL_CACHE:
        from transformers import AutoModel, AutoProcessor

        from .device import pick, strip_vision
        dev, dtype = pick()
        hf = _HF_EQUIV.get(model_id, model_id)
        m = AutoModel.from_pretrained(
            hf, dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
        m = strip_vision(m, "vision_model")
        _MODEL_CACHE[key] = (m, AutoProcessor.from_pretrained(hf), dev)
    return _MODEL_CACHE[key]


def _embed_images(images, model_id):
    if _backend() == "torch":
        import torch
        model, processor, dev = _load_torch(resolve_model(model_id))
        iv = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(
                pixel_values=iv["pixel_values"].to(
                    dev, model.dtype))
        out = out.float().cpu().numpy().astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    import mlx.core as mx
    model, processor = _load_model(resolve_model(model_id))
    iv = processor(images=images, return_tensors="np")
    out = np.array(model.get_image_features(mx.array(iv["pixel_values"])),
                   dtype=np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def embed_text(text, model_id=DEFAULT_MODEL):
    if _backend() == "torch":
        import torch
        model, processor, dev = _load_torch(resolve_model(model_id))
        try:
            max_len = int(
                model.config.text_config.max_position_embeddings)
        except AttributeError:
            max_len = 64
        ti = processor(text=[text], padding="max_length",
                       max_length=max_len, truncation=True,
                       return_tensors="pt")
        with torch.no_grad():
            v = model.get_text_features(
                input_ids=ti["input_ids"].to(dev))
        v = v[0].float().cpu().numpy().astype(np.float32)
        return v / np.linalg.norm(v)
    import mlx.core as mx
    model, processor = _load_model(resolve_model(model_id))
    # Each checkpoint has its own text context (so400m-384: 64 tokens,
    # so400m-224: 16). Ask the model rather than assuming.
    try:
        max_len = int(model.config.text_config.max_position_embeddings)
    except AttributeError:
        max_len = 64
    ti = processor(text=[text], padding="max_length", max_length=max_len,
                   truncation=True, return_tensors="np")
    v = np.array(model.get_text_features(mx.array(ti["input_ids"])),
                 dtype=np.float32)[0]
    return v / np.linalg.norm(v)


_MAT_CACHE: dict = {}


def _vec_table(store, name="embeddings", version=None, column="vector"):
    """Vector table + MEMORY-MAPPED matrix, cached per log version.

    Two generations of this function materialized the matrix in RAM. At
    pilot scale that broke: bridge-full's 180k x 1152 embeddings are 0.83 GB
    of data but cost +4.3 GB peak RSS to load (parquet decode + Arrow
    chunks + combine copy + the cache holding table AND matrix), and the
    desk warms every store — 7 GB before the first query. The fix is the
    store's own law applied to vectors: mmap for reads. The matrix is
    materialized ONCE per (table, version) into a raw .npy sidecar, then
    every process maps it — RSS is only the pages a query touches, startup
    costs a file open, and the OS page cache decides residency.

    Row alignment: the sidecar is written from the same scan() that serves
    the meta columns; scan's ts sort is stable over a deterministic file
    order, so a later projected scan yields the identical permutation. A
    length mismatch (e.g. sidecar from a dead version) forces a rebuild.
    The parquet remains the source of truth — a sidecar is disposable.
    """
    import os
    import uuid as _uuid
    ver = store.table(name).state().version if version is None else version
    key = (str(store.dir), name, column, ver)
    if key in _MAT_CACHE:
        return _MAT_CACHE[key]

    tab = store.table(name)
    cache_dir = tab.dir / "_cache"
    npy = cache_dir / f"{column}-v{ver}.npy"

    t = vecs = None
    if npy.exists():
        st = tab.state(version)
        if st.files:
            names = pq.ParquetFile(
                tab.dir / st.files[0].path).schema_arrow.names
            meta_cols = [c for c in names if c != column]
            t = tab.scan(version=version, columns=meta_cols)
            vecs = np.load(npy, mmap_mode="r")
            if len(vecs) != len(t):
                t = vecs = None                    # stale sidecar: rebuild

    if vecs is None:
        t_full = tab.scan(version=version)
        if len(t_full) == 0:
            extra = ""
            try:
                if store.table("frame_vectors").state().files:
                    extra = (" Per-frame vectors already exist, so this "
                             "costs a numpy mean, not a GPU pass.")
            except Exception:
                pass
            raise RuntimeError(
                f"store '{store.name}' has no '{name}' table — run "
                f"store.embed_windows() first.{extra}")
        col = t_full.column(column)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        try:                               # FixedSizeList: flat buffer reshape
            mat = col.values.to_numpy(zero_copy_only=False) \
                .astype(np.float32, copy=False).reshape(len(t_full), -1)
        except Exception:                  # any other layout: the slow road
            mat = np.stack([np.asarray(v, dtype=np.float32)
                            for v in col.to_pylist()])
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f".{_uuid.uuid4().hex[:8]}.npy"
        np.save(tmp, np.ascontiguousarray(mat))
        os.replace(tmp, npy)               # atomic: readers see whole files
        t = t_full.drop_columns([column])  # meta only — no double storage
        del t_full, mat, col
        vecs = np.load(npy, mmap_mode="r")

    if len(_MAT_CACHE) > 8:
        _MAT_CACHE.clear()
    _MAT_CACHE[key] = (t, vecs)
    return t, vecs


def pool_windows(store, window_s=2.0, stride_s=None, table="frame_vectors"):
    """Build the `embeddings` table by POOLING existing per-frame vectors.

    A window embedding is the mean of its frame embeddings. If `frame_vectors`
    already exists there is nothing to compute with a model: decoding every
    frame again and re-running SigLIP to reach the same answer is pure waste —
    on the Bridge store that was 25 minutes of GPU to reproduce a number a
    numpy mean gives in under a second.

    This is the ordinary database move: two indexes over one scan, not two
    scans.
    """
    fv = store.table(table).scan()
    if len(fv) == 0:
        raise RuntimeError(f"'{table}' is empty — run embed_frames() first")
    win = int(window_s * 1e9)
    stride = int((stride_s or window_s) * 1e9)
    t_start = time.time()
    rows = {"ts": [], "t1": [], "stream": [], "vector": []}
    for s in sorted(set(fv.column("stream").to_pylist())):
        sub = fv.filter(pc.equal(fv.column("stream"), s))
        ts = sub.column("ts").to_numpy()
        order = np.argsort(ts)
        ts = ts[order]
        # zero-copy reshape, NOT to_pylist(): at 1.6M frames the Python-list
        # road needs ~50 GB; the FixedSizeList buffer is already the matrix
        col = sub.column("vector")
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        try:
            vecs = col.values.to_numpy(zero_copy_only=False) \
                .astype(np.float32, copy=False).reshape(len(sub), -1)[order]
        except Exception:
            vecs = np.asarray(col.to_pylist(), dtype=np.float32)[order]
        t = int(ts[0])
        while t <= int(ts[-1]):
            lo, hi = np.searchsorted(ts, [t, t + win])
            if hi > lo:
                v = vecs[lo:hi].mean(axis=0)
                v /= np.linalg.norm(v) + 1e-8
                rows["ts"].append(t)
                rows["t1"].append(min(t + win - 1, int(ts[-1])))
                rows["stream"].append(s)
                rows["vector"].append(v)
            t += stride
    dim = len(rows["vector"][0])
    # FixedSizeListArray straight from the flat float32 buffer. The
    # tolist() road materialises n*dim PYTHON floats — at 1.8M frames /
    # 180k windows that was tens of GB and the process died by jetsam
    # (exit 137) on the very last stage of a 100 h load.
    flat = np.ascontiguousarray(
        np.stack(rows["vector"]).astype(np.float32)).reshape(-1)
    vec_arr = pa.FixedSizeListArray.from_arrays(pa.array(flat), dim)
    tbl = pa.table({
        "ts": pa.array(rows["ts"], pa.int64()),
        "t1": pa.array(rows["t1"], pa.int64()),
        "stream": pa.array(rows["stream"]),
        "vector": vec_arr,
    })
    st = store.table("embeddings").state()
    # `model` must be the id of the model that defines the SPACE — the query
    # path loads it as the text tower. Student-produced vectors live in the
    # TEACHER's space, so when the source was written by an engine (model
    # "fdnnv"), the space id is its `teacher` field. Writing the engine name
    # here sent "fdnnv" to the HF loader as a repo id.
    src = store.table(table).state().meta or {}
    src_model = src.get("model", DEFAULT_MODEL)
    if src_model in (None, "fdnnv"):
        src_model = src.get("teacher", DEFAULT_MODEL)
    meta = {"model": src_model, "built_by": "pooled from frame_vectors",
            "dim": dim, "window_s": window_s, "source_table": table,
            "seconds": round(time.time() - t_start, 2)}
    if st.files:
        import uuid as _uuid

        from .log import FileEntry
        from .store import write_parquet
        fn = f"part-{_uuid.uuid4().hex[:12]}.parquet"
        p = store.dir / "tables" / "embeddings" / fn
        write_parquet(tbl, p)
        tsv = tbl.column("ts").to_numpy()
        version = store.table("embeddings").log.commit(
            op="replace", kind="embeddings", schema=str(tbl.schema),
            add=[FileEntry(fn, len(tbl), p.stat().st_size,
                           int(tsv.min()), int(tsv.max()))],
            remove=[f.path for f in st.files], meta=meta)
    else:
        version = store.table("embeddings").append(tbl, kind="embeddings",
                                                   meta=meta)
    return {"windows": len(tbl), "dim": dim, "version": version,
            "seconds": meta["seconds"], "source": table}


def embed_windows(store, frame_table="frames", window_s=2.0,
                  frames_per_window=2, model=None, batch=16, stride_s=None,
                  incremental=True, reuse_frame_vectors=True):
    """Tumbling windows over every video stream → mean-pooled SigLIP vectors
    → one commit to the `embeddings` table. Frames come through the same
    byte-range path queries use.

    If per-frame vectors already exist, they are pooled instead of re-running
    the model (see `pool_windows`) — same result, no GPU.
    """
    if reuse_frame_vectors:
        try:
            if store.table("frame_vectors").state().files:
                return pool_windows(store, window_s, stride_s)
        except Exception:
            pass
    from PIL import Image  # noqa: F401 (decode happens in FrameSet)
    model = model or DEFAULT_MODEL
    tab = store.table(frame_table)
    st = tab.state()
    win_ns = int(window_s * 1e9)
    stride_ns = int((stride_s or window_s) * 1e9)
    frames = tab.scan()
    streams = sorted(set(frames.column("stream").to_pylist()))
    # Incremental: only embed windows past what the embeddings table already
    # covers per stream — adding a new day of footage costs a new day of
    # embedding, not a re-run of history.
    done_until = {}
    if incremental:
        try:
            prev = store.table("embeddings").scan()
            if len(prev):
                s_arr = prev.column("stream").to_pylist()
                t1_arr = prev.column("t1").to_pylist()
                for s_, e_ in zip(s_arr, t1_arr):
                    done_until[s_] = max(done_until.get(s_, 0), e_)
        except Exception:
            pass
    jobs = []  # (stream, t0, t1)
    for s in streams:
        rows = frames.filter(pc.equal(frames.column("stream"), s))
        ts = rows.column("ts").to_numpy()
        t = (int(ts[0]) // win_ns) * win_ns
        while t <= ts[-1]:
            lo, hi = np.searchsorted(ts, [t, t + win_ns])
            if hi > lo and t >= done_until.get(s, -1):
                jobs.append((s, max(t, int(ts[0])),
                             min(t + win_ns - 1, int(ts[-1]))))
            t += stride_ns
    from .video import FrameSet
    t_start = time.time()
    if not jobs:
        return {"windows": 0, "dim": None, "version": None, "seconds": 0.0,
                "note": "nothing new to embed (incremental)"}
    recs = {"ts": [], "t1": [], "stream": [], "vector": []}
    imgs, owners = [], []

    def flush():
        nonlocal imgs, owners
        if not imgs:
            return
        vecs = _embed_images(imgs, model)
        for (key, v) in zip(owners, vecs):
            pooled.setdefault(key, []).append(v)
        imgs, owners = [], []

    pooled = {}
    for (s, t0, t1) in jobs:
        fs = FrameSet(store, frame_table,
                      frames.filter(pc.and_(
                          pc.equal(frames.column("stream"), s),
                          pc.and_(pc.greater_equal(frames.column("ts"), t0),
                                  pc.less_equal(frames.column("ts"), t1)))))
        n = len(fs)
        picks = np.linspace(0, n - 1, min(frames_per_window, n)).round().astype(int)
        decoded = fs.decode(width=512)
        for p in picks:
            if p < len(decoded):
                from PIL import Image as PILImage
                imgs.append(PILImage.fromarray(decoded[p][1]))
                owners.append((s, t0, t1))
        if len(imgs) >= batch:
            flush()
    flush()
    for (s, t0, t1), vs in pooled.items():
        v = np.mean(vs, axis=0)
        v /= np.linalg.norm(v)
        recs["stream"].append(s)
        recs["ts"].append(t0)
        recs["t1"].append(t1)
        recs["vector"].append(v)
    dim = len(recs["vector"][0])
    t = pa.table({
        "ts": pa.array(recs["ts"], pa.int64()),
        "t1": pa.array(recs["t1"], pa.int64()),
        "stream": pa.array(recs["stream"]),
        "vector": pa.array([v.tolist() for v in recs["vector"]],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("embeddings").append(
        t, kind="embeddings",
        meta={"model": model, "dim": dim, "window_s": window_s,
              "source_table": frame_table,
              "embedded_in_s": round(time.time() - t_start, 1)})
    return {"windows": len(t), "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 1)}


def cluster(store, pca_dims=50, min_cluster_size=8):
    """PCA → HDBSCAN over the embeddings table → cluster ids written back as
    a new embeddings version + a `centroids` table (full-space, normalized:
    the coarse stage must rank in the space the fine stage scores in)."""
    t, vecs = _vec_table(store)
    from sklearn.decomposition import PCA
    import hdbscan
    red = PCA(n_components=min(pca_dims, len(vecs), vecs.shape[1]),
              random_state=0).fit_transform(vecs)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(red)
    out = t.drop_columns(["cluster"]) if "cluster" in t.column_names else t
    out = out.append_column("cluster", pa.array(labels.astype("int32")))
    # replace = remove old files + add the re-clustered ones, one commit
    st = store.table("embeddings").state()
    log = store.table("embeddings").log
    import pyarrow.parquet as pq
    import uuid as _uuid
    fname = f"part-{_uuid.uuid4().hex[:12]}.parquet"
    from .store import write_parquet
    write_parquet(out, store.dir / "tables" / "embeddings" / fname)
    from .log import FileEntry
    p = store.dir / "tables" / "embeddings" / fname
    tsv = out.column("ts").to_numpy()
    log.commit(op="recluster", kind="embeddings", schema=str(out.schema),
               add=[FileEntry(fname, len(out), p.stat().st_size,
                              int(tsv.min()), int(tsv.max()))],
               remove=[f.path for f in st.files],
               meta={**st.meta, "clusters": int(labels.max() + 1),
                     "noise": int((labels < 0).sum())})
    cents = []
    for c in range(labels.max() + 1):
        m = vecs[labels == c].mean(axis=0)
        cents.append(m / np.linalg.norm(m))
    if cents:
        ct = pa.table({
            "ts": pa.array([0] * len(cents), pa.int64()),
            "cluster": pa.array(range(len(cents)), pa.int32()),
            "vector": pa.array([c.tolist() for c in cents],
                               pa.list_(pa.float32(), vecs.shape[1])),
        })
        cst = store.table("centroids").state()
        store.table("centroids").log.commit(
            op="replace", kind="centroids", schema=str(ct.schema),
            add=[], remove=[f.path for f in cst.files])
        store.table("centroids").append(ct, kind="centroids")
    return {"clusters": int(labels.max() + 1),
            "noise": int((labels < 0).sum()), "windows": len(out)}


def _score_windows(vecs, idx, pos_vecs, neg_vecs, neg_weight):
    """Compositional scoring over a candidate set.

    - ONE positive term  → plain cosine (classic semantic search).
    - MANY positive terms → the window's score is the WORST of its per-term
      cosines (min-pool). This is the compositional AND: 'two people' AND
      'a laptop' means a clip of two people with NO laptop scores low on the
      laptop term and is therefore rejected — the fix for 'it returns every
      clip with two people'.
    - negative terms      → each subtracts its cosine (weighted), so
      '... NOT a phone' pushes phone-heavy frames down.
    """
    cand = vecs[idx]                          # [m, d]
    pos = cand @ pos_vecs.T                    # [m, n_pos]
    score = pos.min(axis=1)                    # min-pool = AND
    if neg_vecs is not None and len(neg_vecs):
        score = score - neg_weight * (cand @ neg_vecs.T).max(axis=1)
    return score


def _rank(store, q, k, nprobe, merge=True, t0=None, t1=None, streams=None,
          method="auto", pos_vecs=None, neg_vecs=None, neg_weight=0.5,
          min_score=None, percentile=None, table="embeddings", ctx=None,
          column="vector"):
    t, vecs = _vec_table(store, table, column=column)
    if table != "embeddings":
        # The ANN artifacts (HNSW graph, IVF-PQ codes, HDBSCAN centroids) are
        # built over `embeddings` and index THOSE row ids. Reusing them here
        # would return neighbours of the wrong table — silently, with
        # plausible-looking scores. Any other table scans exactly.
        method = "exact"
    all_t0 = t.column("ts").to_numpy()
    all_t1 = t.column("t1").to_numpy()
    all_s = t.column("stream").to_numpy(zero_copy_only=False)
    # `q` (the coarse retrieval direction) is the mean of positive terms;
    # `pos_vecs` carries the individual terms for compositional scoring.
    if pos_vecs is None:
        pos_vecs = q[None, :]

    # ---- hybrid retrieval: predicates pushed INTO candidate selection ------
    # Time and stream are first-class dimensions of this database; vector
    # search composes with them instead of post-filtering a global top-k
    # (which silently starves filtered queries of results).
    pred = np.ones(len(vecs), bool)
    if t0 is not None:
        pred &= all_t1 >= t0
    if t1 is not None:
        pred &= all_t0 <= t1
    if streams:
        pred &= np.isin(all_s, list(streams))

    labels = (t.column("cluster").to_numpy()
              if "cluster" in t.column_names else None)
    probed = total_clusters = 0
    used = "exact"

    idx = scores = None
    if method in ("auto", "hnsw"):
        from . import ann
        hx = ann.load_hnsw(store)
        if hx is not None:
            # overfetch beyond k so predicate filtering and segment merging
            # still see the event's neighborhood, then score exactly
            fetch = int(min(len(vecs), max(k * 8, 64)))
            hx.set_ef(max(fetch, 64))
            cand, _ = hx.knn_query(q, k=fetch)
            cand = cand[0]
            cand = cand[pred[cand]]
            if len(cand) >= min(k, pred.sum()):
                idx = np.asarray(cand)
                scores = vecs[idx] @ q
                used = "hnsw"
    if idx is None and method in ("auto", "ivfpq"):
        from . import ann
        r = ann.search_ivfpq(store, q, k=max(k * 4, 32), nprobe=max(nprobe, 8),
                             mask=pred) if method == "ivfpq" else None
        if r is not None and r[0]:
            idx = np.array([i for i, _ in r[0]])
            scores = np.array([s for _, s in r[0]])
            used = "ivfpq"
    if idx is None:
        mask = pred.copy()
        if labels is not None and nprobe > 0:
            try:
                _, cents = _vec_table(store, "centroids")
                total_clusters = len(cents)
                order = np.argsort(cents @ q)[::-1][:nprobe]
                probed = len(order)
                mask &= np.isin(labels, order) | (labels < 0)  # noise stays
                used = "ivf"
            except RuntimeError:
                pass
        idx = np.where(mask)[0]
        scores = None
    scanned = len(idx)
    # Final score is ALWAYS the compositional/exact function over the
    # candidate set (the coarse tier only shortlists; it never answers).
    scores = _score_windows(vecs, idx, pos_vecs, neg_vecs, neg_weight)

    # ---- optional fusion with the context index ----------------------------
    # Appearance cosines and context cosines live on different scales (SigLIP
    # image-text similarity is squashed by the modality gap into ~0.01-0.15,
    # while context vectors are mean-free and spread over most of [-1,1]).
    # A raw weighted sum would therefore be governed entirely by the context
    # term regardless of alpha. Standardising each over the CANDIDATE SET
    # first makes alpha mean what it says.
    if ctx is not None and len(idx):
        def _z(a):
            return (a - a.mean()) / (a.std() + 1e-8)
        a = float(ctx["alpha"])
        scores = (1.0 - a) * _z(scores) + a * _z(ctx["vecs"][idx] @ ctx["q"])
    streams_sel = all_s[idx]
    w_t0 = all_t0[idx]
    w_t1 = all_t1[idx]

    # ---- precision floor: an ABSOLUTE cut the user controls ----------------
    # A percentile keeps only the strongest fraction; min_score is a hard
    # cosine floor. Either turns "top-k of everything" into "only real hits",
    # so a query with 6 true matches returns 6, not 50.
    keep = np.ones(len(idx), bool)
    if percentile is not None and len(scores):
        keep &= scores >= np.percentile(scores, percentile)
    if min_score is not None:
        keep &= scores >= min_score
    if not keep.all():
        idx, scores = idx[keep], scores[keep]
        streams_sel, w_t0, w_t1 = streams_sel[keep], w_t0[keep], w_t1[keep]

    stats = {"scanned": scanned, "total": len(vecs), "method": used,
             "clusters_probed": probed, "clusters_total": total_clusters,
             "predicate_candidates": int(pred.sum()),
             "after_floor": int(len(idx))}

    if not merge:
        order = np.argsort(scores)[::-1][:k]
        hits = [{"stream": str(streams_sel[i]), "t0": int(w_t0[i]),
                 "t1": int(w_t1[i]), "score": float(scores[i]),
                 "windows": 1} for i in order]
        return hits, stats
    if len(idx) == 0:
        stats["qualifying_windows"] = 0
        stats["segments"] = 0
        return [], stats

    # ---- dynamic segments: merge, don't chunk -------------------------------
    # Fixed embedding windows are an INDEXING granularity, not an answer
    # granularity. A result is the maximal run of consecutive qualifying
    # windows on one stream: a 20 s event comes back as ONE 20 s hit (its
    # sub-windows are never returned separately), while a query that only
    # matches 2 s of it comes back as that tight 2 s. "Qualifying" is decided
    # per query from the score distribution — an absolute cutoff cannot work
    # because SigLIP cosines live on different scales per query.
    med = float(np.median(scores))
    top = float(scores.max())
    thr = med + 0.55 * (top - med)
    stats["threshold"] = round(thr, 4)
    qual = np.where(scores >= thr)[0]
    order = np.lexsort((w_t0[qual], streams_sel[qual]))
    qual = qual[order]

    gap_ns = int(np.median(w_t1[qual] - w_t0[qual])) + 1 if len(qual) else 0
    segs = []
    for i in qual:
        s, a, b, sc = (str(streams_sel[i]), int(w_t0[i]), int(w_t1[i]),
                       float(scores[i]))
        last = segs[-1] if segs else None
        if last and last["stream"] == s and a - last["t1"] <= gap_ns:
            last["t1"] = max(last["t1"], b)
            last["score"] = max(last["score"], sc)   # peak represents the segment
            last["mean"] = (last["mean"] * last["windows"] + sc) / (last["windows"] + 1)
            last["windows"] += 1
        else:
            segs.append({"stream": s, "t0": a, "t1": b, "score": sc,
                         "mean": sc, "windows": 1})
    segs.sort(key=lambda g: -g["score"])
    stats["qualifying_windows"] = len(qual)
    stats["segments"] = len(segs)
    return segs[:k], stats


def _parse_query(text):
    """Parse a compositional query string into (positive terms, negatives).

    Grammar (all optional, combinable):
      'a AND b'      — every term must match (compositional AND)
      'a NOT b'      — exclude b   (also '-b' or 'a -b')
      'a; b'         — same as AND
    Plain text with none of these is a single positive term (classic search).
    """
    import re
    neg = []
    # split on NOT / leading-minus tokens
    parts = re.split(r'\bNOT\b', text)
    head = parts[0]
    for extra in parts[1:]:
        neg.append(extra.strip())
    pos_raw = re.split(r'\bAND\b|;', head)
    pos = []
    for term in pos_raw:
        term = term.strip()
        # pull out inline -word exclusions
        toks = term.split()
        keep = []
        for tk in toks:
            if tk.startswith("-") and len(tk) > 1:
                neg.append(tk[1:])
            else:
                keep.append(tk)
        if keep:
            pos.append(" ".join(keep))
    pos = [p for p in pos if p]
    neg = [n for n in neg if n]
    return (pos or [text]), neg


def search(store, text, k=10, nprobe=3, merge=True, t0=None, t1=None,
           streams=None, method="auto", neg_weight=0.5, min_score=None,
           percentile=None, rerank=False, rerank_top=12, rerank_alpha=0.7):
    """Compositional text search. `text` may use AND / NOT / -term:
        'two people AND a laptop NOT a phone'
    `min_score` (absolute cosine floor) or `percentile` (keep top X%) turn
    ranked-everything into precise retrieval."""
    st = store.table("embeddings").state()
    model = st.meta.get("model", DEFAULT_MODEL)
    pos_terms, neg_terms = _parse_query(text)
    pos_vecs = np.stack([embed_text(p, model) for p in pos_terms])
    neg_vecs = (np.stack([embed_text(n, model) for n in neg_terms])
                if neg_terms else None)
    q = pos_vecs.mean(axis=0)
    q /= np.linalg.norm(q)  # coarse retrieval direction
    hits, stats = _rank(store, q, k, nprobe, merge=merge, t0=t0, t1=t1,
                        streams=streams, method=method, pos_vecs=pos_vecs,
                        neg_vecs=neg_vecs, neg_weight=neg_weight,
                        min_score=min_score, percentile=percentile)
    stats["positive_terms"] = pos_terms
    stats["negative_terms"] = neg_terms
    if rerank and hits:
        # relational stage: the expensive operator runs LAST, on the pruned set
        from .rerank import rerank_hits
        hits, info = rerank_hits(store, hits, text, top_n=rerank_top,
                                 alpha=rerank_alpha)
        stats["rerank"] = info
    return hits, stats


def search_text(store, text, k=10, nprobe=3, merge=True, t0=None, t1=None,
                streams=None, method="auto", **kw):
    # backward-compatible alias; forwards compositional kwargs too
    return search(store, text, k=k, nprobe=nprobe, merge=merge, t0=t0, t1=t1,
                  streams=streams, method=method, **kw)


def search_clip(store, stream, t0, t1, k=10, nprobe=3, merge=True,
                pt0=None, pt1=None, pstreams=None, method="auto"):
    t, vecs = _vec_table(store)
    s = t.column("stream").to_numpy(zero_copy_only=False)
    a = t.column("ts").to_numpy()
    b = t.column("t1").to_numpy()
    sel = (s == stream) & (a <= t1) & (b >= t0)
    if not sel.any():
        raise ValueError(f"no embedded windows overlap {stream} [{t0},{t1}]")
    q = vecs[sel].mean(axis=0)
    q /= np.linalg.norm(q)
    hits, stats = _rank(store, q, k + 8, nprobe, merge=merge,
                        t0=pt0, t1=pt1, streams=pstreams, method=method)
    hits = [h for h in hits
            if not (h["stream"] == stream and h["t0"] <= t1 and h["t1"] >= t0)]
    return hits[:k], stats
