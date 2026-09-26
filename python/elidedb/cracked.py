"""Sharp text search: student shortlist, teacher verdict, cracked cache.

THE PROBLEM THIS SOLVES
-----------------------
Student (FDNN-V) embeddings rank the whole corpus in ~1 ms but their text
ranking is measurably weak (top-10 agreement 0.5/10 vs the 4.3/10
teacher-self ceiling). The teacher ranks text pristinely but costs 27.7 ms
per frame — unaffordable corpus-wide, affordable on a SHORTLIST.

So the query plan is the database's oldest shape, applied to models:

    cheap operator over everything  ->  expensive operator over survivors
    student ranks all windows (~1ms)    teacher re-scores top-N (~30ms each)

THE CRACKING PART
-----------------
Every teacher vector computed for a query is WRITTEN BACK to the
`teacher_windows` table. The next query that touches those windows pays
nothing. Like database cracking (Idreos et al., CIDR 2007), the index
materialises as a side effect of the workload: hot regions of the corpus
become pristine after their first visit, cold regions never cost a cent.
Cost scales with what users ask, not with what they store.

The teacher here is SigLIP-224 ("fast") — measured within the teacher-self
agreement band of the 384 model at a third of the price — and the query text
is embedded with the SAME checkpoint, so the rerank compares like with like.
"""
from __future__ import annotations

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

TEACHER = "fast"                    # resolved by embeddings.resolve_model
_CACHE: dict = {}                   # (store, version) -> {key: vec}


def _cache_map(store):
    t = store.table("teacher_windows")
    st = t.state()
    key = (str(store.dir), st.version)
    if key not in _CACHE:
        m = {}
        if st.files:
            tb = t.scan()
            for s, a, v in zip(tb.column("stream").to_pylist(),
                               tb.column("ts").to_pylist(),
                               tb.column("vector").to_pylist()):
                m[(s, int(a))] = np.asarray(v, np.float32)
        _CACHE.clear()
        _CACHE[key] = m
    return _CACHE[key]


def _teacher_embed_windows(store, wins):
    """Decode ONE centre frame per window through the byte-range path and
    embed with the teacher. `wins` = [(stream, t0, t1)]."""
    from PIL import Image

    from .embeddings import _embed_images
    frames = store.table("frames").scan()
    imgs, keys = [], []
    for (s, a, b) in wins:
        mid = (a + b) // 2
        sel = frames.filter(pc.and_(
            pc.equal(frames.column("stream"), s),
            pc.and_(pc.greater_equal(frames.column("ts"),
                                     mid - 2_000_000_000),
                    pc.less_equal(frames.column("ts"),
                                  mid + 2_000_000_000))))
        from .video import FrameSet
        dec = FrameSet(store, "frames", sel).decode(stream=s, width=448,
                                                    limit=1)
        if dec:
            imgs.append(Image.fromarray(dec[0][1]))
            keys.append((s, a, b))
    if not imgs:
        return {}
    vecs = _embed_images(imgs, TEACHER)
    return {k: v for k, v in zip(keys, vecs)}


def search_sharp(store, text, k=10, shortlist=48):
    """Student ranks everything; the teacher re-scores the shortlist; misses
    are cached into the store. Returns (hits, stats)."""
    from .embeddings import _vec_table, embed_text
    t_start = time.perf_counter()
    tbl, vecs = _vec_table(store, "embeddings")
    a_t0 = tbl.column("ts").to_numpy()
    a_t1 = tbl.column("t1").to_numpy()
    a_s = tbl.column("stream").to_numpy(zero_copy_only=False)

    q_student = embed_text(text)                       # 384-space (student's)
    order = np.argsort(-(vecs @ q_student))[:shortlist]
    wins = [(str(a_s[i]), int(a_t0[i]), int(a_t1[i])) for i in order]

    cache = _cache_map(store)
    missing = [w for w in wins if (w[0], w[1]) not in cache]
    t_miss = time.perf_counter()
    if missing:
        fresh = _teacher_embed_windows(store, missing)
        if fresh:
            ft = pa.table({
                "ts": pa.array([a for (_, a, _b) in fresh], pa.int64()),
                "t1": pa.array([b for (_, _a, b) in fresh], pa.int64()),
                "stream": pa.array([s for (s, _a, _b) in fresh]),
                "vector": pa.array([v.tolist() for v in fresh.values()],
                                   pa.list_(pa.float32(), 1152)),
            })
            store.table("teacher_windows").append(
                ft, kind="embeddings",
                meta={"model": TEACHER, "written_by": "query cracking"})
            for (s, a, _b), v in fresh.items():
                cache[(s, a)] = v
    miss_ms = (time.perf_counter() - t_miss) * 1e3

    q_teacher = embed_text(text, model_id=TEACHER)     # same checkpoint as
    hits = []                                          # the cached vectors
    for (s, a, b) in wins:
        tv = cache.get((s, a))
        if tv is None:
            continue
        hits.append({"stream": s, "t0": a, "t1": b,
                     "score": float(tv @ q_teacher), "teacher": True})
    hits.sort(key=lambda h: -h["score"])
    stats = {"method": "sharp", "shortlist": len(wins),
             "cache_misses": len(missing),
             "teacher_ms": round(miss_ms, 1),
             "cache_size": len(cache),
             "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    return hits[:k], stats
