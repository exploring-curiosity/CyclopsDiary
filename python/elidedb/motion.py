"""The motion vector channel: motion = temporal DIFFERENCE of appearance.

The Marengo lesson is that motion must be its own vector, never averaged
into appearance. The obvious implementations were measured and lost:
  X-CLIP (video-native, cross-frame attention)    close/open AUC 0.67
  VLM swap-contrast (query-time, 2 model calls)   0.86 (2B) / 0.91 (7B)
  ctx adapter (caption-trained)                   ~chance
  THIS: delta-appearance + query-swap difference  0.98 / 0.98

An event's motion vector is normalize(app(end) - app(start)) in SigLIP
space, from the per-frame vectors ALREADY on disk — ingest cost is a
numpy mean. A directional query's motion vector is
normalize(embed(q) - embed(swap(q))) — the same swap-contrast idea, moved
from VLM logits into embedding space, so it runs index-only in
microseconds. Both sides subtract their shared appearance component, so
what remains on each side is the DIRECTION of state change, and they
were trained (by SigLIP, incidentally) to point the same way.

Limits, measured: works when the state change is visually large
(drawer open<->close: 0.98); chance when a small object moves
(put-in/take-out: 0.46-0.50) — those queries are carried by the
appearance and caption-lexical channels instead. Non-directional queries
have no swap and the channel ABSTAINS (RRF median rank, no veto).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa


def index_motion(store, events_table="context_events", edge_s=1.5,
                 frames_table="frame_vectors"):
    """One motion vector per event span -> `motion_vectors` table.

    Event spans come from the gate-segmented event index so every retrieval
    unit carries both an appearance-family vector and a motion vector —
    the multi-vector layout, aligned. Rebuild = atomic replace commit.
    """
    import time
    import uuid

    from .embeddings import _vec_table
    from .log import FileEntry
    from .store import write_parquet
    t_start = time.time()

    ev = store.table(events_table).scan()
    fv, fvecs = _vec_table(store, frames_table)
    fs = np.asarray(fv.column("stream").to_pylist())
    ft = np.asarray([int(v) for v in fv.column("ts").to_pylist()])
    edge = int(edge_s * 1e9)

    # per-stream sorted timelines + searchsorted per event. The full-array
    # boolean-mask version was O(events x frames): at 100 h that is 50k
    # events over 1.8M frames = 16.5 MINUTES for what is 25 s of means.
    by_stream = {}
    for s in np.unique(fs):
        m = np.where(fs == s)[0]
        o = np.argsort(ft[m], kind="stable")
        by_stream[s] = (ft[m][o], fvecs[m[o]])

    rows_s, rows_a, rows_b, rows_v = [], [], [], []
    for s, a, b in zip(ev.column("stream").to_pylist(),
                       (int(v) for v in ev.column("ts").to_pylist()),
                       (int(v) for v in ev.column("t1").to_pylist())):
        if s not in by_stream:
            continue
        tt, vv = by_stream[s]
        lo, hi = np.searchsorted(tt, [a, b + 1])
        if hi - lo < 6:
            continue
        t0, t1 = int(tt[lo]), int(tt[hi - 1])
        e0 = np.searchsorted(tt, t0 + edge, side="right")
        e1 = np.searchsorted(tt, t1 - edge, side="left")
        v0 = vv[lo:max(e0, lo + 1)].mean(0)
        v1 = vv[min(e1, hi - 1):hi].mean(0)
        d = v1 - v0
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        rows_s.append(s)
        rows_a.append(a)
        rows_b.append(b)
        rows_v.append((d / n).astype(np.float32))
    if not rows_v:
        return {"events": 0}

    dim = len(rows_v[0])
    tbl = pa.table({
        "ts": pa.array(rows_a, pa.int64()),
        "t1": pa.array(rows_b, pa.int64()),
        "stream": pa.array(rows_s),
        "vector": pa.array([v.tolist() for v in rows_v],
                           pa.list_(pa.float32(), dim)),
    })
    tab = store.table("motion_vectors")
    meta = {"model": "delta-appearance", "dim": dim, "edge_s": edge_s,
            "events_table": events_table}
    try:
        existing = tab.state().files
    except Exception:
        existing = []
    if existing:
        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        path = tab.dir / fname
        write_parquet(tbl, path)
        tsv = tbl.column("ts")
        version = tab.log.commit(
            op="replace", kind="embeddings", schema=str(tbl.schema),
            add=[FileEntry(fname, len(tbl), path.stat().st_size,
                           tsv[0].as_py(), tsv[-1].as_py())],
            remove=[f.path for f in existing], meta=meta)
    else:
        version = tab.append(tbl, kind="embeddings", meta=meta)
    return {"events": len(tbl), "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 2)}


def motion_scores(store, qv, qv_swap):
    """(stream, t0, t1, score) per event: motion vectors against the
    query-direction vector. Pure matmul over a version-cached matrix."""
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "motion_vectors")
    qd = np.asarray(qv, np.float32) - np.asarray(qv_swap, np.float32)
    n = float(np.linalg.norm(qd))
    if n < 1e-6:
        return []
    sc = vecs @ (qd / n)
    return list(zip(tbl.column("stream").to_pylist(),
                    (int(v) for v in tbl.column("ts").to_pylist()),
                    (int(v) for v in tbl.column("t1").to_pylist()),
                    sc.tolist()))


_MOT_IDX = {}


def motion_lookup(store, qv, qv_swap):
    """lookup(stream, t0, t1) -> score of the recording CONTAINING the span,
    or nan. The per-stream sorted span index is cached per table version;
    a query pays one matmul + searchsorted per segment. The Python-dict
    scan this replaces cost ~70 ms per directional query at 50k recordings
    (measured at the 100 h scale test)."""
    from .embeddings import _vec_table
    ver = store.table("motion_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _MOT_IDX:
        tbl, _ = _vec_table(store, "motion_vectors")
        ss = np.asarray(tbl.column("stream").to_pylist())
        sa = np.asarray([int(v) for v in tbl.column("ts").to_pylist()])
        sb = np.asarray([int(v) for v in tbl.column("t1").to_pylist()])
        idx = {}
        for s in np.unique(ss):
            m = np.where(ss == s)[0]
            o = np.argsort(sa[m], kind="stable")
            idx[s] = (sa[m][o], sb[m][o], m[o])
        if len(_MOT_IDX) > 8:
            _MOT_IDX.clear()
        _MOT_IDX[key] = idx
    idx = _MOT_IDX[key]
    _, vecs = _vec_table(store, "motion_vectors")
    qd = np.asarray(qv, np.float32) - np.asarray(qv_swap, np.float32)
    n = float(np.linalg.norm(qd))
    if n < 1e-6:
        return lambda s, a, b: float("nan")
    sc = vecs @ (qd / n)

    def lookup(s, a, b):
        if s not in idx:
            return float("nan")
        t0s, t1s, rows = idx[s]
        j = int(np.searchsorted(t0s, a, side="right")) - 1
        if j >= 0 and b <= int(t1s[j]) + 1:
            return float(sc[rows[j]])
        return float("nan")
    return lookup


def motion_candidates(store, qv, qv_swap, streams, t0s, top=24):
    """DIRECTION-FIRST RECALL, content-gated. Given a broad appearance-
    plausible window set (their streams + start times), return the top
    recordings ranked by motion score as (stream, rec_t0, rec_t1, score).

    Why the gate: unrestricted motion recall was measured useless — the
    tiny direction cosines (~0.07) drown corpus-wide in random tabletop
    deltas. Why recall at all: at the 100 h scale, appearance ordering
    alone never surfaces true close/open recordings into the candidate
    pool (put-in clips carry a stronger 'drawer' signal), so ranking-only
    motion had nothing correct to rank. Appearance answers WHAT is
    plausible; motion picks WHICH of those changed the right way."""
    from .embeddings import _vec_table
    ver = store.table("motion_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _MOT_IDX:
        motion_lookup(store, qv, qv_swap)      # builds the cache
    idx = _MOT_IDX[key]
    _, vecs = _vec_table(store, "motion_vectors")
    qd = np.asarray(qv, np.float32) - np.asarray(qv_swap, np.float32)
    n = float(np.linalg.norm(qd))
    if n < 1e-6:
        return []
    sc = vecs @ (qd / n)

    streams = np.asarray(streams)
    t0s = np.asarray(t0s)
    hit_rows = set()
    for s in np.unique(streams):
        if s not in idx:
            continue
        r0, r1, rows = idx[s]
        w = t0s[streams == s]
        j = np.searchsorted(r0, w, side="right") - 1
        ok = j >= 0
        hit_rows.update(int(r) for r in np.unique(rows[j[ok]]))
    if not hit_rows:
        return []
    hits = sorted(hit_rows, key=lambda r: -sc[r])[:top]
    tbl, _ = _vec_table(store, "motion_vectors")
    ss = tbl.column("stream")
    sa = tbl.column("ts")
    sb = tbl.column("t1")
    return [(ss[r].as_py(), int(sa[r].as_py()), int(sb[r].as_py()),
             float(sc[r])) for r in hits]
