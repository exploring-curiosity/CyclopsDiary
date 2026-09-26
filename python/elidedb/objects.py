"""The object channel: every subject in play, discovered — never named.

User-set design rules, enforced here:
  - NOTHING predefined. FastSAM runs in prompt-free segment-everything
    mode: the frame goes in, class-agnostic region masks come out. No
    text prompts, no class lists, no dataset words — a hard rule, because
    this is a general store where anyone uploads anything.
  - NESTED sub-subjects. SAM proposes masks at several granularities
    (a person AND their shirt AND their hat). Containment builds the
    hierarchy: region A is a child of B when most of A's box lies inside
    B's (IoA > 0.75). Children are kept as their own rows — a query about
    the shirt matches the shirt crop, not the whole person.
  - MOTION integrated per subject. Each region carries a motion score:
    mean |pixel delta| inside its box between the keyframe and a nearby
    frame, normalized per frame. The moving subject is the one acting.

Each region crop is embedded with the SAME SigLIP space queries live in,
so retrieval is one matmul: score(query, event) = max over that event's
region embeddings. This is the multi-vector layout at the OBJECT level —
"cloth" matches the cloth crop even when the scene is cluttered.
"""
from __future__ import annotations

import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

_FSAM = {}


def _fastsam():
    if "m" not in _FSAM:
        from ultralytics import FastSAM
        _FSAM["m"] = FastSAM("FastSAM-s.pt")
    return _FSAM["m"]


def _regions(frame_u8, max_regions=24, min_area=0.002, conf=0.4):
    """Prompt-free region proposals -> [(box xyxy, area_frac, parent)].
    Parent index (-1 = top-level) comes from box containment (IoA)."""
    m = _fastsam()
    H, W = frame_u8.shape[:2]
    res = m(frame_u8, device="mps", retina_masks=False, imgsz=448,
            conf=conf, iou=0.9, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return []
    boxes = res[0].boxes.xyxy.cpu().numpy()
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
             / (H * W))
    keep = np.where((areas > min_area) & (areas < 0.9))[0]
    keep = keep[np.argsort(-areas[keep])][:max_regions]
    boxes, areas = boxes[keep], areas[keep]
    parents = np.full(len(boxes), -1, np.int32)
    for i in range(len(boxes)):          # sorted big->small: parents first
        xi0, yi0, xi1, yi1 = boxes[i]
        ai = max((xi1 - xi0) * (yi1 - yi0), 1e-6)
        for j in range(i):               # smallest enclosing earlier box
            xj0, yj0, xj1, yj1 = boxes[j]
            ix = max(0, min(xi1, xj1) - max(xi0, xj0))
            iy = max(0, min(yi1, yj1) - max(yi0, yj0))
            if ix * iy / ai > 0.75:
                parents[i] = j
        # keep the SMALLEST enclosing parent (deepest nesting)
        best = -1
        for j in range(i):
            xj0, yj0, xj1, yj1 = boxes[j]
            ix = max(0, min(xi1, xj1) - max(xi0, xj0))
            iy = max(0, min(yi1, yj1) - max(yi0, yj0))
            if ix * iy / ai > 0.75 and (
                    best < 0 or areas[j] < areas[best]):
                best = j
        parents[i] = best
    return [(boxes[i], float(areas[i]), int(parents[i]))
            for i in range(len(boxes))]


def _motion_energy(frame_a, frame_b, box):
    x0, y0, x1, y1 = (int(v) for v in box)
    a = frame_a[y0:y1, x0:x1].astype(np.float32)
    b = frame_b[y0:y1, x0:x1].astype(np.float32)
    if a.size == 0 or a.shape != b.shape:
        return 0.0
    return float(np.abs(a - b).mean() / 255.0)


def index_objects(store, keyframes_per_recording=2, max_regions=24,
                  verbose=True):
    """Materialize `object_vectors`: one row per discovered region.
    Columns: ts/t1 (recording span), stream, vector (SigLIP crop), area,
    parent (row offset within the same keyframe, -1 top-level), motion."""
    from PIL import Image

    from .embeddings import _embed_images
    from .video import FrameSet
    t_start = time.time()

    try:
        ep = store.table("episodes").scan()
        recs = list(zip(ep.column("stream").to_pylist(),
                        (int(v) for v in ep.column("ts").to_pylist()),
                        (int(v) for v in ep.column("t1").to_pylist())))
    except Exception:
        recs = []
    frames_tbl = store.table("frames").scan()
    if not recs:
        for s in sorted(set(frames_tbl.column("stream").to_pylist())):
            sel = frames_tbl.filter(pc.equal(frames_tbl.column("stream"), s))
            ts = sorted(int(v) for v in sel.column("ts").to_pylist())
            recs.append((s, ts[0], ts[-1]))

    rows = {"ts": [], "t1": [], "stream": [], "area": [], "parent": [],
            "motion": [], "kf": []}
    crops = []
    n_frames = 0
    for ri, (s, a, b) in enumerate(recs):
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 3:
            continue
        pick = np.linspace(0, len(sel) - 2,
                           min(keyframes_per_recording, len(sel) - 1)) \
            .round().astype(int)
        # decode each keyframe AND its next frame for motion energy
        want = sorted({int(p) for p in pick} | {int(p) + 1 for p in pick})
        dec = FrameSet(store, "frames",
                       sel.take(np.array(want))).decode(width=448)
        if len(dec) < 2:
            continue
        by_pos = {want[i]: d[1] for i, d in enumerate(sorted(dec))
                  if i < len(want)}
        n_frames += len(pick)
        for p in pick:
            fr, nxt = by_pos.get(int(p)), by_pos.get(int(p) + 1)
            if fr is None:
                continue
            base_row = len(rows["ts"])
            for box, area, parent in _regions(fr, max_regions=max_regions):
                x0, y0, x1, y1 = (int(v) for v in box)
                crops.append(Image.fromarray(fr[y0:y1, x0:x1]))
                rows["ts"].append(a)
                rows["t1"].append(b)
                rows["stream"].append(s)
                rows["area"].append(area)
                rows["parent"].append(parent if parent < 0
                                      else base_row + parent)
                rows["motion"].append(
                    _motion_energy(fr, nxt, box) if nxt is not None else 0.0)
                rows["kf"].append(int(p))
        if verbose and (ri + 1) % 200 == 0:
            print(f"  {ri + 1}/{len(recs)} recordings, "
                  f"{len(crops)} regions (t={time.time() - t_start:.0f}s)",
                  flush=True)
    if not crops:
        return {"regions": 0}

    # crops MUST live in the same space as the query text tower
    # (DEFAULT_MODEL) — the 'fast' 224 checkpoint is a different space and
    # its cosines would be meaningless against q_full
    from .embeddings import DEFAULT_MODEL
    vecs = []
    B = 64
    for i in range(0, len(crops), B):
        vecs.append(_embed_images(crops[i:i + B], DEFAULT_MODEL))
    vecs = np.concatenate(vecs).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8

    dim = vecs.shape[1]
    order = np.argsort(np.array(rows["ts"]), kind="stable")
    flat = np.ascontiguousarray(vecs[order]).reshape(-1)
    tbl = pa.table({
        "ts": pa.array([rows["ts"][i] for i in order], pa.int64()),
        "t1": pa.array([rows["t1"][i] for i in order], pa.int64()),
        "stream": pa.array([rows["stream"][i] for i in order]),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(flat), dim),
        "area": pa.array([rows["area"][i] for i in order], pa.float32()),
        "parent": pa.array([rows["parent"][i] for i in order], pa.int32()),
        "motion": pa.array([rows["motion"][i] for i in order],
                           pa.float32()),
    })
    tab = store.table("object_vectors")
    meta = {"model": "fastsam-s + siglip crops", "dim": dim,
            "keyframes_per_recording": keyframes_per_recording,
            "max_regions": max_regions, "nested": True}
    try:
        existing = tab.state().files
    except Exception:
        existing = []
    if existing:
        from .log import FileEntry
        from .store import write_parquet
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
    return {"regions": len(tbl), "recordings": len(recs),
            "keyframes": n_frames, "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 1)}


_OBJ_IDX = {}


def object_lookup(store, qv):
    """lookup(stream, t0, t1) -> (best region cosine, its motion score)
    for the recording containing the span; (nan, 0) when absent.
    Per-stream span index cached per version; one matmul per query."""
    from .embeddings import _vec_table
    ver = store.table("object_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _OBJ_IDX:
        tbl, _ = _vec_table(store, "object_vectors")
        ss = np.asarray(tbl.column("stream").to_pylist())
        sa = np.asarray([int(v) for v in tbl.column("ts").to_pylist()])
        sb = np.asarray([int(v) for v in tbl.column("t1").to_pylist()])
        mo = np.asarray(tbl.column("motion").to_pylist(), np.float32)
        idx = {}
        for s in np.unique(ss):
            m = np.where(ss == s)[0]
            o = np.argsort(sa[m], kind="stable")
            idx[s] = (sa[m][o], sb[m][o], m[o])
        if len(_OBJ_IDX) > 8:
            _OBJ_IDX.clear()
        _OBJ_IDX[key] = (idx, mo)
    idx, mo = _OBJ_IDX[key]
    _, vecs = _vec_table(store, "object_vectors")
    # CONJUNCTION OVER REGIONS — the channel's whole point. `qv` may be a
    # matrix of clause/atom embeddings: a compound sentence scored against
    # a small crop is a bag-of-concepts mismatch (measured: green-binding
    # unmoved with whole-sentence scoring), but per-atom max over regions
    # then soft-AND across atoms asks the right question — does this
    # recording contain a green-toy region AND a drawer region?
    Q = np.atleast_2d(np.asarray(qv, np.float32))
    S = vecs @ Q.T                                   # (rows, atoms)

    def lookup(s, a, b):
        if s not in idx:
            return float("nan"), 0.0
        t0s, t1s, rows = idx[s]
        lo = int(np.searchsorted(t0s, a, side="right"))
        j0 = lo - 1
        if j0 < 0 or b > int(t1s[j0]) + 1:
            return float("nan"), 0.0
        # all rows of this recording share (t0, t1): walk the run
        j = j0
        while j >= 0 and t0s[j] == t0s[j0]:
            j -= 1
        run = rows[j + 1:lo]
        if len(run) == 0:
            return float("nan"), 0.0
        per_atom = S[run].max(axis=0)                # best region per atom
        which = int(np.argmax(S[run][:, 0]))
        # CONJUNCTION = the WEAKEST required object. The mean let one
        # strong crop carry a recording whose other object barely matched
        # (attributed live: 'a vessel' matched nothing, so any fork with a
        # good 'lid'-ish crop won; drawer crops are everywhere, so 'green'
        # never had to be real). min() makes every named object earn it.
        return float(per_atom.min()), float(mo[run[which]])
    return lookup


def object_candidates(store, qv, top=24):
    """Top recordings by best motion-weighted region match — the object
    channel's own recall: a small matching crop finds a recording whose
    whole-frame embedding never would."""
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "object_vectors")
    ver = store.table("object_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _OBJ_IDX:
        object_lookup(store, qv)           # builds the cache
    idx, mo = _OBJ_IDX[key]
    Q = np.atleast_2d(np.asarray(qv, np.float32))
    S = (vecs @ Q.T) * (1.0 + mo)[:, None]
    # per-recording soft-AND: mean over atoms of best region per atom
    ss = tbl.column("stream").to_pylist()
    sa = tbl.column("ts").to_pylist()
    sb = tbl.column("t1").to_pylist()
    rec_score = {}
    for s, (t0s, t1s, rows) in idx.items():
        start = 0
        while start < len(rows):
            end = start
            while end < len(rows) and t0s[end] == t0s[start]:
                end += 1
            run = rows[start:end]
            r0 = int(run[0])
            rec_score[(s, int(sa[r0]), int(sb[r0]))] = \
                float(S[run].max(axis=0).min())
            start = end
    out = sorted(rec_score.items(), key=lambda kv: -kv[1])[:top]
    return [(*k, v) for k, v in out]


def bind_lookup(store, qv_mover, qv_landmark):
    """ROLE-AWARE binding: does the MOVER move and the LANDMARK stay?

    The obj channel asks "are both objects present" (min over atoms).
    That is why "put the eggplant into the drawer" scores every episode
    holding an eggplant and a drawer, and why the binding queries sit at
    ~11% precision while direction queries reach 86%.

    A relation has ROLES. In "put X into Y", X is the theme and moves; Y
    is the landmark and largely does not. That asymmetry is already in
    the store - object_vectors carries a per-crop `motion` scalar - it
    was simply never used to tell the two roles apart. This scores:

        min(sim_X, sim_Y)              both objects must really be there
      * relu(motion_X - motion_Y)      and X must be the one that moved

    so an episode where the drawer moves and the eggplant sits still
    scores zero, which is exactly the confusion the flat conjunction
    could not express. No new ingest: pure arithmetic over crops that
    are already embedded.

    MEASURED 2026-07-28 - THIS DOES NOT WORK, AND THE REASON MATTERS.
    Wired as a channel it left the binding queries where it found them
    (q09 best true rank 799 of 1122, q10 rank 441). The role logic is
    not what fails: crop-level object identification is at noise. Across
    all 51,101 crops the best cosine to "an eggplant" is 0.184 (mean
    0.048), and in the two episodes that genuinely contain one the best
    crop scores 0.147 and 0.161 - ranked 110th and 75th corpus-wide. The
    detector cannot find the object, so nothing built on top of it can
    bind it. Fixing binding needs better object GROUNDING (higher-res
    crops, or an open-vocabulary detector), not better relation
    reasoning. Kept, unwired, as the measurement behind that claim.

    Returns lookup(stream, t0, t1) -> float (nan when the recording has
    no crops)."""
    from .embeddings import _vec_table
    ver = store.table("object_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _OBJ_IDX:
        object_lookup(store, np.atleast_2d(np.asarray(qv_mover, np.float32)))
    idx, mo = _OBJ_IDX[key]
    _, vecs = _vec_table(store, "object_vectors")
    m = np.asarray(qv_mover, np.float32).reshape(-1)
    l = np.asarray(qv_landmark, np.float32).reshape(-1)
    m /= np.linalg.norm(m) + 1e-8
    l /= np.linalg.norm(l) + 1e-8
    Sm = vecs @ m
    Sl = vecs @ l

    def lookup(s, a, b):
        if s not in idx:
            return float("nan")
        t0s, t1s, rows = idx[s]
        lo = int(np.searchsorted(t0s, a, side="right"))
        j0 = lo - 1
        if j0 < 0 or b > int(t1s[j0]) + 1:
            return float("nan")
        j = j0
        while j >= 0 and t0s[j] == t0s[j0]:
            j -= 1
        run = rows[j + 1:lo]
        if len(run) == 0:
            return float("nan")
        im = int(np.argmax(Sm[run]))
        il = int(np.argmax(Sl[run]))
        if im == il:
            return 0.0          # one crop cannot play both roles
        present = float(min(Sm[run][im], Sl[run][il]))
        # the mover must out-move the landmark; ties and reversals are
        # evidence AGAINST the relation, not neutral
        delta = float(mo[run[im]] - mo[run[il]])
        return present * max(delta, 0.0)
    return lookup
