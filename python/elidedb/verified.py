"""Verified search: any query, action or not, on any store.

WHY THIS SHAPE (from the measured root cause)
---------------------------------------------
Verbs are erased at both ends of embedding retrieval: the text tower scores
cos("close the drawer","open the drawer")=0.951, and even the full teacher's
frame embeddings separate close/open/put-in at 0.60 — a coin flip. So no
amount of embedding work can enforce "the green object is the SUBJECT of the
action" or "and close it". The only generic component in the system that can
read an EVENT is a VLM shown frames in time order.

The database answer is the same as ever: cheap operators propose, the
expensive operator disposes, and its work is cached.

    RECALL  (cheap, high recall on purpose)
        union of appearance rankings — the full query plus each comma/AND
        atom ("green object", "the drawer"), so a clip that is strong on any
        ingredient reaches the pool. If VLM captions exist in the store, the
        lexical ranking joins the union (verbs live in words).
        The union fixes what killed sharp search: one weak ranker never
        gates the candidate set.

    VERIFY  (expensive, precise, LAST)
        4 frames spanning each candidate segment, in time order, judged by
        the VLM with the whole sentence: logP(yes)-logP(no). Subject-of-
        action and sequential clauses ("...and close it") are enforced here
        — the one place they exist.

    CACHE   (cracking)
        every verdict is written to `vlm_verdicts` keyed by (window, query
        hash). Repeats and refinements of hot queries get cheaper; cold
        corpus costs nothing.

Nothing here is dataset-specific: no object lists, no robot fields, no
task vocabulary. The prompts are the user's own words.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

_VERDICTS: dict = {}
# Background verification: one worker at a time, one job per (store, query).
# The QUERY never waits for a model — verification is index MAINTENANCE that
# happens to be triggered by traffic (cracking without query-time latency).
_BG_LOCK = threading.Lock()
_BG_INFLIGHT: set = set()


def _qhash(text: str) -> int:
    return int.from_bytes(
        hashlib.sha1(text.strip().lower().encode()).digest()[:8],
        "little", signed=True)


def _verdict_map(store):
    t = store.table("vlm_verdicts")
    st = t.state()
    key = (str(store.dir), st.version)
    if key not in _VERDICTS:
        m = {}
        if st.files:
            tb = t.scan()
            for s, a, q, v in zip(tb.column("stream").to_pylist(),
                                  tb.column("ts").to_pylist(),
                                  tb.column("qhash").to_pylist(),
                                  tb.column("margin").to_pylist()):
                m[(s, int(a), int(q))] = float(v)
        _VERDICTS.clear()
        _VERDICTS[key] = m
    return _VERDICTS[key]


def _atoms(text: str):
    """The query plus its ingredients. 'put a green object in the drawer and
    close it' -> the full sentence, 'put a green object in the drawer',
    'close it'. Each is embedded separately for RECALL only — precision is
    the verifier's job, so over-splitting costs nothing but candidates."""
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r",|\band then\b|\bthen\b|\band\b|\bAND\b", t)
    atoms = [p.strip() for p in parts if len(p.strip().split()) >= 2]
    return [t] + [a for a in atoms if a.lower() != t.lower()]


_CAP_CACHE = {}


def _text_candidates(store, text, n, table, column):
    """Lexical recall over any per-span TEXT table (VLM captions, or
    uploader metadata when a store carries it). Verbs live in words, so
    this is the ranker most likely to surface action matches.

    The fitted TF-IDF + document matrix are cached per table VERSION —
    refitting on every query was 50 ms of an 81 ms query (measured), i.e.
    the entire latency gate spent recomputing something that only changes
    when rows are appended. A query pays one sparse transform."""
    try:
        ver = store.table(table).state().version
    except Exception:
        return []
    key = (str(store.dir), table, ver)
    if key not in _CAP_CACHE:
        caps = store.table(table).scan()
        if len(caps) == 0 or column not in caps.column_names:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [t or "" for t in caps.column(column).to_pylist()]
        # char n-grams instead of word tokens: 'closes'/'closing'/'close'
        # share their subword mass, so inflections meet without a stemmer —
        # measured on the regression set, word-token TF-IDF scored 2/10 on
        # a query whose labels used a different inflection of its verb
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                              sublinear_tf=True)
        X = vec.fit_transform(texts)
        wins = list(zip(caps.column("stream").to_pylist(),
                        (int(v) for v in caps.column("ts").to_pylist()),
                        (int(v) for v in caps.column("t1").to_pylist())))
        if len(_CAP_CACHE) > 8:
            _CAP_CACHE.clear()
        _CAP_CACHE[key] = (vec, X, wins)
    vec, X, wins = _CAP_CACHE[key]
    q = vec.transform([text])
    sc = np.asarray((X @ q.T).todense()).ravel()
    order = np.argsort(-sc)[:n]
    return [(wins[int(i)], float(sc[i])) for i in order if sc[i] > 0]


def _caption_candidates(store, text, n):
    return _text_candidates(store, text, n, "context_captions", "caption")


_ADAPTER_CACHE = {}
_REC_CACHE = {}


def _recording_spans(store):
    """Per-stream recording boundaries from the `episodes` table, when the
    store has one. STRUCTURAL ingest metadata — where one recording ends
    and the next begins, the same class of fact as a file boundary; no task
    labels involved. Needed because packed corpora leave NO timeline gap
    between recordings (measured: max frame dt 0.2 s across episode cuts),
    so time alone cannot see the seam."""
    try:
        ver = store.table("episodes").state().version
    except Exception:
        return {}
    key = (str(store.dir), ver)
    if key not in _REC_CACHE:
        ep = store.table("episodes").scan()
        if len(ep) == 0 or not {"stream", "ts", "t1"} <= set(ep.schema.names):
            _REC_CACHE[key] = {}
            return {}
        spans = {}
        for s, a, b in zip(ep.column("stream").to_pylist(),
                           ep.column("ts").to_pylist(),
                           ep.column("t1").to_pylist()):
            spans.setdefault(s, []).append((int(a), int(b)))
        out = {}
        for s, lst in spans.items():
            lst.sort()
            out[s] = (np.array([a for a, _ in lst], np.int64),
                      np.array([b for _, b in lst], np.int64))
        if len(_REC_CACHE) > 8:
            _REC_CACHE.clear()
        _REC_CACHE[key] = out
    return _REC_CACHE[key]


def _ctx_event_scores(store, qv):
    """(stream, t0, t1, ctx-cosine) for every event in the V2 index.

    The adapter is the query-time half of stage B's contrastive pair: `qv`
    is the SigLIP text embedding the appearance ranker ALREADY computed
    (verified identical to the tower stage B trained with, cos 1.0); the
    adapter maps it to the 256-d ctx space. Loaded once per store, then a
    query costs one 2-layer forward + one matmul — no second tower pass.
    """
    from .embeddings import _vec_table
    tbl, vecs = _vec_table(store, "context_events")
    key = str(store.dir)
    if key not in _ADAPTER_CACHE:
        import mlx.core as mx
        from .fdnnv2 import TextAdapter
        from mlx.utils import tree_unflatten
        z = np.load(Path(store.dir) / "models" / "fdnnv2" / "adapter.npz")
        adapter = TextAdapter()
        adapter.update(tree_unflatten([(k_, mx.array(z[k_]))
                                       for k_ in z.files]))
        mx.eval(adapter.parameters())
        _ADAPTER_CACHE[key] = adapter
    import mlx.core as mx
    qc = np.array(_ADAPTER_CACHE[key](mx.array(qv)[None]))[0]
    sc = vecs @ qc
    ss = tbl.column("stream").to_pylist()
    sa = tbl.column("ts").to_pylist()
    sb = tbl.column("t1").to_pylist()
    return [((s, int(a), int(b)), float(c))
            for s, a, b, c in zip(ss, sa, sb, sc)]


def search_verified(store, text, k=8, pool=48, frames_per_clip=2,
                    pad_s=2.0, deep=0, t0=None, t1=None, streams=None,
                    verify="async", verbose=False):
    """Union recall -> segment building -> multi-frame VLM verification.

    Returns (hits, stats); each hit carries `margin` (the VLM's yes/no
    log-odds for the WHOLE event) and `cached` (whether the verdict was
    already in the store).
    """
    from PIL import Image

    from .embeddings import _vec_table
    from .rerank import (DEEP_VLM, as_change_question, as_clip_question,
                         score_clip_sequences)
    from .video import FrameSet
    t_start = time.perf_counter()

    tbl, vecs = _vec_table(store, "embeddings")
    w_t0 = tbl.column("ts").to_numpy()
    w_t1 = tbl.column("t1").to_numpy()
    # stores written before the multi-stream schema have no `stream` column
    # on embeddings; a single-stream store can borrow the frames stream —
    # a KeyError here bricked search on every pre-schema store (measured
    # on lab/oxford during pilot prep)
    if "stream" in tbl.schema.names:
        w_s = tbl.column("stream").to_numpy(zero_copy_only=False)
    else:
        try:
            fs = store.table("frames").scan(columns=["stream"])
            uniq = set(fs.column("stream").to_pylist())
            only = uniq.pop() if len(uniq) == 1 else ""
        except Exception:
            only = ""
        w_s = np.array([only] * len(tbl))

    # hybrid predicates pushed INTO recall, as everywhere else in this store
    pred = np.ones(len(w_t0), bool)
    if t0 is not None:
        pred &= w_t1 >= t0
    if t1 is not None:
        pred &= w_t0 <= t1
    if streams:
        pred &= np.isin(w_s, list(streams))
    idx_all = np.where(pred)[0]
    if len(idx_all) == 0:
        return [], {"method": "verified", "candidates": 0, "segments": 0,
                    "ms": 0.0}

    # ---- RECALL: union over atoms + captions + context vectors ------------
    # The recall budget SCALES with corpus size: 48 candidates out of 7k
    # windows is a 0.7% sample and sees everything worth seeing; 48 out of
    # 180k is 0.03% and starves (measured at the 100 h scale test: verb
    # rank-1s buried under 25x more distractors, verb-strict 10/20 -> 4/20).
    # The VLM budget (`pool`) stays fixed — only the index-side pool grows,
    # and index candidates cost microseconds each.
    atoms = _atoms(text)
    # floor raised to 120: weight-fitting exposed pools where a PERFECT
    # ranker could not reach 10 relevant hits (2 relevant of 48 candidates
    # for an attribute query). Index candidates cost microseconds; recall
    # starvation costs correctness that nothing downstream can repair.
    eff_pool = max(pool, min(max(len(idx_all) // 1500, 120), 256))
    per = max(eff_pool // len(atoms), 12)
    cand = {}
    # ONE tower pass for the query, its atoms, AND the directional swap:
    # per-text embed_text calls cost ~21 ms EACH, which put every compound
    # query over the latency gate (measured 84-106 ms); one batch costs the
    # same as one text.
    from .context import embed_texts
    from .rerank import directional_swap
    sq = directional_swap(text, store)
    # SUBJECT ANCHORING, self-observed: SigLIP only lands near the right
    # images when the corpus's dominant subject is NAMED in the query
    # (measured: "folding the cloth" ranks the true clip #537 bare, #17
    # with "a robot" named — and nine content-blind templates all failed).
    # The subject comes from the store LOOKING AT ITS OWN FRAMES
    # (subjects.py) — never from metadata, keywords, or anything the
    # uploader said. The anchored query is its OWN RRF channel, not a
    # score edit: score-level fusion was measured to trade one query
    # class for another (max: fold 0->12 but close 10->1), while rank
    # consensus lets each channel carry the queries it is good at.
    try:
        from .subjects import subject_prefixes
        prefixes = subject_prefixes(store, build=False)[:1]
    except Exception:
        prefixes = []
    texts_all = list(atoms) + [f"{p_} {text}" for p_ in prefixes]
    qvs = embed_texts(texts_all + ([sq] if sq else []))
    q_full = qvs[0]
    qv_swap = qvs[-1] if sq else None
    qv_anchor = qvs[len(atoms)] if prefixes else None
    qvs = qvs[:len(atoms)]
    # appearance score = MEAN over clause cosines (soft-AND), not the
    # full-sentence cosine alone. Measured: for "pick up a green toy and
    # put it in the drawer" the full-sentence cosine ranked burner/spoon
    # clips top-5 (a compound sentence matches everything and nothing in
    # a bag-of-concepts space), while the clauses individually rank green
    # things and drawer scenes correctly — a clip must score on EVERY
    # clause to rank. The atom embeddings are already paid for by recall.
    # vecs is a memory-map; fancy-indexing it materializes a full copy, so
    # only do that when a predicate actually filtered rows — and ONCE, not
    # per atom (the per-atom form copied 0.83 GB x atoms at pilot scale)
    sub = vecs if len(idx_all) == len(vecs) else vecs[idx_all]
    A = np.stack([sub @ qv for qv in qvs])
    app_all = A.mean(axis=0)
    app_of = {int(i): float(sc) for i, sc in zip(idx_all, app_all)}
    touched = set()
    for row in A:
        top = idx_all[np.argsort(-row)[:per]]
        touched.update(int(i) for i in top)
        for i in top:
            cand.setdefault((str(w_s[i]), int(w_t0[i]), int(w_t1[i])),
                            app_of.get(int(i), 0.0))
    # anchored channel: proposes its own candidates and scores every
    # window-derived candidate (abstains on caption/motion spans). Both
    # restrictions were measured: promote-only loses the fold recall
    # entirely (0/48 in pool), scoring-all keeps it (2/48 -> verified
    # top-5 3/5 fold) at a 2-3 point cold-battery cost inside that
    # metric's own noise band — and the verify tier recovers those at
    # re-ask, while nothing recovers a clip that never enters the pool.
    anc_of = {}
    if qv_anchor is not None:
        anc_all = sub @ qv_anchor
        touched.update(int(i) for i in
                       idx_all[np.argsort(-anc_all)[:per]])
        for i in touched:
            j = int(np.searchsorted(idx_all, i))
            key_ = (str(w_s[i]), int(w_t0[i]), int(w_t1[i]))
            anc_of[key_] = float(anc_all[j])
            cand.setdefault(key_, app_of.get(int(i), 0.0))
    # lexical scores are KEPT, not just used for recall: a caption that
    # says the verb is the strongest index evidence this store has for an
    # action query, and dropping its score buried caption hits at the
    # bottom of the unverified ordering (measured on the verb battery).
    # CAPTIONING IS DEPRECATED (user directive: video-native only). The
    # caption channel is fully disconnected — no recall, no ranking —
    # even where legacy caption tables exist. `met` below is different
    # machinery: uploader-provided labels, search-time only, only on
    # stores that carry them.
    lex_of = {}
    # UPLOADER METADATA channel — search-time only, and only for stores
    # that carry a `meta_text` table by the uploader's choice. The
    # no-metadata rule still binds training and indexing; refusing
    # customer-provided labels at SEARCH time was just leaving quality on
    # the table (the Daft-stack comparison). Control store has none.
    met_of = {}
    for w, sc in _text_candidates(store, text, per, "meta_text", "text"):
        if streams and w[0] not in streams:
            continue
        cand.setdefault(w, 0.0)
        met_of[w] = max(met_of.get(w, 0.0), sc)
    # stores with the full context tier contribute their caption-LSA ranking
    try:
        from .context import _ctx_matrix, caption_space
        if store.table("context").state().files:
            ctx = store.table("context").scan()
            cv = _ctx_matrix(store)
            qc = caption_space(store).transform([text])[0]
            cs = ctx.column("stream").to_pylist()
            ca = ctx.column("ts").to_pylist()
            cb = ctx.column("t1").to_pylist()
            for i in np.argsort(-(cv @ qc))[:per]:
                if streams and cs[i] not in streams:
                    continue
                cand.setdefault((cs[i], int(ca[i]), int(cb[i])), 0.0)
    except Exception:
        pass
    # stores with a V2 context-event index contribute VERB-AWARE ranking:
    # adapter(text) and novelty-pooled event vectors share the 256-d space
    # stage B trained, where "close" and "open" are different directions —
    # the one thing appearance cosines cannot express. Index-only: one
    # matmul plus a 2-layer adapter (~0.1 ms), no VLM.
    ctx_of = {}
    try:
        ev_scores = _ctx_event_scores(store, q_full)
        for (s_, a, b), sc in ev_scores:
            if streams and s_ not in streams:
                continue
            if t0 is not None and b < t0:
                continue
            if t1 is not None and a > t1:
                continue
            ctx_of[(s_, a, b)] = sc
        for w, sc in sorted(ctx_of.items(), key=lambda kv: -kv[1])[:per]:
            cand.setdefault(w, 0.0)
    except Exception:
        pass
    # MOTION channel — the Marengo multi-vector lesson, ElideDB-style:
    # motion = delta-appearance per RECORDING vs (query − swap) direction.
    # Measured: close/open direction AUC 0.98 among drawer recordings —
    # but corpus-wide the tiny direction cosines (~0.07) drown in random
    # tabletop deltas, so this channel answers DIRECTION, not content: it
    # never proposes candidates, it only scores what the content channels
    # (appearance/lexical/ctx) surfaced. Index-only, no model call.
    # OBJECT channel — every subject FastSAM discovered in the frames
    # (nothing predefined, nested sub-subjects, per-region motion). Score
    # of a span = best region cosine in its recording, nudged by that
    # region's motion energy: the moving subject is the one acting.
    # VID channel — the adopted natively co-trained video-text encoder
    # (X-CLIP-large; relational block 1/40 -> 10/40 vs the appearance
    # incumbent on the balanced-sample evaluation). Its OWN text tower
    # (channel-local space; rank fusion makes cross-space fusion legal),
    # one cached forward per query text.
    vid_look = None
    vid_of = {}
    try:
        from .vid import vid_lookup
        vid_look, vid_cands = vid_lookup(store, text)
        for s_, a, b, sc in vid_cands[:per]:
            if streams and s_ not in streams:
                continue
            if t0 is not None and b < t0:
                continue
            if t1 is not None and a > t1:
                continue
            cand.setdefault((s_, a, b), 0.0)
    except Exception:
        pass

    # PE channel — Meta Perception Encoder (published SOTA zero-shot
    # video-text retrieval; best single-encoder p@10 measured here).
    pe_look = None
    try:
        from .pe import pe_lookup
        pe_look, pe_cands = pe_lookup(store, text)
        for s_, a, b, _sc in pe_cands[:per]:
            if streams and s_ not in streams:
                continue
            if t0 is not None and b < t0:
                continue
            if t1 is not None and a > t1:
                continue
            cand.setdefault((s_, a, b), 0.0)
    except Exception:
        pass

    # ACT channel — SSv2 action posteriors (V-JEPA 2 + Meta probe),
    # video-native WHAT-HAPPENED evidence, index-only at query time.
    # Adopted on measurement: put-in vs take-out AUC 0.889 zero-shot —
    # the containment direction no other channel sees.
    act_look = None
    try:
        from .action_channel import act_lookup
        act_look, act_cands = act_lookup(store, text)
        for s_, a, b, _sc in act_cands[:per]:
            if streams and s_ not in streams:
                continue
            if t0 is not None and b < t0:
                continue
            if t1 is not None and a > t1:
                continue
            cand.setdefault((s_, a, b), 0.0)
    except Exception:
        pass

    obj_lookup = None
    try:
        from .objects import object_candidates, object_lookup
        # crops are STATIC objects — the verb in a clause pollutes crop
        # matching (measured: green-binding unmoved with clause atoms).
        # Strip leading verb tokens mechanically; keep the noun phrase.
        # crops are STATIC objects: match them against the query's
        # DETERMINER PHRASES ("a green object", "the drawer"), extracted
        # mechanically — regression-measured: full sentences against small
        # crops scored drawer clips #1 for "the robot moves a green
        # object". Verb-led queries with no noun phrase keep the sentence.
        import re as _re
        _TAIL = {"and", "then", "in", "on", "to", "into", "onto", "of",
                 "from", "at", "with", "it", "is", "robot", "arm"}
        nps = []
        for m in _re.finditer(
                r"\b(?:a|an|the)\s+(?:\w+\s+){0,2}\w+", text.lower()):
            w = m.group(0).split()
            while len(w) > 1 and w[-1] in _TAIL:
                w.pop()
            if len(w) > 1:
                nps.append(" ".join(w))
        np_atoms = nps if nps else list(atoms)
        qvs_obj = (embed_texts(np_atoms)
                   if np_atoms != list(atoms) else qvs)
        obj_lookup = object_lookup(store, qvs_obj)      # per-atom matrix
        for s_, a, b, _sc in object_candidates(store, qvs_obj, top=per):
            if streams and s_ not in streams:
                continue
            if t0 is not None and b < t0:
                continue
            if t1 is not None and a > t1:
                continue
            cand.setdefault((s_, a, b), 0.0)
    except Exception:
        pass

    mot_lookup = None
    if qv_swap is not None:
        try:
            from .motion import motion_candidates, motion_lookup
            mot_lookup = motion_lookup(store, q_full, qv_swap)
            # direction-first recall: appearance gates a broad plausible
            # set, motion picks which of those recordings changed the
            # right way (see motion_candidates for the two measurements
            # that force this split)
            wide = idx_all[np.argsort(-app_all)[:min(4000, len(idx_all))]]
            for s_, a, b, sc in motion_candidates(
                    store, q_full, qv_swap,
                    [str(w_s[i]) for i in wide],
                    [int(w_t0[i]) for i in wide], top=per):
                if streams and s_ not in streams:
                    continue
                if t0 is not None and b < t0:
                    continue
                if t1 is not None and a > t1:
                    continue
                cand.setdefault((s_, a, b), 0.0)
        except Exception:
            pass

    # ---- SEGMENTS: pad each window so the verifier sees the WHOLE event
    # (the close happens after the put; a bare window may hold only one) —
    # but NEVER across a recording boundary. Packed corpora butt recordings
    # together with no time gap, and an unclamped pad+merge produced
    # "clips" spanning two recordings: broken playback, and worse, the
    # before/after verifier judging frames from two DIFFERENT recordings —
    # every such cached verdict was noise (an opening clip scored +0.97
    # for "closing the drawer" this way).
    rec = _recording_spans(store)
    pad = int(pad_s * 1e9)
    by_stream = {}
    for (s, a, b) in cand:
        lo, hi = a - pad, b + pad
        if s in rec:
            r0, r1 = rec[s]
            j = int(np.searchsorted(r0, (a + b) // 2, side="right")) - 1
            if 0 <= j < len(r0):
                lo, hi = max(lo, int(r0[j])), min(hi, int(r1[j]))
        if hi > lo:
            by_stream.setdefault(s, []).append((lo, hi))
    segs = []
    for s, spans in by_stream.items():
        spans.sort()
        cur = list(spans[0])
        for a, b in spans[1:]:
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                segs.append((s, cur[0], cur[1]))
                cur = [a, b]
        segs.append((s, cur[0], cur[1]))
    segs = segs[:eff_pool]

    # ---- segment index score: best member window's appearance cosine,
    # plus (when the store has a V2 event index) best member ctx cosine.
    # NaN = the ctx ranker ABSTAINS on that segment; RRF gives it the
    # median rank rather than the bottom (see fusion.py on why).
    seg_score, seg_ctx, seg_lex, seg_mot, seg_anc = {}, {}, {}, {}, {}
    seg_obj, seg_met, seg_vid, seg_pe, seg_act = {}, {}, {}, {}, {}
    # candidate -> containing segment by per-stream binary search: the
    # nested scan was O(candidates x segments) Python and went to ~700 ms
    # when the recall floor raised both (measured); this is O(n log n)
    seg_sorted = {}
    for si, seg in enumerate(segs):
        seg_sorted.setdefault(seg[0], []).append(seg)
    for s_ in seg_sorted:
        seg_sorted[s_].sort(key=lambda g: g[1])
    import bisect
    starts = {s_: [g[1] for g in lst] for s_, lst in seg_sorted.items()}
    for (s_, a, b), sc in cand.items():
        lst = seg_sorted.get(s_)
        if not lst:
            continue
        j = bisect.bisect_right(starts[s_], a) - 1
        if j < 0:
            continue
        seg = lst[j]
        if not (seg[1] <= a and b <= seg[2]):
            continue
        seg_score[seg] = max(seg_score.get(seg, -1.0), sc)
        if (s_, a, b) in ctx_of:
            seg_ctx[seg] = max(seg_ctx.get(seg, -2.0), ctx_of[(s_, a, b)])
        if (s_, a, b) in lex_of:
            seg_lex[seg] = max(seg_lex.get(seg, 0.0), lex_of[(s_, a, b)])
        if (s_, a, b) in met_of:
            seg_met[seg] = max(seg_met.get(seg, 0.0), met_of[(s_, a, b)])
        if (s_, a, b) in anc_of:
            seg_anc[seg] = max(seg_anc.get(seg, -2.0), anc_of[(s_, a, b)])
    for seg in segs:
        seg_score.setdefault(seg, 0.0)
        seg_ctx.setdefault(seg, float("nan"))
        seg_lex.setdefault(seg, float("nan"))
        seg_met.setdefault(seg, float("nan"))
        seg_anc.setdefault(seg, float("nan"))
        # motion attaches by RECORDING overlap: motion rows are recording
        # spans (the scale where the state change is fully straddled) and
        # every segment is clamped inside exactly one recording
        seg_mot[seg] = (mot_lookup(*seg) if mot_lookup is not None
                        else float("nan"))
        if obj_lookup is not None:
            osc, omo = obj_lookup(*seg)
            # motion-weighted: a matching subject that MOVES outranks the
            # same subject sitting still (the acting subject)
            seg_obj[seg] = (osc * (1.0 + omo) if osc == osc
                            else float("nan"))
        else:
            seg_obj[seg] = float("nan")
        seg_vid[seg] = (vid_look(*seg) if vid_look is not None
                        else float("nan"))
        seg_pe[seg] = (pe_look(*seg) if pe_look is not None
                       else float("nan"))
        seg_act[seg] = (act_look(*seg) if act_look is not None
                        else float("nan"))

    # directional queries hash differently: their margins are swap-CONTRASTS
    # (query minus inverted query), a different quantity than the absolute
    # margins cached before — colliding them would rank with stale semantics
    from .rerank import directional_swap
    dtag = ("dir:" + text) if directional_swap(text) else text
    qh = _qhash(dtag)
    dqh = _qhash("deep:" + dtag)
    vmap = _verdict_map(store)
    # the two tiers are kept SEPARATE because they carry different
    # authority: only the deep (7B) judge may pin a segment to the top or
    # bottom. A 2B screen margin is evidence — one more ranking channel —
    # never a lock: "verified positives first" let stale 2B yeses (minted
    # against an older candidate pool) hold the top of 'folding cloth'
    # forever while the true folds sat unverified below (user-reported,
    # reproduced).
    deep_m, scr_m = {}, {}
    for seg in segs:
        if (seg[0], seg[1], dqh) in vmap:
            deep_m[seg] = vmap[(seg[0], seg[1], dqh)]
        if (seg[0], seg[1], qh) in vmap:
            scr_m[seg] = vmap[(seg[0], seg[1], qh)]
    cached_m = dict(scr_m)
    cached_m.update(deep_m)                    # sync path display: best tier
    fresh = [seg for seg in segs if seg not in cached_m]

    if verify == "sync":
        # the old blocking path — scripts and evals that WANT to wait
        margins = dict(cached_m)
        margins.update(_verify_segments(store, fresh, text, qh,
                                        frames_per_clip))
        hits = [{"stream": s_, "t0": a, "t1": b,
                 "margin": round(margins[(s_, a, b)], 3),
                 "score": margins[(s_, a, b)],
                 "verified": True,
                 "cached": (s_, a, b) in cached_m}
                for (s_, a, b) in margins]
        hits.sort(key=lambda h: -h["score"])
        if deep and hits:
            hits = _deep_rerank(store, hits, text, deep, vmap)
        stats = {"method": "verified", "verify": "sync", "atoms": atoms,
                 "candidates": len(cand), "segments": len(segs),
                 "verified_fresh": len(fresh),
                 "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
        return hits[:k], stats

    # ---- ASYNC (default): the query is INDEX-ONLY — no model call ever ----
    # VLM-FREE (2026-07-24): the 2B/7B court is out of the shipping path
    # entirely — user directive after calibration showed both tiers
    # overclaim scene matches (7B est 1.0 on visually ~10%-pure sets).
    # Cached verdicts carry NO ranking authority; ordering is pure fused
    # index. Video-native verification lives in the channels themselves
    # (act = SSv2 posteriors, mot = delta-appearance contrast) and in the
    # geometric audit tier of the set path. The sync branch above remains
    # for EVALUATION scripts only.
    fresh = list(segs)
    if fresh:
        from .fusion import rrf
        # channel weights are a LEARNED, per-store artifact: fitted offline
        # against a labeled regression set (scripts/fit_weights.py) from
        # captured channel scores — never hand-tuned, reloaded when the
        # file changes, defaults to 1.0 everywhere.
        wfile = Path(store.dir) / "_channel_weights.json"
        learned = {}
        try:
            if wfile.exists():
                _cfg = json.loads(wfile.read_text())
                # per-query-type routing: directional and content queries
                # have different physics; a single shared vector let every
                # new content channel tax direction (measured three times)
                learned = (_cfg.get("weights_dir", _cfg.get("weights", {}))
                           if qv_swap is not None
                           else _cfg.get("weights", {}))
        except Exception:
            pass
        # For directional queries, motion is the ONLY channel measuring the
        # query's discriminating dimension (the others are direction-blind,
        # measured), so it carries extra weight; otherwise it is off.
        fused = rrf({"app": np.array([seg_score[g] for g in fresh]),
                     "ctx": np.array([seg_ctx[g] for g in fresh]),
                     "lex": np.array([seg_lex[g] for g in fresh]),
                     "mot": np.array([seg_mot[g] for g in fresh]),
                     "anc": np.array([seg_anc[g] for g in fresh]),
                     "obj": np.array([seg_obj[g] for g in fresh]),
                     "met": np.array([seg_met[g] for g in fresh]),
                     "vid": np.array([seg_vid[g] for g in fresh]),
                     "pe": np.array([seg_pe[g] for g in fresh]),
                     "act": np.array([seg_act[g] for g in fresh])},
                    weights={**{c: learned.get(c, 1.0) for c in
                                ("app", "ctx", "lex", "met", "anc",
                                 "obj", "vid", "pe", "act")},
                             "mot": (learned.get("mot", 2.5)
                                     if qv_swap is not None else 0.0)})
        # displayed score = the fused score that actually ordered the hit;
        # showing raw appearance while ordering by fusion read as broken
        # attribution hook: which channel put each segment where — the
        # regression bench reads this to give every miss a named cause
        global _LAST_ATTR
        _LAST_ATTR = {"segs": list(fresh),
                      "channels": {"app": [seg_score[g] for g in fresh],
                                   "ctx": [seg_ctx[g] for g in fresh],
                                   "lex": [seg_lex[g] for g in fresh],
                                   "met": [seg_met[g] for g in fresh],
                                   "mot": [seg_mot[g] for g in fresh],
                                   "anc": [seg_anc[g] for g in fresh],
                                   "obj": [seg_obj[g] for g in fresh],
                                   "vid": [seg_vid[g] for g in fresh],
                                   "pe": [seg_pe[g] for g in fresh],
                                   "act": [seg_act[g] for g in fresh]}}
        fused_of = dict(zip(fresh, fused))
        seg_score.update(fused_of)
        fresh = [g for _, g in sorted(zip(-fused, fresh))]

    hits = [{"stream": g[0], "t0": g[1], "t1": g[2], "margin": None,
             "score": seg_score[g], "verified": False, "cached": False}
            for g in fresh]

    stats = {"method": "verified", "verify": "async", "atoms": atoms,
             "candidates": len(cand), "segments": len(segs),
             "ms": round((time.perf_counter() - t_start) * 1e3, 1)}
    return hits[:k], stats


def _verify_segments(store, segs, text, qh, frames_per_clip=2):
    """Decode start/end frames, score with the 2B before/after question,
    append verdicts to the store. Used sync by evals, async by queries."""
    from PIL import Image

    from .rerank import as_change_question, as_clip_question, \
        directional_swap, score_clip_sequences
    from .video import FrameSet
    if not segs:
        return {}
    q_form = (as_change_question if frames_per_clip == 2
              else as_clip_question)
    question = q_form(text)
    frames_tbl = store.table("frames").scan()
    rot = store.meta.get("display", {}).get("rotate", 0)
    need, clips = [], []
    for seg in segs:
        s, a, b = seg
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), s),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), a),
                    pc.less_equal(frames_tbl.column("ts"), b))))
        if len(sel) < 2:
            continue
        pick = np.linspace(0, len(sel) - 1,
                           min(frames_per_clip, len(sel))).round().astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=448)
        if len(dec) < 2:
            continue
        imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
        if rot:
            imgs = [im.rotate(rot, expand=True) for im in imgs]
        need.append(seg)
        clips.append(imgs)
    if not clips:
        return {}
    # SWAP-CONTRAST for directional queries: both VLM tiers are direction-
    # inverted on absolute questions (AUC 0.36 — they score salient
    # interaction, not direction), but the DIFFERENCE against the inverted
    # query cancels the appearance bias: 2B AUC 0.86 (see directional_swap).
    # One extra VLM pass, only when the query has a direction to invert.
    raw = np.array(score_clip_sequences(clips, question), dtype=float)
    sq = directional_swap(text, store)
    if sq:
        raw = raw - np.array(score_clip_sequences(clips, q_form(sq)),
                             dtype=float)
    margins = {seg: float(m) for seg, m in zip(need, raw)}
    # `query` is stored as TEXT, not only qhash: verdicts are the one verb
    # supervision source that survived measurement (yes/no margins, AUC
    # 0.75-0.82, where free-form captions failed at 1-8/24), and as
    # (text, segment, ±margin) triples they can retrain the ctx adapter.
    # A hash can rank cached answers; only the text can teach.
    vt = pa.table({
        "ts": pa.array([g[1] for g in need], pa.int64()),
        "t1": pa.array([g[2] for g in need], pa.int64()),
        "stream": pa.array([g[0] for g in need]),
        "qhash": pa.array([qh] * len(need), pa.int64()),
        "margin": pa.array([margins[g] for g in need], pa.float32()),
        "query": pa.array([text] * len(need)),
    })
    try:
        store.table("vlm_verdicts").append(
            vt, kind="timeseries", evolve=True,
            meta={"written_by": "verified search", "query": text[:120]})
    except Exception:
        pass                                   # concurrent commit: retryable
    vmap = _verdict_map(store)
    for g in need:
        vmap[(g[0], g[1], qh)] = margins[g]
    return margins


def _deep_rerank(store, hits, text, deep, vmap):
    """7B judge over the top hits — sync callers only."""
    from PIL import Image

    from .rerank import (DEEP_VLM, as_clip_question, directional_swap,
                         score_clip_sequences)
    from .video import FrameSet
    frames_tbl = store.table("frames").scan()
    dqh = _qhash("deep:" + (("dir:" + text) if directional_swap(text)
                            else text))
    head = hits[:deep]
    keep, clips7 = [], []
    for h in head:
        key = (h["stream"], h["t0"], dqh)
        if key in vmap:
            h["deep_margin"] = round(vmap[key], 3)
            h["score"] = vmap[key]
            continue
        sel = frames_tbl.filter(pc.and_(
            pc.equal(frames_tbl.column("stream"), h["stream"]),
            pc.and_(pc.greater_equal(frames_tbl.column("ts"), h["t0"]),
                    pc.less_equal(frames_tbl.column("ts"), h["t1"]))))
        if len(sel) < 2:
            continue
        pick = np.linspace(0, len(sel) - 1, min(4, len(sel))).round() \
            .astype(int)
        dec = FrameSet(store, "frames", sel.take(pick)).decode(width=448)
        if len(dec) < 2:
            continue
        clips7.append([Image.fromarray(d[1]) for d in sorted(dec)])
        keep.append(h)
    if clips7:
        deep_m = np.array(score_clip_sequences(
            clips7, as_clip_question(text), model_id=DEEP_VLM), dtype=float)
        # same swap-contrast as the 2B tier: 7B is also direction-inverted
        # on absolute questions (AUC 0.36) and 0.91 on the difference
        sq = directional_swap(text, store)
        if sq:
            deep_m = deep_m - np.array(score_clip_sequences(
                clips7, as_clip_question(sq), model_id=DEEP_VLM),
                dtype=float)
        rows = []
        for h, m in zip(keep, deep_m):
            h["deep_margin"] = round(float(m), 3)
            h["score"] = float(m)
            vmap[(h["stream"], h["t0"], dqh)] = float(m)
            rows.append(h)
        if rows:
            try:
                store.table("vlm_verdicts").append(pa.table({
                    "ts": pa.array([h["t0"] for h in rows], pa.int64()),
                    "t1": pa.array([h["t1"] for h in rows], pa.int64()),
                    "stream": pa.array([h["stream"] for h in rows]),
                    "qhash": pa.array([dqh] * len(rows), pa.int64()),
                    "margin": pa.array([h["score"] for h in rows],
                                       pa.float32()),
                    "query": pa.array([text] * len(rows)),
                }), kind="timeseries", evolve=True,
                    meta={"written_by": "deep verify", "query": text[:120]})
            except Exception:
                pass
    head.sort(key=lambda h: -h["score"])
    return head + hits[deep:]