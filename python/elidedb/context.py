"""Context retrieval: the ingest-side half.

WHY THIS EXISTS
---------------
`embeddings` holds one mean-pooled SigLIP vector per window. Mean pooling is
order-blind: reverse the frames and the vector is identical. So the index can
answer "is there a car and a person here" but never "is the person walking
*toward* the car" — appearance, not context. Reranking with a VLM fixes the
ranking but costs seconds per query, which is the wrong place to spend time in
a database.

The fix is the oldest trick a database has: precompute the expensive operator
into an index and make the query a lookup. Concretely,

    ingest (once, offline)      query (every time, hot)
    ------------------------    -----------------------
    VLM reads 3 frames of the   text -> SigLIP text vector
    window and writes a
    relational caption          ONE matmul against ctx vectors
        |
    SigLIP text tower           (no VLM in the loop, ever)
        |
    target vector  -----------> train a small temporal tower to
                                predict it from cheap frame vectors

The VLM only ever labels a subset. The tower generalises the label to every
window, including windows ingested later, so a new day of footage costs a
forward pass over frame vectors instead of a day of VLM time. This is
pseudo-labelling in the sense of "Distilling Vision-Language Models on
Millions of Videos" (arXiv 2401.06129), applied at ingest instead of at
pretraining scale.

Three tables come out of this module, all ordinary Parquet:

    frame_vectors     ts, stream, vector[d]              one row per FRAME
    context_captions  ts, t1, stream, caption, vector[d] teacher labels
    context           ts, t1, stream, vector[d]          student output
"""
from __future__ import annotations

import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .embeddings import (DEFAULT_MODEL, MODELS, _embed_images,
                         _load_model, resolve_model)

# The teacher is asked for RELATIONS and CHANGE, not for a list of objects.
# An object list is exactly what SigLIP already encodes, so a caption that
# reads "a car, a road, a building" teaches the student nothing it does not
# already know. Everything interesting is in the verbs.
CAPTION_PROMPT = (
    "These frames are in time order from one short video clip. "
    "Reply with ONE sentence of at most 25 words describing what is "
    "happening: who or what is present, what each is doing, how they are "
    "positioned relative to each other, and how the scene moves or changes. "
    "Be concrete and literal. Do not say 'frame', 'image', or 'video'."
)

# The caption is the index. If it does not use the words a user would use, the
# lexical ranker can never fire and the caption-LSA space is built around the
# wrong distinctions. Measured on BridgeData2: the generic prompt above
# produced "a robot arm interacts with a wooden box", while the human label
# for the same clip was "put red object in the drawer" — different noun,
# different granularity, no overlap for retrieval to work with.
#
# So the prompt is a per-domain parameter. This one asks for the ACTION and
# the OBJECT MOVED, which is how manipulation data is described. It stays
# deliberately generic: it never names objects that appear in the labels
# (that would be leaking the eval set into the index), only the SHAPE of the
# description — what moved, and where it ended up.
# The caption IS the index: whatever verbs the prompt teaches are the only
# verbs lexical recall can ever match. The first version of this prompt said
# "which object the arm picks up or moves, and where it puts it" — and the
# resulting 2,348 captions contained picks x2371, puts x1673 and ZERO
# instances of close/open/wipe/push, so "close the drawer" was unfindable by
# construction. The prompt must be VERB-OPEN: describe the action in its own
# words, and always report state changes.
# S0 VIOLATION, FIXED. The previous MANIPULATION_PROMPT enumerated the
# task vocabulary - "picking up, putting, opening, closing, pushing,
# pouring, wiping, pressing" - and this file's own comment stated the
# consequence: "The caption IS the index: whatever verbs the prompt
# teaches are the only verbs lexical recall can ever match." A verb
# outside that list was unfindable by construction, and a non-
# manipulation corpus was unfindable entirely.
#
# It was also TUNED AGAINST THE EVAL LABELS. The removed comment
# recorded the generic prompt producing "a robot arm interacts with a
# wooden box" where "the human label for the same clip was 'put red
# object in the drawer'", and the prompt being rewritten to close that
# gap. bridge_ingest.py deliberately keeps the task strings out of the
# store; this prompt let them back in through the side door.
#
# The replacement names no verb, no object class and no domain. It asks
# for what MOVED and what CHANGED, which is answerable for a kitchen, a
# road, a warehouse or a surgical table, and lets the corpus supply its
# own words.
MANIPULATION_PROMPT = (
    "These frames are in time order from one short clip. "
    "Reply with ONE short sentence in plain English: what moved, what it "
    "did, and what was different at the end. Use whatever words fit; do "
    "not choose from a list. If anything changed state, say so. "
    "Do not say 'frame', 'image', or 'video'."
)

PROMPTS = {"scene": CAPTION_PROMPT, "manipulation": MANIPULATION_PROMPT}

# SigLIP's text tower truncates at 64 tokens, so a rambling caption is
# silently cut mid-clause and the tail is lost anyway. Trim deliberately
# instead: drop the VLM's framing preamble, keep whole sentences.
#
# The preamble pattern is deliberately narrow — a leading PREPOSITIONAL
# phrase only ("In the first frame,", "Across this sequence,"). An earlier
# looser version also matched "The video shows a street with cars," and
# amputated the subject, leaving captions that began "and various buildings".
# Requiring a leading preposition and at most two filler words makes that
# impossible.
_PREAMBLE = re.compile(
    r"^(?:in|across|throughout|over|during)\s+(?:the|this|these)\s+"
    r"(?:[\w-]+\s+){0,2}?(?:frames?|images?|pictures?|sequence|clip|video)\s*,\s*",
    re.I)

# Prompt-echo: asking for "the action verb in plain English" made the VLM
# write 'The action verb is "picking up" as the arm picks up...' — meta-
# language that pollutes the lexical index. Strip the frame, keep the deed.
_VERB_ECHO = re.compile(
    r"^the action(?:\s+verb)?\s+is\s+['\"]?[\w-]+(?:\s+[\w-]+){0,2}?"
    r"['\"]?[,.]?\s+(?:as|because|where|since|:)\s+", re.I)


def tidy_caption(text: str, max_words: int = 32) -> str:
    t = " ".join(text.strip().split())
    t = _PREAMBLE.sub("", t)
    t = _VERB_ECHO.sub("", t)
    parts = re.split(r"(?<=[.!?])\s+", t)
    # A generation cut off at max_tokens ends mid-clause. Keep only sentences
    # that actually terminate, unless that would leave nothing at all.
    whole = [p for p in parts if p.rstrip().endswith((".", "!", "?"))]
    parts = whole or parts[:1]
    out = []
    for p in parts:
        if out and len(" ".join(out + [p]).split()) > max_words:
            break
        out.append(p)
    t = " ".join(out).strip()
    return (t[:1].upper() + t[1:]) if t else " ".join(text.split())[:200]


# ---------------------------------------------------------------------------
# 1. Per-frame vectors — the sequence a temporal model needs
# ---------------------------------------------------------------------------
def embed_frames(store, frame_table="frames", model=None, width=512,
                 batch=32, incremental=True, verbose=True, stride=1,
                 streams=None, engine="siglip"):
    """One SigLIP vector per frame → `frame_vectors`.

    Deliberately NOT pooled. Pooling is the student's job, and pooling here
    would throw away the only signal that distinguishes context from
    appearance. Decode goes through the same byte-range path queries use, so
    this costs the frames it reads and nothing else.
    """
    from PIL import Image

    from .video import FrameSet
    # engine="fdnnv": the distilled streaming encoder — every frame, no
    # stride, state carried per stream. Same output space as the teacher, so
    # everything downstream (pooling, search, context) is unchanged.
    if engine == "fdnnv":
        return _embed_frames_fdnnv(store, frame_table, incremental, verbose,
                                   streams)
    model = resolve_model(model)
    frames = store.table(frame_table).scan()
    allst = sorted(set(frames.column("stream").to_pylist()))
    streams = [s for s in allst if s in streams] if streams else allst

    done = {}
    if incremental:
        try:
            prev = store.table("frame_vectors").scan()
            for s_, t_ in zip(prev.column("stream").to_pylist(),
                              prev.column("ts").to_pylist()):
                done[s_] = max(done.get(s_, -1), t_)
        except Exception:
            pass

    t_start = time.time()
    rows_ts, rows_stream, rows_vec = [], [], []
    for s in streams:
        sel = frames.filter(pc.equal(frames.column("stream"), s))
        if s in done:
            sel = sel.filter(pc.greater(sel.column("ts"), done[s]))
        if len(sel) == 0:
            continue
        # `stride` subsamples the frame index before decoding. A 5 Hz robot
        # camera does not need every frame embedded for a 4 s window to be
        # well described, and the cost here is linear in frames decoded, so
        # this is the dial between ingest time and temporal resolution.
        decoded = FrameSet(store, frame_table, sel).decode(width=width,
                                                           stride=stride)
        if verbose:
            print(f"  {s}: {len(decoded)} frames decoded", flush=True)
        for i in range(0, len(decoded), batch):
            chunk = decoded[i:i + batch]
            vecs = _embed_images([Image.fromarray(a) for _, a in chunk], model)
            for (ts, _), v in zip(chunk, vecs):
                rows_ts.append(int(ts))
                rows_stream.append(s)
                rows_vec.append(v)
    if not rows_ts:
        return {"frames": 0, "note": "nothing new (incremental)"}

    dim = len(rows_vec[0])
    tbl = pa.table({
        "ts": pa.array(rows_ts, pa.int64()),
        "stream": pa.array(rows_stream),
        "vector": pa.array([v.tolist() for v in rows_vec],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("frame_vectors").append(
        tbl, kind="embeddings",
        meta={"model": model, "dim": dim, "decode_width": width,
              "source_table": frame_table})
    return {"frames": len(tbl), "dim": dim, "version": version,
            "seconds": round(time.time() - t_start, 1)}


# ---------------------------------------------------------------------------
# 2. Window plan — shared by teacher and student so labels line up exactly
# ---------------------------------------------------------------------------
def plan_windows(store, window_s=2.0, stride_s=0.5, table="frame_vectors",
                 min_frames=4):
    """Sliding windows over each stream's timeline.

    Stride < window on purpose: overlapping windows are how a *sliding* index
    avoids the boundary problem where an event straddles two tumbling windows
    and lands strongly in neither. Merging overlaps back into one answer is
    already handled downstream by the segment merger.
    """
    fv = store.table(table).scan()
    win = int(window_s * 1e9)
    stride = int(stride_s * 1e9)
    out = []
    for s in sorted(set(fv.column("stream").to_pylist())):
        rows = fv.filter(pc.equal(fv.column("stream"), s))
        ts = np.sort(rows.column("ts").to_numpy())
        if len(ts) == 0:
            continue
        t = int(ts[0])
        end = int(ts[-1])
        while t <= end - win // 2:
            lo, hi = np.searchsorted(ts, [t, t + win])
            if hi - lo >= min_frames:
                out.append((s, t, t + win - 1))
            t += stride
    return out


def window_sequences(store, windows, table="frame_vectors", max_len=32):
    """(stream, t0, t1) → (T, d) float32 stack of that window's frame vectors.

    Subsampled to `max_len` evenly. A 2 s window at 16 Hz is 32 frames; the
    cap keeps the tower's cost independent of frame rate, which is what makes
    the same model valid across a 10 Hz LiDAR-synced camera and a 60 Hz one.
    """
    fv = store.table(table).scan()
    by_stream = {}
    for s in sorted(set(fv.column("stream").to_pylist())):
        rows = fv.filter(pc.equal(fv.column("stream"), s))
        ts = rows.column("ts").to_numpy()
        order = np.argsort(ts)
        vecs = np.asarray(rows.column("vector").to_pylist(), dtype=np.float32)
        by_stream[s] = (ts[order], vecs[order])
    seqs = []
    for (s, t0, t1) in windows:
        ts, vecs = by_stream[s]
        lo, hi = np.searchsorted(ts, [t0, t1 + 1])
        idx = np.arange(lo, hi)
        if len(idx) > max_len:
            idx = idx[np.linspace(0, len(idx) - 1, max_len).round().astype(int)]
        seqs.append(vecs[idx])
    return seqs


# ---------------------------------------------------------------------------
# 3. The teacher — a VLM that actually reads the pixels, run ONCE per window
# ---------------------------------------------------------------------------
def caption_windows(store, windows, frames_per_window=3, model_id=None,
                    max_tokens=64, width=448, verbose=True, limit=None,
                    every=1, prompt="scene"):
    """VLM captions for `windows` → `context_captions` table.

    The VLM is shown several frames of the SAME window in order, so the
    caption can describe motion. A single-frame caption would be another
    appearance label and the student would learn nothing a mean-pool cannot
    already produce.
    """
    import tempfile
    from pathlib import Path

    from PIL import Image

    from .rerank import DEFAULT_VLM, _load
    from .video import FrameSet
    model_id = model_id or DEFAULT_VLM
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    vlm, processor, cfg, _, _ = _load(model_id)
    rot = store.meta.get("display", {}).get("rotate", 0)
    frames = store.table("frames").scan()
    tmpdir = Path(tempfile.mkdtemp(prefix="elidedb_ctx_"))

    # `every` samples the window list uniformly instead of taking a prefix.
    # On a corpus too large to caption in full this matters: a prefix would
    # confine the caption vocabulary to whatever happens early in the
    # timeline, so query terms for anything later would be out of vocabulary
    # and the lexical ranker would abstain on the entire tail.
    if every > 1:
        windows = windows[::every]
    if limit:
        windows = windows[:limit]
    text = PROMPTS.get(prompt, prompt)          # preset name or raw prompt
    prompt = apply_chat_template(processor, cfg, text,
                                 num_images=frames_per_window)

    recs = {"ts": [], "t1": [], "stream": [], "caption": []}
    t_start = time.time()
    for n, (s, t0, t1) in enumerate(windows):
        sel = frames.filter(pc.and_(
            pc.equal(frames.column("stream"), s),
            pc.and_(pc.greater_equal(frames.column("ts"), t0),
                    pc.less_equal(frames.column("ts"), t1))))
        fs = FrameSet(store, "frames", sel)
        dec = fs.decode(width=width)
        if len(dec) < 1:
            continue
        picks = np.linspace(0, len(dec) - 1,
                            min(frames_per_window, len(dec))).round().astype(int)
        paths = []
        for j, p in enumerate(picks):
            im = Image.fromarray(dec[p][1])
            if rot:
                im = im.rotate(rot, expand=True)
            fp = tmpdir / f"w{n}_{j}.jpg"
            im.save(fp, "JPEG", quality=85)
            paths.append(str(fp))
        while len(paths) < frames_per_window:      # pad short windows
            paths.append(paths[-1])
        r = generate(vlm, processor, prompt, image=paths,
                     max_tokens=max_tokens, verbose=False)
        cap = tidy_caption(r.text if hasattr(r, "text") else str(r))
        recs["ts"].append(t0)
        recs["t1"].append(t1)
        recs["stream"].append(s)
        recs["caption"].append(cap)
        if verbose and n % 10 == 0:
            el = time.time() - t_start
            print(f"  [{n + 1}/{len(windows)}] {el:.0f}s  {s} +"
                  f"{(t0 - int(frames.column('ts')[0].as_py())) / 1e9:.1f}s :: "
                  f"{cap[:90]}", flush=True)
    if not recs["ts"]:
        return {"captions": 0}

    # Caption → frozen SigLIP TEXT tower. This is the whole point of using
    # SigLIP as the teacher's codec: the target lands in the SAME space a
    # user's query lands in, so the student is learning to be the image side
    # of a dual encoder — not to regress an arbitrary embedding.
    vecs = embed_texts(recs["caption"])
    dim = vecs.shape[1]
    tbl = pa.table({
        "ts": pa.array(recs["ts"], pa.int64()),
        "t1": pa.array(recs["t1"], pa.int64()),
        "stream": pa.array(recs["stream"]),
        "caption": pa.array(recs["caption"]),
        "vector": pa.array([v.tolist() for v in vecs],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("context_captions").append(
        tbl, kind="embeddings",
        meta={"teacher": model_id, "text_model": DEFAULT_MODEL, "dim": dim,
              "frames_per_window": frames_per_window, "prompt": text[:200],
              "seconds": round(time.time() - t_start, 1)})
    return {"captions": len(tbl), "version": version,
            "seconds": round(time.time() - t_start, 1)}


def embed_texts(texts, model_id=DEFAULT_MODEL, batch=32):
    """Batched SigLIP text tower. Same normalisation as image vectors so the
    two are directly comparable by dot product. Delegates to the
    portable single-text path when mlx is not on this machine."""
    from .embeddings import _backend, embed_text
    if _backend() != "mlx":
        return np.stack([embed_text(t if t.strip() else "a scene",
                                    model_id) for t in texts])
    import mlx.core as mx
    model, processor = _load_model(model_id)
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t if t.strip() else "a scene" for t in texts[i:i + batch]]
        ti = processor(text=chunk, padding="max_length", max_length=64,
                       truncation=True, return_tensors="np")
        v = np.array(model.get_text_features(mx.array(ti["input_ids"])),
                     dtype=np.float32)
        out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 4. CaptionSpace — the output space, chosen by measurement
# ---------------------------------------------------------------------------
# Three candidate spaces were benchmarked against a VLM judge on windows the
# tower never trained on (mean yes/no logprob margin over each method's top-5,
# six relational queries, higher is better):
#
#     appearance only  (SigLIP image-text)                +0.276
#     caption EMBEDDING (SigLIP text tower, oracle)       +0.218   <- worse
#     VLM rerank at query time (2.1 s/query)              +0.314
#     caption TEXT, LSA-48                                +0.339   <- winner
#     caption TEXT, raw TF-IDF, fused with appearance     +0.345
#
# The embedding route loses because SigLIP's text tower is trained to sit
# near IMAGES, not near other text; comparing a query embedding to a caption
# embedding uses a geometry the model was never optimised for. Matching the
# caption as TEXT sidesteps that entirely.
#
# LSA-48 is chosen over raw TF-IDF despite scoring 0.006 lower: it is dense
# and fixed-width, so (a) it is a target a small tower can actually regress,
# which is what lets unlabelled windows get a context vector at all, and
# (b) it is one more fixed_size_list column, so every index already in the
# store consumes it unchanged.


class CaptionSpace:
    """TF-IDF + LSA over the caption corpus. Queries and captions share it.

    Not persisted as a pickle. The captions themselves are the durable
    artifact — they live in `context_captions` as ordinary Parquet — and the
    lexical index is refit from them on load (a few ms for this corpus) and
    cached. That keeps the index unconditionally consistent with the data and
    free of any sklearn version pinning. At corpus sizes where refitting
    stops being free, persist the vocabulary and the SVD basis; the interface
    does not change.
    """

    def __init__(self, vec, svd):
        self.vec, self.svd = vec, svd
        self.dim = svd.n_components

    @staticmethod
    def fit(texts, dim=48):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              sublinear_tf=True).fit(texts)
        X = vec.transform(texts)
        dim = int(min(dim, X.shape[1] - 1, len(texts) - 1))
        svd = TruncatedSVD(n_components=dim, random_state=0).fit(X)
        return CaptionSpace(vec, svd)

    def transform(self, texts):
        v = self.svd.transform(self.vec.transform(list(texts)))
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


_SPACE_CACHE: dict = {}


def caption_space(store, dim=48):
    t = store.table("context_captions")
    key = (str(store.dir), t.state().version, dim)
    if key not in _SPACE_CACHE:
        _SPACE_CACHE.clear()
        _SPACE_CACHE[key] = CaptionSpace.fit(
            t.scan().column("caption").to_pylist(), dim=dim)
    return _SPACE_CACHE[key]


# ---------------------------------------------------------------------------
# 5. The student — train the tower, then materialise context vectors
# ---------------------------------------------------------------------------
def _labelled(store, windows):
    """Join the caption table onto the window plan by (stream, ts)."""
    caps = store.table("context_captions").scan()
    key = {(s, t): i for i, (s, t) in enumerate(
        zip(caps.column("stream").to_pylist(), caps.column("ts").to_pylist()))}
    vecs = np.asarray(caps.column("vector").to_pylist(), dtype=np.float32)
    texts = caps.column("caption").to_pylist()
    keep, tv, tt = [], [], []
    for w in windows:
        i = key.get((w[0], w[1]))
        if i is not None:
            keep.append(w)
            tv.append(vecs[i])
            tt.append(texts[i])
    return keep, np.asarray(tv, dtype=np.float32), tt


def train_context(store, window_s=2.0, stride_s=0.5, val_frac=0.3,
                  d_in=48, d_out=48, epochs=400, cfg=None, verbose=True,
                  seed=0):
    """Fit the input PCA + tower to predict caption-LSA coordinates.

    The validation split is by TIME, never at random: windows slide with 75%
    overlap, so a random split puts near-duplicate windows on both sides and
    reports a score that measures nothing.
    """
    from .ctxtower import ContextCodec, retrieval_r1, save_tower, train_tower

    windows = plan_windows(store, window_s, stride_s)
    windows, _, cap_text = _labelled(store, windows)
    if len(windows) < 16:
        raise RuntimeError(f"only {len(windows)} captioned windows — run "
                           "caption_windows() first")

    fv = store.table("frame_vectors").scan()
    frame_mat = np.asarray(fv.column("vector").to_pylist(), dtype=np.float32)
    codec = ContextCodec.fit(frame_mat, d_in=d_in)
    space = caption_space(store, dim=d_out)

    seqs = [codec.encode(s) for s in window_sequences(store, windows)]
    targets = space.transform(cap_text)

    t0s = np.array([w[1] for w in windows])
    cut = np.quantile(t0s, 1.0 - val_frac)
    val_idx = np.where(t0s >= cut)[0]
    if verbose:
        print(f"  {len(windows)} labelled windows | train "
              f"{len(windows) - len(val_idx)} | val {len(val_idx)} "
              f"(time split at +{(cut - t0s.min()) / 1e9:.1f}s) | "
              f"target dim {space.dim}", flush=True)

    cfg = {**dict(d_in=codec.P_in.shape[0], d_out=space.dim), **(cfg or {})}
    model, info = train_tower(seqs, targets, windows, val_idx, cfg=cfg,
                              epochs=epochs, verbose=verbose, seed=seed)

    import mlx.core as mx
    val = np.zeros(len(seqs), bool)
    val[val_idx] = True
    Xva = _pad_stack([s for s, m in zip(seqs, val) if m])
    pred = np.array(model(mx.array(Xva)))
    pred /= np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8
    tgt_va = targets[val]

    # Baseline: predict the TRAIN-set mean target for every window. That is
    # the best a model can do while knowing nothing about the specific window,
    # so any lift over it is genuine per-window information and not the corpus
    # prior leaking through.
    prior = targets[~val].mean(axis=0)
    prior /= np.linalg.norm(prior) + 1e-8
    prior = np.repeat(prior[None, :], len(tgt_va), axis=0)

    metrics = {
        "labelled_windows": len(windows),
        "train": int((~val).sum()), "val": int(val.sum()),
        "target_dim": space.dim,
        "val_R@1_tower": retrieval_r1(pred, tgt_va),
        "val_cos_tower": float((pred * tgt_va).sum(1).mean()),
        "val_cos_prior": float((prior * tgt_va).sum(1).mean()),
        "params": int(sum(v.size for v in _flat(model).values())),
        **{k: v for k, v in info.items() if k != "history"},
    }
    save_tower(model, codec,
               {"window_s": window_s, "stride_s": stride_s, "d_out": space.dim,
                "metrics": metrics, "history": info["history"]},
               store.dir / "models" / "context")
    return model, codec, metrics


def _flat(model):
    from mlx.utils import tree_flatten
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def _pad_stack(seqs):
    T = max(s.shape[0] for s in seqs)
    X = np.zeros((len(seqs), T, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
        if len(s) < T:
            X[i, len(s):] = s[-1]
    return X


def build_context(store, window_s=None, stride_s=None, batch=64, verbose=True):
    """Materialise the `context` table: one context vector per window.

    Where a window has a teacher caption, its EXACT caption-LSA vector is
    stored. Where it does not, the tower's prediction is stored and the row is
    flagged `estimated`. This is the ordinary database distinction between a
    materialised value and an estimated one, and it is the point of having a
    student at all: the VLM labels what you can afford, the tower covers the
    rest, and the query does not care which it got.
    """
    import time

    import mlx.core as mx

    from .ctxtower import load_tower
    model, codec, meta = load_tower(store.dir / "models" / "context")
    window_s = window_s or meta.get("window_s", 2.0)
    stride_s = stride_s or meta.get("stride_s", 0.5)
    space = caption_space(store, dim=meta.get("d_out", 48))

    windows = plan_windows(store, window_s, stride_s)
    raw = window_sequences(store, windows)
    seqs = [codec.encode(s) for s in raw]
    # The mean-pooled appearance vector for the SAME window, stored alongside.
    # Fusing appearance with context otherwise needs a join between two tables
    # built on different window plans; keeping both columns in one row makes
    # the fused query two matmuls over one Parquet scan and no join at all.
    appear = np.stack([s.mean(axis=0) for s in raw])
    appear /= np.linalg.norm(appear, axis=1, keepdims=True) + 1e-8

    t_start = time.time()
    out = []
    for i in range(0, len(seqs), batch):
        out.append(np.array(model(mx.array(_pad_stack(seqs[i:i + batch])))))
    z = np.concatenate(out, axis=0)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    infer_s = time.time() - t_start

    caps = store.table("context_captions").scan()
    known = {(a, b): c for a, b, c in zip(caps.column("stream").to_pylist(),
                                          caps.column("ts").to_pylist(),
                                          caps.column("caption").to_pylist())}
    have = [(i, known[(w[0], w[1])]) for i, w in enumerate(windows)
            if (w[0], w[1]) in known]
    estimated = np.ones(len(windows), bool)
    if have:
        exact = space.transform([c for _, c in have])
        for (i, _), v in zip(have, exact):
            z[i] = v
            estimated[i] = False

    dim, adim = z.shape[1], appear.shape[1]
    tbl = pa.table({
        "ts": pa.array([w[1] for w in windows], pa.int64()),
        "t1": pa.array([w[2] for w in windows], pa.int64()),
        "stream": pa.array([w[0] for w in windows]),
        "vector": pa.array([v.tolist() for v in z],
                           pa.list_(pa.float32(), dim)),
        "appearance": pa.array([v.tolist() for v in appear],
                               pa.list_(pa.float32(), adim)),
        "estimated": pa.array(estimated.tolist(), pa.bool_()),
    })
    st = store.table("context").state()
    meta_out = {"dim": dim, "window_s": window_s, "stride_s": stride_s,
                "estimated_rows": int(estimated.sum()),
                "exact_rows": int((~estimated).sum())}
    if st.files:                                 # replace: one atomic commit
        import uuid as _uuid

        from .log import FileEntry
        from .store import write_parquet
        fname = f"part-{_uuid.uuid4().hex[:12]}.parquet"
        p = store.dir / "tables" / "context" / fname
        tbl = tbl.take(pc.sort_indices(tbl.column("ts")))
        write_parquet(tbl, p)
        tsv = tbl.column("ts").to_numpy()
        version = store.table("context").log.commit(
            op="replace", kind="embeddings", schema=str(tbl.schema),
            add=[FileEntry(fname, len(tbl), p.stat().st_size,
                           int(tsv.min()), int(tsv.max()))],
            remove=[f.path for f in st.files], meta=meta_out)
    else:
        version = store.table("context").append(tbl, kind="embeddings",
                                                meta=meta_out)
    if verbose:
        print(f"  {len(tbl)} context vectors ({int((~estimated).sum())} exact, "
              f"{int(estimated.sum())} estimated) | tower inference "
              f"{infer_s * 1000:.0f} ms "
              f"({infer_s / len(tbl) * 1e6:.0f} us/window)")
    return {"windows": len(tbl), "dim": dim, "version": version,
            "inference_s": round(infer_s, 3),
            "us_per_window": round(infer_s / len(tbl) * 1e6, 1), **meta_out}


# ---------------------------------------------------------------------------
# 6. Post-training cellular turnover — capacity the data cannot support is
#    both latency and overfitting, so apoptosis pays twice.
# ---------------------------------------------------------------------------
def _prepare(store, window_s, stride_s, codec, val_frac, d_out):
    windows, _, texts = _labelled(store, plan_windows(store, window_s,
                                                      stride_s))
    seqs = [codec.encode(s) for s in window_sequences(store, windows)]
    targets = caption_space(store, dim=d_out).transform(texts)
    t0s = np.array([w[1] for w in windows])
    val = t0s >= np.quantile(t0s, 1.0 - val_frac)
    return windows, seqs, targets, val


def prune_context(store, val_frac=0.3, ppo_iters=40, sparsity_coef=0.15,
                  finetune_epochs=300, rebirth_fraction=0.5,
                  select_tolerance=0.03, verbose=True, seed=0):
    """apoptosis → re-settle → neurogenesis → re-settle, then COMPACT so the
    surviving channels are the only ones that cost anything."""
    import time

    import mlx.core as mx
    import mlx.nn as nn

    from .ctxprune import compact, get_channel_mask, run_pruning_cycle
    from .ctxtower import (load_tower, overlap_mask, retrieval_r1,
                           save_tower, siglip_loss)

    model, codec, meta = load_tower(store.dir / "models" / "context")
    windows, seqs, targets, val = _prepare(
        store, meta["window_s"], meta["stride_s"], codec, val_frac,
        meta.get("d_out", 48))

    Xtr = mx.array(_pad_stack([s for s, m in zip(seqs, ~val) if m]))
    Xva = mx.array(_pad_stack([s for s, m in zip(seqs, val) if m]))
    Ytr, Yva = mx.array(targets[~val]), mx.array(targets[val])
    ig_tr = mx.array(overlap_mask([w for w, m in zip(windows, ~val) if m]))
    ig_va = mx.array(overlap_mask([w for w, m in zip(windows, val) if m]))
    log_t = mx.array(np.float32(meta["metrics"]["log_t"]))
    bias = mx.array(np.float32(meta["metrics"]["bias"]))

    def _norm(a):
        return a * mx.rsqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-8)

    def _loss(m, x, y, ig):
        v = _norm(m(x))
        u = _norm(y)
        return siglip_loss(v, u, ignore=ig, log_t=log_t, bias=bias) \
            + 0.3 * mx.mean(1.0 - mx.sum(v * u, axis=-1))

    def loss_of(m):
        """The number PPO is scored against: VALIDATION retrieval loss. Using
        train loss here would reward a policy for keeping memorisers."""
        m.set_training(False)
        return float(_loss(m, Xva, Yva, ig_va).item())

    lg = nn.value_and_grad(model, lambda m: _loss(m, Xtr, Ytr, ig_tr))

    def r1(m):
        m.set_training(False)
        return retrieval_r1(np.array(_norm(m(Xva))), np.array(_norm(Yva)))

    def latency(m, reps=20):
        m.set_training(False)
        mx.eval(m(Xva))
        t = time.perf_counter()
        for _ in range(reps):
            mx.eval(m(Xva))
        return (time.perf_counter() - t) / reps / Xva.shape[0] * 1e6

    before = {"channels": int(get_channel_mask(model).sum()),
              "val_loss": loss_of(model), "val_R@1": r1(model),
              "us_per_window": latency(model),
              "params": int(sum(v.size for v in _flat(model).values()))}

    rec = run_pruning_cycle(model, np.array(Xtr), loss_of, lg,
                            ppo_iters=ppo_iters, sparsity_coef=sparsity_coef,
                            finetune_epochs=finetune_epochs,
                            rebirth_fraction=rebirth_fraction,
                            select_tolerance=select_tolerance, seed=seed,
                            verbose=verbose)

    # Compaction must be a no-op numerically — it only deletes channels the
    # mask already zeroed. Measure both sides and say so if they disagree,
    # rather than quietly shipping a tower that differs from the one the
    # search selected.
    restored = loss_of(model)
    model, kept = compact(model)
    after = {"channels": int(len(kept)), "val_loss": loss_of(model),
             "loss_before_compaction": restored, "val_R@1": r1(model),
             "us_per_window": latency(model),
             "params": int(sum(v.size for v in _flat(model).values()))}
    rec["before"], rec["after"] = before, after
    drift = abs(after["val_loss"] - restored)
    if drift > 1e-3:
        print(f"  WARNING: compaction changed val loss by {drift:.4f} "
              f"({restored:.4f} -> {after['val_loss']:.4f}) — expected ~0")
    if verbose:
        print(f"\n  channels {before['channels']} -> {after['channels']} | "
              f"params {before['params']:,} -> {after['params']:,} | "
              f"{before['us_per_window']:.1f} -> {after['us_per_window']:.1f} "
              f"us/window | val loss {before['val_loss']:.4f} -> "
              f"{after['val_loss']:.4f}")
    save_tower(model, codec,
               {**{k: v for k, v in meta.items() if k != "history"},
                "pruned": {"before": before, "after": after,
                           "ppo_history": rec.get("ppo_history"),
                           "keep_probs": rec.get("keep_probs"),
                           "reverse_attention": rec.get("reverse_attention"),
                           "selected": rec.get("selected"),
                           "stages": [{k: v for k, v in s.items()
                                       if k != "mask"} for s in rec["stages"]]}},
               store.dir / "models" / "context")
    return model, rec


# ---------------------------------------------------------------------------
# 7. The query path — two matmuls, no VLM, no decode
# ---------------------------------------------------------------------------
_CTX_CACHE: dict = {}


def _ctx_matrix(store):
    t = store.table("context")
    key = (str(store.dir), t.state().version)
    if key not in _CTX_CACHE:
        _CTX_CACHE[key] = np.stack([
            np.asarray(v, dtype=np.float32)
            for v in t.scan().column("vector").to_pylist()])
    return _CTX_CACHE[key]


def _lexical(store):
    """(vectorizer, matrix, has_caption) aligned to the `context` table rows.

    Windows the VLM never captioned have no text, so they are marked as
    abstentions rather than as empty documents — an empty document would score
    0 on every query and be ranked last by the lexical ranker, which is a veto
    dressed up as evidence.
    """
    t = store.table("context")
    key = (str(store.dir), t.state().version, "lex")
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    ctx = t.scan()
    rows = list(zip(ctx.column("stream").to_pylist(),
                    ctx.column("ts").to_pylist()))
    caps = store.table("context_captions").scan()
    known = dict(zip(zip(caps.column("stream").to_pylist(),
                         caps.column("ts").to_pylist()),
                     caps.column("caption").to_pylist()))
    texts = [known.get(r, "") for r in rows]
    has = np.array([bool(x) for x in texts])
    space = caption_space(store)
    X = space.vec.transform(texts)
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel() + 1e-8
    _CTX_CACHE[key] = (space, X, nrm, has)
    return _CTX_CACHE[key]


DEFAULT_WEIGHTS = {"appearance": 1.0, "context": 1.0, "lexical": 1.0}


def search(store, text, k=10, merge=True, t0=None, t1=None, streams=None,
           weights=None, rrf_k=60.0, neg_weight=0.5, min_score=None,
           percentile=None, rerank=False, rerank_top=12, rerank_alpha=0.7,
           explain_top=0):
    """Hybrid contextual search: three rankers fused by reciprocal rank.

        appearance  SigLIP image-text cosine over the window's frames.
                    Knows what OBJECTS are present. Order-blind.
        context     caption-LSA cosine. Knows what is HAPPENING, because the
                    caption was written by a VLM that watched three frames.
        lexical     TF-IDF over the caption text. Exact term evidence — the
                    ranker that actually knows what "red" means.

    Fusing by RRF rather than by a weighted score sum is the fix for
    "crossing red car" returning any clip of someone crossing: RRF rewards
    agreement across rankers, so a candidate that satisfies one strong signal
    alone can no longer win. See elidedb.fusion.

    `rerank=True` adds a final VLM pass over the top `rerank_top` — the
    expensive operator, last, on an already-pruned set.
    """
    from .embeddings import _parse_query, _score_windows, embed_text
    from .fusion import explain_fusion, rrf
    # Fail with a sentence, not a KeyError from three frames down. An empty
    # table has no schema, so the first column access explodes with
    # 'Field "t1" does not exist' — true, useless, and it names the wrong
    # problem.
    if not store.table("context").state().files:
        raise RuntimeError(
            f"store '{store.name}' has no context index. Build it with "
            "store.index_context() (frames -> per-frame vectors -> VLM "
            "captions -> context table), or use store.search_text() for "
            "appearance-only search.")
    pos, neg = _parse_query(text)
    ctx_tbl = store.table("context").scan()
    all_t0 = ctx_tbl.column("ts").to_numpy()
    all_t1 = ctx_tbl.column("t1").to_numpy()
    all_s = ctx_tbl.column("stream").to_numpy(zero_copy_only=False)

    # hybrid retrieval: time and stream predicates are pushed INTO candidate
    # selection, not applied to a global top-k afterwards
    pred = np.ones(len(all_t0), bool)
    if t0 is not None:
        pred &= all_t1 >= t0
    if t1 is not None:
        pred &= all_t0 <= t1
    if streams:
        pred &= np.isin(all_s, list(streams))
    idx = np.where(pred)[0]
    if len(idx) == 0:
        return [], {"index": "context", "total": len(all_t0), "scanned": 0,
                    "segments": 0}

    APP = _appearance_matrix(store)
    CTX = _ctx_matrix(store)
    space, X, nrm, has_cap = _lexical(store)

    pos_app = np.stack([embed_text(p) for p in pos])
    neg_app = np.stack([embed_text(n) for n in neg]) if neg else None
    app = _score_windows(APP, idx, pos_app, neg_app, neg_weight)

    q_join = " ".join(pos)
    ctx_sc = CTX[idx] @ space.transform([q_join])[0]

    qv = space.vec.transform([q_join])
    lex = np.asarray((X[idx] @ qv.T).todense()).ravel() / nrm[idx]
    lex = np.where(has_cap[idx], lex, np.nan)      # abstain, do not veto

    rankings = {"appearance": app, "context": ctx_sc, "lexical": lex}
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    scores = rrf(rankings, w, rrf_k)

    stats = {"index": "context", "total": len(all_t0), "scanned": len(idx),
             "method": "rrf", "rrf_k": rrf_k, "weights": w,
             "predicate_candidates": int(pred.sum()),
             "captioned_candidates": int(has_cap[idx].sum()),
             "positive_terms": pos, "negative_terms": neg}
    if explain_top:
        stats["why"] = explain_fusion(rankings, w, rrf_k, top=explain_top)

    keep = np.ones(len(idx), bool)
    if percentile is not None:
        keep &= scores >= np.percentile(scores, percentile)
    if min_score is not None:
        keep &= scores >= min_score
    idx, scores = idx[keep], scores[keep]
    stats["after_floor"] = int(len(idx))

    hits = _segments(idx, scores, all_s, all_t0, all_t1, k, merge, stats)
    if rerank and hits:
        from .rerank import rerank_hits
        hits, info = rerank_hits(store, hits, text, top_n=rerank_top,
                                 alpha=rerank_alpha)
        stats["rerank"] = info
    return hits, stats


def _segments(idx, scores, all_s, all_t0, all_t1, k, merge, stats):
    """Merge qualifying windows into maximal runs per stream.

    Fixed windows are an INDEXING granularity, not an answer granularity: a
    20 s event should come back as one 20 s hit, and a query matching only 2 s
    of it should come back as that 2 s.
    """
    streams_sel, w_t0, w_t1 = all_s[idx], all_t0[idx], all_t1[idx]
    if len(idx) == 0:
        stats["segments"] = 0
        return []
    if not merge:
        order = np.argsort(scores)[::-1][:k]
        return [{"stream": str(streams_sel[i]), "t0": int(w_t0[i]),
                 "t1": int(w_t1[i]), "score": float(scores[i]),
                 "windows": 1} for i in order]
    med, top = float(np.median(scores)), float(scores.max())
    thr = med + 0.55 * (top - med)
    stats["threshold"] = round(thr, 6)
    qual = np.where(scores >= thr)[0]
    order = np.lexsort((w_t0[qual], streams_sel[qual]))
    qual = qual[order]
    gap = int(np.median(w_t1[qual] - w_t0[qual])) + 1 if len(qual) else 0
    segs = []
    for i in qual:
        s_, a, b, sc = (str(streams_sel[i]), int(w_t0[i]), int(w_t1[i]),
                        float(scores[i]))
        last = segs[-1] if segs else None
        if last and last["stream"] == s_ and a - last["t1"] <= gap:
            last["t1"] = max(last["t1"], b)
            last["score"] = max(last["score"], sc)
            last["mean"] = (last["mean"] * last["windows"] + sc) / (last["windows"] + 1)
            last["windows"] += 1
        else:
            segs.append({"stream": s_, "t0": a, "t1": b, "score": sc,
                         "mean": sc, "windows": 1})
    segs.sort(key=lambda g: -g["score"])
    stats["qualifying_windows"] = len(qual)
    stats["segments"] = len(segs)
    return segs[:k]


def _appearance_matrix(store):
    t = store.table("context")
    key = (str(store.dir), t.state().version, "app")
    if key not in _CTX_CACHE:
        _CTX_CACHE[key] = np.stack([
            np.asarray(v, dtype=np.float32)
            for v in t.scan().column("appearance").to_pylist()])
    return _CTX_CACHE[key]


def explain(store, t0, t1, stream=None):
    """What the database believes is happening in a window — the teacher's own
    words. Lets a result be checked rather than trusted."""
    caps = store.table("context_captions").scan()
    out = []
    if len(caps) == 0 or "stream" not in caps.column_names:
        return out          # store has no captions (yet) — nothing to explain
    for s, a, b, c in zip(caps.column("stream").to_pylist(),
                          caps.column("ts").to_pylist(),
                          caps.column("t1").to_pylist(),
                          caps.column("caption").to_pylist()):
        if stream and s != stream:
            continue
        if b >= t0 and a <= t1:
            out.append({"stream": s, "t0": a, "t1": b, "caption": c})
    return out


def _embed_frames_fdnnv(store, frame_table, incremental, verbose, streams):
    """Every frame through the FDNN-V streaming encoder -> frame_vectors."""
    import pyarrow as pa

    from .fdnnvideo import embed_stream, load_encoder
    mdir = store.dir / "models" / "fdnnv"
    if not (mdir / "encoder.json").exists():
        # a NEW store has no encoder yet — adopt one from a sibling store
        # and COPY it in, so the store stays self-contained and the exact
        # weights that wrote its vectors are pinned with its data
        import shutil
        donors = sorted(store.dir.parent.glob("*/models/fdnnv/encoder.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if not donors:
            raise RuntimeError(
                "no FDNN-V encoder found in this store or any sibling — "
                "train one first (scripts/fdnnv_train.py)")
        shutil.copytree(donors[0].parent, mdir)
        if verbose:
            print(f"  adopted encoder from {donors[0].parent}", flush=True)
    model, meta = load_encoder(mdir)
    # FRESH-DATA FIDELITY GATE: the fast student was distilled on one
    # style of footage; on an arbitrary upload its fidelity to the teacher
    # is unknown. 32 of THIS store's frames go through both encoders; if
    # mean cosine < 0.90 the store gets the TEACHER (slower ingest, right
    # space) instead of a fast-but-wrong index. Automatic — a fresh
    # customer never has to know this exists.
    try:
        from PIL import Image
        from .embeddings import DEFAULT_MODEL, _embed_images
        from .video import FrameSet
        frames_all = store.table(frame_table).scan()
        pick = np.linspace(0, len(frames_all) - 1,
                           min(32, len(frames_all))).round().astype(int)
        rows = frames_all.take(pick)
        ts_s, sv, _, _ = embed_stream(store, model, rows)
        dec = FrameSet(store, frame_table, rows).decode(width=448)
        imgs = [Image.fromarray(d[1]) for d in sorted(dec)]
        tv = _embed_images(imgs, DEFAULT_MODEL)
        n = min(len(sv), len(tv))
        svn = sv[:n] / (np.linalg.norm(sv[:n], axis=1,
                                       keepdims=True) + 1e-8)
        fid = float((svn * tv[:n]).sum(1).mean())
        if verbose:
            print(f"  student fidelity on this corpus: {fid:.3f}",
                  flush=True)
        if fid < 0.90:
            print(f"  fidelity {fid:.3f} < 0.90 — falling back to the "
                  f"TEACHER encoder for this store", flush=True)
            return embed_frames(store, frame_table, model="fast",
                                incremental=incremental, verbose=verbose,
                                streams=streams, engine="siglip")
    except Exception as e:
        if verbose:
            print(f"  fidelity gate skipped ({type(e).__name__})",
                  flush=True)
    frames = store.table(frame_table).scan()
    allst = sorted(set(frames.column("stream").to_pylist()))
    use = [s for s in allst if s in streams] if streams else allst
    done = {}
    if incremental:
        try:
            prev = store.table("frame_vectors").scan()
            for s_, t_ in zip(prev.column("stream").to_pylist(),
                              prev.column("ts").to_pylist()):
                done[s_] = max(done.get(s_, -1), t_)
        except Exception:
            pass
    t_start = time.time()
    rows_ts, rows_stream, rows_vec = [], [], []
    for s in use:
        sel = frames.filter(pc.equal(frames.column("stream"), s))
        if s in done:
            sel = sel.filter(pc.greater(sel.column("ts"), done[s]))
        if len(sel) == 0:
            continue
        sel = sel.take(pc.sort_indices(sel.column("ts")))
        ts, vecs, dec_s, emb_s = embed_stream(store, model, sel)
        if verbose:
            print(f"  {s}: {len(ts):,} frames (decode {dec_s:.1f}s, "
                  f"embed {emb_s:.1f}s)", flush=True)
        rows_ts.extend(int(t) for t in ts)
        rows_stream.extend([s] * len(ts))
        rows_vec.extend(vecs)
    if not rows_ts:
        return {"frames": 0, "note": "nothing new (incremental)"}
    dim = len(rows_vec[0])
    tbl = pa.table({
        "ts": pa.array(rows_ts, pa.int64()),
        "stream": pa.array(rows_stream),
        "vector": pa.array([v.tolist() for v in rows_vec],
                           pa.list_(pa.float32(), dim)),
    })
    version = store.table("frame_vectors").append(
        tbl, kind="embeddings",
        meta={"model": "fdnnv", "teacher": meta.get("teacher"),
              "dim": dim, "every_frame": True,
              "source_table": frame_table})
    return {"frames": len(tbl), "dim": dim, "version": version,
            "engine": "fdnnv",
            "seconds": round(time.time() - t_start, 1)}
