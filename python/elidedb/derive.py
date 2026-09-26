"""Corpus-derived replacements for hand-authored constants.

Every function here answers a question the code used to answer with a
literal. The rule they serve: nothing in code may encode what the data
CONTAINS or what the task IS. A constant is legitimate when it is a
tuning dial measured on the corpus or a property of a model; it is a
violation when it is a fact about the domain, because then the system is
being told the answer rather than recognising it.

The replacements have to work for any physical-AI corpus, not just this
one. A table-top manipulation set, a driving log and a warehouse camera
have nothing in common at the level of "open" or "put_into" - but they
all have objects that move, appear, vanish, approach and separate, and
those are measurable without naming anything.

WHAT WAS REPLACED
-----------------
    MIN_BLOB = 120 (and 40, 60, 60, 30 elsewhere)  -> fit_cut on the
        observed blob-area distribution, in frame fractions so the
        number transfers across resolutions.
    REL_MIN / CAV_MIN thresholds                    -> fit_cut, or gone
        entirely where the quantity feeds clustering instead of an
        if/else.
    hand-listed vocabularies                        -> attested(), words
        the corpus actually produced.
    hand-listed antonym pairs                       -> opposite_pairs(),
        found by reflection in an embedding space.
"""
from __future__ import annotations

import numpy as np


def fit_cut(values, q=None, mode="knee"):
    """A threshold from the data instead of from a keyboard.

    mode="knee"     the largest gap in the sorted values - where the
                    distribution itself separates, if it does.
    mode="quantile" the q-th percentile, for "tighter than all but q% of
                    what I can prove is background".
    mode="otsu"     the between-class variance maximiser, for a
                    distribution with two modes and no prior on where
                    the split sits.

    Returns (cut, diagnostics). The diagnostics matter: a knee found in
    a unimodal distribution is an artefact, and `separation` says so, so
    a caller can refuse to threshold a quantity that has no structure.
    """
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return None, {"reason": "too few samples", "n": int(len(v))}
    s = np.sort(v)
    if mode == "quantile":
        cut = float(np.percentile(s, q if q is not None else 99.0))
    elif mode == "otsu":
        hist, edges = np.histogram(s, bins=min(256, max(16, len(s) // 8)))
        p = hist / hist.sum()
        c = np.cumsum(p)
        m = np.cumsum(p * np.arange(len(p)))
        mt = m[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (mt * c - m) ** 2 / (c * (1 - c))
        i = int(np.nanargmax(var))
        cut = float(edges[i + 1])
    else:
        # knee: the widest gap between consecutive order statistics,
        # ignoring the extreme tails where single outliers dominate
        lo, hi = int(0.05 * len(s)), int(0.95 * len(s))
        seg = s[lo:hi]
        if len(seg) < 4:
            return float(np.median(s)), {"reason": "degenerate", "n": len(s)}
        gaps = np.diff(seg)
        i = int(np.argmax(gaps))
        cut = float((seg[i] + seg[i + 1]) / 2)
    below, above = s[s <= cut], s[s > cut]
    sep = 0.0
    if len(below) and len(above):
        spread = s.std() + 1e-12
        sep = float((above.mean() - below.mean()) / spread)
    return cut, {"n": int(len(s)), "cut": cut, "separation": round(sep, 3),
                 "frac_below": round(float(len(below) / len(s)), 3),
                 "mode": mode}


def attested(texts, min_count=2, max_frac=0.5):
    """Vocabulary the CORPUS produced, not a list someone typed.

    Drops words too rare to be a category and words so common they
    cannot discriminate. No stopword list: "the" is excluded because it
    appears in most documents, which is a measurement, not an opinion
    about English.
    """
    from collections import Counter
    docs = [str(t).lower().split() for t in texts if t]
    if not docs:
        return []
    df = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    return sorted(w for w, c in df.items()
                  if c >= min_count and c / n <= max_frac and w.isalpha())


def opposite_pairs(words, vec, top=1, min_sim=0.25):
    """Antonyms by REFLECTION, not by a table of 81 hand-written pairs.

    In an embedding space trained on natural text, an antonym pair tends
    to be the two ends of one axis: a and b are close in topic and
    opposed in direction once their shared component is removed. So for
    each word, remove the corpus mean, and look for the word whose
    residual points most nearly the other way.

    This finds whatever oppositions the corpus has. On a driving log it
    would find accelerate/brake without anyone having thought of them;
    on this one it finds open/close. That is the whole point - the
    system stops depending on someone having anticipated the domain.
    """
    W = [w for w in words if w]
    if len(W) < 4:
        return []
    V = np.stack([vec(w) for w in W]).astype(np.float32)
    V = V - V.mean(0)                       # the shared component is topic
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    S = V @ V.T
    np.fill_diagonal(S, 0.0)
    out, seen = [], set()
    for i, w in enumerate(W):
        j = int(np.argmin(S[i]))            # most opposed residual
        if S[i, j] > -min_sim:
            continue
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        out.append((w, W[j], round(float(-S[i, j]), 3)))
    return sorted(out, key=lambda r: -r[2])[:max(top * len(W), 1)]


def frame_fraction(px, width, height):
    """Pixel counts are resolution-specific; fractions are not.

    MIN_BLOB=120 meant one thing at 640x480 and something else at
    256x256, which is why five files carried five different values for
    the same idea.
    """
    return float(px) / float(max(width * height, 1))
