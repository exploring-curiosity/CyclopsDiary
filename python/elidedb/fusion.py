"""Reciprocal rank fusion.

WHY NOT JUST ADD THE SCORES
---------------------------
Because they are not on the same scale, and no amount of tuning fixes that.
SigLIP image-text cosines on a homogeneous corpus occupy roughly 0.01-0.15
(the modality gap squashes them); caption-LSA cosines spread over most of
[-1, 1]; TF-IDF cosines are mostly 0 with a thin tail near 1. A weighted sum
of those is governed by whichever signal happens to have the widest spread on
that particular query, so the weight you set is not the weight you get.

Standardising per query (z-scores) helps but still lets one ranker's outlier
drag the result: a single candidate three sigma out on appearance outranks a
candidate that every ranker agrees is second-best.

RRF (Cormack, Clarke & Buettcher, SIGIR 2009) throws the magnitudes away and
keeps only the ORDER:

    score(d) = sum over rankers r of  w_r / (K + rank_r(d))

That makes fusion scale-free by construction, and it rewards CONSENSUS: a
document ranked 2nd by three rankers beats one ranked 1st by a single ranker
and 200th by the rest. Which is exactly the failure being fixed — "crossing
red car" returned any clip of a person crossing, because one strong signal on
"crossing" was allowed to win alone. Under RRF the answer has to look right to
the appearance index AND the caption index AND the lexical index.

K (default 60, the constant from the paper) damps the top of the curve: it is
the number of rank positions over which differences stop mattering much, so a
ranker cannot dominate purely by being extremely confident about its #1.
"""
from __future__ import annotations

import numpy as np

DEFAULT_K = 60.0


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """0-based ranks; highest score gets rank 0.

    NaN means ABSTAIN, and an abstention is given the median rank rather than
    the last. This matters: the lexical ranker can only score windows that
    actually have a caption, and windows whose context vector was estimated by
    the tower have none. Sending those to the bottom would let a ranker veto
    every item it has no opinion about, which is the opposite of what "no
    evidence" should mean. Median rank is the neutral position — it neither
    helps nor hurts.
    """
    s = np.asarray(scores, dtype=float)
    known = ~np.isnan(s)
    r = np.empty(len(s), dtype=float)
    if not known.any():
        r[:] = 0.0
        return r
    sub = s[known]
    order = np.argsort(-sub, kind="stable")
    rr = np.empty(len(sub), dtype=float)
    rr[order] = np.arange(len(sub), dtype=float)
    r[known] = rr
    r[~known] = float(np.median(rr))
    return r


def rrf(rankings: dict[str, np.ndarray], weights: dict[str, float] | None = None,
        k: float = DEFAULT_K) -> np.ndarray:
    """Fuse score vectors (all over the SAME candidate set, same order).

    Pass raw scores, not ranks — this converts. Returns the fused score, where
    higher is better, on an arbitrary but consistent scale.
    """
    if not rankings:
        raise ValueError("rrf() needs at least one ranking")
    n = len(next(iter(rankings.values())))
    out = np.zeros(n, dtype=float)
    for name, sc in rankings.items():
        sc = np.asarray(sc, dtype=float)
        if len(sc) != n:
            raise ValueError(
                f"ranker '{name}' has {len(sc)} scores, expected {n} — every "
                "ranker must score the same candidate set")
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        if w == 0.0:
            continue
        out += w / (k + ranks_from_scores(sc) + 1.0)
    return out


def variant_max(vs):
    """Row-wise max over query-variant score arrays. All-NaN rows
    (episodes absent from a channel's table — they abstain, they are
    not errors) stay NaN without numpy's All-NaN-slice RuntimeWarning,
    which was leaking source lines into bench stdout."""
    m = np.stack(vs)
    out = np.full(m.shape[1], np.nan)
    fin = np.isfinite(m).any(0)
    if fin.any():
        out[fin] = np.nanmax(m[:, fin], 0)
    return out


def explain_fusion(rankings, weights=None, k=DEFAULT_K, idx=None, top=5):
    """Per-ranker rank of each fused winner. Answers 'why is this here?' —
    which is the question a fused score alone can never answer."""
    fused = rrf(rankings, weights, k)
    order = np.argsort(-fused)[:top] if idx is None else idx
    rk = {n: ranks_from_scores(s) for n, s in rankings.items()}
    return [{"i": int(i), "fused": float(fused[i]),
             "ranks": {n: int(r[i]) + 1 for n, r in rk.items()}}
            for i in order]
