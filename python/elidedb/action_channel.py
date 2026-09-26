"""The `act` channel: SSv2 action posteriors as an INDEX-ONLY ranking
signal. Ingested once per episode (scripts/action_ingest.py: V-JEPA 2
ViT-L + Meta's released attentive probe, 174 classes); a query costs a
cached text-vector pass over the 174 class names plus one 174-d dot per
episode.

Adoption measurement (2026-07-24, labeled episodes, zero fitting):
put-in vs take-out AUC 0.889 with the literal class pair — the exact
containment direction the green-drawer failure exposed and no other
channel measures. Directional queries score a CONTRAST of class
weights, w(text) − w(swap), mirroring the swap-contrast law used
everywhere else in this system.
"""
from __future__ import annotations

import numpy as np

_IDX = {}


def _index(store):
    ver = store.table("action_probs").state().version
    key = (str(store.dir), ver)
    if key not in _IDX:
        from .embeddings import _vec_table
        tbl, _ = _vec_table(store, "action_probs")
        ss = tbl.column("stream").to_pylist()
        sa = [int(v) for v in tbl.column("ts").to_pylist()]
        sb = [int(v) for v in tbl.column("t1").to_pylist()]
        idx = {}
        for r, (s, a, b) in enumerate(zip(ss, sa, sb)):
            idx.setdefault(str(s), []).append((a, b, r))
        for s in idx:
            idx[s].sort()
        if len(_IDX) > 8:
            _IDX.clear()
        _IDX[key] = idx
    return _IDX[key]


def act_lookup(store, text, contrast=None):
    """(lookup(stream, t0, t1) -> weighted posterior | nan,
    candidates top-64). Rows in action_probs are one per episode.
    `contrast`: an explicit 174-d weight vector (canonical_contrast)
    overrides the text-mapped weights."""
    from .action_probe import query_class_weights
    from .embeddings import _vec_table
    from .rerank import directional_swap

    idx = _index(store)
    _, probs = _vec_table(store, "action_probs")

    if contrast is not None:
        w = contrast
    else:
        w = query_class_weights(text)
        sq = directional_swap(text, store)
        if sq is not None:
            w = w - query_class_weights(sq)
    sc = np.asarray(probs) @ w

    def lookup(s, a, b):
        lst = idx.get(str(s))
        if not lst:
            return float("nan")
        starts = [x[0] for x in lst]
        j = int(np.searchsorted(starts, a, side="right")) - 1
        if j >= 0 and b <= lst[j][1] + 1:
            return float(sc[lst[j][2]])
        return float("nan")

    cands = []
    for s, lst in idx.items():
        for a, b, r in lst:
            cands.append((s, a, b, float(sc[r])))
    cands.sort(key=lambda x: -x[3])
    return lookup, cands[:64]
