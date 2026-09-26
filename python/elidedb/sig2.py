"""SigLIP 2 channels: `sig2` (content) and `conj` (conjunctive atoms).

sig2: episode score = top-5 mean of frame cosines against the query —
the SigLIP 1 recipe on the improved encoder (fine-grained + better
localization, arXiv 2502.14786).

conj: TEST-TIME COMPOSITIONAL BINDING, mechanical (no LLM): every
determiner phrase in the query is an atom that must independently find
a frame match; the episode score is the MINIMUM over atom scores. A
spoon-on-cloth clip needs a spoon-ish frame AND a cloth-ish frame or
it dies — the decomposition-as-test-time-program idea from the
composed-retrieval literature, with the program being a regex over
articles. Domain-free by construction."""
from __future__ import annotations

import re

import numpy as np

_S = {}
_IDX = {}

MID = "google/siglip2-so400m-patch14-384"


def _text_vec(text):
    import torch
    cache = _S.setdefault("cache", {})
    if text in cache:
        return cache[text]
    if "model" not in _S:
        from transformers import AutoModel, AutoProcessor
        from .device import pick, strip_vision
        dev, dtype = pick()
        _S["proc"] = AutoProcessor.from_pretrained(MID)
        _S["model"] = strip_vision(AutoModel.from_pretrained(
            MID, dtype=dtype,
            low_cpu_mem_usage=True).to(dev).eval(), "vision_model")
        _S["dev"] = dev
    with torch.no_grad():
        tok = _S["proc"](text=[text], padding="max_length",
                         max_length=64, truncation=True,
                         return_tensors="pt").to(_S["dev"])
        t = _S["model"].get_text_features(**tok)
    v = (t / t.norm(dim=-1, keepdim=True))[0].cpu().float().numpy()
    if len(cache) > 256:
        cache.clear()
    cache[text] = v
    return v


def _index(store):
    ver = store.table("sig2_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _IDX:
        from .embeddings import _vec_table
        tbl, _ = _vec_table(store, "sig2_vectors")
        recs = {}
        for r, (s, a) in enumerate(zip(
                tbl.column("stream").to_pylist(),
                (int(v) for v in tbl.column("ts").to_pylist()))):
            recs.setdefault((str(s), a), []).append(r)
        idx = {}
        for (s, a), rows in recs.items():
            idx.setdefault(s, []).append((a, np.array(rows)))
        for s in idx:
            idx[s].sort(key=lambda x: x[0])
        if len(_IDX) > 8:
            _IDX.clear()
        _IDX[key] = idx
    return _IDX[key]


def _frame_scores(store, text):
    from .embeddings import _vec_table
    _, vecs = _vec_table(store, "sig2_vectors")
    return np.asarray(vecs) @ _text_vec(text)


def _lookup_from(idx, sc, pool):
    def lookup(s, a, b):
        lst = idx.get(str(s))
        if not lst:
            return float("nan")
        starts = [x[0] for x in lst]
        j = int(np.searchsorted(starts, a, side="right")) - 1
        if j < 0 or lst[j][0] != a:
            return float("nan")
        return pool(sc[lst[j][1]])
    return lookup


def sig2_lookup(store, text):
    idx = _index(store)
    sc = _frame_scores(store, text)

    def pool(v):
        k = min(5, len(v))
        return float(np.sort(v)[-k:].mean())
    return _lookup_from(idx, sc, pool), None


# closed-class boundary words (English function words — dictionary
# knowledge, corpus-independent): a phrase filler may not contain
# them, and a trailing one is stripped. Without the boundary the
# filler swallowed prepositions ("the eggplant into the") and a
# two-object query collapsed to one corrupt atom — conj abstained.
_STOP = ("a", "an", "the", "and", "then", "it", "of", "to", "on",
         "in", "into", "onto", "from", "at")
_ATOM_RE = re.compile(
    r"\b(?:a|an|the)\s+(?:(?!(?:%s)\b)\w+\s+){0,2}\w+"
    % "|".join(_STOP))


def atoms_of(text):
    """Mechanical atoms: every determiner phrase in the query,
    bounded at closed-class function words."""
    out = []
    for m in _ATOM_RE.finditer(text.lower()):
        w = m.group(0).split()
        while len(w) > 1 and w[-1] in _STOP:
            w.pop()
        if len(w) > 1 and w[1] not in ("table", "robot", "arm"):
            out.append(" ".join(w))
    return list(dict.fromkeys(out))


def conj_lookup(store, text):
    """MIN over atom max-frame scores; abstains (None) when the query
    has fewer than two atoms — nothing to conjoin."""
    atoms = atoms_of(text)
    if len(atoms) < 2:
        return None
    idx = _index(store)
    per_atom = [_frame_scores(store, a) for a in atoms]

    def lookup(s, a, b):
        lst = idx.get(str(s))
        if not lst:
            return float("nan")
        starts = [x[0] for x in lst]
        j = int(np.searchsorted(starts, a, side="right")) - 1
        if j < 0 or lst[j][0] != a:
            return float("nan")
        rows = lst[j][1]
        return float(min(sc[rows].max() for sc in per_atom))
    return lookup
