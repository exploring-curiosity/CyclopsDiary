"""The `pe` channel: Meta Perception Encoder (published SOTA zero-shot
video-text retrieval), adopted by measurement — best single-encoder
precision@10 on the bench (48/140 vs shipping fusion 40), complementary
profile (put-into-drawer 10/10 where others fail). Naive equal fusion
measured WORSE than PE alone, so this channel routes through the learned
per-query-type weights like every other.

8 frame vectors per recording in `pe_vectors`; recording score =
top-5-mean of frame cosines (the winning pooling from the zero-shot
sweep). Query cost: one open_clip text forward (~15 ms, cached)."""
from __future__ import annotations

import numpy as np

_TEXT = {}
_IDX = {}


def _text_vec(text):
    import torch
    if text in _TEXT.get("cache", {}):
        return _TEXT["cache"][text]
    if "model" not in _TEXT:
        import open_clip

        from .device import pick, strip_vision, text_only
        dev, dtype = pick()
        m, _, _ = open_clip.create_model_and_transforms(
            "PE-Core-L-14-336", pretrained="meta")
        m = strip_vision(m.to(dev).eval(), "visual")
        if text_only():
            # serving keeps only the text tower; at reduced precision
            # it is a few hundred MB instead of the fp32 gigabytes
            m = m.to(dtype)
        _TEXT["model"] = m
        _TEXT["dtype"] = m.ln_final.weight.dtype \
            if hasattr(m, "ln_final") else None
        _TEXT["tok"] = open_clip.get_tokenizer("PE-Core-L-14-336")
        _TEXT["dev"] = dev
        _TEXT["cache"] = {}
    with torch.no_grad():
        t = _TEXT["model"].encode_text(
            _TEXT["tok"]([text]).to(_TEXT["dev"]))
    v = (t / t.norm(dim=-1, keepdim=True))[0].cpu().float().numpy()
    if len(_TEXT["cache"]) > 256:
        _TEXT["cache"].clear()
    _TEXT["cache"][text] = v
    return v


def pe_lookup(store, text):
    """(lookup(stream, t0, t1) -> top5-mean cosine | nan, candidates)."""
    from .embeddings import _vec_table
    ver = store.table("pe_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _IDX:
        tbl, _ = _vec_table(store, "pe_vectors")
        ss = np.asarray(tbl.column("stream").to_pylist())
        sa = np.asarray([int(v) for v in tbl.column("ts").to_pylist()])
        sb = np.asarray([int(v) for v in tbl.column("t1").to_pylist()])
        # rows grouped by recording (stream, t0, t1): build run map
        recs = {}
        order = np.lexsort((sa, ss))
        for r in order:
            recs.setdefault((str(ss[r]), int(sa[r]), int(sb[r])),
                            []).append(int(r))
        idx = {}
        for (s, a, b), rows in recs.items():
            idx.setdefault(s, []).append((a, b, np.array(rows)))
        for s in idx:
            idx[s].sort(key=lambda x: x[0])
        if len(_IDX) > 8:
            _IDX.clear()
        _IDX[key] = idx
    idx = _IDX[key]
    _, vecs = _vec_table(store, "pe_vectors")
    sc = vecs @ _text_vec(text)

    def rec_score(rows):
        v = sc[rows]
        k = min(5, len(v))
        return float(np.sort(v)[-k:].mean())

    def lookup(s, a, b):
        lst = idx.get(s)
        if not lst:
            return float("nan")
        starts = [x[0] for x in lst]
        j = int(np.searchsorted(starts, a, side="right")) - 1
        if j >= 0 and b <= lst[j][1] + 1:
            return rec_score(lst[j][2])
        return float("nan")

    cands = []
    for s, lst in idx.items():
        for a, b, rows in lst:
            cands.append((s, a, b, rec_score(rows)))
    cands.sort(key=lambda x: -x[3])
    return lookup, cands[:64]
