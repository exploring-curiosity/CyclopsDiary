"""The `vid` channel: natively co-trained video-text retrieval.

Adopted as a STOCK component by measurement (balanced sample, identical
grading): X-CLIP-large took the relational query block from 1/40 to
10/40 against the SigLIP appearance incumbent — the first candidate to
beat it on the class where the system was measured weakest ("put the lid
on a vessel"). Pretrained by its authors on generic video-text pairs;
nothing of ours; works on any upload, day one.

Clip vectors are ingested per recording (xclip_ingest.py, ~310 ms/clip,
background). Queries pay ONE text-tower forward (~25 ms, cached per
query text) plus a matmul over the mmap sidecar."""
from __future__ import annotations

import numpy as np

_TEXT = {}
_IDX = {}


def _text_vec(text):
    import torch
    if text in _TEXT.get("cache", {}):
        return _TEXT["cache"][text]
    if "model" not in _TEXT:
        from transformers import AutoProcessor, XCLIPModel

        from .device import pick, strip_vision
        mid = "microsoft/xclip-large-patch14"
        dev = pick()[0]
        _TEXT["model"] = strip_vision(
            XCLIPModel.from_pretrained(
                mid, low_cpu_mem_usage=True).to(dev).eval(),
            "vision_model", "mit")
        _TEXT["proc"] = AutoProcessor.from_pretrained(mid)
        _TEXT["dev"] = dev
        _TEXT["cache"] = {}
    with torch.no_grad():
        ti = _TEXT["proc"](text=[text], return_tensors="pt", padding=True)
        ti = {k: v.to(_TEXT["dev"]) for k, v in ti.items()
              if k in ("input_ids", "attention_mask")}
        t = _TEXT["model"].get_text_features(**ti)
        if not torch.is_tensor(t):
            t = _TEXT["model"].text_projection(t.pooler_output)
    v = t[0].cpu().float().numpy()
    v /= np.linalg.norm(v) + 1e-8
    if len(_TEXT["cache"]) > 256:
        _TEXT["cache"].clear()
    _TEXT["cache"][text] = v
    return v


def vid_lookup(store, text):
    """(lookup(stream, t0, t1) -> score | nan, candidates list).
    Recording-span containment via a per-version cached index."""
    from .embeddings import _vec_table
    ver = store.table("xclip_vectors").state().version
    key = (str(store.dir), ver)
    if key not in _IDX:
        tbl, _ = _vec_table(store, "xclip_vectors")
        ss = np.asarray(tbl.column("stream").to_pylist())
        sa = np.asarray([int(v) for v in tbl.column("ts").to_pylist()])
        sb = np.asarray([int(v) for v in tbl.column("t1").to_pylist()])
        idx = {}
        for s in np.unique(ss):
            m = np.where(ss == s)[0]
            o = np.argsort(sa[m], kind="stable")
            idx[s] = (sa[m][o], sb[m][o], m[o])
        if len(_IDX) > 8:
            _IDX.clear()
        _IDX[key] = (idx, ss, sa, sb)
    idx, ss, sa, sb = _IDX[key]
    _, vecs = _vec_table(store, "xclip_vectors")
    sc = vecs @ _text_vec(text)

    def lookup(s, a, b):
        if s not in idx:
            return float("nan")
        t0s, t1s, rows = idx[s]
        j = int(np.searchsorted(t0s, a, side="right")) - 1
        if j >= 0 and b <= int(t1s[j]) + 1:
            return float(sc[rows[j]])
        return float("nan")

    top = np.argsort(-sc)
    cands = [(str(ss[r]), int(sa[r]), int(sb[r]), float(sc[r]))
             for r in top[:64]]
    return lookup, cands
