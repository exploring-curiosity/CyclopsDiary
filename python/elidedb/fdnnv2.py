"""FDNN-V2: the context head. Same speed class, different training signal.

V1's measured dead end: distilled to an appearance teacher whose own space
separates close/open at 0.60, no student can learn verbs — the target lacks
them. V2 keeps V1's skeleton (stem -> FDNN recurrent cell, 0.27 ms/frame)
and warm-starts from its weights; what changes is WHERE the gradient comes
from (docs/FDNNV2_PLAN.md, all research-grounded):

  L1 PREDICTIVE (V-JEPA-style, arXiv 2506.09985): from state h_t, predict
     the embedding of frame t+k at two horizons. The future is free
     supervision on every frame, and dynamics must enter h to predict it.
     This is the biological claim made mechanical: understanding = a scene
     model good enough to predict what happens next.
  L2 ARROW OF TIME (Wei et al., CVPR 2018): classify forward vs reversed
     clips from the final state. Open and close are time-reversals; one
     binary head separates them by construction.
  L3 VERB-FOCUSED CONTRASTIVE (arXiv 2304.06708): sigmoid contrastive
     between clip context embeddings and 7B captions, with hard negatives
     built by verb/direction swaps — the joint space queries actually live
     in. A 2-layer text adapter maps SigLIP text embeddings into the
     256-d context space (~0.1 ms per query).

The appearance head keeps its V1 distillation target so every existing
index and the SigLIP text tower continue to work unchanged.

THE GATE IS AN EVENT DETECTOR. The cell's update gate z_t measures how much
of the scene model this frame is allowed to overwrite. Sustained high gate
activity marks an event boundary — so the encoder emits event segmentation
for free, and event embeddings are pooled with NOVELTY WEIGHTS (gate
activity), not uniformly: the moment the drawer closes outweighs the seconds
it sat still (the TempMe lesson, arXiv 2409.01156, applied at pooling time).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .fdnnvideo import EMBED_DIM, FDNNVideoEncoder

CTX_DIM = 256

# Verb/direction swaps for L3 hard negatives: shared lexicon (moved to
# lexicon.py so the QUERY path can import it without this module's mlx)
from .lexicon import derived_swaps  # noqa: E402,F401


def swap_verbs(text: str, rng, store=None) -> str | None:
    """One randomly chosen applicable swap -> a hard negative. None if no
    swap applies (caption has no directional content to invert)."""
    # Training negatives obey the no-hardwire rule too: a model taught
    # from hand-written oppositions has the hand-writing baked into its
    # weights, which is worse than a lookup table because it cannot be
    # grepped out afterwards.
    pairs = list(derived_swaps(store)) if store is not None else []
    if not pairs:
        return None
    t = " " + text.lower() + " "
    hits = []
    for a, b in pairs:
        if f" {a} " in t:
            hits.append((a, b))
        if f" {b} " in t:
            hits.append((b, a))
    if not hits:
        # clause order flip is the fallback inversion for compounds
        parts = re.split(r"\band then\b|\bthen\b|,", text)
        if len(parts) >= 2:
            return " then ".join(p.strip() for p in reversed(parts)
                                 if p.strip())
        return None
    a, b = hits[rng.integers(len(hits))]
    return re.sub(rf"\b{re.escape(a)}\b", b, text.lower(), count=1)


class TextAdapter(nn.Module):
    """SigLIP text embedding (1152) -> context space (256). Two layers,
    ~0.7M params, ~0.1 ms — the entire query-time cost of verb awareness."""

    def __init__(self, in_dim=EMBED_DIM, dim=CTX_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, dim)

    def __call__(self, x):
        e = self.fc2(nn.silu(self.fc1(x)))
        return e * mx.rsqrt(mx.sum(e * e, axis=-1, keepdims=True) + 1e-8)


class FDNNv2(nn.Module):
    """V1 skeleton + context head + predictors + arrow-of-time head."""

    def __init__(self, base: FDNNVideoEncoder, ctx_dim=CTX_DIM,
                 horizons=(2, 10), motion_dim=64):
        super().__init__()
        self.base = base
        C = base.channels
        G = base.cfg["glimpse"]
        self.horizons = list(horizons)
        self.motion_dim = motion_dim
        # MOTION PATHWAY — the reversal-symmetry breaker. The gated cell is a
        # leaky integrator; for smooth inputs an EMA is nearly order-invariant,
        # and stage-A measured exactly that: AoT stuck at chance (0.47) while
        # dominating the loss. Frame-difference features flip SIGN under time
        # reversal, so direction becomes linearly readable. Velocity is not a
        # nicety here; it is the only anti-symmetric signal in the model.
        self.motion = nn.Linear(G, motion_dim)
        # context head reads (glimpse, state, motion)
        self.head_ctx = nn.Linear(G + C + motion_dim, ctx_dim)
        # Predictors target the model's OWN future glimpse-change (stop-grad).
        # A2 measured why not student-embedding changes: adjacent-frame true
        # change is ~0.2 in norm while the student's own error is ~0.45 — the
        # difference of noisy embeddings is noise, and the predictors sat at
        # cosine 0 for 8 epochs. Own-latent prediction is JEPA's actual
        # recipe; the appearance anchor stops the stem collapsing to constants.
        self.pred = [nn.Linear(C + motion_dim, G) for _ in horizons]
        # AoT reads the velocity SEQUENCE through a temporal conv, not the
        # mean: this corpus is reciprocal motion (arm out, arm back), so mean
        # signed velocity cancels — measured at chance twice. Order within
        # the window is the signal; a conv kernel can be asymmetric in time.
        self.aot_conv = nn.Conv1d(motion_dim, 32, 5, padding=2)
        self.head_aot = nn.Linear(32, 1)
        self.cfg = {"ctx_dim": ctx_dim, "horizons": self.horizons,
                    "motion_dim": motion_dim, "base": base.cfg}

    # ---- forward over a sequence, exposing everything the losses and the
    # event segmenter need ---------------------------------------------------
    def run(self, seq, h0=None):
        """(B,T,H,W,3) -> dict of app (B,T,1152), ctx (B,T,ctx), preds,
        gates (B,T), h_last."""
        b = self.base
        B, T = seq.shape[0], seq.shape[1]
        g = b.stem(seq.reshape(B * T, *seq.shape[2:])).reshape(B, T, -1)
        h = b.init_state(B) if h0 is None else h0
        hs, gates = [], []
        cell = b.cell
        for t in range(T):
            z = mx.sigmoid(g[:, t] @ cell.Wz + h @ cell.Uz + cell.bz)
            h = cell(g[:, t], h)
            hs.append(h)
            gates.append(mx.mean(z, axis=-1))
        hs = mx.stack(hs, axis=1)                    # (B,T,C)
        gates = mx.stack(gates, axis=1)              # (B,T)
        app = b.head_g(g) + b.head_h(hs)
        app = app * mx.rsqrt(mx.sum(app * app, axis=-1, keepdims=True) + 1e-8)
        # signed velocity of the glimpse; first frame gets zero motion.
        # NORMALISED: measured, ||dg|| is real motion (corr 0.63 with pixel
        # motion, 5x on moving frames) but only ~2% of the feature norm — fed
        # raw, it drowned next to signals 50x larger and every motion head
        # starved. Direction is unit-normalised; magnitude re-enters as a
        # bounded gain, so both the WHAT and the HOW-MUCH of motion survive.
        dg = mx.concatenate([mx.zeros_like(g[:, :1]),
                             g[:, 1:] - g[:, :-1]], axis=1)
        mag = mx.sqrt(mx.sum(dg * dg, axis=-1, keepdims=True) + 1e-8)
        mfeat = nn.silu(self.motion(dg / mag)) * mx.tanh(mag)
        cat = mx.concatenate([g, hs, mfeat], axis=-1)
        ctx = self.head_ctx(cat)
        ctx = ctx * mx.rsqrt(mx.sum(ctx * ctx, axis=-1, keepdims=True) + 1e-8)
        hm = mx.concatenate([hs, mfeat], axis=-1)
        preds = [p(hm) for p in self.pred]           # each (B,T,G)
        a = nn.silu(self.aot_conv(mfeat))            # (B,T,32) time-conv
        aot = self.head_aot(mx.mean(a, axis=1))[:, 0]
        return {"app": app, "ctx": ctx, "preds": preds, "gates": gates,
                "aot": aot, "h": h, "g": g}

    # ---- streaming embed (the write path): app + ctx + gate per frame -----
    def embed_stream_np(self, frames_u8, batch=128):
        h = self.base.init_state(1)
        apps, ctxs, gates = [], [], []
        for i in range(0, len(frames_u8), batch):
            x = mx.array(frames_u8[i:i + batch].astype(np.float32)
                         / 127.5 - 1.0)[None]
            out = self.run(x, h0=h)
            h = out["h"]
            apps.append(np.array(out["app"][0], np.float32))
            ctxs.append(np.array(out["ctx"][0], np.float32))
            gates.append(np.array(out["gates"][0], np.float32))
        return (np.concatenate(apps), np.concatenate(ctxs),
                np.concatenate(gates))


# ===========================================================================
# Event segmentation from the gate signal + novelty-weighted pooling
# ===========================================================================
def segment_events(gates, ts, min_len=6, smooth=5, thresh_pct=75.0):
    """Gate-activity peaks -> event boundaries.

    A boundary is where smoothed gate activity crosses above its own
    percentile threshold after having been below — the scene model is being
    rewritten. Percentile (not absolute) because gate scale is a trained
    quantity; per-stream calibration is free.
    Returns list of (start_idx, end_idx) covering the stream.
    """
    g = np.convolve(gates, np.ones(smooth) / smooth, mode="same")
    thr = np.percentile(g, thresh_pct)
    above = g > thr
    bounds = [0]
    for i in range(1, len(g)):
        if above[i] and not above[i - 1] and i - bounds[-1] >= min_len:
            bounds.append(i)
    bounds.append(len(g))
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


def pool_event(vecs, gates, lo, hi):
    """Novelty-weighted pool: frames weighted by gate activity, so change
    dominates stillness. Uniform mean is the verb-eraser; this is not."""
    w = gates[lo:hi] + 1e-3
    w = w / w.sum()
    v = (vecs[lo:hi] * w[:, None]).sum(0)
    return v / (np.linalg.norm(v) + 1e-8)


# ===========================================================================
# persistence
# ===========================================================================
def save_v2(model, adapter, meta, path):
    from mlx.utils import tree_flatten
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "v2.npz",
             **{k: np.array(v) for k, v in tree_flatten(model.parameters())})
    np.savez(path / "adapter.npz",
             **{k: np.array(v)
                for k, v in tree_flatten(adapter.parameters())})
    (path / "v2.json").write_text(json.dumps({**meta, "cfg": model.cfg},
                                             indent=2))


def load_v2(path):
    from mlx.utils import tree_unflatten
    path = Path(path)
    meta = json.loads((path / "v2.json").read_text())
    base = FDNNVideoEncoder(**meta["cfg"]["base"])
    model = FDNNv2(base, ctx_dim=meta["cfg"]["ctx_dim"],
                   horizons=tuple(meta["cfg"]["horizons"]),
                   motion_dim=meta["cfg"].get("motion_dim", 64))
    z = np.load(path / "v2.npz")
    model.update(tree_unflatten([(k, mx.array(z[k])) for k in z.files]))
    model.base.cell.freeze(keys=["basis_types", "mask"], recurse=False)
    adapter = TextAdapter()
    z = np.load(path / "adapter.npz")
    adapter.update(tree_unflatten([(k, mx.array(z[k])) for k in z.files]))
    mx.eval(model.parameters(), adapter.parameters())
    return model, adapter, meta
