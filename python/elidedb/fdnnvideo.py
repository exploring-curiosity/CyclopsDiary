"""FDNN-V: a video-native embedding encoder, distilled from SigLIP.

WHY THIS MODEL EXISTS
---------------------
The database must embed EVERY frame at write time. SigLIP cannot do that at
scale because it is a 428M-parameter model that treats each frame as an
unrelated photograph — measured 13-90 ms/frame on this machine. But video is
not a pile of photographs: frame t is almost entirely explained by frame t-1
(Deep Feature Flow, arXiv 1611.07715, built a whole recognition system on that
observation). A human watching a video does not re-parse the scene 30 times a
second either; they maintain a scene model and update it with what changed.

FDNN-V is that shape, made of the three FDNN rules:

  spatial glimpse (cheap, per frame)     "what is in front of me right now"
      Gabor-initialised first conv        — V1 simple cells ARE Gabor filters
      small conv pyramid                  — ventral stream
          |
  FDNN temporal core (rule 1)            "what has been going on"
      recurrent state h_t; each channel is a KAN-style sum over k
      heterogeneous sub-functions of (glimpse, state):
        FINER   variable-period oscillator — periodic motion (gait, wipers)
        Gabor   temporal wavelet           — bursts (a grasp, a brake light)
        poly    chirp                      — acceleration (pulling away)
      omega bands partition the temporal spectrum: slow = scene identity,
      mid = object motion, fast = transitions
          |
  head -> SigLIP space (1152-d)          so every existing index, text query,
                                          and centroid keeps working unchanged

  rules 2+3 (apoptosis -> fine-tune -> neurogenesis -> fine-tune, decided by
  PPO + reverse attention) run post-training in fdnnv_prune — and unlike the
  attempt to prune SigLIP itself, the fine-tune step EXISTS here, because
  distillation pairs are free: every frame of the corpus already has a teacher
  embedding in `frame_vectors`.

THE STREAMING CONTRACT
----------------------
`step(frame, h) -> (embedding, h')` is causal and O(1) per frame: no
lookahead, no window buffer. That is what makes embed-on-write real — the
encoder can sit inside the ingest loop and emit an embedding as each frame
arrives, like any other index maintenance.

WHAT DISTILLATION CAN AND CANNOT GIVE
-------------------------------------
The student lands in the teacher's embedding space, so text queries (embedded
by the frozen SigLIP text tower) keep working. It can match the teacher ON
THIS CORPUS's manifold; it is not a zero-shot model for arbitrary imagery.
That is the correct trade for a database: specialise the index to the data it
serves, keep the teacher for what it is — an offline labeller.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

EMBED_DIM = 1152          # SigLIP so400m space — compatibility is the point
INPUT_HW = (144, 192)     # decode width 192 -> 192x144 (h, w)


# ===========================================================================
# V1: Gabor-initialised first convolution
# ===========================================================================
def gabor_bank(n_filters, ksize, rng):
    """Oriented Gabor filters spanning orientation x frequency x phase.

    Not decoration: the first layer of every competent visual system —
    biological or learned — converges to oriented band-pass filters. Starting
    there instead of at noise removes the epochs a small model would spend
    rediscovering V1.
    """
    k = np.zeros((n_filters, ksize, ksize, 3), dtype=np.float32)
    half = ksize // 2
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
    for i in range(n_filters):
        theta = np.pi * (i % 8) / 8.0
        lam = ksize / (1.5 + (i // 8) % 3)
        psi = 0.0 if (i // 24) % 2 == 0 else np.pi / 2
        sigma = 0.4 * lam
        xr = xs * np.cos(theta) + ys * np.sin(theta)
        yr = -xs * np.sin(theta) + ys * np.cos(theta)
        g = np.exp(-(xr**2 + 0.8 * yr**2) / (2 * sigma**2)) \
            * np.cos(2 * np.pi * xr / lam + psi)
        g -= g.mean()
        g /= (np.abs(g).sum() + 1e-6)
        # colour-opponent weighting: some filters luminance, some R-G, B-Y
        cw = [(1, 1, 1), (1, -1, 0), (0.5, 0.5, -1)][i % 3]
        for c in range(3):
            k[i, :, :, c] = g * cw[c]
    k += rng.normal(0, 0.01, k.shape).astype(np.float32)
    return k


# ===========================================================================
# Ventral stream: small depthwise-separable pyramid
# ===========================================================================
class DWBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1,
                            groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1)
        self.norm = nn.LayerNorm(c_out)
        self.res = (c_in == c_out and stride == 1)

    def __call__(self, x):
        y = self.norm(self.pw(nn.silu(self.dw(x))))
        return x + y if self.res else y


class ConvStem(nn.Module):
    """Per-frame spatial encoder -> one glimpse vector. Small on purpose:
    fine discrimination is amortised into the temporal state, and capacity
    here is paid for EVERY frame FOREVER."""

    def __init__(self, width=48, out_dim=256, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        w = width
        self.v1 = nn.Conv2d(3, w, 7, stride=2, padding=3)
        self.v1.weight = mx.array(gabor_bank(w, 7, rng))
        self.s1 = DWBlock(w, w * 2, 2)
        self.s2 = DWBlock(w * 2, w * 4, 2)
        self.s3 = DWBlock(w * 4, w * 8, 2)
        self.r1 = DWBlock(w * 8, w * 8, 1)
        self.r2 = DWBlock(w * 8, w * 8, 1)
        self.proj = nn.Linear(w * 16, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def __call__(self, x):                      # (B, H, W, 3) in [-1, 1]
        h = nn.silu(self.v1(x))
        h = self.s1(h)
        h = self.s2(h)
        h = self.s3(h)
        h = self.r2(self.r1(h))
        mean = mx.mean(h, axis=(1, 2))
        peak = mx.max(h, axis=(1, 2))           # mean = layout, max = salient
        return self.norm(self.proj(mx.concatenate([mean, peak], axis=-1)))


class ViTStem(nn.Module):
    """The alternative stem: patchify + tiny transformer. Pure matmuls, which
    Apple-silicon GEMM kernels love; raced against ConvStem by measurement,
    never by taste."""

    def __init__(self, dim=192, depth=4, heads=3, out_dim=256,
                 patch=16, hw=INPUT_HW):
        super().__init__()
        self.patch = patch
        self.nh, self.nw = hw[0] // patch, hw[1] // patch
        self.embed = nn.Linear(patch * patch * 3, dim)
        self.pos = mx.zeros((1, self.nh * self.nw, dim))
        self.blocks = [_TinyBlock(dim, heads) for _ in range(depth)]
        self.proj = nn.Linear(dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def __call__(self, x):                      # (B, H, W, 3)
        B, H, W, _ = x.shape
        p = self.patch
        t = x.reshape(B, self.nh, p, self.nw, p, 3).transpose(0, 1, 3, 2, 4, 5)
        t = t.reshape(B, self.nh * self.nw, p * p * 3)
        t = self.embed(t) + self.pos
        for blk in self.blocks:
            t = blk(t)
        pooled = mx.concatenate([mx.mean(t, axis=1), mx.max(t, axis=1)],
                                axis=-1)
        return self.norm(self.proj(pooled))


class _TinyBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = nn.MultiHeadAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def __call__(self, x):
        y = self.n1(x)
        x = x + self.att(y, y, y)
        return x + self.fc2(nn.silu(self.fc1(self.n2(x))))


# ===========================================================================
# Rule 1: the temporal core — every channel is a sub-network
# ===========================================================================
class FDNNTemporalCell(nn.Module):
    """Recurrent state whose channels are KAN-style sums over k heterogeneous
    temporal sub-functions, gated GRU-style so state persists by default.

    The bases read the JOINT signal (current glimpse, previous state), so a
    FINER channel can oscillate with repeated motion, a Gabor channel can fire
    on a burst of change, and a poly-phase channel can track acceleration —
    while the gate decides how much of the old scene model each step is
    allowed to overwrite. This is FDNN's HybridBiomimeticLayer with time as
    the signal axis instead of a coordinate.
    """

    def __init__(self, in_dim=256, channels=256, k_width=4,
                 omega_bands=(0.8, 2.5, 8.0),
                 band_fractions=(0.34, 0.33, 0.33), seed=0):
        # omega bands are LOWER than the context tower's (2, 6, 18) on
        # purpose: that tower convolved over time feed-forward, this cell
        # FEEDS BACK. sin bases at omega 15 inside a 32-step recurrence give
        # chaotic gradients — measured: stage-2 loss climbed from 0.086 to
        # 0.112 and val fidelity fell 0.019. Same bases, calmer spectrum.
        super().__init__()
        rng = np.random.default_rng(seed)
        C, k = channels, k_width
        self.C, self.k = C, k

        omegas = []
        for om, fr in zip(omega_bands, band_fractions):
            omegas.extend([om] * int(round(fr * C)))
        omegas = (omegas + [omega_bands[-1]] * C)[:C]
        self.omegas_per_neuron = np.array(omegas, dtype=np.float32)
        self.omegas = mx.array(np.repeat(self.omegas_per_neuron, k))

        half, quarter = max(k // 2, 1), max(k // 4, 1)
        per = np.array([0] * half + [1] * quarter
                       + [3] * max(k - half - quarter, 0), np.int32)[:k]
        self.basis_types = mx.array(np.tile(per, C).astype(np.int32))
        # A categorical selector, not a weight: unfrozen, the optimizer
        # promotes it to float and drifts it off its exact values, silently
        # rerouting every Gabor neuron to the poly branch (measured on the
        # context tower: 1.0 -> 0.9992 and `== 1` matched nothing).
        self.freeze(keys=["basis_types"], recurse=False)

        mean_om = float(self.omegas_per_neuron.mean())
        lim_g = float(np.sqrt(6.0 / in_dim) / mean_om)
        lim_h = float(np.sqrt(6.0 / C) / mean_om)
        self.Wg = mx.array(rng.uniform(-lim_g, lim_g,
                                       (in_dim, C * k)).astype(np.float32))
        self.Wh = mx.array(rng.uniform(-lim_h, lim_h,
                                       (C, C * k)).astype(np.float32))
        self.b1 = mx.array(rng.uniform(-2.0, 2.0, (C * k,)).astype(np.float32))
        self.phases = mx.array(rng.uniform(0, 2 * np.pi,
                                           (C * k,)).astype(np.float32))
        self.gabor_s = mx.array(rng.uniform(0.3, 1.5,
                                            (C * k,)).astype(np.float32))
        log_om = np.log(np.clip(np.repeat(self.omegas_per_neuron, k),
                                1e-3, None))
        # alpha starts log-uniform between 1 and omega. With sub-unit omegas
        # (the calmed recurrent bands) log-omega is negative, so the interval
        # must be ordered explicitly — uniform(0, negative) is an error.
        self.log_alpha = mx.array(rng.uniform(np.minimum(0.0, log_om),
                                              np.maximum(0.0, log_om) + 1e-6
                                              ).astype(np.float32))
        w2s = float(np.sqrt(6.0 / (C * k)))
        self.w2 = mx.array(rng.uniform(-w2s, w2s, (C, k)).astype(np.float32))

        zlim = float(np.sqrt(6.0 / (in_dim + C)))
        self.Wz = mx.array(rng.uniform(-zlim, zlim,
                                       (in_dim, C)).astype(np.float32))
        self.Uz = mx.array(rng.uniform(-zlim, zlim,
                                       (C, C)).astype(np.float32))
        # Gate bias starts NEGATIVE: sigmoid(-1) ~ 0.27, so at init the state
        # persists — a scene model that forgets everything every frame is just
        # a per-frame model with extra steps.
        self.bz = mx.array(np.full((C,), -1.0, np.float32))

        self.mask = mx.array(np.ones((C,), np.float32))  # aliveness (rule 2)
        # Aliveness is set by the pruning cycle, never by the optimizer —
        # unfrozen, AdamW weight-decays it off 1.0 and every channel quietly
        # shrinks (the measured context-tower failure mode).
        self.freeze(keys=["mask"], recurse=False)

    def set_active_mask(self, m):
        self.mask = mx.array(np.asarray(m, dtype=np.float32))

    def _cand(self, g, h):
        pre = g @ self.Wg + h @ self.Wh + self.b1
        om_h = self.omegas * pre
        sq = pre * pre
        alpha = mx.exp(self.log_alpha)
        finer = mx.sin(self.omegas * (mx.abs(pre) + 1.0) * pre + self.phases)
        gab = mx.exp(-(self.gabor_s ** 2) * sq) * mx.sin(om_h + self.phases)
        sine = mx.sin(om_h + self.phases)
        poly = mx.sin(alpha * sq + om_h + self.phases)
        acts = mx.where(self.basis_types == 0, finer,
                        mx.where(self.basis_types == 1, gab,
                                 mx.where(self.basis_types == 2, sine, poly)))
        acts = acts.reshape(-1, self.C, self.k)
        return mx.sum(acts * self.w2, axis=-1)

    def neuron_outputs(self, g, h):
        """Per-neuron candidate BEFORE gate and mask — the pruning signal."""
        return self._cand(g, h)

    def __call__(self, g, h):
        cand = self._cand(g, h) * self.mask
        z = mx.sigmoid(g @ self.Wz + h @ self.Uz + self.bz) * self.mask
        return (1.0 - z) * h + z * cand


# ===========================================================================
# The encoder
# ===========================================================================
class FDNNVideoEncoder(nn.Module):
    def __init__(self, stem="conv", stem_width=48, glimpse=256, channels=256,
                 k_width=4, embed_dim=EMBED_DIM, vit_depth=4, seed=0):
        super().__init__()
        self.cfg = dict(stem=stem, stem_width=stem_width, glimpse=glimpse,
                        channels=channels, k_width=k_width,
                        embed_dim=embed_dim, vit_depth=vit_depth, seed=seed)
        if stem == "conv":
            self.stem = ConvStem(width=stem_width, out_dim=glimpse, seed=seed)
        else:
            self.stem = ViTStem(dim=stem_width * 4, depth=vit_depth,
                                out_dim=glimpse)
        self.cell = FDNNTemporalCell(in_dim=glimpse, channels=channels,
                                     k_width=k_width, seed=seed)
        self.head_g = nn.Linear(glimpse, embed_dim)
        # Temporal head starts at zero: at init the model IS the per-frame
        # model (stage 1), and training can only add information from state.
        # Same identity-safe discipline as every other init in this repo.
        self.head_h = nn.Linear(channels, embed_dim)
        self.head_h.weight = mx.zeros(self.head_h.weight.shape)
        self.head_h.bias = mx.zeros((embed_dim,))
        self.channels = channels

    # ---- streaming: this is the embed-on-write contract -------------------
    def init_state(self, batch=1):
        return mx.zeros((batch, self.channels))

    def step(self, frame, h):
        """One frame in, one embedding out, O(1) state carried. Causal."""
        g = self.stem(frame)
        h = self.cell(g, h)
        e = self.head_g(g) + self.head_h(h)
        return e * mx.rsqrt(mx.sum(e * e, axis=-1, keepdims=True) + 1e-8), h

    # ---- batched sequences (training / bulk ingest) -----------------------
    def __call__(self, seq, h0=None):
        """(B, T, H, W, 3) -> (B, T, D). Stem runs on all frames as one big
        batch (the GEMM-friendly part); only the tiny cell recurs."""
        B, T = seq.shape[0], seq.shape[1]
        g = self.stem(seq.reshape(B * T, *seq.shape[2:])).reshape(B, T, -1)
        h = self.init_state(B) if h0 is None else h0
        outs = []
        for t in range(T):
            h = self.cell(g[:, t], h)
            outs.append(h)
        hs = mx.stack(outs, axis=1)
        e = self.head_g(g) + self.head_h(hs)
        return e * mx.rsqrt(mx.sum(e * e, axis=-1, keepdims=True) + 1e-8), h

    def embed_frames_np(self, frames_u8, batch=64, chunk=None):
        """uint8 (N, H, W, 3) of ONE stream, in time order -> (N, 1152).
        Stateful across batches — one continuous pass over the stream."""
        h = self.init_state(1)
        out = []
        for i in range(0, len(frames_u8), batch):
            x = mx.array(frames_u8[i:i + batch].astype(np.float32)
                         / 127.5 - 1.0)[None]
            e, h = self(x, h0=h)
            out.append(np.array(e[0], dtype=np.float32))
        return np.concatenate(out, axis=0)


# ===========================================================================
# Distillation loss: pointwise + affinity mimicking
# ===========================================================================
def distill_loss(student, teacher, affinity_w=0.25, mu=None, centered_w=1.0,
                 anchors=None, anchor_w=50.0):
    """Distillation aimed at RETRIEVAL, not at raw closeness.

    Plain pointwise cosine is a trap on a homogeneous corpus: every teacher
    vector shares a huge common mode, so matching that alone buys ~0.9 cosine
    while scrambling the thin discriminative residual that ranking runs on.
    Measured: a student at fidelity 0.907 kept only 4.4% of the teacher's
    top-10 neighbours, while the teacher AGAINST ITSELF at a different input
    resolution — fidelity 0.921 — keeps 50.7%. Same closeness, 10x the
    retrieval agreement: the difference is WHERE the error lives.

    So three additional terms put the error where it does no harm:
      centered   cosine on (v - mu): the mean-free residual is exactly what
                 ranking compares, so it gets its own gradient.
      affinity   within-batch similarity matching (TinyCLIP, arXiv
                 2309.12314): preserve the teacher's ordering structure.
      anchors    similarity profile against real caption-text embeddings from
                 this store: text queries live in those directions, and
                 image-text sims occupy a band ~50x narrower than image-image
                 sims — hence the weight.
    """
    t = teacher * mx.rsqrt(mx.sum(teacher * teacher, axis=-1,
                                  keepdims=True) + 1e-8)
    loss = mx.mean(1.0 - mx.sum(student * t, axis=-1))
    if mu is not None and centered_w > 0:
        sc = student - mu
        tc = t - mu
        sc = sc * mx.rsqrt(mx.sum(sc * sc, axis=-1, keepdims=True) + 1e-8)
        tc = tc * mx.rsqrt(mx.sum(tc * tc, axis=-1, keepdims=True) + 1e-8)
        loss = loss + centered_w * mx.mean(1.0 - mx.sum(sc * tc, axis=-1))
    if affinity_w > 0:
        s2 = student.reshape(-1, student.shape[-1])
        t2 = t.reshape(-1, t.shape[-1])
        loss = loss + affinity_w * mx.mean(mx.square(s2 @ s2.T - t2 @ t2.T))
    if anchors is not None and anchor_w > 0:
        sa = student.reshape(-1, student.shape[-1]) @ anchors.T
        ta = t.reshape(-1, t.shape[-1]) @ anchors.T
        loss = loss + anchor_w * mx.mean(mx.square(sa - ta))
    return loss


# ===========================================================================
# persistence
# ===========================================================================
def save_encoder(model, meta, path):
    from mlx.utils import tree_flatten
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "weights.npz",
             **{k: np.array(v) for k, v in tree_flatten(model.parameters())})
    (path / "encoder.json").write_text(json.dumps(
        {**meta, "cfg": model.cfg}, indent=2))


def fdnnv_dir() -> Path:
    """Where the FDNN-V encoder lives.

    Repo-level `models/fdnnv`, NOT inside any store. The encoder used to
    sit at `lake/bridge/models/fdnnv`, so clearing the stores deleted a
    trained model along with the data (2026-07-28). A model is not store
    data. The legacy path is still accepted for stores that predate the
    move."""
    here = Path("models/fdnnv")
    if (here / "encoder.json").exists():
        return here
    legacy = Path("lake/bridge/models/fdnnv")
    if (legacy / "encoder.json").exists():
        return legacy
    return here


def load_encoder(path):
    from mlx.utils import tree_unflatten
    path = Path(path)
    meta = json.loads((path / "encoder.json").read_text())
    model = FDNNVideoEncoder(**meta["cfg"])
    z = np.load(path / "weights.npz")
    model.update(tree_unflatten([(k, mx.array(z[k])) for k in z.files]))
    model.cell.freeze(keys=["basis_types", "mask"], recurse=False)
    mx.eval(model.parameters())
    return model, meta


# ===========================================================================
# The write path: chunked byte-range decode feeding the streaming encoder
# ===========================================================================
def embed_stream(store, model, rows, width=192, chunk=512, batch_cb=None):
    """Embed one stream's frames in time order, state carried across chunks.

    `chunk` bounds the decoder subprocess's rawvideo buffer (~1 GB at 512
    frames of 640x480); the encoder state flows straight through, so the
    result is identical to one infinite pass. This loop is the write path:
    ingest can call it as frames land.
    Returns (ts int64 array, vectors float32 (N, D), decode_s, embed_s).
    """
    import time as _time

    from .video import FrameSet
    ts_out, vecs = [], []
    h = model.init_state(1)
    dec_s = emb_s = 0.0
    for i in range(0, len(rows), chunk):
        t0 = _time.perf_counter()
        dec = FrameSet(store, "frames", rows.slice(i, chunk)).decode(
            width=width)
        dec_s += _time.perf_counter() - t0
        if not dec:
            continue
        frames = np.stack([d[1] for d in dec])
        # the stem takes EXACTLY (in_h, in_w); sources with a different
        # aspect ratio decode to other shapes (lab video came back square
        # and crashed the reshape). Stretch — the encoder was distilled on
        # stretched frames, so aspect distortion is in-distribution.
        ih, iw = model.cfg["in_hw"] if "in_hw" in model.cfg else (144, 192)
        if frames.shape[1] != ih or frames.shape[2] != iw:
            xr = np.linspace(0, frames.shape[2] - 1, iw).round().astype(int)
            yr = np.linspace(0, frames.shape[1] - 1, ih).round().astype(int)
            frames = frames[:, yr][:, :, xr]
        t0 = _time.perf_counter()
        x = mx.array(frames.astype(np.float32) / 127.5 - 1.0)[None]
        e, h = model(x, h0=h)
        e = np.array(e[0], dtype=np.float32)
        emb_s += _time.perf_counter() - t0
        ts_out.extend(d[0] for d in dec)
        vecs.append(e)
        if batch_cb:
            batch_cb(len(ts_out))
    if not vecs:
        return np.array([], np.int64), np.zeros((0, EMBED_DIM), np.float32), \
            dec_s, emb_s
    return (np.array(ts_out, np.int64), np.concatenate(vecs), dec_s, emb_s)
