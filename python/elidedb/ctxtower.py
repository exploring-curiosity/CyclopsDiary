"""The context tower — a custom SigLIP video tower over frozen frame features.

WHAT IT IS
----------
SigLIP's image tower is a frozen spatial encoder. This module supplies the
*other* half of an R(2+1)D-style factorisation: a learned TEMPORAL tower that
consumes the sequence of per-frame SigLIP vectors in a window and emits one
vector aligned with SigLIP TEXT embeddings.

    frame vectors (T, 1152)          [frozen SigLIP image tower — the "2D"]
        |  PCA  (frozen, no params)
    (T, d)
        |  dilated temporal conv, taps learned      ← the CNN
        |  each output neuron = KAN sum over k
        |  heterogeneous bases (FINER/Gabor/poly)   ← FDNN thesis 1
    (T, n)
        |  bi-GRU                                    ← the RNN
    (T, 2h)
        |  attention pooling over time
    (2h,)
        |  head -> PCA^T of the caption space
    (1152,)  ~ a SigLIP text embedding

WHY A CONV *AND* A GRU
----------------------
They fail differently, which is the only good reason to have both. The dilated
conv is order-aware but translation-equivariant: it detects "a thing moved
left-to-right" wherever in the window it happens, with a fixed receptive
field. The GRU is order-aware and unbounded: it can carry "the car was
stationary the whole time, then braked" across the window, which no
fixed-width kernel expresses. CLIP4Clip (arXiv 2104.08860) measured exactly
this axis on frozen CLIP features — meanP vs seqLSTM vs seqTransf — and found
learned temporal aggregation helps once the target actually depends on order.
Mean pooling, what `embeddings` does today, is the degenerate case of both.

TSM (Lin et al., ICCV 2019) was the alternative for the conv slot: shifting
channels along time is free. It was not used because its shift is a fixed
±1 tap; here the whole point is that the taps are *frequency-selective*, and
a fixed shift cannot express that.

WHY THE BASES ARE THE FDNN BASES
--------------------------------
A window's feature trajectory is a signal, and these bases were built for
signals. Applied after a temporal convolution:
  - a Gabor sub-function is a temporal wavelet — a burst detector, localised
    in time (a door opening, a brake light);
  - a FINER sub-function is a variable-period oscillator — periodic motion
    (gait, wipers, a turning wheel);
  - a polynomial-phase sub-function is a chirp — monotonic acceleration
    (a vehicle pulling away, a zoom).
The frequency bands then partition the TEMPORAL spectrum instead of every
neuron competing for it: slow = scene identity, mid = object motion, fast =
transitions. Each neuron being a sum over k of these is FDNN thesis 1, and
here it buys genuine expressiveness rather than decoration.

IDENTITY-SAFE INIT
------------------
The head is initialised so the tower emits the mean caption vector for every
window. That is "I know nothing" — a legal, centred prediction — so training
can only add information, and a half-trained tower can never be worse than
the corpus prior. Same discipline as FDNN's identity-init residual.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ===========================================================================
# Temporal hybrid layer — FDNN's HybridBiomimeticLayer with a time axis
# ===========================================================================
class TemporalHybridLayer(nn.Module):
    """Dilated temporal conv whose output channels are KAN sub-network neurons.

    Shapes: (B, T, in_d) -> (B, T, max_out_d).

    The convolution and the sub-function bank are fused: the conv produces
    `max_out_d * k` pre-activations per timestep, the bases are applied
    elementwise, and the k sub-functions of each neuron are summed (KAN).
    So one neuron is not "a channel" — it is a little ensemble of temporal
    filters that disagree about what shape of motion to look for.

    `omega_bands` are TEMPORAL frequencies here. They are an order of
    magnitude smaller than FDNN's coordinate-network defaults on purpose: the
    input is a PCA of unit-norm embeddings, so |h| is O(1) rather than O(100),
    and reusing omega=200 would put every neuron in the chaotic regime where
    gradients are noise.
    """

    def __init__(self, in_d, max_out_d, initial_active=None, k_width=4,
                 kernel=3, dilation=1, omega_bands=(2.0, 6.0, 18.0),
                 band_fractions=(0.34, 0.33, 0.33), bias_range=2.0,
                 use_residual=True, dropout_p=0.0, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.in_d = in_d
        self.max_out_d = max_out_d
        self.k = k_width
        self.kernel = kernel
        self.dilation = dilation
        self.bias_range = bias_range
        self.use_residual = use_residual and (in_d == max_out_d)
        self.dropout_p = dropout_p
        self._training = True

        # ---- frequency-banded omega per neuron (temporal spectrum split) ----
        omegas = []
        for om, fr in zip(omega_bands, band_fractions):
            omegas.extend([om] * int(round(fr * max_out_d)))
        omegas = (omegas + [omega_bands[-1]] * max_out_d)[:max_out_d]
        self.omegas_per_neuron = np.array(omegas, dtype=np.float32)
        om_exp = np.repeat(self.omegas_per_neuron, k_width)
        self.omegas = mx.array(om_exp)

        # ---- heterogeneous basis assignment: 50% FINER, 25% Gabor, 25% poly
        half, quarter = max(k_width // 2, 1), max(k_width // 4, 1)
        per = np.array([0] * half + [1] * quarter
                       + [3] * max(k_width - half - quarter, 0),
                       dtype=np.int32)[:k_width]
        if per.size < k_width:
            per = np.concatenate([per, np.zeros(k_width - per.size, np.int32)])
        self.basis_types = mx.array(np.tile(per, max_out_d).astype(np.int32))

        # ---- the temporal convolution: (kernel, in_d, max_out_d*k) ----------
        # Fan-in is kernel*in_d, and the SIREN convention divides by omega so
        # the pre-activation lands in the basis's useful range at init.
        mean_om = float(self.omegas_per_neuron.mean())
        limit = float(np.sqrt(6.0 / (kernel * in_d)) / mean_om)
        self.w1 = mx.array(rng.uniform(
            -limit, limit, (kernel, in_d, max_out_d * k_width)).astype(np.float32))
        self.b1 = mx.array(rng.uniform(
            -bias_range, bias_range, (max_out_d * k_width,)).astype(np.float32))
        self.phases = mx.array(rng.uniform(
            0, 2 * np.pi, (max_out_d * k_width,)).astype(np.float32))
        self.gabor_s = mx.array(rng.uniform(
            0.3, 1.5, (max_out_d * k_width,)).astype(np.float32))
        log_om = np.log(np.clip(om_exp, 1e-3, None))
        self.log_alpha = mx.array(
            rng.uniform(0.0, log_om).astype(np.float32))

        w2s = float(np.sqrt(6.0 / (max_out_d * k_width)))
        self.w2 = mx.array(rng.uniform(
            -w2s, w2s, (max_out_d, k_width)).astype(np.float32))

        m = np.zeros((max_out_d,), dtype=np.float32)
        m[:(initial_active or max_out_d)] = 1.0
        self.mask = mx.array(m)

        # basis_types is a CATEGORICAL SELECTOR, not a weight. Every mx.array
        # attribute joins the parameter tree, so without this the optimizer
        # takes gradient steps on it: int32 gets promoted to float32 and the
        # values drift (measured: 1.0 -> 0.9992). Dispatch is
        # `where(basis_types == 1, gabor, ...)`, so a drifted selector matches
        # nothing and every Gabor neuron silently falls through to the poly
        # branch — the heterogeneous basis quietly stops being heterogeneous.
        # Freezing keeps it out of trainable_parameters() entirely.
        self.freeze(keys=["basis_types"], recurse=False)

    # -- temporal gather: centred taps so the window is read bidirectionally --
    def _conv(self, x):
        """x: (B, T, in_d) -> (B, T, max_out_d*k)."""
        T = x.shape[1]
        off0 = (self.kernel - 1) // 2
        acc = None
        for t_i in range(self.kernel):
            shift = (t_i - off0) * self.dilation
            if shift == 0:
                xs = x
            elif shift > 0:                       # look forward, edge-pad tail
                xs = mx.concatenate(
                    [x[:, shift:, :],
                     mx.repeat(x[:, -1:, :], shift, axis=1)], axis=1)
            else:                                  # look back, edge-pad head
                s = -shift
                xs = mx.concatenate(
                    [mx.repeat(x[:, :1, :], s, axis=1),
                     x[:, :T - s, :]], axis=1)
            term = xs @ self.w1[t_i]
            acc = term if acc is None else acc + term
        return acc + self.b1

    def _bases(self, h):
        omega_h = self.omegas * h
        h_sq = h * h
        alpha = mx.exp(self.log_alpha)
        finer = mx.sin(self.omegas * (mx.abs(h) + 1.0) * h + self.phases)
        gabor = mx.exp(-(self.gabor_s ** 2) * h_sq) * mx.sin(omega_h + self.phases)
        sine = mx.sin(omega_h + self.phases)
        poly = mx.sin(alpha * h_sq + omega_h + self.phases)
        return mx.where(self.basis_types == 0, finer,
                        mx.where(self.basis_types == 1, gabor,
                                 mx.where(self.basis_types == 2, sine, poly)))

    def neuron_outputs(self, x):
        """(B, T, max_out_d) per-neuron signal BEFORE mask and residual.
        Deterministic regardless of mode — every pruning metric reads this."""
        acts = self._bases(self._conv(x))
        acts = acts.reshape(acts.shape[0], acts.shape[1], self.max_out_d, self.k)
        return mx.sum(acts * self.w2, axis=-1)

    def set_active_mask(self, mask_np):
        self.mask = mx.array(np.asarray(mask_np, dtype=np.float32))

    def __call__(self, x):
        acts = self._bases(self._conv(x))
        acts = acts.reshape(acts.shape[0], acts.shape[1], self.max_out_d, self.k)
        if self._training and self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            drop = (mx.random.uniform(0.0, 1.0, acts.shape) < keep).astype(acts.dtype)
            acts = acts * drop / keep
        out = mx.sum(acts * self.w2, axis=-1) * self.mask
        return out + x if self.use_residual else out


# ===========================================================================
# Attention pooling over time
# ===========================================================================
class AttnPool(nn.Module):
    """Learned soft-argmax over the window.

    Mean pooling says every instant matters equally, which is wrong for a
    driving clip where 1.8 s is empty road and 0.2 s is the pedestrian
    stepping off the kerb. Attention lets the tower spend its output on the
    part of the window that carries the event.
    """

    def __init__(self, d, hidden=32):
        super().__init__()
        self.proj = nn.Linear(d, hidden)
        self.score = nn.Linear(hidden, 1)

    def __call__(self, h):                       # (B, T, d) -> (B, d)
        a = self.score(mx.tanh(self.proj(h)))    # (B, T, 1)
        w = mx.softmax(a, axis=1)
        return mx.sum(h * w, axis=1), w[..., 0]


# ===========================================================================
# The tower
# ===========================================================================
class ContextTower(nn.Module):
    def __init__(self, d_in=48, n_hidden=48, k_width=4, gru_hidden=32,
                 d_out=32, kernel=3, dilations=(1, 2), dropout_p=0.1,
                 input_noise_std=0.02, omega_bands=(2.0, 6.0, 18.0), seed=0):
        super().__init__()
        self.cfg = dict(d_in=d_in, n_hidden=n_hidden, k_width=k_width,
                        gru_hidden=gru_hidden, d_out=d_out, kernel=kernel,
                        dilations=list(dilations), dropout_p=dropout_p,
                        input_noise_std=input_noise_std,
                        omega_bands=list(omega_bands), seed=seed)
        self.input_noise_std = input_noise_std
        self._training = True

        self.blocks = []
        for i, dil in enumerate(dilations):
            self.blocks.append(TemporalHybridLayer(
                in_d=d_in if i == 0 else n_hidden,
                max_out_d=n_hidden, k_width=k_width, kernel=kernel,
                dilation=dil, omega_bands=omega_bands,
                use_residual=(i > 0), dropout_p=dropout_p, seed=seed + i))
        self.gru_f = nn.GRU(n_hidden, gru_hidden)
        self.gru_b = nn.GRU(n_hidden, gru_hidden)
        self.pool = AttnPool(n_hidden + 2 * gru_hidden)
        self.head = nn.Linear(n_hidden + 2 * gru_hidden, d_out)
        # Head starts at exactly zero: the tower's first prediction is the
        # corpus prior (see module docstring). Bias is zero too — the prior
        # itself is added outside, in the frozen decode basis.
        self.head.weight = mx.zeros(self.head.weight.shape)
        self.head.bias = mx.zeros((d_out,))
        self.log_gamma = mx.array(np.float32(np.log(0.5)))

    def set_training(self, mode: bool):
        self._training = mode
        for b in self.blocks:
            b._training = mode
        return self

    def turnover_layers(self):
        return list(self.blocks)

    def trunk(self, x):
        """(B, T, d_in) -> (B, n_hidden + 2*gru_hidden) pooled features."""
        if self._training and self.input_noise_std > 0:
            x = x + mx.random.normal(x.shape) * self.input_noise_std
        h = x
        for b in self.blocks:
            h = b(h)
        f = self.gru_f(h)
        b_rev = self.gru_b(h[:, ::-1, :])[:, ::-1, :]
        cat = mx.concatenate([h, f, b_rev], axis=-1)
        pooled, attn = self.pool(cat)
        # NOTE: deliberately not stashed on `self`. An mx.array assigned to a
        # Module attribute joins the parameter tree, and the optimizer then
        # tries to take a step on it (KeyError: 'last_attn'). Use
        # `attention_weights()` when you want to inspect it.
        return pooled

    def attention_weights(self, x):
        """(B, T) softmax weights — which instants the tower spent itself on."""
        h = x
        for b in self.blocks:
            h = b(h)
        f = self.gru_f(h)
        b_rev = self.gru_b(h[:, ::-1, :])[:, ::-1, :]
        return self.pool(mx.concatenate([h, f, b_rev], axis=-1))[1]

    def __call__(self, x):
        """-> (B, d_out) latent coordinates in the caption PCA basis."""
        return self.head(self.trunk(x)) * mx.exp(self.log_gamma)


# ===========================================================================
# Codec: frozen PCA in, frozen PCA out. Zero learned parameters either side.
# ===========================================================================
class ContextCodec:
    """Frozen PCA on the INPUT side of the tower.

    We have a few hundred labelled windows. A learned 1152->d encoder would be
    ~150k parameters fitted from ~150 examples, which is not learning, it is
    memorising. PCA is the optimal linear compressor under reconstruction
    error and costs zero training examples, so all the sample budget goes to
    the temporal dynamics — the only part mean pooling cannot already do.

    The OUTPUT side is not a PCA of SigLIP text embeddings. That was measured
    and it lost: ranking a query against a predicted caption *embedding*
    scored +0.218 against the VLM judge, below plain appearance search at
    +0.276, because SigLIP is trained for image-text similarity and its text
    tower is not calibrated for text-to-text comparison. The output space is
    now `CaptionSpace` — LSA over the caption text itself — which measured
    +0.339. See docs/CONTEXT.md.
    """

    def __init__(self, P_in, mu_in):
        self.P_in, self.mu_in = P_in, mu_in       # (d_in, D), (D,)

    @staticmethod
    def fit(frame_vecs, d_in=48):
        from sklearn.decomposition import PCA
        d_in = min(d_in, *frame_vecs.shape)
        pi = PCA(n_components=d_in, random_state=0).fit(frame_vecs)
        return ContextCodec(pi.components_.astype(np.float32),
                            pi.mean_.astype(np.float32))

    def encode(self, seq):                        # (T, D) -> (T, d_in)
        return (seq - self.mu_in) @ self.P_in.T

    def save(self, path):
        np.savez(path, P_in=self.P_in, mu_in=self.mu_in)

    @staticmethod
    def load(path):
        z = np.load(path)
        return ContextCodec(z["P_in"], z["mu_in"])


# ===========================================================================
# SigLIP's own loss
# ===========================================================================
def siglip_loss(v, u, ignore=None, log_t=None, bias=None):
    """Pairwise sigmoid loss (Zhai et al., ICCV 2023, arXiv 2303.15343).

    Chosen over softmax-InfoNCE for one concrete reason: the paper's own
    ablation shows sigmoid wins below ~16k batch, and our batch is the whole
    labelled set — a couple of hundred. Softmax normalises over the batch, so
    at this size the partition function is estimated from almost nothing.
    Sigmoid treats every pair as an independent binary problem and never
    needs that global view.

    `ignore` masks pairs that must not be counted as negatives. Windows slide
    with 75% overlap, so window i+1 genuinely depicts the same moment as
    window i; calling it a negative would teach the tower to separate
    identical content. Those pairs are dropped, not down-weighted.
    """
    t = mx.exp(log_t)
    logits = t * (v @ u.T) + bias
    n = v.shape[0]
    y = 2.0 * mx.eye(n) - 1.0                     # +1 diagonal, -1 elsewhere
    z = y * logits
    # -log sigmoid(z) == softplus(-z), stable for both signs
    per_pair = nn.softplus(-z)
    if ignore is not None:
        keep = 1.0 - ignore
        return mx.sum(per_pair * keep) / mx.maximum(mx.sum(keep), 1.0)
    return mx.mean(per_pair)


def overlap_mask(windows):
    """1.0 where two windows are the same stream and overlap in time."""
    n = len(windows)
    m = np.zeros((n, n), dtype=np.float32)
    st = [w[0] for w in windows]
    t0 = np.array([w[1] for w in windows], dtype=np.int64)
    t1 = np.array([w[2] for w in windows], dtype=np.int64)
    for i in range(n):
        same = np.array([s == st[i] for s in st])
        ov = same & (t0 < t1[i]) & (t1 > t0[i])
        m[i] = ov.astype(np.float32)
    np.fill_diagonal(m, 0.0)                      # the positive stays a positive
    return m


# ===========================================================================
# Training
# ===========================================================================
def train_tower(seqs, targets, windows, val_idx, cfg=None, epochs=400,
                lr=3e-3, weight_decay=1e-3, align_w=0.3, verbose=True,
                seed=0):
    """Fit the tower. `seqs` are (T, d_in) encoded sequences, `targets` are
    (d_out,) caption coordinates in the output basis, both already codec-mapped.

    Returns (model, history). Validation is a TIME split supplied by the
    caller — never a random split, because 75%-overlapping windows would put
    near-duplicates on both sides and report a fantasy score.
    """
    import mlx.optimizers as optim
    cfg = cfg or {}
    mx.random.seed(seed)
    model = ContextTower(**cfg)

    T = max(s.shape[0] for s in seqs)
    X = np.zeros((len(seqs), T, seqs[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(seqs):                  # edge-pad short windows
        X[i, :len(s)] = s
        if len(s) < T:
            X[i, len(s):] = s[-1]
    Y = np.asarray(targets, dtype=np.float32)

    val = np.zeros(len(seqs), dtype=bool)
    val[val_idx] = True
    tr = ~val
    Xtr, Ytr = mx.array(X[tr]), mx.array(Y[tr])
    Xva, Yva = mx.array(X[val]), mx.array(Y[val])
    ig_tr = mx.array(overlap_mask([w for w, m in zip(windows, tr) if m]))
    ig_va = mx.array(overlap_mask([w for w, m in zip(windows, val) if m]))

    # logit scale/bias are learned, initialised as in the SigLIP paper
    state = {"log_t": mx.array(np.float32(np.log(10.0))),
             "bias": mx.array(np.float32(-10.0))}

    def _norm(a):
        # eps INSIDE the sqrt, not added to the norm afterwards. The head is
        # zero-initialised on purpose, so the very first forward pass produces
        # an exactly-zero vector — and d||x||/dx = x/||x|| is 0/0 = NaN there.
        # Smoothing the radicand keeps the gradient finite at the origin.
        return a * mx.rsqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-8)

    def loss_fn(m, x, y, ig):
        v = _norm(m(x))
        u = _norm(y)
        # alignment term: pull each prediction onto its own caption. The
        # contrastive term only fixes ORDER; this one fixes absolute position,
        # which is what makes the vectors usable against unseen query text.
        align = mx.mean(1.0 - mx.sum(v * u, axis=-1))
        return siglip_loss(v, u, ignore=ig, log_t=state["log_t"],
                           bias=state["bias"]) + align_w * align

    opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    lg = nn.value_and_grad(model, loss_fn)
    sopt = optim.Adam(learning_rate=lr)

    def state_loss(s):
        model.set_training(False)
        v = _norm(model(Xtr))
        return siglip_loss(v, _norm(Ytr), ignore=ig_tr,
                           log_t=s["log_t"], bias=s["bias"])

    # The aliveness masks are mx.array attributes, so they live in
    # model.parameters() and AdamW would happily weight-decay them off their
    # 0/1 values — silently rescaling every neuron and breaking the
    # `mask == 1.0` alive-check the pruner depends on. Snapshot and restore
    # after every step. (FDNN hit exactly this; it is not hypothetical.)
    frozen_masks = [np.array(l.mask).copy() for l in model.turnover_layers()]

    hist = {"train": [], "val": [], "val_r1": []}
    best, best_w, best_state = 1e9, None, None
    for ep in range(epochs):
        model.set_training(True)
        loss, grads = lg(model, Xtr, Ytr, ig_tr)
        opt.update(model, grads)
        for lyr, fm in zip(model.turnover_layers(), frozen_masks):
            lyr.mask = mx.array(fm)
        sl, sg = mx.value_and_grad(state_loss)(state)
        state = sopt.apply_gradients(sg, state)   # plain dict, not a Module
        mx.eval(model.parameters(), opt.state, state)

        model.set_training(False)
        vl = float(loss_fn(model, Xva, Yva, ig_va).item())
        r1 = retrieval_r1(np.array(_norm(model(Xva))), np.array(_norm(Yva)))
        hist["train"].append(float(loss.item()))
        hist["val"].append(vl)
        hist["val_r1"].append(r1)
        if vl < best:
            best, best_w = vl, _clone_params(model)
            best_state = {k: mx.array(np.array(v)) for k, v in state.items()}
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  ep {ep:4d} train {float(loss.item()):.4f} "
                  f"val {vl:.4f} val_R@1 {r1:.3f}", flush=True)
    if best_w is not None:
        _load_params(model, best_w)
        state = best_state
    model.set_training(False)
    return model, {"history": hist, "best_val": best,
                   "log_t": float(state["log_t"].item()),
                   "bias": float(state["bias"].item())}


def retrieval_r1(v, u):
    """Fraction of windows whose own caption is its nearest caption.

    This is the metric that matters: not "is the vector close to the target"
    but "does the vector RANK the right target first" — the same question the
    query path asks.
    """
    if len(v) < 2:
        return float("nan")
    s = v @ u.T
    return float((s.argmax(axis=1) == np.arange(len(v))).mean())


def _clone_params(model):
    from mlx.utils import tree_flatten
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def _load_params(model, flat):
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in flat.items()]))


def save_tower(model, codec, meta, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "tower.npz", **_clone_params(model))
    codec.save(path / "codec.npz")
    # model.cfg goes LAST and wins. `meta` is often a carried-forward copy of
    # a previous tower.json and still holds that tower's cfg; letting it
    # override would save compacted weights under the pre-compaction shape,
    # and the next load would reshape-crash.
    (path / "tower.json").write_text(json.dumps(
        {**meta, "cfg": model.cfg}, indent=2))


def load_tower(path):
    path = Path(path)
    meta = json.loads((path / "tower.json").read_text())
    model = ContextTower(**meta["cfg"])
    z = np.load(path / "tower.npz")
    _load_params(model, {k: z[k] for k in z.files})
    model.set_training(False)
    return model, ContextCodec.load(path / "codec.npz"), meta
