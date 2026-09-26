"""Post-training cellular turnover for the context tower — PPO + reverse
attention, ported from FDNN_BrainModel/models/pruning.py.

WHY PRUNE AT ALL
----------------
Because the only thing this index is for is answering fast. Every channel the
tower carries is multiplied over T timesteps, for every window, at every
rebuild. Capacity that is not earning its keep is latency the user pays for
forever. And with a few hundred labelled windows, an over-provisioned tower
is also memorising — so apoptosis buys accuracy and speed at the same time,
which is exactly the regime cellular turnover was designed for.

WHAT CHANGED FROM FDNN, AND WHY
-------------------------------
FDNN prunes per (layer, neuron): every layer has its own aliveness mask. That
is right for a plain stack. This tower is a RESIDUAL stack, and in a residual
stack channel j is a wire that runs the whole depth — masking it in layer 2
alone does not remove it, because the skip connection keeps carrying it. So
the prune unit here is the residual-stream CHANNEL, decided once and applied
to every block.

That is not a weakening of the idea, it is the correct form of it for this
topology (the same structured channel pruning people apply to ResNets), and
it has a property per-layer masking does not: a dead channel can be
physically deleted. `compact()` rebuilds the tower without it, so the mask
turns into real wall-clock speed instead of a multiply by zero. Pruning that
does not shrink the matmul is not pruning, it is decoration.

THE THREE MEASURES (unchanged in spirit from FDNN)
--------------------------------------------------
  utilization      Δ validation RETRIEVAL loss when the channel is silenced.
                   Not weight magnitude — the thing we actually care about.
  reverse attention softmax(-utilization): mass on the channels the network
                   attends AWAY from, i.e. the prune candidates.
  PPO              a clipped-PPO contextual bandit over per-channel features
                   that samples masks, scores them by
                   -(loss ratio) - λ·kept_fraction, and learns which channels
                   can actually go.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


# ===========================================================================
# 0. Channel mask plumbing
# ===========================================================================
def n_channels(model):
    return model.blocks[0].max_out_d


def get_channel_mask(model):
    return np.array(model.blocks[0].mask).copy()


def set_channel_mask(model, mask):
    """One decision, applied to every block — see module docstring."""
    m = np.asarray(mask, dtype=np.float32)
    for b in model.turnover_layers():
        b.set_active_mask(m)


# ===========================================================================
# 1. Utilization ("consumption")
# ===========================================================================
def measure_utilization(model, loss_of, verbose=False):
    """Per-channel ablation importance + the cheap descriptive stats.

    `loss_of(model)` must return the validation retrieval loss as a float.
    Ablation is the increase in that loss when a single channel is silenced
    everywhere — the most directly interpretable notion of how much the tower
    relies on it.
    """
    model.set_training(False)
    base = loss_of(model)
    mask = get_channel_mask(model)
    n = len(mask)

    abl = np.zeros(n, dtype=np.float32)
    for c in np.where(mask == 1.0)[0]:
        probe = mask.copy()
        probe[c] = 0.0
        set_channel_mask(model, probe)
        abl[c] = max(loss_of(model) - base, 0.0)
    set_channel_mask(model, mask)
    return {"mask": mask, "ablation": abl, "base_loss": base}


def descriptive_stats(model, X):
    """Activation magnitude per channel (averaged over blocks) and the
    downstream weight norm each channel feeds."""
    model.set_training(False)
    outs = []
    h = mx.array(X)
    for b in model.turnover_layers():
        outs.append(np.array(mx.mean(mx.abs(b.neuron_outputs(h)), axis=(0, 1))))
        h = b(h)
    act = np.mean(np.stack(outs), axis=0)

    n = n_channels(model)
    wf = np.array(model.gru_f.Wx)          # (3*hidden, n_hidden)
    wb = np.array(model.gru_b.Wx)
    hw = np.array(model.head.weight)       # (d_out, n_hidden + 2*gru_hidden)
    pw = np.array(model.pool.proj.weight)
    down = (np.linalg.norm(wf, axis=0) + np.linalg.norm(wb, axis=0)
            + np.linalg.norm(hw[:, :n], axis=0)
            + np.linalg.norm(pw[:, :n], axis=0))
    return act.astype(np.float32), down.astype(np.float32)


# ===========================================================================
# 2. Reverse attention
# ===========================================================================
def reverse_attention(importance, mask, temperature=1.0):
    """Forward attention concentrates on useful units; reverse attention
    inverts it so the LEAST useful surface as prune candidates. Dead channels
    score 0; scores over the live set sum to 1."""
    importance = np.asarray(importance, dtype=np.float64)
    alive = mask == 1.0
    r = np.zeros_like(importance)
    if alive.sum() == 0:
        return r.astype(np.float32)
    imp = importance[alive]
    imp = (imp - imp.mean()) / (imp.std() + 1e-8)
    logits = -imp / max(temperature, 1e-6)
    logits -= logits.max()
    e = np.exp(logits)
    r[alive] = e / (e.sum() + 1e-12)
    return r.astype(np.float32)


# ===========================================================================
# 3. PPO agent
# ===========================================================================
class PrunePolicy(nn.Module):
    def __init__(self, feat_dim, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi_head = nn.Linear(hidden, 1)
        self.v_head = nn.Linear(hidden, 1)

    def __call__(self, feats):
        h = nn.relu(self.fc1(feats))
        h = nn.relu(self.fc2(h))
        return self.pi_head(h)[:, 0], self.v_head(h)[:, 0]


def _bernoulli_logp(actions, logits):
    return actions * logits - mx.logaddexp(mx.zeros_like(logits), logits)


def _entropy(logits):
    p = mx.sigmoid(logits)
    return nn.softplus(logits) - p * logits


class PPOPruner:
    """Contextual bandit over prune masks of the residual stream.

    One step per episode: observe a fixed per-channel feature matrix from the
    frozen tower, emit a keep-probability per channel, sample a binary mask,
    receive one scalar reward. Maximising it means "prune as much as possible
    without inflating retrieval loss".
    """

    def __init__(self, model, X, loss_of, sparsity_coef=0.35,
                 loss_tolerance=1.0, hidden=32, lr=3e-3, clip_eps=0.2,
                 entropy_coef=0.01, value_coef=0.5, seed=0):
        self.model, self.loss_of = model, loss_of
        self.sparsity_coef = sparsity_coef
        self.loss_tolerance = loss_tolerance
        self.clip_eps, self.entropy_coef = clip_eps, entropy_coef
        self.value_coef = value_coef
        np.random.seed(seed)
        mx.random.seed(seed)

        self.base_mask = get_channel_mask(model)
        util = measure_utilization(model, loss_of)
        self.base_loss = util["base_loss"]
        self.ablation = util["ablation"]
        act, down = descriptive_stats(model, X)
        ra = reverse_attention(self.ablation, self.base_mask)
        omega = model.blocks[0].omegas_per_neuron

        self.candidates = np.where(self.base_mask == 1.0)[0]
        f = np.stack([self.ablation[self.candidates], act[self.candidates],
                      down[self.candidates], ra[self.candidates],
                      omega[self.candidates]], axis=1).astype(np.float32)
        mu, sd = f.mean(0, keepdims=True), f.std(0, keepdims=True) + 1e-6
        f[:, :4] = (f[:, :4] - mu[:, :4]) / sd[:, :4]
        self.feats = mx.array(f)
        self.reverse_attn = ra
        self.policy = PrunePolicy(f.shape[1], hidden=hidden)
        self.opt = optim.Adam(learning_rate=lr)
        self.history = {"iter": [], "reward": [], "kept_frac": [],
                        "loss_ratio": [], "entropy": []}

    def _apply(self, action):
        m = self.base_mask.copy()
        m[self.candidates] = action
        set_channel_mask(self.model, m)

    def _reward(self, action):
        self._apply(action)
        loss = self.loss_of(self.model)
        set_channel_mask(self.model, self.base_mask)
        kept = float(action.mean()) if len(action) else 0.0
        ratio = loss / (self.base_loss + 1e-12)
        return (-self.loss_tolerance * ratio - self.sparsity_coef * kept,
                kept, ratio)

    def train(self, n_iters=40, episodes_per_iter=12, ppo_epochs=4,
              verbose=True):
        for it in range(n_iters):
            logits, _ = self.policy(self.feats)
            probs = np.array(mx.sigmoid(logits))
            acts, rews, kfs, lrs = [], [], [], []
            for _ in range(episodes_per_iter):
                a = (np.random.uniform(size=probs.shape) < probs).astype(np.float32)
                r, kf, lr_ = self._reward(a)
                acts.append(a); rews.append(r); kfs.append(kf); lrs.append(lr_)
            acts = np.array(acts, np.float32)
            rews = np.array(rews, np.float32)
            adv = (rews - rews.mean()) / (rews.std() + 1e-6)

            acts_mx, adv_mx, rews_mx = mx.array(acts), mx.array(adv), mx.array(rews)
            old_logits, _ = self.policy(self.feats)
            old_logp = mx.array(np.array(
                mx.sum(_bernoulli_logp(acts_mx, old_logits[None, :]), axis=1)))
            for _ in range(ppo_epochs):
                _, grads = self._loss_and_grad(acts_mx, old_logp, adv_mx, rews_mx)
                self.opt.update(self.policy, grads)
                mx.eval(self.policy.parameters(), self.opt.state)

            ent = float(mx.mean(_entropy(logits)).item())
            self.history["iter"].append(it)
            self.history["reward"].append(float(rews.mean()))
            self.history["kept_frac"].append(float(np.mean(kfs)))
            self.history["loss_ratio"].append(float(np.mean(lrs)))
            self.history["entropy"].append(ent)
            if verbose and (it % 5 == 0 or it == n_iters - 1):
                print(f"  PPO {it:3d} | reward {rews.mean():+.3f} | kept "
                      f"{np.mean(kfs):.2f} | loss x{np.mean(lrs):.3f} | "
                      f"H {ent:.3f}", flush=True)
        return self.history

    def _loss_and_grad(self, actions, old_logp, adv, rewards):
        def loss_fn(policy):
            logits, values = policy(self.feats)
            v = mx.mean(values)
            logp = mx.sum(_bernoulli_logp(actions, logits[None, :]), axis=1)
            ratio = mx.exp(logp - old_logp)
            pl = -mx.mean(mx.minimum(ratio * adv,
                                     mx.clip(ratio, 1 - self.clip_eps,
                                             1 + self.clip_eps) * adv))
            vl = mx.mean(mx.square(v - rewards))
            return pl + self.value_coef * vl \
                - self.entropy_coef * mx.mean(_entropy(logits))
        return nn.value_and_grad(self.policy, loss_fn)(self.policy)

    def apply_greedy(self, keep_threshold=0.5):
        logits, _ = self.policy(self.feats)
        probs = np.array(mx.sigmoid(logits))
        m = self.base_mask.copy()
        m[self.candidates] = (probs >= keep_threshold).astype(np.float32)
        set_channel_mask(self.model, m)
        return m, probs


# ===========================================================================
# 4. Neurogenesis — revive dead channels with fresh, spectrally diverse ones
# ===========================================================================
def neurogenesis(model, n_revive, seed=0):
    """Rebirth. New channels start near-silent (tiny w2) so they cannot shock
    the forward pass, and are re-initialised with FINER's WIDE bias range —
    without that they spawn as low-frequency clones and the population slowly
    loses the spectral diversity that made the KAN basis worth having."""
    rng = np.random.default_rng(seed)
    mask = get_channel_mask(model)
    dead = np.where(mask == 0.0)[0]
    if len(dead) == 0 or n_revive <= 0:
        return 0
    revive = dead[:min(n_revive, len(dead))]

    for b in model.turnover_layers():
        w1 = np.array(b.w1)          # (kernel, in_d, out*k)
        b1 = np.array(b.b1)
        w2 = np.array(b.w2)
        ph = np.array(b.phases)
        gs = np.array(b.gabor_s)
        la = np.array(b.log_alpha)
        om_full = np.repeat(b.omegas_per_neuron, b.k)
        mean_om = float(b.omegas_per_neuron.mean())
        lim = float(np.sqrt(6.0 / (b.kernel * b.in_d)) / mean_om)
        for c in revive:
            s, e = c * b.k, (c + 1) * b.k
            w1[:, :, s:e] = rng.uniform(-lim, lim, (b.kernel, b.in_d, b.k))
            b1[s:e] = rng.uniform(-b.bias_range, b.bias_range, b.k)
            ph[s:e] = rng.uniform(0, 2 * np.pi, b.k)
            gs[s:e] = rng.uniform(0.3, 1.5, b.k)
            lw = np.log(np.clip(om_full[s:e], 1e-3, None))
            la[s:e] = rng.uniform(np.zeros_like(lw), lw)
            w2[c, :] = rng.standard_normal(b.k) * 1e-6
        b.w1 = mx.array(w1.astype(np.float32))
        b.b1 = mx.array(b1.astype(np.float32))
        b.w2 = mx.array(w2.astype(np.float32))
        b.phases = mx.array(ph.astype(np.float32))
        b.gabor_s = mx.array(gs.astype(np.float32))
        b.log_alpha = mx.array(la.astype(np.float32))
    mask[revive] = 1.0
    set_channel_mask(model, mask)
    return len(revive)


# ===========================================================================
# 5. Compaction — where the mask becomes actual speed
# ===========================================================================
def compact(model):
    """Physically delete dead channels: return a smaller, equivalent tower.

    Every dead channel is removed from each block's output slots, from every
    consumer's input slots (both GRUs, the attention pool, the head), and from
    the residual stream itself. Output is numerically equivalent to the masked
    model but the matmuls are genuinely smaller.
    """
    from .ctxtower import ContextTower
    keep = np.where(get_channel_mask(model) == 1.0)[0]
    n_new = len(keep)
    if n_new == 0:
        raise RuntimeError("every channel was pruned")
    if n_new == n_channels(model):
        return model, keep

    cfg = dict(model.cfg)
    cfg["n_hidden"] = int(n_new)
    new = ContextTower(**cfg)
    k = model.blocks[0].k
    sub = np.concatenate([np.arange(c * k, (c + 1) * k) for c in keep])

    for ob, nb in zip(model.blocks, new.blocks):
        w1 = np.array(ob.w1)
        if not ob.use_residual:                  # first block: in_d is d_in
            nb.w1 = mx.array(w1[:, :, sub])
        else:                                    # in_d is the stream: slice both
            nb.w1 = mx.array(w1[:, keep][:, :, sub])
        nb.b1 = mx.array(np.array(ob.b1)[sub])
        nb.phases = mx.array(np.array(ob.phases)[sub])
        nb.gabor_s = mx.array(np.array(ob.gabor_s)[sub])
        nb.log_alpha = mx.array(np.array(ob.log_alpha)[sub])
        nb.w2 = mx.array(np.array(ob.w2)[keep])
        # omegas is TRAINED, so it must be sliced from the live array — not
        # rebuilt from omegas_per_neuron, which is the frozen design band and
        # is stale the moment training starts. Rebuilding it silently reverted
        # every learned frequency and moved val loss by 0.015.
        nb.omegas = mx.array(np.array(ob.omegas)[sub])
        nb.omegas_per_neuron = ob.omegas_per_neuron[keep]   # design band only
        nb.basis_types = mx.array(np.array(ob.basis_types)[sub])
        nb.freeze(keys=["basis_types"], recurse=False)
        nb.set_active_mask(np.ones(n_new, np.float32))

    n_old = n_channels(model)
    g = model.cfg["gru_hidden"]
    for og, ng in ((model.gru_f, new.gru_f), (model.gru_b, new.gru_b)):
        ng.Wx = mx.array(np.array(og.Wx)[:, keep])
        ng.Wh = mx.array(np.array(og.Wh))
        ng.b = mx.array(np.array(og.b))
        ng.bhn = mx.array(np.array(og.bhn))
    # consumers of `cat` = [stream | gru_f | gru_b]: keep the stream slice,
    # carry the two GRU slices through untouched.
    tail = np.arange(n_old, n_old + 2 * g)
    cols = np.concatenate([keep, tail])
    new.pool.proj.weight = mx.array(np.array(model.pool.proj.weight)[:, cols])
    new.pool.proj.bias = mx.array(np.array(model.pool.proj.bias))
    new.pool.score.weight = mx.array(np.array(model.pool.score.weight))
    new.pool.score.bias = mx.array(np.array(model.pool.score.bias))
    new.head.weight = mx.array(np.array(model.head.weight)[:, cols])
    new.head.bias = mx.array(np.array(model.head.bias))
    new.log_gamma = mx.array(np.array(model.log_gamma))
    new.set_training(False)
    return new, keep


# ===========================================================================
# 6. Re-settle
# ===========================================================================
def finetune(model, loss_and_grad_fn, epochs=150, lr=5e-4,
             weight_decay=1e-3, loss_of=None, patience=40):
    """Short re-settling pass with aliveness frozen (see the mask gotcha).

    Validation-aware on purpose. Survivors re-settle on the TRAIN set, and
    with a couple of hundred windows an unguarded 400-epoch re-settle simply
    re-memorises: measured, it took a pruned tower from 0.311 val back up to
    0.357, undoing everything apoptosis had just gained. Keeping the best
    validation state means re-settling can only help, never hurt.
    """
    from .ctxtower import _clone_params, _load_params
    opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
    frozen = get_channel_mask(model)
    best, best_w, since = float("inf"), None, 0
    if loss_of is not None:
        best, best_w = loss_of(model), _clone_params(model)
    for _ in range(epochs):
        model.set_training(True)
        loss, grads = loss_and_grad_fn(model)
        opt.update(model, grads)
        set_channel_mask(model, frozen)
        mx.eval(model.parameters(), opt.state)
        if loss_of is None:
            continue
        v = loss_of(model)
        if v < best - 1e-5:
            best, best_w, since = v, _clone_params(model), 0
        else:
            since += 1
            if since >= patience:
                break
    if best_w is not None:
        _load_params(model, best_w)
        set_channel_mask(model, frozen)
        mx.eval(model.parameters())
    model.set_training(False)
    return best


# ===========================================================================
# 7. The full cycle
# ===========================================================================
def run_pruning_cycle(model, X, loss_of, loss_and_grad_fn, ppo_iters=40,
                      episodes_per_iter=12, sparsity_coef=0.35,
                      keep_threshold=0.5, finetune_epochs=150,
                      rebirth_fraction=0.5, select_tolerance=0.02,
                      seed=0, verbose=True):
    """apoptosis → re-settle → neurogenesis → re-settle, on a frozen tower.

    The cycle is a SEARCH, so it returns the best point it visited, not the
    last one. Rebirth is a bet that the data can support more capacity; on a
    small corpus that bet often loses, and silently shipping the final state
    would hand back a tower worse than the one we started from.
    """
    from .ctxtower import _clone_params, _load_params
    rec = {"stages": []}
    saved = []          # (stage, loss, alive, params, mask) for every stage

    def snap(stage):
        u = measure_utilization(model, loss_of)
        alive = int(u["mask"].sum())
        if verbose:
            print(f"[{stage}] val loss {u['base_loss']:.4f} | "
                  f"alive {alive}/{len(u['mask'])}", flush=True)
        rec["stages"].append({"stage": stage, "loss": u["base_loss"],
                              "alive": alive,
                              "ablation": u["ablation"].tolist(),
                              "mask": u["mask"].tolist()})
        saved.append((stage, u["base_loss"], alive,
                      _clone_params(model), u["mask"].copy()))
        return u

    snap("before")
    if verbose:
        print("\n== PPO reverse-attention pruning search ==", flush=True)
    pruner = PPOPruner(model, X, loss_of, sparsity_coef=sparsity_coef, seed=seed)
    rec["ppo_history"] = pruner.train(n_iters=ppo_iters,
                                      episodes_per_iter=episodes_per_iter,
                                      verbose=verbose)
    _, probs = pruner.apply_greedy(keep_threshold)
    rec["keep_probs"] = probs.tolist()
    rec["reverse_attention"] = pruner.reverse_attn.tolist()
    snap("after_apoptosis")

    if finetune_epochs:
        if verbose:
            print("\n== re-settling survivors ==", flush=True)
        finetune(model, loss_and_grad_fn, epochs=finetune_epochs,
                 loss_of=loss_of)
        snap("after_finetune")

    if rebirth_fraction > 0:
        dead = int((get_channel_mask(model) == 0.0).sum())
        born = neurogenesis(model, int(round(rebirth_fraction * dead)), seed=seed)
        if verbose:
            print(f"\n== neurogenesis: {born} channels reborn ==", flush=True)
        if born:
            snap("after_rebirth")
            if finetune_epochs:
                finetune(model, loss_and_grad_fn,
                         epochs=finetune_epochs, loss_of=loss_of)
                snap("after_rebirth_finetune")

    # ---- operating-point selection ----------------------------------------
    # Not "lowest loss" — that rule can never prune, because the unpruned
    # tower is usually the most accurate one and the whole exercise then
    # returns its own input. The question a database actually asks is: what
    # is the CHEAPEST tower I can serve without giving up measurable quality?
    # So: take the best loss seen, allow `select_tolerance` relative slack,
    # and among everything inside that band keep the fewest channels.
    floor = min(s[1] for s in saved)
    budget = floor * (1.0 + select_tolerance)
    eligible = [s for s in saved if s[1] <= budget]
    stage, loss, alive, params, mask = min(eligible, key=lambda s: (s[2], s[1]))
    _load_params(model, params)
    set_channel_mask(model, mask)
    mx.eval(model.parameters())
    if verbose:
        print(f"\n== operating point: {stage} — {alive} channels, val loss "
              f"{loss:.4f} (best seen {floor:.4f}, "
              f"budget +{select_tolerance:.0%}) ==", flush=True)
    rec["selected"] = {"stage": stage, "loss": loss, "channels": alive,
                       "best_loss_seen": floor,
                       "tolerance": select_tolerance,
                       "candidates": [{"stage": s[0], "loss": s[1],
                                       "channels": s[2]} for s in saved]}
    return rec
