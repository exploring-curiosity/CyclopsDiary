"""Cosmos 3's language model in MLX, and the streaming read on it.

Why: on torch MPS the language model is launch-bound (28 layers of small
kernels; a decoded token costs 65-90 ms), so the continuous read runs at
about real time. MLX builds each step lazily and fuses it: the same read ran
at about 3x real time on an M5 Pro (1-s atoms of two 384x512 frames, window
8192, two readout probes per step; 2026-09-26). Same weights as released
(bf16, or quantized when asked), same math: RMSNorm, GQA 16/8 x 128,
interleaved 3-D M-RoPE (sections 24/20/20, theta 1e8), ReLU^2 MLP. The vision
tower stays in torch; its patch features come over as arrays.

`MLXStream` keeps the torch reader's bookkeeping (`cosmos3.Cosmos3Stream`:
sink, frame blocks, atoms, 3-D positions, window eviction) and moves only the
memory (a preallocated per-layer K/V buffer: append, crop, evict after the
sink) and the arithmetic. It hands over the MEMORY state (the last hidden
state at the atom's last video token) and, with `top_k`, the atom's word
list: the model's own head on the atom's video tokens, the log of their mean
probability, best first. That list is the step's row in the index
`headsearch` scores; nothing is generated and nothing is asked.

Parity with the torch reader is tests/test_cosmos3_mlx.py.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import cosmos3 as C3

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}


# ------------------------------------------------------------------ the language model
class Attention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv, head_dim, bias=False):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=bias)


class MLP(nn.Module):
    def __init__(self, hidden, inter, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter, bias=bias)
        self.fc2 = nn.Linear(inter, hidden, bias=bias)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h * h)                                  # ReLU squared


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                                   cfg.head_dim, bool(getattr(cfg, "attention_bias", False)))
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size, bool(getattr(cfg, "mlp_bias", False)))
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


def rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class KV:
    """Per-layer K/V buffers (1, n_kv, capacity, head_dim), filled up to `n`."""

    def __init__(self, n_layers, n_kv, head_dim, capacity, dtype):
        self.cap = int(capacity)
        self.k = [mx.zeros((1, n_kv, self.cap, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.v = [mx.zeros((1, n_kv, self.cap, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.n = 0

    def put(self, i, k, v):
        """Write the chunk for layer i at [n, n+L); -> the filled keys/values [0, n+L)."""
        L = k.shape[2]
        end = self.n + L
        if end > self.cap:
            raise RuntimeError(f"KV capacity {self.cap} exceeded ({end}): give the reader a window")
        self.k[i][:, :, self.n:end, :] = k
        self.v[i][:, :, self.n:end, :] = v
        return self.k[i][:, :, :end, :], self.v[i][:, :, :end, :]

    def advance(self, L):
        self.n += int(L)

    def crop(self, m):
        """Forget the last m tokens (they are overwritten by the next write)."""
        self.n -= int(m)

    def evict(self, start, m):
        """Drop tokens [start, start+m): the tail moves left over them."""
        if m <= 0:
            return
        tail = self.n - (start + m)
        for i in range(len(self.k)):
            if tail > 0:
                self.k[i][:, :, start:start + tail, :] = self.k[i][:, :, start + m:self.n, :]
                self.v[i][:, :, start:start + tail, :] = self.v[i][:, :, start + m:self.n, :]
        self.n -= int(m)
        mx.eval(self.k, self.v)


class EdgeLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Layer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        rp = cfg.rope_parameters
        d = cfg.head_dim
        self.inv_freq = 1.0 / (float(rp["rope_theta"]) ** (np.arange(0, d, 2, dtype=np.float64) / d))
        sec = rp.get("mrope_section", [24, 20, 20])
        axis = np.zeros(d // 2, np.int64)                        # which of T/H/W drives each frequency slot
        axis[1:sec[1] * 3:3] = 1
        axis[2:sec[2] * 3:3] = 2
        self.axis = axis
        self.n_kv, self.head_dim = cfg.num_key_value_heads, d

    def rope(self, pos3):
        """pos3: (3, L) integer positions (T, H, W) -> cos/sin (L, head_dim), float32."""
        pos3 = np.asarray(pos3, np.float64)
        f = pos3[:, :, None] * self.inv_freq[None, None, :]     # (3, L, d/2)
        sel = np.where(self.axis[None, :] == 1, f[1], np.where(self.axis[None, :] == 2, f[2], f[0]))
        emb = np.concatenate([sel, sel], -1)
        return mx.array(np.cos(emb), dtype=mx.float32), mx.array(np.sin(emb), dtype=mx.float32)

    def new_cache(self, capacity, dtype=mx.bfloat16):
        return KV(len(self.layers), self.n_kv, self.head_dim, capacity, dtype)

    def __call__(self, h, pos3, cache: KV, logits=False):
        """h: (1, L, hidden) input embeddings; pos3: (3, L); appends to `cache`. -> final normed states
        (1, L, hidden) (and the last position's logits when asked)."""
        L = h.shape[1]
        cos, sin = self.rope(pos3)
        cos, sin = cos.astype(h.dtype)[None, None], sin.astype(h.dtype)[None, None]
        mask = "causal" if L > 1 else None
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).reshape(1, L, a.n_heads, a.head_dim).transpose(0, 2, 1, 3)
            k = a.k_proj(x).reshape(1, L, a.n_kv, a.head_dim).transpose(0, 2, 1, 3)
            v = a.v_proj(x).reshape(1, L, a.n_kv, a.head_dim).transpose(0, 2, 1, 3)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            K, V = cache.put(i, k, v)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=a.scale, mask=mask)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, L, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        cache.advance(L)
        out = self.norm(h)
        if logits:
            return out, self.lm_head(out[:, -1, :])
        return out


def from_torch(lm_torch, lm_head_torch, dtype=mx.bfloat16, quant_bits=0):
    """Build the MLX language model from the loaded torch text model (language_model) and lm_head."""
    import torch
    m = EdgeLM(lm_torch.config)
    sd = dict(lm_torch.state_dict())                        # module names match this file's
    sd["lm_head.weight"] = lm_head_torch.weight
    weights = [(k, mx.array(v.detach().to(torch.float32).cpu().numpy()).astype(dtype))
               for k, v in sd.items() if "rotary_emb" not in k]
    m.load_weights(weights, strict=True)
    if quant_bits:
        nn.quantize(m, group_size=64, bits=int(quant_bits),
                    class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and mod.weight.shape[0] >= 256)
    mx.eval(m.parameters())
    return m


# ------------------------------------------------------------------ the streaming read
class MLXStream(C3.Cosmos3Stream):
    """One recording read as one growing video on the MLX language model.

    push(frames, idx, fps) -> the unit memory state of the atom; with `top_k`,
    `last_top` is then (ids int32 (k,), log-probs float32 (k,)) of the atom's
    best words. `window` is the token budget the memory keeps after the sink
    (StreamingLLM: the oldest frames go first); at 384x512 an atom of two
    frames is about 400 tokens, so 8192 is the last ~20 s of video."""

    def __init__(self, enc, window: int | None = 8192, states=("memory",), top_k: int = 0,
                 dtype: str = "bfloat16", quant_bits: int = 0, capacity: int | None = None):
        if tuple(states) != ("memory",):
            raise ValueError("the MLX reader hands over the memory state only")
        super().__init__(enc, window=window, states=states)
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r}: one of {sorted(DTYPES)}")
        self.mdtype = DTYPES[dtype]
        key = f"_mlx_lm_{dtype}_q{quant_bits}"
        if not hasattr(enc, key):                          # one MLX copy of the weights per process
            setattr(enc, key, from_torch(self.lm, enc.model.lm_head, dtype=self.mdtype, quant_bits=quant_bits))
        self.M = getattr(enc, key)
        self.top_k = int(top_k)
        self.last_top = None
        self.capacity = int(capacity or (window or 8192) + 2048)

    def open(self) -> None:
        self.cache = self.M.new_cache(self.capacity, dtype=self.mdtype)
        self.pos = 0
        self._blocks.clear()
        self._atoms, self._tok_total, self._dropped = [], 0, 0
        self.last_top = None
        pos = self._text_positions(len(self.prefix_ids))
        self._prefix_pos = pos[:, 0, :].cpu()
        self.M(self.M.embed_tokens(mx.array([self.prefix_ids])), pos[:, 0, :].cpu().numpy(), self.cache)
        mx.eval(self.cache.k, self.cache.v)

    def cache_len(self) -> int:
        return int(self.cache.n) if self.cache is not None else 0

    def push(self, frames, idx, fps: float) -> np.ndarray:
        import torch
        from transformers.video_utils import VideoMetadata
        if self.cache is None:
            self.open()
        frames = self.enc._frames(frames)
        k = int(frames.shape[0])
        idx = [int(i) for i in idx]
        total = max(idx) + 1
        x = self.proc(text=[C3.PLACEHOLDER], videos=[frames], do_sample_frames=False, return_tensors="pt",
                      video_metadata=[VideoMetadata(total_num_frames=total, fps=float(fps),
                                                    duration=total / float(fps), frames_indices=idx)])
        ids = x["input_ids"][0]
        grid = x["video_grid_thw"].to(self.dev)
        with torch.inference_mode():
            feats = self.core.get_video_features(x["pixel_values_videos"].to(self.dev), grid).pooler_output
            feats = torch.cat(list(feats), dim=0).to(torch.float32).cpu().numpy()
            pos = self._mixed_positions(ids.to(self.dev), grid)
        rows = np.where((ids == self.video_id).numpy())[0]
        emb = self.M.embed_tokens(mx.array(ids.numpy()[None]))
        emb[0, mx.array(rows)] = mx.array(feats).astype(emb.dtype)
        h = self.M(emb, pos[:, 0, :].cpu().numpy(), self.cache)
        mx.eval(h, self.cache.k, self.cache.v)
        L = int(ids.numel())
        per = L // k
        if per * k != L:
            raise RuntimeError(f"{L} tokens for {k} frames: the processor's frame layout is not uniform")
        for j in range(k):
            self._blocks.append((per, pos[:, 0, j * per:(j + 1) * per].cpu()))
        self._atoms.append((self._tok_total, self._tok_total + L, int(self.pos)))
        self._tok_total += L
        if self.top_k:
            self.last_top = self._top(h, rows)
        self._evict(keep=k)
        m = np.array(h[0, -1].astype(mx.float32))
        return m / np.linalg.norm(m)

    def _top(self, h, rows) -> tuple:
        """The atom's best words: the head on its video tokens, log of the mean probability, best first."""
        lg = self.M.lm_head(h[0, mx.array(rows)]).astype(mx.float32)          # (R, V)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        agg = mx.logsumexp(lp, axis=0) - math.log(len(rows))                 # (V,)
        k = min(self.top_k, int(agg.shape[0]))
        top = mx.argpartition(-agg, kth=k - 1)[:k]
        vals = agg[top]
        order = mx.argsort(-vals)
        top, vals = top[order], vals[order]
        mx.eval(top, vals)
        return np.array(top).astype(np.int32), np.array(vals).astype(np.float32)

    def _evict(self, keep: int) -> None:
        if self.window is None:
            return
        drop = 0
        while len(self._blocks) > keep and sum(t for t, _ in self._blocks) > self.window:
            drop += self._blocks.popleft()[0]
        if not drop:
            return
        self._dropped += drop
        self.cache.evict(len(self.prefix_ids), drop)

    def _probe(self):
        raise NotImplementedError("the MLX reader hands over the memory state; questions are the torch reader's")

    def probe_span(self, a, b, top_k=0):
        raise NotImplementedError("the MLX reader hands over the memory state; questions are the torch reader's")
