"""ACTION PROBE — video-native verb grounding, zero LLM/VLM anywhere.

The calibration that forced this: both VLM judge tiers overclaim scene
matches (7B est 1.0 on visually ~10%-pure sets) — a language model shown
frames cannot bind actions. User directive: no LLM/VLM judges, ever.

Replacement: Meta's released SSv2 attentive probe on the V-JEPA 2 ViT-L
encoder ALREADY in the stack (291 ms/16-frame clip on MPS, measured).
Something-Something-v2's 174 classes are literally the failing query
taxonomy of this database — "Closing something", "Opening something",
"Picking something up", "Putting something into something", "Taking
something out of something", "Covering something with something" (= lid
on vessel), "Folding something". The probe is 174-way evidence about
WHAT HAPPENED in the clip, produced by a video model trained on motion,
not a text model guessing from two frames.

Faithful port of vjepa2/src/models/attentive_pooler.py (MIT): 3
self-attn blocks over ALL encoder tokens -> 1-query cross-attn pool ->
linear(1024 -> 174). Class list vendored as data/ssv2_classes.txt —
this is the MODEL's output vocabulary (like a tokenizer), not dataset
metadata; the no-metadata rule stays intact.

Query -> class mapping rides the PE text tower already cached: cosine
between the query and the 174 class names, softmaxed over the top few.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_STATE = {}

PROBE_CKPT = Path(__file__).resolve().parents[2] / \
    "models/ssv2-vitl-16x2x3.pt"
ENCODER_ID = "facebook/vjepa2-vitl-fpc64-256"
N_CLASSES = 174


def ssv2_classes():
    if "classes" not in _STATE:
        p = Path(__file__).parent / "data/ssv2_classes.txt"
        _STATE["classes"] = [ln.strip() for ln in
                             p.read_text().splitlines() if ln.strip()]
        assert len(_STATE["classes"]) == N_CLASSES
    return _STATE["classes"]


def _build_probe():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dim, heads = 1024, 16

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 4)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, heads, C // heads) \
                .permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            return self.proj(y.transpose(1, 2).reshape(B, N, C))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention()
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class CrossAttention(nn.Module):
        # NOTE: Meta's probe CrossAttention has NO output projection
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(dim, dim * 2, bias=True)

        def forward(self, q, x):
            B, n, C = q.shape
            qh = self.q(q).reshape(B, n, heads, C // heads) \
                .transpose(1, 2)
            N = x.shape[1]
            kv = self.kv(x).reshape(B, N, 2, heads, C // heads) \
                .permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(qh, kv[0], kv[1])
            return y.transpose(1, 2).reshape(B, n, C)

    class CrossAttentionBlock(nn.Module):
        # norm1 normalizes the CONTEXT tokens, not the query (Meta)
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.xattn = CrossAttention()
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MLP()

        def forward(self, q, x):
            q = q + self.xattn(q, self.norm1(x))
            return q + self.mlp(self.norm2(q))

    class AttentivePooler(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_tokens = nn.Parameter(torch.zeros(1, 1, dim))
            self.cross_attention_block = CrossAttentionBlock()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            q = self.query_tokens.repeat(len(x), 1, 1)
            return self.cross_attention_block(q, x)

    class AttentiveClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.pooler = AttentivePooler()
            self.linear = nn.Linear(dim, N_CLASSES)

        def forward(self, x):
            return self.linear(self.pooler(x).squeeze(1))

    return AttentiveClassifier()


def _load():
    if "model" in _STATE:
        return _STATE
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    from .device import pick
    dev, dtype = pick()
    enc = AutoModel.from_pretrained(ENCODER_ID, dtype=dtype) \
        .to(dev).eval()
    probe = _build_probe()
    sd = torch.load(PROBE_CKPT, map_location="cpu",
                    weights_only=False)["classifiers"][0]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    probe.load_state_dict(sd, strict=True)
    probe = probe.to(dev).float().eval()
    _STATE.update(model=enc, probe=probe, dev=dev, dtype=dtype,
                  proc=AutoVideoProcessor.from_pretrained(ENCODER_ID))
    return _STATE


def clip_action_probs(frames_u8):
    """16 HWC uint8 frames -> softmax over 174 SSv2 classes."""
    import torch
    st = _load()
    px = st["proc"](videos=[list(frames_u8)], return_tensors="pt")[
        "pixel_values_videos"].to(st["dev"], st["dtype"])
    with torch.no_grad():
        feats = st["model"](pixel_values_videos=px).last_hidden_state
        logits = st["probe"](feats.float())
    return torch.softmax(logits[0], -1).cpu().numpy()


def query_class_weights(text, top=5):
    """Query text -> sparse weights over SSv2 classes via the PE text
    tower (cached). Softmax over the top matches; everything else 0."""
    from .pe import _text_vec
    if "clsvec" not in _STATE:
        _STATE["clsvec"] = np.stack(
            [_text_vec(c.lower()) for c in ssv2_classes()])
    sims = _STATE["clsvec"] @ _text_vec(text)
    w = np.zeros(N_CLASSES)
    ix = np.argsort(-sims)[:top]
    e = np.exp((sims[ix] - sims[ix].max()) / 0.05)
    w[ix] = e / e.sum()
    return w
