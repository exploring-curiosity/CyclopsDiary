"""The world model, wired.

nvidia/Cosmos3-Edge reads each recording as one continuous video
(elidedb.cosmos3_mlx.MLXStream: the vision tower in torch, the language model
in MLX, the model's memory carried across the recording) and hands over, per
1-s step, its memory state and its best words. Nothing is generated and
nothing is asked; the read is the model's own.

Why the side stack: Cosmos3-Edge's own architecture class needs transformers
>= 5, and the main environment carries 4.x for the engine's other models.
data/stacks/d40 holds 5.x and has to be first on the path before anything in
the process imports transformers; `use_stack` puts it there and refuses when
it is too late, rather than letting 4.x fail on the checkpoint later.

What the model sees is what the person saw: the frame keeps its source's
aspect inside the pixel budget (a phone held upright is not squashed into
4:3), and a phone video's display rotation is applied.

One model, many recordings: the weights are one copy per process, and each
recording's `Read` has its own reader, so its own memory. The model takes one
step at a time under LOCK, so a clip from the inbox and several live phones
can be read at once, interleaved step by step, each read continuous and none
mixed with another.
"""
from __future__ import annotations

import math
import os
import sys
import threading

import numpy as np

from . import config

_ENC = None
LOCK = threading.RLock()


def use_stack() -> None:
    if not config.STACK.is_dir():
        raise RuntimeError(f"{config.STACK} is missing; rebuild it with the command in data/stacks/README.md")
    mod = sys.modules.get("transformers")
    if mod is not None and not mod.__version__.startswith("5."):
        raise RuntimeError(f"transformers {mod.__version__} was imported before the side stack went on the path")
    if str(config.STACK) not in sys.path:
        sys.path.insert(0, str(config.STACK))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the weights are in the local cache; no venue network needed
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def frame_hw(width: int, height: int, budget: int, multiple: int = 32) -> tuple:
    """(H, W) of a frame with the source's aspect and about `budget` pixels, each a multiple of 32
    (the tower's 16-pixel patch, merged 2 x 2)."""
    a = float(width) / float(height)
    h = math.sqrt(budget / a)
    return (max(multiple, int(round(h / multiple)) * multiple),
            max(multiple, int(round(h * a / multiple)) * multiple))


def rotation_of(path: str) -> int:
    """Quarter turns (np.rot90's k, counter-clockwise) that stand the stored frames upright: the
    container's display rotation, read off the first decoded frame."""
    import av
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            return int(round(float(fr.rotation or 0) / 90.0)) % 4
    return 0


def upright(img: np.ndarray, k: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def encoder(model: str | None = None):
    """The world model, loaded once per process: about 8.5 GB of weights in torch, then an MLX copy
    of the language model when the first reader is made."""
    global _ENC
    if _ENC is None:
        use_stack()
        from elidedb import cosmos3 as C3
        _ENC = C3.Cosmos3Embed(model_id=model or config.get().model)
    return _ENC


def reader(enc, window: int, top_k: int):
    from elidedb import cosmos3_mlx as CM
    return CM.MLXStream(enc, window=window, top_k=top_k)


class Read:
    """One recording read on the shared model: frames in with their times (the source rate), rows out, one per
    step: (t0, t1, memory state, the step's best word ids, their log-probabilities). `due` is the time the next
    sampled frame is due; the frames in between may be passed as None (they only teach the source rate).
    Every model step is taken under LOCK with this recording's frame size; `model` stands in for the model's
    reader in tests."""

    def __init__(self, enc, hw, model=None):
        from elidedb import streamtext as ST
        s = config.get()
        self.enc, self.hw = enc, tuple(hw)
        with LOCK:
            self._size()
            self.rd = model if model is not None else reader(enc, s.window, s.top_k)
            self.emb = ST.StreamEmbedder(self.rd, rate=s.rate, n_frames=s.n_frames)

    @property
    def due(self) -> float:
        return self.emb.due

    def push(self, t: float, frame) -> list:
        with LOCK:
            self._size()
            return self._rows(self.emb.push(t, frame))

    def flush(self) -> list:
        with LOCK:
            self._size()
            return self._rows(self.emb.flush())

    def _size(self) -> None:
        if self.enc is not None:
            self.enc.frame_hw = self.hw                   # the model's frame size is per recording

    def _rows(self, got) -> list:
        return [(span[0], span[1], vec, *self.rd.last_top) for span, vec in got]


def cached(model: str) -> bool:
    """Whether the model's weights are in the local Hugging Face cache (nothing is downloaded here)."""
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    return os.path.isdir(os.path.join(root, "models--" + model.replace("/", "--")))
