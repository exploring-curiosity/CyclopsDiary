"""Continuous Cosmos stream sampling, pixel change and trace serialization.

Retrieval lives in streamstore/eventsearch. No exposure-search fallback.
"""


from __future__ import annotations


import json
from pathlib import Path
import numpy as np


DEFAULT_MODEL = "nvidia/Cosmos3-Edge"


N_FRAMES = 8


CHANGE_SIZE = 112
SENTENCES_FILE = Path(__file__).resolve().parent / "data" / "generic_sentences.txt"


class StreamEmbedder:
    """Frames in at the source rate, one state per ATOM out, through a
    streaming reader (`cosmos3.Cosmos3Stream`): the tower reads the
    recording as one growing video at its own rate, and a row is the moment,
    not an exposure.

    The tower samples at `rate` (4 fps, the rate the exposures were read
    at); an atom is `n_frames` sampled frames (a frame pair) handed to the
    reader with their source indices and the source rate, which is what the
    model's own timestamps are written from. A row spans the sampling grid
    the atom was due on, so rows tile the recording (0-0.5, 0.5-1.0, ...)
    whatever the source's jitter. A partial atom at the end of a recording
    is read as it is: a moment is not dropped for ending the recording.
    Memory holds one atom of frames, never the stream; the reader holds the
    model's memory of the recording."""

    def __init__(self, reader, rate: float, n_frames: int = 2, fps=None):
        self.rd, self.rate, self.n = reader, float(rate), int(n_frames)
        self.dt = (1.0 / fps) if fps else None       # the source interval, learned from the first two frames
        self.t_last = None
        self.due = 0.0                               # the grid time the next sample is due
        self.t0 = 0.0                                # the grid time the open atom started on
        self.buf: list = []                          # (t, frame) sampled into the open atom
        self.rd.open()

    def push(self, t: float, frame) -> list:
        if self.t_last is not None and self.dt is None and t > self.t_last:
            self.dt = t - self.t_last
        self.t_last = t
        out = []
        if t >= self.due - 1e-9:
            if not self.buf:
                self.t0 = self.due
            self.buf.append((t, frame))
            self.due = round(self.due + 1.0 / self.rate, 9)
            if len(self.buf) >= self.n:
                out += self._close()
        return out

    def flush(self) -> list:
        return self._close() if self.buf else []

    def _close(self) -> list:
        fps = (1.0 / self.dt) if self.dt else self.rate
        idx = [int(round(tt * fps)) for tt, _ in self.buf]
        frames = np.stack([f for _, f in self.buf])
        span = (round(self.t0, 6), round(self.t0 + len(self.buf) / self.rate, 6))
        vec = np.asarray(self.rd.push(frames, idx, fps))
        self.buf = []
        return [(span, vec)]


GENERIC_SENTENCES = 512


def generic_sentences() -> list[str]:
    """The fixed sample of generic English this mean is taken over: 512
    WordNet glosses of verbs and nouns, drawn once at seed 0 and written
    down.

    Written down rather than drawn on demand because the draw needed nltk
    and a 10 MB corpus download. Neither is in the encoder image and a
    customer deployment has no way to add them, so the first ingest into a
    new store ended in ModuleNotFoundError with the segment already in the
    bucket. The draw was seeded, so the sample was never varying; only its
    source was. scripts/gen_generic_sentences.py rebuilds this file with
    the same draw.
    """
    return SENTENCES_FILE.read_text().splitlines()


def generic_text_mean(encoder, n: int = GENERIC_SENTENCES, batch: int = 32):
    """The text tower's common direction over generic English: no task, no
    domain, one vector per model."""
    from tqdm import tqdm
    xs = generic_sentences()[:n]
    out = []
    bar = tqdm(total=len(xs), unit="sentence", desc=f"text mean: {getattr(encoder, 'id', '?')}")
    with bar:
        for i in range(0, len(xs), batch):
            out.append(encoder.embed_text(xs[i:i + batch]))
            bar.update(len(out[-1]))
    return np.concatenate(out).mean(0)


def text_mean_path(base, model_id: str) -> Path:
    return Path(base) / "streamtext" / model_id.replace("/", "__") / "text_mean.npz"


def load_text_mean(base, encoder):
    """The cached model-level text mean, computed once per model."""
    p = text_mean_path(base, encoder.id)
    if p.exists():
        return np.load(p)["mean"]
    m = generic_text_mean(encoder)
    if m is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, mean=m, n=GENERIC_SENTENCES, seed=0)
    return m


class ChangeChannel:
    """The stream's change at frame rate, read in whatever decode is already
    running: the mean absolute difference of the grey CHANGE_SIZE picture
    against the previous frame, one scalar per frame, the first frame zero.

    This pixel-change channel augments memory-state novelty when proposing
    event boundaries. Frames arrive in RGB from video.scan_object; the
    channel does not classify actions or decide their final query extents.

    Cheap enough to sit in every ingest: one resize and one subtract per
    frame, no model.
    """

    def __init__(self, fps: float | None = None, size: int = CHANGE_SIZE):
        self.fps, self.size = (float(fps) if fps else None), int(size)
        self.t: list = []
        self.d: list = []
        self._prev = None

    def push(self, t: float, frame) -> None:
        import cv2
        g = cv2.resize(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY), (self.size, self.size),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        self.t.append(float(t))
        self.d.append(0.0 if self._prev is None else float(np.abs(g - self._prev).mean()))
        self._prev = g

    def done(self) -> tuple:
        return np.asarray(self.t, np.float64), np.asarray(self.d, np.float32)

    def save(self, path, source="") -> None:
        t, d = self.done()
        if not len(t):
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=t, d=d, meta=json.dumps(dict(fps=float(self.fps or 0.0), size=self.size, source=str(source))))


def load_change(path) -> tuple:
    """The stream's change channel (scripts/build_change_channel.py): the
    mean absolute change of the grey picture from one frame to the next,
    at the stream's own frame rate. Returns (t, d)."""
    z = np.load(path, allow_pickle=False)
    return np.asarray(z["t"], np.float64), np.asarray(z["d"], np.float32)


def save(path, vecs, spans, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vecs=np.asarray(vecs, np.float32), spans=np.asarray(spans, np.float64),
             meta=np.array(json.dumps(meta)))
    return path


def load(path):
    z = np.load(path, allow_pickle=False)
    spans = [(float(a), float(b)) for a, b in z["spans"]]
    return z["vecs"], spans, json.loads(str(z["meta"]))


def make_encoder(model_id: str | None = None, **kw):
    """Load the current world-model tower; never substitute another family."""
    model_id = model_id or DEFAULT_MODEL
    if not _is_cosmos3(model_id):
        raise ValueError(f"Unsupported tower {model_id!r}; rebuild with nvidia/Cosmos3-Edge.")
    from .cosmos3 import Cosmos3Embed
    return Cosmos3Embed(model_id, **kw)


def _is_cosmos3(model_id: str) -> bool:
    return model_id == DEFAULT_MODEL


def tower_traits(model_id: str | None = None) -> dict:
    """The current world's text question and uncentred state space."""
    model_id = model_id or DEFAULT_MODEL
    if not _is_cosmos3(model_id):
        raise ValueError(f"Unsupported tower {model_id!r}; rebuild with nvidia/Cosmos3-Edge.")
    from . import cosmos3 as C3
    return dict(centre=False, prompt=C3.VIDEO_PROMPT)
