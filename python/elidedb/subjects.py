"""Self-observed query anchoring: the store learns its own salient subject.

MEASURED PROBLEM (100 h corpus, "folding the cloth"): SigLIP's text tower
only lands near the right images when the visually DOMINANT subject is
named — "a robot folding the cloth" ranks the true clip #17, while the
same query with "someone"/"an arm"/"a hand"/no subject ranks it #200-500.
Nine content-blind prompt templates were measured and none recovered it:
the anchor is the subject WORD, and it differs per corpus.

ABSOLUTE RULE (user-set): the system must never be told what a store
contains — no keywords, no metadata, no per-dataset tuning. So the store
finds the subject itself, by LOOKING: caption a small time-spread sample
of its own frames with the local VLM and keep the most frequent leading
noun phrase. On a manipulation corpus this discovers "the robot arm"; on
a dashcam corpus it would discover "a car". Derived purely from pixels,
recomputed per store version, identical machinery for any upload.
"""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

_SUBJ_CACHE = {}

# leading noun phrase = words before the first finite verb / gerund.
_VERBISH = re.compile(
    r"\b(is|are|was|were|has|have|had|can|will|appears?|seems?|sits?|"
    r"stands?|holds?|holding|moves?|moving|picks?|picking|puts?|putting|"
    r"reaches|reaching|grabs?|grabbing|lifts?|lifting|folds?|folding|"
    r"opens?|opening|closes?|closing|places?|placing|\w+ing|\w+s)\b")


_PREAMBLE = re.compile(
    r"^(the\s+(image|photo|picture|video|scene|frame)\s+"
    r"(shows|depicts|features|contains|is\s+of|is\s+showing)\s*|"
    r"the\s+main\s+(thing|object|subject|focus)"
    r"(\s+(in|of)\s+(the|this)\s+\w+)?\s+is\s*|"
    r"(in\s+(the|this)\s+\w+\s*,?\s*)|it\s+is\s+|there\s+is\s+)",
    re.IGNORECASE)


def _leading_phrase(caption: str) -> str | None:
    # VLMs frame answers with meta-language ("The image shows...", "The
    # main thing is...") — strip the framing, keep the subject it names
    prev = None
    caption = caption.strip()
    while prev != caption:
        prev = caption
        caption = _PREAMBLE.sub("", caption).strip()
    words = re.findall(r"[a-zA-Z']+", caption.lower())
    out = []
    for w in words[:6]:
        if _VERBISH.fullmatch(w) and len(out) >= 2:
            break
        out.append(w)
    # relative-clause residue is not part of the subject
    while out and out[-1] in ("which", "that", "who", "with", "and",
                              "in", "on", "of", "whose"):
        out.pop()
    # a usable subject is 2-4 words starting with a determiner-ish token
    if 2 <= len(out) <= 4 and out[0] in ("a", "an", "the", "two", "some"):
        return " ".join(out)
    return None


def subject_prefixes(store, sample=128, top=2, force=False, build=True):
    """The store's dominant subject phrases, mined from its own frames.
    Cached in the store as a derived artifact keyed by frames version.

    `build=False` (the QUERY path) only reads the cache — mining is a
    ~60 s VLM pass and must never run inside a search; the desk warms it
    and ingest can call it explicitly."""
    ver = store.table("frames").state().version
    key = (str(store.dir), ver)
    if key in _SUBJ_CACHE and not force:
        return _SUBJ_CACHE[key]
    cache = store.dir / "tables" / "frames" / "_subjects.json"
    if cache.exists() and not force:
        c = json.loads(cache.read_text())
        if c.get("version") == ver:
            _SUBJ_CACHE[key] = c["prefixes"]
            return c["prefixes"]
    if not build:
        return []

    from PIL import Image

    from .rerank import DEFAULT_VLM, _load
    from .video import FrameSet
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    frames = store.table("frames").scan()
    n = len(frames)
    if n == 0:
        return []
    idx = np.linspace(0, n - 1, min(sample, n)).round().astype(int)
    vlm, processor, cfg, _, _ = _load(DEFAULT_VLM)
    prompt = apply_chat_template(
        processor, cfg,
        "In one short sentence, what is the main thing in this image and "
        "what is it doing?", num_images=1)
    tmp = Path(tempfile.mkdtemp(prefix="elidedb_subj_"))
    phrases = Counter()
    for j, i in enumerate(idx):
        try:
            dec = FrameSet(store, "frames",
                           frames.take(np.array([int(i)]))).decode(width=448)
            if not dec:
                continue
            p = tmp / f"s{j}.jpg"
            Image.fromarray(dec[0][1]).save(p)
            r = generate(vlm, processor, prompt, image=[str(p)],
                         max_tokens=24, temperature=0.0, verbose=False)
            cap = r.text if hasattr(r, "text") else str(r)
            ph = _leading_phrase(cap)
            if ph:
                phrases[ph] += 1
        except Exception:
            continue
    # keep phrases that describe a MAJORITY view of the corpus, not a
    # one-off frame: at least 10% of sampled captions must agree
    floor = max(3, len(idx) // 10)
    out = [p for p, c in phrases.most_common(top) if c >= floor]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"version": ver, "prefixes": out,
                                 "sampled": int(len(idx)),
                                 "counts": dict(phrases.most_common(8))}))
    _SUBJ_CACHE[key] = out
    return out
