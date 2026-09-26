"""Relational reranking with a vision-language model.

WHY THIS EXISTS
---------------
SigLIP-style embeddings encode *what is present* in a frame, not *how things
relate*. "two people working together on a laptop" and "two people standing
near a laptop" land in almost the same place in embedding space, which is why
appearance-only search returns every clip containing two people.

A VLM actually reads the pixels and answers a grounded question about the
relation. It is ~1000x more expensive per window, so it is used exactly as a
database uses an expensive operator: LAST, on a small candidate set that
cheap operators already pruned.

    ts/stream predicates  ->  ANN/IVF shortlist  ->  exact cosine  ->  VLM
    (log zone maps)           (~40-80 vectors)      (rank)            (top N)

SCORING
-------
Generation is not used for scoring: small instruct VLMs are heavily
yes-biased and answer "Yes" to nearly any yes/no question (measured — both a
true and a false frame generated "Yes"). Instead we take ONE forward pass and
read the next-token distribution:

    score = max logP("Yes"|image,question) - max logP("No"|image,question)

That is a calibrated, continuous margin: positive leans yes, negative leans
no, and the magnitude is comparable across candidates. Measured on a known
true/false pair from the lab capture: TRUE +1.09 vs FALSE +0.53.

The final ordering fuses the retrieval score with the relational margin, so
a candidate must be both visually similar AND relationally correct.
"""
from __future__ import annotations

import re

import numpy as np

from .lexicon import derived_swaps

DEFAULT_VLM = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
_VLM_CACHE: dict = {}


def _load(model_id: str):
    if model_id not in _VLM_CACHE:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, processor = load(model_id)
        cfg = load_config(model_id)
        tok = processor.tokenizer
        yes = sorted({tok.encode(s)[0] for s in ("Yes", "yes", " Yes")})
        no = sorted({tok.encode(s)[0] for s in ("No", "no", " No")})
        _VLM_CACHE[model_id] = (model, processor, cfg, yes, no)
    return _VLM_CACHE[model_id]


def as_question(query: str) -> str:
    """Turn a retrieval phrase into a grounded yes/no question.

    The compositional operators are prose here: the VLM reasons over the
    whole sentence, which is precisely the capability embeddings lack."""
    q = query.strip()
    q = re.sub(r"\s+AND\s+", " and ", q, flags=re.I)
    q = re.sub(r"\s+NOT\s+", " but no ", q, flags=re.I)
    q = re.sub(r"(^|\s)-(\w)", r"\1no \2", q)
    if not q:
        return "Is anything notable happening? Answer yes or no."
    return (f"Does this image show: {q}? "
            "Answer only yes or no.")


def score_images(images, question: str, model_id: str = DEFAULT_VLM):
    """P(yes) - P(no) margin per image, in one forward pass each."""
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg, yes_ids, no_ids = _load(model_id)
    prompt = apply_chat_template(processor, cfg, question, num_images=1)
    out = []
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.gettempdir()) / "_elidedb_rerank.jpg"
    for im in images:
        im.save(tmp, "JPEG", quality=88)
        r = generate(model, processor, prompt, image=[str(tmp)],
                     max_tokens=1, verbose=False)
        if r.logprobs is None:
            out.append(0.0)
            continue
        a = mx.array(r.logprobs).reshape(-1)
        y = max(float(a[i]) for i in yes_ids)
        n = max(float(a[i]) for i in no_ids)
        out.append(y - n)
    return out


def rerank_hits(store, hits, query, top_n: int = 12, alpha: float = 0.7,
                model_id: str = DEFAULT_VLM, width: int = 512):
    """Re-order retrieval hits by relational correctness.

    Only the first `top_n` hits are examined (the expensive operator runs
    last, on a pruned set). `alpha` weights the VLM margin against the
    retrieval score; both are rank-normalised so they are commensurable.
    Returns (hits, info) with `vlm` and `fused` attached to each reranked hit.
    """
    from PIL import Image
    if not hits:
        return hits, {"reranked": 0}
    head, tail = hits[:top_n], hits[top_n:]
    rot = store.meta.get("display", {}).get("rotate", 0)

    frames, keep = [], []
    for h in head:
        mid = (h["t0"] + h["t1"]) // 2
        w, _ = store.window(mid - 500_000_000, mid + 500_000_000,
                            tables=["frames"])
        fs = w.get("frames")
        dec = fs.decode(stream=h["stream"], width=width, limit=1) if fs else []
        if not dec:
            continue
        im = Image.fromarray(dec[0][1])
        if rot:
            im = im.rotate(rot, expand=True)
        frames.append(im)
        keep.append(h)
    if not frames:
        return hits, {"reranked": 0, "note": "no frames decodable"}

    question = as_question(query)
    margins = score_images(frames, question, model_id)

    # rank-normalise both signals to [0,1] so alpha is meaningful
    def ranknorm(v):
        v = np.asarray(v, dtype=float)
        if len(v) < 2:
            return np.ones_like(v)
        r = v.argsort().argsort().astype(float)
        return r / (len(v) - 1)
    rn_vlm = ranknorm(margins)
    rn_ret = ranknorm([h["score"] for h in keep])
    for h, m, fv, fr in zip(keep, margins, rn_vlm, rn_ret):
        h["vlm"] = round(float(m), 3)
        h["fused"] = round(float(alpha * fv + (1 - alpha) * fr), 4)
    keep.sort(key=lambda h: -h["fused"])
    return keep + tail, {
        "reranked": len(keep), "question": question, "model": model_id,
        "vlm_min": round(float(min(margins)), 3),
        "vlm_max": round(float(max(margins)), 3)}


# ===========================================================================
# Multi-frame (clip-level) verification — actions live BETWEEN frames
# ===========================================================================
def directional_swap(query: str, store=None) -> str | None:
    """The query with its first STATE-REVERSAL term inverted (open<->close,
    into<->out of, ...), or None when the query has no true direction.

    WHY: measured on ground-truth close/open episode clips, BOTH VLM tiers
    are direction-INVERTED on absolute before/after questions (2B AUC 0.36,
    7B 0.36, and a 'direction matters' phrasing made 7B worse at 0.25) —
    they score salient-drawer-interaction, not direction. But the direction
    information exists: scoring the query AND its swap and taking the
    DIFFERENCE cancels the appearance bias by construction — 2B 0.86,
    7B 0.91. Deterministic (first applicable swap) so cached margins keyed
    by query stay stable.

    Bare adverb pairs (up/down, left/right, front/back) are EXCLUDED here:
    they match incidental particles and produce nonsense swaps — measured
    live when "pick up a green toy..." swapped to "pick DOWN..." and the
    motion channel fired at weight 2.5 on a garbage direction, putting a
    pot-on-stove clip at rank 1. A swap that is not a meaningful sentence
    is worse than no swap.
    Multiword pairs are tried first so "picks up" wins before any single
    word could."""
    # THE CORPUS IS THE ONLY SOURCE. This used to import VERB_SWAPS -
    # 81 hand-authored pairs - and never call derived_swaps at all, so
    # the direction mechanism, the one thing separating "opens the
    # drawer" from "closes the drawer", was hand-written English. Making
    # it a "fallback" was not a fix: derived_swaps returns 0 pairs on
    # every store measured, so the fallback fired every time and the
    # hand list remained the mechanism.
    #
    # Gone. A corpus that cannot attest an opposition yields None, and
    # the caller loses its contrast rather than borrowing one. That is
    # the honest state of a corpus with no vocabulary, and it is visible
    # in the numbers instead of hidden behind a list.
    pairs = list(derived_swaps(store)) if store is not None else []
    if not pairs:
        return None
    weak = {frozenset(p) for p in ((("left", "right")),
                                   (("up", "down")),
                                   (("front", "back")))}
    pairs = [p for p in pairs if frozenset(p) not in weak]
    pairs.sort(key=lambda p: -max(len(p[0]), len(p[1])))
    t = " " + query.lower() + " "
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            if f" {x} " in t:
                return re.sub(rf"\b{re.escape(x)}\b", y, query.lower(),
                              count=1)
    return None


def as_clip_question(query: str) -> str:
    """The clip-level question. Unlike the single-frame form, this one hands
    the VLM the frames in time order and asks about the EVENT — which is the
    only place 'put X in the drawer AND CLOSE IT' can be checked, because no
    single frame contains a verb."""
    q = re.sub(r"\s+AND\s+", " and ", query.strip(), flags=re.I)
    q = re.sub(r"\s+NOT\s+", " but not ", q, flags=re.I)
    return (f"These frames are in time order from one short clip. "
            f"Does the clip show this happening: {q}? "
            "Answer only yes or no.")


def score_clip_sequences(clips, question, model_id: str = DEFAULT_VLM):
    """logP(yes)-logP(no) margin per CLIP, each clip = frames in time order.

    Same calibrated-margin trick as score_images (generation is yes-biased),
    but the evidence is a sequence, so the verb is finally visible to the
    scorer. `clips` = list of lists of PIL images (2-6 frames each).
    """
    import tempfile
    from pathlib import Path

    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, cfg, yes_ids, no_ids = _load(model_id)
    tmpdir = Path(tempfile.mkdtemp(prefix="elidedb_clipv_"))
    out = []
    for ci, frames in enumerate(clips):
        prompt = apply_chat_template(processor, cfg, question,
                                     num_images=len(frames))
        paths = []
        for j, im in enumerate(frames):
            fp = tmpdir / f"c{ci}_{j}.jpg"
            im.save(fp, "JPEG", quality=85)
            paths.append(str(fp))
        r = generate(model, processor, prompt, image=paths,
                     max_tokens=1, verbose=False)
        if r.logprobs is None:
            out.append(0.0)
            continue
        a = mx.array(r.logprobs).reshape(-1)
        y = max(float(a[i]) for i in yes_ids)
        n = max(float(a[i]) for i in no_ids)
        out.append(y - n)
    return out


def as_change_question(query: str) -> str:
    """Before/after formulation: the FIRST and LAST frame of a clip.

    Measured on ground-truth clips (put-green-on-drawer, 5 true / 8 false):
    the 4-frame 'time order' question at 2B scored AUC 0.40 — literal true
    clips ranked BELOW false ones. The same 2B judging only the first and
    last frame scored AUC 0.75 at 0.5 s/clip. An action is a state change,
    and a before/after pair is the smallest complete evidence of one — and
    small models reason far better over 2 images than 4.
    (7B, 4-frame: AUC 0.82 at ~10x the cost — the `deep` tier.)
    """
    q = re.sub(r"\s+AND\s+", " and ", query.strip(), flags=re.I)
    return (f"The first image is the start of a short clip and the second "
            f"is the end. Did this happen in between: {q}? "
            "Answer only yes or no.")


DEEP_VLM = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
