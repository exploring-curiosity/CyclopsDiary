"""How a search found what it found, laid out for the demo stage: the tracker's own decision, step by step.

`explain_object` redoes what objects.track does by name -- every step of every camera scored by the
output-head formula (q uniform over the name's words, centred on the workspace's steps), cut where the
scores themselves split (corpus.split) -- and returns it as lanes, one per recording, so the stage can draw
the scores, the cut and the steps that stand apart. The trail and the last known location come from the
sightings the tracker stored, so what the stage shows is what the tracker decided. Private like the
tracker: an object is read for its owner only.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from . import memory, objects
from .tools import plain


def explain_object(db, owner: str, name: str, tok) -> dict:
    from elidedb import corpus
    tr = objects.get(db, owner, name)
    if tr is None:
        raise ValueError(f"{owner} has no object called {name!r}")
    mem = memory.load(db, tr["workspace"])
    s = objects.name_scores(mem, tr["words"]) if mem.n and tr.get("words") else np.zeros(mem.n)
    cut = corpus.split(s) if mem.n and tr.get("words") else None
    present = s > cut if cut is not None else np.zeros(mem.n, bool)
    lanes: dict = {}
    for j in range(mem.n):
        fid = mem.rec[j]
        f = mem.footage[fid]
        ln = lanes.setdefault(fid, dict(footage=fid, source=f["source"], person=f.get("person"),
                                        started_at=f["started_at"], status=f.get("status"),
                                        example=f.get("role") == "example", steps=[]))
        t0 = float(mem.t0[j])
        ln["steps"].append(dict(t0=t0, t1=t0 + mem.atom_s, observed_at=f["started_at"] + timedelta(seconds=t0),
                                score=float(s[j]), present=bool(present[j]) and not ln["example"]))
    words = [list(dict.fromkeys(tok.decode([t]).strip() for t in toks)) for toks in (tr.get("words") or [[]])[0]]
    return plain(dict(object=name, owner=owner, words=words, cut=None if cut is None else float(cut),
                      scores=[float(x) for x in s], lanes=list(lanes.values()),
                      trail=objects.trail(db, owner, name), last=objects.last_known(db, owner, name)))
