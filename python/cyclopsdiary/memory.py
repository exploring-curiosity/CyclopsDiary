"""The shared memory read back: every step of a workspace's footage as one set,
searched by example with the output-head formula (elidedb.headsearch).

The set is the workspace's -- every person's cameras -- so the mean the
formula centres on is that set's own. Footage read under different pins (a
different model, rate or word count) is not one space and is refused rather
than mixed.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from elidedb import headsearch as HS

from . import config

PIN_KEYS = ("model", "rate", "n_frames", "top_k")


class Memory:
    def __init__(self, footage: dict, ids, lp, rec, t0, atom_s: float, union_k: int):
        self.footage, self.atom_s, self.union_k = footage, float(atom_s), int(union_k)
        self.ids, self.lp = ids, lp
        self.rec, self.t0 = np.asarray(rec), np.asarray(t0, np.float64)

    @property
    def n(self) -> int:
        return len(self.rec)

    def search(self, footage: str, t0: float, t1: float, top: int = 10, reverse: bool = False,
               recs=None) -> list:
        """The example -- a span of one footage -- asked of every step of the set.
        reverse: the example's steps in the opposite order (the example undone). recs: the footage answers
        may come from (all by default); the formula still centres on the whole set.
        -> ranked [dict(footage, source, person, t0, t1, score, observed_at)], the example itself excluded."""
        if self.n == 0:
            return []
        hs = HS.HeadSearch.from_lists(self.ids, self.lp, self.rec, self.t0, self.atom_s, self.union_k)
        A = hs.atoms_of(footage, float(t0), float(t1))
        if len(A) == 0:
            raise ValueError(f"no steps of footage {footage} inside {t0}-{t1} s")
        if reverse:
            A = A[::-1].copy()
        out = []
        for c in hs.find(A, exclude=(float(t0), float(t1)), top=top, recs=recs):
            f = self.footage[c["rec"]]
            a, b = c["span"]
            out.append(dict(footage=c["rec"], source=f["source"], person=f.get("person"), t0=float(a),
                            t1=float(b), score=float(c["score"]),
                            observed_at=f["started_at"] + timedelta(seconds=float(a))))
        return out


def load(db, workspace: str) -> Memory:
    s = config.get()
    # "live": a phone still streaming (live.py); its steps so far are final, so it is searched as it grows
    fl = list(db.footage.find({"workspace": workspace, "status": {"$in": ["ready", "live"]}}).sort("started_at", 1))
    pins = {tuple(f["pin"][k] for k in PIN_KEYS) for f in fl}
    if len(pins) > 1:
        raise ValueError(f"workspace {workspace!r} holds footage read under {len(pins)} different pins: {sorted(pins)}")
    ids, lp, rec, t0 = [], [], [], []
    for f in fl:
        for st in db.steps.find({"footage": f["_id"]}, {"ids": 1, "lp": 1, "t0": 1}).sort("i", 1):
            ids.append(np.frombuffer(st["ids"], np.int32))
            lp.append(np.frombuffer(st["lp"], np.float16).astype(np.float64))
            rec.append(f["_id"])
            t0.append(float(st["t0"]))
    atom = (float(fl[0]["pin"]["n_frames"]) / float(fl[0]["pin"]["rate"])) if fl else s.atom_s
    return Memory({f["_id"]: f for f in fl}, np.array(ids), np.array(lp), rec, t0, atom, s.union_k)
