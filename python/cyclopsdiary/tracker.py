"""The object tracker: private to its owner, searching footage the whole workspace shares.

An object is registered by EXAMPLE: a span of any footage in the owner's
workspace where it is in view ("these seconds show my keys"). That moment is
the first sighting, confirmed by the act of registering it. `locate` asks
every step of every camera in the workspace for that moment again
(memory.search: the example's own word distribution against every step's,
aligned by DTW), keeps the answers that stand apart from the rest -- the cut
is the scores' own (elidedb.corpus.split), not a constant, so a search can
also say that nothing stands out -- and records each answer as a sighting.

The last known location is a belief read off the records every time: the
latest sighting that the owner confirmed or the latest search set apart,
and that the owner has not rejected. A new sighting or a rejection changes
it at once; nothing has to be recomputed or invalidated.

Every read and write here filters on the owner.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

from elidedb import corpus

from . import memory


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get(db, owner: str, name: str) -> dict | None:
    return db.trackers.find_one({"owner": owner, "name": name})


def trackers(db, owner: str) -> list:
    return list(db.trackers.find({"owner": owner}).sort("name", 1))


def _record(db, tr: dict, f: dict, t0: float, t1: float, now: datetime, kind: str, score=None, apart=False,
            confirmed_at=None) -> None:
    key = dict(tracker=tr["_id"], footage=f["_id"], t0=float(t0), t1=float(t1))
    db.sightings.update_one(key, {
        "$setOnInsert": dict(owner=tr["owner"], workspace=tr["workspace"], source=f["source"],
                             person=f.get("person"), kind=kind,
                             observed_at=f["started_at"] + timedelta(seconds=float(t0)),
                             recorded_at=now, confirmed_at=confirmed_at, rejected_at=None),
        "$set": dict(score=score, apart=bool(apart), found_at=now)}, upsert=True)


def register(db, owner: str, name: str, footage: str, t0: float, t1: float) -> dict:
    p = db.people.find_one({"_id": owner})
    if p is None:
        raise ValueError(f"no person {owner!r}")
    f = db.footage.find_one({"_id": footage, "workspace": p["workspace"]})
    if f is None:
        raise ValueError(f"no footage {footage!r} in {owner}'s workspace")
    if not float(t1) > float(t0):
        raise ValueError("the example must end after it starts")
    now = _now()
    doc = dict(workspace=p["workspace"], owner=owner, name=name,
               example=dict(footage=footage, t0=float(t0), t1=float(t1)), created_at=now)
    try:
        doc["_id"] = db.trackers.insert_one(doc).inserted_id
    except DuplicateKeyError:
        raise ValueError(f"{owner} already tracks {name!r}") from None
    _record(db, doc, f, t0, t1, now, kind="example", apart=True, confirmed_at=now)
    return doc


def locate(db, owner: str, name: str, top: int = 10) -> dict:
    tr = get(db, owner, name)
    if tr is None:
        raise ValueError(f"{owner} tracks nothing called {name!r}")
    mem = memory.load(db, tr["workspace"])
    ex = tr["example"]
    found = mem.search(ex["footage"], ex["t0"], ex["t1"], top=top)
    cut = corpus.split([c["score"] for c in found]) if found else None
    now = _now()
    db.sightings.update_many({"tracker": tr["_id"], "owner": owner, "kind": "found"}, {"$set": {"apart": False}})
    for c in found:
        c["apart"] = cut is not None and c["score"] > cut
        _record(db, tr, mem.footage[c["footage"]], c["t0"], c["t1"], now, kind="found", score=c["score"],
                apart=c["apart"])
    return dict(tracker=tr, found=found, cut=cut, searched=dict(footage=len(mem.footage), steps=mem.n),
                last_known=last_known(db, owner, name))


def last_known(db, owner: str, name: str) -> dict | None:
    tr = get(db, owner, name)
    if tr is None:
        return None
    return db.sightings.find_one({"tracker": tr["_id"], "owner": owner, "rejected_at": None,
                                  "$or": [{"confirmed_at": {"$ne": None}}, {"apart": True}]},
                                 sort=[("observed_at", -1)])


def confirm(db, owner: str, sighting) -> bool:
    return db.sightings.update_one({"_id": sighting, "owner": owner, "rejected_at": None},
                                   {"$set": {"confirmed_at": _now()}}).modified_count == 1


def reject(db, owner: str, sighting) -> bool:
    """A sighting the owner says is not the object. The example itself cannot be rejected: it is the
    definition of the object."""
    return db.sightings.update_one({"_id": sighting, "owner": owner, "kind": "found"},
                                   {"$set": {"rejected_at": _now()}}).modified_count == 1
