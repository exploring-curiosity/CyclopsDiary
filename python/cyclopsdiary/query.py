"""The query layer: one example clip in, the moments it names out, as rows in MongoDB.

The example is a span of footage already in the memory (footage id, t0, t1),
or a clip file, which is first read by the world model like any footage and
kept with role "example" so it never comes back as an answer. The search is
the tracker's: the output-head formula per step, aligned over time by the
subsequence DTW (memory.Memory.search).

direction
  forward   moments like the example (a key put on the table: other put-downs).
  reverse   the example undone: its steps asked in the opposite order, so the
            alignment wants the example's last moment first and its first
            moment last (a key put on the table, reversed: the key leaving
            the table). Only the order of the example's rows changes; the
            footage and the model's read are untouched. Unmeasured: no labelled
            removals exist yet, so it carries no accuracy figure.

scope (which footage may answer; the set the formula centres on is always
the whole workspace)
  all       every camera, the example's own span excluded (what `where` does)
  others    only other recordings than the example's
  after     only moments observed after the example ends

A query is a row in `queries`, so anything that can write MongoDB can ask
(the partner's agent, another machine): insert it with status "queued" and
`bin/cyclopsdiary serve` answers it; or answer it at once with `ask`. Each answer
is a row in `events`: the moment, its camera and person, when it happened in
the world (observed_at), its score, and whether it stands apart from the rest
(elidedb.corpus.split, the scores' own cut -- no constant).
"""
from __future__ import annotations

import os
import re
import socket
import time
from datetime import datetime, timedelta, timezone

from pymongo import ReturnDocument

from . import atlas

DIRECTIONS = ("forward", "reverse")
SCOPES = ("all", "others", "after")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue(db, workspace: str, footage: str, t0: float, t1: float, direction: str = "forward",
            scope: str = "all", top: int = 10, label: str | None = None, asked_by: str | None = None) -> dict:
    """A query row, status "queued": `serve` answers it."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r}: one of {', '.join(DIRECTIONS)}")
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r}: one of {', '.join(SCOPES)}")
    if not float(t1) > float(t0):
        raise ValueError("the example must end after it starts")
    f = db.footage.find_one({"_id": footage, "workspace": workspace})
    if f is None:
        raise ValueError(f"no footage {footage!r} in workspace {workspace!r}")
    doc = dict(workspace=workspace, example=dict(footage=footage, t0=float(t0), t1=float(t1)),
               direction=direction, scope=scope, top=int(top), label=label, asked_by=asked_by,
               status="queued", asked_at=_now())
    doc["_id"] = db.queries.insert_one(doc).inserted_id
    return doc


def claim(db, query_id=None) -> dict | None:
    """The oldest queued query (or query_id, if still queued), marked running by this worker in one atomic
    step, so several workers are safe."""
    return db.queries.find_one_and_update(
        {"status": "queued"} if query_id is None else {"_id": query_id, "status": "queued"},
        {"$set": {"status": "running", "started_at": _now(), "worker": f"{socket.gethostname()}:{os.getpid()}"}},
        sort=[("asked_at", 1)], return_document=ReturnDocument.AFTER)


def _answers(mem, q: dict) -> list:
    ex = q["example"]
    ef = mem.footage.get(ex["footage"])
    if ef is None:
        raise ValueError(f"footage {ex['footage']} is not ready in workspace {q['workspace']}")
    recs = {fid for fid, f in mem.footage.items() if f.get("role") != "example"}
    if q["scope"] == "others":
        recs.discard(ex["footage"])
    end = ef["started_at"] + timedelta(seconds=float(ex["t1"]))
    if q["scope"] == "after":
        recs = {r for r in recs
                if mem.footage[r]["started_at"] + timedelta(seconds=float(mem.footage[r].get("duration_s") or 0)) > end}
    if not recs:
        return []
    got = mem.search(ex["footage"], ex["t0"], ex["t1"], top=mem.n if q["scope"] == "after" else q["top"],
                     reverse=q["direction"] == "reverse", recs=recs)
    if q["scope"] == "after":
        got = [c for c in got if c["observed_at"] >= end]
    return got[:q["top"]]


def ask(db, workspace: str, footage: str, t0: float, t1: float, here: bool | None = None, **kw) -> dict:
    """Queue a query and answer it here and now (the CLI's and the agent tools' path); if a worker claimed it
    first, or this Python cannot run the search (here=False; by default: no numpy, as in the agent's own
    environment), wait for `serve` to answer. -> the query row as stored."""
    import importlib.util
    q = enqueue(db, workspace, footage, t0, t1, **kw)
    here = importlib.util.find_spec("numpy") is not None if here is None else here
    got = claim(db, q["_id"]) if here else None
    if got is None:
        return wait(db, q["_id"])
    try:
        return answer(db, got)
    except Exception as e:
        fail(db, got, e)
        raise


def wait(db, query_id, timeout_s: float = 60.0) -> dict:
    """The query row once it is done or failed (a change stream where the server has one, else polling)."""
    end = time.monotonic() + timeout_s
    cs = atlas.change_stream(db.queries, {"documentKey._id": query_id}, 1.0)
    try:
        while time.monotonic() < end:
            q = db.queries.find_one({"_id": query_id})
            if q is None or q["status"] in ("done", "failed"):
                return q
            if cs is not None:
                cs.try_next()
            else:
                time.sleep(0.25)
    finally:
        if cs is not None:
            cs.close()
    raise TimeoutError(f"query {query_id} not answered in {timeout_s:g} s: is `bin/cyclopsdiary serve` running?")


def requeue_stale(db) -> int:
    """Queries left "running" by a worker on this machine that is no longer alive go back to "queued"
    (a worker killed mid-answer); a live worker's claims are left alone."""
    host = socket.gethostname()
    n = 0
    for q in db.queries.find({"status": "running", "worker": {"$regex": f"^{re.escape(host)}:"}}, {"worker": 1}):
        pid = int(q["worker"].rsplit(":", 1)[1])
        try:
            os.kill(pid, 0)
            continue                                       # alive
        except ProcessLookupError:
            pass
        except PermissionError:
            continue                                       # alive, someone else's
        n += db.queries.update_one({"_id": q["_id"], "status": "running"},
                                   {"$set": {"status": "queued"}, "$unset": {"worker": ""}}).modified_count
    return n


def answer(db, q: dict) -> dict:
    """Search for query row q, write its events, mark it done. -> the query row as stored."""
    from elidedb import corpus
    from . import memory
    mem = memory.load(db, q["workspace"])
    found = _answers(mem, q)
    cut = corpus.split([c["score"] for c in found]) if found else None
    now = _now()
    db.events.delete_many({"query": q["_id"]})                  # a re-answered query replaces its events
    rows = []
    for rank, c in enumerate(found, 1):
        c["apart"] = cut is not None and c["score"] > cut
        rows.append(dict(query=q["_id"], workspace=q["workspace"], kind=q["direction"], label=q.get("label"),
                         rank=rank, footage=c["footage"], source=c["source"], person=c["person"],
                         t0=c["t0"], t1=c["t1"], observed_at=c["observed_at"], score=c["score"],
                         apart=c["apart"], recorded_at=now))
    if rows:
        db.events.insert_many(rows)
    db.queries.update_one({"_id": q["_id"]}, {"$set": dict(
        status="done", answered_at=now, cut=cut, events=len(rows),
        apart=sum(r["apart"] for r in rows), searched=dict(footage=len(mem.footage), steps=mem.n))})
    return db.queries.find_one({"_id": q["_id"]})


def fail(db, q: dict, err: BaseException) -> None:
    db.queries.update_one({"_id": q["_id"]}, {"$set": dict(status="failed", answered_at=_now(),
                                                            error=f"{type(err).__name__}: {err}"[:500])})


def events(db, query_id) -> list:
    return list(db.events.find({"query": query_id}).sort("rank", 1))


def timeline(db, workspace: str, label: str | None = None, kind: str | None = None, since: datetime | None = None,
             until: datetime | None = None, apart_only: bool = True, limit: int = 20) -> list:
    """The moments the workspace's queries found, newest first in the world's time. A moment found by several
    queries counts once, with its best score and how many queries found it (found_by)."""
    m = {"workspace": workspace}
    if apart_only:
        m["apart"] = True
    if kind:
        m["kind"] = kind
    if label:
        m["label"] = {"$regex": re.escape(label), "$options": "i"}
    if since or until:
        m["observed_at"] = {**({"$gte": since} if since else {}), **({"$lte": until} if until else {})}
    return list(db.events.aggregate([
        {"$match": m},
        {"$sort": {"score": -1}},
        {"$group": {"_id": {"footage": "$footage", "t0": "$t0", "t1": "$t1", "kind": "$kind"},
                    "best": {"$first": "$$ROOT"}, "found_by": {"$sum": 1}}},
        {"$replaceWith": {"$mergeObjects": ["$best", {"found_by": "$found_by"}]}},
        {"$sort": {"observed_at": -1}},
        {"$limit": int(limit)},
        {"$project": {"_id": 0, "workspace": 0, "recorded_at": 0}}]))
