"""The agent's tools: the Cosmos side's queries and the agent's own memory, as plain functions over MongoDB.

mcp_server.py serves them over MCP, so any MCP client -- the Strands agent (agent/), Claude Code, the
MongoDB-side tooling -- calls the same code. Every result is plain JSON (ids and times as strings).

The only writes are a query row (checked by the database's validator, answered by `serve` or here) and a
note in the agent's memory; nothing the cameras stored is ever written through a tool.
"""
from __future__ import annotations

from datetime import datetime

from bson import ObjectId

from . import agentmemory, objects, query


def plain(x):
    """BSON values made JSON: ObjectId -> str, datetime -> ISO 8601."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, ObjectId):
        return str(x)
    if isinstance(x, datetime):
        return x.isoformat()
    return x


def _when(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def footage(db, workspace: str) -> list:
    """The recordings in the workspace, oldest first: the ids and seconds an example is taken from."""
    return plain(list(db.footage.find(
        {"workspace": workspace, "status": "ready", "role": {"$ne": "example"}},
        {"source": 1, "person": 1, "started_at": 1, "duration_s": 1, "steps": 1}).sort("started_at", 1)))


def find_moments(db, workspace: str, footage: str, start_s: float, end_s: float, direction: str = "forward",
                 scope: str = "all", top: int = 10, label: str | None = None, person: str | None = None) -> dict:
    q = query.ask(db, workspace, footage, start_s, end_s, direction=direction, scope=scope, top=top, label=label,
                  asked_by=person)
    return _answer(db, q)


def moments(db, workspace: str, query_id: str) -> dict:
    q = db.queries.find_one({"_id": ObjectId(query_id), "workspace": workspace})
    if q is None:
        raise ValueError(f"no query {query_id} in workspace {workspace!r}")
    return _answer(db, q)


def _answer(db, q: dict) -> dict:
    out = dict(query=q["_id"], status=q["status"], direction=q["direction"], scope=q["scope"], label=q.get("label"),
               example=q["example"])
    if q["status"] == "failed":
        out["error"] = q.get("error")
    if q["status"] == "done":
        out.update(searched_steps=q["searched"]["steps"], stands_apart=q.get("cut") is not None,
                   moments=[{k: e[k] for k in ("rank", "footage", "source", "person", "t0", "t1", "observed_at",
                                                "score", "apart")} for e in query.events(db, q["_id"])])
    return plain(out)


def timeline(db, workspace: str, label: str | None = None, kind: str | None = None, since: str | None = None,
             until: str | None = None, apart_only: bool = True, limit: int = 20) -> list:
    return plain(query.timeline(db, workspace, label=label, kind=kind, since=_when(since), until=_when(until),
                                apart_only=apart_only, limit=limit))


def remember(db, workspace: str, person: str, subject: str, note: str, query_id: str | None = None,
             session: str | None = None) -> dict:
    return plain(agentmemory.remember(db, workspace, person, subject, note,
                                      query_id=ObjectId(query_id) if query_id else None, session=session))


def recall(db, workspace: str, person: str, text: str | None = None, subject: str | None = None,
           limit: int = 10) -> dict:
    return plain(agentmemory.recall(db, workspace, person, text=text, subject=subject, limit=limit))


def context(db, workspace: str, person: str | None = None) -> str:
    return agentmemory.render(agentmemory.context(db, workspace, person))


# ---- private objects (objects.py): each person's own, found by the world model's words and their examples

def _sighting(s: dict) -> dict:
    return dict(id=s["_id"], camera=s["source"], person=s.get("person"), footage=s["footage"], t0=s["t0"],
                t1=s["t1"], seen_from=s["observed_at"], seen_until=s["until"], near=s.get("place") or [],
                by=s["by"], confirmed=s.get("confirmed_at") is not None)


def _where(w: dict) -> dict:
    last = w["last"]
    return plain(dict(object=w["object"], trail=[_sighting(s) for s in w["trail"]],
                      last_seen=None if last is None else dict(_sighting(last),
                                                               near=last.get("end_place") or last.get("place") or [])))


def my_objects(db, person: str) -> list:
    return plain([dict(name=o["name"], aliases=o["aliases"], examples=o["examples"],
                       last_seen=None if o["last"] is None else _sighting(o["last"])) for o in objects.objects(db, person)])


def add_object(db, person: str, name: str, tok, aliases=(), footage: str | None = None, start_s: float | None = None,
               end_s: float | None = None, describe: str | None = None) -> dict:
    return _where(objects.add(db, person, name, tok, aliases=aliases, footage=footage, t0=start_s, t1=end_s,
                              describe=describe))


def where_is(db, person: str, name: str, tok=None) -> dict:
    """tok: track it again first (the model's tokenizer names the places); without it, what is stored."""
    if tok is not None:
        objects.track(db, person, name, tok)
    return _where(objects.where(db, person, name))


def confirm_sighting(db, person: str, sighting_id: str) -> dict:
    return dict(confirmed=objects.confirm(db, person, ObjectId(sighting_id)))


def reject_sighting(db, person: str, sighting_id: str) -> dict:
    return dict(rejected=objects.reject(db, person, ObjectId(sighting_id)))
