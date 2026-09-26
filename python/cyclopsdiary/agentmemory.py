"""The agent's memory in MongoDB: its conversations, its notes, and the context it starts from.

The agent (Strands; its model from OpenRouter) keeps nothing on its own disk. Three things live in the
workspace's database next to what the cameras saw:

  conversation  agent_sessions / agent_agents / agent_messages hold Strands' session model (Session,
                SessionAgent, SessionMessage) stored as its own to_dict(), so a conversation survives a
                restart and any machine can carry it on (agent/src/session.py is the Strands side).
                The model sees a sliding window of the conversation; every message stays here and
                `recall` finds the older ones again by their words.
  notes         agent_notes: what the agent was told or concluded ("the spare key lives in the blue
                bowl"), private to one person like a tracker, with the query that backs it when there
                is one. Found by subject or by words: a text index, no embeddings and no cosine.
  context       `context()`: the workspace as it stands, read fresh every time -- its cameras and
                recordings, the latest moments that stood apart, the queries still open, the person's
                latest notes -- for the agent's system prompt when a session starts.

The model's context window holds the last few turns; MongoDB holds every turn and everything the
cameras saw, and the agent pulls what it needs back in through its tools.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError

from . import query


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _strip(d: dict | None, *keys) -> dict | None:
    if d is not None:
        for k in ("_id", *keys):
            d.pop(k, None)
    return d


# ---- the conversation: Strands' session model, as dicts (Session/SessionAgent/SessionMessage.to_dict())

def session_create(db, session: dict, workspace: str, person: str | None = None) -> None:
    try:
        db.agent_sessions.insert_one(dict(session, _id=session["session_id"], workspace=workspace, person=person))
    except DuplicateKeyError:
        raise ValueError(f"session {session['session_id']} already exists") from None


def session_read(db, session_id: str) -> dict | None:
    return _strip(db.agent_sessions.find_one({"_id": session_id}), "workspace", "person")


def agent_create(db, session_id: str, agent: dict) -> None:
    db.agent_agents.replace_one({"_id": f"{session_id}/{agent['agent_id']}"}, dict(agent, session=session_id),
                                upsert=True)


def agent_read(db, session_id: str, agent_id: str) -> dict | None:
    return _strip(db.agent_agents.find_one({"_id": f"{session_id}/{agent_id}"}), "session")


def agent_update(db, session_id: str, agent: dict) -> bool:
    """False when there is no such agent. created_at stays the first write's."""
    got = db.agent_agents.update_one({"_id": f"{session_id}/{agent['agent_id']}"},
                                     {"$set": {k: v for k, v in agent.items() if k != "created_at"}})
    return got.matched_count == 1


def message_create(db, session_id: str, agent_id: str, message: dict) -> None:
    db.agent_messages.replace_one({"_id": f"{session_id}/{agent_id}/{int(message['message_id'])}"},
                                  dict(message, session=session_id, agent=agent_id), upsert=True)


def message_read(db, session_id: str, agent_id: str, message_id: int) -> dict | None:
    return _strip(db.agent_messages.find_one({"_id": f"{session_id}/{agent_id}/{int(message_id)}"}),
                  "session", "agent")


def message_update(db, session_id: str, agent_id: str, message: dict) -> bool:
    """False when there is no such message (Strands updates one only to redact it)."""
    got = db.agent_messages.update_one({"_id": f"{session_id}/{agent_id}/{int(message['message_id'])}"},
                                       {"$set": {k: v for k, v in message.items() if k != "created_at"}})
    return got.matched_count == 1


def messages(db, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0) -> list:
    cur = db.agent_messages.find({"session": session_id, "agent": agent_id},
                                 {"_id": 0, "session": 0, "agent": 0}).sort("message_id", 1).skip(int(offset))
    return list(cur.limit(int(limit)) if limit is not None else cur)


# ---- notes: private to a person, recalled by subject or by words

def remember(db, workspace: str, person: str, subject: str, note: str, query_id=None,
             session: str | None = None) -> dict:
    if db.people.find_one({"_id": person, "workspace": workspace}) is None:
        raise ValueError(f"no person {person!r} in workspace {workspace!r}")
    doc = dict(workspace=workspace, person=person, subject=subject.strip(), note=note.strip(),
               query=query_id, session=session, created_at=_now())
    doc["_id"] = db.agent_notes.insert_one(doc).inserted_id
    return doc


def recall(db, workspace: str, person: str, text: str | None = None, subject: str | None = None,
           limit: int = 10, exclude_session: str | None = None) -> dict:
    """The person's notes (by subject, by words, else the latest) and, when words are given, the lines of
    the person's conversations that carry them (exclude_session: leave out the one in progress, whose
    recent turns the model already sees). With words it is one aggregation, one round trip: the notes by
    text score, then ($unionWith) the matching lines whose session ($lookup) is the person's."""
    f = {"workspace": workspace, "person": person}
    if subject:
        f["subject"] = {"$regex": f"^{re.escape(subject.strip())}$", "$options": "i"}
    if not text:
        return dict(notes=list(db.agent_notes.find(f).sort("created_at", -1).limit(limit)), said=[])
    lines = [
        {"$match": {"$text": {"$search": text}, **({"session": {"$ne": exclude_session}} if exclude_session else {})}},
        {"$set": {"score": {"$meta": "textScore"}}},
        {"$sort": {"score": -1}},
        {"$lookup": {"from": "agent_sessions", "localField": "session", "foreignField": "_id", "as": "of",
                     "pipeline": [{"$project": {"workspace": 1, "person": 1}}]}},
        {"$match": {"of.workspace": workspace, "of.person": person}},
        {"$limit": limit},
        {"$project": {"_id": 0, "said": {"$literal": True}, "session": 1, "message": 1, "created_at": 1}}]
    out = dict(notes=[], said=[])
    for d in db.agent_notes.aggregate([{"$match": dict(f, **{"$text": {"$search": text}})},
                                       {"$set": {"score": {"$meta": "textScore"}}}, {"$sort": {"score": -1}},
                                       {"$limit": limit},
                                       {"$unionWith": {"coll": "agent_messages", "pipeline": lines}}]):
        if d.get("said") is not True:
            out["notes"].append(d)
            continue
        words = " ".join(c["text"] for c in d["message"].get("content", []) if "text" in c)
        out["said"].append(dict(session=d["session"], role=d["message"].get("role"), text=words[:1000],
                                at=d.get("created_at")))
    return out


# ---- the context a session starts from

def context(db, workspace: str, person: str | None = None, moments: int = 5, notes: int = 5) -> dict:
    cameras = list(db.footage.aggregate([
        {"$match": {"workspace": workspace, "status": "ready", "role": {"$ne": "example"}}},
        {"$group": {"_id": "$source", "person": {"$first": "$person"}, "recordings": {"$sum": 1},
                    "steps": {"$sum": "$steps"}, "first": {"$min": "$started_at"}, "last": {"$max": "$started_at"}}},
        {"$lookup": {"from": "sources", "localField": "_id", "foreignField": "_id", "as": "src"}},
        {"$project": {"_id": 0, "source": "$_id", "person": 1, "recordings": 1, "steps": 1, "first": 1, "last": 1,
                      "kind": {"$first": "$src.kind"}}},
        {"$sort": {"source": 1}}]))
    open_q = db.queries.count_documents({"workspace": workspace, "status": {"$in": ["queued", "running"]}})
    failed = list(db.queries.find({"workspace": workspace, "status": "failed"}, {"label": 1, "error": 1})
                  .sort("asked_at", -1).limit(3))
    mine = [] if person is None else list(db.agent_notes.find({"workspace": workspace, "person": person},
                                                              {"subject": 1, "note": 1, "created_at": 1})
                                          .sort("created_at", -1).limit(notes))
    return dict(workspace=workspace, person=person, cameras=cameras, open_queries=open_q, failed_queries=failed,
                moments=query.timeline(db, workspace, limit=moments), notes=mine, at=_now())


def render(ctx: dict) -> str:
    """The context as a few lines for a system prompt."""
    t = lambda d: d.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if d else "-"  # noqa: E731
    lines = [f"Workspace {ctx['workspace']}" + (f", speaking with {ctx['person']}" if ctx["person"] else "")
             + f"; as of {t(ctx['at'])}."]
    lines.append("Cameras: " + ("; ".join(
        f"{c['source']} ({c.get('kind') or '?'}, {c.get('person') or '?'}): {c['recordings']} recordings, "
        f"{c['steps']} s read, {t(c['first'])} to {t(c['last'])}" for c in ctx["cameras"]) or "none yet") + ".")
    if ctx["moments"]:
        lines.append("Latest moments that stood apart: " + "; ".join(
            f"{m.get('label') or m['kind']} at {t(m['observed_at'])} on {m['source']} "
            f"({m['footage']} {m['t0']:g}-{m['t1']:g} s)" for m in ctx["moments"]) + ".")
    if ctx["open_queries"]:
        lines.append(f"{ctx['open_queries']} queries still being answered.")
    for q in ctx["failed_queries"]:
        lines.append(f"A query failed ({q.get('label') or q['_id']}): {q.get('error')}.")
    if ctx["notes"]:
        lines.append("Notes: " + "; ".join(f"{n['subject']}: {n['note']}" for n in ctx["notes"]) + ".")
    return "\n".join(lines)
