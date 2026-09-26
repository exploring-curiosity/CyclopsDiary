"""MongoDB Atlas as the memory: the connection, the collections, their indexes.

Twelve collections in one database (MONGODB_DB, default "cyclopsdiary"): eight for what the
cameras saw, four for the agent's own memory (agentmemory.py).

  people     {_id, workspace, name}
  sources    {_id, workspace, person, kind, label}       one camera: glasses, phone, webcam
  footage    {_id, workspace, source, person, path, sha256, bytes, width, height, fps,
              duration_s, rotation, started_at, pin, status, steps, bytes_read}
  steps      {_id "<footage>:<i>", workspace, source, person, footage, i, t0, t1,
              observed_at, state, ids, lp}               one per 1-s step of the model's read
  trackers   {_id, workspace, owner, name, example {footage, t0, t1}, created_at}
  sightings  {_id, tracker, owner, workspace, footage, source, person, t0, t1, kind,
              score, apart, observed_at, recorded_at, found_at, confirmed_at, rejected_at}
  queries    {_id, workspace, example {footage, t0, t1}, direction, scope, top, label, asked_by,
              status queued|running|done|failed, asked_at, started_at, answered_at, worker,
              cut, events, apart, searched, error}             one example asked of the workspace
  events     {_id, query, workspace, kind, label, rank, footage, source, person, t0, t1,
              observed_at, score, apart, recorded_at}          the moments a query found (query.py)

  agent_sessions  {_id session_id, session_type, workspace, person, created_at, updated_at}
  agent_agents    {_id "<session>/<agent>", session, agent_id, state, conversation_manager_state, ...}
  agent_messages  {_id "<session>/<agent>/<n>", session, agent, message_id, message, redact_message, ...}
  agent_notes     {_id, workspace, person, subject, note, query, session, created_at}

The first three are Strands' session model stored as its own to_dict(); agent_notes is what the
agent was told or concluded. Messages and notes carry a text index for recall by words.

A footage row made from an example clip (`ask --clip`) has role "example": it
is searched as part of the set but never returned as an answer. A source may
carry clock_offset_s, added to its recordings' own clock at ingest (the sync
shot: both phones film the same clock at the start of a session).

Shared and private are decided in the data. Footage and steps belong to a
WORKSPACE, so everyone in it searches everyone's cameras. Trackers and
sightings belong to an OWNER, and every read of them (tracker.py) filters on
it; two people can each track "keys".

A step holds the world model's own output-head row -- its 512 best word ids
(int32) and their log-probabilities (float16), 3 KB -- and the memory state
(float16, 4 KB), as BSON binary. The search reads the word rows; the state is
kept so a later reader never needs the video again. Its _id "<footage>:<i>"
makes a second write of the same step a duplicate-key no-op, which is what
makes an interrupted ingest safe to run again.

Record vs belief: a sighting keeps three clocks apart -- observed_at, when it
happened in the world (the footage's clock); recorded_at, when the system
found it; confirmed_at, when the owner said yes. The last known location is
never stored; it is read off the sightings each time.
"""
from __future__ import annotations

import re

import pymongo
from pymongo import ASCENDING, DESCENDING, TEXT

from . import config

COLLECTIONS = ("people", "sources", "footage", "steps", "trackers", "sightings", "queries", "events",
               "agent_sessions", "agent_agents", "agent_messages", "agent_notes")
INDEXES = {
    "people": [[("workspace", ASCENDING)]],
    "sources": [[("workspace", ASCENDING)], [("person", ASCENDING)]],
    "footage": [[("workspace", ASCENDING), ("started_at", ASCENDING)], [("source", ASCENDING)],
                [("sha256", ASCENDING)]],
    "steps": [[("footage", ASCENDING), ("i", ASCENDING)], [("workspace", ASCENDING), ("observed_at", ASCENDING)]],
    "trackers": [([("owner", ASCENDING), ("name", ASCENDING)], {"unique": True})],
    "sightings": [[("owner", ASCENDING), ("tracker", ASCENDING), ("observed_at", DESCENDING)],
                  ([("tracker", ASCENDING), ("footage", ASCENDING), ("t0", ASCENDING), ("t1", ASCENDING)],
                   {"unique": True})],
    "queries": [[("status", ASCENDING), ("asked_at", ASCENDING)], [("workspace", ASCENDING), ("asked_at", DESCENDING)]],
    "events": [[("query", ASCENDING), ("rank", ASCENDING)],
               [("workspace", ASCENDING), ("kind", ASCENDING), ("observed_at", DESCENDING)]],
    "agent_sessions": [[("workspace", ASCENDING), ("person", ASCENDING), ("created_at", DESCENDING)]],
    "agent_agents": [[("session", ASCENDING)]],
    "agent_messages": [[("session", ASCENDING), ("agent", ASCENDING), ("message_id", ASCENDING)],
                       ([("message.content.text", TEXT)], {"name": "recall"})],
    "agent_notes": [[("workspace", ASCENDING), ("person", ASCENDING), ("created_at", DESCENDING)],
                    ([("subject", TEXT), ("note", TEXT)], {"name": "recall", "weights": {"subject": 3}})],
}


NUMBER = ["double", "int", "long", "decimal"]
VALIDATORS = {
    # The query row is the contract with whoever asks (the partner's agent, another machine): the database
    # refuses a malformed one (error 121) instead of the worker failing on it later.
    "queries": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["workspace", "example", "direction", "scope", "top", "status", "asked_at"],
            "properties": {
                "workspace": {"bsonType": "string"},
                "example": {"bsonType": "object", "required": ["footage", "t0", "t1"],
                            "properties": {"footage": {"bsonType": "string"},
                                           "t0": {"bsonType": NUMBER, "minimum": 0},
                                           "t1": {"bsonType": NUMBER, "minimum": 0}}},
                "direction": {"enum": ["forward", "reverse"]},
                "scope": {"enum": ["all", "others", "after"]},
                "top": {"bsonType": ["int", "long"], "minimum": 1, "maximum": 100},
                "status": {"enum": ["queued", "running", "done", "failed"]},
                "asked_at": {"bsonType": "date"},
                "label": {"bsonType": ["string", "null"]},
                "asked_by": {"bsonType": ["string", "null"]},
            },
        },
        "$expr": {"$gt": ["$example.t1", "$example.t0"]},
    },
    "agent_notes": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["workspace", "person", "subject", "note", "created_at"],
            "properties": {
                "workspace": {"bsonType": "string"},
                "person": {"bsonType": "string"},
                "subject": {"bsonType": "string", "minLength": 1, "maxLength": 200},
                "note": {"bsonType": "string", "minLength": 1, "maxLength": 4000},
                "query": {"bsonType": ["objectId", "null"]},
                "session": {"bsonType": ["string", "null"]},
                "created_at": {"bsonType": "date"},
            },
        },
    },
}


def redact(uri: str) -> str:
    """The connection string with its password hidden, for printing."""
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:****@", uri)


def client(uri: str | None = None, timeout_ms: int = 10000) -> pymongo.MongoClient:
    uri = uri or config.get().mongodb_uri
    if not uri:
        raise RuntimeError("MONGODB_URI is not set: put the Atlas connection string in the project's .env "
                           "(see .env.example)")
    kw = dict(serverSelectionTimeoutMS=timeout_ms, appname="cyclopsdiary", tz_aware=True)
    if uri.startswith("mongodb+srv://"):
        try:
            import certifi                                # a known CA bundle, whatever OpenSSL this Python was built on
            kw["tlsCAFile"] = certifi.where()
        except ImportError:
            pass
    return pymongo.MongoClient(uri, **kw)


def database(uri: str | None = None, name: str | None = None):
    return client(uri)[name or config.get().db]


def ensure(db) -> dict:
    """Make what is missing: the collections and their indexes. Running it again changes nothing.
    A collection of the same name that is not a plain collection (a teammate's time-series diary, a view)
    is refused rather than written into: point MONGODB_DB at another database."""
    have = set(db.list_collection_names())
    for info in db.list_collections(filter={"name": {"$in": list(COLLECTIONS)}}):
        if info.get("type", "collection") != "collection":
            raise RuntimeError(f"{db.name}.{info['name']} already exists as a {info['type']}, not CyclopsDiary's: "
                               "set MONGODB_DB to another database")
    made = [c for c in COLLECTIONS if c not in have]
    for c in made:
        db.create_collection(c)
    for c, specs in INDEXES.items():
        for spec in specs:
            keys, kw = spec if isinstance(spec, tuple) else (spec, {})
            db[c].create_index(keys, **kw)
    for c, v in VALIDATORS.items():                    # moderate: rows already stored are not re-judged
        db.command("collMod", c, validator=v, validationLevel="moderate", validationAction="error")
    return dict(created=made, collections=list(COLLECTIONS))


def change_stream(coll, match: dict, wait_s: float):
    """A change stream on coll for the changes matching `match`, or None where the server has none (a
    standalone mongod): the caller then polls. Atlas is a replica set, so there a new row wakes the worker
    at once instead of on the next poll."""
    from pymongo.errors import OperationFailure
    try:
        return coll.watch([{"$match": match}], max_await_time_ms=max(1, int(wait_s * 1000)))
    except OperationFailure:
        return None


def ping(db) -> dict:
    ok = db.client.admin.command("ping").get("ok") == 1.0
    return dict(ok=ok, version=db.client.server_info().get("version"))


def counts(db) -> dict:
    return {c: db[c].estimated_document_count() for c in COLLECTIONS}
