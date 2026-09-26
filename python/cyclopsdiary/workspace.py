"""People and their cameras.

A person belongs to one workspace. A source is one person's camera; its
footage is shared with everyone in the workspace, while what that person
tracks stays theirs (tracker.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

KINDS = ("glasses", "phone", "webcam", "file")


def now() -> datetime:
    return datetime.now(timezone.utc)


def add_person(db, person: str, workspace: str, name: str | None = None) -> dict:
    db.people.update_one({"_id": person},
                         {"$setOnInsert": dict(workspace=workspace, name=name or person, created_at=now())},
                         upsert=True)
    got = db.people.find_one({"_id": person})
    if got["workspace"] != workspace:
        raise ValueError(f"{person!r} is already in workspace {got['workspace']!r}")
    return got


def add_source(db, source: str, person: str, kind: str, label: str | None = None,
               clock_offset_s: float | None = None) -> dict:
    """clock_offset_s: seconds added to this camera's own clock at ingest (from a sync shot); given again,
    it replaces the old one."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r}: one of {', '.join(KINDS)}")
    p = db.people.find_one({"_id": person})
    if p is None:
        raise ValueError(f"no person {person!r}: add them first")
    db.sources.update_one({"_id": source},
                          {"$setOnInsert": dict(workspace=p["workspace"], person=person, kind=kind,
                                                label=label or source, created_at=now())},
                          upsert=True)
    got = db.sources.find_one({"_id": source})
    if got["person"] != person:
        raise ValueError(f"source {source!r} is {got['person']!r}'s")
    if clock_offset_s is not None:
        db.sources.update_one({"_id": source}, {"$set": {"clock_offset_s": float(clock_offset_s)}})
        got = db.sources.find_one({"_id": source})
    return got
