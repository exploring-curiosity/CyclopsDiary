"""The question router: the agent's four read-only tools (design doc section 7), answered at MongoDB speed.

  find_object(name)       which of the person's objects answer to that name, and where each was last seen
  object_belief(name)     where it is believed to be now, with the clip that shows it
  at_place(place)         what of the person's was last seen at a place
  between(start, end)     what happened between two times (an aggregation over sightings and moments)

A question goes to rules first: "where are my keys", "what's on the table", "what happened since 3pm" need
no model call. What the rules miss goes to one model call that only picks the tool and its arguments (the
fastest OpenRouter route measured, reasoning off); a question that is none of the four returns None, and
the full agent takes it. After three failed calls in a row the model is left alone for a minute and the
rules still answer.

The tracking already happened when the footage landed (`serve` re-tracks every object after each clip), so
answering is reading -- and in a long-running process (the MCP server) not even that: `watch(db)` keeps the
reads in memory and a MongoDB change stream on trackers, sightings, footage and events drops them as soon
as it reports a change (the stream's lag behind the write). A read that races a change is not kept.
Without change streams (a standalone mongod) nothing is kept and every answer reads. The answer is worded from what was
found -- seen, believed, or not seen -- and never guessed. The clip (the footage file and its seconds) and
the MongoDB queries behind the answer (the evidence view) come with it.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

from . import config

TOOLS = [
    {"type": "function", "function": {
        "name": "find_object", "description": "Which of the person's objects answer to a name (all of them when "
        "the name is empty), and where each was last seen.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "object_belief", "description": "Where one object is believed to be now (last seen where, when, "
        "by which camera), with the clip that shows it. For 'where is X', 'have you seen X', 'who has X'.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "at_place", "description": "Which of the person's objects were last seen at a place (table, "
        "chair, counter, drawer ...).",
        "parameters": {"type": "object", "properties": {"place": {"type": "string"}}, "required": ["place"]}}},
    {"type": "function", "function": {
        "name": "between", "description": "What happened between two times: sightings of the person's objects and "
        "moments the cameras found. Times as ISO 8601 with the time zone.",
        "parameters": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                       "required": ["start", "end"]}}},
]
FADE_S = float(os.environ.get("CYCLOPSDIARY_FADE_S") or 1800)    # the design's tunable: older -> "probably still"
MODEL_ROUTE = {"order": ["together", "deepinfra", "baseten"], "allow_fallbacks": True}   # ledger L-11
_breaker = dict(fails=0, until=0.0)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Kept:
    """Reads kept in memory while a change stream says nothing they depend on has changed."""

    def __init__(self):
        self.version, self.live, self.data, self.lock = 0, False, {}, threading.Lock()

    def get(self, key, compute):
        if not self.live:
            return compute()
        v = self.version
        with self.lock:
            if (v, key) in self.data:
                return self.data[(v, key)]
        got = compute()
        with self.lock:
            if v == self.version:
                self.data[(v, key)] = got
        return got

    def changed(self):
        with self.lock:
            self.version += 1
            self.data.clear()


_kept = _Kept()


def watch(db) -> bool:
    """Keep the router's reads in memory, dropped on any change to trackers, sightings, footage or events
    (a change stream on the database). -> False where the server has no change streams."""
    from . import atlas
    if _kept.live:
        return True
    cs = atlas.change_stream(db, {"ns.coll": {"$in": ["trackers", "sightings", "footage", "events", "people"]}}, 5.0)
    if cs is None:
        return False

    def run():
        try:
            while cs.alive:
                if cs.try_next() is not None:
                    _kept.changed()
        except Exception:                                      # noqa: BLE001 -- lost: stop keeping, read again
            pass
        _kept.live = False
        _kept.changed()

    _kept.live = True
    threading.Thread(target=run, name="router-watch", daemon=True).start()
    return True


def warm(db, person: str) -> None:
    """Read (and keep) what the first question will need, so it is not the slow one."""
    find_object(db, person)
    _workspace(db, person)


def _local():
    return datetime.now().astimezone().tzinfo


# ---- the four tools

def _bare(w: str) -> str:
    """For saying it back: 'my keys' -> 'keys', 'the chair' -> 'chair'."""
    return re.sub(r"^(my|the|our|a|an|your|his|her|their)\s+", "", w.strip().lower())


def _norm(w: str) -> str:
    w = re.sub(r"[^a-z0-9 ]", "", w.lower()).strip()
    w = re.sub(r"^(my|the|our|a|an|your|his|her|their)\s+", "", w)
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _trackers(db, person: str, name: str = "") -> list:
    rows = _kept.get(("trackers", person), lambda: list(db.trackers.find({"owner": person}, {"words": 0})))
    if not name.strip():
        return rows
    want = _norm(name)
    return [t for t in rows if want in {_norm(n) for n in [t["name"], *t.get("aliases", [])]}]


def names(db, person: str) -> dict:
    """The person's objects by every name and alias, normalized -> the object's name."""
    return {_norm(n): t["name"] for t in _trackers(db, person) for n in [t["name"], *t.get("aliases", [])] if _norm(n)}


def _workspace(db, person: str):
    return _kept.get(("workspace", person),
                     lambda: (db.people.find_one({"_id": person}, {"workspace": 1}) or {}).get("workspace"))


def _clip(db, s: dict) -> dict:
    f = _kept.get(("footage", s["footage"]), lambda: db.footage.find_one({"_id": s["footage"]}, {"path": 1, "source": 1}))
    return dict(footage=s["footage"], path=f and f.get("path"), camera=s.get("source"), t0=s["t0"], t1=s["t1"])


def _last(db, person: str, t: dict):
    from . import objects                                      # the trail's merge rules are objects.py's
    return _kept.get(("last", person, t["name"]), lambda: objects.last_known(db, person, t["name"]))


def _sight(db, s: dict) -> dict:
    return dict(camera=s.get("source"), person=s.get("person"), seen_from=s["observed_at"], seen_until=s["until"],
                near=s.get("end_place") or s.get("place") or [], by=s.get("by"),
                confirmed=s.get("confirmed_at") is not None, clip=_clip(db, s))


def _evidence_sightings(person: str, t: dict) -> dict:
    return dict(collection="sightings", find={"tracker": t["_id"], "owner": person, "rejected_at": None,
                                              "$or": [{"apart": True}, {"confirmed_at": {"$ne": None}}]},
                sort={"observed_at": 1})


def find_object(db, person: str, name: str = "") -> dict:
    ts = _trackers(db, person, name)
    ev = [dict(collection="trackers", find={"owner": person})]
    out = []
    for t in ts:
        s = _last(db, person, t)
        ev.append(_evidence_sightings(person, t))
        out.append(dict(name=t["name"], aliases=t.get("aliases", []), last=None if s is None else _sight(db, s)))
    return dict(tool="find_object", name=_bare(name), objects=out, evidence=ev)


def object_belief(db, person: str, name: str) -> dict:
    ts = _trackers(db, person, name)
    ev = [dict(collection="trackers", find={"owner": person}, then=f"name or alias is {name!r}")]
    if not ts:
        return dict(tool="object_belief", name=_bare(name), status="unknown", evidence=ev)
    t = ts[0]
    s = _last(db, person, t)
    ev.append(_evidence_sightings(person, t))
    if s is None:
        return dict(tool="object_belief", name=t["name"], status="not_seen", evidence=ev)
    ago = (_now() - s["until"]).total_seconds()
    return dict(tool="object_belief", name=t["name"], status="seen" if ago <= FADE_S else "believed",
                ago_s=round(ago), last=_sight(db, s), evidence=ev)


def at_place(db, person: str, place: str) -> dict:
    want = _norm(place)
    ev = [dict(collection="trackers", find={"owner": person})]
    out = []
    for t in _trackers(db, person):
        s = _last(db, person, t)
        ev.append(_evidence_sightings(person, t))
        if s is not None and want in {_norm(w) for w in (s.get("end_place") or s.get("place") or [])}:
            out.append(dict(name=t["name"], last=_sight(db, s)))
    return dict(tool="at_place", place=_bare(place), objects=out, evidence=ev)


def between(db, person: str, start: datetime, end: datetime) -> dict:
    """One aggregation, one round trip: the person's sightings, then ($unionWith) the workspace's moments."""
    moments = [
        {"$match": {"workspace": _workspace(db, person), "apart": True, "observed_at": {"$gte": start, "$lte": end}}},
        {"$group": {"_id": {"footage": "$footage", "t0": "$t0", "kind": "$kind"}, "e": {"$first": "$$ROOT"}}},
        {"$project": {"_id": 0, "at": "$e.observed_at", "what": {"$ifNull": ["$e.label", "$e.kind"]},
                      "camera": "$e.source", "person": "$e.person", "footage": "$e.footage", "t0": "$e.t0",
                      "t1": "$e.t1", "source": "moment"}}]
    pipeline = [
        {"$match": {"owner": person, "rejected_at": None, "observed_at": {"$gte": start, "$lte": end},
                    "$or": [{"apart": True}, {"confirmed_at": {"$ne": None}}]}},
        {"$lookup": {"from": "trackers", "localField": "tracker", "foreignField": "_id", "as": "object"}},
        {"$project": {"_id": 0, "at": "$observed_at", "until": 1, "what": {"$first": "$object.name"},
                      "camera": "$source", "person": 1, "near": {"$ifNull": ["$end_place", "$place"]},
                      "footage": 1, "t0": 1, "t1": 1, "source": "sighting"}},
        {"$unionWith": {"coll": "events", "pipeline": moments}},
        {"$sort": {"at": 1, "source": 1}}]
    return dict(tool="between", start=start, end=end, happened=list(db.sightings.aggregate(pipeline)),
                evidence=[dict(collection="sightings", aggregate=pipeline)])


# ---- rules: the questions people actually ask, no model call

_WHERE = re.compile(r"^(?:where(?:'s| is| are| was| were| did (?:i|we|you) (?:leave|put|see|last see))?|"
                    r"have you seen|did (?:you|anyone) see|who has|who took|find|locate|last (?:seen|known) "
                    r"(?:location of )?)\s+(?P<obj>.+?)(?:\s+(?:now|last|again|today|go|gone))*$")
_PLACE = re.compile(r"^(?:what(?:'s| is| was| are| were)?|is there anything|anything|which things are|"
                    r"what did i leave|what have i left)\s+(?:on|at|in|near|by|under)\s+(?P<place>.+)$")
_LIST = re.compile(r"^(?:what|which) (?:objects|things|items)(?: do i have| am i tracking| are you tracking|"
                   r" are tracked)?$|^list (?:my )?(?:objects|things|items)$")
_SPAN = re.compile(r"^what (?:happened|did (?:i|we) do|went on)(?P<rest>.*)$")


def _clock(s: str, day: datetime) -> datetime | None:
    """'3pm', '3:10', '15:10:05', '3:10 pm', 'noon', 'now', '10 minutes ago' -> an aware datetime."""
    s = s.strip()
    if s in ("now", "right now"):
        return _now()
    if s == "noon":
        return day.replace(hour=12, minute=0, second=0, microsecond=0)
    if s == "midnight":
        return day.replace(hour=0, minute=0, second=0, microsecond=0)
    m = re.match(r"^(?P<n>\d+|an?|one) (?P<u>sec|second|min|minute|hour|hr)s? ago$", s)
    if m:
        n = 1 if m["n"] in ("a", "an", "one") else int(m["n"])
        return _now() - timedelta(seconds=n * {"sec": 1, "second": 1, "min": 60, "minute": 60, "hour": 3600,
                                                "hr": 3600}[m["u"]])
    m = re.match(r"^(?P<h>\d{1,2})(?::(?P<m>\d{2}))?(?::(?P<s>\d{2}))?\s*(?P<ap>am|pm|a\.m\.|p\.m\.)?$", s)
    if not m:
        return None
    h, mi, se = int(m["h"]), int(m["m"] or 0), int(m["s"] or 0)
    if m["ap"] and m["ap"].startswith("p") and h < 12:
        h += 12
    if m["ap"] and m["ap"].startswith("a") and h == 12:
        h = 0
    if h > 23 or mi > 59 or se > 59:
        return None
    return day.replace(hour=h, minute=mi, second=se, microsecond=0)


def _span(rest: str):
    rest = rest.strip()
    now = _now()
    day = now.astimezone(_local())
    if rest in ("", "today"):
        return day.replace(hour=0, minute=0, second=0, microsecond=0), now
    m = re.match(r"^(?:in the |over the )?(?:last|past) (?P<n>\d+|an?|one|few) ?(?P<u>sec|second|min|minute|hour|hr)s?$",
                 rest)
    if m:
        n = {"a": 1, "an": 1, "one": 1, "few": 3}.get(m["n"]) or int(m["n"])
        return now - timedelta(seconds=n * {"sec": 1, "second": 1, "min": 60, "minute": 60, "hour": 3600,
                                            "hr": 3600}[m["u"]]), now
    m = re.match(r"^(?:since|after) (?P<a>.+)$", rest)
    if m and (a := _clock(m["a"], day)):
        return a, now
    m = re.match(r"^(?:between|from) (?P<a>.+?) (?:and|to|until|till) (?P<b>.+)$", rest)
    if m and (a := _clock(m["a"], day)) and (b := _clock(m["b"], day)):
        return (a, b) if a <= b else (b, a)
    for word, (h0, h1) in (("this morning", (5, 12)), ("this afternoon", (12, 17)), ("this evening", (17, 23))):
        if rest == word:
            return day.replace(hour=h0, minute=0, second=0, microsecond=0), day.replace(hour=h1, minute=0, second=0,
                                                                                         microsecond=0)
    return None


_ASKING = re.compile(r"^(?:where|which|who|when|what|have|has|had|did|do|does|is|are|was|were|can|could|any|"
                     r"anyone|know|tell me|show me)\b")
_CUE = re.compile(r"\b(?:where|which|who|when|seen|see|saw|find|found|lost|misplaced|dropped|left|leave|put|last|"
                  r"moved|took|taken)\b")
_TELLING = re.compile(r"^(?:remember|note|save|add|forget|remind|name|rename|track|stop|call)\b")


def _one(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _named(question: str, q: str, names: dict) -> list:
    """The known objects a question asks after in its own words ('any idea where I dropped my keys?',
    'which camera saw my keys last?'); none for a statement or a note to keep."""
    if _TELLING.match(q) or not _CUE.search(q) or not (question.strip().endswith("?") or _ASKING.match(q)):
        return []
    words = " " + " ".join(_one(w) for w in re.findall(r"[a-z0-9]+", q)) + " "
    hits = [k for k in names if f" {k} " in words]
    hits = [k for k in hits if not any(k != o and f" {k} " in f" {o} " for o in hits)]   # 'car key', not 'key'
    return sorted({names[k] for k in hits})


def rules(question: str, names: dict | None = None):
    """-> (tool, args) or None. names (names(db, person)): also a question naming one of the person's objects."""
    q = re.sub(r"\s+", " ", question.lower().strip().rstrip("?.! ")).replace("where're", "where are")
    if _LIST.match(q):
        return "find_object", dict(name="")
    if (m := _SPAN.match(q)) and (span := _span(m["rest"])):
        return "between", dict(start=span[0], end=span[1])
    if m := _PLACE.match(q):
        return "at_place", dict(place=m["place"])
    if m := _WHERE.match(q):
        if names is not None and _norm(m["obj"]) not in names:
            found = _named(question, q, names)
            if len(found) != 1:
                return None if found else ("object_belief", dict(name=m["obj"]))   # two objects: the model
            return "object_belief", dict(name=found[0])
        return "object_belief", dict(name=m["obj"])
    if names and len(found := _named(question, q, names)) == 1:
        return "object_belief", dict(name=found[0])
    return None


# ---- the one model call, for what the rules miss

def model_route(db, person: str, question: str, timeout_s: float = 6.0):
    """-> (tool, args), or None when the question is none of the four (or the model is unavailable)."""
    import httpx
    config.load_env()
    key, model = os.environ.get("OPENROUTER_API_KEY"), os.environ.get("AGENT_MODEL")
    if not key or not model or time.monotonic() < _breaker["until"]:
        return None
    names = [t["name"] for t in _trackers(db, person)]
    now = _now().astimezone(_local())
    body = dict(model=model, temperature=0, max_tokens=200, tools=TOOLS, tool_choice="auto",
                reasoning={"enabled": False}, provider=MODEL_ROUTE, messages=[
                    {"role": "system", "content": "Pick the one tool that answers the question, with its arguments, "
                     f"or no tool if none fits. Now: {now.isoformat(timespec='seconds')}. The person's objects: "
                     f"{', '.join(names) or 'none yet'}."},
                    {"role": "user", "content": question}])
    try:
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=timeout_s,
                       headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        calls = r.json()["choices"][0]["message"].get("tool_calls") or []
        _breaker["fails"] = 0
    except Exception:                                          # noqa: BLE001 -- the rules still answer
        _breaker["fails"] += 1
        if _breaker["fails"] >= 3:
            _breaker.update(fails=0, until=time.monotonic() + 60)
        return None
    if not calls:
        return None
    fn = calls[0]["function"]
    args = json.loads(fn.get("arguments") or "{}")
    if fn["name"] == "between":
        try:
            args = {k: datetime.fromisoformat(args[k].replace("Z", "+00:00")) for k in ("start", "end")}
        except (KeyError, ValueError):
            return None
        args = {k: v if v.tzinfo else v.replace(tzinfo=_local()) for k, v in args.items()}
    return fn["name"], args


# ---- wording: from the status, never a guess

def _t(d: datetime) -> str:
    return d.astimezone(_local()).strftime("%-I:%M:%S %p").lower()


def _ago(s: float) -> str:
    s = int(s)
    return (f"{s} s ago" if s < 90 else f"{s // 60} min ago" if s < 5400 else f"{s // 3600} h ago" if s < 172800
            else f"{s // 86400} days ago")


def _near(words) -> str:
    return f" near the {' / '.join(words[:3])}" if words else ""


def say(r: dict) -> str:
    tool = r["tool"]
    if tool == "object_belief":
        name = r["name"]
        if r["status"] == "unknown":
            return f"I don't know anything called {name}. Name it first (bin/cyclopsdiary object add)."
        if r["status"] == "not_seen":
            return f"I haven't seen your {name}."
        s = r["last"]
        who = f"{s['camera']}" + (f" ({s['person']}'s camera)" if s.get("person") else "")
        head = (f"Your {name}: last seen{_near(s['near'])} on {who} at {_t(s['seen_until'])}, {_ago(r['ago_s'])}"
                + (" (you confirmed it)" if s["confirmed"] else "") + ".")
        return head + (" Probably still there." if r["status"] == "believed" else "")
    if tool == "find_object":
        if not r["objects"]:
            return "I don't know anything by that name." if r["name"] else "You haven't named any objects yet."
        return " ".join(f"{o['name']}: " + (f"last seen{_near(o['last']['near'])} on {o['last']['camera']} at "
                                             f"{_t(o['last']['seen_until'])}." if o["last"] else "not seen yet.")
                        for o in r["objects"])
    if tool == "at_place":
        if not r["objects"]:
            return f"I haven't seen anything of yours at the {r['place']}."
        return f"Last seen at the {r['place']}: " + ", ".join(
            f"your {o['name']} ({_t(o['last']['seen_until'])} on {o['last']['camera']})" for o in r["objects"]) + "."
    if tool == "between":
        span = f"between {_t(r['start'])} and {_t(r['end'])}"
        if not r["happened"]:
            return f"Nothing I saw happened {span}."
        return f"{span[0].upper()}{span[1:]}: " + "; ".join(
            f"{_t(h['at'])} {h['what']}{_near(h.get('near') or [])} ({h['camera']})" for h in r["happened"][:8]) + "."
    return "I don't know."


def ask(db, person: str, question: str, use_model: bool = True) -> dict:
    """-> dict(answer, tool, args, how, result, ms), or answer None when it is none of the four questions."""
    t0 = time.perf_counter()
    how, got = "rules", rules(question, names(db, person))
    if got is None and use_model:
        how, got = "model", model_route(db, person, question)
    t1 = time.perf_counter()
    if got is None:
        return dict(answer=None, how="none", ms=dict(route=round((t1 - t0) * 1000)))
    tool, args = got
    fn = {"find_object": find_object, "object_belief": object_belief, "at_place": at_place, "between": between}[tool]
    r = fn(db, person, **args)
    t2 = time.perf_counter()
    return dict(answer=say(r), tool=tool, args=args, how=how, result=r,
                ms=dict(route=round((t1 - t0) * 1000), tool=round((t2 - t1) * 1000), total=round((t2 - t0) * 1000)))
