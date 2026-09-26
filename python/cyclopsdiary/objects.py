"""Private objects: named by their owner, identified and tracked by the world model's own words.

An object belongs to one person in a workspace. The store is `trackers`, and every read and write here
filters on the owner. An object has a name, the world model's own token ids for the name and its aliases
(`words`), and the spans of footage the owner confirmed show it (`examples`). Two people can each have
"keys", and neither sees the other's.

Identification uses the output-head formula (elidedb.headsearch) for both routes. There is no second model
and no cosine.

  by name     q is uniform over the name's words. A step's score is the mean over those words of
              (log p_step(word) - that word's mean over the workspace's steps). A word's log p is its best
              spelling's (" keys", "Keys", ...). A word outside a step's top-K takes that step's floor.
              This is text through the model's own head (ElideDB ledger N-65/N-66). An alias is an
              alternative name: a step takes its best phrase.
  by example  each confirmed span is asked of every camera (memory.search: the formula aligned by DTW).
              A match counts only where the name also shows the object, so the example marks which
              sightings look like the owner's own one and never adds a moment the name does not see.

A step shows the object when its score stands apart from the rest. The cut is elidedb.corpus.split, the
scores' own cut, with no constant. Consecutive steps of one recording make one sighting. A sighting keeps
the model's other top words there, which name the place ("marble, table").

The trail is the object's sightings in world time across every camera. The last known location is the
latest sighting the owner has not rejected. `serve` re-tracks every object after each clip it reads, so
the trail grows as footage lands.
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import memory, tracker

STOP = frozenset("""the a an and or of to in on at from into onto with for by is are was were be been being it its
this that these those there here as up down out over under near next left right top bottom front back side view
camera image video frame scene shot while then also very can could has have had not no his her their our your my
one two some any all each which what who when where how she he they them we you""".split())
# A place is named by things, not by how they look: colours never name where something is.
COLOURS = frozenset("""black white blue red green yellow gray grey brown orange pink purple beige cream tan navy teal
silver gold golden dark light bright pale colorful colourful colored coloured""".split())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---- words: the object's name in the model's own vocabulary

def name_words(tok, name: str, aliases=()) -> list:
    """[phrase][word] -> the model's own token ids for that word (the first token of each spelling)."""
    phrases = []
    for phrase in (name, *aliases):
        words = []
        said = phrase.lower().split()
        for w in [w for w in said if w not in STOP] or said:     # "my keys" -> keys; "can" alone stays a can
            ids = set()
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                got = tok.encode(v, add_special_tokens=False)
                if got:
                    ids.add(int(got[0]))
            if ids:
                words.append(sorted(ids))
        if words:
            phrases.append(words)
    if not phrases:
        raise ValueError(f"no words to look for in {name!r}")
    return phrases


def _word_lp(IDS, LP, FLOOR, tokens) -> np.ndarray:
    """log p of one word at every step: its best spelling's, or the step's floor when none is in its top-K."""
    best = np.full(len(IDS), -np.inf)
    for t in tokens:
        best = np.maximum(best, np.where(IDS == t, LP, -np.inf).max(1))
    return np.where(np.isfinite(best), best, FLOOR)


def name_scores(mem, phrases) -> np.ndarray:
    """The by-name score of every step of the set (the formula, q uniform over a phrase's words)."""
    IDS, LP = np.asarray(mem.ids), np.asarray(mem.lp)
    FLOOR = LP.min(1)
    per = []
    for words in phrases:
        rel = []
        for toks in words:
            w = _word_lp(IDS, LP, FLOOR, toks)
            rel.append(w - w.mean())
        per.append(np.mean(rel, axis=0))
    return np.max(per, axis=0)


def place(mem, rows, tok, skip=(), n: int = 5) -> list:
    """The model's own top words over some steps, the object's words left out: what is there with it."""
    mass: dict = {}
    for j in rows:
        for w, lp in zip(np.asarray(mem.ids[j])[:64], np.asarray(mem.lp[j])[:64]):
            if int(w) not in skip:
                mass[int(w)] = mass.get(int(w), 0.0) + float(np.exp(lp))
    own = {tok.decode([int(w)]).strip().lower() for w in skip}
    own |= {w + "s" for w in own} | {w[:-1] for w in own if w.endswith("s")}   # "key" is the keys' own word too
    out, seen = [], set(own)
    for w in sorted(mass, key=mass.get, reverse=True):
        s = tok.decode([w]).strip()
        if s.isalpha() and len(s) > 2 and s.lower() not in STOP | COLOURS and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s.lower())
            if len(out) == n:
                break
    return out


# ---- the store

def get(db, owner: str, name: str) -> dict | None:
    return tracker.get(db, owner, name)


def add(db, owner: str, name: str, tok, aliases=(), footage: str | None = None, t0: float | None = None,
        t1: float | None = None, describe: str | None = None) -> dict:
    """Name an object (and, optionally, a span of footage that shows it), then track it. -> where().

    describe: the words to look for when the name alone also names other things ("keys library card": a
    keyboard has keys too). The object keeps its name for questions; kept until described again."""
    p = db.people.find_one({"_id": owner})
    if p is None:
        raise ValueError(f"no person {owner!r}")
    now = _now()
    if describe is None:
        describe = (get(db, owner, name) or {}).get("describe")
    upd = {"$setOnInsert": dict(workspace=p["workspace"], owner=owner, name=name, created_at=now),
           "$set": dict(words=name_words(tok, describe or name, aliases), aliases=list(aliases), describe=describe,
                        updated_at=now)}
    ex = None
    if footage is not None:
        f = db.footage.find_one({"_id": footage, "workspace": p["workspace"]})
        if f is None:
            raise ValueError(f"no footage {footage!r} in {owner}'s workspace")
        if t0 is None or t1 is None or not float(t1) > float(t0):
            raise ValueError("an example needs a start and an end after it")
        ex = dict(footage=footage, t0=float(t0), t1=float(t1))
        upd["$addToSet"] = {"examples": ex}
    db.trackers.update_one({"owner": owner, "name": name}, upd, upsert=True)
    tr = get(db, owner, name)
    if ex is not None:
        if "example" not in tr:                              # the first example, for tracker.locate
            db.trackers.update_one({"_id": tr["_id"]}, {"$set": {"example": ex}})
        tracker._record(db, tr, f, ex["t0"], ex["t1"], now, kind="example", apart=True, confirmed_at=now)
    track(db, owner, name, tok)
    return where(db, owner, name)


def objects(db, owner: str) -> list:
    """The owner's objects, each with where it was last seen."""
    return [dict(name=t["name"], aliases=t.get("aliases", []), examples=len(t.get("examples", [])),
                 last=last_known(db, owner, t["name"])) for t in tracker.trackers(db, owner)]


# ---- tracking

def _segments(mem, present: np.ndarray, answerable: set) -> list:
    """Runs of consecutive present steps inside one recording -> [(footage, [step rows])]."""
    out, cur, prev = [], None, None
    for j in np.argsort([f"{mem.rec[k]}:{mem.t0[k]:012.3f}" for k in range(mem.n)], kind="stable"):
        j = int(j)
        if not present[j] or mem.rec[j] not in answerable:
            cur = None
            continue
        if cur is not None and mem.rec[j] == cur[0] and abs(mem.t0[j] - mem.t0[prev] - mem.atom_s) < 1e-6:
            cur[1].append(j)
        else:
            cur = (mem.rec[j], [j])
            out.append(cur)
        prev = j
    return out


def _put(db, tr: dict, f: dict, t0: float, t1: float, now: datetime, by: str, score: float, where: list,
         end: list) -> None:
    key = dict(tracker=tr["_id"], footage=f["_id"], t0=float(t0), t1=float(t1))
    db.sightings.update_one(key, {
        "$setOnInsert": dict(owner=tr["owner"], workspace=tr["workspace"], source=f["source"],
                             person=f.get("person"), kind="found",
                             observed_at=f["started_at"] + timedelta(seconds=float(t0)),
                             recorded_at=now, confirmed_at=None, rejected_at=None),
        "$set": dict(score=float(score), apart=True, found_at=now, by=by, place=where, end_place=end,
                     until=f["started_at"] + timedelta(seconds=float(t1)))}, upsert=True)


def track(db, owner: str, name: str, tok, mem=None) -> dict:
    """Look for the object in every camera's footage now: by its name, and by each of its examples.
    Sightings found before and not found now stop standing apart (a confirmed one stays believed)."""
    from elidedb import corpus
    tr = get(db, owner, name)
    if tr is None:
        raise ValueError(f"{owner} has no object called {name!r}")
    mem = mem if mem is not None else memory.load(db, tr["workspace"])
    now = _now()
    n = dict(by_name=0, by_example=0)
    if mem.n:
        answerable = {fid for fid, f in mem.footage.items() if f.get("role") != "example"}
        own = {t for words in tr.get("words", []) for toks in words for t in toks}
        present = np.zeros(mem.n, bool)
        if tr.get("words"):
            s = name_scores(mem, tr["words"])
            cut = corpus.split(s)
            if cut is not None:
                present = s > cut
                for fid, rows in _segments(mem, present, answerable):
                    a, b = float(mem.t0[rows[0]]), float(mem.t0[rows[-1]]) + mem.atom_s
                    _put(db, tr, mem.footage[fid], a, b, now, "name", float(s[rows].max()),
                         place(mem, rows, tok, own), place(mem, rows[-1:], tok, own))
                    n["by_name"] += 1
        for ex in tr.get("examples", []):
            if ex["footage"] not in mem.footage:
                continue
            found = mem.search(ex["footage"], ex["t0"], ex["t1"], top=10, recs=answerable)
            cut = corpus.split([c["score"] for c in found]) if found else None
            for c in found:
                if cut is None or c["score"] <= cut:
                    continue
                rows = [j for j in range(mem.n) if mem.rec[j] == c["footage"] and c["t0"] <= mem.t0[j] < c["t1"]]
                if tr.get("words") and not present[rows].any():
                    continue                                 # like the example, but the object's name is not there
                _put(db, tr, mem.footage[c["footage"]], c["t0"], c["t1"], now, "example", c["score"],
                     place(mem, rows, tok, own) if rows else [], place(mem, rows[-1:], tok, own) if rows else [])
                n["by_example"] += 1
    db.sightings.update_many({"tracker": tr["_id"], "owner": owner, "kind": "found", "found_at": {"$ne": now}},
                             {"$set": {"apart": False}})
    return n


def track_workspace(db, workspace: str, tok) -> dict:
    """Every object in the workspace, every owner's, tracked over the footage as it is now (after a clip)."""
    mem = memory.load(db, workspace)
    out = {}
    for tr in db.trackers.find({"workspace": workspace}, {"owner": 1, "name": 1}):
        out[f"{tr['owner']}/{tr['name']}"] = track(db, tr["owner"], tr["name"], tok, mem=mem)
    return out


# ---- the trail and the belief

def trail(db, owner: str, name: str) -> list:
    """The object's sightings in world time, every camera: those that stand apart or were confirmed, not
    rejected. Overlapping sightings of one recording (by name and by example) count once."""
    tr = get(db, owner, name)
    if tr is None:
        raise ValueError(f"{owner} has no object called {name!r}")
    rows = list(db.sightings.find({"tracker": tr["_id"], "owner": owner, "rejected_at": None,
                                   "$or": [{"apart": True}, {"confirmed_at": {"$ne": None}}]}).sort("observed_at", 1))
    no = list(db.sightings.find({"tracker": tr["_id"], "owner": owner, "rejected_at": {"$ne": None}},
                                {"footage": 1, "t0": 1, "t1": 1}))
    out = []
    for s in rows:
        if not s.get("confirmed_at") and any(r["footage"] == s["footage"] and s["t0"] < r["t1"] and r["t0"] < s["t1"]
                                             for r in no):
            continue                                         # the owner said that span is not it
        s.setdefault("until", s["observed_at"] + timedelta(seconds=float(s["t1"]) - float(s["t0"])))
        s["by"] = {s.get("by") or s["kind"]}
        prev = next((o for o in out if o["footage"] == s["footage"] and s["t0"] < o["t1"] and o["t0"] < s["t1"]), None)
        if prev is None:
            out.append(s)
            continue
        prev["by"] |= s["by"]
        prev["confirmed_at"] = prev.get("confirmed_at") or s.get("confirmed_at")
        if (s.get("score") or 0) > (prev.get("score") or 0) and s.get("place"):
            prev["place"] = s["place"]
        if s["until"] > prev["until"] and s.get("end_place"):
            prev["end_place"] = s["end_place"]
        prev["t0"], prev["t1"] = min(prev["t0"], s["t0"]), max(prev["t1"], s["t1"])
        prev["observed_at"], prev["until"] = min(prev["observed_at"], s["observed_at"]), max(prev["until"], s["until"])
    for s in out:
        s["by"] = sorted(s["by"])
    return sorted(out, key=lambda s: s["observed_at"])


def last_known(db, owner: str, name: str) -> dict | None:
    t = trail(db, owner, name)
    return max(t, key=lambda s: s["until"]) if t else None


def where(db, owner: str, name: str) -> dict:
    t = trail(db, owner, name)
    return dict(object=name, owner=owner, trail=t, last=max(t, key=lambda s: s["until"]) if t else None)


def confirm(db, owner: str, sighting) -> bool:
    """Yes, that is it: the sighting is believed and its span becomes one of the object's examples."""
    s = db.sightings.find_one({"_id": sighting, "owner": owner, "rejected_at": None})
    if s is None or not tracker.confirm(db, owner, sighting):
        return False
    db.trackers.update_one({"_id": s["tracker"], "owner": owner},
                           {"$addToSet": {"examples": dict(footage=s["footage"], t0=float(s["t0"]), t1=float(s["t1"]))}})
    return True


def reject(db, owner: str, sighting) -> bool:
    return tracker.reject(db, owner, sighting)


# ---- the page

def page(db, owner: str, path) -> Path:
    """The owner's objects: each one's trail across the cameras and where it was last seen, every span played
    from the original file (no copies)."""
    from .lastseen import _src
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    utc = lambda d: d.astimezone(timezone.utc).strftime("%H:%M:%S") + " UTC"             # noqa: E731
    e = html.escape
    feet = {}

    def clip(s, cls=""):
        f = feet.setdefault(s["footage"], db.footage.find_one({"_id": s["footage"]}))
        where_ = ", ".join(s.get("place") or []) or "-"
        tag = "confirmed" if s.get("confirmed_at") else "+".join(s["by"])
        return (f'<figure class="{cls}"><video src="{_src(f, path.parent)}#t={s["t0"]:g},{s["t1"]:g}" '
                f'data-t0="{s["t0"]:g}" data-t1="{s["t1"]:g}" controls muted playsinline autoplay></video>'
                f'<figcaption><b>{e(where_)}</b><br>{utc(s["observed_at"])}–{utc(s["until"])} · {e(f["source"])} '
                f'({e(str(f.get("person")))}) · {e(tag)}</figcaption></figure>')

    cards = []
    for o in objects(db, owner):
        w = where(db, owner, o["name"])
        if w["last"] is None:
            cards.append(f'<section><h2>{e(o["name"])}</h2><p>not seen yet</p></section>')
            continue
        steps = []
        for s in w["trail"]:
            head = (s.get("place") or ["?"])[0]
            if not steps or steps[-1] != head:
                steps.append(head)
        last = w["last"]
        strip = '<span class="arrow">→</span>'.join(clip(s) for s in w["trail"])
        cards.append(f'<section><h2>{e(o["name"])}</h2>'
                     f'<p class="verdict">last seen {utc(last["until"])} on {e(last["source"])}, near: '
                     f'{e(", ".join(last.get("end_place") or last.get("place") or []))}</p>'
                     f'<p>trail: {" → ".join(e(x) for x in steps)}</p>'
                     f'<div class="last">{clip(last, "big")}</div><div class="strip">{strip}</div></section>')
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(owner)}'s objects</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --dim:#57534e; --line:#e7e5e4; --accent:#b45309; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --dim:#a8a29e; --line:#44403c; --accent:#f59e0b; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:24px 0 4px; }}
p {{ margin:0 0 10px; color:var(--dim); }} .verdict {{ font-size:18px; color:var(--accent); }}
section {{ border-top:1px solid var(--line); padding-top:8px; }}
.last figure {{ max-width:560px; }}
.strip {{ display:flex; gap:10px; align-items:center; overflow-x:auto; padding:8px 0; }}
.strip figure {{ flex:0 0 220px; }} .arrow {{ color:var(--accent); font-size:22px; }}
figure {{ margin:0; }} video {{ width:100%; max-height:52vh; background:#000; border-radius:8px; }}
figcaption {{ font-size:12px; color:var(--dim); margin-top:4px; }}
</style></head><body><main>
<h1>{e(owner)}'s objects</h1>
<p>Private to {e(owner)}. Every camera in the workspace was read by the world model; each object is found by the
model's own words for it and by the moments {e(owner)} confirmed. Near: the model's other top words there.</p>
{"".join(cards) or "<p>no objects yet</p>"}
</main><script>
for (const v of document.querySelectorAll("video[data-t0]")) {{
  const t0 = +v.dataset.t0, t1 = +v.dataset.t1;
  v.addEventListener("loadedmetadata", () => {{ v.currentTime = t0; v.play().catch(() => {{}}); }});
  v.addEventListener("timeupdate", () => {{ if (v.currentTime < t0 - 0.25 || v.currentTime >= t1 - 0.05) v.currentTime = t0; }});
  if (v.readyState >= 1) {{ v.currentTime = t0; v.play().catch(() => {{}}); }}
}}
</script></body></html>
""")
    return path
