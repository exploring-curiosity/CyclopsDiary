"""bin/cyclopsdiary -- the command line.

  check                                            is Atlas reachable; make the schema; what is stored
  person add ID --workspace WS [--name NAME]
  source add ID --person P --kind glasses|phone|webcam|file [--label L]
  ingest VIDEO... --source ID [--started-at ISO]   footage -> the world model -> steps in MongoDB
  footage [--workspace WS]                         footage ids and clocks (to pick an example from)
  object add NAME --person P [--also ALIAS...] [--footage F --start S --end S]
                                                   a private object: the model's words for its name (and a span
                                                   that shows it); tracked over every camera at once
  object list --person P  /  object page --person P [--out HTML]
  track NAME --person P --footage F --start S --end S     the same, from an example span
  where NAME --person P                            track it now; its trail across the cameras and where it was last seen
  confirm SIGHTING --person P  /  reject SIGHTING --person P      (a confirmed sighting becomes an example)
  ask (--footage F --start S --end S | --clip VIDEO --source ID) [--reverse] [--scope all|others|after]
      [--top N] [--label TEXT] [--person P] [--queue]
                                                   one example asked of every camera -> events in MongoDB;
                                                   --reverse asks for the example undone (put down -> taken away)
  events QUERY                                     a query's status and the moments it found
  serve [--inbox DIR] [--poll S] [--once]          ingest DIR/<source>/ clips as they land, answer queued queries
        [--live [PORT] [--tunnel]]                 phones stream their cameras from the browser, read as they arrive;
                                                   --tunnel prints a public https link for phones on any network
  mcp --workspace WS [--person P]                  the agent's tools over MCP (stdio): queries, timeline, memory
  say QUESTION --person P [--no-model] [--json]    the router: one of the agent's four tools, answered at once
  lastseen [--object O --person P --source S --start ISO --end ISO] [--forward] [--scope S] [--play]
                                                   video in (an object's sighting), tracked through time, video out
                                                   (where it went); defaults: the keys demo on the sample clips
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from bson import ObjectId

from . import atlas, config, query, workspace
# tower and tracker need numpy (the model's environment); each command imports them itself, so
# `mcp` also runs in the agent's .venv-agent


def _db():
    return atlas.database()


def _when(t: datetime) -> str:
    return t.astimezone().strftime("%Y-%m-%d %H:%M:%S") if t else "-"


def cmd_check(a) -> int:
    from . import tower
    s = config.get()
    if not s.mongodb_uri:
        print("MONGODB_URI is not set: copy .env.example to .env and put the Atlas connection string in it")
        return 1
    print(f"MongoDB     {atlas.redact(s.mongodb_uri)}  (database {s.db})")
    db = _db()
    try:
        p = atlas.ping(db)
    except Exception as e:                                 # noqa: BLE001 -- the reason is the message
        print(f"unreachable: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
        print("Atlas: is this machine's IP on the project's Network Access list, and is the user/password right?")
        return 1
    print(f"server      MongoDB {p['version']}, ping ok")
    made = atlas.ensure(db)["created"]
    print(f"schema      {', '.join(atlas.COLLECTIONS)}" + (f"  (made now: {', '.join(made)})" if made else ""))
    print("stored      " + ", ".join(f"{k} {v}" for k, v in atlas.counts(db).items()))
    print(f"model       {s.model}: {'in the local cache' if tower.cached(s.model) else 'NOT in the local cache'}; "
          f"side stack {'present' if config.STACK.is_dir() else 'MISSING'} ({config.STACK})")
    return 0


def cmd_person(a) -> int:
    p = workspace.add_person(_db(), a.id, a.workspace, a.name)
    print(f"{p['_id']} in workspace {p['workspace']}")
    return 0


def cmd_source(a) -> int:
    s = workspace.add_source(_db(), a.id, a.person, a.kind, a.label, a.clock_offset)
    off = s.get("clock_offset_s")
    print(f"{s['_id']}: {s['person']}'s {s['kind']} in workspace {s['workspace']}"
          + (f", clock {off:+g} s" if off else ""))
    return 0


def cmd_ingest(a) -> int:
    from . import tower
    tower.use_stack()                                      # before anything imports transformers
    from . import ingest
    db = _db()
    atlas.ensure(db)
    if a.started_at and len(a.video) > 1:
        raise ValueError("--started-at is one recording's clock; ingest one video with it")
    started = datetime.fromisoformat(a.started_at) if a.started_at else None
    if started is not None and started.tzinfo is None:
        started = started.astimezone()                     # a bare time is this machine's local time
    for path in a.video:
        r = ingest.ingest(db, path, a.source, started_at=started)
        if r.get("skipped"):
            print(f"{path}: already in the memory as footage {r['footage']} ({r['steps']} steps)")
            continue
        print(f"{path}: footage {r['footage']}, {r['steps']} steps from {_when(r['started_at'])}, "
              f"{r['x_real_time']}x real time, read {r['bytes_read'] / 1e6:.1f} of {r['bytes'] / 1e6:.1f} MB")
    return 0


def cmd_footage(a) -> int:
    q = {"workspace": a.workspace} if a.workspace else {}
    for f in _db().footage.find(q).sort("started_at", 1):
        print(f"{f['_id']}  {f['source']:<16} {_when(f['started_at'])}  {f.get('duration_s', 0):7.1f} s  "
              f"{f.get('steps', 0):5d} steps  {f['status']}  {f['path']}")
    return 0


def _tok():
    from . import lastseen, tower
    tower.use_stack()                                      # before anything imports transformers
    print(f"loading {config.get().model}'s tokenizer (the model's own words; not the model) ...", flush=True)
    return lastseen.tokenizer()


def _print_where(w: dict) -> None:
    utc = lambda d: d.astimezone(timezone.utc).strftime("%H:%M:%S") + " UTC"      # noqa: E731
    if w["last"] is None:
        print(f"{w['owner']}'s {w['object']}: not seen in any camera's footage")
        return
    for s in w["trail"]:
        print(f"  {utc(s['observed_at'])}-{utc(s['until'])}  {s['source']:<10} {s['footage']} {s['t0']:g}-{s['t1']:g} s"
              f"  [{'confirmed' if s.get('confirmed_at') else '+'.join(s['by'])}; {s['_id']}]"
              f"  near: {', '.join(s.get('place') or [])}")
    L = w["last"]
    print(f"last seen  {utc(L['until'])} on {L['source']} ({L.get('person')}), near: "
          f"{', '.join(L.get('end_place') or L.get('place') or [])}")


def cmd_object(a) -> int:
    from . import objects
    db = _db()
    if a.op == "list":
        for o in objects.objects(db, a.person):
            L = o["last"]
            print(f"{o['name']:<16} " + (f"last seen {_when(L['until'])} on {L['source']}, near: "
                                        f"{', '.join(L.get('end_place') or L.get('place') or [])}" if L else "not seen yet"))
        return 0
    if a.op == "page":
        print(f"page       {objects.page(db, a.person, a.out)}")
        return 0
    if (a.footage is None) != (a.start is None) or (a.start is None) != (a.end is None):
        raise ValueError("an example needs --footage, --start and --end together")
    w = objects.add(db, a.person, a.name, _tok(), aliases=a.also or (), footage=a.footage, t0=a.start, t1=a.end,
                    describe=a.describe)
    _print_where(w)
    return 0


def cmd_track(a) -> int:
    from . import objects
    _print_where(objects.add(_db(), a.person, a.name, _tok(), footage=a.footage, t0=a.start, t1=a.end))
    return 0


def cmd_where(a) -> int:
    from . import objects
    db, tok = _db(), _tok()
    objects.track(db, a.person, a.name, tok)
    _print_where(objects.where(db, a.person, a.name))
    return 0


def _print_query(db, q) -> None:
    ex = q["example"]
    print(f"query {q['_id']}: {q['direction']} of {ex['footage']} {ex['t0']:g}-{ex['t1']:g} s, scope {q['scope']}"
          + (f", {q['label']!r}" if q.get("label") else "") + f"  [{q['status']}]")
    if q["status"] == "failed":
        print(f"  {q.get('error')}")
    if q["status"] != "done":
        return
    print(f"searched {q['searched']['steps']} steps of {q['searched']['footage']} footage"
          + ("" if q.get("cut") is not None else "; no moment stands apart from the rest"))
    for e in query.events(db, q["_id"]):
        print(f"  {'*' if e['apart'] else ' '} {e['score']:+.4f}  {_when(e['observed_at'])}  {e['source']:<16} "
              f"{e['footage']} {e['t0']:g}-{e['t1']:g} s")


def cmd_ask(a) -> int:
    from . import tower
    db = _db()
    atlas.ensure(db)
    if a.clip:
        if not a.source or a.footage:
            raise ValueError("--clip needs --source (the camera that shot it) and no --footage")
        tower.use_stack()                                  # before anything imports transformers
        from . import ingest
        r = ingest.ingest(db, a.clip, a.source, role="example")
        f = db.footage.find_one({"_id": r["footage"]})
        atom = float(f["pin"]["n_frames"]) / float(f["pin"]["rate"])
        t0, t1 = 0.0, max(float(f["duration_s"]), f["steps"] * atom)   # the whole clip, its last partial step too
    else:
        if not a.footage or a.start is None or a.end is None:
            raise ValueError("give --footage F --start S --end S, or --clip VIDEO --source ID")
        f = db.footage.find_one({"_id": a.footage})
        if f is None:
            raise ValueError(f"no footage {a.footage!r}")
        t0, t1 = a.start, a.end
    kw = dict(direction="reverse" if a.reverse else "forward", scope=a.scope, top=a.top, label=a.label,
              asked_by=a.person)
    if a.queue:
        q = query.enqueue(db, f["workspace"], f["_id"], t0, t1, **kw)
        print(f"queued {q['_id']}: bin/cyclopsdiary serve answers it; bin/cyclopsdiary events {q['_id']} shows it")
        return 0
    _print_query(db, query.ask(db, f["workspace"], f["_id"], t0, t1, **kw))
    return 0


def cmd_events(a) -> int:
    db = _db()
    q = db.queries.find_one({"_id": ObjectId(a.query)})
    if q is None:
        print(f"no query {a.query}")
        return 1
    _print_query(db, q)
    return 0


def _interrupt(*_):
    raise KeyboardInterrupt


def cmd_serve(a) -> int:
    import signal
    from . import tower
    if a.inbox or a.live:
        tower.use_stack()                                  # before anything imports transformers
    from . import stream
    db = _db()
    atlas.ensure(db)
    signal.signal(signal.SIGTERM, _interrupt)              # SIGTERM stops it like Ctrl-C: the tunnel is closed too
    try:
        n = stream.serve(db, inbox=a.inbox, poll=a.poll, once=a.once, live=a.live, tunnel=a.tunnel)
    except KeyboardInterrupt:
        print("stopped")
        return 0
    print(", ".join(f"{k} {v}" for k, v in n.items()))
    return 0


def cmd_say(a) -> int:
    import json
    from . import router
    from .tools import plain
    r = router.ask(_db(), a.person, a.question, use_model=not a.no_model)
    if a.json:
        print(json.dumps(plain(r), indent=1, default=str))
    else:
        print(r["answer"] or "(none of the four questions: the full agent takes it)")
        print(f"  [{r['how']}{' -> ' + r['tool'] if r.get('tool') else ''}, {r['ms'].get('total', r['ms']['route'])} ms]")
    return 0 if r["answer"] else 1


def cmd_mcp(a) -> int:
    from . import mcp_server
    mcp_server.main(a.workspace, a.person)
    return 0


def cmd_stage(a) -> int:
    from . import stage
    print(f"loading {config.get().model}'s tokenizer (the model's words for the feed; not the model) ...", flush=True)
    stage.main(port=a.port, track_every=a.track_every)
    return 0


def cmd_lastseen(a) -> int:
    from . import tower
    import subprocess
    import time
    from . import lastseen
    obs = dict(lastseen.DEMO, **{k: v for k, v in dict(object=a.object, person=a.person, source=a.source,
                                                        start=a.start, end=a.end).items() if v})
    tower.use_stack()                                      # before anything imports transformers
    print(f"loading {config.get().model}'s tokenizer (to read the model's words; not the model) ...", flush=True)
    tok = lastseen.tokenizer()
    db = _db()
    t = time.monotonic()
    r = lastseen.run(db, obs, direction="forward" if a.forward else "reverse", scope=a.scope, tokenizer=tok)
    took = time.monotonic() - t
    ex, q = r["example"], r["query"]
    f = ex["footage"]
    utc = lambda d: d.strftime("%H:%M:%S") + " UTC"        # noqa: E731
    print(f"video in   {obs['object']}: {obs['person']} on {f['source']}, {utc(lastseen._t(obs['start']))} to "
          f"{utc(lastseen._t(obs['end']))}  ({f['path'].rsplit('/', 1)[-1]} {ex['t0']:g}-{ex['t1']:g} s)")
    print(f"through    {q['direction']}, scope {q['scope']}: {q['searched']['steps']} steps of "
          f"{q['searched']['footage']} recordings; {q['events']} moments, {q['apart']} stand apart "
          f"(query {q['_id']}, {took:.1f} s)")
    if "last" not in r:
        print(f"last seen  not seen since {utc(lastseen._t(obs['end']))}")
        return 0
    m, last = r["moved"], r["last"]
    g = last["footage"]
    print(f"moved      {g['source']} ({g.get('person')}) at {utc(m['observed_at'])}: {g['path'].rsplit('/', 1)[-1]} "
          f"{m['t0']:g}-{m['t1']:g} s, score {m['score']:+.3f}" + (" *" if m["apart"] else "  (nothing stood apart)"))
    print(f"last seen  {g['source']} at {utc(last['observed_at'])}, the end of that recording"
          + (f"; the model's words there: {', '.join(last['words'])}" if last["words"] else ""))
    page = lastseen.page(r, a.page)
    print(f"video out  {page}")
    if a.play:
        for path, t0, t1, title in ((f["path"], ex["t0"], ex["t1"], "video in"),
                                    (g["path"], last["t0"], last["t1"], "video out: last seen")):
            subprocess.run(["ffplay", "-hide_banner", "-loglevel", "error", "-autoexit", "-ss", f"{t0:g}",
                            "-t", f"{t1 - t0:g}", "-window_title", title, path], check=False)
    return 0


def cmd_mark(a) -> int:
    from . import objects
    fn = objects.confirm if a.cmd == "confirm" else objects.reject
    ok = fn(_db(), a.person, ObjectId(a.sighting))
    print(("done" if ok else f"no such {'open ' if a.cmd == 'confirm' else 'found '}sighting of {a.person}'s"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cyclopsdiary", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("person").add_subparsers(dest="op", required=True).add_parser("add")
    p.add_argument("id"); p.add_argument("--workspace", required=True); p.add_argument("--name")
    p.set_defaults(fn=cmd_person)
    p = sub.add_parser("source").add_subparsers(dest="op", required=True).add_parser("add")
    p.add_argument("id"); p.add_argument("--person", required=True)
    p.add_argument("--kind", required=True, choices=workspace.KINDS); p.add_argument("--label")
    p.add_argument("--clock-offset", type=float, help="seconds added to this camera's clock (from a sync shot)")
    p.set_defaults(fn=cmd_source)
    p = sub.add_parser("ingest")
    p.add_argument("video", nargs="+"); p.add_argument("--source", required=True); p.add_argument("--started-at")
    p.set_defaults(fn=cmd_ingest)
    p = sub.add_parser("footage"); p.add_argument("--workspace"); p.set_defaults(fn=cmd_footage)
    ob = sub.add_parser("object").add_subparsers(dest="op", required=True)
    p = ob.add_parser("add")
    p.add_argument("name"); p.add_argument("--person", required=True)
    p.add_argument("--also", nargs="*", help="other names for it (key, car keys)")
    p.add_argument("--describe", help="the words to look for when the name also names other things "
                                      "(\"keys library card\": a keyboard has keys too)")
    p.add_argument("--footage"); p.add_argument("--start", type=float); p.add_argument("--end", type=float)
    p.set_defaults(fn=cmd_object)
    p = ob.add_parser("list"); p.add_argument("--person", required=True); p.set_defaults(fn=cmd_object)
    p = ob.add_parser("page"); p.add_argument("--person", required=True)
    p.add_argument("--out", default=str(config.ROOT / ".local" / "demo" / "objects.html")); p.set_defaults(fn=cmd_object)
    p = sub.add_parser("track")
    p.add_argument("name"); p.add_argument("--person", required=True); p.add_argument("--footage", required=True)
    p.add_argument("--start", type=float, required=True); p.add_argument("--end", type=float, required=True)
    p.set_defaults(fn=cmd_track)
    p = sub.add_parser("where")
    p.add_argument("name"); p.add_argument("--person", required=True); p.add_argument("--top", type=int, default=10)
    p.set_defaults(fn=cmd_where)
    p = sub.add_parser("ask")
    p.add_argument("--footage"); p.add_argument("--start", type=float); p.add_argument("--end", type=float)
    p.add_argument("--clip"); p.add_argument("--source")
    p.add_argument("--reverse", action="store_true", help="the example undone: its steps asked in reverse order")
    p.add_argument("--scope", choices=query.SCOPES, default="all")
    p.add_argument("--top", type=int, default=10); p.add_argument("--label"); p.add_argument("--person")
    p.add_argument("--queue", action="store_true", help="only queue it; `serve` answers")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("events"); p.add_argument("query"); p.set_defaults(fn=cmd_events)
    p = sub.add_parser("serve")
    p.add_argument("--inbox"); p.add_argument("--poll", type=float, default=2.0); p.add_argument("--once", action="store_true")
    p.add_argument("--live", type=int, nargs="?", const=8780, metavar="PORT",
                   help="phones stream their cameras from the browser (camera service on PORT, default 8780)")
    p.add_argument("--tunnel", action="store_true", help="a public HTTPS link for the phones (Cloudflare quick tunnel)")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("say"); p.add_argument("question"); p.add_argument("--person", required=True)
    p.add_argument("--no-model", action="store_true", help="rules only: never call the model")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_say)
    p = sub.add_parser("mcp"); p.add_argument("--workspace", required=True); p.add_argument("--person")
    p.set_defaults(fn=cmd_mcp)
    p = sub.add_parser("stage", help="the demo stage: its API and the ui/ app, on this machine only")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--track-every", type=float, default=5.0, help="seconds between tracker passes while a phone is live")
    p.set_defaults(fn=cmd_stage)
    p = sub.add_parser("lastseen")
    for k in ("object", "person", "source", "start", "end"):
        p.add_argument(f"--{k}")
    p.add_argument("--forward", action="store_true", help="moments like the example instead of the example undone")
    p.add_argument("--scope", choices=query.SCOPES, default="after")
    p.add_argument("--page", default=str(config.ROOT / ".local" / "demo" / "lastseen.html"))
    p.add_argument("--play", action="store_true", help="play the example and the answer with ffplay")
    p.set_defaults(fn=cmd_lastseen)
    for c in ("confirm", "reject"):
        p = sub.add_parser(c); p.add_argument("sighting"); p.add_argument("--person", required=True)
        p.set_defaults(fn=cmd_mark)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, RuntimeError) as e:
        print(f"cyclopsdiary {a.cmd}: {e}", file=sys.stderr)
        return 2
