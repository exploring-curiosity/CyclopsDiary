"""The demo stage's API: what the cameras stream, what the memory holds, how a question is answered. It serves
the ui/ app and runs as its own process (`bin/cyclopsdiary stage`), fed by MongoDB rather than by the worker:

  GET  /api/state?person=P            the workspace's cameras and recordings (live ones too), P's objects and
                                      their trails
  GET  /api/steps?footage=F&person=P  a recording's steps: the model's top words, where P's objects rank in them
  GET  /api/object?person=P&name=N    how the tracker finds N: every step's score per camera, the scores' own
                                      cut, the trail (searchview.py)
  POST /api/ask {person, question}    router.ask: the answer, its clip, the MongoDB queries behind it
  GET  /media/{footage}               the original file, with byte ranges (the players seek to spans)
  WS   /feed?person=P                 MongoDB change streams on steps, footage, sightings, queries and events as
                                      they happen. A step comes with the model's own top words and where each of
                                      P's objects ranks among them; sightings are P's only
  WS   /video/{footage}               a recording from its first byte, then its bytes as the file grows, until
                                      its row says it ended: the phone's own encoding, played as it arrives
                                      (Media Source Extensions); nothing is re-encoded
  GET  /                              the built ui/ app (ui/dist), when there is one

Localhost only: the Cloudflare tunnel carries the camera service's port, never this one. The one thing it
writes is the tracker's work: while a phone is live, objects.track_workspace runs every few seconds (and once
more when the recording ends), so the objects' sightings follow the stream as they follow clips in `serve`.
"""
from __future__ import annotations

import asyncio
import contextlib
import mimetypes
import threading
import time
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from . import config, objects, searchview, tracker
from .tools import plain

UI_DIST = config.ROOT / "ui" / "dist"
WATCHED = ("steps", "footage", "sightings", "queries", "events", "trackers")


class Hub:
    """Fan-out from threads (the change stream, the tracker) to the stage's sockets on the server's loop."""

    def __init__(self):
        self.loop, self.subs, self.lock = None, {}, threading.Lock()
        self.change_streams = False

    def bind(self, loop) -> None:
        self.loop = loop

    def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self.lock:
            self.subs.setdefault(topic, set()).add(q)
        return q

    def unsubscribe(self, topic: str, q) -> None:
        with self.lock:
            self.subs.get(topic, set()).discard(q)

    def publish(self, topic: str, msg) -> None:
        if self.loop is None:
            return
        with self.lock:
            qs = list(self.subs.get(topic, ()))
        for q in qs:
            self.loop.call_soon_threadsafe(q.put_nowait, msg)


class TrackerWords:
    """Every tracked object's word ids ("owner/name" -> ids), to say where each ranks among a step's words."""

    def __init__(self, db):
        self.db, self.stale, self.words = db, True, {}

    def get(self) -> dict:
        if self.stale:
            self.words = {f"{t['owner']}/{t['name']}": {int(i) for ph in t.get("words") or [] for w in ph for i in w}
                          for t in self.db.trackers.find({}, {"owner": 1, "name": 1, "words": 1})}
            self.stale = False
        return self.words


def top_words(tok, ids, n: int = 6) -> list:
    """The model's own best words at a step, as text: the first n that are words (not stop words or marks)."""
    out = []
    for i in ids:
        w = tok.decode([int(i)]).strip().lower()
        if len(w) >= 2 and any(ch.isalpha() for ch in w) and w not in objects.STOP and w not in out:
            out.append(w)
            if len(out) == n:
                break
    return out


def ranks(ids, words: dict) -> dict:
    """Where each object ("owner/name" -> its word ids) first appears among a step's words, best first (0 = the
    model's top word); objects not among them are left out."""
    out = {}
    for key, want in words.items():
        hit = np.flatnonzero(np.isin(ids, list(want)))
        if len(hit):
            out[key] = int(hit[0])
    return out


def shape(ch: dict, tok, tw: TrackerWords) -> dict | None:
    """A change from MongoDB as the stage's event, or None."""
    coll, d = ch["ns"]["coll"], ch.get("fullDocument") or {}
    if coll == "trackers":
        tw.stale = True
        return None
    if coll == "steps":
        if ch["operationType"] != "insert":
            return None
        ids = np.frombuffer(d["ids"], np.int32)
        return dict(kind="step", footage=d["footage"], source=d["source"], person=d.get("person"), i=d.get("i"),
                    t0=d["t0"], t1=d["t1"], observed_at=d.get("observed_at"), words=top_words(tok, ids[:64]),
                    ranks=ranks(ids, tw.get()))
    keep = {"footage": ("_id", "source", "person", "status", "steps", "duration_s", "started_at", "mime"),
            "sightings": ("_id", "tracker", "owner", "source", "footage", "t0", "t1", "observed_at", "until", "by",
                          "place", "end_place", "apart", "confirmed_at", "rejected_at"),
            "queries": ("_id", "status", "label", "direction", "scope", "asked_by", "asked_at", "answered_at",
                        "events", "apart"),
            "events": ("_id", "query", "rank", "kind", "label", "footage", "source", "t0", "t1", "observed_at",
                       "score", "apart")}.get(coll)
    if keep is None or not d:
        return None
    ev = {k: d.get(k) for k in keep}
    if coll == "events":
        ev["direction"] = ev.pop("kind")
    return dict(ev, kind={"footage": "footage", "sightings": "sighting", "queries": "query",
                          "events": "event"}[coll])


def visible(ev: dict, person: str | None) -> dict | None:
    """What a person's stage may see: their own sightings, and only their own objects' ranks in a step."""
    if ev["kind"] == "sighting":
        return ev if ev.get("owner") == person else None
    if ev["kind"] == "step":
        return dict(ev, ranks={k: v for k, v in ev["ranks"].items() if k.split("/", 1)[0] == person})
    return ev


def watch(db, hub: Hub, tok, stop: threading.Event) -> None:
    """MongoDB change streams -> the feed. A standalone mongod has none: the stage then shows state, not a feed."""
    from pymongo.errors import OperationFailure, PyMongoError
    tw = TrackerWords(db)
    pipeline = [{"$match": {"ns.coll": {"$in": list(WATCHED)}, "operationType": {"$in": ["insert", "update", "replace"]}}},
                {"$project": {"fullDocument.state": 0, "fullDocument.lp": 0}}]
    while not stop.is_set():
        try:
            with db.watch(pipeline, full_document="updateLookup", max_await_time_ms=1000) as cs:
                hub.change_streams = True
                while not stop.is_set():
                    ch = cs.try_next()
                    if ch is not None and (ev := shape(ch, tok, tw)) is not None:
                        hub.publish("feed", plain(ev))
        except OperationFailure as e:
            print(f"stage: no change streams on this server ({e.code}); the feed stays quiet", flush=True)
            hub.change_streams = False
            return
        except PyMongoError as e:
            print(f"stage: change stream lost ({type(e).__name__}); reopening", flush=True)
            time.sleep(1.0)


def track_live(db, tok, stop: threading.Event, every: float = 5.0) -> None:
    """While a phone is live, run the tracker over its workspace every `every` seconds, and once more when the
    recording ends, so the objects' sightings follow the stream."""
    was: dict = {}
    while not stop.wait(every):
        try:
            now = {f["_id"]: f["workspace"] for f in db.footage.find({"status": "live"}, {"workspace": 1})}
            for ws in set(now.values()) | {w for fid, w in was.items() if fid not in now}:
                if db.trackers.find_one({"workspace": ws}, {"_id": 1}) is not None:
                    objects.track_workspace(db, ws, tok)
            was = now
        except Exception as e:                              # noqa: BLE001 -- the stage keeps serving
            print(f"stage: tracking failed: {type(e).__name__}: {e}", flush=True)


def _state(db, person: str | None) -> dict:
    p = db.people.find_one({"_id": person}) if person else None
    ws = p["workspace"] if p else (db.people.find_one({}, {"workspace": 1}) or {}).get("workspace")
    trs = tracker.trackers(db, person) if p else []
    trails = {t["name"]: objects.trail(db, person, t["name"]) for t in trs}      # each trail read once
    objs = [dict(name=t["name"], aliases=t.get("aliases", []), examples=len(t.get("examples", [])),
                 last=max(trails[t["name"]], key=lambda s: s["until"]) if trails[t["name"]] else None) for t in trs]
    return plain(dict(
        workspace=ws, person=person if p else None,
        people=list(db.people.find({"workspace": ws}, {"name": 1}).sort("_id", 1)),
        cameras=list(db.sources.find({"workspace": ws}, {"person": 1, "kind": 1, "label": 1}).sort("_id", 1)),
        footage=list(db.footage.find({"workspace": ws, "status": {"$in": ["ready", "live"]}, "role": {"$ne": "example"}},
                                     {"source": 1, "person": 1, "started_at": 1, "duration_s": 1, "steps": 1,
                                      "status": 1, "mime": 1}).sort("started_at", 1)),
        objects=objs, trails=trails))


def _steps(db, tok, footage: str, person: str | None) -> list:
    """A recording's steps as the stage draws them: the model's top words, and where the person's own objects
    rank among them (no one else's)."""
    words = {f"{t['owner']}/{t['name']}": {int(i) for ph in t.get("words") or [] for w in ph for i in w}
             for t in db.trackers.find({"owner": person}, {"owner": 1, "name": 1, "words": 1})} if person else {}
    out = []
    for s in db.steps.find({"footage": footage}, {"i": 1, "t0": 1, "t1": 1, "observed_at": 1, "ids": 1}).sort("t0", 1):
        ids = np.frombuffer(s["ids"], np.int32)
        out.append(dict(i=s.get("i"), t0=s["t0"], t1=s["t1"], observed_at=s.get("observed_at"),
                        words=top_words(tok, ids[:64]), ranks=ranks(ids, words)))
    return plain(out)


def _mime(row: dict) -> str:
    m = row.get("mime") or mimetypes.guess_type(row["path"])[0] or "video/mp4"
    return "video/mp4" if m.split(";")[0] == "video/quicktime" else m      # a .mov is ISO media: players take it as mp4


def codec(path, mime: str) -> str:
    """The type a Media Source needs (codec included), read from the recording's own header: the phone page's
    mime can be bare ("video/webm"), and a codec the stream does not have stops the player."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    kind = mime.split(";")[0].strip()
    if kind in ("video/webm", "video/x-matroska"):
        for tag, c in ((b"V_VP9", "vp9"), (b"V_VP8", "vp8"), (b"V_AV1", "av01.0.08M.08"), (b"V_MPEG4/ISO/AVC", "avc1.42E01E")):
            if tag in head:
                return f'video/webm; codecs="{c}"'
    elif kind == "video/mp4":
        i = head.find(b"avcC")
        if i >= 0 and len(head) >= i + 8:
            return f'video/mp4; codecs="avc1.{head[i + 5]:02X}{head[i + 6]:02X}{head[i + 7]:02X}"'
        i = head.find(b"hvcC")
        if i >= 0 and len(head) >= i + 17:
            p, lvl = head[i + 5], head[i + 16]
            return f'video/mp4; codecs="hvc1.{p & 31}.6.{"H" if p & 32 else "L"}{lvl}.B0"'
    return mime


def app(db, hub: Hub, tok, dist: Path = UI_DIST) -> Starlette:
    async def state(req):
        return JSONResponse(await asyncio.to_thread(_state, db, req.query_params.get("person")))

    async def object_(req):
        try:
            return JSONResponse(await asyncio.to_thread(searchview.explain_object, db, req.query_params.get("person"),
                                                        req.query_params.get("name"), tok))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def ask(req):
        from . import router
        b = await req.json()
        r = await asyncio.to_thread(router.ask, db, b.get("person"), b.get("question") or "",
                                    bool(b.get("use_model", True)))
        return JSONResponse(plain(r))

    async def steps(req):
        return JSONResponse(await asyncio.to_thread(_steps, db, tok, req.query_params.get("footage"),
                                                    req.query_params.get("person")))

    async def media(req):
        row = await asyncio.to_thread(db.footage.find_one, {"_id": req.path_params["footage"]}, {"path": 1, "mime": 1})
        if row is None or not row.get("path") or not Path(row["path"]).is_file():
            return Response(status_code=404)
        return FileResponse(row["path"], media_type=_mime(row).split(";")[0])

    async def feed(ws: WebSocket):
        await ws.accept()
        person, q = ws.query_params.get("person"), hub.subscribe("feed")
        try:
            await ws.send_json({"kind": "hello", "change_streams": hub.change_streams})
            while True:
                ev = visible(await q.get(), person)
                if ev is not None:
                    await ws.send_json(plain(ev))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe("feed", q)

    async def video(ws: WebSocket):
        fid = ws.path_params["footage"]
        row = await asyncio.to_thread(db.footage.find_one, {"_id": fid}, {"path": 1, "mime": 1, "status": 1})
        if row is None or not row.get("path") or not Path(row["path"]).is_file():
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            for _ in range(50):                                 # the header is in the first chunk: wait for it
                if Path(row["path"]).stat().st_size > 0:
                    break
                await asyncio.sleep(0.1)
            await ws.send_json({"mime": _mime(row), "type": codec(row["path"], _mime(row)), "status": row.get("status")})
            ended, checked = row.get("status") != "live", time.monotonic()
            with open(row["path"], "rb") as f:
                while True:
                    b = f.read(1 << 16)
                    if b:
                        await ws.send_bytes(b)
                        continue
                    if ended:
                        break
                    if time.monotonic() - checked > 0.5:
                        checked = time.monotonic()
                        r = await asyncio.to_thread(db.footage.find_one, {"_id": fid}, {"status": 1})
                        ended = r is None or r.get("status") != "live"
                        continue                                # read what arrived before it ended
                    await asyncio.sleep(0.1)
            await ws.send_json({"end": True})
            await ws.close()
        except (WebSocketDisconnect, RuntimeError):
            pass

    @contextlib.asynccontextmanager
    async def lifespan(_):
        hub.bind(asyncio.get_running_loop())
        yield

    routes = [Route("/api/state", state), Route("/api/object", object_), Route("/api/ask", ask, methods=["POST"]),
              Route("/api/steps", steps),
              Route("/media/{footage}", media), WebSocketRoute("/feed", feed), WebSocketRoute("/video/{footage}", video)]
    if Path(dist).is_dir():
        routes.append(Mount("/", StaticFiles(directory=dist, html=True)))
    return Starlette(routes=routes, lifespan=lifespan)


def main(port: int = 8790, track_every: float = 5.0) -> None:
    import uvicorn
    from . import atlas, lastseen, tower
    tower.use_stack()                                           # the tokenizer comes from the side stack
    tok = lastseen.tokenizer()
    db = atlas.database()
    hub, stop = Hub(), threading.Event()
    threading.Thread(target=watch, args=(db, hub, tok, stop), name="stage feed", daemon=True).start()
    threading.Thread(target=track_live, args=(db, tok, stop, track_every), name="stage tracker", daemon=True).start()
    print(f"stage: http://127.0.0.1:{port}/ (this machine only)"
          + ("" if Path(UI_DIST).is_dir() else "; ui/ not built yet: cd ui && npm run dev"), flush=True)
    try:
        uvicorn.run(app(db, hub, tok), host="127.0.0.1", port=port, log_level="warning", ws_max_size=16 << 20)
    finally:
        stop.set()
