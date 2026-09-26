"""Many cameras into one memory, and the queries answered, by one long-running worker.

`bin/cyclopsdiary serve --inbox DIR` loads the world model once, then loops:

  inbox    DIR/<source id>/ holds one camera's clips as they arrive (AirDrop, cable, a
           sync client). A clip is ingested once its size and mtime have held still for one
           poll, so a copy still in flight is left alone, under the source its folder names.
           Clips are never moved, renamed or changed: the footage row points at the file by
           path and sha256, and a clip already in the memory is skipped.
  queries  rows in `queries` with status "queued" are claimed one at a time (an atomic
           find-and-modify, so several workers are safe) and answered into `events`
           (query.py). On Atlas a change stream on `queries` wakes the worker the moment a
           row is inserted; on a standalone mongod it polls. At start, claims left
           "running" by a dead worker on this machine are re-queued. Answering needs
           MongoDB and the formula, not the model, so `serve` without --inbox is a query
           worker that loads nothing.

  live     with --live, phones stream their cameras from the browser into this same process
           (camserver.py, live.py); --tunnel gives them a public HTTPS link, so they can be on
           any network. Each phone's recording is read as it arrives.

The model is one 8.5 GB copy. Each recording is read as one continuous video with the
model's memory carried: inbox clips one at a time, live phones alongside them, each with
its own memory, the model taking one step at a time (tower.Read). An interrupted ingest
resumes where its steps stop (ingest.py); a live recording cut off by a stopped service is
closed at the next start from what reached the disk.
"""
from __future__ import annotations

import time
from pathlib import Path

from pymongo.errors import PyMongoError

from . import atlas, config, query, tower

VIDEO = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


class Inbox:
    """DIR/<source>/<clip>: the clips whose size and mtime held still since the last scan."""

    def __init__(self, root):
        self.root = Path(root)
        self.last: dict = {}            # path -> (size, mtime) at the previous scan
        self.done: set = set()          # (path, size, mtime) handled in this process
        self.warned: set = set()

    def ready(self, db) -> list:
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            for f in sorted(d.iterdir()):
                if f.name.startswith(".") or f.suffix.lower() not in VIDEO or not f.is_file():
                    continue
                st = f.stat()
                key = (str(f), st.st_size, st.st_mtime)
                if key in self.done:
                    continue
                prev, self.last[str(f)] = self.last.get(str(f)), key[1:]
                if prev != key[1:]:
                    continue                                    # new, or still being written: wait a poll
                if db.sources.find_one({"_id": d.name}) is None:
                    if d.name not in self.warned:
                        print(f"inbox: no source {d.name!r} (bin/cyclopsdiary source add); its clips wait", flush=True)
                        self.warned.add(d.name)
                    continue
                out.append((key, f, d.name))
        return out


def serve(db, inbox=None, poll: float = 2.0, once: bool = False, live: int | None = None,
          tunnel: bool = False) -> dict:
    """once: one pass over the inbox and the queue, then return (tests, cron). live: the camera service's port
    (phones stream from the browser); tunnel: a public HTTPS link to it."""
    from . import ingest
    s = config.get()
    enc = None
    box = Inbox(inbox) if inbox else None
    if box is not None or live:
        print(f"loading {s.model} once for every camera (vision tower in torch, language model in MLX) ...",
              flush=True)
        enc = tower.encoder(s.model)
    if live:
        _cameras(db, live, tunnel)
    if box is not None:
        from . import lastseen
        tok = lastseen.tokenizer()                              # the model's own words, for the objects' places
        print(f"watching {box.root}/<source>/ for clips", flush=True)
    back = query.requeue_stale(db)
    cs = None if once else atlas.change_stream(db.queries, {"operationType": "insert"}, poll)
    print(("answering queries as they are inserted (change stream)" if cs is not None else
           f"answering queued queries every {poll:g} s (no change streams on this server)")
          + (f"; {back} left running by a dead worker re-queued" if back else "")
          + "; Ctrl-C stops (an interrupted clip resumes)", flush=True)
    if box is not None:
        box.ready(db)                                           # the first look: a clip is ready once it holds still
        if once:
            time.sleep(poll)
    n = dict(ingested=0, skipped=0, failed_clips=0, answered=0, failed_queries=0)
    while True:
        worked = False
        for key, f, src in (box.ready(db) if box is not None else []):
            try:
                r = ingest.ingest(db, f, src, enc=enc)
                if r.get("skipped"):
                    n["skipped"] += 1
                else:
                    n["ingested"] += 1
                    print(f"{src}: {f.name} -> footage {r['footage']}, {r['steps']} steps, "
                          f"{r['x_real_time']}x real time", flush=True)
                    from . import objects
                    ws = db.sources.find_one({"_id": src})["workspace"]
                    for k, v in objects.track_workspace(db, ws, tok).items():    # every owner's objects, privately
                        print(f"  tracked {k}: {v['by_name']} by name, {v['by_example']} by example", flush=True)
            except Exception as e:                              # noqa: BLE001 -- one bad clip must not stop the rest
                n["failed_clips"] += 1
                print(f"{src}: {f.name} failed: {type(e).__name__}: {e}", flush=True)
            box.done.add(key)
            worked = True
        while (q := query.claim(db)) is not None:
            try:
                done = query.answer(db, q)
                n["answered"] += 1
                print(f"query {q['_id']} ({q['direction']}, {q['scope']}): {done['events']} moments, "
                      f"{done['apart']} stand apart, over {done['searched']['steps']} steps", flush=True)
            except Exception as e:                              # noqa: BLE001 -- the query row carries the reason
                n["failed_queries"] += 1
                query.fail(db, q, e)
                print(f"query {q['_id']} failed: {type(e).__name__}: {e}", flush=True)
            worked = True
        if once:
            return n
        if not worked:
            if cs is not None:
                try:
                    cs.try_next()                               # returns on the next insert, or after `poll`
                except PyMongoError as e:
                    print(f"change stream lost ({type(e).__name__}: {e}); polling every {poll:g} s", flush=True)
                    cs = None
            else:
                time.sleep(poll)


def _cameras(db, port: int, tunnel: bool) -> dict:
    """The camera service beside the worker, on the same model (camserver.py, live.py); with a tunnel, the
    public link the phones open."""
    import atexit
    from . import camserver
    from . import live as LV
    stale = LV.close_stale(db)
    cam = camserver.start(db, port=port, tunnel=tunnel)
    if cam["tunnel"] is not None:
        atexit.register(cam["tunnel"].terminate)
    print((f"{stale} live recordings left open by a stopped service closed; " if stale else "")
          + f"cameras: open {cam['public'] or cam['local']} on each phone"
          + ("" if cam["public"] else " (this machine only; --tunnel gives phones a link)"), flush=True)
    return cam
