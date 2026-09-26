"""Footage into the shared memory.

One recording from one person's camera, read by the world model as one
continuous video, lands in MongoDB as a footage row and one step row per
second. The raw file is never changed or copied (raw data is immutable):
the footage row points at it and carries its sha256, so the same bytes
ingested twice are one footage.

The read is the engine's: one sequential decode (elidedb.video.scan_object,
every byte counted, the frame index built in the same pass), the tower's
rate sampled (streamtext.StreamEmbedder), each 1-s step of two frames handed
to the model with its memory carried from the start of the recording.

Resumable: a step is keyed "<footage>:<i>", so a re-run after an interrupt
re-reads the recording -- the model's memory has to be rebuilt from its
start -- but writes only the steps that are missing; footage marked ready is
skipped. The count that marks it ready is the rows in MongoDB, not the exit
of the read.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from bson.binary import Binary
from pymongo.errors import BulkWriteError

from . import config, tower


def sha256_of(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


PIXEL_NAME = re.compile(r"^PXL_(\d{8})_(\d{6})(\d{3})")


def _iso(v) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(v).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def capture_start(name: str, meta: dict) -> tuple:
    """When the phone started recording, by the phone's own convention. -> (datetime | None, where from).

    A Pixel names the file after the start in UTC (PXL_20260926_151010310 is 15:10:10.310) while its
    creation_time is later (the save: 20 s later on a 9.3 s clip, 2026-09-26). An iPhone writes the start,
    with its time zone, as com.apple.quicktime.creationdate. Anything else: the container's creation_time,
    which may be the save rather than the start; a sync shot's clock_offset_s corrects it."""
    m = PIXEL_NAME.match(name)
    if m:
        t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return t + timedelta(milliseconds=int(m.group(3))), "pixel file name"
    for key, where in (("com.apple.quicktime.creationdate", "apple creationdate"), ("creation_time", "creation_time")):
        if meta.get(key) and _iso(meta[key]) is not None:
            return _iso(meta[key]), where
    return None, None


def probe(path) -> dict:
    """The container's own facts, from its header and first frame."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else 0.0)
        meta = {**dict(s.metadata), **dict(c.metadata)}
        info = dict(width=int(s.codec_context.width), height=int(s.codec_context.height),
                    fps=float(s.average_rate) if s.average_rate else None, duration_s=dur)
    when, where = capture_start(Path(path).name, meta)
    info.update(created=when, clock_from=where, rotation=tower.rotation_of(path))
    return info


def write_steps(db, footage: dict, rows) -> int:
    """rows: (t0, t1, state, ids, lp) of one footage, t0 in seconds from its start. -> rows newly written
    (a step already stored is left as it is)."""
    pin = footage["pin"]
    atom = float(pin["n_frames"]) / float(pin["rate"])
    docs = []
    for t0, t1, state, ids, lp in rows:
        i = int(round(float(t0) / atom))
        docs.append({"_id": f"{footage['_id']}:{i}", "workspace": footage["workspace"], "source": footage["source"],
                     "person": footage.get("person"), "footage": footage["_id"], "i": i,
                     "t0": float(t0), "t1": float(t1),
                     "observed_at": footage["started_at"] + timedelta(seconds=float(t0)),
                     "state": Binary(np.asarray(state, np.float16).tobytes()),
                     "ids": Binary(np.asarray(ids, np.int32).tobytes()),
                     "lp": Binary(np.asarray(lp, np.float16).tobytes())})
    if not docs:
        return 0
    try:
        return len(db.steps.insert_many(docs, ordered=False).inserted_ids)
    except BulkWriteError as e:
        if any(w.get("code") != 11000 for w in e.details.get("writeErrors", [])):
            raise
        return int(e.details.get("nInserted", 0))


def ingest(db, path, source: str, started_at: datetime | None = None, enc=None, batch: int = 32,
           role: str = "footage") -> dict:
    """role "example": a query's clip, searched as part of the set but never an answer (query.py).
    The clock is started_at when given, else the file's own creation time (or its mtime less its
    duration), plus the source's clock_offset_s (its sync-shot correction)."""
    s = config.get()
    path = Path(path).resolve()
    src = db.sources.find_one({"_id": source})
    if src is None:
        raise ValueError(f"no source {source!r}: add it first (bin/cyclopsdiary source add)")
    sha = sha256_of(path)
    fid = sha[:20]
    old = db.footage.find_one({"_id": fid})
    if old is not None and old.get("status") == "ready":
        return dict(footage=fid, skipped=True, steps=old.get("steps", 0))
    info = probe(path)
    upright_wh = (info["width"], info["height"]) if info["rotation"] % 2 == 0 else (info["height"], info["width"])
    hw = tower.frame_hw(*upright_wh, s.frame_px)
    pin = dict(model=s.model, rate=s.rate, n_frames=s.n_frames, frame_hw=list(hw), window=s.window,
               top_k=s.top_k, row="memory state + the atom's head top-k")
    if old is not None and old.get("pin") != pin:
        raise ValueError(f"footage {fid} was partly read under another pin; delete its steps first")
    started = started_at or info["created"] or (
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) - timedelta(seconds=info["duration_s"]))
    clock_from = "given" if started_at else (info["clock_from"] or "file mtime less duration")
    if started_at is None:
        started += timedelta(seconds=float(src.get("clock_offset_s") or 0.0))
    db.footage.update_one({"_id": fid}, {"$setOnInsert": dict(
        workspace=src["workspace"], source=source, person=src["person"], path=str(path), sha256=sha,
        bytes=path.stat().st_size, width=info["width"], height=info["height"], fps=info["fps"],
        duration_s=info["duration_s"], rotation=info["rotation"], started_at=started, clock_from=clock_from,
        clock_offset_s=None if started_at else float(src.get("clock_offset_s") or 0.0), pin=pin, role=role,
        status="reading", steps=0)}, upsert=True)
    footage = db.footage.find_one({"_id": fid})

    from tqdm import tqdm
    if enc is None:
        if tower._ENC is None:
            print(f"loading {s.model} (vision tower in torch, language model in MLX) ...", flush=True)
        enc = tower.encoder(s.model)
    rd = tower.Read(enc, hw)
    print(f"read at {hw[1]}x{hw[0]}, {s.n_frames} frames a step at {s.rate:g} fps", flush=True)
    from elidedb import video as V
    bar = tqdm(total=max(1, math.ceil(info["duration_s"] * s.rate / s.n_frames)), unit="step",
               desc=f"{source}: {path.name}")
    buf: list = []
    wrote = 0
    k = info["rotation"]

    def take(rows):
        nonlocal wrote
        buf.extend(rows)
        bar.update(len(rows))                               # counted once the step has been read
        if len(buf) >= batch:
            wrote += write_steps(db, footage, buf)
            buf.clear()

    # Resampled at the decode, in memory (the file is never re-encoded): every frame is still decoded, as
    # later frames reference it, but only the frames the embedder samples (its own test, `due`) are turned
    # into RGB and stood upright; the rest pass as None with their times, which the embedder still needs
    # to learn the source rate. A 60 fps phone clip keeps one frame in 30; the decoder runs on several
    # threads. The model's input is unchanged: rows byte-identical on both sample clips (2026-09-26).
    def want(t):
        return t >= rd.due - 1e-9

    def on_frame(t, img):
        take(rd.push(t, tower.upright(img, k) if img is not None else None))

    t0 = time.perf_counter()
    d = V.scan_object(str(path), on_frame=on_frame, want=want, thread_type="AUTO")
    take(rd.flush())
    wrote += write_steps(db, footage, buf)
    bar.close()
    dt = time.perf_counter() - t0
    n = db.steps.count_documents({"footage": fid})
    db.footage.update_one({"_id": fid}, {"$set": dict(
        status="ready", steps=n, fps=d["fps"], duration_s=d["duration_s"], bytes_read=d["bytes_read"],
        read_s=round(dt, 2), x_real_time=round(d["duration_s"] / dt, 2) if dt > 0 else None,
        ingested_at=datetime.now(timezone.utc))})
    return dict(footage=fid, steps=n, written=wrote, started_at=started, duration_s=d["duration_s"],
                x_real_time=round(d["duration_s"] / dt, 2) if dt > 0 else None,
                bytes=int(path.stat().st_size), bytes_read=int(d["bytes_read"]))
