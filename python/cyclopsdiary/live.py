"""Live cameras: a phone's recording, streamed in chunks, read by the world model as it arrives.

A phone records with its own encoder -- VP8/VP9 WebM on Android Chrome, fragmented H.264 MP4 on iPhone
Safari -- and sends each chunk as it comes (camserver.py). Each recording is one LiveSession:

  raw      the chunks are appended, as received, to live/<source>/<footage id>.<webm|mp4>. The footage is
           exactly what the phone's encoder wrote, written once and never re-encoded; its sha256 is taken
           when the recording ends.
  read     a decoder follows that file as it grows and hands every frame to the read a clip gets
           (tower.Read: one continuous read with the model's memory carried across the chunks, two frames a
           step at 2 fps, only the sampled frames converted and stood upright). Each step is written to
           `steps` as soon as it is read, so the recording can be searched while it is still going
           (memory.py reads "live" footage too).
  footage  the row is made when the recording starts: status "live", started_at the phone's clock at record
           start plus the source's clock_offset_s. When it ends it becomes "ready" with the file's size and
           sha256 and the steps read. A recording cut off mid-chunk (the socket dropped) keeps what arrived:
           the decoder stops at the torn chunk and the error is kept on the row.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config, ingest, tower

ROOT = config.ROOT / "live"
KINDS = {"video/webm": ("webm", "matroska"), "video/x-matroska": ("mkv", "matroska"), "video/mp4": ("mp4", "mp4")}


def close_stale(db) -> int:
    """Recordings left "live" by a service that stopped mid-recording, closed from what reached the disk:
    ready with the steps that were read (failed with none), and the file's size and sha256."""
    n = 0
    for f in db.footage.find({"status": "live"}):
        p = Path(f["path"])
        steps = db.steps.count_documents({"footage": f["_id"]})
        n += db.footage.update_one({"_id": f["_id"], "status": "live"}, {"$set": dict(
            status="ready" if steps else "failed", steps=steps, sha256=ingest.sha256_of(p) if p.exists() else None,
            bytes=p.stat().st_size if p.exists() else 0,
            error="the service stopped before the recording ended")}).modified_count
    return n


class _Tail:
    """The recording's file read as it grows: read() waits for more bytes until the recording has ended.
    It has no seek, so the demuxer treats it as the stream it is."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.grew = threading.Condition()
        self.ended = False

    def more(self) -> None:
        with self.grew:
            self.grew.notify_all()

    def end(self) -> None:
        with self.grew:
            self.ended = True
            self.grew.notify_all()

    def read(self, n: int = -1) -> bytes:
        with self.grew:
            while True:
                b = self.f.read(n)
                if b or self.ended:
                    return b
                self.grew.wait(0.5)

    def close(self) -> None:
        self.f.close()


class LiveSession:
    def __init__(self, db, source: str, started_at: datetime, mime: str, make_read=None, root: Path = ROOT):
        """make_read(hw) -> a tower.Read for this recording (default: on the process's one model)."""
        s = config.get()
        src = db.sources.find_one({"_id": source})
        if src is None:
            raise ValueError(f"no source {source!r}: add it first (bin/cyclopsdiary source add)")
        kind = mime.split(";")[0].strip().lower()
        if kind not in KINDS:
            raise ValueError(f"{mime!r}: the phone must send one of {', '.join(KINDS)}")
        ext, self.fmt = KINDS[kind]
        offset = float(src.get("clock_offset_s") or 0.0)
        started = started_at.astimezone(timezone.utc) + timedelta(seconds=offset)
        fid = f"{source}-{started:%Y%m%dT%H%M%S}{started.microsecond // 1000:03d}"
        self.path = Path(root) / source / f"{fid}.{ext}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "xb")                      # a recording's file is written once, never over
        self.db, self.make_read = db, make_read or (lambda hw: tower.Read(tower.encoder(s.model), hw))
        self.footage = dict(
            _id=fid, workspace=src["workspace"], source=source, person=src["person"], path=str(self.path),
            sha256=None, bytes=0, mime=mime, started_at=started, clock_from="the phone's clock at record start",
            clock_offset_s=offset, pin=dict(model=s.model, rate=s.rate, n_frames=s.n_frames, frame_hw=None,
                                            window=s.window, top_k=s.top_k,
                                            row="memory state + the atom's head top-k"),
            role="footage", status="live", steps=0, duration_s=0.0)
        db.footage.insert_one(self.footage)
        self.bytes, self.steps, self.frames, self.t_last, self.dt = 0, 0, 0, None, None
        self.read, self.error, self.t_start = None, None, time.perf_counter()
        self.tail = _Tail(self.path)
        self.thread = threading.Thread(target=self._run, name=f"live {fid}", daemon=True)
        self.thread.start()

    def feed(self, chunk: bytes) -> None:
        self.f.write(chunk)
        self.f.flush()
        self.bytes += len(chunk)
        self.tail.more()

    def stop(self) -> dict:
        """The recording has ended (the phone stopped, or its socket dropped): read what is left, then close
        the row. -> dict(footage, steps, bytes, duration_s, status, error)"""
        if not self.f.closed:
            self.f.close()
        self.tail.end()
        self.thread.join()
        self.tail.close()
        size, sha = self.path.stat().st_size, ingest.sha256_of(self.path)
        n = self.db.steps.count_documents({"footage": self.footage["_id"]})
        dur = (self.t_last + (self.dt or 0.0)) if self.t_last is not None else 0.0
        dt = time.perf_counter() - self.t_start
        done = dict(status="ready" if n else "failed", sha256=sha, bytes=size, steps=n, duration_s=round(dur, 3),
                    fps=round(self.frames / dur, 3) if dur > 0 else None, read_s=round(dt, 2),
                    ingested_at=datetime.now(timezone.utc), error=self.error)
        self.db.footage.update_one({"_id": self.footage["_id"]}, {"$set": done})
        return dict(footage=self.footage["_id"], steps=n, bytes=size, duration_s=done["duration_s"],
                    status=done["status"], error=self.error)

    def _begin(self, width: int, height: int, k: int) -> None:
        """The first frame decides the read's frame size: the source's aspect, stood upright."""
        wh = (width, height) if k % 2 == 0 else (height, width)
        hw = tower.frame_hw(*wh, config.get().frame_px)
        self.read = self.make_read(hw)
        self.footage["pin"]["frame_hw"] = list(hw)
        self.db.footage.update_one({"_id": self.footage["_id"]}, {"$set": {
            "pin.frame_hw": list(hw), "width": int(width), "height": int(height), "rotation": int(k)}})

    def _take(self, rows) -> None:
        if not rows:
            return
        self.steps += ingest.write_steps(self.db, self.footage, rows)
        self.db.footage.update_one({"_id": self.footage["_id"]}, {"$set": {
            "steps": self.steps, "duration_s": round(float(rows[-1][1]), 3)}})

    def _run(self) -> None:
        import av
        try:
            c = av.open(self.tail, format=self.fmt, options={"probesize": "32768", "analyzeduration": "0"})
            try:
                st = c.streams.video[0]
                st.thread_type = "AUTO"
                tb, start, k = st.time_base, None, 0
                for pkt in c.demux(st):
                    for fr in pkt.decode():
                        if fr.pts is None:
                            continue
                        if start is None:
                            start = fr.pts
                            k = int(round(float(fr.rotation or 0) / 90.0)) % 4
                            self._begin(fr.width, fr.height, k)
                        t = float((fr.pts - start) * tb)
                        if self.t_last is not None and t > self.t_last:
                            self.dt = min(self.dt or t - self.t_last, t - self.t_last)
                        self.t_last, self.frames = t, self.frames + 1
                        img = tower.upright(fr.to_ndarray(format="rgb24"), k) if t >= self.read.due - 1e-9 else None
                        self._take(self.read.push(t, img))
            finally:
                c.close()
        except Exception as e:                              # noqa: BLE001 -- a torn last chunk ends the read here
            self.error = f"{type(e).__name__}: {e}"[:300]
        finally:
            if self.read is not None:
                try:
                    self._take(self.read.flush())
                except Exception as e:                      # noqa: BLE001
                    self.error = self.error or f"{type(e).__name__}: {e}"[:300]
