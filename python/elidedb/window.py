"""Windows: a moment back with the minimum bytes.

Video: the source stays in its bucket. The frame index is our own sample
table: for a window it names the keyframe governing t0 and the last
packet at or before t1, in FILE order, so the bytes between them are ONE
GetObject Range and nothing else. The container is never demuxed at
query time: the header cached at ingest (ftyp, moov, the mdat box header;
nothing that is a picture) describes the stream, the packets are sliced
out of the one Range by the index, and a decoder built from that
description is fed them. Timestamps are the index's, so a frame's time is
the stream's clock. An object without an index falls back to a seek over
a counting file, still through the same API.

Telemetry: the recording's rows inside the window (pruned to its own row
groups through the rec_id value pushdown), resampled onto one timeline at
the caller's rate by nearest or linear interpolation. Alignment is a
query parameter, never a storage commitment.
"""
from __future__ import annotations

import io
from fractions import Fraction

import numpy as np
import pyarrow as pa

from .connect import parse_url


class CountingFile:
    """A binary file-like that counts what is actually read.

    `counts=False` for bytes that never crossed the wire -- a pushed
    segment the process is still holding. The reads happen, but charging
    them to the elision figure would report traffic that did not occur,
    and that figure is only worth having if it is the wire.
    """

    def __init__(self, f, counts: bool = True):
        self._f = f
        self._counts = counts
        self.bytes_read = 0
        self.reads = 0

    def read(self, n=-1):
        b = self._f.read(n)
        if self._counts:
            self.bytes_read += len(b)
            self.reads += 1
        return b

    def readinto(self, buf):
        b = self._f.read(len(buf))
        n = len(b)
        buf[:n] = b
        if self._counts:
            self.bytes_read += n
            self.reads += 1
        return n

    def seek(self, off, whence=0):
        return self._f.seek(off, whence)

    def tell(self):
        return self._f.tell()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


#: One sequential GET's worth. Large enough that a 206 MB object is a few
#: dozen round trips rather than a few thousand, small enough to hold in
#: memory beside a tower.
SCAN_BLOCK = 8 << 20


def open_source(url: str, block_size: int = 1 << 20, sequential: bool = False) -> CountingFile:
    """A counting file over an object.

    `cache_type="none"` by default, so a planned read asks the bucket for
    exactly its range: read-ahead here is the bug that made a clip plan
    301,412 bytes and request 52,730,212.

    `sequential=True` for a pass that reads the whole object in order --
    the ingest scan, which has to. Read-ahead costs nothing there, because
    every byte is going to be read, and without it the demuxer's own 32 kB
    buffer set the request size: 123 GETs for a 3.9 MB object, about 6,600
    for a 206 MB one, each a serial round trip.
    """
    scheme, fs, path = parse_url(url)
    if scheme == "file":
        from .store import _uncached          # the store's own no-cache policy
        return CountingFile(_uncached(path))
    if sequential:
        return CountingFile(fs.open(path, "rb", block_size=SCAN_BLOCK,
                                    cache_type="readahead"))
    return CountingFile(fs.open(path, "rb", block_size=block_size, cache_type="none"))


def decode_span(url: str, t0: float, t1: float, width=None, probesize=1 << 16):
    """-> ([(ts_seconds, HWC uint8), ...], bytes_read). Frames with
    t0 <= ts <= t1 from a seek to the governing keyframe."""
    import av
    frames = []
    with open_source(url) as f:
        # a small probe window: the container index says where things are,
        # a 5 MB default probe would read a short file whole
        c = av.open(f, options={"probesize": str(probesize), "analyzeduration": "0"})
        try:
            s = c.streams.video[0]
            tb = float(s.time_base)
            c.seek(int(t0 / tb), stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                if fr.pts is None:
                    continue
                t = fr.pts * tb
                if t < t0 - 1e-9:
                    continue                  # inside the governing GOP, before t0
                if t > t1 + 1e-9:
                    break
                img = fr.to_ndarray(format="rgb24")
                if width and img.shape[1] > width:
                    import cv2
                    h = int(img.shape[0] * width / img.shape[1])
                    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
                frames.append((float(t), img))
        finally:
            c.close()
        return frames, f.bytes_read


class HeldFile(io.RawIOBase):
    """A read-only file object over the bytes of an object that are held
    in memory (the header ranges cached at ingest): reads inside them are
    served, reads anywhere else are the end of the file. Nothing is
    fetched. A container opened over it yields the stream's description
    (codec, extradata, dimensions, time base) without touching media."""

    def __init__(self, size: int, held):
        self.size = int(size)
        self._held = sorted(((int(o), b) for o, b in held), key=lambda r: r[0])
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        base = {0: 0, 1: self._pos, 2: self.size}[whence]
        self._pos = max(0, base + int(off))
        return self._pos

    def readinto(self, buf):
        pos = self._pos
        for off, data in self._held:
            if off <= pos < off + len(data):
                k = min(len(buf), off + len(data) - pos)
                buf[:k] = data[pos - off:pos - off + k]
                self._pos += k
                return k
        return 0


def open_header(header: dict, size: int):
    """The container opened over its cached header alone: no media, no
    request. -> (container, video stream). The stream is the template a
    decoder or a muxer is built from; the caller closes the container."""
    import av
    held = list(zip((o for o, _ in header["ranges"]), header["bytes"]))
    c = av.open(HeldFile(size, held), format=header.get("format") or "mov")
    return c, c.streams.video[0]


def plan_span(frames_rows: pa.Table, t0_ns: int, t1_ns: int):
    """The one Range a window costs, from the frame index alone. Rows go
    into FILE order (offset order): the governing keyframe is the last
    keyframe at or before t0, the end is the last packet at or before t1,
    and every packet between them is in the plan, so a frame stored after
    the frames it references (B-frames) is decodable. -> {offset, length,
    key_ts_ns, ts, off, size, key} with the packet arrays in file order,
    or None when the index has no rows."""
    if frames_rows is None or len(frames_rows) == 0 or "byte_offset" not in frames_rows.column_names:
        return None
    ts = frames_rows.column("ts").to_numpy().astype(np.int64)
    off = frames_rows.column("byte_offset").to_numpy().astype(np.int64)
    size = frames_rows.column("packet_size").to_numpy().astype(np.int64)
    key = np.asarray(frames_rows.column("keyframe").to_pylist(), bool)
    order = np.argsort(off, kind="stable")
    ts, off, size, key = ts[order], off[order], size[order], key[order]
    keys = np.flatnonzero(key & (ts <= int(t0_ns)))
    if len(keys):
        k = int(keys[-1])
    else:                                    # before the first keyframe: start there
        anyk = np.flatnonzero(key)
        k = int(anyk[0]) if len(anyk) else 0
    js = np.flatnonzero(ts <= int(t1_ns))
    j = int(js[-1]) if len(js) and int(js[-1]) >= k else k
    sel = slice(k, j + 1)
    return {"offset": int(off[k]), "length": int(off[j] + size[j] - off[k]), "key_ts_ns": int(ts[k]),
            "ts": ts[sel], "off": off[sel], "size": size[sel], "key": key[sel]}


def fetch_plan(url: str, plan: dict) -> bytes:
    """The one request a plan costs: GetObject with Range on a bucket, a
    seek and a read on a local file."""
    from .connect import read_range
    return read_range(url, plan["offset"], plan["length"])


def packets_of(plan: dict, span: bytes):
    """The packets of a fetched plan as (ts_ns, bytes, keyframe), file order."""
    base = plan["offset"]
    for t, o, n, k in zip(plan["ts"], plan["off"], plan["size"], plan["key"]):
        a = int(o - base)
        yield int(t), span[a:a + int(n)], bool(k)


NS = Fraction(1, 10**9)


def decode_plan(template, plan: dict, span: bytes, t0_ns: int, t1_ns: int,
                width=None, limit=None) -> list:
    """Frames with t0 <= ts <= t1 from the packets of one fetched plan: a
    decoder built from the cached header's stream (codec, extradata) is
    fed the packets in file order with the index's timestamps, so what
    comes out carries the stream's clock. Stops at the first frame past
    t1, or after `limit` frames. -> [(seconds, HWC uint8), ...]"""
    import av
    cc = template.codec_context
    dec = av.CodecContext.create(cc.name, "r")
    if cc.extradata:
        dec.extradata = cc.extradata
    frames = []

    def take(fr) -> bool:
        if fr.pts is None:
            return False
        t = int(fr.pts)
        if t < t0_ns:
            return False
        if t > t1_ns:
            return True
        img = fr.to_ndarray(format="rgb24")
        if width and img.shape[1] > width:
            import cv2
            h = int(img.shape[0] * width / img.shape[1])
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        frames.append((t / 1e9, img))
        return bool(limit) and len(frames) >= limit

    done = False
    for t, data, _ in packets_of(plan, span):
        p = av.Packet(data)
        p.pts = t
        p.time_base = NS
        for fr in dec.decode(p):
            if take(fr):
                done = True
                break
        if done:
            break
    if not done:
        for fr in dec.decode(None):
            if take(fr):
                break
    return frames


def align(rows: pa.Table, t0_ns: int, t1_ns: int, rate_hz: float,
          interp: str = "nearest") -> dict:
    """Numeric columns of `rows` resampled onto [t0, t1] at rate_hz."""
    step = max(int(round(1e9 / float(rate_hz))), 1)
    timeline = np.arange(int(t0_ns), int(t1_ns) + 1, step, dtype=np.int64)
    out = {"timeline_ns": timeline.tolist()}
    if len(rows) == 0:
        return out
    ts = rows.column("ts").to_numpy().astype(np.int64)
    for name in rows.column_names:
        if name in ("ts", "rec_id"):
            continue
        col = rows.column(name)
        if not (pa.types.is_floating(col.type) or pa.types.is_integer(col.type)):
            continue
        v = col.to_numpy(zero_copy_only=False).astype(np.float64)
        if interp == "linear":
            out[name] = np.interp(timeline, ts, v).tolist()
        elif interp == "nearest":
            idx = np.clip(np.searchsorted(ts, timeline), 0, len(ts) - 1)
            prev = np.clip(idx - 1, 0, len(ts) - 1)
            use_prev = (timeline - ts[prev]) <= (ts[idx] - timeline)
            out[name] = v[np.where(use_prev, prev, idx)].tolist()
        else:
            raise ValueError("interp must be 'nearest' or 'linear'")
    return out
