"""Video path: the frame index is a Parquet table; pixels stay in the
source files. A window query filters index ROWS (cheap, columnar), then
decode preads exactly the byte ranges those rows point at — late
materialization, media never copied or re-encoded into the store."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa

from .store import _uncached          # media pages are DATABASE pages:
# the pixel path moves far more bytes than every Parquet read combined,
# so leaving it on the OS page cache would have kept the cache the
# engine claims not to use. Same F_NOCACHE policy, same ELIDEDB_CACHE
# switch, scoped to the store's own media files.


def scan_video_packets(video_path, timestamps_ns=None) -> dict:
    """Packet-level scan (no decode) via PyAV if present, else ffprobe.
    Returns columnar dict for the frame_index schema."""
    import json
    import subprocess

    from .fftools import find
    r = subprocess.run(
        [find("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_packets",
         "-show_entries", "packet=pos,size,flags,pts",
         "-show_entries", "stream=codec_name,width,height,time_base",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True)
    doc = json.loads(r.stdout)
    stream = doc["streams"][0]
    pk = doc["packets"]
    n = len(pk)
    if timestamps_ns is not None and len(timestamps_ns) < n:
        n = len(timestamps_ns)  # truncated sidecar: index the known prefix
    if timestamps_ns is not None:
        ts = list(timestamps_ns[:n])
    else:
        # container pts are TIMEBASE TICKS, not seconds — convert via the
        # stream's time_base (e.g. AVI at 1/30: pts 0,1,2 = 0, 33.3, 66.7 ms)
        num, den = (int(x) for x in stream.get("time_base", "1/1000000000").split("/"))
        ts = [int(float(p.get("pts", i)) * num / den * 1e9)
              for i, p in enumerate(pk[:n])]
    return {
        "ts": pa.array(ts, pa.int64()),
        "byte_offset": pa.array([int(p["pos"]) for p in pk[:n]], pa.int64()),
        "packet_size": pa.array([int(p["size"]) for p in pk[:n]], pa.int32()),
        "keyframe": pa.array([p.get("flags", "").startswith("K") for p in pk[:n]]),
        "width": pa.array([stream["width"]] * n, pa.int32()),
        "height": pa.array([stream["height"]] * n, pa.int32()),
        "codec": pa.array([stream["codec_name"]] * n),
        "source": pa.array([str(Path(video_path).resolve())] * n),
    }


class FrameSet:
    """Lazy handle over frame_index rows in a query window."""

    def __init__(self, store, table_name, rows: pa.Table):
        self.store = store
        self.table_name = table_name
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        streams = (set(self.rows.column("stream").to_pylist())
                   if "stream" in self.rows.column_names else set())
        return f"<FrameSet {len(self.rows)} frames, streams={sorted(streams)}>"

    def streams(self):
        if "stream" not in self.rows.column_names:
            return []
        return sorted(set(self.rows.column("stream").to_pylist()))

    def _resolve(self, src: str) -> str:
        # "@media/..." = store-managed media (standalone store); anything
        # else is an external reference-in-place.
        return str(self.store.dir / src[1:]) if src.startswith("@") else src

    def decode(self, stream=None, stride=1, width=None, limit=None,
               workers=8):
        """pread + decode selected frames → list of (ts_ns, np.uint8 HWC).

        Reads EXACTLY the byte ranges the selection requires. Two paths:
        - intra codecs (mjpeg): each packet is a standalone JPEG — read the
          selected packets, decode them in parallel (cv2 releases the GIL).
        - inter codecs (hevc/h264 elementary streams, the compressed
          `transcode=` tier): decode is GOP-granular — read from the
          preceding keyframe through the last selected packet, pipe through
          ffmpeg, keep the selected frames. Keyframe interval = the
          random-access dial chosen at ingest."""
        import pyarrow.compute as pc
        rows = self.rows
        if stream is not None and "stream" in rows.column_names:
            rows = rows.filter(pc.equal(rows.column("stream"), stream))
        if len(rows) == 0:
            self.last_bytes_read = 0
            return []
        sel = list(range(0, len(rows), stride))
        if limit:
            sel = sel[:limit]
        codec = rows.column("codec")[0].as_py() if "codec" in rows.column_names \
            else "mjpeg"
        if codec in ("hevc", "h264"):
            return self._decode_gop(rows, sel, codec, width)

        # ---- intra path: parallel per-packet decode -------------------------
        import cv2
        from concurrent.futures import ThreadPoolExecutor
        offs = rows.column("byte_offset").to_pylist()
        sizes = rows.column("packet_size").to_pylist()
        tss = rows.column("ts").to_pylist()
        srcs = rows.column("source").to_pylist()
        bufs, bytes_read, handles = [], 0, {}
        try:
            for i in sel:
                src = srcs[i]
                if src not in handles:
                    handles[src] = _uncached(self._resolve(src))
                f = handles[src]
                f.seek(offs[i])
                b = f.read(sizes[i])
                bytes_read += len(b)
                bufs.append((tss[i], b))
        finally:
            for f in handles.values():
                f.close()

        def _one(item):
            ts, b = item
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None  # torn tail packet of a mid-write segment
            if width and img.shape[1] > width:
                h = int(img.shape[0] * width / img.shape[1])
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            return (ts, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = [r for r in ex.map(_one, bufs) if r is not None]
        self.last_bytes_read = bytes_read
        return out

    def _decode_gop(self, rows, sel, codec, width):
        """GOP-granular decode for inter-codec elementary streams.

        The selection is partitioned into CONTIGUOUS RUNS and each run is read
        and decoded separately. That partitioning is the whole point: the
        earlier version read one span from the first selected frame's keyframe
        to the last selected packet, so a scattered selection — exactly what
        `stride=N` sampling produces — read and decoded the entire file to
        return a handful of frames. Measured on a 4103 s Bridge stream: 96
        frames spread across the file cost 188 ms/frame, against 2.7 ms/frame
        for a contiguous window, because 20,515 frames were being decoded to
        hand back 96.

        Runs whose byte ranges touch are merged, so a dense selection still
        becomes one large sequential read (one ffmpeg call, not one per GOP)
        while a sparse one becomes many small GOP reads. Both regimes read
        only the bytes they need.
        """
        import subprocess

        import cv2
        import pyarrow.compute as pc

        from .fftools import find
        src = rows.column("source")[sel[0]].as_py()
        w = rows.column("width")[0].as_py()
        h = rows.column("height")[0].as_py()

        # Full frame index for this source: needed to find each selected
        # frame's governing keyframe. An index read, not a media read.
        idx = self.store.table(self.table_name).scan()
        idx = idx.filter(pc.equal(idx.column("source"), src))
        ts_all = idx.column("ts").to_numpy()
        off_all = idx.column("byte_offset").to_numpy()
        size_all = idx.column("packet_size").to_numpy()
        key_all = np.asarray(idx.column("keyframe").to_pylist(), dtype=bool)
        key_pos = np.where(key_all)[0]
        if len(key_pos) == 0:
            key_pos = np.array([0])

        sel_ts = np.array([rows.column("ts")[i].as_py() for i in sel],
                          dtype=np.int64)
        pos = np.searchsorted(ts_all, sel_ts)
        pos = np.clip(pos, 0, len(ts_all) - 1)
        gov = key_pos[np.clip(np.searchsorted(key_pos, pos, side="right") - 1,
                              0, len(key_pos) - 1)]

        # Build merged [byte_start, byte_end) runs, each tagged with the index
        # position its first decoded frame corresponds to.
        runs = []
        for p, g in zip(pos, gov):
            a = int(off_all[g])
            b = int(off_all[p]) + int(size_all[p])
            if runs and a <= runs[-1]["end"]:
                runs[-1]["end"] = max(runs[-1]["end"], b)
                runs[-1]["want"].add(int(p))
            else:
                runs.append({"start": a, "end": b, "first": int(g),
                             "want": {int(p)}})

        frame_bytes = w * h * 3
        ff = find("ffmpeg")
        # Each run begins at a keyframe, so the runs concatenate into one
        # valid elementary stream: N GOP reads still cost only ONE decoder
        # invocation. Process spawn was the dominant cost once the byte ranges
        # were correct (96 runs = 96 ffmpeg starts ~= 20 ms each).
        payloads, spans = [], []
        total = 0
        with _uncached(self._resolve(src)) as f:
            for r in runs:
                f.seek(r["start"])
                buf = f.read(r["end"] - r["start"])
                total += len(buf)
                payloads.append(buf)
                spans.append((r["first"], max(r["want"]) - r["first"] + 1,
                              r["want"]))
        self.last_bytes_read = total

        proc = subprocess.run(
            [ff, "-v", "error", "-f", codec, "-i", "pipe:0",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            input=b"".join(payloads), capture_output=True)
        n_out = len(proc.stdout) // frame_bytes
        expected = sum(n for _, n, _ in spans)

        def _take(buf, k):
            img = np.frombuffer(buf, np.uint8, count=frame_bytes,
                                offset=k * frame_bytes).reshape(h, w, 3)
            if width and w > width:
                nh = int(h * width / w)
                img = cv2.resize(img, (width, nh),
                                 interpolation=cv2.INTER_AREA)
            return img.copy()

        out = []
        if n_out == expected:
            base = 0
            for first, n, want in spans:
                for k in range(n):
                    ip = first + k
                    if ip in want:
                        out.append((int(ts_all[ip]), _take(proc.stdout,
                                                           base + k)))
                base += n
        else:
            # The concatenated decode did not line up (open GOPs, a truncated
            # tail). Fall back to decoding each run on its own rather than
            # returning frames under the wrong timestamps — a silently
            # misaligned frame is worse than a slow one.
            for buf, (first, _n, want) in zip(payloads, spans):
                pr = subprocess.run(
                    [ff, "-v", "error", "-f", codec, "-i", "pipe:0",
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                    input=buf, capture_output=True)
                for k in range(len(pr.stdout) // frame_bytes):
                    ip = first + k
                    if ip in want:
                        out.append((int(ts_all[ip]), _take(pr.stdout, k)))
        out.sort(key=lambda x: x[0])
        return out


# ------------------------------------------------------------ over S3
_CODEC_NAMES = {"libdav1d": "av1", "libaom-av1": "av1", "libvpx-vp9": "vp9", "libvpx": "vp8", "hevc": "hevc"}


def mp4_header_ranges(f, size: int) -> list:
    """(offset, length) of every byte of an MP4-family file that is not
    media: the top-level atoms walked by their headers (eight to sixteen
    bytes read per atom), with mdat's payload left out and its own box
    header kept, adjacent ranges merged. A file with moov at the head
    yields one range up to the first media byte; moov at the tail adds a
    second. A demuxer opened over these bytes alone has everything that
    describes the stream (sample table, codec parameters, clock) and
    nothing that is a picture. Empty for anything that is not an MP4
    family container."""
    ranges, off = [], 0
    while off + 8 <= size:
        f.seek(off)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        sz, typ, hlen = int.from_bytes(hdr[:4], "big"), hdr[4:8], 8
        if sz == 1:
            sz, hlen = int.from_bytes(f.read(8), "big"), 16
        elif sz == 0:
            sz = size - off
        if off == 0 and typ not in (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"):
            return []
        if sz < hlen:
            break
        keep = hlen if typ == b"mdat" else min(sz, size - off)
        if ranges and ranges[-1][0] + ranges[-1][1] == off:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + keep)
        else:
            ranges.append((off, keep))
        off += sz
    return ranges


def _held(body: bytes):
    """The bytes we already have, behind the same counting interface a
    bucket read has. It reports zero bytes read, which is the truth: the
    elision figure counts what came off the wire."""
    import io

    from .window import CountingFile
    return CountingFile(io.BytesIO(body), counts=False)


def scan_object(url: str, on_frame=None, label: str | None = None, probesize: int = 1 << 16,
                body: bytes | None = None, ref=None, want=None, thread_type: str | None = None) -> dict:
    """One sequential pass over an object through the S3 API (or any
    fsspec URL): every packet becomes a frame-index row (pts relative to
    the first, byte offset, size, keyframe flag) and, when `on_frame` is
    given, every frame is decoded and handed over as (seconds, HWC uint8)
    in the same pass. Also returns the container's own facts (fps,
    duration, first pts, creation time), the header ranges worth caching
    (everything that is not media) with their bytes and the container's
    format name, the object size, and the bytes the pass read, counted at
    the file object.

    `body` is the object's bytes when the caller already has them -- a
    pushed segment, which arrived over HTTP and was written to the bucket
    in this same request. The offsets recorded are offsets into the
    object, which is the same file either way, so a later read plans
    against the bucket exactly as if this pass had read it from there.
    Passing them saves downloading what the process is still holding:
    measured at 1.94x the segment, in six GETs.

    `ref` is the object's description when the caller already has it. It
    is a HeadObject either way; doing it twice per object is a round trip
    a ten-thousand-object backfill pays ten thousand times for a fact in
    hand.

    `want(seconds)`, when given, says which frames the caller will keep: the
    others are still decoded (a later frame may reference them) but handed
    over as None, without the conversion to RGB. A reader sampling 2 fps
    from a 60 fps phone clip keeps one frame in 30. `thread_type` ("AUTO")
    lets the decoder use several threads; the frames are the same either way.
    """
    import av
    from .connect import stat
    from .window import open_source

    ref = ref if ref is not None else stat(url)
    size = len(body) if body is not None else int(ref.size)
    pts, pos, psize, key = [], [], [], []
    with (_held(body) if body is not None else open_source(url, sequential=True)) as f:
        ranges = mp4_header_ranges(f, size)
        header_bytes = []
        for off, ln in ranges:
            f.seek(off)
            header_bytes.append(f.read(ln))
        f.seek(0)
        c = av.open(f, options={"probesize": str(probesize), "analyzeduration": "0"})
        try:
            s = c.streams.video[0]
            if thread_type and on_frame is not None:
                s.thread_type = thread_type
            fmt = c.format.name.split(",")[0]          # the demuxer's short name
            tb = s.time_base
            start = int(s.start_time) if s.start_time is not None else None
            total = int(s.frames) if s.frames else None
            demux = c.demux(s)
            if label:
                from tqdm import tqdm
                demux = tqdm(demux, total=total, unit="frame", desc=label)
            for pkt in demux:
                if pkt.size and pkt.pts is not None and pkt.pos is not None and pkt.pos >= 0:
                    pts.append(int(pkt.pts))
                    pos.append(int(pkt.pos))
                    psize.append(int(pkt.size))
                    key.append(bool(pkt.is_keyframe))
                if on_frame is not None:
                    for fr in pkt.decode():
                        if fr.pts is None:
                            continue
                        if start is None:
                            start = int(fr.pts)
                        t = float((fr.pts - start) * tb)
                        on_frame(t, fr.to_ndarray(format="rgb24") if want is None or want(t) else None)
            cc = s.codec_context
            name = _CODEC_NAMES.get(cc.name, cc.name)
            width, height = int(cc.width), int(cc.height)
            fps = float(s.average_rate) if s.average_rate else None
            duration = float(s.duration * tb) if s.duration else (float(c.duration) / 1e6 if c.duration else 0.0)
            created = c.metadata.get("creation_time")
        finally:
            c.close()
        bytes_read = f.bytes_read
    if pts:
        t0 = min(pts)
        ts = [int(round(float((p - t0) * tb) * 1e9)) for p in pts]
        first_pts_ns = int(round(float(t0 * tb) * 1e9))
    else:
        ts, first_pts_ns = [], 0
    if not fps and duration and pts:
        fps = len(pts) / duration
    n = len(pts)
    created_ns = None
    if created:
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            created_ns = int((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp() * 1e9)
        except ValueError:
            created_ns = None
    return {
        "ts": pa.array(ts, pa.int64()), "byte_offset": pa.array(pos, pa.int64()),
        "packet_size": pa.array(psize, pa.int32()), "keyframe": pa.array(key, pa.bool_()),
        "width": pa.array([width] * n, pa.int32()), "height": pa.array([height] * n, pa.int32()),
        "codec": pa.array([name] * n), "source": pa.array([url] * n),
        "fps": fps, "duration_s": float(duration), "first_pts_ns": first_pts_ns, "creation_time_ns": created_ns,
        "size": size, "etag": ref.etag, "tier": ref.tier, "storage_class": ref.storage_class,
        "bytes_read": int(bytes_read), "header": {"ranges": ranges, "bytes": header_bytes, "format": fmt},
    }
