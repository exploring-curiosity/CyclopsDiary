"""Transaction log: the Delta-Lake idea at StreetDex scale.

A table's state is not "the files in the directory" — it is the fold of an
append-only log of JSON commits. That one move buys, with plain files:

- **snapshot isolation / time travel**: version N is immutable forever; a
  reader at version N never sees version N+1's files.
- **atomic multi-file commits**: a commit lands as one exclusively-created
  log entry (O_EXCL is the lock — single writer, many readers).
- **file-level zone maps**: every added file records rows/bytes/min_ts/max_ts,
  so a time-window query prunes whole files from the log alone, before any
  Parquet footer is opened. (Row-group pruning inside surviving files is
  Parquet's own statistics — two layers, same idea.)
- **schema-on-log**: the schema travels with the commit, so evolution is an
  append, never a rewrite.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileEntry:
    path: str  # relative to the table dir
    rows: int
    bytes: int
    min_ts: int
    max_ts: int
    # Zone map for columns BEYOND ts: {column: [min, max]}. min_ts/max_ts
    # are the same idea hard-coded for the one column every table has;
    # this generalises it to whatever a table is clustered on, so
    # "object_id == 7" can drop a file from the LOG - kilobytes of JSON
    # already in memory - instead of opening its Parquet footer. That is
    # a layer below Parquet: layer 0 costs nothing per file, the footer
    # costs a seek and a read per surviving file, and at a thousand
    # files the difference is the query.
    #
    # Only meaningful for a column the file is CLUSTERED on. A min/max
    # over an unsorted column spans nearly the whole domain and prunes
    # nothing, so the writer records these only for its sort keys -
    # a zone map that never prunes is pure metadata cost.
    zone: dict = field(default_factory=dict)
    # END of the latest interval in this file, i.e. max(t1) - NOT
    # max(ts). For point-like rows the two are the same and this is
    # redundant; for INTERVALS they are not, and the difference is a
    # wrong answer rather than a slow one.
    #
    # An object present from t=100 to t=900 in a file whose starts span
    # 100..200 has max_ts=200. A query for [400, 500] overlaps that
    # object, but `max_ts < t0` prunes the whole file and returns
    # nothing. Verified: 0 rows where 1 was correct. Every table until
    # now was point-like or short-lived so it never fired; presence
    # intervals - "this object was on the table for four hours" - are
    # exactly the shape that breaks it.
    max_end: int = 0

    def to_json(self):
        d = dict(self.__dict__)
        if not d["zone"]:
            d.pop("zone")     # old readers, and old files, see no change
        if not d["max_end"]:
            d.pop("max_end")
        return d

    @staticmethod
    def from_json(d):
        # a file written before max_end existed falls back to max_ts,
        # which is CONSERVATIVE only for point-like rows. It is recorded
        # as such rather than silently trusted: see overlaps().
        return FileEntry(d["path"], d["rows"], d["bytes"], d["min_ts"],
                         d["max_ts"], d.get("zone", {}),
                         int(d.get("max_end", 0)))

    def overlaps(self, t0, t1) -> bool:
        """True unless this file provably holds no interval meeting
        [t0, t1]. Compares against the interval END, so a long-lived row
        that starts before the window is not pruned away."""
        end = self.max_end or self.max_ts
        if t0 is not None and end < t0:
            return False
        if t1 is not None and self.min_ts > t1:
            return False
        return True

    def may_contain(self, column, lo, hi) -> bool:
        """False only when this file PROVABLY holds nothing in [lo, hi].

        Absent statistics must answer True: a missing zone map means
        unknown, never empty. Every file written before zone maps
        existed takes that branch, so the optimisation degrades to the
        old behaviour instead of silently losing rows.
        """
        z = self.zone.get(column)
        if not z:
            return True
        return not (hi < z[0] or lo > z[1])


@dataclass
class TableState:
    version: int = 0
    kind: str = "timeseries"
    schema: str = ""
    files: list[FileEntry] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def rows(self):
        return sum(f.rows for f in self.files)

    @property
    def bytes(self):
        return sum(f.bytes for f in self.files)

    @property
    def min_ts(self):
        return min((f.min_ts for f in self.files), default=0)

    @property
    def max_ts(self):
        return max((f.max_ts for f in self.files), default=0)


CHECKPOINT_EVERY = 10  # Delta checkpoints every 10th commit; same dial here


def _fsync_dir(path: Path):
    """Durability of a file's *existence* requires fsyncing its directory —
    the metadata write that makes the entry reachable after power loss."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_file(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CommitConflict(RuntimeError):
    """A concurrent commit removed files this transaction depended on."""


class TableLog:
    def __init__(self, table_dir: Path):
        self.dir = Path(table_dir)
        self.log_dir = self.dir / "_log"

    def versions(self) -> list[int]:
        if not self.log_dir.is_dir():
            return []
        return sorted(int(p.stem) for p in self.log_dir.glob("*.json")
                      if p.stem.isdigit())

    def _checkpoints(self) -> list[int]:
        if not self.log_dir.is_dir():
            return []
        return sorted(int(p.name.split(".")[0])
                      for p in self.log_dir.glob("*.checkpoint.json"))

    def read_state(self, version: int | None = None) -> TableState:
        st = TableState()
        start = 0
        # Checkpoints make the fold O(commits since checkpoint) instead of
        # O(all commits) — the log stays an audit trail without becoming a
        # read cost. Same move as Delta's _last_checkpoint.
        for v in reversed(self._checkpoints()):
            if version is None or v <= version:
                c = json.loads(
                    (self.log_dir / f"{v:020d}.checkpoint.json").read_text())
                st.version = c["version"]
                st.kind = c["kind"]
                st.schema = c["schema"]
                st.meta = dict(c["meta"])
                st.files = [FileEntry.from_json(f) for f in c["files"]]
                start = v
                break
        for v in self.versions():
            if v <= start:
                continue
            if version is not None and v > version:
                break
            entry = json.loads((self.log_dir / f"{v:020d}.json").read_text())
            st.version = v
            st.kind = entry.get("table_kind", st.kind)
            st.schema = entry.get("schema", st.schema)
            st.meta.update(entry.get("meta", {}))
            removed = set(entry.get("remove", []))
            if removed:
                st.files = [f for f in st.files if f.path not in removed]
            st.files += [FileEntry.from_json(f) for f in entry.get("add", [])]
        return st

    def commit(self, *, op: str, kind: str, schema: str = "",
               add: list[FileEntry] = (), remove: list[str] = (),
               meta: dict | None = None, retries: int = 5) -> int:
        """Optimistic concurrency, Delta-style: O_EXCL on the next log entry
        is the lock. Losing the race means retrying at the next version —
        append-vs-append never truly conflicts. Commits that REMOVE files
        revalidate against the fresh state first: if a concurrent writer
        already removed one of ours, that is a real conflict and we fail
        cleanly instead of double-applying."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(retries):
            version = (self.versions() or [0])[-1] + 1
            if remove:
                active = {f.path for f in self.read_state().files}
                missing = [r for r in remove if r not in active]
                if missing:
                    raise CommitConflict(
                        "files no longer active (concurrent rewrite?): "
                        f"{missing[:3]}")
            entry = {
                "version": version,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "op": op,
                "table_kind": kind,
                "schema": schema,
                "add": [f.to_json() for f in add],
                "remove": list(remove),
                "meta": meta or {},
            }
            path = self.log_dir / f"{version:020d}.json"
            # Torn-write-proof commit: the entry is fully written and fsynced
            # under a temp name, then hard-LINKED to its final name. link()
            # is atomic AND exclusive (fails if the name exists), so a
            # partially-written entry can never appear under a version name —
            # a crash leaves only an ignorable *.tmp. Atomicity + durability
            # in one primitive; fsync of the directory makes the rename
            # itself survive power loss.
            tmp = self.log_dir / f".{version:020d}.{os.getpid()}.tmp"
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(entry, indent=1))
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                os.unlink(tmp)
                continue  # lost the race — re-read state, take the next slot
            os.unlink(tmp)
            _fsync_dir(self.log_dir)
            if version % CHECKPOINT_EVERY == 0:
                st = self.read_state(version)
                cp_tmp = self.log_dir / f".cp{version}.{os.getpid()}.tmp"
                cp_tmp.write_text(
                    json.dumps({"version": st.version, "kind": st.kind,
                                "schema": st.schema, "meta": st.meta,
                                "files": [f.to_json() for f in st.files]}))
                os.replace(cp_tmp,  # checkpoints are derived: replace is fine
                           self.log_dir / f"{version:020d}.checkpoint.json")
            # `_meta.json` beside the table: the human/tool-readable summary
            # of CURRENT state (schema, rows, bytes, ts range) so a store
            # browser or a pilot's script answers "what is in this table?"
            # with one file read, not a log replay. Derived — a failure
            # here must never fail the commit.
            try:
                st = self.read_state(version)
                meta_tmp = self.dir / f".meta.{os.getpid()}.tmp"
                meta_tmp.write_text(json.dumps({
                    "version": st.version, "op": op, "kind": st.kind,
                    "rows": sum(f.rows for f in st.files),
                    "bytes": sum(f.bytes for f in st.files),
                    "files": len(st.files),
                    "min_ts": min((f.min_ts for f in st.files), default=None),
                    "max_ts": max((f.max_ts for f in st.files), default=None),
                    "schema": st.schema, "meta": st.meta,
                    "ts_utc": entry["ts_utc"]}, indent=1))
                os.replace(meta_tmp, self.dir / "_meta.json")
            except Exception:
                pass
            return version
        raise CommitConflict(f"lost the commit race {retries} times")

    def history(self) -> list[dict]:
        out = []
        for v in self.versions():
            e = json.loads((self.log_dir / f"{v:020d}.json").read_text())
            out.append({"version": v, "ts_utc": e["ts_utc"], "op": e["op"],
                        "added_files": len(e.get("add", [])),
                        "added_rows": sum(a["rows"] for a in e.get("add", [])),
                        "meta": e.get("meta", {})})
        return out
