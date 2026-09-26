"""Store: a directory of Parquet tables under a transaction log.

Everything is Parquet — sensor rows, the video frame index, embeddings,
centroids. No custom byte formats. The design borrows one idea from each
system it wants to be judged against:

- warehouses (Vertica/BigQuery): columnar + statistics pruning, projection
  pushdown — Parquet row groups and column chunks give both natively.
- Spark: predicate pushdown to the scan; a query touches the minimum files,
  row groups, and columns.
- Delta/Iceberg: the table is a log fold; snapshots, time travel, atomic
  appends (log.py).
- C-Store: late materialization — video pixels are produced last, from byte
  ranges the frame-index table points at; the raw media file is never copied
  into the store.
- Kafka/streaming: time is the primary axis; every table MUST carry `ts`
  (int64 ns, sorted within a file) — that is the one schema law here.
- lakehouse: open format means other engines read the store for free;
  `Store.sql()` is DuckDB pointed at the very same files.
"""
from __future__ import annotations

import io
import json
import time
import contextlib
import fcntl
import importlib
import os
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .log import FileEntry, TableLog, fsync_file

ROW_GROUP_ROWS = 64 * 1024  # the amortize-vs-overfetch dial (Parquet's
                            # row-group size == SDX's chunk_target_rows)


ROW_GROUP_TARGET_BYTES = 8 * 1024 * 1024


def _row_width(schema: pa.Schema) -> int:
    """Approximate uncompressed bytes per row, for row-group sizing."""
    w = 0
    for f in schema:
        t = f.type
        try:
            if pa.types.is_fixed_size_list(t):
                w += t.list_size * (t.value_type.bit_width // 8)
            else:
                w += t.bit_width // 8
        except (ValueError, AttributeError):
            w += 32                       # strings/lists: a guess is fine
    return max(w, 1)


# THE DATABASE DOES NOT CACHE. Not yet, deliberately: a buffer pool is
# real work with real invalidation rules, and it is the LAST thing to
# build, after the layout and the pruning are right. Until then the
# honest position is that repeat reads should cost what first reads
# cost, so a benchmark number cannot be quietly borrowed from the OS.
#
# Reads therefore open store files with F_NOCACHE (macOS) / O_DIRECT-ish
# advice, which tells the kernel not to keep this file's pages in the
# unified buffer cache. That is scoped to OUR files: the system cache
# and every other process's data are untouched, which `sudo purge` can
# never say. Set ELIDEDB_CACHE=1 to opt back into the OS page cache.
_NOCACHE = os.environ.get("ELIDEDB_CACHE", "0") not in ("1", "true")
_F_NOCACHE = 48                      # <sys/fcntl.h>, Darwin


def _uncached(path):
    """Open a store file so its pages are not retained by the kernel.

    Returns a plain binary file object; pyarrow accepts any file-like,
    so this drops in wherever a path was passed. Falls back to a normal
    open anywhere the fcntl is unavailable (Linux, odd filesystems) -
    losing the no-cache property is worth strictly less than crashing,
    and the caller is told by ELIDEDB_CACHE what it asked for.
    """
    f = open(path, "rb", buffering=0)
    try:
        fcntl.fcntl(f.fileno(), _F_NOCACHE, 1)
    except Exception:
        pass
    return f


class _PF:
    """pq.ParquetFile that OWNS its file object and closes it.

    The first version handed `_uncached(path)` straight to ParquetFile
    and returned. ParquetFile does not take ownership, so every footer
    read leaked one open file object - and a leaked Python file object
    with a live buffer makes the interpreter hang at SHUTDOWN, not at
    the point of the leak. That is why an inspection script printed all
    of its output and then sat at 0% CPU for 27 minutes holding nothing
    visible: it was stuck tearing down, and `lsof` showed zero because
    the fd table was already gone.

    Every read path since the no-cache change was leaking. Closing is
    the fix; being a context manager as well means callers can be
    explicit where it matters.
    """

    __slots__ = ("_fh", "pf")

    def __init__(self, path):
        self._fh = _uncached(path) if _NOCACHE else None
        self.pf = pq.ParquetFile(self._fh if self._fh is not None
                                 else str(path))

    def __getattr__(self, k):
        return getattr(self.pf, k)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def close(self):
        try:
            self.pf.close()
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def __del__(self):
        self.close()


def _pf(path):
    """pq.ParquetFile honouring the no-cache policy, and closing."""
    return _PF(path)


def _read(path, **kw):
    """pq.read_table honouring the no-cache policy.

    Reads through a _PF and COPIES the result, so the file object cannot
    outlive this call. `with _uncached(path) as fh: pq.read_table(fh)`
    looks safe and is not - pyarrow keeps a reference to the handle for
    lazy access, so the `with` closes nothing and the leaked handle hangs
    the interpreter at shutdown. That is the same failure the _pf fix
    addressed, surviving in the other reader: a two-table scan would sit
    at 0% CPU forever while a single-table one exited fine.
    """
    if not _NOCACHE:
        return pq.read_table(str(path), **kw)
    f = _PF(path)
    try:
        cols = kw.get("columns")
        filters = kw.get("filters")
        t = f.pf.read(columns=cols)
        if filters is not None:
            t = t.filter(filters)
        return t.combine_chunks()      # materialise before the handle dies
    finally:
        f.close()


def _top(md, c: int) -> str:
    """Top-level column name for row-group column index `c`.

    Parquet flattens nested types to LEAVES, and md.schema.names is the
    leaf list: a fixed-size-list column named `vector` appears there as
    `element`, with path_in_schema `vector.list.element`. Matching a
    projection against the leaf name therefore never matched a vector
    column, so its bytes were never charged - 8.7 MB per row group on
    frame_vectors, silently absent from every elision number in a store
    that is 87% vectors. Metrics that undercount are worse than no
    metrics: they make the headline claim look better than it is.
    """
    return md.row_group(0).column(c).path_in_schema.split(".")[0]


def _col_index(md, name: str):
    """Row-group column index for a TOP-LEVEL column name, or None."""
    for c in range(md.num_columns):
        if _top(md, c) == name:
            return c
    return None


def _max_end(table: pa.Table) -> int:
    """Latest interval END in this table: max(t1), falling back to
    max(ts) for a table that has no t1 (point rows end where they
    start)."""
    col = "t1" if "t1" in table.column_names else "ts"
    try:
        v = pc.max(table.column(col)).as_py()
    except Exception:
        return 0
    return int(v or 0)


def _zone(table: pa.Table, columns) -> dict:
    """File-level min/max for the columns a file is clustered on.

    Recorded in the commit log, so a value predicate can drop whole
    files from JSON already in memory - before any Parquet footer is
    opened. Strings are kept as strings and numbers as numbers; the
    comparison at read time is the caller's, and it must compare like
    with like or it will prune wrongly rather than merely badly.
    """
    z = {}
    for c in columns or ():
        if c not in table.column_names or c == "ts":
            continue      # ts already has its own dedicated pair
        col = table.column(c)
        try:
            lo, hi = pc.min(col).as_py(), pc.max(col).as_py()
        except pa.ArrowNotImplementedError:
            continue      # lists, structs: no order, no zone map
        if lo is not None and hi is not None:
            z[c] = [lo, hi]
    return z


def write_parquet(table: pa.Table, path):
    """One writer for every file in the store. `ts` gets
    DELTA_BINARY_PACKED — timestamps are near-arithmetic, so delta encoding
    beats generic zstd ~3x on that column (the Gorilla/TSDB observation);
    string columns keep dictionary encoding; everything rides zstd.

    Row groups are sized by ROW WIDTH to a byte target, not a fixed row
    count: at 64k rows a 1152-d float32 vector table packed ~295 MB into
    ONE group, so time-range pruning inside a file could skip nothing —
    the elision law applied to layout. A narrow sensor table still gets
    tens of thousands of rows per group; a vector table gets ~1.8k, and a
    2 s window read touches one group instead of the whole file."""
    rows = min(ROW_GROUP_ROWS,
               max(4096, ROW_GROUP_TARGET_BYTES // _row_width(table.schema)))
    dict_cols = [f.name for f in table.schema
                 if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    pq.write_table(table, path, row_group_size=rows,
                   compression="zstd",
                   use_dictionary=dict_cols,
                   # The PAGE INDEX (per-page min/max + offsets) is what
                   # lets a reader skip pages INSIDE a surviving row group
                   # — the layer that makes a columnar file behave like an
                   # index for selective reads. pyarrow omits it by
                   # default, so every file written before 2026-07-28 can
                   # only prune to row-group granularity. It costs a small
                   # constant in the footer and is read only when a query
                   # has a predicate that can use it.
                   write_page_index=True,
                   column_encoding={"ts": "DELTA_BINARY_PACKED"})
    fsync_file(path)  # durability: data reaches disk BEFORE the commit that
                      # references it — the write-ahead ordering rule


class QueryStats:
    """Bytes accounting: the elision number, file-granular.

    Log-level pruning is exact (a pruned file contributes 0 bytes). Inside a
    surviving file, Parquet's own row-group pruning + column projection cut
    further; we report the surviving files' bytes as the upper bound actually
    mapped, plus rows returned."""

    def __init__(self):
        self.corpus_bytes = 0
        self.files_total = 0
        self.files_touched = 0
        self.bytes_touched = 0
        self.rows_returned = 0
        self.wall_ms = 0.0

    @property
    def elided_pct(self):
        if not self.corpus_bytes:
            return 0.0
        b = min(self.bytes_touched, self.corpus_bytes)
        return 100.0 * (self.corpus_bytes - b) / self.corpus_bytes

    def __repr__(self):
        return (f"<{self.files_touched}/{self.files_total} files, "
                f"{self.bytes_touched:,} B touched of {self.corpus_bytes:,} B "
                f"({self.elided_pct:.3f}% elided), {self.rows_returned:,} rows, "
                f"{self.wall_ms:.1f} ms>")


class Table:
    def __init__(self, store: "Store", name: str):
        self.store = store
        self.name = name
        self.dir = store.dir / "tables" / name
        self.log = TableLog(self.dir)

    def state(self, version=None):
        # A snapshot pin is per table -- {table: version} -- because tables
        # commit independently and a consistent read holds each at its own
        # number. Resolving it here rather than at every call site is what
        # keeps a caller holding a pin from having to know it holds one;
        # before this, any read that reached a table directly with a pin
        # compared a dict against an integer and crashed.
        if isinstance(version, dict):
            version = version.get(self.name)
        return self.log.read_state(version)

    def history(self):
        return self.log.history()

    # ---- write path -------------------------------------------------------
    def _validate(self, table: pa.Table, evolve: bool):
        """Consistency guarantees enforced at the door: ts present, int64,
        never null; and the schema must match the table's existing schema
        (same names ⇒ same types). evolve=True permits ADDING columns —
        widening reads promote missing columns to null — but a type change
        for an existing name is always an error, never a silent coercion."""
        if "ts" not in table.column_names:
            raise ValueError(f"table '{self.name}': a `ts` int64-ns column is "
                             "required — timestamps are the one schema law")
        ts = table.column("ts")
        if ts.type != pa.int64():
            raise ValueError("`ts` must be int64 nanoseconds since epoch")
        if ts.null_count:
            raise ValueError(f"table '{self.name}': `ts` contains "
                             f"{ts.null_count} null(s) — every row must be "
                             "timestamped")
        st = self.state()
        if st.files:
            # compare against the LATEST file: after additive evolution the
            # newest schema is the table's current contract
            existing = _pf(
                self.dir / st.files[-1].path).schema_arrow
            have = {f.name: f.type for f in existing}
            new = {f.name: f.type for f in table.schema}
            for name, typ in new.items():
                if name in have and have[name] != typ:
                    raise ValueError(
                        f"table '{self.name}': column '{name}' is "
                        f"{have[name]} but incoming batch has {typ} — "
                        "type changes are never implicit")
            added = set(new) - set(have)
            missing = set(have) - set(new)
            if (added or missing) and not evolve:
                raise ValueError(
                    f"table '{self.name}': schema differs (new columns "
                    f"{sorted(added)}, absent columns {sorted(missing)}). "
                    "Pass evolve=True to allow additive evolution.")
        return st

    def _sorted(self, table: pa.Table) -> pa.Table:
        order = pc.sort_indices(table.column("ts"))
        if not pc.all(pc.equal(order, pa.array(range(len(table))))).as_py():
            table = table.take(order)  # ts-sorted files ⇒ tight zone maps
        return table

    # ---- layout policy ----------------------------------------------------
    def set_layout(self, cluster_by: str, *, sort_by=None,
                   min_group_rows=256) -> int:
        """Declare the table's CLUSTER KEY once, for every writer.

        Physical layout was a per-call argument, and the `events` table
        is what that costs. write_once wrote it grouped by `kind`;
        build_teacher rewrote it with plain replace(), ts-sorted. Same
        table, two writers, and the store ended up holding the
        unclustered one - so a lookup on `kind`, the verb, the single
        most natural predicate in the corpus, had to open 100% of row
        groups. Nothing errored. The table was simply no longer an
        index, and only a footer audit would ever have said so.

        Clustering is a property of a TABLE, the way a clustered index
        is, not a decision each INSERT gets to re-make. Declared here,
        it lands in the log - so it is versioned, travels with time
        travel, and any writer that goes through append/replace honours
        it without knowing it exists.
        """
        st = self.state()
        # A pure METADATA commit: no file added, none removed. The
        # declaration is a new version, so it is auditable and time
        # travel still lands on the layout that was in force then, but
        # it does not rewrite a byte. Existing files keep whatever
        # layout they were written with until something rebuilds them -
        # declaring an index does not reorganise the table.
        return self.log.commit(
            op="layout", kind=st.kind, schema=st.schema,
            meta={"layout": {
                "cluster_by": cluster_by,
                "sort_by": list(sort_by or [cluster_by, "ts"]),
                "min_group_rows": int(min_group_rows)}})

    def layout(self) -> dict | None:
        return self.state().meta.get("layout")

    def append(self, table: pa.Table, *, kind="timeseries", meta=None,
               evolve=False) -> int:
        lay = self.layout()
        if lay:
            return self.append_grouped(
                table, lay["cluster_by"], kind=kind, meta=meta,
                evolve=evolve, sort_by=lay["sort_by"],
                min_group_rows=lay["min_group_rows"])
        return self.append_batches([table], kind=kind, meta=meta,
                                   evolve=evolve)

    def append_batches(self, batches, *, kind="timeseries", meta=None,
                       evolve=False) -> int:
        """Write one file per batch, commit ONCE — a multi-gigabyte load is
        a single atomic transaction with bounded memory."""
        self.dir.mkdir(parents=True, exist_ok=True)
        adds, schema = [], None
        for batch in batches:
            if len(batch) == 0:
                continue
            self._validate(batch, evolve)
            batch = self._sorted(batch)
            schema = batch.schema
            fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
            path = self.dir / fname
            write_parquet(batch, path)
            tsv = batch.column("ts")
            adds.append(FileEntry(fname, len(batch), path.stat().st_size,
                                  tsv[0].as_py(), tsv[-1].as_py(),
                                  {}, _max_end(batch)))
        if not adds:
            raise ValueError("nothing to append (all batches empty)")
        return self.log.commit(op="append", kind=kind,
                               schema=str(schema), add=adds, meta=meta)

    def append_grouped(self, table: pa.Table, group_col: str, *,
                       kind="timeseries", meta=None, evolve=False,
                       sort_by=None, replace=False,
                       min_group_rows: int = 1, max_group_rows: int = 0) -> int:
        """Write with ROW GROUPS ALIGNED TO A LOGICAL UNIT.

        The default writer sizes row groups by bytes, which is right for a
        sensor stream and wrong for anything whose query unit is an
        object. The frames table is the proof: 39,026 rows landed in ONE
        row group of 981 KB, so a query for a single episode decompressed
        every frame in the store and layer-2 pruning measured 0.45%.

        Here each distinct `group_col` value becomes its own row group, so
        "give me episode X" touches exactly one. The trade is footer size
        — 1,122 groups x 9 columns is ~10k column-chunk entries of
        metadata, and the footer is read on EVERY query — which is the
        random-access-vs-metadata tension made explicit rather than
        inherited from a default.

        `max_group_rows` is the same dial from the other end, and it
        matters once one value is BIG. The frames table proved that too,
        from the opposite direction: with a whole recording as one object
        rather than 58 cut copies, one group per rec_id put 17,268 rows in
        a single group, so a four-second window decompressed the entire
        recording's index. A ceiling splits a large value across several
        groups; the table is sorted within the value, so each split group
        still carries a narrow ts range and a window reaches one or two.
        The floor and the ceiling answer different questions -- too many
        tiny values, and one value too large -- and a table can need both.

        `sort_by` is recorded in the footer as Parquet sorting_columns, so
        a reader can know the file is clustered without trusting us.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        if len(table) == 0:
            raise ValueError("nothing to write")
        self._validate(table, evolve)
        sort_by = sort_by or [group_col, "ts"]
        sort_by = [c for c in sort_by if c in table.column_names]
        table = table.take(pc.sort_indices(
            table, sort_keys=[(c, "ascending") for c in sort_by]))

        g = table.column(group_col).to_pylist()
        raw, start = [], 0
        for i in range(1, len(g) + 1):
            if i == len(g) or g[i] != g[start]:
                raw.append((start, i - start))
                start = i
        # MERGE ADJACENT GROUPS UP TO A ROW FLOOR. One group per distinct
        # value sounds maximally prunable and is a trap: metadata costs a
        # fixed amount per group PER COLUMN, so tiny groups invert the
        # ratio. Measured on the labels table at one-group-per-value -
        # 406 groups over 7,573 rows - the footer reached 286,796 bytes
        # against 283,957 bytes of actual column data. The footer was
        # LARGER THAN THE DATA, and it is read on every query, so the
        # "index" cost more to consult than the table cost to scan.
        #
        # The SORT is what makes statistics prunable; group size is an
        # independent dial. Merging keeps values contiguous, so each
        # group still covers a narrow min/max range, at a fraction of the
        # metadata.
        bounds = []
        if min_group_rows <= 1:
            bounds = raw
        else:
            off = cur = 0
            for s, n in raw:
                cur += n
                if cur >= min_group_rows:
                    bounds.append((off, cur))
                    off, cur = s + n, 0
            if cur:
                bounds.append((off, cur))
        if max_group_rows and max_group_rows > 0:
            split = []
            for off, n in bounds:
                for a in range(0, n, max_group_rows):
                    split.append((off + a, min(max_group_rows, n - a)))
            bounds = split

        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        path = self.dir / fname
        idx = {c: table.column_names.index(c) for c in sort_by}
        sc = [pq.SortingColumn(idx[c]) for c in sort_by]
        dict_cols = [f.name for f in table.schema
                     if pa.types.is_string(f.type)
                     or pa.types.is_large_string(f.type)]
        w = pq.ParquetWriter(path, table.schema, compression="zstd",
                             use_dictionary=dict_cols,
                             write_page_index=True,
                             sorting_columns=sc,
                             column_encoding={"ts": "DELTA_BINARY_PACKED"})
        try:
            for off, n in bounds:
                w.write_table(table.slice(off, n), row_group_size=n)
        finally:
            w.close()
        fsync_file(path)

        st = self.state()
        tsv = table.column("ts")
        add = [FileEntry(fname, len(table), path.stat().st_size,
                         tsv[0].as_py(), tsv[-1].as_py(),
                         _zone(table, sort_by), _max_end(table))]
        return self.log.commit(
            op="replace" if replace else "append", kind=kind,
            schema=str(table.schema), add=add,
            remove=[f.path for f in st.files] if replace else [],
            meta=dict(meta or {}, row_groups=len(bounds),
                      grouped_by=group_col, sorted_by=sort_by))

    def replace(self, table: pa.Table, *, kind="timeseries", meta=None,
                evolve=False) -> int:
        """REBUILD IN PLACE: these rows become the table, in one commit.

        Derived tables — events, answers, anything recomputed from the
        frames — need this and `append` is wrong for them. Recomputing
        the teacher's events with `append` took the table from 8,263 to
        16,526 rows, with `agent` at exactly 2x the demo count, and left
        every demo holding BOTH the old and the new typing of the same
        transition. Nothing errored; consumers that read a demo's kinds
        as a set just started seeing contradictions.

        Old files are removed in the same transaction that adds the new
        ones, so readers see one or the other and never the union, and
        the previous version stays addressable through the log.

        A rebuild MUST NOT quietly restore the table to ts-sorted: if a
        cluster key is declared, this rewrites through the grouped
        writer. That is exactly how `events` lost its layout - a
        recompute through replace() undid write_once's grouping."""
        lay = self.layout()
        if lay:
            return self.append_grouped(
                table, lay["cluster_by"], kind=kind, meta=meta,
                evolve=evolve, sort_by=lay["sort_by"], replace=True,
                min_group_rows=lay["min_group_rows"])
        st = self.state()
        prev = [f.path for f in st.files]
        self.dir.mkdir(parents=True, exist_ok=True)
        if len(table) == 0:
            raise ValueError("replace() with an empty table would drop "
                             "the table; use delete_range to truncate")
        self._validate(table, evolve)
        table = self._sorted(table)
        fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
        path = self.dir / fname
        write_parquet(table, path)
        tsv = table.column("ts")
        add = [FileEntry(fname, len(table), path.stat().st_size,
                         tsv[0].as_py(), tsv[-1].as_py(),
                         {}, _max_end(table))]
        return self.log.commit(op="replace", kind=kind,
                               schema=str(table.schema), add=add,
                               remove=prev,
                               meta=dict(meta or {},
                                         rows_before=st.rows,
                                         files_replaced=len(prev)))

    def compact(self, target_rows_per_file: int = 8_000_000) -> dict:
        """OPTIMIZE: rewrite the active file set into few large, ts-sorted,
        delta-encoded files — one atomic replace-commit. Fixes the many-
        small-files tax that every append-only log accumulates, and applies
        the current encodings to data written before them."""
        st = self.state()
        if not st.files:
            return {"files_before": 0, "files_after": 0}
        data = self.scan()
        before_bytes = st.bytes
        from .log import FileEntry as FE
        adds = []
        for lo in range(0, len(data), target_rows_per_file):
            part = data.slice(lo, target_rows_per_file)
            fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
            path = self.dir / fname
            write_parquet(part, path)
            tsv = part.column("ts")
            adds.append(FE(fname, len(part), path.stat().st_size,
                           tsv[0].as_py(), tsv[-1].as_py()))
        self.log.commit(op="compact", kind=st.kind, schema=str(data.schema),
                        add=adds, remove=[f.path for f in st.files],
                        meta={"files_before": len(st.files),
                              "bytes_before": before_bytes})
        after = sum(a.bytes for a in adds)
        return {"files_before": len(st.files), "files_after": len(adds),
                "bytes_before": before_bytes, "bytes_after": after,
                "ratio": round(before_bytes / max(after, 1), 2)}

    def delete_range(self, t0: int, t1: int) -> dict:
        """Delete rows with ts in [t0, t1] — the 'scrub that run' operation
        (bad takes, PII, retention). Files fully inside the range are just
        dropped; overlapping files are rewritten without the range; files
        outside are untouched. One atomic commit; prior versions still see
        the data (time travel is the audit trail) until their files are
        garbage-collected."""
        import pyarrow.compute as pc
        from .log import FileEntry as FE
        st = self.state()
        removes, adds, dropped = [], [], 0
        for f in st.files:
            if not f.overlaps(t0, t1):
                continue  # untouched
            removes.append(f.path)
            if t0 <= f.min_ts and f.max_ts <= t1:
                dropped += f.rows
                continue  # fully covered: no rewrite needed
            t = _read(self.dir / f.path)
            keep = t.filter(pc.or_(pc.less(t.column("ts"), t0),
                                   pc.greater(t.column("ts"), t1)))
            dropped += len(t) - len(keep)
            if len(keep):
                fname = f"part-{uuid.uuid4().hex[:12]}.parquet"
                write_parquet(keep, self.dir / fname)
                tsv = keep.column("ts")
                adds.append(FE(fname, len(keep),
                               (self.dir / fname).stat().st_size,
                               tsv[0].as_py(), tsv[-1].as_py()))
        if not removes:
            return {"rows_deleted": 0}
        self.log.commit(op="delete", kind=st.kind, schema=st.schema,
                        add=adds, remove=removes,
                        meta={"deleted_range": [t0, t1],
                              "rows_deleted": dropped})
        return {"rows_deleted": dropped, "files_rewritten": len(adds),
                "files_removed": len(removes)}

    # ---- secondary indexes (B+ tree over any numeric column) --------------
    def create_index(self, column: str, order: int = 256) -> dict:
        """Build an immutable, bulk-loaded B+ tree over `column` for the
        CURRENT version (BPT1 — same bytes the C++ engine reads). Zone maps
        prune nothing on unsorted columns; this re-sorts (value → row
        location) once, so point/range predicates touch only the row groups
        that actually contain hits. Rebuild after appends (`create_index`
        again) — the artifact is version-suffixed like any derived state."""
        from . import bptree
        st = self.state()
        keys, vals = [], []
        for fi, f in enumerate(st.files):
            col = _read(self.dir / f.path, columns=[column]) \
                .column(column).to_numpy(zero_copy_only=False)
            keys.append(bptree.encode_key(col))
            vals.append((np.uint64(fi) << np.uint64(40)) |
                        np.arange(len(col), dtype=np.uint64))
        img = bptree.build(np.concatenate(keys) if keys else
                           np.array([], np.int64),
                           np.concatenate(vals) if vals else
                           np.array([], np.uint64), order=order)
        ixdir = self.dir / "_index"
        ixdir.mkdir(exist_ok=True)
        # A secondary index is a DERIVED sidecar, not table data: building it
        # does NOT advance the log (that would be a version with identical
        # data). The artifact is keyed to the version it was built for; a
        # reader accepts it only while that version's FILE SET still matches
        # the current one — appends invalidate it, so rebuild after appends.
        path = ixdir / f"{column}.v{st.version}.bpt"
        tmp = ixdir / f".{column}.tmp"
        tmp.write_bytes(img)
        fsync_file(tmp)
        tmp.replace(path)
        return {"column": column, "keys": int(sum(len(k) for k in keys)),
                "bytes": len(img), "version": st.version}

    def _open_index(self, column: str, st):
        from . import bptree
        ixdir = self.dir / "_index"
        if not ixdir.is_dir():
            return None
        current = {f.path for f in st.files}
        for p in sorted(ixdir.glob(f"{column}.v*.bpt"), reverse=True):
            v = int(p.stem.split(".v")[-1])
            if {f.path for f in self.state(v).files} == current:
                return bptree.Reader.open(p), p.stat().st_size
        return None

    def where(self, column: str, op: str, value, value2=None, columns=None,
              stats: QueryStats | None = None) -> pa.Table:
        """Predicate pushdown on a NON-time column. With a B+ index: descend,
        map hits to (file, row group), read ONLY those row groups, then take
        the exact rows. Without one: zone-map scan + filter, honestly counted
        as such. ops: ==, >=, <=, between."""
        from . import bptree
        stats = stats if stats is not None else QueryStats()
        st = self.state()
        stats.files_total += len(st.files)
        stats.corpus_bytes += st.bytes
        NEG_INF, POS_INF = -(1 << 63), (1 << 63) - 1  # full i64 range:
        # float encodings legitimately occupy the far ends of int64
        lo, hi = {"==": (value, value),
                  ">=": (value, None), "<=": (None, value),
                  "between": (value, value2)}[op]
        ix = self._open_index(column, st)
        if ix is None:
            # fallback: full scan with an honest bill
            t = self.scan(stats=stats)
            arr = t.column(column)
            m = None
            if lo is not None:
                m = pc.greater_equal(arr, lo)
            if hi is not None:
                c = pc.less_equal(arr, hi)
                m = c if m is None else pc.and_(m, c)
            out = t.filter(m)
            stats.rows_returned += len(out)
            return out
        reader, ix_bytes = ix
        stats.bytes_touched += ix_bytes  # the index read is real I/O too
        locs = reader.range(
            bptree.encode_scalar(lo) if lo is not None else NEG_INF,
            bptree.encode_scalar(hi) if hi is not None else POS_INF)
        locs = np.sort(np.asarray(locs, dtype=np.uint64))
        file_ids = (locs >> np.uint64(40)).astype(np.int64)
        rows = (locs & np.uint64((1 << 40) - 1)).astype(np.int64)
        parts = []
        want_cols = None if columns is None else \
            (["ts", *columns] if "ts" not in columns else list(columns))
        for fi in np.unique(file_ids):
            fmask = file_ids == fi
            frows = rows[fmask]
            pf = _pf(self.dir / st.files[fi].path)
            md = pf.metadata
            # metadata-driven row->group mapping: group sizes are whatever
            # the writer chose (now width-adaptive), so boundaries come
            # from the footer, never from an assumed constant
            rg_start = np.cumsum(
                [0] + [md.row_group(g).num_rows
                       for g in range(md.num_row_groups)])
            g_of = np.searchsorted(rg_start, frows, side="right") - 1
            groups = np.unique(g_of).tolist()
            for g in groups:
                for c in range(md.row_group(g).num_columns):
                    if want_cols is None or _top(md, c) in want_cols:
                        stats.bytes_touched += \
                            md.row_group(g).column(c).total_compressed_size
            tbl = pf.read_row_groups(groups, columns=want_cols)
            # map absolute rows -> positions inside the concatenated groups
            base = np.cumsum([0] + [md.row_group(g).num_rows for g in groups])
            gsel = {g: k for k, g in enumerate(groups)}
            gpos = np.array([gsel[g] for g in g_of])
            local = frows - rg_start[g_of] + base[gpos]
            parts.append(tbl.take(pa.array(np.sort(local))))
            stats.files_touched += 1
        out = pa.concat_tables(parts) if parts else \
            self.scan(0, -1, columns=columns)  # empty, right schema
        stats.rows_returned += len(out)
        return out

    def files(self, version=None) -> list[str]:
        """Absolute paths of this snapshot's Parquet files — the open-format
        contract: hand these to ANY engine (DuckDB, Spark, Polars, a
        distributed dataframe library) and it reads the table with zero
        export and zero ElideDB code."""
        return [str(self.dir / f.path) for f in self.state(version).files]

    # ---- read path --------------------------------------------------------
    def scan_values(self, column: str, values, columns=None,
                    version=None, stats: QueryStats | None = None,
                    t0=None, t1=None):
        """VALUE PREDICATE PUSHED INTO THE READER, not applied after it.

        Reading the whole table and calling .filter() afterwards is what
        makes a store behave like a filesystem: correct answer, no
        pruning. Measured on the labels table before this existed - a
        lookup for one name touched 502 KB of 711 KB, 71% of the table,
        to return 26 episodes.

        Here the predicate goes to the Parquet reader, so row groups
        whose min/max for `column` cannot contain any requested value are
        never decompressed. It works because build_index sorts labels by
        (kind, value) and gives each value its own row group - sorted
        layout is what turns statistics into an index.

        bytes_touched is charged the same way: footer plus only the row
        groups whose statistics overlap the requested values.

        `t0`/`t1` add a SECOND predicate, on ts, pushed the same way. One
        value is not always a small read: an object's frame index is one
        rec_id and tens of thousands of rows, so a value predicate alone
        makes a four-second window pay for the whole recording. With the
        range, the same window pays for the row groups whose ts statistics
        reach it. Both layers are exact -- a group is dropped only when its
        own min/max prove it holds nothing asked for -- so the rows that
        come back are a superset of the window and never a truncation of
        it, which is what lets the caller filter afterwards and still be
        right.
        """
        st = self.state(version)
        stats = stats if stats is not None else QueryStats()
        stats.files_total += len(st.files)
        stats.corpus_bytes += st.bytes
        # Values keep their OWN type. Stringifying them first, as this
        # did, silently breaks numeric columns: "51" < "7" lexically, so
        # a lookup for object 7 would drop the group holding it. A
        # comparison against Parquet statistics has to be in the
        # column's order, not in string order.
        want = sorted(set(values))
        if columns is not None:
            columns = list(dict.fromkeys([*columns, column, "ts"]))
        lo, hi = want[0], want[-1]
        parts = []
        for f in st.files:
            # LAYER 0: the commit log's zone map, already in memory. A
            # file that provably holds nothing in [lo, hi] is dropped
            # here, before its footer is even opened. Same for the time
            # range, which the log carries for every file as its own pair.
            if not f.may_contain(column, lo, hi):
                continue
            if (t0 is not None or t1 is not None) and not f.overlaps(t0, t1):
                continue
            p = self.dir / f.path
            pf = _pf(p)
            md = pf.metadata
            ci = _col_index(md, column)
            if ci is None:
                continue
            ti = _col_index(md, "ts") if (t0 is not None or t1 is not None) else None
            groups, touched = [], 0
            for g in range(md.num_row_groups):
                rg = md.row_group(g)
                s = rg.column(ci).statistics
                # LAYER 1: row-group statistics. min/max is a RANGE
                # test, so a group is skipped only when every requested
                # value falls outside [min, max].
                if s is not None and s.min is not None:
                    try:
                        if hi < s.min or lo > s.max:
                            continue
                    except TypeError:
                        pass          # mixed types: cannot prove empty
                if ti is not None:
                    ts = rg.column(ti).statistics
                    if ts is not None and ts.min is not None:
                        if (t1 is not None and ts.min > t1) or (t0 is not None and ts.max < t0):
                            continue
                groups.append(g)
                for c in range(rg.num_columns):
                    if columns is None or _top(md, c) in columns:
                        touched += rg.column(c).total_compressed_size
            stats.files_touched += 1
            stats.bytes_touched += md.serialized_size + touched
            if not groups:
                continue
            tb = pf.read_row_groups(groups, columns=columns)
            parts.append(tb.filter(pc.field(column).isin(want)))
        if not parts:
            return pa.table({"ts": pa.array([], pa.int64())}), stats
        out = pa.concat_tables(parts, promote_options="permissive")
        stats.rows_returned += len(out)
        return out, stats

    def scan(self, t0=None, t1=None, columns=None, version=None,
             stats: QueryStats | None = None) -> pa.Table:
        st = self.state(version)
        stats = stats if stats is not None else QueryStats()
        stats.files_total += len(st.files)
        stats.corpus_bytes += st.bytes
        if columns is not None and "ts" not in columns:
            columns = ["ts", *columns]  # ts always rides along: it is the
                                        # sort key and the alignment axis
        # layer 1: log-level file pruning (zone maps in the commit entries)
        # INTERVAL overlap, not start-point containment. See
        # FileEntry.overlaps: comparing t0 against max(ts) drops any row
        # whose interval began before the window and had not ended.
        files = [f for f in st.files if f.overlaps(t0, t1)]
        stats.files_touched += len(files)
        # layer 2 accounting: row-group zone maps from the Parquet footer.
        # A surviving file is charged its footer + only the row groups whose
        # ts min/max overlap the window — which is exactly what the reader
        # below will materialize. Same math as warehouse skip-indexes.
        for f in files:
            pf = _pf(self.dir / f.path)
            md = pf.metadata
            footer_bytes = md.serialized_size
            ts_idx = _col_index(md, "ts") or 0
            # SAME INTERVAL FIX, one layer down. A row group is skipped
            # only if no interval in it can reach the window: compare t0
            # against max(t1) where the table has a t1, and against
            # max(ts) only when it does not.
            e_idx = _col_index(md, "t1")
            touched = 0
            for rg in range(md.num_row_groups):
                g = md.row_group(rg)
                st_ts = g.column(ts_idx).statistics
                st_e = (g.column(e_idx).statistics
                        if e_idx is not None else st_ts)
                end = (st_e.max if st_e is not None and st_e.max is not None
                       else (st_ts.max if st_ts is not None else None))
                if end is not None and t0 is not None and end < t0:
                    continue
                if st_ts is not None and t1 is not None and st_ts.min > t1:
                    continue
                for c in range(g.num_columns):
                    col = g.column(c)
                    if (columns is None or _top(md, c) in columns
                            or c == ts_idx):
                        touched += col.total_compressed_size
            stats.bytes_touched += footer_bytes + touched
        if not files:
            empty = pa.schema([("ts", pa.int64())])
            return pa.table({"ts": pa.array([], pa.int64())}).cast(empty)
        # layer 2: Parquet row-group pruning + projection pushdown
        # INTERVAL OVERLAP, not start containment. `ts >= t0` asks
        # "did it START inside the window", which is a different
        # question and drops every row still in progress when the window
        # opens. Overlap is (row.t1 >= t0) AND (row.ts <= t1); with no
        # t1 column a row is a point and the two coincide.
        has_end = "t1" in (pq.ParquetFile(
            _uncached(self.dir / files[0].path) if _NOCACHE
            else str(self.dir / files[0].path)).schema_arrow.names)
        end_f = pc.field("t1") if has_end else pc.field("ts")
        filt = None
        if t0 is not None:
            filt = end_f >= t0
        if t1 is not None:
            c = pc.field("ts") <= t1
            filt = c if filt is None else filt & c
        if has_end and columns is not None and "t1" not in columns:
            columns = [*columns, "t1"]      # the predicate needs it
        parts = [_read(self.dir / f.path, columns=columns,
                               filters=filt) for f in files]
        out = pa.concat_tables(parts, promote_options="permissive")
        if len(parts) > 1:  # files may interleave in time across streams
            out = out.take(pc.sort_indices(out.column("ts")))
        stats.rows_returned += len(out)
        return out


class Store:
    FORMAT = "elidedb/2"

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        meta_path = self.dir / "_store.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{path} is not a store (no _store.json) — Store.create() it")
        self.meta = json.loads(meta_path.read_text())

    # ---- lifecycle --------------------------------------------------------
    @staticmethod
    def create(path: str | Path, name: str) -> "Store":
        p = Path(path)
        (p / "tables").mkdir(parents=True, exist_ok=True)
        meta = {"format": Store.FORMAT, "name": name,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (p / "_store.json").write_text(json.dumps(meta, indent=1))
        return Store(p)

    @staticmethod
    def open(path: str | Path) -> "Store":
        return Store(path)

    @property
    def name(self):
        return self.meta["name"]

    def snapshot(self) -> dict:
        """Pin every table's current version in one call. Pass the result as
        `version=` to window/aligned/sql/scan for a CONSISTENT multi-table
        read: no writer that commits after this call can skew your view.
        (Per-table commits are serializable on their own log — the same
        isolation model as Delta and Iceberg; the pin extends it across
        tables for readers.)"""
        return {name: self.table(name).state().version
                for name in self.tables()}

    @staticmethod
    def _ver(version, name):
        if isinstance(version, dict):
            return version.get(name)
        return version

    def vacuum(self, retain_versions: int = 3, dry_run: bool = False) -> dict:
        """Garbage-collect files no snapshot within the retention window can
        reach: parquet parts removed by compaction/delete, and managed media
        no retained frame-index version references. Time travel remains
        intact for the last `retain_versions` versions of every table;
        earlier versions become unreadable — that is the explicit trade this
        command makes, and why it is manual."""
        freed = 0
        removed = []
        media_refs: set[str] = set()
        for name in self.tables():
            tab = self.table(name)
            versions = tab.log.versions()
            keep_versions = versions[-retain_versions:] if versions else []
            referenced = set()
            for v in keep_versions:
                st = tab.state(v)
                referenced |= {f.path for f in st.files}
                if st.kind == "frame_index":
                    tbl = tab.scan(version=v)
                    if "source" in tbl.column_names:
                        media_refs |= {s for s in
                                       set(tbl.column("source").to_pylist())
                                       if s.startswith("@")}
            for f in tab.dir.glob("part-*.parquet"):
                if f.name not in referenced:
                    freed += f.stat().st_size
                    removed.append(str(f.relative_to(self.dir)))
                    if not dry_run:
                        f.unlink()
        media = self.dir / "media"
        if media.is_dir():
            for m in media.iterdir():
                if f"@media/{m.name}" not in media_refs:
                    freed += m.stat().st_size
                    removed.append(f"media/{m.name}")
                    if not dry_run:
                        m.unlink()
        return {"files_removed": len(removed), "bytes_freed": freed,
                "dry_run": dry_run, "removed": removed[:10]}

    def tables(self) -> list[str]:
        root = self.dir / "tables"
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if (p / "_log").is_dir())

    def table(self, name: str) -> Table:
        return Table(self, name)

    def drop_caches(self) -> dict:
        """Forget everything this store has memoised. Nothing else.

        The point is scope. `sudo purge` empties the whole machine's
        unified buffer cache - every other process's working set with
        it - which makes a benchmark both unrepeatable and rude. What a
        database should be able to say is "drop MY cache", so this
        clears only what ElideDB itself holds: the derived matrices
        memoised per (store, version) in context.py and cracked.py.

        File pages are handled separately and earlier: store reads open
        with F_NOCACHE, so the kernel is never asked to retain them in
        the first place. Nothing to evict beats evicting.

        Loaded model weights are NOT dropped - they are not a cache of
        the data, they are the program.
        """
        out = {}
        for mod, names in ((".context", ("_CTX_CACHE", "_SPACE_CACHE")),
                           (".cracked", ("_CACHE",))):
            try:
                m = importlib.import_module(mod, __package__)
            except Exception:
                continue
            for n in names:
                d = getattr(m, n, None)
                if isinstance(d, dict):
                    out[f"{mod.lstrip('.')}.{n}"] = len(d)
                    d.clear()
        out["file_pages"] = "not cached (F_NOCACHE)" if _NOCACHE else "OS"
        return out

    @contextlib.contextmanager
    def measure(self):
        """Charge every read inside this block to one QueryStats.

        `scan` and `scan_values` already account honestly, but they
        account PER CALL, and a retrieval query fans out across a dozen
        channel tables through code that never threads a stats object
        through. Threading one through every channel would touch every
        caller for a number none of them care about; wrapping the two
        entry points for the duration of a block gets the same figure
        with the accounting living in exactly one place.

            with db.measure() as st:
                search_set(db, "open the drawer")
            st.bytes_touched, st.elided_pct

        corpus_bytes is the whole store, not the sum of the tables that
        happened to be touched - the elision claim is against everything
        that could have been read, or it means nothing.
        """
        stats = QueryStats()
        stats.corpus_bytes = sum(self.table(t).state().bytes
                                 for t in self.tables())
        scan, values = Table.scan, Table.scan_values

        def scan_m(self_, *a, stats=None, **kw):
            local = QueryStats()
            out = scan(self_, *a, stats=local, **kw)
            _fold(stats or _NULL, local)
            _fold(stats_outer, local)
            return out

        def values_m(self_, *a, stats=None, **kw):
            local = QueryStats()
            out = values(self_, *a, stats=local, **kw)
            _fold(stats or _NULL, local)
            _fold(stats_outer, local)
            return out

        stats_outer = stats
        Table.scan, Table.scan_values = scan_m, values_m
        try:
            yield stats
        finally:
            Table.scan, Table.scan_values = scan, values


    def describe(self) -> list[dict]:
        out = []
        for name in self.tables():
            st = self.table(name).state()
            out.append({"table": name, "kind": st.kind, "version": st.version,
                        "rows": st.rows, "bytes": st.bytes,
                        "min_ts": st.min_ts, "max_ts": st.max_ts,
                        "files": len(st.files), "meta": st.meta})
        return out

    # ---- friendly ingest --------------------------------------------------
    def ingest_rows(self, table: str, data, ts_column="ts", ts_unit="auto",
                    meta=None, evolve=False) -> int:
        """Append rows from a pandas DataFrame / dict of arrays / pyarrow
        Table / CSV / Parquet path.

        Friendly on purpose: `ts_column` may be a datetime column, an ISO-8601
        string column, or epoch numbers in s/ms/us/ns — `ts_unit="auto"`
        detects the epoch unit by magnitude (an explicit unit always wins)."""
        import os as _os

        import pandas as pd
        if isinstance(data, (str, Path)):
            p = str(data)
            if p.endswith((".parquet", ".pq")):
                data = pd.read_parquet(p)
            elif _os.path.getsize(p) > 128 * 1024 * 1024:
                # memory-bounded load: stream the CSV in chunks, one file per
                # chunk, ONE atomic commit for the whole load
                def gen():
                    for chunk in pd.read_csv(p, chunksize=2_000_000):
                        yield self._normalize_ts(chunk, ts_column, ts_unit)
                return self.table(table).append_batches(gen(), meta=meta)
            else:
                data = pd.read_csv(p)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        if isinstance(data, pd.DataFrame):
            data = self._normalize_ts(data, ts_column, ts_unit)
        return self.table(table).append(data, meta=meta, evolve=evolve)

    @staticmethod
    def _normalize_ts(df, ts_column, ts_unit) -> pa.Table:
        import pandas as pd
        if ts_column not in df.columns:
            raise ValueError(
                f"no column '{ts_column}' — available: "
                f"{list(df.columns)} (pass ts_column=...)")
        df = df.rename(columns={ts_column: "ts"}).copy()
        col = df["ts"]
        if pd.api.types.is_datetime64_any_dtype(col):
            df["ts"] = col.astype("int64")  # datetime64 is already ns
        elif col.dtype == object or pd.api.types.is_string_dtype(col):
            df["ts"] = pd.to_datetime(col).astype("int64")  # ISO strings
        else:
            if ts_unit == "auto":
                # epoch magnitude: seconds ~1e9, ms ~1e12, us ~1e15, ns ~1e18
                m = float(pd.Series(col).abs().median())
                ts_unit = ("s" if m < 1e11 else "ms" if m < 1e14
                           else "us" if m < 1e17 else "ns")
            mult = {"ns": 1, "us": 1_000, "ms": 1_000_000,
                    "s": 1_000_000_000}[ts_unit]
            df["ts"] = (col.astype("float64") * mult).round().astype("int64")
        return pa.Table.from_pandas(df, preserve_index=False)

    def _media_dest(self, src: Path) -> Path:
        import hashlib
        h = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:8]
        media = self.dir / "media"
        media.mkdir(exist_ok=True)
        return media / f"{src.stem}-{h}{src.suffix}"

    def ingest_video(self, table: str, video_path, timestamps_ns=None,
                     stream=None, meta=None, copy=True, transcode=None,
                     gop_s: float = 1.0, crf: int = 26) -> int:
        """Index a video file: packet scan → frame_index Parquet rows.

        copy=True (default): the media file is copied into the store's
        `media/` directory first, so the store directory IS the complete,
        portable database. copy=False indexes the file in place.

        transcode="hevc"|"h264": re-encode the managed copy as a compressed
        elementary stream (≈10-25x smaller than MJPEG at like quality) with a
        forced keyframe every `gop_s` seconds. Random access becomes
        GOP-granular instead of frame-exact — `gop_s` IS the seekability-vs-
        compression dial, chosen per table at ingest, and decode reads
        exactly one GOP span per window."""
        import shutil
        import subprocess

        from .fftools import find
        from .video import scan_video_packets
        src = Path(video_path)
        if transcode:
            assert transcode in ("hevc", "h264")
            if timestamps_ns is None:  # take pts from the source container
                probe = scan_video_packets(src)
                timestamps_ns = probe["ts"].to_pylist()
            n_in = len(timestamps_ns)
            span_s = max((timestamps_ns[-1] - timestamps_ns[0]) / 1e9, 0.1)
            fps = max((n_in - 1) / span_s, 1.0)
            g = max(1, round(gop_s * fps))
            dest = self._media_dest(src).with_suffix(f".{transcode}")
            if not dest.exists():
                enc = "libx265" if transcode == "hevc" else "libx264"
                subprocess.run(
                    [find("ffmpeg"), "-v", "error", "-y", "-i", str(src),
                     "-c:v", enc, "-preset", "fast", "-crf", str(crf),
                     "-g", str(g), "-keyint_min", str(g), "-an",
                     "-f", transcode, str(dest)], check=True)
            scanned_path, source_ref = dest, f"@media/{dest.name}"
        elif copy:
            dest = self._media_dest(src)
            if not dest.exists():
                shutil.copy2(src, dest)
            scanned_path, source_ref = dest, f"@media/{dest.name}"
        else:
            scanned_path = src
            source_ref = str(src.resolve())
        rows = scan_video_packets(scanned_path, timestamps_ns)
        n = len(rows["ts"])
        rows["source"] = pa.array([source_ref] * n)
        rows["stream"] = [stream or src.stem] * n
        t = pa.table(rows)
        return self.table(table).append(
            t, kind="frame_index",
            meta={"source": source_ref, "original": str(src.resolve()),
                  **({"transcode": transcode, "gop_s": gop_s, "crf": crf}
                     if transcode else {}),
                  **(meta or {})})

    def adopt_media(self, table: str = "frames", verbose=True) -> dict:
        """Make the store standalone: copy every externally-referenced media
        file into `media/` and rewrite the frame index to store-relative
        paths. One replace-commit per call — old index versions still resolve
        (the external files are not deleted)."""
        import shutil
        import uuid as _uuid
        from .log import FileEntry
        tab = self.table(table)
        st = tab.state()
        if st.kind != "frame_index":
            raise ValueError(f"{table} is not a frame_index table")
        t = tab.scan()
        srcs = t.column("source").to_pylist()
        external = sorted({s for s in srcs if not s.startswith("@")})
        if not external:
            return {"adopted": 0, "bytes": 0}
        mapping, copied = {}, 0
        for s in external:
            p = Path(s)
            if not p.exists():
                raise FileNotFoundError(f"referenced media missing: {s}")
            dest = self._media_dest(p)
            if not dest.exists():
                shutil.copy2(p, dest)
            copied += dest.stat().st_size
            mapping[s] = f"@media/{dest.name}"
            if verbose:
                print(f"  adopted {p.name} -> media/{dest.name}")
        new_src = pa.array([mapping.get(s, s) for s in srcs])
        t = t.set_column(t.column_names.index("source"), "source", new_src)
        fname = f"part-{_uuid.uuid4().hex[:12]}.parquet"
        path = self.dir / "tables" / table / fname
        write_parquet(t, path)
        tsv = t.column("ts").to_numpy()
        tab.log.commit(op="adopt-media", kind="frame_index",
                       schema=str(t.schema),
                       add=[FileEntry(fname, len(t), path.stat().st_size,
                                      int(tsv.min()), int(tsv.max()))],
                       remove=[f.path for f in st.files],
                       meta={"media_files": len(external)})
        return {"adopted": len(external), "bytes": copied}

    # ---- queries ----------------------------------------------------------
    def window(self, t0: int, t1: int, tables=None, columns=None,
               version=None):
        """The multimodal read: every requested table filtered to [t0, t1].
        frame_index tables come back as FrameSet (lazy byte-range decode)."""
        from .video import FrameSet
        stats = QueryStats()
        start = time.perf_counter()
        out = {}
        names = tables
        if names is None:
            # Default to the DATA tables. Index artifacts (embeddings,
            # centroids, frame_vectors, context, ...) are timestamped too, so
            # they would otherwise be dragged into every window read and drag
            # thousands of 1152-d vectors with them. Ask for them by name and
            # you still get them.
            names = [n for n in self.tables()
                     if self.table(n).state(self._ver(version, n)).kind
                     not in ("embeddings", "centroids")]
        for name in names:
            tab = self.table(name)
            v = self._ver(version, name)
            st = tab.state(v)
            cols = columns.get(name) if isinstance(columns, dict) else columns
            data = tab.scan(t0, t1, columns=cols, version=v, stats=stats)
            out[name] = FrameSet(self, name, data) if st.kind == "frame_index" \
                else data
        stats.wall_ms = (time.perf_counter() - start) * 1e3
        return out, stats

    def aligned(self, t0, t1, rate_hz, tables=None, interp="nearest",
                version=None, edge_guard_s: float = 1.0):
        """Query-time alignment: resample numeric columns of the requested
        timeseries tables onto one [t0, t1] timeline at rate_hz."""
        timeline = np.arange(t0, t1 + 1, int(1e9 / rate_hz), dtype=np.int64)
        guard = int(edge_guard_s * 1e9)  # neighbors just outside the window
                                         # make edge interpolation exact
        out = {"timeline_ns": timeline}
        stats = QueryStats()
        for name in (tables or self.tables()):
            tab = self.table(name)
            v = self._ver(version, name)
            if tab.state(v).kind != "timeseries":
                continue
            data = tab.scan(t0 - guard, t1 + guard, version=v, stats=stats)
            if len(data) == 0:
                continue
            ts = data.column("ts").to_numpy()
            cols = {}
            for cname in data.column_names:
                if cname == "ts":
                    continue
                arr = data.column(cname)
                if not pa.types.is_floating(arr.type) and \
                   not pa.types.is_integer(arr.type):
                    continue
                v = arr.to_numpy().astype(np.float64)
                if interp == "linear":
                    cols[cname] = np.interp(timeline, ts, v)
                else:  # nearest
                    idx = np.searchsorted(ts, timeline)
                    idx = np.clip(idx, 0, len(ts) - 1)
                    prev = np.clip(idx - 1, 0, len(ts) - 1)
                    use_prev = (timeline - ts[prev]) <= (ts[idx] - timeline)
                    cols[cname] = v[np.where(use_prev, prev, idx)]
            out[name] = cols
        return out, stats

    def sql(self, query: str, version=None):
        """DuckDB over the store's own Parquet files — the lakehouse dividend:
        because the format is open, a whole second engine comes for free.
        Table names in the query = store table names."""
        import duckdb
        con = duckdb.connect()
        for name in self.tables():
            st = self.table(name).state(self._ver(version, name))
            files = [str(self.dir / "tables" / name / f.path) for f in st.files]
            if files:
                quoted = ", ".join(f"'{f}'" for f in files)
                con.execute(
                    f'CREATE VIEW "{name}" AS SELECT * FROM '
                    f"read_parquet([{quoted}])")
        return con.execute(query).fetchdf()

    # ---- semantic layer (see embeddings.py) --------------------------------
    def embed_windows(self, frame_table="frames", window_s=2.0,
                      frames_per_window=2, model=None, batch=16):
        from .embeddings import embed_windows
        return embed_windows(self, frame_table, window_s, frames_per_window,
                             model, batch)

    def search(self, text: str, k=10, nprobe=3, merge=True, t0=None, t1=None,
               streams=None, method="auto", neg_weight=0.5, min_score=None,
               percentile=None, rerank=False, rerank_top=12,
               rerank_alpha=0.7):
        """Compositional text search. `text` supports AND / NOT / -term;
        `min_score`/`percentile` add a precision floor. See embeddings.search."""
        from .embeddings import search
        return search(self, text, k=k, nprobe=nprobe, merge=merge, t0=t0,
                      t1=t1, streams=streams, method=method,
                      neg_weight=neg_weight, min_score=min_score,
                      percentile=percentile, rerank=rerank,
                      rerank_top=rerank_top, rerank_alpha=rerank_alpha)

    def search_text(self, text: str, k=10, nprobe=3, **kw):
        from .embeddings import search_text
        return search_text(self, text, k=k, nprobe=nprobe, **kw)

    def search_clip(self, stream: str, t0: int, t1: int, k=10, nprobe=3, **kw):
        """Query-by-example. When the store carries a V-JEPA clip index
        the neighbor space is the WORLD MODEL's (video-native, motion-
        structured, no text anywhere); otherwise appearance windows.
        Measured (bridge4h): V-JEPA beats appearance on action-class
        neighbor purity for 'open' (0.51 vs 0.40) and ties elsewhere."""
        try:
            from .embeddings import _vec_table
            import numpy as np
            tbl, vecs = _vec_table(self, "vjepa_vectors")
            ss = tbl.column("stream").to_pylist()
            sa = [int(v) for v in tbl.column("ts").to_pylist()]
            sb = [int(v) for v in tbl.column("t1").to_pylist()]
            mid = (t0 + t1) // 2
            qi = next((i for i in range(len(ss))
                       if ss[i] == stream and sa[i] <= mid <= sb[i]), None)
            if qi is not None:
                sc = vecs @ np.asarray(vecs[qi])
                sc[qi] = -9
                order = np.argsort(-sc)[:k]
                hits = [{"stream": ss[i], "t0": sa[i], "t1": sb[i],
                         "score": float(sc[i])} for i in order]
                return hits, {"method": "vjepa-qbe", "k": k}
        except Exception:
            pass
        from .embeddings import search_clip
        return search_clip(self, stream, t0, t1, k=k, nprobe=nprobe, **kw)

    # ---- context retrieval -------------------------------------------------
    def index_context(self, window_s=2.0, stride_s=0.5, label_fraction=1.0,
                      prune=True, epochs=300, verbose=True, model=None,
                      frame_stride=1, prompt="scene"):
        """Build the context index end to end.

        frames -> per-frame vectors -> VLM captions on `label_fraction` of
        windows -> caption-LSA space -> temporal tower -> cellular turnover
        -> materialised `context` table.

        Cost is dominated by the image encoder, so the knobs that matter are:

          model="fast"      3.3x faster encoder, same 1152-d space
          frame_stride=N    embed every Nth frame (5 Hz video rarely needs all)
          label_fraction<1  caption only part of the corpus; the tower covers
                            the rest
          prompt=           "scene" or "manipulation" — the caption IS the
                            index, so it has to use the words a user would

        Measured: decode 2.7 ms/frame, encode 90.3 ms/frame (quality) or
        27.7 ms/frame (fast). Embedding a large corpus is a batch job measured
        in hours; nothing here hides that.
        """
        from . import context as C
        out = {"frame_vectors": C.embed_frames(self, verbose=verbose,
                                               model=model,
                                               stride=frame_stride)}
        windows = C.plan_windows(self, window_s, stride_s)
        if label_fraction < 1.0:
            # Label a TIME PREFIX, not a random sample: the realistic shape of
            # this problem is "we captioned what we had, then more footage
            # arrived", and a random sample would quietly hand the tower
            # neighbours of every held-out window.
            n = max(int(len(windows) * label_fraction), 16)
            windows = sorted(windows, key=lambda w: w[1])[:n]
        out["captions"] = C.caption_windows(self, windows, verbose=verbose,
                                            prompt=prompt)
        _, _, out["train"] = C.train_context(self, window_s, stride_s,
                                             epochs=epochs, verbose=verbose)
        if prune:
            _, rec = C.prune_context(self, verbose=verbose)
            out["prune"] = rec.get("selected")
        out["build"] = C.build_context(self, verbose=verbose)
        return out

    def search_context(self, text: str, k=8, pool=48, deep=0, t0=None,
                       t1=None, streams=None, rerank=False, verify="async",
                       **_legacy):
        """THE search: any query, action or not, on any store.

        Union recall over every tier the store has (appearance embeddings,
        caption words, caption-LSA vectors) proposes candidates; a VLM
        reading each clip's start/end frames verifies WHAT IS HAPPENING;
        verdicts are cached into the store so hot queries get cheap.
        `deep=N` (or rerank=True) re-judges the top N with the larger VLM
        over 4 ordered frames. Legacy RRF-only search remains at
        elidedb.context.search for stores where a model-free path matters."""
        from .verified import search_verified
        if rerank and not deep:
            deep = 6
        if deep and verify == "async":
            verify = "sync"          # deep judging is an explicit wait
        return search_verified(self, text, k=k, pool=pool, deep=deep,
                               t0=t0, t1=t1, streams=streams, verify=verify)

    def search_verified(self, text: str, k=8, pool=48, deep=0):
        """Any query, action or not: union recall proposes, a VLM shown
        frames IN TIME ORDER disposes, verdicts are cached into the store.
        The only path that can enforce 'the green object is the one being
        moved' or '...and close it'. See elidedb.verified."""
        from .verified import search_verified
        return search_verified(self, text, k=k, pool=pool, deep=deep)

    def search_sharp(self, text: str, k=10, shortlist=48):
        """Text search at teacher quality, student price: the student ranks
        every window (~1 ms), the teacher re-scores only the shortlist, and
        every teacher vector is cached into the store — quality accumulates
        where users query (database cracking). See elidedb.cracked."""
        from .cracked import search_sharp
        return search_sharp(self, text, k=k, shortlist=shortlist)

    def explain(self, t0: int, t1: int, stream=None):
        """The teacher's own description of what happens in a window."""
        from .context import explain
        return explain(self, t0, t1, stream=stream)


_NULL = None


def _fold(dst, src):
    """Accumulate one QueryStats into another.

    corpus_bytes takes a MAX, not a sum: it is a denominator, and adding
    denominators across calls would inflate it until the elision figure
    became meaningless. The measure() block seeds it with the whole
    store, which is the largest and the correct one; a caller's own
    stats keeps whatever per-table denominator it had.
    """
    if dst is None:
        return
    dst.files_total += src.files_total
    dst.files_touched += src.files_touched
    dst.bytes_touched += src.bytes_touched
    dst.rows_returned += src.rows_returned
    dst.corpus_bytes = max(dst.corpus_bytes, src.corpus_bytes)
