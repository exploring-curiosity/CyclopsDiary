"""BPT1 — immutable, bulk-loaded B+ tree (numpy twin of the C++20 reader in
src/streetdex/index/bptree.{hpp,cpp}; identical bytes on disk).

Why a B+ tree here at all: zone maps prune brilliantly on `ts` because files
are time-sorted, and prune *nothing* on an unsorted column (every row group
spans nearly the whole domain). A secondary index re-sorts (key → location)
once, so point/range predicates on ANY numeric column become a descent plus
a contiguous leaf scan instead of a full-table read. Immutability makes the
classic hard parts vanish: bulk load at 100% fill, no splits, no
rebalancing, rebuilt per table version like every other derived artifact.

Values are opaque u64s; the store packs (file_idx << 40) | row_in_file, so a
leaf hit maps straight to one Parquet row group — the index prunes I/O, not
just rows.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"BPT1"
DEFAULT_ORDER = 256  # 256 × 8B keys = one 4 KiB page per node touch


def encode_key(values: np.ndarray) -> np.ndarray:
    """Order-preserving i64 encoding: ints pass through; IEEE-754 doubles use
    the sign-flip trick (identical to bpt_encode_f64 in C++)."""
    if np.issubdtype(values.dtype, np.integer):
        return values.astype("<i8")
    if np.issubdtype(values.dtype, np.floating):
        u = values.astype("<f8").view("<u8").copy()
        neg = (u & 0x8000000000000000) != 0
        u[neg] = ~u[neg]
        u[~neg] |= 0x8000000000000000
        return (u ^ 0x8000000000000000).view("<i8")
    raise TypeError(f"unindexable dtype {values.dtype}")


def encode_scalar(v) -> int:
    return int(encode_key(np.array([v]))[0])


def build(keys: np.ndarray, values: np.ndarray,
          order: int = DEFAULT_ORDER) -> bytes:
    order_idx = np.argsort(keys, kind="stable")
    k = np.ascontiguousarray(keys[order_idx], dtype="<i8")
    v = np.ascontiguousarray(values[order_idx], dtype="<u8")
    n = len(k)

    levels = []  # bottom-up internal levels: first key of each child node
    below = k
    while len(below) > order:
        firsts = below[::order].copy()
        levels.append(firsts)
        below = firsts
    levels.reverse()  # root first, as the C++ builder writes them

    out = bytearray()
    out += MAGIC
    out += struct.pack("<IQII", order, n, len(levels), 0)
    dir_pos = len(out)
    out += b"\0" * (16 * (len(levels) + 1))
    directory = []
    for lv in levels:
        while len(out) % 8:
            out += b"\0"
        directory.append((len(out), len(lv)))
        out += lv.astype("<i8").tobytes()
    while len(out) % 8:
        out += b"\0"
    directory.append((len(out), n))
    leaf = np.empty(n, dtype=[("k", "<i8"), ("v", "<u8")])
    leaf["k"] = k
    leaf["v"] = v
    out += leaf.tobytes()
    for i, (off, cnt) in enumerate(directory):
        struct.pack_into("<QQ", out, dir_pos + 16 * i, off, cnt)
    return bytes(out)


class Reader:
    def __init__(self, data: bytes | np.memmap):
        buf = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) \
            else data
        if bytes(buf[:4]) != MAGIC:
            raise ValueError("not a BPT1 index")
        self.order, self.n, nlevels, _ = struct.unpack_from(
            "<IQII", buf.tobytes()[:24] if isinstance(buf, np.memmap)
            else data, 4)
        raw = buf.tobytes() if isinstance(buf, np.memmap) else data
        dirs = [struct.unpack_from("<QQ", raw, 24 + 16 * i)
                for i in range(nlevels + 1)]
        self.levels = [np.frombuffer(raw, "<i8", cnt, off)
                       for off, cnt in dirs[:-1]]
        loff, lcnt = dirs[-1]
        leaf = np.frombuffer(raw, dtype=[("k", "<i8"), ("v", "<u8")],
                             count=lcnt, offset=loff)
        self.keys = leaf["k"]
        self.vals = leaf["v"]

    @classmethod
    def open(cls, path: str | Path):
        return cls(np.memmap(path, dtype=np.uint8, mode="r"))

    def lower_bound(self, k: int) -> int:
        node = 0
        for lv in self.levels:
            begin = node * self.order
            end = min(begin + self.order, len(lv))
            # lower_bound + step-back (see C++ twin): duplicates of k may
            # start in the child before the first child whose first-key == k
            pos = int(np.searchsorted(lv[begin:end], k, side="left")) + begin
            node = pos - 1 if pos > begin else begin
        begin = node * self.order
        end = min(begin + self.order, self.n)
        return int(np.searchsorted(self.keys[begin:end], k, side="left")) + begin

    def range(self, lo: int, hi: int) -> np.ndarray:
        a = self.lower_bound(lo)
        b = int(np.searchsorted(self.keys, hi, side="right"))
        return self.vals[a:b]
