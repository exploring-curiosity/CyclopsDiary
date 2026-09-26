"""The event search over a stream store's memory rows.

Everything here is arithmetic over the rows the streaming read wrote
(`streamstore`, read="stream", the MEMORY state per atom) and the change
channel read in the same pass. No model runs, no label enters, no constant
is fitted to a corpus: every scale is the stream's own (its median change,
its mean fused score, the anchor's own length).

Four pieces, in the order a query meets them:

  fused_edges / proposals      where events end, from the stream itself;
                               spans of one to M consecutive events
  SubsequenceSearch            the anchor cuts the answers: the anchor's
                               atom sequence aligned over every recording
                               with a free start; the cost curve's minima
                               are the candidates, their extents from the
                               alignment; re-anchoring; the gated contrast
  rank                         best first, overlaps and adjacent
                               duplicates suppressed
  chain / rank_fusion          a request of several clauses as a chain
                               over the proposals, ordered by the fusion
                               of the mean and the weakest clause fit

Measured (BENCHMARKS 2026-09-21, "The anchor cuts the answers", "The
anchor's neighbours as negatives", "Bridge through the same system").
"""
from __future__ import annotations

import numpy as np

LAGS = (1, 2, 4, 8)          # the memory change pyramid: one atom to eight, each over the stream's median at that lag
NMS_S = 0.5                  # two edges closer than this are one edge
IOU_DUP = 0.5                # two answers overlapping more than this are one answer


# ------------------------------------------------------------------ small helpers
def unit(X):
    X = np.asarray(X, np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return X / n


def iou(a, b) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return float(inter / (max(a[1], b[1]) - min(a[0], b[0])))


def _rank01(v) -> np.ndarray:
    """Percentile rank in (0, 1], the order statistic of a signal inside its own recording."""
    v = np.asarray(v, np.float64)
    return (np.argsort(np.argsort(v)) + 1) / len(v)


def centre_by_recording(X, rec) -> np.ndarray:
    """Rows less their recording's mean, unit again: the room, the camera and the
    lighting are what every row of a recording shares, and are not the event."""
    X = np.asarray(X, np.float64).copy()
    rec = np.asarray(rec)
    for r in np.unique(rec):
        m = rec == r
        X[m] -= X[m].mean(0)
    return unit(X)


# ------------------------------------------------------------------ edges and proposals
def fused_edges(X, spans, change, bounds, atom_s: float) -> dict:
    """Where events end, from the stream itself.

    X: unit memory rows of one stream in order; spans: their (t0, t1);
    change: (t, d) the change channel at frame rate, or None; bounds:
    [(t0, t1, rec_id)] the recordings of the stream. Per atom, two signals:
    the pixel change (the largest frame change inside the atom) and the
    memory change pyramid (1 - cos at lags 1, 2, 4, 8 atoms, each divided
    by the stream's median change at that lag, credited to the lag's
    midpoint atom). Each is ranked inside its recording and the two ranks
    are summed; local maxima of the sum, thinned to one per NMS_S, are the
    edges. The coarse set is the edges above the mean score of the
    stream's own edges: a count the stream sets, not a threshold (2.1x /
    2.0x / 2.4x the true boundaries on the three AgiBot streams, BENCHMARKS
    2026-09-20). The bar is the mean over the EDGES, not over all atoms:
    two percentile ranks average to 1 on every stream, so the all-atom
    mean is a constant in disguise, and it passed 2.8x the true count on
    dagger where the edges' mean gives 2.1x (2026-09-21).

    -> dict(times, scores, coarse (bool), mean, rec): edges in stream order.
    An edge's time is the START of its atom: the change at lag L is
    credited to the atom after it, so the peak atom is the first atom of
    the new event and the boundary lies at its start (the event pass closes
    the event before it).
    """
    X = unit(X)
    starts = np.asarray([a for a, _ in spans], np.float64)
    ends = np.asarray([b for _, b in spans], np.float64)
    n = len(starts)
    if n == 0:
        return dict(times=np.zeros(0), scores=np.zeros(0), coarse=np.zeros(0, bool), mean=0.0, rec=np.zeros(0, object))
    if not bounds:
        bounds = [(float(starts.min()), float(ends.max()), "")]
    rec_of = np.full(n, "", object)
    for lo, hi, rid in bounds:
        rec_of[(starts >= lo - 1e-9) & (starts < hi - 1e-9)] = rid
    # the pixel channel per atom
    pix = np.zeros(n)
    if change is not None and change[0] is not None and len(change[0]):
        ct, cd = np.asarray(change[0], np.float64), np.asarray(change[1], np.float64)
        order = np.argsort(ct)
        ct, cd = ct[order], cd[order]
        lo_i = np.searchsorted(ct, starts, side="left")
        hi_i = np.searchsorted(ct, ends, side="left")
        for i in range(n):
            if hi_i[i] > lo_i[i]:
                pix[i] = cd[lo_i[i]:hi_i[i]].max()
    # the stream's median change at every lag, over all its recordings
    lagmed = {}
    for L in LAGS:
        v = []
        for lo, hi, rid in bounds:
            idx = np.where(rec_of == rid)[0]
            if len(idx) > L:
                M = X[idx]
                v.append(1.0 - np.einsum("ij,ij->i", M[L:], M[:-L]))
        lagmed[L] = float(np.median(np.concatenate(v))) if v else 1.0
        if not np.isfinite(lagmed[L]) or lagmed[L] <= 0:
            lagmed[L] = 1.0
    fused = np.zeros(n)
    for lo, hi, rid in bounds:
        idx = np.where(rec_of == rid)[0]
        if len(idx) < 3:
            continue
        M = X[idx]
        pyr = np.zeros(len(idx))
        for L in LAGS:
            if len(M) > L:
                v = (1.0 - np.einsum("ij,ij->i", M[L:], M[:-L])) / lagmed[L]
                mid = np.arange(L, len(M)) - (L - 1) // 2 if L > 1 else np.arange(1, len(M))
                np.add.at(pyr, np.clip(mid, 0, len(M) - 1), v)
        f = _rank01(pyr)
        if change is not None and change[0] is not None and len(change[0]):
            f = f + _rank01(pix[idx])
        fused[idx] = f
    # local maxima, one per NMS_S, each at its atom's start
    cand = []
    for lo, hi, rid in bounds:
        idx = np.where(rec_of == rid)[0]
        for k in range(1, len(idx) - 1):
            i = idx[k]
            if fused[i] >= fused[idx[k - 1]] and fused[i] > fused[idx[k + 1]]:
                cand.append((fused[i], starts[i], rid))
    cand.sort(key=lambda c: -c[0])
    kept = []
    for sc, t, rid in cand:
        if all(abs(t - u) > NMS_S or r != rid for _, u, r in kept):
            kept.append((sc, t, rid))
    kept.sort(key=lambda c: c[1])
    times = np.array([t for _, t, _ in kept], np.float64)
    scores = np.array([s for s, _, _ in kept], np.float64)
    recs = np.array([r for _, _, r in kept], object)
    mean = float(scores.mean()) if len(scores) else 0.0          # the stream's scale: its own edges
    return dict(times=times, scores=scores, coarse=scores > mean, mean=mean, rec=recs)


def proposals(edge_times, t0: float, t1: float, max_ev: int = 3) -> list:
    """Spans of one to `max_ev` consecutive events of one recording, the
    recording's own start and end counted as edges. -> [(t0, t1, n_ev)],
    in the order of their start and then their length."""
    cuts = [float(t0)] + [float(t) for t in np.sort(np.asarray(edge_times, np.float64)) if t0 < t < t1] + [float(t1)]
    out = []
    for i in range(len(cuts) - 1):
        for m in range(1, max_ev + 1):
            if i + m < len(cuts):
                out.append((cuts[i], cuts[i + m], m))
    return out


def predecessors(props) -> list:
    """For every proposal, the proposals that end where it starts."""
    by_end = {}
    for i, (a, b, _) in enumerate(props):
        by_end.setdefault(round(b, 6), []).append(i)
    return [by_end.get(round(a, 6), []) for a, b, _ in props]


def inside(props) -> list:
    """For every proposal, the other proposals it contains."""
    out = []
    for i, (a, b, _) in enumerate(props):
        out.append([j for j, (c, d, _) in enumerate(props) if j != i and c >= a - 1e-9 and d <= b + 1e-9])
    return out


# ------------------------------------------------------------------ the anchor cuts the answers
class SubsequenceSearch:
    """The anchor's atom sequence aligned over every recording of a store.

    X: unit rows (the memory half, centred per recording); rec: the
    recording of every row; t0: the start time of every row; atom_s: the
    atom. A recording longer than `chunk` atoms is searched as overlapping
    chunks (`overlap` atoms shared) that the vectorised recurrence treats
    as recordings; a candidate is mapped back to its global atoms and its
    one recording. A compute layout, never a cut of the stream.
    """

    def __init__(self, X, rec, t0, atom_s: float, chunk: int = 120, overlap: int = 40):
        self.X = unit(X)
        self.rec = np.asarray(rec)
        self.t0 = np.asarray(t0, np.float64)
        self.atom_s = float(atom_s)
        if not np.isfinite(self.X).all():
            raise ValueError("non-finite rows in the store")
        self.N = len(self.X)
        self.idx, self.rec_of_chunk = [], []
        for r in _ordered_unique(self.rec):
            rows = np.where(self.rec == r)[0]
            if len(rows) <= chunk:
                self.idx.append(rows); self.rec_of_chunk.append(r)
                continue
            starts = list(range(0, len(rows) - overlap, chunk - overlap))
            for s in starts:
                self.idx.append(rows[s:s + chunk]); self.rec_of_chunk.append(r)
        self.n_chunks = len(self.idx)
        m = max((len(v) for v in self.idx), default=1)
        self.pad = np.zeros((self.n_chunks, m), np.int64)
        self.lens = np.zeros(self.n_chunks, np.int64)
        for k, v in enumerate(self.idx):
            self.pad[k, :len(v)] = v
            self.pad[k, len(v):] = v[-1]
            self.lens[k] = len(v)
        self.bounds = {}
        for r in _ordered_unique(self.rec):
            rows = np.where(self.rec == r)[0]
            self.bounds[r] = (int(rows[0]), int(rows[-1]))

    # ---- geometry
    def sim(self, A) -> np.ndarray:
        """Cosine of the anchor's atoms to every row: (n, N), computed per query rather than
        held as an N x N Gram matrix (a three-hour store is 20,000 rows)."""
        with np.errstate(all="ignore"):                 # Accelerate raises spurious FP flags on a finite product
            return self.X[np.asarray(A, np.int64)] @ self.X.T

    def atoms_of(self, rec, t0: float, t1: float) -> np.ndarray:
        """The atoms of one recording inside [t0, t1); a span shorter than an atom takes the atoms it overlaps."""
        m = (self.rec == rec) & (self.t0 >= t0 - 1e-6) & (self.t0 + self.atom_s <= t1 + 1e-6)
        if not m.any():
            m = (self.rec == rec) & (self.t0 < t1 - 1e-6) & (self.t0 + self.atom_s > t0 + 1e-6)
        return np.where(m)[0]

    def span_of(self, atoms) -> tuple:
        atoms = np.asarray(atoms)
        return (float(self.t0[atoms[0]]), float(self.t0[atoms[-1]]) + self.atom_s)

    # ---- the alignment
    def curve(self, A):
        """Subsequence DTW of anchor A (atoms) over every chunk, free start.
        -> (norm (chunks, m), start (chunks, m)): for every end position the
        least cost of aligning the whole anchor to a subsequence ending
        there, divided by (anchor length + matched length), and the start
        the alignment came from."""
        A = np.asarray(A, np.int64)
        n, R, m = len(A), self.n_chunks, self.pad.shape[1]
        D = 1.0 - self.sim(A)[:, self.pad.reshape(-1)].reshape(n, R, m)
        C = np.full((n + 1, m + 1, R), np.inf)
        C[0, :, :] = 0.0
        St = np.zeros((n + 1, m + 1, R), np.int64)
        St[0, :, :] = np.arange(m + 1)[:, None]
        ar = np.arange(R)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cands = np.stack([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]])      # diagonal first on ties
                starts = np.stack([St[i - 1, j - 1], St[i - 1, j], St[i, j - 1]])
                b = np.argmin(cands, axis=0)
                C[i, j] = D[i - 1, :, j - 1] + cands[b, ar]
                St[i, j] = starts[b, ar]
        cost = C[n, 1:, :].T
        start = St[n, 1:, :].T
        length = np.arange(1, m + 1)[None, :] - start
        norm = cost / (n + np.maximum(length, 1))
        for k in range(R):
            norm[k, self.lens[k]:] = np.inf
        return norm, start

    def candidates(self, norm, start) -> list:
        """Local minima of every chunk's cost curve, with their extents."""
        out = []
        for k in range(self.n_chunks):
            c = norm[k, :self.lens[k]]
            L = len(c)
            for j in range(L):
                if not np.isfinite(c[j]):
                    continue
                if (j == 0 or c[j] <= c[j - 1]) and (j == L - 1 or c[j] < c[j + 1]):
                    atoms = self.idx[k][int(start[k, j]):j + 1]
                    if len(atoms) == 0:
                        continue
                    out.append(dict(span=self.span_of(atoms), score=-float(c[j]), rec=self.rec_of_chunk[k], atoms=atoms))
        return out

    def plain(self, A, seqs) -> np.ndarray:
        """End-to-end DTW similarity of anchor A to every atom sequence in seqs (-cost / (n + len))."""
        A = np.asarray(A, np.int64)
        n = len(A)
        lens = np.array([len(s) for s in seqs], np.int64)
        if len(seqs) == 0:
            return np.zeros(0)
        L = int(lens.max())
        pad = np.zeros((len(seqs), L), np.int64)
        for p, s in enumerate(seqs):
            pad[p, :len(s)] = s
            pad[p, len(s):] = s[-1]
        D = 1.0 - self.sim(A)[:, pad.reshape(-1)].reshape(n, len(seqs), L)
        C = np.full((n + 1, L + 1, len(seqs)), np.inf)
        C[0, 0, :] = 0.0
        for i in range(1, n + 1):
            for j in range(1, L + 1):
                C[i, j] = D[i - 1, :, j - 1] + np.minimum(np.minimum(C[i - 1, j], C[i, j - 1]), C[i - 1, j - 1])
        return -C[n, lens, np.arange(len(seqs))] / (n + lens)

    # ---- the neighbours as negatives (gated)
    def neighbours(self, A, peaks) -> list:
        """The one-event-long windows before and after the anchor in its own
        recording, one event being what the coarse edges inside the anchor
        say (an anchor with none inside is one event long)."""
        A = np.asarray(A, np.int64)
        lo, hi = self.bounds[self.rec[A[0]]]
        t0, t1 = self.t0[A[0]], self.t0[A[-1]] + self.atom_s
        peaks = np.asarray(peaks, np.float64)
        ins = peaks[(peaks > t0 + self.atom_s) & (peaks < t1 - self.atom_s)]
        first = int(round((ins[0] - t0) / self.atom_s)) if len(ins) else len(A)
        last = int(round((t1 - ins[-1]) / self.atom_s)) if len(ins) else len(A)
        first, last = max(1, first), max(1, last)
        before = np.arange(max(lo, A[0] - first), A[0])
        after = np.arange(A[-1] + 1, min(hi, A[-1] + last) + 1)
        return [g for g in (before, after) if len(g)]

    def _peaks_for(self, A, contrast):
        if isinstance(contrast, dict):
            return contrast.get(self.rec[np.asarray(A)[0]])
        return contrast

    def contrasted(self, A, seqs, peaks, pos=None) -> np.ndarray:
        """Similarity to the anchor less the mean similarity to its neighbours,
        applied only when every neighbour matches the anchor no better than
        the anchor's median candidate (the self-gated form: under blocked
        collection the neighbours are the class, and the gate reads that
        off the stream; measured never to lose more than .02)."""
        pos = self.plain(A, seqs) if pos is None else pos
        if peaks is None or len(seqs) == 0:
            return pos
        nb = self.neighbours(A, peaks)
        if not nb:
            return pos
        med = float(np.median(pos))
        if all(float(self.plain(A, [g])[0]) <= med for g in nb):
            return pos - np.mean([self.plain(g, seqs) for g in nb], axis=0)
        return pos

    # ---- the entry
    def find(self, anchors, exclude=None, top: int = 10, shots: int = 1, contrast=None, recs=None) -> list:
        """anchors: one atom array, or several (an example and the words'
        chain, or several examples). exclude: a (t0, t1) on the first
        anchor's recording that no answer may overlap (the example itself),
        or a list of (rec, t0, t1). shots > 1: the best candidate of every
        recording that scores above the mean of the recordings' bests
        re-anchors the search, the scores being the mean over anchors of
        the end-to-end similarity. contrast: the coarse edge
        times of the anchors' recording (or {rec: times}), to gate the
        neighbour contrast. recs: the recordings answers may come from (all
        by default). -> ranked candidates [dict(span, score, rec, atoms)]."""
        anchors = [np.asarray(a, np.int64) for a in (anchors if isinstance(anchors, (list, tuple)) else [anchors])]
        anchors = [a for a in anchors if len(a)]
        if not anchors:
            return []
        if exclude is None:
            excl = []
        elif len(exclude) == 2 and not isinstance(exclude[0], (tuple, list)):
            excl = [(self.rec[anchors[0][0]], float(exclude[0]), float(exclude[1]))]
        else:
            excl = [(r, float(a), float(b)) for r, a, b in exclude]

        def keep(c):
            if recs is not None and c["rec"] not in recs:
                return False
            return not any(c["rec"] == r and c["span"][1] > a and c["span"][0] < b for r, a, b in excl)

        pool = {}
        for a in anchors:
            for c in self.candidates(*self.curve(a)):
                if keep(c):
                    pool[(c["span"], c["rec"])] = c
        cands = list(pool.values())
        if not cands:
            return []
        if len(anchors) > 1 or shots > 1 or contrast is not None:
            seqs = [c["atoms"] for c in cands]
            if shots > 1:
                # The best candidate of every recording re-anchors -- of every recording that looks like it holds
                # the moment: the ones whose best scores above the mean of the recordings' bests, a count the
                # store sets. On a one-task stream every recording holds it and the rule keeps the closer half;
                # on a store of several tasks it keeps the task's recordings and leaves the others out, where
                # re-anchoring on all of them pulled the search to whatever the majority of the store held
                # (measured 2026-09-21: dagger example P@1 .975 -> .208 over a four-task store).
                best = {}
                for c in cands:
                    if c["rec"] not in best or c["score"] > best[c["rec"]]["score"]:
                        best[c["rec"]] = c
                bar = float(np.mean([b["score"] for b in best.values()]))
                new = [b["atoms"] for b in best.values() if b["score"] >= bar] or [b["atoms"] for b in best.values()]
                for a in new:
                    for c in self.candidates(*self.curve(a)):
                        if keep(c):
                            pool[(c["span"], c["rec"])] = c
                cands = list(pool.values())
                seqs = [c["atoms"] for c in cands]
                anchors = new + anchors
            sims = []
            for a in anchors:
                s = self.plain(a, seqs)
                sims.append(self.contrasted(a, seqs, self._peaks_for(a, contrast), pos=s) if contrast is not None else s)
            sims = np.mean(sims, axis=0)
            cands = [dict(c, score=float(sims[i])) for i, c in enumerate(cands)]
        return rank(cands, top)


def _ordered_unique(a):
    seen, out = set(), []
    for x in a:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def rank(cands, top: int) -> list:
    """Best first, a candidate that overlaps a chosen one of its recording by
    more than IOU_DUP or touches it suppressed: the alignment answers on both
    sides of a strong match and both are the same moment. A plain global
    order, not a round robin over recordings: measured the same or better on
    one-task streams (BENCHMARKS 2026-09-21, "Strict regimes": the round
    robin is not a prior the eval rewards), and on a store of several tasks
    the round robin would list every recording's best before any recording's
    second, whatever their scores."""
    chosen: dict = {}
    out = []

    def touch(a, b):
        return min(a[1], b[1]) - max(a[0], b[0]) > -1e-6

    for c in sorted(cands, key=lambda c: -c["score"]):
        taken = chosen.setdefault(c["rec"], [])
        if all(iou(c["span"], s) <= IOU_DUP and not touch(c["span"], s) for s in taken):
            taken.append(c["span"])
            out.append(c)
            if len(out) == top:
                break
    return out


# ------------------------------------------------------------------ the clause chain
def chain(scores, props, pred, inside_, relations, agg: str = "both") -> list:
    """scores: (k, N) the fit of every clause on every proposal; props:
    [(t0, t1, n_ev)]; pred / inside_: from `predecessors` / `inside`;
    relations: one per clause ('start' first, then 'then' / 'after' /
    'during'). A 'then' or 'after' clause sits on a proposal that follows
    the previous clause's proposal; a 'during' clause sits on a proposal
    inside the previous clause's, which stays the anchor. One chain per
    end proposal; agg 'mean', 'min' (the weakest clause) or 'both' (the
    chains of both, pooled, see `rank_fusion`). -> [dict(span, score, path, fits)]."""
    if agg == "both":
        pool = {}
        for a in ("mean", "min"):
            for c in chain(scores, props, pred, inside_, relations, a):
                pool[c["path"]] = c
        return rank_fusion(list(pool.values()))
    P = np.asarray(scores, np.float64)
    k, N = P.shape
    comb = (lambda x, y: x + y) if agg == "mean" else min
    best = np.full((k, N), -np.inf)
    prev = np.full((k, N), -1, np.int64)
    at = np.full((k, N), -1, np.int64)
    best[0] = P[0]
    at[0] = np.arange(N)
    for j in range(1, k):
        rel = relations[j] if j < len(relations) else "then"
        if rel == "during":
            for anc in range(N):
                if best[j - 1][anc] == -np.inf:
                    continue
                for i in inside_[anc]:
                    c = comb(best[j - 1][anc], P[j][i])
                    if c > best[j][anc]:
                        best[j][anc], prev[j][anc], at[j][anc] = c, anc, i
        else:
            for i in range(N):
                for p in pred[i]:
                    if best[j - 1][p] > -np.inf:
                        c = comb(best[j - 1][p], P[j][i])
                        if c > best[j][i]:
                            best[j][i], prev[j][i], at[j][i] = c, p, i
    out = []
    for anc in range(N):
        if best[k - 1][anc] == -np.inf:
            continue
        path, a_ = [], anc
        for j in range(k - 1, -1, -1):
            path.append(int(at[j][a_]))
            a_ = prev[j][a_]
        path = tuple(path[::-1])
        fits = tuple(float(P[j][path[j]]) for j in range(k))
        out.append(dict(span=(props[path[0]][0], props[anc][1]),
                        score=float(best[k - 1][anc]) / (k if agg == "mean" else 1), path=path, fits=fits))
    return out


def rank_fusion(chains) -> list:
    """Every chain ranked by its mean clause fit and by its weakest clause
    fit; ordered by the sum of the two ranks (no weight, no threshold).
    Measured: three- and four-clause chains .53 / .57 -> .98 / 1.0 at one
    with the one- and two-clause results untouched. -> the chains, best first,
    each with `score` = -rank sum (ties broken by the mean)."""
    if not chains:
        return []
    means = np.array([np.mean(c["fits"]) for c in chains])
    mins = np.array([np.min(c["fits"]) for c in chains])
    r = np.argsort(np.argsort(-means)) + np.argsort(np.argsort(-mins))
    out = [dict(c, score=-float(r[i]) + 1e-3 * float(means[i])) for i, c in enumerate(chains)]
    return sorted(out, key=lambda c: -c["score"])
