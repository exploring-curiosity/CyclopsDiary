"""What a corpus can be asked about itself.

Retrieval answers "what looks like this". Pointed differently, the same
arithmetic answers four more questions that a fleet pays people to answer by
hand:

    novelty     has anything ever looked like this
    redundant   which footage adds nothing
    cover       which few spans stand for all of it
    apart       is this week unlike last week
    align       what is different about this route since the last run
    outliers    what in this shift was not like normal
    frozen      where did the feed stop moving
    stretches   where does one stretch of this stream end and the next begin
    shifted     which of those boundaries held, and how surely
    uncovered   which places did this run never reach
    spend       which few of these are worth adding to what is held

Everything here is a fold over the vectors a stream already has. Nothing is
trained, nothing is labelled, and no threshold is declared: where a cut is
needed it is read off the data with `split`, because a constant in this file
would be the system deciding how similar is similar in a building it has
never seen.

All inputs are (n, d) float arrays of one store's exposures. Rows are taken
to be, or are made into, unit vectors, so an inner product is a cosine and a
distance is `1 - cosine` in [0, 2].
"""
from __future__ import annotations

import numpy as np


def unit(V) -> np.ndarray:
    """Rows as unit vectors. A zero row stays zero rather than becoming a
    direction nobody chose."""
    X = np.asarray(V, np.float32)
    if X.ndim == 1:
        X = X[None]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def _sim(A, B) -> np.ndarray:
    """Cosine of every row of A against every row of B."""
    return unit(A) @ unit(B).T


# Below this, a spread is arithmetic noise rather than a difference. A
# cosine here is a float32 dot product over a couple of thousand terms, and
# its rounding error accumulates to a few parts in a million; two identical
# clips come back differing in the seventh decimal. This is a fact about the
# machine, not a judgement about video, which is why it is allowed to be a
# number in this file when nothing else is.
NOISE = 1e-5


def split(x, anchor: float | None = None) -> float | None:
    """A cut read off the numbers themselves, or None if there is one group.

    Otsu's method -- the threshold that leaves the least variance inside the
    two sides -- used rather than a constant because the only honest answer
    to "how similar is too similar" is whatever this corpus's own two clumps
    say, and a corpus with one clump has to be able to say so.

    Otsu will always find a best cut, including in a cloud of noise that has
    no two sides, so two things have to hold before a cut counts. The groups
    must not reach each other: their means further apart than their own
    spreads. And, when an `anchor` is given, the group on its side must be
    nearer to the anchor than to the other group. The anchor is the value
    that means "the same thing" -- 1 for a cosine, 0 for a distance -- and
    it is geometry rather than a tuned number: without it, twelve unrelated
    clips whose similarities happen to fall in two accidental clumps get
    called duplicates of each other.
    """
    v = np.asarray(x, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) <= NOISE:
        return None
    edges = np.linspace(v.min(), v.max(), 65)
    counts, _ = np.histogram(v, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(counts)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(counts * mids) / w0
        m1 = (np.sum(counts * mids) - np.cumsum(counts * mids)) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between = np.where(np.isfinite(between), between, -np.inf)
    if not np.any(np.isfinite(between)) or np.max(between) <= 0:
        return None
    lo, hi = v[v <= mids[int(np.argmax(between))]], v[v > mids[int(np.argmax(between))]]
    if lo.size == 0 or hi.size == 0:
        return None
    gap = hi.mean() - lo.mean()
    if gap <= (lo.std() + hi.std()):
        return None                                    # the groups reach each other
    if anchor is not None:
        near = hi if abs(anchor - hi.mean()) < abs(anchor - lo.mean()) else lo
        if abs(anchor - near.mean()) >= gap:
            return None                                # neither side means "the same"
    # The boundary BETWEEN them, not the top of the lower one. Every caller
    # asks "is this row on the high side", and the top of the low group
    # answers yes for the whole corpus.
    return float((lo.mean() + hi.mean()) / 2)


# ---------------------------------------------------------------- novelty
def novelty(V, against, k: int = 5) -> np.ndarray:
    """How far each row is from the k-th nearest thing already seen.

    The k-th and not the first: one near-duplicate in a corpus says somebody
    once stood here, and it should not make a place familiar. The k-th says
    the region is populated.

    With nothing to compare against the answer is NaN, not 1.0. A fleet's
    first hour is not the most surprising hour it will ever have, and a
    system that said it was would make every robot's first day an emergency.
    """
    X, C = unit(V), np.asarray(against, np.float32)
    if C.size == 0 or C.shape[0] == 0:
        return np.full(X.shape[0], np.nan, np.float32)
    C = unit(C)
    kk = int(min(max(1, k), C.shape[0]))
    out = np.empty(X.shape[0], np.float32)
    # In blocks: a fleet-sized corpus against a day of exposures is a matrix
    # nobody needs to hold whole.
    for i in range(0, X.shape[0], 1024):
        s = X[i:i + 1024] @ C.T
        kth = np.partition(s, -kk, axis=1)[:, -kk]
        out[i:i + 1024] = 1.0 - kth
    return out


# -------------------------------------------------------------- redundant
def redundant(V, at: float | None = None) -> list:
    """Rows that say nothing the rows before them did not already say.

    Each row is compared with everything earlier; the ones whose nearest
    earlier neighbour is closer than the corpus's own cut are redundant. The
    cut comes from `split` over those nearest-neighbour similarities, so a
    corpus with no repetition in it surrenders nothing rather than giving up
    its least interesting tenth.
    """
    X = unit(V)
    if X.shape[0] < 3:
        return []
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    # Only earlier rows count: redundancy is a property of what a row adds
    # when it arrives, and "the future repeats me" is not a reason to drop
    # the first sighting of something.
    S[np.triu_indices(S.shape[0], k=0)] = -np.inf
    best = S.max(axis=1)
    best[0] = -np.inf                                  # nothing precedes the first
    live = best[np.isfinite(best)]
    cut = at if at is not None else split(live, anchor=1.0)
    if cut is None:
        return []
    return [int(i) for i in np.flatnonzero(best >= cut)]


# ------------------------------------------------------------------ cover
def cover(V, n: int, against=None) -> list:
    """The n rows that leave nothing far from something chosen.

    Farthest-point traversal: start at the row furthest from the centre,
    then repeatedly take whatever is currently worst represented. It is the
    greedy k-centre, which is within a factor of two of the best possible
    cover, and it is what "show me what a day looks like" means when nobody
    will watch the day.

    `against` is a corpus that already exists, and it turns the same walk
    into a budget. A robot with one bar of signal and a day of footage is
    not choosing at random, and the most interesting clip of its day is the
    wrong thing to send if the fleet already has it: with `against`, the
    first pick is what the fleet is furthest from rather than what this day
    is furthest from, and every pick after that keeps both in view.
    """
    X = unit(V)
    m = X.shape[0]
    if m == 0 or n <= 0:
        return []
    if n >= m:
        return list(range(m))
    return [i for i, _ in _greedy(X, n, _start(X, against))]


def _start(X, against) -> np.ndarray:
    """How far every row is from what is already held, before anything is
    picked. With nothing held, everything is equally unheld."""
    C = None if against is None else np.asarray(against, np.float32)
    if C is None or not C.size or not C.shape[0]:
        return np.full(X.shape[0], np.inf, np.float32)
    return novelty(X, C, k=1)


def _greedy(X, n: int, far) -> list:
    """Farthest-point traversal from wherever `far` starts it.

    Returns (row, gain) pairs in the order they were taken, where the gain
    is how far that row was from everything -- held or already picked --
    at the moment it was chosen. The gain is what makes this a budget
    rather than a quota: a pick whose gain has collapsed is a byte spent
    on something the corpus already has.
    """
    far = np.array(far, np.float32, copy=True)
    if np.all(np.isinf(far)):
        # Nothing is held, so nothing says where to start: begin at the row
        # furthest from the corpus's own centre, which is the one a single
        # pick represents worst.
        centre = unit(X.mean(axis=0))
        first = int(np.argmin((X @ centre.T).ravel()))
        out = [(first, float("inf"))]
        far = 1.0 - (X @ X[first])
    else:
        first = int(np.argmax(far))
        out = [(first, float(far[first]))]
        far = np.minimum(far, 1.0 - (X @ X[first]))
    # A row already taken is out of the running, rather than relying on its
    # distance to itself being zero: in a corpus of near-identical rows every
    # remaining distance is zero too, and the walk would hand back the same
    # row n times.
    far[first] = -np.inf
    for _ in range(int(n) - 1):
        nxt = int(np.argmax(far))
        out.append((nxt, float(far[nxt])))
        far = np.minimum(far, 1.0 - (X @ X[nxt]))
        far[nxt] = -np.inf
    return out


def cover_radius(V, picked) -> float:
    """The furthest any row is from the nearest chosen one.

    The number that says whether a cover is worth anything: after these
    picks, nothing in the corpus is further than this from one of them.
    """
    X = unit(V)
    if X.shape[0] == 0 or not len(picked):
        return float("nan")
    return float(np.max(1.0 - (X @ X[list(picked)].T).max(axis=1)))


# ------------------------------------------------------------------ apart
def apart(A, B, trials: int = 0, seed: int = 0) -> dict:
    """How far two sets of exposures are from each other, and how surely.

    The distance is between the two means, which on unit vectors is exactly
    the maximum mean discrepancy under a linear kernel -- so it is a real
    two-sample statistic and not a summary that happens to look like one.

    `trials` shuffles the two sets together and re-measures, which gives a
    p-value with nothing fitted and no distribution assumed: the null is the
    data's own. Without it the distance stands alone, which is enough to
    plot and not enough to alert on.
    """
    X, Y = unit(A), unit(B)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return {"distance": float("nan"), "p": None, "n": [int(X.shape[0]), int(Y.shape[0])]}
    obs = float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))
    out = {"distance": obs, "p": None, "n": [int(X.shape[0]), int(Y.shape[0])],
           "trials": int(trials)}
    if trials <= 0:
        return out
    both = np.vstack([X, Y])
    cut = X.shape[0]
    g = np.random.default_rng(seed)
    worse = 0
    for _ in range(int(trials)):
        g.shuffle(both)
        if np.linalg.norm(both[:cut].mean(axis=0) - both[cut:].mean(axis=0)) >= obs:
            worse += 1
    # +1 on both sides: a permutation test can never honestly report zero,
    # and an alert built on p == 0 fires on the first fluke it meets.
    out["p"] = (worse + 1) / (trials + 1)
    return out


# ------------------------------------------------------------------ align
def align(A, B) -> list:
    """Line two runs of the same route up, step by step.

    Dynamic time warping over the exposures, so the comparison is against
    the same PLACE rather than the same second: a robot that drove the aisle
    slowly today is not a robot that saw something different.

    Returns the path as (i, j) pairs, oldest first.
    """
    X, Y = unit(A), unit(B)
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return []
    D = 1.0 - (X @ Y.T)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def changed(A, B, at: float | None = None) -> list:
    """Where two runs of one route stop agreeing.

    Not a score: the steps. "This aisle looks different today" is worth
    nothing to anybody without which seconds of it, and the cut between
    agreeing and not is read off this pair of runs rather than declared.
    """
    path = align(A, B)
    if not path:
        return []
    X, Y = unit(A), unit(B)
    d = np.array([1.0 - float(X[i] @ Y[j]) for i, j in path], np.float32)
    cut = at if at is not None else split(d, anchor=0.0)
    if cut is None:
        return []
    out = []
    for (i, j), dist in zip(path, d):
        if dist >= cut:
            out.append({"a": int(i), "b": int(j), "distance": round(float(dist), 4)})
    return out


# --------------------------------------------------------------- outliers
def outliers(V, against, k: int = 5) -> list:
    """Which of these exposures are unlike the period they are measured
    against.

    Triage, and the job a person does today by watching a shift back at
    four times speed: here is what normal looked like, show me what was
    not it. The cut is `split` over the novelties, so a shift that was like
    every other shift raises nothing -- which matters more than finding
    incidents, because a triage that always finds something is one nobody
    reads by the second week.
    """
    s = novelty(V, against, k=k)
    live = s[np.isfinite(s)]
    if live.size == 0:
        return []
    cut = split(live, anchor=0.0)
    if cut is None:
        return []
    return [{"i": int(i), "novelty": round(float(s[i]), 4)}
            for i in np.flatnonzero(np.isfinite(s) & (s >= cut))]


# ----------------------------------------------------------------- frozen
def _consecutive(X) -> np.ndarray:
    """Cosine of every exposure with the one after it."""
    return np.einsum("ij,ij->i", X[:-1], X[1:]) if X.shape[0] > 1 else np.zeros(0, np.float32)


def frozen(V, at: float | None = None) -> list:
    """Runs where nothing changed at all: a feed that stopped moving.

    A stuck camera fills a disk exactly as fast as a working one, and it
    will never be found by looking for a strange scene -- what says it is a
    fault is that nothing changed for a long time. Returns (first, last)
    row pairs, longest first, and no verdict: forty seconds of an idle
    loading bay and forty seconds of a frozen encoder look identical from
    here, and which one it is depends on where the camera points.
    """
    X = unit(V)
    if X.shape[0] < 3:
        return []
    same = _consecutive(X)
    cut = at if at is not None else split(same, anchor=1.0)
    if cut is None:
        return []
    runs, i = [], 0
    while i < same.size:
        if same[i] < cut:
            i += 1
            continue
        j = i
        while j + 1 < same.size and same[j + 1] >= cut:
            j += 1
        runs.append((int(i), int(j + 1)))
        i = j + 1
    return sorted(runs, key=lambda r: r[0] - r[1])


# -------------------------------------------------------------- stretches
def stretches(V, at: float | None = None) -> list:
    """The stream cut at its own change points, as (first, last) rows.

    Chaptering with nothing declared: what separates one stretch from the
    next is a step far above the steps this stream usually takes. A stream
    that never changes is one stretch, which is the answer and not a
    failure to find any.
    """
    X = unit(V)
    n = X.shape[0]
    if n == 0:
        return []
    if n < 3:
        return [(0, n - 1)]
    d = 1.0 - _consecutive(X)
    cut = at if at is not None else split(d, anchor=0.0)
    if cut is None:
        return [(0, n - 1)]
    edges = [int(i) + 1 for i in np.flatnonzero(d >= cut)]
    out, start = [], 0
    for e in edges:
        out.append((start, e - 1))
        start = e
    out.append((start, n - 1))
    return [(a, b) for a, b in out if b >= a]


def shifted(V, trials: int = 200, seed: int = 0) -> list:
    """Every boundary in the stream, with how sure it is that it held.

    The difference between a scene changing and a camera being knocked is
    not the size of the step -- it is whether the stream goes back. So each
    boundary is measured as a two-sample test between the stretch before it
    and the stretch after it, which is a statement about the two periods
    rather than about one frame. The p-value is the stream's own shuffle:
    nothing is fitted, and no alerting constant is declared here, because
    what counts as an incident is a property of the site and not of this
    file.
    """
    X = unit(V)
    segs = stretches(X)
    out = []
    for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
        got = apart(X[a0:a1 + 1], X[b0:b1 + 1], trials=trials, seed=seed)
        out.append({"at": int(b0), "distance": round(float(got["distance"]), 4),
                    "p": got["p"], "before": [a0, a1], "after": [b0, b1]})
    return out


def _beyond(d, base) -> float | None:
    """The cut between a distance that means "there is one of these already"
    and one that means "there is nothing like this".

    A distance is not far or near on its own: 0.05 is a twin in a corpus of
    near-duplicates and a stranger in a corpus of one scene. So the cut is
    read off a pool of the two populations together -- the corpus's own
    neighbour distances, which are by definition what "the same thing"
    looks like here, and the distances being judged. A pool with one clump
    in it means every one of them has a partner, and the answer is nothing.
    """
    pool = np.concatenate([np.asarray(base, np.float64).ravel(),
                           np.asarray(d, np.float64).ravel()])
    return split(pool[np.isfinite(pool)], anchor=0.0)


def spend(V, against, n: int, k: int = 1) -> list:
    """The few rows worth adding to a corpus that already exists.

    The uplink, and every labelling budget: `n` is a ceiling and not a
    quota. Each pick's gain is how far it was from everything held and
    everything already picked, and the picks stop where that gain falls
    back into what the held corpus calls the same thing -- so a day with
    one new thing in it costs one clip, whatever budget it was offered.
    """
    X = unit(V)
    if X.shape[0] == 0 or n <= 0:
        return []
    C = np.asarray(against, np.float32) if against is not None else None
    took = _greedy(X, min(int(n), X.shape[0]), _start(X, C))
    out = [{"i": int(i), "gain": (None if np.isinf(g) else round(float(g), 4))}
           for i, g in took]
    if C is None or not C.size or not C.shape[0]:
        return out                                     # nothing held: all of it is new
    gains = np.array([g for _, g in took], np.float64)
    cut = _beyond(gains, novelty(C, C, k=k + 1))
    if cut is None:
        return out
    return [o for o, g in zip(out, gains) if g >= cut]


# -------------------------------------------------------------- uncovered
def uncovered(ref, run, k: int = 1) -> list:
    """Which of the reference's places this run never reached.

    Inspection completeness, which is the question an inspection robot
    exists to answer and the one nobody can answer from a video file. Not
    `changed`: a place that looks different was visited, and a place with
    no partner anywhere in the run was not. Nearest neighbour over the
    whole run and not the aligned one, for exactly that reason -- an
    alignment gives every row a partner whether or not it deserves one.
    """
    s = novelty(ref, run, k=k)
    if not np.any(np.isfinite(s)):
        return []
    # The run's own leave-one-out neighbour distances are what "the same
    # place, seen again" measures here, so they are the population the
    # reference's distances are judged against. Judging them on their own
    # would report a third of a route missed because one end of it was
    # slightly noisier than the other.
    cut = _beyond(s, novelty(run, run, k=k + 1))
    if cut is None:
        return []
    return [int(i) for i in np.flatnonzero(np.isfinite(s) & (s >= cut))]
