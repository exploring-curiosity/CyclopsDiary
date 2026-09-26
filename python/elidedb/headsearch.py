"""The output-head score as query by example: one formula for the words and the example.

    score(q, step) = sum_w q(w) * (log p_step(w) - mean over the set's steps of log p(w))

q is uniform over a sentence's content words for text, and the example step's own
output-head distribution for an example (ledger N-65..N-67). A step's distribution is
the world model's own head on the atom's video tokens (the log of their mean
probability, `cosmos3_mlx.MLXStream(top_k=...)`), and the index keeps only each step's
top-K words: a word outside a step's list takes that list's last value (its floor), and
the words scored are the union of every step's top `union_k`.

Why this and not a cosine of memory states: the owner's rule is no cosine anywhere,
example search included, and it is also the better number. On the dev set the formula
puts the right recording first 87% of the time where the centred-state cosine does 27%,
and pooled over three sets the block examples read P@1 .280 / P@support .240 against
.200 / .140 (ledger N-70). Where it is weak is localisation inside the recording (.187
given the recording); the alignment below is what localises.

The alignment is eventsearch's subsequence DTW with only the per-step score changed
(cost = 1 - score): free start, the anchor's own length, candidates at the cost curve's
minima. The mean is taken over the set being searched, so no constant is fitted to a
corpus; searching a different set moves the mean with it.
"""
from __future__ import annotations

import numpy as np

from . import eventsearch as EV

UNION_K = 128            # the words scored: those reaching some step's top 128, the setting the numbers above used


def dense(ids, lps, union_k: int = UNION_K):
    """ids, lps: (steps, K), each row a step's top-K word ids and log-probs, best first.
    -> (words (V,), LP (steps, V)): log p over the union of every step's top `union_k`
    words, a word a step did not list at that step's last listed value."""
    ids = np.asarray(ids, np.int64)
    lps = np.asarray(lps, np.float64)
    if ids.ndim != 2 or ids.shape != lps.shape or ids.shape[1] < 1:
        raise ValueError(f"ids {ids.shape} and lps {lps.shape}: one (steps, K) list each")
    words = np.unique(ids[:, :union_k])
    LP = np.repeat(lps[:, -1:], len(words), axis=1)
    at = np.minimum(np.searchsorted(words, ids), len(words) - 1)
    r, c = np.nonzero(words[at] == ids)
    LP[r, at[r, c]] = lps[r, c]
    return words, LP


class HeadSearch(EV.SubsequenceSearch):
    """eventsearch.SubsequenceSearch over output-head rows. LP: (steps, V) from `dense`;
    rec, t0, atom_s as there. The state rows the parent class keeps are not read."""

    def __init__(self, LP, rec, t0, atom_s: float, **kw):
        LP = np.asarray(LP, np.float64)
        if LP.ndim != 2 or not np.isfinite(LP).all():
            raise ValueError("LP must be a finite (steps, words) matrix")
        super().__init__(np.ones((len(LP), 1)), rec, t0, atom_s, **kw)
        self.LP = LP
        self.PM = LP - LP.mean(0, keepdims=True)

    @classmethod
    def from_lists(cls, ids, lps, rec, t0, atom_s: float, union_k: int = UNION_K, **kw) -> "HeadSearch":
        return cls(dense(ids, lps, union_k)[1], rec, t0, atom_s, **kw)

    def sim(self, A) -> np.ndarray:
        """(len(A), steps): each anchor step's own distribution against every step's centred log p."""
        A = np.asarray(A, np.int64)
        P = np.exp(self.LP[A] - self.LP[A].max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        with np.errstate(all="ignore"):                 # Accelerate raises spurious FP flags on a finite product
            return P @ self.PM.T
