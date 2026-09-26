"""The output-head score as query by example (`headsearch`).

One formula serves the words and the example (ledger N-65..67):
score(q, step) = sum_w q(w) * (log p_step(w) - mean over the set's steps of log p(w)),
with q the example step's own distribution. The index keeps each step's
top-K words; a word outside a step's list takes the list's last value.
No cosine anywhere: the alignment cost is 1 - this score.

Run: python3 -m pytest tests/test_headsearch.py -v -p no:django
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np                                    # noqa: E402

from elidedb import headsearch as HS                  # noqa: E402


def test_dense_fills_a_word_outside_a_steps_list_with_that_lists_floor():
    """The union is every word that reaches some step's top `union_k`; a step
    that did not list a word holds its own last listed value for it."""
    ids = np.array([[5, 3, 9], [3, 7, 5]])
    lps = np.array([[-1.0, -2.0, -4.0], [-0.5, -3.0, -6.0]])
    words, LP = HS.dense(ids, lps, union_k=2)
    assert words.tolist() == [3, 5, 7]
    assert LP.tolist() == [[-2.0, -1.0, -4.0], [-0.5, -6.0, -3.0]]


def test_the_score_is_the_examples_distribution_against_the_centred_log_probs():
    rng = np.random.default_rng(0)
    LP = rng.normal(size=(6, 4)) - 5.0
    hs = HS.HeadSearch(LP, rec=[0, 0, 0, 1, 1, 1], t0=[0, 1, 2, 0, 1, 2], atom_s=1.0)
    A = [1, 2]
    S = hs.sim(A)
    assert S.shape == (2, 6)
    for i, a in enumerate(A):
        q = np.exp(LP[a]) / np.exp(LP[a]).sum()
        for j in range(6):
            want = sum(q[w] * (LP[j, w] - LP[:, w].mean()) for w in range(4))
            assert abs(S[i, j] - want) < 1e-9


def _stream(n, events, rng, vocab=60, k=24, lift=6.0):
    """Top-k lists of n steps: background words 10.. at random, the event's
    steps with words 0..3 lifted (the same thing seen again)."""
    ids, lps = [], []
    for j in range(n):
        logit = rng.normal(size=vocab)
        logit[:10] -= 3.0
        if any(a <= j < b for a, b in events):
            logit[:4] += lift
        lp = logit - np.log(np.exp(logit).sum())
        top = np.argsort(-lp)[:k]
        ids.append(top)
        lps.append(lp[top])
    return np.array(ids), np.array(lps)


def test_an_example_finds_the_same_moment_in_another_recording():
    rng = np.random.default_rng(1)
    ia, la = _stream(30, [(5, 9)], rng)
    ib, lb = _stream(30, [(20, 24)], rng)
    words, LP = HS.dense(np.vstack([ia, ib]), np.vstack([la, lb]), union_k=12)
    rec = np.array(["a"] * 30 + ["b"] * 30)
    t0 = np.concatenate([np.arange(30.0), np.arange(30.0)])
    hs = HS.HeadSearch(LP, rec, t0, atom_s=1.0)
    top = hs.find(np.arange(5, 9), recs={"b"}, top=3)
    assert top and top[0]["rec"] == "b"
    s, e = top[0]["span"]
    inter = max(0.0, min(e, 24.0) - max(s, 20.0))
    assert inter / (max(e, 24.0) - min(s, 20.0)) >= 0.5, top[0]["span"]
