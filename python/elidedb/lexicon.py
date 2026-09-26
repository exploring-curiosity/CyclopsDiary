"""Closed-class English lexicon shared by query routing and training.

Dependency-free ON PURPOSE: the query path imports this (via
rerank.directional_swap), and it previously reached the same tables
through fdnnv2, whose module level imports mlx. That worked on a Mac
and crashed every Linux deployment. Lexicon is language knowledge, not
model code; it lives where nothing heavier than a list comprehension
runs.

Verb/direction swaps: GENERIC english antonyms, no dataset nouns, per
the no-hardwire rule.
"""
from __future__ import annotations

# ---------------------------------------------------------------------
# DERIVED OPPOSITES. The 81 pairs below and the 16 `un-` bases were
# hand-authored, and fold/stack/cover are conspicuously this corpus's
# vocabulary - a task prior in a file that claims to hold generic
# English. The swap-contrast MECHANISM is sound and measured; the list
# was not derived.
#
# derived_swaps() replaces it: opposites are found by reflection in an
# embedding space over the vocabulary the CORPUS attested. On a driving
# log that finds accelerate/brake without anyone having thought of them;
# here it finds open/close. The hand list stays only as a cold-start
# fallback for an empty store, and is labelled as such rather than
# presented as knowledge.
# ---------------------------------------------------------------------


def derived_swaps(store, vec=None, min_count=2):
    """Opposite pairs this corpus supports, or () if it cannot say."""
    from .derive import attested, opposite_pairs
    try:
        if "labels" not in store.tables():
            return ()
        vals = store.table("labels").scan().column("value").to_pylist()
        vocab = attested(vals, min_count=min_count)
        if len(vocab) < 8:
            return ()
        if vec is None:
            from .sig2 import _text_vec as vec
        return tuple((a, b) for a, b, _ in opposite_pairs(vocab, vec))
    except Exception:
        return ()


# THERE IS NO VERB TABLE. VERB_SWAPS (17 hand-authored pairs) and
# _UN_BASES (16 hand-authored stems that generated ~64 more) lived here
# as a "cold-start fallback", labelled a violation by their own comment
# and kept anyway. A fallback that always fires is not a fallback: on
# every store measured, derived_swaps returned 0 pairs, so the hand list
# WAS the direction mechanism, and it was also feeding the student's
# training negatives.
#
# Removed. derived_swaps() above is the only source of oppositions: what
# THIS corpus attests, scored in its own embedding space. A corpus that
# cannot express an opposition now yields None, and callers must handle
# that honestly rather than borrow English from a list. A forklift
# corpus gets forklift oppositions or it gets nothing - which is the
# rule, and the point.
