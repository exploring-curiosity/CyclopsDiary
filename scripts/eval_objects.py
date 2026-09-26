#!/usr/bin/env python3
"""Evaluate objects.py on the sample clips: identification by the world model's own words, and tracking
through time. The labels and the protocol are eval/cyclopsdiary_objects.json, written before the run.

Reads Atlas (the team's database for the phone clips, lastseen_smoke for the negatives) and writes nothing
there. Output: eval_logs/objects_sample_<UTC time>.json and a short summary.

    python3 scripts/eval_objects.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from cyclopsdiary import tower  # noqa: E402

tower.use_stack()                                          # before anything imports transformers

import numpy as np  # noqa: E402
from elidedb import corpus  # noqa: E402

from cyclopsdiary import atlas, memory, objects  # noqa: E402
from cyclopsdiary.lastseen import tokenizer  # noqa: E402

P = json.loads((ROOT / "eval" / "cyclopsdiary_objects.json").read_text())
PATTERN = "wiring-cam-20260926T182226809"


def combine(*mems):
    fl = {}
    for m in mems:
        fl.update(m.footage)
    return memory.Memory(fl, np.concatenate([m.ids for m in mems]), np.concatenate([m.lp for m in mems]),
                         np.concatenate([m.rec for m in mems]), np.concatenate([m.t0 for m in mems]),
                         mems[0].atom_s, mems[0].union_k)


def prefix(m, k):
    return memory.Memory(m.footage, m.ids[:k], m.lp[:k], m.rec[:k], m.t0[:k], m.atom_s, m.union_k)


def only(m, recs):
    keep = np.isin(m.rec, list(recs))
    return memory.Memory({r: m.footage[r] for r in recs}, m.ids[keep], m.lp[keep], m.rec[keep], m.t0[keep],
                         m.atom_s, m.union_k)


def label(m, j, name):
    """1 / 0 / None (cannot tell) / "u" (unlabelled robot footage)."""
    rec, i = str(m.rec[j]), int(round(float(m.t0[j]) / m.atom_s))
    if rec == PATTERN:
        return 0
    if rec in P["footage"]:
        return P["visible"][name][rec][i] if name in P["visible"] else 0
    return "u"


def detect(m, name, tok):
    s = objects.name_scores(m, objects.name_words(tok, name))
    cut = corpus.split(s)
    return s, cut, (s > cut) if cut is not None else np.zeros(m.n, bool)


def identify(m, name, tok, labels_as=None):
    y = [label(m, j, labels_as or name) for j in range(m.n)]
    try:
        s, cut, present = detect(m, name, tok)
    except ValueError as e:                                # the tracker cannot take the name: every positive missed
        pos = sum(1 for v in y if v == 1)
        return dict(steps=m.n, positives=pos, error=str(e), cut=None, present=0, tp=0, fp=0, fn=pos,
                    unlabelled_hits=0, **(dict(p_at_1=0.0, p_at_support=0.0) if pos else {}))
    scored = [j for j in range(m.n) if y[j] in (0, 1)]
    order = sorted(scored, key=lambda j: -s[j])
    pos = sum(y[j] for j in scored)
    out = dict(steps=m.n, positives=pos, cut=None if cut is None else float(cut),
               present=int(present.sum()),
               tp=sum(1 for j in scored if present[j] and y[j] == 1),
               fp=sum(1 for j in scored if present[j] and y[j] == 0),
               fn=sum(1 for j in scored if not present[j] and y[j] == 1),
               unlabelled_hits=sum(1 for j in range(m.n) if present[j] and y[j] == "u"))
    if pos:
        out.update(p_at_1=float(y[order[0]]), p_at_support=float(np.mean([y[j] for j in order[:pos]])))
    return out


LEX = {loc: set(ws) for loc, ws in P["place_lexicon"].items() if loc != "note"}


def location(words):
    for w in words:
        for loc, ws in LEX.items():
            if w.lower() in ws:
                return loc
    return "unknown"


def truth_after(m, k):
    """Where a perfect tracker says the keys were last seen, after the first k steps: the location during the
    last of those steps in which the keys were in view."""
    for j in range(k - 1, -1, -1):
        rec, i = str(m.rec[j]), int(round(float(m.t0[j]) / m.atom_s))
        if P["visible"]["keys"][rec][i] == 1:
            return P["keys_location"][rec][i]
    return None


def last_seen(m, tok, own):
    """The tracker's answer: the latest sighting (a run of steps where "keys" stands apart) and the model's
    other top words at its last step, as objects.track and objects.where give them."""
    s, cut, present = detect(m, "keys", tok)
    segs = objects._segments(m, present, set(m.footage))
    if not segs:
        return None
    end = lambda seg: m.footage[seg[0]]["started_at"].timestamp() + float(m.t0[seg[1][-1]])   # noqa: E731
    fid, rows = max(segs, key=end)
    words = objects.place(m, rows[-1:], tok, own)
    return dict(footage=fid, t0=float(m.t0[rows[0]]), t1=float(m.t0[rows[-1]]) + m.atom_s, words=words,
                location=location(words))


def main():
    t = time.monotonic()
    print("loading the world model's tokenizer (its words only, not the model) ...", flush=True)
    tok = tokenizer()
    c = atlas.client()
    phones = memory.load(c["cyclopsdiary"], "home")
    smoke_home = memory.load(c["lastseen_smoke"], "home")
    pattern = memory.load(c["lastseen_smoke"], "wiring-check")
    sets = {"phones": phones, "phones+pattern": combine(phones, pattern),
            "phones+pattern+robot": combine(phones, pattern, smoke_home)}
    names = [n for n in P["visible"]]
    out = dict(protocol="eval/cyclopsdiary_objects.json", at=datetime.now(timezone.utc).isoformat(), sets={})
    for sname, m in sets.items():
        per = {n: identify(m, n, tok) for n in names}
        absent = {n: identify(m, n, tok) for n in P["absent"]}
        deviation = {"soda can": identify(m, "soda can", tok, labels_as="can")}   # after the fact: not in the summary
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn"))
        out["sets"][sname] = dict(
            steps=m.n, present=per, absent=absent, deviation=deviation,
            summary=dict(
                names=len(names),
                p_at_1=float(np.mean([v["p_at_1"] for v in per.values()])),
                p_at_support=float(np.mean([v["p_at_support"] for v in per.values()])),
                step_precision=tp / (tp + fp) if tp + fp else None, step_recall=tp / (tp + fn) if tp + fn else None,
                names_with_no_cut=sum(1 for v in per.values() if v["cut"] is None),
                absent_names_with_a_sighting=sum(1 for v in absent.values() if v["present"] > 0),
                absent_names=len(P["absent"]),
                unlabelled_hits=sum(v["unlabelled_hits"] for v in list(per.values()) + list(absent.values()))))
    own = {t_ for words in objects.name_words(tok, "keys") for toks in words for t_ in toks}
    order = only(phones, P["footage"])                                        # world-time order: iPhone, then Pixel
    through = []
    for k in range(2, order.n + 1):
        got = last_seen(prefix(order, k), tok, own)
        want = truth_after(order, k)
        through.append(dict(steps=k, after=f"{order.rec[k - 1]}:{float(order.t0[k - 1]):g}", truth=want,
                            answer=got, correct=bool(got and got["location"] == want)))
    clips = [r for r in through if r["steps"] in (7, order.n)]                 # when each clip has landed (serve)
    full = objects._segments(order, detect(order, "keys", tok)[2], set(order.footage))
    trail = []
    for fid, rows in full:
        w = objects.place(order, rows, tok, own)
        trail.append(dict(footage=fid, t0=float(order.t0[rows[0]]), t1=float(order.t0[rows[-1]]) + order.atom_s,
                          words=w, location=location(w),
                          end_location=location(objects.place(order, rows[-1:], tok, own))))
    out["through_time"] = dict(per_step=through, after_each_clip=clips, trail=trail,
                               summary=dict(after_each_clip=f"{sum(r['correct'] for r in clips)}/{len(clips)}",
                                            per_step=f"{sum(r['correct'] for r in through)}/{len(through)}"))
    out["seconds"] = round(time.monotonic() - t, 1)
    path = ROOT / "eval_logs" / f"objects_sample_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    for sname, v in out["sets"].items():
        s = v["summary"]
        print(f"{sname:22s} {v['steps']:3d} steps  P@1 {s['p_at_1']:.2f}  P@support {s['p_at_support']:.2f}  "
              f"step precision {s['step_precision'] if s['step_precision'] is None else round(s['step_precision'], 2)}  "
              f"recall {s['step_recall'] if s['step_recall'] is None else round(s['step_recall'], 2)}  "
              f"no cut {s['names_with_no_cut']}/{s['names']}  absent with a sighting "
              f"{s['absent_names_with_a_sighting']}/{s['absent_names']}  unlabelled hits {s['unlabelled_hits']}")
    print(f"through time: after each clip {out['through_time']['summary']['after_each_clip']}, "
          f"after each step {out['through_time']['summary']['per_step']}")
    print(f"written {path.relative_to(ROOT)} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
