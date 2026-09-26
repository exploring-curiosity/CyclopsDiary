#!/usr/bin/env python3
"""Score objects.py's decision for person-a's keys on the three live demo takes (2026-09-26, 19:40-19:43 UTC,
workspace home on Atlas), recomputed from the memory as track() does, read-only.

Labels: Claude, from 1-fps contact sheets (the frame at t + 0.5 s of each 1-s step), written before any score
was looked at: 1 = the keys in view, 0 = not, None = cannot tell / no frame (not scored).

  - P@1, P@support: the labelled steps of the three takes ranked by the name score
  - step precision and recall of the decision (score above corpus.split's cut over the whole workspace)
  - last known: the last step decided present, against the last step the keys are in view
  - the decision's false steps, by what the frame shows

    python3 scripts/eval_objects_live.py [--name "keys library card"]

--name scores that description instead of the stored tracker's words (same formula, same cut), on the live
takes and on the sample clips' keys labels (eval/cyclopsdiary_objects.json).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def spans(n, ones=(), nones=()):
    lab = [0] * n
    for a, b in ones:
        for i in range(a, b + 1):
            lab[i] = 1
    for i in nones:
        lab[i] = None
    return lab


LABELS = {                                             # footage id -> per-step keys label
    "phone-a-20260926T194008040": spans(93, [(39, 83)], [84, 85, 92]),              # laptops 0-38, palm 39-83, keyboard 86-91
    "phone-b-20260926T194201230": spans(87, [(0, 29), (36, 38), (43, 85)], [39, 86]),  # on the table; away 30-35, 40-42
    "phone-a-20260926T194204142": spans(45, [(22, 37)], [44]),                        # laptop 0-21, table 22-37, floor 38-43
}
WHAT = {"phone-a-20260926T194008040": {range(0, 39): "laptops, people", range(86, 93): "laptop keyboard"},
        "phone-b-20260926T194201230": {range(30, 36): "table edge, can", range(40, 43): "floor, chair"},
        "phone-a-20260926T194204142": {range(0, 22): "laptop keyboard", range(38, 44): "legs, floor"}}


def main() -> int:
    from cyclopsdiary import atlas, memory, objects
    from elidedb import corpus

    db = atlas.database()
    tr = objects.get(db, "person-a", "keys")
    mem = memory.load(db, tr["workspace"])
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else None
    words = tr["words"]
    if name:
        from cyclopsdiary import lastseen, tower
        tower.use_stack()
        words = objects.name_words(lastseen.tokenizer(), name)
    s = objects.name_scores(mem, words)
    cut = corpus.split(s)
    present = s > cut if cut is not None else np.zeros(mem.n, bool)

    rows = {}                                          # (footage, step) -> row
    for j in range(mem.n):
        rows[(mem.rec[j], int(round(float(mem.t0[j]))))] = j
    scored = [(fid, i) for fid, lab in LABELS.items() for i, v in enumerate(lab) if v is not None and (fid, i) in rows]
    lab = {(fid, i): LABELS[fid][i] for fid, i in scored}
    visible = [k for k in scored if lab[k] == 1]
    ranked = sorted(scored, key=lambda k: -s[rows[k]])
    seen = [k for k in scored if present[rows[k]]]
    tp = [k for k in seen if lab[k] == 1]
    false = [k for k in seen if lab[k] == 0]

    def when(k):
        f = mem.footage[k[0]]
        return f["started_at"] + timedelta(seconds=k[1])

    last_seen = max((k for k in rows if k[0] in LABELS and present[rows[k]]), key=when, default=None)
    last_true = max(visible, key=when)
    what = lambda k: next((v for r, v in WHAT.get(k[0], {}).items() if k[1] in r), "?")
    false_by = {}
    for k in false:
        false_by[what(k)] = false_by.get(what(k), 0) + 1
    sample = json.loads((ROOT / "eval" / "cyclopsdiary_objects.json").read_text())["visible"]["keys"]
    sk = [(fid, i) for fid, lab_ in sample.items() for i, v in enumerate(lab_) if v is not None and (fid, i) in rows]
    sv = [k for k in sk if sample[k[0]][k[1]] == 1]
    sr = sorted(sk, key=lambda k: -s[rows[k]])
    sseen = [k for k in sk if present[rows[k]]]
    sample_res = dict(p_at_1=float(sample[sr[0][0]][sr[0][1]] == 1),
                      p_at_support=sum(sample[k[0]][k[1]] == 1 for k in sr[:len(sv)]) / len(sv),
                      recall=sum(sample[k[0]][k[1]] == 1 for k in sseen) / len(sv),
                      false_steps=sum(sample[k[0]][k[1]] == 0 for k in sseen), negatives=len(sk) - len(sv))
    res = dict(at=datetime.now(timezone.utc).isoformat(), workspace=tr["workspace"], steps_in_memory=int(mem.n),
               name=name or tr["name"], sample=sample_res,
               cut=None if cut is None else float(cut), scored=len(scored), visible=len(visible),
               p_at_1=float(lab[ranked[0]] == 1), p_at_support=sum(lab[k] == 1 for k in ranked[:len(visible)]) / len(visible),
               precision=len(tp) / len(seen) if seen else None, recall=len(tp) / len(visible),
               false_steps=len(false), false_by_what=false_by,
               last_seen=dict(footage=last_seen[0], step=last_seen[1], at=when(last_seen).isoformat()) if last_seen else None,
               last_in_view=dict(footage=last_true[0], step=last_true[1], at=when(last_true).isoformat()),
               labels=LABELS)
    tag = ("_" + name.replace(" ", "-")) if name else ""
    out = ROOT / "eval_logs" / f"objects_live{tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"live takes, keys: P@1 {res['p_at_1']:.0f}  P@support {res['p_at_support']:.2f}  "
          f"precision {res['precision']:.2f}  recall {res['recall']:.2f}  ({len(tp)}/{len(visible)} in-view steps, "
          f"{len(false)} false: {false_by})")
    print(f"last seen {res['last_seen']}  |  truth {res['last_in_view']}")
    print(f"sample clips, keys: P@1 {sample_res['p_at_1']:.0f}  P@support {sample_res['p_at_support']:.2f}  "
          f"recall {sample_res['recall']:.2f}  false {sample_res['false_steps']}/{sample_res['negatives']}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
