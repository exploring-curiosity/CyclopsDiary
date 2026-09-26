"""Where did it go? One object's sighting in, tracked through time, the footage of where it went out.

The demo: person-a's iPhone saw the keys picked up off the floor and put on the table; later person-b's
Pixel saw them taken off that table and put on a chair. person-a asks where the keys are.

  video in   an observation: which object, whose camera, when it was in view (world time). The partner's
             detector will hand these over; until then DEMO is written by hand from the sample clips
             (docs/memory/ledger.md L-5). That span of the camera's footage is the example.
  through    every camera's footage after it is asked for the example undone (query.py: reverse, scope
  time       after -- the keys put on the table, reversed, is the keys moved off it). The best answer is
             the moment the object moved; the query and its events are stored in MongoDB.
  video out  the recording that saw it move, from that moment to the recording's end: the last the
             cameras saw of it. Both ends play from the original files, never cut, copied or re-encoded.
             The world model's own top words at that last step say what is there.
"""
from __future__ import annotations

import html
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config, query

DEMO = dict(object="keys", person="person-a", source="phone-a",           # IMG_2658.MOV 3-6 s: put on the table
            start="2026-09-26T15:10:02+00:00", end="2026-09-26T15:10:05+00:00")


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def example(db, source: str, start: str, end: str) -> tuple:
    """The footage of `source` that has the object in view from start to end (world time)
    -> (footage row, t0, t1) in that footage's seconds."""
    a, b = _t(start), _t(end)
    if not b > a:
        raise ValueError("the observation must end after it starts")
    for f in db.footage.find({"source": source, "status": "ready", "started_at": {"$lte": a}}).sort("started_at", -1).limit(1):
        t0, t1 = (a - f["started_at"]).total_seconds(), (b - f["started_at"]).total_seconds()
        if t0 < float(f["duration_s"]):
            return f, t0, min(t1, float(f["duration_s"]))
    raise ValueError(f"no footage of {source} has {start} to {end} in it")


def run(db, obs: dict = DEMO, direction: str = "reverse", scope: str = "after", tokenizer=None) -> dict:
    f, t0, t1 = example(db, obs["source"], obs["start"], obs["end"])
    q = query.ask(db, f["workspace"], f["_id"], t0, t1, direction=direction, scope=scope, top=5,
                  label=f"{obs['object']} moved", asked_by=obs["person"])
    out = dict(obs=obs, example=dict(footage=f, t0=t0, t1=t1), query=q, events=query.events(db, q["_id"]))
    if out["events"]:
        moved = out["events"][0]
        g = db.footage.find_one({"_id": moved["footage"]})
        end_s = float(g["duration_s"])
        out.update(moved=moved, last=dict(footage=g, t0=float(moved["t0"]), t1=end_s,
                                          observed_at=g["started_at"] + timedelta(seconds=end_s),
                                          words=words(db, g, tokenizer) if tokenizer is not None else []))
    return out


def tokenizer():
    """The world model's own tokenizer, to read its word ids (tower.use_stack() first)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.get().model, local_files_only=True)


def words(db, g: dict, tok, n: int = 6) -> list:
    """The world model's own top words at g's last step, most probable first (its output head on that
    second of video; digits and word pieces skipped)."""
    st = db.steps.find_one({"footage": g["_id"]}, {"ids": 1}, sort=[("i", -1)])
    out = []
    for w in np.frombuffer(st["ids"], np.int32):
        s = tok.decode([int(w)]).strip()
        if s.isalpha() and len(s) > 2 and s.lower() not in {o.lower() for o in out}:
            out.append(s)
            if len(out) == n:
                break
    return out


def _src(f: dict, page_dir: Path) -> str:
    """media/<footage id><suffix> beside the page: a symlink to the original file (never a copy), so the page
    works from file:// and from a server rooted at the page's folder."""
    link = page_dir / "media" / (f["_id"] + Path(f["path"]).suffix.lower())
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink() or os.readlink(link) != f["path"]:
        link.unlink(missing_ok=True)
        link.symlink_to(f["path"])
    return html.escape(link.relative_to(page_dir).as_posix())


def page(r: dict, path) -> Path:
    """A page that plays the example and the answer from the original files (media fragments, no copies)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    utc = lambda d: d.strftime("%H:%M:%S") + " UTC"                                    # noqa: E731
    ex, obs = r["example"], r["obs"]
    f = ex["footage"]
    card_in = (f'<figure><video src="{_src(f, path.parent)}#t={ex["t0"]:g},{ex["t1"]:g}" data-t0="{ex["t0"]:g}" '
               f'data-t1="{ex["t1"]:g}" controls muted playsinline autoplay></video><figcaption><b>Video in</b> · {html.escape(obs["object"])}, seen by '
               f'{html.escape(obs["person"])} on {html.escape(f["source"])}, {utc(_t(obs["start"]))} to '
               f'{utc(_t(obs["end"]))}</figcaption></figure>')
    if "last" in r:
        m, last = r["moved"], r["last"]
        g = last["footage"]
        said = ", ".join(html.escape(w) for w in last["words"])
        card_out = (f'<figure><video src="{_src(g, path.parent)}#t={last["t0"]:g},{last["t1"]:g}" '
                    f'data-t0="{last["t0"]:g}" data-t1="{last["t1"]:g}" controls muted playsinline autoplay></video><figcaption><b>Video out</b> · moved at '
                    f'{utc(m["observed_at"])} on {html.escape(g["source"])} ({html.escape(str(g.get("person")))}), '
                    f'last seen {utc(last["observed_at"])}' + (f'<br>the model\'s words there: {said}' if said else "")
                    + '</figcaption></figure>')
        verdict = (f'{html.escape(obs["object"])} last seen on {html.escape(g["source"])} at '
                   f'{utc(last["observed_at"])}' + ("" if m["apart"] else " (best guess: nothing stood apart)"))
    else:
        card_out, verdict = "<figure><figcaption>not seen since</figcaption></figure>", "not seen since"
    q = r["query"]
    rows = "".join(f'<li>{"★" if e["apart"] else "·"} {utc(e["observed_at"])} {html.escape(e["source"])} '
                   f'{e["t0"]:g}–{e["t1"]:g} s, score {e["score"]:+.3f}</li>' for e in r["events"])
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Last seen</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --dim:#57534e; --line:#e7e5e4; --accent:#b45309; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --dim:#a8a29e; --line:#44403c; --accent:#f59e0b; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:24px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p {{ margin:0 0 16px; color:var(--dim); }}
.row {{ display:grid; grid-template-columns:1fr auto 1fr; gap:16px; align-items:center; }}
.arrow {{ color:var(--accent); font-size:28px; }}
figure {{ margin:0; }} video {{ width:100%; max-height:62vh; background:#000; border-radius:8px; }}
figcaption {{ font-size:13px; color:var(--dim); margin-top:6px; }}
.verdict {{ font-size:18px; margin:20px 0 8px; color:var(--accent); }}
ul {{ padding-left:18px; color:var(--dim); font-size:13px; }}
@media (max-width:640px) {{ .row {{ grid-template-columns:1fr; }} .arrow {{ transform:rotate(90deg); justify-self:center; }} }}
</style></head><body><main>
<h1>Where are the {html.escape(obs["object"])}?</h1>
<p>{html.escape(obs["person"])} asked. The example was their own footage; every camera after it was searched
for it undone ({html.escape(q["direction"])}, {q["searched"]["steps"]} steps of {q["searched"]["footage"]} recordings).</p>
<div class="row">{card_in}<div class="arrow">→</div>{card_out}</div>
<div class="verdict">{verdict}</div>
<ul>{rows}</ul>
<p>query {q["_id"]} in MongoDB · ★ stands apart from the rest</p>
</main><script>
// each player loops over its own span of the original file (media fragments alone are not honoured everywhere)
for (const v of document.querySelectorAll("video[data-t0]")) {{
  const t0 = +v.dataset.t0, t1 = +v.dataset.t1;
  v.addEventListener("loadedmetadata", () => {{ v.currentTime = t0; v.play().catch(() => {{}}); }});
  v.addEventListener("timeupdate", () => {{ if (v.currentTime < t0 - 0.25 || v.currentTime >= t1 - 0.05) v.currentTime = t0; }});
  if (v.readyState >= 1) {{ v.currentTime = t0; v.play().catch(() => {{}}); }}
}}
</script></body></html>
""")
    return path
