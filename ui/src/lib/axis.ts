import { gap, secs } from "./time";
import type { Footage } from "./types";

// The memory's x axis is world time, shared by every camera's lane, so a hand-off from one camera to the next
// reads left to right. Long stretches with nothing recorded shrink to a marker.

const JOIN_S = 60; //  recordings closer than this share one stretch of the sheet
export const GAP_W = 64;
export const PAD = 16;
const MIN_PX = 24, MAX_PX = 120;

export type Axis = { width: number; px: number; x: Map<string, number>; gaps: { x: number; label: string }[] };

export const seconds = (f: Footage) => Math.max(f.duration_s || 0, f.steps || 0, 1);

type Stretch = { start: number; end: number; ids: { id: string; at: number }[] };

function stretches(footage: Footage[]): Stretch[] {
  const out: Stretch[] = [];
  for (const f of [...footage].sort((a, b) => secs(a.started_at) - secs(b.started_at))) {
    const at = secs(f.started_at), cur = out.at(-1);
    if (!cur || at - cur.end > JOIN_S) out.push({ start: at, end: at + seconds(f), ids: [{ id: f._id, at }] });
    else {
      cur.ids.push({ id: f._id, at });
      cur.end = Math.max(cur.end, at + seconds(f));
    }
  }
  return out;
}

/** Pixels a second: the whole memory in the width there is, within what keeps a frame legible. */
export function fit(footage: Footage[], width: number) {
  const s = stretches(footage);
  if (!s.length || width <= 0) return 56;
  const total = s.reduce((n, x) => n + x.end - x.start, 0);
  return Math.max(MIN_PX, Math.min(MAX_PX, Math.floor((width - 2 * PAD - (s.length - 1) * GAP_W) / total)));
}

export function axis(footage: Footage[], px: number): Axis {
  const x = new Map<string, number>(), gaps: Axis["gaps"] = [];
  let x0 = PAD, prev: Stretch | null = null;
  for (const s of stretches(footage)) {
    if (prev) {
      gaps.push({ x: x0 + GAP_W / 2, label: gap(s.start - prev.end) });
      x0 += GAP_W;
    }
    for (const r of s.ids) x.set(r.id, x0 + (r.at - s.start) * px);
    x0 += (s.end - s.start) * px;
    prev = s;
  }
  return { width: x0 + PAD, px, x, gaps };
}
