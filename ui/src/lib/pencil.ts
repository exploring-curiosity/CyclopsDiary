// A grease-pencil loop, the mark a photographer makes on a contact sheet around the frames that matter.
// Deterministic for a given seed, so a loop keeps its shape across renders.

type P = [number, number];

function rng(seed: string) {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 16777619);
  return () => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    return (h >>> 0) / 4294967296;
  };
}

/** Catmull-Rom through the points, as cubic Béziers. */
function smooth(p: P[]) {
  const f = (n: number) => n.toFixed(1);
  let d = `M${f(p[0][0])},${f(p[0][1])}`;
  for (let i = 0; i < p.length - 1; i++) {
    const a = p[i - 1] ?? p[i], b = p[i], c = p[i + 1], e = p[i + 2] ?? c;
    d += ` C${f(b[0] + (c[0] - a[0]) / 6)},${f(b[1] + (c[1] - a[1]) / 6)} ${f(c[0] - (e[0] - b[0]) / 6)},${f(c[1] - (e[1] - b[1]) / 6)} ${f(c[0])},${f(c[1])}`;
  }
  return d;
}

/** A loop around a box, drawn clockwise from its top left and carried past the start, as a hand does. */
export function loop(x0: number, y0: number, x1: number, y1: number, seed: string) {
  const r = rng(seed);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, a = (x1 - x0) / 2, b = (y1 - y0) / 2;
  const n = Math.max(24, Math.round((a + b) / 14)), start = -2.35, turn = Math.PI * 2 + 0.5, phase = r() * 6;
  const pts: P[] = [];
  for (let i = 0; i <= n; i++) {
    const t = start + (turn * i) / n, c = Math.cos(t), s = Math.sin(t);
    const drift = 1 + 0.02 * Math.sin(t * 1.7 + phase) + (0.05 * i) / n; // the hand opens as it comes round
    pts.push([
      cx + a * drift * Math.sign(c) * Math.abs(c) ** 0.3 + (r() - 0.5) * 2,
      cy + b * drift * Math.sign(s) * Math.abs(s) ** 0.3 + (r() - 0.5) * 2,
    ]);
  }
  return smooth(pts);
}
