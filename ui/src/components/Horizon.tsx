import { useEffect, useState } from "react";
import { gap, secs } from "../lib/time";
import type { Camera, Footage } from "../lib/types";

// The model's own memory, from the pinned setting: a window of 8,192 tokens, 360 video tokens a second at the
// phones' frame size (two frames of 18 x 10 merged patches; ledger L-10's setting).
const WINDOW_TOKENS = 8192, TOKENS_PER_S = 360;
const MODEL_S = WINDOW_TOKENS / TOKENS_PER_S;
const W = 340;

/** The whole memory to scale, from the first recording to now, with the model's own memory drawn beside it. */
export function Horizon({ footage, cameras, people, hues }: { footage: Footage[]; cameras: Camera[]; people: number; hues: string[] }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 5000);
    return () => window.clearInterval(t);
  }, []);
  if (!footage.length) return null;
  const start = Math.min(...footage.map((f) => secs(f.started_at)));
  const span = Math.max(1, now - start);
  const x = (t: number) => ((t - start) / span) * W;
  const lane = new Map(cameras.map((c, i) => [c._id, i]));
  const seconds = footage.reduce((n, f) => n + (f.steps || 0), 0);
  const tokens = seconds * TOKENS_PER_S;
  return (
    <div className="horizon" title="Every second every camera saw is a row in MongoDB; the model alone keeps its last 8,192 tokens">
      <svg width={W + 60} height={30} aria-hidden="true">
        <line className="axis" x1={0} x2={W} y1={14} y2={14} />
        {footage.map((f) => (
          <rect
            key={f._id}
            x={x(secs(f.started_at))}
            y={5 + 5 * (lane.get(f.source) ?? 0)}
            width={Math.max(2, x(secs(f.started_at) + (f.duration_s || f.steps || 1)) - x(secs(f.started_at)))}
            height={8}
            style={{ fill: hues[(lane.get(f.source) ?? 0) % hues.length] }}
          />
        ))}
        <path className="model" d={`M${W - Math.max(1.5, (MODEL_S / span) * W)},22 v4 H${W} v-4`} />
        <text x={W + 4} y={18}>now</text>
      </svg>
      <span className="horizon-text">
        <b>MongoDB</b> {seconds.toLocaleString()} s of video ({(tokens / 1e3).toFixed(0)}k tokens) · {footage.length} recordings ·{" "}
        {cameras.length} cameras · {people} people · over {gap(span).replace("+", "")}
        <br />
        <b>the model alone</b> its last {WINDOW_TOKENS.toLocaleString()} tokens ≈ {Math.round(MODEL_S)} s, the tick at the right
      </span>
    </div>
  );
}
