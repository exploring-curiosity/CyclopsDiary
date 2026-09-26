import { motion } from "motion/react";
import type { Axis } from "../lib/axis";
import { clock, secs } from "../lib/time";
import type { Camera, Sighting, TrackedObject } from "../lib/types";

type Props = {
  cameras: Camera[];
  hues: string[];
  objects: TrackedObject[];
  trails: Record<string, Sighting[]>;
  ax: Axis;
  laneH: number;
  gapY: number;
  picTop: number;
  picH: number;
  width: number;
  asked?: string;
  run: number;
  delay: number;
};

const words = (s: Sighting) => (s.end_place?.length ? s.end_place : (s.place ?? [])).slice(0, 2).join(" · ");

/** The graph the memory builds for one object: its sightings in world time as nodes on each camera's lane, joined
 *  in order, so a hand-off to another person's camera crosses lanes; the last one runs on to now. */
export function Trail(p: Props) {
  const lane = new Map(p.cameras.map((c, i) => [c._id, i]));
  const name = p.asked ?? p.objects[0]?.name;
  if (!name) return null;
  const pts = (p.trails[name] ?? [])
    .filter((s) => p.ax.x.has(s.footage) && lane.has(s.source))
    .sort((a, b) => secs(a.observed_at) - secs(b.observed_at))
    .map((s) => {
      const i = lane.get(s.source)!;
      return {
        s,
        x: p.ax.x.get(s.footage)! + ((s.t0 + s.t1) / 2) * p.ax.px,
        y: i * (p.laneH + p.gapY) + p.picTop + p.picH / 2,
        hue: p.hues[i % p.hues.length],
      };
    });
  if (!pts.length) return null;
  const last = pts.at(-1)!;
  const step = Math.min(0.22, 2.4 / pts.length);
  return (
    <svg className="trail" width={p.width} height={p.cameras.length * (p.laneH + p.gapY)} aria-hidden="true">
      <g key={`${name}:${p.run}`}>
        {pts.slice(1).map((b, k) => {
          const a = pts[k], mx = (a.x + b.x) / 2, hand = a.s.source !== b.s.source;
          return (
            <g key={`e${b.s._id}`} className={hand ? "edge hand" : "edge"}>
              <motion.path
                d={`M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ delay: p.delay + k * step, duration: step * 1.4, ease: "easeInOut" }}
              />
              {hand && (
                <motion.text
                  x={mx}
                  y={(a.y + b.y) / 2 - 6}
                  textAnchor="middle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: p.delay + (k + 1) * step }}
                >
                  {`→ ${b.s.person ?? b.s.source}'s camera`}
                </motion.text>
              )}
            </g>
          );
        })}
        {pts.map((q, k) => {
          const show = k === 0 || k === pts.length - 1 || pts[k - 1].s.source !== q.s.source;
          return (
            <motion.g
              key={`n${q.s._id}`}
              className={q === last ? "node last" : "node"}
              initial={{ opacity: 0, scale: 0.4 }}
              animate={{ opacity: 1, scale: 1 }}
              style={{ transformOrigin: `${q.x}px ${q.y}px` }}
              transition={{ delay: p.delay + Math.max(0, k - 1) * step + (k ? step : 0) }}
            >
              <circle cx={q.x} cy={q.y} r={q === last ? 7 : 4.5} style={{ fill: q.hue }} />
              {show && q !== last && (
                <text x={q.x} y={q.y + 18} textAnchor="middle">
                  {`${clock(q.s.observed_at)}${words(q.s) ? ` · ${words(q.s)}` : ""}`}
                </text>
              )}
            </motion.g>
          );
        })}
        <motion.g
          className="now"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: p.delay + pts.length * step }}
        >
          <circle className="pulse" cx={last.x} cy={last.y} r={7} />
          <line x1={last.x + 8} y1={last.y} x2={p.width - 12} y2={last.y} />
          <text x={last.x + 12} y={last.y - 10}>
            {`${name}: last seen ${clock(last.s.until)}${words(last.s) ? `, near ${words(last.s)}` : ""}`}
          </text>
          <text x={p.width - 12} y={last.y - 10} textAnchor="end">
            now
          </text>
        </motion.g>
      </g>
    </svg>
  );
}
