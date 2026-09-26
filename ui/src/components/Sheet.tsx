import { motion } from "motion/react";
import { type CSSProperties, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { axis, fit, seconds, type Axis } from "../lib/axis";
import { loop } from "../lib/pencil";
import { sheet, thumb, useThumbs } from "../lib/thumbs";
import { clock, secs } from "../lib/time";
import type { Camera, Explain, Footage, Sighting, Step, TrackedObject } from "../lib/types";
import { Trail } from "./Trail";

type Props = {
  cameras: Camera[];
  hues: string[];
  footage: Footage[];
  steps: Record<string, Step[]>;
  objects: TrackedObject[];
  trails: Record<string, Sighting[]>;
  explain: Explain | null;
  run: number;
  laneH: number;
  gapY: number;
  onPick: (f: Footage, t: number) => void;
};

/** Where things sit inside a lane, from its height. */
function metrics(laneH: number) {
  const barsH = 40, edgeH = 18;
  const picH = Math.round(Math.max(64, Math.min(200, laneH * 0.46)));
  const block = picH + edgeH + 10 + barsH;
  const picTop = Math.max(30, Math.round((laneH - block) / 2));
  return { picH, picTop, edgeH, barsTop: picTop + picH + edgeH + 10, barsH };
}
type M = ReturnType<typeof metrics>;

// the beats of an answer, in seconds from the moment it lands
export const BEATS = { bars: 0.1, spread: 0.8, cut: 1.0, ink: 1.6, loops: 1.8, print: 2.6 };

/** How the tracker's scores sit: every step's score by footage and second, their range and their own cut. */
function scored(ex: Explain | null) {
  if (!ex) return null;
  const at = new Map<string, { score: number; present: boolean }>();
  let lo = Infinity, hi = -Infinity;
  for (const l of ex.lanes) {
    if (l.example) continue;
    for (const s of l.steps) {
      at.set(`${l.footage}:${s.t0}`, { score: s.score, present: s.present });
      lo = Math.min(lo, s.score);
      hi = Math.max(hi, s.score);
    }
  }
  return { at, lo, hi: hi > lo ? hi : lo + 1, cut: ex.cut, object: ex.object };
}

/** The memory: every camera's recordings as strips of frames, one per second the model read, on one world-time
 *  axis; the tracker's sightings as grease-pencil loops; after a question, each second's score and the cut. */
export function Sheet(p: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = box.current!;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useThumbs();
  const ax = useMemo(() => axis(p.footage, fit(p.footage, w)), [p.footage, w]);
  const m = metrics(p.laneH);
  const sc = useMemo(() => scored(p.explain), [p.explain]);

  // frames for finished recordings, read once by a player nobody sees
  useEffect(() => {
    for (const f of p.footage) {
      const st = p.steps[f._id];
      if (f.status === "ready" && st?.length) void sheet(f._id, st.map((s) => s.t0));
    }
  }, [p.footage, p.steps]);

  // keep the newest second in view, unless the viewer has scrolled back
  const follow = useRef(true);
  useEffect(() => {
    const el = box.current;
    if (el && follow.current) el.scrollLeft = el.scrollWidth;
  }, [ax.width]);

  const height = p.cameras.length * p.laneH + Math.max(0, p.cameras.length - 1) * p.gapY;
  return (
    <div
      className="sheet"
      ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        follow.current = el.scrollLeft + el.clientWidth >= el.scrollWidth - 24;
      }}
    >
      <div className="sheet-inner" style={{ width: Math.max(ax.width, w), height }}>
        {ax.gaps.map((g) => (
          <div key={g.x} className="gap" style={{ left: g.x }}>
            <span>{g.label}</span>
          </div>
        ))}
        {p.cameras.map((c, i) => (
          <Lane
            key={c._id}
            first={i === 0}
            top={i * (p.laneH + p.gapY)}
            height={p.laneH}
            hue={p.hues[i % p.hues.length]}
            recs={p.footage.filter((f) => f.source === c._id)}
            steps={p.steps}
            ax={ax}
            m={m}
            sc={sc}
            run={p.run}
            objects={p.objects}
            trails={p.trails}
            camera={c._id}
            onPick={p.onPick}
          />
        ))}
        <Trail
          cameras={p.cameras}
          hues={p.hues}
          objects={p.objects}
          trails={p.trails}
          ax={ax}
          laneH={p.laneH}
          gapY={p.gapY}
          picTop={m.picTop}
          picH={m.picH}
          width={Math.max(ax.width, w)}
          asked={sc?.object}
          run={p.run}
          delay={sc ? BEATS.loops + 0.5 : 0.4}
        />
      </div>
    </div>
  );
}

type LaneProps = {
  first: boolean;
  top: number;
  height: number;
  hue: string;
  camera: string;
  recs: Footage[];
  steps: Record<string, Step[]>;
  ax: Axis;
  m: M;
  sc: ReturnType<typeof scored>;
  run: number;
  objects: TrackedObject[];
  trails: Record<string, Sighting[]>;
  onPick: (f: Footage, t: number) => void;
};

function Lane(p: LaneProps) {
  const { ax, m } = p;
  const x = (f: string) => p.ax.x.get(f) ?? 0;
  return (
    <div className="lane" style={{ top: p.top, height: p.height, "--hue": p.hue, "--pic-h": `${m.picH}px` } as CSSProperties}>
      {p.recs.map((f) => (
        <div key={f._id} className="strip" style={{ left: x(f._id), top: m.picTop, width: seconds(f) * ax.px }}>
          <time className="strip-at">{clock(f.started_at)}</time>
          {(p.steps[f._id] ?? []).map((s) => (
            <Cell key={s.t0} f={f} s={s} px={ax.px} picH={m.picH} onPick={p.onPick} />
          ))}
        </div>
      ))}
      {p.sc && <Bars {...p} x={x} />}
      <Loops {...p} x={x} />
    </div>
  );
}

function Cell({ f, s, px, picH, onPick }: { f: Footage; s: Step; px: number; picH: number; onPick: LaneProps["onPick"] }) {
  const url = thumb(f._id, s.t0);
  return (
    <button
      type="button"
      className="cell"
      style={{ left: s.t0 * px, width: px - 3 }}
      onClick={() => onPick(f, s.t0)}
      title={`${clock(secs(f.started_at) + s.t0)}: ${s.words.join(", ")}`}
    >
      <span className="pic" style={{ height: picH }}>
        {url ? <img src={url} alt="" draggable={false} /> : <span className="sec">{s.t0}</span>}
      </span>
      {px >= 34 && <span className="edge">{s.words[0] ?? ""}</span>}
    </button>
  );
}

/** Each second's score for the object asked about, rising left to right; the scores' own cut drawn across;
 *  the seconds above it inked in the pencil's colour. */
function Bars(p: LaneProps & { x: (f: string) => number }) {
  const sc = p.sc!, { m, ax } = p;
  const h = (v: number) => 2 + ((v - sc.lo) / (sc.hi - sc.lo)) * (m.barsH - 4);
  const bars = p.recs.flatMap((f) =>
    (p.steps[f._id] ?? []).flatMap((s) => {
      const v = sc.at.get(`${f._id}:${s.t0}`);
      if (!v) return [];
      const bx = p.x(f._id) + s.t0 * ax.px;
      return [{ key: `${f._id}:${s.t0}`, bx, v }];
    }),
  );
  const cutY = sc.cut === null ? null : m.barsH - h(sc.cut);
  return (
    <svg className="bars" key={p.run} width={ax.width} height={m.barsH} style={{ top: m.barsTop }} aria-hidden="true">
      {bars.map((b) => (
        <rect
          key={b.key}
          className={b.v.present ? "bar present" : "bar"}
          x={b.bx + 1}
          width={Math.max(2, ax.px - 5)}
          y={m.barsH - h(b.v.score)}
          height={h(b.v.score)}
          style={{ "--d": `${BEATS.bars + (b.bx / Math.max(1, ax.width)) * BEATS.spread}s`, "--ink": `${BEATS.ink}s` } as CSSProperties}
        />
      ))}
      {cutY !== null && (
        <>
          <motion.line
            className="cut"
            x1={0}
            x2={ax.width}
            y1={cutY}
            y2={cutY}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ delay: BEATS.cut, duration: 0.6, ease: "easeInOut" }}
          />
          {p.first && (
            <motion.text
              className="cut-note"
              x={ax.width - 6}
              y={cutY - 4}
              textAnchor="end"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: BEATS.cut + 0.5 }}
            >
              the scores' own cut
            </motion.text>
          )}
        </>
      )}
    </svg>
  );
}

/** The tracker's sightings of the viewer's objects, each a loop around its seconds; the latest one says so. */
function Loops(p: LaneProps & { x: (f: string) => number }) {
  const { m, ax } = p;
  const asked = p.sc?.object;
  const marks = p.objects.flatMap((o, k) =>
    (p.trails[o.name] ?? [])
      .filter((s) => s.source === p.camera && ax.x.has(s.footage))
      .map((s, i) => {
        const x0 = p.x(s.footage) + s.t0 * ax.px - 5 - k * 3, x1 = p.x(s.footage) + s.t1 * ax.px + 2 + k * 3;
        const y0 = m.picTop - 7 - k * 3, y1 = m.picTop + m.picH + 5 + k * 3;
        const last = o.last?._id === s._id;
        const isAsked = asked === o.name;
        const delay = isAsked ? BEATS.loops + i * 0.25 : 0.3 + i * 0.12;
        return { s, o, x0, x1, y0, y1, last, isAsked, delay, key: isAsked ? `${s._id}:${p.run}` : s._id };
      }),
  );
  return (
    <svg className="marks" width={ax.width} height={p.height} aria-hidden="true">
      {marks.map((k) => (
        <g key={k.key} className={asked && !k.isAsked ? "ring dim" : "ring"}>
          <motion.path
            d={loop(k.x0, k.y0, k.x1, k.y1, k.s._id)}
            filter="url(#wax)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ delay: k.delay, duration: 0.55, ease: "easeInOut" }}
          />
          <motion.text
            x={k.x1}
            y={k.y0 - 7}
            textAnchor="end"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: k.delay + 0.45 }}
          >
            {k.last ? `${k.o.name}, last seen ${clock(k.s.until)}` : k.o.name}
          </motion.text>
        </g>
      ))}
    </svg>
  );
}
