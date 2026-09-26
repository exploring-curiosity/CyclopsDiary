import { AnimatePresence, motion } from "motion/react";
import { type CSSProperties, useEffect, useRef, useState } from "react";
import { media } from "../lib/api";
import { capture } from "../lib/thumbs";
import { clock, secs } from "../lib/time";
import type { Camera, Footage, Sighting, Step, TrackedObject } from "../lib/types";
import { LivePlayer } from "./LivePlayer";

export type Seek = { footage: string; t: number; n: number };

type Props = {
  camera: Camera;
  hue: string;
  footage: Footage | null;
  steps: Step[];
  person: string;
  objects: TrackedObject[];
  trails: Record<string, Sighting[]>;
  seek: Seek | null;
  height: number;
  aspect: number;
  onAspect: (camera: string, a: number) => void;
};

export const CAPTION = 34;

const nth = (n: number) =>
  `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : (["th", "st", "nd", "rd"][n % 10] ?? "th")}`;

/** One camera: what it shows now (live, or its latest recording replayed), the viewer's objects the tracker
 *  sees in it at this moment, and the world model's own words for the second on screen. */
export function CameraTile(p: Props) {
  const [t, setT] = useState(0);
  const [trouble, setTrouble] = useState<string | null>(null);
  const video = useRef<HTMLVideoElement | null>(null);
  const f = p.footage;
  const live = f?.status === "live";
  const step = live ? p.steps.at(-1) : p.steps.find((s) => s.t0 <= t && t < s.t1);
  const at = live ? (step?.t0 ?? -1) : t;

  // the contact sheet's frame for this second, from this player
  useEffect(() => {
    const v = video.current, s = p.steps.find((x) => x.t0 + 0.2 <= t && t < x.t1);
    if (v && f && s) capture(v, f._id, s.t0);
  }, [t, f, p.steps]);

  // a replay asked for (a frame picked in the memory, an answer's clip)
  useEffect(() => {
    const v = video.current;
    if (!v || !p.seek || live || p.seek.footage !== f?._id) return;
    v.currentTime = p.seek.t;
    void v.play().catch(() => {});
  }, [p.seek, f?._id, live]);

  useEffect(() => setTrouble(null), [f?._id]);

  const labels = p.objects
    .map((o) => ({
      name: o.name,
      rank: step?.ranks[`${p.person}/${o.name}`],
      seen: !!f && (p.trails[o.name] ?? []).some((s) => s.footage === f._id && s.t0 <= at && at < s.t1),
    }))
    .filter((l) => l.seen) // the tracker's decision; the word's rank alone is evidence, not a sighting
    .sort((a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9));
  const names = new Set(p.objects.flatMap((o) => [o.name, ...o.aliases]).map((n) => n.toLowerCase()));
  const world = f && step ? clock(secs(f.started_at) + (live ? step.t0 : t)) : null;
  const style = { "--hue": p.hue, height: p.height, width: Math.round((p.height - CAPTION) * p.aspect) } as CSSProperties;
  const aspect = (a: number) => p.onAspect(p.camera._id, a);

  return (
    <figure className="tile" style={style}>
      <div className="frame">
        {!f ? (
          <p className="empty">Nothing recorded on {p.camera._id} yet.</p>
        ) : live ? (
          <LivePlayer footage={f._id} onTime={setT} onVideo={(v) => (video.current = v)} onAspect={aspect} />
        ) : (
          <video
            key={f._id}
            ref={video}
            src={media(f._id)}
            muted
            loop
            playsInline
            autoPlay
            preload="auto"
            onTimeUpdate={(e) => setT(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              if (v.videoWidth && v.videoHeight) aspect(v.videoWidth / v.videoHeight);
            }}
            onError={() => setTrouble("This browser can't play this recording's format.")}
            onPause={(e) => {
              // the browser pauses muted video it thinks is out of sight; the stage keeps every camera running
              const v = e.currentTarget;
              window.setTimeout(() => v.paused && !v.ended && void v.play().catch(() => {}), 400);
            }}
          />
        )}
        {trouble && <p className="trouble">{trouble}</p>}
        <div className="tile-head">
          <span className="cam">{p.camera._id}</span>
          {f && (
            <span className={live ? "state live" : "state"}>
              {live ? "Live" : "Replay"}
              {world && <time>{world}</time>}
            </span>
          )}
          <span className="who">{p.camera.person}</span>
        </div>
        <ul className="labels" aria-label="Your things in view">
          <AnimatePresence initial={false}>
            {labels.map((l) => (
              <motion.li
                key={l.name}
                layout
                className="label"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                title={
                  `The tracker has your ${l.name} here.` +
                  (l.rank !== undefined ? ` "${l.name}" is the model's ${nth(l.rank + 1)} word for this second.` : "")
                }
              >
                {l.name}
                {l.rank !== undefined && <small>#{l.rank + 1}</small>}
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      </div>
      <figcaption className="ticker">
        <AnimatePresence mode="wait" initial={false}>
          {step ? (
            <motion.span
              key={`${f?._id}:${step.t0}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              {step.words.map((w) => (
                <span key={w} className={names.has(w) ? "hit" : undefined}>
                  {w}
                </span>
              ))}
            </motion.span>
          ) : (
            <span className="quiet">{f ? "Waiting for the model's first words" : " "}</span>
          )}
        </AnimatePresence>
      </figcaption>
    </figure>
  );
}
