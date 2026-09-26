import { motion } from "motion/react";
import { useState } from "react";
import { media } from "../lib/api";
import { clock } from "../lib/time";
import type { Answer, Clip, Explain, TrackedObject } from "../lib/types";
import { BEATS } from "./Sheet";

export type QA = { question: string; answer?: Answer; error?: string };

type Props = {
  objects: TrackedObject[];
  qa: QA | null;
  explain: Explain | null;
  run: number;
  onAsk: (question: string) => void;
};

/** The diary: ask where something is; the answer comes back as a sentence, the moment itself, and the way it
 *  was found. */
export function Diary({ objects, qa, explain, run, onAsk }: Props) {
  const [text, setText] = useState("");
  const busy = !!qa && !qa.answer && !qa.error;
  const send = (q: string) => {
    if (!q.trim() || busy) return;
    setText(q);
    onAsk(q.trim());
  };
  return (
    <aside className="diary">
      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          send(text);
        }}
      >
        <label className="sr" htmlFor="question">
          Ask about your things
        </label>
        <input
          id="question"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={`Where are my ${objects[0]?.name ?? "keys"}?`}
          autoComplete="off"
        />
        <button type="submit" disabled={busy || !text.trim()}>
          Ask
        </button>
      </form>
      {objects.length > 0 && (
        <div className="suggest">
          {objects.map((o) => (
            <button key={o.name} type="button" disabled={busy} onClick={() => send(`Where are my ${o.name}?`)}>
              Where are my {o.name}?
            </button>
          ))}
        </div>
      )}
      {qa ? (
        <Entry key={run} qa={qa} explain={explain} />
      ) : (
        <p className="idle">
          Every second a camera sees becomes a step in MongoDB: the world model's memory of it, and its own words for
          what was there. {objects.length ? "Ask where something is to watch the answer being found." : "Name an object to follow it (bin/cyclopsdiary object add)."}
        </p>
      )}
    </aside>
  );
}

const NOT_ONE =
  "That isn't a question the diary answers yet. Ask where something is, what is at a place, or what happened between two times.";

function Entry({ qa, explain }: { qa: QA; explain: Explain | null }) {
  const a = qa.answer;
  const r = a?.result;
  const last = r?.last ?? r?.objects?.find((o) => o.last)?.last ?? undefined;
  const clip = last?.clip;
  const counted = explain?.lanes.filter((l) => !l.example).flatMap((l) => l.steps) ?? [];
  const at = explain ? BEATS.print : 0.15;
  const args = a?.args ? Object.values(a.args).map((v) => JSON.stringify(v)).join(", ") : "";
  return (
    <article className="entry">
      <p className="question">{qa.question}</p>
      {qa.error ? (
        <p className="trouble-text">The stage could not answer: {qa.error}</p>
      ) : !a ? (
        <p className="reading">Reading the memory…</p>
      ) : (
        <>
          <motion.p className="answer" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: at, duration: 0.6 }}>
            {a.answer ?? NOT_ONE}
          </motion.p>
          {clip && last && (
            <Print clip={clip} delay={at} caption={`${clip.camera}, ${clock(last.seen_from)} to ${clock(last.seen_until)}`} />
          )}
          {a.tool && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: at + 0.4, duration: 0.5 }}>
              <h2 className="how-title">How it was found</h2>
              <ol className="how">
                <li>
                  <span>
                    {a.how === "rules" ? "The rules" : "The model"} read the question as{" "}
                    <code>
                      {a.tool}({args})
                    </code>
                  </span>
                  <span className="ms">{a.ms.route} ms</span>
                </li>
                <li>
                  <span>MongoDB returned the sightings the tracker had stored</span>
                  {a.ms.tool !== undefined && <span className="ms">{a.ms.tool} ms</span>}
                </li>
                {explain && (
                  <li>
                    <span>
                      The tracker scored all {counted.length} seconds of memory by the model's own words for “
                      {explain.words[0]?.[0] ?? explain.object}”: {counted.filter((s) => s.present).length} stood apart,
                      above the scores' own cut.
                    </span>
                  </li>
                )}
                {clip && (
                  <li>
                    <span>
                      The latest sighting is the answer: {clip.camera}, {clip.t0}–{clip.t1} s into its recording.
                    </span>
                  </li>
                )}
              </ol>
              {!!r?.evidence?.length && (
                <details className="queries">
                  <summary>The MongoDB queries</summary>
                  <pre>
                    {r.evidence
                      .map(
                        (e) =>
                          `db.${e.collection}.find(${JSON.stringify(e.find)})${e.sort ? `.sort(${JSON.stringify(e.sort)})` : ""}`,
                      )
                      .join("\n")}
                  </pre>
                </details>
              )}
            </motion.div>
          )}
        </>
      )}
    </article>
  );
}

/** The answer's moment, printed: the original footage's span, looping, developing into view. */
function Print({ clip, caption, delay }: { clip: Clip; caption: string; delay: number }) {
  // a sighting's last second can run past the end of its recording: the span ends where the file does
  const end = (v: HTMLVideoElement) => Math.min(clip.t1, (v.duration || Infinity) - 0.1);
  const again = (v: HTMLVideoElement) => {
    v.currentTime = clip.t0;
    void v.play().catch(() => {});
  };
  return (
    <motion.figure
      className="print"
      initial={{ opacity: 0, filter: "grayscale(1) brightness(0.15) contrast(1.8)" }}
      animate={{ opacity: 1, filter: "grayscale(0) brightness(1) contrast(1)" }}
      transition={{ delay, duration: 1.8, ease: [0.3, 0, 0.2, 1] }}
    >
      <video
        src={`${media(clip.footage)}#t=${clip.t0},${clip.t1}`}
        muted
        playsInline
        autoPlay
        onTimeUpdate={(e) => {
          const v = e.currentTarget;
          if (v.currentTime < clip.t0 - 0.3 || v.currentTime >= end(v)) again(v);
        }}
        onPause={(e) => {
          const v = e.currentTarget;
          if (v.ended || v.currentTime >= end(v) - 0.1) again(v);
        }}
        onEnded={(e) => again(e.currentTarget)}
      />
      <figcaption>{caption}</figcaption>
    </motion.figure>
  );
}
