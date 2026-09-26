import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { CameraTile, type Seek } from "./components/CameraTile";
import { Diary, type QA } from "./components/Diary";
import { Eye } from "./components/Eye";
import { Horizon } from "./components/Horizon";
import { Sheet } from "./components/Sheet";
import * as api from "./lib/api";
import { type FeedStatus, useFeed } from "./lib/feed";
import { secs } from "./lib/time";
import type { Explain, FeedEvent, Footage, State, Step } from "./lib/types";

const HUES = ["#56C2D6", "#A48CF0", "#86C58A", "#E3A857"];
const GAP_Y = 16;

const FEED: Record<FeedStatus, string> = {
  connecting: "Connecting to the stage",
  live: "Live from MongoDB",
  quiet: "This MongoDB has no change streams: reload to see new data",
  down: "The stage stopped: start bin/cyclopsdiary stage",
};

const remembered = () => {
  try {
    return new URLSearchParams(location.search).get("person") ?? localStorage.getItem("stage.person");
  } catch {
    return null;
  }
};

const byTime = (a: Step, b: Step) => a.t0 - b.t0;
const merge = (old: Step[] = [], add: Step[]) => {
  const seen = new Map(old.map((s) => [s.t0, s]));
  for (const s of add) seen.set(s.t0, s);
  return [...seen.values()].sort(byTime);
};

export function App() {
  const [person, setPerson] = useState<string | null>(remembered);
  const [state, setState] = useState<State | null>(null);
  const [offline, setOffline] = useState(false);
  const [steps, setSteps] = useState<Record<string, Step[]>>({});
  const [explain, setExplain] = useState<Explain | null>(null);
  const [qa, setQA] = useState<QA | null>(null);
  const [run, setRun] = useState(0);
  const [seek, setSeek] = useState<Seek | null>(null);
  const [aspects, setAspects] = useState<Record<string, number>>({});
  const asked = useRef<Set<string>>(new Set());

  const load = useCallback(async () => {
    try {
      const s = await api.getState(person ?? undefined);
      setOffline(false);
      setState(s);
      if (!person && s.people[0]) setPerson(s.people[0]._id);
    } catch {
      setOffline(true);
    }
  }, [person]);
  useEffect(() => {
    void load();
  }, [load]);

  const choose = (p: string) => {
    try {
      localStorage.setItem("stage.person", p);
    } catch {
      /* private window: the choice lasts this visit */
    }
    history.replaceState(null, "", `?person=${encodeURIComponent(p)}`);
    asked.current = new Set();
    setSteps({});
    setExplain(null);
    setQA(null);
    setPerson(p);
  };

  // each recording's steps, once; the feed brings the ones that land later
  useEffect(() => {
    if (!state || !person) return;
    for (const f of state.footage) {
      if (asked.current.has(f._id)) continue;
      asked.current.add(f._id);
      api
        .getSteps(f._id, person)
        .then((list) => setSteps((prev) => ({ ...prev, [f._id]: merge(prev[f._id], list) })))
        .catch(() => asked.current.delete(f._id));
    }
  }, [state, person]);

  // sightings move the trails: read the state again, once things settle
  const later = useRef(0);
  const reload = useCallback(() => {
    window.clearTimeout(later.current);
    later.current = window.setTimeout(() => void load(), 400);
  }, [load]);

  const status = useFeed(person, (e: FeedEvent) => {
    if (e.kind === "step") {
      const { footage, t0, t1, i, words, ranks, observed_at } = e;
      setSteps((prev) => ({ ...prev, [footage]: merge(prev[footage], [{ t0, t1, i, words, ranks, observed_at }]) }));
      if (!state?.footage.some((f) => f._id === footage)) reload();
    } else if (e.kind === "footage") {
      const f = e as Footage;
      if (!state?.cameras.some((c) => c._id === f.source)) return reload();
      setState((s) => {
        if (!s) return s;
        const rest = s.footage.filter((x) => x._id !== f._id);
        const footage = f.status === "failed" ? rest : [...rest, { ...s.footage.find((x) => x._id === f._id), ...f }];
        return { ...s, footage: footage.sort((a, b) => secs(a.started_at) - secs(b.started_at)) };
      });
    } else if (e.kind === "sighting") reload();
  });
  const away = useRef(false);
  useEffect(() => {
    if (status === "down") away.current = true;
    if (status === "live" && away.current) {
      away.current = false;
      void load(); // whatever landed while the feed was away
    }
  }, [status, load]);

  const onAsk = async (question: string) => {
    if (!person) return;
    const n = run + 1;
    setRun(n);
    setExplain(null);
    setQA({ question });
    // the tracker's view of an object the question names is read alongside the answer
    const named = state?.objects.find((o) => question.toLowerCase().includes(o.name.toLowerCase()));
    const early = named ? api.explain(person, named.name).catch(() => null) : null;
    try {
      const a = await api.ask(person, question);
      const name = a.result?.name;
      const ex =
        name && a.result?.status !== "unknown" && (a.tool === "object_belief" || a.tool === "find_object")
          ? await (named?.name === name && early ? early : api.explain(person, name).catch(() => null))
          : null;
      setExplain(ex);
      setQA({ question, answer: a });
      const clip = a.result?.last?.clip;
      if (clip) setSeek({ footage: clip.footage, t: clip.t0, n });
    } catch (err) {
      setQA({ question, error: err instanceof Error ? err.message : String(err) });
    }
  };

  const world = useRef<HTMLElement>(null);
  const [h, setH] = useState(0);
  useLayoutEffect(() => {
    const el = world.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setH(el.clientHeight - 32));
    ro.observe(el);
    return () => ro.disconnect();
  }, [state === null]);

  const cams = state?.cameras ?? [];
  const narrow = typeof window !== "undefined" && window.innerWidth <= 1000;
  const laneH = narrow
    ? 300
    : Math.max(220, Math.floor((h - GAP_Y * Math.max(0, cams.length - 1)) / Math.max(1, cams.length)));
  const shown = (cam: string): Footage | null => {
    const mine = (state?.footage ?? []).filter((f) => f.source === cam);
    return (
      mine.findLast((f) => f.status === "live") ??
      (seek ? mine.find((f) => f._id === seek.footage) : undefined) ??
      mine.at(-1) ??
      null
    );
  };
  const anyLive = !!state?.footage.some((f) => f.status === "live");

  return (
    <div className="app">
      <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
        <filter id="wax">
          <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="2" seed="7" />
          <feDisplacementMap in="SourceGraphic" scale="1.8" />
        </filter>
      </svg>
      <header className="masthead">
        <h1 className="brand">
          <Eye live={anyLive} />
          CyclopsDiary
        </h1>
        {state && <Horizon footage={state.footage} cameras={state.cameras} people={state.people.length} hues={HUES} />}
        <div className="bar-end">
          {state && state.people.length > 0 && (
            <label className="who">
              Viewing as
              <select value={person ?? ""} onChange={(e) => choose(e.target.value)}>
                {state.people.map((x) => (
                  <option key={x._id} value={x._id}>
                    {x.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span className={`feed ${status}`}>{FEED[status]}</span>
        </div>
      </header>
      <main className="world" ref={world}>
        {offline ? (
          <p className="notice">
            The stage isn't answering. Start it with <code>bin/cyclopsdiary stage</code>, then reload.
          </p>
        ) : !state ? (
          <p className="notice">Opening the memory…</p>
        ) : !cams.length ? (
          <p className="notice">No cameras in this workspace yet. Add one with bin/cyclopsdiary source add.</p>
        ) : (
          <>
            <Sheet
              cameras={cams}
              hues={HUES}
              footage={state.footage}
              steps={steps}
              objects={state.objects}
              trails={state.trails}
              explain={explain}
              run={run}
              laneH={laneH}
              gapY={GAP_Y}
              onPick={(f, t) => setSeek({ footage: f._id, t, n: Date.now() })}
            />
            <div className="tiles" style={{ gap: GAP_Y }}>
              {cams.map((c, i) => {
                const f = shown(c._id);
                return (
                  <CameraTile
                    key={c._id}
                    camera={c}
                    hue={HUES[i % HUES.length]}
                    footage={f}
                    steps={f ? (steps[f._id] ?? []) : []}
                    person={person ?? ""}
                    objects={state.objects}
                    trails={state.trails}
                    seek={seek}
                    height={laneH}
                    aspect={aspects[c._id] ?? 9 / 16}
                    onAspect={(cam, a) => setAspects((x) => (Math.abs((x[cam] ?? 0) - a) < 0.01 ? x : { ...x, [cam]: a }))}
                  />
                );
              })}
            </div>
          </>
        )}
      </main>
      <Diary objects={state?.objects ?? []} qa={qa} explain={explain} run={run} onAsk={onAsk} />
    </div>
  );
}
