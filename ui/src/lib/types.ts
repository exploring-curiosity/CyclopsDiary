// What the stage API (python/cyclopsdiary/stage.py) sends. Times are ISO 8601 strings in UTC.

export type Camera = { _id: string; person: string; kind: string; label?: string };

export type Footage = {
  _id: string;
  source: string;
  person: string;
  started_at: string;
  duration_s: number;
  steps: number;
  status: "live" | "ready" | "failed";
  mime?: string;
};

/** One sighting of a tracked object: a run of steps where its score stood apart (objects.py). */
export type Sighting = {
  _id: string;
  footage: string;
  source: string;
  person?: string;
  t0: number;
  t1: number;
  observed_at: string;
  until: string;
  place?: string[];
  end_place?: string[];
  by?: string[] | string;
  confirmed_at?: string | null;
};

export type TrackedObject = { name: string; aliases: string[]; examples: number; last: Sighting | null };

export type State = {
  workspace: string | null;
  person: string | null;
  people: { _id: string; name: string }[];
  cameras: Camera[];
  footage: Footage[];
  objects: TrackedObject[];
  trails: Record<string, Sighting[]>;
};

/** One second of a recording as the world model read it: its top words, and where the viewer's objects rank. */
export type Step = {
  i: number;
  t0: number;
  t1: number;
  observed_at?: string;
  words: string[];
  ranks: Record<string, number>;
};

export type ExplainLane = {
  footage: string;
  source: string;
  person: string;
  started_at: string;
  status: string;
  example: boolean;
  steps: { t0: number; t1: number; observed_at: string; score: number; present: boolean }[];
};

/** How the tracker found an object: every step's score, the scores' own cut, the trail it made. */
export type Explain = {
  object: string;
  owner: string;
  words: string[][];
  cut: number | null;
  scores: number[];
  lanes: ExplainLane[];
  trail: Sighting[];
  last: Sighting | null;
};

export type Clip = { footage: string; camera: string; t0: number; t1: number };

export type Evidence = { collection: string; find: unknown; sort?: unknown; then?: string };

/** A sighting as the router words it (router._sight). */
export type Seen = {
  camera: string;
  person?: string;
  seen_from: string;
  seen_until: string;
  near: string[];
  confirmed: boolean;
  clip: Clip;
};

export type Answer = {
  answer: string | null;
  tool?: string;
  args?: Record<string, unknown>;
  how: "rules" | "model" | "none";
  result?: {
    tool: string;
    name?: string;
    status?: string;
    last?: Seen;
    objects?: { name: string; last: Seen | null }[];
    evidence?: Evidence[];
  };
  ms: { route: number; tool?: number; total?: number };
};

export type FeedEvent =
  | { kind: "hello"; change_streams: boolean }
  | ({ kind: "step"; footage: string; source: string; person?: string } & Step)
  | ({ kind: "footage" } & Footage)
  | ({ kind: "sighting"; owner?: string } & Sighting)
  | { kind: "query" | "event"; [k: string]: unknown };
