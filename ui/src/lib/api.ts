import type { Answer, Explain, State, Step } from "./types";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

const q = (o: Record<string, string | undefined>) =>
  new URLSearchParams(Object.entries(o).filter((e): e is [string, string] => !!e[1])).toString();

export const getState = (person?: string) => json<State>(`/api/state?${q({ person })}`);
export const getSteps = (footage: string, person?: string) => json<Step[]>(`/api/steps?${q({ footage, person })}`);
export const explain = (person: string, name: string) => json<Explain>(`/api/object?${q({ person, name })}`);
export const ask = (person: string, question: string) =>
  json<Answer>("/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ person, question }),
  });

export const media = (footage: string) => `/media/${encodeURIComponent(footage)}`;
export const socket = (path: string) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
