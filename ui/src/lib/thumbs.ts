import { useSyncExternalStore } from "react";
import { media } from "./api";

// Frames for the memory's contact sheet, drawn from the players in this page: one per step, at its middle.
// Display only -- nothing is stored, and the footage itself is never touched.

const store = new Map<string, Map<number, string>>();
const listeners = new Set<() => void>();
let version = 0;

const W = 120, H = 320;

function put(footage: string, t0: number, url: string) {
  let m = store.get(footage);
  if (!m) store.set(footage, (m = new Map()));
  m.set(t0, url);
  version++;
  listeners.forEach((l) => l());
}

export const thumb = (footage: string, t0: number) => store.get(footage)?.get(t0);

export function useThumbs() {
  return useSyncExternalStore(
    (l) => (listeners.add(l), () => listeners.delete(l)),
    () => version,
  );
}

/** The frame a player shows now, as the step's picture (cropped to fill, like a contact sheet). */
export function capture(video: HTMLVideoElement, footage: string, t0: number) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh || thumb(footage, t0)) return;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const s = Math.max(W / vw, H / vh);
  try {
    c.getContext("2d")!.drawImage(video, (W - vw * s) / 2, (H - vh * s) / 2, vw * s, vh * s);
    put(footage, t0, c.toDataURL("image/jpeg", 0.72));
  } catch {
    /* a frame the browser will not hand over: the cell keeps its second */
  }
}

const once = (el: HTMLElement, ok: string) =>
  new Promise<boolean>((res) => {
    const done = (v: boolean) => () => {
      el.removeEventListener(ok, yes);
      el.removeEventListener("error", no);
      res(v);
    };
    const yes = done(true), no = done(false);
    el.addEventListener(ok, yes, { once: true });
    el.addEventListener("error", no, { once: true });
  });

const started = new Set<string>();

/** A finished recording's frames, read one step at a time by a player nobody sees. */
export async function sheet(footage: string, t0s: number[]) {
  const want = t0s.filter((t) => !thumb(footage, t));
  const key = `${footage}:${want.length}`;
  if (!want.length || started.has(key)) return;
  started.add(key);
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.src = media(footage);
  if (!(await once(v, "loadeddata"))) return;
  for (const t0 of want) {
    v.currentTime = Math.min(t0 + 0.5, Math.max(0, v.duration - 0.05));
    if (!(await once(v, "seeked"))) break;
    capture(v, footage, t0);
  }
  v.removeAttribute("src");
  v.load();
}
