import { useEffect, useRef, useState } from "react";
import { socket } from "./api";
import type { FeedEvent } from "./types";

export type FeedStatus = "connecting" | "live" | "quiet" | "down";

/** MongoDB's change streams, as the stage relays them for one person; reconnects when the stage restarts. */
export function useFeed(person: string | null, onEvent: (e: FeedEvent) => void): FeedStatus {
  const [status, setStatus] = useState<FeedStatus>("connecting");
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    if (!person) return;
    let ws: WebSocket | null = null, stop = false, retry = 0, timer = 0;
    const open = () => {
      ws = new WebSocket(socket(`/feed?person=${encodeURIComponent(person)}`));
      ws.onmessage = (m) => {
        const e = JSON.parse(m.data as string) as FeedEvent;
        if (e.kind === "hello") {
          retry = 0;
          setStatus(e.change_streams ? "live" : "quiet");
        }
        handler.current(e);
      };
      ws.onclose = () => {
        if (stop) return;
        setStatus("down");
        timer = window.setTimeout(open, Math.min(5000, 400 * 2 ** retry++));
      };
    };
    setStatus("connecting");
    open();
    return () => {
      stop = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, [person]);
  return status;
}
