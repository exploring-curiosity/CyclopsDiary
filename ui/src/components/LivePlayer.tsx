import { useEffect, useRef, useState } from "react";
import { socket } from "../lib/api";

type Props = {
  footage: string;
  onTime: (t: number) => void;
  onVideo: (v: HTMLVideoElement | null) => void;
  onAspect: (a: number) => void;
};

/** A recording while it is being made: the phone's own bytes, relayed by the stage from the file as it grows,
 *  played as they arrive (Media Source Extensions). Nothing is re-encoded; it runs a second or two behind. */
export function LivePlayer({ footage, onTime, onVideo, onAspect }: Props) {
  const ref = useRef<HTMLVideoElement>(null);
  const report = useRef(onVideo);
  report.current = onVideo;
  const [trouble, setTrouble] = useState<string | null>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    report.current(video);
    setTrouble(null);
    const ms = new MediaSource();
    const url = URL.createObjectURL(ms);
    video.src = url;
    const queue: ArrayBuffer[] = [];
    let ws: WebSocket | null = null, sb: SourceBuffer | null = null, ended = false, gone = false;

    const pump = () => {
      if (gone || !sb || sb.updating) return;
      const next = queue.shift();
      if (next) {
        try {
          sb.appendBuffer(next);
        } catch (e) {
          const b = video.buffered;
          if ((e as DOMException).name === "QuotaExceededError" && b.length) {
            queue.unshift(next); // make room behind the playhead, then try again
            sb.remove(b.start(0), Math.max(b.start(0) + 1, video.currentTime - 10));
          } else setTrouble("The live picture stopped here; the recording itself goes on.");
        }
        return;
      }
      if (ended && ms.readyState === "open") {
        try {
          ms.endOfStream();
        } catch {
          /* already ended */
        }
      }
    };

    const follow = () => {
      const b = video.buffered;
      if (!b.length || !sb) return;
      const edge = b.end(b.length - 1);
      if (!ended && edge - video.currentTime > 3) video.currentTime = Math.max(b.start(b.length - 1), edge - 1);
      if (video.paused) void video.play().catch(() => {});
      if (!sb.updating && video.currentTime - b.start(0) > 90) sb.remove(b.start(0), video.currentTime - 30);
    };

    ms.addEventListener(
      "sourceopen",
      () => {
        ws = new WebSocket(socket(`/video/${encodeURIComponent(footage)}`));
        ws.binaryType = "arraybuffer";
        ws.onmessage = (m) => {
          if (typeof m.data !== "string") {
            queue.push(m.data as ArrayBuffer);
            pump();
            return;
          }
          const head = JSON.parse(m.data) as { type?: string; mime?: string; end?: boolean };
          if (head.end) {
            ended = true;
            pump();
            return;
          }
          const type = [head.type, head.mime].find((t): t is string => !!t && MediaSource.isTypeSupported(t));
          if (!type) {
            setTrouble(`This browser can't play ${head.type ?? head.mime} as it streams.`);
            ws?.close();
            return;
          }
          sb = ms.addSourceBuffer(type);
          sb.addEventListener("updateend", () => {
            follow();
            pump();
          });
          pump();
        };
        ws.onclose = () => {
          ended = true;
          pump();
        };
      },
      { once: true },
    );

    return () => {
      gone = true;
      ws?.close();
      report.current(null);
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
    };
  }, [footage]);

  return (
    <>
      <video
        ref={ref}
        muted
        playsInline
        autoPlay
        onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => {
          const v = e.currentTarget;
          if (v.videoWidth && v.videoHeight) onAspect(v.videoWidth / v.videoHeight);
        }}
      />
      {trouble && <p className="trouble">{trouble}</p>}
    </>
  );
}
