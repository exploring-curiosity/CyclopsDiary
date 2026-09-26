"""The camera service: phones stream their cameras into the memory from the browser, no app to install.

  GET  /?t=TOKEN          the capture page (camera.html): pick the camera, Start, Stop
  GET  /sources?t=TOKEN   the cameras (sources) the page offers
  WS   /ingest?t=TOKEN    {"source", "started_at" (ms, the phone's clock at record start), "mime"}, then the
                          recording's chunks as binary messages; {"stop": true} or a dropped socket ends it.
                          The service answers {"footage"} at once, {"steps", "bytes"} as it reads, and
                          {"done": ...} once the recording is closed (live.LiveSession).

A phone's browser gives the camera only to an HTTPS page, and a phone on public Wi-Fi or on cellular cannot
reach this machine directly, so the service is reached through a tunnel with a public HTTPS name:
`serve --live --tunnel` starts a Cloudflare quick tunnel (no account; its URL changes every run) and prints
the link. The token is new each run and travels in the link; without it every route answers 403.
"""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from . import config, live

PAGE = Path(__file__).with_name("camera.html")


def app(db, token: str, make_read=None, root: Path = live.ROOT) -> Starlette:
    def ok(conn) -> bool:
        return secrets.compare_digest(conn.query_params.get("t") or "", token)

    async def page(req):
        if not ok(req):
            return Response("this camera link needs its token\n", status_code=403)
        return HTMLResponse(PAGE.read_text(), headers={"Cache-Control": "no-store"})

    async def sources(req):
        if not ok(req):
            return Response(status_code=403)
        rows = await asyncio.to_thread(lambda: list(db.sources.find({}, {"person": 1, "kind": 1, "label": 1})
                                                    .sort("_id", 1)))
        return JSONResponse(rows)

    async def ingest(ws: WebSocket):
        if not ok(ws):
            await ws.close(code=1008)
            return
        await ws.accept()
        hello = await ws.receive_json()
        try:
            started = datetime.fromtimestamp(float(hello["started_at"]) / 1000.0, timezone.utc)
            s = await asyncio.to_thread(live.LiveSession, db, str(hello["source"]), started, str(hello["mime"]),
                                        make_read, root)
        except (KeyError, TypeError, ValueError) as e:
            await ws.send_json({"error": str(e)})
            await ws.close(code=1003)
            return
        print(f"live: {s.footage['source']} started {s.footage['_id']} ({s.footage['mime']})", flush=True)
        await ws.send_json({"footage": s.footage["_id"]})
        told = 0
        try:
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes"):
                    s.feed(m["bytes"])
                elif m.get("text") and json.loads(m["text"]).get("stop"):
                    break
                if s.steps != told:
                    told = s.steps
                    await ws.send_json({"steps": s.steps, "bytes": s.bytes})
        except WebSocketDisconnect:
            pass
        finally:
            done = await asyncio.to_thread(s.stop)
            print(f"live: {s.footage['source']} {done['footage']}: {done['steps']} steps, "
                  f"{done['duration_s']:g} s, {done['status']}" + (f" ({done['error']})" if done["error"] else ""),
                  flush=True)
        try:
            await ws.send_json({"done": done})
            await ws.close()
        except Exception:                                   # noqa: BLE001 -- the phone has already gone
            pass

    return Starlette(routes=[Route("/", page), Route("/sources", sources), WebSocketRoute("/ingest", ingest)])


def start(db, port: int = 8780, tunnel: bool = False, make_read=None) -> dict:
    """The service on 127.0.0.1:port in a background thread, and with tunnel=True a public HTTPS link to it.
    -> dict(local, public, token, tunnel process)"""
    import uvicorn
    token = secrets.token_urlsafe(12)
    server = uvicorn.Server(uvicorn.Config(app(db, token, make_read), host="127.0.0.1", port=port,
                                           log_level="warning", ws_max_size=16 << 20))
    threading.Thread(target=server.run, name="camserver", daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"the camera service did not start on port {port}")
    out = dict(local=f"http://127.0.0.1:{port}/?t={token}", public=None, token=token, tunnel=None)
    if tunnel:
        out["tunnel"], url = cloudflare(port)
        out["public"] = f"{url}/?t={token}"
        threading.Thread(target=_announce, args=(url,), name="tunnel dns", daemon=True).start()
    return out


def _announce(url: str, timeout: float = 300.0) -> None:
    """Say when the tunnel's name resolves. A phone that looks it up earlier is told it does not exist and may
    remember that for a minute or more (measured: 76.7 s on this Mac), so it is asked at a public resolver,
    which leaves the local caches alone."""
    host, t0 = url.split("//", 1)[1], time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            got = subprocess.run(["dig", "+short", "@1.1.1.1", host], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return
        if got.strip():
            print(f"cameras: the link resolves now ({time.monotonic() - t0:.0f} s): open it on the phones", flush=True)
            return
        time.sleep(2)


def cloudflare(port: int, timeout: float = 45.0):
    """A Cloudflare quick tunnel to 127.0.0.1:port: (process, https://<random>.trycloudflare.com). No account;
    it needs outbound port 7844, and its log goes to .local/logs/cloudflared.log."""
    exe = shutil.which("cloudflared")
    if exe is None:
        raise RuntimeError("cloudflared is not installed (brew install cloudflared)")
    log = config.ROOT / ".local" / "logs" / "cloudflared.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        p = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                             stdout=f, stderr=subprocess.STDOUT)
    end = time.monotonic() + timeout
    while time.monotonic() < end and p.poll() is None:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log.read_text(errors="replace"))
        if m:
            return p, m.group(0)
        time.sleep(0.25)
    p.terminate()
    raise RuntimeError(f"cloudflared gave no link in {timeout:g} s; see {log}")
