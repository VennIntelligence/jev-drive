#!/usr/bin/env python3
"""HUMAN-LABEL server: serves the labelling page (web/), one pack (hl_pack.py output) and an append-only label log.

    LABEL_PASSWORD=... python hl_server.py --pack DIR [--port 6006] [--host 127.0.0.1] [--env-file FILE]

API (all but the login need the cookie): GET /api/clips (manifest + per-clip label status), GET /api/clip/<id> (clip JSON),
GET /video/<id> (mp4, HTTP range), GET /api/labels[?clip=<id>] (records), POST /api/labels (a key press -> one record; {"undo": <id>}
appends a tombstone, nothing is ever deleted). Records go to <pack>/labels.jsonl, one JSON object per line, fsynced.
Login: POST /login {"password"} sets an HttpOnly cookie for 30 days (HMAC of the password: changing the password logs everyone out).
Every URL the page uses is relative, so the app also works under a path prefix (tailscale serve --set-path /label).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

from aiohttp import web

WEB = Path(__file__).resolve().parents[1] / "web"
REPO = Path(__file__).resolve().parents[3]
COOKIE, MAX_AGE = "hl_auth", 30 * 86400
KEYS = {"1": "observable", "2": "brake", "3": "turn", "4": "unwind", "9": "verdict_wrong", "n": "note", "d": "done"}
STATIC = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
log = logging.getLogger("hl")

LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Human label - login</title>
<script>if(!location.pathname.endsWith('/'))location.replace(location.pathname+'/'+location.search)</script>
<style>body{font:15px system-ui,sans-serif;background:#f4f5f7;display:grid;place-items:center;height:100vh;margin:0}
form{background:#fff;padding:28px 32px;border-radius:8px;box-shadow:0 1px 4px #0002;display:grid;gap:12px;width:280px}
input,button{font:inherit;padding:8px 10px;border:1px solid #ccd;border-radius:5px}button{background:#246;color:#fff;border:0;cursor:pointer}
#e{color:#b22;min-height:1.2em;font-size:13px}</style></head><body>
<form id="f"><b>jev-drive human labels</b><input id="p" type="password" placeholder="password" autofocus>
<button>Log in</button><div id="e"></div></form>
<script>f.onsubmit=async e=>{e.preventDefault();const r=await fetch('login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({password:p.value})});if(r.ok)location.reload();else{document.getElementById('e').textContent='Wrong password';p.select()}}</script>
</body></html>"""


def git_commit():
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "experiments/human_label"], capture_output=True,
                               text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Store:
    """The pack's manifest and the append-only label log (kept in memory, appended + fsynced per record)."""

    def __init__(self, pack: Path):
        self.pack = pack
        self.man = json.loads((pack / "manifest.json").read_text())
        self.clips = {c["id"]: c for c in self.man["clips"]}
        self.path = pack / "labels.jsonl"
        self.path.touch()
        self.recs = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
        self.lock = asyncio.Lock()
        self.commit = git_commit()

    def status(self):
        undone = {r["ref"] for r in self.recs if r.get("type") == "undo"}
        st = {}
        for r in self.recs:
            if r.get("type") != "label" or r["id"] in undone:
                continue
            s = st.setdefault(r["clip"], dict(n=0, done=False))
            if r["key"] == "d":
                s["done"] = True
            else:
                s["n"] += 1
        return st

    async def append(self, rec):
        async with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.recs.append(rec)
        return rec


def token(pw):
    return hmac.new(pw.encode(), b"jev-drive human_label v1", hashlib.sha256).hexdigest()


@web.middleware
async def auth(request, handler):
    app = request.app
    if app["token"] is None or request.path == "/login" or hmac.compare_digest(request.cookies.get(COOKIE, ""), app["token"]):
        return await handler(request)
    if request.path == "/":
        return web.Response(text=LOGIN, content_type="text/html", headers={"Cache-Control": "no-store"})
    return web.json_response({"error": "login required"}, status=401)


async def login(request):
    try:
        pw = str((await request.json()).get("password", ""))
    except (ValueError, AttributeError):
        pw = ""
    if request.app["token"] is None or hmac.compare_digest(token(pw), request.app["token"]):
        r = web.json_response({"ok": True})
        r.set_cookie(COOKIE, request.app["token"] or "", max_age=MAX_AGE, httponly=True, samesite="Lax", path="/")
        log.info("login ok from %s", request.headers.get("X-Forwarded-For") or request.remote)
        return r
    log.warning("login failed from %s", request.headers.get("X-Forwarded-For") or request.remote)
    await asyncio.sleep(1.0)
    return web.json_response({"error": "wrong password"}, status=403)


async def static(request):
    r = web.FileResponse(WEB / STATIC[request.path])
    r.headers["Cache-Control"] = "no-cache"
    return r


async def clips(request):
    s: Store = request.app["store"]
    m = {k: v for k, v in s.man.items() if k != "scenes"}
    return web.json_response(dict(m, status=s.status(), commit=s.commit, keys=KEYS))


def clip_id(request):
    cid = request.match_info["cid"].removesuffix(".mp4").removesuffix(".json")
    if cid not in request.app["store"].clips:
        raise web.HTTPNotFound(text="no such clip")
    return cid


async def clip(request):
    r = web.FileResponse(request.app["store"].pack / "clips" / f"{clip_id(request)}.json")
    r.headers["Cache-Control"] = "no-cache"
    return r


async def video(request):
    r = web.FileResponse(request.app["store"].pack / "video" / f"{clip_id(request)}.mp4")
    r.headers["Cache-Control"] = "private, max-age=86400"
    return r


async def labels_get(request):
    s: Store = request.app["store"]
    c = request.query.get("clip")
    return web.json_response([r for r in s.recs if c is None or r.get("clip") == c])


def num(x, nd=3):
    return None if x is None else round(float(x), nd)


async def labels_post(request):
    s: Store = request.app["store"]
    try:
        b = await request.json()
    except ValueError:
        raise web.HTTPBadRequest(text="json body expected")
    base = dict(id=uuid.uuid4().hex[:12], labeller=str(b.get("labeller") or "anon")[:40], wall=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                commit=s.commit, batch=s.man.get("batch"))
    if b.get("undo"):
        ref = next((r for r in s.recs if r["id"] == b["undo"] and r.get("type") == "label"), None)
        if ref is None:
            raise web.HTTPBadRequest(text="unknown label id")
        rec = dict(base, type="undo", ref=ref["id"], clip=ref["clip"], key=ref["key"])
    else:
        c, key = s.clips.get(b.get("clip")), str(b.get("key"))
        if c is None or key not in KEYS:
            raise web.HTTPBadRequest(text="unknown clip or key")
        note = b.get("note")
        rec = dict(base, type="label", clip=c["id"], scene=c["scene"], arm=c["arm"], tag=c["tag"], rollout=c["rollout"], key=key,
                   value=KEYS[key], t=num(b.get("t")), frame=None if b.get("frame") is None else int(b["frame"]),
                   k=None if b.get("k") is None else int(b["k"]), d_entry=num(b.get("d_entry"), 2), speed=num(b.get("speed"), 2),
                   cmd=b.get("cmd"), note=None if note is None else str(note)[:500])
    await s.append(rec)
    log.info("%s %s %s key %s t %s d_entry %s%s", rec["labeller"], rec["type"], rec["clip"], rec["key"], rec.get("t"), rec.get("d_entry"),
             f" note {rec['note']!r}" if rec.get("note") else "")
    return web.json_response(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--env-file", type=Path, help="KEY=VALUE lines (LABEL_PASSWORD) read before the environment")
    ap.add_argument("--no-auth", action="store_true", help="serve without a password (local testing only)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S")
    if a.env_file and a.env_file.exists():
        for line in a.env_file.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#"):
                os.environ.setdefault(k, v.strip().strip("'\""))
    pw = os.environ.get("LABEL_PASSWORD")
    if not pw and not a.no_auth:
        raise SystemExit("LABEL_PASSWORD is not set (env or --env-file); use --no-auth only for local tests")
    app = web.Application(middlewares=[auth], client_max_size=64 * 1024)
    app["token"] = None if a.no_auth else token(pw)
    app["store"] = st = Store(a.pack)
    app.router.add_post("/login", login)
    for p in STATIC:
        app.router.add_get(p, static)
    app.router.add_get("/api/clips", clips)
    app.router.add_get("/api/clip/{cid}", clip)
    app.router.add_get("/video/{cid}", video)
    app.router.add_get("/api/labels", labels_get)
    app.router.add_post("/api/labels", labels_post)
    log.info("pack %s: %d clips, %d label records, commit %s; http://%s:%d%s", a.pack, len(st.clips), len(st.recs), st.commit, a.host, a.port,
             " (no auth)" if a.no_auth else "")
    web.run_app(app, host=a.host, port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
