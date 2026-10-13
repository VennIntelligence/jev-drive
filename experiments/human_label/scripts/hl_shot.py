#!/usr/bin/env python3
"""Headless-Chrome smoke of the labelling page (CDP over aiohttp): log in, open a clip, seek, optionally press keys, save a screenshot.

    hl_shot.py --url http://127.0.0.1:8790/ --password-file <.env> --out shot.png [--clip ID] [--frame 80] [--keys 1,3,u,v]
Keys are sent as real key events, so they write labels: point it at a scratch copy of the pack, never the live one.
"""
import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time

import aiohttp


async def main(a):
    pw = next(x.split("=", 1)[1].strip() for x in open(a.password_file) if x.startswith("LABEL_PASSWORD="))
    prof = tempfile.mkdtemp(prefix="hl_chrome_")
    ch = subprocess.Popen([a.chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--mute-audio", f"--remote-debugging-port={a.cdp}",
                           f"--user-data-dir={prof}", f"--window-size={a.size}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as s:
            for _ in range(50):
                try:
                    pages = await (await s.get(f"http://127.0.0.1:{a.cdp}/json")).json()
                    ws_url = next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page")
                    break
                except (aiohttp.ClientError, StopIteration):
                    await asyncio.sleep(0.2)
            ws = await s.ws_connect(ws_url, max_msg_size=0)
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send_json(dict(id=n, method=method, params=params))
                while True:
                    m = json.loads((await ws.receive()).data)
                    if m.get("id") == n:
                        if "error" in m:
                            raise RuntimeError(m["error"])
                        return m.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
                return r.get("result", {}).get("value")

            w, h = map(int, a.size.split(","))
            await cmd("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            await cmd("Page.enable")
            await cmd("Page.navigate", url=a.url)
            await asyncio.sleep(1.5)
            print("login page:", await js("document.title"))
            await js(f"fetch('login',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{password:{json.dumps(pw)}}})}}).then(r=>r.status)")
            if a.clip:
                await js(f"localStorage.setItem('hl.cur', {json.dumps(a.clip)}); localStorage.setItem('hl.who', 'smoke-test')")
            await cmd("Page.reload")
            await asyncio.sleep(3)
            await js(f"new Promise(r => {{ seekFrame({a.frame}); setTimeout(r, 800); }})")
            for k in [x for x in a.keys.split(",") if x]:
                code = {"left": "ArrowLeft", "right": "ArrowRight", "space": " ", "enter": "Enter", "esc": "Escape"}.get(k, k)
                vk = {"ArrowLeft": 37, "ArrowRight": 39, " ": 32, "Enter": 13, "Escape": 27}.get(code) or ord(code.upper()[0])
                for typ in ("keyDown", "keyUp"):
                    await cmd("Input.dispatchKeyEvent", type=typ, key=code, windowsVirtualKeyCode=vk,
                              text=(code if len(code) == 1 else "\r" if code == "Enter" else "") if typ == "keyDown" else "")
                await asyncio.sleep(0.3 if len(code) == 1 and code.isalpha() else 0.6)
            await asyncio.sleep(1.0)
            state = await js("JSON.stringify({clip: S.cur, frame: frame(), t: simT(frame()), labels: activeLabels().map(r => [r.key, r.t, r.d_entry]),"
                             " ready: vid.readyState, dur: vid.duration, err: vid.error && vid.error.code, toast: document.getElementById('toast').textContent})")
            print("state:", state)
            png = await cmd("Page.captureScreenshot", format="png")
            open(a.out, "wb").write(base64.b64decode(png["data"]))
            print("screenshot ->", a.out)
    finally:
        ch.terminate()
        time.sleep(0.5)
        ch.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--password-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="")
    ap.add_argument("--frame", type=int, default=80)
    ap.add_argument("--keys", default="")
    ap.add_argument("--size", default="1440,900")
    ap.add_argument("--cdp", type=int, default=9333)
    ap.add_argument("--chrome", default="google-chrome")
    asyncio.run(main(ap.parse_args()))
