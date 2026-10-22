"""Web front end of the exam control panel (stdlib only, Python 3.8+): the same Backend and control.json / loop.json protocol as the Tk
panel in lib/exam_panel.py (its docstring has the protocol); this module only replaces the window with one page (lib/exam_web.html).

    serve(backend, port)        # 127.0.0.1:<port> and <tailscale IPv4>:<port> (never 0.0.0.0)

API (JSON): GET /api/state?run=<dir> (everything the page shows: phone address, input status, loop, batches, the items of the shown
batch), GET /api/meta (title, sets, options), GET /api/catalog (backend.catalog(), for the custom pick), GET /api/official?run=,
POST /api/control {paused | playlist | option + value}, /api/act {kind, item} (refused when no loop runs), /api/order {run, order},
/api/new_batch {key, ids, force}, /api/continue {run, force}, /api/play_start {run, item, force} (continue, then drive that item
first), /api/delete {run}, /api/ensure_side. new_batch / continue answer {"confirm": text} when
backend.busy_other() reports another run; the page asks and posts again with force (stop_other() first), as the Tk panel does.
The shown batch is per request (?run=), so the desktop window and a phone can look at different batches.
"""
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from exam_panel import WHEEL_STATUS, Backend, Control, du, human_size, phone_urls, read_json

PAGE = Path(__file__).with_name("exam_web.html")
BOOT = "%x" % int(time.time() * 1000)      # server start id: an open page reloads itself when it changes (new code after --restart)


def wheel_status(path=WHEEL_STATUS):
    """Input status from the wheel daemon's status file as chips: [(text, level)], level ok / warn / bad."""
    st = read_json(path)
    if not st or time.time() - st.get("t", 0) > 5:
        return [("wheel daemon not running", "bad")]
    ph, kb = st.get("phone") or {}, st.get("keyboard") or {}
    pa, ka = ph.get("age_s"), kb.get("age_s")
    if not ph.get("clients"):
        phone = ("phone not connected", "bad")
    elif pa is None:
        phone = ("phone connected, no frame yet", "warn")
    else:
        phone = ("phone live" if pa <= 0.3 else "phone: last frame %.1f s ago" % pa, "ok" if pa <= 0.3 else "warn")
    key = ("keyboard active%s" % (" (%s)" % " ".join(kb["keys"]) if kb.get("keys") else ""), "ok") if ka is not None and ka < 0.3 \
        else ("keyboard idle", "idle")
    src = st.get("source")
    return [phone, key, ("driving: %s" % src, "ok") if src else ("driving: nobody (car brakes)", "warn")]


def tailscale_ip():
    try:
        return subprocess.run(["tailscale", "ip", "-4"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                              timeout=5).stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


class App(object):
    def __init__(self, backend, side=True):
        self.b, self.side = backend, side
        self.ctl = Control(backend.live)
        self.lock = threading.Lock()           # backend.selected is shared: items / official run under it
        self.url, self.lan = phone_urls()
        self.sizes, self.cache, self.notices = {}, {}, []
        c = self.ctl.read()                    # write option defaults once, as the Tk panel does
        if any(n not in c["options"] for n in backend.options):
            self.ctl.update(options=dict({n: o[1] for n, o in backend.options.items()}, **c["options"]))
        threading.Thread(target=self.sizer, daemon=True).start()

    def notice(self, msg, level="info"):
        if msg:
            self.notices = (self.notices + [{"t": time.time(), "msg": msg, "level": level}])[-20:]

    def cached(self, key, ttl, fn):
        t, v = self.cache.get(key, (0, None))
        if time.time() - t > ttl:
            v = fn()
            self.cache[key] = (time.time(), v)
        return v

    def sizer(self):
        """du of every batch once, of the running batch every 60 s (background: du can take seconds)."""
        while True:
            try:
                cur = self.b.loop_run()
                for r in self.b.runs():
                    if r not in self.sizes or (r == cur and time.time() - self.sizes[r][0] > 60):
                        self.sizes[r] = (time.time(), du(r))
            except Exception:  # noqa: BLE001  (a half-written batch must not stop the sizes)
                pass
            time.sleep(3)

    def batches(self):
        out = []
        for r in self.b.runs():
            try:
                i = self.b.batch_info(r)
            except Exception:  # noqa: BLE001
                i = dict(label="?", created="?", driven=0, total=0, mean=None, extra="")
            out.append(dict(run=r, name=os.path.basename(r), label=i["label"], created=i["created"], driven=i["driven"],
                            total=i["total"], mean=i["mean"], extra=i.get("extra", "")))
        return out

    def fmt(self, v):
        return "" if v is None else self.b.score_fmt % v

    def items(self, run):
        with self.lock:
            self.b.selected = run
            shown = self.b.run_dir()
            try:
                items = self.b.items()
            except Exception:  # noqa: BLE001  (never die on a half-written file)
                items = []
        return shown, [dict(id=it["id"], label=it.get("label") or it["id"], group=it["group"], scenario=it["scenario"],
                            human=it["human"], model=it["model"], human_s=self.fmt(it["human"]),
                            model_s=self.fmt(it["model"]), model_src=it.get("model_src") if it["model"] is not None else None,
                            status=it["status"]) for it in items]

    def state(self, run):
        st = read_json(Path(self.b.live) / "loop.json", {}) or {}
        alive = bool(self.b.loop_run())        # a "stopped" loop only drains its scoring: not running for the page
        c = self.ctl.read()
        shown, items = self.items(run)
        cur = self.b.loop_run()
        bat = self.cached("batches", 4, self.batches)
        for x in bat:
            x["size"] = human_size(self.sizes[x["run"]][1]) if x["run"] in self.sizes else "..."
            x["mean_s"] = self.fmt(x["mean"])
            x["current"] = x["run"] == cur
        return dict(
            phone=dict(url=self.url, lan=self.lan), inputs=wheel_status(),
            loop=dict(alive=alive, state=st.get("state") if alive else "not running", item=st.get("item") if alive else None,
                      msg=st.get("msg", "") if alive else "", run=st.get("run") if alive else None,
                      other="" if alive else self.cached("busy", 5, self.b.busy_other)),
            control=dict(paused=c["paused"], playlist=c["playlist"], options=c["options"]),
            shown=shown, batches=bat, items=items, notices=self.notices, boot=BOOT, t=time.time())

    def meta(self):
        dec = re.search(r"\.(\d)", self.b.score_fmt)
        return dict(title=self.b.title, batch_extra=self.b.batch_extra, score_max=getattr(self.b, "score_max", 1.0),
                    score_dec=int(dec.group(1)) if dec else 2, sets=[list(s) for s in self.b.batch_sets()],
                    options=[dict(name=n, label=o[0], default=o[1], choices=o[2] if len(o) > 2 else None) for n, o in self.b.options.items()],
                    catalog=type(self.b).catalog is not Backend.catalog)

    # ---- actions ----
    def take_over(self, force):
        other = self.b.busy_other()
        if other and not force:
            return {"confirm": "%s\n\nStop it now? The current item is aborted and can be driven again." % other}
        if other:
            self.notice(self.b.stop_other())
        return None

    def new_batch(self, key, ids=None, force=False):
        ask = self.take_over(force)
        if ask:
            return ask
        if self.ctl.read()["paused"]:
            self.ctl.update(paused=False)
        before = set(self.b.runs())
        try:
            ok, msg = self.b.new_batch(key, ids=ids) if ids is not None else self.b.new_batch(key)
        except TypeError:
            ok, msg = False, "This backend has no custom pick (new_batch takes no id list)."
        self.cache.clear()
        new = [r for r in self.b.runs() if r not in before]
        return dict(ok=ok, msg=msg, run=new[0] if new else None)

    def loop_pid(self):
        st = read_json(Path(self.b.live) / "loop.json", {}) or {}
        return st.get("pid") if self.b.loop_run() else None

    def continue_batch(self, run, force=False, item=None):
        """Continue `run`; with `item`, that item is driven first: its play action is posted once the new loop reports alive (a loop
        ignores actions older than itself, so a play queued before it starts is lost)."""
        if run not in self.b.runs():
            return dict(ok=False, msg="%s is not a batch of this panel" % run)
        ask = self.take_over(force)
        if ask:
            return ask
        old = self.loop_pid()
        if self.ctl.read()["paused"]:          # a pause left over from an earlier loop would hold the new one
            self.ctl.update(paused=False)
        ok, msg = self.b.continue_batch(run)
        self.cache.clear()
        if ok and item:
            threading.Thread(target=self.play_when_up, args=(run, item, old), daemon=True).start()
            msg += "\n\nDriving %s first." % item
        return dict(ok=ok, msg=msg, run=run)

    def play_when_up(self, run, item, old, wait=180):
        t = time.time()
        while time.time() - t < wait:
            pid = self.loop_pid()
            if pid and pid != old and self.b.loop_run() == run:
                self.ctl.act("play", item)
                return
            time.sleep(0.3)
        self.notice("the loop did not start within %d s; %s was not queued" % (wait, item), "bad")

    def delete(self, run):
        if run == self.b.loop_run():
            return dict(ok=False, msg="The loop is running on this batch:\n%s\n\nStop it first (Stop), then delete." % run)
        ok, msg = self.b.delete_batch(run)
        self.sizes.pop(run, None)
        self.cache.clear()
        return dict(ok=ok, msg=msg)

    def order(self, run, order):
        st = read_json(Path(self.b.live) / "loop.json", {}) or {}
        alive = bool(st.get("pid")) and os.path.exists("/proc/%s" % st["pid"])
        if alive and st.get("run") != run:          # viewing another batch never reorders the loop's
            return dict(ok=False)
        self.ctl.update(order=list(order))
        return dict(ok=True)

    def control(self, d):
        if "paused" in d:
            self.ctl.update(paused=bool(d["paused"]))
        if "playlist" in d:
            self.ctl.update(playlist=bool(d["playlist"]))
        o = self.b.options.get(d.get("option"))
        if o:                                  # a toggle takes a bool, a choice one of its values
            v = bool(d.get("value")) if len(o) < 3 else str(d.get("value"))
            vals = [x[0] if isinstance(x, (list, tuple)) else x for x in o[2]] if len(o) > 2 else None   # [value, label, group] entries
            if vals is not None and v not in vals:
                return dict(ok=False, msg="%s: %r is not one of %s" % (d["option"], v, vals))
            c = self.ctl.read()
            self.ctl.update(options=dict(c["options"], **{d["option"]: v}))
        return dict(ok=True, control=self.ctl.read())

    def official(self, run):
        with self.lock:
            self.b.selected = run
            try:
                return dict(run=self.b.run_dir(), text=self.b.official() or "(nothing yet)")
            except Exception as e:  # noqa: BLE001
                return dict(run=run, text="official table failed: %r" % e)

    def post(self, path, d):
        if path == "/api/control":
            return self.control(d)
        if path == "/api/act":
            if d.get("kind") not in ("play", "next", "restart", "stop"):
                return dict(ok=False, msg="unknown action")
            cur = self.b.loop_run()
            if not cur:                       # never queue silently: a later loop drops actions older than itself
                return dict(ok=False, msg="No loop is running: %s not sent. Continue a batch (or use an item's play button)." % d["kind"])
            if d["kind"] == "play" and d.get("run") and d["run"] != cur:
                return dict(ok=False, msg="The loop runs on another batch (%s)." % os.path.basename(cur))
            return dict(ok=True, id=self.ctl.act(d["kind"], d.get("item")))
        if path == "/api/order":
            return self.order(d.get("run"), d.get("order") or [])
        if path == "/api/new_batch":
            return self.new_batch(d.get("key"), d.get("ids"), bool(d.get("force")))
        if path == "/api/continue":
            return self.continue_batch(d.get("run"), bool(d.get("force")))
        if path == "/api/play_start":
            return self.continue_batch(d.get("run"), bool(d.get("force")), d.get("item"))
        if path == "/api/delete":
            return self.delete(d.get("run"))
        if path == "/api/ensure_side":
            msg = self.b.ensure_side() if self.side else ""
            self.notice(msg)
            return dict(ok=True, msg=msg)
        return None

    def get(self, path, q):
        run = q.get("run", [None])[0] or None
        if path == "/api/state":
            return self.state(run)
        if path == "/api/meta":
            return self.meta()
        if path == "/api/catalog":
            return dict(rows=self.b.catalog())
        if path == "/api/official":
            return self.official(run)
        return None


def handler(app):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def guard(self, fn):
            try:
                out = fn()
            except Exception as e:  # noqa: BLE001  (one failed call answers 500, the server stays up)
                return self.send(500, {"ok": False, "msg": "%s: %s" % (type(e).__name__, e)})
            self.send(404, {"ok": False, "msg": "no such endpoint"}) if out is None else self.send(200, out)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            self.guard(lambda: app.get(u.path, parse_qs(u.query)))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self.send(400, {"ok": False, "msg": "bad JSON"})
            self.guard(lambda: app.post(urlparse(self.path).path, d))
    return H


def serve(backend, port, hosts=None, side=True):
    """Serve the panel on 127.0.0.1 and the Tailscale IPv4 (or `hosts`); blocks."""
    app = App(backend, side)
    hosts = hosts or [h for h in ("127.0.0.1", tailscale_ip()) if h]
    servers, bound = [], []
    for h in hosts:
        try:
            servers.append(ThreadingHTTPServer((h, port), handler(app)))
            bound.append(h)
        except OSError as e:
            print("cannot bind %s:%d: %s" % (h, port, e), flush=True)
    if not servers:
        raise SystemExit("no address to serve on")
    for s in servers:
        s.daemon_threads = True
    print("%s panel on %s" % (backend.title, ", ".join("http://%s:%d/" % (h, port) for h in bound)), flush=True)
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    servers[0].serve_forever()
