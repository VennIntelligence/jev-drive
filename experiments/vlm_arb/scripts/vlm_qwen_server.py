"""Qwen3-VL-4B traffic-light server: zero-shot, one forward pass, option scoring, both cameras (plan 2026-10-03-vlm-vred.md).

  python vlm_qwen_server.py serve --port 8200 [--max-batch 1] [--res r1153] [--log FILE]
  python vlm_qwen_server.py supervise --cards 0,1,2 --run DIR     one `serve` per card, restarted if it dies; STOP file ends it
  python vlm_qwen_server.py bench [--res r1153]                    batch 1 / 2 / 4 service time on this card, JPEG in, answer out

Answers the Q-light question only, with the same prompt, preprocessing and scoring as `vlm_thin_stage.py` (`fwd_<res>`):
the first token after a forced `ANSWER:` is scored over the four options (argmax). No head, no truncation.

Wire format: POST /light with the JPEG bytes of the wide and the road frame concatenated, header `X-Sizes: n_wide,n_road`.
Reply JSON {ans, p[4], queue_ms, svc_ms, batch, depth}. GET /health -> 200 once warm.

One process = one card = the routes of that card. A single GPU worker thread takes whatever is queued (up to --max-batch
requests of the same frame size) and runs them as one batch (default 1: the forward is compute bound, batching
bought 3-6% throughput and delays the first request of a batch); HTTP handler threads only enqueue bytes and wait.
"""
import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_thin_common import LIGHTS, RES, Thin, log, sync  # noqa: E402

PORT0 = 8200                                           # port of card g = PORT0 + g


class Job:
    __slots__ = ("jpgs", "t_in", "done", "out")

    def __init__(self, jpgs):
        self.jpgs, self.t_in, self.done, self.out = jpgs, time.perf_counter(), threading.Event(), None


class Engine:
    def __init__(self, res, max_batch, log_path=None):
        import torch
        self.torch, self.res, self.max_batch = torch, RES[res], max_batch
        self.th = Thin()
        self.th.restore()                              # the whole model: no cut, final norm on
        self.q = queue.Queue()
        self.log = open(log_path, "a", buffering=1) if log_path else None
        self.ready = False
        self.n = 0

    def scores(self, xs):
        """Option logits [B, 4] of requests with identical frame sizes, as one batch."""
        t, th = self.torch, self.th
        cat = lambda k: t.cat([x[k] for x in xs])      # noqa: E731
        out = th.m.model(input_ids=cat("input_ids"), attention_mask=cat("attention_mask"), pixel_values=cat("pixel_values"),
                         image_grid_thw=cat("image_grid_thw"), mm_token_type_ids=cat("mm_token_type_ids"), use_cache=False)
        return out.last_hidden_state[:, -1].float() @ th.W.T

    def run_batch(self, jobs):
        t, th = self.torch, self.th
        t0 = time.perf_counter()
        xs = [th.prep(j.jpgs, self.res) for j in jobs]
        groups = {}
        for j, x in zip(jobs, xs):
            groups.setdefault(tuple(x["n_img"]), []).append((j, x))
        for grp in groups.values():
            lg = self.scores([x for _, x in grp])
            p = t.softmax(lg, -1).cpu().numpy()
            lg = lg.cpu().numpy()
            now = time.perf_counter()
            for (j, _), pi, li in zip(grp, p, lg):
                j.out = dict(ans=LIGHTS[int(li.argmax())], p=[round(float(v), 5) for v in pi], logits=[round(float(v), 3) for v in li],
                             queue_ms=round((t0 - j.t_in) * 1e3, 1), svc_ms=round((now - t0) * 1e3, 1), batch=len(grp),
                             depth=self.q.qsize())
        return t0

    def loop(self):
        t = self.torch
        with t.no_grad():
            while True:
                first = self.q.get()
                jobs = [first]
                while len(jobs) < self.max_batch:
                    try:
                        jobs.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                try:
                    self.run_batch(jobs)
                except Exception as e:  # noqa: BLE001 - one bad request must not end the server
                    for j in jobs:
                        j.out = dict(error=str(e)[:200])
                    log("batch failed: %r" % e)
                for j in jobs:
                    j.done.set()
                    self.n += 1
                    if self.log and "error" not in j.out:
                        self.log.write(json.dumps(dict(t=round(time.time(), 3), **{k: j.out[k] for k in ("ans", "queue_ms", "svc_ms", "batch", "depth")})) + "\n")

    def warm(self):
        import io
        import numpy as np
        from PIL import Image
        rng = np.random.default_rng(0)
        b = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (1208, 1928, 3), dtype=np.uint8)).save(b, "JPEG", quality=85)
        jp = [b.getvalue()] * 2
        with self.torch.no_grad():
            for k in (1, 2, 1, 2, 4, 1):
                self.run_batch([Job(jp) for _ in range(min(k, self.max_batch))])
        sync(self.torch)
        self.ready = True


def handler(eng):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body):
            b = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200 if eng.ready else 503, dict(ready=eng.ready, served=eng.n, depth=eng.q.qsize()))

        def do_POST(self):
            try:
                sizes = [int(s) for s in self.headers["X-Sizes"].split(",")]
                body = self.rfile.read(int(self.headers["Content-Length"]))
                job = Job([body[:sizes[0]], body[sizes[0]:sizes[0] + sizes[1]]])
                eng.q.put(job)
                job.done.wait()
                self._send(200 if "error" not in job.out else 500, job.out)
            except Exception as e:  # noqa: BLE001
                self._send(400, dict(error=str(e)[:200]))
    return H


def serve(a):
    eng = Engine(a.res, a.max_batch, a.log)
    threading.Thread(target=eng.loop, daemon=True).start()
    eng.warm()
    log("warm; serving on port %d (batch <= %d, %s)" % (a.port, a.max_batch, a.res))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(eng))
    srv.daemon_threads = True
    srv.serve_forever()


def supervise(a):
    """One server per card; a dead server is restarted; the STOP file (or SIGTERM) stops them by their own pids."""
    run = Path(a.run)
    (run / "servers").mkdir(parents=True, exist_ok=True)
    stop = run / "STOP"
    stop.unlink(missing_ok=True)
    procs, running = {}, [True]
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))

    def start(g):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        lf = open(run / "servers" / ("srv%d.log" % g), "a")
        procs[g] = subprocess.Popen([sys.executable, __file__, "serve", "--port", str(PORT0 + g), "--max-batch", str(a.max_batch),
                                     "--res", a.res, "--log", str(run / "servers" / ("req%d.jsonl" % g))], env=env, stdout=lf, stderr=subprocess.STDOUT)
        (run / "servers" / ("srv%d.pid" % g)).write_text(str(procs[g].pid))
        log("card %d: server pid %d" % (g, procs[g].pid))
    for g in a.cards:
        start(g)
    while running[0] and not stop.exists():
        time.sleep(3)
        for g, p in list(procs.items()):
            if p.poll() is not None:
                log("card %d: server exited rc=%s, restarting" % (g, p.returncode))
                start(g)
    for p in procs.values():
        p.terminate()
    for p in procs.values():
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
    log("servers stopped")


def bench(a):
    import io
    import numpy as np
    from PIL import Image
    from vlm_thin_common import load_frames, read_jpgs
    eng = Engine(a.res, 4)
    df = load_frames()
    sel = df.iloc[np.linspace(0, len(df) - 1, 48).astype(int)]
    jp = [read_jpgs(r) for r in sel.itertuples()]
    torch = eng.torch
    with torch.no_grad():
        eng.warm()
        for B in (1, 2, 3, 4):
            ms = []
            for k in range(0, len(jp) - B + 1, B):
                sync(torch)
                t = time.perf_counter()
                eng.run_batch([Job(j) for j in jp[k:k + B]])
                sync(torch)
                ms.append((time.perf_counter() - t) * 1e3)
            m = np.array(ms)
            log("batch %d: %.1f ms per batch (p50 %.1f p95 %.1f), %.1f ms per request" % (B, m.mean(), np.percentile(m, 50), np.percentile(m, 95), m.mean() / B))
        # answers: batch 4 against batch 1 on the same frames
        a1 = []
        for j in jp:
            job = Job(j)
            eng.run_batch([job])
            a1.append(job.out["ans"])
        a4 = []
        for k in range(0, len(jp), 4):
            jobs = [Job(j) for j in jp[k:k + 4]]
            eng.run_batch(jobs)
            a4 += [j.out["ans"] for j in jobs]
        log("answer agreement batch 4 vs batch 1: %d / %d" % (sum(x == y for x, y in zip(a1, a4)), len(a1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "supervise", "bench"])
    ap.add_argument("--port", type=int, default=PORT0)
    ap.add_argument("--max-batch", type=int, default=1)       # measured: 126 / 122 / 119 / 121 ms per request at batch 1 / 2 / 3 / 4
    ap.add_argument("--res", default="r1153")
    ap.add_argument("--log", default="")
    ap.add_argument("--cards", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    {"serve": serve, "supervise": supervise, "bench": bench}[a.cmd](a)
