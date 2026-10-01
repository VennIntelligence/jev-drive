#!/usr/bin/env python3
"""Per-card CARLA knee on the real G workload: N b2d_run workers of one examinee on one card, sweep N.

  carla_knee.py run --agent simlingo --n 6 8 --gpu 6 --cpus 144-167 --idx 20 --window 240
  carla_knee.py summary                      markdown table of every point in $DATA_DIR/runs/infra/knee-6000d

Each point starts one runner exactly as experiments/night_queue_4/lib/nq4_g_lane.py does (its launch(): nq3_b_cl10.sh / b2d_run, reduced CARLA
pools, the given core slice), on the G ghost seed-1 routes of that examinee in a fixed shuffled order, but writing to
its own directory (the G arms are untouched). After every worker has ticked, it measures for --window seconds:
game time advanced by all route attempts (1 tick = 0.05 s), split into useful (attempts that finished or were still
running at the end) and wasted (attempts that ended any other way, e.g. a server crash), completed routes, server
starts and deaths, RenderThread timeouts in the server logs, peak VRAM and mean utilisation of the card, and how busy
the core slice was. Then it stops the runner with SIGINT (routes cancelled, servers stopped) and waits for the card to
be free of its servers. One JSON row per point in results.jsonl.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/night_queue_4/lib",)]
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import nq4_g_lane as L   # noqa: E402

ROOT = L.DATA / "runs/infra/knee-6000d"
TICK_S = 0.05


def cpu_times(cores):
    out = {}
    for line in open("/proc/stat"):
        f = line.split()
        if f[0].startswith("cpu") and f[0][3:].isdigit() and int(f[0][3:]) in cores:
            v = list(map(int, f[1:]))
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out


def cores_of(spec):
    s = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        s |= set(range(int(a), int(b or a) + 1))
    return s


def gpu_sample(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split(",")
    return float(q[0]) / 1024, float(q[1])


def attempts(out):
    """attempt dir -> (ticks so far, final status or None while running)."""
    res = {}
    for d in (out / "attempts").glob("*/*"):
        try:
            t = json.loads((d / "heartbeat.json").read_text()).get("ticks") or 0
        except (OSError, ValueError):
            t = 0
        try:
            st = json.loads((d / "attempt.json").read_text()).get("status")
        except (OSError, ValueError):
            st = None
        res[str(d)] = (t, st)
    return res


def carla_on(gpu):
    n = 0
    for p in Path("/proc").iterdir():
        try:
            a = (p / "cmdline").read_bytes()
        except OSError:
            continue
        if b"CarlaUE4-Linux-Shipping" in a and f"-graphicsadapter={gpu}".encode() in a:
            n += 1
    return n


def point(a, n, rep):
    tag = f"{a.agent}-n{n}-{rep}"
    out = ROOT / tag
    L.OUT = ROOT / "lane"
    L.CPUS = a.cpus
    L.arm_dir = lambda c, v, s: out
    ids = list(L.requested(a.agent, "ghost", 1))
    random.Random(0).shuffle(ids)
    others = carla_on(a.gpu)
    t_launch = time.time()
    r = L.launch(dict(cand=a.agent, variant="ghost", seed=1), a.gpu, n, a.idx, 2 * n, ids)
    pid = r["pid"]
    print(time.strftime("%T"), tag, "runner", pid, "foreign CARLA on card:", others, flush=True)
    # warm-up: every worker has an attempt that ticks, then half a minute more
    deadline = time.time() + 900
    while time.time() < deadline:
        running = [t for t, st in attempts(out).values() if st is None and t > 0]
        if len(running) >= n:
            break
        time.sleep(10)
    warm = time.time() - t_launch
    time.sleep(30)
    cores = cores_of(a.cpus)
    a0, c0, t0 = attempts(out), cpu_times(cores), time.time()
    vram, util = [], []
    while time.time() - t0 < a.window:
        m, u = gpu_sample(a.gpu)
        vram.append(m)
        util.append(u)
        time.sleep(15)
    a1, c1, t1 = attempts(out), cpu_times(cores), time.time()
    useful = wasted = 0
    done = 0
    for k, (tk, st) in a1.items():
        d = tk - a0.get(k, (0, None))[0]
        if st in (None, "finished"):
            useful += d
        else:
            wasted += d
        if st == "finished" and a0.get(k, (0, None))[1] is None:
            done += 1
    busy = sum((c1[c][0] - c0[c][0]) - (c1[c][1] - c0[c][1]) for c in cores) / max(1, sum(c1[c][0] - c0[c][0] for c in cores))
    os.kill(pid, signal.SIGINT)
    for _ in range(90):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(2)
    for _ in range(60):
        if carla_on(a.gpu) <= others:
            break
        time.sleep(2)
    ev = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines() if l.strip()]
    starts = sum(e.get("event", e.get("kind")) == "server_start" for e in ev)
    deaths = sum(str(e.get("status", "")).startswith("server_died") for e in ev
                 if e.get("event", e.get("kind")) == "route_end")
    rt = 0
    for f in (out / "servers").glob("carla-*.log"):
        try:
            rt += "waiting for RenderThread" in f.read_text(errors="replace")
        except OSError:
            pass
    subprocess.run(["rm", "-rf", str(out / "viz")])
    w = t1 - t0
    row = dict(agent=a.agent, n=n, rep=rep, gpu=a.gpu, cpus=a.cpus, window_s=round(w), warmup_s=round(warm),
               foreign_carla=others, sim_s_per_s=round(useful * TICK_S / w, 3), wasted_sim_s_per_s=round(wasted * TICK_S / w, 3),
               per_worker=round(useful * TICK_S / w / n, 3), routes_done=done, routes_per_h=round(done * 3600 / w, 1),
               server_starts=starts, server_deaths=deaths, renderthread_timeouts=rt,
               vram_peak_gb=round(max(vram), 1), gpu_util=round(sum(util) / len(util)), cpu_busy=round(100 * busy),
               t=time.strftime("%F %T"))
    with (ROOT / "results.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def summary(_a):
    rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text().splitlines() if l.strip()]
    print("| agent | N | game s / wall s (useful) | wasted | per worker | routes / h | server starts / deaths / RT timeouts | VRAM peak GB | GPU util % | CPU busy % |")
    print("|:--|--:|--:|--:|--:|--:|:--|--:|--:|--:|")
    for r in sorted(rows, key=lambda r: (r["agent"], r["n"], r["rep"])):
        print(f"| {r['agent']} | {r['n']} | {r['sim_s_per_s']} | {r['wasted_sim_s_per_s']} | {r['per_worker']} | {r['routes_per_h']} "
              f"| {r['server_starts']} / {r['server_deaths']} / {r['renderthread_timeouts']} | {r['vram_peak_gb']} | {r['gpu_util']} | {r['cpu_busy']} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "summary"))
    ap.add_argument("--agent", default="simlingo", choices=list(L.NEED_GB))
    ap.add_argument("--n", type=int, nargs="+", default=[5, 8, 6, 7])
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--cpus", default="144-167")
    ap.add_argument("--idx", type=int, default=20)
    ap.add_argument("--window", type=float, default=240)
    ap.add_argument("--extend", type=int, default=10, help="largest N the automatic extension may reach (0: off)")
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "summary":
        summary(a)
    else:
        for n in a.n:
            point(a, n, a.rep)
        while a.extend:   # the best point is the largest N and still gains > 3 % over the next: try N + 2
            rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text().splitlines() if l.strip()]
            v = {r["n"]: r["sim_s_per_s"] for r in rows if r["agent"] == a.agent and r["rep"] == a.rep}
            top = max(v)
            lower = max((k for k in v if k < top), default=None)
            if lower is None or max(v, key=v.get) != top or v[top] < 1.03 * v[lower] or top + 2 > a.extend:
                break
            point(a, top + 2, a.rep)
