#!/usr/bin/env python
"""Scaling ladder for the Bench2Drive harness: how aggregate tick rate and per-worker cost change with the number
of CARLA servers per GPU. docs/bench2drive-cost.md ("Harness cost and layout, 2026-09-25") has the results.

Every worker runs the SAME route with the same seed and --max-ticks, so a rung measures N copies of one fixed piece
of work and per-worker cost is comparable across rungs. Servers are b2d_run.Server objects (our launch flags:
-RenderOffScreen, Epic, -graphicsadapter=<gpu>), started staggered and reused across rungs; each worker is a
b2d_run.Runner with one lent server, so routes run through exactly the production path (b2d_route.py ->
leaderboard + scenario_runner + traffic manager in the route process).

A sampler thread records every --sample-s: cgroup cpu.stat (usage, throttling) of the whole container, the load
average, CPU / RSS / threads of every process we started (CarlaUE4 server binary vs b2d_route.py client), GPU
memory and utilisation, and each worker's heartbeat tick count. A rung's numbers come from its steady window: the
span in which every worker is past warm-up and none has finished, so route setup and teardown are excluded.

    $DATA_DIR/envs/carla/bin/python scripts/b2d_scale.py --out $DATA_DIR/runs/infra-acceptance/scale/t12-alp4 \
        --route-id 1773 --gpus 4 --rungs 1,2,4,8 --max-ticks 1200 --cpus 52-96 -- --rig alp4 --decimate 2

Arguments after `--` go to b2d_run.py unchanged (rig, decimation, optimisation flags, --agent ...).
Python 3.8: runs in envs/carla.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2d_run  # noqa: E402

CLK = os.sysconf("SC_CLK_TCK")
CGROUP = Path("/sys/fs/cgroup")


def parse_args():
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--route-id", required=True)
    p.add_argument("--gpus", default="4", help="comma-separated CUDA indices (== -graphicsadapter on this box)")
    p.add_argument("--rungs", default="1,2,4", help="servers per GPU, one rung each, ascending")
    p.add_argument("--max-ticks", type=int, default=1200)
    p.add_argument("--warm-ticks", type=int, default=40, help="a worker enters the steady window past this tick")
    p.add_argument("--stagger-s", type=float, default=15.0)
    p.add_argument("--index-lo", type=int, default=740)
    p.add_argument("--index-hi", type=int, default=789)
    p.add_argument("--cpus", default="", help="pin this process and everything it starts to these CPUs (taskset list)")
    p.add_argument("--sample-s", type=float, default=2.0)
    p.add_argument("--restart-servers", action="store_true", help="fresh servers for every rung")
    a = p.parse_args(argv)
    a.passthrough = rest
    a.gpus = [int(g) for g in a.gpus.split(",")]
    a.rungs = [int(n) for n in a.rungs.split(",")]
    return a


def cpu_list(spec):
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


class Indices(object):
    """CARLA server indices from our block. An index is never reused within one invocation (a stopped server's
    ports sit in TIME_WAIT and CARLA dies on a busy port), and the next invocation continues where this one stopped,
    via a small state file, so back-to-back runs do not land on each other's sockets either."""

    def __init__(self, lo, hi, state):
        self.lo, self.hi, self.state = lo, hi, Path(state)
        try:
            self.next = int(self.state.read_text())
        except (OSError, ValueError):
            self.next = lo

    def take(self):
        for _ in range(self.hi - self.lo + 1):
            i = self.next if self.lo <= self.next <= self.hi else self.lo
            self.next = i + 1
            self.state.write_text(str(self.next))
            port = b2d_run.PORT_BASE + b2d_run.PORT_STRIDE * i
            tm = b2d_run.TM_BASE + b2d_run.PORT_STRIDE * i
            if all(b2d_run.port_free(q) for q in (port, port + 1, port + 2, tm)):
                return i
        raise RuntimeError("no free CARLA index in %d-%d" % (self.lo, self.hi))


def read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def proc_table():
    """pid -> (ppid, comm, cpu ticks, rss kB, threads) for every process the container can see."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        s = read("/proc/%s/stat" % d)
        if not s:
            continue
        r = s.rfind(")")
        comm, f = s[s.find("(") + 1:r], s[r + 2:].split()
        # fields after comm: state(0) ppid(1) ... utime(11) stime(12) ... num_threads(17) ... rss pages(21)
        out[int(d)] = (int(f[1]), comm, int(f[11]) + int(f[12]), int(f[21]) * 4, int(f[17]))
    return out


def classify(pid, comm):
    if comm.startswith("CarlaUE4"):
        return "server" if comm.startswith("CarlaUE4-Linux") else "wrapper"
    cmd = read("/proc/%d/cmdline" % pid)
    if "b2d_route.py" in cmd:
        return "route"
    if "py-spy" in comm:
        return "pyspy"
    return "other"


class Sampler(threading.Thread):
    def __init__(self, rdir, dt):
        super().__init__(daemon=True)
        self.rdir = str(rdir)
        self.fh = open(os.path.join(self.rdir, "samples.jsonl"), "a", buffering=1)
        self.dt, self.stop_flag = dt, False

    def run(self):
        me = os.getpid()
        while not self.stop_flag:
            t0 = time.time()
            table = proc_table()
            kids = {}
            for pid, row in table.items():
                ppid = row[0]
                kids.setdefault(ppid, []).append(pid)
            mine, stack = [], [me]
            while stack:
                pid = stack.pop()
                for k in kids.get(pid, []):
                    mine.append(k)
                    stack.append(k)
            procs = {}
            for pid in mine:
                _, comm, cpu, rss, thr = table[pid]
                cls = classify(pid, comm)
                if cls != "wrapper":
                    procs[pid] = [cls, cpu, rss, thr]
            stat = dict(line.split() for line in read(str(CGROUP / "cpu.stat")).splitlines() if line)
            gpu = []
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                gpu = [[int(x) for x in line.split(",")] for line in q.stdout.strip().splitlines()]
            except (subprocess.SubprocessError, ValueError):
                pass
            beats = {}
            for f in glob.glob(os.path.join(self.rdir, "w*/attempts/*/*/heartbeat.json")):
                try:
                    beat = json.loads(read(f))
                    beats[os.path.relpath(f, self.rdir).split("/")[0]] = [beat["ticks"], round(beat["t"], 2)]
                except (ValueError, KeyError):
                    pass
            rec = {"t": round(t0, 3), "cg_usage_us": int(stat.get("usage_usec", 0)),
                   "cg_throttled_us": int(stat.get("throttled_usec", 0)),
                   "cg_nr_throttled": int(stat.get("nr_throttled", 0)),
                   "load1": float(read("/proc/loadavg").split()[0]), "procs": procs, "gpu": gpu, "beats": beats}
            self.fh.write(json.dumps(rec) + "\n")
            time.sleep(max(0.0, self.dt - (time.time() - t0)))


def window_stats(samples, n_workers, warm, gpus, t_start):
    """Numbers over the steady window: every worker past `warm` ticks and still ticking (its heartbeat, written every
    2 s while it ticks, is fresh), i.e. no route loading and none finished yet."""
    def steady(s):
        b = s["beats"]
        return len(b) == n_workers and all(v[0] >= warm and s["t"] - v[1] < 6.0 for v in b.values())
    first = {}
    for s in samples:
        for w, v in s["beats"].items():
            first.setdefault(w, v[1] - t_start)
    setup = sorted(first.values())
    idx = [i for i, s in enumerate(samples) if steady(s)]
    if len(idx) < 3:
        return {"window_s": 0.0, "setup_s_median": setup[len(setup) // 2] if setup else None}
    a, b = samples[idx[0]], samples[idx[-1]]
    dt = b["t"] - a["t"]
    ticks = sum(v[0] for v in b["beats"].values()) - sum(v[0] for v in a["beats"].values())
    per = {w: (b["beats"][w][0] - a["beats"][w][0]) / dt for w in b["beats"]}
    out = {"window_s": round(dt, 1), "agg_ticks_s": round(ticks / dt, 2),
           "setup_s_median": round(setup[len(setup) // 2], 1), "setup_s_max": round(setup[-1], 1),
           "worker_ticks_s_min": round(min(per.values()), 2), "worker_ticks_s_max": round(max(per.values()), 2),
           "ms_per_tick_worker": round(1e3 * n_workers * dt / ticks, 1) if ticks else None}
    cores = {}
    for cls in ("server", "route", "other", "pyspy"):
        pids = [p for p, r in b["procs"].items() if r[0] == cls and p in a["procs"]]
        if pids:
            c = sum(b["procs"][p][1] - a["procs"][p][1] for p in pids) / CLK / dt
            cores[cls] = {"n": len(pids), "cores": round(c, 2), "cores_each": round(c / len(pids), 2),
                          "rss_gb_each": round(sum(b["procs"][p][2] for p in pids) / len(pids) / 2 ** 20, 2),
                          "threads_each": round(sum(b["procs"][p][3] for p in pids) / len(pids), 1)}
    out["cores"] = cores
    mine = sum(v["cores"] for v in cores.values())
    cg = (b["cg_usage_us"] - a["cg_usage_us"]) / 1e6 / dt
    out.update(cores_mine=round(mine, 2), cores_container=round(cg, 2), cores_background=round(cg - mine, 2),
               throttled_s_per_s=round((b["cg_throttled_us"] - a["cg_throttled_us"]) / 1e6 / dt, 3),
               nr_throttled=b["cg_nr_throttled"] - a["cg_nr_throttled"])
    win = samples[idx[0]:idx[-1] + 1]
    out["load1_mean"] = round(sum(s["load1"] for s in win) / len(win), 1)
    for g in gpus:
        mem = [row[1] for s in win for row in s["gpu"] if row[0] == g]
        util = [row[2] for s in win for row in s["gpu"] if row[0] == g]
        if mem:
            out["gpu%d" % g] = {"mem_gb_mean": round(sum(mem) / len(mem) / 1024, 1),
                                "mem_gb_max": round(max(mem) / 1024, 1),
                                "util_mean": round(sum(util) / len(util), 1)}
    return out


def main():
    a = parse_args()
    if a.cpus:
        os.sched_setaffinity(0, cpu_list(a.cpus))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    events = open(str(out / "events.jsonl"), "a", buffering=1)
    log = open(str(out / "log.txt"), "a", buffering=1)

    def event(kind, **kw):
        rec = dict(kw, t=round(time.time(), 3), kind=kind)
        events.write(json.dumps(rec) + "\n")
        line = "%s %s %s" % (time.strftime("%H:%M:%S"), kind, json.dumps(kw))
        log.write(line + "\n")
        print(line, flush=True)

    event("start", argv=sys.argv, cpus=a.cpus, affinity=len(os.sched_getaffinity(0)))
    indices = Indices(a.index_lo, a.index_hi, Path(os.environ["DATA_DIR"]) / "runs/infra-acceptance/.next_index")
    base = b2d_run.parse_args(["--out", str(out)] + a.passthrough)
    routes = b2d_run.select_routes(base.routes, "all", a.route_id, 0)
    pool = {g: [] for g in a.gpus}
    last_start = [0.0]

    def start(server):
        wait = a.stagger_s - (time.time() - last_start[0])
        if wait > 0:
            time.sleep(wait)
        server.index = indices.take()
        server.port = b2d_run.PORT_BASE + b2d_run.PORT_STRIDE * server.index
        server.tm_port = b2d_run.TM_BASE + b2d_run.PORT_STRIDE * server.index
        t0 = time.time()
        server.start()
        last_start[0] = time.time()
        event("server_start", gpu=server.gpu_rank, index=server.index, startup_s=round(time.time() - t0, 1))

    rc = 0
    try:
        for n in a.rungs:
            rdir = out / ("rung-%02d" % n)
            rdir.mkdir(exist_ok=True)
            if a.restart_servers:
                for g in a.gpus:
                    for s in pool[g]:
                        s.stop()
            for g in a.gpus:
                while len(pool[g]) < n:
                    pool[g].append(b2d_run.Server(0, out / "servers", base.quality, g, stride=10 ** 6,
                                                  windowed=False))
                for s in pool[g][n:]:
                    s.stop()   # an idle server free-runs in async mode and would load the rung
                for s in pool[g][:n]:
                    if not s.alive():
                        start(s)
            servers = [s for g in a.gpus for s in pool[g][:n]]
            runners, threads = [], []
            for w, s in enumerate(servers):
                args = b2d_run.parse_args(["--out", str(rdir / ("w%02d" % w)), "--route-ids", a.route_id,
                                           "--workers", "1", "--max-attempts", "1", "--no-reap", "--fresh",
                                           "--max-ticks", str(a.max_ticks), "--gpu-rank", str(s.gpu_rank)]
                                          + a.passthrough)
                runners.append(b2d_run.Runner(args, routes, servers=[s]))
            sampler = Sampler(rdir, a.sample_s)
            sampler.start()
            event("rung_start", servers_per_gpu=n, workers=len(servers),
                  indices=[s.index for s in servers])
            t0 = time.time()
            for r in runners:
                threads.append(threading.Thread(target=r.run))
                threads[-1].start()
            for t in threads:
                t.join()
            time.sleep(a.sample_s)
            sampler.stop_flag = True
            sampler.join()
            samples = [json.loads(line) for line in open(str(rdir / "samples.jsonl"))]
            stats = window_stats(samples, len(servers), a.warm_ticks, a.gpus, t0)
            prof = []
            for f in sorted(glob.glob(str(rdir / "w*/attempts/*/*/route_result.json"))):
                d = json.loads(read(f) or "{}")
                p = d.get("profile", {})
                prof.append({"status": d.get("status"), "ticks": p.get("ticks"), "wall_s": d.get("wall_s"),
                             **{k: p.get(k + "_ms_mean") for k in
                                ("total", "world_tick", "tree", "agent", "provider", "control")}})
            fin = [r for r in prof if (r["ticks"] or 0) >= a.max_ticks - 1]
            stats.update(servers_per_gpu=n, gpus=a.gpus, workers=len(servers), rung_wall_s=round(time.time() - t0, 1),
                         routes_capped=len(fin), routes=prof, passthrough=a.passthrough, route_id=a.route_id,
                         max_ticks=a.max_ticks, cpus=a.cpus)
            for k in ("total", "world_tick", "tree", "agent"):
                xs = [r[k] for r in prof if r.get(k) is not None]
                stats[k + "_ms_mean"] = round(sum(xs) / len(xs), 1) if xs else None
            (rdir / "stats.json").write_text(json.dumps(stats, indent=1))
            with open(str(out / "rungs.jsonl"), "a") as fh:
                fh.write(json.dumps(stats) + "\n")
            event("rung_end", **{k: v for k, v in stats.items() if k not in ("routes",)})
            if len(fin) < len(servers):
                rc = 1
                event("rung_incomplete", capped=len(fin), workers=len(servers))
    finally:
        for g in a.gpus:
            for s in pool[g]:
                s.stop()
        event("end", rc=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
