#!/usr/bin/env python
"""CARLA server thread census and a reduced-thread-pool probe. See docs/carla.md, "Threads per server".

A CARLA 0.9.15 server sizes its three asio pools (rpclib RPC, sensor streaming, multi-GPU secondary) from
std::thread::hardware_concurrency(), which in a container is the *host* count (208 here), not the CPU quota or the
affinity mask: (208 - 2) / 3 = 68 threads each, ~204 of a server's ~300. The shipped binary reads
-RPCThreads=N -StreamingThreads=N -SecondaryThreads=N to override them (Plugins/Carla/.../Server/CarlaServer.cpp),
so no rebuild or LD_PRELOAD is needed. UE4's own pools (TaskGraph, PoolThread) follow the affinity mask instead.

  carla_threads.py count                                  every live CarlaUE4 server: threads by pool, affinity
  carla_threads.py probe --tag base --servers 3 ...       start K servers on our own index block, time the start,
        load a map, drive a blocking six-camera loop (the leaderboard waits for every sensor each tick), sample
        threads / CPU / VRAM, stop them by PID; one JSON row per server to --out
  carla_threads.py bench --port P ...                     (internal) the client side of one probe server
  carla_threads.py summary FILE.jsonl ...                 per-round / per-arm markdown tables of probe output
  carla_threads.py compare base-a=DIR reduced-a=DIR ... --out DIR
        route equivalence of PDM-Lite runs (scripts/carla_threads_routes.sh): DS, infractions, game time and
        trajectory deviation for every pair of arms, grouped by the arm kinds (the name before '-')

Python 3.8: runs in envs/carla. Starts only servers it owns and stops them by process group it created.
"""
import argparse
import collections
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", DATA_DIR / "third_party/carla/CARLA_0.9.15"))
HZ = os.sysconf("SC_CLK_TCK")
REDUCED = "-RPCThreads=4 -StreamingThreads=4 -SecondaryThreads=4"   # what a 16-thread desktop gets by default


def pool(name):
    """Thread name -> pool. Unnamed threads carry the executable's truncated name: the main thread plus the
    streaming and secondary asio pools (carla::ThreadPool does not name its threads); rpclib names its 'server'."""
    for key in ("TaskGraph", "PoolThread", "server"):
        if name.startswith(key):
            return key
    return "unnamed" if name.startswith("CarlaUE4") else "other"


def census(pid):
    names = []
    for t in Path("/proc/%d/task" % pid).iterdir():
        try:
            names.append((t / "comm").read_text().strip())
        except OSError:
            pass
    c = collections.Counter(pool(n) for n in names)
    c["total"] = len(names)
    return dict(c)


def cpu_s(pid):
    f = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / HZ


def affinity(pid):
    return len(os.sched_getaffinity(pid))


def servers():
    return [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()
            and _read(p / "comm") == "CarlaUE4-Linux-"]


def _read(p):
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.splitlines() if "," in l)}


def cmd_count(a):
    rows = []
    for pid in sorted(servers()):
        c = census(pid)
        args = _read(Path("/proc/%d/cmdline" % pid)).replace("\0", " ")
        rows.append(dict(pid=pid, affinity=affinity(pid), reduced="-RPCThreads" in args, **c))
    keys = ["pid", "affinity", "reduced", "total", "unnamed", "server", "TaskGraph", "PoolThread", "other"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, 0)) for k in keys))
    print("servers %d, threads %d, pids.current %s" % (len(rows), sum(r["total"] for r in rows),
                                                      _read(Path("/sys/fs/cgroup/pids.current"))))


# ------------------------------------------------------------------------------------------------ server side
def bindable(port):
    with socket.socket() as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def listening(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


class Server:
    def __init__(self, index, gpu, cpus, extra, log):
        self.index, self.port, self.tm = index, 2000 + 50 * index, 8000 + 50 * index
        ports = (self.port, self.port + 1, self.port + 2, self.tm, self.tm + 1)
        if not all(bindable(p) for p in ports):
            raise RuntimeError("index %d: a port of %s is taken" % (index, ports))
        env = dict(os.environ, VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json")
        env.pop("SDL_VIDEODRIVER", None)
        self.log = log
        self.t0 = time.time()
        with open(log, "wb") as fh:
            self.proc = subprocess.Popen(
                ["taskset", "-c", cpus, str(CARLA_ROOT / "CarlaUE4.sh"), "-RenderOffScreen", "-nosound",
                 "-carla-rpc-port=%d" % self.port, "-quality-level=Epic", "-graphicsadapter=%d" % gpu] + extra.split(),
                stdout=fh, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
        self.pid = None

    def wait_port(self, timeout=180):
        while time.time() - self.t0 < timeout:
            if self.proc.poll() is not None:
                return None
            if listening(self.port):
                self.find_pid()
                return time.time() - self.t0
            time.sleep(0.5)
        return None

    def find_pid(self):
        for pid in servers():
            try:
                if os.getpgid(pid) == self.proc.pid:
                    self.pid = pid
            except OSError:
                pass

    def stop(self):
        # Our own group: setsid made pgid == our child's pid. The wrapper dies at once on SIGTERM, but the binary's
        # graceful shutdown can take minutes with VRAM still held, so wait for the whole group, then SIGKILL it.
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            try:
                os.killpg(self.proc.pid, sig)
            except OSError:
                break
            deadline = time.time() + wait
            while time.time() < deadline:
                self.proc.poll()
                try:
                    os.killpg(self.proc.pid, 0)
                except OSError:
                    return
                time.sleep(1)


def cmd_probe(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = out.parent / "logs"
    logs.mkdir(exist_ok=True)
    for rep in range(a.reps):
        pids_now = int(_read(Path("/sys/fs/cgroup/pids.current")))
        if pids_now > a.pids_guard:
            sys.exit("pids.current %d above --pids-guard %d; not starting" % (pids_now, a.pids_guard))
        srvs, rows = [], []
        start_at = time.time() + a.stagger * a.servers + a.settle
        try:
            for k in range(a.servers):
                s = Server(a.index + k, a.gpu, a.cpus, a.server_args,
                           logs / ("%s-r%d-i%d-%d.log" % (a.tag, rep, a.index + k, int(time.time()))))
                srvs.append(s)
                up = s.wait_port()
                row = dict(tag=a.tag, rep=rep, index=s.index, server_args=a.server_args, servers=a.servers,
                           port_open_s=None if up is None else round(up, 1), log=str(s.log))
                rows.append(row)
                if up is None:
                    continue
                time.sleep(3)
                row["threads_idle"] = census(s.pid) if s.pid else None
                s.bench = subprocess.Popen(
                    [sys.executable, __file__, "bench", "--port", str(s.port), "--tm-port", str(s.tm),
                     "--town", a.town, "--cameras", str(a.cameras), "--width", str(a.width), "--height", str(a.height),
                     "--traffic", str(a.traffic), "--ticks", str(a.ticks), "--start-at", str(start_at)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if k < a.servers - 1:
                    time.sleep(a.stagger)
            # sample threads / CPU / VRAM of every server while the measured loops run
            while time.time() < start_at:
                time.sleep(1)
            live = [s for s in srvs if s.pid]
            c0 = {s.pid: cpu_s(s.pid) for s in live if s.proc.poll() is None}
            t0, peak, vram = time.time(), collections.defaultdict(dict), collections.defaultdict(int)
            while any(getattr(s, "bench", None) and s.bench.poll() is None for s in live):
                for s in live:
                    if s.proc.poll() is None:
                        c = census(s.pid)
                        if c["total"] > peak[s.pid].get("total", 0):
                            peak[s.pid] = c
                v = vram_mib()
                for s in live:
                    vram[s.pid] = max(vram[s.pid], v.get(s.pid, 0))
                time.sleep(2)
            dt = time.time() - t0
            for s, row in zip(srvs, rows):
                if not s.pid:
                    row["status"] = "start_failed"
                    continue
                text = s.bench.communicate()[0] if getattr(s, "bench", None) else ""
                res = next((json.loads(l) for l in reversed(text.splitlines()) if l.startswith("{")), None)
                row.update(bench=res, bench_tail=None if res else text[-600:],
                           threads_peak=peak.get(s.pid), vram_mib=vram.get(s.pid),
                           server_cores=round((cpu_s(s.pid) - c0[s.pid]) / dt, 2)
                           if s.pid in c0 and s.proc.poll() is None else None,
                           server_alive=s.proc.poll() is None, affinity=affinity(s.pid) if s.proc.poll() is None else None)
                logtext = s.log.read_text(errors="replace")
                row["render_timeout"] = "waiting for RenderThread" in logtext
                row["signal"] = bool(re.search(r"Signal[ =]11", logtext))
                row["status"] = "ok" if res and res.get("ok") and row["server_alive"] else "failed"
        finally:
            for s in srvs:
                if getattr(s, "bench", None) and s.bench.poll() is None:
                    s.bench.kill()
                s.stop()
        with out.open("a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                b = row.get("bench") or {}
                print("%s rep %d idx %d: %s start %.1fs threads %s/%s ms/tick %s fps %s cores %s vram %s" % (
                    a.tag, rep, row["index"], row["status"], row["port_open_s"] or -1,
                    (row.get("threads_idle") or {}).get("total"), (row.get("threads_peak") or {}).get("total"),
                    b.get("ms_median"), b.get("fps"), row.get("server_cores"), row.get("vram_mib")), flush=True)
        time.sleep(a.gap)


# ------------------------------------------------------------------------------------------------ client side
def cmd_bench(a):
    import numpy as np
    import carla
    client = carla.Client("127.0.0.1", a.port, worker_threads=8)
    client.set_timeout(300.0)
    t = time.time()
    world = client.load_world(a.town)
    load_s = time.time() - t
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, 0.05
    world.apply_settings(st)
    bp, spawns = world.get_blueprint_library(), world.get_map().get_spawn_points()
    ego = next(v for v in (world.try_spawn_actor(bp.find("vehicle.lincoln.mkz_2017"), sp) for sp in spawns) if v)
    tm = client.get_trafficmanager(a.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(0)
    ego.set_autopilot(True, a.tm_port)
    actors = [ego]
    for sp in spawns[1:1 + a.traffic]:
        v = world.try_spawn_actor(bp.filter("vehicle.*")[0], sp)
        if v:
            v.set_autopilot(True, a.tm_port)
            actors.append(v)
    got, cond = {}, threading.Condition()

    def cb(img, k):
        with cond:
            got[k] = img
            cond.notify_all()
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(a.width))
    cam_bp.set_attribute("image_size_y", str(a.height))
    cams = []
    for i, yaw in enumerate([0.0, -55.0, 55.0, 180.0, -110.0, 110.0][:a.cameras]):
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=0.8, z=1.6), carla.Rotation(yaw=yaw)),
                                attach_to=ego)
        cam.listen(lambda img, k=i: cb(img, k))
        cams.append(cam)
        actors.append(cam)

    def step():  # the leaderboard's pattern: tick, then block until every sensor delivered this frame
        f = world.tick()
        with cond:
            ok = cond.wait_for(lambda: len(got) == len(cams) and all(g.frame >= f for g in got.values()), 20.0)
        return ok
    for _ in range(40):
        step()
    stds = [float(np.frombuffer(g.raw_data, dtype=np.uint8).reshape(g.height, g.width, 4)[::8, ::8, :3].std())
            for g in got.values()]
    while time.time() < a.start_at:     # all probe servers measure over the same window
        step()
    dts, missed = [], 0
    for _ in range(a.ticks):
        t = time.perf_counter()
        missed += not step()
        dts.append(time.perf_counter() - t)
    for cam in cams:
        cam.stop()
    world.tick()
    client.apply_batch_sync([carla.command.DestroyActor(x) for x in reversed(actors)], True)
    st.synchronous_mode = False
    world.apply_settings(st)
    ms = sorted(1e3 * d for d in dts)
    print(json.dumps(dict(ok=missed == 0 and min(stds or [0]) >= 1.0, load_s=round(load_s, 1), ticks=a.ticks,
                          missed=missed, ms_median=round(statistics.median(ms), 1), ms_mean=round(statistics.mean(ms), 1),
                          ms_p95=round(ms[int(0.95 * len(ms))], 1), fps=round(len(ms) / (sum(ms) / 1e3), 2),
                          cam_std_min=round(min(stds or [0]), 1))), flush=True)


# ------------------------------------------------------------------------------------------------ route equivalence
def route_record(arm_dir, rid):
    done = Path(arm_dir) / "done" / ("%s.json" % rid)
    if not done.exists():
        return None
    d = json.loads(done.read_text())
    adir = Path(arm_dir) / "attempts" / rid / str(d["attempt"])
    rec = json.loads((adir / "results.json").read_text())["_checkpoint"]["records"][0]
    traj = {}
    for line in (adir / "expert.jsonl").read_text().splitlines():
        e = json.loads(line)
        traj[round(e["t"] / 0.05)] = (e["x"], e["y"])
    s = rec["scores"]
    return dict(status=rec["status"], ds=s["score_composed"], rc=s["score_route"], pen=s["score_penalty"],
                game_s=rec["meta"]["duration_game"], attempt=d["attempt"], traj=traj,
                infr={k: len(v) for k, v in rec["infractions"].items() if k != "min_speed_infractions" and v})


def cmd_compare(a):
    arms = dict(x.split("=", 1) for x in a.arms)          # name=dir; the name's prefix before '-' is its kind
    recs = {n: {} for n in arms}
    for n, d in arms.items():
        for f in sorted((Path(d) / "done").glob("*.json")):
            recs[n][f.stem] = route_record(d, f.stem)
    routes = sorted(set.intersection(*(set(r) for r in recs.values())), key=int)
    per_route, pairs = [], collections.defaultdict(list)
    names = list(arms)
    for rid in routes:
        row = dict(route=rid)
        for n in names:
            r = recs[n][rid]
            row.update({n + "_ds": round(r["ds"], 2), n + "_game_s": r["game_s"], n + "_infr": json.dumps(r["infr"])})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = recs[names[i]][rid], recs[names[j]][rid]
                common = sorted(set(A["traj"]) & set(B["traj"]))
                dist = [((A["traj"][k][0] - B["traj"][k][0]) ** 2 + (A["traj"][k][1] - B["traj"][k][1]) ** 2) ** .5
                        for k in common]
                split = next((k * 0.05 for k, x in zip(common, dist) if x > 0.1), None)
                kinds = "-".join(sorted(n.split("-")[0] for n in (names[i], names[j])))
                pr = dict(route=rid, pair=names[i] + "|" + names[j], kinds=kinds, d_ds=round(abs(A["ds"] - B["ds"]), 2),
                          d_game_s=round(abs(A["game_s"] - B["game_s"]), 2), same_infr=A["infr"] == B["infr"],
                          dev_max_m=round(max(dist or [0]), 2), dev_med_m=round(statistics.median(dist or [0]), 3),
                          split_t=split)
                pairs[kinds].append(pr)
        per_route.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("routes.csv", per_route), ("pairs.csv", [p for v in pairs.values() for p in v])):
        keys = list(rows[0])
        (out / name).write_text("\n".join([",".join(keys)] + [",".join('"%s"' % r[k] if "," in str(r[k]) else str(r[k])
                                                                   for k in keys) for r in rows]) + "\n")
    lines = ["| pair kind | pairs | DS identical | mean abs dDS | max abs dDS | same infractions | "
             "mean abs d game time (s) | median of per-route max deviation (m) | trajectories bit-identical |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kinds, v in sorted(pairs.items()):
        lines.append("| %s | %d | %d | %.2f | %.2f | %d | %.2f | %.2f | %d |" % (
            kinds, len(v), sum(p["d_ds"] == 0 for p in v), statistics.mean(p["d_ds"] for p in v),
            max(p["d_ds"] for p in v), sum(p["same_infr"] for p in v), statistics.mean(p["d_game_s"] for p in v),
            statistics.median(p["dev_max_m"] for p in v), sum(p["dev_max_m"] == 0 for p in v)))
    means = ["| arm | routes | mean DS | completed |", "|---|---:|---:|---:|"]
    for n in names:
        rs = [recs[n][r] for r in routes]
        means.append("| %s | %d | %.2f | %d |" % (n, len(rs), statistics.mean(r["ds"] for r in rs),
                                                  sum(r["status"] == "Completed" for r in rs)))
    (out / "summary.md").write_text("\n".join(means + [""] + lines) + "\n")
    print("\n".join(means + [""] + lines))


def cmd_summary(a):
    """Per-round and per-arm tables of probe JSONL files (a round = one probe call; it starts at its first index)."""
    out = []
    for path in a.files:
        rows, rnd, first = [json.loads(l) for l in open(path)], -1, None
        for r in rows:
            first = r["index"] if first is None else first
            rnd += r["index"] == first
            r["round"] = rnd
        out += ["### " + Path(path).name, "",
                "| round | arm | starts | crashed at start (RenderThread / other) | threads per server | "
                "aggregate FPS | ms/tick mean (median) | server cores | server core-s per tick | VRAM GB |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        arms = collections.defaultdict(lambda: collections.Counter())
        for k in range(rnd + 1):
            R = [r for r in rows if r["round"] == k]
            ok = [r for r in R if r["status"] == "ok"]
            rt = sum(bool(r.get("render_timeout")) for r in R)
            bad = len(R) - len(ok)
            arms[R[0]["tag"]].update(starts=len(R), rt=rt, other=bad - rt)
            m = lambda f: statistics.mean(f(r) for r in ok) if ok else float("nan")
            out.append("| %d | %s | %d | %d (%d / %d) | %d | %.1f | %.0f (%.0f) | %.2f | %.3f | %.1f |" % (
                k, R[0]["tag"], len(R), bad, rt, bad - rt, R[0]["threads_idle"]["total"] if R[0].get("threads_idle") else -1,
                sum(r["bench"]["fps"] for r in ok), m(lambda r: r["bench"]["ms_mean"]), m(lambda r: r["bench"]["ms_median"]),
                m(lambda r: r["server_cores"]), m(lambda r: r["server_cores"] * r["bench"]["ms_mean"] / 1e3),
                m(lambda r: r["vram_mib"] / 1024)))
        out += [""] + ["- %s: %d starts, %d RenderThread-timeout crashes, %d other failures" % (t, c["starts"], c["rt"], c["other"])
                       for t, c in arms.items()] + [""]
    print("\n".join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count")
    s = sub.add_parser("summary", help="markdown tables of probe JSONL files")
    s.add_argument("files", nargs="+")
    c = sub.add_parser("compare", help="route equivalence: name=arm_dir ... (b2d_run --out dirs of the PDM-Lite runs)")
    c.add_argument("arms", nargs="+")
    c.add_argument("--out", required=True)
    q = sub.add_parser("probe")
    q.add_argument("--tag", required=True)
    q.add_argument("--server-args", default="", help="extra CarlaUE4 arguments; reduced pools: '%s'" % REDUCED)
    q.add_argument("--index", type=int, default=210, help="first server index (rpc 2000+50i, TM 8000+50i)")
    q.add_argument("--servers", type=int, default=1)
    q.add_argument("--reps", type=int, default=1)
    q.add_argument("--gpu", type=int, default=1)
    q.add_argument("--cpus", default="110-117,200-207", help="taskset list, as the chains pin their runners")
    q.add_argument("--stagger", type=float, default=20.0)
    q.add_argument("--settle", type=float, default=150.0, help="s after the last start before the measured window")
    q.add_argument("--gap", type=float, default=10.0, help="s between repetitions")
    q.add_argument("--pids-guard", type=int, default=14000)
    q.add_argument("--out", required=True)
    for b in (q, sub.add_parser("bench")):
        b.add_argument("--town", default="Town10HD_Opt")
        b.add_argument("--cameras", type=int, default=6)
        b.add_argument("--width", type=int, default=1600)
        b.add_argument("--height", type=int, default=900)
        b.add_argument("--traffic", type=int, default=30)
        b.add_argument("--ticks", type=int, default=300)
    b.add_argument("--port", type=int, required=True)
    b.add_argument("--tm-port", type=int, required=True)
    b.add_argument("--start-at", type=float, default=0.0)
    a = p.parse_args()
    return {"count": cmd_count, "probe": cmd_probe, "bench": cmd_bench, "compare": cmd_compare, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
