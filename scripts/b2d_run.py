#!/usr/bin/env python
"""Run a set of Bench2Drive routes with per-route isolation, a tick-progress watchdog and resume.
See research/carla-efficiency.md, section "可靠性和续跑".

The properties this exists for, in the order they matter:

  R1  a crash costs one route, not the run. One process and one CARLA server per worker; the route
      runs in a child process, so a segfault in the CARLA client library kills that child only, and
      the worker restarts its server and moves on.
  R2  resume works after a crash, not only after a clean exit. A route is "done" iff
      `done/<id>.json` exists, written after the route finished. Re-running the same command skips
      those and redoes everything else. Nothing relies on the leaderboard's own `--resume`, whose
      checkpoint is rewritten by the process that is crashing.
  R3  slow is not hung. The watchdog reads the heartbeat the tick loop writes and asks whether
      ticks advanced, not whether the process is alive. It also watches the *server*: when the
      server process is gone the route is killed at once instead of waiting out the client's RPC
      timeout, which is most of the wall clock of a crashed route.
  R7  a server that has not died yet is still recycled, optionally, every `--recycle-routes` N
      routes. Off by default: see the note on that flag for why the interval is a measurement.
  R4/R5 every attempt keeps its own directory with the route result, the leaderboard checkpoint and
      the server log at the time it died.
  R6  `summary.json` reports attempts per route and which routes never finished. Read it before
      quoting a score: a score computed over the routes that happened to finish is a score on a
      selected subset.

Multi-GPU is route sharding, not a split simulation, so several runners may share one `--out`
directory - one per card, or one per machine on a shared filesystem. The bookkeeping is idempotent
across them: `done/<id>.json` means finished on *any* card, and a route is claimed with an
exclusive `claims/<id>.lock` before it starts, so two cards never run the same route. A claim whose
owner died is stolen after `--claim-stale-s`. Give each card its own `--server-index` (ports are
2000+50i) and its own `--gpu-rank`.

    scripts/b2d_run.py --out $DATA_DIR/runs/b2d/base55 --workers 4 --towns base --rig front3 \
        --policy sleep --infer-ms 129

Python 3.8: runs in envs/carla.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(os.environ["DATA_DIR"])
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", DATA_DIR / "third_party/carla/CARLA_0.9.15"))
BENCH2DRIVE = Path(os.environ.get("BENCH2DRIVE_ROOT", DATA_DIR / "third_party/Bench2Drive"))
PYTHON = str(DATA_DIR / "envs/carla/bin/python")
HERE = Path(__file__).resolve().parent
# A CARLA server claims several ports above its RPC port (streaming, secondary), and the traffic
# manager wants room of its own, so the slots are 50 apart - the same spacing scripts/carla_server.sh
# uses. Our first 44-route run used 4 and lost two worker slots to a traffic-manager bind error.
PORT_BASE, TM_BASE, PORT_STRIDE = 2000, 8000, 50

# Towns the base 0.9.15 package actually ships, read off a running server with
# `client.get_available_maps()`. Town06, Town07 and Town11-15 are in AdditionalMaps, which is
# deliberately not downloaded (the Waymo training split owns the link), so those routes are out of
# scope. This list is the fallback; the run re-checks it against the live server, because being
# wrong about it costs three failed attempts per route and looks like a flaky simulator.
BASE_TOWNS = {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD", "Town10HD_Opt"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--routes", default=str(BENCH2DRIVE / "leaderboard/data/bench2drive220.xml"))
    p.add_argument("--towns", default="base", help="'base', 'all', or a comma-separated list")
    p.add_argument("--route-ids", default="", help="comma-separated route ids, overrides --towns")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4, help="4 is the measured operating point; "
                   "5 is slower and a 6th server segfaults at startup (docs/carla.md)")
    p.add_argument("--stagger-s", type=float, default=20.0,
                   help="gap between server launches; launching a pool at once costs 17%% "
                        "throughput and half-fails (docs/carla.md)")
    p.add_argument("--server-index", type=int, default=0,
                   help="first CARLA server index; rpc port 2000+50i, as scripts/carla_server.sh")
    p.add_argument("--gpu-rank", type=int, default=0, help="which card this runner's servers use")
    p.add_argument("--claim-stale-s", type=float, default=7200.0,
                   help="a claim older than this is assumed to belong to a dead runner")
    p.add_argument("--quality", default="Epic", choices=["Epic", "Low"])
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--recycle-routes", type=int, default=0,
                   help="R7: stop and restart a worker's server every N routes even when it looks "
                        "healthy, to get ahead of UE4's slow death instead of paying for the crash. "
                        "0 (default) is off ON PURPOSE: a recycle costs 30-60 s against about four "
                        "minutes per route per worker, so every route is 15-25%% overhead and every "
                        "fifth is 3-5%%. Whether that buys anything depends on how the crash rate "
                        "grows with the number of consecutive route attempts one server has "
                        "served (a failed attempt counts, which recycles a suspect server sooner), "
                        "which "
                        "summary.json now reports as by_server_age. Set N from that curve, not from "
                        "taste.")
    p.add_argument("--stall-s", type=float, default=240.0, help="no tick progress for this long = hung")
    p.add_argument("--route-timeout-s", type=float, default=5400.0)
    p.add_argument("--fresh", action="store_true", help="ignore existing results and redo everything")
    # passed through to b2d_route.py
    p.add_argument("--rig", default="front3")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--policy", default="none")
    p.add_argument("--infer-ms", type=float, default=0.0)
    p.add_argument("--policy-socket", default="")
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--no-spectator", action="store_true")
    p.add_argument("--fast-copy", action="store_true")
    p.add_argument("--zero-copy", action="store_true")
    p.add_argument("--cache-lights", action="store_true")
    p.add_argument("--max-ticks", type=int, default=0)
    return p.parse_args()


def select_routes(routes_xml, towns, route_ids, limit):
    """Returns [(route_id, town)]."""
    root = ET.parse(routes_xml).getroot()
    town_of = dict((r.get("id"), r.get("town")) for r in root.findall("route"))
    if route_ids:
        wanted = [(r.strip(), town_of.get(r.strip())) for r in route_ids.split(",") if r.strip()]
    else:
        allowed = None
        if towns == "base":
            allowed = BASE_TOWNS
        elif towns != "all":
            allowed = set(towns.split(","))
        wanted = [(r.get("id"), r.get("town")) for r in root.findall("route")
                  if allowed is None or r.get("town") in allowed]
    return wanted[:limit] if limit else wanted


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read().decode("utf-8", "replace").replace("\0", " ")
    except (OSError, IOError):
        return ""


def kill_group(pid, why=""):
    """Kill a process group by pid, after checking what that pid actually is. Never pkill -f:
    -f matches our own command line and other sessions' (docs/long-runs.md)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            return
        time.sleep(2)
        if not pid_alive(pid):
            return


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


class Server(object):
    """One headless CARLA server. Owns the process group so it can be killed without pkill -f,
    which docs/long-runs.md forbids for good reason."""

    def __init__(self, index, log_dir, quality, gpu_rank=0, stride=0):
        self.index = index
        self.gpu_rank = gpu_rank
        self.stride = stride or 1
        self.routes_served = 0   # how many routes this process has run; the R7 curve needs it
        self.started_at = None
        self.port = PORT_BASE + PORT_STRIDE * index
        self.tm_port = TM_BASE + PORT_STRIDE * index
        self.log_dir = Path(log_dir)
        self.quality = quality
        self.proc = None
        self.pgid = None
        self.log = None
        self.starts = 0

    def start(self, timeout=180.0):
        self.stop()
        self.starts += 1
        self.log = self.log_dir / ("carla-%d-%d.log" % (self.index, self.starts))
        env = dict(os.environ, VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json")
        with open(self.log, "wb") as fh:
            self.proc = subprocess.Popen(
                [str(CARLA_ROOT / "CarlaUE4.sh"), "-RenderOffScreen", "-nosound",
                 "-carla-rpc-port=%d" % self.port, "-quality-level=%s" % self.quality,
                 "-graphicsadapter=%d" % self.gpu_rank],
                stdout=fh, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
        # setsid in preexec_fn makes the child its own group leader, so pgid == pid. Record it:
        # stop() must not look it up later, when the wrapper may already be gone.
        self.pgid = self.proc.pid
        (self.log_dir / ("carla-%d.pid" % self.index)).write_text(str(self.proc.pid))
        self.started_at, self.routes_served = time.time(), 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not port_free(self.port):
                time.sleep(3)  # the port opens slightly before the RPC server is usable
                return True
            if self.proc.poll() is not None:
                raise RuntimeError("CARLA server %d died at startup, see %s" % (self.index, self.log))
            time.sleep(2)
        self.stop()
        raise RuntimeError("CARLA server %d never opened port %d" % (self.index, self.port))

    def move(self):
        """Take the next port slot. Twice in the 44-route run a worker's slot became unusable -
        every attempt on it hung before the route printed a line, while the same routes ran first
        time on a fresh slot - so a worker that fails a route does not keep asking the same ports
        to work."""
        self.stop()
        self.index += self.stride
        self.port = PORT_BASE + PORT_STRIDE * self.index
        self.tm_port = TM_BASE + PORT_STRIDE * self.index

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            # Kill the group by the pgid recorded at launch, never by looking it up now.
            # `CarlaUE4.sh` is a wrapper that execs nothing: the real binary is its child, and
            # when the binary crashes the wrapper exits first. os.getpgid(wrapper) then raises
            # ESRCH, the loop breaks, and the server binary survives its own runner - one did,
            # holding an RPC port and 6 GB of VRAM for half an hour. setsid made the wrapper a
            # group leader, so its pid IS the pgid and stays valid for as long as any member of
            # the group is alive.
            try:
                os.killpg(self.pgid, sig)
            except OSError:
                break
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                continue
            # The wrapper is reaped; the binary may not be. Poll the group rather than trust it.
            time.sleep(1)
            try:
                os.killpg(self.pgid, 0)
            except OSError:
                break
        self.proc, self.pgid = None, None


class Runner(object):
    def __init__(self, a, routes):
        self.a = a
        self.out = Path(a.out)
        for sub in ("done", "attempts", "servers", "claims"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.events = open(str(self.out / "events.jsonl"), "a", buffering=1)
        # Reentrant: next_route() holds the lock and logs an event, and event() locks too. With a
        # plain Lock the second resume run deadlocked on the first "already done" skip - which is
        # only reachable when there is something to resume from, i.e. never in a first run.
        self.lock = threading.RLock()
        self.queue = list(routes)
        self.requested = list(routes)
        self.available_maps = None
        self.no_map = []
        self.reap_orphans()
        self.claimed = set()
        self.stop_flag = False
        # Servers must not be launched simultaneously: four at once produced a world-load timeout
        # and 17% less throughput than the same four staggered by 20 s.
        self.start_lock = threading.Lock()
        self.last_start = 0.0

    def reap_orphans(self):
        """A runner that is SIGKILLed leaves its CARLA servers and route processes behind, holding
        the ports the next run wants. Recorded pids make that recoverable without pattern-matching
        command lines: check that the pid really is the thing we started, then kill its group."""
        for pidfile in sorted(self.out.glob("servers/*.pid")) + sorted(
                self.out.glob("attempts/*/*/route.pid")):
            try:
                pid = int(pidfile.read_text().strip())
            except (OSError, ValueError):
                continue
            cmd = cmdline(pid)
            if pid_alive(pid) and ("CarlaUE4" in cmd or "b2d_route.py" in cmd):
                self.event("orphan_killed", pid=pid, cmd=cmd[:120])
                kill_group(pid)
            try:
                pidfile.unlink()
            except OSError:
                pass

    def event(self, kind, **kw):
        rec = dict(kw)
        rec["t"], rec["kind"] = time.time(), kind
        with self.lock:
            self.events.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def next_route(self):
        with self.lock:
            while self.queue:
                rid, town = self.queue.pop(0)
                if self.available_maps is not None and town not in self.available_maps:
                    self.no_map.append(rid)
                    self.event("skip", route_id=rid, town=town, reason="map not installed")
                    continue
                if not self.a.fresh and (self.out / "done" / (rid + ".json")).exists():
                    self.event("skip", route_id=rid, reason="already done")
                    continue
                if not self.claim(rid):
                    self.event("skip", route_id=rid, reason="claimed by another runner")
                    continue
                return rid
            return None

    def claim(self, rid):
        """O_EXCL is the whole mechanism: on one filesystem it is atomic between processes and
        between machines, which is what makes several cards safe to point at one --out."""
        path = self.out / "claims" / (rid + ".lock")
        body = json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "t": time.time(),
                           "gpu_rank": self.a.gpu_rank}).encode()
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                return False
            owner = {}
            try:
                owner = json.loads(path.read_text())
            except (OSError, ValueError):
                pass
            mine = owner.get("host") == socket.gethostname()
            dead = mine and not pid_alive(int(owner.get("pid", -1)))
            if not dead and age < self.a.claim_stale_s:
                return False
            self.event("claim_stolen", route_id=rid, age_s=round(age))
            fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        self.claimed.add(rid)
        return True

    def release(self, rid):
        try:
            os.unlink(str(self.out / "claims" / (rid + ".lock")))
        except OSError:
            pass
        self.claimed.discard(rid)

    def run(self):
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True)
                   for i in range(self.a.workers)]
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            self.stop_flag = True
            raise
        self.summarise()

    def worker(self, wi):
        server = Server(self.a.server_index + wi, self.out / "servers", self.a.quality,
                        self.a.gpu_rank, stride=self.a.workers)
        try:
            while not self.stop_flag:
                rid = self.next_route()
                if rid is None:
                    return
                finished = False
                for attempt in range(1, self.a.max_attempts + 1):
                    if self.stop_flag:
                        self.release(rid)
                        return
                    if not server.alive():
                        self.event("server_start", worker=wi, index=server.index, port=server.port)
                        self.staggered_start(server)
                        self.learn_maps(server)
                    ok, record = self.run_once(wi, server, rid, attempt)
                    server.routes_served += 1
                    if ok:
                        (self.out / "done" / (rid + ".json")).write_text(json.dumps(record, indent=2))
                        finished = True
                        break
                    # Anything that is not a clean finish means the server is suspect: a hung or
                    # segfaulted server answers RPCs for a while and then fails the next route too.
                    # Come back on a different port as well, in case the slot is what is broken.
                    server.move()
                    self.event("server_moved", worker=wi, index=server.index, port=server.port)
                # Release either way: the claim exists to stop two runners racing, not to record
                # the outcome. A route that never finished must be retryable by the next run.
                self.release(rid)
                if not finished:
                    self.event("route_abandoned", route_id=rid, attempts=self.a.max_attempts)
                if self.a.recycle_routes and server.routes_served >= self.a.recycle_routes:
                    self.event("server_recycled", worker=wi, index=server.index,
                               routes_served=server.routes_served,
                               age_s=round(time.time() - (server.started_at or time.time())))
                    server.stop()  # the next route starts it again, under the stagger lock
        finally:
            server.stop()

    def learn_maps(self, server):
        """Ask the server which maps it has, once. A route whose town is not installed is not a
        failure to retry, it is out of scope, and it has to be reported as such rather than
        dropped (research/carla-efficiency.md R6)."""
        if self.available_maps is not None:
            return
        import carla
        try:
            client = carla.Client("127.0.0.1", server.port)
            client.set_timeout(60.0)
            maps = set(m.split("/")[-1] for m in client.get_available_maps())
        except Exception as e:
            self.event("map_probe_failed", error=str(e))
            return
        with self.lock:
            self.available_maps = maps | set(m.replace("_Opt", "") for m in maps)
        self.event("maps", maps=sorted(maps))

    def staggered_start(self, server):
        with self.start_lock:
            wait = self.a.stagger_s - (time.time() - self.last_start)
            if wait > 0:
                time.sleep(wait)
            server.start()
            self.last_start = time.time()

    def run_once(self, wi, server, rid, attempt):
        adir = self.out / "attempts" / rid / ("%d" % attempt)
        adir.mkdir(parents=True, exist_ok=True)
        # The traffic manager's RPC server lives in the *client* process, so a route that is still
        # dying holds its port and the next attempt fails with "bind error" before it ticks once.
        # Take the first free port instead of insisting on one.
        # Stay inside this server's own 50-port block so the scan cannot wander into a neighbour's.
        tm_port = next(p for p in range(server.tm_port, server.tm_port + PORT_STRIDE) if port_free(p))
        cmd = [PYTHON, str(HERE / "b2d_route.py"), "--routes", self.a.routes, "--route-id", rid,
               "--port", str(server.port), "--tm-port", str(tm_port),
               "--out", str(adir), "--rig", self.a.rig, "--width", str(self.a.width),
               "--height", str(self.a.height), "--policy", self.a.policy,
               "--infer-ms", str(self.a.infer_ms), "--decimate", str(self.a.decimate)]
        if self.a.policy_socket:
            cmd += ["--policy-socket", self.a.policy_socket]
        for flag in ("overlap", "no_spectator", "fast_copy", "zero_copy", "cache_lights"):
            if getattr(self.a, flag):
                cmd.append("--" + flag.replace("_", "-"))
        if self.a.max_ticks:
            cmd += ["--max-ticks", str(self.a.max_ticks)]

        self.event("route_start", worker=wi, route_id=rid, attempt=attempt, port=server.port,
                   tm_port=tm_port)
        t0 = time.time()
        log = open(str(adir / "route.log"), "wb")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        (adir / "route.pid").write_text(str(proc.pid))
        reason = self.supervise(proc, adir, t0, server)
        log.close()

        record = {"route_id": rid, "attempt": attempt, "wall_s": round(time.time() - t0, 1),
                  "worker": wi, "server_index": server.index, "server_log": str(server.log),
                  # R7's raw material: how many route attempts this server process had already
                  # served before this one, and how long it had been up. summary.json bins on them.
                  "server_age_routes": server.routes_served,
                  "server_age_s": round(time.time() - (server.started_at or time.time()))}
        rfile = adir / "route_result.json"
        if rfile.exists():
            try:
                record.update(json.loads(rfile.read_text()))
            except ValueError:
                record["status"] = "unreadable_result"
        record["ticks"] = record.get("profile", {}).get("ticks")
        if reason:
            # A route can write its result and then hang on the way out - the CARLA client's
            # threads do not always stop - so a kill does not mean the route failed. Keep what the
            # route said about itself and record the kill beside it.
            record["killed"] = reason
            record.setdefault("status", reason)
            if record["status"] == "harness_error":
                pass  # the route's own verdict is the more informative one
            elif "profile" not in record:
                record["status"] = reason
        elif "status" not in record:
            record["status"] = "no_result_file"
        record["returncode"] = proc.returncode
        # The route process cannot know which server it ran on or how old that server was, so the
        # runner writes its own view beside the route's. summary.json bins on it (R7).
        (adir / "attempt.json").write_text(json.dumps(
            dict((k, v) for k, v in record.items() if k != "profile"), indent=2))
        ok = record["status"] == "finished"
        self.event("route_end", worker=wi, route_id=rid, attempt=attempt,
                   status=record["status"], wall_s=record["wall_s"], ticks=record.get("ticks"))
        return ok, record

    def supervise(self, proc, adir, t0, server=None):
        """Return None if the process exited on its own, else why we killed it.

        Two conditions, and the cheap one first. When the server process is gone the route is
        finished whatever it thinks: the client will sit on its RPC until `client_timeout`
        expires, which in the Town12 camera crashes was most of the five to seven minutes each
        one cost. Killing on a dead server turns that into seconds. The heartbeat check stays for
        the case the cheap one cannot see - a server that is alive and wedged."""
        beat = adir / "heartbeat.json"
        last_ticks, last_change = -1, time.time()
        while True:
            try:
                proc.wait(timeout=5)
                return None
            except subprocess.TimeoutExpired:
                pass
            if server is not None and server.proc is not None and not server.alive():
                self.kill(proc)
                return "server_died_rc%s" % server.proc.returncode
            now = time.time()
            ticks = None
            if beat.exists():
                try:
                    ticks = json.loads(beat.read_text()).get("ticks")
                except ValueError:
                    ticks = None
            if ticks is not None and ticks != last_ticks:
                last_ticks, last_change = ticks, now
            stalled = now - last_change
            if stalled > self.a.stall_s:
                self.kill(proc)
                return "hung_no_tick_progress_%ds" % int(stalled)
            if now - t0 > self.a.route_timeout_s:
                self.kill(proc)
                return "route_timeout"

    @staticmethod
    def kill(proc):
        # proc.pid is the pgid: preexec_fn=os.setsid made it a group leader. Do not call
        # os.getpgid here - see Server.stop() for what happens when the leader has already exited.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=10)
                return
            except (OSError, subprocess.TimeoutExpired):
                continue

    def summarise(self):
        """Built from what is on disk, not from this process's memory: a run that resumes after a
        crash must still report the whole history, including the attempts its predecessor made.
        R6 in research/carla-efficiency.md - the number of restarts and the routes that never
        finished are results, not logistics, because a score over the routes that happened to
        finish is a score on a selected subset."""
        requested = [rid for rid, _ in self.requested]
        history, restarts = {}, 0
        for rid in requested:
            tries = []
            for adir in sorted((self.out / "attempts" / rid).glob("*"),
                               key=lambda p: int(p.name) if p.name.isdigit() else 0):
                runner = _read_json(adir / "attempt.json")
                d = runner or _read_json(adir / "route_result.json")
                if d is None:
                    tries.append({"attempt": adir.name, "status": "no result file"})
                    continue
                tries.append({"attempt": adir.name, "status": d.get("status"),
                              "wall_s": d.get("wall_s"),
                              "ticks": d.get("ticks", d.get("profile", {}).get("ticks")),
                              "killed": d.get("killed"),
                              "server_age_routes": d.get("server_age_routes")})
            if tries:
                history[rid] = tries
                restarts += len(tries) - 1
        done = {}
        for rid in requested:
            f = self.out / "done" / (rid + ".json")
            if f.exists():
                done[rid] = json.loads(f.read_text())
        never = [r for r in requested if r not in done and r not in self.no_map]
        walls = [r.get("wall_s", 0) for r in done.values()]
        ticks = [r.get("profile", {}).get("ticks", 0) for r in done.values()]
        summary = {
            "routes_requested": len(requested),
            "routes_finished": len(done),
            "routes_skipped_no_map": sorted(set(self.no_map)),
            "routes_never_finished": never,
            "repeat_offenders": sorted(r for r, t in history.items() if len(t) > 1),
            "restarts": restarts,
            "attempts": history,
            "wall_s_total": round(sum(walls), 1),
            "wall_s_median": _median(walls),
            "ticks_total": sum(ticks),
            "s_per_tick_median": round(_median(walls) / max(1, _median(ticks)), 4) if ticks else None,
            "by_server_age": _by_server_age(history),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
        self.event("summary", **dict((k, v) for k, v in summary.items() if k != "attempts"))


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _by_server_age(history):
    """R7's curve: does a server get more likely to fail the longer it has been serving routes?

    Buckets are the number of routes the server had already run when this attempt started. Read
    `failed / attempts` per bucket; a rate that climbs with age is the argument for recycling, a
    flat one says recycling only costs. One run is not the curve - accumulate across runs before
    setting --recycle-routes. Attempts whose age was not recorded (older runs) are left out."""
    buckets = {}
    for tries in history.values():
        for t in tries:
            age = t.get("server_age_routes")
            if age is None:
                continue
            b = buckets.setdefault(str(age), {"attempts": 0, "failed": 0})
            b["attempts"] += 1
            if t.get("status") != "finished":
                b["failed"] += 1
    return dict(sorted(buckets.items(), key=lambda kv: int(kv[0])))


def _median(xs):
    xs = sorted(x for x in xs if x)
    return round(xs[len(xs) // 2], 1) if xs else 0.0


def main():
    a = parse_args()
    routes = select_routes(a.routes, a.towns, a.route_ids, a.limit)
    if not routes:
        print("no routes selected", file=sys.stderr)
        return 2
    print("%d routes, %d workers, gpu %d -> %s" % (len(routes), a.workers, a.gpu_rank, a.out),
          flush=True)
    Runner(a, routes).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
