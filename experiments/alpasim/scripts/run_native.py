"""Run an AlpaSim wizard preset without Docker: same services, same commands, plain processes on localhost.

The wizard is run as shipped with `wizard.run_method=NONE`, which writes docker-compose.yaml and every service
config into the log dir. This launcher then starts each compose service as a host process: the command is the
compose command with container mount points rewritten to their host paths, the environment is the compose
environment, the working dir is the AlpaSim checkout. The runtime starts last and its exit code ends the run
(what `docker compose up --exit-code-from runtime-0` does). Nothing in the AlpaSim source is edited.

A driver command (`--driver`, a shell string that honours ALPASIM_DRIVER_HOST / ALPASIM_DRIVER_PORT) is started
first on a free port; `--tap` puts driver_tap.py between runtime and driver. A sampler records per-service GPU
memory, CPU seconds and RSS once a second into usage.jsonl; native_summary.json holds peaks and wall times.

A runtime worker that dies (SIGKILLed from outside on this box, docs/alpasim.md "Killed workers") leaves the runtime
alive but stuck: "Worker N died with exit code -9", "Result pump failed", then no rollout ever finishes. The launcher
watches the runtime's log for those lines and ends the run with rc 1 at once, so the pool job fails in seconds.

Run it as a GPU-pool job with the AlpaSim env's python (the pool sets CUDA_VISIBLE_DEVICES):
  python -m jevdrive.cl submit --name alpasim-dev --vram 30 --cpu 12 -- bash experiments/alpasim/scripts/run.sh ...
"""
import argparse, json, os, random, re, shlex, signal, socket, subprocess, sys, threading, time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORKER_CRASH = re.compile(rb"Worker \d+ died with exit code[^\n]*|Result pump failed[^\n]*")


def worker_crash(log: Path, off: int) -> tuple:
    """(first crash line in the runtime log past byte `off` or "", new offset). Re-reads 200 bytes of overlap so a
    line split across two reads is still matched."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, off - 200))
            data = f.read()
            m = WORKER_CRASH.search(data)
            return (m.group(0).decode(errors="replace").strip() if m else ""), f.tell()
    except OSError:
        return "", off


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def proc_table() -> dict:
    """pid -> (ppid, cpu seconds, rss bytes) for every visible process."""
    tick, page, out = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE"), {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                f = open(f"/proc/{p}/stat").read().rsplit(") ", 1)[1].split()
                out[int(p)] = (int(f[1]), (int(f[11]) + int(f[12])) / tick, int(f[21]) * page)
            except (OSError, IndexError):
                pass
    return out


def tree(table: dict, root: int) -> set:
    kids = {}
    for pid, (ppid, _, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    out, todo = set(), [root]
    while todo:
        pid = todo.pop()
        if pid in table and pid not in out:
            out.add(pid)
            todo += kids.get(pid, [])
    return out


class Procs:
    def __init__(self, out: Path):
        self.out, self.p, self.t0, self.peak, self.stop = out, {}, {}, {}, threading.Event()
        (out / "native-logs").mkdir(parents=True, exist_ok=True)

    def start(self, name: str, cmd: str, env: dict, cwd: str) -> None:
        log = open(self.out / "native-logs" / f"{name}.log", "w")
        log.write(f"$ {cmd}\n")
        log.flush()
        self.p[name] = subprocess.Popen(["bash", "-c", cmd], env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.t0[name] = time.time()
        print(f"[native] started {name} pid {self.p[name].pid}", flush=True)

    def wait_port(self, name: str, port: int, timeout: float) -> float:
        t0 = time.time()
        while not port_open(port):
            if self.p[name].poll() is not None:
                raise RuntimeError(f"{name} exited with {self.p[name].returncode} before listening on {port}")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"{name} not listening on {port} after {timeout:.0f} s")
            time.sleep(0.5)
        return time.time() - self.t0[name]

    def sample(self) -> None:
        with open(self.out / "usage.jsonl", "w", buffering=1) as f:
            while not self.stop.wait(1.0):
                table, gpu = proc_table(), {}
                try:
                    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    gpu = {int(a): float(b) for a, b in (l.split(",") for l in smi.stdout.splitlines() if "," in l)}
                except Exception:
                    pass
                rec = {"t": time.time()}
                for name, p in self.p.items():
                    pids = tree(table, p.pid)
                    v = {"gpu_mib": sum(gpu.get(x, 0.0) for x in pids), "cpu_s": sum(table[x][1] for x in pids),
                         "rss_gib": sum(table[x][2] for x in pids) / 2**30, "procs": len(pids)}
                    rec[name] = v
                    pk = self.peak.setdefault(name, {"gpu_mib": 0.0, "cpu_s": 0.0, "rss_gib": 0.0})
                    for k in pk:
                        pk[k] = max(pk[k], v[k])
                f.write(json.dumps(rec) + "\n")

    def kill_all(self) -> None:
        for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 2)):
            live = [p for p in self.p.values() if p.poll() is None]
            for p in live:  # each service is its own session: the group holds only what we started
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
            end = time.time() + wait
            while time.time() < end and any(p.poll() is None for p in live):
                time.sleep(0.2)


def host_cmd(svc: dict, src: str, subs: tuple = ()) -> str:
    """The compose command with container paths replaced by the host paths of its volume mounts."""
    cmd = svc["command"][-1].replace("umask 0000\n", "", 1).replace("$$", "$")
    mounts = dict(reversed(v.rsplit(":", 1)) for v in svc.get("volumes", []))  # container -> host
    mounts["/repo"] = src
    for cont in sorted(mounts, key=len, reverse=True):
        if cont != mounts[cont]:
            cmd = cmd.replace(cont, mounts[cont])
    for old, new in subs:
        cmd = cmd.replace(old, new)
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=os.path.expandvars("$DATA_DIR/third_party/alpasim"))
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--nuplan-root", default=os.path.expandvars("$DATA_DIR/datasets/alpasim_nuplan"))
    ap.add_argument("--driver", help="shell command of the driver; reads ALPASIM_DRIVER_HOST / ALPASIM_DRIVER_PORT")
    ap.add_argument("--driver-cwd")
    ap.add_argument("--driver-port", type=int, help="port of an already running driver (no --driver)")
    ap.add_argument("--tap", action="store_true", help="log every driver RPC to driver_tap.jsonl")
    ap.add_argument("--scene-list", help="file with one scene id per line; overrides the preset's scenes.scene_ids")
    ap.add_argument("--sub", action="append", default=[], metavar="OLD=NEW",
                    help="replace OLD by NEW in every service command after the mount rewrite (repeatable; NEW may be empty). PAI track: "
                         "--sub /app=<unpacked renderer image>/app --sub ' --enable-harmonizer=' --sub /tmp/nre-cache-dir=<run dir>/nre-cache")
    ap.add_argument("--ready-timeout", type=float, default=900)
    ap.add_argument("overrides", nargs="+", help="wizard arguments, e.g. +e2e_challenge_nuplan=dev scenes.limit_to_first_n=1")
    a = ap.parse_args()
    src, out = str(Path(a.src).resolve()), Path(a.log_dir).resolve()
    subs = tuple(x.split("=", 1) for x in a.sub)
    out.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ, UV_NO_SYNC="1", PYTHONUNBUFFERED="1", ALPASIM_NUPLAN_ROOT=a.nuplan_root)
    procs, times, rc = Procs(out), {"t_start": time.time()}, 1
    threading.Thread(target=procs.sample, daemon=True).start()
    try:
        port = a.driver_port
        if a.driver:
            port = free_port()
            procs.start("driver", a.driver, dict(base, ALPASIM_DRIVER_HOST="127.0.0.1", ALPASIM_DRIVER_PORT=str(port)),
                        a.driver_cwd or src)
            times["driver_ready_s"] = procs.wait_port("driver", port, a.ready_timeout)
        if a.tap:
            up, port = port, free_port()
            procs.start("tap", f"uv run python {HERE / 'driver_tap.py'} --listen 127.0.0.1:{port} "
                               f"--upstream 127.0.0.1:{up} --out {out / 'driver_tap.jsonl'}", base, src)
            procs.wait_port("tap", port, 120)

        # The wizard hands out the first free ports at or above baseport; start from a random one per run.
        if a.scene_list:
            ids = ",".join(f'"{x}"' for x in Path(a.scene_list).read_text().split())
            a.overrides += [f"scenes.scene_ids=[{ids}]", "scenes.limit_to_first_n=0"]
        wiz = ["uv", "run", "alpasim_wizard", *a.overrides, "wizard.run_method=NONE", f"wizard.log_dir={out}",
               f"wizard.baseport={random.randrange(20000, 30000)}"]
        t0 = time.time()
        with open(out / "native-logs" / "wizard.log", "w") as log:
            log.write("$ " + shlex.join(wiz) + "\n")
            log.flush()
            subprocess.run(wiz, cwd=src, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(base, ALPASIM_DRIVER_HOST="localhost", ALPASIM_DRIVER_PORT=str(port)))
        times["wizard_s"] = time.time() - t0

        services = yaml.safe_load(open(out / "docker-compose.yaml"))["services"]
        runtime = [n for n in services if n.startswith("runtime")]
        for name in [n for n in services if n not in runtime] + runtime:
            svc = services[name]
            env = dict(base, **dict(e.split("=", 1) for e in svc.get("environment", [])))
            if name in runtime:
                times["services_ready_s"] = time.time() - t0
                times["t_runtime"] = time.time()
            procs.start(name, host_cmd(svc, src, subs), env, src)
            for p in svc.get("ports", []):
                times[f"{name}_ready_s"] = procs.wait_port(name, int(str(p).split(":")[0]), a.ready_timeout)
        rt_log, rt_off = out / "native-logs" / f"{runtime[0]}.log", 0
        while (rc := procs.p[runtime[0]].poll()) is None:
            dead = [n for n, p in procs.p.items() if n not in runtime and p.poll() is not None]
            if dead:
                raise RuntimeError(f"service {dead} died while the runtime was running")
            crash, rt_off = worker_crash(rt_log, rt_off)
            if crash:
                raise RuntimeError(f"runtime worker crashed and the runtime does not recover: {crash}")
            time.sleep(1)
        times["runtime_s"] = time.time() - times["t_runtime"]
        # The runtime exits 0 with failed rollouts (allow_aggregation_with_failed_rollouts): count them here.
        rows = json.load(open(out / "aggregate" / "results-summary.json"))["rollouts"]
        times["rollouts"], times["rollouts_failed"] = len(rows), sum(bool(r.get("failure_reason")) for r in rows)
        if rc == 0 and rows and times["rollouts_failed"] == len(rows):
            raise RuntimeError("every rollout failed: " + str(rows[0]["failure_reason"])[:300])
    except Exception as e:
        print(f"[native] ERROR {e}", flush=True)
        times["error"], rc = str(e), rc or 1
    finally:
        procs.stop.set()
        procs.kill_all()
        times["total_s"] = time.time() - times["t_start"]
        json.dump({"rc": rc, "times": times, "peak": procs.peak, "overrides": a.overrides, "driver": a.driver,
                   "tap": a.tap, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                  open(out / "native_summary.json", "w"), indent=1)
        print(f"[native] rc {rc}, {times}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
