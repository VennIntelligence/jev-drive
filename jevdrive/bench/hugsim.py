"""HUGSIM closed-loop readouts through the harness every HUGSIM result used (experiments/hugsim/archive/zs_run.py: HUGSIM's
closed_loop.py from patched private trees, scored by HUGSIM itself), with the serving of experiments/op_parity/scripts/pp_hugsim.sh.

A run = (model, preset) over a scenario set; units are scenarios. Stages:
  onnx       parity arms: the arm's serving ONNX (pp_hugsim.py onnx: trained initializers + `intent_bias` input); an existing build
             under $DATA_DIR/runs/op_parity/hugsim/onnx is linked instead of rebuilt
  w<i>       K worker jobs (one per card by default), each starting its own servers (parity: bias server pp_hugsim.py serve +
             policy server hugsim_zs_server.py --onnx; onnx: policy server; WA-JEPA: none, its client loads the model) and W
             scenario slots. The slots of all jobs pull from one shared queue (claim files), longest expected scenario first,
             so the cards finish together. Every scenario is one `zs_run.py run` of that single scenario (resumable rows in
             <run>/results.csv), watched: no sim.log growth for `stall_s` or longer than `timeout_s` -> its process tree is
             stopped (jevdrive.cl.procs) and it is retried; before every scenario each server is checked (process alive, socket
             accepting; a policy-server reset would build a model instance) and the job's servers are restarted when one is dead
             or refusing. So a refused / dead socket costs one retry, not the 5400 s timeout of the old shell runners.
  collect    (envs/hugsim) units.csv: one row per scenario, HUGSIM's scores (hdscore, rc, nc, dac, ttc, c, pdms), the runner's
             end class and the behaviour of experiments/op_parity/results/hugsim_spin10.md (spin = heading error >= 60 deg vs the
             recorded route, experiments/hugsim/scripts/spin_analysis.py; launch stall = peak speed over the first 40 steps
             < 1.6 m/s; stuck = max_steps end; cls = spin, else the end class); summary.json; DONE.

Presets (jevdrive.openpilot.interface.HUGSIM_PRESETS, docs/openpilot-interface.md): exam = the wajepa_ref harness (preset exam, tree
`fixed`, PR #57 controller; every result before 2026-10-05), spec = openpilot's lateral path (tree opctrl, decision 118),
spec_plan = spec with the lateral curvature from the model's own plan. WA-JEPA runs exam only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import runner as R
from .models import REPO, Model, data_dir, resolve

ZS_RUN = REPO / "experiments/hugsim/archive/zs_run.py"
SERVER = REPO / "experiments/hugsim/archive/hugsim_zs_server.py"
PPH = REPO / "experiments/op_parity/scripts/pp_hugsim.py"
LEGACY_ONNX = lambda: data_dir() / "runs/op_parity/hugsim/onnx"  # noqa: E731
from ..openpilot.interface import HUGSIM_PRESETS

PRESETS = tuple(HUGSIM_PRESETS)
CONTROLLERS = ("official", "fixed", "ideal", "fixed2", "lowspeed", "lowsel", "opctrl", "opctrl_long", "fixedc")
CONTROLLER_ENV = ("OP_CTRL", "OP_CTRL_LONG", "LOWSPEED_CTRL", "LOWSPEED_SEL")
TAG = "bench"
AD = {"cinque": "zs", "lebowski": "zs", "small": "zs", "wajepa": "wj"}
STALL_S, TIMEOUT_S = 420.0, 1500.0      # sim.log silent this long -> stuck; any scenario longer than this -> stuck (max seen 723 s)


def configuration(preset="exam", opts=None, controller="", controller_env=None, repeat="", onnx="") -> dict:
    """Canonical behaviour identity; scenario sets and worker counts are execution choices, not arms."""
    preset = "spec" if preset == "opctrl_d118" else preset
    if preset not in HUGSIM_PRESETS:
        raise ValueError(f"unknown HUGSIM preset {preset!r}: {PRESETS}")
    p = HUGSIM_PRESETS[preset]
    tree = controller or p["controller"] or "fixed"
    if tree not in CONTROLLERS or (p["controller"] and tree != p["controller"]):
        raise ValueError(f"preset {preset} cannot use controller {tree!r}")
    def obj(value):
        value = json.loads(value) if isinstance(value, str) else {} if value is None else value
        if not isinstance(value, dict):
            raise ValueError("HUGSIM opts / controller-env must be JSON objects")
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    options, env = obj(opts), obj(controller_env)
    if "parity" in options and not isinstance(options["parity"], dict):
        raise ValueError("parity options must be an object")
    if options.get("parity", {}).get("socket"):
        raise ValueError("bench manages the parity socket")
    for k, v in env.items():
        if k not in CONTROLLER_ENV:
            raise ValueError(f"unsupported controller env {k!r}: {CONTROLLER_ENV}")
        env[k] = obj(v)
    repeat = str(repeat) if repeat is not None else ""
    if repeat and not re.fullmatch(r"[A-Za-z0-9_.-]+", repeat):
        raise ValueError("repeat must be a filename-safe label")
    from .models import expand
    return dict(preset=preset, controller=tree, opts=options, controller_env=env, repeat=repeat,
                onnx=str(Path(expand(onnx)).resolve()) if onnx else "")


def run_key(m: Model, preset: str, **kw) -> str:
    cfg = configuration(preset, **kw)
    key = f"{m.key('hugsim')}_{cfg['preset']}"
    identity = {k: cfg[k] for k in ("controller", "opts", "controller_env", "onnx")}
    default = configuration(preset)
    if identity != {k: default[k] for k in identity}:
        key += "-c" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    if cfg["repeat"]:
        key += "-r" + cfg["repeat"]
    return key


def onnx_path(m: Model) -> Path:
    if m.family == "adapt_h":
        return R.bench_root("hugsim", "_onnx", f"{m.name}.onnx")
    return R.bench_root("hugsim", "_onnx", "pp-shipped.onnx" if m.shipped else f"pp-{m.name}.onnx")


def agent_of(m: Model) -> str:
    return "wajepa" if m.family == "wajepa" else m.base


# ---------------------------------------------------------------- planning
def expected_walls(run_dir: Path) -> dict:
    """scenario -> median wall (s) over earlier HUGSIM runs (LPT order of the queue)."""
    import statistics
    w = {}
    for f in (data_dir() / "runs/op_parity/hugsim/results.csv", Path(run_dir) / "results.csv"):
        if f.exists():
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    try:
                        wall = float(r["wall_s"])
                        w.setdefault(r["scenario"], []).append(wall)
                    except (KeyError, ValueError):
                        pass
    return {k: statistics.median(v) for k, v in w.items()}


def done_set(run_dir: Path) -> set:
    f = Path(run_dir) / "results.csv"
    if not f.exists():
        return set()
    with open(f) as fh:
        return {r["scenario"] for r in csv.DictReader(fh) if r["end"] != "crash" and r["tag"] == TAG}


def stages(m: Model, preset: str, run_dir: Path, scenarios: list, workers: int = 6, jobs: int = 0, stall_s: float = STALL_S,
           timeout_s: float = TIMEOUT_S, retries: int = 2, opts=None, controller="", controller_env=None, repeat="", onnx="") -> list:
    identity = configuration(preset, opts, controller, controller_env, repeat, onnx)
    if identity["onnx"]:
        if m.family != "onnx":
            raise ValueError("--onnx overrides apply only to ONNX models")
        if not Path(identity["onnx"]).is_file():
            raise FileNotFoundError(identity["onnx"])
    preset = identity["preset"]
    if m.family == "wajepa" and preset != "exam":
        raise SystemExit("WA-JEPA runs the exam preset only (its client has no openpilot lateral path)")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cf = run_dir / "config.json"
    old = json.loads(cf.read_text()) if cf.exists() else {}
    if old and configuration(**{k: old.get(k) for k in identity}) != identity:
        raise ValueError(f"different HUGSIM configuration already exists in {run_dir}")
    want = list(dict.fromkeys(list(old.get("scenarios", [])) + list(scenarios)))
    walls = expected_walls(run_dir)
    med = sorted(walls.values())[len(walls) // 2] if walls else 100.0
    want.sort(key=lambda s: -walls.get(Path(s).stem, med))
    done = done_set(run_dir)
    todo = [s for s in want if Path(s).stem not in done]
    for f in (run_dir / "claims").glob("*.failed") if (run_dir / "claims").is_dir() else []:
        f.unlink()                                                    # a new submit retries scenarios that failed before
    w = max(1, min(workers, len(todo) or 1))
    k = jobs or max(1, min(len(R.box_cards()), -(-len(todo) // w)))
    cfg = dict(model=m.spec, **identity, agent=agent_of(m), scenarios=want, workers=w, jobs=k, stall_s=stall_s,
               timeout_s=timeout_s, retries=retries, tag=TAG)
    p = HUGSIM_PRESETS[preset]
    cfg["resolved_opts"] = dict(p["opts"], **identity["opts"])
    cfg["resolved_controller_env"] = dict(p["env"], **identity["controller_env"])
    R.atomic_write(cf, json.dumps(cfg, indent=1))
    S = []
    if m.family in ("parity", "adapt_h"):
        S.append(R.Stage("onnx", R.stage_cmd("jev", "hugsim-onnx", m.spec), done=str(onnx_path(m)), vram=10, cpu=4, ram=24))
    per = 12.5 if m.family == "wajepa" else 8.5
    base = 6 if m.family == "wajepa" else 8
    if todo:
        for i in range(k):
            S.append(R.Stage(f"w{i}", R.stage_cmd("jev", "hugsim-worker", run_dir, i), done=str(run_dir / "workers" / f"w{i}.DONE"),
                             vram=round(base + per * w, 1), cpu=2 * w + 3, ram=6 * w + 8, after=[s.name for s in S if s.name == "onnx"],
                             tries=2))
    S.append(R.Stage("collect", R.stage_cmd("hugsim", "hugsim-collect", run_dir), done=str(run_dir / "DONE"), vram=0.5, cpu=4, ram=16,
                     after=[s.name for s in S if s.name.startswith("w")]))
    for f in [run_dir / "DONE", run_dir / "ERROR", run_dir / "WAIT_TIMEOUT"] + [run_dir / "workers" / f"w{i}.DONE" for i in range(k)]:
        if todo and f.exists():                                       # new work: the run is open again
            f.rename(f.with_name(f"{f.name}.{time.strftime('%Y%m%d-%H%M%S')}"))
    return S


# ---------------------------------------------------------------- stage: ONNX
def build_onnx(spec: str) -> None:
    m = resolve(spec, check=True)
    out = onnx_path(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    if m.family == "adapt_h":
        build_h_onnx(m, out)
        return
    leg = LEGACY_ONNX() / out.name
    if leg.exists():
        if not out.exists():
            out.symlink_to(leg)
        return
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.onnx")
    subprocess.run([R.py("op-train"), str(PPH), "onnx", "--tag", "P0" if m.shipped else m.name, "--out", str(tmp)], check=True, cwd=REPO)
    os.replace(tmp, out)


def build_h_onnx(m: Model, out: Path) -> None:
    """H lane's no-adapter export and the same stream equivalence gate, before publishing the serving ONNX."""
    script = REPO / "experiments/op_adapt_l/scripts/op_l_onnx.py"
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.onnx")
    ref = out.with_suffix(".ref.npz")
    log = out.with_suffix(".check.txt")
    subprocess.run([R.py("op-train"), str(script), "build", "--ckpt", m.ckpt, "--out", str(tmp), "--no-adapter"], check=True, cwd=REPO)
    subprocess.run([R.py("op-train"), str(script), "ref", "--ckpt", m.ckpt, "--out", str(ref)], check=True, cwd=REPO)
    with open(log, "w") as f:
        subprocess.run([R.py("openpilot"), str(script), "check", "--onnx", str(tmp), "--ref", str(ref)], check=True, cwd=REPO, stdout=f,
                       stderr=subprocess.STDOUT)
    streams = [line.split() for line in log.read_text().splitlines() if line.startswith("stream")]
    if not streams or any(float(row[15]) > 0.5 for row in streams):
        raise RuntimeError(f"H ONNX plan xy differs from training port by > 0.5 m: {log}")
    os.replace(tmp, out)


# ---------------------------------------------------------------- servers
def _wire():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import zeroshot_wire as wire
    return wire


def ping(path: str, timeout: float = 20.0) -> bool:
    """A reset round trip (bias server only: a reset on the policy server builds a model instance)."""
    wire = _wire()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        wire.send(s, {"cmd": "reset"}, {})
        meta, _ = wire.recv(s)
        return bool(meta.get("ok"))
    except (OSError, ValueError, KeyError):
        return False
    finally:
        s.close()


def accepts(path: str, timeout: float = 10.0) -> bool:
    """Does the server's listening socket accept a connection? (no request: the server's connection thread sees EOF and ends)"""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        return True
    except OSError:
        return False
    finally:
        s.close()


class Server:
    def __init__(self, name: str, cmd: list, sock: str, ready: Path, log: Path, env: dict):
        self.name, self.cmd, self.sock, self.ready, self.log, self.env = name, cmd, sock, ready, log, env
        self.p, self.rec = None, ready.with_suffix(".procs.json")

    def start(self, timeout: float = 900.0) -> None:
        from ..cl import procs
        self.ready.unlink(missing_ok=True)
        with open(self.log, "a") as lf:
            lf.write(f"\n==> {time.strftime('%F %T')} start {' '.join(self.cmd)}\n")
            lf.flush()
            self.p = subprocess.Popen(self.cmd, cwd=REPO, env=self.env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        procs.capture(self.rec, self.p.pid)
        t0 = time.time()
        while not self.ready.exists():
            if self.p.poll() is not None:
                raise RuntimeError(f"{self.name} died at start (rc {self.p.returncode}): {self.log}")
            if time.time() - t0 > timeout:
                self.stop()
                raise RuntimeError(f"{self.name} not ready after {timeout:.0f} s: {self.log}")
            time.sleep(2)

    def alive(self) -> bool:
        """Process running and its socket accepting (twice, 5 s apart, before calling it dead)."""
        if self.p is None or self.p.poll() is not None:
            return False
        return accepts(self.sock) or (time.sleep(5) is None and accepts(self.sock))

    def stop(self) -> None:
        from ..cl import procs
        if self.rec.exists():
            procs.refresh(self.rec)
            procs.stop(self.rec, grace=10)


class Servers:
    """The model's resident servers of one worker job; `ensure` pings them and restarts them when needed."""

    def __init__(self, m: Model, run_dir: Path, i: int):
        self.m, self.lock, self.restarts = m, threading.Lock(), 0
        d = run_dir / "servers"
        d.mkdir(parents=True, exist_ok=True)
        tmp = Path("/tmp") / f"jb-{os.getpid()}-{i}"
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        self.list, self.op_sock, self.bias_sock = [], "", ""
        if m.family == "parity":
            self.bias_sock = str(tmp / "bias.sock")
            tag = "P3-init" if m.name == "P0" else m.name          # P0: shipped weights + an untrained P3 adapter (bias exactly 0)
            self.list.append(Server("bias", [R.py("op-train"), "-u", str(PPH), "serve", "--tag", tag, "--socket", self.bias_sock,
                                             "--ready-file", str(d / f"w{i}-bias.ready")], self.bias_sock, d / f"w{i}-bias.ready",
                                    d / f"w{i}-bias.log", env))
        if m.family in ("parity", "onnx", "adapt_h"):
            self.op_sock = str(tmp / "op.sock")
            onnx = str(onnx_path(m)) if m.family in ("parity", "adapt_h") else m.onnx
            self.list.append(Server("policy", [R.py("openpilot"), "-u", str(SERVER), m.base] + (["--onnx", onnx] if onnx else [])
                                    + ["--socket", self.op_sock, "--ready-file", str(d / f"w{i}-op.ready")], self.op_sock,
                                    d / f"w{i}-op.ready", d / f"w{i}-op.log", env))

    def start(self) -> None:
        for s in self.list:
            s.start()

    def ensure(self) -> None:
        with self.lock:
            bad = [s.name for s in self.list if not s.alive()]
            if not bad:
                return
            self.restarts += 1
            print(f"{time.strftime('%T')} servers {bad} unhealthy: restart {self.restarts}", flush=True)
            if self.restarts > 6:
                raise RuntimeError(f"servers {bad} keep failing (6 restarts)")
            self.stop()
            self.start()

    def stop(self) -> None:
        for s in self.list:
            s.stop()


# ---------------------------------------------------------------- the shared scenario queue
class Queue:
    """Claim files <run>/claims/<stem>.claim (O_EXCL), heartbeat = the file's mtime; stale claims (no heartbeat for 300 s) are
    taken over; <stem>.failed marks a scenario that exhausted its retries in this submission."""

    STALE_S = 300.0

    def __init__(self, run_dir: Path, scenarios: list, i: int):
        self.dir = run_dir / "claims"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_dir, self.scen, self.i, self.mine, self.lock = run_dir, scenarios, i, set(), threading.Lock()
        for f in self.dir.glob("*.claim"):                          # claims of an earlier attempt of this worker job
            try:
                if json.loads(f.read_text()).get("worker") == i:
                    f.unlink()
            except (OSError, ValueError):
                pass

    def _claim(self, f: Path) -> bool:
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - f.stat().st_mtime > self.STALE_S:
                    f.rename(f.with_name(f"{f.name}.stale.{time.time():.0f}"))
                    return self._claim(f)
            except OSError:
                pass
            return False
        os.write(fd, json.dumps(dict(worker=self.i, job=os.environ.get("CL_POOL_JOB", ""), pid=os.getpid(), t=time.time())).encode())
        os.close(fd)
        return True

    def next(self):
        with self.lock:
            done = done_set(self.run_dir)
            for s in self.scen:
                st = Path(s).stem
                if st in done or (self.dir / f"{st}.failed").exists():
                    continue
                f = self.dir / f"{st}.claim"
                if self._claim(f):
                    self.mine.add(f)
                    return s
            return None

    def release(self, s: str, failed: str = "") -> None:
        f = self.dir / f"{Path(s).stem}.claim"
        if failed:
            R.atomic_write(self.dir / f"{Path(s).stem}.failed", failed + "\n")
        with self.lock:
            self.mine.discard(f)
        f.unlink(missing_ok=True)

    def beat(self) -> None:
        with self.lock:
            for f in list(self.mine):
                try:
                    os.utime(f)
                except OSError:
                    pass


# ---------------------------------------------------------------- stage: worker
def scenario_dir(run_dir: Path, scen: str, agent: str) -> Path:
    import yaml
    c = yaml.safe_load((data_dir() / "datasets/hugsim/scenarios" / scen).read_text())
    return run_dir / TAG / AD[agent] / f"{c['scene_name']}_{c['mode']}"


def preset_args(preset: str, controller="") -> list:
    cfg = configuration(preset, controller=controller)
    return ["--preset", cfg["preset"], "--controller", cfg["controller"]]


def run_one(cfg: dict, run_dir: Path, scen: str, srv: Servers, gpu: str, log: Path) -> str:
    """One scenario through zs_run.py under the watchdog -> 'done' | 'crash' | 'stall' | 'timeout'."""
    from ..cl import procs
    opts = dict(cfg.get("opts", {}))
    if srv.bias_sock:
        opts["parity"] = dict(opts.get("parity", {}), socket=srv.bias_sock)
    cmd = [R.py("hugsim"), str(ZS_RUN), "run", "--out", str(run_dir), "--agent", cfg["agent"],
           *preset_args(cfg["preset"], cfg.get("controller", "")), "--controller-env", json.dumps(cfg.get("controller_env", {})),
           "--gpu", gpu, "--workers", "1", "--scenarios", scen, "--tag", TAG, "--timeout", str(cfg["timeout_s"]), "--retries", "0",
           "--max-fail", "0", "--opts", json.dumps(opts)] + (["--socket", srv.op_sock] if srv.op_sock else [])
    sd = scenario_dir(run_dir, scen, cfg["agent"])
    rec = run_dir / "workers" / f"{Path(scen).stem}.procs.json"
    t0 = last = time.time()
    size = -1
    with open(log, "a") as lf:
        lf.write(f"\n==> {time.strftime('%F %T')} {scen}\n")
        lf.flush()
        env = {k: v for k, v in os.environ.items() if k not in CONTROLLER_ENV}
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.capture(rec, p.pid)
    why = ""
    while p.poll() is None:
        time.sleep(5)
        procs.refresh(rec)
        try:
            sz = (sd / "sim.log").stat().st_size
        except OSError:
            sz = -1
        if sz != size:
            size, last = sz, time.time()
        if time.time() - last > cfg["stall_s"]:
            why = "stall"
        elif time.time() - t0 > cfg["timeout_s"] + 60:
            why = "timeout"
        if why:
            procs.stop(rec, grace=10)
            p.wait()
            break
    rec.unlink(missing_ok=True)
    if why:
        return why
    return "done" if Path(scen).stem in done_set(run_dir) else "crash"


def worker(run_dir: str, i: int) -> None:
    run_dir = Path(run_dir)
    i = int(i)
    cfg = json.loads((run_dir / "config.json").read_text())
    m = resolve(cfg["model"])
    if cfg.get("onnx"):
        m = replace(m, onnx=cfg["onnx"])
    gpu = os.environ.get("CL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    (run_dir / "workers").mkdir(parents=True, exist_ok=True)
    tree = cfg.get("controller") or configuration(cfg["preset"])["controller"]
    subprocess.run([R.py("hugsim"), str(ZS_RUN), "setup-trees", tree], check=True, cwd=REPO)
    q = Queue(run_dir, cfg["scenarios"], i)
    srv = Servers(m, run_dir, i)
    stop_beat = threading.Event()

    def beat():
        while not stop_beat.wait(30):
            q.beat()
    threading.Thread(target=beat, daemon=True).start()
    stats = dict(done=0, failed=0, retries=0)
    lock = threading.Lock()

    def slot(k: int):
        time.sleep(8 * k)                                            # staggered scene loads
        log = run_dir / "workers" / f"w{i}-slot{k}.log"
        while True:
            s = q.next()
            if s is None:
                return
            res = ""
            for att in range(1 + cfg["retries"]):
                srv.ensure()
                res = run_one(cfg, run_dir, s, srv, gpu, log)
                if res == "done":
                    break
                with lock:
                    stats["retries"] += 1
                print(f"{time.strftime('%T')} w{i}: {Path(s).stem} {res} (attempt {att + 1})", flush=True)
            q.release(s, "" if res == "done" else f"{res} after {1 + cfg['retries']} attempts")
            with lock:
                stats["done" if res == "done" else "failed"] += 1
                n = len(done_set(run_dir))
            R.status(run_dir, f"w{i}: {n} / {len(cfg['scenarios'])} scenarios done, this job {stats}")

    try:
        srv.start()
        th = [threading.Thread(target=slot, args=(k,)) for k in range(cfg["workers"])]
        for t in th:
            t.start()
        for t in th:
            t.join()
    finally:
        stop_beat.set()
        srv.stop()
    R.atomic_write(run_dir / "workers" / f"w{i}.DONE", json.dumps(dict(t=time.strftime("%F %T"), **stats, restarts=srv.restarts)) + "\n")


# ---------------------------------------------------------------- stage: collect (envs/hugsim)
def routes() -> dict:
    f = data_dir() / "runs/op_parity/hugsim/routes.json"
    if not f.exists():
        f = R.bench_root("hugsim", "routes.json")
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(REPO / "experiments/hugsim/scripts/spin_export_routes.py"),
                            str(REPO / "experiments/hugsim/results/hugsim-exam/scored_op.csv"), str(f)], check=True)
    return json.loads(f.read_text())


def behaviour(row: dict, rts: dict) -> dict:
    """Spin / launch stall / standing of one finished run (pp_hugsim_report.py extract)."""
    sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
    import spin_analysis as SA
    d = Path(row["run_dir"])
    pos, th, v, steer, plans = SA.load_run(d, row["agent"])
    res, _ = SA.analyse(pos, th, v, steer, plans, rts[row["scene"]])
    return dict(spin=bool(res["spin"]), max_abs_e=res["max_abs_e"], k60=res.get("k60", -1), v_max40=float(v[:40].max()),
                v_max=float(v.max()), v_end=float(v[-1]), standing=float((v < 0.3).mean()), n_steps=len(v))


CLS = {"max_steps": "stuck", "bg_collision": "bg_coll", "fg_collision": "fg_coll"}
NUM = ("hdscore", "rc", "nc", "dac", "ttc", "c", "pdms")


def units_from_rows(rows: list, rts: dict, want=None) -> list:
    """Last finished row per scenario (+ behaviour) -> unit dicts."""
    last = {}
    for r in rows:
        if r["end"] != "crash" and (want is None or r["scenario"] in want):
            last[r["scenario"]] = r
    out = []
    for sc, r in last.items():
        u = dict(scenario=sc, dataset=r["dataset"], difficulty=r["difficulty"], scene=r["scene"], end=r["end"], steps=int(r["steps"]),
                 wall_s=float(r["wall_s"]), run_dir=r["run_dir"], **{k: float(r[k]) if r[k] != "" else float("nan") for k in NUM})
        try:
            u.update(behaviour(r, rts))
        except Exception as e:                                       # a run without infos.pkl: scores only
            u.update(spin=None, behaviour_error=repr(e)[:200])
        u["cls"] = "spin" if u.get("spin") else CLS.get(u["end"], u["end"])
        u["launch_stall"] = bool(u.get("v_max40", 99) < 1.6)
        u["stuck"] = u["end"] == "max_steps"
        out.append(u)
    return out


def collect(run_dir: str) -> None:
    import pandas as pd
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    rows = [r for r in csv.DictReader(open(run_dir / "results.csv")) if r["tag"] == TAG] if (run_dir / "results.csv").exists() else []
    want = {Path(s).stem for s in cfg["scenarios"]}
    u = pd.DataFrame(units_from_rows(rows, routes(), want))
    u.to_csv(run_dir / "units.csv", index=False)
    missing = sorted(want - set(u.scenario)) if len(u) else sorted(want)
    failed = {f.stem: f.read_text().strip() for f in (run_dir / "claims").glob("*.failed")} if (run_dir / "claims").is_dir() else {}
    summ = dict(model=cfg["model"], preset=cfg["preset"], n=len(u), missing=missing, failed=failed,
                HD=float(u.hdscore.mean()) if len(u) else float("nan"), RC=float(u.rc.mean()) if len(u) else float("nan"))
    if len(u):
        summ |= {f"n_{c}": int((u.cls == c).sum()) for c in ("complete", "fg_coll", "bg_coll", "off_route", "stuck", "spin")}
        summ |= dict(n_launch_stall=int(u.launch_stall.sum()), n_spin_any=int(u.spin.fillna(False).astype(bool).sum()))
    R.atomic_write(run_dir / "summary.json", json.dumps(summ, indent=1, default=str))
    if missing:
        R.atomic_write(run_dir / "ERROR", f"missing {len(missing)} scenarios: {', '.join(missing)}\n")
        raise RuntimeError(f"HUGSIM incomplete: {len(missing)} scenarios missing; {run_dir}/summary.json")
    R.status(run_dir, f"done: {len(u)} scenarios, HD {summ['HD']:.3f}" + (f", MISSING {len(missing)}" if missing else ""))
    R.atomic_write(run_dir / "DONE", json.dumps(dict(t=time.strftime("%F %T"), **summ), default=str) + "\n")
