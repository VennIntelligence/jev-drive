"""The GPU pool: agents submit jobs, one dispatcher puts each on whichever card has room ("any free card takes the next
job"). Nobody picks cards, cores or CARLA ports by hand.

Spool, under $DATA_DIR/runs/pool/ (the dispatcher is its only writer, except inbox/, cancel/, retarget/ and holds.json):

  inbox/<id>.json   a submitted job (written by `submit`, atomic rename); the dispatcher moves it into its state
  cancel/<id>       a cancel request (`cancel`; content "drain" = touch the job's DRAIN file instead of stopping it)
  retarget/<id>     new allowed cards for a queued job (`retarget`; content "0,2", empty = any); the id stays, so
                    `after` chains keep working
  holds.json        resources held outside the pool (`hold`): a card or part of it, server indices, cores; a hold
                    with a pid ends by itself when that process exits
  state.json        every job: spec, state, card, server-index block, cores, tries, why it waits (dispatcher only)
  status.json       the last round: per-card accounting, heartbeat (read by `queue` / `top`)
  jobs/<id>/        owned.json (process tree by pid + start time), rc.<k>; log_dir defaults here
  dispatch.lock     one dispatcher at a time (flock)

A job: cmd (argv, or one shell string), cwd, log_dir, env, need = {vram_gb, carla servers, cpu cores, ram_gb, train},
priority (higher first, then submit order), owner, after (job ids that must be done first), when_exists (a file gate),
gpus (allowed cards), exclusive, tries, timeout_h, max_rss_gb, profile (thread env from profiles.py).

Every round (default 20 s) the dispatcher reaps finished jobs, then admits queued ones in priority order. A job fits a
card when (all measured live, nvidia-smi cached <= 15 s):
  VRAM      total - headroom - demand >= vram_gb, demand = (VRAM used by processes outside the pool, at least what
            holds declare) + sum over the card's pool jobs of max(declared, measured). Declared VRAM stays reserved for
            a job's whole life, so a job that has not reached its peak yet is never overbooked.
  CARLA     pool servers (max(declared, live)) + servers outside the pool + carla <= carla_per_card (default 6); at most
            one CARLA job starts per card per round (staggered server starts).
  training  train jobs on the card < train_per_card (default 2).
  ports     a free block of 2 x carla server indices: every RPC (2000 + 50 i) and traffic-manager (8000 + 50 i = the RPC
            block of i + 120) port block clear of pool jobs, holds and every LISTENING port on the box (/proc/net/tcp),
            inside capacity.index_bounds. A block is freed only when every process of the job's tree has exited.
  CPU       sum of declared cores (pool + holds) <= cpu budget (cgroup quota x cpu_overcommit); cpu > 0 pins the job
            (taskset) to free physical cores, NUMA-local to the card first.
  PIDs/RAM  pids.current + threads of jobs younger than 5 min + the job's estimate <= 0.80 pids.max; cgroup memory
            without page cache (anon + shmem + kernel) + young ram_gb + ram_gb <= 0.85 memory.max.
Among the cards that fit, the least loaded (fewest pool jobs + foreign GPU processes) wins and VRAM best-fit breaks ties:
spreading keeps every card computing, best-fit keeps room for a large job. A job blocked > hold_s (15 min) at the head
of the queue reserves the card closest to fitting it: lower-priority jobs stop starting there until it starts.

A job's process tree is recorded by (pid, start time) plus a CL_TOKEN env entry (procs.py). When its root exits, any
member still alive is stopped through a pidfd; resources are freed only after the tree is empty. Cancel, timeout and the
RSS cap stop the tree the same way. Nothing is ever found or killed by pattern.

Job env: the dispatcher's env + the profile's thread env + the job's env + CUDA_VISIBLE_DEVICES={gpu}, CL_GPU, CL_IDX,
CL_SPAN, CL_CPUS, CL_POOL_JOB, CL_JOB_DIR, B2D_DRAIN_FILE, B2D_PIDS_WAIT. cmd and env may use {gpu} {idx} {span}
{carla} {workers} {cpus} {server_args} {client_threads} {pids_wait} {job_dir} {id}.
log_dir gets log.txt (all tries), STATUS (one line), DONE (JSON) or ERROR (reason + log tail).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import capacity, procs, profiles
from .box import CARLA_BIN, format_cpus, parse_cpus, probe, processes, smi

REPO = Path(__file__).resolve().parents[2]
YOUNG_S = 300.0
STALE_S = 120.0                       # status.json older than this: the dispatcher is not running
DEFAULTS = dict(poll_s=20.0, headroom_gb=4.0, carla_per_card=capacity.GPU_KNEE, train_per_card=2, cpu_overcommit=1.0,
                max_starts=4, hold_s=900.0, cards=None, smi_age_s=15.0)
FINAL = ("done", "failed", "cancelled")


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", str(Path.home() / "data")))


def pool_dir() -> Path:
    return Path(os.environ.get("CL_POOL_DIR", str(data_dir() / "runs" / "pool")))


def _read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


@contextmanager
def _flock(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


# ---------------------------------------------------------------------------------------------------- job spec
@dataclass
class Spec:
    cmd: object                                   # argv list, or one shell string (run with bash -c)
    name: str = "job"
    owner: str = ""
    cwd: str = ""                                 # default: the repo
    log_dir: str = ""                             # default: <pool>/jobs/<id>
    env: dict = field(default_factory=dict)
    vram_gb: float = 0.0
    carla: int = 0                                # CARLA servers the job starts on its card
    span: int = 0                                 # server indices to reserve (default 2 x carla)
    cpu: int = 0                                  # cores; > 0 pins the job to that many free physical cores
    ram_gb: float = 0.0
    threads: int = 0                              # PID estimate (default from the thread model)
    train: bool = False                           # counts against train_per_card
    exclusive: bool = False                       # alone on its card
    gpus: list = field(default_factory=list)      # allowed cards (empty: any)
    priority: float = 0.0                         # higher first
    after: list = field(default_factory=list)     # job ids that must be done (a failed one fails this job)
    when_exists: str = ""                         # stay queued until this path exists
    tries: int = 1
    timeout_h: float = 0.0
    max_rss_gb: float = 0.0                       # stop the job when its tree's RSS exceeds this
    profile: str = ""                             # thread profile (profiles.py); default for CARLA jobs: profiles.DEFAULT

    def check(self) -> "Spec":
        if not self.cmd:
            raise ValueError("empty cmd")
        if self.vram_gb <= 0 and self.carla:
            self.vram_gb = self.carla * capacity.VRAM_PER_WORKER_GB
        if self.vram_gb <= 0 and not self.exclusive:
            raise ValueError("declare vram_gb (the job's peak VRAM on its card) or exclusive")
        if self.profile and self.profile not in profiles.PROFILES:
            raise ValueError("unknown profile %s" % self.profile)
        self.after = [a for a in (self.after or []) if a]
        self.gpus = [int(g) for g in self.gpus or []]
        return self

    def est_threads(self, host_cpus: int) -> int:
        if self.threads:
            return self.threads
        prof = profiles.get(self.profile or None)
        per_core = max(self.cpu / max(self.carla, 1), 4) if self.cpu else 8
        return 100 + self.carla * capacity.worker_threads(prof, per_core, host_cpus)


def new_id(name: str = "") -> str:
    return "%s-%s" % (time.strftime("%m%d-%H%M%S"), secrets.token_hex(2))


def submit(cmd, name: str = "job", pool: Path = None, **kw) -> str:
    """Queue a job; returns its id. kw: any Spec field."""
    kw.setdefault("owner", os.environ.get("CL_OWNER") or os.environ.get("CL_POOL_JOB", "") or os.environ.get("USER", ""))
    spec = Spec(cmd=cmd, name=name, **kw).check()
    pool = Path(pool or pool_dir())
    jid = new_id(name)
    procs.atomic_json(pool / "inbox" / ("%s.json" % jid), dict(id=jid, t_submit=time.time(), spec=spec.__dict__))
    return jid


def preflight(cmd, name: str, timeout_min: float = 15.0, pool: Path = None, **kw) -> str:
    """Queue a short smoke run of `cmd` (the caller's job then goes `after` it) and return its id. It starts ahead of
    normal work (priority + 1000) with the caller's resources, so a broken script fails in one smoke run instead of in
    every job of a batch. Identical smoke commands (same cmd and cwd, last 6 h, not failed / cancelled) are reused."""
    pool = Path(pool or pool_dir())
    key = hashlib.sha1(json.dumps([cmd, kw.get("cwd") or ""], sort_keys=True).encode()).hexdigest()[:12]
    st, _, inbox, _ = snapshot(pool)
    for j in list(st["jobs"].values()) + [dict(x, state="inbox") for x in inbox]:
        if (j["spec"].get("env") or {}).get("CL_PREFLIGHT_KEY") == key and j["state"] not in ("failed", "cancelled") \
                and time.time() - j.get("t_submit", 0) < 6 * 3600:
            return j["id"]
    kw = {k: v for k, v in kw.items() if k not in ("after", "when_exists", "tries", "timeout_h", "priority")}
    kw["env"] = dict(kw.get("env") or {}, CL_PREFLIGHT="1", CL_PREFLIGHT_KEY=key)
    if kw.get("log_dir"):
        kw["log_dir"] = str(Path(kw["log_dir"]) / "preflight")
    return submit(cmd, name=name + "-pf", pool=pool, priority=1000.0, tries=1,
                  timeout_h=timeout_min / 60, **kw)


def static_check(cmd, cwd: str = "") -> list:
    """Problems visible without running: script paths in the command that do not exist, Python files that do not
    compile. Returns messages (empty = fine)."""
    toks = shlex.split(cmd) if isinstance(cmd, str) else [str(x) for x in cmd]
    base, out = Path(cwd or REPO), []
    for t in toks:
        if not re.fullmatch(r"[\w./~-]+\.(py|sh)", t):
            continue
        f = Path(os.path.expanduser(t))
        f = f if f.is_absolute() else base / f
        if not f.exists():
            out.append("missing %s" % f)
        elif f.suffix == ".py":
            try:
                compile(f.read_bytes(), str(f), "exec")
            except SyntaxError as e:
                out.append("syntax error %s:%s %s" % (f, e.lineno, e.msg))
    return out


def retarget(jid: str, gpus, pool: Path = None) -> None:
    """Change the allowed cards of a queued job (empty = any card)."""
    p = Path(pool or pool_dir()) / "retarget" / jid
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(",".join(str(int(g)) for g in gpus))


def cancel(jid: str, drain: bool = False, pool: Path = None) -> None:
    p = Path(pool or pool_dir()) / "cancel" / jid
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("drain" if drain else "stop")


# ---------------------------------------------------------------------------------------------------- holds
def load_holds(pool: Path = None) -> list:
    return _read_json(Path(pool or pool_dir()) / "holds.json", []) or []


def add_hold(card: int, note: str, whole: bool = False, vram_gb: float = 0.0, carla: int = 0, cpu: int = 0,
             cpus: str = "", idx: str = "", pid: int = 0, train: bool = False, pool: Path = None) -> dict:
    """Resources held outside the pool (a job started by hand, a running legacy chain). With pid, the hold ends when
    that process (identified by pid + start time) exits."""
    pool = Path(pool or pool_dir())
    h = dict(id=secrets.token_hex(3), card=int(card), note=note, whole=whole, vram_gb=vram_gb, carla=carla,
             cpu=cpu or len(parse_cpus(cpus)), cpus=cpus, idx=idx, train=train, t=time.time())
    if pid:
        rows = processes()
        if pid not in rows:
            raise RuntimeError("pid %d is not running" % pid)
        h["proc"] = procs.identity(rows[pid])
    with _flock(pool / "holds.lock"):
        procs.atomic_json(pool / "holds.json", load_holds(pool) + [h])
    return h


def drop_hold(hid: str, pool: Path = None) -> bool:
    pool = Path(pool or pool_dir())
    with _flock(pool / "holds.lock"):
        hs = load_holds(pool)
        keep = [h for h in hs if h["id"] != hid]
        procs.atomic_json(pool / "holds.json", keep)
    return len(keep) < len(hs)


def live_holds(holds: list, rows: dict) -> list:
    return [h for h in holds if not h.get("proc") or procs.same(h["proc"], rows)]


def idx_range(spec: str) -> list:
    return parse_cpus(spec) if spec else []


# ---------------------------------------------------------------------------------------------------- live state
def listening_ports(root: Path = Path("/")) -> set:
    out = set()
    for f in ("proc/net/tcp", "proc/net/tcp6"):
        try:
            lines = (root / f).read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            p = ln.split()
            if len(p) > 3 and p[3] == "0A":
                out.add(int(p[1].rsplit(":", 1)[1], 16))
    return out


def port_blocks(ports) -> set:
    """Port blocks in use: block j = ports 2000 + 50 j .. +49 (index j's RPC block, or index j - 120's TM block)."""
    S, B = capacity.PORT_STRIDE, capacity.PORT_BASE
    return {(p - B) // S for p in ports if p >= B}


TM_SHIFT = (capacity.TM_BASE - capacity.PORT_BASE) // capacity.PORT_STRIDE     # 120


def blocks_of(indices) -> set:
    """Port blocks a server index uses: its RPC block i and its traffic-manager block (= RPC block of i + 120)."""
    return {b for i in indices for b in (i, i + TM_SHIFT)}


def free_index_block(used_blocks: set, span: int, lo: int, hi: int):
    for i0 in range(lo, hi - span + 2):
        if not blocks_of(range(i0, i0 + span)) & used_blocks:
            return i0
    return None


def rss_gb(pids) -> float:
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    t = 0
    for p in pids:
        try:
            t += int(Path("/proc/%d/statm" % p).read_text().split()[1]) * page
        except (OSError, ValueError, IndexError):
            pass
    return t / 2 ** 30


@dataclass
class CardAcct:
    """One card's accounting for a round."""
    index: int
    numa: int
    total_gb: float
    used_gb: float
    foreign_gb: float = 0.0          # used by processes outside the pool (>= what holds declare)
    pool_gb: float = 0.0             # sum over pool jobs of max(declared, measured)
    carla_foreign: int = 0
    carla_pool: int = 0
    train: int = 0
    jobs: int = 0                    # pool jobs on the card
    foreign_procs: int = 0
    exclusive: bool = False          # a pool job or a hold owns the whole card
    whole_hold: str = ""
    started_carla: bool = False
    reserved_for: str = ""
    util: int = 0

    def avail_gb(self, headroom: float) -> float:
        return self.total_gb - headroom - self.foreign_gb - self.pool_gb


# ---------------------------------------------------------------------------------------------------- placement
def fits(spec: Spec, a: CardAcct, cfg: dict) -> str:
    """'' when the job fits card `a` this round, else why not."""
    if spec.gpus and a.index not in spec.gpus:
        return "card not allowed"
    if a.whole_hold:
        return "held: " + a.whole_hold
    if a.exclusive:
        return "exclusive job on card"
    if spec.exclusive and (a.jobs or a.carla_foreign or a.foreign_gb > 2):
        return "exclusive: card not empty"
    need = spec.vram_gb if not spec.exclusive else a.total_gb - cfg["headroom_gb"]
    av = a.avail_gb(cfg["headroom_gb"])
    if av < need:
        return "VRAM %.1f GB free of the pool's view < %.1f" % (max(av, 0), need)
    if spec.carla and a.carla_pool + a.carla_foreign + spec.carla > cfg["carla_per_card"]:
        return "CARLA %d + %d > %d per card" % (a.carla_pool + a.carla_foreign, spec.carla, cfg["carla_per_card"])
    if spec.carla and a.started_carla:
        return "a CARLA job started on this card this round"
    if spec.train and a.train >= cfg["train_per_card"]:
        return "%d training jobs on card" % a.train
    return ""


def choose(spec: Spec, cards: list, cfg: dict, jid: str):
    """(card, {card: why}) for the best card that fits, or (None, reasons)."""
    why, ok = {}, []
    for a in cards:
        if a.reserved_for and a.reserved_for != jid:
            why[a.index] = "reserved for blocked job " + a.reserved_for
            continue
        w = fits(spec, a, cfg)
        if w:
            why[a.index] = w
        else:
            ok.append(a)
    if not ok:
        return None, why
    best = min(ok, key=lambda a: (a.jobs + a.foreign_procs, a.avail_gb(cfg["headroom_gb"]) - spec.vram_gb))
    return best, why


def pick_cpus(n: int, numa: int, box, taken: set):
    if n <= 0:
        return ""
    pool = [c for c in box.primary_cpus(numa) if c not in taken]
    pool += [c for c in box.primary_cpus() if c not in taken and c not in pool]
    pool += [c for c in box.affinity if c not in taken and c not in pool]
    return format_cpus(pool[:n]) if len(pool) >= n else None


# ---------------------------------------------------------------------------------------------------- dispatcher
class Dispatcher:
    def __init__(self, pool: Path = None, probe_fn=None, ports_fn=None, cfg: dict = None, smi_fn=None):
        self.pool = Path(pool or pool_dir())
        self.state_path = self.pool / "state.json"
        self.st = _read_json(self.state_path, None) or {"jobs": {}}
        self.children = {}
        self.probe_fn = probe_fn
        self.ports_fn = ports_fn or listening_ports
        self.smi_fn = smi_fn or smi
        self.cfg_over = cfg or {}
        self.halt = False
        self.log_f = None
        self.last = {}

    # -------------------------------------------------------------- helpers
    def cfg(self) -> dict:
        c = dict(DEFAULTS)
        c.update(_read_json(self.pool / "config.json", {}) or {})
        c.update(self.cfg_over)
        return c

    def log(self, kind: str, **kw) -> None:
        line = json.dumps(dict(t=time.strftime("%F %T"), kind=kind, **kw), default=str)
        with (self.pool / "events.jsonl").open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save(self) -> None:
        procs.atomic_json(self.state_path, self.st)

    def jdir(self, jid: str) -> Path:
        return self.pool / "jobs" / jid

    def logdir(self, j: dict) -> Path:
        return Path(j["spec"].get("log_dir") or self.jdir(j["id"]))

    def status_line(self, j: dict, text: str) -> None:
        d = self.logdir(j)
        d.mkdir(parents=True, exist_ok=True)
        (d / "STATUS").write_text("%s %s (%s): %s\n" % (time.strftime("%F %T"), j["spec"]["name"], j["id"], text))

    def jobs(self, *states) -> list:
        return [j for j in self.st["jobs"].values() if j["state"] in states]

    # -------------------------------------------------------------- inbox / cancel
    def ingest(self) -> None:
        for p in sorted((self.pool / "inbox").glob("*.json")):
            j = _read_json(p)
            if not j or j.get("id") in self.st["jobs"]:
                p.unlink(missing_ok=True)
                continue
            j.update(state="queued", tries=0, why="new")
            self.st["jobs"][j["id"]] = j
            self.jdir(j["id"]).mkdir(parents=True, exist_ok=True)
            procs.atomic_json(self.jdir(j["id"]) / "spec.json", j)
            self.status_line(j, "queued")
            self.log("submit", id=j["id"], name=j["spec"]["name"], owner=j["spec"].get("owner"))
            p.unlink(missing_ok=True)
        for p in sorted((self.pool / "retarget").glob("*")):
            j = self.st["jobs"].get(p.name)
            gpus = [int(g) for g in p.read_text().split(",") if g.strip()] if p.exists() else []
            p.unlink(missing_ok=True)
            if j is not None and j["state"] == "queued":
                j["spec"]["gpus"] = gpus
                self.log("retarget", id=j["id"], gpus=gpus)
        for p in sorted((self.pool / "cancel").glob("*")):
            j = self.st["jobs"].get(p.name)
            mode = p.read_text().strip() if p.exists() else "stop"
            p.unlink(missing_ok=True)
            if j is None or j["state"] in FINAL:
                continue
            if j["state"] == "queued":
                self.finish(j, "cancelled", None, "cancelled while queued")
            elif mode == "drain":
                (self.jdir(j["id"]) / "DRAIN").touch()
                self.log("drain", id=j["id"])
            else:
                j["stop"] = "cancelled"

    # -------------------------------------------------------------- end of jobs
    def finish(self, j: dict, state: str, rc, why: str) -> None:
        j.update(state=state, rc=rc, t1=time.time(), why=why)
        d = self.logdir(j)
        d.mkdir(parents=True, exist_ok=True)
        info = dict(id=j["id"], name=j["spec"]["name"], state=state, rc=rc, tries=j["tries"], gpu=j.get("gpu"),
                    wall_s=round(time.time() - j.get("t0", time.time()), 1), why=why)
        if state == "done":
            procs.atomic_json(d / "DONE", info)
        else:
            tail = ""
            try:
                tail = (d / "log.txt").read_bytes()[-3000:].decode(errors="replace")
            except OSError:
                pass
            (d / "ERROR").write_text("# job %s %s: %s (rc %s, try %d) %s\n\n%s\n" % (
                j["id"], state, why, rc, j["tries"], time.strftime("%F %T"), tail))
        self.status_line(j, "%s%s" % (state, ": " + why if why else ""))
        self.log("end", id=j["id"], state=state, rc=rc, why=why, gpu=j.get("gpu"), wall_s=info["wall_s"])

    def reap(self, rows: dict) -> dict:
        """Refresh running jobs; finish the ones whose tree is empty. Returns id -> live member pids."""
        live = {}
        for j in self.jobs("running"):
            jid, d = j["id"], self.jdir(j["id"])
            p = self.children.get(jid)
            if p is not None:
                p.poll()
            try:
                rec, rows = procs.refresh(d / "owned.json", rows)
            except (OSError, ValueError):
                rec = None
            members = procs.live_members(rec, rows) if rec else []
            root_alive = bool(rec) and procs.same(rec["root"], rows)
            spec = j["spec"]
            if root_alive and not j.get("stop"):
                if spec.get("timeout_h") and time.time() - j["t0"] > spec["timeout_h"] * 3600:
                    j["stop"] = "timeout after %.1f h" % spec["timeout_h"]
                elif spec.get("max_rss_gb") and rss_gb(members) > spec["max_rss_gb"]:
                    j["stop"] = "RSS %.1f GB > cap %.1f" % (rss_gb(members), spec["max_rss_gb"])
            if members and (j.get("stop") or not root_alive):
                self.log("stop", id=jid, pids=members, why=j.get("stop") or "root exited, members left")
                try:
                    procs.stop(d / "owned.json")
                except RuntimeError as e:             # a member would not die (D state): keep holding its resources
                    self.log("stop_failed", id=jid, err=str(e))
                    live[jid] = members
                    continue
                rows = processes()
                members = []
            if members:
                live[jid] = members
                continue
            self.children.pop(jid, None)
            try:
                rc = int((d / ("rc.%d" % j["tries"])).read_text().strip())
            except (OSError, ValueError):
                rc = None
            if j.get("stop"):
                self.finish(j, "cancelled" if j["stop"] == "cancelled" else "failed", rc, j.pop("stop"))
            elif rc == 0:
                self.finish(j, "done", rc, "")
            elif j["tries"] < int(spec.get("tries") or 1):
                j.update(state="queued", why="retry after rc %s" % rc)
                self.log("retry", id=jid, rc=rc, tries=j["tries"])
            else:
                self.finish(j, "failed", rc, "rc %s" % rc)
        return live

    # -------------------------------------------------------------- accounting
    def account(self, box, rows: dict, live: dict, holds: list, cfg: dict):
        allowed = cfg.get("cards")
        cards = {c.index: CardAcct(c.index, c.numa, c.mem_total_mib / 1024, c.mem_used_mib / 1024, util=c.util)
                 for c in box.cards if allowed is None or c.index in allowed}
        uuid = {c.uuid: c.index for c in box.cards}
        per_pid = {}                                           # (card, pid) -> GB
        for ln in self.smi_fn("compute-apps=gpu_uuid,pid,used_memory", cfg["smi_age_s"]):
            f = [x.strip() for x in ln.split(",")]
            if len(f) == 3 and f[0] in uuid and f[1].isdigit():
                per_pid[(uuid[f[0]], int(f[1]))] = float(f[2]) / 1024 if f[2].replace(".", "").isdigit() else 0.0
        pool_pids, pool_carla = set(), {}
        for j in self.jobs("running"):
            g, mem = j.get("gpu"), set(live.get(j["id"], []))
            pool_pids |= mem
            servers = [p for p in mem if rows.get(p, {}).get("argv") and CARLA_BIN in rows[p]["argv"][0]]
            for p in servers:
                pool_carla.setdefault(p, g)
            if g not in cards:
                continue
            a, s = cards[g], j["spec"]
            act = sum(v for (cg, p), v in per_pid.items() if cg == g and p in mem)
            j["vram_now"] = round(act, 1)
            a.pool_gb += max(float(s["vram_gb"]), act)
            a.foreign_gb -= act                                # foreign = used - pool actual (used added below)
            a.exclusive = a.exclusive or bool(s.get("exclusive"))
            a.carla_pool += max(s.get("carla", 0), len(servers))
            a.train += bool(s.get("train"))
            a.jobs += 1
        for g, a in cards.items():
            a.foreign_gb += a.used_gb
            a.foreign_procs = len({p for (cg, p) in per_pid if cg == g and p not in pool_pids})
        for c in box.cards:
            if c.index in cards:
                ours = sum(1 for p, g in pool_carla.items() if g == c.index)
                cards[c.index].carla_foreign = max(0, c.carla - ours)
        hold_gb, hold_carla = {}, {}
        for h in holds:
            g = h["card"]
            if g not in cards:
                continue
            if h.get("whole"):
                cards[g].whole_hold = h.get("note") or h["id"]
            hold_gb[g] = hold_gb.get(g, 0) + float(h.get("vram_gb") or 0)
            hold_carla[g] = hold_carla.get(g, 0) + int(h.get("carla") or 0)
            cards[g].train += bool(h.get("train"))
        for g, a in cards.items():
            a.foreign_gb = max(a.foreign_gb, hold_gb.get(g, 0.0), 0.0)
            a.carla_foreign = max(a.carla_foreign, hold_carla.get(g, 0))
        return cards

    # -------------------------------------------------------------- admission
    def admit(self, box, rows: dict, live: dict, holds: list, cfg: dict) -> None:
        cards = self.account(box, rows, live, holds, cfg)
        lo, hi = capacity.index_bounds(box.ephemeral[0])
        used_blocks = port_blocks(self.ports_fn())
        for h in holds:
            used_blocks |= blocks_of(idx_range(h.get("idx", "")))
        taken_cpu = set()
        for h in holds:
            taken_cpu |= set(parse_cpus(h.get("cpus") or ""))
        cpu_used = sum(int(h.get("cpu") or 0) for h in holds)
        now = time.time()
        young_threads = young_ram = 0
        for j in self.jobs("running"):
            if j.get("span"):
                used_blocks |= blocks_of(range(j["idx"], j["idx"] + j["span"]))
            taken_cpu |= set(parse_cpus(j.get("cpus") or ""))
            cpu_used += max(int(j["spec"].get("cpu") or 0), 1)
            if now - j.get("t0", 0) < YOUNG_S:
                young_threads += Spec(**j["spec"]).est_threads(box.host_cpus)
                young_ram += float(j["spec"].get("ram_gb") or 0)
        budget = box.cores * cfg["cpu_overcommit"] if not cfg.get("cpu_budget") else cfg["cpu_budget"]
        adm = capacity.Admission.from_box(box)
        state = {j["id"]: j["state"] for j in self.st["jobs"].values()}
        queue = sorted(self.jobs("queued"), key=lambda j: (-float(j["spec"].get("priority") or 0), j["t_submit"]))
        starts, head_reserved = 0, False
        for j in queue:
            s = Spec(**j["spec"])
            bad = [a for a in s.after if state.get(a) in ("failed", "cancelled")]
            if bad:
                self.finish(j, "failed", None, "dependency %s %s" % (bad[0], state[bad[0]]))
                continue
            wait = [a for a in s.after if state.get(a) != "done"]
            why = ""
            if wait:
                why = "after %s (%s)" % (wait[0], state.get(wait[0], "unknown id"))
            elif s.when_exists and not Path(s.when_exists).exists():
                why = "waiting for %s" % s.when_exists
            elif starts >= cfg["max_starts"]:
                why = "start limit this round"
            if why:
                self.set_why(j, why)
                continue
            n_cpu = max(s.cpu, 1)
            pid_need = s.est_threads(box.host_cpus)
            if cpu_used + n_cpu > budget:
                why = "CPU %d + %d cores > budget %.0f" % (cpu_used, n_cpu, budget)
            elif box.pids_current + young_threads + pid_need > adm.plan_cap:
                why = "PIDs %d + %d + %d > %d" % (box.pids_current, young_threads, pid_need, adm.plan_cap)
            elif box.mem_max_gb and box.mem_used_gb + young_ram + s.ram_gb > 0.85 * box.mem_max_gb:
                why = "memory %.0f + %.0f GB > 85%% of %.0f" % (box.mem_used_gb + young_ram, s.ram_gb, box.mem_max_gb)
            if why:
                self.set_why(j, why)
                continue
            a, reasons = choose(s, list(cards.values()), cfg, j["id"])
            span = s.span or 2 * s.carla
            i0 = None
            if a is not None and span:
                i0 = free_index_block(used_blocks, span, lo, hi)
                if i0 is None:
                    a, reasons = None, {"all": "no free block of %d server indices in %d-%d" % (span, lo, hi)}
            cpus = ""
            if a is not None and s.cpu:
                cpus = pick_cpus(s.cpu, a.numa, box, taken_cpu)
                if cpus is None:
                    a, reasons = None, {"all": "no %d free cores to pin" % s.cpu}
            if a is None:
                self.set_why(j, "; ".join("card %s: %s" % kv for kv in sorted(reasons.items(), key=str)) or "no card")
                j.setdefault("t_blocked", now)
                if not head_reserved and now - j["t_blocked"] >= cfg["hold_s"]:
                    head_reserved = True               # the oldest blocked head job reserves the card closest to fit
                    cand = [c for c in cards.values() if not c.whole_hold and (not s.gpus or c.index in s.gpus)]
                    if cand:
                        best = max(cand, key=lambda c: c.avail_gb(cfg["headroom_gb"]))
                        best.reserved_for = j["id"]
                        self.set_why(j, j["why"] + " [reserves card %d]" % best.index)
                continue
            self.launch(j, s, a.index, i0, span if i0 is not None else 0, cpus, box, adm)
            starts += 1
            a.jobs += 1
            a.pool_gb += s.vram_gb
            a.carla_pool += s.carla
            a.train += s.train
            a.exclusive = a.exclusive or s.exclusive
            a.started_carla = a.started_carla or bool(s.carla)
            if a.reserved_for == j["id"]:
                a.reserved_for = ""
            if i0 is not None:
                used_blocks |= blocks_of(range(i0, i0 + span))
            taken_cpu |= set(parse_cpus(cpus))
            cpu_used += n_cpu
            young_threads += pid_need
            young_ram += s.ram_gb
        self.last = cards

    def set_why(self, j: dict, why: str) -> None:
        if j.get("why") != why:
            j["why"] = why
            self.status_line(j, "queued: " + why)

    def launch(self, j: dict, s: Spec, g: int, idx, span: int, cpus: str, box, adm) -> None:
        jid, d, ld = j["id"], self.jdir(j["id"]), self.logdir(j)
        d.mkdir(parents=True, exist_ok=True)
        ld.mkdir(parents=True, exist_ok=True)
        if j["tries"] == 0:
            for f in ("DONE", "ERROR"):
                if (ld / f).exists():
                    (ld / f).rename(ld / ("%s.%s" % (f, time.strftime("%Y%m%d-%H%M%S"))))
        j["tries"] += 1
        k = j["tries"]
        prof = profiles.get(s.profile or None)
        subs = dict(gpu=g, idx=idx if idx is not None else "", span=span, carla=s.carla, workers=s.carla, cpus=cpus,
                    server_args=" ".join(prof.server_args()), client_threads=prof.client_threads,
                    pids_wait=adm.wait_cap, job_dir=ld, id=jid)

        def fill(x):
            x = str(x)
            for key, v in subs.items():
                x = x.replace("{%s}" % key, str(v))
            return x
        argv = ["bash", "-c", fill(s.cmd)] if isinstance(s.cmd, str) else [fill(a) for a in s.cmd]
        full = (["taskset", "-c", cpus] if cpus else []) + argv
        env = dict(os.environ)
        if s.profile or s.carla:
            env.update(prof.environ())
        env.update({kk: fill(v) for kk, v in s.env.items()})
        env.update(CUDA_VISIBLE_DEVICES=str(g), CL_GPU=str(g), CL_IDX=str(subs["idx"]), CL_SPAN=str(span),
                   CL_CPUS=cpus, CL_POOL_JOB=jid, CL_JOB_DIR=str(ld), CL_OWNER=s.owner or jid,
                   B2D_DRAIN_FILE=str(d / "DRAIN"), B2D_PIDS_WAIT=str(adm.wait_cap), PYTHONUNBUFFERED="1",
                   CL_RC=str(d / ("rc.%d" % k)), CL_TOKEN="pool/%s/%d/%d" % (jid, k, time.time_ns()))
        (d / ("rc.%d" % k)).unlink(missing_ok=True)
        (d / "DRAIN").unlink(missing_ok=True)
        with (ld / "log.txt").open("ab") as f:
            f.write(("\n==> %s try %d on card %d idx %s span %d cpus %s\n$ %s\n" % (
                time.strftime("%F %T"), k, g, subs["idx"], span, cpus or "-", shlex.join(full))).encode())
            f.flush()
            p = subprocess.Popen(["sh", "-c", '"$@"; echo $? > "$CL_RC"', "sh"] + full, cwd=s.cwd or str(REPO), env=env,
                                 stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        procs.capture(d / "owned.json", p.pid, token="CL_TOKEN=" + env["CL_TOKEN"])
        self.children[jid] = p
        j.update(state="running", gpu=g, idx=idx, span=span, cpus=cpus, t0=time.time(), pid=p.pid, why="",
                 argv=full)
        j.pop("t_blocked", None)
        self.status_line(j, "running on card %d (try %d, pid %d%s)" % (g, k, p.pid, ", idx %s" % idx if span else ""))
        self.log("launch", id=jid, name=s.name, gpu=g, idx=idx, span=span, cpus=cpus, pid=p.pid, try_=k,
                 vram_gb=s.vram_gb, carla=s.carla)

    def prune_holds(self, rows: dict) -> None:
        """Drop holds whose process has exited (logged once)."""
        with _flock(self.pool / "holds.lock"):
            hs = load_holds(self.pool)
            keep = live_holds(hs, rows)
            procs.atomic_json(self.pool / "holds.json", keep)
        for h in hs:
            if h not in keep:
                self.log("hold_ended", hold=h["id"], card=h["card"], note=h.get("note"))

    # -------------------------------------------------------------- loop
    def round(self) -> None:
        cfg = self.cfg()
        self.ingest()
        rows = processes()
        live = self.reap(rows)
        rows = processes()
        holds = live_holds(load_holds(self.pool), rows)
        if len(holds) < len(load_holds(self.pool)):
            self.prune_holds(rows)
        box = self.probe_fn(rows) if self.probe_fn else probe(rows=rows, query=lambda q: smi(q, cfg["smi_age_s"]))
        if not self.halt:
            self.admit(box, rows, live, holds, cfg)
        self.save()
        procs.atomic_json(self.pool / "status.json", dict(
            t=time.time(), pid=os.getpid(), cfg=cfg, holds=holds,
            cards={g: a.__dict__ for g, a in self.last.items()},
            counts={s: len(self.jobs(s)) for s in ("queued", "running", "done", "failed", "cancelled")},
            box=dict(cores=box.cores, pids=box.pids_current, pids_max=box.pids_max, mem_gb=round(box.mem_used_gb, 1),
                     mem_max_gb=round(box.mem_max_gb, 1), load=box.load)))

    def run(self, once: bool = False) -> int:
        self.pool.mkdir(parents=True, exist_ok=True)
        for sub in ("inbox", "cancel", "retarget", "jobs"):
            (self.pool / sub).mkdir(exist_ok=True)
        with (self.pool / "dispatch.lock").open("a") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise SystemExit("another dispatcher owns %s" % self.pool)
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
            self.log("dispatcher_start", pid=os.getpid(), running=[j["id"] for j in self.jobs("running")])
            fails = 0
            while not getattr(self, "stopping", False):
                try:
                    self.round()
                    fails = 0
                except Exception:                           # noqa: BLE001 - a bad round is logged and retried
                    fails += 1
                    self.log("round_error", n=fails, tb=traceback.format_exc()[-2000:])
                    if fails >= 20:
                        return 1
                if once:
                    break
                time.sleep(self.cfg()["poll_s"])
            self.log("dispatcher_stop", running=[j["id"] for j in self.jobs("running")])
        return 0


# ---------------------------------------------------------------------------------------------------- readers
def snapshot(pool: Path = None) -> tuple:
    """(state, status, inbox jobs, heartbeat age s) for the CLI readers."""
    pool = Path(pool or pool_dir())
    st = _read_json(pool / "state.json", {"jobs": {}}) or {"jobs": {}}
    status = _read_json(pool / "status.json", {}) or {}
    inbox = [x for x in (_read_json(p) for p in sorted((pool / "inbox").glob("*.json"))) if x]
    age = time.time() - status["t"] if status.get("t") else float("inf")
    return st, status, inbox, age


def job_states(ids, pool: Path = None) -> dict:
    st, _, inbox, _ = snapshot(pool)
    out = {i: "inbox" for i in (x["id"] for x in inbox)}
    out.update({i: j["state"] for i, j in st["jobs"].items()})
    return {i: out.get(i, "unknown") for i in ids}


def wait(ids, poll_s: float = 30.0, pool: Path = None, on_poll=None) -> dict:
    """Block until every job in `ids` is done / failed / cancelled; returns id -> state. on_poll(states) each poll."""
    while True:
        s = job_states(ids, pool)
        if on_poll:
            on_poll(s)
        if all(v in FINAL or v == "unknown" for v in s.values()):
            return s
        time.sleep(poll_s)


def b2d_cmd(out, route_ids, routes: str = None, agent: str = "", agent_config: str = "", python: str = None,
            tm_seed: int = 0, max_attempts: int = 3, stall_s: float = 480, route_timeout_s: float = 3600,
            extra=()) -> list:
    """argv of one scripts/b2d_run.py invocation on the pool's placeholders (submit it with carla=<workers>)."""
    d = data_dir()
    cmd = [str(d / "envs/carla/bin/python"), str(REPO / "scripts/b2d_run.py"),
           "--routes", str(routes or d / "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"),
           "--route-ids", ",".join(str(r) for r in route_ids), "--out", str(out), "--workers", "{carla}",
           "--server-index", "{idx}", "--index-span", "{span}", "--gpu-rank", "{gpu}",
           "--client-threads", "{client_threads}", "--server-args", "{server_args}", "--tm-seed", str(tm_seed),
           "--max-attempts", str(max_attempts), "--stall-s", str(stall_s), "--route-timeout-s", str(route_timeout_s),
           "--no-spectator"]
    if agent:
        cmd += ["--agent", str(agent), "--agent-config", str(agent_config)]
    if python:
        cmd += ["--python", str(python)]
    return cmd + [str(x) for x in extra]
