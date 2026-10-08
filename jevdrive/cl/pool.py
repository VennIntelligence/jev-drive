"""The GPU pool: agents submit jobs, one dispatcher puts each on whichever card has room ("any free card takes the next
job"). Nobody picks cards, cores or CARLA ports by hand.

Spool, under $DATA_DIR/runs/pool/ (the dispatcher is its only writer, except inbox/, cancel/, retarget/ and holds.json):

  inbox/<id>.json   a submitted job (written by `submit`, atomic rename); the dispatcher moves it into its state
  cancel/<id>       a cancel request (`cancel`; content "drain" = touch the job's DRAIN file instead of stopping it)
  retarget/<id>     new allowed cards for a queued job (`retarget`; content "0,2", empty = any); the id stays, so
                    `after` chains keep working
  history.json      per finished job-name prefix: recent peak VRAM and peak cores (dispatcher only; `submit` reads it)
  holds.json        resources held outside the pool (`hold`): a card or part of it, server indices, cores; a hold
                    with a pid ends by itself when that process exits
  state.json        every job: spec, state, card, server-index block, cores, tries, why it waits (dispatcher only)
  status.json       the last round: per-card accounting, heartbeat (read by `queue` / `top`)
  usage.jsonl       one line a minute: per-card util / booked VRAM / GPU jobs, cores measured / charged, the queue
                    split by wait cause (dispatcher only; `usage` reads it)
  jobs/<id>/        owned.json (process tree by pid + start time), rc.<k>; log_dir defaults here
  dispatch.lock     one dispatcher at a time (flock)

A job: cmd (argv, or one shell string), cwd, log_dir, env, need = {vram_gb, carla servers, cpu cores, ram_gb, train},
priority (higher first, then submit order), owner, after (job ids that must be done first), when_exists (a file gate),
gpus (allowed cards), pin_strict (never widen gpus), exclusive, tries, timeout_h, max_rss_gb, profile (thread env from
profiles.py).

Every round (default 20 s) the dispatcher reaps finished jobs, then admits queued ones in priority order. A job fits a
card when (all measured live, nvidia-smi cached <= 15 s):
  VRAM      total - headroom - demand >= need, demand = (VRAM used by processes outside the pool, at least what
            holds declare) + sum over the card's pool jobs of max(measured now, booked). A declaration is an upper
            bound: `booked` (and a queued job's `need`) is min(declared, 1.2 x max peak of the name prefix's finished
            runs + 1 GB) once history has >= 3 of them; without history a job books its declaration for 10 min, then
            min(declared, 1.2 x own peak + 1 GB). CARLA and exclusive jobs always book the declaration, and
            config trust_measured = false restores that for every job.
  CARLA     pool servers (max(declared, live)) + servers outside the pool + carla <= carla_per_card (default 6); at most
            one CARLA job starts per card per round (staggered server starts).
  training  train jobs on the card < train_per_card (default 2).
  ports     a free block of 2 x carla server indices: every RPC (2000 + 50 i) and traffic-manager (8000 + 50 i = the RPC
            block of i + 120) port block clear of pool jobs, holds and every LISTENING port on the box (/proc/net/tcp),
            inside capacity.index_bounds. A block is freed only when every process of the job's tree has exited.
  CPU       sum of charged cores (pool + holds) <= cpu budget (cgroup quota x cpu_overcommit); cpu > 0 pins the job
            (taskset) to free physical cores, NUMA-local to the card first. A job is charged its declared cores while
            younger than 5 min (and when no measurement exists yet), then max(1, 1.2 x its peak measured cores over the
            last 5 min): utime + stime deltas over its process tree, sampled every round. Holds keep declared cores.
            With >= 3 finished runs in history a young or queued job is charged min(declared, 1.2 x their max).
  PIDs/RAM  pids.current + threads of jobs younger than 5 min + the job's estimate <= 0.80 pids.max; cgroup memory
            without page cache (anon + shmem + kernel) + what young jobs have not allocated yet (ram_gb - their
            tree's RSS, tapering to 0 at 5 min) + ram_gb <= 0.85 memory.max (ram_gb at most 1.2 x history's max
            RSS + 1).
Among the cards that fit, the least loaded (fewest GPU jobs of the pool + foreign GPU processes; CPU-only jobs with
vram_gb <= 1 do not count and do not make a card busy) wins and VRAM best-fit breaks ties:
spreading keeps every card computing, best-fit keeps room for a large job. A job blocked > hold_s (15 min) at the head
of the queue by a card-level reason reserves the card closest to fitting it: lower-priority jobs stop starting there
until it starts. Every queued job accumulates waited = {cause: seconds} (after, gate, vram, ram, cpu, train, ...; in
state.json and in its `launch` event), so "why did it wait" has an answer afterwards.

Work conservation (an idle card never waits on bookkeeping). A card is idle when it has no pool job, no whole hold and
at most idle_vram_gb (1 GB) used outside the pool, for >= idle_s (120 s, config). Then:
  admit_idle     a queued job with vram_gb > 1 blocked only by the CPU budget or the PID plan cap starts on that card
                 anyway (highest priority first; logged `admit_idle`). VRAM, RAM, CARLA, ports and pinning are never
                 relaxed, and CPU-only jobs (vram_gb <= 1) never use this rule.
  auto_retarget  a queued job restricted by `gpus` whose allowed cards cannot take it, while a card outside `gpus` is idle
                 and would fit it, gets that card added to its gpus and starts there (logged `auto_retarget`).
                 `submit --pin-strict` opts out.
History: when a job ends done (>= 60 s), its peak VRAM (measured) and peak cores are appended to history.json under
its history key (env CL_HIST_KEY if the submitter set one, else its name; then hist_prefix: lower case, trailing -s3 / -k2 / _t0 / -007 / -pf / trailing digits stripped repeatedly,
so `op-eval-s3-pf` and `op-eval-k2` are `op-eval`), last 20 kept. `submit` without --vram / --cpu defaults to p95 x 1.2
of that history and says so on stderr; explicit values win.

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
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
import traceback
from collections import deque
import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import capacity, procs, profiles
from .box import CARLA_BIN, format_cpus, parse_cpus, probe, processes, smi

REPO = Path(__file__).resolve().parents[2]
YOUNG_S = 300.0
CPU_WINDOW_S = 300.0                  # peak of measured cores over this window
CPU_MARGIN = 1.2                      # charge = margin x recent peak cores
HIST_KEEP = 20
HIST_MIN_WALL_S = 60.0
HIST_MIN_N = 3                        # finished runs of a name prefix before its measured peaks replace a declaration
VRAM_SETTLE_S = 600.0                 # without history: a job books its declared VRAM this long, then its own peak
VRAM_MARGIN, VRAM_PAD_GB = 1.2, 1.0   # booked = margin x measured peak + pad, never above the declaration
USAGE_EVERY_S = 60.0                  # one usage.jsonl line per this many seconds
SERIAL_HINT_S = 1800.0                # `top` flags a GPU job older than this while another card has been idle
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
STALE_S = 120.0                       # status.json older than this: the dispatcher is not running
DEFAULTS = dict(poll_s=20.0, headroom_gb=4.0, carla_per_card=capacity.GPU_KNEE, train_per_card=2, cpu_overcommit=1.0,
                max_starts=4, hold_s=900.0, cards=None, smi_age_s=15.0, idle_s=120.0, idle_vram_gb=1.0,
                trust_measured=True)
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


# ---------------------------------------------------------------------------------------------------- measured CPU, history
def proc_ticks(pid: int):
    """utime + stime of a process in clock ticks, None when it is gone."""
    try:
        f = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except (OSError, IndexError, ValueError):
        return None


def cpu_charge(j: dict, now: float, known: float = None) -> float:
    """Cores a running job is charged against the CPU budget: declared while young or unmeasured (at most `known`, the
    name prefix's measured cap, when history has one), then 1.2 x its peak."""
    declared = float(max(int(j["spec"].get("cpu") or 0), 1))
    peak = j.get("cores_peak")
    if peak is None or now - j.get("t0", 0) < YOUNG_S:
        return declared if known is None else max(1.0, min(declared, max(known, CPU_MARGIN * float(peak or 0))))
    return max(1.0, CPU_MARGIN * float(peak))


def is_gpu(spec: dict) -> bool:
    """A job that computes on its card (CPU-only stages declare a token vram_gb <= 1)."""
    return bool(float(spec.get("vram_gb") or 0) > 1 or spec.get("carla") or spec.get("exclusive"))


def known_caps(hist: dict, job) -> dict:
    """{vram_gb, cpu, ram_gb}: what finished runs under this job's history key (hist_key; `job` is a spec dict or a
    name) needed at most, with margin; only the keys with at least HIST_MIN_N samples. The dispatcher books a job at min(declared, this): a declaration is an upper bound,
    the record of what the same job used is the estimate."""
    e = (hist or {}).get(hist_key(job)) or {}
    out = {}
    for key, src, scale, pad in (("vram_gb", "vram", VRAM_MARGIN, VRAM_PAD_GB), ("cpu", "cores", CPU_MARGIN, 0.0),
                                 ("ram_gb", "ram", 1.2, 1.0)):
        xs = e.get(src) or []
        if len(xs) >= HIST_MIN_N:
            out[key] = round(max(xs) * scale + pad, 1)
    return out


def vram_need(spec: dict, known: dict, trust: bool = True) -> float:
    """GB a queued job needs free on a card: its declaration, or less when history knows the name prefix."""
    declared = float(spec.get("vram_gb") or 0)
    if not trust or spec.get("carla") or spec.get("exclusive") or "vram_gb" not in (known or {}):
        return declared
    return min(declared, known["vram_gb"])


def vram_charge(j: dict, now: float, known: dict = None, trust: bool = True) -> float:
    """GB a running job books on its card besides what it uses right now (the caller takes the max with the live
    measurement). CARLA and exclusive jobs, and trust_measured = false: the declaration, for life. Otherwise the
    declaration until the job's own peak can be believed (history of the prefix, or VRAM_SETTLE_S of running), then
    min(declared, max(history cap, 1.2 x own peak + 1 GB))."""
    s = j["spec"]
    declared = float(s.get("vram_gb") or 0)
    if not trust or s.get("carla") or s.get("exclusive"):
        return declared
    peak = float(j.get("vram_peak") or 0)
    own = VRAM_MARGIN * peak + VRAM_PAD_GB
    if "vram_gb" in (known or {}):
        return min(declared, max(known["vram_gb"], own))
    if peak <= 0 or now - j.get("t0", 0) < VRAM_SETTLE_S:
        return declared
    return min(declared, own)


def ram_reserve(j: dict, now: float, known: float = None) -> float:
    """GB of host RAM still to come from a young job: what it is expected to need (declared, at most `known` from
    history) minus what its tree already holds (that part is in the cgroup's measured memory; adding the whole
    declaration on top counted young jobs twice), tapering linearly to 0 at YOUNG_S (the old rule held the full
    declaration until YOUNG_S and nothing after)."""
    age = now - j.get("t0", 0)
    if age >= YOUNG_S:
        return 0.0
    need = float(j["spec"].get("ram_gb") or 0)
    need = need if known is None else min(need, known)
    return max(0.0, need - float(j.get("rss_now") or 0)) * (1.0 - max(age, 0.0) / YOUNG_S)


CAUSES = (("after ", "after"), ("waiting for ", "gate"), ("start limit", "start_limit"), ("CPU ", "cpu"), ("PIDs ", "pids"),
          ("memory ", "ram"), ("no free block", "ports"), ("free cores to pin", "cpu_pin"), ("VRAM ", "vram"),
          ("CARLA ", "carla"), ("training jobs", "train"), ("exclusive", "exclusive"), ("held: ", "held"),
          ("reserved for", "reserved"), ("card not allowed", "pinned"), ("retry", "retry"), ("new", "new"))
NOT_READY = ("after", "gate", "new", "retry")


def wait_cause(why: str) -> str:
    """One word for why a queued job waits. Cards may differ: the first card it is allowed on decides."""
    cs = [next((c for k, c in CAUSES if k in part), "other") for part in (why or "").split("; card")]
    return next((c for c in cs if c != "pinned"), cs[0])


_SUFFIX = re.compile(r"(?:[-_.](?:[a-z]?\d+|pf)|\d+)+$")


def hist_prefix(name: str) -> str:
    """Key of a job name in history.json: lower case, trailing -s3 / -k2 / _t0 / -007 / -pf / digits stripped."""
    n = name.lower()
    return _SUFFIX.sub("", n) or n


def hist_key(job) -> str:
    """History key of a job (a spec dict, or just a name): env CL_HIST_KEY when the submitter sets one, else the name;
    hist_prefix of that. CL_HIST_KEY is for jobs whose names differ per model or run while their resource use does
    not (jevdrive.bench sets it per stage kind), so a new model's first run already has history."""
    if isinstance(job, str):
        return hist_prefix(job)
    return hist_prefix((job.get("env") or {}).get("CL_HIST_KEY") or job["name"])


def hist_record(pool: Path, job, vram_gb: float, cores: float, ram_gb: float = 0.0) -> None:
    pool = Path(pool)
    h = _read_json(pool / "history.json", {}) or {}
    e = h.setdefault(hist_key(job), dict(vram=[], cores=[]))
    for k, v in (("vram", vram_gb), ("cores", cores), ("ram", ram_gb)):
        if v > 0:
            e[k] = (e.get(k, []) + [round(v, 2)])[-HIST_KEEP:]
    procs.atomic_json(pool / "history.json", h)


def _p95(xs: list) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)]


def history_defaults(name: str, pool: Path = None) -> dict:
    """{vram_gb, cpu} = p95 x 1.2 of the finished jobs under this name prefix (only the keys that have samples)."""
    e = (_read_json(Path(pool or pool_dir()) / "history.json", {}) or {}).get(hist_prefix(name)) or {}
    out = {}
    if e.get("vram"):
        out["vram_gb"] = round(math.ceil(_p95(e["vram"]) * 1.2 * 10) / 10, 1)
    if e.get("cores"):
        out["cpu"] = max(1, math.ceil(_p95(e["cores"]) * 1.2))
    return out


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
    pin_strict: bool = False                      # never widen gpus (auto_retarget opt-out)
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
    jobs: int = 0                    # pool jobs computing on the card (is_gpu); what "least loaded" and "idle" count
    cpu_jobs: int = 0                # CPU-only pool jobs parked on the card (vram_gb <= 1)
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
        self.ticks_fn = proc_ticks
        self.cpu_prev = {}                                    # job id -> ({pid: ticks}, t)
        self.cpu_win = {}                                     # job id -> deque of (t, cores)
        self.idle_since = {}                                  # card -> t since which it has been idle
        self.cpu_info = {}
        self.hist = {}                                        # history.json, re-read every round
        self.usage_t = 0.0

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
        if state == "done" and info["wall_s"] >= HIST_MIN_WALL_S and not j["spec"].get("env", {}).get("CL_PREFLIGHT"):
            try:
                hist_record(self.pool, j["spec"], float(j.get("vram_peak") or 0), float(j.get("cores_max") or 0),
                            float(j.get("rss_peak") or 0))
            except OSError:
                pass
        self.status_line(j, "%s%s" % (state, ": " + why if why else ""))
        self.log("end", id=j["id"], state=state, rc=rc, why=why, gpu=j.get("gpu"), wall_s=info["wall_s"],
                 vram_peak=j.get("vram_peak"), cores_max=j.get("cores_max"), rss_peak=j.get("rss_peak"))

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
        allowed, now = cfg.get("cards"), time.time()
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
            j["vram_peak"] = max(float(j.get("vram_peak") or 0), round(act, 1))
            j["vram_booked"] = round(max(act, vram_charge(j, now, known_caps(self.hist, s), cfg["trust_measured"])), 1)
            a.pool_gb += j["vram_booked"]
            a.foreign_gb -= act                                # foreign = used - pool actual (used added below)
            a.exclusive = a.exclusive or bool(s.get("exclusive"))
            a.carla_pool += max(s.get("carla", 0), len(servers))
            a.train += bool(s.get("train"))
            a.jobs += is_gpu(s)
            a.cpu_jobs += not is_gpu(s)
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

    # -------------------------------------------------------------- measured CPU, idle cards
    def measure_cpu(self, live: dict, now: float) -> None:
        """Cores each running job used since the last round (utime + stime over its live tree); keeps the peak over the
        last CPU_WINDOW_S in j["cores_peak"] (what admission charges) and the lifetime max in j["cores_max"]."""
        for jid in list(self.cpu_prev):
            if jid not in live:
                self.cpu_prev.pop(jid, None)
                self.cpu_win.pop(jid, None)
        for jid, mem in live.items():
            j = self.st["jobs"].get(jid)
            if j is None:
                continue
            cur = {p: t for p in mem if (t := self.ticks_fn(p)) is not None}
            prev, t_prev = self.cpu_prev.get(jid, (None, 0.0))
            self.cpu_prev[jid] = (cur, now)
            if prev is None or now - t_prev < 1.0:
                continue
            d = sum(max(t - prev.get(p, 0), 0) for p, t in cur.items())
            cores = d / CLK_TCK / (now - t_prev)
            w = self.cpu_win.get(jid)
            if w is None:
                w = self.cpu_win[jid] = deque()
                if j.get("cores_peak") is not None:           # after a restart: keep the old peak for one window
                    w.append((now - 1.0, float(j["cores_peak"])))
            w.append((now, cores))
            while w and now - w[0][0] > CPU_WINDOW_S:
                w.popleft()
            j["cores_now"] = round(cores, 1)
            j["cores_peak"] = round(max(c for _, c in w), 1)
            j["cores_max"] = max(float(j.get("cores_max") or 0), j["cores_peak"])

    def track_idle(self, cards: dict, now: float, cfg: dict) -> None:
        for g, a in cards.items():
            if a.jobs == 0 and not a.whole_hold and not a.exclusive and a.foreign_gb <= cfg["idle_vram_gb"]:
                self.idle_since.setdefault(g, now)
            else:
                self.idle_since.pop(g, None)

    def idle_cards(self, cards, now: float, cfg: dict) -> list:
        """Cards idle for >= idle_s that still have no job this round."""
        return [a for a in cards if a.jobs == 0 and not a.whole_hold and not a.exclusive and a.index in self.idle_since
                and now - self.idle_since[a.index] >= cfg["idle_s"]]

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
        cpu_used = float(sum(int(h.get("cpu") or 0) for h in holds))
        now, trust = time.time(), cfg["trust_measured"]
        self.track_idle(cards, now, cfg)
        young_threads = young_ram = 0
        for j in self.jobs("running"):
            if j.get("span"):
                used_blocks |= blocks_of(range(j["idx"], j["idx"] + j["span"]))
            taken_cpu |= set(parse_cpus(j.get("cpus") or ""))
            known = known_caps(self.hist, j["spec"]) if trust else {}
            cpu_used += cpu_charge(j, now, known.get("cpu"))
            if now - j.get("t0", 0) < YOUNG_S:
                young_threads += Spec(**j["spec"]).est_threads(box.host_cpus)
                young_ram += ram_reserve(j, now, known.get("ram_gb")) if trust else float(j["spec"].get("ram_gb") or 0)
        budget = box.cores * cfg["cpu_overcommit"] if not cfg.get("cpu_budget") else cfg["cpu_budget"]
        self.cpu_info = dict(charged=round(cpu_used, 1), budget=round(budget, 1))
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
            # What the job is booked at: its declaration, or less when finished runs of the name prefix say so.
            known = known_caps(self.hist, j["spec"]) if trust else {}
            n_cpu = max(1, math.ceil(min(max(s.cpu, 1), known.get("cpu", max(s.cpu, 1)))))
            ram_need = min(s.ram_gb, known.get("ram_gb", s.ram_gb))
            v_need = vram_need(j["spec"], known, trust)
            eff = dataclasses.replace(s, vram_gb=v_need)
            pid_need = s.est_threads(box.host_cpus)
            soft = hard = ""                           # soft: CPU / PID plan cap, relaxed on an idle card
            if cpu_used + n_cpu > budget:
                soft = "CPU %.0f + %d cores > budget %.0f" % (cpu_used, n_cpu, budget)
            elif box.pids_current + young_threads + pid_need > adm.plan_cap:
                soft = "PIDs %d + %d + %d > %d" % (box.pids_current, young_threads, pid_need, adm.plan_cap)
            if box.mem_max_gb and box.mem_used_gb + young_ram + ram_need > 0.85 * box.mem_max_gb:
                hard = "memory %.0f + %.0f GB > 85%% of %.0f" % (box.mem_used_gb + young_ram, ram_need, box.mem_max_gb)
            if hard or (soft and s.vram_gb <= 1):
                self.set_why(j, hard or soft)
                continue
            every = list(cards.values())
            idle = self.idle_cards(every, now, cfg) if soft else None
            a, reasons = choose(eff, idle if soft else every, cfg, j["id"])
            widened = None
            if a is None and s.gpus and not s.pin_strict:      # pinning watchdog: an idle card outside gpus would fit
                extra = [c for c in (idle if soft else self.idle_cards(every, now, cfg)) if c.index not in s.gpus]
                a, _ = choose(dataclasses.replace(eff, gpus=[]), extra, cfg, j["id"])
                widened = a
            if soft and a is None:
                self.set_why(j, soft)
                continue
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
                # Only a card-level block reserves a card: a reservation cannot produce ports or cores to pin.
                if not head_reserved and "all" not in reasons and now - j["t_blocked"] >= cfg["hold_s"]:
                    head_reserved = True               # the oldest blocked head job reserves the card closest to fit
                    cand = [c for c in cards.values() if not c.whole_hold and (not s.gpus or c.index in s.gpus)]
                    if cand:
                        best = max(cand, key=lambda c: c.avail_gb(cfg["headroom_gb"]))
                        best.reserved_for = j["id"]
                        self.set_why(j, j["why"] + " [reserves card %d]" % best.index)
                continue
            if widened is not None:
                s.gpus = j["spec"]["gpus"] = sorted(set(s.gpus) | {a.index})
                self.log("auto_retarget", id=j["id"], name=s.name, gpus=s.gpus, card=a.index)
            if soft:
                self.log("admit_idle", id=j["id"], name=s.name, card=a.index, blocked=soft,
                         idle_s=round(now - self.idle_since[a.index]))
            j["vram_booked"] = round(v_need, 1)
            self.launch(j, s, a.index, i0, span if i0 is not None else 0, cpus, box, adm)
            starts += 1
            a.jobs += is_gpu(j["spec"])
            a.cpu_jobs += not is_gpu(j["spec"])
            a.pool_gb += v_need
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
            young_ram += ram_need
        self.last = cards

    def note_wait(self, j: dict, cause: str = "") -> None:
        """Add the time since the last note to j["waited"][previous cause]; `cause` is what the job waits on now."""
        now = time.time()
        if j.get("cause"):
            w = j.setdefault("waited", {})
            w[j["cause"]] = round(w.get(j["cause"], 0.0) + now - j.get("t_cause", now), 1)
        j["cause"], j["t_cause"] = cause, now

    def set_why(self, j: dict, why: str) -> None:
        self.note_wait(j, wait_cause(why))
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
        self.note_wait(j)
        j.pop("t_cause", None)
        self.status_line(j, "running on card %d (try %d, pid %d%s)" % (g, k, p.pid, ", idx %s" % idx if span else ""))
        self.log("launch", id=jid, name=s.name, gpu=g, idx=idx, span=span, cpus=cpus, pid=p.pid, try_=k,
                 vram_gb=s.vram_gb, booked_gb=j.get("vram_booked"), carla=s.carla, waited=j.get("waited"))

    def usage(self, box) -> None:
        """One line per USAGE_EVERY_S in usage.jsonl: per card [util %, GB used, GB booked, GPU jobs, whole hold], cores
        [measured, charged, budget], and the queue split by wait cause (`usage_report` and `cl usage` read it)."""
        now = time.time()
        if now - self.usage_t < USAGE_EVERY_S:
            return
        self.usage_t = now
        q_gpu, q_cpu, blocked = {}, {}, 0
        for j in self.jobs("queued"):
            c = j.get("cause") or "new"
            if c in NOT_READY:
                blocked += 1
            else:
                d = q_gpu if is_gpu(j["spec"]) else q_cpu
                d[c] = d.get(c, 0) + 1
        run = self.jobs("running")
        line = dict(t=round(now), cards={g: [a.util, round(a.used_gb, 1), round(a.pool_gb, 1), a.jobs, int(bool(a.whole_hold))]
                                         for g, a in self.last.items()},
                    cpu=[round(sum(float(j.get("cores_now") or 0) for j in run), 1), self.cpu_info.get("charged"),
                         self.cpu_info.get("budget")], quota=box.cores, q_gpu=q_gpu, q_cpu=q_cpu, not_ready=blocked,
                    oldest={str(g): round(now - min((j["t0"] for j in run if j.get("gpu") == g and is_gpu(j["spec"])),
                                                    default=now)) for g in self.last})
        with (self.pool / "usage.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")

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
        self.measure_cpu(live, time.time())
        self.hist = _read_json(self.pool / "history.json", {}) or {}
        for jid, mem in live.items():
            j = self.st["jobs"].get(jid)
            if j is not None:
                j["rss_now"] = round(rss_gb(mem), 1)
                j["rss_peak"] = max(float(j.get("rss_peak") or 0), j["rss_now"])
        holds = live_holds(load_holds(self.pool), rows)
        if len(holds) < len(load_holds(self.pool)):
            self.prune_holds(rows)
        box = self.probe_fn(rows) if self.probe_fn else probe(rows=rows, query=lambda q: smi(q, cfg["smi_age_s"]))
        if not self.halt:
            self.admit(box, rows, live, holds, cfg)
            self.usage(box)
        self.save()
        procs.atomic_json(self.pool / "status.json", dict(
            t=time.time(), pid=os.getpid(), cfg=cfg, holds=holds,
            cards={g: a.__dict__ for g, a in self.last.items()}, cpu=self.cpu_info,
            idle={g: round(time.time() - t) for g, t in self.idle_since.items()},
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


def usage_report(hours: float = 24.0, pool: Path = None, now: float = None) -> dict:
    """What the box did over the last `hours`, from usage.jsonl: card-hours and core-hours available and used, and
    the idle card-hours split by cause. A card sample is
      computing          a GPU job of the pool on it and util >= 10 %
      job, GPU idle      a GPU job on it but util < 10 % (a CPU phase inside a GPU job, a loader-bound job)
      held               a whole-card hold
      queued: <cause>    no GPU job on it while a ready GPU job waits (cause = what most of them wait on)
      serial             no GPU job on it, nothing ready, another card runs a GPU job older than SERIAL_HINT_S
                         (one job doing several runs in a row; fan it out)
      no work queued     no GPU job on it and nothing ready"""
    now = now or time.time()
    rows = []
    try:
        for ln in (Path(pool or pool_dir()) / "usage.jsonl").read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("t", 0) >= now - hours * 3600:
                rows.append(r)
    except OSError:
        pass
    out = dict(hours=0.0, card_h=0.0, util_h=0.0, cards={}, core_h=0.0, core_used_h=0.0, core_charged_h=0.0, samples=len(rows))
    for r, nxt in zip(rows, rows[1:] + [None]):
        dt = min((nxt["t"] - r["t"]) if nxt else USAGE_EVERY_S, 5 * USAGE_EVERY_S) / 3600      # a gap = dispatcher down
        out["hours"] += dt
        out["core_h"] += dt * float(r.get("quota") or 0)
        out["core_used_h"] += dt * float(r["cpu"][0] or 0)
        out["core_charged_h"] += dt * float(r["cpu"][1] or 0)
        ready = r.get("q_gpu") or {}
        for g, (util, used, booked, jobs, held) in r["cards"].items():
            out["card_h"] += dt
            out["util_h"] += dt * util / 100
            others_old = any(x[3] and (r.get("oldest") or {}).get(h, 0) >= SERIAL_HINT_S for h, x in r["cards"].items() if h != g)
            if jobs:
                k = "computing" if util >= 10 else "job, GPU idle"
            elif held:
                k = "held"
            elif ready:
                k = "queued: " + max(ready, key=ready.get)
            else:
                k = "serial" if others_old else "no work queued"
            out["cards"][k] = out["cards"].get(k, 0.0) + dt
    return out


def serial_hint(cmd) -> str:
    """'' or why this command looks like several runs in a row inside one job (shell `&&` / `;` / `for` around more
    than one python or script call). Such a job holds one card while the others idle: submit one job per run."""
    if not isinstance(cmd, str):
        cmd = " ".join(str(x) for x in cmd) if len(cmd) >= 3 and cmd[:2] == ["bash", "-c"] else ""
    if not cmd:
        return ""
    parts = [p for p in re.split(r"&&|;|\n", cmd) if re.search(r"\bpython[\d.]*\b|\.py\b|\.sh\b", p)]
    if re.search(r"\bfor\b.+\bdo\b", cmd, re.S) and parts:
        return "a shell loop around a run"
    return "%d runs chained in one shell command" % len(parts) if len(parts) >= 2 else ""


def fanout(cmd, arms, name: str, collect=None, collect_kw: dict = None, pool: Path = None, **kw) -> dict:
    """One pool job per arm, plus an optional collect job after all of them: the way to run N independent arms
    (seeds, variants, shards). `{arm}` in cmd, env values, log_dir and in the collect command is replaced by the arm
    ({arms} in collect: all of them, space separated). Jobs are named <name>-<arm>; log_dir without {arm} gets
    /<arm> appended. The collect job is CPU-only (vram 0.5, cpu 2) unless collect_kw says otherwise. Returns {"arms": {arm: id}, "collect": id or None}."""
    arms = [str(a) for a in arms]
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("fanout needs distinct arms")

    def fill(x, arm):
        if isinstance(x, (list, tuple)):
            return [fill(y, arm) for y in x]
        return x.replace("{arm}", arm).replace("{arms}", " ".join(arms)) if isinstance(x, str) else x
    ids = {}
    for arm in arms:
        k = dict(kw, env={e: fill(v, arm) for e, v in (kw.get("env") or {}).items()})
        if kw.get("log_dir"):
            k["log_dir"] = fill(kw["log_dir"], arm) if "{arm}" in kw["log_dir"] else str(Path(kw["log_dir"]) / arm)
        ids[arm] = submit(fill(cmd, arm), name="%s-%s" % (name, arm), pool=pool, **k)
    cid = None
    if collect:
        ck = dict(vram_gb=0.5, cpu=2, owner=kw.get("owner"), cwd=kw.get("cwd", ""), env=kw.get("env") or {},
                  priority=kw.get("priority", 0.0))
        if kw.get("log_dir") and "{arm}" not in kw["log_dir"]:
            ck["log_dir"] = str(Path(kw["log_dir"]) / "collect")
        ck.update(collect_kw or {})
        if ck.get("owner") is None:
            ck.pop("owner")
        cid = submit(fill(collect, ""), name="%s-collect" % name, pool=pool, after=list(ids.values()), **ck)
    return dict(arms=ids, collect=cid)


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
