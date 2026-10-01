"""A lane: a list of jobs pulled dynamically onto the cards of a lease, with retries, hand-off files and safe cleanup.

A job is one command that needs `workers` CARLA workers on one card, usually one scripts/b2d_run.py invocation (b2d()
builds it); b2d_run itself does per-route isolation, attempts, the stall watchdog, the box-wide start-slot flock and the
drain file. The lane adds what every lane script used to re-implement:

  placement  each round, per card of the lease: running workers + the job's <= workers per card, free VRAM (measured,
             minus what jobs launched in the last 5 min will still take, minus 8 GB headroom), PID room (capacity.Admission
             with the thread model for the job's profile and the card's core slice), a free sub-block of the card's
             server indices; one launch per card per round, so servers start staggered. `exclusive` jobs run alone.
  lease      re-read from the schedule table every round: a card that leaves the row takes no new job (running ones
             finish), a revoked / done row drains the lane. CLI overrides (--gpus, --cpus, ...) pin a fixed lease.
  retries    a job that fails is re-queued up to `tries` times (b2d_run resumes its --out, so only unfinished routes
             re-run); then ERROR.<job> is written and the lane goes on (or stops, --fail-fast).
  files      <root>/STATUS (one sentence), status.json, DONE / ERROR, util.csv (sampler), lane/<ts>/{log.txt,
             events.jsonl, tb/}; per job jobs/<name>/{job.<k>.json, log.<k>.txt, rc.<k>, owned.json, DONE, DRAIN}.
  cleanup    every job runs in its own session; its process tree is recorded by (pid, start time) (procs) and refreshed
             each round, and when the job ends any member still alive is stopped through a pidfd. Nothing is ever
             found by pattern. A restarted lane adopts live jobs from state.json.
  drain      <root>/DRAIN (or a revoked lease): no new launches, every running job's B2D_DRAIN_FILE is touched, b2d_run
             finishes the routes it is on and exits; drained jobs stay queued for the next run.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import capacity, procs
from .box import format_cpus, parse_cpus, probe, processes
from .capacity import Admission, worker_threads
from .lease import Lease, get as get_lease, load as load_table
from .profiles import Profile, get as get_profile
from .sampler import Sampler

REPO = Path(__file__).resolve().parents[2]
YOUNG_S = 300.0                    # a job launched this recently may not show its VRAM / threads yet


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", str(Path.home() / "data")))


@dataclass
class Job:
    """`cmd` and `env` values may use {gpu} {idx} {span} {workers} {cpus} {out} {server_args} {client_threads}
    {pids_wait} {job_dir}."""
    name: str
    cmd: list
    workers: int = 1
    vram_gb: float = capacity.VRAM_PER_WORKER_GB      # per worker: CARLA server + the agent's model
    deps: tuple = ()
    tries: int = 2
    profile: Profile = None                           # None: the lane's profile
    env: dict = field(default_factory=dict)
    server_args: tuple = ()                           # appended to the profile's pool flags
    exclusive: bool = False                           # alone on its card (measurements)
    gpus: tuple = ()                                  # restrict to these cards
    ok: object = None                                 # callable(job) -> bool, checked after rc == 0
    retry_check: bool = False                         # re-run when `ok` fails (default: a failed check is final)
    ready: object = None                              # callable(job) -> bool: stays queued until True (file gates)
    cores: int = 0                                    # own sub-slice of the card's cores (0: the whole card slice)
    priority: float = 0.0
    out: str = ""                                     # {out}; default <root>/jobs/<name>/out
    agent_threads: int = capacity.AGENT_THREADS
    cuda: bool = True                                 # CUDA_VISIBLE_DEVICES = the card (CARLA itself ignores it)
    meta: dict = field(default_factory=dict)


def b2d(name: str, out, route_ids, routes: str = None, agent: str = "", agent_config: str = "", python: str = None,
        workers: int = 6, tm_seed: int = 0, max_attempts: int = 3, stall_s: float = 480, route_timeout_s: float = 3600,
        extra=(), min_done: float = 1.0, **kw) -> Job:
    """One scripts/b2d_run.py invocation as a job; it succeeds when >= min_done of `route_ids` have done/<id>.json."""
    d = data_dir()
    ids = [str(r) for r in route_ids]
    cmd = [str(d / "envs/carla/bin/python"), str(REPO / "scripts/b2d_run.py"),
           "--routes", str(routes or d / "third_party/Bench2Drive/leaderboard/data/bench2drive220.xml"),
           "--route-ids", ",".join(ids), "--out", str(out), "--workers", "{workers}", "--server-index", "{idx}",
           "--index-span", "{span}", "--gpu-rank", "{gpu}", "--client-threads", "{client_threads}",
           "--server-args", "{server_args}", "--tm-seed", str(tm_seed), "--max-attempts", str(max_attempts),
           "--stall-s", str(stall_s), "--route-timeout-s", str(route_timeout_s), "--no-spectator"]
    if agent:
        cmd += ["--agent", str(agent), "--agent-config", str(agent_config)]
    if python:
        cmd += ["--python", str(python)]
    job = Job(name, cmd + [str(x) for x in extra], workers=workers, out=str(out), **kw)
    job.ok = lambda j: b2d_finished(Path(out), ids, min_done)
    job.meta.setdefault("routes", len(ids))
    return job


def b2d_finished(out: Path, ids, min_done: float = 1.0) -> bool:
    done = {p.stem for p in (Path(out) / "done").glob("*.json")}
    return len(done & set(ids)) >= math.ceil(min_done * len(ids))


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


class Lane:
    def __init__(self, name: str, jobs: list, root, lease: Lease = None, profile: Profile = None,
                 workers_per_card: int = None, poll_s: float = 20.0, stagger_s: float = 10.0, fail_fast: bool = False,
                 sample_s: float = 15.0, table: Path = None, probe_fn=None, generate=None):
        names = [j.name for j in jobs]
        if len(set(names)) != len(names):
            raise ValueError("job names must be unique")
        missing = {d for j in jobs for d in j.deps} - set(names)
        if missing:
            raise ValueError("unknown deps: %s" % sorted(missing))
        self.name, self.root, self.jobs = name, Path(root), {j.name: j for j in jobs}
        self.fixed, self.table = lease, table
        self.profile = profile or get_profile()
        self.wpc, self.poll_s, self.stagger_s, self.fail_fast, self.sample_s = (workers_per_card, poll_s, stagger_s,
                                                                                 fail_fast, sample_s)
        self.probe = probe_fn or (lambda rows: probe(rows=rows))
        self.generate = generate                    # callable(lane) -> [Job]: new jobs from results (dynamic lanes)
        self.state_path = self.root / "state.json"
        self.st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"jobs": {}}
        for n in names:
            self.st["jobs"].setdefault(n, {"state": "queued", "tries": 0})
        self.children, self.halt, self.blocked, self.lease = {}, False, {}, None
        self.log = self.bar = None

    # ------------------------------------------------------------------ bookkeeping
    def jst(self, name: str) -> dict:
        return self.st["jobs"][name]

    def jdir(self, name: str) -> Path:
        return self.root / "jobs" / name

    def save(self) -> None:
        procs.atomic_json(self.state_path, self.st)

    def event(self, kind: str, **kw) -> None:
        if self.log is not None:
            self.log.event(kind, **kw)
            self.log.info(kind + " " + " ".join("%s=%s" % (k, v) for k, v in kw.items()))

    def running(self) -> list:
        return [n for n in self.jobs if self.jst(n)["state"] == "running"]

    def counts(self) -> dict:
        c = {}
        for n in self.jobs:
            s = self.jst(n)["state"]
            c[s] = c.get(s, 0) + 1
        return c

    def current_lease(self):
        if self.fixed is not None:
            return self.fixed
        return get_lease(self.name, load_table(self.table))

    def draining(self, lease) -> bool:
        return (self.root / "DRAIN").exists() or lease is None or lease.revoked or self.halt

    # ------------------------------------------------------------------ job end
    def reap(self, rows: dict, draining: bool) -> None:
        for n in self.running():
            st, jd = self.jst(n), self.jdir(n)
            p = self.children.get(n)
            if p is not None:
                p.poll()                                   # collect our own zombie
            rec, rows = procs.refresh(jd / "owned.json", rows)
            if procs.same(rec["root"], rows):
                continue
            left = procs.live_members(rec, rows)
            if left:                                       # runner gone, servers / routes left behind: ours, stop them
                self.event("reap", job=n, pids=left)
                procs.stop(jd / "owned.json")
            self.children.pop(n, None)
            k = st["tries"]
            try:
                rc = int((jd / ("rc.%d" % k)).read_text().strip())
            except (OSError, ValueError):
                rc = None
            job = self.jobs[n]
            ok = rc == 0 and (job.ok is None or bool(job.ok(job)))
            st.update(rc=rc, t1=time.time(), wall_s=round(time.time() - st.get("t0", time.time()), 1))
            drained = (jd / "DRAIN").exists()
            if ok:
                st["state"] = "done"
                procs.atomic_json(jd / "DONE", dict(rc=rc, tries=k, wall_s=st["wall_s"], gpu=st.get("gpu")))
            elif rc == 0 and not job.retry_check:          # the output check failed: a verdict, not a crash
                st["state"] = "failed"
                self._error(n, k, rc, "output check failed")
            elif drained:
                st["state"], st["tries"] = "queued", k - 1  # a drained job did not fail
            elif k < job.tries and not draining:
                st["state"] = "queued"
            else:
                st["state"] = "failed"
                self._error(n, k, rc, "")
            self.event("job_end", job=n, rc=rc, ok=ok, state=st["state"], tries=k, wall_s=st["wall_s"],
                       gpu=st.get("gpu"))
            if self.bar is not None and st["state"] in ("done", "failed"):
                self.bar.update(1)
        self.save()

    def _error(self, n: str, k: int, rc, why: str) -> None:
        tail = ""
        try:
            tail = (self.jdir(n) / ("log.%d.txt" % k)).read_bytes()[-3000:].decode(errors="replace")
        except OSError:
            pass
        (self.root / ("ERROR." + n)).write_text("# job %s failed after %d tries (rc %s%s) %s\n\n%s\n" % (
            n, k, rc, ", " + why if why else "", time.strftime("%F %T %Z"), tail))
        if self.fail_fast:
            self.halt = True

    def add_generated(self) -> None:
        for j in (self.generate(self) if self.generate else []):
            if j.name not in self.jobs:
                self.jobs[j.name] = j
                self.st["jobs"].setdefault(j.name, {"state": "queued", "tries": 0})
                self.event("job_added", job=j.name)

    def sub_slice(self, lease: Lease, g: int, j: Job):
        """The card's core list, or for a job with `cores` its own contiguous-first share not used by running jobs."""
        spec = lease.cards[g]["cpus"]
        if not j.cores:
            return spec
        taken = set()
        for n in self.running():
            s = self.jst(n)
            if s.get("gpu") == g and self.jobs[n].cores:
                taken |= set(parse_cpus(s["cpus"]))
        free = [c for c in parse_cpus(spec) if c not in taken]
        return format_cpus(free[:j.cores]) if len(free) >= j.cores else None

    # ------------------------------------------------------------------ placement
    def schedule(self, lease: Lease, rows: dict) -> None:
        box = self.probe(rows)
        adm = Admission.from_box(box)
        now = time.time()
        run = [(n, self.jst(n)) for n in self.running()]
        young = [(n, s) for n, s in run if now - s.get("t0", 0) < YOUNG_S]
        pending = sum(self.jobs[n].workers for n, _ in young)
        done = {n for n in self.jobs if self.jst(n)["state"] == "done"}
        ready = sorted((j for j in self.jobs.values() if self.jst(j.name)["state"] == "queued" and set(j.deps) <= done
                        and (j.ready is None or j.ready(j))),
                       key=lambda j: j.priority)
        self.blocked = {}
        cap = self.wpc or lease.workers or capacity.GPU_KNEE
        used = {g: sum(self.jobs[n].workers for n, s in run if s.get("gpu") == g) for g in lease.cards}
        launched = False
        for g in sorted(lease.cards, key=lambda g: used[g]):
            if not ready:
                break
            card = next((c for c in box.cards if c.index == g), None)
            if card is None:
                self.blocked[g] = "card %d not present on the box" % g
                continue
            mine_here = [(n, s) for n, s in run if s.get("gpu") == g]
            if any(self.jobs[n].exclusive for n, _ in mine_here):
                self.blocked[g] = "exclusive job running"
                continue
            ours = sum(self.jobs[n].workers for n, _ in mine_here)
            foreign = max(0, card.carla - ours)       # other lanes' servers on this card
            spec = lease.cards[g]["cpus"]
            ncores = len(parse_cpus(spec)) if spec else box.cores / max(len(box.cards), 1)
            reserve = sum(self.jobs[n].workers * self.jobs[n].vram_gb for n, s in young if s.get("gpu") == g)
            for j in ready:
                if j.gpus and g not in j.gpus:
                    continue
                why = None
                prof = j.profile or self.profile
                per = worker_threads(prof, ncores, box.host_cpus, j.agent_threads)
                if j.workers * j.vram_gb > card.mem_total_mib / 1024 - capacity.VRAM_HEADROOM_GB or j.workers > cap:
                    st = self.jst(j.name)                    # can never fit this card: say so instead of waiting forever
                    if not (set(j.gpus) - {g}) and len(lease.cards) and all(
                            j.workers * j.vram_gb > c.mem_total_mib / 1024 - capacity.VRAM_HEADROOM_GB or j.workers > cap
                            for c in box.cards if c.index in lease.cards and (not j.gpus or c.index in j.gpus)):
                        st["state"] = "failed"
                        self._error(j.name, st["tries"], None, "never fits: %d workers x %.1f GB, %d per card" % (
                            j.workers, j.vram_gb, cap))
                        ready.remove(j)
                        break
                    continue
                if j.exclusive and (ours or foreign):
                    why = "%s: wants the card alone" % j.name
                elif used[g] + foreign + j.workers > cap:
                    why = "%s: %d + %d workers > %d per card" % (j.name, used[g] + foreign, j.workers, cap)
                elif card.free_gb - capacity.VRAM_HEADROOM_GB - reserve < j.workers * j.vram_gb:
                    why = "%s: VRAM free %.0f GB - reserve %.0f < %.0f" % (j.name, card.free_gb, reserve, j.workers * j.vram_gb)
                elif adm.room(box.pids_current, per, pending) < j.workers:
                    why = "%s: pids %d, %d per worker, cap %d" % (j.name, box.pids_current, per, adm.plan_cap)
                block = None if why else self.sub_block(lease, g, j)
                if why is None and block is None:
                    why = "%s: no free index sub-block" % j.name
                cpus = None if why else self.sub_slice(lease, g, j)
                if why is None and cpus is None:
                    why = "%s: no %d free cores in the card slice" % (j.name, j.cores)
                if why:
                    self.blocked.setdefault(g, why)
                    continue
                if launched and self.stagger_s:
                    time.sleep(self.stagger_s)
                self.launch(j, g, lease, block, adm, cpus)
                launched = True
                used[g] += j.workers
                pending += j.workers
                ready.remove(j)
                self.blocked.pop(g, None)
                break                                       # one launch per card per round

    def sub_block(self, lease: Lease, g: int, j: Job):
        """(idx, span) for a job on card g: 2 x workers indices (b2d_run moves a failing worker to a fresh slot inside
        its span), the whole block for an exclusive job, never overlapping a running job of this lane."""
        base, span = lease.cards[g]["idx0"], lease.span
        if j.exclusive:
            return (base, span)
        want = min(span, 2 * j.workers)
        taken = set()
        for n in self.running():
            s = self.jst(n)
            if s.get("gpu") == g:
                taken |= set(range(s["idx"], s["idx"] + s["span"]))
        for i in range(base, base + span - want + 1):
            if not taken & set(range(i, i + want)):
                return (i, want)
        return None

    def launch(self, j: Job, g: int, lease: Lease, block: tuple, adm: Admission, cpus: str = None) -> None:
        st, jd = self.jst(j.name), self.jdir(j.name)
        jd.mkdir(parents=True, exist_ok=True)
        st["tries"] += 1
        k, (idx, span) = st["tries"], block
        prof = j.profile or self.profile
        cpus = lease.cards[g]["cpus"] if cpus is None else cpus
        out = j.out or str(jd / "out")
        subs = dict(gpu=g, idx=idx, span=span, workers=j.workers, cpus=cpus, out=out, job_dir=jd,
                    server_args=" ".join(list(prof.server_args()) + list(j.server_args)),
                    client_threads=prof.client_threads, pids_wait=adm.wait_cap)
        def fill(a):
            a = str(a)
            for key, v in subs.items():
                a = a.replace("{%s}" % key, str(v))
            return a
        argv = [fill(a) for a in j.cmd]
        full = (["taskset", "-c", cpus] if cpus else []) + argv
        env = dict(os.environ)
        env.update(prof.environ())
        env.update({k_: fill(v) for k_, v in j.env.items()})
        env.update(B2D_DRAIN_FILE=str(jd / "DRAIN"), B2D_PIDS_WAIT=str(adm.wait_cap), PYTHONUNBUFFERED="1",
                   CL_LANE=self.name, CL_JOB=j.name, CL_RC=str(jd / ("rc.%d" % k)),
                   CL_TOKEN="%s/%s/%d/%d" % (self.name, j.name, k, time.time_ns()))
        if j.cuda:
            env["CUDA_VISIBLE_DEVICES"] = str(g)
        _unlink(jd / "DRAIN")
        _unlink(jd / ("rc.%d" % k))
        with (jd / ("log.%d.txt" % k)).open("ab") as log:
            p = subprocess.Popen(["sh", "-c", '"$@"; echo $? > "$CL_RC"', "sh"] + full, cwd=str(REPO), env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.capture(jd / "owned.json", p.pid, token="CL_TOKEN=" + env["CL_TOKEN"])
        self.children[j.name] = p
        st.update(state="running", gpu=g, idx=idx, span=span, cpus=cpus, t0=time.time(), pid=p.pid, rc=None)
        procs.atomic_json(jd / ("job.%d.json" % k), dict(
            job=j.name, try_=k, gpu=g, idx=idx, span=span, cpus=cpus, argv=full, profile=prof.describe(),
            env={x: env[x] for x in sorted(set(prof.environ()) | set(j.env) | {"B2D_PIDS_WAIT", "CUDA_VISIBLE_DEVICES"})
                 if x in env}, meta=j.meta, t0=st["t0"]))
        self.save()
        self.event("launch", job=j.name, gpu=g, workers=j.workers, idx=idx, span=span, cpus=cpus, try_=k, pid=p.pid,
                   profile=prof.name)

    # ------------------------------------------------------------------ status
    def write_status(self, lease) -> None:
        c = self.counts()
        on = ", ".join("%s@%s" % (n, self.jst(n).get("gpu")) for n in self.running())
        text = "%s %s: %d running%s, %d queued, %d done, %d failed%s" % (
            time.strftime("%F %T"), self.name, c.get("running", 0), " (%s)" % on if on else "", c.get("queued", 0),
            c.get("done", 0), c.get("failed", 0), "; draining" if self.draining(lease) else "")
        (self.root / "STATUS").write_text(text + "\n")
        procs.atomic_json(self.root / "status.json", dict(
            t=time.time(), lane=self.name, counts=c, blocked=self.blocked,
            lease=None if lease is None else lease.row(), profile=self.profile.describe(),
            jobs={n: self.jst(n) for n in self.jobs}))

    # ------------------------------------------------------------------ main loop
    def slices(self) -> dict:
        lease = self.lease
        return {g: v["cpus"] for g, v in lease.cards.items()} if lease else {}

    def records(self) -> dict:
        out = {}
        for n in self.running():
            path = self.jdir(n) / "owned.json"
            try:
                out.setdefault(self.jst(n)["gpu"], []).append(json.loads(path.read_text()))
            except (OSError, ValueError):
                pass
        return out

    def run(self, log=None) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "lane.lock").open("a") as lockf:
            try:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("another lane process owns %s" % self.root)
            return self._run(log)

    def _run(self, log) -> int:
        procs.capture(self.root / "lane.owned.json", os.getpid())
        for f in ("DONE", "ERROR"):
            if (self.root / f).exists():
                (self.root / f).rename(self.root / ("%s.%s" % (f, time.strftime("%Y%m%d-%H%M%S"))))
        self.log = log
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "halt", True))
        try:
            from tqdm import tqdm
            self.bar = tqdm(total=len(self.jobs), desc=self.name, initial=sum(
                self.jst(n)["state"] in ("done", "failed") for n in self.jobs))
        except ImportError:
            self.bar = None
        self.lease = self.current_lease()
        sampler = Sampler(self.root / "util.csv", self.slices, self.records, self.sample_s,
                          tb=getattr(log, "tb", None)) if self.sample_s else None
        if sampler:
            sampler.start()
        adopted = [n for n in self.running()]
        self.event("start", lane=self.name, jobs=len(self.jobs), adopted=adopted, profile=self.profile.describe(),
                   lease=None if self.lease is None else self.lease.row())
        fails = 0
        try:
            while True:
                try:
                    self.add_generated()
                    self.lease = lease = self.current_lease()
                    drain = self.draining(lease)
                    self.reap(processes(), drain)
                    if drain:
                        for n in self.running():
                            (self.jdir(n) / "DRAIN").touch()
                    elif lease is not None:
                        self.schedule(lease, processes())
                    self.write_status(lease)
                    fails = 0
                    c = self.counts()
                    if not c.get("running") and (drain or not c.get("queued")):
                        break
                    if not c.get("running") and not self.ready_any():
                        self.event("deadlock", queued=[n for n in self.jobs if self.jst(n)["state"] == "queued"])
                        break
                except Exception:                           # noqa: BLE001 - a bad round is retried, ten in a row stop
                    fails += 1
                    self.event("loop_error", n=fails, tb=traceback.format_exc()[-2000:])
                    if fails >= 10:
                        (self.root / "ERROR").write_text(traceback.format_exc())
                        return 1
                time.sleep(self.poll_s)
        finally:
            if sampler:
                sampler.halt.set()
            if self.bar is not None:
                self.bar.close()
            self.save()
        c = self.counts()
        summary = dict(t=time.strftime("%F %T"), counts=c, jobs={n: self.jst(n) for n in self.jobs})
        if c.get("failed"):
            procs.atomic_json(self.root / "ERROR", summary)
        elif not c.get("queued") and not c.get("running"):
            procs.atomic_json(self.root / "DONE", summary)
        self.event("end", **c)
        self.write_status(self.lease)
        return 1 if c.get("failed") else (0 if not c.get("queued") else 3)

    def ready_any(self) -> bool:
        """Can any queued job ever start (its deps are done or still on their way)?"""
        bad = {n for n in self.jobs if self.jst(n)["state"] == "failed"}
        return any(self.jst(j.name)["state"] == "queued" and not (set(j.deps) & bad) for j in self.jobs.values())


# ---------------------------------------------------------------------- operator actions on a lane root
def stop(root: Path) -> list:
    """Stop the lane process, then every running job's recorded tree. Never by pattern."""
    root = Path(root)
    stopped = []
    lane_rec = root / "lane.owned.json"
    if lane_rec.exists():
        rec = json.loads(lane_rec.read_text())
        if procs.same(rec["root"], processes()):
            procs.signal_member(rec["root"], signal.SIGTERM)
            stopped.append(rec["root"]["pid"])
            time.sleep(2)
    st = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {"jobs": {}}
    for n, s in st["jobs"].items():
        rec = root / "jobs" / n / "owned.json"
        if s.get("state") == "running" and rec.exists():
            stopped += procs.stop(rec)
    return stopped
