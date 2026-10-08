"""Execution layer: a benchmark run is a small graph of stages, each one GPU-pool job (jevdrive.cl). Submitting the graph puts
every stage into the pool at once, chained with `after`, so the cards fill as soon as the pool admits the jobs; the CLI returns
(or waits with --wait).

Run dir = $DATA_DIR/runs/bench/<bench>/<run key>/ :
  config.json    what was asked (model, bench, preset, units, sizing)
  jobs.json      stage -> pool job id (re-submitting reuses a live job, skips a finished stage)
  pool/<stage>/  the pool's log dir of the stage: log.txt, STATUS, DONE / ERROR
  STATUS         one line, overwritten by the stages as they progress
  DONE           written by the last stage (collect) with the run's summary; ERROR when a stage failed (by `status` / `wait`)
  units.csv      one row per unit (token / group / scenario / route) with the metrics; summary.json
A stage is finished when its `done` output exists (outputs are written atomically), so re-running a command resumes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .models import REPO, data_dir


def py(env: str) -> str:
    """Interpreter of a named env on the box (`jev` = the repo venv that runs jevdrive.cl / jevdrive.bench)."""
    if env == "jev":
        v = REPO / ".venv/bin/python"
        return str(v if v.exists() else Path(sys.executable))
    return str(data_dir() / "envs" / env / "bin" / "python")


def bench_root(*p) -> Path:
    return data_dir() / "runs" / "bench" / Path(*p)


@dataclass
class Stage:
    name: str                                   # unique within the run (pool job name = <run name>-<stage>)
    cmd: object                                 # argv list or a shell string
    done: str                                   # output whose existence marks the stage finished
    vram: float = 0.5                           # declared peak VRAM (GB); <= 1 = CPU job
    cpu: int = 2
    ram: float = 0.0
    after: list = field(default_factory=list)   # stage names
    env: dict = field(default_factory=dict)
    tries: int = 1
    timeout_h: float = 0.0
    preflight: str = ""                         # smoke command (shell string) run first with the same resources
    carla: int = 0                              # CARLA servers the job starts (the pool places them and their ports)


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def status(run_dir: Path, text: str) -> None:
    line = f"{time.strftime('%F %T')} {Path(run_dir).parent.name}/{Path(run_dir).name}: {text}"
    atomic_write(Path(run_dir) / "STATUS", line + "\n")
    print(line, flush=True)


def submit(run_dir: Path, name: str, stages: list, owner: str = "bench", dry: bool = False, priority: float = 0.0,
           gpus=None) -> dict:
    """Queue the stages of one run; returns stage -> job id ('done' for finished stages). Idempotent."""
    from ..cl import pool as P
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    jf = run_dir / "jobs.json"
    old = json.loads(jf.read_text()) if jf.exists() else {}
    live = P.job_states([v for v in old.values() if v and v != "done"]) if old else {}
    ids = {}
    for s in stages:
        if Path(s.done).exists():
            ids[s.name] = "done"
            continue
        j = old.get(s.name)
        if j and live.get(j) in ("inbox", "queued", "running"):
            ids[s.name] = j
            continue
        after = [ids[a] for a in s.after if ids.get(a) not in (None, "done")]
        missing = [a for a in s.after if a not in ids]
        if missing:
            raise ValueError(f"stage {s.name}: unknown dependencies {missing} (stages must be listed after what they need)")
        log_dir = run_dir / "pool" / s.name
        for f in ("ERROR",):
            if (log_dir / f).exists():
                (log_dir / f).rename(log_dir / f"{f}.{time.strftime('%Y%m%d-%H%M%S')}")
        job_name = f"{name}-{s.name}"
        if len(job_name) > 60:
            suffix = f"-{hashlib.sha256(name.encode()).hexdigest()[:8]}-{s.name}"
            job_name = name[:60 - len(suffix)] + suffix
        key = hist_key(s.cmd, max(s.vram, 0.5))
        kw = dict(name=job_name, owner=owner, vram_gb=max(s.vram, 0.5), cpu=s.cpu, ram_gb=s.ram, after=after,
                  env=dict({"CL_HIST_KEY": key} if key else {}, **s.env), tries=s.tries, carla=s.carla, timeout_h=s.timeout_h, log_dir=str(log_dir), cwd=str(REPO), priority=priority)
        if gpus:
            kw["gpus"] = list(gpus)
        if dry:
            ids[s.name] = f"dry:{s.name}"
            print(f"[dry] {kw['name']}: vram {kw['vram_gb']} cpu {s.cpu} after {s.after}\n      "
                  + (s.cmd if isinstance(s.cmd, str) else shlex.join(map(str, s.cmd))))
            continue
        if s.preflight:
            pf = P.preflight(s.preflight, kw["name"], **{k: v for k, v in kw.items() if k not in ("name",)})
            kw["after"] = kw["after"] + [pf]
        ids[s.name] = P.submit(s.cmd, **kw)
    if not dry:
        atomic_write(jf, json.dumps(ids, indent=1))
    return ids


def state(run_dir: Path) -> dict:
    """{stage: pool state or 'done'}, plus the run's STATUS line; writes ERROR when a stage failed or was cancelled."""
    from ..cl import pool as P
    run_dir = Path(run_dir)
    jf = run_dir / "jobs.json"
    ids = json.loads(jf.read_text()) if jf.exists() else {}
    st = P.job_states([v for v in ids.values() if v != "done"])
    out = {k: ("done" if v == "done" else st.get(v, "unknown")) for k, v in ids.items()}
    bad = {k: v for k, v in out.items() if v in ("failed", "cancelled")}
    if bad and not (run_dir / "DONE").exists():
        errs = []
        for k in bad:
            e = run_dir / "pool" / k / "ERROR"
            errs.append(f"{k}: {bad[k]}" + (f"\n{e.read_text()[-1500:]}" if e.exists() else ""))
        atomic_write(run_dir / "ERROR", "\n\n".join(errs) + "\n")
    return out


def wait(run_dirs: list, poll_s: float = 30.0, quiet: bool = False, timeout_s: float = 0.0) -> bool:
    """Block until every run has DONE or a failed stage; True when all are DONE."""
    last, started = None, time.monotonic()
    while True:
        rows = []
        for d in run_dirs:
            s = state(d)
            stl = (Path(d) / "STATUS").read_text().strip() if (Path(d) / "STATUS").exists() else ""
            rows.append((d, s, stl))
        fin = [(Path(d) / "DONE").exists() or (Path(d) / "ERROR").exists() for d, _, _ in rows]
        txt = "\n".join(f"{Path(d).parent.name}/{Path(d).name}: " + " ".join(f"{k}={v}" for k, v in s.items()) + (f"\n  {stl}" if stl else "")
                        for d, s, stl in rows)
        if txt != last and not quiet:
            print(time.strftime("%T"), txt, flush=True)
            last = txt
        if all(fin):
            return all((Path(d) / "DONE").exists() for d, _, _ in rows)
        if timeout_s and time.monotonic() - started >= timeout_s:
            from ..cl import pool as P
            for d, st, _ in rows:
                if (Path(d) / "DONE").exists():
                    continue
                ids = json.loads((Path(d) / "jobs.json").read_text())
                for stage, job_state in st.items():
                    if job_state in ("inbox", "queued", "running"):
                        P.cancel(ids[stage])
                atomic_write(Path(d) / "WAIT_TIMEOUT", "wait deadline exceeded; cancellation requested\n")
                atomic_write(Path(d) / "ERROR", "wait deadline exceeded; cancellation requested\n")
            return False
        remaining = max(0, timeout_s - (time.monotonic() - started)) if timeout_s else poll_s
        time.sleep(min(poll_s, remaining))


def hist_key(cmd, vram: float) -> str:
    """Pool history key of a stage: its kind and declared VRAM, not the run name. Run names carry the model, so every
    new model would start without history and be booked at its declarations; one kind of stage at one size uses the
    same resources whatever the model. '' when the command is not a stage_cmd."""
    if isinstance(cmd, (list, tuple)) and len(cmd) > 3 and list(cmd[1:3]) == ["-m", "jevdrive.bench.stage"]:
        return "bn-%s@%ggb" % (cmd[3], vram)
    return ""


def stage_cmd(env: str, fn: str, *args) -> list:
    """argv of `python -m jevdrive.bench.stage <fn> args` in a named env (cwd = repo, so jevdrive imports from the checkout)."""
    return [py(env), "-m", "jevdrive.bench.stage", fn, *map(str, args)]


def box_cards() -> list:
    """Indices of the box's cards (live probe; [0] off the box)."""
    try:
        from ..cl import box
        return [c.index for c in box.probe().cards] or [0]
    except Exception:
        return [0]
