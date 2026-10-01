"""The run-directory contract every new experiment uses (docs/lib.md, docs/long-runs.md).

    with Run("nq5", "pilot", seed=0, config=vars(args)) as run:
        for x in run.tqdm(items, desc="units"): ...
        run.scalar("loss", 0.1, step=1); run.summary["acc"] = 0.9

Run dir `$DATA_DIR/runs/<experiment>/<tag>/<YYYYmmdd-HHMMSS>/`: log.txt, events.jsonl, tb/, meta.json, STATUS,
and DONE (JSON summary) or ERROR (traceback), with the semantics of jevdrive.cl lanes.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..cl.procs import atomic_json

REPO = Path(__file__).resolve().parents[2]
ENV_KEYS = ("DATA_DIR", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONHASHSEED", "TMUX_PANE", "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV")
FMT = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def seed_everything(seed: int, deterministic: bool = False):
    """Seed random, numpy and torch (if installed); PYTHONHASHSEED for child processes. Returns a numpy Generator.
    deterministic=True also forces deterministic torch kernels (slower; errors on ops that have none)."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % 2**32)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def git_state(repo: Path = REPO) -> dict:
    """{sha, dirty, branch}: dirty means tracked files differ from HEAD (untracked files are ignored)."""
    def git(*a):
        r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    return dict(sha=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


def box_summary() -> dict | None:
    """Cores / memory / cards of this machine through jevdrive.cl.box.probe; None where it cannot run (no /proc)."""
    try:
        from ..cl.box import probe
        b = probe()
    except Exception:                                   # noqa: BLE001 - a probe failure must not stop a run
        return None
    return dict(cores=b.cores, host_cpus=b.host_cpus, mem_max_gb=round(b.mem_max_gb, 1),
                cards=[dict(index=c.index, uuid=c.uuid, mem_total_mib=c.mem_total_mib,
                            mem_used_mib=c.mem_used_mib) for c in b.cards])


def _default_root() -> Path:
    from ..common import data_dir
    return data_dir() / "runs"


class Run:
    """Context manager owning one run dir. Exiting cleanly writes DONE (JSON: wall time + `summary`); an exception
    (incl. KeyboardInterrupt) writes ERROR (traceback text) and is re-raised. Stale DONE / ERROR from an earlier
    attempt in the same dir (`resume=`) are renamed to DONE.<ts> / ERROR.<ts> on entry, as jevdrive.cl lanes do."""

    def __init__(self, experiment: str, tag: str = "", *, root: Path | None = None, resume: Path | None = None,
                 seed: int | None = None, config: dict | None = None, env: tuple = ENV_KEYS, probe: bool = True):
        self.name = "/".join(p for p in (experiment, tag) if p)
        self.seed, self.summary = seed, {}
        self.rng = None
        self._config, self._env, self._probe, self._resume = dict(config or {}), env, probe, resume
        self._root = Path(root) if root else None
        self._tb, self._bar_t = None, 0.0
        self.dir: Path | None = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Run:
        if self._resume:
            self.dir = Path(self._resume)
            self.dir.mkdir(parents=True, exist_ok=True)
        else:
            base = (self._root or _default_root()) / self.name / time.strftime("%Y%m%d-%H%M%S")
            self.dir, k = base, 1
            while True:                                 # runs started in the same second get -2, -3, ...
                try:
                    self.dir.mkdir(parents=True)
                    break
                except FileExistsError:
                    k += 1
                    self.dir = base.with_name(f"{base.name}-{k}")
        ts = time.strftime("%Y%m%d-%H%M%S")
        for f in ("DONE", "ERROR"):
            if (self.dir / f).exists():
                (self.dir / f).rename(self.dir / f"{f}.{ts}")
        self.log = logging.getLogger(self.name)
        self.log.setLevel(logging.INFO)
        root = logging.getLogger()
        if not root.handlers:                           # terminal output, same format as jevdrive.common.get_logger
            logging.basicConfig(level=logging.INFO, format=FMT._fmt, datefmt=FMT.datefmt)
        self._fh = logging.FileHandler(self.dir / "log.txt")
        self._fh.setFormatter(FMT)
        root.addHandler(self._fh)
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self.t0 = time.time()
        if self.seed is not None:
            self.rng = seed_everything(self.seed)
        self.meta = dict(name=self.name, dir=str(self.dir), argv=sys.argv, cwd=os.getcwd(), host=platform.node(),
                         python=sys.version.split()[0], pid=os.getpid(), seed=self.seed, git=git_state(),
                         env={k: os.environ[k] for k in self._env if k in os.environ}, config=self._config,
                         splits={}, start=time.strftime("%F %T %Z"), end=None, wall_s=None, ok=None,
                         resumed=bool(self._resume), box=box_summary() if self._probe else None)
        self._save_meta()
        self.event("start", name=self.name, seed=self.seed, git=self.meta["git"]["sha"])
        self.status("started")
        self.log.info("run dir %s", self.dir)
        return self

    def __exit__(self, et, ev, tb) -> bool:
        wall = round(time.time() - self.t0, 1)
        ok = et is None or (et is SystemExit and ev.code in (0, None))
        self.meta.update(end=time.strftime("%F %T %Z"), wall_s=wall, ok=ok)
        self._save_meta()
        if ok:
            atomic_json(self.dir / "DONE", dict(t=time.strftime("%F %T"), wall_s=wall, **self.summary))
            self.status(f"done in {wall:.0f} s")
        else:
            text = "".join(traceback.format_exception(et, ev, tb))
            (self.dir / "ERROR").write_text(f"# {self.name} failed {time.strftime('%F %T %Z')} after {wall:.0f} s\n\n"
                                            f"{text}\n")
            self.event("error", type=et.__name__, msg=str(ev)[:500], tb=text[-3000:])
            self.log.error("failed: %s: %s", et.__name__, ev)
            self.status(f"failed: {et.__name__}: {str(ev)[:200]}")
        self.event("end", ok=ok, wall_s=wall)
        self.close()
        return False

    def close(self) -> None:
        if self._tb:
            self._tb.close()
        logging.getLogger().removeHandler(self._fh)
        self._fh.close()
        self._events.close()

    # ------------------------------------------------------------------ outputs
    def path(self, *parts: str) -> Path:
        """A file inside the run dir (parents created)."""
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def info(self, msg: str, *args) -> None:
        self.log.info(msg, *args)

    def event(self, kind: str, **fields) -> None:
        """One line of events.jsonl: {"t": unix time, "kind": kind, **fields}, flushed at once."""
        self._events.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **fields}, default=_jsonable) + "\n")

    def scalar(self, tag: str, value: float, step: int) -> None:
        """A curve point: TensorBoard (when torch is installed) and an events.jsonl `scalar` line."""
        if self._tb is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.dir / "tb"), flush_secs=10)
            except ImportError:
                self._tb = False
        if self._tb:
            self._tb.add_scalar(tag, value, step)
        self.event("scalar", tag=tag, value=float(value), step=int(step))

    def scalars(self, values: dict, step: int, prefix: str = "") -> None:
        for k, v in values.items():
            self.scalar(prefix + k, v, step)

    def status(self, text: str) -> None:
        """STATUS: one current sentence (overwritten, not a log), `<date time> <name>: <text>`."""
        tmp = self.dir / ".STATUS.tmp"
        tmp.write_text(f"{time.strftime('%F %T')} {self.name}: {text}\n")
        os.replace(tmp, self.dir / "STATUS")

    def note(self, **fields) -> None:
        """Add fields to meta.json now (e.g. dataset versions, model checkpoint hashes)."""
        self.meta.update(fields)
        self._save_meta()

    def use_split(self, split) -> None:
        """Record a jevdrive.data.splits.Split (its split_id) in meta.json; mandatory for every split a run reads."""
        self.meta["splits"][f"{split.dataset}/{split.name}"] = split.id
        self._save_meta()
        self.event("split", dataset=split.dataset, name=split.name, id=split.id, n=len(split))

    def tqdm(self, iterable=None, status_s: float = 60.0, **kw):
        """tqdm on the terminal; every `status_s` seconds and at the end also STATUS and a `progress` event."""
        from tqdm import tqdm
        run = self

        class Bar(tqdm):
            def update(self, n=1):
                r = super().update(n)
                if time.time() - run._bar_t >= status_s:
                    run._progress(self)
                return r

            def close(self):
                if not self.disable and not getattr(self, "_jev_closed", False):
                    self._jev_closed = True
                    run._progress(self)
                super().close()

        return Bar(iterable, **kw)

    def _progress(self, bar) -> None:
        self._bar_t = time.time()
        d = bar.format_dict
        n, total, rate = d["n"], d["total"], d["rate"] or (d["n"] / d["elapsed"] if d["elapsed"] else None)
        eta = (total - n) / rate if total and rate else None
        self.event("progress", desc=d.get("prefix") or "", n=n, total=total, rate=rate, eta_s=eta)
        self.status("%s %d/%s%s%s" % (d.get("prefix") or "progress", n, total or "?",
                                      " (%.0f%%)" % (100 * n / total) if total else "",
                                      ", eta %s" % _hms(eta) if eta else ""))

    def _save_meta(self) -> None:
        atomic_json(self.dir / "meta.json", self.meta)


def _hms(s: float) -> str:
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)


def _jsonable(x):
    if hasattr(x, "item"):                              # numpy / torch scalars
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)


def cli_args(parser) -> None:
    """Add the shared flags: --seed, --resume DIR (reopen a run dir), --force (recompute cached artifacts)."""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None, help="reopen this run dir instead of a new one")
    parser.add_argument("--force", action="store_true", help="recompute cached artifacts (jevdrive.cache)")
