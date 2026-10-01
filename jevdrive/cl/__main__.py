"""python -m jevdrive.cl <command>: see docs/closed-loop-runbook.md.

  probe [--json]                       what the box has now: quota, pids, memory, NUMA, cards, CARLA per card, table rows
  plan [--profile P] [--gpus 1,2] ...  per-card sizing the defaults would give (workers, threads, PID caps, indices)
  profiles                             the named worker profiles
  lease LANE --gpus N|1,2 [--cores-per-card C] [--workers-per-card W] [--span S] [--status TEXT] [--dry-run]
                                       find free cards / NUMA-local cores / index blocks and write the lane's table row
  release LANE [NOTE]                  archive the lane's row (sch_table.py finish)
  run LANEFILE [--gpus ..] [--cpus ..] [--workers-per-card W] [--profile P] [--num-threads T] [--client-threads N]
      [--only a,b] [--fail-fast] [--arg k=v ...] [--dry-run]
                                       run the jobs a lane file declares on the lane's lease (re-read every round)
  status ROOT | drain ROOT | stop ROOT one lane root: its STATUS / stop new work and let routes finish / stop by record
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

from . import capacity, lease as L, profiles
from .box import probe
from .lane import Lane, data_dir, stop as stop_lane


def _gpus(spec):
    return [int(x) for x in str(spec).split(",") if x.strip() != ""] if spec else []


def _profile(a) -> profiles.Profile:
    p = profiles.get(getattr(a, "profile", None), client_threads=getattr(a, "client_threads", None),
                     pool_threads=getattr(a, "pool_threads", None))
    nt = getattr(a, "num_threads", None)
    return p if nt is None else dataclasses.replace(p, num_threads=nt or None)


def cmd_probe(a):
    box = probe()
    rows = L.load()
    if a.json:
        print(json.dumps(dict(box=box.to_dict(), table=rows), indent=2, default=str))
        return 0
    print("host CPUs %d, cgroup quota %s cores, affinity %d CPUs, NUMA %s" % (
        box.host_cpus, box.quota_cores or "none", len(box.affinity),
        "; ".join("%d: %d CPUs" % (n, len(c)) for n, c in sorted(box.numa.items())) or "-"))
    print("pids %d / %s, memory %.0f / %s GiB, load %s, ephemeral ports %d-%d" % (
        box.pids_current, box.pids_max or "max", box.mem_current_gb, "%.0f" % box.mem_max_gb if box.mem_max_gb else "max",
        " ".join(box.load), *box.ephemeral))
    held = {g: r["lane"] for r in L.live(rows) for g in L.gpus(r)}
    print("\n| card | NUMA | VRAM used / total GB | util % | CARLA | compute procs | table lane |\n|--:|--:|--:|--:|--:|--:|:--|")
    for c in box.cards:
        print("| %d | %d | %.1f / %.1f | %d | %d | %d | %s |" % (c.index, c.numa, c.mem_used_mib / 1024, c.mem_total_mib / 1024,
                                                           c.util, c.carla, len(c.compute_pids), held.get(c.index, "")))
    print("\nlive table rows:")
    for r in L.live(rows):
        print("  " + "\t".join(r[k] for k in L.COLS))
    for b in L.conflicts(rows, box.ephemeral[0]):
        print("CONFLICT:", b)
    p = capacity.plan(box, profiles.get())
    print("\ndefault plan (%s): %s cores per card, %d workers per card (%.1f cores each), ~%d threads per worker, "
          "PID plan cap %d / wait cap %d, server indices %d-%d" % (
              p["profile"], p["cores_per_card"], p["workers_per_card"], p["cores_per_worker"], p["threads_per_worker"],
              p["pids_plan_cap"], p["pids_wait_cap"], *p["index_bounds"]))
    return 0


def cmd_plan(a):
    box = probe()
    print(json.dumps(capacity.plan(box, _profile(a), _gpus(a.gpus), a.cores_per_card, a.workers_per_card,
                                   a.agent_threads), indent=2))
    return 0


def cmd_profiles(a):
    for n, p in profiles.PROFILES.items():
        print(("* " if n == profiles.DEFAULT else "  ") + p.describe() + "  server args: %s" % (" ".join(p.server_args()) or "-"))
    return 0


def cmd_lease(a):
    box = probe()
    want = _gpus(a.gpus) if "," in str(a.gpus) or a.explicit else None
    n = len(want) if want else int(a.gpus)
    cpc = a.cores_per_card or int(box.cores // max(len(box.cards), 1))
    w = a.workers_per_card or capacity.workers_per_card(cpc, min(c.mem_total_mib for c in box.cards) / 1024)
    ls = L.find_free(box, L.load(), a.lane, n, want, cpc, a.span, w)
    ls.status = a.status or "running since %s" % time.strftime("%F %H:%M")
    print("\t".join(ls.row()[k] for k in L.COLS))
    if not a.dry_run:
        L.grant(ls, ephemeral_lo=box.ephemeral[0])
        print("granted")
    return 0


def cmd_release(a):
    L.finish(a.lane, " ".join(a.note))
    print("archived", a.lane)
    return 0


def load_lanefile(path: str):
    spec = importlib.util.spec_from_file_location("lanefile_" + Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_run(a):
    mod = load_lanefile(a.lanefile)
    name = a.lane or mod.NAME
    root = Path(a.root or getattr(mod, "ROOT", name))
    root = root if root.is_absolute() else data_dir() / "runs" / root
    args = dict(kv.split("=", 1) for kv in a.arg)
    jobs = mod.jobs(args)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j.name in keep]
    prof = _profile(a) if (a.profile or a.num_threads is not None or a.client_threads is not None) else \
        getattr(mod, "PROFILE", None) or profiles.get()
    row = L.get(name)
    fixed = None
    if a.gpus or a.cpus or a.idx0:
        if row is None:
            sys.exit("lane %s has no table row: lease it first (python -m jevdrive.cl lease %s ...)" % (name, name))
        fixed = L.Lease(row.lane, dict(row.cards), row.span, row.workers, row.status)
        if a.gpus:
            fixed.cards = {g: fixed.cards[g] for g in _gpus(a.gpus)}
        for g, v in L._map(a.cpus or "").items():
            fixed.cards[g]["cpus"] = v
        for g, v in L._map(a.idx0 or "").items():
            fixed.cards[g]["idx0"] = int(v)
    elif row is None:
        sys.exit("lane %s has no table row: lease it first (python -m jevdrive.cl lease %s ...)" % (name, name))
    lane = Lane(name, jobs, root, lease=fixed, profile=prof, workers_per_card=a.workers_per_card or getattr(
        mod, "WORKERS_PER_CARD", None), poll_s=a.poll_s, fail_fast=a.fail_fast)
    if a.dry_run:
        ls = fixed or row
        print("lane %s -> %s, profile %s, lease %s" % (name, root, prof.describe(), ls.row()))
        for j in jobs:
            print("%-28s workers %2d deps %-20s %s" % (j.name, j.workers, ",".join(j.deps) or "-", " ".join(j.cmd)[:160]))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    try:
        rel = root.relative_to(data_dir() / "runs")
        from ..runlog import RunLog
        log = RunLog(*rel.parts, "lane")
    except (ValueError, ImportError):
        log = None
    return lane.run(log)


def cmd_status(a):
    root = Path(a.root)
    print((root / "STATUS").read_text().strip() if (root / "STATUS").exists() else "no STATUS in %s" % root)
    if (root / "status.json").exists():
        s = json.loads((root / "status.json").read_text())
        for g, why in s.get("blocked", {}).items():
            print("  card %s blocked: %s" % (g, why))
    for f in sorted(root.glob("ERROR*")):
        print("  %s" % f.name)
    return 0


def cmd_drain(a):
    (Path(a.root) / "DRAIN").touch()
    print("drain requested: no new jobs; running ones finish their current routes")
    return 0


def cmd_stop(a):
    print("stopped pids:", stop_lane(Path(a.root)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevdrive.cl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("probe")
    s.add_argument("--json", action="store_true")

    def knobs(p):
        p.add_argument("--profile", default=None, choices=sorted(profiles.PROFILES))
        p.add_argument("--num-threads", type=int, default=None, help="OMP/MKL/OPENBLAS/NUMBA threads; 0 = leave unset")
        p.add_argument("--client-threads", type=int, default=None, help="carla.Client worker threads; 0 = CARLA default")
        p.add_argument("--pool-threads", type=int, default=None, help="CARLA RPC / streaming / secondary pool size")
        p.add_argument("--workers-per-card", type=int, default=None)
        p.add_argument("--gpus", default=None)
    s = sub.add_parser("plan")
    knobs(s)
    s.add_argument("--cores-per-card", type=float, default=None)
    s.add_argument("--agent-threads", type=int, default=capacity.AGENT_THREADS)
    sub.add_parser("profiles")
    s = sub.add_parser("lease")
    s.add_argument("lane")
    s.add_argument("--gpus", default="1", help="a count, or an explicit list '1,2' (a single card: --gpus 2 --explicit)")
    s.add_argument("--explicit", action="store_true")
    s.add_argument("--cores-per-card", type=int, default=None)
    s.add_argument("--workers-per-card", type=int, default=None)
    s.add_argument("--span", type=int, default=24)
    s.add_argument("--status", default="")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("release")
    s.add_argument("lane")
    s.add_argument("note", nargs="*")
    s = sub.add_parser("run")
    s.add_argument("lanefile")
    knobs(s)
    s.add_argument("--lane", default=None)
    s.add_argument("--root", default=None)
    s.add_argument("--cpus", default=None, help="per-card core lists '1:52-76,2:77-101' (override the lease)")
    s.add_argument("--idx0", default=None, help="per-card first server index '1:160,2:200' (override the lease)")
    s.add_argument("--only", default="")
    s.add_argument("--fail-fast", action="store_true")
    s.add_argument("--poll-s", type=float, default=20.0)
    s.add_argument("--arg", action="append", default=[], help="k=v passed to the lane file's jobs(args)")
    s.add_argument("--dry-run", action="store_true")
    for n in ("status", "drain", "stop"):
        sub.add_parser(n).add_argument("root")
    a = ap.parse_args(argv)
    return globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
