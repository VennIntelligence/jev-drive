"""python -m jevdrive.cl <command>: see docs/closed-loop-runbook.md.

GPU pool (jevdrive/cl/pool.py): agents submit jobs, the dispatcher (tmux jev:pool) runs each on a card with room.
  submit --name N --vram GB [--carla N] [--cpu N] [--train] [--priority P] [--after ID,..] [...] -- CMD ...
                                       queue a job; prints its id (one shell string or an argv after --)
  queue [--all] | show ID | cancel ID.. [--drain] | top
  hold --card G (--whole | --vram GB [--carla N]) [--idx a-b] [--cpus ..] [--pid PID] --note TEXT | holds | unhold HID
                                       resources used outside the pool (ends by itself when --pid exits)
  dispatch [--once]                    the dispatcher (run it once, in tmux jev:pool)
Box:
  probe [--json]                       what the box has now: quota, pids, memory, NUMA, cards, CARLA per card, pool
  plan [--profile P] [--gpus 1,2] ...  per-card sizing the defaults would give (workers, threads, PID caps, indices)
  profiles                             the named worker profiles
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

from . import capacity, pool as P, profiles
from .box import probe


def _gpus(spec):
    return [int(x) for x in str(spec).split(",") if x.strip() != ""] if spec else []


def _profile(a) -> profiles.Profile:
    p = profiles.get(getattr(a, "profile", None), client_threads=getattr(a, "client_threads", None),
                     pool_threads=getattr(a, "pool_threads", None))
    nt = getattr(a, "num_threads", None)
    return p if nt is None else dataclasses.replace(p, num_threads=nt or None)


def cmd_probe(a):
    box = probe()
    st, status, inbox, age = P.snapshot()
    if a.json:
        print(json.dumps(dict(box=box.to_dict(), pool=status), indent=2, default=str))
        return 0
    print("host CPUs %d, cgroup quota %s cores, affinity %d CPUs, NUMA %s" % (
        box.host_cpus, box.quota_cores or "none", len(box.affinity),
        "; ".join("%d: %d CPUs" % (n, len(c)) for n, c in sorted(box.numa.items())) or "-"))
    print("pids %d / %s, memory (no page cache) %.0f / %s GiB, load %s, ephemeral ports %d-%d" % (
        box.pids_current, box.pids_max or "max", box.mem_used_gb, "%.0f" % box.mem_max_gb if box.mem_max_gb else "max",
        " ".join(box.load), *box.ephemeral))
    held = {}
    for j in st["jobs"].values():
        if j["state"] == "running":
            held.setdefault(j.get("gpu"), []).append(j["spec"]["name"])
    print("\n| card | NUMA | VRAM used / total GB | util % | CARLA | compute procs | pool jobs |\n|--:|--:|--:|--:|--:|--:|:--|")
    for c in box.cards:
        print("| %d | %d | %.1f / %.1f | %d | %d | %d | %s |" % (c.index, c.numa, c.mem_used_mib / 1024, c.mem_total_mib / 1024,
                                                           c.util, c.carla, len(c.compute_pids), ", ".join(held.get(c.index, []))))
    print("\npool dispatcher: %s" % ("heartbeat %.0f s ago" % age if age < P.STALE_S else "NOT RUNNING"))
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


def _warn_dispatcher(age):
    if age > P.STALE_S:
        print("WARNING: the pool dispatcher is not running (no heartbeat for %s); jobs stay queued. Start it: "
              "scripts/tmux_run.sh pool .venv/bin/python -m jevdrive.cl dispatch" % (
                  "ever" if age == float("inf") else "%.0f s" % age), file=sys.stderr)


def cmd_submit(a):
    cmd = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
    if not cmd:
        sys.exit("no command (put it after --)")
    cmd = cmd[0] if len(cmd) == 1 and " " in cmd[0] else cmd
    env = dict(os.environ) if a.copy_env else {}
    env.update(dict(kv.split("=", 1) for kv in a.env))
    cwd = a.cwd or os.getcwd()
    if not a.no_check:
        bad = P.static_check(cmd, cwd) + (P.static_check(a.preflight, cwd) if a.preflight else [])
        if bad:
            sys.exit("not submitted (--no-check to override): " + "; ".join(bad))
    res = dict(owner=a.owner or None, cwd=cwd, log_dir=a.log_dir, env=env, vram_gb=a.vram, carla=a.carla, span=a.span,
               cpu=a.cpu, ram_gb=a.ram, threads=a.threads, train=a.train, exclusive=a.exclusive,
               gpus=[int(g) for g in a.gpus.split(",") if g] if a.gpus else [], max_rss_gb=a.max_rss,
               profile=a.profile or "")
    after = a.after.split(",") if a.after else []
    if a.preflight:
        pf = P.preflight(a.preflight, a.name, timeout_min=a.preflight_min, **res)
        print("preflight", pf, file=sys.stderr)
        after.append(pf)
    jid = P.submit(cmd, name=a.name, priority=a.priority, after=after, when_exists=a.when_exists, tries=a.tries,
                   timeout_h=a.timeout_h, **res)
    print(jid)
    _warn_dispatcher(P.snapshot()[3])
    return 0


def _age(t):
    s = time.time() - t
    return "%dm" % (s / 60) if s < 5400 else "%.1fh" % (s / 3600)


def cmd_queue(a):
    st, status, inbox, age = P.snapshot()
    jobs = list(st["jobs"].values()) + [dict(j, state="inbox", why="not yet read by the dispatcher") for j in inbox]
    order = {"running": 0, "queued": 1, "inbox": 2, "failed": 3, "cancelled": 4, "done": 5}
    if not a.all:
        cut = time.time() - 6 * 3600
        jobs = [j for j in jobs if j["state"] in ("running", "queued", "inbox") or j.get("t1", 0) > cut]
    jobs.sort(key=lambda j: (order.get(j["state"], 9), -float(j["spec"].get("priority") or 0), j["t_submit"]))
    print("%-16s %-9s %4s %-22s %-10s %6s %5s %4s %5s  %s" % ("id", "state", "card", "name", "owner", "vram", "carla",
                                                            "cpu", "age", "why / log"))
    for j in jobs:
        s = j["spec"]
        tail = j.get("why") or ""
        if j["state"] in ("running", "failed", "done"):
            tail = (tail + " " if tail else "") + str(Path(s.get("log_dir") or P.pool_dir() / "jobs" / j["id"]) / "log.txt")
        print("%-16s %-9s %4s %-22s %-10s %6s %5s %4s %5s  %s" % (
            j["id"], j["state"], j.get("gpu", "") if j.get("gpu") is not None else "", s["name"][:22],
            (s.get("owner") or "")[:10], "%.0f" % s["vram_gb"] if not s.get("exclusive") else "card", s.get("carla") or "",
            s.get("cpu") or "", _age(j.get("t0") or j["t_submit"]), tail[:160]))
    _warn_dispatcher(age)
    return 0


def cmd_show(a):
    st, _, inbox, _ = P.snapshot()
    j = st["jobs"].get(a.id) or next((x for x in inbox if x["id"] == a.id), None)
    if j is None:
        sys.exit("no job %s" % a.id)
    print(json.dumps(j, indent=2, default=str))
    log = Path(j["spec"].get("log_dir") or P.pool_dir() / "jobs" / a.id) / "log.txt"
    if log.exists():
        print("\n--- tail of %s" % log)
        print(log.read_bytes()[-3000:].decode(errors="replace"))
    return 0


def cmd_cancel(a):
    for jid in a.ids:
        P.cancel(jid, a.drain)
        print("%s requested for %s (the dispatcher acts within one round)" % ("drain" if a.drain else "cancel", jid))
    return 0


def cmd_top(a):
    st, status, inbox, age = P.snapshot()
    box = probe()
    acct = {int(g): v for g, v in (status.get("cards") or {}).items()}
    run = [j for j in st["jobs"].values() if j["state"] == "running"]
    q = [j for j in st["jobs"].values() if j["state"] == "queued"]
    b = status.get("box", {})
    print("box: %s cores, pids %s / %s, memory %s / %s GiB, load %s; pool: %d running, %d queued, %d in inbox" % (
        b.get("cores"), b.get("pids"), b.get("pids_max"), b.get("mem_gb"), b.get("mem_max_gb"), " ".join(b.get("load", [])),
        len(run), len(q), len(inbox)))
    print("\n| card | util % | VRAM used / total | pool booked | outside pool | free for pool | CARLA pool / other | train | jobs |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    hd = (status.get("cfg") or {}).get("headroom_gb", P.DEFAULTS["headroom_gb"])
    for c in box.cards:
        x = acct.get(c.index, {})
        jobs = ", ".join("%s %s (%.0f/%.0f GB)" % (j["id"], j["spec"]["name"][:18], j.get("vram_now", 0), j["spec"]["vram_gb"])
                         for j in run if j.get("gpu") == c.index)
        free = x.get("total_gb", 0) - hd - x.get("foreign_gb", 0) - x.get("pool_gb", 0)
        print("| %d | %d | %.0f / %.0f | %.0f | %.0f | %s | %s / %s | %s | %s |" % (
            c.index, c.util, c.mem_used_mib / 1024, c.mem_total_mib / 1024, x.get("pool_gb", 0), x.get("foreign_gb", 0),
            "held: " + x["whole_hold"] if x.get("whole_hold") else "%.0f" % max(free, 0), x.get("carla_pool", 0),
            x.get("carla_foreign", 0), x.get("train", 0), jobs or "-"))
    for h in status.get("holds", []):
        print("hold %s card %d: %s" % (h["id"], h["card"], h.get("note")))
    _warn_dispatcher(age)
    return 0


def cmd_hold(a):
    if not a.whole and a.vram <= 0:
        sys.exit("a hold declares --whole or --vram")
    h = P.add_hold(a.card, " ".join(a.note), a.whole, a.vram, a.carla, a.cpu, a.cpus, a.idx, a.pid, a.train)
    print("hold", h["id"], json.dumps(h))
    return 0


def cmd_holds(a):
    for h in P.load_holds():
        print(json.dumps(h))
    return 0


def cmd_unhold(a):
    print("dropped" if P.drop_hold(a.hid) else "no hold %s" % a.hid)
    return 0


def cmd_retarget(a):
    for jid in a.ids:
        P.retarget(jid, [int(g) for g in a.gpus.split(",") if g] if a.gpus else [])
    return 0


def cmd_dispatch(a):
    return P.Dispatcher().run(once=a.once)

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
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--owner", default="", help="who to ask about it (default: $CL_OWNER, the submitting pool job, $USER)")
    s.add_argument("--vram", type=float, default=0.0, help="peak VRAM GB on the card (default 9 per CARLA server)")
    s.add_argument("--carla", type=int, default=0, help="CARLA servers the job starts ({carla}, {idx}, {span} in cmd)")
    s.add_argument("--span", type=int, default=0, help="server indices to reserve (default 2 x carla)")
    s.add_argument("--cpu", type=int, default=0, help="cores; > 0 pins the job (taskset) to that many free cores")
    s.add_argument("--ram", type=float, default=0.0, help="host RAM GB (admission)")
    s.add_argument("--max-rss", type=float, default=0.0, help="stop the job when its process tree's RSS exceeds this GB")
    s.add_argument("--threads", type=int, default=0, help="PID estimate (default from the thread model)")
    s.add_argument("--train", action="store_true", help="a training job (per-card cap)")
    s.add_argument("--exclusive", action="store_true", help="alone on its card")
    s.add_argument("--gpus", default="", help="allowed cards, e.g. 1,2 (default any)")
    s.add_argument("--priority", type=float, default=0.0, help="higher starts first")
    s.add_argument("--after", default="", help="job ids that must finish with rc 0 first")
    s.add_argument("--when-exists", default="", help="stay queued until this path exists")
    s.add_argument("--tries", type=int, default=1)
    s.add_argument("--timeout-h", type=float, default=0.0)
    s.add_argument("--profile", default=None, choices=sorted(profiles.PROFILES))
    s.add_argument("--env", action="append", default=[], help="K=V (repeatable)")
    s.add_argument("--copy-env", action="store_true", help="run with this shell's whole environment")
    s.add_argument("--cwd", default="")
    s.add_argument("--log-dir", default="", help="log.txt / STATUS / DONE / ERROR here (default runs/pool/jobs/<id>)")
    s.add_argument("--preflight", default="", help="smoke command (one shell string, e.g. the job with --limit 2): runs "
                   "first at top priority with the same resources, the job waits for it and fails if it fails; batches "
                   "sharing the same smoke command share one run")
    s.add_argument("--preflight-min", type=float, default=15.0, help="timeout of the smoke run, minutes")
    s.add_argument("--no-check", action="store_true", help="skip the static check (script paths exist, .py compiles)")
    s.add_argument("argv", nargs=argparse.REMAINDER, metavar="CMD")
    s = sub.add_parser("queue")
    s.add_argument("--all", action="store_true")
    sub.add_parser("show").add_argument("id")
    s = sub.add_parser("cancel")
    s.add_argument("ids", nargs="+")
    s.add_argument("--drain", action="store_true", help="touch the job's DRAIN file (b2d_run finishes routes) instead")
    s = sub.add_parser("retarget", help="change the allowed cards of queued jobs (ids and after-chains stay)")
    s.add_argument("ids", nargs="+")
    s.add_argument("--gpus", default="", help="allowed cards, e.g. 0,2 (empty = any)")
    sub.add_parser("top")
    s = sub.add_parser("hold")
    s.add_argument("--card", type=int, required=True)
    s.add_argument("--whole", action="store_true")
    s.add_argument("--vram", type=float, default=0.0)
    s.add_argument("--carla", type=int, default=0)
    s.add_argument("--cpu", type=int, default=0, help="cores counted against the pool's CPU budget")
    s.add_argument("--cpus", default="", help="core list the pool must not pin jobs to")
    s.add_argument("--idx", default="", help="server indices in use, e.g. 170-175")
    s.add_argument("--train", action="store_true")
    s.add_argument("--pid", type=int, default=0, help="the hold ends when this process exits")
    s.add_argument("--note", nargs="+", required=True)
    sub.add_parser("holds")
    sub.add_parser("unhold").add_argument("hid")
    s = sub.add_parser("dispatch")
    s.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    return globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
