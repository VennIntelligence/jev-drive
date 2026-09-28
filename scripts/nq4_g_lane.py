#!/usr/bin/env python3
"""Night queue 4, lane G (user scope 2026-09-27): the ghost / perturbation matrix for the five leaderboard examinees
(PDM-Lite control, TFv6, BridgeDrive, BLUE, SimLingo), work-conserving on all seven GPUs.

  nq4_g_lane.py plan      cells, routes left, worker-hour and wall-time estimate (read-only)
  nq4_g_lane.py run       the lane (tmux jev:nq4-g); idempotent, resumes from state, adopts its live runners
  nq4_g_lane.py stop      stop this lane's runners by their recorded identities (operator only)

Cells = examinee x world x TM seed on the G route set, in priority tiers: shift seed 0 -> swap seed 0 -> orig + ghost
seed 0 -> orig + ghost seeds 1-2 (user 2026-09-28; before: seed 0, seeds 1-2, shift, swap). Outputs are the registered ones of scripts/nq4_gk.sh: runs/nq4/gk/arms/
<cand>/<variant>/s<seed> (requested.json, done/, attempts/, runner-g<gpu>.log, DONE), so the G readouts
(jevdrive.nq4_g report) and the orig reuse of night queue 3 runs work unchanged.
Staged gate per examinee x world, automatic: 1 route -> check -> 10 routes -> check (jevdrive.nq4_g pilot-check, the
written checklist) -> full. A pilot PASS already in arms_pilot counts; a failed pilot blocks only that examinee x world
(arms_blocked/<cand>.<variant>, ERROR.<cand>.<variant>), the rest goes on. A step whose seed-0 routes are all reused
from night queue 3 needs no pilot (as in the chain).
Scheduling: every POLL_S the lane measures each card (CARLA servers on it, VRAM) and starts one b2d_run runner per card
with room for the most urgent cell that still has unclaimed routes (pilot stages first); several runners, also on
different cards, share a cell (b2d_run's route claims). Room = CARD_CAP - CARLA servers of others - this lane's
workers there, limited by free VRAM (per-examinee need) and by the thread cap (pids + PIDS_PER_WORKER per new worker
<= PIDS_CAP). A route gets at most MAX_TRIES attempts in all (3 per runner, one registered retry); a cell with more than
10 % of its routes unfinished after that is failed (ERROR.cell.*), the rest goes on. The lane never kills a runner on
exit; a restarted lane adopts the live ones.
Grant: the lane's SCH row (lane nq4-g in $DATA_DIR/runs/sched/table.tsv, scripts/sch_table.py) is re-read every round:
gpus = the cards it may start runners on, workers = CARD_CAP, cpus = the runners' taskset, either one core list for
all cards or a per-card map "g:a-b,g:a-b" (UE4 sizes its TaskGraph and PoolThread pools from the affinity mask, so a
server pinned to 24 cores has ~110 threads instead of ~260 on 128; docs/carla.md "Threads per server"), idx0 = the
per-card index blocks as a map "g:i,g:i" (idx_span indices each). Another lane gets cards or cores by editing that row (sch_table.py
grant nq4-g ...); runners already on a card that left the grant finish their routes. Pilot stages run only on the
cards of the `debug` row (the test card, user 2026-09-28); cells only on the lane's own cards, filled up to CARD_CAP
counting the CARLA servers actually on each card (work-conserving).
Priority for G_RESPECT lanes (default wm-loop): such a lane writes runs/sched/demand/<lane>.json = {"<gpu>": workers}
when it wants slots. On those cards the lane counts max(demand, their actual servers) as taken, starts nothing that
would exceed CARD_CAP, and drains just enough of its own runners (B2D_DRAIN_FILE: a drained runner takes no new route
and exits after its current ones), so no route is ever killed. Without a demand file nothing is held back. A revoked / done row stops new
launches. Without a row: the old fixed layout (cards 0-6, 5 per card, indices 420 + 10 g).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cx_controller import atomic, claim, process_snapshot, same   # noqa: E402
from cx_owned_process import refresh                             # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
G = DATA / "runs/nq4/gk"
OUT = DATA / "runs/nq4/g-lane"
CANDS = ["pdm", "tfv6", "bridgedrive", "blue", "simlingo"]
# Order (user 2026-09-28, after the seed-0 peek showed no position memorisation): shift seed 0, swap seed 0, the rest of
# seed 0, then ghost + orig seeds 1-2 as filler (still needed for the registered 3-seed ghost verdict). shift / swap pair
# with orig on the same base and seed from the leaderboard record (jevdrive.nq4_g report), so the seed-0 orig reference
# (night queue 3 CL10, TFv6's own run) needs no new trajectory runs.
TIERS = ([[(c, "shift", 0) for c in CANDS],
          [(c, "swap", 0) for c in CANDS],
          [(c, v, 0) for c in CANDS for v in ("ghost", "orig")],
          [(c, v, s) for s in (1, 2) for c in CANDS for v in ("ghost", "orig")]])
# User 2026-09-28: no more closed-loop capacity on the privileged expert beyond its ghost baseline (seeds 0-2); orig seed 0
# stays reused from night queue 3. Dropped cells take no new runner; routes already running finish.
DROPPED = {"pdm.orig.1", "pdm.orig.2", "pdm.shift.0", "pdm.swap.0"}
END = ("DONE", "FAILED", "BLOCKED", "DROPPED")
PILOT_SEED = {("pdm", "orig"): 1}          # PDM-Lite orig seed 0 is night queue 3's cl1_expert (registered step 1,2)
CARD_CAP = int(os.environ.get("CARD_CAP", 5))
MAX_W = int(os.environ.get("RUNNER_WORKERS", 6))       # a runner's workers, further capped by CARD_CAP
MAX_TRIES = 6
POLL_S = 20
# Thread budget per new worker, measured 2026-09-27 with stock servers: CARLA server 457 threads + route client 150-280.
PIDS_CAP = int(os.environ.get("G_PIDS_CAP", 16000))
PIDS_PER_WORKER = int(os.environ.get("G_PIDS_PER_WORKER", 700))
# Per-card budget (user 2026-09-28): a worker of agent a takes 1 / cap(a) of the card's render budget and CPU_A[a] cores
# of its slice; heavy agents' cap is CARD_CAP (the knee, SCH row workers), light agents render little and go further.
# A card is full when the render shares sum to 1, the slice is CPU_FILL busy, or it holds SERVER_MAX servers.
HEAVY_PER_CARD = int(os.environ.get("G_HEAVY_PER_CARD", 3))   # while light work is queued: keeps every card mixed
HEAVY = {"simlingo", "blue"}      # VLM-sized (~14 GB, high render + compute share); user 2026-09-28: BridgeDrive is light
CAP_A = {"tfv6": 10, "pdm": 12, **{k: int(v) for k, v in (e.split("=") for e in os.environ.get("G_CAPS", "").split(",") if e)}}
CPU_A = {"simlingo": 2.0, "blue": 2.0, "bridgedrive": 2.0, "tfv6": 2.0, "pdm": 1.2}   # cores / worker, measured 2026-09-28
SERVER_MAX, CPU_FILL = int(os.environ.get("G_SERVER_MAX", 10)), 0.85
NEED_GB = {"pdm": 8, "tfv6": 11, "bridgedrive": 11, "blue": 14, "simlingo": 14}   # CARLA server + author model, per worker
IDX0, IDX_SPAN = 420, 10                    # fallback layout when the SCH row is missing
CPUS = os.environ.get("G_CPUS", "0-39,56-81,118-179")
LANE, SCH = "nq4-g", DATA / "runs/sched/table.tsv"
GPUS, BLOCKS = list(range(7)), {g: range(IDX0 + IDX_SPAN * g, IDX0 + IDX_SPAN * (g + 1)) for g in range(7)}
RESPECT, DEMAND, PILOT_GPUS = os.environ.get("G_RESPECT", "wm-loop").split(","), {}, [6]
WMIN = {"pdm": 2.0, "tfv6": 8.4, "bridgedrive": 5.2, "blue": 4.4, "simlingo": 15.0}   # prior worker-min / route
PY_CARLA, PY_SL, PY_VENV = DATA / "envs/carla/bin/python", DATA / "envs/simlingo/bin/python", REPO / ".venv/bin/python"
SIM = DATA / "third_party/simlingo"


def now():
    return time.time()


def load_grant():
    """Re-read this lane's SCH row into GPUS, CARD_CAP, CPUS and BLOCKS, the debug row into PILOT_GPUS and the
    G_RESPECT lanes' demand files into DEMAND (module docstring, "Grant")."""
    global GPUS, CARD_CAP, CPUS, BLOCKS, DEMAND, PILOT_GPUS
    import csv
    DEMAND = {}
    for lane in RESPECT:
        try:
            for g, n in json.loads((SCH.parent / "demand" / f"{lane}.json").read_text()).items():
                DEMAND[int(g)] = DEMAND.get(int(g), 0) + int(n)
        except (OSError, ValueError, AttributeError):
            pass
    if not SCH.exists():
        return
    with SCH.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    row = next((r for r in rows if r["lane"] == LANE), None)
    dbg = next((r for r in rows if r["lane"] == "debug"), None)
    if dbg is not None:
        PILOT_GPUS = [int(g) for g in dbg["gpus"].split(",") if g.isdigit()]
    if row is None:
        return
    if row["status"].startswith(("done", "revoked")):
        GPUS = []
        return
    span = int(row["idx_span"])
    blocks = {int(g): range(int(i), int(i) + span) for g, i in (e.split(":") for e in row["idx0"].split(","))}
    GPUS = [int(g) for g in row["gpus"].split(",") if int(g) in blocks]
    CARD_CAP, CPUS, BLOCKS = int(row["workers"]), row["cpus"], blocks
    PILOT_GPUS = [g for g in PILOT_GPUS if g in blocks]


def event(kind, **kw):
    row = dict(t=round(now(), 3), kind=kind, **kw)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    line = time.strftime("%F %T ") + kind + " " + " ".join(f"{k}={v}" for k, v in kw.items())
    with (OUT / "log.txt").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def arm_dir(c, v, s):
    return G / "arms" / c / v / f"s{s}"


# ---------------------------------------------------------------- route lists (the chain's ids_for / step_ids)
def requested(c, v, s) -> list[str]:
    out = arm_dir(c, v, s)
    f = out / "requested.json"
    if not f.exists():
        import pandas as pd
        sys.path.insert(0, str(REPO))
        from jevdrive import nq4_g as NG
        t = pd.read_csv(NG.root() / "routes.csv", dtype={"base": str})
        var = pd.read_csv(NG.root() / "variants.csv", dtype={"id": str, "base": str})
        var = var[var.variant == v].merge(t[["base", "obstacle"]], on="base")
        if v == "orig":
            have = set()
            for d in NG.reuse_dirs().get((c, s), []):
                have |= {p.stem for p in (d / "done").glob("*.json")} if (d / "done").exists() else set()
            var = var[~var.base.isin(have)]
        var = var.merge(t[["base", "town"]], on="base").sort_values(["town", "id"])
        out.mkdir(parents=True, exist_ok=True)
        atomic(f, list(var.id))
    return json.loads(f.read_text())


def done(c, v, s) -> set[str]:
    d = arm_dir(c, v, s) / "done"
    return {p.stem for p in d.glob("*.json")} if d.exists() else set()


def tries(c, v, s, rid) -> tuple[int, int]:
    """(attempts that ran the route, all attempts). An attempt whose server died before the route ticked once never ran
    it (2026-09-28: the driver-lock storm killed every new server at map load), so it does not use up one of the route's
    MAX_TRIES; all attempts together stay under 2 * MAX_TRIES so a route that always kills its server still ends."""
    real = total = 0
    for p in (arm_dir(c, v, s) / "attempts" / rid).glob("*"):
        if not p.name.isdigit():
            continue
        total += 1
        try:
            a = json.loads((p / "attempt.json").read_text())
            real += not (str(a.get("status", "")).startswith("server_died") and not a.get("ticks"))
        except (OSError, ValueError):
            real += 1                     # running, or ended without a record: counts
    return real, total


def left(c, v, s, ids=None) -> list[str]:
    ids = requested(c, v, s) if ids is None else ids
    d = done(c, v, s)
    return [r for r in ids if r not in d and (lambda t: t[0] < MAX_TRIES and t[1] < 2 * MAX_TRIES)(tries(c, v, s, r))]


# ---------------------------------------------------------------- cards
_SMI = {}


def smi(q, max_age=60.0):
    """nvidia-smi query, cached: every call takes the driver's device lock, and on 2026-09-28 per-round queries from
    several lanes helped CARLA start-ups pile up on it (D state). At most one real query per kind per minute."""
    t, out = _SMI.get(q, (0.0, None))
    if out is None or now() - t > max_age:
        out = subprocess.run(["nvidia-smi", f"--query-{q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=60).stdout.splitlines()
        _SMI[q] = (now(), out)
    return out


def card_cpus(gpu=None):
    """The taskset list for a runner on `gpu` (None: every core the lane holds, for its CPU-side helpers)."""
    if ":" not in CPUS:
        return CPUS
    m = dict(e.split(":") for e in CPUS.split(","))
    return m[str(gpu)] if gpu is not None else ",".join(m.values())


def cards(rows, own=frozenset()):
    """Per card: VRAM (cached query), CARLA servers of this lane (RPC port index in `own`, the live runners' blocks) and
    of others, counted from the process table (-graphicsadapter = CUDA card on this box), and other lanes' big GPU jobs."""
    bus, by_gpu = {}, {}
    for line in smi("gpu=index,pci.bus_id,memory.used,memory.total"):
        i, b, u, t = [x.strip() for x in line.split(",")]
        bus[b.lower()[-12:]] = by_gpu[int(i)] = dict(gpu=int(i), used=float(u) / 1024, total=float(t) / 1024, other=0,
                                                     mine=0, foreign=0)
    for line in smi("compute-apps=gpu_bus_id,pid,used_memory,process_name"):
        b, pid, mem, name = [x.strip() for x in line.split(",", 3)]
        if b.lower()[-12:] in bus and "CarlaUE4" not in name and mem.isdigit() and int(mem) >= 2048 and not any(
                m in " ".join(rows.get(int(pid), {}).get("argv", [])) for m in ("/runs/nq4/", "b2d_", "leaderboard")):
            bus[b.lower()[-12:]]["foreign"] += 1      # another lane's GPU job (e.g. a 3DGS training): pilots avoid it
    for r in rows.values():
        a = r.get("argv", [])
        if not a or "CarlaUE4-Linux-Shipping" not in a[0]:
            continue
        g = next((int(x.split("=")[1]) for x in a if x.startswith("-graphicsadapter=")), None)
        port = next((int(x.split("=")[1]) for x in a if x.startswith("-carla-rpc-port=")), None)
        if g in by_gpu:
            by_gpu[g]["mine" if port is not None and (port - 2000) // 50 in own else "other"] += 1
    return by_gpu


def cap(c):
    return CAP_A.get(c, CARD_CAP)


def n_cores(spec):
    return sum(int(b or a) - int(a) + 1 for a, _, b in (x.partition("-") for x in spec.split(",")))


def up_indices(rows):
    """Server indices of every live CARLA server (from its RPC port)."""
    out = set()
    for r in rows.values():
        a = r.get("argv", [])
        if a and "CarlaUE4-Linux-Shipping" in a[0]:
            out |= {(int(x.split("=")[1]) - 2000) // 50 for x in a if x.startswith("-carla-rpc-port=")}
    return out


def pids_now():
    return int(Path("/sys/fs/cgroup/pids.current").read_text())


# ---------------------------------------------------------------- runners
def launch(job, gpu, workers, idx, span, ids):
    c, v, s = job["cand"], job["variant"], job["seed"]
    out = arm_dir(c, v, s)
    out.mkdir(parents=True, exist_ok=True)
    xml = G / "g_routes.xml"
    drain = OUT / "drain" / f"{time.time_ns()}"
    env = dict(os.environ, B2D_DRAIN_FILE=str(drain), B2D_NQ4_TRACE="1", B2D_PIDS_WAIT=str(PIDS_CAP), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", NUMBA_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    env.pop("B2D_NQ4_REUSE_WORLD", None)
    if c == "pdm":
        env.update(BENCH2DRIVE_ROOT=str(SIM / "Bench2Drive"), WORK_DIR=str(SIM))
        cmd = ["taskset", "-c", card_cpus(gpu), str(PY_CARLA), "scripts/b2d_run.py", "--routes", str(xml), "--route-ids", ",".join(ids),
               "--out", str(out), "--workers", str(workers), "--server-index", str(idx), "--index-span", str(span),
               "--gpu-rank", str(gpu), "--tm-seed", str(s), "--no-spectator", "--no-reap", "--client-threads", "8",
               "--max-attempts", "3", "--stall-s", "480", "--route-timeout-s", "3600", "--python", str(PY_SL),
               "--agent", "scripts/b2d_expert_agent.py", "--agent-config", "expert+nq4"]
    else:
        env.update(CL10_ROUTES=str(xml), INDEX_SPAN=str(span), B_CPUS=card_cpus(gpu))
        cmd = ["scripts/nq3_b_cl10.sh", c, str(gpu), str(workers), str(idx), str(s), str(out), ",".join(ids)]
    with (out / f"runner-g{gpu}.log").open("a") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rec = OUT / "runners" / f"{p.pid}.owned.json"
    refresh(rec, p.pid)
    (out / "runner.pids").open("a").write(f"{p.pid}\n")
    return dict(pid=p.pid, record=str(rec), drain=str(drain), root=json.loads(rec.read_text())["root"], gpu=gpu, workers=workers,
                idx=idx, span=span, t0=now(), n_ids=len(ids))


def alive(r, rows):
    return same(r["root"], rows)


# ---------------------------------------------------------------- the lane
class Lane:
    def __init__(self):
        self.path = OUT / "state.json"
        self.st = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.st.setdefault("runners", {})     # pid -> {job..., gpu, workers, idx, span}
        self.st.setdefault("pilots", {})      # "c.v" -> {state, stage, t0}
        self.st.setdefault("cells", {})       # "c.v.s" -> {state}
        self.st.setdefault("tier_reported", [])
        self.children = {}

    def save(self):
        atomic(self.path, self.st)

    # pilots ---------------------------------------------------------
    def pilot(self, c, v):
        """State of the examinee x world gate: PASS / WAIVED / BLOCKED, or STAGE with stage 1 | 2."""
        key = f"{c}.{v}"
        p = self.st["pilots"].setdefault(key, {})
        if p.get("state"):
            return p
        seed = PILOT_SEED.get((c, v), 0)
        if (G / "arms_pilot" / key / "PASS").exists():
            p.update(state="PASS", how="existing PASS")
        elif (G / "arms_blocked" / key).exists():
            p.update(state="BLOCKED", how="existing block")
        elif not requested(c, v, seed):
            p.update(state="WAIVED", how="seed routes all reused from night queue 3")
        else:
            p.update(state="STAGE", stage=1, seed=seed, launched=False, runner=None, t0=None)
        event("pilot", cell=key, state=p["state"], how=p.get("how", ""))
        return p

    def pilot_ids(self, c, v, p):
        ids = requested(c, v, p["seed"])
        return ids[:1] if p["stage"] == 1 else ids[:10]

    def check_pilot(self, c, v, p):
        key = f"{c}.{v}"
        d = G / "arms_pilot" / key
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["taskset", "-c", card_cpus(), str(PY_VENV), "-m", "jevdrive.nq4_g", "pilot-check", "--cand", c,
                            "--variant", v, "--out", str(arm_dir(c, v, p["seed"])), "--ids", ",".join(self.pilot_ids(c, v, p)),
                            "--since", str(p["t0"])], cwd=REPO, capture_output=True, text=True)
        (d / "check.err").open("a").write(r.stderr)
        (d / f"stage{p['stage']}.json").write_text(r.stdout)
        try:
            ok = json.loads(r.stdout)["pass"] is True
        except (ValueError, KeyError):
            ok = False
        event("pilot_check", cell=key, stage=p["stage"], ok=ok, result=r.stdout.strip()[:400])
        if not ok:
            (G / "arms_blocked").mkdir(exist_ok=True)
            (G / "arms_blocked" / key).write_text(r.stdout or r.stderr[-2000:])
            (G / f"ERROR.{key}").write_text(f"# nq4-g pilot FAILED: {c} {v} stage {p['stage']} "
                                            f"({time.strftime('%F %T %Z')})\n\n{r.stdout}\n")
            p.update(state="BLOCKED", how=f"stage {p['stage']} failed")
        elif p["stage"] == 1:
            p.update(stage=2, launched=False, runner=None, t0=None)
        else:
            (d / "PASS").write_text(time.strftime("%F %T") + "\n")
            p.update(state="PASS", how="staged pilot")

    # work list -------------------------------------------------------
    def jobs(self):
        """Units that may take a runner now, most urgent first: by tier, a tier's pilot stages ahead of its cells."""
        out = []
        for tier, cells in enumerate(TIERS):
            for c, v, s in cells:
                key = f"{c}.{v}.{s}"
                if key in DROPPED and self.st["cells"].get(key, {}).get("state") != "DROPPED":
                    self.st["cells"][key] = dict(state="DROPPED", at=time.strftime("%F %T"), done=len(done(c, v, s)))
                    event("cell_end", cell=key, state="DROPPED")
                if self.st["cells"].get(key, {}).get("state") in END:
                    continue
                p = self.pilot(c, v)
                if p["state"] == "BLOCKED":
                    self.st["cells"][key] = dict(state="BLOCKED")
                    event("cell_end", cell=key, state="BLOCKED")
                elif p["state"] == "STAGE":
                    if s == p["seed"] and not p["launched"]:
                        out.append(dict(kind="pilot", cand=c, variant=v, seed=s, tier=tier - 0.5, key=f"pilot:{c}.{v}",
                                        ids=self.pilot_ids(c, v, p)))
                else:
                    out.append(dict(kind="cell", cand=c, variant=v, seed=s, tier=tier, key=key, ids=requested(c, v, s)))
        return sorted(out, key=lambda j: j["tier"])

    def live(self, key, rows):
        return [r for r in self.st["runners"].values() if r["key"] == key and alive(r, rows)]

    def settle(self, rows):
        """A launched pilot stage whose runner ended is checked; a cell with nothing left and no runner is closed."""
        for key, p in self.st["pilots"].items():
            if p.get("state") == "STAGE" and p.get("launched"):
                r = self.st["runners"].get(str(p.get("runner")))
                if r is None or not alive(r, rows):
                    self.check_pilot(*key.split("."), p)
        for cells in TIERS:
            for c, v, s in cells:
                key = f"{c}.{v}.{s}"
                cell = self.st["cells"].setdefault(key, {})
                if cell.get("state") in END:
                    continue
                if self.pilot(c, v)["state"] not in ("PASS", "WAIVED") or self.live(key, rows) or left(c, v, s):
                    continue
                req = requested(c, v, s)
                n = len(done(c, v, s) & set(req))
                ok = len(req) - n <= 0.10 * max(len(req), 1)
                out = arm_dir(c, v, s)
                if ok:
                    atomic(out / "DONE", dict(cand=c, variant=v, seed=s, done=n, requested=len(req),
                                              finished=time.strftime("%F %T"), owner="nq4_g_lane"))
                else:
                    (G / f"ERROR.cell.{key}").write_text(f"# nq4-g cell FAILED: {key} ({time.strftime('%F %T %Z')})\n\n"
                                                         f"{len(req) - n} / {len(req)} routes unfinished after {MAX_TRIES} attempts\n")
                if c in ("simlingo", "blue"):
                    subprocess.run(["rm", "-rf", str(out / "viz")])   # the author agents' debug images: output only
                cell.update(state="DONE" if ok else "FAILED", done=n, requested=len(req), at=time.strftime("%F %T"))
                event("cell_end", cell=key, **cell)

    # scheduling ------------------------------------------------------
    def own(self, rows):
        """Server indices of this lane's live runners, whatever card or grant they started under."""
        return {i for r in self.st["runners"].values() if alive(r, rows) for i in range(r["idx"], r["idx"] + r["span"])}

    def blocks_free(self, gpu, rows):
        used = self.own(rows)
        return [i for i in BLOCKS.get(gpu, ()) if i not in used]

    def eff(self, r, up):
        """Workers of a runner that still hold a server. A worker whose share of routes ran out exits for good, so after
        start-up (300 s) a count below the last one that holds for three polls lowers it (a server restart is shorter)."""
        n = sum(1 for i in range(r["idx"], r["idx"] + r["span"]) if i in up)
        e = r.get("eff", r["workers"])
        if now() - r["t0"] < 300 or n >= e:
            r["low"] = 0
        else:
            r["low"] = r.get("low", 0) + 1
            if r["low"] >= 3:
                r["eff"], e = n, n
        return e

    def drain(self, rows, other):
        """Only where a G_RESPECT lane has written a demand: if its claim plus this lane's undrained render share exceed
        the card, drain runners (smallest share that covers the excess first); each finishes its current routes and
        exits. Without a demand nothing is drained."""
        for g in GPUS:
            if not DEMAND.get(g):
                continue
            live = [r for r in self.st["runners"].values() if r["gpu"] == g and alive(r, rows) and r.get("drain")]
            sh = lambda r: r.get("eff", r["workers"]) / cap(r["cand"])
            excess = other.get(g, 0) / CARD_CAP + sum(sh(r) for r in live if not r.get("drained")) - 1
            while excess > 1e-6:
                cand = [r for r in live if not r.get("drained")]
                if not cand:
                    break
                r = min((r for r in cand if sh(r) >= excess), key=sh, default=max(cand, key=sh))
                Path(r["drain"]).parent.mkdir(parents=True, exist_ok=True)
                Path(r["drain"]).touch()
                r["drained"] = now()
                excess -= sh(r)
                event("drain", gpu=g, pid=r["pid"], workers=r["workers"], job=r["key"], demand=DEMAND.get(g, 0))

    def schedule(self, rows):
        jobs = self.jobs()
        for j in jobs:                        # routes still to run and workers already on them, once per round
            j["rest"] = left(j["cand"], j["variant"], j["seed"], j["ids"])
            j["busy"] = sum(r.get("eff", r["workers"]) for r in self.live(j["key"], rows) if not r.get("drained"))
        if not jobs:
            return
        info = cards(rows, self.own(rows))
        up = up_indices(rows)
        live = [r for r in self.st["runners"].values() if alive(r, rows)]
        young = [r for r in live if now() - r["t0"] < 300]
        pids = pids_now() + PIDS_PER_WORKER * sum(r["workers"] for r in young)   # servers that may still be starting
        use = {g: {} for g in info}           # card -> agent -> workers that hold a server (or are still starting)
        for r in live:
            if r["gpu"] in use:
                use[r["gpu"]][r["cand"]] = use[r["gpu"]].get(r["cand"], 0) + self.eff(r, up)
        other = {g: max(info[g]["other"], DEMAND.get(g, 0)) for g in info}
        self.drain(rows, other)

        def budget(g):                        # render share, cores and servers in use on card g
            u = use[g]
            return (sum(n / cap(c) for c, n in u.items()) + other[g] / CARD_CAP,
                    sum(n * CPU_A[c] for c, n in u.items()) + 2.0 * other[g], sum(u.values()) + other[g])

        load = {g: budget(g)[0] + 0.5 * info[g]["foreign"] for g in info}
        # Pilot stages go only to the test card(s); a pilot's crash-rate check must not measure a crowded card, so
        # among those the least loaded one with room.
        granted = [g for g in info if g in GPUS or g in PILOT_GPUS]
        calm = min((load[g] for g in granted if g in PILOT_GPUS and budget(g)[0] < 1), default=None)
        # Placement (user 2026-09-28): heavy workers spread evenly over the batch cards - a card takes a heavy worker only
        # while it holds no more heavy workers than the least-heavy batch card (+1 inside one launch); the rest of its
        # room goes to light cells. With no light work queued a card still takes heavy work (work-conserving).
        nheavy = {g: sum(n for c, n in use[g].items() if c in HEAVY) for g in info}
        floor = min((nheavy[g] for g in granted if g in GPUS), default=0)
        light_left = any(j["kind"] == "cell" and j["cand"] not in HEAVY and len(j["rest"]) > j["busy"] for j in jobs)
        for gpu in sorted(granted, key=lambda g: load[g]):
            card = info[gpu]
            share, cpu, nsrv = budget(gpu)
            cores = n_cores(card_cpus(gpu)) if gpu in BLOCKS else 24
            unstarted = sum(r["workers"] for r in young if r["gpu"] == gpu)
            free_gb = card["total"] - card["used"] - 8 * unstarted - 8   # keep >= 8 GB per card (sch_table.py)
            heavy_ok = gpu not in GPUS or not light_left or (nheavy[gpu] <= floor and nheavy[gpu] < HEAVY_PER_CARD)
            want_heavy = heavy_ok and nheavy[gpu] <= floor
            for j in sorted(jobs, key=lambda j: (j["cand"] in HEAVY) != want_heavy):
                c, v, s = j["cand"], j["variant"], j["seed"]
                if (j["kind"] == "pilot") != (gpu in PILOT_GPUS):
                    continue                  # pilots only on the test card, cells only on batch cards
                if j["kind"] == "cell" and c in HEAVY and not heavy_ok:
                    continue
                if j["kind"] == "pilot":
                    p = self.st["pilots"][f"{c}.{v}"]
                    if p["launched"]:
                        continue              # one runner per stage, checked when it ends
                    if j["rest"] and load[gpu] > calm:
                        continue
                    if not j["rest"]:         # every stage route already finished earlier: check at once
                        p.update(launched=True, runner=None, t0=now())
                        continue
                need = len(j["rest"]) - j["busy"]
                w = min(need, MAX_W, int((1 - share) * cap(c) + 1e-6), int((CPU_FILL * cores - cpu) // CPU_A[c]),
                        SERVER_MAX - nsrv, int(free_gb // NEED_GB[c]), (PIDS_CAP - pids) // PIDS_PER_WORKER)
                if j["kind"] == "pilot" and p["stage"] == 1:
                    w = min(w, 1)
                if j["kind"] == "cell" and c in HEAVY and gpu in GPUS and light_left:
                    w = min(w, floor + 1 - nheavy[gpu], HEAVY_PER_CARD - nheavy[gpu])
                free = self.blocks_free(gpu, rows)
                fits = lambda n: next((i for i in free if all(i + k in free for k in range(2 * n))), None)
                while w > 0 and fits(w) is None:
                    w -= 1
                if w <= 0:
                    continue
                r = launch(j, gpu, w, fits(w), 2 * w, j["rest"])
                r.update(key=j["key"], kind=j["kind"], cand=c, variant=v, seed=s)
                self.st["runners"][str(r["pid"])] = r
                use[gpu][c] = use[gpu].get(c, 0) + w
                if c in HEAVY:
                    nheavy[gpu] += w
                if j["kind"] == "pilot":
                    p.update(launched=True, runner=r["pid"], t0=r["t0"])
                j["busy"] += w
                event("launch", job=j["key"], gpu=gpu, workers=w, idx=r["idx"], routes=len(j["rest"]), pid=r["pid"],
                      share=round(share, 2), cpu=round(cpu, 1), servers=nsrv)
                self.save()
                pids += PIDS_PER_WORKER * w
                time.sleep(10)
                break                         # one launch per card per round: servers start staggered

    def status(self, rows):
        info = cards(rows, self.own(rows))
        lines = [f"# nq4 G lane {time.strftime('%F %T %Z')}", "",
                 f"pids {pids_now()} (cap {PIDS_CAP}, {PIDS_PER_WORKER} per new worker); CARD_CAP {CARD_CAP}; "
                 f"granted GPUs {','.join(map(str, GPUS)) or 'none'}; cpus {CPUS}", "", "| GPU | CARLA others | this lane | VRAM used |",
                 "|--:|--:|--:|--:|"]
        for g, c in sorted(info.items()):
            mine = sum(r["workers"] for r in self.st["runners"].values() if r["gpu"] == g and alive(r, rows))
            lines.append(f"| {g} | {c['other']} | {mine} ({c['mine']} up) | {c['used']:.0f} / {c['total']:.0f} GB |")
        lines += ["", "| cell | state | done / requested | live workers |", "|:--|:--|--:|--:|"]
        for cells in TIERS:
            for c, v, s in cells:
                key = f"{c}.{v}.{s}"
                req = requested(c, v, s)
                w = sum(r["workers"] for r in self.live(key, rows)) + sum(r["workers"] for r in self.live(f"pilot:{c}.{v}", rows) if s == r["seed"])
                p = self.st["pilots"].get(f"{c}.{v}", {})
                state = self.st["cells"].get(key, {}).get("state") or (
                    f"pilot stage {p.get('stage')}" if p.get("state") == "STAGE" else "queued/running")
                lines.append(f"| {key} | {state} | {len(done(c, v, s) & set(req))} / {len(req)} | {w} |")
        tmp = OUT / "STATUS.md.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(OUT / "STATUS.md")

    def tiers_report(self):
        for t, cells in enumerate(TIERS):
            if t in self.st["tier_reported"]:
                continue
            if all(self.st["cells"].get(f"{c}.{v}.{s}", {}).get("state") in END for c, v, s in cells):
                with (OUT / "report.log").open("a") as log:   # G readout tables, runs/nq4/gk/results/g (background)
                    subprocess.Popen(["taskset", "-c", card_cpus(), str(PY_VENV), "-m", "jevdrive.nq4_g", "report"], cwd=REPO,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.st["tier_reported"].append(t)
                event("tier_done", tier=t)

    def run(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with claim(OUT / "owner.lock"):
            (OUT / "pid").write_text(str(os.getpid()))
            for f in ("DONE", "ERROR"):
                (OUT / f).unlink(missing_ok=True)
            rows = process_snapshot()
            n = sum(alive(r, rows) for r in self.st["runners"].values())
            load_grant()
            event("start", pid=os.getpid(), adopted_runners=n, card_cap=CARD_CAP, cpus=CPUS, gpus=GPUS,
                  pids_per_worker=PIDS_PER_WORKER, pids_cap=PIDS_CAP)
            fails, last_status, grant = 0, 0, None
            while True:
                try:
                    load_grant()
                    if grant != (GPUS, CARD_CAP, CPUS, BLOCKS, DEMAND, PILOT_GPUS):
                        grant = (GPUS, CARD_CAP, CPUS, BLOCKS, dict(DEMAND), PILOT_GPUS)
                        event("grant", gpus=GPUS, pilot_gpus=PILOT_GPUS, card_cap=CARD_CAP, cpus=CPUS, demand=DEMAND,
                              blocks={g: f"{b.start}-{b.stop - 1}" for g, b in BLOCKS.items()})
                    rows = process_snapshot()
                    for pid, r in list(self.st["runners"].items()):
                        if not alive(r, rows):
                            if not r.get("ended"):
                                r["ended"] = now()
                                event("runner_end", job=r["key"], gpu=r["gpu"], pid=pid)
                    self.settle(rows)
                    self.st["runners"] = {k: r for k, r in self.st["runners"].items() if not r.get("ended") or now() - r["ended"] < 3600}
                    self.schedule(process_snapshot())
                    self.tiers_report()
                    self.save()
                    if now() - last_status > 60:
                        self.status(process_snapshot())
                        last_status = now()
                    fails = 0
                    if all(self.st["cells"].get(f"{c}.{v}.{s}", {}).get("state") in END
                           for cells in TIERS for c, v, s in cells):
                        atomic(OUT / "DONE", dict(finished=time.strftime("%F %T"), cells=self.st["cells"]))
                        event("end")
                        return 0
                except Exception:
                    fails += 1
                    event("loop_error", n=fails, tb=traceback.format_exc()[-1500:])
                    if fails >= 10:
                        (OUT / "ERROR").write_text(traceback.format_exc())
                        return 1
                time.sleep(POLL_S)


def plan():
    rows = []
    tot = 0.0
    for t, cells in enumerate(TIERS):
        wh = 0.0
        for c, v, s in cells:
            if f"{c}.{v}.{s}" in DROPPED:
                continue
            req = requested(c, v, s)
            n = len(left(c, v, s))
            wh += n * WMIN[c] / 60
            rows.append(f"tier {t} {c:12s} {v:6s} s{s}: {len(req):3d} routes, {len(done(c, v, s) & set(req)):3d} done, {n:3d} left")
        rows.append(f"tier {t}: {wh:.1f} worker-h")
        tot += wh
    print("\n".join(rows))
    print(f"total {tot:.1f} worker-h; at 30 workers {tot / 30:.1f} h, at 20 workers {tot / 20:.1f} h")


def stop():
    rows = process_snapshot()
    st = json.loads((OUT / "state.json").read_text())
    for r in st["runners"].values():
        if alive(r, rows):
            subprocess.run([sys.executable, str(REPO / "scripts/cx_owned_process.py"), "stop", r["record"]])
            print("stopped", r["pid"], r["key"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run", "stop"))
    a = ap.parse_args()
    raise SystemExit({"plan": plan, "run": lambda: Lane().run(), "stop": stop}[a.cmd]())
