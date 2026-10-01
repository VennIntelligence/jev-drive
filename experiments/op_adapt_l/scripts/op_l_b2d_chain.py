#!/usr/bin/env python
"""op-adapt L in Bench2Drive: one self-advancing chain (experiments/op_adapt_l/plans/2026-10-01-op-adapt-L-b2d-prereg.md).

A unit = one `experiments/op_closed_loop/archive/op_arb.sh set <2|h> <tag>` call = one arm x one TM seed x the route set, on one card (op-drive recipe: 6 CARLA
workers + 1 openpilot server per card). Three card threads take ready units by priority (units of the same served model first, so the
openpilot server is kept alive across units: KEEP_SRV). Pace-matched controls are dynamic: after an arm finishes, `dbaseslow`
(linear match), then `dbaseslow2..4` (registration P: next speeds = prev x v_arm / v_slow) until the mean-speed ratio is in [0.9, 1.1].
Every finished unit writes its result rows (experiments.op_adapt_l.lib.op_l_b2d_report unit) into $ROOT/results/.

  .venv/bin/python experiments/op_adapt_l/scripts/op_l_b2d_chain.py --plan stage1|stage2|full|heldout [--workers 6] [--cards 0,1,2]

Hand-offs in $DATA_DIR/runs/op_l_b2d: STATUS (one sentence), DONE-<plan>, ERROR, chain/<ts>/{log.txt, events.jsonl, tb/}, util.csv
(nvidia-smi + per-core busy of each card's core list every 15 s). Only PIDs recorded by op_arb.sh are ever stopped (never pkill -f).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[3] / "scripts"
REPO = HERE.parent
sys.path.insert(0, str(REPO))
DATA = Path(os.environ["DATA_DIR"])
ROOT = DATA / "runs" / "op_l_b2d"
AD = ROOT / "arms"
ONNX = DATA / "runs" / "op_adapt_L" / "b2d" / "onnx"
DEV_ROUTES = 10
TZ = '"zone_m": {"3": [5.0, 5.0], "5": [5.0, 10.0], "6": [5.0, 10.0]}'      # STRAIGHT and lane changes only: LEFT / RIGHT handed to openpilot
R1 = '"resume": "nored"'
INT = '"intent": "route"'
# arm -> (served model, DESIRE, DRIVE_ARGS)
ARMS = {
    "dbase": ("base", "true", ""), "drive": ("base", "true", R1), "dnod": ("base", "false", R1),
    "dtz": ("base", "true", f"{R1}, {TZ}"),
    "lmain": ("lmain-s0", "false", f"{R1}, {INT}"), "lkd": ("lmain-s0", "true", f"{R1}, {INT}"),
    "ltz": ("lmain-s0", "false", f"{R1}, {INT}, {TZ}"),
    "lnoint": ("lnoint-s0", "false", R1), "ldw10": ("ldw10-s0", "false", f"{R1}, {INT}"),
    "lmain1": ("lmain1-s1", "false", f"{R1}, {INT}"), "lmain2": ("lmain2-s2", "false", f"{R1}, {INT}"),
}
FAMILY = {"drive": "ld", "lmain": "lm", "lnoint": "ln", "ldw10": "lw", "lmain1": "l1", "lmain2": "l2"}     # arm judged -> tag of its pace family
TAG_OF = {a: FAMILY.get(a, "ld") for a in ARMS} | {"lkd": "lm", "ltz": "lm", "dnod": "ld", "dtz": "ld", "dbase": "ld"}


class Unit:
    def __init__(self, tag, arm, seed, prio, model="base", deps=(), routes="2", ids="", match=""):
        self.tag, self.arm, self.seed, self.prio, self.model = tag, arm, seed, prio, model
        self.deps, self.routes, self.ids, self.match = list(deps), routes, ids, match
        self.id = f"{tag}-{arm}-s{seed}"
        self.state, self.card, self.tries = "new", None, 0

    @property
    def dir(self):
        return AD / self.id


class Chain:
    def __init__(self, a):
        from jevdrive.runlog import RunLog
        self.a = a
        self.log = RunLog("op_l_b2d", "chain")
        self.cards = [int(x) for x in a.cards.split(",")]
        self.lock = threading.Lock()
        self.units: dict[str, Unit] = {}
        self.slots = [(g, k) for g in self.cards for k in range(a.slots)]      # a slot = one openpilot server + its CARLA workers on a card
        self.cur_model = {sl: None for sl in self.slots}
        self.stop = False
        self.fatal = None
        self.card_cpus = dict(zip(self.cards, a.cpus.split(";")))
        self.cpus = {}
        for g in self.cards:                                  # a card's core list split evenly over its slots
            cores = self.cores(self.card_cpus[g])
            n = len(cores) // a.slots
            for k in range(a.slots):
                part = cores[k * n:(k + 1) * n if k < a.slots - 1 else len(cores)]
                self.cpus[(g, k)] = f"{part[0]}-{part[-1]}"
        ROOT.mkdir(parents=True, exist_ok=True)
        AD.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cores(spec):
        r = []
        for part in spec.split(","):
            lo, _, hi = part.partition("-")
            r += list(range(int(lo), int(hi or lo) + 1))
        return r

    # ---------------------------------------------------------------- plan
    def add(self, u):
        self.units[u.id] = u
        return u

    def family(self, arm, seeds, prio, chain=True, routes="2", ids="", tag=None, dbase_dep=True):
        """An arm over seeds, plus (chain) its pace-matched control iterations."""
        tag = tag or TAG_OF[arm]
        for s in seeds:
            deps = [f"ld-dbase-s{s}"] if (chain and dbase_dep and arm != "dbase") else []
            u = self.add(Unit(tag, arm, s, prio, ARMS[arm][0], routes=routes, ids=ids))
            if chain:
                self.add(Unit(tag, "dbaseslow", s, prio - 0.5, "base", deps=[u.id] + deps, routes=routes, ids=ids, match=arm))

    def build(self, plan):
        if plan == "stage1":
            for arm in ("lmain", "drive"):
                self.add(Unit("s1", arm, 0, 0, ARMS[arm][0], ids="24944"))
            return
        seeds = (0, 1)
        if plan in ("stage2", "full"):
            s = (0,) if plan == "stage2" else seeds
            for arm, prio in (("dbase", 0), ("drive", 0), ("lmain", 0)):
                self.family(arm, s, prio, chain=False)
        if plan == "full":
            for arm in ("drive", "lmain"):                       # pace chains of the two main arms
                for sd in seeds:
                    self.add(Unit(TAG_OF[arm], "dbaseslow", sd, 0.5, "base", deps=[f"{TAG_OF[arm]}-{arm}-s{sd}", f"ld-dbase-s{sd}"], match=arm))
            self.family("lnoint", seeds, 1)
            self.family("dnod", seeds, 2, chain=False)
            for arm, prio in (("lmain1", 3), ("lmain2", 3), ("ldw10", 3)):
                self.family(arm, seeds, prio)
            for arm in ("lkd", "dtz", "ltz"):
                self.family(arm, seeds, 4, chain=False)
        if plan == "heldout":
            self.plan_heldout()
        if self.a.only:                                           # resume: only these arms, and only units that are not complete yet
            keep = set(self.a.only.split(","))
            for uid, u in list(self.units.items()):
                full = len(glob.glob(str(u.dir / "done" / "*.json"))) == self.n_routes(u)
                if u.arm not in keep or (full and (u.dir / "DONE").exists()):
                    del self.units[uid]

    def plan_heldout(self):
        for arm in ("drive", "lmain"):
            self.family(arm, (0, 1), 0, routes="h", tag="h" + TAG_OF[arm], dbase_dep=False)
        for s in (0, 1):
            self.add(Unit("hld", "dbase", s, 0, "base", routes="h"))
        for u in list(self.units.values()):
            if u.arm == "dbaseslow":
                u.deps.append(f"hld-dbase-s{u.seed}")

    # ---------------------------------------------------------------- running
    def event(self, kind, **kw):
        self.log.event(kind, **kw)

    def status(self, text):
        (ROOT / "STATUS").write_text(f"{time.strftime('%F %T')} {text}\n")

    def fail(self, why):
        self.fatal = why
        self.stop = True
        (ROOT / "ERROR").write_text(f"# op_l_b2d chain ERROR {time.strftime('%F %T %Z')}\n\n{why}\n")
        self.log.info("ERROR: " + why)
        self.event("error", reason=why)

    def ready(self, card):
        """card is a slot (gpu, k)."""
        with self.lock:
            cand = [u for u in self.units.values() if u.state == "new" and all(self.units[d].state == "done" for d in u.deps)]
            if not cand:
                return None
            cand.sort(key=lambda u: (u.prio, u.model != self.cur_model[card], u.id))
            u = cand[0]
            u.state, u.card = "run", card
            self.cur_model[card] = u.model
            return u

    def env_for(self, u, slot):
        model, desire, args = ARMS.get(u.arm, ("base", "true", R1))
        env = dict(os.environ)
        g, k = slot
        idx0 = self.a.idx0 + 12 * self.cards.index(g) + (12 // self.a.slots) * k
        env.update(GPU=str(g), IDX0=str(idx0), WORKERS=str(1 if u.ids else self.a.workers), CPUS=self.cpus[slot], SEEDS=str(u.seed),
                   ARMS=u.arm, LAT_EXEC="curv", RESUME_S="5", KEEP_SRV="1", SRV_NO_TWIN="1", DESIRE=desire, DRIVE_ARGS=args,
                   OP_ARB_DIR=str(ROOT / f"card{g}" if self.a.slots == 1 else ROOT / f"card{g}s{k}"), OP_ARB_ARMS=str(AD), OPENBLAS_CORETYPE="Haswell")
        if u.model != "base":
            env["SRV_ONNX"] = str(ONNX / f"{u.model}.onnx")
        else:
            env.pop("SRV_ONNX", None)
        if u.match:
            env["MATCH_ARM"] = u.match
        if u.ids:
            env["OPL_IDS"] = u.ids
        return env

    def n_routes(self, u):
        return len(u.ids.split(",")) if u.ids else (DEV_ROUTES if u.routes == "2" else 19)

    def run_unit(self, u, card):
        if u.arm == "dbaseslow":
            self.prep_slow(u)
        t0 = time.time()
        self.event("unit_start", unit=u.id, card=list(card), arm=u.arm, seed=u.seed, model=u.model)
        self.log.info(f"[slot {card}] start {u.id} ({u.model}, workers {self.env_for(u, card)['WORKERS']})")
        self.status(f"running: " + ", ".join(f"{x.id}@{x.card[0]}.{x.card[1]}" for x in self.units.values() if x.state == "run"))
        for attempt in (1, 2):
            u.tries = attempt
            arm = u.arm
            if arm == "dbaseslow":
                arm = u.slow_arm
            stale = AD / f"{u.tag}-{arm}-s{u.seed}" / "DONE"
            if stale.exists() and len(glob.glob(str(stale.parent / "done" / "*.json"))) != self.n_routes(u):
                stale.unlink()      # op_arb.sh writes DONE even when routes never finished; without this the retry is a no-op and the unit "fails twice (rc 0)"
            env = self.env_for(u, card)
            env["ARMS"] = arm
            with open(ROOT / f"unit-{u.id}.log", "a") as lf:
                rc = subprocess.run(["bash", str(REPO / "experiments/op_closed_loop/archive/op_arb.sh"), "set", u.routes, u.tag], env=env, cwd=REPO,
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            d = AD / f"{u.tag}-{arm}-s{u.seed}"
            n = len(glob.glob(str(d / "done" / "*.json")))
            if rc == 0 and (d / "DONE").exists() and n == self.n_routes(u):
                break
            self.log.info(f"[slot {card}] {u.id} attempt {attempt}: rc {rc}, DONE {(d / 'DONE').exists()}, routes {n}/{self.n_routes(u)}")
        else:
            self.fail(f"unit {u.id} failed twice (rc {rc}); see {ROOT}/unit-{u.id}.log and the slot's log.txt")
            return
        wall = (time.time() - t0) / 60
        self.report_unit(u, arm)
        with self.lock:
            u.state = "done"
            self.after(u, arm)
        self.log.info(f"[slot {card}] done {u.id} in {wall:.1f} min")
        self.event("unit_end", unit=u.id, card=list(card), wall_min=round(wall, 1))
        self.log.scalar("unit_wall_min", wall, len([x for x in self.units.values() if x.state == "done"]))

    # ---------------------------------------------------------------- pace-matched controls
    def prep_slow(self, u):
        """dbase link for the family tag, and which dbaseslow version this unit is (set by after())."""
        if not hasattr(u, "slow_arm"):
            u.slow_arm = "dbaseslow"
        link = AD / f"{u.tag}-dbase-s{u.seed}"
        if u.tag != "ld" and not link.exists() and u.tag[0] != "h":
            os.symlink(AD / f"ld-dbase-s{u.seed}", link)
        if u.tag[0] == "h" and not link.exists():
            os.symlink(AD / f"hld-dbase-s{u.seed}", link)

    def speeds(self, d):
        out = {}
        for f in glob.glob(str(d / "done" / "*.json")):
            rid = os.path.basename(f)[:-5]
            p = d / "attempts" / rid / str(json.load(open(f))["attempt"]) / "plans.jsonl"
            v = [r["v"] for r in map(json.loads, open(p)) if not r["warm"]]
            out[rid] = sum(v) / max(len(v), 1)
        return out

    def ratio(self, tag, arm, slow, seed):
        a, b = self.speeds(AD / f"{tag}-{arm}-s{seed}"), self.speeds(AD / f"{tag}-{slow}-s{seed}")
        k = [r for r in a if r in b]
        return sum(a[r] for r in k) / max(sum(b[r] for r in k), 1e-6)

    def after(self, u, arm):
        """A finished dbaseslow version: in the band [0.9, 1.1] -> the family's control is this one, else queue the next P iteration."""
        if u.arm != "dbaseslow":
            return
        r = self.ratio(u.tag, u.match, arm, u.seed)
        self.log.info(f"pacing {u.tag} s{u.seed}: v({u.match}) / v({arm}) = {r:.3f}")
        self.event("pacing", tag=u.tag, seed=u.seed, slow=arm, ratio=round(r, 3))
        n = 1 if arm == "dbaseslow" else int(arm[len("dbaseslow"):])
        (AD / f"{u.tag}-{arm}-s{u.seed}" / "PACE").write_text(f"{r:.4f}\n")
        if 0.9 <= r <= 1.1 or n >= 4:
            return
        nxt = "dbaseslow2" if n == 1 else f"dbaseslow{n + 1}"
        v = Unit(u.tag, "dbaseslow", u.seed, u.prio, "base", deps=[], routes=u.routes, ids=u.ids, match=u.match)
        v.id = f"{u.tag}-{nxt}-s{u.seed}"
        v.slow_arm = nxt
        self.units[v.id] = v

    def report_unit(self, u, arm):
        try:
            subprocess.run([str(REPO / ".venv/bin/python"), "-m", "experiments.op_adapt_l.lib.op_l_b2d_report", "unit", u.tag, arm, str(u.seed)], cwd=REPO,
                           env=dict(os.environ, OPL_ROOT=str(ROOT)), check=False, timeout=600,
                           stdout=open(ROOT / "report.log", "a"), stderr=subprocess.STDOUT)
        except Exception as e:  # noqa: BLE001 - a report failure never stops the runs
            self.log.info(f"report for {u.id} failed: {e}")

    # ---------------------------------------------------------------- util sampler
    def sampler(self):
        def cpu_busy():
            out = {}
            for line in open("/proc/stat"):
                if line.startswith("cpu") and line[3].isdigit():
                    f = line.split()
                    v = list(map(int, f[1:9]))
                    out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
            return out

        cores = self.cores
        prev = cpu_busy()
        path = ROOT / "util.csv"
        new = not path.exists()
        with open(path, "a", buffering=1) as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["t", "gpu", "util_pct", "mem_mib", "cores_busy", "n_carla"])
            while not self.stop:
                time.sleep(15)
                cur = cpu_busy()
                try:
                    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
                    g = {int(x.split(",")[0]): (int(x.split(",")[1]), int(x.split(",")[2])) for x in smi}
                    ncar = subprocess.run(["pgrep", "-c", "CarlaUE4-Linux"], capture_output=True, text=True).stdout.strip()
                except Exception:  # noqa: BLE001
                    continue
                for c in self.cards:
                    cs = cores(self.card_cpus[c])
                    busy = sum((cur[i][0] - prev[i][0]) / max(cur[i][1] - prev[i][1], 1) for i in cs if i in cur and i in prev)
                    w.writerow([int(time.time()), c, g.get(c, (0, 0))[0], g.get(c, (0, 0))[1], round(busy, 2), ncar])
                prev = cur

    def card_loop(self, card):
        while not self.stop:
            u = self.ready(card)
            if u is None:
                with self.lock:
                    left = [x for x in self.units.values() if x.state in ("new", "run")]
                if not left:
                    return
                if not any(x.state == "run" for x in left) and not self.stop:
                    # only units with unmet deps remain and nothing runs: a dependency cannot complete
                    self.fail("deadlock: " + ", ".join(x.id for x in left))
                    return
                time.sleep(20)
                continue
            self.run_unit(u, card)

    def run(self):
        self.build(self.a.plan)
        self.log.info(f"plan {self.a.plan}: {len(self.units)} initial units on cards {self.cards}")
        self.event("start", plan=self.a.plan, units=sorted(self.units))
        try:
            (ROOT / "ERROR").unlink()
        except FileNotFoundError:
            pass
        threading.Thread(target=self.sampler, daemon=True).start()
        ts = [threading.Thread(target=self.card_loop, args=(sl,)) for sl in self.slots]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.stop = True
        for g, k in self.slots:                              # stop only the openpilot servers this chain started (recorded PIDs)
            p = ROOT / (f"card{g}" if self.a.slots == 1 else f"card{g}s{k}") / "srv" / "op.pid"
            if p.exists():
                try:
                    pid = int(p.read_text())
                    os.killpg(pid, 15)
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
                p.unlink(missing_ok=True)
        if self.fatal:
            self.status("ERROR " + self.fatal)
            return 1
        (ROOT / (f"DONE-{self.a.plan}" if not self.a.only else f"DONE-resume-{self.a.only.replace(',', '+')}")).write_text(time.strftime("%F %T") + "\n")
        self.status(f"plan {self.a.plan} done")
        self.event("end", plan=self.a.plan)
        self.log.info("done")
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["stage1", "stage2", "full", "heldout"])
    ap.add_argument("--workers", type=int, default=6, help="CARLA workers per slot")
    ap.add_argument("--slots", type=int, default=1, help="slots (an openpilot server + its workers) per card; each takes an even share of the card's cores and index block")
    ap.add_argument("--only", default="", help="comma list of arms: keep only their units that are not complete (resume after a failed unit)")
    ap.add_argument("--cards", default="0,1,2")
    ap.add_argument("--cpus", default="8-29;30-51;52-73", help="core list per card, ';' separated")
    ap.add_argument("--idx0", type=int, default=200)
    sys.exit(Chain(ap.parse_args()).run())
