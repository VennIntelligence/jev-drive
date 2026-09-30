"""op-adapt r2 staged launch (§6 of todos/2026-09-29-op-adapt-r2-prereg.md): one self-advancing lane per stage with
STATUS / DONE / ERROR files, the §6 checklists evaluated by code into checklist.json, and a GPU packer that puts runs
on the lane's cards (default 1-4) while each card's total stays under --cap-gb (shared with WL-2 CARLA).

  stage1    A seed 0, lambda_s = 1, 2 000 steps -> the 1-unit checklist
  stage10   every §6 ~10-unit arm (A at both lambda_s, D, A-real, A-sim, A-noC, A-noK, D-only, A-bhv) at 10 % of the
            steps, packed on the cards -> the 10-unit checklist (items 1-7), any failure stops the batch
  full      lambda_s selection first (A seed 0 at 0.3 and 1, then `select`), then every arm x seed of §2 with the chosen
            lambda_s; Z is not run (R0-gated)
  check     re-evaluate a stage checklist from existing run dirs (no training)

Lane dir: R2/stage/<stage>/{STATUS, DONE, ERROR, checklist.json, jobs.json}; the runs are R2/stage/<stage>/runs/<tag>/
(stage 1 / 10) or R2/train-<tag>/ (full).
  scripts/tmux_run.sh r2-stage1 python scripts/op_adapt_r2_lane.py stage1 --cores 40-47
"""
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import op_adapt_r2 as R  # noqa: E402

PY = sys.executable
TRAIN = str(Path(__file__).resolve().parent / "op_adapt_r2_train.py")
EST_SEQ_PER_S = R.SEQ_PER_ARM / (1.1 * 3600)        # §7: 1.1 GPU·h per arm of 1.2 M labelled sequences
EST_VRAM_R1 = (8.3, 32)                              # round-1 bench: B fp16 peak 8.3 GB at batch 32 (scaled by batch size)


# ---------------------------------------------------------------- GPU packer
def parse_cores(spec: str) -> list:
    """'48-95,100' -> [48, ..., 95, 100]."""
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def gpu_used(gpus) -> dict:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode()
    u = {int(a): int(b) / 1024 for a, b in (l.split(",") for l in out.strip().splitlines())}
    return {g: u[g] for g in gpus}


class Packer:
    """Launch jobs [{tag, argv, vram_gb}] on the cards: a job starts on the card with the most room when
    (card used now) + (reserved by our jobs started < 3 min ago that have not reached their peak) + vram_gb <= cap."""

    def __init__(self, gpus, cap_gb, cores, lane: Path, log, per_card=2):
        self.gpus, self.cap, self.lane, self.log = gpus, cap_gb, lane, log
        self.running = {}
        # every job gets its own core slice: the range is split into per_card x cards slices (a run is GPU-bound, so two
        # runs per card is the packing that fits 2 x ~32 GB in 84 GB; the CPU side is dev eval bursts and 4 batch threads)
        allc = sorted(os.sched_getaffinity(0) & set(parse_cores(cores))) or sorted(os.sched_getaffinity(0))
        n = max(1, min(len(allc) // 8, per_card * len(gpus)))
        k = len(allc) // n
        self.slices = [allc[i * k:(i + 1) * k] for i in range(n)]
        self.free = list(range(n))

    def room(self):
        used = gpu_used(self.gpus)
        now = time.time()
        for j in self.running.values():
            if now - j["t0"] < 180:                                     # still allocating: count its estimate
                used[j["gpu"]] += j["vram_gb"]
        return {g: self.cap - u for g, u in used.items()}

    def run(self, jobs, poll=60):
        pending, done = list(jobs), {}
        while pending or self.running:
            for j in list(pending):
                room = self.room()
                g = max(room, key=room.get)
                if room[g] >= j["vram_gb"] and self.free:
                    sl = self.free.pop(0)
                    cs = ",".join(map(str, self.slices[sl]))
                    nc = len(self.slices[sl])
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="2", OPENBLAS_CORETYPE="Haswell",
                               R2_SCORE_WORKERS=str(max(1, nc // 3)), R2_GATHER_THREADS=str(max(1, nc // 3)))
                    lf = open(self.lane / "logs" / f"{j['tag']}.log", "a")
                    p = subprocess.Popen(["taskset", "-c", cs, *j["argv"]], env=env, stdout=lf, stderr=subprocess.STDOUT)
                    self.running[j["tag"]] = j | {"p": p, "gpu": g, "t0": time.time(), "slice": sl}
                    pending.remove(j)
                    self.log(f"start {j['tag']} on GPU {g} cores {self.slices[sl][0]}-{self.slices[sl][-1]} "
                             f"(room {room[g]:.1f} GB, needs {j['vram_gb']:.1f}), pid {p.pid}")
                    time.sleep(20)
            for tag, j in list(self.running.items()):
                rc = j["p"].poll()
                if rc is not None:
                    done[tag] = {"rc": rc, "gpu": j["gpu"], "wall_s": time.time() - j["t0"]}
                    self.free.append(j["slice"])
                    del self.running[tag]
                    self.log(f"end {tag}: rc {rc}, {done[tag]['wall_s'] / 60:.1f} min")
            write(self.lane / "STATUS", {"phase": "training", "pending": [j["tag"] for j in pending],
                                         "running": {t: j["gpu"] for t, j in self.running.items()}, "done": done})
            if pending or self.running:
                time.sleep(poll)
        return done


def write(p: Path, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(p)


def train_argv(arm, seed, lam, steps, run_dir, extra=()):
    return [PY, TRAIN, "train", "--arm", arm, "--seed", str(seed), "--lam-s", str(lam), "--max-steps", str(steps),
            "--run-dir", str(run_dir), *extra]


def est_vram(batch=R.BATCH):
    """Planning estimate of one run's card memory: round-1 bench scaled to the r2 batch (labelled + distillation
    stream + twins); replaced by the 1-unit measurement once it exists."""
    n = batch * (1 + 0.25 + 0.1)
    return EST_VRAM_R1[0] * n / EST_VRAM_R1[1]


# ---------------------------------------------------------------- checklists
def events(run: Path):
    return [json.loads(l) for l in open(run / "events.jsonl")] if (run / "events.jsonl").exists() else []


def loss_curve(ev, tag):
    v = [(e["step"], e["value"]) for e in ev if e["kind"] == "scalar" and e["tag"] == tag]
    return np.array(v) if v else np.zeros((0, 2))


def item(name, ok, value=None, line=None, note=""):
    return {"item": name, "pass": None if ok is None else bool(ok), "value": value, "line": line, "note": note}


def _dev(run):
    p = run / "dev.json"
    return json.loads(p.read_text()) if p.exists() else {}


def check_stage1(run: Path) -> list:
    """§6 1-unit checklist on the A seed-0 run."""
    ev, dv = events(run), _dev(run)
    out = []
    fin = all(np.isfinite(e["value"]) for e in ev if e["kind"] == "scalar" and e["tag"].startswith("loss/"))
    trend = {}
    for t in ("loss/total", "loss/aux", "loss/score", "loss/pair", "loss/dir", "loss/distill"):
        c = loss_curve(ev, t)
        if len(c) >= 4:
            k = max(1, len(c) // 5)
            trend[t] = (float(c[:k, 1].mean()), float(c[-k:, 1].mean()))
    dec = all(trend[t][1] < trend[t][0] for t in ("loss/total", "loss/aux") if t in trend) and "loss/total" in trend
    out.append(item("1 losses finite, total and L_aux decreasing", fin and dec and not (run / "ERROR").exists(), trend,
                    "finite; last 20 % < first 20 %",
                    "L_distill starts at 0 by construction (student = teacher) and can only grow; its trend is reported, not gated"))
    out.append(item("2 dev drift median <= 0.10 m", dv.get("drift_median", np.inf) <= 0.10, dv.get("drift_median"), 0.10))
    aucs = {k: (dv.get(f"aux_auc_{k}"), dv.get(f"aux_auc_{k}_orig_probe")) for k in ("simC", "simK", "nus", "wod")}
    ok = [a > o for a, o in aucs.values() if a is not None and o is not None and np.isfinite(a) and np.isfinite(o)]
    out.append(item("3 dev L_aux AUC above O in sim and real", (all(ok) and len(ok) == 4) if ok else None, aucs,
                    "aux head AUC > O temporal probe AUC (simC, simK, nus, wod)"))
    sps, vram = dv.get("labelled_seq_per_s"), dv.get("peak_reserved_gb")
    ev_v = est_vram()
    out.append(item("4 throughput and memory within +-30 % of §7", None if sps is None else
                    (abs(sps / EST_SEQ_PER_S - 1) <= 0.3 and abs(vram / ev_v - 1) <= 0.3),
                    {"labelled_seq_per_s": sps, "est": EST_SEQ_PER_S, "peak_gb": vram, "est_gb": ev_v}, "+-30 %",
                    "§7 gives GPU·h only; memory estimate = round-1 bench (8.3 GB at batch 32) scaled to the r2 batch"))
    ff = run / "full_forward.json"
    out.append(item("5 cache -> stage 4 -> policy vs full port forward, temporal corr >= 0.9999 on 3 pairs",
                    json.loads(ff.read_text())["pass"] if ff.exists() else None,
                    json.loads(ff.read_text()) if ff.exists() else None, 0.9999, "needs package C's frame loader"))
    s0 = [e for e in ev if e["kind"] == "score0"]
    out.append(item("6 step-0 L_score = 0 where op is in Top", (s0[0]["max_loss_on_top"] <= 1e-3) if s0 else None,
                    s0[0] if s0 else None, "max <= 1e-3 (fp16 batch-shape noise; exact value reported)"))
    return out


def check_stage10(runs: dict) -> list:
    """§6 10-unit checklist, items 1-7, per arm."""
    out = []
    for tag, run in runs.items():
        arm = R.ARMS[tag.split("@")[0]]
        dv, ev = _dev(run), events(run)
        A = lambda n, ok, v=None, line=None, note="": out.append(item(f"{tag}: {n}", ok, v, line, note))  # noqa: E731
        A("1 completed, no NaN / OOM restart", (run / "DONE").exists() and not (run / "ERROR").exists()
          and dv.get("nonfinite", 1) == 0 and not any(e["kind"] == "resume" for e in ev), dv.get("steps"))
        A("2 dev drift median <= 0.15 m", dv.get("drift_median", np.inf) <= 0.15, dv.get("drift_median"), 0.15)
        if "pair" in arm.losses and arm.sim:
            pa = [dv.get(f"pair_acc_sim{r}") for r in arm.sim]
            A("3a sim L_pair pair accuracy > 0.6", all(p is not None and p > 0.6 for p in pa), pa, 0.6)
        if arm.real and "aux" in arm.losses:
            r = {k: (dv.get(f"aux_auc_{k}"), dv.get(f"aux_auc_{k}_orig_probe")) for k in ("nus", "wod")}
            A("3b real L_aux AUC >= O", all(a is not None and o is not None and a >= o for a, o in r.values()), r)
        if arm.det:
            A("4 adapter gate |alpha| > 0", dv.get("det_gate_abs", 0) > 0, dv.get("det_gate_abs"))
        elif arm.name == "A-real":
            A("4 A-real has no gate (structure)", "det_gate_abs" not in dv, None)
        if "score" in arm.losses or "dir" in arm.losses:
            sims = [f"sim{r}" for r in arm.sim]
            b = [dv.get(f"sjev_better_notop_{s}") for s in sims] + ([dv.get("sjev_better_notop_nus")] if "nus" in arm.score_domains and "sjev_better_notop_nus" in dv else [])
            b = [x for x in b if x is not None]
            vio = [(dv.get(f"dir_violation_{s}_adapt"), dv.get(f"dir_violation_{s}_orig")) for s in sims]
            ok_b = all(x > 0.5 for x in b) if b else None
            ok_v = all(a < o for a, o in vio if a is not None) if any(a is not None for a, _ in vio) else None
            A("5 S_jev better than O on op-not-in-Top dev x+ (> 0.5) and fewer v+ > v- violations than O",
              None if ok_b is None and ok_v is None else (ok_b is not False and ok_v is not False), {"better": b, "violation": vio})
        if "dplan" in arm.losses:
            s = [dv.get(f"dplan_sign_agree_sim{r}") for r in arm.sim]
            A("5 (A-bhv) dev pair dv-hat vs dv* sign agreement > 0.5", all(x is not None and x > 0.5 for x in s), s, 0.5)
        A("6a native speed distribution on dev normal frames, KS < 0.1", dv.get("ks_v2_normal", 1) < 0.1, dv.get("ks_v2_normal"), 0.1)
        xs = [(dv.get(f"xminus_slow_nearest_sim{r}_adapt"), dv.get(f"xminus_slow_nearest_sim{r}_orig")) for r in (arm.sim or "")]
        xs = [x for x in xs if x[0] is not None]
        if xs:
            A("6b dev x- nearest-Top slow candidates <= O + 5 pp", all(a <= o + 0.05 for a, o in xs), xs)
        if arm.offset:
            f = dv.get("off_train_op_fail")
            A("7a offset train: op fails DAC or DDC in 5-60 %", f is not None and 0.05 <= f <= 0.60, f, "[0.05, 0.60]",
              "< 5 %: the set is empty of signal; > 60 %: warp or offsets broke the image -> stop and report to main")
            A("7b offset dev: DAC*DDC pass above O", dv.get("off_dev_pass_adapt", -1) > dv.get("off_dev_pass_orig", 2),
              (dv.get("off_dev_pass_adapt"), dv.get("off_dev_pass_orig")))
            A("7c twin frames drift <= ordinary navtrain dev drift", dv.get("twin_drift_median", np.inf) <= dv.get("drift_nav_median", -1),
              (dv.get("twin_drift_median"), dv.get("drift_nav_median")))
    return out


def verdict(items) -> dict:
    fails = [i["item"] for i in items if i["pass"] is False]
    pend = [i["item"] for i in items if i["pass"] is None]
    return {"pass": not fails and not pend, "failed": fails, "pending": pend, "items": items}


# ---------------------------------------------------------------- data preparation (self-advancing, idempotent)
DOMAINS = ("simC", "simK", "nus", "wod", "nav", "p5")          # v5: no "off"


def det_ready(dn) -> bool:
    """Every cache file of the domain has package D's token file."""
    try:
        from jevdrive import op_adapt_det as DT
    except ImportError:
        return False
    import pandas as pd
    src = R.r2("t") / "trunk" / f"{dn}.src.parquet"
    return src.exists() and all(DT.tok_path(f).exists() for f in pd.read_parquet(src, columns=["file"]).file.unique())


def prep(a, d, log, det=False):
    """pack (mapped caches) every domain whose package-C index is newer than its sample table, lay D's tokens on
    them when `det`, and run the teacher on every domain without one (one GPU job)."""
    t = R.r2("t")
    todo = []
    for dn in DOMAINS:
        ix = R.r2("index") / f"{dn}.parquet"
        sm = t / "samples" / f"{dn}.parquet"
        if not ix.exists():
            log(f"prep: no package-C index for {dn}")
            continue
        if not sm.exists() or sm.stat().st_mtime < ix.stat().st_mtime:
            subprocess.check_call([PY, TRAIN, "pack", "--domains", dn, "--source", "c"])
            log(f"prep: packed {dn}")
        if det and dn != "p5" and not (t / "det" / f"{dn}.tok.npy").exists():
            if not det_ready(dn):
                raise RuntimeError(f"package D tokens missing for {dn}")
            subprocess.check_call([PY, "-c", f"from jevdrive import op_adapt_r2 as R; R.pack_det('{dn}')"])
            log(f"prep: detection tokens laid on {dn}")
        if not (t / "teacher" / dn / "uid.npy").exists():
            todo.append(dn)
    if todo:
        Packer(a.gpus, a.cap_gb, a.cores, d, log).run(
            [{"tag": "teacher", "argv": [PY, TRAIN, "teacher", "--domains", *todo], "vram_gb": 14.0}])
        missing = [dn for dn in todo if not (t / "teacher" / dn / "uid.npy").exists()]
        if missing:
            raise RuntimeError(f"teacher failed for {missing} (see logs/teacher.log)")
        log(f"prep: teacher on {todo}")


# ---------------------------------------------------------------- stages
def lane(stage, a, body):
    d = R.r2("stage", stage)
    (d / "logs").mkdir(exist_ok=True)
    for f in ("DONE", "ERROR"):
        (d / f).unlink(missing_ok=True)
    logf = open(d / "lane.log", "a", buffering=1)

    def log(m):
        line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
    try:
        write(d / "STATUS", {"phase": "start"})
        res = body(d, log)
        write(d / "checklist.json", res)
        if res.get("pass"):
            (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            log("checklist passed")
        else:
            (d / "ERROR").write_text(json.dumps({"failed": res.get("failed"), "pending": res.get("pending")}, indent=1))
            log(f"checklist not passed: failed {res.get('failed')}, pending {res.get('pending')}")
        write(d / "STATUS", {"phase": "end", "pass": res.get("pass")})
    except BaseException as e:  # noqa: BLE001
        (d / "ERROR").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        log(f"ERROR {e}")
        raise


def vram_from_stage1(default):
    p = R.r2("stage", "stage1") / "runs" / "A-s0-ls1" / "dev.json"
    return json.loads(p.read_text())["peak_reserved_gb"] + 1.5 if p.exists() else default


def prep_stage(a):
    """pack every domain's mapped cache and run the teacher where missing (needed before any readout, M1 included)."""
    def body(d, log):
        prep(a, d, log)
        return {"pass": True, "failed": [], "pending": []}
    lane("prep", a, body)


def stage1(a):
    def body(d, log):
        prep(a, d, log)
        run = d / "runs" / "A-s0-ls1"
        pk = Packer(a.gpus, a.cap_gb, a.cores, d, log)
        pk.run([{"tag": "A-s0-ls1", "argv": train_argv("A", 0, 1.0, a.steps or 2000, run, a.extra), "vram_gb": a.vram_gb or est_vram() + 4}])
        if a.full_forward:
            subprocess.call(["taskset", "-c", a.cores, PY, str(Path(__file__).parent / "op_adapt_r2_readout.py"), "fullfwd",
                             "--out", str(run / "full_forward.json")], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpus[0])))
        return verdict(check_stage1(run))
    lane("stage1", a, body)


def stage10(a):
    arms = a.arms or list(R.STAGE10)
    steps = a.steps or int(np.ceil(0.1 * R.RunCfg(arm="A").steps))

    def body(d, log):
        prep(a, d, log, det=any(R.ARMS[x.split("@")[0]].det for x in arms))
        runs, jobs = {}, []
        vr = a.vram_gb or vram_from_stage1(est_vram() + 4)
        for spec in arms:
            name, lam = (spec.split("@") + ["1"])[:2]
            tag = spec if "@" in spec else name
            run = d / "runs" / tag
            runs[tag] = run
            jobs.append({"tag": tag, "argv": train_argv(name, 0, float(lam), steps, run, a.extra), "vram_gb": vr})
        write(d / "jobs.json", {"steps": steps, "jobs": [{k: v for k, v in j.items()} for j in jobs]})
        Packer(a.gpus, a.cap_gb, a.cores, d, log).run(jobs)
        return verdict(check_stage10(runs))
    lane("stage10", a, body)


def full(a):
    def body(d, log):
        prep(a, d, log, det=True)
        pk = Packer(a.gpus, a.cap_gb, a.cores, d, log)
        vr = a.vram_gb or vram_from_stage1(est_vram() + 4)
        sel = R.r2() / "lambda_s.json"
        if not sel.exists():
            runs = [R.r2(f"train-A-s0-ls{lam:g}") for lam in R.LAM_S_GRID]
            pk.run([{"tag": f"A-s0-ls{lam:g}", "argv": train_argv("A", 0, lam, 0, r, a.extra), "vram_gb": vr}
                    for lam, r in zip(R.LAM_S_GRID, runs)])
            subprocess.check_call([PY, TRAIN, "select", "--runs", *map(str, runs)])
        lam = json.loads(sel.read_text())["lam_s"]
        log(f"lambda_s = {lam}")
        jobs = []
        for name, arm in R.ARMS.items():
            if name in ("O", "Z"):
                continue
            for s in range(arm.seeds):
                if name == "A" and s == 0:
                    continue                                  # the selected A seed-0 run is already trained
                cfg = R.RunCfg(arm=name, seed=s, lam_s=lam)
                jobs.append({"tag": cfg.tag, "argv": train_argv(name, s, lam, 0, R.r2(f"train-{cfg.tag}"), a.extra), "vram_gb": vr})
        done = pk.run(jobs)
        bad = [t for t, r in done.items() if r["rc"] != 0]
        return {"pass": not bad, "failed": bad, "pending": [], "lambda_s": lam, "runs": list(done)}
    lane("full", a, body)


def check(a):
    d = R.r2("stage", a.stage)
    if a.stage == "stage1":
        res = verdict(check_stage1(d / "runs" / "A-s0-ls1"))
    else:
        runs = {p.name: p for p in sorted((d / "runs").iterdir()) if p.is_dir()}
        res = verdict(check_stage10(runs))
    write(d / "checklist.json", res)
    for i in res["items"]:
        print(f"[{ {True: 'PASS', False: 'FAIL', None: 'PEND'}[i['pass']] }] {i['item']}: {i['value']}")
    print("PASS" if res["pass"] else f"NOT PASSED failed={res['failed']} pending={res['pending']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prep", "stage1", "stage10", "full", "check"))
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3, 4])
    ap.add_argument("--cap-gb", type=float, default=76.0, help="card total must stay below 78 GB")
    ap.add_argument("--cores", default="40-47")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--vram-gb", type=float, default=0.0)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--check-stage", dest="stage_name", default="stage1")
    ap.add_argument("--full-forward", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed through to the train command")
    a = ap.parse_args()
    if a.stage == "check":
        a.stage = a.stage_name
        return check(a)
    {"prep": prep_stage, "stage1": stage1, "stage10": stage10, "full": full}[a.stage](a)


if __name__ == "__main__":
    main()
