#!/usr/bin/env python3
"""op-adapt L: the whole queue as ONE self-advancing chain (prereg sections 4 and 7).

  wave 1   the four selection candidates (seed 0) + the adapter-only run + the original model's readout pass (6 jobs on 6 slots)
  select   `op_adapt_l_train.py select` -> $L/selection.json (main = the registered choice); runs/main-s0 -> the selected s0 run
  wave 2   the ablation queue in the registered order; every job = train -> eval (WOD val / nuScenes val / rater) -> read
  wave 3   navtest PDMS of main s0-2 and noint s0 (one pass, port O in the same run)

Slots: 2 per card on the lane's cards (a stage-4 run is GPU-bound at about 32 GB, so two per card overlap one run's CPU-side dev
eval with the other's training) x an equal share of the lane's cores. Idempotent: each step leaves a marker (runs/<tag>/DONE,
readout/<tag>/eval/rater.npz, readout/<tag>/metrics.csv), so a re-run of the same command after a restart resumes.
Files in $L/chain/: STATUS (json, current), events.jsonl, log.txt, logs/<job>.<step>.log, ERROR (a stop line), DONE.
A single failed job is recorded (failed/<job>) and the queue goes on; three failures stop the chain (ERROR).

  scripts/tmux_run.sh L-chain python3 experiments/op_adapt_l/scripts/op_adapt_l_chain.py --gpus 0,1,2 --cores 8-74
Stop only by exact PID. Never pkill -f.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402

DATA = Path(os.environ["DATA_DIR"])
PY = str(DATA / "envs" / "op-train" / "bin" / "python")
ROOT = L.lroot()
C = L.lroot("chain")
(C / "logs").mkdir(exist_ok=True)
(C / "failed").mkdir(exist_ok=True)
LOCK = threading.Lock()
STATE = {"phase": "start", "running": {}, "done": [], "failed": []}


def log(m):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {m}"
    print(line, flush=True)
    with LOCK:
        with open(C / "log.txt", "a") as f:
            f.write(line + "\n")


def ev(kind, **kw):
    with LOCK:
        with open(C / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **kw}) + "\n")


def status():
    with LOCK:
        (C / "STATUS").write_text(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **STATE}, indent=1))


def parse_cores(spec: str) -> list:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def sh(job, step, argv, gpu, cores):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_CORETYPE="Haswell", OMP_NUM_THREADS="2")
    cmd = ["taskset", "-c", ",".join(map(str, cores))] + argv
    with open(C / "logs" / f"{job}.{step}.log", "a") as f:
        f.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode


def tag(job):
    return f"{job['arm']}-s{job['seed']}"


def run_job(job, gpu, cores):
    """train -> eval -> read for one arm; markers make every step skippable."""
    t = tag(job)
    steps = []
    if not (ROOT / "runs" / t / "DONE").exists():
        a = [PY, "experiments/op_adapt_l/scripts/op_adapt_l_train.py", "train", "--arm", job["arm"], "--seed", str(job["seed"])]
        if job.get("steps"):
            a += ["--steps", str(job["steps"])]
        steps.append(("train", a))
    if not (ROOT / "readout" / t / "eval" / "rater.npz").exists():
        steps.append(("eval", [PY, "experiments/op_adapt_l/scripts/op_adapt_l_readout.py", "eval", "--model", t]))
    if not (ROOT / "readout" / t / "metrics.csv").exists() and t != "O":
        steps.append(("read", [PY, "experiments/op_adapt_l/scripts/op_adapt_l_readout.py", "read", "--model", t]))
    for step, argv in steps:
        with LOCK:
            STATE["running"][t] = {"step": step, "gpu": gpu, "since": time.strftime("%H:%M:%S")}
        status()
        ev("start", job=t, step=step, gpu=gpu)
        t0 = time.time()
        rc = sh(t, step, argv, gpu, cores)
        ev("end", job=t, step=step, rc=rc, wall_s=round(time.time() - t0))
        log(f"{t} {step} rc={rc} {time.time() - t0:.0f}s")
        if rc != 0:
            (C / "failed" / t).write_text(f"{step} rc={rc} {time.strftime('%F %T')}\n")
            with LOCK:
                STATE["running"].pop(t, None)
                STATE["failed"].append(t)
            status()
            return False
    with LOCK:
        STATE["running"].pop(t, None)
        STATE["done"].append(t)
    status()
    return True


def run_phase(name, jobs, slots):
    """Run the jobs on the slots (one thread per slot pulls the next job); returns when the queue is empty."""
    STATE["phase"] = name
    status()
    log(f"phase {name}: {len(jobs)} jobs on {len(slots)} slots")
    it = iter(list(jobs))
    lk = threading.Lock()

    def worker(slot):
        gpu, cores = slot
        while True:
            with lk:
                job = next(it, None)
            if job is None:
                return
            if len(STATE["failed"]) >= 3:
                return
            if job.get("fn"):
                job["fn"](gpu, cores)
            else:
                run_job(job, gpu, cores)
    th = [threading.Thread(target=worker, args=(s,), daemon=True) for s in slots]
    for i, t in enumerate(th):
        t.start()
        time.sleep(20)                                   # stagger the model loads
    for t in th:
        t.join()
    if len(STATE["failed"]) >= 3:
        (C / "ERROR").write_text("three jobs failed: " + ", ".join(STATE["failed"]) + "\n")
        ev("error", failed=STATE["failed"])
        log("ERROR: three jobs failed, chain stops")
        sys.exit(1)


def o_eval(gpu, cores):
    t = "O"
    if (ROOT / "readout" / t / "eval" / "rater.npz").exists():
        return
    with LOCK:
        STATE["running"][t] = {"step": "eval", "gpu": gpu, "since": time.strftime("%H:%M:%S")}
    status()
    rc = sh(t, "eval", [PY, "experiments/op_adapt_l/scripts/op_adapt_l_readout.py", "eval", "--model", "O"], gpu, cores)
    ev("end", job=t, step="eval", rc=rc)
    log(f"O eval rc={rc}")
    with LOCK:
        STATE["running"].pop(t, None)
    if rc != 0:
        (C / "failed" / t).write_text("eval failed\n")
        STATE["failed"].append(t)


def gpu_monitor(gpus, every=20, window=30):
    """Sample nvidia-smi utilisation of the lane's cards into chain/gpu_util.csv; when a card's 10-minute mean drops below 70% while
    jobs are running, log it (and put it in STATUS) so a stalled slot is visible."""
    hist = {g: [] for g in gpus}
    last_warn = {g: 0.0 for g in gpus}
    with open(C / "gpu_util.csv", "a") as f:
        while True:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                              text=True)
                t = time.strftime("%F %T")
                for line in out.strip().splitlines():
                    i, u, m = (int(x) for x in line.split(","))
                    if i in hist:
                        hist[i] = (hist[i] + [u])[-window:]
                        f.write(f"{t},{i},{u},{m}\n")
                f.flush()
                low = {g: round(sum(h) / len(h)) for g, h in hist.items() if len(h) >= window and sum(h) / len(h) < 70}
                with LOCK:
                    STATE["gpu_util_10min"] = {g: round(sum(h) / max(len(h), 1)) for g, h in hist.items()}
                    STATE["gpu_low"] = low
                for g, u in low.items():
                    if STATE["running"] and time.time() - last_warn[g] > 900:
                        last_warn[g] = time.time()
                        log(f"GPU {g} 10-min mean utilisation {u}% (< 70%) with {len(STATE['running'])} jobs running")
            except Exception as e:  # noqa: BLE001
                log(f"gpu monitor: {e}")
            time.sleep(every)


def selection_checklist() -> dict:
    """Prereg section 7, 'about 10 units': completion (DONE, no non-finite loss), final dev drift median <= 0.15 m, speed distribution
    KS < 0.1, at least one qualifying candidate. A missing / crashed / non-finite run is a bug-type failure and stops the chain;
    the substance items (drift, KS, no qualifying candidate) are logged and the registered ablation queue still runs."""
    from experiments.op_adapt_l.lib import op_adapt_l_arms as ARMS
    out = {"runs": {}, "bug": [], "substance": []}
    for name in list(ARMS.SELECTION) + ["tr_ad"]:
        run = ROOT / "runs" / f"{name}-s0"
        dj = run / "dev.json"
        if not (run / "DONE").exists() or not dj.exists():
            out["bug"].append(f"{name}: not completed")
            continue
        d = json.loads(dj.read_text())
        row = {"nonfinite": d.get("nonfinite"), "drift_median": d["drift_median"], "drift_p95": d["drift_p95"], "ks_v2": d.get("ks_v2_other"),
               "qualifies": d["select"]["qualifies"], "cap_gain": d["select"]["cap_gain"], "seq_per_s": d.get("seq_per_s"),
               "peak_gb": d.get("peak_reserved_gb"), "data_wait_frac": d.get("data_wait_frac")}
        out["runs"][name] = row
        if d.get("nonfinite"):
            out["bug"].append(f"{name}: {d['nonfinite']} non-finite losses")
        if d["drift_median"] > 0.15:
            out["substance"].append(f"{name}: drift median {d['drift_median']:.3f} > 0.15")
        if d.get("ks_v2_other", 0) >= 0.1:
            out["substance"].append(f"{name}: KS {d['ks_v2_other']:.3f} >= 0.1")
    if not any(v["qualifies"] for v in out["runs"].values() if v):
        out["substance"].append("no candidate qualifies under the registered dev lines")
    out["pass"] = not out["bug"]
    (C / "selection_checklist.json").write_text(json.dumps(out, indent=1))
    return out


def alias_main(sel: str):
    for sub in ("runs", "readout"):
        src, dst = ROOT / sub / f"{sel}-s0", ROOT / sub / "main-s0"
        if src.exists() and not dst.exists():
            dst.symlink_to(src.name)


def navtest(cores_spec):
    def f(gpu, cores):
        models = [m for m in ("main-s0", "main-s1", "main-s2", "noint-s0") if (ROOT / "runs" / m / "ckpt-final.pt").exists()
                  and not (ROOT / "readout" / m / "navtest.json").exists()]
        if not models:
            return
        STATE["running"]["navtest"] = {"step": "navtest", "gpu": gpu, "since": time.strftime("%H:%M:%S")}
        status()
        rc = sh("navtest", "navtest", [PY, "experiments/op_adapt_l/scripts/op_adapt_l_readout.py", "navtest", "--models", *models, "--cpus", cores_spec], gpu, cores)
        ev("end", job="navtest", step="navtest", rc=rc)
        log(f"navtest {models} rc={rc}")
        STATE["running"].pop("navtest", None)
        if rc != 0:
            (C / "failed" / "navtest").write_text("failed\n")
            STATE["failed"].append("navtest")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--cores", default="8-74")
    ap.add_argument("--per-card", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0, help="override the steps of every job (tests)")
    ap.add_argument("--seed2", action="store_true", help="also queue seed 2 of the single-seed ablations (a second pass after the queue)")
    a = ap.parse_args()
    (C / "ERROR").unlink(missing_ok=True)
    (C / "DONE").unlink(missing_ok=True)
    gpus = [int(g) for g in a.gpus.split(",")]
    cores = parse_cores(a.cores)
    n = len(gpus) * a.per_card
    k = len(cores) // n
    slots = [(gpus[i // a.per_card], cores[i * k:(i + 1) * k]) for i in range(n)]
    threading.Thread(target=gpu_monitor, args=(gpus,), daemon=True).start()
    log(f"chain start: slots {[(g, f'{c[0]}-{c[-1]}') for g, c in slots]}")
    J = lambda arm, seed=0, steps=0: {"arm": arm, "seed": seed, "steps": a.steps or steps}  # noqa: E731

    # longest first (stage-4 runs), so the tail of the wave is the cheap frozen-stage-4 runs
    o_eval(gpus[0], cores[:12])                          # the original model's readout pass first: every `read` needs it
    if STATE["failed"]:
        (C / "ERROR").write_text("O readout pass failed\n")
        sys.exit(1)
    wave1 = [J(x) for x in ("sel_s4ia", "sel_s4ia_dw3", "sel_s4polia", "sel_s4polia_dw3")] + \
        [J(x) for x in ("sel_polia", "sel_polia_dw3", "sel_polid", "sel_polid_dw3", "tr_ad")]
    run_phase("wave1 (selection candidates + adapter-only + O readout)", wave1, slots)
    ck = selection_checklist()
    log(f"selection checklist: bug {ck['bug']}; substance {ck['substance']}")
    ev("selection_checklist", **{k: ck[k] for k in ("bug", "substance", "pass")})
    if not ck["pass"]:
        (C / "ERROR").write_text("selection-wave checklist failed (bug-type): " + "; ".join(ck["bug"]) + "\n")
        sys.exit(1)
    while (C / "PAUSE").exists():                        # a manual look before the full queue: touch chain/PAUSE, rm it to go on
        STATE["phase"] = "paused before wave 2"
        status()
        time.sleep(30)
    # the registered selection
    if not (ROOT / "selection.json").exists():
        rc = sh("select", "select", [PY, "experiments/op_adapt_l/scripts/op_adapt_l_train.py", "select"], gpus[0], cores[:4])
        if rc != 0:
            (C / "ERROR").write_text("selection failed\n")
            sys.exit(1)
    sel = json.loads((ROOT / "selection.json").read_text())["config"]
    alias_main(sel)
    log(f"selection: main = {sel} ({json.loads((ROOT / 'selection.json').read_text())['rule']})")
    ev("selection", config=sel)
    from experiments.op_adapt_l.lib import op_adapt_l_arms as ARMS
    dws = [x for x in ("dw03", "dw1", "dw3", "dw10") if not ARMS.same_as_main(x)]
    wave2 = [J("main", 1), J("main", 2), J("tr_ad_dw3"), J("noint", 0), J("only_start"), J("only_stop"), J("only_turn")] + [J(x) for x in dws] + \
        [J("nocontrast"), J("stayheavy"), J("noint", 1), J("noint", 2)] + \
        [J(x, 1) for x in ["only_start", "only_stop", "only_turn"] + dws + ["nocontrast", "stayheavy"]] + [J("long", 0)]
    if a.seed2:
        wave2 += [J(x, 2) for x in ["only_start", "only_stop", "only_turn"] + dws + ["nocontrast", "stayheavy"]]
    run_phase("wave2 (main seeds, intent, per-slice, distillation, contrast)", wave2, slots)
    run_phase("wave3 (navtest PDMS)", [{"fn": navtest("8-40")}], slots[:1])
    STATE["phase"] = "end"
    status()
    (C / "DONE").write_text(time.strftime("%F %T\n"))
    ev("end", failed=STATE["failed"])
    log(f"chain done; failed: {STATE['failed']}")


if __name__ == "__main__":
    main()
