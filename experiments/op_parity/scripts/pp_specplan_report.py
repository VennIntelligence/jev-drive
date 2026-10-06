"""spec_plan run: P2 / P2+hinge under the HUGSIM `spec_plan` preset (lateral curvature from the model's own plan) vs the stored `spec` and `exam` runs.

  extract  (box, envs/hugsim)  per-run behaviour of the 12 (arm, preset) sets on the 64 scenarios -> results/hugsim_specplan/extract.csv,
                               traces.json.gz (per-step heading rate, speed, requested curvature)
  report   (Mac)               tables.md, the CSVs behind them, figures figs/*.png
Harness, scoring and classes as results/hugsim_full.md (spin = heading error >= 60 deg vs the recorded route; class = spin, else the runner's end).
Oscillation: heading rate w_i = (theta_{i+1} - theta_i) in deg per 0.25 s step; a flip is a sign change between consecutive moving steps
(v > 1 m/s) whose |w| both exceed FLIP_DEG; a run oscillates when it has >= 8 flips and >= 15% of its moving steps are flips.
Turning scenario = recorded route yaw range >= 30 deg (as experiments/op_probe/scripts/opj_turn_gain.py: 23 scenarios).
"""
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/op_parity/results/hugsim_specplan"
ARMS = ["P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1"]
PRE = {"exam": "pp-", "spec": "pp-spec-", "specplan": "pp-specplan-"}
FLIP_DEG, MOVING = 0.3, 1.0
CLS = ["complete", "spin", "bg_coll", "fg_coll", "off_route", "stuck"]


def osc(w, v):
    m = (v[:-1] > MOVING) & (v[1:] > MOVING) if len(v) > 1 else np.zeros(0, bool)
    w = np.asarray(w)
    sig = np.abs(w) > FLIP_DEG
    pair = m[: len(w) - 1] & sig[:-1] & sig[1:] if len(w) > 1 else np.zeros(0, bool)
    flips = int((np.sign(w[:-1]) * np.sign(w[1:]) < 0)[pair].sum()) if len(w) > 1 else 0
    nm = int(max(1, (v > MOVING).sum()))
    return flips, flips / nm


def extract():
    sys.path.insert(0, str(REPO / "experiments/hugsim/scripts"))
    import spin_analysis as SA
    D = Path(os.environ["DATA_DIR"])
    OUT.mkdir(parents=True, exist_ok=True)
    scen = {Path(p).stem for p in open(REPO / "experiments/hugsim/scripts/derot_all64.txt").read().split()}
    routes = json.load(open(D / "runs/op_parity/hugsim/routes.json"))
    turn = {k: float(np.degrees(np.ptp(np.unwrap(np.asarray(v["yaw"], float))))) for k, v in routes.items()}
    want = {pfx + a: (pr, a) for a in ARMS for pr, pfx in PRE.items()}
    last = {}
    for r in csv.DictReader(open(D / "runs/op_parity/hugsim/results.csv")):
        if r["tag"] in want and r["scenario"] in scen and r["end"] != "crash":
            last[(r["tag"], r["scenario"])] = r
    rows, traces = [], {}
    for (tag, sc), r in sorted(last.items()):
        d = Path(r["run_dir"])
        pos, th, v, steer, plans = SA.load_run(d, r["agent"])
        res, _ = SA.analyse(pos, th, v, steer, plans, routes[r["scene"]])
        w = np.degrees(np.diff(th))
        fl, fr = osc(w, v)
        kap = []
        zf = d / "zs_steps.jsonl"
        if zf.exists():
            kap = [x.get("kappa") for x in (json.loads(l) for l in open(zf)) if "kappa" in x]
        rows.append(dict(r, preset=want[tag][0], arm=want[tag][1], max_abs_e=res["max_abs_e"], spin=bool(res["spin"]), v_max40=float(v[:40].max()),
                         standing=float((v < 0.3).mean()), n=len(v), flips=fl, flip_rate=fr, w_rms=float(np.sqrt(np.mean(w ** 2))), w_max=float(np.abs(w).max()),
                         oscillates=bool(fl >= 8 and fr >= 0.15), route_turn=turn[r["scene"]], turning=turn[r["scene"]] >= 30))
        traces[f"{want[tag][0]}|{want[tag][1]}|{sc}"] = dict(w=[round(float(x), 2) for x in w], v=[round(float(x), 2) for x in v],
                                                           kappa=[round(float(x), 4) for x in kap if x is not None])
    with open(OUT / "extract.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with gzip.open(OUT / "traces.json.gz", "wt") as f:
        json.dump(traces, f)
    print(len(rows), "rows ->", OUT)


if __name__ == "__main__":
    {"extract": extract}[sys.argv[1]]()
