#!/usr/bin/env python3
"""ONPOL-LONG zero-score clips of one checkpoint's held read (default YS-L18-s1): table of zeros by type and failure decision, a stratified
sample, and the case list for on_clips.py's msgs / maps / replay / clip steps (unedited; ol_clips.sh chains them).
  table   (AlpaSim venv)  needs $OC/read_sim.pkl (c1_extract.py logs of the read run) -> $OC/zeros.csv
  cases   --n corridor=6 offroad=5 collision=4 --seed 0 -> sample.csv, cases.csv, cases.pkl, clip_scenes.txt, rows.json
$OC defaults to $DATA_DIR/runs/onpol_long/zeros/<tag>. Failure decision i = index (0 based) of the driver call in force at the first metric
timestamp where the failure metric is nonzero (decisions at 2 Hz, 18 per rollout)."""
import argparse
import bisect
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/onpol_long"
KIND = [("collision_at_fault", "collision"), ("offroad", "offroad"), ("left_corridor_laterally", "corridor")]


def cmd_table(a):
    OC = Path(a.oc)
    sims = pickle.load(open(OC / "read_sim.pkl", "rb"))
    P = {r["scene"]: r for r in csv.DictReader(open(a.per_scene)) if r["label"] == f"{a.tag}@18"}
    rows = []
    for s, o in sorted(sims.items()):
        if s not in P or float(P[s]["score"]) != 0:
            continue
        now = np.array([d["now"] for d in o["drive"]])
        first = {}
        for m, k in KIND:
            v = o["metrics"].get(m)
            if v is not None and (v[:, 1] > 0).any():
                t = v[(v[:, 1] > 0), 0].min()
                first[k] = (int(np.searchsorted(now, t, side="right")) - 1, (t - now[0]) / 1e6)
        typ = min(first, key=lambda k: first[k][1]) if first else "other"       # type = the earliest failure
        rows.append(dict(scene=s, type=typ, decision=first[typ][0] if first else -1, t_s=round(first[typ][1], 2) if first else "",
                         kinds="+".join(first) or "-", n_drive=len(now), dist_m=P[s]["dist_traveled_m"], gt_m=P[s]["gt_dist_traveled_m"]))
    with open(OC / "zeros.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()), None
        w[0].writeheader(), w[0].writerows(rows)
    print(len(rows), "zeros with a log;", {t: sum(r["type"] == t for r in rows) for t in ("corridor", "offroad", "collision", "other")})


def cmd_cases(a):
    OC = Path(a.oc)
    Z = list(csv.DictReader(open(OC / "zeros.csv")))
    sims = pickle.load(open(OC / "read_sim.pkl", "rb"))
    rng = np.random.default_rng(a.seed)
    pick = []
    for kv in a.n:
        t, n = kv.split("=")
        pool = sorted(r["scene"] for r in Z if r["type"] == t)
        pick += [dict(type=t, scene=s) for s in sorted(rng.choice(pool, int(n), replace=False))]
    run = str(W / f"read/{a.tag}/run")
    cases, C = {}, {}
    for p in pick:
        cid = p["scene"].rsplit("-", 1)[1][:8] + "_" + a.tag
        C[a.tag, p["scene"]] = sims[p["scene"]] | {"run": run}
        cases[cid] = dict(id=cid, set=a.tag, scene=p["scene"], obj="", kind=p["type"], origin="ONPOL-LONG zeros", look="")
    (OC / "cases.csv").write_text("id,set,scene,obj,kind,origin,look\n" + "".join(",".join(c.values()) + "\n" for c in cases.values()))
    pickle.dump(C, open(OC / "cases.pkl", "wb"), protocol=4)
    (OC / "clip_scenes.txt").write_text("".join(f"{c['set']} {c['scene']}\n" for c in cases.values()))
    (OC / "rows.json").write_text(json.dumps([dict(arm=a.tag, what="zero", scene=p["scene"], turn="", tags=[dict(tag=a.tag, side="arm", kind=p["type"], clip=p["scene"].rsplit("-", 1)[1][:8] + "_" + a.tag)]) for p in pick]))
    (OC / "sample.csv").write_text("type,scene,clip\n" + "".join(f"{p['type']},{p['scene']},{p['scene'].rsplit('-', 1)[1][:8]}_{a.tag}\n" for p in pick))
    print(len(cases), "cases")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for n, f in (("table", cmd_table), ("cases", cmd_cases)):
    p = sub.add_parser(n)
    p.set_defaults(fn=f)
    p.add_argument("--tag", default="YS-L18-s1")
    p.add_argument("--oc", default=os.environ.get("ON_CLIPS", ""))
    p.add_argument("--per-scene", default=str(W / "report/Y-S/per_scene.csv"))
    p.add_argument("--n", nargs="*", default=["corridor=6", "offroad=5", "collision=4"])
    p.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
a.fn(a)
