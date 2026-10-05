#!/usr/bin/env python
"""Offline description of standstills in existing `spec`-like runs, to set the resume rule's thresholds before any scored run
(plans/2026-10-05-op-resume-prereg.md section 2). Reads logs only; nothing is re-simulated.

  HUGSIM  decision 118's opctrl arm (= preset spec), 64 scenes: $DATA_DIR/runs/opctrl/closed/cinque-opctrl/zs/*/
          zs_steps.jsonl (model heads) + infos.pkl (ground-truth boxes, evaluation only)
  B2D     rig122 `ol` arm (= preset spec, zones on, seed 2, 20 turn routes) and decision 118.5's `opc` arm (drive + op lateral,
          19 routes x seeds 2 / 3): plans.jsonl (model heads + ground-truth ctx, evaluation only)

One row per standstill episode (v < V_STAND for >= 1 s, outside warm-up). Model-own features (what a rule may read): lead_prob,
lead x, the plan's speed at 5 s and position at 10 s (HUGSIM) / 5 s (B2D), the meta gas-press probability (B2D only).
Ground truth (only to describe, never for the rule): HUGSIM nearest box in the ego lane ahead; B2D light state / distance,
lead gap. Output: <out>/episodes.csv; <out>/first_fire.csv = the pre-registered rule (jevdrive/openpilot/resume.py defaults) replayed on
each logged run up to its first firing (counterfactual: after a firing the logged run no longer applies), with the ground truth there.
  python experiments/op_resume/scripts/or_offline.py --out $DATA_DIR/runs/op_resume/offline
"""
import argparse
import csv
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
V_STAND = 0.1
HUG = D / "runs/opctrl/closed/cinque-opctrl/zs"
HUG_RES = D / "runs/opctrl/closed/results.csv"
B2D = {"ol": sorted(glob.glob(str(D / "runs/rig122/arms/ol-s2-k*/attempts/*"))),
       "opc": sorted(glob.glob(str(D / "runs/vlm_arb/arms/v2-opc-s*-q*/attempts/*")))}


def episodes(v, dt, min_s=1.0):
    """[(i0, i1)] index ranges (i1 exclusive) where v < V_STAND for >= min_s."""
    out, i0 = [], None
    for i, x in enumerate(list(v) + [1e9]):
        if x < V_STAND and i0 is None:
            i0 = i
        elif x >= V_STAND and i0 is not None:
            if (i - i0) * dt >= min_s:
                out.append((i0, i))
            i0 = None
    return out


def q(a, p):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return round(float(np.percentile(a, p)), 3) if len(a) else ""


def hugsim_rows():
    ends = {}
    if HUG_RES.exists():
        for r in csv.DictReader(open(HUG_RES)):
            if r["tag"] == "cinque-opctrl":
                ends[Path(r["run_dir"]).name] = (r["scenario"], r["end"], r["hdscore"])
    rows = []
    for d in sorted(HUG.iterdir()):
        f = d / "zs_steps.jsonl"
        if not f.exists():
            continue
        L = [json.loads(x) for x in open(f) if '"setup"' not in x[:12]]
        try:
            infos = pickle.load(open(d / "infos.pkl", "rb"))
        except Exception:
            infos = None
        v = [r["v"] for r in L]
        scen, end, hd = ends.get(d.name, (d.name, "", ""))
        for i0, i1 in episodes(v, 0.25):
            seg = L[i0:i1]
            gt = []
            if infos is not None:
                for k in range(i0, min(i1, len(infos))):
                    b = [bb for bb in infos[k]["obj_boxes"] if 0 < bb[0] < 30 and abs(bb[1]) < 2.0]
                    gt.append(min(bb[0] for bb in b) if b else np.inf)
            later = any(r["v"] > 1.0 for r in L[i1:])
            rows.append(dict(board="hugsim", arm="spec", run=scen, end=end, hd=hd, t0=round(seg[0]["t"], 2),
                             dur_s=round(0.25 * len(seg), 2), last=i1 >= len(L), relaunched=later,
                             lp_med=q([r.get("lead_prob") for r in seg], 50), lp_max=q([r.get("lead_prob") for r in seg], 100),
                             lx_med=q([r.get("lead_x") for r in seg], 50),
                             frac_lead15=round(np.mean([(r.get("lead_prob") or 0) > 0.5 and (r.get("lead_x") or 99) < 15 for r in seg]), 3),
                             v5_med=q([r["model_v"][3] for r in seg], 50), x10_med=q([r["model_pos"][-1][0] for r in seg], 50),
                             x25_med=q([r["model_pos"][3][0] for r in seg], 50),
                             gt_lane_min=round(float(np.min(gt)), 1) if gt and np.isfinite(np.min(gt)) else "",
                             gt_frac_lead15=round(float(np.mean(np.asarray(gt) < 15)), 3) if gt else ""))
    return rows


def b2d_rows():
    rows = []
    for arm, dirs in B2D.items():
        for rd in dirs:
            att = sorted(glob.glob(rd + "/*/plans.jsonl"), key=lambda p: int(Path(p).parent.name))
            if not att:
                continue
            f = att[-1]
            unit = Path(rd).parents[1].name
            L = [json.loads(x) for x in open(f)]
            L = [r for r in L if not r.get("warm")]
            if len(L) < 10:
                continue
            dt = float(np.median(np.diff([r["t"] for r in L])))
            v = [r["v"] for r in L]
            for i0, i1 in episodes(v, dt):
                seg = L[i0:i1]
                ctx = [r.get("ctx") or {} for r in seg]
                red = [c.get("tl") in (1, 2) and c.get("tl_dist", 99) < 40 for c in ctx]
                lgap = [c.get("lead_gap", np.inf) for c in ctx]
                rel = [r.get("rel") for r in seg if isinstance(r.get("rel"), str)]
                later = any(r["v"] > 1.0 for r in L[i1:])
                rows.append(dict(board="b2d", arm=arm, run="%s/%s" % (unit, Path(rd).name), end="", hd="", t0=round(seg[0]["t"], 2),
                                 dur_s=round(dt * len(seg), 2), last=i1 >= len(L), relaunched=later,
                                 lp_med=q([r["lp"][0] for r in seg], 50), lp_max=q([r["lp"][0] for r in seg], 100),
                                 lx_med=q([r["lead"][0][0] for r in seg], 50),
                                 frac_lead15=round(np.mean([r["lp"][0] > 0.5 and r["lead"][0][0] < 15 for r in seg]), 3),
                                 v5_med=q([r["vplan"][4] for r in seg], 50), x10_med="", x25_med=q([r["op_xy"][3][0] for r in seg], 50),
                                 gas2_med=q([r["gas"][1] for r in seg], 50), gas0_med=q([r["gas"][0] for r in seg], 50),
                                 latch_frac=round(np.mean([bool(r.get("latch")) for r in seg]), 3), rel=";".join(sorted(set(rel))),
                                 red_frac=round(float(np.mean(red)), 3), gt_lane_min=round(float(min(lgap)), 1) if np.isfinite(min(lgap)) else "",
                                 gt_frac_lead15=round(float(np.mean(np.asarray(lgap) < 15)), 3)))
    return rows


def first_fire():
    """Replay ResumeRule (defaults) on every logged run until its first firing; -> rows with the ground truth at that step."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from jevdrive.openpilot import resume as RR
    rows = []
    ends = {Path(r["run_dir"]).name: (r["scenario"], r["end"]) for r in csv.DictReader(open(HUG_RES)) if r["tag"] == "cinque-opctrl"}
    for d in sorted(HUG.iterdir()):
        f = d / "zs_steps.jsonl"
        if not f.exists():
            continue
        L = [json.loads(x) for x in open(f) if '"setup"' not in x[:12]]
        infos = pickle.load(open(d / "infos.pkl", "rb"))
        rule, hit = RR.ResumeRule(), None
        for k, r in enumerate(L):
            on, why = rule.step(r["t"], r["v"], r.get("lead_prob"), r.get("lead_x"), RR.plan_s1_from_points(r["plan"], 0.5))
            if why == "fire":
                b = [bb for bb in infos[min(k, len(infos) - 1)]["obj_boxes"] if 0 < bb[0] < 30 and abs(bb[1]) < 2.0]
                hit = dict(t=r["t"], lead_prob=round(r.get("lead_prob") or 0, 3), lead_x=round(r.get("lead_x") or 0, 1),
                           gt_lane_min=round(min(bb[0] for bb in b), 1) if b else "")
                break
        sc, end = ends.get(d.name, (d.name, ""))
        rows.append(dict(board="hugsim", run=sc, end=end, fires=hit is not None, **(hit or {})))
    for arm, dirs in B2D.items():
        for rd in dirs:
            att = sorted(glob.glob(rd + "/*/plans.jsonl"), key=lambda p: int(Path(p).parent.name))
            if not att:
                continue
            rule, hit = RR.ResumeRule(), None
            for r in (json.loads(x) for x in open(att[-1])):
                if r.get("warm"):
                    continue
                s1 = float(np.hypot(*r["op_xy"][0])) if r.get("op_xy") else None      # op_xy[0] = plan at 1 s (rear frame)
                on, why = rule.step(r["t"], r["v"], r["lp"][0], r["lead"][0][0], s1)
                if why == "fire":
                    c = r.get("ctx") or {}
                    hit = dict(t=r["t"], lead_prob=r["lp"][0], lead_x=round(r["lead"][0][0], 1), tl=c.get("tl"), tl_dist=c.get("tl_dist"),
                               gt_lane_min=c.get("lead_gap", ""), latch=r.get("latch"))
                    break
            rows.append(dict(board="b2d", run="%s:%s/%s" % (arm, Path(rd).parents[1].name, Path(rd).name), end="", fires=hit is not None,
                             **(hit or {})))
    return rows


def write_csv(rows, path):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = hugsim_rows() + b2d_rows()
    write_csv(rows, out / "episodes.csv")
    print("%d episodes -> %s" % (len(rows), out / "episodes.csv"))
    ff = first_fire()
    write_csv(ff, out / "first_fire.csv")
    print("%d runs, %d fire -> %s" % (len(ff), sum(r["fires"] for r in ff), out / "first_fire.csv"))


if __name__ == "__main__":
    main()
