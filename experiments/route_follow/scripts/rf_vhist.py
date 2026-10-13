#!/usr/bin/env python3
"""VHIST read-out (experiments/route_follow/plans/2026-10-13-vhist-prereg.md): the serving-side speed-history switch (JEV_VHIST,
experiments/alpasim/lib/vhist.py) on held 460 at 18 decisions. CPU only, read-only. Box, AlpaSim's venv.

  rf_vhist.py same --a <run dir> --b <run dir>[,<run dir>..]     plans of a against b at every common (scene, k) of the last driver session:
                                                                 count equal (bit for bit, as logged) and the largest difference (m)
  rf_vhist.py check --run <run dir>                              every logged `vhist` record against vhist.py's rule, recomputed from the
                                                                 logged measured row, command and previous plan (mismatches listed)
  rf_vhist.py report --runs LABEL=<dir>[,<dir>..] ... --scenes <list>[,<list>..] --pairs A:B .. --out DIR
      per label: mean score and zeros over the listed scenes (a scene without a rollout counts 0, zero-taxonomy's convention), zero kinds,
      GT distance (mean of the aggregate dist_to_gt_trajectory over rollouts), rollouts with collision_rear > 0 (all / scored > 0), mean
      progress_score and progress_rel, plus rf_deficit.py (run unchanged as a subprocess into DIR/deficit). Pairs: paired differences over
      the listed scenes, log-clustered 95 % CIs (jevdrive.stats.paired). -> DIR/vhist.json, vhist.md, scenes.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]
sys.path[:0] = [str(R), str(R / "experiments/alpasim/lib"), str(R / "experiments/anchor_offset/scripts"), str(HERE)]


def records(spec):
    from ao_e1 import drive_records
    out = {}
    for D in spec.split(","):
        for sc, rr in drive_records(D).items():
            out.setdefault(sc, rr)
    return out


def same(a):
    A, B = records(a.a), records(a.b)
    n = eq = 0
    worst = (0.0, None)
    for sc in sorted(set(A) & set(B)):
        kb = {r["k"]: r for r in B[sc]}
        for r in A[sc]:
            if r["k"] in kb:
                p, q = np.asarray(r["poses"]), np.asarray(kb[r["k"]]["poses"])
                n += 1
                eq += bool(np.array_equal(p, q))
                d = float(np.abs(p - q).max())
                worst = max(worst, (d, (sc, r["k"])), key=lambda x: x[0])
    print(f"{len(set(A) & set(B))} common scenes, {n} common decisions, {eq} plans equal, largest difference {worst[0]:.4f} m at {worst[1]}")
    return 0 if n and eq == n else 1


def check(a):
    os.environ["JEV_VHIST"] = "hold"
    import vhist as VH
    bad = n = 0
    for sc, rr in records(a.run).items():
        st = VH.new()
        for r in rr:
            vh = r.get("vhist")
            if vh is None:
                print(f"{sc} k {r['k']}: no vhist record")
                bad += 1
                continue
            n += 1
            ego, fed = np.asarray(r["ego"], np.float32), np.asarray(vh["ego_fed"])
            if vh["mode"] == "zero":
                exp = ego.copy()
                exp[VH.EGO_SPEED] = 0
            else:
                exp, info = VH.feed(st, ego, np.eye(4)[r["cmd"]])
                if info["release"] != vh["release"] or abs(info["v_fed"] - vh["v_fed"]) > 2e-3:
                    print(f"{sc} k {r['k']}: release / v_fed {info['release']!r} {info['v_fed']} logged {vh['release']!r} {vh['v_fed']}")
                    bad += 1
                VH.observe(st, r.get("poses_model", r["poses"]))
            if np.abs(exp - fed).max() > 2e-4:                  # the logged rows are rounded to 5 decimals
                print(f"{sc} k {r['k']}: fed row differs by {np.abs(exp - fed).max():.5f}")
                bad += 1
    print(f"{n} vhist records checked, {bad} mismatches")
    return int(bad > 0)


def report(a):
    import pandas as pd
    from jevdrive import stats
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = sorted({s for f in a.scenes.split(",") for s in Path(f).read_text().split()})
    rows = []
    for spec in a.runs:
        label, dirs = spec.split("=", 1)
        got = {}
        for D in dirs.split(","):
            for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]:
                got.setdefault(r["clipgt_id"], r)
        for s in scenes:
            r = got.get(s)
            m, sm = (r.get("metrics") or {}, r.get("score_metrics") or {}) if r else ({}, {})
            sc = float(r["score"] or 0) if r and not r.get("failure_reason") else 0.0
            z = sc == 0
            rows.append(dict(label=label, scene=s, rollout=r is not None, score=sc, zero=int(z),
                             z_col=int(z and (m.get("collision_at_fault") or 0) > 0), z_off=int(z and (m.get("offroad") or 0) > 0),
                             z_cor=int(z and (m.get("left_corridor_laterally") or 0) > 0),
                             gt=m.get("dist_to_gt_trajectory", np.nan) if r else np.nan,
                             rear=float((m.get("collision_rear") or 0) > 0) if r else np.nan,
                             rear_nz=float((m.get("collision_rear") or 0) > 0 and sc > 0) if r else np.nan,
                             prog=sm.get("progress_score", np.nan) if r else np.nan, prel=sm.get("progress_rel", np.nan) if r else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(out / "scenes.csv", index=False)
    py = sys.executable
    subprocess.run([py, str(HERE / "rf_deficit.py"), "--runs", *a.runs, "--out", str(out / "deficit"), "--workers", str(a.workers),
                    *(["--pairs", *a.pairs] if a.pairs else [])], check=True, stdout=subprocess.DEVNULL)
    D = json.loads((out / "deficit/deficit.json").read_text())
    res = dict(labels={}, pairs={})
    for lab, x in S.groupby("label", sort=False):
        d = D["labels"].get(lab, {})
        res["labels"][lab] = dict(scenes=len(x), rollouts=int(x.rollout.sum()), mean=float(x.score.mean()), zeros=int(x.zero.sum()),
                                  col=int(x.z_col.sum()), off=int(x.z_off.sum()), cor=int(x.z_cor.sum()), gt=float(x.gt.mean()),
                                  rear=int(np.nansum(x.rear)), rear_nz=int(np.nansum(x.rear_nz)), prog=float(x.prog.mean()), prel=float(x.prel.mean()),
                                  def_med=d.get("median_dec", np.nan), def_mean=d.get("mean_scene", np.nan))
    logs = np.array([s.rsplit("-", 1)[0] for s in scenes])
    P = {c: S.pivot(index="scene", columns="label", values=c).reindex(scenes) for c in ("score", "zero", "gt", "rear", "prog")}
    for p in a.pairs or []:
        A, B = p.split(":")
        res["pairs"][p] = {c: stats.paired(P[c][A].to_numpy(float), P[c][B].to_numpy(float), groups=logs) for c in P}
        res["pairs"][p]["deficit"] = D["pairs"].get(p)
        ra, rb = res["labels"][A]["def_mean"], res["labels"][B]["def_mean"]
        res["pairs"][p]["share_removed"] = float(1 - ra / rb) if rb else np.nan
    (out / "vhist.json").write_text(json.dumps(res, indent=1, default=float))
    md = ["# VHIST read: held 460 at 18 decisions (plans/2026-10-13-vhist-prereg.md)", "",
          "| label | scenes / rollouts | mean | zeros (coll / off / corr) | GT (m) | rear-ended (all / scored > 0) | progress score | progress rel | "
          "deficit median (m) | deficit mean of rollout means (m) |", "|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|"]
    md += [f"| {k} | {r['scenes']} / {r['rollouts']} | {r['mean']:.4f} | {r['zeros']} ({r['col']} / {r['off']} / {r['cor']}) | {r['gt']:.3f} | "
           f"{r['rear']} / {r['rear_nz']} | {r['prog']:.4f} | {r['prel']:.4f} | {r['def_med']:+.3f} | {r['def_mean']:+.3f} |" for k, r in res["labels"].items()]
    if res["pairs"]:
        md += ["", "| A - B | mean score | zeros | GT (m) | rear-ended | progress score | deficit (rollout means) | share of B's shortness removed |",
               "|:--|:--|:--|:--|:--|:--|:--|--:|"]
        for k, r in res["pairs"].items():
            n = len(scenes)
            md.append(f"| {k} | {stats.fmt(r['score'], '+.4f')} | {stats.fmt(r['zero'], '+.1f', n)} | {stats.fmt(r['gt'], '+.3f')} | "
                      f"{stats.fmt(r['rear'], '+.1f', r['rear']['n'])} | {stats.fmt(r['prog'], '+.4f')} | "
                      f"{stats.fmt(r['deficit'], '+.3f') if r['deficit'] else '-'} | {r['share_removed']:.2f} |")
        md += ["", "Zeros and rear-ended differences are totals (mean x scenes in the pair); CIs log-clustered, 95 %."]
    (out / "vhist.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("same")
    s.add_argument("--a", required=True), s.add_argument("--b", required=True)
    s = sp.add_parser("check")
    s.add_argument("--run", required=True)
    s = sp.add_parser("report")
    s.add_argument("--runs", nargs="+", required=True), s.add_argument("--scenes", required=True), s.add_argument("--pairs", nargs="*")
    s.add_argument("--out", required=True), s.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    sys.exit({"same": same, "check": check, "report": report}[a.cmd](a) or 0)
