#!/usr/bin/env python3
"""nuPlan JEV_ACC (plans/2026-10-16-nuplan-acc-prereg.md): the smoke gate and the held report. AlpaSim's venv on the box.

  rf_nuacc.py ident --pair NAME=OLD:NEW --on NAME=DIR --out ident.json
      the driver's off-identity (every decision's poses equal, rf_tts.py ident's comparison) and the ACC smoke checks: every inference
      record has fix.acc, no inference / input errors, v_hand locked at the 0.5 s call (k = 1, the nuPlan handoff), `set` wins somewhere.
      rc 2 when anything fails (the pool then fails every held read chained after it)
  rf_nuacc.py report --dir R --ref RUNS --acc RUNS --vl RUNS --stored RUNS [--workers N]
      after ol_confirm (R/held18) and rf_follow (R/follow): the registered lines (acc vs ref), the GT split per rollout (gt_decomp/gd.py's
      evaluator: at the step of maximal GT, cross-track = lateral_dist_to_gt_trajectory, past-end = GT - cross-track) with log-clustered
      paired CIs, ACC term / guard shares and handoff speeds, new and fixed zeros. -> R/nuacc.md, R/nuacc.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R_ = HERE.parents[2]
sys.path[:0] = [str(HERE), str(R_ / "experiments/gt_decomp/scripts")]
from rf_tts import drive_logs  # noqa: E402

ARMS = ("ref", "acc", "vl", "stored")


def ident(a):
    res, ok = {}, True
    for p in a.pair:
        name, dirs = p.split("=", 1)
        A, B = (drive_logs(x) for x in dirs.split(":"))
        d, miss = [], 0
        for sc in sorted(set(A) | set(B)):
            ka = {r["k"]: np.asarray(r["poses"]) for r in A.get(sc, [])}
            kb = {r["k"]: np.asarray(r["poses"]) for r in B.get(sc, [])}
            miss += len(set(ka) ^ set(kb))
            d += [float(np.abs(ka[k] - kb[k]).max()) for k in set(ka) & set(kb)]
        d = np.array(d)
        e = dict(scenes=len(set(A) & set(B)), decisions=len(d), unmatched=miss, max_abs_m=float(d.max()) if len(d) else None,
                 identical=bool(len(d) and (d == 0).all() and miss == 0 and len(set(A)) == len(set(B)) >= 2))
        ok &= e["identical"]
        res[name] = e
    for p in a.on:
        name, run = p.split("=", 1)
        X = drive_logs(run)
        recs = [r for v in X.values() for r in v]
        acc = [r["fix"]["acc"] for r in recs if "acc" in (r.get("fix") or {})]
        hand = {sc: [(r["k"], (r.get("fix") or {}).get("acc", {}).get("v_hand"), (r.get("fix") or {}).get("v0")) for r in v[:3]] for sc, v in X.items()}
        hand_ok = all(len(h) >= 2 and h[0][1] is None and h[1][1] is not None and abs(h[1][1] - h[1][2]) < 1e-3 for h in hand.values())
        err = 0
        for f in glob.glob(str(Path(run) / "**/driver-logs/drive.jsonl"), recursive=True):
            for line in open(f):
                if '"close"' in line:
                    c = json.loads(line)
                    err += int(c.get("inference_error", 0)) + int(c.get("input_error", 0))
        won = collections.Counter(x["won"] for x in acc)
        e = dict(scenes=len(X), inferences=len(recs), with_acc=len(acc), errors=err, won=dict(won), guard=dict(collections.Counter(x["guard"] for x in acc)),
                 handoff=hand, handoff_ok=hand_ok,
                 ok=bool(len(X) >= 6 and recs and len(acc) == len(recs) and err == 0 and hand_ok and won.get("set", 0) > 0))
        ok &= e["ok"]
        res[name] = e
    res["pass"] = bool(ok)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 2)


def _split(job):
    """(scene, run dir) -> GT split of its rollout (gd.py's evaluator re-implementation)."""
    import gd
    sc, run = job
    f = sorted(glob.glob(f"{run}/rollouts/{sc}/*/metrics.parquet"))
    if not f:
        return sc, None
    rv, _ = gd.rollout_values(gd.series(run, sc, Path(f[0]).parent.name))
    return sc, dict(gt=rv["gt"], gt_lat=rv["gt_lat"], gt_over=rv["gt_over"])


def report(a):
    import pandas as pd
    R = Path(a.dir)
    runs = dict(ref=a.ref, acc=a.acc, vl=a.vl, stored=a.stored)
    tag = {k: Path(v).name for k, v in runs.items()}
    C = json.load(open(R / "held18/confirm.json"))["held18"]
    S = pd.read_csv(R / "held18/per_scene.csv")
    S["tag"] = S.label.str.split("@").str[0]
    inv = {v: k for k, v in tag.items()}
    S["arm"] = S.tag.map(inv)
    sc = sorted(set.intersection(*[set(X[X.offroad_or_collision_at_fault.notna()].scene) for _, X in S.groupby("arm")]))
    S = S[S.scene.isin(sc)].set_index(["arm", "scene"])
    jobs = [(s, S.loc[(k, s), "run"]) for k in ARMS for s in sc]
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_split, jobs, chunksize=8))
    G = pd.DataFrame([dict(arm=k, scene=s, **(v or dict(gt=np.nan, gt_lat=np.nan, gt_over=np.nan))) for (s, _), v, k in
                      zip(jobs, (o[1] for o in out), [k for k in ARMS for _ in sc])]).set_index(["arm", "scene"])
    logs = np.array([s.rsplit("-", 1)[0] for s in sc])
    L = np.unique(logs)
    W = np.stack([np.bincount(d, minlength=len(L)) for d in np.random.default_rng(0).integers(0, len(L), (10000, len(L)))])[:, np.searchsorted(L, logs)]

    def paired(col, x, y="ref", src=G):
        d = src.loc[x, col].reindex(sc).to_numpy() - src.loc[y, col].reindex(sc).to_numpy()
        ok = np.isfinite(d)
        bs = (W[:, ok] @ d[ok]) / W[:, ok].sum(1)
        return [float(d[ok].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    z = {k: S.loc[k].score.reindex(sc) == 0 for k in ARMS}
    nz = {k: int(z[k].sum()) for k in ARMS}
    km = C["at_fault_km"]
    acc_t, ref_t = tag["acc"], tag["ref"]
    dm, dl = C[acc_t]["score"], paired("gt_lat", "acc")
    lines = [("held mean paired d, CI lower > -0.01", f"{dm[0]:+.4f} [{dm[1]:+.4f}, {dm[2]:+.4f}]", dm[1] > -0.01),
             ("zeros <= ref + 2", f"{nz['acc']} vs {nz['ref']} (d {C[acc_t]['zero'][0]:+.0f} [{C[acc_t]['zero'][1]:+.1f}, {C[acc_t]['zero'][2]:+.1f}])", nz["acc"] <= nz["ref"] + 2),
             ("cross-track GT paired d (point) <= +0.05 m", f"{dl[0]:+.3f} [{dl[1]:+.3f}, {dl[2]:+.3f}] m", dl[0] <= 0.05),
             ("at-fault km >= 0.9 x ref (point)", f"{km[acc_t]:.3f} vs {km[ref_t]:.3f} km (ratio {km[acc_t] / max(km[ref_t], 1e-9):.2f})", km[acc_t] >= 0.9 * km[ref_t])]
    passed = all(x[2] for x in lines)
    f = lambda v, p=3: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]"  # noqa: E731
    md = [f"# nuPlan JEV_ACC: held 460 @18, {len(sc)} paired scenes, {len(L)} logs; acc vs ref (same-code re-read of RSG-L18-s0)", "",
          "Arms: ref = RSG-L18-s0 no switch (re-read 2026-10-16), acc = JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1, vl = JEV_VCONT=1.0 JEV_LEAD=1, "
          "stored = the 2026-10-13 RSG-L18-s0 read. Official camera set (ALPASIM_FAST unset). Log-clustered 95 % CI, 10 000 draws.", "",
          "| registered line | value | result |", "|:--|:--|:--|"]
    md += [f"| {n} | {v} | {'pass' if p else 'FAIL'} |" for n, v, p in lines]
    md += ["", f"**Verdict: {'PASS (held); the public 1491 guard may be read' if passed else 'FAIL (held); no public read'}.**", "",
           "| arm | mean score | zeros | at-fault zeros | GT (m) | cross-track GT (m) | past-end GT (m) | at-fault km |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for k in ARMS:
        X = S.loc[k].reindex(sc)
        atf = int(((X.score == 0) & (X.offroad_or_collision_at_fault > 0)).sum())
        md.append(f"| {k} ({tag[k]}) | {X.score.mean():.4f} | {nz[k]} | {atf} | {G.loc[k, 'gt'].mean():.3f} | {G.loc[k, 'gt_lat'].mean():.3f} | "
                  f"{G.loc[k, 'gt_over'].mean():.3f} | {km[tag[k]]:.3f} |")
    md += ["", "| a - b | mean score | zeros | at-fault zeros | GT (m) | cross-track GT (m) | past-end GT (m) | at-fault km |", "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    pairs = {}
    for x, y in (("acc", "ref"), ("vl", "ref"), ("acc", "vl"), ("stored", "ref")):
        key = tag[x] if y == "ref" else f"{tag[x]} - {tag[y]}"
        c = C.get(key)
        if c is None:                                     # a pair ol_confirm was not asked for: score / zeros from the table here
            c = {k2: paired(k2, x, y, S.assign(zero=(S.score == 0).astype(float), z_atf=((S.score == 0) & (S.offroad_or_collision_at_fault > 0)).astype(float))
                            .rename(columns={"dist_to_gt_trajectory": "gt"})) for k2 in ("score", "zero", "z_atf", "gt")}
            c["zero"] = [v * len(sc) for v in c["zero"]]
            c["z_atf"] = [v * len(sc) for v in c["z_atf"]]
            c["at_fault_km"] = [km[tag[x]] - km[tag[y]], float("nan"), float("nan")]
        pr = dict(score=c["score"], zero=c["zero"], z_atf=c["z_atf"], gt=c["gt"], gt_lat=paired("gt_lat", x, y), gt_over=paired("gt_over", x, y), km=c["at_fault_km"])
        pairs[f"{x} - {y}"] = pr
        md.append(f"| {x} - {y} | {f(pr['score'], 4)} | {f(pr['zero'], 1)} | {f(pr['z_atf'], 1)} | {f(pr['gt'])} | {f(pr['gt_lat'])} | {f(pr['gt_over'])} | "
                  f"{pr['km'][0]:+.3f} |")
    # zeros that moved
    kind = lambda X: "col" if X.collision_at_fault > 0 else "off" if X.offroad > 0 else "cor" if X.left_corridor_laterally > 0 else "?"  # noqa: E731
    new = [s for s in sc if z["acc"][s] and not z["ref"][s]]
    fixed = [s for s in sc if z["ref"][s] and not z["acc"][s]]
    # ACC statistics from the drive logs
    X = drive_logs(runs["acc"].rstrip("/") + "/")
    recs = [r for v in X.values() for r in v]
    acc = [r["fix"]["acc"] for r in recs if "acc" in (r.get("fix") or {})]
    won = collections.Counter(x["won"] for x in acc)
    near = collections.Counter(x["near"] for x in acc)
    guard = collections.Counter(x["guard"] or "none" for x in acc)
    vh = np.array([next((r["fix"]["acc"]["v_hand"] for r in v if r["fix"]["acc"]["v_hand"] is not None), np.nan) for v in X.values() if v and "acc" in (v[0].get("fix") or {})])
    cov = np.array([x["cover"] for x in acc])
    vhand = {s: next((r["fix"]["acc"]["v_hand"] for r in v if (r.get("fix") or {}).get("acc", {}).get("v_hand") is not None), None) for s, v in X.items()}
    md += ["", f"New zeros under acc ({len(new)}): " + (", ".join(f"{s[-16:]} ({kind(S.loc[('acc', s)])}, v_hand {vhand.get(s)})" for s in new) or "none"),
           f"Fixed by acc ({len(fixed)}): " + (", ".join(f"{s[-16:]} ({kind(S.loc[('ref', s)])})" for s in fixed) or "none"), "",
           f"ACC terms at 1.5 s over {len(acc)} decisions: won {dict(won.most_common())}; nearest {dict(near.most_common())}; guard {dict(guard.most_common())}.",
           f"Handoff speed v_hand (m/s) over {np.isfinite(vh).sum()} sessions: p10 / p50 / p90 {np.nanpercentile(vh, [10, 50, 90]).round(2).tolist()}, "
           f"share < 3 m/s {np.nanmean(vh < 3):.2f}; route cover > 0 on {np.mean(cov > 0):.2f} of decisions.", ""]
    fl = R / "follow"
    if fl.exists():
        md += [f"Turn completion (rf_follow): {fl}/ (descriptive)."]
    (R / "nuacc.md").write_text("\n".join(md) + "\n")
    json.dump(dict(scenes=len(sc), lines=[dict(line=n, value=v, ok=bool(p)) for n, v, p in lines], passed=bool(passed), zeros=nz, pairs=pairs,
                   at_fault_km={k: km[tag[k]] for k in ARMS}, new_zeros=new, fixed_zeros=fixed, won=dict(won), guard=dict(guard)),
              open(R / "nuacc.json", "w"), indent=1, default=float)
    G.reset_index().to_csv(R / "gt_split.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ident")
    p.add_argument("--pair", nargs="+", required=True)
    p.add_argument("--on", nargs="+", default=[])
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--dir", required=True)
    for k in ARMS:
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dict(ident=ident, report=report)[a.cmd](a)
