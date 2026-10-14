#!/usr/bin/env python3
"""Turning school phase 2 (RSGSELF-L18, self-branch rows in `--on-mode mixed`): the anti-lateral-ratchet guard and the ladder table.
Driven by rf_self_train.sh report / dry. Box.

  lat    (AlpaSim venv, CPU)  rf_selfread.py lat --runs LABEL=<run dir>[,<run dir>] ... --out DIR [--mean NAME=TAG,TAG ..] [--pairs A:B ..]
             [--scenes FILE] [--workers N]                          -> DIR/lat_decisions.parquet, DIR/lat.json, DIR/lat.md
         Held 460 @18 reads. Per decision (drive.jsonl, last session of a scene, rf_follow's reader) up to the first zero-score event (rf_follow's
         k_z: at-fault offroad / collision or dist_to_gt >= 4 m) and at fed speed >= 1 m/s: the route = AlpaSim's RouteGenerator rebuilt at every
         logged anchor (rm_build.regen, the full polyline right after the decision); class of the anchor against it = rf_offroad.rclass
         (junction / approach [junction within 30 m ahead] / jog / curve / straight, map-based intersection test). Kept: straight and curve
         (the non-junction, open-road decisions). dlat2 = signed lateral offset (left +) of the served plan's 2.0 s pose from the route minus
         the anchor's own offset: the plan's sideways move relative to the lane over 2 s. Statistic per label = mean over scenes of the
         scene's mean |dlat2| (a scene counts once); seed-mean labels (--mean) average the seeds' scene means; pairs A:B on the common scenes,
         log-clustered bootstrap 95 % CI (jevdrive.stats, log = scene id without its last "-<hash>"). Guard: A - RSG-L18 <= +0.02 m (point).
         Also reported: mean signed dlat2 (a one-sided drift) and mean |lateral| of the 2 s pose itself.
  table  (any python with pandas)  rf_selfread.py table --held DIR --guard DIR --lat DIR --arm NAME --ref NAME --seeds 0,1 --out FILE
         The ladder table of results/tschool/summary.md for one arm against its control: held 460 @18 mean / zeros, public 1479 @10 mean / zeros
         (each per seed, absolute numbers on the scenes every label of the confirm read shares), paired CIs from ol_confirm's confirm.json, the
         lateral guard; step-2 lines of the turning-school prereg Amendment 2 (held mean >= ref - 0.01, held zeros s0 <= ref s0 + 3, public zeros s0
         <= ref + 5) and the lateral guard (<= +0.02 m). Exam entry speed: read later on Tokyo (not here). A set that is absent (dry run: no public
         guard) is printed as "-".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/route_mem/scripts"),
                str(R_ / "experiments/alpasim/scripts"), str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/anchor_offset/scripts"),
                str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts"), str(R_ / "experiments/route_follow/scripts")]

T0, DT, V_MIN, I2S, KEEP, LINE = 17_000, 500_000, 1.0, 3, ("straight", "curve"), 0.02


def lat_scene(job):
    """(scene, [(label, run dir, drive records)], cfg) -> decision rows of every label on this scene."""
    scene, items, cfg = job
    import long_read as LR
    import rmem as RM
    from ao_e1 import loader, project, stations
    from rf_offroad import Road, rclass
    from rm_build import regen
    road = Road(loader(str(cfg)).get_data_source(scene).map, np.concatenate([np.asarray([r["anchor"] for r in recs], float)[:, :2] for _, _, recs in items]))
    out = []
    for label, D, recs in items:
        ev = LR.one((label, D, scene))
        kz = min([r["k"] for r in ev if r["atf"] or (r["dist_gt"] or 0) >= 4.0], default=10 ** 6)
        Fu = {}
        regen(scene, recs, cfg, Fu)
        for r in recs:
            k = int((r["t0"] - T0) // DT)
            e = np.asarray(r["ego"], float)
            v = float(np.hypot(e[4], e[5]) * 10.0)
            Rk = Fu.get(r["k"])
            if k > kz or v < V_MIN or Rk is None or len(Rk) < 2:
                continue
            Rk = Rk[np.isfinite(Rk).all(1)]
            Rk = Rk[np.r_[True, np.hypot(*np.diff(Rk, axis=0).T) > 1e-6]]
            a = np.asarray(r["anchor"], float)
            S = stations(Rk)
            st, la = project(Rk, S, a[:2])
            p2 = RM.to_local(a, np.asarray(r["poses"], float)[I2S:I2S + 1, :2])[0]
            _, lp = project(Rk, S, p2, st - 5.0, st + 60.0)
            out.append(dict(label=label, scene=scene, k=k, v=v, cls=rclass(road, Rk, a[:2]), lat_a=la, lat_p2=lp, dlat2=lp - la))
    return out


def cmd_lat(a):
    import pandas as pd
    import ev as EV
    from ao_e1 import drive_records
    from jevdrive import par, stats
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(Path(a.scenes).read_text().split()) if a.scenes else None
    with Run("route_follow", "selflat", config=vars(a)) as run:
        by, labels = {}, []
        for spec in a.runs:
            label, dirs = spec.split("=", 1)
            labels.append(label)
            seen = set()
            for D in dirs.split(","):
                for sc, rr in sorted(drive_records(D).items()):
                    if sc in seen or (keep is not None and sc not in keep):
                        continue
                    seen.add(sc)
                    for r in rr:
                        r.pop("ms", None), r.pop("hist", None)
                    by.setdefault(sc, []).append((label, D, rr))
            run.info(f"{label}: {len(seen)} rollouts")
        jobs = [(sc, it, EV.CFG) for sc, it in sorted(by.items())]
        res = par.pmap(lat_scene, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="scenes")
        err = {jobs[i][0]: str(e)[-300:] for i, e in res.errors.items()}
        X = pd.DataFrame([d for v in res.values if v for d in v])
        X.to_parquet(out / "lat_decisions.parquet", index=False)
        K = X[X.cls.isin(KEEP)].assign(adl=lambda x: x.dlat2.abs(), alp=lambda x: x.lat_p2.abs())
        per = K.groupby(["label", "scene"])[["adl", "dlat2", "alp"]].mean()
        ms = {n: v.split(",") for n, v in (m.split("=", 1) for m in a.mean)}
        P = {c: per[c].unstack("label") for c in per.columns}
        for n, v in ms.items():
            for c in P:
                P[c][n] = P[c][[t for t in v if t in P[c]]].mean(1, skipna=False)
        R = dict(errors=err, classes={lab: X[X.label == lab].cls.value_counts().to_dict() for lab in labels}, labels={}, pairs={})
        for lab in list(labels) + list(ms):
            m = P["adl"][lab].notna()
            R["labels"][lab] = dict(scenes=int(m.sum()), decisions=int((K.label == lab).sum()) if lab in labels else None,
                                    abs_dlat2=float(P["adl"][lab][m].mean()), signed_dlat2=float(P["dlat2"][lab][m].mean()),
                                    abs_lat_p2=float(P["alp"][lab][m].mean()))
        for p in a.pairs:
            A, B = p.split(":")
            d = (P["adl"][A] - P["adl"][B]).dropna()
            r = stats.bootstrap(d.to_numpy(), groups=[s.rsplit("-", 1)[0] for s in d.index])
            ds = (P["dlat2"][A] - P["dlat2"][B]).dropna()
            rs = stats.bootstrap(ds.to_numpy(), groups=[s.rsplit("-", 1)[0] for s in ds.index])
            R["pairs"][p] = dict(scenes=int(r["n"]), logs=int(r["units"]), abs_dlat2=[r["mean"], r["lo"], r["hi"]],
                                 signed_dlat2=[rs["mean"], rs["lo"], rs["hi"]], guard_pass=bool(r["mean"] <= LINE))
        (out / "lat.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
        md = ["# Lateral guard: the plan's 2 s sideways move relative to the route on non-junction held decisions (rf_selfread.py lat)", "",
              f"Decisions up to the first zero event, fed speed >= {V_MIN} m/s, anchor class straight / curve (rf_offroad.rclass); per scene mean, then the "
              f"mean over scenes. Guard: arm - control <= +{LINE} m on mean |dlat2|.", "",
              "| label | scenes | decisions | mean abs dlat2 (m) | mean signed dlat2 (m, left +) | mean abs lateral of the 2 s pose (m) |", "|:--|--:|--:|--:|--:|--:|"]
        md += [f"| {k} | {v['scenes']} | {v['decisions'] if v['decisions'] is not None else 'seed mean'} | {v['abs_dlat2']:.4f} | {v['signed_dlat2']:+.4f} | "
               f"{v['abs_lat_p2']:.4f} |" for k, v in R["labels"].items()]
        if R["pairs"]:
            md += ["", "| A - B | scenes / logs | abs dlat2 [95 % CI] | signed dlat2 [95 % CI] | guard (<= +0.02 m) |", "|:--|:--|:--|:--|:--|"]
            md += [f"| {p.replace(':', ' - ')} | {v['scenes']} / {v['logs']} | {v['abs_dlat2'][0]:+.4f} [{v['abs_dlat2'][1]:+.4f}, {v['abs_dlat2'][2]:+.4f}] | "
                   f"{v['signed_dlat2'][0]:+.4f} [{v['signed_dlat2'][1]:+.4f}, {v['signed_dlat2'][2]:+.4f}] | {'pass' if v['guard_pass'] else 'FAIL'} |"
                   for p, v in R["pairs"].items()]
        if err:
            md += ["", f"{len(err)} scenes failed (excluded): " + ", ".join(sorted(err)[:10])]
        (out / "lat.md").write_text("\n".join(md) + "\n")
        run.summary.update(out=str(out), labels=R["labels"], pairs=R["pairs"])
        print("\n".join(md))


def cmd_table(a):
    import pandas as pd
    seeds = a.seeds.split(",")
    arm, ref = a.arm, a.ref
    md = [f"# {arm} vs {ref}: ladder table (rf_selfread.py table)", ""]
    A = {}
    for nm, d, st in (("held", a.held, "held18"), ("public", a.guard, "all10")):
        f = Path(d or "/nonexistent") / "per_scene.csv"
        if not f.exists():
            continue
        S = pd.read_csv(f)
        S["tag"] = S.label.str.split("@").str[0]
        S = S[S.offroad_or_collision_at_fault.notna()]
        sc = set.intersection(*[set(S[S.tag == t].scene) for t in S.tag.unique()])
        S = S[S.scene.isin(sc)]
        A[nm] = dict(n=len(sc), mean=S.groupby("tag").score.mean().to_dict(), zeros=S.assign(z=S.score == 0).groupby("tag").z.sum().to_dict(),
                     confirm=json.loads((Path(d) / "confirm.json").read_text()).get(st, {}))
    L = json.loads((Path(a.lat) / "lat.json").read_text()) if a.lat and (Path(a.lat) / "lat.json").exists() else None
    tg = lambda b: [f"{b}-s{s}" for s in seeds]  # noqa: E731
    cell = lambda nm, k, b, f: " / ".join(f.format(A[nm][k][t]) if nm in A and t in A[nm][k] else "-" for t in tg(b))  # noqa: E731
    md += ["| arm | held mean " + " / ".join(f"s{s}" for s in seeds) + " | held zeros | public mean | public zeros | lateral abs dlat2 (m) |", "|:--|:--|:--|:--|:--|:--|"]
    for b in (ref, arm):
        la = " / ".join(f"{L['labels'][t]['abs_dlat2']:.4f}" if L and t in L["labels"] else "-" for t in tg(b))
        md.append(f"| {b} | {cell('held', 'mean', b, '{:.4f}')} | {cell('held', 'zeros', b, '{:g}')} | {cell('public', 'mean', b, '{:.4f}')} | "
                  f"{cell('public', 'zeros', b, '{:g}')} | {la} |")
    md += ["", f"Paired differences {arm} - {ref} (log-clustered 95 % CI; ol_confirm on the common scenes):", "",
           "| pair | held mean | held zeros | public mean | public zeros | lateral abs dlat2 (m) |", "|:--|:--|:--|:--|:--|:--|"]
    f3 = lambda v, p: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]" if v else "-"  # noqa: E731
    for x, y in [(arm, ref)] + list(zip(tg(arm), tg(ref))):
        k = f"{x} - {y}"
        g = lambda nm, c: (A[nm]["confirm"].get(k) or {}).get(c) if nm in A else None  # noqa: E731
        lp = (L or {}).get("pairs", {}).get(f"{x}:{y}", {}).get("abs_dlat2")
        md.append(f"| {k} | {f3(g('held', 'score'), 4)} | {f3(g('held', 'zero'), 1)} | {f3(g('public', 'score'), 4)} | {f3(g('public', 'zero'), 1)} | {f3(lp, 4)} |")
    r0, a0 = f"{ref}-s0", f"{arm}-s0"
    lines = []
    if "held" in A:
        hm = np.mean([A["held"]["mean"][t] for t in tg(arm) if t in A["held"]["mean"]]) if any(t in A["held"]["mean"] for t in tg(arm)) else np.nan
        rm = np.mean([A["held"]["mean"][t] for t in tg(ref) if t in A["held"]["mean"]])
        lines.append(("held mean (seed mean) >= ref - 0.01", f"{hm:.4f} vs {rm - 0.01:.4f}", bool(hm >= rm - 0.01)))
        if a0 in A["held"]["zeros"]:
            lines.append(("held zeros s0 <= ref s0 + 3", f"{A['held']['zeros'][a0]:g} vs {A['held']['zeros'][r0] + 3:g}",
                          bool(A["held"]["zeros"][a0] <= A["held"]["zeros"][r0] + 3)))
    if "public" in A and a0 in A["public"]["zeros"]:
        rz = np.mean([A["public"]["zeros"][t] for t in tg(ref)])
        lines.append(("public zeros s0 <= ref (seed mean) + 5", f"{A['public']['zeros'][a0]:g} vs {rz + 5:g}", bool(A["public"]["zeros"][a0] <= rz + 5)))
    if L and f"{arm}:{ref}" in L["pairs"]:
        v = L["pairs"][f"{arm}:{ref}"]["abs_dlat2"][0]
        lines.append((f"lateral abs dlat2, seed means, arm - ref <= +{LINE} m", f"{v:+.4f}", bool(v <= LINE)))
    md += ["", "| line | value | pass |", "|:--|:--|:--|"] + [f"| {n} | {v} | {'yes' if p else '**no**'} |" for n, v, p in lines]
    md += ["| exam entry speed (Tokyo) | read later on Tokyo | pending |", ""]
    md += ["Sets: " + ", ".join(f"{nm} {A[nm]['n']} scenes" for nm in A) + (". Lateral: " + Path(a.lat).name if L else ". Lateral: absent") + "."]
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lat")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--out", required=True)
    p.add_argument("--mean", nargs="*", default=[]), p.add_argument("--pairs", nargs="*", default=[])
    p.add_argument("--scenes", default=""), p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("table")
    p.add_argument("--held", default=""), p.add_argument("--guard", default=""), p.add_argument("--lat", default="")
    p.add_argument("--arm", required=True), p.add_argument("--ref", default="RSG-L18"), p.add_argument("--seeds", default="0,1")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(lat=cmd_lat, table=cmd_table)[a.cmd](a)
