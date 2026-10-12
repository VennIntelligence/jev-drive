#!/usr/bin/env python3
"""E1 (plans/2026-10-12-e1-prereg.md): does the served plan's lateral offset from the route at a near anchor (5 / 10 / 20 m ahead) warn of
corridor and offroad zeros on stored 18-decision held rollouts? CPU only, read-only on the run dirs. Box, AlpaSim's venv.

  ao_e1.py collect --runs <label>=<run dir> ... [--jobs N]     -> run dir: per_decision.parquet, per_rollout.csv
  ao_e1.py report --in <collect run dir> --md <out.md> --json <out.json>

Route: AlpaSim's own RouteGeneratorMap built as event_loop.py builds it (the scene's GT rig trajectory + the scene's vector map, start
offset 40 m), re-sampled at the driver's logged `anchor` pose of every decision; its first waypoint is checked against the logged `route0`
(gate G0). Geometry (offsets, stations) uses the generator's full local polyline after every extension the rollouts triggered.
Events from the evaluator's 2 Hz series in metrics.parquet (t = 17 000 us + 500 000 us k), as experiments/alpasim_long/scripts/long_read.py.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[3])]
from jevdrive.run import Run  # noqa: E402

T0, DT, NS = 17000, 500000, (5, 10, 20)
START_M = 40.0          # route_start_offset_m of the challenge config
_L = {}


# ---------------------------------------------------------------- geometry
def stations(R):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]


def project(R, S, p, lo=-np.inf, hi=np.inf):
    """Point p (2,) onto polyline R (n, 2) with cumulative stations S, segments restricted to [lo, hi] -> (station, signed lateral, left +)."""
    a, b = R[:-1], R[1:]
    seg = b - a
    L2 = (seg * seg).sum(1)
    ok = (L2 > 1e-8) & (S[1:] >= lo) & (S[:-1] <= hi)
    if not ok.any():
        ok = L2 > 1e-8
    a, seg, L2, s0 = a[ok], seg[ok], L2[ok], S[:-1][ok]
    t = np.clip(((p - a) * seg).sum(1) / L2, 0, 1)
    q = a + t[:, None] * seg
    d = np.hypot(*(p - q).T)
    i = int(d.argmin())
    u, v = seg[i] / np.sqrt(L2[i]), p - a[i]
    lat = float(u[0] * v[1] - u[1] * v[0])
    return float(s0[i] + t[i] * np.sqrt(L2[i])), lat


def loader(cfg):
    if cfg not in _L:
        from alpasim_runtime.config import UserSimulatorConfig
        from alpasim_runtime.scene_loader import build_scene_loader
        from omegaconf import OmegaConf
        _L[cfg] = build_scene_loader(OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(cfg)))
    return _L[cfg]


def one_scene(job):
    """(scene, cfg, {label: [drive records sorted by k]}) -> per-decision rows, or an error row."""
    scene, cfg, recs = job
    try:
        from alpasim_runtime.config import RouteGeneratorType
        from alpasim_runtime.route_generator import RouteGenerator
        from alpasim_utils.geometry import Pose
        ds = loader(cfg).get_data_source(scene)
        tr = ds.rig.trajectory
        P, Tt = np.asarray(tr.positions, float), np.asarray(tr.timestamps_us, float)
        g = RouteGenerator.create(P, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=START_M)
        g0 = {}
        for lab, rr in recs.items():
            for r in rr:
                x, y, yaw = r["anchor"]
                pose = Pose(np.array([x, y, np.interp(r["t0"], Tt, P[:, 2])]), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]))
                wp = g.generate_route(int(r["t0"]), pose).waypoints
                g0[lab, r["k"]] = (float(np.hypot(*(wp[0, :2] - np.asarray(r["route0"])))) if len(wp) and r.get("route0") is not None
                                   else np.nan)
        R = np.asarray(g.route_polyline_in_local.points, float)[:, :2]
        S = stations(R)
        rows = []
        for lab, rr in recs.items():
            hint = None
            for r in rr:
                a = np.asarray(r["anchor"], float)
                lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30, hint + 60)
                se, e0 = project(R, S, a[:2], lo, hi)
                hint = se
                c, s = np.cos(a[2]), np.sin(a[2])
                Pm = a[:2] + np.asarray(r["poses"], float)[:, :2] @ np.array([[c, s], [-s, c]])
                pr = np.array([(se, e0)] + [project(R, S, p, se - 10, se + 80) for p in Pm])
                st, la = np.maximum.accumulate(pr[:, 0]), pr[:, 1]
                row = dict(label=lab, scene=scene, k=int((r["t0"] - T0) // DT), drv_k=int(r["k"]), s_e=se, e0=e0, g0=g0[lab, r["k"]])
                for N in NS:
                    short = se + N > st[-1]
                    row[f"o{N}"] = float(la[-1] if short else np.interp(se + N, st, la))
                    row[f"short{N}"] = bool(short)
                rows.append(row)
        return rows
    except Exception as e:  # noqa: BLE001  (counted against G1)
        return [dict(label="__error__", scene=scene, err=repr(e)[:300])]


def drive_records(run):
    S = {}
    for line in open(Path(run) / "driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "start":
            S[r["session"]] = (r["scene"], r["n"], [])
        elif r["kind"] == "drive" and r["session"] in S:
            S[r["session"]][2].append(r)
    best = {}
    for sc, n, rr in S.values():                     # the last session of a scene (a retried rollout), ol_mech.py's rule
        if rr and (sc not in best or n > best[sc][0]):
            best[sc] = (n, sorted(rr, key=lambda r: r["k"]))
    return {sc: rr for sc, (n, rr) in best.items()}


def series(job):
    label, run, scene, rid = job
    import polars as pl
    f = glob.glob(f"{run}/rollouts/{scene}/{rid}/metrics.parquet")
    if not f:
        return []
    m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(["dist_to_gt_trajectory", "offroad", "collision_any"])).select(
        "name", pl.col("timestamps_us").cast(pl.Int64), "values")
    ser = {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}
    ks = sorted((t - T0) // DT for t in ser["dist_to_gt_trajectory"] if t >= T0 and (t - T0) % DT == 0)
    return [dict(label=label, scene=scene, k=int(k), dist_gt=ser["dist_to_gt_trajectory"][T0 + DT * k],
                 offroad=(ser.get("offroad", {}).get(T0 + DT * k) or 0) > 0, coll=(ser.get("collision_any", {}).get(T0 + DT * k) or 0) > 0)
            for k in ks]


def collect(a):
    import pandas as pd
    from jevdrive.data import splits
    held = splits.load("alpasim_nuplan/onpol-held")
    runs = dict(s.split("=", 1) for s in a.runs)
    with Run("anchor_offset", "e1", config=vars(a)) as run:
        run.use_split(held)
        keep = set(held.members)
        roll, sjobs, recs, cfg = [], [], {}, None
        for lab, D in runs.items():
            cfg = cfg or str(Path(D) / "generated-user-config-0.yaml")
            for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]:
                s = r["clipgt_id"]
                if s not in keep:
                    continue
                mm = r.get("metrics") or {}
                roll.append(dict(label=lab, scene=s, run=D, rollout_id=r.get("rollout_id"), score=r.get("score"),
                                 **{n: mm.get(n) for n in ("collision_at_fault", "offroad", "left_corridor_laterally")}))
                sjobs.append((lab, D, s, r.get("rollout_id")))
            for sc, rr in drive_records(D).items():
                if sc in keep:
                    recs.setdefault(sc, {})[lab] = rr
        run.info(f"{len(roll)} scored rollouts, {len(recs)} scenes with driver records, scene config {cfg}")
        with ProcessPoolExecutor(a.jobs) as ex:
            K = pd.DataFrame([x for xx in ex.map(series, sjobs, chunksize=16) for x in xx])
            rows = [x for xx in run.tqdm(ex.map(one_scene, [(sc, cfg, v) for sc, v in sorted(recs.items())], chunksize=1), total=len(recs),
                                         desc="scenes") for x in xx]
        D = pd.DataFrame(rows)
        err = D[D.label == "__error__"]
        D = D[D.label != "__error__"].drop(columns=[c for c in ("err",) if c in D])
        D = D.merge(K, on=["label", "scene", "k"], how="left")
        out = Path(run.dir)
        D.to_parquet(out / "per_decision.parquet", index=False)
        pd.DataFrame(roll).to_csv(out / "per_rollout.csv", index=False)
        err.to_csv(out / "errors.csv", index=False)
        run.summary.update(rollouts=len(roll), scenes=len(recs), decisions=len(D), scene_errors=len(err),
                           g0_le_005=float((D.g0 <= 0.05).mean()), g0_nan=int(D.g0.isna().sum()))
        run.info(json.dumps(run.summary))


# ---------------------------------------------------------------- report
def report(a):
    import pandas as pd
    I = Path(a.inp)
    D, Rl = pd.read_parquet(I / "per_decision.parquet"), pd.read_csv(I / "per_rollout.csv")
    nerr = len(pd.read_csv(I / "errors.csv")) if (I / "errors.csv").stat().st_size > 1 else 0
    labels = sorted(Rl.label.unique())
    # scenes with a scored rollout in every label (ol_report.py: route-fold scenes have none)
    has = Rl[Rl.collision_at_fault.notna()].groupby("scene").label.nunique()
    scenes = set(has[has == len(labels)].index)
    Rl, D = Rl[Rl.scene.isin(scenes)].copy(), D[D.scene.isin(scenes)].copy()
    out = dict(labels=labels, scenes=len(scenes), rollouts=len(Rl), scene_errors=nerr)
    # G0, G1
    out["G0"] = dict(decisions=int(D.g0.notna().sum()), share_le_005=float((D.g0 <= 0.05).mean()), p99=float(D.g0.quantile(0.99)),
                     max=float(D.g0.max()), nan=int(D.g0.isna().sum()))
    out["G0"]["pass"] = out["G0"]["share_le_005"] >= 0.99
    n_dec = D.groupby(["label", "scene"]).size()
    full = n_dec.reindex(pd.MultiIndex.from_frame(Rl[["label", "scene"]])).fillna(0)
    ok = D.groupby(["label", "scene"])[[f"o{N}" for N in NS]].apply(lambda x: bool(x.notna().all().all()))
    out["G1"] = dict(rollouts_with_records=int((full > 0).sum()), rollouts_all_offsets=int(ok.reindex(full.index).fillna(False).sum()))
    out["G1"]["share"] = out["G1"]["rollouts_all_offsets"] / len(Rl)
    out["G1"]["pass"] = out["G1"]["share"] >= 0.95
    # events
    D["b_route"], D["b_log"] = D.e0.abs(), D.dist_gt
    for N in NS:
        D[f"s{N}"] = D[f"o{N}"].abs()
    g = D.sort_values("k").groupby(["label", "scene"])
    kc = g.apply(lambda x: x.k[x.dist_gt >= 4.0].min(), include_groups=False).rename("k_c")
    ko = g.apply(lambda x: x.k[x.offroad.astype(bool)].min(), include_groups=False).rename("k_o")
    Rl = Rl.join(kc, on=["label", "scene"]).join(ko, on=["label", "scene"])
    z = Rl.score == 0
    Rl["corr"] = z & (Rl.left_corridor_laterally > 0)
    Rl["offr"] = z & (Rl.offroad > 0)
    Rl["clean"] = ~((Rl.left_corridor_laterally > 0) | (Rl.offroad > 0) | (Rl.collision_at_fault > 0))
    Rl["log"] = Rl.scene.str.rsplit("-", n=1).str[0]
    Rl["fam"] = Rl.label.str.replace(r"-s\d+$", "", regex=True)
    logs = np.array(sorted(Rl.log.unique()))
    li = {l: i for i, l in enumerate(logs)}
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(d, minlength=len(logs)) for d in rng.integers(0, len(logs), (10000, len(logs)))]).astype(np.float32)
    Dk = {key: x for key, x in D.groupby(["label", "scene"])}
    measures = [f"s{N}" for N in NS] + ["b_route", "b_log"]

    def wauc(sc, y, w):
        """Weighted AUC (ties count 1/2): scores sc (n,), labels y (n,) bool, weights w (B, n) -> (B,)."""
        o = np.argsort(sc, kind="stable")
        sc, y, w = sc[o], y[o], w[:, o]
        first = np.unique(sc, return_index=True)[1]
        gn, gp = np.add.reduceat(w * ~y, first, axis=1), np.add.reduceat(w * y, first, axis=1)
        num = (gp * (np.cumsum(gn, 1) - gn + 0.5 * gn)).sum(1)
        return num / (gp.sum(1) * gn.sum(1))

    res = {}
    for ev, kcol in (("corridor", "k_c"), ("offroad", "k_o")):
        pos = Rl[Rl["corr" if ev == "corridor" else "offr"] & Rl[kcol].notna()]
        P_ok = pos[pos[kcol] >= 2]
        neg = Rl[Rl.clean]
        res[ev] = dict(events=int(len(pos)), events_k_le_1=int((pos[kcol] < 2).sum()), clean=int(len(neg)),
                       by_family={f: int(n) for f, n in P_ok.fam.value_counts().items()})
        for m in measures:
            def score(r, upto):
                x = Dk.get((r.label, r.scene))
                if x is None:
                    return np.nan
                x = x[x.k < upto] if upto is not None else x
                return float(x[m].max()) if len(x) else np.nan
            sp = np.array([score(r, getattr(r, kcol)) for r in P_ok.itertuples()])
            sn = np.array([score(r, None) for r in neg.itertuples()])
            sc = np.r_[sp, sn]
            y = np.r_[np.ones(len(sp), bool), np.zeros(len(sn), bool)]
            lg = np.r_[[li[l] for l in P_ok.log], [li[l] for l in neg.log]]
            keepm = ~np.isnan(sc)
            sc, y, lg = sc[keepm], y[keepm], lg[keepm]
            auc = float(wauc(sc, y, np.ones((1, len(sc)), np.float32))[0])
            bs = np.concatenate([wauc(sc, y, W[i:i + 500][:, lg]) for i in range(0, len(W), 500)])
            tau = float(np.nanpercentile(sn, 95))
            leads, dirs = [], []
            for r in P_ok.itertuples():
                x = Dk.get((r.label, r.scene))
                if x is None:
                    leads.append(np.nan)
                    continue
                pre = x[x.k < getattr(r, kcol)]
                hit = pre[pre[m] >= tau]
                if len(hit):
                    kw = int(hit.k.min())
                    leads.append((getattr(r, kcol) - kw) * 0.5)
                    if m.startswith("s"):
                        at = x[x.k >= getattr(r, kcol)]
                        e_ev = float(at.e0.iloc[0]) if len(at) else float(x.e0.iloc[-1])
                        dirs.append(np.sign(hit.sort_values("k")[f"o{m[1:]}"].iloc[0]) == np.sign(e_ev))
                else:
                    leads.append(np.nan)
            leads = np.array(leads)
            res[ev][m] = dict(n_pos=int(y.sum()), n_neg=int((~y).sum()), auc=auc, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              tau=tau, recall_lead15=float(np.mean(leads >= 1.5)), warned=float(np.mean(~np.isnan(leads))),
                              lead_median_s=float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else None,
                              direction=float(np.mean(dirs)) if dirs else None)
            for f in sorted(P_ok.fam.unique()):
                mf = np.r_[(P_ok.fam == f).to_numpy(), (neg.fam == f).to_numpy()][keepm]
                res[ev][m].setdefault("auc_by_family", {})[f] = float(wauc(sc[mf], y[mf], np.ones((1, mf.sum()), np.float32))[0])
    # E3: does the plan shrink the offset at N m when the car is >= 1 m off the route
    e3 = {}
    posr = Rl[(Rl.corr | Rl.offr)]
    pre = []
    for r in posr.itertuples():
        x = Dk.get((r.label, r.scene))
        if x is None:
            continue
        kev = np.nanmin([r.k_c if r.corr else np.nan, r.k_o if r.offr else np.nan])
        pre.append(x[x.k < kev])
    pre = pd.concat(pre) if pre else D.iloc[:0]
    cl = pd.concat([Dk[(r.label, r.scene)] for r in Rl[Rl.clean].itertuples() if (r.label, r.scene) in Dk])
    for N in NS:
        f = lambda x: x[x.e0.abs() >= 1.0]  # noqa: E731
        a_, b_ = f(pre), f(cl)
        e3[N] = dict(pre_n=int(len(a_)), pre_share=float((a_[f"o{N}"].abs() >= a_.e0.abs()).mean()) if len(a_) else None,
                     clean_n=int(len(b_)), clean_share=float((b_[f"o{N}"].abs() >= b_.e0.abs()).mean()) if len(b_) else None)
    out["events"], out["E3"] = res, {str(k): v for k, v in e3.items()}
    out["describe"] = {f"short{N}": float(D[f"short{N}"].mean()) for N in NS} | {
        f"visible{N}": float((D.s_e - D.groupby(["label", "scene"]).s_e.transform("min") >= START_M - N).mean()) for N in NS}
    # verdicts (prereg section 4)
    def e2_pass(ev):
        best = None
        for N in NS:
            r = res[ev][f"s{N}"]
            c = (r["ci"][0] >= 0.75, r["recall_lead15"] >= 0.5, r["auc"] - max(res[ev]["b_route"]["auc"], res[ev]["b_log"]["auc"]) >= 0.05,
                 (r["direction"] or 0) >= 0.8)
            if all(c):
                best = best or N
            r["e2_lines"] = list(map(bool, c))
        return best
    gates = out["G0"]["pass"] and out["G1"]["pass"] and nerr <= 0.05 * len(scenes)
    n_c, n_o = e2_pass("corridor"), e2_pass("offroad")
    out["E2"] = "room (corridor), N = %d" % n_c if n_c else ("offroad only, N = %d" % n_o if n_o else "closed")
    e3ok = [N for N in (5, 10) if e3[N]["pre_share"] is not None and e3[N]["pre_share"] >= 0.5 and e3[N]["pre_share"] - e3[N]["clean_share"] >= 0.2]
    out["E3_verdict"] = f"room, N = {e3ok[0]}" if e3ok else "closed"
    out["gates_pass"] = bool(gates)
    Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    # markdown
    L = [f"# E1: near-anchor plan offset as a drift warning (held {len(scenes)} scenes x {len(labels)} labels, 18 decisions)", "",
         f"Registered in [plans/2026-10-12-e1-prereg.md](../plans/2026-10-12-e1-prereg.md); script `scripts/ao_e1.py`; labels: {', '.join(labels)}.", "",
         f"G0 route rebuild: {out['G0']['share_le_005']:.4f} of {out['G0']['decisions']} decisions within 0.05 m of the logged route0 "
         f"(p99 {out['G0']['p99']:.3f} m, max {out['G0']['max']:.3f} m) -> {'pass' if out['G0']['pass'] else 'FAIL'}. "
         f"G1 coverage: {out['G1']['share']:.4f} of rollouts with every offset -> {'pass' if out['G1']['pass'] else 'FAIL'}; scene errors {nerr}.", ""]
    for ev in ("corridor", "offroad"):
        r = res[ev]
        L += [f"## {ev} zeros: {r['events']} events ({r['events_k_le_1']} at k <= 1 excluded), {r['clean']} clean rollouts; by family {r['by_family']}", "",
              "| score | AUC [95% CI, log-clustered] | threshold at 5% clean false alarms (m) | warned before the event | warned >= 1.5 s ahead | median lead (s) | direction agrees | AUC YNONE / YR / YS |",
              "|:--|:--|--:|--:|--:|--:|--:|:--|"]
        for m in measures:
            x = r[m]
            fam = " / ".join(f"{x['auc_by_family'].get(f, np.nan):.3f}" for f in ("YNONE-L18", "YR10m10-F", "YS-L18"))
            L.append(f"| {m} | {x['auc']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}] | {x['tau']:.2f} | {x['warned']:.2f} | **{x['recall_lead15']:.2f}** | "
                     f"{x['lead_median_s'] if x['lead_median_s'] is None else round(x['lead_median_s'], 1)} | "
                     f"{'-' if x['direction'] is None else round(x['direction'], 2)} | {fam} |")
        L.append("")
    L += ["## Does the plan shrink the offset (E3 line): decisions with the car >= 1 m off the route", "",
          "| N | pre-event decisions | share with plan offset at N >= car offset | clean decisions | share |", "|--:|--:|--:|--:|--:|"]
    for N in NS:
        x = e3[N]
        L.append(f"| {N} m | {x['pre_n']} | {x['pre_share'] if x['pre_share'] is None else round(x['pre_share'], 3)} | {x['clean_n']} | "
                 f"{x['clean_share'] if x['clean_share'] is None else round(x['clean_share'], 3)} |")
    L += ["", "Describe: " + ", ".join(f"{k} {v:.3f}" for k, v in out["describe"].items()) +
          " (short = the plan does not reach the anchor station; visible = the anchor was already in a route message the driver received).", "",
          f"**Verdicts (registered lines):** gates {'pass' if gates else 'FAIL'}; E2: {out['E2']}; E3: {out['E3_verdict']}."]
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 8))
    r = sp.add_parser("report")
    r.add_argument("--in", dest="inp", required=True), r.add_argument("--md", required=True), r.add_argument("--json", required=True)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else report(a)
