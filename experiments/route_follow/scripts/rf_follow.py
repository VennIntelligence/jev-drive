#!/usr/bin/env python3
"""ROUTE-FOLLOW closed-loop read (research/next-round/route-night.md, stage 1 D; line L5): does the ego follow the route through turns on
stored AlpaSim reads? CPU only, read-only on the run dirs. Box, AlpaSim's venv.

  rf_follow.py --runs LABEL@18=<run dir>[,<run dir>..] ... --out DIR [--cover auto|served|offline|none] [--pairs A:B ..] [--limit N]
      -> DIR/follow.json, follow.md, scenes.csv (one row per label x scene), decisions.parquet (one row per label x scene x decision)

Per rollout (the last driver session of a scene, ao_e1.drive_records): the ego path = the driver's logged `anchor` of every decision
(local frame, 2 Hz; the stored runs keep no rollout.asl), the served plan `poses`, the fed ego row, the served memory `rmem` when the
checkpoint has the ROUTE-MEM branch. The route = AlpaSim's own RouteGenerator.create (MAP, 40 m) on the scene's after160 data source
(rm_build pool / ao_e1's way), re-generated at every logged anchor in decision order, then its full local polyline (every extension the
rollout triggered). Zero-score events from the evaluator's 2 Hz series (long_read.one): offroad, front / lateral collision, or
dist_to_gt >= 4 m (corridor); the first such step k_z ends the rollout for this read.

  turn       the route stretch the logged ego covers up to decision 17 ([s0, s_log], s0 = the first anchor's station, s_log = the log's
             own pose at t0 of decision 17 projected on the route; the same window for every checkpoint) holds a 30 m window whose net
             heading change (chord heading over +-5 m) is >= 60 deg and whose change TH over [a - 10, a + 40] m is >= 45 deg (the first such a):
             entry / exit = where 10 % / 90 % of TH is reached; left = TH > 0 (counter-clockwise). Scenes whose exit + 10 m lies beyond
             s_log are not turns of this read (the log itself did not finish the turn within the 18 decisions).
             outcome of a turn: completed | event (zero-score event before the crossing) | off (crossed > 2.5 m off, or last seen > 2.5 m
             off: took another way) | short (no event, still on the route when the rollout ended: too slow)
  completed  the ego path (anchors, linear between decisions) crosses station exit + 10 m at a decision k < k_z, and its distance to the
             route at the crossing is <= 2.5 m (projections are local: within -30 / +60 m of the previous decision's station)
  cover      memory covers +20 m (rmem.I20 valid) at the decision 20 m before the entry: the last decision with station <= entry - 20 m
             (decision 0 when the rollout starts closer). served = the logged `rmem`; offline = the same memory rebuilt from the
             regenerated messages (rm_build feats_of; G-serve showed both equal); auto = served when every decision logs `rmem`, else no
             buckets (old checkpoints: completion only)
  junction   decisions k <= k_z with station in [entry - 20, exit + 10] m: |distance to the route|, pooled over scenes (median / p90)
  plan 2 s   arc length of the served plan from the rear axle through its 0.5 .. 2.0 s poses minus the fed speed x 2 s, decisions
             k <= k_z with speed >= 1 m/s (all, and the junction stretch); reported only (decision 285's "plan shorter than the car")
CIs: log-clustered bootstrap (10 000 draws, jevdrive.stats), log = scene id without its last "-<hash>"; --pairs A:B gives the rate
difference A - B with one draw of logs applied to both (bucket members may differ between labels: the ego paths differ).
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
                str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts")]
import rmem as RM  # noqa: E402

T0, DT = 17_000, 500_000
WIN, TURN_DEG, LAT_OK, PAST, PRE, HALF = 30.0, 60.0, 2.5, 10.0, 20.0, 5.0
K_END = 17                                         # the last decision of an 18-decision read: the turn window ends at the log's pose then
N_BOOT = 10_000


def heading(R, S, step=1.0):
    """Polyline R (n, 2) with stations S -> (stations at `step` m, unwrapped heading rad of the chord over +-HALF m: the generator's 3 m lane-join
    steps read as < 20 deg, not as turns)."""
    s = np.arange(0.0, S[-1] + 1e-9, step)
    at = lambda q: np.c_[np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])]  # noqa: E731
    d = at(np.minimum(s + HALF, S[-1])) - at(np.maximum(s - HALF, 0.0))
    return s, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def find_turn(s, h, lo, hi):
    """First 30 m window inside [lo, hi] whose net heading change is >= 60 deg -> (entry, exit, TH rad) or None."""
    i0, i1 = np.searchsorted(s, lo), np.searchsorted(s, hi, side="right")
    w = int(round(WIN / (s[1] - s[0]))) if len(s) > 1 else 0
    for a in range(i0, max(i0, i1 - w)):
        if np.degrees(abs(h[a + w] - h[a])) >= TURN_DEG:
            j0, j1 = max(0, a - 10), min(len(s) - 1, a + 40)
            th = h[j1] - h[j0]
            if np.degrees(abs(th)) < 0.75 * TURN_DEG:            # a window edge on a lane-join step or an S-bend: not a turn, look further
                continue
            f = (h[j0:j1 + 1] - h[j0]) / th
            k10, k90 = np.flatnonzero(f >= 0.1), np.flatnonzero(f >= 0.9)
            return float(s[j0 + k10[0]]), float(s[j0 + k90[0]]), float(th)
    return None


def regen_full(scene, recs, cfg):
    """rm_build.regen plus the generator's full polyline after every extension -> ({k: (20, 2) rig-frame waypoints}, {k: anchor}, route0 gap
    per k, (n, 2) local-frame route polyline, the logged ego position at decision K_END)."""
    import logging
    from ao_e1 import loader
    from alpasim_runtime.config import RouteGeneratorType
    from alpasim_runtime.route_generator import RouteGenerator
    from alpasim_utils.geometry import Pose
    logging.disable(logging.WARNING)
    ds = loader(str(cfg)).get_data_source(scene)
    tr = ds.rig.trajectory
    P, Tt = np.asarray(tr.positions, float), np.asarray(tr.timestamps_us, float)
    g = RouteGenerator.create(P, vector_map=ds.map, route_generator_type=RouteGeneratorType.MAP, route_start_offset_m=40.0)
    W, A, G0 = {}, {}, {}
    for r in recs:
        x, y, yaw = r["anchor"]
        pose = Pose(np.array([x, y, np.interp(r["t0"], Tt, P[:, 2])]), np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]))
        W[r["k"]] = np.asarray(RouteGenerator.prepare_for_policy(g.generate_route(int(r["t0"]), pose)).waypoints, float)[:, :2]
        A[r["k"]] = np.asarray(r["anchor"], float)
        G0[r["k"]] = (float(np.hypot(*(W[r["k"]][0] - np.asarray(r["route0"])))) if r.get("route0") is not None and np.isfinite(W[r["k"]][0]).all()
                      else np.nan)
    t_end = T0 + DT * K_END
    log_end = np.array([np.interp(t_end, Tt, P[:, 0]), np.interp(t_end, Tt, P[:, 1])])
    return W, A, G0, np.asarray(g.route_polyline_in_local.points, float)[:, :2], log_end


def plan_arc2(poses):
    p = np.r_[[[0.0, 0.0]], np.asarray(poses, float)[:4, :2]]
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def one(job):
    """(label, run dir, scene, drive records, metric summary, cfg, cover mode) -> (scene row, decision rows)."""
    label, D, scene, recs, summ, cfg, cover = job
    import long_read as LR
    from ao_e1 import project, stations
    from rm_build import feats_of
    ev = {r["k"]: r for r in LR.one((label, D, scene))}
    kz = min([k for k, r in ev.items() if r["atf"] or (r["dist_gt"] or 0) >= 4.0], default=10 ** 6)
    W, A, G0, Rl, log_end = regen_full(scene, recs, cfg)
    keep = np.r_[True, np.hypot(*np.diff(Rl, axis=0).T) > 1e-6]
    Rl = Rl[keep]
    S = stations(Rl)
    served = all("rmem" in r for r in recs)
    mode = ("served" if served else "none") if cover == "auto" else cover
    if mode == "served" and not served:
        raise ValueError(f"{scene}: --cover served but the drive records have no rmem")
    F = feats_of(W, A) if mode == "offline" else {r["k"]: np.asarray(r["rmem"], np.float32) for r in recs} if mode == "served" else None
    dec, hint = [], None
    for r in recs:
        a = np.asarray(r["anchor"], float)
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        st, lat = project(Rl, S, a[:2], lo, hi)
        hint = st
        e = np.asarray(r["ego"], float)[:RM.EGO_DIM]
        v = float(np.hypot(e[4], e[5]) * 10.0)
        dec.append(dict(label=label, scene=scene, k=int((r["t0"] - T0) // DT), drv_k=int(r["k"]), st=st, lat=lat, v=v,
                        deficit=plan_arc2(r["poses"]) - 2.0 * v, g0=G0[r["k"]],
                        c20=(bool(F[r["k"]].reshape(RM.N_PT, RM.NF)[RM.I20, 0] > 0.5) if F is not None else None)))
    st = np.array([d["st"] for d in dec])
    ks = np.array([d["k"] for d in dec])
    s0 = st[0]
    s_end = project(Rl, S, log_end, s0 - 10.0, s0 + 150.0)[0]                # the log's station at decision K_END
    s, h = heading(Rl, S)
    t = find_turn(s, h, s0, s_end)
    row = dict(label=label, scene=scene, log=scene.rsplit("-", 1)[0], score=summ.get("score"), kz=kz if kz < 10 ** 6 else -1, decisions=len(dec),
               s0=s0, s_log_end=s_end, route_m=float(S[-1]), cover_mode=mode, turn=False)
    for d in dec:
        d["junction"] = False
    if t is not None and t[1] + PAST <= s_end:
        ent, ext, th = t
        tgt = ext + PAST
        reached, lat_x, k_x = False, np.nan, -1
        for i in range(len(dec)):                                           # first crossing of exit + 10 m, linear between anchors
            if st[i] >= tgt:
                if i == 0:
                    lat_x = abs(dec[0]["lat"])
                else:
                    w = (tgt - st[i - 1]) / max(st[i] - st[i - 1], 1e-9)
                    lat_x = abs((1 - w) * dec[i - 1]["lat"] + w * dec[i]["lat"])
                reached, k_x = ks[i] < kz, int(ks[i])
                break
        ic = np.flatnonzero(st <= ent - PRE)
        ic = int(ic[-1]) if len(ic) else 0
        for d in dec:
            d["junction"] = bool(ent - PRE <= d["st"] <= ext + PAST and d["k"] <= kz)
        ok = bool(reached and lat_x <= LAT_OK)
        last = np.flatnonzero(ks < kz)
        lat_last = abs(dec[int(last[-1])]["lat"]) if len(last) else np.nan
        outcome = ("completed" if ok else "event" if kz < 10 ** 6 and (k_x < 0 or kz <= k_x) else "off" if k_x >= 0 or lat_last > LAT_OK
                   else "short")
        row.update(turn=True, side="L" if th > 0 else "R", turn_deg=float(np.degrees(th)), entry=ent, exit=ext, k_cross=k_x, lat_cross=lat_x,
                   reached=bool(reached), completed=ok, outcome=outcome, k_cover=int(ks[ic]),
                   cover=(None if mode == "none" else bool(dec[ic]["c20"])))
    return row, dec


def rate(x, logs):
    from jevdrive import stats
    r = stats.bootstrap(np.asarray(x, float), logs, n_boot=N_BOOT)
    return dict(n=int(r["n"]), rate=r["mean"], lo=r["lo"], hi=r["hi"])


def diff_rate(T, A, B, mask_a, mask_b):
    """Completion rate A - B over their own bucket members, logs resampled once for both (ratio of sums per label)."""
    xa, xb = T[A][mask_a(T[A])], T[B][mask_b(T[B])]
    L = np.array(sorted(set(xa.log) | set(xb.log)))
    rng = np.random.default_rng(0)
    idx = rng.integers(len(L), size=(N_BOOT, len(L)))

    def per(x):
        c = np.searchsorted(L, x.log.to_numpy())
        return np.bincount(c, x.completed.astype(float), len(L)), np.bincount(c, minlength=len(L)).astype(float)
    sa, ca = per(xa)
    sb, cb = per(xb)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)
    b = b[np.isfinite(b)]
    d = xa.completed.mean() - xb.completed.mean() if len(xa) and len(xb) else np.nan
    return dict(diff=float(d), lo=float(np.quantile(b, 0.025)) if len(b) else np.nan, hi=float(np.quantile(b, 0.975)) if len(b) else np.nan,
                n_a=int(len(xa)), n_b=int(len(xb)))


def main(a):
    import pandas as pd
    import ev as EV
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(Path(a.scenes).read_text().split()) if a.scenes else None
    with Run("route_follow", "follow", config=vars(a)) as run:
        jobs, labels = [], []
        for spec in a.runs:
            label, dirs = spec.split("=", 1)
            labels.append(label)
            seen, n = set(), 0
            for D in dirs.split(","):
                summ = {r["clipgt_id"]: r for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]}
                for sc, rr in sorted(drive_records(D).items()):
                    if sc in seen or sc not in summ or (keep is not None and sc not in keep) or (a.limit and n >= a.limit):
                        continue
                    seen.add(sc)
                    n += 1
                    for r in rr:
                        r.pop("ms", None), r.pop("hist", None)
                    jobs.append((label, D, sc, rr, dict(score=summ[sc].get("score"), **(summ[sc].get("metrics") or {})), EV.CFG, a.cover))
            run.info(f"{label}: {n} rollouts from {len(dirs.split(','))} dirs")
        res = par.pmap(one, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="rollouts")
        err = {jobs[i][0] + " " + jobs[i][2]: str(e)[-300:] for i, e in res.errors.items()}
        if err:
            run.info(f"{len(err)} rollouts failed (no route: AlpaSim fails the same scenes, route folds back): {list(err.items())[:3]}")
        Sx = pd.DataFrame([v[0] for v in res.values if v is not None])
        Dx = pd.DataFrame([d for v in res.values if v is not None for d in v[1]])
        Sx.to_csv(out / "scenes.csv", index=False)
        Dx.to_parquet(out / "decisions.parquet", index=False)
        T = {lab: Sx[(Sx.label == lab) & Sx.turn].copy() for lab in labels}
        for lab in labels:
            if len(T[lab]):
                T[lab]["completed"] = T[lab].completed.astype(bool)
        Rj = dict(errors=err, labels={}, pairs={})
        buckets = {"all": lambda x: np.ones(len(x), bool), "left": lambda x: (x.side == "L").to_numpy(), "right": lambda x: (x.side == "R").to_numpy(),
                   "covered": lambda x: (x.cover == True).to_numpy(), "uncovered": lambda x: (x.cover == False).to_numpy()}  # noqa: E712
        for lab in labels:
            x, d = T[lab], Dx[Dx.label == lab]
            s_lab = Sx[Sx.label == lab]
            r = dict(rollouts=int(len(s_lab)), turns=int(len(x)), cover_mode=str(s_lab.cover_mode.iloc[0]) if len(s_lab) else "",
                     g0_share_005=float((d.g0 <= 0.05).mean()) if len(d) else np.nan)
            for b, m in buckets.items():
                if b in ("covered", "uncovered") and r["cover_mode"] == "none":
                    continue
                mm = m(x) if len(x) else np.zeros(0, bool)
                r[b] = rate(x.completed[mm].astype(float), x.log[mm]) if mm.any() else dict(n=0, rate=np.nan, lo=np.nan, hi=np.nan)
            r["outcomes"] = {o: int((x.outcome == o).sum()) if len(x) else 0 for o in ("completed", "event", "off", "short")}
            j = d[d.junction]
            r["junction_lat"] = dict(n=int(len(j)), median=float(j.lat.abs().median()) if len(j) else np.nan,
                                     p90=float(j.lat.abs().quantile(0.9)) if len(j) else np.nan)
            mv = (d.v >= 1.0) & (d.k <= d.scene.map(s_lab.set_index("scene").kz.replace(-1, 10 ** 6)))
            r["plan2s_deficit"] = dict(n=int(mv.sum()), median=float(d.deficit[mv].median()), p10=float(d.deficit[mv].quantile(0.1)),
                                       junction_median=float(d.deficit[mv & d.junction].median()) if (mv & d.junction).any() else np.nan)
            Rj["labels"][lab] = r
            run.info(f"{lab}: {json.dumps(r, default=float)[:600]}")
        for p in a.pairs or []:
            A, B = p.split(":")
            Rj["pairs"][p] = {b: diff_rate(T, A, B, buckets[b], buckets[b]) for b in buckets
                              if b in ("all", "left", "right") or (Rj["labels"][A]["cover_mode"] != "none" and Rj["labels"][B]["cover_mode"] != "none")}
        (out / "follow.json").write_text(json.dumps(Rj, indent=1, default=float) + "\n")
        f = lambda r: "-" if not r or not r.get("n") else f"{100 * r['rate']:.1f} % ({r['n']}) [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"  # noqa: E731
        md = ["# ROUTE-FOLLOW closed-loop read (rf_follow.py)", "",
              f"Turn = route net heading change >= {TURN_DEG:.0f} deg within {WIN:.0f} m inside the log's own travel to decision 17; completed = the ego crosses exit + {PAST:.0f} m "
              f"before any zero-score event, <= {LAT_OK} m from the route. Covered = the memory's +20 m point valid at the decision {PRE:.0f} m before the entry. "
              "Rates with log-clustered 95 % CI (scenes); junction = entry - 20 m .. exit + 10 m, |ego - route| over decisions; "
              "plan 2 s deficit = plan arc to 2 s - speed x 2 s (m, decisions at >= 1 m/s). Outcomes of the turns: event = a zero-score event before the crossing, "
              "off = crossed > 2.5 m from the route or last seen > 2.5 m off, short = still on the route when the rollout ended (slow).", "",
              "| label | rollouts / turns | completion all | left | right | covered | uncovered | completed / event / off / short | junction lat median / p90 (m) | plan 2 s deficit median (junction) | route0 <= 5 cm |",
              "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|--:|"]
        for lab, r in Rj["labels"].items():
            md.append(f"| {lab} | {r['rollouts']} / {r['turns']} | {f(r['all'])} | {f(r['left'])} | {f(r['right'])} | {f(r.get('covered'))} | {f(r.get('uncovered'))} | "
                      f"{' / '.join(str(v) for v in r['outcomes'].values())} | "
                      f"{r['junction_lat']['median']:.2f} / {r['junction_lat']['p90']:.2f} ({r['junction_lat']['n']}) | "
                      f"{r['plan2s_deficit']['median']:+.2f} ({r['plan2s_deficit']['junction_median']:+.2f}) | {r['g0_share_005']:.3f} |")
        if Rj["pairs"]:
            md += ["", "| pair A - B | bucket | completion difference (pp) [95 % CI] | n A / n B |", "|:--|:--|:--|:--|"]
            for p, v in Rj["pairs"].items():
                for b, q in v.items():
                    md.append(f"| {p} | {b} | {100 * q['diff']:+.1f} [{100 * q['lo']:+.1f}, {100 * q['hi']:+.1f}] | {q['n_a']} / {q['n_b']} |")
        if err:
            md += ["", f"{len(err)} rollouts without a route rebuild (excluded): " + ", ".join(sorted(err)[:10])]
        (out / "follow.md").write_text("\n".join(md) + "\n")
        run.summary.update(out=str(out), labels=Rj["labels"])
        print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=dir1,dir2 (ol_report syntax, e.g. YS-L18-s1@18=<run>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cover", default="auto", choices=("auto", "served", "offline", "none"))
    ap.add_argument("--pairs", nargs="*", help="A:B label pairs for completion differences")
    ap.add_argument("--scenes", default="", help="text file of scene ids to keep")
    ap.add_argument("--limit", type=int, default=0, help="first N rollouts per label (smoke)")
    ap.add_argument("--workers", type=int, default=16)
    main(ap.parse_args())
