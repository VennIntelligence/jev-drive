#!/usr/bin/env python3
"""ACC-GAP diagnosis (experiments/route_follow/plans/2026-10-16-acc-gap-prereg.md): how the lead limit of the PAI ACC serve
(serve_fix JEV_LEAD inside JEV_ACC) sets the following gap, against the human at the same time and the same place, on a stored PAI 281 read.
CPU only; run dirs are read, never written.

  slim   (box, repo .venv)  rf_accgap.py slim --runs <read>/runs/* --out <pkl>
         every decision of every session (driver/drive.jsonl, infer records): k, now, fed speed, fix.lead (filtered probabilities, d, vl,
         a_mpc, a_e2e, src, cut) and fix.acc (won, near, v_set, v_lead, v15, v_model, guard, v_curve)
  table  (Mac, .venv)  rf_accgap.py table --tax <rf_tax extract pkl> --dec <slim pkl> --out <parquet>
         per decision: the nearest vehicle in AlpaSim's route corridor 0 .. 60 m ahead of the front bumper (rf_londiag.corridor_vehicles:
         stationary, or moving in the route's direction) = the true lead, its gap (front bumper to its rear, m) and speed; the human's gap
         to the same actor (a) at the same time (the log's ego pose at t: the actors replay the log, so gap_e - gap_h is how far the ego
         lags the human behind that lead) and (b) when the log was at the ego's station; valid = before the first zero-score event /
         rear contact (rf_londiag.t_trunc)
  report (Mac, .venv)  rf_accgap.py report --table <parquet> [--table-ref <parquet> --label-ref g10] --out <md> [--subset <txt>]
         following = valid decisions, true lead moving >= 1 m/s, gap <= 40 m, ego >= 2 m/s; per-scene medians of the time gap
         (gap / own speed) ego vs human, by winning term; lead-term decisions far above the human (ratio >= 1.5); phantom leads (serve
         lead present, filtered p > 0.5, and no true vehicle within its d + 15 m); missed leads (true lead <= 30 m, serve p <= 0.5);
         the model's lead distance error d - true gap; the following subset of the prereg (--subset writes it).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
D_LEAD, D_FOLLOW, V_LEAD_MOV, V_EGO_MIN = 60.0, 40.0, 1.0, 2.0
FAR = 1.5                                  # "far above the human": ego time gap >= 1.5 x the human's
PHANTOM_M = 15.0                           # a serve lead with no true vehicle within its distance + 15 m


def slim_one(f):
    S = {}
    for line in open(f):
        if '"infer": true' not in line:
            continue
        r = json.loads(line)
        fx = r.get("fix") or {}
        L, A = fx.get("lead") or {}, fx.get("acc") or {}
        nan = lambda x: np.nan if x is None else float(x)  # noqa: E731
        p = L.get("p") or [np.nan, np.nan]
        S.setdefault(r["session"], dict(scene=r["scene"], rows=[]))["rows"].append(
            (int(r["k"]), int(r["now"]), nan(fx.get("v0")), nan(p[0]), nan(L.get("d")), nan(L.get("vl")), nan(L.get("a_mpc")), nan(L.get("a_e2e")),
             L.get("src") == "mpc", nan(L.get("cut")), str(A.get("won") or ""), str(A.get("near") or ""), nan(A.get("v_set")), nan(A.get("v_lead")),
             nan(A.get("v15")), nan(A.get("v_model")), str(A.get("guard") or ""), nan(A.get("v_curve")), nan(L.get("vm"))))
    cols = ("k", "now", "v0", "p0", "d", "vl", "a_mpc", "a_e2e", "mpc", "cut", "won", "near", "v_set", "v_lead", "v15", "v_model", "guard", "v_curve", "vm")
    out = {}
    for s, d in S.items():
        c = list(zip(*sorted(d["rows"], key=lambda x: x[0])))
        out[s] = dict(scene=d["scene"], **{k: np.array(v) for k, v in zip(cols, c)})
    return out


def slim(a):
    from concurrent.futures import ProcessPoolExecutor
    fs = [str(f) for r in a.runs for f in (Path(r) / "driver/drive.jsonl", Path(r) / "driver-logs/drive.jsonl") if f.exists()]
    fs = list(dict.fromkeys(os.path.realpath(f) for f in fs))
    out = {}
    with ProcessPoolExecutor(max(1, min(a.workers, len(fs)))) as ex:
        for d in ex.map(slim_one, fs):
            out.update(d)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(len(fs), "drive logs,", len(out), "sessions ->", a.out, flush=True)


def _mods():
    sys.path.insert(0, str(HERE))
    import rf_londiag as LD
    import rf_tax as TX
    return TX, LD


def roll_rows(job):
    o, d = job
    TX, LD = _mods()
    if "ctrl" not in o or o.get("ser") is None:
        return []
    row, _, _ = TX.classify(o, None, None)
    r = TX.Roll(o)
    rt = LD.Route(r.route, [], TX)
    tr = LD.t_trunc(TX, o, row, r)
    t = d["now"].astype(float)
    tc = np.clip(t, r.t[0], r.t[-1])
    P = np.c_[np.interp(tc, r.t, r.x), np.interp(tc, r.t, r.y)]
    _, s_rt, lat_rt, _ = rt.proj(P)
    s_log = r.log_station(P)
    LT = r.LT
    LS = r.LS[:len(LT)]
    # the human at the same time: the log pose at t (the log clock = the evaluator's)
    sl_t = np.interp(tc, LT, LS)
    Ph_t = np.c_[np.interp(sl_t, r.LS, r.LP[:, 0]), np.interp(sl_t, r.LS, r.LP[:, 1])]
    sh_t = rt.proj(Ph_t)[1]
    rows = []
    for n in range(len(t)):
        sf = s_rt[n] + r.front
        V = LD.corridor_vehicles(r, rt, tc[n], sf, D_LEAD)
        g = dict(scene=o["scene"], k=int(d["k"][n]), t=(t[n] - r.t0) / 1e6, valid=bool(t[n] <= tr), v=float(d["v0"][n]), v_log_t=float(np.interp(tc[n], LT, r.Lv[:len(LT)])),
                 v_log_s=float(np.interp(s_log[n], r.LS, r.Lv)) if len(r.Lv) == len(r.LS) else np.nan, lag=float(sl_t[n] - s_log[n]), route_lat=float(lat_rt[n]),
                 p0=float(d["p0"][n]), d=float(d["d"][n]), vl=float(d["vl"][n]), a_mpc=float(d["a_mpc"][n]), a_e2e=float(d["a_e2e"][n]),
                 mpc=bool(d["mpc"][n]), cut=float(d["cut"][n]), won=str(d["won"][n]), near=str(d["near"][n]), v_set=float(d["v_set"][n]),
                 v_lead=float(d["v_lead"][n]), v15=float(d["v15"][n]), v_model=float(d["v_model"][n]), guard=str(d["guard"][n]),
                 vm=float(d["vm"][n]) if "vm" in d else np.nan,
                 gap=np.nan, veh_v=np.nan, veh_id="", gap_h_t=np.nan, gap_h_s=np.nan, v_h_s=np.nan)
        if V:
            gap, vv, vid, _ = V[0]
            g.update(gap=gap, veh_v=vv, veh_id=vid)
            st = r.objs_at(tc[n])
            k = np.flatnonzero(st["id"] == vid)
            if len(k):
                so = float(rt.proj(np.c_[st["x"][k[0]], st["y"][k[0]]])[1][0])
                g["gap_h_t"] = so - st["L"][k[0]] / 2 - (float(sh_t[n]) + r.front)
            tl = float(np.interp(s_log[n], LS, LT))               # same place: when the log was at the ego's station
            if r.tg[0] <= tl <= r.tg[-1] and s_log[n] < r.LS[-1] - 0.5:
                st = r.objs_at(tl)
                k = np.flatnonzero(st["id"] == vid)
                if len(k):
                    Ph = np.c_[np.interp(s_log[n], r.LS, r.LP[:, 0]), np.interp(s_log[n], r.LS, r.LP[:, 1])]
                    sh, so = float(rt.proj(Ph)[1][0]), float(rt.proj(np.c_[st["x"][k[0]], st["y"][k[0]]])[1][0])
                    g.update(gap_h_s=so - st["L"][k[0]] / 2 - (sh + r.front), v_h_s=float(np.interp(tl, LT, r.Lv[:len(LT)])))
        rows.append(g)
    return rows


def human_rows(o):
    """The human's own following on the log: per 0.5 s step of the log clock, the nearest vehicle in the logged path's corridor 0 .. 60 m
    ahead of the human's front bumper (corridor_vehicles on the log path), its gap and speed, and the human's speed."""
    TX, LD = _mods()
    if "ctrl" not in o or o.get("ser") is None:
        return []
    r = TX.Roll(o)
    rt = LD.Route(r.LP, [], TX)
    LT, LS = r.LT, r.LS[:len(r.LT)]
    rows = []
    for t in r.tg:
        if not (LT[0] <= t <= LT[-1]):
            continue
        s_h = float(np.interp(t, LT, LS))
        if s_h >= r.LS[-1] - 0.5:
            continue
        V = LD.corridor_vehicles(r, rt, t, s_h + r.front, D_LEAD)
        g = dict(scene=o["scene"], t=(t - r.t0) / 1e6, v_h=float(np.interp(t, LT, r.Lv[:len(LT)])), gap=np.nan, veh_v=np.nan)
        if V:
            g.update(gap=V[0][0], veh_v=V[0][1])
        rows.append(g)
    return rows


class _LogShim:
    """What corridor_vehicles reads from rf_tax.Roll, for a log without a rollout: objects on a 0.5 s grid, the ego half width."""

    def __init__(self, tg, traffic, hw):
        self.tg, self.tr, self.hw = tg, traffic, hw

    def objs_at(self, t):
        return self.tr[int(np.clip(np.searchsorted(self.tg, t), 0, len(self.tg) - 1))]


def humanlog_scene(job):
    """(scene, cfg) -> the human's following steps on the log alone (no rollout): log clock 0.5 s grid."""
    scene, cfg = job
    TX, LD = _mods()
    ds = TX.loader(cfg).get_data_source(scene)
    tr = ds.rig.trajectory
    T, P = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)[:, :2]
    vc = ds.rig.vehicle_config
    front, hw = float(vc.aabb_x_offset_m + vc.aabb_x_m), float(vc.aabb_y_m) / 2
    LP = TX.dedupe(P)
    if len(LP) < 3:
        return []
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    v = np.gradient(S, T * 1e-6)
    tg = T[::5]
    LogShim = _LogShim(tg, TX.traffic_at(TX.logged_objects(ds), tg), hw)
    rt = LD.Route(LP, [], TX)
    s_rt = rt.proj(P[::5])[1]
    rows = []
    for n, t in enumerate(tg):
        V = LD.corridor_vehicles(LogShim, rt, t, float(s_rt[n]) + front, D_LEAD)
        g = dict(scene=scene, t=(t - T[0]) / 1e6, v_h=float(v[::5][n]), gap=np.nan, veh_v=np.nan)
        if V:
            g.update(gap=V[0][0], veh_v=V[0][1])
        rows.append(g)
    return rows


def humanlog(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    scenes = [x for x in Path(a.scenes).read_text().split() if x]
    rows, err = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(humanlog_scene, (s, a.cfg)): s for s in scenes}
        for f in futs:
            try:
                rows += f.result()
            except Exception as e:  # noqa: BLE001
                err.append((futs[f], repr(e)[:200]))
    print(len(scenes), "scenes,", len(err), "errors", err[:3])
    fit(pd.DataFrame(rows), a.out)


def fit(H, out):
    import pandas as pd
    H.to_parquet(out)
    st = H[(H.gap > 0.5) & (H.gap <= D_FOLLOW) & (H.veh_v >= V_LEAD_MOV) & ((H.v_h - H.veh_v).abs() < 1.0) & (H.v_h >= 1.0)]
    c = np.linalg.lstsq(np.c_[np.ones(len(st)), st.v_h], st.gap, rcond=None)[0]
    print(f"{H.scene.nunique()} scenes, {len(H)} steps; steady following {len(st)} steps in {st.scene.nunique()} scenes: gap = {c[0]:.2f} + {c[1]:.3f} v")
    b = pd.cut(st.v_h, [1, 3, 5, 8, 12, 18, 40])
    print(st.groupby(b, observed=True).agg(n=("gap", "size"), scenes=("scene", "nunique"), gap=("gap", "median"), v=("v_h", "median")).round(2).to_string())
    m = st.groupby(b, observed=True).agg(gap=("gap", "median"), v=("v_h", "median"))
    print("openpilot 6 + 1.45 v at the bin medians:", (6 + 1.45 * m.v).round(2).tolist())
    ss = H[(H.v_h < 0.3) & (H.gap <= 20)]
    print(f"standstill gap median {ss.gap.median():.2f} m (n {len(ss)}, {ss.scene.nunique()} scenes)")


def human(a):
    """Fit the human's following law on the logs of the given extracts (train scenes only): steady following (true lead moving >= 1 m/s,
    gap <= 40 m, |v_h - v_lead| < 1 m/s, v_h >= 1) by speed bin, least squares gap = s0 + T v, and the standstill gap (v_h < 0.3,
    lead <= 20 m)."""
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    seen, rolls = set(), []
    for f in a.tax:
        for o in pickle.load(open(f, "rb"))["rolls"]:
            if o["scene"] not in seen and "ctrl" in o:
                seen.add(o["scene"])
                rolls.append(o)
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for rr in ex.map(human_rows, rolls, chunksize=2):
            rows += rr
    fit(pd.DataFrame(rows), a.out)


def table(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    rolls = pickle.load(open(a.tax, "rb"))["rolls"]
    D = pickle.load(open(a.dec, "rb"))
    jobs = [(o, D[o["rid"]]) for o in rolls if o["rid"] in D]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for rr in ex.map(roll_rows, jobs, chunksize=2):
            rows += rr
    T = pd.DataFrame(rows)
    T.to_parquet(a.out)
    print(len(jobs), "rollouts,", len(T), "decisions ->", a.out)


def report(a):
    import pandas as pd
    T = pd.read_parquet(a.table)
    md = [f"# ACC-GAP diagnosis: {a.label}", "", f"table `{a.table}`, {T.scene.nunique()} scenes, {len(T)} decisions, valid {int(T.valid.sum())}", ""]
    V = T[T.valid]
    md += ["## Winning term at 1.5 s (valid decisions)", "", "| term | share |", "|:--|--:|"]
    md += [f"| {k} | {x:.3f} |" for k, x in V.won.value_counts(normalize=True).items()]
    F = V[(V.veh_v >= V_LEAD_MOV) & (V.gap <= D_FOLLOW) & (V.v >= V_EGO_MIN)].copy()
    F["th_e"] = F.gap / F.v
    F["th_h"] = F.gap_h_s / F.v_h_s
    F["th_ht"] = F.gap_h_t / F.v_log_t
    F["r"] = F.th_e / F.th_h
    md += ["", f"## Following decisions (true moving lead <= {D_FOLLOW:g} m, ego >= {V_EGO_MIN:g} m/s): {len(F)} in {F.scene.nunique()} scenes", ""]
    md += ["| winning term | decisions | scenes | ego time gap s (median) | human same place s | ratio median | ratio >= 1.5 | ego gap m | human gap m (same t) | ego lags human m |",
           "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, x in [("all", F)] + [(k, F[F.won == k]) for k in ("lead", "set", "ramp", "model", "curve")]:
        if not len(x):
            continue
        y = x[np.isfinite(x.r) & (x.v_h_s >= V_EGO_MIN)]
        md.append(f"| {k} | {len(x)} | {x.scene.nunique()} | {x.th_e.median():.2f} | {y.th_h.median():.2f} | {y.r.median():.2f} | "
                  f"{(y.r >= FAR).mean():.2f} | {x.gap.median():.1f} | {x.gap_h_t.median():.1f} | {x.lag.median():.1f} |")
    # per scene
    y = F[np.isfinite(F.r) & (F.v_h_s >= V_EGO_MIN)]
    sc = y.groupby("scene").agg(n=("k", "size"), th_e=("th_e", "median"), th_h=("th_h", "median"), lead=("won", lambda w: (w == "lead").mean()))
    sc["ratio"] = sc.th_e / sc.th_h
    md += ["", f"Per scene (>= 5 paired following decisions): {int((sc.n >= 5).sum())} scenes; median ego {sc[sc.n >= 5].th_e.median():.2f} s, "
           f"human {sc[sc.n >= 5].th_h.median():.2f} s, ratio median {sc[sc.n >= 5].ratio.median():.2f}"]
    # perception
    L = F[F.p0 > 0.5]
    md += ["", "## The model's lead against the true lead (following decisions)", "",
           f"- serve lead present (filtered p > 0.5): {len(L)} / {len(F)} ({len(L) / max(len(F), 1):.2f})",
           f"- distance error d - true gap: median {np.nanmedian(L.d - L.gap):+.2f} m, p10 {np.nanpercentile(L.d - L.gap, 10):+.2f}, p90 {np.nanpercentile(L.d - L.gap, 90):+.2f}",
           f"- relative speed error (vl - v_lead_true): median {np.nanmedian(L.vl - L.veh_v):+.2f} m/s, p10 {np.nanpercentile(L.vl - L.veh_v, 10):+.2f}, p90 {np.nanpercentile(L.vl - L.veh_v, 90):+.2f}"]
    for lo, hi in ((0, 15), (15, 25), (25, 40)):
        z = L[(L.gap >= lo) & (L.gap < hi)]
        if len(z):
            md.append(f"  - true gap {lo}-{hi} m: n {len(z)}, d - gap median {np.nanmedian(z.d - z.gap):+.2f} m, vl - v median {np.nanmedian(z.vl - z.veh_v):+.2f}")
    # phantom / missed
    P = V[(V.p0 > 0.5) & np.isfinite(V.d)]
    ph = P[~(P.gap <= P.d + PHANTOM_M)]
    md += ["", "## Phantom and missed leads (valid decisions)", "",
           f"- serve lead present: {len(P)} decisions; no true corridor vehicle within d + {PHANTOM_M:g} m: {len(ph)} ({len(ph) / max(len(P), 1):.2f}), "
           f"lead term wins on {int((ph.won == 'lead').sum())} of them, in {ph[ph.won == 'lead'].scene.nunique()} scenes",
           f"- true moving lead <= 30 m and serve p <= 0.5: {int(((V.gap <= 30) & (V.veh_v >= V_LEAD_MOV) & (V.p0 <= 0.5)).sum())}"]
    # lead wins with no true lead / stationary
    W = V[V.won == "lead"]
    md += ["", "## Decisions the lead term wins", "",
           f"- {len(W)} decisions in {W.scene.nunique()} scenes; true lead moving <= 40 m {int(((W.veh_v >= V_LEAD_MOV) & (W.gap <= 40)).sum())}, "
           f"stationary vehicle <= 40 m {int(((W.veh_v < V_LEAD_MOV) & (W.gap <= 40)).sum())}, none within 40 m {int((~(W.gap <= 40)).sum())}",
           f"- with no true vehicle within 40 m: serve p > 0.5 {int(((~(W.gap <= 40)) & (W.p0 > 0.5)).sum())}, p <= 0.5 (the MPC's no-lead "
           f"acceleration limit) {int(((~(W.gap <= 40)) & (W.p0 <= 0.5)).sum())}",
           f"- served speed at 1.5 s below the set speed by: median {np.nanmedian(W.v_set - W.v15):.2f} m/s"]
    # speed-binned time gap: ego vs openpilot's steady state vs human
    md += ["", "## Time gap by speed (following decisions where lead wins)", "", "| ego speed m/s | n | ego s | human same place s | openpilot steady state (1.45 + 6 / v) s |",
           "|:--|--:|--:|--:|--:|"]
    Wf = F[F.won == "lead"]
    for lo, hi in ((2, 5), (5, 8), (8, 12), (12, 18), (18, 40)):
        z = Wf[(Wf.v >= lo) & (Wf.v < hi)]
        zz = z[np.isfinite(z.th_h) & (z.v_h_s >= V_EGO_MIN)]
        if len(z):
            vm = z.v.median()
            md.append(f"| {lo}-{hi} | {len(z)} | {z.th_e.median():.2f} | {zz.th_h.median() if len(zz) else np.nan:.2f} | {1.45 + 6 / vm:.2f} |")
    if a.subset:
        lw = F[F.won == "lead"].groupby("scene").size()
        sub = sorted(lw.index[lw >= a.min_lead])
        Path(a.subset).write_text("\n".join(sub) + "\n")
        md += ["", f"Following subset (lead wins on >= {a.min_lead} following decisions): {len(sub)} scenes -> `{a.subset}`"]
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("slim")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    s = sp.add_parser("table")
    s.add_argument("--tax", required=True)
    s.add_argument("--dec", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    s = sp.add_parser("report")
    s.add_argument("--table", required=True)
    s.add_argument("--label", default="acc")
    s.add_argument("--out", required=True)
    s.add_argument("--subset")
    s.add_argument("--min-lead", type=int, default=10)
    s = sp.add_parser("human")
    s.add_argument("--tax", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    s = sp.add_parser("humanlog")
    s.add_argument("--scenes", required=True)
    s.add_argument("--cfg", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dict(slim=slim, table=table, report=report, human=human, humanlog=humanlog)[a.cmd](a)


if __name__ == "__main__":
    main()
