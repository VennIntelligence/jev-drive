#!/usr/bin/env python3
"""APP1X zero attribution (decision 303 Amendment 5, descriptive): which channel of the approach loss -- the lateral pull toward the entry
state or the one-sided speed term -- produced APP1X-L18-s0's new held zeros against RSG-L18-s0 (APP2W05-L18-s0 as a replication, APP0C-L18-s0
the W 0 control). CPU only, read-only on the stored held 452 / 18-decision reads.

  rf_appzeros.py extract --runs LABEL=<run dir>[,..] .. --out DIR [--workers N]      box, AlpaSim venv: per rollout the route polyline
        (rf_follow.regen_full: AlpaSim's MAP RouteGenerator re-run at every logged anchor), anchors, fed speed, served plan, per-decision
        zero-score flags -> DIR/rollouts.pkl
  rf_appzeros.py read --pkl DIR/rollouts.pkl --follow DIR/scenes.csv --out DIR [--fig PNG]   any python with numpy / pandas / matplotlib

Read (per scene, one common route = the base label's polyline, every label's anchors and plans projected on it):
  zero step kz    first decision with offroad, front / lateral collision or dist_to_gt >= 4 m (rf_follow's k_z); kind = the summary's
                  failure reason
  route turns     every >= 45 deg turn of the route polyline (rf_follow.find_turn's rule at 45 deg, the approach rows' threshold);
                  strict = >= 60 deg; the scene class: held turn (rf_follow turn of the base) / route turn (a >= 45 deg turn within
                  [s0 - 10, s_zero + 60] m) / bend (net heading change 20 - 45 deg over that stretch) / straight
  phase           the zero station vs the nearest >= 45 deg entry: before (entry - 40 .. entry), inside (entry .. exit + 10), after,
                  far (> 40 m before any entry or no turn)
  channels        last 3 s before the zero (decisions kz - 5 .. kz) of the zero label, against the base at the same station (the base's
                  decisions linear in station): speed ratio, signed lateral to the route (+ toward the inside of the nearest turn, else toward
                  the local curvature, else left), heading error (+ toward the inside), and the served plan's 1.0 / 1.5 / 2.0 s poses: lateral
                  to the route (mean) and lateral shift plan - ego
  no-turn split   no >= 45 deg entry within [st - 10, st + 60] m of any decision up to the zero: the approach rows' L2 / L3 window is closed
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
T0, DT = 17_000, 500_000
NAMES = ("dist_to_gt_trajectory", "collision_any", "offroad", "collision_front", "collision_lateral")


# ---------------------------------------------------------------------------------------------------------------------------- extract (box)
def one(job):
    import glob
    import polars as pl
    import rf_follow as RF
    label, D, scene, recs, summ, cfg = job
    f = sorted(glob.glob(f"{D}/rollouts/{scene}/*/metrics.parquet"))
    ev = {}
    if f:
        m = pl.read_parquet(f[0]).filter(pl.col("name").is_in(NAMES)).select("name", pl.col("timestamps_us").cast(pl.Int64), "values")
        ser = {n: dict(zip(x["timestamps_us"].to_list(), x["values"].to_list())) for (n,), x in m.group_by("name")}
        g = lambda n, t: float(ser.get(n, {}).get(t) or 0.0)  # noqa: E731
        for t in ser.get("dist_to_gt_trajectory", {}):
            if t >= T0 and (t - T0) % DT == 0:
                ev[int((t - T0) // DT)] = dict(gt=g("dist_to_gt_trajectory", t), off=g("offroad", t), colf=g("collision_front", t),
                                               coll=g("collision_lateral", t), col=g("collision_any", t))
    W, A, G0, Rl, log_end = RF.regen_full(scene, recs, cfg)
    keep = np.r_[True, np.hypot(*np.diff(Rl, axis=0).T) > 1e-6]
    return dict(label=label, scene=scene, score=summ.get("score"), fail=summ.get("failure_reason"), route=Rl[keep], log_end=log_end,
                k=np.array([int((r["t0"] - T0) // DT) for r in recs]), anchor=np.array([r["anchor"] for r in recs], float),
                v=np.array([float(np.hypot(*np.asarray(r["ego"], float)[4:6]) * 10.0) for r in recs]),
                poses=np.array([np.asarray(r["poses"], float)[:8, :3] for r in recs]), cmd=np.array([r.get("cmd", -1) for r in recs]), ev=ev)


def extract(a):
    import ev as EV
    from ao_e1 import drive_records
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "appzeros", config=vars(a)) as run:
        jobs = []
        for spec in a.runs:
            label, dirs = spec.split("=", 1)
            seen = set()
            for D in dirs.split(","):
                summ = {r["clipgt_id"]: r for r in json.loads((Path(D) / "aggregate/results-summary.json").read_text())["rollouts"]}
                for sc, rr in sorted(drive_records(D).items()):
                    if sc in seen or sc not in summ:
                        continue
                    seen.add(sc)
                    for r in rr:
                        for x in ("ms", "hist", "rmem"):
                            r.pop(x, None)
                    jobs.append((label, D, sc, rr, summ[sc], EV.CFG))
            run.info(f"{label}: {len(seen)} rollouts")
        res = par.pmap(one, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="rollouts")
        if res.errors:
            run.info(f"{len(res.errors)} rollouts failed: {[(jobs[i][0], jobs[i][2], str(e)[-200:]) for i, e in list(res.errors.items())[:3]]}")
        R = [v for v in res.values if v is not None]
        with open(out / "rollouts.pkl", "wb") as fh:
            pickle.dump(R, fh)
        run.summary.update(out=str(out), rollouts=len(R), errors=len(res.errors))


# ---------------------------------------------------------------------------------------------------------------------------- read (any)
def stations(R):
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(R, axis=0).T))]


def project(R, S, p, lo=-np.inf, hi=np.inf):
    """ao_e1.project: point onto polyline -> (station, signed lateral, left +)."""
    a, b = R[:-1], R[1:]
    seg = b - a
    L2 = (seg * seg).sum(1)
    ok = (L2 > 1e-8) & (S[1:] >= lo) & (S[:-1] <= hi)
    if not ok.any():
        ok = L2 > 1e-8
    a, seg, L2, s0 = a[ok], seg[ok], L2[ok], S[:-1][ok]
    t = np.clip(((p - a) * seg).sum(1) / L2, 0, 1)
    q = a + t[:, None] * seg
    i = int(np.hypot(*(p - q).T).argmin())
    u, v = seg[i] / np.sqrt(L2[i]), p - a[i]
    return float(s0[i] + t[i] * np.sqrt(L2[i])), float(u[0] * v[1] - u[1] * v[0])


def heading(R, S, step=1.0, half=5.0):
    s = np.arange(0.0, S[-1] + 1e-9, step)
    at = lambda q: np.c_[np.interp(q, S, R[:, 0]), np.interp(q, S, R[:, 1])]  # noqa: E731
    d = at(np.minimum(s + half, S[-1])) - at(np.maximum(s - half, 0.0))
    return s, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def turns(s, h, deg=45.0, win=30.0):
    """Every turn of the polyline, rf_follow.find_turn's rule at `deg` (net change over a 30 m window, >= 0.75 x deg over [a - 10, a + 40] m,
    entry / exit at 10 % / 90 %), scanning on after each exit -> [(entry, exit, TH rad)]."""
    out, w, a = [], int(round(win / (s[1] - s[0]))) if len(s) > 1 else 0, 0
    while a + w < len(s):
        if np.degrees(abs(h[a + w] - h[a])) >= deg:
            j0, j1 = max(0, a - 10), min(len(s) - 1, a + 40)
            th = h[j1] - h[j0]
            if np.degrees(abs(th)) >= 0.75 * deg:
                f = (h[j0:j1 + 1] - h[j0]) / th
                e, x = j0 + np.flatnonzero(f >= 0.1)[0], j0 + np.flatnonzero(f >= 0.9)[0]
                out.append((float(s[e]), float(s[x]), float(th)))
                a = max(x, a + 1)
                continue
        a += 1
    return out


def track(r, R, S, s_h, h_h):
    """Project a rollout's anchors and plans on route R -> per-decision arrays."""
    n = len(r["k"])
    st, lat, he, pl, pl2 = (np.full(n, np.nan) for _ in range(5))
    hint = None
    for i in range(n):
        x, y, yaw = r["anchor"][i]
        lo, hi = (-np.inf, np.inf) if hint is None else (hint - 30.0, hint + 60.0)
        st[i], lat[i] = project(R, S, np.array([x, y]), lo, hi)
        hint = st[i]
        he[i] = (yaw - np.interp(st[i], s_h, h_h) + np.pi) % (2 * np.pi) - np.pi
        c, sn = np.cos(yaw), np.sin(yaw)
        P = r["poses"][i][1:4, :2]                                                   # 1.0 / 1.5 / 2.0 s (poses at 0.5 s steps)
        L = np.c_[x + c * P[:, 0] - sn * P[:, 1], y + sn * P[:, 0] + c * P[:, 1]]
        q = [project(R, S, p, st[i] - 5.0, st[i] + 60.0) for p in L]
        pl[i] = np.mean([b for _, b in q])
        pl2[i] = q[-1][1]
    return dict(st=st, lat=lat, he=he, plat=pl, plat2=pl2)


def zero_k(r):
    ks = sorted(k for k, e in r["ev"].items() if e["off"] > 0 or e["colf"] > 0 or e["coll"] > 0 or e["gt"] >= 4.0)
    return ks[0] if ks else None


def read(a):
    import pandas as pd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(a.pkl, "rb") as fh:
        RR = pickle.load(fh)
    by = {}
    for r in RR:
        by.setdefault(r["scene"], {})[r["label"]] = r
    labs = sorted({r["label"] for r in RR})
    base = a.base
    F = pd.read_csv(a.follow)
    held = set(F[(F.label.str.startswith(base)) & (F.turn == True)].scene)                          # noqa: E712
    srows, drows, routes = [], [], {}
    for sc, d in sorted(by.items()):
        if base not in d:
            continue
        R = d[base]["route"]
        # the longest polyline among the labels (extensions append; checked below: a shorter one is a prefix within 5 cm)
        for lb, r in d.items():
            if len(r["route"]) > len(R) and np.allclose(r["route"][:len(R)], R, atol=0.05):
                R = r["route"]
        S = stations(R)
        s_h, h_h = heading(R, S)
        T = turns(s_h, h_h, 45.0)
        routes[sc] = R
        tr = {lb: track(r, R, S, s_h, h_h) for lb, r in d.items()}
        for lb, r in d.items():
            t = tr[lb]
            kz = zero_k(r)
            z = r["score"] == 0
            iz = (int(np.flatnonzero(r["k"] <= kz)[-1]) if kz is not None and (r["k"] <= kz).any() else len(r["k"]) - 1) if z else len(r["k"]) - 1
            sz = t["st"][iz]
            s0 = t["st"][0]
            # nearest >= 45 deg entry to the zero station, phase, inside sign
            ent = min(T, key=lambda q: min(abs(sz - q[0]), abs(sz - q[1])) if q[0] <= sz <= q[1] + 10 else abs(sz - q[0]) if sz < q[0] else abs(sz - q[1]) + 0.0) if T else None
            if ent is None:
                phase, dent = "none", np.nan
            else:
                dent = sz - ent[0]
                phase = ("inside" if ent[0] <= sz <= ent[1] + 10 else "before" if -40 <= dent < 0 else "after" if sz > ent[1] + 10 else "far")
            win_hi = sz + 60.0
            tw = [q for q in T if s0 - 10 <= q[0] <= win_hi]
            k_ = (s_h >= s0 - 10) & (s_h <= win_hi)
            dh = np.degrees(abs(h_h[k_].max() - h_h[k_].min())) if k_.any() else 0.0
            cls = "held_turn" if sc in held else "route_turn" if tw else "bend" if dh >= 20 else "straight"
            # the approach window opened at any decision up to the zero: a >= 45 deg entry in [st - 10, st + 60]
            opened = any(any(st_ - 10 <= q[0] <= st_ + 60 for q in T) for st_ in t["st"][:iz + 1])
            if ent is not None and (ent[0] - 40 <= sz <= ent[1] + 40):
                sg = float(np.sign(ent[2]))
            else:
                kk = (s_h >= sz - 10) & (s_h <= sz + 40)
                c = h_h[kk][-1] - h_h[kk][0] if kk.sum() > 1 else 0.0
                sg = float(np.sign(c)) if np.degrees(abs(c)) >= 5 else 1.0
            ev = r["ev"].get(kz, {}) if kz is not None else {}
            srows.append(dict(label=lb, scene=sc, log=sc.rsplit("-", 1)[0], score=r["score"], zero=bool(z), fail=r["fail"] if z else "",
                              kz=kz if z else -1, iz=iz, s0=s0, s_zero=sz, travel=sz - s0, cls=cls, phase=phase, d_entry=dent,
                              turn_deg=float(np.degrees(ent[2])) if ent is not None else np.nan, opened=opened, sign=sg,
                              n_turns=len(T), held=sc in held, ev_off=ev.get("off", np.nan), ev_col=max(ev.get("colf", 0), ev.get("coll", 0)) if ev else np.nan,
                              ev_gt=ev.get("gt", np.nan)))
            for i in range(len(r["k"])):
                drows.append(dict(label=lb, scene=sc, k=int(r["k"][i]), i=i, st=t["st"][i], lat=sg * t["lat"][i], he=sg * t["he"][i], v=r["v"][i],
                                  plat=sg * t["plat"][i], plat2=sg * t["plat2"][i], dplat=sg * (t["plat"][i] - t["lat"][i]), sign=sg))
    Sx, Dx = pd.DataFrame(srows), pd.DataFrame(drows)
    Sx.to_csv(out / "app1x_zeros_scenes.csv", index=False)
    Dx.to_parquet(out / "app1x_zeros_decisions.parquet", index=False)
    with open(out / "routes.pkl", "wb") as fh:
        pickle.dump(routes, fh)
    print(f"{len(Sx)} scene rows, labels {labs}")


def summarise(a):
    """Channel table per zero label vs the base -> app1x_zeros.{md,json,csv}, the BEV figure."""
    import pandas as pd
    out = Path(a.out)
    Sx = pd.read_csv(out / "app1x_zeros_scenes.csv")
    Dx = pd.read_parquet(out / "app1x_zeros_decisions.parquet")
    base = a.base
    B = Sx[Sx.label == base].set_index("scene")
    rows = []
    for lb in a.labels:
        X = Sx[Sx.label == lb].set_index("scene")
        new = X.index[(X.zero) & (~B.reindex(X.index).zero.fillna(True).astype(bool))]
        for sc in new:
            s = X.loc[sc]
            da = Dx[(Dx.label == lb) & (Dx.scene == sc)].sort_values("i")
            db = Dx[(Dx.label == base) & (Dx.scene == sc)].sort_values("i")
            w = da[(da.i <= s.iz) & (da.i >= s.iz - 5)]                                       # last 3 s (6 decisions at 2 Hz) up to the zero
            stb = db.st.to_numpy()
            mono = np.maximum.accumulate(stb)
            inb = lambda col, x: np.interp(x, mono, db[col].to_numpy(), left=np.nan, right=np.nan)  # noqa: E731
            vb = inb("v", w.st)
            reach = np.isfinite(vb)
            # same decision index (time-paired) as a second view
            wb = db.set_index("i").reindex(w.i)
            rows.append(dict(label=lb, scene=sc, log=s.log, fail=s.fail, cls=s.cls, phase=s.phase, d_entry=s.d_entry, opened=s.opened, kz=s.kz,
                             travel_a=s.travel, travel_b=B.loc[sc].travel, n_reach=int(reach.sum()),
                             v_a=w.v.mean(), v_b_st=np.nanmean(vb) if reach.any() else np.nan,
                             v_ratio_st=np.nanmedian(w.v.to_numpy()[reach] / np.maximum(vb[reach], 0.5)) if reach.any() else np.nan,
                             v_ratio_k=np.nanmedian(w.v.to_numpy() / np.maximum(wb.v.to_numpy(), 0.5)),
                             lat_a=w.lat.mean(), lat_b_st=np.nanmean(inb("lat", w.st)) if reach.any() else np.nan,
                             lat_zero=w.lat.iloc[-1], he_a=np.degrees(w.he.mean()), he_b_st=np.degrees(np.nanmean(inb("he", w.st))) if reach.any() else np.nan,
                             plat_a=w.plat.mean(), plat_b_st=np.nanmean(inb("plat", w.st)) if reach.any() else np.nan,
                             dplat_a=w.dplat.mean(), dplat_b_st=np.nanmean(inb("dplat", w.st)) if reach.any() else np.nan,
                             lat_abs_a=w.lat.abs().mean(), lat_abs_b_st=np.nanmean(np.abs(inb("lat", w.st))) if reach.any() else np.nan))
    Z = pd.DataFrame(rows)
    Z["d_lat"], Z["d_he"], Z["d_plat"], Z["d_dplat"] = Z.lat_a - Z.lat_b_st, Z.he_a - Z.he_b_st, Z.plat_a - Z.plat_b_st, Z.dplat_a - Z.dplat_b_st
    Z["d_lat_abs"] = Z.lat_abs_a - Z.lat_abs_b_st
    sp = np.where(Z.v_ratio_st < 0.9, "slower", np.where(Z.v_ratio_st > 1.1, "faster", "same"))
    sp = np.where(Z.v_ratio_st.isna(), "base_never_there", sp)
    la = np.where(Z.d_lat > 0.3, "inside", np.where(Z.d_lat < -0.3, "outside", "same_line"))
    la = np.where(Z.d_lat.isna(), "n/a", la)
    Z["mode"] = [f"{x}/{y}" for x, y in zip(sp, la)]
    Z.to_csv(out / "app1x_zeros.csv", index=False)
    res = dict(base=base, labels={})
    q = lambda x: [float(np.nanpercentile(x, p)) if np.isfinite(x).any() else None for p in (25, 50, 75)]  # noqa: E731
    for lb, g in Z.groupby("label"):
        X = Sx[Sx.label == lb]
        r = dict(new_zeros=int(len(g)), zeros=int(X.zero.sum()), base_zeros=int(B.zero.sum()),
                 fail=g.fail.value_counts().to_dict(), cls=g.cls.value_counts().to_dict(), phase=g.phase.value_counts().to_dict(),
                 opened=g.opened.value_counts().to_dict(), mode=g["mode"].value_counts().to_dict(),
                 cls_all_scenes=X.cls.value_counts().to_dict(), opened_all_scenes=X.opened.value_counts().to_dict(),
                 d_entry=q(g.d_entry), v_ratio_st=q(g.v_ratio_st), v_ratio_k=q(g.v_ratio_k), d_lat=q(g.d_lat), d_lat_abs=q(g.d_lat_abs),
                 d_he_deg=q(g.d_he), d_plat=q(g.d_plat), d_dplat=q(g.d_dplat), lat_a=q(g.lat_a), lat_b_st=q(g.lat_b_st), lat_zero=q(g.lat_zero),
                 dplat_a=q(g.dplat_a), dplat_b_st=q(g.dplat_b_st), v_a=q(g.v_a), v_b_st=q(g.v_b_st), travel_d=q(g.travel_a - g.travel_b),
                 by_opened={str(o): dict(n=int(len(h)), d_lat=q(h.d_lat), v_ratio_st=q(h.v_ratio_st), d_plat=q(h.d_plat), fail=h.fail.value_counts().to_dict())
                            for o, h in g.groupby("opened")})
        res["labels"][lb] = r
    # whole-rollout lateral bias (all decisions before a zero, every scene, paired on the station): is the inside lean everywhere?
    res["all_scene_bias"] = {}
    for lb in a.labels:
        dd = []
        for sc, da in Dx[Dx.label == lb].groupby("scene"):
            db = Dx[(Dx.label == base) & (Dx.scene == sc)].sort_values("i")
            if not len(db):
                continue
            iz = int(Sx[(Sx.label == lb) & (Sx.scene == sc)].iz.iloc[0])
            da = da.sort_values("i")
            da = da[da.i <= iz]
            mono = np.maximum.accumulate(db.st.to_numpy())
            for col in ("lat", "dplat", "v"):
                x = np.interp(da.st, mono, db[col].to_numpy(), left=np.nan, right=np.nan)
                da = da.assign(**{col + "_b": x})
            dd.append(da)
        dd = pd.concat(dd)
        op = dd.merge(Sx[Sx.label == lb][["scene", "opened", "cls"]], on="scene")
        res["all_scene_bias"][lb] = {g: dict(n=int(len(h)), d_lat=float(np.nanmedian(h.lat - h.lat_b)), d_dplat=float(np.nanmedian(h.dplat - h.dplat_b)),
                                             d_abslat=float(np.nanmedian(h.lat.abs() - h.lat_b.abs())), v_ratio=float(np.nanmedian(h.v / np.maximum(h.v_b, 0.5))))
                                     for g, h in op.groupby("cls")}
    (out / "app1x_zeros.json").write_text(json.dumps(res, indent=1, default=lambda x: x if not isinstance(x, (np.integer, np.floating, np.bool_)) else x.item()))
    print(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--runs", nargs="+", required=True), e.add_argument("--out", required=True), e.add_argument("--workers", type=int, default=0)
    r = sp.add_parser("read")
    r.add_argument("--pkl", required=True), r.add_argument("--follow", required=True), r.add_argument("--out", required=True)
    r.add_argument("--base", default="RSG-L18-s0")
    m = sp.add_parser("summary")
    m.add_argument("--out", required=True), m.add_argument("--base", default="RSG-L18-s0")
    m.add_argument("--labels", nargs="+", default=["APP1X-L18-s0", "APP2W05-L18-s0", "APP0C-L18-s0"])
    a = ap.parse_args()
    if a.cmd == "extract":
        R_ = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/route_mem/scripts"),
                        str(R_ / "experiments/alpasim/scripts"), str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/anchor_offset/scripts"),
                        str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts")]
        extract(a)
    else:
        {"read": read, "summary": summarise}[a.cmd](a)
