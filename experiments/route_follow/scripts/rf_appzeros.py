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


def onset(a):
    """Where the zero label's track first leaves the base's: every >= 45 deg route turn whose entry the base passes within its rollout, the
    decisions of both labels with station in [entry - 40, entry + 30] m before their own zero, binned by station - entry; per bin the zero label
    minus the base at the same station (base linear in station): lateral (+ inside), heading (+ inside), the plan's 1 - 2 s shift over the ego
    (+ inside) and the speed ratio. Groups: the zero label's new-zero scenes vs every other scene -> app1x_zeros_onset.{csv,json}."""
    import pandas as pd
    out = Path(a.out)
    Sx = pd.read_csv(out / "app1x_zeros_scenes.csv")
    Dx = pd.read_parquet(out / "app1x_zeros_decisions.parquet")
    with open(out / "routes.pkl", "rb") as fh:
        routes = pickle.load(fh)
    base, bins = a.base, np.array([-40, -20, -10, 0, 10, 20, 30])
    rows = []
    B = Sx[Sx.label == base].set_index("scene")
    for lb in a.labels:
        X = Sx[Sx.label == lb].set_index("scene")
        new = set(X.index[X.zero & ~B.reindex(X.index).zero.astype(bool)])
        for sc, R in routes.items():
            S = stations(R)
            s_h, h_h = heading(R, S)
            da = Dx[(Dx.label == lb) & (Dx.scene == sc)].sort_values("i")
            db = Dx[(Dx.label == base) & (Dx.scene == sc)].sort_values("i")
            da, db = da[da.i <= X.loc[sc].iz], db[db.i <= B.loc[sc].iz]
            mono = np.maximum.accumulate(db.st.to_numpy())
            for e, x, th in turns(s_h, h_h, 45.0):
                if not (mono[0] <= e + 30 and mono[-1] >= e):                            # the base reaches the entry
                    continue
                sg = float(np.sign(th)) * da.sign.to_numpy()                            # undo the stored sign, apply this turn's
                sgb = float(np.sign(th)) * db.sign.to_numpy()
                w = (da.st >= e - 40) & (da.st < e + 30)
                if not w.any():
                    continue
                q = lambda col, sgn: np.interp(da.st[w], mono, (db[col] * sgn).to_numpy(), left=np.nan, right=np.nan)  # noqa: E731
                d = pd.DataFrame(dict(label=lb, scene=sc, new=sc in new, entry=e, rel=(da.st[w] - e).to_numpy(),
                                      d_lat=(da.lat * sg)[w].to_numpy() - q("lat", sgb), d_he=np.degrees((da.he * sg)[w].to_numpy() - q("he", sgb)),
                                      d_dplat=(da.dplat * sg)[w].to_numpy() - q("dplat", sgb), dplat_a=(da.dplat * sg)[w].to_numpy(),
                                      v_ratio=da.v[w].to_numpy() / np.maximum(q("v", 1.0), 0.5), lat_a=(da.lat * sg)[w].to_numpy()))
                rows.append(d)
    O = pd.concat(rows)
    O["bin"] = pd.cut(O.rel, bins, right=False).astype(str)
    O.to_csv(out / "app1x_zeros_onset.csv", index=False)
    res = {}
    for (lb, nw), g in O.groupby(["label", "new"]):
        res[f"{lb}|{'new_zero' if nw else 'other'}"] = {b: dict(n=int(len(h)), scenes=int(h.scene.nunique()), d_lat=float(np.nanmedian(h.d_lat)),
                                                              d_he=float(np.nanmedian(h.d_he)), d_dplat=float(np.nanmedian(h.d_dplat)),
                                                              d_lat_mean=float(np.nanmean(h.d_lat)), d_dplat_mean=float(np.nanmean(h.d_dplat)),
                                                              v_ratio=float(np.nanmedian(h.v_ratio)))
                                                        for b, h in g.groupby("bin", sort=False)}
    (out / "app1x_zeros_onset.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k)
        for b in sorted(v, key=lambda s: float(s.split(",")[0][1:])):
            x = v[b]
            print(f"  {b:10s} n {x['n']:4d} ({x['scenes']:3d})  d_lat {x['d_lat']:+.2f} ({x['d_lat_mean']:+.2f})  d_he {x['d_he']:+.1f}  "
                  f"d_dplat {x['d_dplat']:+.2f} ({x['d_dplat_mean']:+.2f})  v_ratio {x['v_ratio']:.2f}")


def figure(a):
    """BEV of example scenes: route, the base's and the zero label's anchors, both served plans at the zero label's last decision before the
    zero (the base's at the nearest station), the zero point; frame = the first anchor, heading up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    out = Path(a.out)
    with open(a.pkl, "rb") as fh:
        RR = pickle.load(fh)
    with open(out / "routes.pkl", "rb") as fh:
        routes = pickle.load(fh)
    Z = pd.read_csv(out / "app1x_zeros.csv")
    by = {}
    for r in RR:
        by.setdefault(r["scene"], {})[r["label"]] = r
    C = {a.base: "#2a78d6", a.label: "#eb6834"}
    fig, axs = plt.subplots(1, len(a.scenes), figsize=(4.2 * len(a.scenes), 5.6))
    for ax, key in zip(np.atleast_1d(axs), a.scenes):
        sc = next(x for x in by if x.endswith(key))
        z = Z[(Z.label == a.label) & (Z.scene == sc)].iloc[0]
        x0, y0, h0 = by[sc][a.base]["anchor"][0]
        c, s = np.cos(np.pi / 2 - h0), np.sin(np.pi / 2 - h0)
        T = lambda P: np.c_[c * (P[:, 0] - x0) - s * (P[:, 1] - y0), s * (P[:, 0] - x0) + c * (P[:, 1] - y0)]  # noqa: E731
        R = routes[sc]
        A = T(by[sc][a.label]["anchor"][:, :2])
        lim = np.r_[A, T(by[sc][a.base]["anchor"][:, :2])]
        lo, hi = lim.min(0) - 12, lim.max(0) + 12
        Rt = T(R)
        m = (Rt[:, 0] > lo[0] - 20) & (Rt[:, 0] < hi[0] + 20) & (Rt[:, 1] > lo[1] - 20) & (Rt[:, 1] < hi[1] + 20)
        ax.plot(*Rt[m].T, color="#b5b4ae", lw=6, solid_capstyle="round", zorder=1, label="route (AlpaSim MAP)")
        iz = int(z.kz) if np.isfinite(z.kz) else len(A) - 1
        ka = by[sc][a.label]["k"]
        izi = int(np.flatnonzero(ka <= iz)[-1])
        for lb in (a.base, a.label):
            r = by[sc][lb]
            P = T(r["anchor"][:, :2])
            n = izi + 1 if lb == a.label else len(P)
            ax.plot(*P[:n].T, "-o", color=C[lb], lw=2, ms=3.5, zorder=3, label=lb.replace("-L18-s0", "") + (" (to the zero)" if lb == a.label else ""))
        # plans 1 s before the zero
        j = max(izi - 2, 0)
        sa = T(by[sc][a.label]["anchor"][j:j + 1, :2])[0]
        jb = int(np.argmin(np.hypot(*(T(by[sc][a.base]["anchor"][:, :2]) - sa).T)))
        for lb, jj in ((a.label, j), (a.base, jb)):
            x, y, yaw = by[sc][lb]["anchor"][jj]
            Pp = by[sc][lb]["poses"][jj][:, :2]
            cc, ss = np.cos(yaw), np.sin(yaw)
            L = np.r_[[[x, y]], np.c_[x + cc * Pp[:, 0] - ss * Pp[:, 1], y + ss * Pp[:, 0] + cc * Pp[:, 1]]]
            ax.plot(*T(L).T, "--", color=C[lb], lw=1.5, zorder=4)
        ax.plot(*A[izi], marker="X", ms=12, color="#0b0b0b", mec="#fcfcfb", mew=1.5, zorder=5, label="zero (first event)")
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal")
        ax.set_title(f"...{key}  {z.fail.replace('left_corridor_laterally', 'corridor')}\n{z.cls}, {z.phase}"
                     f" ({z.d_entry:+.0f} m from entry)" if np.isfinite(z.d_entry) else f"...{key}  {z.fail}\n{z.cls}", fontsize=9, color="#0b0b0b")
        ax.text(0.02, 0.02, f"last 3 s vs RSG at same station:\nspeed x{z.v_ratio_st:.2f}, lat {z.d_lat:+.2f} m, heading {z.d_he:+.1f} deg\n(+ = toward the turn's inside)",
                transform=ax.transAxes, fontsize=7.5, color="#52514e", va="bottom")
        ax.tick_params(labelsize=7, colors="#52514e")
        for sp_ in ax.spines.values():
            sp_.set_color("#d0cfca")
        ax.grid(color="#ecebe7", lw=0.6)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle(f"{a.label} new held zeros vs {a.base}: anchors (solid), served plans 1 s before the zero (dashed); metres, start heading up",
                 fontsize=10, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(a.fig, dpi=130)
    print(a.fig)


def plan0(a):
    """Identical inputs: decisions 0 and 1 have the same state in every label (the rollouts split from decision 2), so the served plans there
    are a pure policy difference. Per label minus the base, binned by the ego's station - the nearest >= 45 deg entry: the 1 - 2 s plan
    lateral shift over the ego and the 4 s pose lateral / yaw (+ toward the inside), the plan arc to 2 s / 4 s -> app1x_zeros_plan01.{csv,json}."""
    import pandas as pd
    out = Path(a.out)
    with open(a.pkl, "rb") as fh:
        RR = pickle.load(fh)
    with open(out / "routes.pkl", "rb") as fh:
        routes = pickle.load(fh)
    by = {}
    for r in RR:
        by.setdefault(r["scene"], {})[r["label"]] = r
    labs = [a.base] + a.labels
    rows = []
    arc = lambda P: float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], P[:, :2]], axis=0).T).sum())  # noqa: E731
    for sc, d in by.items():
        R = routes[sc]
        S = stations(R)
        s_h, h_h = heading(R, S)
        T = turns(s_h, h_h, 45.0)
        for i in (0, 1):
            assert all(np.allclose(d[lb]["anchor"][i], d[a.base]["anchor"][i], atol=1e-3) for lb in labs)
            x, y, yaw = d[a.base]["anchor"][i]
            st, lat = project(R, S, np.array([x, y]))
            near = [t for t in T if t[0] - 40 <= st <= t[1] + 10]
            if near:
                t = min(near, key=lambda t: abs(st - t[0]))
                rel, sg = st - t[0], float(np.sign(t[2]))
                b = str(pd.cut([rel], [-40, -20, -10, 0, 10, 30], right=False)[0])
            else:
                rel, sg = np.nan, 1.0
                b = "no_turn_60" if not any(st - 10 <= t[0] <= st + 60 for t in T) else "other"
            rec = dict(scene=sc, i=i, rel=rel, bin=b)
            c, s = np.cos(yaw), np.sin(yaw)
            for lb in labs:
                P = d[lb]["poses"][i]
                L = np.c_[x + c * P[:, 0] - s * P[:, 1], y + s * P[:, 0] + c * P[:, 1]]
                rec[lb + "|dplat"] = sg * (np.mean([project(R, S, p, st - 5, st + 80)[1] for p in L[1:4]]) - lat)
                rec[lb + "|y4"], rec[lb + "|h4"] = sg * P[7, 1], np.degrees(sg * P[7, 2])
                rec[lb + "|arc2"], rec[lb + "|arc4"] = arc(P[:4]), arc(P[:8])
            rows.append(rec)
    X = pd.DataFrame(rows)
    X.to_csv(out / "app1x_zeros_plan01.csv", index=False)
    res = {}
    for lb in a.labels:
        res[lb] = {}
        for b, g in X.groupby("bin"):
            res[lb][b] = dict(n=int(len(g)), **{f"{m}_{f}": float(fn(g[f"{lb}|{m}"] - g[f"{a.base}|{m}"])) for m in ("dplat", "y4", "h4", "arc2", "arc4")
                                               for f, fn in (("med", np.median), ("mean", np.mean))},
                              base_y4=float(g[f"{a.base}|y4"].median()), base_h4=float(g[f"{a.base}|h4"].median()))
    (out / "app1x_zeros_plan01.json").write_text(json.dumps(res, indent=1))
    for lb, v in res.items():
        print(lb)
        for b, x in v.items():
            print(f"  {b:14s} n {x['n']:4d} dplat {x['dplat_med']:+.3f} ({x['dplat_mean']:+.3f}) y4 {x['y4_med']:+.3f} ({x['y4_mean']:+.3f}) "
                  f"h4 {x['h4_med']:+.2f} ({x['h4_mean']:+.2f}) arc2 {x['arc2_med']:+.3f} arc4 {x['arc4_med']:+.3f} ({x['arc4_mean']:+.3f}) base h4 {x['base_h4']:+.1f}")


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
    o = sp.add_parser("onset")
    o.add_argument("--out", required=True), o.add_argument("--base", default="RSG-L18-s0")
    o.add_argument("--labels", nargs="+", default=["APP1X-L18-s0", "APP2W05-L18-s0", "APP0C-L18-s0"])
    f = sp.add_parser("fig")
    f.add_argument("--pkl", required=True), f.add_argument("--out", required=True), f.add_argument("--fig", required=True)
    f.add_argument("--base", default="RSG-L18-s0"), f.add_argument("--label", default="APP1X-L18-s0"), f.add_argument("--scenes", nargs="+", required=True)
    q = sp.add_parser("plan01")
    q.add_argument("--pkl", required=True), q.add_argument("--out", required=True), q.add_argument("--base", default="RSG-L18-s0")
    q.add_argument("--labels", nargs="+", default=["APP1X-L18-s0", "APP2W05-L18-s0", "APP0C-L18-s0"])
    a = ap.parse_args()
    if a.cmd == "extract":
        R_ = Path(__file__).resolve().parents[3]
        sys.path[:0] = [str(R_), str(R_ / "lib"), str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/route_mem/scripts"),
                        str(R_ / "experiments/alpasim/scripts"), str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/anchor_offset/scripts"),
                        str(R_ / "experiments/eyes_vs_head/scripts"), str(R_ / "experiments/alpasim_long/scripts")]
        extract(a)
    else:
        {"read": read, "summary": summarise, "onset": onset, "fig": figure, "plan01": plan0}[a.cmd](a)
