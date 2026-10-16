#!/usr/bin/env python3
"""JOINT-DAGGER route check (results/joint_dagger/route_check.md): on every DAgger row (labels.npz valid: on1_L18RSG own rows k >= 2 with a
clean joint target; train + dev logs, the full label set and Amendment 1's junction + failure-window subset), in the row's rear-axle frame:
  (a) the route memory the trainer feeds (feat/on1_JDG.npz pool-d rows for train logs, feat/on1_JDGdev.npz for dev logs; f16 -> f32, rows with
      ok False zeroed, the 40 slots at -6 .. +72 m with valid bits; the Amendment 1 pool feat/on1_JDGJ.npz checked equal row by row),
  (b) the expert pack's path (rule_expert's lane-chain centreline, rf_expert.py pack; the row's station by rule_expert's own sequential
      projection: the scene's rows in k order, window [prev - 10, prev + 40] m),
  (c) the joint target's 8 poses (labels.npz target = the trained `fut`), the served plan (meta poses),
  (d) the human log path (meta gpath: logged path s0 - 15 .. s0 + 70 m, 1 m, already in the row frame).
Reads: lateral distance (a)-(b) at +10 / 20 / 40 m, exit disagreement (heading at +40 m differs > 30 deg), rows with no valid slot, plan / label
distance to (a), (b), (d); by segment (DAGGER-CHECK seg; failure window apart). Join: pool tab / feat / labels / meta row by row, the fed memory
against the driver's served memory (meta rmem) and the route-hinge polyline (route.npz), the pack's log against the row's logged poses (meta logfut, same times), and for 20 random rows
the rule expert re-run at the meta anchor against the stored label.
  rf_jdroute.py read [--workers N]   CPU -> $JD/route_check/{rows.npz, route_check.json, join20.txt}, copies in results/joint_dagger/
Box, repo .venv (numpy only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import rf_jdagger as JG  # noqa: E402  (sets the repo paths)
import rmem as RM  # noqa: E402

D, CR, ROWS, JD, RES = JG.D, JG.CR, JG.ROWS, JG.JD, JG.RES
FEAT = D / "runs/route_mem/feat"
OUT = JD / "route_check"
XS = (10.0, 20.0, 40.0)
EXIT_DEG = 30.0
SEGS = ("junction", "approach", "window", "straight", "other")


def fed(name):
    """feat/<name>.npz -> {row name without the merge prefix: (DIM,) f32 as the trainer feeds it}, raw names."""
    z = np.load(FEAT / f"{name}.npz")
    f = z["f"].astype(np.float32)
    f[~z["ok"]] = 0
    nm = [("on:" + n.split(":", 2)[2]) if n.startswith(("on:d:", "on:s:")) else n for n in z["names"].tolist()]
    return dict(zip(nm, f)), z


def pdist(P, X):
    """Polyline P (n, 2), points X (m, 2) -> distance (m,), nan where the nearest point is an end of P with the point beyond it."""
    X = np.asarray(X, float).reshape(-1, 2)
    if len(P) < 2:
        return np.full(len(X), np.nan)
    A, d = P[:-1], np.diff(P, axis=0)
    L2 = np.maximum((d * d).sum(1), 1e-12)
    ur = ((X[:, None] - A[None]) * d[None]).sum(-1) / L2[None]
    u = np.clip(ur, 0, 1)
    e = np.hypot(*(A[None] + u[..., None] * d[None] - X[:, None]).transpose(2, 0, 1))
    j = e.argmin(1)
    r = np.arange(len(X))
    out = e[r, j]
    end = ((j == 0) & (ur[r, j] < 0)) | ((j == len(A) - 1) & (ur[r, j] > 1))
    out[end] = np.nan
    return out


def wrapd(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def scene_job(job):
    """One scene: every row in k order (rule_expert's session) -> per wanted row the metrics."""
    import rule_expert as XP
    sc, pf, idx, anc, want, F, gp, T, S, t0, lf = job
    Z = XP.Scene(Path(pf))
    P, St = Z.P, Z.S
    hx = np.degrees(np.unwrap(np.arctan2(*np.gradient(P, axis=0)[:, ::-1].T)))
    zp = np.load(pf)
    lxy, lt = zp["log_xy"].astype(float), zp["log_t"].astype(float)
    hint, out = None, {}
    for j, i in enumerate(idx):
        a = anc[j]
        s0, _ = Z.project(np.asarray(a[:2], float), *((hint - 10.0, hint + 40.0) if hint is not None else (None, None)))
        hint = s0
        if not want[j]:
            continue
        r = {}
        # (b) the pack path in the row frame
        w = (St >= s0 - 20) & (St <= s0 + 130)
        B = RM.in_ego(a, P[w][::2])
        bx = lambda s: RM.in_ego(a, np.c_[np.interp(s0 + np.asarray(s), St, P[:, 0]), np.interp(s0 + np.asarray(s), St, P[:, 1])])  # noqa: E731
        hb = lambda s: float(np.interp(s0 + s, St, hx)) - np.degrees(a[2])  # noqa: E731
        # (a) the fed memory
        Q = RM.pts(F[j])
        v = Q[:, 0] > 0.5
        axy = Q[:, 1:3] * 20.0
        ha = np.degrees(np.arctan2(Q[:, 4], Q[:, 3]))
        Am = axy[v]
        r["nA"] = int(v.sum())
        r["firstA"] = float(RM.OFFS[v][0]) if v.any() else np.nan
        # (d) the human log path (gpath -15 .. +70 m, 1 m)
        G0 = gp[j].astype(float)
        G = G0[np.isfinite(G0).all(1)]
        hd40 = float(np.degrees(np.arctan2(*(G0[56] - G0[54])[::-1])))
        for x in XS:
            ia = int(np.flatnonzero(RM.OFFS == x)[0])
            pa = axy[ia] if v[ia] else None
            pb = bx(x)[0]
            pd = G0[15 + int(x)]
            r[f"ab{x:g}"] = float(pdist(B, pa[None])[0]) if pa is not None else np.nan          # (a)'s slot to (b)
            r[f"ba{x:g}"] = float(pdist(Am, pb[None])[0]) if len(Am) >= 2 else np.nan            # (b)'s point to (a)
            r[f"ad{x:g}"] = float(pdist(G, pa[None])[0]) if pa is not None else np.nan
            r[f"bd{x:g}"] = float(pdist(G, pb[None])[0])
            r[f"abp{x:g}"] = float(np.hypot(*(pa - pb))) if pa is not None else np.nan             # same arc offset, point to point
            r[f"dd{x:g}"] = float(np.hypot(*(pd - pb)))
        i40 = int(np.flatnonzero(RM.OFFS == 40.0)[0])
        r["hA40"] = float(wrapd(ha[i40])) if v[i40] else np.nan
        r["hB40"] = float(wrapd(hb(40.0)))
        r["hD40"] = float(wrapd(hd40))
        i0 = int(np.flatnonzero(RM.OFFS == 0.0)[0])
        p0 = bx(0.0)[0]
        r["a0_lon"] = float(axy[i0, 0] - p0[0]) if v[i0] else np.nan                              # memory anchor vs pack anchor (x of the +0 points)
        r["a0_d"] = float(np.hypot(*(axy[i0] - p0))) if v[i0] else np.nan
        r["b0_lat"] = float(np.hypot(*p0))                                                       # ego off the pack path
        # pack frame / time vs the row: the pack's log at t0 + 0 .. 4 s in the row frame against meta logfut (the log's own poses, row frame)
        tt = t0[j] + 5e5 * np.arange(9)
        ok = (tt >= lt[0]) & (tt <= lt[-1]) & np.isfinite(lf[j][:, 0])
        X = np.c_[np.interp(tt[ok], lt, lxy[:, 0]), np.interp(tt[ok], lt, lxy[:, 1])]
        r["frame_logfut"] = float(np.hypot(*(RM.in_ego(a, X) - lf[j][ok, :2]).T).max()) if ok.any() else np.nan
        Lg = RM.in_ego(a, lxy)
        r["log40"] = float(pdist(Lg, G0[55:56])[0] < 1.0) if np.isfinite(G0[55]).all() else 0.0     # gpath is the human log at +40 m
        for nm, X in (("lab", T[j]), ("srv", S[j])):
            xy = X[:, :2].astype(float)
            da, db, dg = pdist(Am, xy) if len(Am) >= 2 else np.full(8, np.nan), pdist(B, xy), pdist(G, xy)
            for q, dq in (("a", da), ("b", db), ("d", dg)):
                r[f"{nm}_{q}_mean"] = float(np.nanmean(dq)) if np.isfinite(dq).any() else np.nan
                r[f"{nm}_{q}_4s"] = float(dq[-1])
            r[f"{nm}_arc4"] = float(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], xy], axis=0).T).sum())
        out[int(i)] = r
    return out


def summ(R, m, logs):
    """Summary of the row metrics R (dict of (N,) arrays) over mask m."""
    from jevdrive import stats
    n = int(m.sum())
    if not n:
        return dict(n=0)
    q = lambda k, f=np.nanmedian: (float(f(R[k][m])) if np.isfinite(R[k][m]).any() else None)  # noqa: E731
    out = dict(n=n, logs=len(set(logs[m].tolist())), no_valid=float((R["nA"][m] == 0).mean()), slot40_invalid=float(np.mean(~np.isfinite(R["hA40"][m]))),
               nA_med=q("nA"), firstA_med=q("firstA"), a0_d_med=q("a0_d"), a0_lon_med=q("a0_lon"), b0_lat_med=q("b0_lat"))
    for x in XS:
        k = f"ab{x:g}"
        z = R[k][m]
        f = np.isfinite(z)
        out[f"ab{x:g}"] = dict(n=int(f.sum()), med=float(np.median(z[f])) if f.any() else None, p90=float(np.percentile(z[f], 90)) if f.any() else None,
                               gt1=float((z[f] > 1.0).mean()) if f.any() else None, gt3=float((z[f] > 3.0).mean()) if f.any() else None,
                               ba_med=q(f"ba{x:g}"), abp_med=q(f"abp{x:g}"), ad_med=q(f"ad{x:g}"), bd_med=q(f"bd{x:g}"))
    both = np.isfinite(R["hA40"]) & m
    dab = np.abs(wrapd(R["hA40"] - R["hB40"]))
    dad = np.abs(wrapd(R["hA40"] - R["hD40"]))
    dbd = np.abs(wrapd(R["hB40"] - R["hD40"]))
    ex = (dab > EXIT_DEG).astype(float)
    ex[~both] = np.nan
    ci = stats.bootstrap(ex[m], groups=logs[m])
    out["exit_ab"] = dict(n=int(both.sum()), share=float(ex[both].mean()) if both.any() else None, ci=[ci["lo"], ci["hi"]],
                          n_rows=int((ex[both] > 0).sum()),
                          of_which_a_matches_d=int(((dab > EXIT_DEG) & (dad <= EXIT_DEG) & both).sum()),
                          of_which_b_matches_d=int(((dab > EXIT_DEG) & (dbd <= EXIT_DEG) & both).sum()))
    geo = (dab > EXIT_DEG) & both & (R["ab40"] > 2.0)                      # a heading split that is also a > 2 m geometric split at +40 m
    lg = R["log40"] > 0.5
    out["exit_geo"] = dict(n_rows=int(geo.sum()), share=float(geo.sum() / max(both.sum(), 1)), with_log40=int((geo & lg).sum()),
                           b_matches_d=int((geo & lg & (dbd <= EXIT_DEG)).sum()), a_matches_d=int((geo & lg & (dad <= EXIT_DEG)).sum()))
    out["log40_share"] = float(lg[m].mean())
    fa = R["firstA"][m]
    out["first_gt18"] = float(np.mean(fa > 18))
    out["firstA_iqr"] = [float(np.nanpercentile(fa, 25)), float(np.nanpercentile(fa, 75))]
    out["exit_ad"] = float(((dad > EXIT_DEG) & both).sum() / max(both.sum(), 1))
    out["exit_bd"] = float(((dbd > EXIT_DEG) & m).sum() / n)
    for nm in ("lab", "srv"):
        out[nm] = {f"{q_}_{w}": q(f"{nm}_{q_}_{w}") for q_ in ("a", "b", "d") for w in ("mean", "4s")}
        out[nm]["a_cov"] = float(np.isfinite(R[f"{nm}_a_4s"][m]).mean())
        out[nm]["arc4_med"] = q(f"{nm}_arc4")
    # does the label sit closer to the route than the served plan? (paired, row level, mean over the 8 poses)
    for q_ in ("a", "b"):
        dl, ds = R[f"lab_{q_}_mean"][m], R[f"srv_{q_}_mean"][m]
        ok = np.isfinite(dl) & np.isfinite(ds)
        out[f"lab_minus_srv_{q_}"] = dict(n=int(ok.sum()), mean=float((dl - ds)[ok].mean()) if ok.any() else None,
                                          lab_closer=float((dl < ds)[ok].mean()) if ok.any() else None)
    out["frame_logfut_max"] = float(np.nanmax(R["frame_logfut"][m]))
    return out


def cmd_read(a):
    from concurrent.futures import ProcessPoolExecutor
    from jevdrive.run import Run
    OUT.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "jdg-routecheck", config=vars(a)) as run:
        M = dict(np.load(ROWS / "L18RSG" / "meta.npz"))
        Lz = dict(np.load(JD / "labels.npz"))
        names = M["names"]
        assert (Lz["names"] == names).all()
        N = len(names)
        scene, k, logs = M["scene"].astype(str), M["k"].astype(int), M["log"].astype(str)
        valid, dev, win, seg = Lz["valid"], Lz["dev"], Lz["win"], Lz["seg"].astype(str)
        sub = valid & ((seg == "junction") | win)
        # ---- the fed memory, and the join of pool tab / feat / labels / meta
        Fd, zd = fed("on1_JDG")
        Fv, zv = fed("on1_JDGdev")
        Fj, zj = fed("on1_JDGJ")
        Fsrc, _ = fed("on1_L18RSG")
        F = np.zeros((N, RM.DIM), np.float32)
        miss = []
        for i in np.flatnonzero(valid):
            src = Fv if dev[i] else Fd
            if names[i] not in src:
                miss.append(names[i])
                continue
            F[i] = src[names[i]]
        join = dict(valid=int(valid.sum()), missing_fed=len(miss))
        jd = [max(float(np.abs(Fj[n] - F[i]).max()), 0.0) for i, n in zip(np.flatnonzero(sub & ~dev), names[sub & ~dev])]
        join["A1_feat_vs_full_max"] = max(jd) if jd else None
        join["A1_rows"] = dict(train=int((sub & ~dev).sum()), dev=int((sub & dev).sum()), in_JDGJ=sum(n in Fj for n in names[sub & ~dev]))
        join["fed_vs_source_feat_max"] = float(max(np.abs(F[i] - Fsrc[names[i]]).max() for i in np.flatnonzero(valid)))
        P = RM.pts(F)
        Ps = RM.pts(M["rmem"].astype(np.float32))
        vb = (P[..., 0] > 0.5) | (Ps[..., 0] > 0.5)
        dm = np.where(vb, np.abs(P - Ps).max(-1), 0).max(-1)
        mask_eq = ((P[..., 0] > 0.5) == (Ps[..., 0] > 0.5)).all(-1)
        join["fed_vs_served_rmem"] = dict(rows=int(valid.sum()), max_p50=float(np.median(dm[valid])), max_p99=float(np.percentile(dm[valid], 99)),
                                          rows_mask_differs=int((~mask_eq & valid).sum()), rows_gt_0p01=int(((dm > 0.01) & valid).sum()))
        rt = np.load(CR / "on1_L18RSG" / "route.npz")
        assert (rt["tokens"] == names).all()
        poly = rt["poly"][:, 2:2 + RM.N_PT].astype(float)                          # LAB -10 + 2 i: slot i (-6 + 2 i) is LAB index i + 2
        vv = (P[..., 0] > 0.5) & np.isfinite(poly).all(-1)
        dpl = np.where(vv, np.hypot(*(P[..., 1:3] * 20.0 - poly).transpose(2, 0, 1)), 0).max(-1)
        join["fed_vs_hinge_poly_max"] = dict(p50=float(np.median(dpl[valid])), p99=float(np.percentile(dpl[valid], 99)), max=float(dpl[valid].max()))
        for pool in ("JDGd", "JDGdev"):
            T = dict(np.load(CR / f"on1_{pool}" / "tab.npz"))
            ix = {n: i for i, n in enumerate(names.tolist())}
            r = np.array([ix[n] for n in T["names"].tolist()])
            join[pool] = dict(rows=len(r), fut_vs_target_max=float(np.abs(T["fut"] - Lz["target"][r]).max()),
                              served_vs_meta_max=float(np.abs(T["served"] - M["poses"][r]).max()),
                              speed_vs_meta_max=float(np.abs(T["speed"] - M["speed"][r]).max()), log_equal=bool((T["log"].astype(str) == M["logu"][r].astype(str)).all() or (T["log"].astype(str) == logs[r]).all()))
        kk = {n: int(x) for n, x in zip(names.tolist(), k.tolist())}
        kd = [(("on:" + n.split(":", 2)[2]) if n.startswith("on:d:") else n, int(x)) for z_ in (zd, zv) for n, x in zip(z_["names"].tolist(), z_["k"].tolist())
              if not n.startswith("on:s:")]
        join["feat_k_vs_meta_k_mismatch"] = int(sum(kk[n] != x for n, x in kd))
        join["feat_d_rows"] = len(kd)
        run.info(json.dumps(join, default=float))
        # ---- per scene metrics
        pack = {}
        for d_ in JG.PACKS:
            for sc in np.unique(scene):
                f = d_ / f"{sc}.npz"
                if sc not in pack and f.exists():
                    pack[sc] = str(f)
        jobs = []
        for sc in sorted(pack):
            i = np.flatnonzero(scene == sc)
            i = i[np.argsort(k[i])]
            if not valid[i].any():
                continue
            jobs.append((sc, pack[sc], i, M["anchor"][i].astype(float), valid[i], F[i], M["gpath"][i], Lz["target"][i], M["poses"][i],
                         M["t_us"][i].astype(float), M["logfut"][i].astype(float)))
        res = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for r in run.tqdm(ex.map(scene_job, jobs, chunksize=4), total=len(jobs), desc="scenes"):
                res |= r
        keys = sorted(next(iter(res.values())).keys())
        R = {q: np.full(N, np.nan) for q in keys}
        for i, r in res.items():
            for q in keys:
                R[q][i] = r[q]
        done = np.zeros(N, bool)
        done[list(res)] = True
        np.savez(OUT / "rows.npz", names=names, done=done, valid=valid, dev=dev, win=win, seg=seg, sub=sub, **R)
        segm = dict(junction=seg == "junction", approach=seg == "turn_appr", window=win, straight=seg == "straight",
                    other=~np.isin(seg, ["junction", "turn_appr", "straight"]))
        sets = dict(full_train=valid & ~dev, full_dev=valid & dev, A1_train=sub & ~dev, A1_dev=sub & dev)
        cov = {}                                                      # the memory's near field on the other row sets the trainer feeds
        for nm in ("navtrain_full.s0of12", "on1_L18RB0"):
            z = np.load(FEAT / f"{nm}.npz")
            Q = RM.pts(z["f"][z["ok"]].astype(np.float32))
            vq = Q[..., 0] > 0.5
            fa = np.where(vq.any(-1), RM.OFFS[np.argmax(vq, -1)], np.nan)
            cov[nm] = dict(rows=int(len(fa)), no_slot=float(np.isnan(fa).mean()), first_med=float(np.nanmedian(fa)), first_gt18=float(np.mean(fa > 18)))
        out = dict(memory_cover_other=cov, join=join, rows_done=int(done.sum()), rows_valid=int(valid.sum()), xs=XS, exit_deg=EXIT_DEG, sets={})
        for s_, m in sets.items():
            out["sets"][s_] = dict(all=summ(R, m & done, logs), seg={g: summ(R, m & done & segm[g], logs) for g in SEGS})
        # ---- 20 random rows: the rule expert re-run at the meta anchor vs the stored label; ids side by side
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(np.flatnonzero(valid & done), 20, replace=False))
        Td = {p_: dict(np.load(CR / f"on1_{p_}" / "tab.npz")) for p_ in ("JDGd", "JDGdev")}
        tix = {p_: {n: i for i, n in enumerate(T_["names"].tolist())} for p_, T_ in Td.items()}
        lines = ["row | pool | tab name == meta name | meta scene / k / log | anchor (x, y, yaw) | speed meta / tab | feat name, k | "
                 "fed vs served rmem max | label: expert re-run vs stored max | tab fut vs label target max | seg / win"]
        for i in pick:
            p_ = "JDGdev" if dev[i] else "JDGd"
            T_ = Td[p_]
            ti = tix[p_][names[i]]
            z_ = zv if dev[i] else zd
            fn = [j for j, n in enumerate(z_["names"].tolist()) if n.endswith(names[i][3:]) and (n == names[i] or n == "on:d:" + names[i][3:])][0]
            sc = scene[i]
            ii = np.flatnonzero(scene == sc)
            ii = ii[np.argsort(k[ii])]
            sf = JD / "stops/train" / f"{sc}.npz"
            _, _, P1, _, _ = JG.expert_scene((sc, pack[sc], str(sf) if sf.exists() else "", ii, M["anchor"][ii].astype(float), M["speed"][ii].astype(float),
                                              M["t_us"][ii].astype(np.int64)))
            xr = float(np.nanmax(np.abs(P1[np.flatnonzero(ii == i)[0]] - Lz["expert_st"][i])))
            lines.append(f"{i} | {p_} | {T_['names'][ti] == names[i]} | {sc[:40]} / {k[i]} / {logs[i][:26]} | "
                         f"({M['anchor'][i][0]:.2f}, {M['anchor'][i][1]:.2f}, {M['anchor'][i][2]:.3f}) | {M['speed'][i]:.3f} / {T_['speed'][ti]:.3f} | "
                         f"{z_['names'][fn][-20:]}, {int(z_['k'][fn])} | {dm[i]:.4f} | {xr:.2e} | {float(np.abs(T_['fut'][ti] - Lz['target'][i]).max()):.1e} | "
                         f"{seg[i]} / {int(win[i])}")
        (OUT / "join20.txt").write_text("\n".join(lines) + "\n")
        out["join20"] = lines
        (OUT / "route_check.json").write_text(json.dumps(out, indent=1, default=float))
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "route_check.json").write_text(json.dumps(out, indent=1, default=float))
        print("\n".join(lines))
        print(json.dumps({s_: v["all"] for s_, v in out["sets"].items()}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("read")
    p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dict(read=cmd_read)[a.cmd](a)
