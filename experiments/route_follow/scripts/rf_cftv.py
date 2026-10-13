#!/usr/bin/env python3
"""Turning school, demonstration source 2 variant: pool `on1_CFTv` = on1_CFT (rf_cftpool.py; demos with the human's speed profile, src "human")
+ the rows on1_CFT dropped ONLY by its e2 rule (the human's speed does not fit the sharper counterfactual exit: demo peak lateral
acceleration > 4 m/s^2) with a SYNTHETIC longitudinal profile on the same path (src "synthetic"; rm_train --on-weight src=synthetic:W weights
them separately). Prereg experiments/route_follow/plans/2026-10-13-turning-school-prereg.md; note research/next-round/turning-school-rows.md.
Box, AlpaSim venv (rf_cftpool's), CPU. Not trained here.

Synthetic profile (per rescued row, on rf_cftpool.path's demo path P, 0.1 m steps):
  v_lim(u) = sqrt(A_LAT / |kappa(u)|) with kappa the path curvature smoothed over 2 m, capped at the human's highest speed over the 4 s;
  backward pass v(u) <= sqrt(v(u + du)^2 + 2 DECEL du) (braking <= 2.5 m/s^2 ahead of every curvature limit); start at the token's speed v0
  (tab `speed`); forward pass v(u + du) <= sqrt(v(u)^2 + 2 ACC du) (ACC 1.5 m/s^2). A row whose v0 exceeds the backward-pass speed at u = 0
  by more than 0.1 m/s cannot meet the limit with 2.5 m/s^2 braking: excluded (e2v). fut = P at the arc reached at t = 0.5 .. 4 s, yaw = tangent.
Re-checked on the rescued rows: e3 (actor overlap at the new timing), e4 (raster); e1 (branch continuity) is the path's and kept from on1_CFT.
Gates: G-v1 = rf_cftpool G-cft1 on the synthetic rows (demo inside sdf > 0 >= 99 %); G-v2 speed-profile sanity: the 8-step profile has peak
|v x yaw rate| <= 3.3 m/s^2 (A_LAT + 10 % for the 0.5 s discretisation), braking <= 2.75 m/s^2 between steps and first-step mean speed within
10 % of v0 (or 0.5 m/s) on >= 99 % of the synthetic rows; G-v3 the replicated on1_CFT selection equals on1_CFT's row names exactly;
G-v4 6 BEV figures of synthetic rows (figs/cftv_bev_{0..5}.png).

  $P experiments/route_follow/scripts/rf_cftv.py --data navtrain_full.s{0..11}of12 [--name CFTv]
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_cfpool as CF  # noqa: E402
import rf_cftpool as CT  # noqa: E402

A_LAT, DECEL, ACC, V_TOL = 3.0, 2.5, 1.5, 0.1
TK = 0.5 * np.arange(1, 9)


def profile(cum, hd, v0, vcap, alat=A_LAT):
    """Arc grid cum (n,), tangent hd (n,) -> (arc at t = 0.5 .. 4 s (8,), feasible, v (n,))."""
    du = np.diff(cum)
    k = np.gradient(hd) / np.maximum(np.gradient(cum), 1e-6)
    k = np.convolve(np.r_[[k[0]] * 10, k, [k[-1]] * 10], np.ones(21) / 21, "valid")
    v = np.minimum(np.sqrt(alat / np.maximum(np.abs(k), 1e-4)), vcap)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * DECEL * du[i]))
    ok = v0 <= v[0] + V_TOL
    v[0] = min(v0, v[0])
    for i in range(len(v) - 1):
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * ACC * du[i]))
    t = np.r_[0.0, np.cumsum(du / np.maximum(0.5 * (v[1:] + v[:-1]), 0.05))]
    return np.interp(TK, t, cum), bool(ok and t[-1] >= TK[-1]), v


def synth(job):
    """(C, fut, v0) -> fut_syn (8, 3), feasible, peak alat, max braking, first-step speed."""
    C, fut, v0 = job
    P, cum, hd, S, *_ = CT.path(C, fut.astype(np.float64))
    vcap = max(float((np.diff(S) / 0.5).max()), v0)
    lim = A_LAT
    for _ in range(4):                                                # the 2 m smoothing under-reads sharp centreline corners: tighten the limit
        sk, ok, _ = profile(cum, hd, v0, vcap, lim)                   # until the 0.5 s measure (the e2 rule's) is within A_LAT + 10 %
        out = np.stack([np.interp(sk, cum, P[:, 0]), np.interp(sk, cum, P[:, 1]), np.interp(sk, cum, hd)], 1).astype(np.float32)
        vs = np.diff(np.r_[0.0, sk]) / 0.5
        al = float(np.abs(vs * np.diff(np.r_[0.0, np.unwrap(out[:, 2])]) / 0.5).max())
        if al <= 1.1 * A_LAT:
            break
        lim *= (A_LAT / al) ** 2
    br = float(max(0.0, -np.diff(np.r_[v0, vs]).min() / 0.5))
    return out, ok, al, br, float(vs[0])


def synth_log(u):
    """(log, [(row index, token, centreline in the log frame, logged future, v0)]) -> [(row index, *synth)] (rf_cftpool.demo_one's frames)."""
    import corr_geom as CG
    log, items = u
    fr, xy, yaw = CG.log_frames(CF.split_dir(log), log)
    pos = {f["token"]: i for i, f in enumerate(fr)}
    out = []
    for g, tok, cl, fut, v0 in items:
        i = pos[tok]
        out.append((g,) + synth((CG.to_ego(cl, xy[i], yaw[i]), fut, v0)))
    return out


def main(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    dvs = splits.load(CF.DEV)
    with Run("route_follow", f"cftv-{a.name}", config=vars(a)) as run:
        run.use_split(dvs)
        zs = [dict(np.load(CF.OUT / "rows" / f"{d}.npz")) for d in a.data]
        ds = [dict(np.load(CF.OUT / "cft" / f"{d}.npz")) for d in a.data]
        Z = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
        Q = {k: np.concatenate([q[k] for q in ds]) for k in ds[0]}
        Z["data"] = np.concatenate([np.full(len(z["row"]), s) for s, z in enumerate(zs)])
        tabs = [np.load(CF.CR / d / "tab.npz") for d in a.data]
        flog = np.stack([tabs[s]["fut"][r] for s, r in zip(Z["data"], Z["row"])])
        spd = np.array([tabs[s]["speed"][r] for s, r in zip(Z["data"], Z["row"])], np.float64)
        dev_logs = set()
        for t in tabs:
            dev_logs |= set(t["log"][dvs.mask(t["names"])].tolist())
        cf_good = Z["ok"] & np.isfinite(flog).all((1, 2)) & ~np.isin(Z["log"], list(dev_logs))
        cand = np.flatnonzero(cf_good & np.isfinite(Q["fut"]).all((1, 2)))
        road = np.load(CF.ROAD)
        road = dict(tokens=road["tokens"], ok=road["ok"], sdf=road["sdf"])
        rpos = {t: j for j, t in enumerate(road["tokens"].tolist())}

        def sdf_rows(rows, F):
            out = []
            for m, g in enumerate(rows):
                j = rpos.get(Z["token"][g], -1)
                out.append(CF.sdf_at(road["sdf"][j], F[m, :, :2]) if j >= 0 and road["ok"][j] else np.full(8, np.nan))
            return np.array(out).reshape(-1, 8)
        AG = np.load(CT.AGENTS)
        AG = dict(tokens=AG["tokens"], ok=AG["ok"], box=AG["box"], valid=AG["valid"])
        sd = sdf_rows(cand, Q["fut"][cand])
        e1, e2 = Q["cont"][cand] > CT.CONT, Q["alat"][cand] > CT.ALAT
        e3 = CT.collide(Q["fut"][cand], Z["token"][cand], AG)[0]
        e4 = (np.nan_to_num(sd, nan=1.0) <= 0).any(1)
        keep = cand[~(e1 | e2 | e3 | e4)]
        ref = np.load(CF.CR / "on1_CFT" / "tab.npz")["names"]
        names_k = np.array([f"on:cft:{t}:{c}" for t, c in zip(Z["token"][keep], Z["cls"][keep])])
        gv3 = bool(len(ref) == len(keep) and np.array_equal(np.sort(ref), np.sort(names_k)))
        only2 = cand[e2 & ~e1 & ~e3 & ~e4]
        run.info(f"candidates {len(cand)}, on1_CFT replicated {len(keep)} (equal {gv3}), e2 {int(e2.sum())}, e2 only {len(only2)}")
        geo = {}
        for s, d in enumerate(a.data):
            geo.update({(s, r["token"], r["cls"]): r["cl"] for r in pickle.load(open(CF.OUT / "geom" / f"{d}.pkl", "rb"))})
        by = {}
        for g in only2:
            by.setdefault(Z["log"][g], []).append((int(g), Z["token"][g], geo[(Z["data"][g], Z["token"][g], Z["cls"][g])], flog[g], float(spd[g])))
        R = par.pmap(synth_log, sorted(by.items(), key=lambda x: -len(x[1])), run=run, workers=a.workers or None, desc="synth")
        R.raise_if_failed()
        res = sorted(x for v in R.values for x in v)
        gs = np.array([x[0] for x in res], int)
        Fs = np.stack([x[1] for x in res])
        feas, al, br, v1 = (np.array([x[k] for x in res]) for k in (2, 3, 4, 5))
        sds = sdf_rows(gs, Fs)
        e3s = CT.collide(Fs, Z["token"][gs], AG)[0]
        e4s = (np.nan_to_num(sds, nan=1.0) <= 0).any(1)
        ok = feas & ~e3s & ~e4s
        gsel, Fsel = gs[ok], Fs[ok]
        v0s = spd[gs]
        san = (al <= A_LAT * 1.1) & (br <= DECEL * 1.1) & (np.abs(v1 - v0s) <= np.maximum(0.1 * v0s, 0.5))
        fin = np.isfinite(sds[ok])
        arc_s = np.hypot(*np.diff(np.concatenate([np.zeros((len(Fsel), 1, 2)), Fsel[..., :2]], 1), axis=1).transpose(2, 0, 1)).sum(1)
        arc_h = Q["arc_log"][gsel]
        G = {"data": a.data, "cft_candidates": int(len(cand)), "cft_replicated": int(len(keep)), "G_v3_equals_on1_CFT": gv3,
             "e2_rows": int(e2.sum()), "e2_only_rows": int(len(only2)),
             "synthetic": {"infeasible_e2v": int((~feas).sum()), "e3_actor_overlap": int((feas & e3s).sum()), "e4_offroad": int((feas & ~e3s & e4s).sum()),
                           "kept": int(ok.sum()), "by_class": {c: int((Z["cls"][gsel] == c).sum()) for c in ("j0", "j1", "lc")}},
             "G_v1": {"demo_inside": float((sds[ok][fin] > 0).mean()) if fin.any() else None},
             "G_v2": {"sane_share": float(san[ok].mean()) if ok.any() else None, "alat_p99": float(np.quantile(al[ok], 0.99)) if ok.any() else None,
                      "brake_p99": float(np.quantile(br[ok], 0.99)) if ok.any() else None,
                      "arc4_synthetic_over_human_median": float(np.median(arc_s / np.maximum(arc_h, 1e-6))) if ok.any() else None,
                      "v0_median": float(np.median(v0s[ok])) if ok.any() else None,
                      "speed_at_4s_median": float(np.median(np.hypot(*(Fsel[:, 7, :2] - Fsel[:, 6, :2]).T) / 0.5)) if ok.any() else None}}
        G["G_v2"].update(alat_ok=float((al <= A_LAT * 1.1)[ok].mean()), brake_ok=float((br <= DECEL * 1.1)[ok].mean()),
                         v0_ok=float((np.abs(v1 - v0s) <= np.maximum(0.1 * v0s, 0.5))[ok].mean()))
        G["G_v1"]["pass"] = bool(G["G_v1"]["demo_inside"] is not None and G["G_v1"]["demo_inside"] >= 0.99)
        G["G_v2"]["pass"] = bool(G["G_v2"]["sane_share"] is not None and G["G_v2"]["sane_share"] >= 0.99)
        sel = np.r_[keep, gsel]
        fut = np.concatenate([Q["fut"][keep], Fsel]).astype(np.float32)
        if not a.gates_only:
            CF.write_pool(a.name, Z, sel, a.data, road, run, prefix="cftv", fut=fut)
            tp = CF.CR / f"on1_{a.name}" / "tab.npz"
            tb = dict(np.load(tp))
            tb["src"] = np.r_[np.full(len(keep), "human"), np.full(len(gsel), "synthetic")].astype("U16")
            tb["cls"] = Z["cls"][sel].astype("U4")
            np.savez(tp, **tb)
            G["pool"] = dict(rows=int(len(sel)), human=int(len(keep)), synthetic=int(len(gsel)), dir=str(CF.CR / f"on1_{a.name}"))
        figs = CF.R / "experiments/route_follow/figs"
        rng = np.random.default_rng(0)
        G["G_v4"] = []
        for q, m in enumerate(rng.permutation(len(gsel))[:6]):
            g = gsel[m]
            j = rpos.get(Z["token"][g], -1)
            p = figs / f"cftv_bev_{q}.png"
            info = (f"SYNTHETIC speed: v0 {spd[g]:.1f} m/s, arc4 human {Q['arc_log'][g]:.1f} / synth {arc_s[m]:.1f} m, alat human-speed {Q['alat'][g]:.1f} "
                    f"-> {al[ok][m]:.1f}")
            CT.figure(p, {k: v[g] for k, v in Z.items()}, road["sdf"][j] if j >= 0 else np.zeros((128, 96), np.float16), flog[g], Fsel[m], info)
            G["G_v4"].append(dict(fig=str(p.relative_to(CF.R)), token=str(Z["token"][g]), cls=str(Z["cls"][g])))
        (CF.OUT / f"cftv_gates_{a.name}.json").write_text(json.dumps(G, indent=1))
        shutil.copy(CF.OUT / f"cftv_gates_{a.name}.json", CF.R / f"experiments/route_follow/results/cftv_gates_{a.name}.json")
        run.summary.update(G)
        print(json.dumps(G, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True), ap.add_argument("--name", default="CFTv"), ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--gates-only", action="store_true")
    main(ap.parse_args())
