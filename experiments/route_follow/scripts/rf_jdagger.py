#!/usr/bin/env python3
"""JOINT-DAGGER (experiments/route_follow/plans/2026-10-16-joint-dagger-prereg.md): the rule expert's stop rule data, the joint (lateral +
longitudinal) DAgger labels on RSG-L18-s0's own states, the training pools and the pilot gates. Box; envs/op-train (numpy + torch + pandas).

  rf_jdagger.py stops --pack DIR --out DIR [--workers N]      CPU: rule_expert's stop-rule files (<scene>.npz) for the scenes of a pack dir:
        LONG-PROBE's stop lines of the scene (onpol_nuplan ctx.npz: route in the origin frame, ln_*, kf_ts) mapped onto the pack's path. The
        simulation frame is the nuPlan map frame shifted by O (on_rows: map = local + O, no rotation) and simulation time = log time - T0;
        O and T0 per scene from every on-policy pool's meta (pose_map - anchor, t_log_us - t_us). A line whose point lies > 3 m from the path
        is not eligible. Check: ctx keyframe poses (map) - O against the pack's log path at kf_ts - T0 -> DIR/index.json
  rf_jdagger.py label [--workers N]                           CPU: for every row of on1_L18RSG with a pack (rf_dagger's d319 stage packs):
        (1) bit check: the expert with no stop file reproduces DAGGER-CHECK's stored plans (expert_L18RSG.npz) exactly;
        (2) the expert with the stop rule (stops/train), its 41 simulated states;
        (3) the joint target: the expert's path re-timed by v(a) = min(the expert's own speed at that arc, on_envelope's upper bounds at the
            row's own state: stop line (vuf, front-bumper arc), turn entry (vur, rear-axle arc), lead in time (lsig per 0.1 s step)), the
            expert's speed controller (A_ACC up, A_MAX down) from the row's speed; with no bound binding it is the expert's plan exactly;
        (4) validity: the target's 0.1-4 s footprint against the pool's agent boxes and road raster (rf_dagger.contacts, rf_rdom's hinge
            geometry); rows with a contact, a t = 0 contact or no agent / road label are dropped;
        (5) weights: hi = the 3 s before a failed turn (d319 units' k_ref - 6 .. k_ref - 1, d318's failure states) or the junction / turn
            approach segment (DAGGER-CHECK rows.parquet); dev = the crc32 log rule.
        -> $JD/labels.npz, $JD/labels.json (counts, how often each bound binds, target vs served / expert)
  rf_jdagger.py pack                                          CPU: on1_JDGd (train-log rows, fut = target, jw = hi / lo), on1_JDGdev (dev-log rows,
        never trained), on1_JDG = on1_L18RB0 (S rows, self) + on1_JDGd (both), and the flags per dose -> $JD/flags_<dose>.txt, $JD/pack.json
  rf_jdagger.py packj                                         CPU, Amendment 1: on1_JDGjd = the on1_JDGd rows where the teacher beats the policy
        (junction segment or the 3 s before a failed turn; turn approach and straight / free-road rows dropped), on1_JDGJ = on1_L18RB0 (S, self)
        + on1_JDGjd (both), uniform draw inside the subset, flags per dose (j10 / j15: 13 / 19 DAgger rows per 128-row batch, expected)
        -> $JD/flags_j<share>.txt, $JD/pack_j.json
  rf_jdagger.py gate --tag T [--base RSG-L18-s0] [--sub] [--submit-held]   CPU: the registered pilot gates (G-real dev ADE <= 0.56, G-rely, equal-arc
        turning >= +0.3 deg, DAgger-label ADE on dev rows drop >= 30 %; failure-window rows reported apart) -> $JD/gate/<tag>.json and a copy in
        results/joint_dagger/; --submit-held: on PASS, the held 460 @18 read and its report (rf_recov_post.sh submit-held)
  rf_jdagger.py xpgate --dir DIR                              (AlpaSim venv) the expert's held read with the stop rule against the stored one
        (rf_follow / ol_confirm outputs in DIR + per-scene scores): completed turns, zeros, zeros in scenes where the human stopped at a line
  rf_jdagger.py bitcmp                                        the 60-step identity runs: JDGG-off (RSG-L18 recipe) vs JDGG-w0 (pool on1_JDG,
        DAgger rows at weight 0, --on-mode mixed): scalars, weights, on_hash_60 = RSG-L18-s0's stored hash -> $JD/gates.json
  rf_jdagger.py table                                         every pilot gate + held lines -> results/joint_dagger/pilot_gates.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

R_ = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(R_), str(HERE), str(R_ / "lib"), str(R_ / "experiments/onpol_nuplan/lib"), str(R_ / "experiments/onpol_nuplan/scripts"),
                str(R_ / "experiments/alpasim/lib"), str(R_ / "experiments/long_contact/lib"), str(R_ / "experiments/body1/lib"),
                str(R_ / "experiments/route_mem/lib"), str(R_ / "experiments/corridor/lib")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, ROWS = D / "runs/op_parity/cache", D / "runs/onpol_nuplan/rows"
RF = D / "runs/route_follow"
JD = RF / "joint_dagger"
RES = R_ / "experiments/route_follow/results/joint_dagger"
DC = RF / "dagger_check"
RW = RF / "rewind_onpol_all"
PACKS = [RW / s / "pack" for s in ("rest", "s50", "smoke")] + [RF / "rewind_train64/pack"]
BIG, LAT_ROUTE, FRONT = 1e3, 3.5, 4.049
A_ACC, A_DEC, A_MAX, STOP_M, MARGIN_UP, DT = 1.5, 3.0, 6.0, 1.5, 1.0, 0.1
LINE_LAT = 3.0
K_S = 13
DOSES = {"25": 32, "50": 64}                      # DAgger share of the 128-row batch -> DAgger rows per batch
DOSES_J = {"j10": 13, "j15": 19}                  # Amendment 1: 10 / 15 % of 128 = 12.8 / 19.2 -> the nearest whole rows (10.2 / 14.8 %)
HI_W = 3.0
LINES = dict(dev_ade=0.56, turn=0.3, lab_drop=0.30)


def seg_proj(P, X):
    """Points X (m, 2) on polyline P (n, 2) -> (arc as segment index + u, signed lateral left +), global nearest segment."""
    A, B = P[:-1], P[1:]
    d = B - A
    L2 = np.maximum((d * d).sum(1), 1e-12)
    u = np.clip(((X[:, None, :] - A[None]) * d[None]).sum(-1) / L2[None], 0, 1)
    q = A[None] + u[..., None] * d[None]
    e = ((q - X[:, None, :]) ** 2).sum(-1)
    j = e.argmin(1)
    r = np.arange(len(X))
    cr = d[j, 0] * (X[:, 1] - A[j, 1]) - d[j, 1] * (X[:, 0] - A[j, 0])
    return j + u[r, j], np.sign(cr) * np.sqrt(e[r, j])


def stations_of(P, arc):
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    return np.interp(arc, np.arange(len(P)), S)


# ---------------------------------------------------------------- stop-rule files
def scene_frames():
    """{scene: (T0 us, O (2,), T0 spread us, O spread m, rows)} from every on-policy pool meta."""
    acc = {}
    for f in sorted(ROWS.glob("*/meta.npz")):
        try:
            m = np.load(f)
            if not {"t_us", "t_log_us", "pose_map", "anchor", "scene"} <= set(m.files):
                continue
            sc = m["scene"].astype(str)
            T0 = m["t_log_us"].astype(np.int64) - m["t_us"].astype(np.int64)
            Oxy = m["pose_map"][:, :2].astype(float) - m["anchor"][:, :2].astype(float)
        except Exception:  # noqa: BLE001
            continue
        order = np.argsort(sc, kind="stable")
        u, st = np.unique(sc[order], return_index=True)
        for s, a, b in zip(u, st, np.r_[st[1:], len(order)]):
            i = order[a:b]
            acc.setdefault(s, ([], []))
            acc[s][0].append(T0[i])
            acc[s][1].append(Oxy[i])
    out = {}
    for s, (T, Oo) in acc.items():
        T, Oo = np.concatenate(T), np.concatenate(Oo)
        om = np.median(Oo, 0)
        out[s] = (int(np.median(T)), om, int(T.max() - T.min()), float(np.hypot(*(Oo - om).T).max()), int(len(T)))
    return out


def stops_scene(job):
    sc, pf, cx, T0, O, out = job
    z = np.load(pf)
    P = z["path"].astype(float)
    S = z["s"].astype(float)
    ox, oy, oyaw = (float(v) for v in cx["origin"])
    R = cx["route"].astype(float)
    c, s = np.cos(oyaw), np.sin(oyaw)
    G = np.c_[ox + c * R[:, 0] - s * R[:, 1], oy + s * R[:, 0] + c * R[:, 1]] - O       # the ctx route in the simulation frame
    ty, ld = cx["ln_type"].astype(np.int8), cx["ln_d"].astype(float)
    n = len(ty)
    st_s, lat = np.full(n, np.nan), np.full(n, np.nan)
    have = (ty > 0) & np.isfinite(ld)
    if have.any():
        g = np.arange(len(G), dtype=float)
        X = np.c_[np.interp(ld[have], g, G[:, 0]), np.interp(ld[have], g, G[:, 1])]
        a, l_ = seg_proj(P, X)
        st_s[have], lat[have] = np.interp(a, np.arange(len(P)), S), np.abs(l_)
    elig = cx["ln_elig"].astype(bool) & have & (lat <= LINE_LAT)
    kts = cx["kf_ts"].astype(np.int64)
    kf_t = np.where(kts > 0, kts - T0, -1).astype(np.int64)
    # frame / time check: the ctx keyframe rear-axle poses (map) - O against the pack's log path at the keyframe's simulation time
    lt, lxy = z["log_t"].astype(float), z["log_xy"].astype(float)
    kp = cx["kf_pose"][:, :2].astype(float) - O
    ok = (kf_t >= lt[0]) & (kf_t <= lt[-1]) & np.isfinite(kp).all(1)
    err = np.hypot(kp[ok, 0] - np.interp(kf_t[ok], lt, lxy[:, 0]), kp[ok, 1] - np.interp(kf_t[ok], lt, lxy[:, 1])) if ok.any() else np.zeros(0)
    np.savez(Path(out) / f"{sc}.npz", st_s=st_s, typ=ty, elig=elig, hs=np.isfinite(cx["ln_tstop"].astype(float)), kf_t=kf_t,
             st=cx["ln_st"].astype(np.int8), lat=lat, ln_d=ld)
    return dict(scene=sc, lines=int(have.sum()), elig=int(elig.sum()), elig_ctx=int((cx["ln_elig"].astype(bool) & have).sum()),
                far=int((have & (lat > LINE_LAT)).sum()), kf_err_med=float(np.median(err)) if len(err) else None,
                kf_err_max=float(err.max()) if len(err) else None, kf_n=int(len(err)))


def cmd_stops(a):
    from concurrent.futures import ProcessPoolExecutor
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", f"jdg-stops-{out.name}", config=vars(a)) as run:
        X = dict(np.load(D / "runs/onpol_nuplan/ctx/ctx.npz"))
        cidx = {s: i for i, s in enumerate(X["scene"].astype(str).tolist())}
        fr = scene_frames()
        packs = {}
        for p in a.pack.split(","):
            for f in sorted(Path(p).glob("*.npz")):
                packs.setdefault(f.stem, str(f))
        keys = ("origin", "route", "ln_type", "ln_d", "ln_elig", "ln_tstop", "ln_st", "kf_ts", "kf_pose")
        jobs, miss = [], dict(ctx=[], frame=[])
        for sc, pf in sorted(packs.items()):
            if sc not in cidx:
                miss["ctx"].append(sc)
                continue
            if sc not in fr:
                miss["frame"].append(sc)
                continue
            T0, O, dT, dO, _ = fr[sc]
            jobs.append((sc, pf, {k: X[k][cidx[sc]] for k in keys}, T0, O, str(out)))
        res = []
        with ProcessPoolExecutor(a.workers) as ex:
            for r in run.tqdm(ex.map(stops_scene, jobs, chunksize=4), total=len(jobs), desc="scenes"):
                res.append(r)
        e = np.array([r["kf_err_max"] for r in res if r["kf_err_max"] is not None])
        rec = dict(packs=len(packs), written=len(res), no_ctx=len(miss["ctx"]), no_frame=len(miss["frame"]), missing=miss,
                   lines=sum(r["lines"] for r in res), elig_ctx=sum(r["elig_ctx"] for r in res), elig=sum(r["elig"] for r in res),
                   far=sum(r["far"] for r in res), kf_err_max_median=float(np.median(e)) if len(e) else None,
                   kf_err_max_p95=float(np.percentile(e, 95)) if len(e) else None, kf_err_max_max=float(e.max()) if len(e) else None,
                   frame_spread_T0_us_max=max((fr[r["scene"]][2] for r in res), default=0),
                   frame_spread_O_m_max=max((fr[r["scene"]][3] for r in res), default=0.0), scenes=res)
        (out / "index.json").write_text(json.dumps(rec, indent=1, default=float))
        run.summary.update({k: v for k, v in rec.items() if k not in ("scenes", "missing")})
        print(json.dumps({k: v for k, v in rec.items() if k != "scenes"}, indent=1, default=float))
        assert rec["kf_err_max_p95"] is None or rec["kf_err_max_p95"] < 0.5, "frame / time mapping off (keyframe check)"


# ---------------------------------------------------------------- labels
def expert_scene(job):
    """One scene's rows (k order, one session per variant) -> plans without stop file (bit check), plans + states with the stop rule, info."""
    import rule_expert as XP
    name, f, sf, idx, anc, v, t = job
    XP._scenes[name] = XP.Scene(Path(f))
    XP._scenes[name + "|st"] = XP.Scene(Path(f), Path(sf) if sf else None)
    n = len(idx)
    P0, P1 = np.full((n, 8, 3), np.nan, np.float32), np.full((n, 8, 3), np.nan, np.float32)
    ST = np.full((n, XP.NSTEP + 1, 5), np.nan)
    Q = np.full((n, 5), np.nan, np.float32)                         # hits, line distance, line type, stop distance, err
    for j in range(n):
        try:
            P0[j] = XP.plan(name, name, int(t[j]), anc[j], float(v[j]))[0]
            p, info, st = XP.plan(name + "|st", name + "|st", int(t[j]), anc[j], float(v[j]), dense=True)
            P1[j], ST[j] = p, st
            ln = info.get("line")
            Q[j] = (len(info["hits"]), np.nan if ln is None else ln[0], np.nan if ln is None else ln[1],
                    np.nan if info["stop"] is None else info["stop"], 0)
        except Exception:  # noqa: BLE001
            Q[j, 4] = 1
    for k in (name, name + "|st"):
        XP._scenes.pop(k)
        XP.close(k)
    return idx, P0, P1, ST, Q


def retime(st, anc, v0, route, vuf, vur, lsig, sf0, ok):
    """The joint target of one row: the expert's dense states st (41, 5) re-timed under the row's upper bounds -> (poses (8, 3) anchor frame,
    binding flags dict, re-timed speeds (41,))."""
    x0, y0, h0 = (float(q) for q in anc)
    v = st[:, 3]
    a = np.r_[0.0, np.cumsum(0.5 * (v[:-1] + v[1:]) * DT)]
    ai = a + 1e-9 * np.arange(len(a))                               # strictly increasing for interp (a stopped expert repeats its arc)
    vnext = np.r_[v[1:], v[-1]]
    c, s = np.cos(h0), np.sin(h0)
    Rm = np.array([[c, s], [-s, c]])
    xy = (st[:, :2] - [x0, y0]) @ Rm.T                              # expert dense rear axle in the row frame
    hh = np.unwrap(st[:, 2])
    bounded = ok and route is not None
    if bounded:
        NR = len(route)
        fx = xy + FRONT * np.c_[np.cos(hh - h0), np.sin(hh - h0)]
        sr, lr = seg_proj(route.astype(float), xy)
        sf, _ = seg_proj(route.astype(float), fx)
        on = (np.abs(lr) <= LAT_ROUTE) & (sr <= NR - 1) & (sf <= NR - 1)
        g = np.arange(NR, dtype=float)
    hit = dict(stop=False, turn=False, lead=False)
    ar, vv = 0.0, float(v[0])
    A, V = [0.0], [vv]
    for n in range(len(v) - 1):
        U = float(np.interp(ar, ai, vnext))
        if bounded:
            k = min(int(np.searchsorted(ai, ar, side="right")) - 1, len(ai) - 1)
            if on[max(k, 0)]:
                r_, f_ = float(np.interp(ar, ai, sr)), float(np.interp(ar, ai, sf))
                bu, bt = float(np.interp(f_, g, vuf)), float(np.interp(r_, g, vur))
                bl = BIG
                if lsig[n] < BIG / 2:
                    rem = float(lsig[n]) - STOP_M - f_
                    vl = np.sqrt(2 * A_DEC * rem) + MARGIN_UP if rem > 0 else 0.0
                    bl = max(vl, np.sqrt(max(v0 * v0 - 2 * A_MAX * (f_ - sf0), 0.0)))
                for key, b in (("stop", bu), ("turn", bt), ("lead", bl)):
                    if b < BIG / 2 and b < U - 1e-6:
                        hit[key] = True
                U = min(U, bu, bt, bl)
        vn = max(0.0, vv + float(np.clip((U - vv) / DT, -A_MAX, A_ACC)) * DT)
        ar = min(ar + 0.5 * (vv + vn) * DT, a[-1])
        vv = vn
        A.append(ar)
        V.append(vv)
    A = np.array(A)[5::5]
    q = np.c_[np.interp(A, ai, xy[:, 0]), np.interp(A, ai, xy[:, 1]), np.interp(A, ai, hh) - h0]
    q[:, 2] = (q[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return q.astype(np.float32), hit, np.array(V)


def retime_scene(job):
    idx, ST, anc, v0, route, vuf, vur, lsig, sf0, ok = job
    n = len(idx)
    T = np.full((n, 8, 3), np.nan, np.float32)
    H = np.zeros((n, 3), bool)
    V4 = np.full(n, np.nan, np.float32)
    for j in range(n):
        if not np.isfinite(ST[j]).all():
            continue
        q, hit, V = retime(ST[j], anc[j], float(v0[j]), route[j], vuf[j], vur[j], lsig[j], float(sf0[j]), bool(ok[j]))
        T[j], H[j], V4[j] = q, (hit["stop"], hit["turn"], hit["lead"]), V[-1]
    return idx, T, H, V4


def windows(scene, k):
    """Row indices in the 3 s before a failed turn: d319's event / off units (k_ref - 6 .. k_ref - 1) and d318's failure states (train)."""
    at = {(s, int(kk)): i for i, (s, kk) in enumerate(zip(scene.tolist(), k.tolist()))}
    out = set()
    for u in csv.DictReader(open(R_ / "experiments/route_follow/results/rewind_onpol_all/units.csv")):
        if u["outcome"] in ("event", "off") and u["k_ref"] != "":
            kr = int(u["k_ref"])
            out |= {at[(u["scene"], kk)] for kk in range(kr - 6, kr) if (u["scene"], kk) in at}
    S_ = np.load(RF / "route_dominance/states.npz")
    out |= set(S_["row"][(S_["set"] == "train") & (S_["kind"] == "fail")].astype(int).tolist())
    w = np.zeros(len(scene), bool)
    w[sorted(out)] = True
    return w


def cmd_label(a):
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    import rf_dagger as DG
    import rf_recov as RC
    from jevdrive.run import Run
    with Run("route_follow", "jdg-label", config=vars(a)) as run:
        M = dict(np.load(ROWS / "L18RSG" / "meta.npz"))
        Lb = dict(np.load(RF / "recovery_envelope/lab_L18RSG.npz"))
        tab = np.load(CR / "on1_L18RSG" / "tab.npz")
        old = np.load(DC / "expert_L18RSG.npz")
        names = M["names"]
        assert (Lb["names"] == names).all() and (tab["names"] == names).all() and (old["names"] == names).all()
        N = len(names)
        scene, k = M["scene"].astype(str), M["k"].astype(int)
        pack = {}
        for d in PACKS:
            for sc in np.unique(scene):
                f = d / f"{sc}.npz"
                if sc not in pack and f.exists():
                    pack[sc] = str(f)
        sdir = JD / "stops/train"
        jobs = []
        for sc in sorted(pack):
            i = np.flatnonzero(scene == sc)
            i = i[np.argsort(k[i])]
            sf = sdir / f"{sc}.npz"
            jobs.append((sc, pack[sc], str(sf) if sf.exists() else "", i, M["anchor"][i].astype(float), M["speed"][i].astype(float),
                         M["t_us"][i].astype(np.int64)))
        P0, P1 = np.full((N, 8, 3), np.nan, np.float32), np.full((N, 8, 3), np.nan, np.float32)
        ST = np.full((N, 41, 5), np.nan)
        Q = np.full((N, 5), np.nan, np.float32)
        with ProcessPoolExecutor(a.workers) as ex:
            for idx, p0, p1, st, q in run.tqdm(ex.map(expert_scene, jobs, chunksize=4), total=len(jobs), desc="expert"):
                P0[idx], P1[idx], ST[idx], Q[idx] = p0, p1, st, q
        eq = (np.isnan(P0) == np.isnan(old["poses"])).all() and np.array_equal(np.nan_to_num(P0), np.nan_to_num(old["poses"]))
        bit = dict(stop_rule_off_equals_stored=bool(eq), max_abs=float(np.nanmax(np.abs(P0 - old["poses"]))) if np.isfinite(P0).any() else None)
        run.info(f"bit check (expert without stop file vs DAGGER-CHECK's stored plans): {bit}")
        assert eq, f"rule_expert with the stop rule off differs from the stored plans: {bit}"
        jobs = []
        for sc in sorted(pack):
            i = np.flatnonzero(scene == sc)
            jobs.append((i, ST[i], M["anchor"][i].astype(float), Lb["v0"][i].astype(float), Lb["route"][i], Lb["vuf"][i], Lb["vur"][i], Lb["lsig"][i],
                         Lb["sf0"][i], Lb["ok"][i]))
        T = np.full((N, 8, 3), np.nan, np.float32)
        H = np.zeros((N, 3), bool)
        V4 = np.full(N, np.nan, np.float32)
        with ProcessPoolExecutor(a.workers) as ex:
            for idx, t_, h_, v_ in run.tqdm(ex.map(retime_scene, jobs, chunksize=4), total=len(jobs), desc="retime"):
                T[idx], H[idx], V4[idx] = t_, h_, v_
        has = np.isfinite(T).all((1, 2))
        C, okA, okR = DG.contacts("on1_L18RSG", {"target": T, "expert_st": P1, "served": M["poses"].astype(np.float32)}, np.arange(N))
        own = k >= 2
        dev = np.array([RC.is_dev(x) for x in M["log"].astype(str)])
        clean = has & okA & okR & ~C["target"]["agent"] & ~C["target"]["road"] & ~C["target"]["t0"]
        valid = own & clean
        rp = pd.read_parquet(DC / "rows.parquet")
        assert (rp.name.to_numpy() == names).all()
        seg = rp.seg.to_numpy().astype(str)
        win = windows(scene, k)
        hi = win | np.isin(seg, ["junction", "turn_appr"])
        same = has & np.isfinite(P1).all((1, 2)) & ~H.any(1)
        dpe = np.hypot(*(T[same][..., :2] - P1[same][..., :2]).transpose(2, 0, 1)).max(1) if same.any() else np.zeros(0)
        Dg = DG.disagree(M["poses"].astype(np.float32), T)
        np.savez(JD / "labels.npz", names=names, target=T, expert_st=P1, expert_nostop=P0, has=has, valid=valid, own=own, dev=dev, hi=hi, win=win,
                 seg=seg, bind=H, v4=V4, line=Q[:, 1:3], hits=Q[:, 0], okA=okA, okR=okR,
                 **{f"c_{c}_{q}": C[c][q] for c in C for q in ("agent", "road", "t0")})
        cnt = lambda s: dict(rows=int(s.sum()), scenes=len(set(scene[s].tolist())), logs=len(set(M["log"][s].astype(str).tolist())))  # noqa: E731
        sfree = lambda c, s: float((~C[c]["agent"][s] & ~C[c]["road"][s] & ~C[c]["t0"][s]).mean()) if s.any() else None  # noqa: E731
        rec = dict(rows=N, scenes_with_pack=len(pack), stop_files=int(sum((sdir / f"{s_}.npz").exists() for s_ in pack)),
                   bit=bit, expert_rows=int(np.isfinite(P1).all((1, 2)).sum()), expert_errors=int((Q[:, 4] == 1).sum()),
                   stop_rule_rows=int(np.isfinite(Q[:, 1]).sum()), target_rows=int(has.sum()),
                   own=cnt(own & has), valid=cnt(valid), valid_train=cnt(valid & ~dev), valid_dev=cnt(valid & dev),
                   dropped=dict(no_label=int((own & has & ~(okA & okR)).sum()), agent=int((own & has & C["target"]["agent"]).sum()),
                                road=int((own & has & C["target"]["road"]).sum()), t0=int((own & has & C["target"]["t0"]).sum())),
                   hi=dict(all=cnt(valid & hi), train=cnt(valid & hi & ~dev), dev=cnt(valid & hi & dev), win_train=cnt(valid & win & ~dev),
                           win_dev=cnt(valid & win & dev), junction_appr_train=cnt(valid & ~win & hi & ~dev)),
                   seg={s_: dict(rows=int((valid & (seg == s_)).sum()), bind_stop=float(H[valid & (seg == s_), 0].mean()) if (valid & (seg == s_)).any() else None,
                                 bind_turn=float(H[valid & (seg == s_), 1].mean()) if (valid & (seg == s_)).any() else None,
                                 bind_lead=float(H[valid & (seg == s_), 2].mean()) if (valid & (seg == s_)).any() else None) for s_ in DG.SEGS},
                   bind_share=dict(stop=float(H[valid, 0].mean()), turn=float(H[valid, 1].mean()), lead=float(H[valid, 2].mean()), any=float(H[valid].any(1).mean())),
                   unbound_target_vs_expert_max_m=dict(p50=float(np.median(dpe)) if len(dpe) else None, max=float(dpe.max()) if len(dpe) else None),
                   contact_free_own=dict(target=sfree("target", own & has & okA & okR), expert_stop=sfree("expert_st", own & has & okA & okR),
                                         served=sfree("served", own & has & okA & okR)),
                   disagree_valid=dict(d2_med=float(np.median(Dg["d2"][valid])), d4_med=float(np.median(Dg["d4"][valid])),
                                       lat4_abs_med=float(np.median(np.abs(Dg["lat4"][valid]))), lon4_mean=float(np.mean(Dg["lon4"][valid])),
                                       dh4_abs_med=float(np.median(np.abs(Dg["dh4"][valid])))))
        (JD / "labels.json").write_text(json.dumps(rec, indent=1, default=float))
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "labels.json").write_text(json.dumps(rec, indent=1, default=float))
        run.summary.update(dict(valid=rec["valid"], bit=bit))
        print(json.dumps(rec, indent=1, default=float))


# ---------------------------------------------------------------- pools
def cmd_pack(a):
    import rf_recov as RC
    assert not (CR / "on1_JDG" / "tab.npz").exists() or os.environ.get("JDG_FORCE"), "on1_JDG exists: JDG_FORCE=1 to rebuild"
    Lz = dict(np.load(JD / "labels.npz"))
    tab = np.load(CR / "on1_L18RSG" / "tab.npz")
    assert (tab["names"] == Lz["names"]).all()
    jw = np.where(Lz["hi"], "hi", "lo")
    rec = {}
    for nm, keep in (("JDGd", Lz["valid"] & ~Lz["dev"]), ("JDGdev", Lz["valid"] & Lz["dev"])):
        r = np.flatnonzero(keep)
        RC.subset("L18RSG", r, nm, dict(fut=Lz["target"][r].astype(tab["fut"].dtype), jw=jw[r].astype("U4"), jwin=np.where(Lz["win"][r], "1", "0"),
                                       jseg=Lz["seg"][r].astype("U9"), kind=np.where(Lz["win"][r], "win", "own").astype("U4"),
                                       dev=np.full(len(r), "1" if nm == "JDGdev" else "0")))   # rf_recov.dev_of reads it (the tab `log` is not the nuPlan log file)
        rec[nm] = dict(rows=int(len(r)), hi=int(Lz["hi"][r].sum()), win=int(Lz["win"][r].sum()))
    RC.merge(["on1_L18RB0:s:self", "on1_JDGd:d:both"], "JDG")
    Mt = dict(np.load(CR / "on1_JDG" / "tab.npz"))
    pool, jwm = Mt["pool"].astype(str), Mt["jw"].astype(str)
    n_s, n_lo, n_hi = int((pool == "s").sum()), int(((pool == "d") & (jwm == "lo")).sum()), int(((pool == "d") & (jwm == "hi")).sum())
    assert n_s and n_lo + n_hi == rec["JDGd"]["rows"] and (pool[:n_s] == "s").all(), "merge order: S rows first"
    n_eff = n_lo + HI_W * n_hi
    flags = {}
    for share, kd in DOSES.items():
        w = kd * n_s / (K_S * n_eff)
        flags[share] = f"--on on1_JDG --on-k {K_S + kd} --on-mode mixed --on-weight pool=d:{w:.6f},jw=hi:{HI_W:g}"
    flags["w0"] = f"--on on1_JDG --on-k {K_S} --on-mode mixed --on-weight pool=d:0"
    for k_, f in flags.items():
        (JD / f"flags_{k_}.txt").write_text(f + "\n")
    rec |= dict(s_rows=n_s, d_lo=n_lo, d_hi=n_hi, hi_weight=HI_W, hi_draw_share=float(HI_W * n_hi / n_eff), flags=flags,
                batch=dict(total=128, yaw_rate=13, s=K_S, dagger={s: kd for s, kd in DOSES.items()}, normal={s: 128 - 13 - K_S - kd for s, kd in DOSES.items()}))
    (JD / "pack.json").write_text(json.dumps(rec, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pack.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_packj(a):
    """Amendment 1: the junction + failure-window subset of on1_JDGd, merged with the S rows; uniform draw inside the subset."""
    import rf_recov as RC
    assert not (CR / "on1_JDGJ" / "tab.npz").exists() or os.environ.get("JDG_FORCE"), "on1_JDGJ exists: JDG_FORCE=1 to rebuild"
    T = dict(np.load(CR / "on1_JDGd" / "tab.npz"))
    seg, win = T["jseg"].astype(str), T["jwin"].astype(str) == "1"
    keep = np.flatnonzero((seg == "junction") | win)
    RC.subset("JDGd", keep, "JDGjd", {})
    RC.merge(["on1_L18RB0:s:self", "on1_JDGjd:d:both"], "JDGJ")
    Mt = dict(np.load(CR / "on1_JDGJ" / "tab.npz"))
    pool = Mt["pool"].astype(str)
    n_s, n_d = int((pool == "s").sum()), int((pool == "d").sum())
    assert n_d == len(keep) and (pool[:n_s] == "s").all(), "merge order: S rows first"
    Td = dict(np.load(CR / "on1_JDGdev" / "tab.npz"))
    dsub = (Td["jseg"].astype(str) == "junction") | (Td["jwin"].astype(str) == "1")
    flags, batch = {}, {}
    for share, kd in DOSES_J.items():
        w = kd * n_s / (K_S * n_d)                                  # expected S rows 13, DAgger rows kd per batch (weighted draw with replacement)
        flags[share] = f"--on on1_JDGJ --on-k {K_S + kd} --on-mode mixed --on-weight pool=d:{w:.6f}"
        p = kd / (K_S + kd)
        batch[share] = dict(total=128, yaw_rate=13, s_expected=K_S, dagger_expected=kd, dagger_share=kd / 128, dagger_sd=float(np.sqrt((K_S + kd) * p * (1 - p))),
                            normal=128 - 13 - K_S - kd, draws_per_row_1000=float(1000 * kd / n_d))
    for k_, f in flags.items():
        (JD / f"flags_{k_}.txt").write_text(f + "\n")
    rec = dict(src="on1_JDGd", rows=int(len(keep)), junction=int((seg[keep] == "junction").sum()), win=int(win[keep].sum()),
               both=int(((seg == "junction") & win).sum()), logs=len(set(T["log"][keep].astype(str))), dropped=dict(turn_appr=int(((seg == "turn_appr") & ~win).sum()), other=int(((seg != "turn_appr") & (seg != "junction") & ~win).sum())),
               dev=dict(rows=int(dsub.sum()), junction=int((Td["jseg"].astype(str) == "junction").sum()), win=int((Td["jwin"].astype(str) == "1").sum()),
                        logs=len(set(Td["log"][dsub].astype(str)))),
               s_rows=n_s, flags=flags, batch=batch)
    (JD / "pack_j.json").write_text(json.dumps(rec, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pack_j.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def turn_seg(tag, base):
    """Amendment 1: rf_recov's equal-arc turning read split by the row's place on the held turn: approach rel in [-40, 0) m, junction rel in
    [0, 10) m (both inside the registered [-40, 10) window), and the whole turn rel >= 0 (to exit + 10 m, descriptive)."""
    import rf_recov as RC
    (A, PA), (B, PB) = RC.turn_read(tag), RC.turn_read(base)
    X = A.merge(B, on=["scene", "k", "row", "rel", "bin", "log"], suffixes=("", "_b"))
    ha, hb, da = [], [], []
    for _, x in X.iterrows():
        pa, pb = PA[int(x.j)], PB[int(x.j_b)]
        a_ = min(RC.arc4(pa), RC.arc4(pb))
        ha.append(np.degrees(x.sg * RC.heading_at_arc(pa, a_)))
        hb.append(np.degrees(x.sg * RC.heading_at_arc(pb, a_)))
        da.append(RC.arc4(pa) - RC.arc4(pb))
    ha, hb, da = np.array(ha), np.array(hb), np.array(da)
    rel, lg = X.rel.to_numpy(), X.log.to_numpy()
    out = {}
    for nm, m in (("window", (rel >= -40) & (rel < 10)), ("approach", (rel >= -40) & (rel < 0)), ("junction", (rel >= 0) & (rel < 10)), ("turn_all", rel >= 0)):
        if m.sum() < 2:
            out[nm] = None
            continue
        r = RC.boot_diff(ha[m], hb[m], lg[m])
        out[nm] = dict(n=int(m.sum()), scenes=int(X[m].scene.nunique()), dh_mean=float((ha - hb)[m].mean()), ci=[r["lo"], r["hi"]],
                       d_arc4_mean=float(da[m].mean()))
    return out


# ---------------------------------------------------------------- gates
def lab_read(tag, base, pool, rows):
    """ADE of the tag's and the base's plans to the DAgger target (tab fut) on a fwd row set -> dict (all rows, window rows)."""
    import rf_recov as RC
    from jevdrive import stats
    rr, Pt = RC.plans(pool, rows, tag)
    rb, Pb = RC.plans(pool, rows, base)
    assert np.array_equal(rr, rb)
    Tb = np.load(CR / f"on1_{pool}" / "tab.npz")
    f, lg, w = Tb["fut"][rr], Tb["log"][rr].astype(str), Tb["jwin"][rr].astype(str) == "1"
    hi = Tb["jw"][rr].astype(str) == "hi"
    sg = Tb["jseg"][rr].astype(str)
    ade = lambda P: np.hypot(*(P[..., :2] - f[..., :2]).transpose(2, 0, 1)).mean(1)  # noqa: E731
    at, ab = ade(Pt), ade(Pb)
    out = {}
    for nm, m in (("all", np.ones(len(rr), bool)), ("hi", hi), ("win", w), ("junction", sg == "junction"), ("sub", (sg == "junction") | w),
                  ("approach", (sg == "turn_appr") & ~w), ("rest", (sg != "turn_appr") & (sg != "junction") & ~w)):
        if not m.any():
            out[nm] = None
            continue
        r = stats.paired(at[m], ab[m], groups=lg[m])
        out[nm] = dict(n=int(m.sum()), logs=len(set(lg[m])), pilot=float(at[m].mean()), base=float(ab[m].mean()), drop=float(1 - at[m].mean() / ab[m].mean()),
                       diff=[r["mean"], r["lo"], r["hi"]])
    return out


def cmd_gate(a):
    import rf_recov as RC
    sys.path[:0] = [str(R_ / "experiments/op_parity/scripts")]
    res, L = dict(tag=a.tag, base=a.base, lines=LINES), {}
    tg, _ = RC.trainer_gates(a.tag)
    res["trainer"] = tg
    L["dev_ade"] = tg["dev_ade"] is not None and tg["dev_ade"] <= LINES["dev_ade"]
    rl = tg["rely"] or {}
    L["rely"] = bool(rl["read"]) if "read" in rl else None
    res["turn"], _ = RC.gate_turn(a.tag, a.base)
    L["turn"] = res["turn"]["equal_arc"]["dh_mean"] >= LINES["turn"]
    if a.sub:                                                       # Amendment 1: turning by segment, the label line on the trained row classes
        res["turn_seg"] = turn_seg(a.tag, a.base)
    res["lab_dev"] = lab_read(a.tag, a.base, "JDGdev", "dev-rec")
    res["lab_key"] = "sub" if a.sub else "all"
    L["lab"] = res["lab_dev"][res["lab_key"]]["drop"] >= LINES["lab_drop"]
    try:
        res["lab_train"] = lab_read(a.tag, a.base, "JDGd", "train-rec")         # descriptive: the rows it trains on
    except FileNotFoundError:
        res["lab_train"] = None
    res["lines_ok"] = L
    res["pass"] = all(v for v in L.values() if v is not None) and L.get("rely") is not False and None not in (L["dev_ade"], L["turn"], L["lab"])
    (JD / "gate").mkdir(parents=True, exist_ok=True)
    (JD / "gate" / f"{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"gate_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("PASS" if res["pass"] else "FAIL", flush=True)
    if res["pass"] and a.submit_held:
        env = dict(os.environ, ARMROOT=str(JD), KIND="J", RESSUB="joint_dagger", PRIO=os.environ.get("PRIO", "8"))
        p = subprocess.run(["bash", str(HERE / "rf_recov_post.sh"), "submit-held", a.tag, a.tag, "-", "-", "-"], env=env, capture_output=True, text=True)
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        if p.returncode:
            raise SystemExit(f"held submit failed rc {p.returncode}")


def cmd_xpgate(a):
    """Expert held read with the stop rule (ST) vs the stored one (E46): from DIR/follow/follow.json, DIR/held18/confirm.json and the reads'
    per-scene scores; human-stop scenes = held scenes with an eligible line the human stopped before (ctx)."""
    d = Path(a.dir)
    lab_st, lab_x = a.st, a.xp
    J = json.loads((d / "follow/follow.json").read_text())
    turns = {lb: J["labels"][f"{lb}@18"]["outcomes"] | dict(turns=J["labels"][f"{lb}@18"]["turns"]) for lb in (lab_st, lab_x) if f"{lb}@18" in J["labels"]}
    conf = json.loads((d / "held18/confirm.json").read_text())["held18"].get(lab_st)          # ol_confirm: pairs keyed by the first label
    X = dict(np.load(D / "runs/onpol_nuplan/ctx/ctx.npz"))
    hs = (X["ln_elig"].astype(bool) & (X["ln_type"] > 0) & np.isfinite(X["ln_tstop"].astype(float))).any(1)
    hstop = set(X["scene"].astype(str)[hs].tolist())
    M = D / "runs/route_mem/read"

    def scores(lb):
        out = {}
        for f in glob.glob(str(M / lb / "c*/run/aggregate/results-summary.json")):
            for r in json.loads(open(f).read())["rollouts"]:
                if r.get("score") is not None:
                    out[r["clipgt_id"]] = float(r["score"])
        return out
    S = {lb: scores(lb) for lb in (lab_st, lab_x)}
    common = sorted(set(S[lab_st]) & set(S[lab_x]))
    zs = lambda lb, ss: int(sum(S[lb][s] == 0 for s in ss))  # noqa: E731
    hsc = [s for s in common if s in hstop]
    st_lines, xp_dec = 0, 0                                         # expert decisions, and those the stop rule stopped (info `line`)
    for f in glob.glob(str(M / lab_st / "c*/run/runs/*/driver/drive.jsonl")) + glob.glob(str(M / lab_st / "c*/run/driver-logs/drive.jsonl")):
        for ln in open(f, errors="replace"):
            if '"expert"' in ln:
                x = json.loads(ln).get("expert") or {}
                xp_dec += 1
                st_lines += "line" in x
    res = dict(turns=turns, paired=conf, scenes=len(common), zeros={lb: zs(lb, common) for lb in (lab_st, lab_x)}, human_stop_scenes=len(hsc),
               zeros_human_stop={lb: zs(lb, hsc) for lb in (lab_st, lab_x)}, mean={lb: float(np.mean([S[lb][s] for s in common])) for lb in (lab_st, lab_x)},
               expert_decisions=xp_dec, decisions_with_line_stop=st_lines,
               gross_regression=bool(turns.get(lab_st, {}).get("completed", 0) < turns.get(lab_x, {}).get("completed", 0) - 2))
    (d / "xpgate.json").write_text(json.dumps(res, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "xpgate.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


def cmd_bitcmp(a):
    import torch
    RSG_HASH60 = "ea8cacc3e451c577"

    def run(tag):
        d = sorted(glob.glob(f"{D}/runs/route_mem/train-{tag}/*/"))[-1]
        sc = {}
        for line in open(d + "events.jsonl"):
            e = json.loads(line)
            if e["kind"] == "scalar":
                sc[(e["tag"], e["step"])] = e["value"]
        return sc, json.load(open(d + "DONE")), torch.load(f"{D}/runs/op_parity/runs/{tag}/ckpt-final.pt", map_location="cpu", weights_only=False)["model"]
    (sa, da, ma), (sb, db, mb) = run("JDGG-off"), run("JDGG-w0")
    keys = [k for k in sorted(set(sa) & set(sb), key=str) if not k[0].startswith(("throughput/", "loss/drop/"))]
    dl = max(abs(sa[k] - sb[k]) for k in keys)
    dw = max(float((ma[g][k].float() - mb[g][k].float()).abs().max()) for g in ("net", "parity") for k in ma[g])
    same = all(set(ma[g]) == set(mb[g]) for g in ("net", "parity"))
    r = dict(a="JDGG-off", b="JDGG-w0", note="RSG-L18 recipe 60 steps vs pool on1_JDG with the DAgger rows at weight 0 (--on-mode mixed, on-k 13)",
             scalars=len(keys), scalar_max_abs_diff=dl, weight_max_abs_diff=dw, weight_keys_equal=same, on_hash_60=[da.get("on_hash_60"), db.get("on_hash_60")],
             rsg_l18_s0_on_hash_60=RSG_HASH60, dev_ade=[da.get("dev_ade"), db.get("dev_ade")])
    r["identical"] = bool(dl == 0 and dw == 0 and same and r["on_hash_60"][0] == r["on_hash_60"][1] == RSG_HASH60)
    (JD / "gates.json").write_text(json.dumps(r, indent=1, default=float))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gates.json").write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps(r, indent=1, default=float))
    raise SystemExit(0 if r["identical"] else 2)


def cmd_table(a):
    rows = []
    for f in sorted((JD / "gate").glob("*.json")):
        g = json.loads(f.read_text())
        t = g["tag"]
        ln = JD / t / "lines.json"
        h = json.loads(ln.read_text()) if ln.exists() else None
        H = (h or {}).get("lines", {})
        fmt = lambda k, p=1: (lambda v: f"{v[0]:+.{p}f} [{v[1]:+.{p}f}, {v[2]:+.{p}f}]")(H[k]["v"]) if k in H else "-"  # noqa: E731
        la, tu = g["lab_dev"], g["turn"]["equal_arc"]
        lt = g.get("lab_train") or {}
        rows.append(f"| {t} | {g['trainer']['dev_ade']:.4f} | {g['lines_ok']['rely']} | {tu['dh_mean']:+.2f} [{tu['ci'][0]:+.2f}, {tu['ci'][1]:+.2f}] | "
                    f"{la['all']['base']:.3f} -> {la['all']['pilot']:.3f} ({-100 * la['all']['drop']:+.1f} %) | "
                    + (f"{-100 * la['win']['drop']:+.1f} % (n {la['win']['n']})" if la.get("win") else "-") + " | "
                    + (f"{-100 * lt['all']['drop']:+.1f} %" if lt.get("all") else "-") + f" | {'PASS' if g['pass'] else 'FAIL'} | "
                    f"{fmt('H1_zeros')} | {fmt('H1_at_fault')} | {fmt('H1_gt', 3)} | {fmt('H2_held_turns')} | "
                    + ({None: "-", True: "PASS", False: "FAIL"}[h.get('H1')] if h else "-") + " |")
    md = ["# JOINT-DAGGER pilot gates", "", "Lines: dev ADE <= 0.56; G-rely true; equal-arc turning (pilot - RSG-L18-s0) >= +0.3 deg; DAgger-label ADE on dev-log rows "
          "drop >= 30 %; then held 460 @18 H1 vs the RSG-L18 2-seed mean (zeros <= +3, at-fault <= +3, GT <= +0.03 m).", "",
          "| pilot | dev ADE | rely | equal-arc turn (deg) | label ADE dev | window rows dev | label ADE train rows | offline | held zeros d | at-fault d | GT d (m) | held turns d | H1 |",
          "|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"] + rows
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pilot_gates.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("stops")
    p.add_argument("--pack", required=True), p.add_argument("--out", required=True), p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("label")
    p.add_argument("--workers", type=int, default=24)
    sp.add_parser("pack")
    sp.add_parser("packj")
    p = sp.add_parser("gate")
    p.add_argument("--tag", required=True), p.add_argument("--base", default="RSG-L18-s0"), p.add_argument("--submit-held", action="store_true")
    p.add_argument("--sub", action="store_true", help="Amendment 1: label line on the junction + failure-window dev rows, turning by segment")
    p = sp.add_parser("xpgate")
    p.add_argument("--dir", required=True), p.add_argument("--st", default="RSG-L18-s0-XP-L12-E46-ST"), p.add_argument("--xp", default="RSG-L18-s0-XP-L12-E46")
    sp.add_parser("bitcmp")
    sp.add_parser("table")
    a = ap.parse_args()
    JD.mkdir(parents=True, exist_ok=True)
    dict(stops=cmd_stops, label=cmd_label, pack=cmd_pack, packj=cmd_packj, gate=cmd_gate, xpgate=cmd_xpgate, bitcmp=cmd_bitcmp, table=cmd_table)[a.cmd](a)
