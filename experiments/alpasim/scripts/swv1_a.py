#!/usr/bin/env python3
"""Lane SWV1, part A (plans/2026-10-10-swv1-prereg.md): per decision, the served plan's swept ego box against the recorded object boxes,
plan kinematics, what the simulator's controller delivered; per collision rollout, the class (i) straight into an in-path vehicle,
(ii) manoeuvre whose planned sweep intersects the struck object, (iii) manoeuvre whose planned sweeps clear, (iv) drift, (v) other.

  swv1_a.py [--dom nuplan pai pai-v1]  -> tmp/swv1/dec_<dom>.pkl (per-decision rows, reused by swv1_b.py / swv1_c.py),
                                          results/swerve/{decisions_<dom>.csv, classes.csv, classes.md}
Simulator boxes and the logged path are labels only."""
import argparse
import csv
import pickle
from collections import Counter, defaultdict

import numpy as np

import swv1_lib as S

CL = S.CL
OUT = S.RES / "swerve"
LEADS = ("lead stopped", "slow lead")


def p0_plan(r, i):
    """The shipped policy's plan of decision i in the world frame (41, 3), or None."""
    rp = r.rp
    if rp is None:
        return None
    if r.dom == "nuplan":
        j = next((j for j, x in enumerate(rp) if x["now"] == r.now[i]), None)
        return None if j is None else S.dense(r.plan[i, 0], rp[j]["poses_p0"])
    j = np.flatnonzero(rp["now"] == r.now[i])
    if not len(j):
        return None
    return S.dense(r.ego_rig(rp["t0"][j[0]])[0], rp["p0"][j[0]])


def rel(r, k, t):
    """Object k in the ego rig frame at sim time t -> (gap from the front bumper to its nearest corner, y_min, y_max, centre y)."""
    e = r.ego_rig(t)[0]
    p, ok = r.obj_pose(k, t)
    c = CL.into(e, CL.corners(*p[0], *r.size(k)))
    return float(c[:, 0].min() - (r.off + r.Le / 2)), float(c[:, 1].min()), float(c[:, 1].max()), float(c[:, 1].mean()), bool(ok[0])


def decisions(r):
    rows = []
    lat = r.lat(r.plan[:, 0, :2]) if len(r.plan) else np.zeros((0, 2))
    for i in range(len(r.plan)):
        now, plan = r.now[i], r.plan[i]
        if r.t_ev is not None and now >= r.t_ev:
            break
        v, kap, alat, dpsi, arc = S.plan_kin(plan)
        sw = r.sweep(i)
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(plan[:, :2], axis=0).T))]
        hits = {k: x for k, x in sw.items() if x[1] >= 0}
        ka = min(sw, key=lambda k: sw[k][0]) if sw else None
        kf = min(hits, key=lambda k: hits[k][1]) if hits else None
        e1, e0 = r.ego_rig(now + 0.5e6)[0], r.ego_rig(now)[0]
        d = e1[:2] - plan[5, :2]
        row = dict(dom=r.dom, set=r.set, scene=r.scene, log=r.log, grp=r.grp, k=i, now=round((now - r.T0) * 1e-6, 2),
                   t_to_ev=None if r.t_ev is None else round((r.t_ev - now) * 1e-6, 2), v_e=float(r.ego_v(now)[0]), lat_off=float(lat[i, 0]),
                   clr_any=sw[ka][0] if ka else 9.0, obj_any=ka, hit_any=bool(hits), ti_any=kf and hits[kf][1] * S.DT, di_any=kf and float(cum[hits[kf][1]]),
                   obj_hit=kf, v_plan0=float(v[0]), v_plan_max=float(v.max()), kap_max=float(np.abs(kap).max()), alat=alat, dpsi4=float(np.degrees(dpsi)),
                   arc4=arc, yaw_dem=float((plan[5, 2] - plan[0, 2]) / 0.5), yaw_del=float(CL.wrap(e1[2] - e0[2]) / 0.5),
                   track_lat=float(-d[0] * np.sin(plan[5, 2]) + d[1] * np.cos(plan[5, 2])) if now + 0.5e6 <= r.T1 else np.nan)
        fz = r.sweep(i, frozen=True)
        tr = r.sweep(i, horizon=r.T1)
        row["hit_frozen"], row["clr_frozen"] = any(x[1] >= 0 for x in fz.values()), min([x[0] for x in fz.values()], default=9.0)
        row["hit_trunc"] = any(x[1] >= 0 for x in tr.values())
        q = p0_plan(r, i)
        if q is not None:
            s0 = r.sweep(i, q)
            v0, _, al0, dp0, arc0 = S.plan_kin(q)
            row.update(hit_any_p0=any(x[1] >= 0 for x in s0.values()), clr_any_p0=min([x[0] for x in s0.values()], default=9.0), arc4_p0=arc0,
                       base_gap=float(np.hypot(*(q[20, :2] - plan[20, :2]))), base_lat=float(-(q[20, 0] - plan[20, 0]) * np.sin(plan[20, 2]) + (q[20, 1] - plan[20, 1]) * np.cos(plan[20, 2])),
                       alat_p0=al0)
        if r.struck:
            k = r.struck
            x = sw.get(k, (9.0, -1))
            g, ylo, yhi, yc, ok = rel(r, k, now)
            row.update(clr_st=x[0], hit_st=x[1] >= 0, ti_st=x[1] * S.DT if x[1] >= 0 else None, di_st=float(cum[x[1]]) if x[1] >= 0 else None,
                       short_st=r.shortfall(i, k) if x[1] >= 0 else 0.0, gap_st=g, ylo_st=ylo, yhi_st=yhi, yc_st=yc, obj_v=float(r.obj_v(k, now)[0]),
                       in_corr=bool(ok and g > -1.0 and ylo < r.We / 2 and yhi > -r.We / 2))
            # lateral position of the plan (ego frame) where it reaches the object's range
            pe = CL.into(plan[0], plan[:, :2])
            xr = g + r.off + r.Le / 2
            row["plan_y_at_obj"] = float(np.interp(xr, pe[:, 0], pe[:, 1])) if pe[-1, 0] >= xr > 0 else 0.0
            if q is not None:
                x0 = s0.get(k, (9.0, -1))
                row.update(clr_st_p0=x0[0], hit_st_p0=x0[1] >= 0)
        rows.append(row)
    return rows


def classify(r, D):
    """D: this rollout's decision rows (before impact). -> dict(cls, sub, facts)."""
    te = r.t_ev
    tx = r.tax or {}
    w3 = [d for d in D if 0.5 < d["t_to_ev"] <= 3.0] or [d for d in D if d["t_to_ev"] <= 3.0]
    w8 = [d for d in D if d["t_to_ev"] <= 8.0]
    e_ev = r.ego_rig(te)[0]
    lat_ev = float(tx["lat_off"]) if r.dom == "nuplan" else float(r.lat(e_ev[None, :2])[0, 0])      # nuPlan: COL1's published offset (box centres)
    kind = r.kind or ""
    on_path = abs(lat_ev) < 0.5
    # (i)
    is_i = kind in LEADS and on_path
    own = [d for d in D if d["t_to_ev"] <= 3.0]
    own_i = bool(own) and np.mean([d["in_corr"] for d in own]) >= 0.5 and all(abs(d["plan_y_at_obj"]) <= 0.5 for d in own)
    if r.dom == "pai" and own_i and kind not in ("crossing",):
        is_i = True
    # manoeuvre
    g = r.gt_rig
    gw = g[(g[:, 0] >= te - 8e6) & (g[:, 0] <= te + 8e6)]                # the ego may start a turn before the log does
    log_turn = float(np.degrees(np.unwrap(gw[:, 3])[-1] - gw[0, 3])) if len(gw) > 1 else 0.0
    plan_turn = max((abs(d["dpsi4"]) for d in w8), default=0.0)
    gs = r.o["logged"][0]["traj"]
    scene_turn = float(np.degrees(np.unwrap(gs[:, 3])[-1] - gs[0, 3]))
    turn = abs(log_turn) >= 20 or (r.dom == "nuplan" and abs(scene_turn) >= 20)          # refinement 1: the log turns; a turning plan on a straight log is drift
    go, go_obj = False, None
    on = next((j for j, d in enumerate(D) if abs(d["lat_off"]) > 0.3), None)
    if on is not None:
        d = D[on]
        t = r.T0 + d["now"] * 1e6
        t2 = min(t + 1e6, te)
        dirn = np.sign(float(r.lat(r.ego_rig(t2)[:, :2])[0, 0]) - d["lat_off"]) or np.sign(d["lat_off"])
        reach = max(10.0, 4 * d["v_e"])
        for k in r.ids:
            gg, ylo, yhi, yc, ok = rel(r, k, t)
            close = d["v_e"] - float(r.obj_v(k, t)[0]) * float(np.cos(r.obj_pose(k, t)[0][0, 2] - r.ego_rig(t)[0, 2]))
            # refinement 2: something to go around = it would be reached within 4 s at the current speeds
            if ok and -1.0 < gg < reach and gg < 4.0 * max(close, 0.0) + 2.0 and ylo < r.We / 2 and yhi > -r.We / 2 and (abs(yc) < 0.3 or dirn * yc < 0):
                go, go_obj = True, k
                break
    # refinement 3: the ego is on the logged path and clips a standing object beside it = a pass the log makes with room and the plan does not
    tight = bool(not is_i and on_path and float(r.obj_v(r.struck, te)[0]) < 0.5)
    man = turn or go or tight
    sweep_hit = any(d["hit_st"] for d in w3)
    first = next((d for d in w8 if d["hit_st"]), None)
    cls = "i" if is_i else ("ii" if sweep_hit else "iii") if man else "iv" if not on_path else "v"
    hits = [d for d in w3 if d["hit_st"]]
    last = D[-1] if D else {}
    return dict(dom=r.dom, set=r.set, scene=r.scene, log=r.log, kind=kind, cls=cls, manoeuvre="turn" if turn else "go-around" if go else "tight pass" if tight else "",
                go_around_same_obj=bool(go and go_obj == r.struck), t_ev=round((te - r.T0) * 1e-6, 2), ego_v_ev=round(float(r.ego_v(te)[0]), 2),
                obj_v_ev=round(float(r.obj_v(r.struck, te)[0]), 2), lat_ev=round(lat_ev, 2), log_turn=round(log_turn, 1), plan_turn_max=round(plan_turn, 1),
                n_dec=len(w8), n_w3=len(w3), sweep_hit_w3=sweep_hit, share_hit_w3=round(float(np.mean([d["hit_st"] for d in w3])), 2) if w3 else None,
                share_hit_w3_p0=round(float(np.mean([d["hit_st_p0"] for d in w3])), 2) if w3 and "hit_st_p0" in w3[0] else None,
                first_hit_lead_s=first and first["t_to_ev"], min_clr_w3=round(min((d["clr_st"] for d in w3), default=np.nan), 2),
                short_med=round(float(np.median([abs(d["short_st"]) for d in hits])), 2) if hits and np.isfinite([d["short_st"] for d in hits]).any() else None,
                short_max3=bool(hits) and bool(np.isnan([d["short_st"] for d in hits]).any()),
                alat_max=round(max((d["alat"] for d in w8), default=0.0), 2), plan_turn_on_straight=bool(plan_turn >= 20 and not turn), v_plan_last=round(last.get("v_plan_max", np.nan), 2),
                yaw_dem_rms=round(float(np.sqrt(np.mean([d["yaw_dem"] ** 2 for d in w8]))), 3) if w8 else None,
                yaw_del_rms=round(float(np.sqrt(np.mean([d["yaw_del"] ** 2 for d in w8]))), 3) if w8 else None,
                track_p95=round(float(np.nanpercentile([abs(d["track_lat"]) for d in w8], 95)), 2) if w8 and np.isfinite([d["track_lat"] for d in w8]).any() else None,
                in_corr_w3=round(float(np.mean([d["in_corr"] for d in own])), 2) if own else None, plan_y_at_obj_max=round(max((abs(d["plan_y_at_obj"]) for d in own), default=0.0), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dom", nargs="*", default=["nuplan", "pai", "pai-v1"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for dom in a.dom:
        R = S.nuplan() if dom == "nuplan" else S.pai("base" if dom == "pai" else "v1")
        rows, by = [], {}
        for r in R:
            if r.grp == "X":
                continue
            D = decisions(r)
            rows += D
            by[r.set, r.scene] = D
            if r.grp == "C":
                cases.append(classify(r, D))
                print(dom, r.set, r.scene[-16:], cases[-1]["kind"], "->", cases[-1]["cls"], cases[-1]["manoeuvre"], flush=True)
        pickle.dump(rows, open(S.TMP / f"dec_{dom}.pkl", "wb"))
        keys = list(dict.fromkeys(k for d in rows for k in d))
        with open(OUT / f"decisions_{dom}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for d in rows if d["grp"] == "C"])
    old = []
    if (OUT / "classes.csv").exists():
        old = [r for r in csv.DictReader(open(OUT / "classes.csv")) if not any(r["dom"] == c["dom"] and r["set"] == c["set"] for c in cases)]
    with open(OUT / "classes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(cases[0]))
        w.writeheader()
        w.writerows(old + cases)


if __name__ == "__main__":
    main()
