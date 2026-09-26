"""Pre-registered readout for the state-space policy behaviour exam (todos/2026-09-26-state-space-policies.md).

Reads the rollouts written by statepol_eval.py and the variant metadata from statepol_build.py, and prints /
writes one table per exam (A negotiation, B bypass, C recovery) plus the validity gate.
Rollout index k is the state after sim step k, i.e. time t0 + (k + 1) * 0.1 s (checked with the expert ego).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

T0 = 0  # rollout starts at log frame 0
k_at = lambda sec: int(round(sec * 10)) - 1
OBS_HALF = 2.4


def proj(path, pts):
    """Arc length along `path` of the closest point and signed lateral offset (left +) for each point."""
    a, b = path[:-1], path[1:]
    d = b - a
    L = np.maximum((d ** 2).sum(-1), 1e-9)
    s_cum = np.concatenate([[0], np.cumsum(np.sqrt(L))])
    t = np.clip(((pts[:, None] - a[None]) * d[None]).sum(-1) / L[None], 0, 1)
    foot = a[None] + t[..., None] * d[None]
    dist = np.linalg.norm(pts[:, None] - foot, axis=-1)
    i = dist.argmin(1)
    r = np.arange(len(pts))
    cross = d[i, 0] * (pts[:, 1] - a[i, 1]) - d[i, 1] * (pts[:, 0] - a[i, 0])
    return s_cum[i] + t[r, i] * np.sqrt(L[i]), np.sign(cross) * dist[r, i]


def log_path(m):
    v = m["ego_valid"][T0:]
    return m["ego_xy"][T0:][v]


def alive(ego, k):
    return len(ego) > k and ego[k, 0] > -9000 and not (ego[: k + 1, 5] > 0).any()


def ego_frame(m, xy):
    o, h = m["ego_xy"][T0], m["ego_h"][T0]
    R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    return (xy - o) @ R.T


def outcome(r):
    e = r["ego"]
    col = bool((e[:, 5] == 1).any())
    off = bool((e[:, 5] == 2).any())
    met = r.get("metrics") or {}
    return col, off, bool(met.get("goal_reached", False))


def load(run, variant, planner, traffic):
    """Rollouts with sim positions moved back into the log (JSON) frame."""
    p = Path(run) / f"{variant}__{planner}__{traffic}.pkl"
    if not p.exists():
        return None
    data = pickle.load(open(p, "rb"))
    metas = {m["map_id"]: m for m in pickle.load(open(Path(DATA) / variant / "meta.pkl", "rb"))}
    for mid, r in data.items():
        m = metas[mid]
        e0 = m["ego_xy"][0] + (m.get("shift_vec", 0) if variant == "shift" else 0)
        off = e0 - r["traj0"]
        for key in ("ego", "partner"):
            a = r[key]
            if len(a):
                ok = a[:, 0] > -9000
                a[ok, :2] += off.astype(a.dtype)
    return data


def exam_a(data, run, planner, traffic):
    base, nop, nul = (load(run, v, planner, traffic) for v in ("base", "nopartner", "nullrm"))
    if base is None or nop is None or nul is None:
        return None
    mb = {m["episode"]: m for m in pickle.load(open(Path(data) / "base/meta.pkl", "rb"))}
    idx = {v: {m["episode"]: m["map_id"] for m in pickle.load(open(Path(data) / v / "meta.pkl", "rb"))}
           for v in ("base", "nopartner", "nullrm")}

    def deltas(rp, rm, m):
        path = log_path(m)
        out = {}
        for sec in (3, 5):
            k = k_at(sec)
            if alive(rp["ego"], k) and alive(rm["ego"], k):
                sp, _ = proj(path, rp["ego"][k: k + 1, :2])
                sm, _ = proj(path, rm["ego"][k: k + 1, :2])
                out[f"dprog{sec}"] = float(sp[0] - sm[0])
        k = k_at(3)
        if alive(rp["ego"], k) and alive(rm["ego"], k):
            out["dv3"] = float(np.hypot(*rp["ego"][k, 2:4]) - np.hypot(*rm["ego"][k, 2:4]))
            for sec in (2, 3):
                kk = k_at(sec)
                yp, ym = ego_frame(m, rp["ego"][[kk], :2])[0, 1], ego_frame(m, rm["ego"][[kk], :2])[0, 1]
                py = ego_frame(m, m["partner_xy"][[T0 + kk + 1]])[0, 1]
                out[f"dlat{sec}"] = float((yp - ym) * (-np.sign(py) if py != 0 else 1))
        return out

    eff, nulls, order, cols = [], [], [], []
    for ep, m in mb.items():
        rb = base.get(idx["base"][ep])
        if rb is None:
            continue
        if ep in idx["nopartner"] and idx["nopartner"][ep] in nop:
            rn = nop[idx["nopartner"][ep]]
            eff.append(deltas(rb, rn, m))
            cols.append((outcome(rb)[0], outcome(rn)[0]))
        if ep in idx["nullrm"] and idx["nullrm"][ep] in nul:
            nulls.append(deltas(rb, nul[idx["nullrm"][ep]], m))
        # passing order on crossing pairs (log vs sim), partner = recorded partner state
        pv = m["partner_valid"]
        lp, pp = m["ego_xy"][T0 + 1:], m["partner_xy"][T0 + 1:]
        lv = m["ego_valid"][T0 + 1:] & pv[T0 + 1:]
        if lv.sum() < 20:
            continue
        D = np.linalg.norm(lp[:, None] - pp[None], axis=-1)
        D[~m["ego_valid"][T0 + 1:]] = np.inf
        D[:, ~pv[T0 + 1:]] = np.inf
        if D.min() >= 3:
            continue
        i, j = np.unravel_index(D.argmin(), D.shape)
        if i == j:
            continue
        log_first = i < j
        e = rb["ego"]
        par = rb["partner"] if len(rb["partner"]) else pp
        n = min(len(e), len(par))
        ok = (e[:n, 0] > -9000)
        if ok.sum() < 10:
            continue
        Ds = np.linalg.norm(e[:n, None, :2] - par[None, :n, :2], axis=-1)
        Ds[~ok] = np.inf
        conflict = lp[i]
        te = np.where(ok & (np.linalg.norm(e[:n, :2] - conflict, axis=1) < 3))[0]
        tp = np.where(np.linalg.norm(par[:n, :2] - conflict, axis=1) < 3)[0]
        if len(tp) == 0:
            continue
        if len(te) == 0:
            sim_first = False  # ego never reached the conflict point: it went second / did not go
        else:
            sim_first = te[0] < tp[0]
        order.append((log_first, sim_first))

    def arr(rows, key):
        return np.array([r[key] for r in rows if key in r])

    res = dict(n_eff=len(eff), n_null=len(nulls))
    tau_p = max(1.0, float(np.percentile(np.abs(arr(nulls, "dprog5")), 95))) if len(nulls) else 1.0
    tau_l = max(0.3, float(np.percentile(np.abs(arr(nulls, "dlat3")), 95))) if len(nulls) else 0.3
    ep5, np5 = arr(eff, "dprog5"), arr(nulls, "dprog5")
    el3, nl3 = arr(eff, "dlat3"), arr(nulls, "dlat3")
    res.update(tau_prog=tau_p, tau_lat=tau_l,
               yield_rate=float((ep5 < -tau_p).mean()), yield_null=float((np5 < -tau_p).mean()),
               faster_rate=float((ep5 > tau_p).mean()),
               lat_rate=float((el3 > tau_l).mean()), lat_null=float((nl3 > tau_l).mean()),
               dprog5_median=float(np.median(ep5)), dv3_median=float(np.median(arr(eff, "dv3"))))
    c = np.array(cols)
    res.update(col_with=float(c[:, 0].mean()), col_without=float(c[:, 1].mean()))
    o = np.array(order)
    if len(o):
        res.update(n_cross=len(o), order_agree=float((o[:, 0] == o[:, 1]).mean()),
                   ego_second_sim=float((~o[:, 1]).mean()), ego_second_log=float((~o[:, 0]).mean()))
    a = res["yield_rate"] >= max(0.10, 2 * res["yield_null"])
    b = res.get("order_agree", 0) >= 0.70
    cc = res["col_with"] <= res["col_without"] + 0.05
    res.update(crit_a=bool(a), crit_b=bool(b), crit_c=bool(cc))
    res["verdict"] = ("negotiates" if a and b and cc else "yields, order not human-like" if a and not b
                      else "yields, human-like order, but more collisions with partner present" if a
                      else "no reaction to partner")
    return res


def exam_b(data, run, planner, traffic):
    ob, base = load(run, "obstacle", planner, traffic), load(run, "obsctrl", planner, traffic)
    if ob is None or base is None:
        return None
    mo = pickle.load(open(Path(data) / "obstacle/meta.pkl", "rb"))
    bidx = {m["episode"]: m["map_id"] for m in pickle.load(open(Path(data) / "obsctrl/meta.pkl", "rb"))}
    cls, nul = [], []
    for m in mo:
        r = ob.get(m["map_id"])
        if r is None:
            continue
        c, hd = m["obs_xy"], m["obs_h"]
        fwd, left = np.array([np.cos(hd), np.sin(hd)]), np.array([-np.sin(hd), np.cos(hd)])
        path = log_path(m)

        def window_lat(e):
            ok = e[:, 0] > -9000
            p = e[ok, :2]
            if not len(p):
                return 0.0, -np.inf
            u = (p - c) @ fwd
            _, lat = proj(path, p)
            w = np.abs(u) <= 10
            return (float(np.abs(lat[w]).max()) if w.any() else 0.0), float(u.max())

        col, off, _ = outcome(r)
        lat, umax = window_lat(r["ego"])
        e = r["ego"]
        last = e[e[:, 0] > -9000]
        v_end = float(np.hypot(*last[-1, 2:4])) if len(last) else 0.0
        passed = umax >= OBS_HALF + 3
        k = ("collide" if col else "offroad" if off else "bypass" if passed and lat >= 1.0
             else "stop" if not passed and v_end < 0.5 else "other")
        cls.append(k)
        rb = base.get(bidx[m["episode"]])
        if rb is not None:
            nul.append(window_lat(rb["ego"])[0] >= 1.0)
    cls = np.array(cls)
    res = {f"{k}": float((cls == k).mean()) for k in ("bypass", "stop", "collide", "offroad", "other")}
    res.update(n=len(cls), null_lat=float(np.mean(nul)) if nul else float("nan"))
    if res["bypass"] >= 0.30 and res["bypass"] - res["null_lat"] >= 0.20 and res["collide"] <= 0.20:
        res["verdict"] = "bypasses"
    elif res["stop"] >= 0.50 and res["bypass"] < 0.10:
        res["verdict"] = "only stops"
    elif res["collide"] >= 0.30:
        res["verdict"] = "does not handle static obstacle"
    else:
        res["verdict"] = "mixed (no verdict)"
    return res


def exam_c(data, run, planner, traffic):
    sh, base = load(run, "shift", planner, traffic), load(run, "base", planner, traffic)
    if sh is None or base is None:
        return None
    ms = pickle.load(open(Path(data) / "shift/meta.pkl", "rb"))
    bidx = {m["episode"]: m["map_id"] for m in pickle.load(open(Path(data) / "base/meta.pkl", "rb"))}
    rows = {"shift": [], "base": []}
    for m in ms:
        for name, r in (("shift", sh.get(m["map_id"])), ("base", base.get(bidx[m["episode"]]))):
            if r is None:
                continue
            col, off, _ = outcome(r)
            d = {}
            for sec in (1, 3, 5):
                k = k_at(sec)
                if alive(r["ego"], k):
                    d[sec] = abs(float(proj(log_path(m), r["ego"][k: k + 1, :2])[1][0]))
            rows[name].append((col, off, d))
    res = {}
    for name, rr in rows.items():
        res[f"{name}_n"] = len(rr)
        res[f"{name}_col"] = float(np.mean([x[0] for x in rr]))
        res[f"{name}_off"] = float(np.mean([x[1] for x in rr]))
        for sec in (1, 3, 5):
            v = np.array([x[2][sec] for x in rr if sec in x[2]])
            res[f"{name}_d{sec}_median"] = float(np.median(v)) if len(v) else float("nan")
        v3 = [x[2].get(3, np.inf) for x in rr]  # dead by 3 s counts as not recovered
        res[f"{name}_rec3"] = float(np.mean(np.array(v3) < 0.5))
    ok = res["shift_rec3"] >= 0.70 and res["shift_off"] <= 0.05 and res["shift_col"] <= res["base_col"] + 0.05
    res["verdict"] = "recovers" if ok else "does not recover (by criterion)"
    return res


def validity(data, run, planner, traffic):
    b = load(run, "nopartner", planner, traffic)
    if b is None:
        return None
    o = np.array([outcome(r) for r in b.values()])
    return dict(n=len(o), goal=float(o[:, 2].mean()), collide=float(o[:, 0].mean()), offroad=float(o[:, 1].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--planners", default="ppo,cond_normal,cond_caut,cond_aggr,idm,pdm,cv")
    ap.add_argument("--traffic", default="expert,ppo")
    a = ap.parse_args()
    global DATA
    DATA = a.data
    out = {}
    for tr in a.traffic.split(","):
        for p in a.planners.split(","):
            key = f"{p}|{tr}"
            out[key] = dict(validity=validity(a.data, a.run, p, tr), A=exam_a(a.data, a.run, p, tr),
                            B=exam_b(a.data, a.run, p, tr), C=exam_c(a.data, a.run, p, tr))
            print(key, json.dumps(out[key], default=float, indent=None))
    json.dump(out, open(Path(a.run) / "readout.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
