#!/usr/bin/env python3
"""ROUTE-FOLLOW counterfactual (registered in research/next-round/route-follow.md section 2 before the run): does the served policy's plan
at the controller's horizon (1-2 s) move when the route it is given changes? The route reaches these models only as the 3-way command in
the adapter's ego row (columns 1..3 of lib/parity_adapter.ego_features: L, S, R; "unknown" = all zero), so the counterfactual edits those
three numbers on cached served tokens. No training, no simulator.

  cf    (envs/op-train, GPU)  groups held / zero (runs/road_probe/tok/<g>: the served frozen tokens + fed ego rows), every checkpoint in --tags,
                              5 command variants (served, L, S, R, none) -> $W/cf/<g>/<tag>.npz poses (5, N, 8, 3) rear axle 0.5..4 s, left +
  nav   (envs/op-train, CPU)  training side: NAVSIM driving_command of the navtrain shards and navtest against the logged 4 s future, and the
                              AlpaSim route command (ap2_route rebuild, m = 4 variant) on the same tokens -> $W/nav.json
  read  (envs/op-train, CPU)  buckets from runs/road_probe/labels/<g>.parquet, the registered quantities, log-cluster bootstrap -> $W/read/<ts>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/op_parity/scripts")]
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run  # noqa: E402

DATA = data_dir()
RP = DATA / "runs/road_probe"
W = DATA / "runs/route_follow"
TAGS = ("YS-L18-s1", "YS2G-L18-s1", "YR10m10-F-s0", "APY10m10-AB-s0", "P2H10-F-s0", "P0")
VARS = ("served", "L", "S", "R", "none")
CMD = {"L": (1, 0, 0), "S": (0, 1, 0), "R": (0, 0, 1), "none": (0, 0, 0)}
TI = {1.0: 1, 1.5: 2, 2.0: 3, 4.0: 7}                       # index of t (s) in the 8 output poses 0.5 .. 4 s


def logu(scene):
    return "_".join(str(scene).split("_")[:2])               # rp.logu / c0b_report.log_of


def variants(ego):
    V = np.repeat(ego[None].astype(np.float32), len(VARS), 0)
    for v, name in enumerate(VARS):
        if name in CMD:
            V[v][:, 1:4] = CMD[name]
    return V


# ---------------------------------------------------------------- cf (GPU)
def cf(a):
    import torch
    import pp_train as T
    from jevdrive import navsim_zs as Z
    from jevdrive import op_adapt as A
    from jevdrive import op_interp as I
    with Run("route_follow", "cf", config=vars(a)) as run:
        run.use_split(splits.load("alpasim_nuplan/onpol-held"))
        dev = torch.device("cuda")
        tc = torch.tensor([[1.0, 0.0]], device=dev)          # as rp.py tok (its G-tok reproduced the served plans with it)
        gates = {}
        for g in a.groups:
            src, out = RP / f"tok/{g}", W / "cf" / g
            out.mkdir(parents=True, exist_ok=True)
            front, tb = np.load(src / "front.npy", mmap_mode="r"), dict(np.load(src / "tab.npz"))
            N, V, cam = len(tb["ego"]), variants(tb["ego"]), tb["cam"].astype(np.float64)
            served_tag = tb["label"].astype(str) if "label" in tb else np.full(N, "YS-L18-s1")
            for tag in a.tags:
                f = out / f"{tag}.npz"
                if not f.exists() or a.force:
                    m = T.load_pmodel(tag, dev)
                    pi = A.plan_index(m.net.slices)
                    P = np.zeros((len(VARS), N, 8, 3), np.float32)
                    t_ = time.time()
                    with torch.no_grad():
                        for i in range(0, N, a.bs):
                            r = slice(i, min(i + a.bs, N))
                            H = torch.from_numpy(np.ascontiguousarray(front[r])).to(dev)
                            t = tc.repeat(len(H), 1)
                            for v in range(len(VARS)):
                                mu = m(H, torch.from_numpy(V[v, r]).to(dev), t).float()[:, pi].view(-1, 33, 15).cpu().numpy()
                                P[v, r] = np.stack([I.to_rear(q[:, 0:3], q[:, 11], I.T_IDXS, c[:2], Z.T_OUT, "lever") for q, c in zip(mu, cam[r])])
                    np.savez(f, poses=P)
                    run.info(f"{g} {tag}: {N} rows x {len(VARS)} variants in {time.time() - t_:.0f} s")
                    del m
                    torch.cuda.empty_cache()
                P = np.load(f)["poses"]
                own = served_tag == tag
                if own.any():                                  # G-rep: the served variant of the served checkpoint reproduces the stored plan
                    d = np.abs(P[0, own, :, :2] - tb["poses"][own, :, :2]).max((1, 2))
                    gates[f"G_rep/{g}/{tag}"] = dict(n=int(own.sum()), share=float((d <= 0.10).mean()), median=float(np.median(d)), max=float(d.max()),
                                                     pass_=bool((d <= 0.10).mean() >= 0.95))
                if tag == "P0":                                # G-null: no adapter, the command cannot act
                    d = np.abs(P[1:, :, :, :2] - P[0][None, :, :, :2]).max()
                    gates[f"G_null/{g}"] = dict(max=float(d), pass_=bool(d < 1e-3))
        (W / "cf" / "gates.json").write_text(json.dumps(gates, indent=1))
        run.summary.update(gates=gates)
        run.info(json.dumps(gates))


# ---------------------------------------------------------------- nav (training side, CPU)
def route_cmd(wp):
    """ap2 route waypoints (N, 20, 2) rig frame, NaN padding -> command index 0 L / 1 S / 2 R / 3 unknown (navigation.command_from_route)."""
    out = np.full(len(wp), 3, int)
    for i, w in enumerate(wp):
        ok = np.isfinite(w).all(1)
        if not ok.any():
            continue
        d = np.hypot(*w[ok].T)
        j = np.flatnonzero(d >= 5.0)
        if not len(j):
            out[i] = 1
            continue
        y = w[ok][j[0], 1]
        out[i] = 0 if y > 2.0 else 2 if y < -2.0 else 1
    return out


def fut_bins(fut):
    """Logged 4 s future (N, 8, 3) x, y (left +), yaw -> bin names."""
    y4, h4 = fut[:, 7, 1], np.degrees(fut[:, 7, 2])
    b = np.full(len(fut), "other", object)
    b[(np.abs(h4) < 5) & (np.abs(y4) < 1.0)] = "straight"
    b[(np.abs(h4) < 10) & (y4 >= 2.0)] = "shift_L"
    b[(np.abs(h4) < 10) & (y4 <= -2.0)] = "shift_R"
    b[(h4 >= 20) & (h4 < 45)] = "turn_L20"
    b[(h4 <= -20) & (h4 > -45)] = "turn_R20"
    b[h4 >= 45] = "turn_L45"
    b[h4 <= -45] = "turn_R45"
    return b


def nav(a):
    with Run("route_follow", "nav", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        res = {}
        for name, shards in (("navtrain", [f"navtrain_full.s{i}of12" for i in range(12)]), ("navtest", ["lb_navtest"])):
            nc, rc, B, Y4, H4 = [], [], [], [], []
            for s in shards:
                tf = DATA / "runs/op_parity/cache" / s / "tab.npz"
                rf = DATA / "runs/alpasim/ap2/route" / f"{s}.npz"
                if not tf.exists():
                    run.info(f"missing {tf}")
                    continue
                tb = dict(np.load(tf))
                ok = np.isfinite(tb["fut"][:, 7, 0])
                c_nav = tb["cmd"][:, -1].argmax(1)
                c_rt = np.full(len(c_nav), -1)
                if rf.exists():
                    z = dict(np.load(rf))
                    pos = {n: i for i, n in enumerate(z["names"].tolist())}
                    idx = np.array([pos.get(n, -1) for n in tb["names"].tolist()])
                    have = (idx >= 0)
                    c_rt[have] = route_cmd(z["wp"][idx[have], 3])
                    c_rt[have & ~z["ok"][np.maximum(idx, 0), 3]] = -1
                nc.append(c_nav[ok]); rc.append(c_rt[ok]); B.append(fut_bins(tb["fut"][ok]))
                Y4.append(tb["fut"][ok, 7, 1]); H4.append(np.degrees(tb["fut"][ok, 7, 2]))
            if not nc:
                continue
            nc, rc, B = np.concatenate(nc), np.concatenate(rc), np.concatenate(B)
            r = dict(n=int(len(nc)), route_rebuilt=int((rc >= 0).sum()))
            for b in ("straight", "shift_L", "shift_R", "turn_L20", "turn_R20", "turn_L45", "turn_R45", "other"):
                m = B == b
                r[b] = dict(n=int(m.sum()),
                            nav=[float((nc[m] == k).mean()) if m.any() else np.nan for k in range(4)],
                            route=[float((rc[m & (rc >= 0)] == k).mean()) if (m & (rc >= 0)).any() else np.nan for k in range(4)])
            ok2 = rc >= 0
            r["agree_nav_route"] = float((nc[ok2] == rc[ok2]).mean()) if ok2.any() else np.nan
            for lab, c in (("nav", nc), ("route", rc)):
                turn = c[(c == 0) | (c == 2)] if lab == "nav" else c[ok2 & ((c == 0) | (c == 2))]
                Bt = B[(c == 0) | (c == 2)] if lab == "nav" else B[ok2 & ((c == 0) | (c == 2))]
                vis = np.isin(Bt, ["shift_L", "shift_R", "turn_L20", "turn_R20", "turn_L45", "turn_R45"])
                r[f"{lab}_LR_share"] = float(len(turn) / max(1, len(c if lab == 'nav' else c[ok2])))
                r[f"{lab}_LR_visible_in_4s"] = float(vis.mean()) if len(vis) else np.nan     # L / R rows whose 4 s target already shows the manoeuvre
                r[f"{lab}_LR_on_straight_4s"] = float((Bt == "straight").mean()) if len(Bt) else np.nan
            res[name] = r
            run.info(f"{name}: {json.dumps(r)[:400]}")
        W.mkdir(parents=True, exist_ok=True)
        (W / "nav.json").write_text(json.dumps(res, indent=1))
        run.summary.update(nav=res)


# ---------------------------------------------------------------- read (CPU)
def boot_med(e, cl, B=2000, seed=0):
    m = np.isfinite(e)
    e, cl = e[m], cl[m]
    if not len(e):
        return dict(n=0, v=np.nan, lo=np.nan, hi=np.nan, clusters=0)
    u, inv = np.unique(cl, return_inverse=True)
    groups = [e[inv == k] for k in range(len(u))]
    rng = np.random.default_rng(seed)
    bs = [np.median(np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])) for _ in range(B)]
    return dict(n=int(len(e)), clusters=int(len(u)), v=float(np.median(e)), mean=float(e.mean()), lo=float(np.quantile(bs, 0.025)),
                hi=float(np.quantile(bs, 0.975)))


def y_at_x(P, xs):
    """Plan poses (N, 8, 3) with the origin prepended -> lateral where the path first crosses x = xs (N,), NaN if it never does."""
    Q = np.concatenate([np.zeros((len(P), 1, 2)), P[:, :, :2]], 1)
    out = np.full(len(P), np.nan)
    for i, (q, x) in enumerate(zip(Q, xs)):
        j = np.flatnonzero((q[:-1, 0] < x) & (q[1:, 0] >= x))
        if len(j):
            k = j[0]
            t = (x - q[k, 0]) / max(q[k + 1, 0] - q[k, 0], 1e-9)
            out[i] = q[k, 1] + t * (q[k + 1, 1] - q[k, 1])
    return out


def read(a):
    import pandas as pd
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = W / "read" / ts
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "read", config=vars(a)) as run:
        run.use_split(splits.load("alpasim_nuplan/onpol-held"))
        R = {"gates": json.loads((W / "cf/gates.json").read_text())}
        for g in a.groups:
            tb = dict(np.load(RP / f"tok/{g}/tab.npz"))
            lab = pd.read_parquet(RP / f"labels/{g}.parquet").sort_values("row").reset_index(drop=True)
            assert len(lab) == len(tb["ego"]) and (lab["k"].to_numpy() == tb["k"]).all()
            N = len(lab)
            served_tag = tb["label"].astype(str) if "label" in tb else np.full(N, "YS-L18-s1")
            cl = np.array([logu(s) for s in tb["scene"]])
            e = tb["ego"]
            scmd = np.where(e[:, 1:4].max(1) > 0.5, e[:, 1:4].argmax(1), 3)            # served command 0 L 1 S 2 R 3 none
            t40, tsg, jd = lab["turn40"].to_numpy(), lab["turn_sign"].to_numpy(), lab["jdist"].to_numpy()
            yl = {d: lab[f"ylog{d}"].to_numpy() for d in (10, 20, 40)}
            jt = (t40 >= 45) & np.isfinite(jd) & (jd <= 40)
            lc = (t40 < 15) & (np.abs(yl[40]) >= 2.5) & np.isfinite(yl[40])
            st = (t40 < 10) & (np.abs(yl[40]) < 1.0)
            bend = (t40 >= 20) & ~np.isfinite(jd)
            sgn = np.where(jt | bend, tsg, np.where(lc, np.sign(yl[40]), 1.0))
            toward = np.where(sgn > 0, 0, 2)                                            # command index on the route's side
            buckets = {"JT": jt, "JT_far(20-40m)": jt & (jd > 20), "JT_near(2-20m)": jt & (jd > 0) & (jd <= 20), "JT_in(0m)": jt & (jd == 0),
                       "JT_cmd_on": jt & (scmd == toward), "JT_cmd_S": jt & (scmd == 1), "LC": lc, "LC_cmd_on": lc & (scmd == toward),
                       "BEND": bend, "ST": st, "ST_cmd_S": st & (scmd == 1), "ST_cmd_LR": st & ((scmd == 0) | (scmd == 2))}
            Rg = {"n": N, "buckets": {k: int(v.sum()) for k, v in buckets.items()}}
            for tag in a.tags:
                f = W / "cf" / g / f"{tag}.npz"
                if not f.exists():
                    continue
                P = np.load(f)["poses"].astype(np.float64)                             # (5, N, 8, 3)
                own = served_tag == tag
                if g == "zero" and not own.any():
                    continue
                vi = {v: i for i, v in enumerate(VARS)}
                ytow = np.where(toward == 0, P[vi["L"], :, :, 1].T, P[vi["R"], :, :, 1].T).T  # (N, 8)
                yaway = np.where(toward == 0, P[vi["R"], :, :, 1].T, P[vi["L"], :, :, 1].T).T
                yS, ysv, ynone = P[vi["S"], :, :, 1], P[vi["served"], :, :, 1], P[vi["none"], :, :, 1]
                q = {}
                for t, k in TI.items():
                    q[f"swing@{t}s"] = sgn * (ytow[:, k] - yaway[:, k])                  # toward-route command minus away command
                    q[f"towardS@{t}s"] = sgn * (ytow[:, k] - yS[:, k])                   # toward-route command minus straight
                    q[f"servedS@{t}s"] = sgn * (ysv[:, k] - yS[:, k])                    # served minus straight
                    q[f"servedNone@{t}s"] = sgn * (ysv[:, k] - ynone[:, k])
                    q[f"absLR@{t}s"] = np.abs(P[vi["L"], :, k, 1] - P[vi["R"], :, k, 1])
                cam = tb["cam"][:, 0].astype(np.float64)
                for d in (10, 20):
                    xs = cam + d
                    yd = {v: y_at_x(P[vi[v]], xs) for v in ("L", "S", "R", "served")}
                    ytd = np.where(toward == 0, yd["L"], yd["R"])
                    yad = np.where(toward == 0, yd["R"], yd["L"])
                    q[f"need@{d}m"] = sgn * yl[d]                                         # the route's (logged path's) own lateral there
                    q[f"swing@{d}m"] = sgn * (ytd - yad)
                    q[f"towardS@{d}m"] = sgn * (ytd - yd["S"])
                    q[f"planS@{d}m"] = sgn * yd["S"]                                       # what the plan does with the command straight
                    q[f"planServed@{d}m"] = sgn * yd["served"]
                    q[f"errServed@{d}m"] = sgn * (yd["served"] - yl[d])                    # + = beyond the route's side, - = short of it
                rows = own if g == "zero" else np.ones(N, bool)
                Rt = {}
                for bn, bm in buckets.items():
                    m = bm & rows
                    Rt[bn] = {qn: boot_med(qv[m], cl[m], B=a.B) for qn, qv in q.items()}
                Rg[tag] = Rt
                run.info(f"{g} {tag} JT swing@2s {Rt['JT']['swing@2.0s']} ST absLR@2s {Rt['ST']['absLR@2.0s']}")
            R[g] = Rg
        (out / "rf_read.json").write_text(json.dumps(R, indent=1, default=float))
        run.summary.update(read_dir=str(out))
        run.info(f"read -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cf")
    p.add_argument("--groups", nargs="+", default=["held", "zero"])
    p.add_argument("--tags", nargs="+", default=list(TAGS))
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--force", action="store_true")
    sub.add_parser("nav")
    p = sub.add_parser("read")
    p.add_argument("--groups", nargs="+", default=["held", "zero"])
    p.add_argument("--tags", nargs="+", default=list(TAGS))
    p.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    dict(cf=cf, nav=nav, read=read)[a.cmd](a)


if __name__ == "__main__":
    main()
