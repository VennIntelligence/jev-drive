#!/usr/bin/env python3
"""ROUND9 HINGE-AUDIT (plans/2026-10-22-hinge-audit.md): are the S consequence hinges satisfied by the human's own log future and by the
teacher labels; root causes of every human violation; expert-calibrated hinge candidates. CPU only, stored files + a CPU forward of
R6DAG-L18-s1. Chain: hinge_audit.sh. Box outputs: $DATA_DIR/runs/round9/hinge_audit/ -> experiments/round9/results/hinge_audit/.

  fwd    (op-train)  --parts S|ph|nav --shard i/n: the trainer's build with R6DAG's argv (rtc.py), eval-mode plans of R6DAG-L18-s1 on a
                     seeded --frac of the chosen pool rows (S = nuPlan S rows rb0 / rtb1 / y / ys, ph = PAI human rows; nav = navtrain
                     imitation rows) + the trainer's own terms on them (cross-check) -> $O/fwd_<part>_<shard>.npz
  steps  (op-train)  per pool (S | PAID | PAIH) and path (log / fut / plan): per-step raw clearances of every term (agent corridor / side
                     distance, road corner SDF (+ drivable SDF on nuPlan), onc / soft region SDF, side distance) plus the root-cause
                     counterfactuals of the log path -> $O/steps_<pool>_<path>.npz
  report (op-train)  every table of the result page from the steps files -> $O/audit.json, results dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/round9/scripts"), str(REPO / "experiments/defect_fix/lib"),
                str(REPO / "experiments/body1/lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
O = DATA / "runs/round9/hinge_audit"
CR = DATA / "runs/op_parity/cache"
S_PARTS = ("rb0", "rtb1", "y", "ys")
CKPT = "R6DAG-L18-s1"


# ---------------------------------------------------------------- fwd (op-train, CPU)
def cmd_fwd(a):
    import torch
    import rtc
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    rt = REPO / "experiments/route_mem/scripts/rm_train.py"
    src = rt.read_text()
    tail = "dict(train=cmd_train, rely=cmd_rely, mixcheck=cmd_mixcheck)[a.cmd](a)"
    assert src.count(tail) == 1
    g = {"__name__": "__main__", "__file__": str(rt)}
    sys.argv = [str(rt), "rely", *rtc.R6DAG_ARGV.format(tag=CKPT, data=DATA).split()]
    exec(compile(src.replace(tail, "_HA_ARGS = a"), str(rt), "exec"), g)
    ra = g["_HA_ARGS"]
    ra.rm_from = CKPT
    dev = torch.device("cpu")
    ns, *_ = g["build"](ra, dev)
    LS = ns.losses(ns.mode)
    import pp_train as T
    S, model = ns.S, ns.model.eval()
    names = S.tab["names"].astype(str)
    part = np.array([x.split(":")[1] if x.startswith("on:") else "" for x in names])
    i, n = (int(x) for x in a.shard.split("/"))
    rng = np.random.default_rng(a.seed)
    if a.parts == "nav":
        hz = LS.hinge
        cand = np.flatnonzero((np.arange(len(names)) < ns.on0) & S.has_fut.cpu().numpy() & hz.ok.cpu().numpy())
    else:
        want = S_PARTS if a.parts == "S" else ("ph",)
        cand = np.flatnonzero(np.isin(part, want) & (np.arange(len(names)) >= ns.on0))
    sel = np.sort(rng.choice(cand, int(round(a.frac * len(cand))), replace=False))
    sel = sel[i::n]
    print(f"{a.parts} shard {a.shard}: {len(sel)} of {len(cand)} rows", flush=True)
    pi_ = torch.as_tensor(S.pi, device=dev)
    P, TR = [], {k: [] for k in ("agent", "road", "lat_onc", "lat_soft", "lat_side", "nav", "nav_fut")}
    t0 = time.time()
    for j in range(0, len(sel), a.bs):
        r = torch.as_tensor(sel[j:j + a.bs], device=dev)
        with torch.no_grad():
            o = model(S.front[r.cpu().numpy()], S.ego[r], S.tc[r], None, None, nv=None if ns.nv is None else ns.nv[r]).float()
            x, y, psi = T.rear(o[:, pi_].view(-1, 33, 15), S.cam_x[r], LS.W)
            if a.parts == "nav":
                TR["nav"].append(torch.relu(LS.hinge.margin - LS.hinge.margins(x, y, psi, r)).mean(1).numpy())
                F = S.fut[r]
                TR["nav_fut"].append(torch.relu(LS.hinge.margin - LS.hinge.margins(F[..., 0], F[..., 1], F[..., 2], r)).mean(1).numpy())
            else:
                TR["agent"].append((LS.agent2.per_step(x, y, psi, r).mean(1) * LS.on_aok[r]).numpy())
                if LS.road is not None:
                    TR["road"].append((torch.relu(LS.road.margin - LS.road.margins(x, y, psi, r)).mean(1) * LS.road.ok[r]).numpy())
                lt = LS.lon.loss(torch.stack([x, y, psi], -1), r - LS.on0, LS.pack)
                for k in ("lat_onc", "lat_soft", "lat_side"):
                    if k in lt:
                        TR[k].append(lt[k].numpy())
        P.append(torch.stack([x, y, psi], -1).numpy())
        if j == 0:
            print(f"first batch {time.time() - t0:.1f} s", flush=True)
    O.mkdir(parents=True, exist_ok=True)
    extra = {}
    if a.parts == "nav":                                                   # the navtrain human label on EVERY labelled row (no forward needed)
        allc = cand
        hv = []
        with torch.no_grad():
            for j in range(0, len(allc), 4096):
                r = torch.as_tensor(allc[j:j + 4096])
                F = S.fut[r]
                m = LS.hinge.margins(F[..., 0], F[..., 1], F[..., 2], r)
                hv.append(torch.stack([torch.relu(LS.hinge.margin - m).mean(1), m.amin(1)], 1).numpy())
        hv = np.concatenate(hv)
        extra = dict(all_rows=allc, all_fut_hinge=hv[:, 0], all_fut_minsdf=hv[:, 1], speed_all=S.tab["speed"][allc] if "speed" in S.tab else np.zeros(0),
                     fut_all=S.fut[torch.as_tensor(allc)].numpy(), margin=LS.hinge.margin)
    np.savez(O / f"fwd_{a.parts}_{i}.npz", rows=sel, names=names[sel], poses=np.concatenate(P), on0=ns.on0,
             **{f"tr_{k}": np.concatenate(v) for k, v in TR.items() if v}, **extra)
    print(json.dumps(dict(parts=a.parts, shard=a.shard, rows=len(sel), sec=round(time.time() - t0, 1))), flush=True)


# ---------------------------------------------------------------- geometry (torch; the trainer's terms, generalised for the counterfactuals)
FRONT, REAR, HALF_W, LAT_MAX, BC = 4.049, -1.127, 1.1485, 3.0, (4.049 - 1.127) / 2
X0, Y0 = -8.0, -24.0
MARGIN, BUF = 0.3, 0.4
W = dict(agent=30.0, road=30.0, onc=15.0, soft=3.0, side=15.0)        # R6DAG: ho_w 3 x (agent / road lam 10; lat 10 x 0.5 / 0.1 / 0.5)
BIG = 1e3


def interp_at(times):
    """(len(times), 9): linear interpolation of the 9 knots (0, 0.5 .. 4 s) at `times` (clamped to [0, 4])."""
    t = np.r_[0, np.arange(1, 9) * 0.5]
    tt = np.clip(np.asarray(times, float), 0, 4)
    M = np.zeros((len(tt), 9))
    for k, x in enumerate(tt):
        j = min(np.searchsorted(t, x, side="right") - 1, 7)
        f = (x - t[j]) / 0.5
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


T41 = np.arange(41) * 0.1


def densify(P9):
    """(n, 9, 3) knots (start pose + 8 poses) -> (n, 41, 3) torch, yaw unwrapped along the knots first."""
    import torch
    P = np.array(P9, np.float64)
    P[..., 2] = np.unwrap(P[..., 2], axis=1)
    return torch.einsum("kj,bjc->bkc", torch.as_tensor(interp_at(T41), dtype=torch.float32), torch.as_tensor(P, dtype=torch.float32))


def boxes_at(box, val, times):
    """Boxes (B, 9, K, 5), valid (B, 9, K) -> interpolated at `times` (B, T, K, 5), valid (B, T, K) (both neighbours valid)."""
    import torch
    M = torch.as_tensor(interp_at(times), dtype=torch.float32)
    bx = torch.einsum("kj,bjoc->bkoc", M, box)
    bad = torch.einsum("kj,bjo->bko", (M > 0).float(), (~val).float())
    return bx, bad < 0.5


def box_terms(D, bx, valid, ext=MARGIN, eg_shrink=0.0, ob_shrink=0.0):
    """Dense ego poses D (B, T, 3), boxes (B, T, K, 5) -> dict of (B, T): ag_c / ag_s = min distance of the front-extended ego box to the
    counted objects in / beside the corridor (AgentHinge.distances), agp_c = the same with the plain box, sd = nr1_lat.side_dist (plain box,
    side objects not behind). BIG where none is counted."""
    import torch
    from agent_hinge import _box_sdf, _corners
    ex, ey, eps = D[..., 0:1], D[..., 1:2], D[..., 2:3]
    c, s = torch.cos(eps), torch.sin(eps)
    ox, oy, oyaw = bx[..., 0], bx[..., 1], bx[..., 2]
    ohl, ohw = (bx[..., 3] / 2 - ob_shrink).clamp_min(0.05), (bx[..., 4] / 2 - ob_shrink).clamp_min(0.05)
    px, py = _corners(ox, oy, oyaw, ohl, ohw)

    def dist(e):
        f, r, hw = FRONT + e - eg_shrink, REAR + eg_shrink, HALF_W - eg_shrink
        off, hl = (f + r) / 2, (f - r) / 2
        ecx, ecy = ex + c * off, ey + s * off
        hl_e, hw_e = torch.full_like(ecx, hl), torch.full_like(ecx, hw)
        qx, qy = _corners(ecx, ecy, eps, hl_e, hw_e)
        d1 = _box_sdf(qx, qy, ox[..., None], oy[..., None], oyaw[..., None], ohl[..., None], ohw[..., None]).amin(-1)
        d2 = _box_sdf(px, py, ecx[..., None], ecy[..., None], eps[..., None], hl_e[..., None], hw_e[..., None]).amin(-1)
        return torch.minimum(d1, d2), ecx, ecy, hl
    de, _, _, _ = dist(ext)
    dp, pcx, pcy, phl = dist(0.0)
    c_off = (FRONT + REAR) / 2
    rx, ry = ox - (ex + c * c_off), oy - (ey + s * c_off)
    lon, lat = c * rx + s * ry, -s * rx + c * ry
    counted = valid & (lon > 0) & ((lat.abs() < LAT_MAX) | (de < 0))
    lc = -s[..., None] * (px - ex[..., None]) + c[..., None] * (py - ey[..., None])
    corridor = (lc.amax(-1) > -HALF_W) & (lc.amin(-1) < HALF_W)
    lon_s = c * (ox - pcx) + s * (oy - pcy)
    counted_s = valid & ~corridor & (lon_s > -(phl + ohl))
    f = lambda d, m: d.masked_fill(~m, BIG).amin(-1)                                          # noqa: E731
    return dict(ag_c=f(de, counted & corridor), ag_s=f(de, counted & ~corridor), agp_c=f(dp, counted & corridor),
                agp_s=f(dp, counted & ~corridor), sd=f(dp, counted_s))


def sample(sdf, xy):
    """sdf (B, 1, 128, 96), points (B, N, 2) -> (B, N) bilinear, border clamp (drivable_hinge's convention)."""
    import torch
    g = torch.stack([(xy[..., 1] - Y0) / 24.0 - 1.0, (xy[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
    return torch.nn.functional.grid_sample(sdf.float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]


def foot(D):
    """Dense poses (B, T, 3) -> footprint corners (B, T, 4, 2) and rear axle + box centre (B, T, 2, 2)."""
    import torch
    from drivable_hinge import CORNERS
    C = torch.as_tensor(CORNERS, dtype=torch.float32)
    c, s = torch.cos(D[..., 2:3]), torch.sin(D[..., 2:3])
    cx = D[..., :1] + c * C[:, 0] - s * C[:, 1]
    cy = D[..., 1:2] + s * C[:, 0] + c * C[:, 1]
    pts = torch.stack([D[..., :2], torch.stack([D[..., 0] + BC * c[..., 0], D[..., 1] + BC * s[..., 0]], -1)], 2)
    return torch.stack([cx, cy], -1), pts


# ---------------------------------------------------------------- pools: rows, labels, paths
def pool_spec(name):
    """-> dict: names (pool-local rows), speed, label files, paths {log, fut, plan} as (n, 9, 3) knots with NaN rows where missing, extras."""
    import pandas as pd
    if name == "S":
        P = CR / "on1_JTX"
        tab = np.load(P / "tab.npz")
        pool, mode = tab["pool"].astype(str), tab["mode"].astype(str)
        sel = np.flatnonzero(np.isin(pool, S_PARTS) & (mode == "self"))
        names = tab["names"][sel].astype(str)
        n = len(sel)
        log = np.full((n, 9, 3), np.nan, np.float32)
        e0, k = np.zeros(n), np.zeros(n, int)
        RD = dict(rb0="L18RB0", rtb1="L18RTB1", ys="L18YS", y="L18Y")
        for p, rd in RD.items():
            m = np.load(DATA / "runs/onpol_nuplan/rows" / rd / "meta.npz", allow_pickle=True)
            pos = {x: i for i, x in enumerate(m["names"].tolist())}
            idx = np.flatnonzero(pool[sel] == p)
            j = np.array([pos[x.replace(f"on:{p}:", "on:", 1)] for x in names[idx].tolist()])
            log[idx], e0[idx], k[idx] = m["logfut"][j], m["e0"][j], m["k"][j]
        fut = np.concatenate([np.zeros((n, 1, 3), np.float32), tab["fut"][sel].astype(np.float32)], 1)
        plan = np.full((n, 9, 3), np.nan, np.float32)
        f = O / "fwd_S_0.npz"
        if f.exists():
            z = np.load(f)
            pos = {x: i for i, x in enumerate(names.tolist())}
            j = np.array([pos[x] for x in z["names"].astype(str).tolist()])
            plan[j, 1:] = z["poses"]
            plan[j, 0] = 0
        lat = DATA / "runs/defect_fix/lat/L18NR1v2"
        lm = np.load(lat / "meta.npz")
        nr = len(lm["names"])
        assert np.array_equal(lm["names"].astype(str), tab["names"][:nr].astype(str))
        junc = np.full(n, np.nan)
        jj = sel < nr
        junc[jj] = lm["junction_share"][sel[jj]]
        return dict(name=name, dir=P, names=names, jtx=sel, speed=tab["speed"][sel].astype(np.float64), road=P / "road.npz", drv=P / "drivable.npz",
                    agent=P / "agent.npz", lat=lat, lat_rows=np.where(jj, sel, -1), paths=dict(log=log, fut=fut, plan=plan), junc=junc, e0=e0, k=k,
                    part=pool[sel], draw=np.ones(n, bool))
    if name == "PAID":
        P = CR / "on1_PAID"
        tab = np.load(P / "tab.npz")
        names = tab["names"].astype(str)
        n = len(names)
        M = np.load(DATA / "runs/round9/pai_s_gate/meta.npz")
        assert np.array_equal(M["names"].astype(str), names)
        hum_ok = np.isfinite(M["log_fut"]).all((1, 2)) & M["log_fut_ok"].all(1) & np.isfinite(M["log0"]).all(1)
        log = np.full((n, 9, 3), np.nan, np.float32)
        log[hum_ok, 0], log[hum_ok, 1:] = M["log0"][hum_ok], M["log_fut"][hum_ok]
        fut = np.concatenate([np.zeros((n, 1, 3), np.float32), tab["fut"].astype(np.float32)], 1)
        plan = np.full((n, 9, 3), np.nan, np.float32)
        z = np.load(DATA / "runs/round9/pai_s_gate" / f"fwd_{CKPT}.npz")
        plan[z["idx"], 1:], plan[z["idx"], 0] = z["poses"], 0
        dg = np.load(P / "dagger.npz")
        return dict(name=name, dir=P, names=names, speed=tab["speed"].astype(np.float64), road=DATA / "runs/round9/pai_s_gate/road/road_on1_PAID.npz",
                    drv=None, agent=P / "agent.npz", lat=None, paths=dict(log=log, fut=fut, plan=plan), draw=dg["draw"].astype(bool),
                    ev_off=M["ev_off"], ev_col=M["ev_col"], t0=M["t0"], scene=M["scene"])
    if name == "PAIH":
        P = CR / "on1_PAIH"
        tab = np.load(P / "tab.npz")
        names = tab["names"].astype(str)
        n = len(names)
        fut = np.concatenate([np.zeros((n, 1, 3), np.float32), tab["fut"].astype(np.float32)], 1)
        plan = np.full((n, 9, 3), np.nan, np.float32)
        f = O / "fwd_ph_0.npz"
        if f.exists():
            z = np.load(f)
            key = {":".join(x.split(":")[-2:]): i for i, x in enumerate(names.tolist())}
            j = np.array([key[":".join(x.split(":")[-2:])] for x in z["names"].astype(str).tolist()])
            plan[j, 1:], plan[j, 0] = z["poses"], 0
        return dict(name=name, dir=P, names=names, speed=tab["speed"].astype(np.float64), road=DATA / "runs/round5/road_pai/road_on1_PAIH.npz",
                    drv=None, agent=P / "agent.npz", lat=None, paths=dict(log=fut, plan=plan), draw=np.ones(n, bool))
    raise ValueError(name)


def cmd_steps(a):
    """Per-step raw clearances of every term on every path of one pool (module docstring) -> $O/steps_<pool>.npz."""
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from jevdrive.common import n_cpus
    import nr1_lat
    torch.set_num_threads(a.threads or n_cpus())
    sp = pool_spec(a.pool)
    names = sp["names"]
    n = len(names) if not a.limit else a.limit
    names = names[:n]
    t_ = time.time()
    ah = AgentHinge(sp["agent"], names, "cpu", MARGIN, 0.0)
    road = Hinge(sp["road"], names, "cpu", MARGIN, bank=True)
    drv = Hinge(sp["drv"], names, "cpu", MARGIN, bank=True) if sp["drv"] else None
    print(f"labels loaded {time.time() - t_:.0f} s: agent ok {float(ah.ok.float().mean()):.4f}, road ok {float(road.ok.float().mean()):.4f}", flush=True)
    # the trainer's road raster per row: rows whose t0 footprint is off the road raster take drivable.npz (on_train note (x))
    rows_all = torch.arange(n)
    z = torch.zeros(n, 8)
    m0 = torch.cat([road.margins(z[i:i + 8192], z[i:i + 8192], z[i:i + 8192], rows_all[i:i + 8192]).amin(1) for i in range(0, n, 8192)])
    off_t0 = (m0 < 0) & road.ok
    if drv is not None:
        fb = torch.nonzero(off_t0)[:, 0]
        trn = Hinge(sp["road"], names, "cpu", MARGIN, bank=True)
        if len(fb):
            trn.sdf[fb], trn.ok[fb] = drv.sdf[fb], drv.ok[fb]
    else:
        trn = road
    pack = None
    if sp["lat"] is not None:
        lm = np.load(sp["lat"] / "meta.npz")
        pack = dict(bins=torch.as_tensor(lm["bins"].astype(np.float32)), points=str(lm["points"]) if "points" in lm.files else "both",
                    onc=np.load(sp["lat"] / "onc.npy", mmap_mode="r"), soft=np.load(sp["lat"] / "soft.npy", mmap_mode="r"))
        lat_rows = sp["lat_rows"][:n]
        lat_ok = lat_rows >= 0
        lat_ok[lat_ok] &= lm["ok"][lat_rows[lat_ok]].astype(bool)
    out = dict(names=names, speed=sp["speed"][:n], agent_ok=ah.ok.numpy(), road_ok=trn.ok.numpy(), off_t0=off_t0.numpy(), m0=m0.numpy())
    if pack is not None:
        out["lat_ok"] = lat_ok
    # scene context from the boxes at t0 (pool-local): lead within 25 m in the lane, parked object beside, moving share
    B, V = ah.box, ah.val
    b0, v0 = B[:, 0], V[:, 0]
    mv = torch.where(V[:, 0] & V[:, 8], torch.hypot(B[:, 8, :, 0] - B[:, 0, :, 0], B[:, 8, :, 1] - B[:, 0, :, 1]), torch.full_like(b0[..., 0], np.nan))
    lead = v0 & (b0[..., 0] > 0) & (b0[..., 0] < 25) & (b0[..., 1].abs() < 2.0)
    parked = v0 & (b0[..., 0] > -2) & (b0[..., 0] < 25) & (b0[..., 1].abs() >= 1.5) & (b0[..., 1].abs() < 4.0) & (mv < 1.0)
    lead_d = torch.where(lead, b0[..., 0], torch.full_like(b0[..., 0], BIG)).amin(1)
    out |= dict(lead_d=lead_d.numpy(), lead_moving=(lead & (mv >= 1.0)).any(1).numpy(), parked=parked.any(1).numpy(),
                box_t0=v0.sum(1).numpy(), box_t8=V[:, 8].sum(1).numpy(), moved_share=np.nanmean(np.where(V[:, 0].numpy(), (mv >= 1.0).numpy(), np.nan), 1))
    for k_ in ("junc", "e0", "k", "part", "draw", "ev_off", "ev_col", "t0", "scene", "jtx"):
        if k_ in sp:
            out[k_] = np.asarray(sp[k_])[:n]
    bs = 1024
    for path, P9 in sp["paths"].items():
        P9 = P9[:n]
        okp = np.isfinite(P9).all((1, 2))
        res = {q: np.full((n, 41) if q not in ("road", "drv", "pts") else (n, 41, 4 if q != "pts" else 2), BIG, np.float32)
               for q in ("ag_c", "ag_s", "agp_c", "agp_s", "sd", "road", "drv", "pts")}
        res["oow"] = np.zeros((n, 41), bool)
        if pack is not None:
            nw = 40 if pack["points"] == "ra" else 80
            res["onc"], res["soft"] = (np.full((n, nw), BIG, np.float32) for _ in range(2))
        idx = np.flatnonzero(okp)
        t1 = time.time()
        for i in range(0, len(idx), bs):
            r = idx[i:i + bs]
            rt = torch.as_tensor(r)
            with torch.no_grad():
                D = densify(P9[r])
                bx, vv = boxes_at(B[rt], V[rt], T41)
                for q, v in box_terms(D, bx, vv).items():
                    res[q][r] = v.numpy()
                C, pts = foot(D)
                res["road"][r] = sample(trn.sdf[rt], C.reshape(len(r), -1, 2)).view(len(r), 41, 4).numpy()
                res["pts"][r] = sample(trn.sdf[rt], pts.reshape(len(r), -1, 2)).view(len(r), 41, 2).numpy()
                if drv is not None:
                    res["drv"][r] = sample(drv.sdf[rt], C.reshape(len(r), -1, 2)).view(len(r), 41, 4).numpy()
                res["oow"][r] = ((C[..., 0] < X0) | (C[..., 0] > X0 + 64) | (C[..., 1].abs() > 24)).any(-1).numpy()
                if pack is not None:
                    lr = lat_rows[r]
                    lo = lr >= 0
                    if lo.any():
                        tl = torch.as_tensor(lr[lo])
                        for key in ("onc", "soft"):
                            res[key][r[lo]] = nr1_lat.region_m(D[torch.as_tensor(lo)], tl, pack, key).numpy()
            if i == 0:
                print(f"{a.pool} {path}: first batch {time.time() - t1:.1f} s ({len(idx)} rows)", flush=True)
        # counterfactuals of the agent / side geometry, only on rows where this path violates agent or side (log and fut paths)
        if path in ("log", "fut"):
            ag = np.maximum(np.maximum(MARGIN - res["ag_c"], 0), np.maximum(-res["ag_s"], 0)).mean(1) * out["agent_ok"]
            sdv = np.maximum(BUF - res["sd"][:, 1:], 0).mean(1) * out["agent_ok"]
            cf = np.flatnonzero(okp & ((ag > 0) | (sdv > 0)))
            VAR = dict(shift_m05=dict(dt=-0.5), shift_m025=dict(dt=-0.25), shift_p025=dict(dt=0.25), shift_p05=dict(dt=0.5), static=dict(static=True),
                       obj_shrink=dict(ob=0.25), ego_shrink=dict(eg=0.25))
            for vn, kw in VAR.items():
                cag, cas, csd = (np.full((len(cf), 41), BIG, np.float32) for _ in range(3))
                for i in range(0, len(cf), bs):
                    r = cf[i:i + bs]
                    rt = torch.as_tensor(r)
                    with torch.no_grad():
                        D = densify(P9[r])
                        if kw.get("static"):
                            bx, vv = boxes_at(B[rt], V[rt], np.zeros(41))
                        else:
                            bx, vv = boxes_at(B[rt], V[rt], T41 + kw.get("dt", 0.0))
                        q = box_terms(D, bx, vv, eg_shrink=kw.get("eg", 0.0), ob_shrink=kw.get("ob", 0.0))
                    cag[i:i + len(r)], cas[i:i + len(r)], csd[i:i + len(r)] = q["ag_c"].numpy(), q["ag_s"].numpy(), q["sd"].numpy()
                res[f"cf_{vn}_ag_c"], res[f"cf_{vn}_ag_s"], res[f"cf_{vn}_sd"] = cag, cas, csd
            res["cf_rows"] = cf
            # lag sweep on every row (agent hinge activity of this path vs box time shift)
            lag = {}
            for dt in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5):
                act = np.zeros(n, bool)
                for i in range(0, len(idx), 4 * bs):
                    r = idx[i:i + 4 * bs]
                    rt = torch.as_tensor(r)
                    with torch.no_grad():
                        q = box_terms(densify(P9[r]), *boxes_at(B[rt], V[rt], T41 + dt))
                    act[r] = ((np.maximum(MARGIN - q["ag_c"].numpy(), 0).sum(1) + np.maximum(-q["ag_s"].numpy(), 0).sum(1)) > 0) & out["agent_ok"][r]
                lag[f"{dt:+.1f}"] = float(act[idx].mean())
            res["lag_sweep"] = np.array(json.dumps(lag))
            print(f"{a.pool} {path}: lag sweep {lag}", flush=True)
        for q, v in res.items():
            out[f"{path}__{q}"] = v.astype(np.float16) if v.dtype == np.float32 and q not in ("ag_c", "ag_s", "agp_c", "agp_s", "sd") else v
        out[f"{path}__ok"] = okp
        out[f"{path}__P9"] = P9.astype(np.float32)
        print(f"{a.pool} {path}: done {time.time() - t1:.0f} s", flush=True)
    O.mkdir(parents=True, exist_ok=True)
    np.savez(O / f"steps_{a.pool}{'_lim' if a.limit else ''}.npz", **out)
    print("written", O / f"steps_{a.pool}.npz", flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fwd")
    f.add_argument("--parts", choices=["S", "ph", "nav"], required=True)
    f.add_argument("--shard", default="0/1")
    f.add_argument("--frac", type=float, default=0.3)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--bs", type=int, default=256)
    f.add_argument("--threads", type=int, default=0)
    s = sp.add_parser("steps")
    s.add_argument("--pool", choices=["S", "PAID", "PAIH"], required=True)
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--limit", type=int, default=0)
    sp.add_parser("report")
    a = p.parse_args()
    dict(fwd=cmd_fwd, steps=globals().get("cmd_steps"), report=globals().get("cmd_report"))[a.cmd](a)


if __name__ == "__main__":
    main()
