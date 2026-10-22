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
  report (op-train)  every table of the result page from the steps files -> $O/audit.json
  g5     (op-train)  --pool P --designs ..: one shape-only step of each design's summed terms on LA rows -> $O/g5_<pool>.json
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


# ---------------------------------------------------------------- report
EPS = 0.05
TERMS = ("agent", "road", "onc", "soft", "side")
HARD = ("agent", "road", "onc", "side")


class Pool:
    """A steps file with the term values of every path under a design (fixed margins or per-row margins)."""

    def __init__(self, name, f):
        z = np.load(f, allow_pickle=True)
        self.name, self.z = name, z
        self.n = len(z["names"])
        self.paths = [p for p in ("log", "fut", "plan") if f"{p}__ok" in z.files]
        if name == "PAIH":
            self.paths = ["log", "plan"]
        self.has_lat = "lat_ok" in z.files
        self.terms = TERMS if self.has_lat else ("agent", "road", "side")
        self.C = {p: self._clear(p) for p in self.paths}

    def g(self, p, q):
        return self.z[f"{p}__{q}"].astype(np.float64)

    def _clear(self, p):
        """Row clearances of path p in each term's own metric (+ = room): agent corridor / side, road, onc / soft depth (negated), side."""
        win = ~self.z[f"{p}__oow"].astype(bool)
        c = dict(ag_c=self.g(p, "ag_c").min(1), ag_s=self.g(p, "ag_s").min(1), road=self.g(p, "road").min((1, 2)), sd=self.g(p, "sd")[:, 1:].min(1),
                 agp_c=self.g(p, "agp_c").min(1), agp_s=self.g(p, "agp_s").min(1),
                 road_w=np.where(win[..., None], self.g(p, "road"), BIG).min((1, 2)))
        if self.has_lat:
            c["onc"], c["soft"] = self.g(p, "onc").min(1), self.g(p, "soft").min(1)
        return c

    def ok(self, t):
        z = self.z
        if t in ("agent", "side"):
            return z["agent_ok"].astype(bool)
        if t == "road":
            return z["road_ok"].astype(bool)
        return z["lat_ok"].astype(bool) if self.has_lat else np.zeros(self.n, bool)

    def margins(self, design, ref=None):
        """Per-row margins (dict of (n,)) of a design. ref: list of paths whose clearances calibrate the margins (min over them)."""
        n = self.n
        m = dict(ag_c=np.full(n, MARGIN), ag_s=np.zeros(n), road=np.full(n, MARGIN), sd=np.full(n, BUF), onc=np.zeros(n), soft=np.zeros(n))
        if design in ("fixed", "gate"):
            return m
        if design in ("ec_log", "ec_wit"):
            ref = ["log"] if design == "ec_log" else [p for p in ("log", "fut") if p in self.paths]
            for k in m:
                if k not in self.C["log"]:
                    continue
                vals = []
                for p in ref:
                    v = self.C[p][k].copy()
                    v[~self.z[f"{p}__ok"]] = np.inf
                    vals.append(v)
                c = np.minimum.reduce(vals)
                if k in ("onc", "soft"):                     # depth allowed = max(0, human depth + eps); clearance metric is the sdf (- = inside)
                    m[k] = np.where(np.isfinite(c), np.maximum(0.0, -c + EPS), 0.0)
                else:
                    m[k] = np.where(np.isfinite(c), np.minimum(m[k], c - EPS), m[k])
            return m
        if design == "ctx":                                  # per-context quantile margins (calibrated on even scene hashes, see ctx_margins)
            return self.ctx_m
        if design == "spec":                                 # the recommended spec: witness-calibrated (min over log, fut), single front margin
            ref = [p for p in ("log", "fut") if p in self.paths]   # (plain ego box), in-window road clearance, overlap floor (term off)
            def wc(k):
                vals = []
                for p in ref:
                    v = self.C[p][k].copy()
                    v[~self.z[f"{p}__ok"]] = np.inf
                    vals.append(v)
                return np.minimum.reduce(vals)
            ca, cs, cr, cd = wc("agp_c"), wc("agp_s"), wc("road_w"), wc("sd")
            m["ag_c"] = np.where(np.isfinite(ca), np.minimum(MARGIN, ca - EPS), MARGIN)
            m["ag_s"] = np.where(np.isfinite(cs), np.minimum(0.0, cs - EPS), 0.0)
            m["road"] = np.where(np.isfinite(cr), np.minimum(MARGIN, cr - EPS), MARGIN)
            m["sd"] = np.where(np.isfinite(cd), np.minimum(BUF, cd - EPS), BUF)
            m["off_agent"] = np.minimum(ca, cs) < 0                 # a witness overlaps an object: no agent term on the row
            m["off_side"] = cd < 0
            if self.has_lat:
                for k in ("onc", "soft"):
                    c_ = wc(k)
                    m[k] = np.where(np.isfinite(c_), np.maximum(0.0, -c_ + EPS), 0.0)
            return m
        raise ValueError(design)

    def values(self, p, design="fixed", rows=None):
        """Term values (weighted, as the trainer) of path p under a design -> dict term -> (n,)."""
        m = self.margins(design)
        r = slice(None) if rows is None else rows
        ag_c, ag_s = self.g(p, "ag_c")[r], self.g(p, "ag_s")[r]
        mc, ms = m["ag_c"][r][:, None], m["ag_s"][r][:, None]
        out = dict(agent=W["agent"] * np.maximum(np.maximum(mc - ag_c, 0), np.maximum(ms - ag_s, 0)).mean(1),
                   road=W["road"] * np.maximum(m["road"][r][:, None, None] - self.g(p, "road")[r], 0).mean((1, 2)),
                   side=W["side"] * np.maximum(m["sd"][r][:, None] - self.g(p, "sd")[r][:, 1:], 0).mean(1))
        if self.has_lat:
            for t in ("onc", "soft"):
                out[t] = W[t] * np.maximum(-self.g(p, t)[r] - m[t][r][:, None], 0).mean(1)
        if design == "spec":
            agp_c, agp_s = self.g(p, "agp_c")[r], self.g(p, "agp_s")[r]
            out["agent"] = W["agent"] * np.maximum(np.maximum(mc - agp_c, 0), np.maximum(ms - agp_s, 0)).mean(1) * ~m["off_agent"][r]
            win = ~self.z[f"{p}__oow"][r].astype(bool)
            out["road"] = W["road"] * (np.maximum(m["road"][r][:, None, None] - self.g(p, "road")[r], 0) * win[..., None]).mean((1, 2))
            out["side"] = out["side"] * ~m["off_side"][r]
        okp = self.z[f"{p}__ok"][r]
        for t in out:
            out[t] = np.where(self.ok(t)[r] & okp, out[t], 0.0)
        if design == "gate":                                 # term kept on a row only where the human paths satisfy it
            hum = {q: self.values(q, "fixed", rows) for q in ("log", "fut") if q in self.paths}
            for t in out:
                bad = np.zeros_like(out[t], bool)
                for q, hv in hum.items():
                    bad |= hv[t] > 0
                out[t] = np.where(bad, 0.0, out[t])
        return out


def ctx_bins(P):
    """Context bin per row: speed band x scene (for the context-quantile margins and the strata)."""
    z = P.z
    v = z["speed"].astype(float)
    L = P.z["log__P9"] if "log__P9" in z.files else None
    dpsi = np.degrees(np.abs(np.unwrap(L[..., 2], axis=1)[:, 8] - np.unwrap(L[..., 2], axis=1)[:, 0])) if L is not None else np.zeros(P.n)
    junc = z["junc"].astype(float) if "junc" in z.files else np.full(P.n, np.nan)
    scene = np.select([v < 0.5, dpsi > 30, np.nan_to_num(junc) > 0.1, z["lead_d"] < 20, z["parked"].astype(bool)],
                      ["standing", "turn>30", "junction", "lead<20m", "parked beside"], "straight free")
    spd = np.select([v < 0.5, v < 3, v < 8, v < 13, v < 17], ["<0.5", "0.5-3", "3-8", "8-13", "13-17"], ">17")
    return spd, scene, dpsi


def scene_hash(P):
    z = P.z
    if "scene" in z.files:
        s = z["scene"].astype(str)
    else:
        s = np.array([":".join(x.split(":")[2:-2]) if x.count(":") >= 4 else ":".join(x.split(":")[:-1]) for x in z["names"].astype(str)])
    import zlib
    return np.array([zlib.crc32(x.encode()) % 2 for x in s])


def ctx_margins(P, spd, scene, q=0.05):
    """Context-quantile margins: per (speed band, scene) the q-quantile of the human log clearance, clipped to [lo, m0]; calibrated on the
    rows of even-hash scenes only (the odd half is the test)."""
    h = scene_hash(P)
    cal = (h == 0) & P.z["log__ok"]
    m0 = P.margins("fixed")
    out, tab = {k: v.copy() for k, v in m0.items()}, {}
    key = np.char.add(np.char.add(spd.astype(str), "|"), scene.astype(str))
    for k in m0:
        if k not in P.C["log"]:
            continue
        c = P.C["log"][k]
        for b in np.unique(key):
            sel = key == b
            cc = c[sel & cal & np.isfinite(c) & (c < BIG / 2)]
            if k in ("onc", "soft"):
                val = max(0.0, float(-np.quantile(cc, q)) + EPS) if len(cc) > 20 else 0.0
                out[k][sel] = val
            else:
                val = min(m0[k][0], float(np.quantile(cc, q)) - EPS) if len(cc) > 20 else m0[k][0]
                out[k][sel] = val
            tab[f"{k}|{b}"] = round(float(val), 3)
    return out, tab, h


def cover(v, dpsi, act, la):
    """LA rows in the loss-map bands (pai_s_gate.cover): counts relative to all rows."""
    rs = (act["road"] > 0) | (act["side"] > 0)
    ag = act["agent"] > 0
    return dict(turn_inside=la & rs & (dpsi > 30) & (v >= 2.6) & (v <= 12.2), straight_offroad=la & rs & (dpsi < 5) & (v >= 1.3) & (v <= 24.2),
                drift=la & rs & (dpsi < 5) & (v >= 17.2), collision=la & ag & (v >= 2.7) & (v <= 16.1), gt13=la & (v > 13))


def cmd_report(a):
    import pandas as pd
    R = {}
    md = []
    pools = {}
    for name in ("S", "PAID", "PAIH"):
        f = O / f"steps_{name}.npz"
        if f.exists():
            pools[name] = Pool(name, f)
    wz = np.load(DATA / "runs/round6/dag/w/R6DAG.npz")
    wmap = dict(zip(wz["names"].astype(str).tolist(), wz["w"].tolist()))
    DOSE = dict(S=32, PAID=16, PAIH=32)
    for name, P in pools.items():
        z = P.z
        spd, scene, dpsi = ctx_bins(P)
        v = z["speed"].astype(float)
        P.ctx_m, ctx_tab, hsh = ctx_margins(P, spd, scene)
        live = np.array([wmap.get(x, 0.0) for x in z["names"].astype(str)]) if name == "S" else np.ones(P.n)
        cand = z["draw"].astype(bool) & (live > 0) if "draw" in z.files else live > 0
        res = dict(rows=P.n, candidates=int(cand.sum()), ctx_margins=ctx_tab)
        fut_dpsi = np.degrees(np.abs(z["fut__P9"][:, 8, 2])) if "fut__P9" in z.files else dpsi
        # ---- 1. unconditional human / teacher violation per term, fixed design
        V = {p: P.values(p, "fixed") for p in P.paths}
        aud = {}
        for t in P.terms:
            okt = P.ok(t)
            row = {}
            for p in P.paths:
                m = okt & z[f"{p}__ok"] & cand
                row[p] = dict(n=int(m.sum()), viol=float((V[p][t][m] > 0).mean()) if m.any() else None, mass_mean=float(V[p][t][m].mean()) if m.any() else None)
            if "plan" in V:
                pm = okt & z["plan__ok"] & cand
                act = pm & (V["plan"][t] > 0)
                row["plan_active"] = float(act.sum() / max(pm.sum(), 1))
                for p in ("log", "fut"):
                    if p in V:
                        mm = act & z[f"{p}__ok"]
                        row[f"{p}_viol_given_active"] = float((V[p][t][mm] > 0).mean()) if mm.any() else None
                        row[f"{p}_viol_uncond_same_rows"] = float((V[p][t][pm & z[f"{p}__ok"]] > 0).mean()) if pm.any() else None
            aud[t] = row
        res["audit"] = aud
        # strata of the human log violation (unconditional, candidates)
        st = []
        for t in P.terms:
            m = P.ok(t) & z["log__ok"] & cand
            hv = V["log"][t] > 0
            ck = {"agent": "ag_c", "road": "road", "onc": "onc", "soft": "soft", "side": "sd"}[t]
            c = P.C["log"][ck]
            cb = np.select([c < 0, c < 0.1, c < 0.3, c < 1.0, c < BIG / 2], ["<0", "0-0.1", "0.1-0.3", "0.3-1", ">1"], "none")
            for lab, arr in (("speed", spd), ("scene", scene), ("clearance", cb)):
                for b in np.unique(arr[m]):
                    s = m & (arr == b)
                    st.append(dict(term=t, by=lab, bin=str(b), rows=int(s.sum()), share=float(s.sum() / max(m.sum(), 1)),
                                   human_viol=float(hv[s].mean()), human_mass_share=float(V["log"][t][s].sum() / max(V["log"][t][m].sum(), 1e-12))))
        res["strata"] = st
        # ---- 2. root causes of the human (log) and the recovery label / teacher (fut) violations
        RC = {}
        for RP in [q for q in ("log", "fut") if q in P.paths]:
            rc = {}
            cf_rows = z[f"{RP}__cf_rows"] if f"{RP}__cf_rows" in z.files else np.zeros(0, int)
            cpos = {int(r): i for i, r in enumerate(cf_rows)}

            def cf_ok(var, rows):
                i = np.array([cpos[int(r)] for r in rows], int)
                ac, as_ = z[f"{RP}__cf_{var}_ag_c"][i].astype(float), z[f"{RP}__cf_{var}_ag_s"][i].astype(float)
                return np.minimum(ac, as_).min(1) >= 0, z[f"{RP}__cf_{var}_sd"][i].astype(float)[:, 1:].min(1) >= 0
            L9 = z[f"{RP}__P9"]
            for t in P.terms:
                m = P.ok(t) & z[f"{RP}__ok"] & cand & (V[RP][t] > 0)
                rows = np.flatnonzero(m)
                if not len(rows):
                    rc[t] = dict(n=0)
                    continue
                cause = np.full(len(rows), "residual", object)
                inc = {}
                if t == "agent":
                    ac, as_ = P.g(RP, "ag_c")[rows], P.g(RP, "ag_s")[rows]
                    init = (np.maximum(MARGIN - ac[:, 0], 0) + np.maximum(-as_[:, 0], 0)) > 0
                    pc, ps = P.g(RP, "agp_c")[rows].min(1), P.g(RP, "agp_s")[rows].min(1)
                    real = np.minimum(pc, ps) < 0
                    front_only = ~real & (pc >= MARGIN)
                    o_geo, _ = cf_ok("ego_shrink", rows)
                    o_box, _ = cf_ok("obj_shrink", rows)
                    o_t = np.zeros(len(rows), bool)
                    for vn in ("shift_m05", "shift_m025", "shift_p025", "shift_p05"):
                        o_t |= cf_ok(vn, rows)[0]
                    o_st, _ = cf_ok("static", rows)
                    inc = dict(initial=init, margin_front_extension=front_only, margin=~real & ~front_only, geometry=real & o_geo, box_inflation=real & o_box,
                               time_shift=real & o_t, real_overlap=real, static_boxes_would_clear=o_st)
                    order = ["initial", "margin_front_extension", "margin", "geometry", "box_inflation", "time_shift"]
                elif t == "road":
                    rd = P.g(RP, "road")[rows]
                    init = rd[:, 0].min(1) < MARGIN
                    mn = rd.min((1, 2))
                    marg = mn >= 0
                    gap = np.zeros(len(rows), bool)
                    if P.name == "S":
                        dv = P.g(RP, "drv")[rows]
                        neg = rd < 0
                        gap = ~marg & np.where(neg, dv >= 0, True).all((1, 2))
                    oow = z[f"{RP}__oow"][rows].any(1)
                    pts = P.g(RP, "pts")[rows].min((1, 2)) >= 0
                    geo = ~marg & pts & (mn > -0.5)
                    inc = dict(initial=init, margin=marg, raster_gap=gap, coverage=oow, geometry=geo, centre_outside=P.g(RP, "pts")[rows].min((1, 2)) < 0)
                    order = ["initial", "margin", "raster_gap", "coverage", "geometry"]
                elif t in ("onc", "soft"):
                    x = P.g(RP, t)[rows]
                    init = x[:, :2].min(1) < 0 if x.shape[1] == 80 else x[:, :1].min(1) < 0
                    shallow = (-x.min(1)) < 0.3
                    Lr = L9[rows].astype(float)
                    c0, s0 = np.cos(Lr[:, 0, 2]), np.sin(Lr[:, 0, 2])
                    lat_disp = np.abs(-s0 * (Lr[:, 8, 0] - Lr[:, 0, 0]) + c0 * (Lr[:, 8, 1] - Lr[:, 0, 1]))
                    if t == "onc":
                        legit = (dpsi[rows] > 30) | z["parked"][rows].astype(bool) | ((z["lead_d"][rows] < 25) & ~z["lead_moving"][rows].astype(bool))
                        inc = dict(initial=init, depth_lt_0p3=shallow, legit_turn_or_pass=legit, turn=dpsi[rows] > 30)
                    else:
                        legit = lat_disp > 1.5
                        inc = dict(initial=init, depth_lt_0p3=shallow, legit_lane_change=legit)
                    order = list(inc)[:3]
                else:                                            # side
                    sd = P.g(RP, "sd")[rows]
                    init = sd[:, 1] < BUF
                    marg = sd[:, 1:].min(1) >= 0
                    _, g_ = cf_ok("ego_shrink", rows)
                    _, b_ = cf_ok("obj_shrink", rows)
                    tt = np.zeros(len(rows), bool)
                    for vn in ("shift_m05", "shift_m025", "shift_p025", "shift_p05"):
                        tt |= cf_ok(vn, rows)[1]
                    inc = dict(initial=init, margin=marg, geometry=~marg & g_, box_inflation=~marg & b_, time_shift=~marg & tt)
                    order = ["initial", "margin", "geometry", "box_inflation", "time_shift"]
                for k in reversed(order):
                    cause = np.where(inc[k], k, cause)
                order2 = [k for k in order if k != "initial"] + ["initial"]          # clearance-first reading (initial is a context, not a cause)
                cause2 = np.full(len(rows), "residual", object)
                for k in reversed(order2):
                    cause2 = np.where(inc[k], k, cause2)
                vals = V[RP][t][rows]
                rc[t] = dict(n=int(len(rows)), first_match={k: float((cause == k).mean()) for k in order + ["residual"]},
                             first_match_mass={k: float(vals[cause == k].sum() / vals.sum()) for k in order + ["residual"]},
                             clearance_first={k: float((cause2 == k).mean()) for k in order2 + ["residual"]},
                             clearance_first_mass={k: float(vals[cause2 == k].sum() / vals.sum()) for k in order2 + ["residual"]},
                             incidence={k: float(np.mean(x)) for k, x in inc.items()})
            RC[RP] = rc
        rc = RC
        if "log__lag_sweep" in z.files:
            rc["lag_sweep_log"] = json.loads(str(z["log__lag_sweep"]))
        if "fut__lag_sweep" in z.files:
            rc["lag_sweep_fut"] = json.loads(str(z["fut__lag_sweep"]))
        res["root_causes"] = rc
        # ---- 4. designs
        DES = ["fixed", "gate", "ec_log", "ec_wit", "ctx", "spec"]
        if "plan" in P.paths:                                # the plan's road mass from corners outside the raster window (border clamp)
            pm_ = z["plan__ok"] & cand
            rv = np.maximum(MARGIN - P.g("plan", "road")[pm_], 0) * P.ok("road")[pm_][:, None, None]
            ow = z["plan__oow"][pm_].astype(bool)
            res["plan_road_oow"] = dict(rows_with_oow=float(ow.any(1).mean()), road_mass_share_oow=float(rv[ow].sum() / max(rv.sum(), 1e-12)),
                                        by_speed={b: float(ow[spd[pm_] == b].any(1).mean()) for b in np.unique(spd[pm_])})
        if "plan" in P.paths and (z["plan__ok"] & cand).any():
            pm = z["plan__ok"] & cand
            w_ = live[pm] if name == "S" else np.ones(pm.sum())
            base_cov = None
            dres = {}
            for d in DES:
                Vd = {p: P.values(p, d) for p in P.paths}
                tv = {t: Vd["plan"][t][pm] for t in P.terms}
                act = {t: tv[t] > 0 for t in P.terms}
                anyact = np.any([act[t] for t in P.terms], 0)
                fut_k = "fut" if "fut" in Vd else "log"
                un = (z["off_t0"][pm].astype(bool) if name != "PAIH" else np.zeros(pm.sum(), bool)).copy()
                un_ar = un.copy()
                for t in P.terms:
                    u_t = act[t] & (Vd[fut_k][t][pm] > 0)
                    un |= u_t
                    if t in ("agent", "road"):
                        un_ar |= u_t
                la = anyact & ~un
                mass = np.sum([tv[t] for t in P.terms], 0)
                hard_mass = np.sum([tv[t] for t in P.terms if t != "soft"], 0)
                row = dict(active=float(np.average(anyact, weights=w_)), la=float(np.average(la, weights=w_)),
                           la_per_batch=float(DOSE[name] * np.average(la, weights=w_)), unlearnable=float(np.average(un, weights=w_)),
                           mass_mean=float(np.average(mass, weights=w_)), learnable_mass=float((mass * w_)[~un].sum() / max((mass * w_).sum(), 1e-12)),
                           learnable_mass_agent_road_def=float((mass * w_)[~un_ar].sum() / max((mass * w_).sum(), 1e-12)),
                           hard_mass_mean=float(np.average(hard_mass, weights=w_)))
                for t in P.terms:
                    okm = P.ok(t)
                    hm = okm & z["log__ok"] & cand
                    row[f"{t}_human_viol"] = float((Vd["log"][t][hm] > 0).mean()) if hm.any() else None
                    if d == "ctx":                                  # out-of-sample half: odd scene hashes
                        hm2 = hm & (hsh == 1)
                        row[f"{t}_human_viol_test_half"] = float((Vd["log"][t][hm2] > 0).mean()) if hm2.any() else None
                    if "fut" in Vd:
                        fm = okm & z["fut__ok"] & cand
                        row[f"{t}_fut_viol"] = float((Vd["fut"][t][fm] > 0).mean()) if fm.any() else None
                        a_ = act[t]
                        row[f"{t}_fut_viol_given_active"] = float((Vd["fut"][t][pm][a_] > 0).mean()) if a_.any() else None
                    row[f"{t}_active"] = float(np.average(act[t], weights=w_))
                    row[f"{t}_la"] = float(np.average(act[t] & ~un, weights=w_))
                    row[f"{t}_mass_share"] = float((tv[t] * w_).sum() / max((mass * w_).sum(), 1e-12))
                cv = cover(v[pm], fut_dpsi[pm], {t: tv.get(t, np.zeros(pm.sum())) for t in ("agent", "road", "side")}, la)
                cvr = {k: float(DOSE[name] * np.average(x, weights=w_)) for k, x in cv.items()}
                row["cover_per_batch"] = cvr
                if d == "fixed":
                    base_cov = cvr
                row["cover_rel_fixed"] = {k: (cvr[k] / base_cov[k] if base_cov[k] > 0 else None) for k in cvr}
                dres[d] = row
            res["designs"] = dres
        R[name] = res
        print(name, json.dumps({k: res[k] for k in ("rows", "candidates")}), flush=True)
    # ---- navtrain drivable hinge (human label on every labelled row; plan on the fwd sample)
    f = O / "fwd_nav_0.npz"
    if f.exists():
        z = np.load(f)
        h, ms = z["all_fut_hinge"], z["all_fut_minsdf"]
        nav = dict(rows=int(len(h)), human_viol=float((h > 0).mean()), human_overlap=float((ms < 0).mean()),
                   human_minsdf_q=[float(x) for x in np.quantile(ms, [0.01, 0.05, 0.25, 0.5])],
                   plan_active=float((z["tr_nav"] > 0).mean()), human_viol_given_plan_active=float((z["tr_nav_fut"][z["tr_nav"] > 0] > 0).mean()))
        for mg in (0.2, 0.1, 0.0):
            nav[f"human_viol_margin_{mg}"] = float((ms < mg).mean())
        R["navtrain"] = nav
    O.mkdir(parents=True, exist_ok=True)
    (O / "audit.json").write_text(json.dumps(R, indent=1, default=float))
    print("written", O / "audit.json", flush=True)


def cmd_g5(a):
    """G5-style gradient sanity of a design on LA rows of a pool: one shape-only step (max 0.2 m per pose) of the summed S terms on the
    R6DAG-L18-s1 plan -> term reduction, 4 s arc / end-speed change, cos toward the recovery label `fut` and toward the human log line."""
    import torch
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    from nr1_lat import shape_only
    import nr1_lat
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    P = Pool(a.pool, O / f"steps_{a.pool}.npz")
    z = P.z
    spd, scene, dpsi = ctx_bins(P)
    P.ctx_m, _, _ = ctx_margins(P, spd, scene)
    sp = pool_spec(a.pool)
    out = {}
    for d in a.designs.split(","):
        Vd = {p: P.values(p, d) for p in P.paths}
        fut_k = "fut" if "fut" in Vd else "log"
        act = np.any([Vd["plan"][t] > 0 for t in P.terms], 0) & z["plan__ok"]
        un = z["off_t0"].astype(bool) if a.pool != "PAIH" else np.zeros(P.n, bool)
        for t in P.terms:
            un = un | ((Vd["plan"][t] > 0) & (Vd[fut_k][t] > 0))
        la = np.flatnonzero(act & ~un & z["draw"].astype(bool))
        rows = np.sort(np.random.default_rng(0).choice(la, min(a.n, len(la)), replace=False))
        names = z["names"][rows].astype(str)
        ah = AgentHinge(sp["agent"], names, "cpu", MARGIN, 0.0)
        road = Hinge(sp["road"], names, "cpu", MARGIN)
        road.sdf[~road.ok] = 0.0                                          # unlabelled rows hold uninitialised rasters (NaN gradients)
        if sp["drv"] is not None:
            fb = np.flatnonzero(z["off_t0"][rows])
            if len(fb):
                dr = Hinge(sp["drv"], names, "cpu", MARGIN)
                fb = torch.as_tensor(fb)
                road.sdf[fb], road.ok[fb] = dr.sdf[fb], dr.ok[fb]
        pack = None
        if P.has_lat:
            lm = np.load(sp["lat"] / "meta.npz")
            pack = dict(bins=torch.as_tensor(lm["bins"].astype(np.float32)), points=str(lm["points"]) if "points" in lm.files else "both",
                        onc=np.load(sp["lat"] / "onc.npy", mmap_mode="r"), soft=np.load(sp["lat"] / "soft.npy", mmap_mode="r"))
            lat_rows = sp["lat_rows"][rows]
        m = P.margins(d)
        M = {k: torch.as_tensor(np.asarray(v)[rows], dtype=torch.float32) for k, v in m.items()}
        keep = {t: torch.as_tensor((P.values("plan", d)[t][rows] > 0) | True) for t in P.terms}
        if d == "gate":                                       # rows / terms the gate switched off stay off
            hv = {q: P.values(q, "fixed") for q in ("log", "fut") if q in P.paths}
            for t in P.terms:
                bad = np.zeros(len(rows), bool)
                for q in hv:
                    bad |= hv[q][t][rows] > 0
                keep[t] = torch.as_tensor(~bad)
        okA = torch.as_tensor(z["agent_ok"][rows].astype(bool))
        okR = torch.as_tensor(z["road_ok"][rows].astype(bool))
        okL = torch.as_tensor(z["lat_ok"][rows].astype(bool)) if P.has_lat else None
        Pp = torch.as_tensor(z["plan__P9"][rows, 1:].astype(np.float32))
        ri = torch.arange(len(rows))

        def f(x, y, psi):
            xs, ys = shape_only(x, y, psi)
            P9 = torch.cat([torch.zeros(len(x), 1, 3), torch.stack([xs, ys, psi], -1)], 1)
            D = torch.einsum("kj,bjc->bkc", torch.as_tensor(interp_at(T41), dtype=torch.float32), P9)
            bx, vv = boxes_at(ah.box[ri], ah.val[ri], T41)
            q = box_terms(D, bx, vv)
            tot = {}
            sp_ = d == "spec"
            kc, ks = ("agp_c", "agp_s") if sp_ else ("ag_c", "ag_s")
            tot["agent"] = W["agent"] * torch.maximum((M["ag_c"][:, None] - q[kc]).clamp_min(0), (M["ag_s"][:, None] - q[ks]).clamp_min(0)).mean(1) * okA
            tot["side"] = W["side"] * (M["sd"][:, None] - q["sd"][:, 1:]).clamp_min(0).mean(1) * okA
            C, _ = foot(D)
            rr = (M["road"][:, None] - sample(road.sdf[ri], C.reshape(len(x), -1, 2))).clamp_min(0)
            if sp_:
                tot["agent"] = tot["agent"] * (1 - M["off_agent"])
                tot["side"] = tot["side"] * (1 - M["off_side"])
                win = ~((C[..., 0] < X0) | (C[..., 0] > X0 + 64) | (C[..., 1].abs() > 24)).any(-1)          # (B, 41) step in the window
                rr = rr * win.detach().float().repeat_interleave(4, 1)
            tot["road"] = torch.where(okR, W["road"] * rr.mean(1), torch.zeros(()))         # unlabelled rows hold uninitialised rasters
            if pack is not None:
                lo = lat_rows >= 0
                for k in ("onc", "soft"):
                    v_ = torch.zeros(len(x))
                    if lo.any():
                        sdf = nr1_lat.region_m(D[torch.as_tensor(lo)], torch.as_tensor(lat_rows[lo]), pack, k)
                        v_[torch.as_tensor(lo)] = W[k] * (-sdf - M[k][torch.as_tensor(lo)][:, None]).clamp_min(0).mean(1)
                    tot[k] = v_ * okL
            return sum(tot[t] * keep[t] for t in tot)
        x, y, psi = Pp[..., 0].clone().requires_grad_(True), Pp[..., 1].clone().requires_grad_(True), Pp[..., 2]
        v0 = f(x, y, psi)
        v0.sum().backward()
        gx, gy = x.grad, y.grad
        nrm = torch.hypot(gx, gy).amax(1, keepdim=True).clamp_min(1e-9)
        nx, ny = (x - 0.2 * gx / nrm).detach(), (y - 0.2 * gy / nrm).detach()
        with torch.no_grad():
            v1 = f(nx, ny, psi)
        P0, P1 = Pp.numpy(), torch.stack([nx, ny, psi], -1).numpy()

        def arc(Q):
            Q = np.concatenate([np.zeros((len(Q), 1, 2)), Q[..., :2]], 1)
            sg = np.hypot(*np.diff(Q, axis=1).transpose(2, 0, 1))
            return sg.sum(1), sg[:, -1] / 0.5
        a0, e0 = arc(P0)
        a1, e1 = arc(P1)
        dv = (P1 - P0)[..., :2].reshape(len(P0), -1)
        res = dict(rows=len(rows), la_rows=int(len(la)), term_reduced=float((v1 < v0.detach()).float().mean()),
                   term_rel_reduction_med=float(np.median(((v0.detach() - v1) / v0.detach().clamp_min(1e-9)).numpy())),
                   arc_rel_change_med=float(np.median((a1 - a0) / np.maximum(a0, 1e-3))), end_speed_change_med=float(np.median(e1 - e0)))
        for ref in [q for q in ("fut", "log") if f"{q}__P9" in z.files]:
            L = z[f"{ref}__P9"][rows, 1:].astype(np.float64)
            okr = np.isfinite(L).all((1, 2))
            to = (L - P0)[..., :2].reshape(len(P0), -1)
            cos = (dv * to).sum(1) / np.maximum(np.linalg.norm(dv, axis=1) * np.linalg.norm(to, axis=1), 1e-9)
            d0 = np.linalg.norm((L - P0)[..., :2], axis=2).mean(1)
            d1 = np.linalg.norm((L - P1)[..., :2], axis=2).mean(1)
            res[f"toward_{ref}"] = float((cos[okr] > 0).mean())
            res[f"away_from_{ref}_line"] = float((d1[okr] > d0[okr] + 1e-4).mean())
        out[d] = res
        print(a.pool, d, json.dumps(res), flush=True)
    (O / f"g5_{a.pool}.json").write_text(json.dumps(out, indent=1))


def retime(G, S_new):
    """Path knots G (n, 9, 3) (start + 8 poses) -> poses at arc lengths S_new (n, 8) along G's polyline (extended straight past its end)."""
    out = np.zeros(S_new.shape + (3,))
    for i in range(len(G)):
        g = G[i].astype(float)
        seg = np.hypot(*np.diff(g[:, :2], axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        yaw = np.unwrap(g[:, 2])
        for k, sk in enumerate(S_new[i]):
            if sk <= s[-1] and s[-1] > 1e-6:
                j = min(np.searchsorted(s, sk, side="right") - 1, 7)
                f = 0.0 if seg[j] < 1e-9 else (sk - s[j]) / seg[j]
                out[i, k] = g[j] * (1 - f) + g[j + 1] * f
                out[i, k, 2] = yaw[j] * (1 - f) + yaw[j + 1] * f
            else:
                e = sk - s[-1]
                out[i, k] = [g[-1, 0] + e * np.cos(yaw[-1]), g[-1, 1] + e * np.sin(yaw[-1]), yaw[-1]]
    return out


def cmd_retime(a):
    """PAID teacher label, shape vs speed: the agent / side terms of the teacher's path re-timed to the human's arc lengths (and to
    min(teacher, human) arc lengths) on every candidate row where the label violates agent -> $O/retime_PAID.json."""
    import torch
    from agent_hinge import AgentHinge
    from jevdrive.common import n_cpus
    torch.set_num_threads(a.threads or n_cpus())
    P = Pool("PAID", O / "steps_PAID.npz")
    z = P.z
    Vf = P.values("fut", "fixed")
    Vl = P.values("log", "fixed")
    rows = np.flatnonzero((Vf["agent"] > 0) & z["log__ok"] & z["draw"].astype(bool))
    sp = pool_spec("PAID")
    ah = AgentHinge(sp["agent"], z["names"][rows].astype(str), "cpu", MARGIN, 0.0)
    F, L = z["fut__P9"][rows].astype(float), z["log__P9"][rows].astype(float)
    arc = lambda G: np.cumsum(np.hypot(*np.diff(G[..., :2], axis=1).transpose(2, 0, 1)), 1)     # noqa: E731
    sf, sl = arc(F), arc(L)
    res = dict(rows=int(len(rows)), label_arc4_over_human_med=float(np.median(sf[:, -1] / np.maximum(sl[:, -1], 0.1))),
               label_longer_than_human=float((sf[:, -1] > sl[:, -1] + 0.5).mean()), human_also_violates=float((Vl["agent"][rows] > 0).mean()))
    st = np.maximum(L[:, :1, 0] + sl, 0.0)                     # the human's station along the ego heading at each step (station-matched)
    res["human_t0_ahead_of_student_med"] = float(np.median(L[:, 0, 0]))
    res["human_t0_behind_student"] = float((L[:, 0, 0] < -0.5).mean())
    for nm, S_new in (("human_timing", sl), ("min_timing", np.minimum(sf, sl)), ("station_matched", st), ("min_station", np.minimum(sf, st))):
        G = np.concatenate([F[:, :1], retime(F, S_new)], 1)
        v = []
        for i in range(0, len(rows), 1024):
            rt = torch.arange(i, min(i + 1024, len(rows)))
            with torch.no_grad():
                q = box_terms(densify(G[rt.numpy()]), *boxes_at(ah.box[rt], ah.val[rt], T41))
            v.append((np.maximum(np.maximum(MARGIN - q["ag_c"].numpy(), 0), np.maximum(-q["ag_s"].numpy(), 0)).mean(1) > 0))
        res[f"{nm}_still_violates"] = float(np.concatenate(v).mean())
    (O / "retime_PAID.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


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
    g = sp.add_parser("g5")
    g.add_argument("--pool", choices=["S", "PAID", "PAIH"], required=True)
    g.add_argument("--designs", default="fixed,gate,ec_log,ec_wit,ctx,spec")
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--threads", type=int, default=0)
    r = sp.add_parser("retime")
    r.add_argument("--threads", type=int, default=0)
    a = p.parse_args()
    dict(fwd=cmd_fwd, steps=cmd_steps, report=cmd_report, g5=cmd_g5, retime=cmd_retime)[a.cmd](a)


if __name__ == "__main__":
    main()
