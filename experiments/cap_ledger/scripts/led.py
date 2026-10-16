#!/usr/bin/env python3
"""CAP-LEDGER (plans/2026-10-16-cap-ledger-prereg.md): which driving capabilities of the shipped Cinque (P0) does each stage of our lineage
lose, keep or gain? Decision 326's method (wait_cap) extended from waiting to the main behaviours: open loop on all 2 Hz frames of the 147
NAVSIM test logs (navtest filter not applied, never trained on), cells by the human's own logged behaviour, frozen Cinque encoder on the W
protocol, the same 19 checkpoints. No training.

  select  (box, navsim2 python, CPU)  led.py select     per frame: wait_cap's row (wc.log_rows: speeds, future, reference path, in-path
                                      lead) + the hindsight route's stop lines and light states (long_contact lc_ctx.label_log) + map lanes /
                                      intersections, pedestrians and crossing vehicles -> cells; sampled rows (one per token, a membership
                                      column per cell) -> $DATA_DIR/runs/cap_ledger/select/rows.npz, cells.md
  define  (box, any python)           led.py define     registers navsim/capledger_test (jevdrive.data.splits)
  index   (box, navsim2 python)       led.py index      wait_cap's index on these tokens -> runs/navsim_zs/index/capledger_test{.pkl,_future.npz}
  (then)  pp_prep.py --data idx_capledger_test --frames warp --no-side ; rm_build.py nav --data idx_capledger_test
  plans   (box, op-train python, GPU) led.py plans --tags ..   per checkpoint and history condition (H = the human's history, F = constant
                                      velocity: vx kept, vy = ax = ay = 0, the past poses re-spaced at vx, rf_lonol.flat_rows) the rear-axle plan
                                      -> runs/cap_ledger/plans/<H|F>/<tag>.npz
  ledger  (Mac or box, CPU)           led.py ledger --rows R --plans DIR [--wc-rows R --wc-plans DIR] --out RESULTS   the registered
                                      success measures per cell, per stage, the paired CIs and the classification (prereg section 4)

Frames: rear axle at t0, x forward, y left (OpenScene lidar2ego is the identity). Shared code is wait_cap's wc.py, imported unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


WC = _load("wc", "experiments/wait_cap/scripts/wc.py")
DATA = WC.DATA
NAV = WC.NAV
OUT = DATA / "runs/cap_ledger"
SPLIT = "capledger_test"
FRONT = WC.FRONT
CAP, GAP_FR, SEED, SPARSE = 1200, 4, 20261016, 300
MIN_ROWS, MIN_LOGS = 40, 10                    # a cell is classified only with >= 40 rows from >= 10 logs
NC = 61                                          # lane centreline: 0 .. 60 m every 1 m from the rear axle's projection
CELLS = ("S_L", "S_R", "S_SS", "GO_G", "F", "C", "LC_L", "LC_R", "T_L", "T_R", "K", "Y_P", "Y_V")
DECEL = ("S_L", "S_R", "S_SS", "Y_P", "Y_V")      # primary read = constant-velocity history (F)
GROUPS = {"wait": ("W",), "launch": ("G",), "stop_lead": ("S_L",), "stop_red": ("S_R",), "stop_sign": ("S_SS",), "go_green": ("GO_G",),
          "follow": ("F",), "cruise": ("C",), "lane_change": ("LC_L", "LC_R"), "turn": ("T_L", "T_R"), "curve": ("K",), "yield": ("Y_P", "Y_V")}
PATCH = ("wait", "launch", "stop_lead", "stop_red", "stop_sign")      # the scope of the parallel wait / stop patch (wait-train design rows)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def signed_lat(C, pts):
    """Signed lateral offset (left +) of points (k, 2) from the polyline C (n, 2) (nearest segment, clamped)."""
    a, d = C[:-1], C[1:] - C[:-1]
    t = np.clip((((pts[:, None] - a[None]) * d[None]).sum(-1)) / np.maximum((d * d).sum(-1), 1e-12)[None], 0, 1)
    q = a[None] + t[..., None] * d[None]
    D = np.hypot(*(pts[:, None] - q).transpose(2, 0, 1))
    j = D.argmin(1)
    k = np.arange(len(pts))
    cr = d[j, 0] * (pts[:, 1] - a[j, 1]) - d[j, 1] * (pts[:, 0] - a[j, 0])
    return np.sign(cr) * D[k, j]


# ---------------------------------------------------------------- select (navsim2 venv, CPU)
_LM = {}
_LCX = None


def lane_map(loc):
    if loc in _LM:
        return _LM[loc]
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
    from shapely.strtree import STRtree
    m = NuPlanMapFactory(get_maps_db(os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"])).build_map_from_name(loc)
    ln, bp, ix = (m._get_vector_map_layer(x) for x in (L.LANE, L.BASELINE_PATHS, L.INTERSECTION))
    base = {str(int(f)): g for f, g in zip(bp.lane_fid, bp.geometry) if f == f}
    lg = list(ln.geometry)
    _LM[loc] = dict(lg=lg, tree=STRtree(lg), fid=[str(x) for x in ln.fid], grp=[str(x) for x in ln.lane_group_fid], base=base,
                    ig=list(ix.geometry), itree=STRtree(list(ix.geometry)))
    return _LM[loc]


def lane_at(M, xy, yaw):
    """(lane fid, lane group) of the lane polygon containing the global point, the best heading match among overlaps; (None, None) if none."""
    from shapely.geometry import Point
    p = Point(*xy)
    best, bd = (None, None), 1e9
    for c in M["tree"].query(p):
        if not M["lg"][c].covers(p):
            continue
        b = M["base"].get(M["fid"][c])
        if b is None:
            continue
        s = b.project(p)
        q0, q1 = b.interpolate(max(s - 1.0, 0.0)), b.interpolate(min(s + 1.0, b.length))
        dh = abs(wrap(np.arctan2(q1.y - q0.y, q1.x - q0.x) - yaw))
        if dh < bd:
            best, bd = (M["fid"][c], M["grp"][c]), dh
    return best if bd < np.radians(60) else (None, None)


def centreline(M, fid, xy, yaw):
    """Lane fid's baseline from the rear axle's projection, 0 .. 60 m every 1 m in the t0 ego frame (straight beyond its end)."""
    from shapely.geometry import Point
    b = M["base"][fid]
    s0 = b.project(Point(*xy))
    s = np.arange(NC, dtype=float) + s0
    L = b.length
    pts = np.array([[b.interpolate(min(v, L)).x, b.interpolate(min(v, L)).y] for v in s])
    q0, q1 = b.interpolate(max(L - 2.0, 0.0)), b.interpolate(L)
    h = np.arctan2(q1.y - q0.y, q1.x - q0.x)
    far = s > L
    pts[far] = np.array([q1.x, q1.y]) + (s[far] - L)[:, None] * [np.cos(h), np.sin(h)]
    return WC.to_ego(pts, np.asarray(xy), yaw)


def log_cells(path):
    """One test log -> list of per-frame dicts (wait_cap's row + stop lines + map + conflicts)."""
    global _LCX
    if _LCX is None:                                                 # once per worker: its stop-line map cache lives in the module
        _LCX = _load("lc_ctx", "experiments/long_contact/scripts/lc_ctx.py")
    LCX = _LCX
    from shapely.geometry import Point
    base = WC.log_rows(path)
    if not base:
        return []
    fr = pickle.load(open(path, "rb"))
    M = lane_map(fr[0]["map_location"])
    idx = {f["token"]: i for i, f in enumerate(fr)}
    ctx = dict(LCX.label_log((path, [(k, r["token"]) for k, r in enumerate(base)])))
    P = np.array([f["ego2global_translation"][:2] for f in fr], float)
    Y = np.array([WC.yaw_of(f["ego2global_rotation"]) for f in fr])
    out = []
    for k, r in enumerate(base):
        i = idx[r["token"]]
        c = ctx.get(k)
        if c is None:
            continue
        R = r["ref"].astype(float)
        fut = r["fut"].astype(float)
        seg = np.hypot(*np.diff(np.r_[np.zeros((1, 2)), fut], axis=0).T) / 0.5
        e = dict(dpsi2=float(wrap(Y[i + 4] - Y[i])), dpsi4=float(wrap(Y[i + 8] - Y[i])), vmin4=float(seg.min()),
                 ln_type=c["ln_type"], ln_d=c["ln_d"], ln_st0=c["ln_st"][:, 0], ln_conn=c["ln_conn"], ln_elig=c["ln_elig"], ln_tstop=c["ln_tstop"])
        l0, g0 = lane_at(M, P[i], Y[i])
        l4, g4 = lane_at(M, P[i + 8], Y[i + 8])
        e |= dict(lane0=l0 or "", lane4=l4 or "", grp0=g0 or "", grp4=g4 or "")
        e["junc"] = bool(any(M["ig"][q].covers(Point(*P[j])) for j in range(i, i + 9) for q in M["itree"].query(Point(*P[j]))))
        if l0:
            C = centreline(M, l0, P[i], Y[i])
            o = signed_lat(C, np.r_[np.zeros((1, 2)), fut[[3, 7]]])
            e |= dict(lane_c=C.astype(np.float32), lane_ok=True, off0=float(o[0]), off2=float(o[1]), off4=float(o[2]))
        else:
            e |= dict(lane_c=np.zeros((NC, 2), np.float32), lane_ok=False, off0=np.nan, off2=np.nan, off4=np.nan)
        # conflicts: pedestrians (any heading) and crossing vehicles (not in path at t0, >= 30 deg off the path, >= 1 m/s), constant velocity
        an = fr[i]["anns"]
        B = np.asarray(an["gt_boxes"], float).reshape(-1, 7)
        nm, vel = np.asarray(an["gt_names"]), np.asarray(an["gt_velocity_3d"], float).reshape(-1, 3)
        tt = np.arange(9) * 0.5
        yp, yv = np.inf, np.inf
        hd = np.arctan2(np.diff(R[:, 1]), np.diff(R[:, 0]))
        for kind, sel, lat_in, s_max in (("p", nm == "pedestrian", 2.0, 30.0), ("v", np.isin(nm, WC.VEH), 1.5, 35.0)):
            for b, v in zip(B[sel], vel[sel]):
                pts = b[None, :2] + tt[:, None] * v[None, :2]
                d, s = WC.lat_s(R, pts)
                if kind == "v":
                    if d[0] <= 1.5 or np.hypot(*v[:2]) < 1.0:
                        continue
                    j = min(int(s[0] / WC.REF_DS), len(hd) - 1)
                    if abs(wrap(np.arctan2(v[1], v[0]) - hd[j])) < np.radians(30):
                        continue
                hit = (d <= lat_in) & (s >= FRONT - 2.0) & (s <= s_max) & (s < r["ref_len"] + 1.0)
                if hit.any():
                    if kind == "p":
                        yp = min(yp, float(s[hit].min()))
                    else:
                        yv = min(yv, float(s[hit].min()))
        e |= dict(yp_s=yp, yv_s=yv)
        out.append(r | e)
    return out


def cells_of(r):
    """Registered cell membership of one frame (prereg section 2) -> dict cell -> bool."""
    v0, vend, a4, dps = r["v0"], r["vend"], r["a4"], np.degrees(abs(r["dpsi4"]))
    mv = 2.0 <= v0 <= 12.0
    ty, dF, st0, conn, tstop = r["ln_type"], r["ln_d"] - FRONT, r["ln_st0"], r["ln_conn"], r["ln_tstop"]
    lines = ty > 0
    q = int(np.argmax(lines)) if lines.any() else -1                         # the first stop line ahead on the hindsight route
    line_ok = q >= 0 and 0.0 <= dF[q] <= 40.0 and r["lead_any"] > dF[q] + 2.0
    any_line60 = bool((lines & np.isfinite(dF) & (dF <= 60.0)).any())
    c = {}
    c["S_L"] = WC.cell_of(r) == "S_L"
    red60 = (q >= 0 and ty[q] == 2 and conn[q] and st0[q] == 1 and 0.0 <= dF[q] <= 60.0 and r["lead_any"] > dF[q] + 2.0)
    c["S_R"] = bool(mv and red60 and a4 < dF[q] + 0.5 and vend <= 0.7 * v0)
    c["S_SS"] = bool(mv and vend < 0.5 and line_ok and ty[q] == 1 and a4 < dF[q] + 0.5)
    c["GO_G"] = bool(v0 >= 3.0 and q >= 0 and ty[q] == 2 and conn[q] and st0[q] == 0 and 5.0 <= dF[q] <= 40.0 and a4 >= dF[q]
                     and vend >= 0.8 * v0 and r["lead_any"] > dF[q] + 10.0)
    c["F"] = WC.cell_of(r) == "F_L"
    noconf = not np.isfinite(r["yp_s"]) and not np.isfinite(r["yv_s"])
    c["C"] = bool(5.0 <= v0 <= 15.0 and abs(vend - v0) <= 1.0 and dps < 10.0 and r["lead_any"] > 40.0 and not any_line60 and noconf and not r["junc"])
    lc = (v0 >= 3.0 and r["lane_ok"] and r["lane4"] != "" and r["lane4"] != r["lane0"] and r["grp4"] == r["grp0"] and abs(r["off0"]) <= 1.5
          and abs(r["off4"]) >= 1.5 and abs(r["off4"]) >= abs(r["off0"]) + 1.0 and not r["junc"] and dps < 20.0)
    c["LC_L"], c["LC_R"] = bool(lc and r["off4"] > 0), bool(lc and r["off4"] < 0)
    tn = v0 >= 2.0 and a4 >= 6.0 and dps >= 30.0 and r["junc"]
    c["T_L"], c["T_R"] = bool(tn and r["dpsi4"] > 0), bool(tn and r["dpsi4"] < 0)
    c["K"] = bool(v0 >= 5.0 and 10.0 <= dps < 30.0 and not r["junc"] and r["lane0"] != "" and r["lane4"] != ""
                  and not (r["lane4"] != r["lane0"] and r["grp4"] == r["grp0"]))
    for cell, s in (("Y_P", r["yp_s"]), ("Y_V", r["yv_s"])):
        blocked = bool((lines & np.isfinite(dF) & (r["ln_d"] <= s + 5.0) & ((ty == 1) | (st0 == 1))).any())
        c[cell] = bool(np.isfinite(s) and v0 >= 1.0 and (vend <= 0.6 * v0 or vend < 0.5) and a4 + FRONT < s and r["lead_any"] > s - FRONT and not blocked)
    return c


def cmd_select(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(NAV / "maps"))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    logs = sorted((NAV / "navsim_logs/test").glob("*.pkl"))
    if a.limit_logs:
        logs = logs[:a.limit_logs]
    with Run("cap_ledger", "select", config=vars(a), seed=SEED) as run:
        res = par.pmap(log_cells, logs, run=run, workers=min(a.workers or n_cpus(), len(logs)), desc="logs")
        res.raise_if_failed()
        rows = [r for v in res.values for r in v]
        for r in rows:
            r["cells"] = cells_of(r)
        run.info(f"{len(rows)} eligible frames, {len(logs)} logs")
        rng = np.random.default_rng(SEED)
        pick, census = {}, {}
        for c in CELLS:
            cand = sorted((r for r in rows if r["cells"][c]), key=lambda r: (r["log"], r["frame"]))
            thin, last = [], {}
            for r in cand:
                if r["frame"] - last.get(r["log"], -99) >= GAP_FR:
                    thin.append(r)
                    last[r["log"]] = r["frame"]
            if len(thin) < SPARSE:                                     # sparse cell: every qualifying frame (log-clustered CI)
                thin = cand
            got = [thin[k] for k in sorted(rng.permutation(len(thin))[:CAP])]
            census[c] = dict(frames=len(cand), thinned=len(thin), rows=len(got), logs=len({r["log"] for r in got}))
            for r in got:
                pick.setdefault(r["token"], r)["in_" + c] = True
        keep = sorted(pick.values(), key=lambda r: (r["log"], r["frame"]))
        for r in keep:
            for c in CELLS:
                r.setdefault("in_" + c, False)
        arr = ("fut", "ref", "lane_c", "ln_type", "ln_d", "ln_st0", "ln_conn", "ln_elig", "ln_tstop")
        cols = [k for k in keep[0] if k not in arr + ("cells",)]
        out = OUT / "select"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "rows.npz", **{k: np.stack([np.asarray(r[k]) for r in keep]) for k in arr}, **{k: np.array([r[k] for r in keep]) for k in cols})
        sc = [k for k in cols if not k.startswith("in_") and np.asarray(rows[0][k]).ndim == 0]       # human-only census table of every eligible frame (scalars + cells + first line)
        q1 = [int(np.argmax(r["ln_type"] > 0)) if (r["ln_type"] > 0).any() else -1 for r in rows]
        np.savez(out / "frames.npz", **{k: np.array([r[k] for r in rows]) for k in sc if not k.startswith("in_")},
                 **{"c_" + c: np.array([r["cells"][c] for r in rows]) for c in CELLS},
                 l1_type=np.array([r["ln_type"][q] if q >= 0 else 0 for r, q in zip(rows, q1)]),
                 l1_d=np.array([r["ln_d"][q] - FRONT if q >= 0 else np.nan for r, q in zip(rows, q1)]),
                 l1_st=np.array([r["ln_st0"][q] if q >= 0 else -9 for r, q in zip(rows, q1)]),
                 l1_conn=np.array([r["ln_conn"][q] if q >= 0 else False for r, q in zip(rows, q1)]),
                 l1_tstop=np.array([r["ln_tstop"][q] if q >= 0 else np.nan for r, q in zip(rows, q1)]))
        (out / "tokens.txt").write_text("\n".join(r["token"] for r in keep) + "\n")
        vx = [float(np.median([r["lead_v"] for r in rows if r["cells"]["F"]] or [np.nan])), float(np.median([r["v0"] for r in rows if r["cells"]["F"]] or [np.nan]))]
        md = [f"eligible frames {len(rows)} in {len(logs)} test logs; unique tokens sampled {len(keep)}; F lead speed median {vx[0]:.2f} vs ego {vx[1]:.2f} m/s", "",
              "| cell | frames | thinned | rows | logs |", "|---|--:|--:|--:|--:|"]
        md += [f"| {c} | {v['frames']} | {v['thinned']} | {v['rows']} | {v['logs']} |" for c, v in census.items()]
        (out / "cells.md").write_text("\n".join(md) + "\n")
        run.summary |= dict(eligible=len(rows), unique=len(keep), census=census)
        print((out / "cells.md").read_text())


def cmd_define(a):
    from jevdrive.data import splits
    z = np.load(OUT / "select/rows.npz")
    splits.define("navsim", SPLIT, sorted(z["token"].tolist()), unit="token", origin="cap_ledger select (experiments/cap_ledger/scripts/led.py): sampled "
                  "behaviour cells of NAVSIM test-log frames, plans/2026-10-16-cap-ledger-prereg.md; rows $DATA_DIR/runs/cap_ledger/select/rows.npz",
                  status="frozen", used_by=["experiments/cap_ledger"], notes="open-loop evaluation rows only; test logs, never trained on")


def cmd_index(a):
    WC.OUT, WC.SPLIT = OUT, SPLIT                                    # wait_cap's index, unchanged, on this topic's rows (unique tokens)
    WC.cmd_index(a)


# ---------------------------------------------------------------- plans (op-train venv, GPU)
def cmd_plans(a):
    import torch
    sys.path[:0] = [str(REPO / p) for p in ("lib", "scripts", "experiments/op_adapt_r2/lib", "experiments/op_parity/scripts", "experiments/route_mem/lib",
                                             "experiments/route_mem/scripts", "experiments/alpasim/lib")]
    import pp_train as T
    import rmem as RM
    import vhist as VH
    from jevdrive.run import Run
    from jevdrive.data import splits
    data = f"idx_{SPLIT}"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    with Run("cap_ledger", "plans", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{SPLIT}"))
        S = T.Store([data], dev, need_side=False, frames="warp")
        z = np.load(DATA / "runs/route_mem/feat" / f"{data}.npz")
        assert np.array_equal(z["names"], S.tab["names"]), "route features rows != cache rows"
        F = z["f"].astype(np.float32)
        F[~z["ok"]] = 0
        F = torch.as_tensor(F, device=dev)
        W = torch.as_tensor(T.R2.t_weights(T.T8), device=dev)
        E0 = S.ego[:, :20].float().cpu().numpy().astype(np.float64)    # rf_lonol.flat_rows: vx kept, vy = ax = ay = 0, past poses re-spaced at vx
        EF = E0.copy()
        EF[:, 5:8] = 0.0
        for i in range(len(EF)):
            P = E0[i, 8:20].reshape(4, 3) * [10.0, 10.0, 1.0]
            EF[i, 8:20] = (VH.respace(P, max(E0[i, 4] * 10.0, 0.0)) / [10.0, 10.0, 1.0]).reshape(12)
        egoF = S.ego.clone()
        egoF[:, :20] = torch.as_tensor(EF.astype(np.float32), device=dev).to(S.ego.dtype)
        keep = [c for c in range(20) if c not in WC_SPEED_COLS]
        assert np.array_equal(egoF[:, keep].cpu().numpy(), S.ego[:, keep].cpu().numpy()), "F row changed a non-speed column"
        run.info(f"rows {S.n}; F rows: mean |vx| {np.abs(EF[:, 4]).mean() * 10:.2f} m/s, history poses moved {np.abs(EF[:, 8:20] - E0[:, 8:20]).max() * 10:.2f} m max")
        for tag in a.tags:
            todo = [c for c in a.conds if a.force or not (OUT / "plans" / c / f"{tag}.npz").exists()]
            if not todo:
                continue
            if tag == "P0":
                model, br = T.load_pmodel("P0", dev), False
            else:
                ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
                model = T.PModel(ck["model"]["arm"]).to(dev).eval()
                br = RM.has_branch(tag)
                if br:
                    sd = json.loads((T.proot("runs", tag) / RM.SIDECAR).read_text())
                    assert not sd.get("reflex") and not sd["head"].get("order") and not sd["head"].get("latfeat"), f"{tag}: unsupported branch variant"
                    RM.attach(model, **{k: v for k, v in sd["head"].items() if k not in ("order", "latfeat")})
                    model.to(dev)
                model.load_state(ck["model"])
                model.eval()
            sl = model.net.slices
            pi = torch.arange(sl["plan"].start, sl["plan"].start + 495, device=dev)
            for cond in todo:
                EG = S.ego if cond == "H" else egoF
                xs, ys, ps = [], [], []
                with torch.no_grad():
                    for i in range(0, S.n, 256):
                        r = torch.arange(i, min(i + 256, S.n), device=dev)
                        e = torch.cat([EG[r].float(), F[r]], 1) if br else EG[r]
                        p = model(S.front[r], e, S.tc[r], None, None, nv=None).float()[:, pi].view(-1, 33, 15)
                        x, y, psi = T.rear(p, S.cam_x[r], W)
                        xs.append(x.cpu()), ys.append(y.cpu()), ps.append(psi.cpu())
                x, y, psi = (torch.cat(v).numpy() for v in (xs, ys, ps))
                fut = S.tab["fut"]
                ade = float(np.nanmean(np.hypot(x - fut[..., 0], y - fut[..., 1])))
                o = OUT / "plans" / cond
                o.mkdir(parents=True, exist_ok=True)
                np.savez(o / f"{tag}.npz", names=S.tab["names"], plan=np.stack([x, y, psi], -1).astype(np.float32), branch=br)
                run.info(f"{tag} {cond}: rows {S.n}, branch {br}, ADE vs log {ade:.3f}")
                run.summary[f"{tag}/{cond}"] = dict(ade=ade, branch=br)
            del model
            torch.cuda.empty_cache()


WC_SPEED_COLS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]   # vx vy ax ay + the 4 history poses (x, y, yaw)


# ---------------------------------------------------------------- ledger (CPU)
def measures(plan, z):
    """plan (n, 8, 3) rear-axle poses on the rows z -> dict of per-row arrays used by the success rules."""
    P = plan[..., :2].astype(float)
    seg = np.hypot(*np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P], 1), axis=1).transpose(2, 0, 1))
    arc = seg.sum(1)
    n = len(P)
    lat, smax, herr, offp4, offp2, sline_ok = (np.full(n, np.nan) for _ in range(6))
    ref = z["ref"].astype(float)
    # human path heading on the reference grid (0.5 m)
    fut = z["fut"].astype(float)
    latf = np.full(n, np.nan)
    for k in range(n):
        d, s = WC.lat_s(ref[k], P[k])
        lat[k], smax[k] = d.max(), s.max()
        Hk = np.r_[np.zeros((1, 2)), fut[k]]                        # the human's own 4 s path, 20 m straight beyond (the reference path stops at 40 m)
        u = Hk[-1] - Hk[-2]
        Hk = np.r_[Hk, Hk[-1:] + 20.0 * u / max(np.hypot(*u), 1e-6)]
        latf[k] = WC.lat_s(Hk, P[k])[0].max()
        R = ref[k]
        h = np.unwrap(np.arctan2(np.diff(R[:, 1]), np.diff(R[:, 0])))
        g = np.arange(len(h)) * WC.REF_DS + WC.REF_DS / 2
        herr[k] = abs(wrap(plan[k, 7, 2] - np.interp(arc[k], g, h)))
        if z["lane_ok"][k]:
            o = signed_lat(z["lane_c"][k].astype(float), P[k][[3, 7]])
            offp2[k], offp4[k] = o
    # the first stop line ahead: rear-axle arc at which the front bumper reaches it
    ty, dF = z["ln_type"], z["ln_d"] - FRONT
    q = np.where((ty > 0).any(1), (ty > 0).argmax(1), 0)
    line_s = np.where((ty > 0).any(1), dF[np.arange(n), q], np.nan)
    # plan's projected arc per pose (monotone envelope) and speed per segment
    s_pose = np.zeros((n, 8))
    for k in range(n):
        s_pose[k] = np.maximum.accumulate(WC.lat_s(ref[k], P[k])[1])
    crossed = s_pose > line_s[:, None] + 0.5
    return dict(arc=arc, lat=lat, latf=latf, smax=smax, herr=np.degrees(herr), offp2=offp2, offp4=offp4, line_s=line_s,
                never_cross=~crossed.any(1))


def success(cell, m, z):
    """The registered 'does what the human does' rule of each cell (prereg section 3) -> bool per row."""
    a4 = z["a4"]
    if cell in ("S_L", "Y_P", "Y_V"):
        return m["arc"] <= a4 + 2.0
    if cell in ("S_R", "S_SS"):
        return m["never_cross"] & (m["arc"] <= a4 + 2.0)
    if cell == "GO_G":
        return m["arc"] >= a4 - 2.0
    if cell == "F":
        return np.abs(m["arc"] - a4) <= np.maximum(2.0, 0.10 * a4)
    if cell == "C":
        return np.abs(m["arc"] / a4 - 1.0) <= 0.10
    if cell in ("LC_L", "LC_R"):
        return (np.sign(m["offp4"]) == np.sign(z["off4"])) & (np.abs(m["offp4"]) >= 0.5 * np.abs(z["off4"]))
    if cell in ("T_L", "T_R"):
        return (m["herr"] <= 15.0) & (m["arc"] >= 2.0)
    if cell == "K":
        return m["latf"] <= 1.0
    raise KeyError(cell)


def secondary(cell, m, z):
    """One continuous companion per cell (reported, not classified)."""
    a4 = z["a4"]
    if cell in ("LC_L", "LC_R"):
        return "offset ratio 4 s", m["offp4"] / z["off4"]
    if cell in ("T_L", "T_R"):
        return "heading err (deg)", m["herr"]
    if cell == "K":
        return "max lat (m)", m["latf"]
    if cell == "C":
        return "arc ratio", m["arc"] / a4
    return "arc - human (m)", m["arc"] - a4


def cmd_ledger(a):
    from jevdrive import stats
    stages = WC.STAGES
    pred = WC.PRED
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(a.rows, allow_pickle=False))
    toks = z["token"].tolist()
    succ = {}                                 # (cell, cond) -> {stage: 2-seed row mean of success}, plus rows mask and logs
    sec = {}
    meta = {}

    def plans_of(d, tag, names):
        p = np.load(Path(d) / f"{tag}.npz")
        pos = {k: i for i, k in enumerate(p["names"].tolist())}
        return p["plan"][np.array([pos[k] for k in names])]

    for cond in ("H", "F"):
        for st, tags in stages:
            M = [measures(plans_of(Path(a.plans) / cond, t, toks), z) for t in tags]
            for c in CELLS:
                msk = z["in_" + c]
                if not msk.any():
                    continue
                zc = {k: v[msk] for k, v in z.items()}
                s = np.mean([success(c, {k: v[msk] for k, v in m.items()}, zc).astype(float) for m in M], 0)
                name, v = secondary(c, {k: v[msk] for k, v in M[0].items()}, zc)
                succ.setdefault((c, cond), {})[st] = s
                sec.setdefault((c, cond), {})[st] = (name, float(np.nanmedian(v)))
                meta[c] = dict(n=int(msk.sum()), logs=z["log"][msk], human_a4=float(np.median(z["a4"][msk])))
    # waiting / launching: decision 326's rows and plans (H only; the history is static there)
    if a.wc_rows:
        w = dict(np.load(a.wc_rows, allow_pickle=False))
        lead = {k: w[f"lead_{k}"] for k in ("x", "y", "yaw", "l", "w")}
        for st, tags in stages:
            R = [WC.classify(plans_of(a.wc_plans, t, w["token"].tolist()), w["ref"], lead, None) for t in tags]
            for c, cl, want in (("W", ("W_L", "W_N"), 0), ("G", ("G_L", "G_N"), 1)):
                msk = np.isin(w["cell"], cl)
                s = np.mean([((r["cls"][msk] == 0) if want == 0 else (r["cls"][msk] != 0)).astype(float) for r in R], 0)
                for cond in ("H", "F"):
                    succ.setdefault((c, cond), {})[st] = s
                    sec.setdefault((c, cond), {})[st] = ("arc4 med (m)", float(np.median(np.mean([r["arc4"][msk] for r in R], 0))))
                meta[c] = dict(n=int(msk.sum()), logs=w["log"][msk], human_a4=float(np.median(w["a4"][msk])))
    cells = [c for c in ("W", "G") + CELLS if c in meta]
    order = [st for st, _ in stages]
    led, L = {}, ["# CAP-LEDGER (2-seed row means; CI = log-clustered bootstrap B 10 000 seed 0)", "",
                  "Primary read: H (the human's history) except the decelerating cells " + ", ".join(DECEL) + " (F = constant-velocity history).", ""]
    for cond in ("primary", "H", "F"):
        L += [f"## read: {cond}", "", "| cell | n / logs | " + " | ".join(order) + " |", "|---|--:|" + "--:|" * len(order)]
        for c in cells:
            cd = ("F" if c in DECEL else "H") if cond == "primary" else cond
            S, g = succ[(c, cd)], meta[c]["logs"]
            L.append(f"| {c} | {meta[c]['n']} / {len(set(g.tolist()))} | " + " | ".join(f"{S[st].mean():.3f}" for st in order) + " |")
        L.append("")
    # classification on the primary read
    L += ["## classification (primary read; stage vs P0, paired)", "",
          "| cell | P0 | stage | rate | vs P0 | class | vs predecessor |", "|---|--:|---|--:|--:|---|--:|"]
    for c in cells:
        cd = "F" if c in DECEL else "H"
        S, g = succ[(c, cd)], meta[c]["logs"]
        p0 = stats.bootstrap(S["P0"], groups=g)
        p0s = "good" if p0["mean"] >= 0.60 else "poor" if p0["mean"] < 0.40 else "partial"
        judged = meta[c]["n"] >= MIN_ROWS and len(set(g.tolist())) >= MIN_LOGS
        led[c] = dict(read=cd, n=meta[c]["n"], logs=len(set(g.tolist())), judged=judged, P0=p0["mean"], P0_status=p0s, stages={})
        first_loss = None
        for st in order[1:]:
            d = stats.paired(S[st], S["P0"], groups=g)
            dp = stats.paired(S[st], S[pred[st]], groups=g)
            if d["mean"] <= -0.15 and d["hi"] < 0:
                cls = "LOST" if p0s == "good" else "DEGRADED"
            elif d["mean"] >= 0.15 and d["lo"] > 0:
                cls = "GAINED"
            elif p0s == "poor" and S[st].mean() < 0.40:
                cls = "NEVER"
            else:
                cls = "UNCHANGED" + (" (eroded)" if d["mean"] <= -0.05 and d["hi"] < 0 else "")
            if not judged:
                cls = "(too few: " + cls + ")"
            step_loss = dp["mean"] <= -0.10 and dp["hi"] < 0
            if cls in ("LOST", "DEGRADED") and first_loss is None and step_loss:
                first_loss = st
            led[c]["stages"][st] = dict(rate=float(S[st].mean()), d=d["mean"], lo=d["lo"], hi=d["hi"], cls=cls, dp=dp["mean"], dp_lo=dp["lo"],
                                        dp_hi=dp["hi"], step_loss=bool(step_loss), sec=sec[(c, cd)][st])
            L.append(f"| {c} | {stats.fmt(p0)} {p0s} | {st} | {S[st].mean():.3f} | {stats.fmt(d)} | {cls} | {stats.fmt(dp)} ({pred[st]}) |")
        led[c]["lost_at"] = first_loss
        L.append("")
    # companion read (the other history), the same rules, classes only
    L += ["## companion read (the other history condition), classes vs P0", "", "| cell | read | " + " | ".join(order[1:]) + " |",
          "|---|---|" + "---|" * (len(order) - 1)]
    for c in cells:
        cd = "H" if c in DECEL else "F"
        if c in ("W", "G"):
            continue
        S, g = succ[(c, cd)], meta[c]["logs"]
        p0g = S["P0"].mean() >= 0.60
        row = []
        for st in order[1:]:
            d = stats.paired(S[st], S["P0"], groups=g)
            row.append(f"{S[st].mean():.2f} " + ("LOST" if (d["mean"] <= -0.15 and d["hi"] < 0 and p0g) else "DEGR" if (d["mean"] <= -0.15 and d["hi"] < 0)
                                                 else "GAIN" if (d["mean"] >= 0.15 and d["lo"] > 0) else "-"))
        L.append(f"| {c} | {cd} (P0 {S['P0'].mean():.2f}) | " + " | ".join(row) + " |")
    L += ["", "## secondary (median of the cell's continuous companion, seed 0)", "", "| cell | quantity | " + " | ".join(order) + " |",
          "|---|---|" + "--:|" * len(order)]
    for c in cells:
        cd = "F" if c in DECEL else "H"
        s = sec[(c, cd)]
        L.append(f"| {c} | {s['P0'][0]} | " + " | ".join(f"{s[st][1]:.2f}" for st in order) + f" |  (human arc4 median {meta[c]['human_a4']:.1f} m)")
    # registered verdict (prereg section 5)
    lost = {gname: sorted({st for c in cs if c in led for st, v in led[c]["stages"].items() if v["cls"] in ("LOST", "DEGRADED")})
            for gname, cs in GROUPS.items()}
    lost = {k: v for k, v in lost.items() if v}
    base_lost = [k for k, v in lost.items() if k not in PATCH and set(v) & {"P2-F", "RB0-F"}]
    later_only = [k for k, v in lost.items() if k not in PATCH and not set(v) & {"P2-F", "RB0-F"}]
    verdict = ("REBUILD supported" if len(base_lost) >= 2 else "PATCH-PLUS (one more capability lost at the base)" if len(base_lost) == 1
               else "PATCH suffices")
    L += ["", "## verdict (prereg section 5)", "", f"- groups with a LOST / DEGRADED stage: {json.dumps(lost)}",
          f"- outside the wait / stop patch scope, lost at a base stage (P2-F or RB0-F): {base_lost}; lost only at later stages: {later_only}",
          f"- first stage of the loss per cell: {json.dumps({c: v['lost_at'] for c, v in led.items() if v['lost_at']})}", f"- **{verdict}**"]
    (out / "ledger.md").write_text("\n".join(L) + "\n")
    (out / "ledger.json").write_text(json.dumps(dict(ledger=led, lost=lost, base_lost=base_lost, later_only=later_only, verdict=verdict), indent=1, default=float))
    print((out / "ledger.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit-logs", type=int, default=0)
    sp.add_parser("define")
    sp.add_parser("index")
    p = sp.add_parser("plans")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--conds", nargs="+", default=["H", "F"])
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("ledger")
    p.add_argument("--rows", required=True)
    p.add_argument("--plans", required=True)
    p.add_argument("--wc-rows", default="")
    p.add_argument("--wc-plans", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "define": cmd_define, "index": cmd_index, "plans": cmd_plans, "ledger": cmd_ledger}[a.cmd](a)
