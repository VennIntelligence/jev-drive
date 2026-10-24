#!/usr/bin/env python3
"""ROUND9 NTV1 S-PAI (plans/2026-10-24-ntv1-spai.md = the registration; lane scripts/spai.sh): S rows on the student's own PAI states.
Student D3 = NT4D3-F-s0 (step 0 of the plan: D3's 2-seed PAI 281 mean 0.859 >= D1's 0.852). D3 is rolled on the 719 curated train scenes with
step 2a's serve; round 6's dag.sh builds its DAgger rows (pool on1_PAIDC3, iteration 3: kept, not used here). The same states, 2 Hz, become
hinge-only `self` rows (part `ps`) of on1_JTDC2S = on1_JTDC2's parts + ps, with a road_pai raster whose window is stretched to x -8..152 m
(lib/drivable_hinge `geom`). Box, repo root.

  raster --chunk C   (AlpaSim venv)  every row-file scene of the chunk: the 2 Hz candidate rows (k % 5 = crc32(scene) % 5, before the scoring
                     event and the log end) + the G2 decision (the last decision >= 2.5 s before an offroad event); the evaluator-rule surface
                     (road_pai.pai_raster's rule) on the stretched canvas -> $O/spai/parts/<scene>.npz (+ human path, model plan, scene facts)
  index              (op-train)  parts -> $O/spai/sdf.npy + road_spai.npz (Hinge layout + geom; tokens = the merged names on:ps:paidc3:..),
                     fallbacks (a) t0 off, (b) human path < -0.5 m, (c) no surface, (d) human path triggers; per-row road class, speed;
                     gates G1-G3 -> results/g13.json (exit 2 on a failed line)
  weights            NTV1D2S (S 32 = 16 rb0 + 16 ps, ps by road class and the standing cap; DAgger 8 as NTV1D2) and NTV1D2SZ (NTV1D2 + ps at 0);
                     G4 -> results/g4.json, w_*.json (exit 2 on a failed line)
  wcheck --out J     both draw files against NTV1D2 bit for bit where they must be, shares, held logs, modes
  argv ARM TAG [--steps N] [--on-start T] [--save-at S|none] [--mix]   one token a line; ARM: D3 (dg4.py's = stored), SP (the arm), SPZ (D3 on
                     on1_JTDC2S / NTV1D2SZ + the raster: the weight-0 pair)
  diff --out J       D3's built argv == its stored argv; SP differs only in --tag / --on / --on-wfile + the appended --on-road-pai FILE
  ident --tag T --ref R --upto N --stop M --out J -- CMD   pcl.py ident; the eval-mode held / fit slice scalars (pool_held/*, pool_fit/*) are
                     compared apart (their row sample depends on the pool's `self` rows) and reported, not gated
  pair --a A --b B --upto N --out J [--strict]       two logged runs: every scalar equal (pool_held / pool_fit apart unless --strict); pool rows drawn
  smoke --tag T --on-start T0 --out J              dgc.py smoke's checks against w_NTV1D2S.json + the raster rows the trainer took (= the index)
  build              build summary (pool, rows, rasters, gates, disk) -> results/build.json
  bs --out DIR --tags T,..                         behavior_stats (dgt.py bs)
  verdict --dir O --out P                          PAI 281 (new + old), val160, classes, flips, highway, lines, cost -> P.{md,json}
  final arm (plans/2026-10-24-ntv1-final.md; spai.sh final with SPAI_SEED):
  fdiff --tag T --seed S --out J                   S-PAI-s0's built argv == stored; T's differs only in --tag / --seed; pool / draw / raster files untouched since
                                                   S-PAI-s0's training started (mtime) + their sha256
  fverdict --dir O --tag T --out P                 S-PAI 2-seed mean (s0 + T) on PAI 281 + val160 vs R8K-s2, INTP75-NR, D3 2-seed mean; per seed; classes, flips,
                                                   highway, gates, cost -> P.{md,json}
"""
import argparse
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pcl as C  # noqa: E402

DD, REPO = C.DD, C.REPO
O = Path(os.environ.get("SPAI_O") or DD / "runs/round9/ntv1_spai")      # SPAI_O / SPAI_NP / SPAI_TAG: a smoke on another roll
SP_ = O / "spai"
O4 = DD / "runs/round9/ntv1_step4"
O3, OV = DD / "runs/round9/ntv1_step3", DD / "runs/round9/val160"
RES = REPO / "experiments/round9/results/ntv1_spai"
CR, F_ = DD / "runs/op_parity/cache", DD / "runs/pai_fix/facts"
D3, D3S1, D1, D1S1 = "NT4D3-F-s0", "NT4D3-F-s1", "NT3D1-F-s0", "NT3D1-F-s1"
SP = "NT5SP-F-s0"
NP3, POOL2, POOLS = os.environ.get("SPAI_NP", "on1_PAIDC3"), "on1_JTDC2", "on1_JTDC2S"
W2, WREL = "runs/round9/ntv1_step4/w/NTV1D2.npz", "runs/round9/ntv1_spai/w"
RIDX = "runs/round9/ntv1_spai/spai/road_spai.npz"
# registered (plan sections 2-4)
GEOM = (-8.0, -24.0, 1.25, 0.5)                          # x0, y0, cell x, cell y: x -8..152 m (128 x 1.25), y -24..24 m (96 x 0.5)
NH, NW, FINE, PAD = 128, 96, 0.25, 8.0
CX0, CY0 = GEOM[0] - PAD, GEOM[1] - PAD
NX, NY = int((NH * GEOM[2] + 2 * PAD) / FINE), int((NW * GEOM[3] + 2 * PAD) / FINE)
BX, BY = int(GEOM[2] / FINE), int(GEOM[3] / FINE)       # fine cells per stored cell (5 x 2)
BANDS = ((0, 320, 0), (256, 576, 288), (512, 704, 544))      # fine-canvas rows (x = -16 + 0.25 r): (lo, hi, rows from here on taken from this band)
EDGE_R = MAP_R = 170.0                                   # road edges / lanes within reach of the stretched canvas
EDGE_DZ, MARGIN, B_OUT = 3.0, 0.3, -0.5
HZ = 5                                                   # 10 Hz decisions -> 2 Hz: k % 5 == crc32(scene) % 5
G2_DT = 2.5e6                                            # us before the offroad event
CLS_W = dict(highway=1.42, arterial=1.13, urban=0.80, ramp=0.82)   # pool design (6): exam share / train share by distance, capped [0.5, 1.5]
W_CAP = (0.5, 1.5)
V_STAND, STAND_MAX, V_FAST = 0.5, 0.05, 13.0
HOLD_PCT = 15                                            # the recipe's --on-hold-pct: crc32(log) % 100 < 15 over the `self` rows' logs
K_ON, K_DAG, K_PS = 32, 8, 16
LINE_G1, LINE_G2, LINE_G3, LINE_G4 = 0.07, 0.60, 0.10, 0.30
LINE = 0.015
MONI = re.compile(r"^pool_(held|fit)/")                  # eval-mode held / fit slice scalars (monitoring; sample depends on the self rows)


def phase(scene):
    return zlib.crc32(scene.encode()) % HZ


# ---------------------------------------------------------------- raster (AlpaSim venv)
def _ring(xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(xy)[:, :2] - pose[:2]
    x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return np.round(np.stack([(y - CY0) / FINE - 0.5, (x - CX0) / FINE - 0.5], -1) * 16).astype(np.int32)


def _fill(u, pose):
    import cv2
    img = np.zeros((NX, NY), np.uint8)
    if u is not None:
        for g in (u.geoms if hasattr(u, "geoms") else [u]):
            cv2.fillPoly(img, [_ring(g.exterior.coords, pose)], 1, shift=4)
            for h in g.interiors:
                cv2.fillPoly(img, [_ring(h.coords, pose)], 0, shift=4)
    return img


def _sdf(ins):
    """opb_labels.mask_sdf on the stretched canvas: EDT both sides at 0.25 m, block mean 5 x 2 -> (128, 96) at 1.25 x 0.5 m."""
    from scipy.ndimage import distance_transform_edt as edt
    if not ins.any():
        return np.full((NH, NW), -20.0, np.float32)
    d = edt(ins) - edt(~ins)
    d = np.where(ins, d - 0.5, d + 0.5) * FINE
    p = int(PAD / FINE)
    return d[p:NX - p, p:NY - p].reshape(NH, BX, NW, BY).mean((1, 3)).astype(np.float32)


def raster(g, pose):
    """road_pai.pai_raster's rule (the evaluator's PAI offroad surface: the box centre's component among cells no same-level road edge
    crosses, plus the lane union) on the stretched canvas -> (NH, NW) float16, inside share, edges kept. The component is grown in 80 m bands
    (BANDS, each the stock canvas's length): on one 176 m canvas it escapes through any gap in the road edges (a junction 140 m ahead made
    a narrow road 99 % `inside` in the smoke), the stock canvas closed such strips at its border."""
    import cv2
    from scipy.ndimage import label
    P, pid, tr = g["P"], g["pid"], g["traj"]
    z_ego = tr[np.argmin(((tr[:, :2] - pose[:2]) ** 2).sum(1)), 2] + g["zc"]
    bar, keep = np.zeros((NX, NY), np.uint8), []
    d2 = ((P[:, :2] - pose[:2]) ** 2).sum(1)
    near = d2 < EDGE_R ** 2
    if near.any():
        q, dq, zq = pid[near], d2[near], P[near, 2]
        o = np.lexsort((dq, q))
        first = o[np.r_[True, np.diff(q[o]) != 0]]
        keep = q[first][~(np.abs(zq[first] - z_ego) > EDGE_DZ)].tolist()
        if keep:
            cv2.polylines(bar, [_ring(g["E"][i], pose) for i in keep], False, 1, shift=4)
    free = bar == 0
    lab, _ = label(free[:BANDS[0][1]])                           # band 0 = the stock road_pai canvas (x -16..64): its rule unchanged
    i0, j0 = int((g["xc"] - CX0) / FINE), int(-CY0 / FINE)
    w = lab[i0 - 8:i0 + 9, j0 - 8:j0 + 9]
    ii, jj = np.nonzero(w)
    seed = w[ii, jj][np.argmin((ii - 8) ** 2 + (jj - 8) ** 2)] if len(ii) else 0
    comp = (lab == seed) if seed else np.zeros(lab.shape, bool)
    ins = np.zeros_like(free)
    ins[:BANDS[0][1]] = comp
    for (plo, phi, _), (lo, hi, take) in zip(BANDS, BANDS[1:]):  # each further 80 m band: the components its overlap with the previous
        lab, _ = label(free[lo:hi])                               # band's component touches (no leak through gaps outside the band)
        ids = np.unique(lab[:phi - lo][comp[lo - plo:]])
        comp = np.isin(lab, ids[ids > 0])
        ins[take:hi] = comp[take - lo:]
    if g["U"] is not None:
        ins |= _fill(g["U"], pose) > 0
    sdf = np.full((NH, NW), 20.0, np.float32) if ins.all() else np.clip(_sdf(ins), -20, 20)
    return sdf.astype(np.float16), float(ins.mean()), len(keep)


def scene_geom(scene, cfg, xy):
    """road_pai.scene_geom with the lane union taken MAP_R around the anchors (the canvas reaches 152 m ahead)."""
    import rf_tax as RT
    from trajdata.maps.vec_map_elements import MapElementType as MT
    ds = RT.loader(cfg).get_data_source(scene)
    vm, vc = ds.map, ds.rig.vehicle_config
    tr = np.asarray(ds.rig.trajectory.positions, float)
    g = dict(U=None, E=[], traj=np.c_[tr[:, :2], tr[:, 2] if tr.shape[1] > 2 else np.full(len(tr), np.nan)],
             xc=float(vc.aabb_x_offset_m + vc.aabb_x_m / 2), zc=float(vc.aabb_z_offset_m + vc.aabb_z_m / 2))
    if vm is not None:
        for e in vm.elements.get(MT.ROAD_EDGE, {}).values():
            q = np.asarray(e.polyline.points, float)
            if len(q) > 1:
                g["E"].append(np.c_[q[:, :2], q[:, 2] if q.shape[1] > 2 else np.full(len(q), np.nan)])
        if len(vm.lanes):
            U = RT.Map(vm, xy, rad=MAP_R).U
            g["U"] = None if U.is_empty else U
    g["P"] = np.concatenate(g["E"]) if g["E"] else np.zeros((0, 3))
    g["pid"] = np.repeat(np.arange(len(g["E"])), [len(e) for e in g["E"]])
    return g


def _raster_one(job):
    scene, cfg, rowf, part = job
    z = np.load(rowf)
    k, t0, ae, aend = z["k"].astype(np.int64), z["t0_us"].astype(np.int64), z["after_event"].astype(bool), z["after_end"].astype(bool)
    live = ~ae & ~aend
    s2 = live & (k % HZ == phase(scene))
    zk, tev = z["zero_kind"].astype(bool), z["t_event_us"].astype(np.int64)
    g2 = np.zeros(len(k), bool)
    if zk[1]:                                                    # offroad event: the last decision at least 2.5 s before it
        c = np.flatnonzero(t0 <= tev[1] - G2_DT)
        if len(c):
            g2[c[-1]] = True
    sel = np.flatnonzero(s2 | g2)
    A = z["anchor"][sel].astype(np.float64)
    t = time.time()
    g = scene_geom(scene, cfg, A[:, :2]) if len(sel) else None
    out = [raster(g, p) for p in A] if len(sel) else []
    sdf = np.stack([o[0] for o in out]) if out else np.zeros((0, NH, NW), np.float16)
    empty = g is None or (g["U"] is None and not g["E"])
    tmp = part.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, k=k[sel].astype(np.int32), anchor=A, sdf=sdf, frac=np.array([o[1] for o in out], np.float32),
                        n_edges=np.array([o[2] for o in out], np.int32), empty=empty, s2=s2[sel], g2=g2[sel], v0=z["v0"][sel], t0=t0[sel],
                        log0=z["log0"][sel], log_fut=z["log_fut"][sel], log_fut_ok=z["log_fut_ok"][sel], plan=z["poses_model"][sel],
                        served=z["poses"][sel], score=z["score"], zero_kind=zk, t_event=tev, tag=z["tag"], log_xyyaw=z["log_xyyaw"], log_s=z["log_s"])
    os.replace(tmp, part)
    return dict(scene=scene, rows=len(k), live=int(live.sum()), sel=len(sel), s2=int(s2.sum()), g2=int(g2.sum()), sec=time.time() - t)


def cmd_raster(a):
    sys.path[:0] = [str(REPO / "experiments/round5/scripts"), str(REPO / "experiments/route_follow/scripts")]
    import road_pai as RP
    from jevdrive import par
    from jevdrive.common import n_cpus
    rows = sorted(Path(a.rows).glob("clipgt-*.npz"))
    cfg = RP.scene_cfgs(a.cfg_glob)
    miss = [p.stem for p in rows if p.stem not in cfg]
    assert not miss, f"no config for {miss[:3]}"
    (SP_ / "parts").mkdir(parents=True, exist_ok=True)
    jobs = [(p.stem, cfg[p.stem], p, SP_ / "parts" / f"{p.stem}.npz") for p in rows]
    res = par.pmap(_raster_one, jobs, workers=a.workers or n_cpus(), desc=f"raster {a.chunk}")
    res.raise_if_failed()
    st = [v for v in res.values if v]
    rec = dict(chunk=a.chunk, scenes=len(st), rows=sum(v["rows"] for v in st), sel=sum(v["sel"] for v in st), s2=sum(v["s2"] for v in st),
               g2=sum(v["g2"] for v in st), sec_med=float(np.median([v["sec"] for v in st])) if st else None)
    (SP_ / "chunks").mkdir(exist_ok=True)
    C.dump(SP_ / "chunks" / f"{a.chunk}.json", rec)
    print(json.dumps(rec))


# ---------------------------------------------------------------- index (op-train)
def path_m(H, P9, rows, bs=4096):
    """(n, 9, 3) poses [start, 0.5 .. 4 s] in the row frame -> (n, 41, 4) SDF of the 4 footprint corners at 0 .. 4 s (H's own sampling)."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            P0 = torch.as_tensor(np.asarray(P9[i:i + bs], np.float32))
            r = torch.as_tensor(rows[i:i + bs])
            d = torch.einsum("kj,bjc->bkc", H.M, P0)
            Cn = H.C[H.src[r]]
            c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
            cx = d[..., :1] + c * Cn[:, None, :, 0] - s * Cn[:, None, :, 1]
            cy = d[..., 1:2] + s * Cn[:, None, :, 0] + c * Cn[:, None, :, 1]
            xy = torch.stack([cx, cy], -1).reshape(len(P0), -1, 2)
            out.append(H._sample(xy, r).reshape(len(P0), 41, 4).numpy())
    return np.concatenate(out) if out else np.zeros((0, 41, 4), np.float32)


def stations(L):
    """behavior_stats.scene_facts' 1 m stations along the human log (rf_tax.dedupe, arc, interp; the row file's log_xyyaw is maps_train's log P,
    same frame) -> (m, 2), in the order of stations_train.parquet's rows of the scene."""
    P = np.asarray(L[:, :2], float)
    P = P[np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-3]]
    if len(P) < 2:
        P = np.r_[P[:1], P[:1] + [[0.01, 0]]]
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    sg = np.arange(0, S[-1] + 1e-6, 1.0) if S[-1] >= 1 else np.array([0.0])
    return np.c_[np.interp(sg, S, P[:, 0]), np.interp(sg, S, P[:, 1])]


def ffill(P9, ok8):
    """(n, 9, 3) [start, 8 poses] with ok8 (n, 8): poses after the first missing one take the last valid pose (0-weight terms stay finite)."""
    P = P9.copy()
    v = np.cumprod(ok8, 1).astype(bool)
    for j in range(1, 9):
        P[:, j] = np.where(v[:, j - 1:j], P[:, j], P[:, j - 1])
    return P, v.sum(1) * 0.5


def cmd_index(a):
    import pandas as pd
    import torch
    sys.path.insert(0, str(REPO / "lib"))
    from drivable_hinge import Hinge
    from jevdrive.common import n_cpus
    torch.set_num_threads(n_cpus())
    tb = np.load(CR / NP3 / "tab.npz")
    pn = tb["names"].astype(str)
    pk = {(n.split(":")[-2], int(n.split(":")[-1])): i for i, n in enumerate(pn)}
    parts = sorted((SP_ / "parts").glob("clipgt-*.npz"))
    ST = pd.read_parquet(DD / "runs/round9/pool_design/stations_train.parquet")
    stc = {s: g.cls.to_numpy() for s, g in ST.groupby("scene", sort=False)}
    K = dict(scene=[], k=[], s2=[], g2=[], v0=[], plan=[], hum=[], hok=[], score=[], cls=[], empty=[], frac=[], tag=[], zoff=[])
    sdf = np.lib.format.open_memmap(SP_ / "sdf.tmp.npy", "w+", np.float16, (sum(len(np.load(p)["k"]) for p in parts), NH, NW))
    o, st_match = 0, []
    for p in parts:
        with np.load(p) as z:
            m = len(z["k"])
            sdf[o:o + m] = z["sdf"]
            o += m
            s = p.stem
            Q = stations(z["log_xyyaw"])
            cl = stc.get(s)
            st_match.append(cl is not None and len(cl) == len(Q))
            if cl is not None and len(cl) == len(Q):
                j = np.argmin(((z["anchor"][:, None, :2] - Q[None]) ** 2).sum(-1), 1)
                K["cls"] += cl[j].tolist()
            else:
                K["cls"] += ["unknown"] * m
            K["scene"] += [s] * m
            K["k"].append(z["k"]), K["s2"].append(z["s2"]), K["g2"].append(z["g2"]), K["v0"].append(z["v0"]), K["frac"].append(z["frac"])
            K["plan"].append(np.concatenate([np.zeros((m, 1, 3), np.float32), z["plan"]], 1))
            K["hum"].append(np.concatenate([z["log0"][:, None], z["log_fut"]], 1).astype(np.float32))
            K["hok"].append(z["log_fut_ok"])
            K["score"] += [float(z["score"])] * m
            K["empty"] += [bool(z["empty"])] * m
            K["tag"] += [str(z["tag"])] * m
            K["zoff"] += [bool(z["zero_kind"][1])] * m
    sdf.flush()
    del sdf
    os.replace(SP_ / "sdf.tmp.npy", SP_ / "sdf.npy")
    K = {q: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)) for q, v in K.items()}
    tg = os.environ.get("SPAI_TAG", D3)
    assert set(K["tag"].tolist()) == {tg}, f"row files not all {tg}'s: {set(K['tag'].tolist())}"
    n = len(K["k"])
    ip = np.array([pk.get((s, int(k)), -1) for s, k in zip(K["scene"], K["k"])])
    inpool = ip >= 0
    tok = np.array([f"on:ps:{pn[i][3:]}" if i >= 0 else f"on:ps:nopool:{s}:{k}" for i, s, k in zip(ip, K["scene"], K["k"])])
    meta = dict(sdf_file="sdf.npy", geom=np.array(GEOM), layers=np.array(["NUREC_ROAD_EDGE_SIDE", "NUREC_LANE", "NUREC_ROAD_AREA"]), margin=MARGIN)
    f = SP_ / "road_spai.npz"
    np.savez(f, tokens=tok, ok=~K["empty"], ix=np.arange(n), **meta)            # provisional: every raster with a surface
    H = Hinge(f, tok, "cpu", MARGIN)
    rows = np.arange(n)
    mp = path_m(H, K["plan"], rows)                                              # the model's own plan (poses_model), from the row origin
    hum, t_last = ffill(np.nan_to_num(K["hum"]), K["hok"])
    mh = path_m(H, hum, rows)                                                    # the human's logged path, from the human's pose
    hv = np.arange(41)[None] * 0.1 <= t_last[:, None] + 1e-9                     # dense steps the human log covers
    h_avail = (t_last >= 1.0) & np.isfinite(K["hum"][:, 0]).all(1)
    m_t0 = mp[:, 0].min(1)
    m_h = np.where(hv[..., None], mh, np.inf).min((1, 2))
    m_p = mp.min((1, 2))
    beyond56 = float((np.abs(K["plan"][:, 1:, 0]) > 56).any(1).mean())
    no_s = K["empty"]
    ra = ~no_s & (m_t0 < 0)
    rb = ~no_s & ~ra & h_avail & (m_h < B_OUT)
    okp = ~no_s & ~ra & ~rb
    trig_h = okp & h_avail & (m_h < MARGIN)
    ok = okp & h_avail & ~trig_h                                                 # fallback (d) (registered, d418): the human path triggers -> no hinge
    why = np.select([no_s, ra, rb, okp & ~h_avail, trig_h], [3, 1, 2, 5, 4], 0).astype(np.int8)   # ok t0_off label_out no_surface human_trig no_human
    hinge_raw = m_p < MARGIN                                                     # plan hinge active (includes the t0 footprint)
    cand = K["s2"] & inpool
    np.savez(f, tokens=tok, ok=ok & cand, ok_primary=okp, ix=np.arange(n), m_t0=m_t0, m_h=m_h, m_plan=m_p, why=why, **meta)
    Hc = Hinge(f, tok, "cpu", MARGIN, bank=True)                                 # read back through the trainer's loader
    rr = np.flatnonzero(Hc.ok.numpy())[:20000]
    rb_ok = bool(np.array_equal(path_m(Hc, K["plan"][rr], rr).min((1, 2)), m_p[rr]))
    g2r = K["g2"]
    full = K["s2"] & inpool & (K["score"] >= 0.98)
    g1 = float(trig_h[okp & h_avail & cand].mean())
    g2_raw, g2_tr = float(hinge_raw[g2r].mean()) if g2r.any() else None, float((hinge_raw & ok)[g2r].mean()) if g2r.any() else None
    g3_raw, g3_tr = float(hinge_raw[full].mean()), float((hinge_raw & ok)[full].mean())
    c = cand
    by_cls = {q: dict(rows=int((K["cls"][c] == q).sum()), ok=float(ok[c][K["cls"][c] == q].mean()) if (K["cls"][c] == q).any() else None)
              for q in sorted(set(K["cls"][c].tolist()))}
    spd = {f"{lo}-{hi}": dict(rows=int(m.sum()), ok=float(ok[c][m].mean()) if m.any() else None, trig_h=float(trig_h[c][m].mean()) if m.any() else None,
                              plan_active_of_ok=float(hinge_raw[c][m & ok[c]].mean()) if (m & ok[c]).any() else None)
           for lo, hi in ((0, 0.5), (0.5, 3), (3, 8), (8, 13), (13, 20), (20, 99)) for m in [(K["v0"][c] >= lo) & (K["v0"][c] < hi)]}
    rec = dict(rasters=n, rows_in_pool=int(inpool.sum()), candidates=int(c.sum()), pool_rows=len(pn), g2_rows=int(g2r.sum()),
               g2_scenes=int(len(set(K["scene"][g2r].tolist()))), offroad_scenes=int(len(set(K["scene"][K["zoff"]].tolist()))),
               full_rows=int(full.sum()), station_match=float(np.mean(st_match)),
               fallback={w: float((why[c] == i).mean()) for i, w in enumerate(("ok", "t0_off", "label_out", "no_surface", "human_trig", "no_human"))},
               ok_share=float(ok[c].mean()), human_avail=float(h_avail[c].mean()), plan_beyond_56m=beyond56,
               G1=dict(line=LINE_G1, value=g1, pass_=g1 <= LINE_G1, what="human log path triggers the road hinge (margin 0.3), of primary-ok candidates"),
               G2=dict(line=LINE_G2, value=g2_raw, as_trained=g2_tr, pass_=g2_raw is not None and g2_raw >= LINE_G2,
                       what="the model plan's road hinge active at the last decision >= 2.5 s before each offroad event (raw; as trained reported)"),
               G3=dict(line=LINE_G3, value=g3_raw, as_trained=g3_tr, pass_=g3_raw <= LINE_G3,
                       what="the model plan's road hinge active on 2 Hz decisions of scenes the roll scored >= 0.98 (raw; as trained reported)"),
               by_cls=by_cls, by_speed=spd, readback_equal=rb_ok, file=str(f), sdf_gib=(SP_ / "sdf.npy").stat().st_size / 2 ** 30)
    np.savez(SP_ / "rows.npz", tokens=tok, scene=K["scene"], k=K["k"], v0=K["v0"], cls=K["cls"], cand=cand, ok=ok, g2=g2r, full=full,
             hinge_raw=hinge_raw, trig_h=trig_h, score=K["score"])
    rec["pass_"] = bool(rec["G1"]["pass_"] and rec["G2"]["pass_"] and rec["G3"]["pass_"] and rb_ok and rec["station_match"] >= 0.95)
    (O / "results").mkdir(parents=True, exist_ok=True)
    C.dump(O / "results/g13.json", rec)
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if rec["pass_"] else 2)


# ---------------------------------------------------------------- weights
def cmd_weights(a):
    W = np.load(DD / W2)
    tb = np.load(CR / POOLS / "tab.npz")
    names, part, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["log"].astype(str)
    n2 = len(W["names"])
    assert np.array_equal(names[:n2], W["names"].astype(str)), f"{POOLS}: prefix != NTV1D2's rows (= {POOL2})"
    assert set(part[n2:].tolist()) == {"ps"} and (tb["mode"].astype(str)[n2:] == "self").all(), "tail: part ps, mode self"
    R = np.load(SP_ / "rows.npz")
    at = {t: i for i, t in enumerate(names[n2:].tolist())}
    ci = np.flatnonzero(R["cand"])
    pos = n2 + np.array([at[t] for t in R["tokens"][ci]])
    sl = sorted(set(logs[tb["mode"].astype(str) == "self"].tolist()))
    held = {g for g in sl if zlib.crc32(g.encode()) % 100 < HOLD_PCT}
    keep = ~np.isin(logs[pos], list(held))
    pos, ci = pos[keep], ci[keep]
    cls, v0 = R["cls"][ci].astype(str), R["v0"][ci].astype(np.float64)
    cw = np.clip(np.array([CLS_W.get(c, 1.0) for c in cls]), *W_CAP)
    st = v0 < V_STAND
    sh = cw[st].sum() / cw.sum()
    if sh > STAND_MAX:                                           # standing rows down to STAND_MAX of the ps draws (d377: shape-only hinges push them sideways)
        cw[st] *= STAND_MAX * (1 - sh) / (sh * (1 - STAND_MAX))
    w1 = W["w"].astype(np.float64)
    s_part = W["grp"].astype(str) == "S"                         # NTV1D2's S group = the 13 793 live rb0 rows
    assert abs(w1[s_part].sum() - K_ON / (K_ON + K_DAG)) < 1e-9 and set(tb["pool"].astype(str)[:n2][s_part & (w1 > 0)].tolist()) == {"rb0"}, \
        "NTV1D2: S group = rb0 rows, share 0.8"
    w = np.zeros(len(names))
    w[:n2] = np.where(s_part, w1 * (K_ON - K_PS) / K_ON, w1)      # rb0 16 / 40 (scaled), DAgger 8 / 40 as NTV1D2
    w[pos] = cw / cw.sum() * K_PS / (K_ON + K_DAG)
    wz = np.concatenate([w1, np.zeros(len(names) - n2)])
    grp = np.concatenate([W["grp"].astype(str), np.full(len(names) - n2, "ps")])
    assert abs(w.sum() - 1) < 1e-9 and abs(wz.sum() - 1) < 1e-9
    (O / "w").mkdir(parents=True, exist_ok=True)
    for nm, ww in (("NTV1D2SZ", wz), ("NTV1D2S", w)):
        np.savez(O / "w" / f"{nm}.npz", names=names, w=ww.astype(w1.dtype) if nm == "NTV1D2SZ" else ww, grp=grp)
        rec = dict(base=W2, pool=POOLS, rows=len(names), prefix_rows=n2, ps_rows=int(len(names) - n2), ps_drawn=int((ww[n2:] > 0).sum()),
                   group_share=C_gshare(ww, grp), out=f"{WREL}/{nm}.npz")
        C.dump(O / "results" / f"w_{nm}.json", rec)
    pw = w[pos] / w[pos].sum()
    sv = lambda m: float(pw[m].sum())  # noqa: E731
    rb0_fast = _rb0_fast(W, w1, s_part)
    s_all = (rb0_fast * (K_ON - K_PS) + sv(v0 >= V_FAST) * K_PS) / K_ON
    rec = dict(ps_candidates=int(R["cand"].sum()), ps_held_logs=len(held & set(logs[n2:].tolist())), ps_drawn=int(len(pos)),
               ps_road_ok=float(R["ok"][ci].mean()), ps_fast=sv(v0 >= V_FAST), ps_standing=sv(st), rb0_fast=rb0_fast, s_fast_all=s_all,
               ps_by_cls={c: dict(rows=int((cls == c).sum()), share=sv(cls == c), w=float(np.clip(CLS_W.get(c, 1.0), *W_CAP))) for c in sorted(set(cls))},
               G4=dict(line=LINE_G4, value=sv(v0 >= V_FAST), pass_=sv(v0 >= V_FAST) >= LINE_G4,
                       what="share of the PAI-S draws (16 of the 32 S slots) at v0 >= 13 m/s; all S draws reported as s_fast_all"))
    rec["pass_"] = bool(rec["G4"]["pass_"] and rec["ps_standing"] <= STAND_MAX + 1e-9)
    C.dump(O / "results/g4.json", rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["pass_"] else 2)


def C_gshare(w, grp):
    return {str(g): float(w[grp == g].sum() / w.sum()) for g in sorted(set(grp.tolist())) if w[grp == g].sum() > 0}


def _rb0_fast(W, w1, s_part):
    tb = np.load(CR / POOL2 / "tab.npz")
    v = tb["speed"].astype(np.float64)
    m = s_part & (w1 > 0)
    return float(w1[m & (v >= V_FAST)].sum() / w1[m].sum())


def cmd_wcheck(a):
    W, Z, N = np.load(DD / W2), np.load(O / "w/NTV1D2SZ.npz"), np.load(O / "w/NTV1D2S.npz")
    tb = np.load(CR / POOLS / "tab.npz")
    names, part, mode, logs = tb["names"].astype(str), tb["pool"].astype(str), tb["mode"].astype(str), tb["log"].astype(str)
    n2 = len(W["names"])
    w1, wz, w = W["w"], Z["w"], N["w"]
    sh = lambda m: float(w[m].sum() / w.sum())  # noqa: E731
    ps = part == "ps"
    held = {g for g in set(logs[mode == "self"].tolist()) if zlib.crc32(g.encode()) % 100 < HOLD_PCT}
    rb0 = np.zeros(len(w), bool)
    rb0[:n2] = W["grp"].astype(str) == "S"
    ck = dict(names_equal=bool(np.array_equal(N["names"].astype(str), names) and np.array_equal(Z["names"].astype(str), names)),
              prefix_names=bool(np.array_equal(names[:n2], W["names"].astype(str))),
              z_prefix_bitwise=bool(wz.dtype == w1.dtype and np.array_equal(wz[:n2], w1)), z_tail_zero=bool(not wz[n2:].any()),
              dagger_bitwise=bool(np.array_equal(w[:n2][~rb0[:n2]], w1[~rb0[:n2]].astype(w.dtype))),
              rb0_scaled=bool(np.allclose(w[:n2][rb0[:n2]], w1[rb0[:n2]] * 0.5, rtol=0, atol=1e-15)),
              ps_modes=sorted(set(mode[ps & (w > 0)])), ps_held_drawn=int((ps & (w > 0) & np.isin(logs, list(held))).sum()),
              share_rb0=sh(rb0), share_rb0_parts=sorted(set(part[rb0 & (w > 0)])), share_ps=sh(ps), share_pd=sh(part == "pd"), share_pd2=sh(part == "pd2"),
              share_pd_ntv1d2=float(w1[W["grp"].astype(str) == "pd"].sum()), share_pd2_ntv1d2=float(w1[W["grp"].astype(str) == "pd2"].sum()),
              drawn_parts=sorted(set(part[w > 0])))
    ok = (ck["names_equal"] and ck["prefix_names"] and ck["z_prefix_bitwise"] and ck["z_tail_zero"] and ck["dagger_bitwise"] and ck["rb0_scaled"]
          and ck["ps_modes"] == ["self"] and ck["ps_held_drawn"] == 0 and abs(ck["share_rb0"] - 0.4) < 1e-9 and abs(ck["share_ps"] - 0.4) < 1e-9
          and abs(ck["share_pd"] - ck["share_pd_ntv1d2"]) < 1e-9 and abs(ck["share_pd2"] - ck["share_pd2_ntv1d2"]) < 1e-9
          and ck["drawn_parts"] == ["pd", "pd2", "ps", "rb0"] and ck["share_rb0_parts"] == ["rb0"])
    rec = dict(checks=ck, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 2)


# ---------------------------------------------------------------- argv / identity / smoke
def build(arm, tag, steps=None, on_start=None, save_at=None, mix=False, seed=None):
    import dg4
    a = dg4.build("D3", tag, steps, on_start, save_at, seed, mix)       # seed None = the stored --seed 0, token for token
    if arm in ("SP", "SPZ"):
        C.put(a, "--on", POOLS)
        C.put(a, "--on-wfile", f"{WREL}/{'NTV1D2S' if arm == 'SP' else 'NTV1D2SZ'}.npz")
        a += ["--on-road-pai", str(DD / RIDX)]
    return a


def cmd_argv(a):
    print("\n".join(build(a.arm, a.tag, a.steps, a.on_start, a.save_at, a.mix, a.seed)))


def cmd_diff(a):
    b = C.stored_argv(D3)
    d3, x = build("D3", D3), build("SP", SP)
    head = x[:len(b)]
    d = [(b[i - 1], b[i], head[i]) for i in range(len(b)) if b[i] != head[i]]
    want = {"--tag": SP, "--on": POOLS, "--on-wfile": f"{WREL}/NTV1D2S.npz"}
    ok = d3 == b and {q[0]: q[2] for q in d} == want and len(d) == len(want) and x[len(b):] == ["--on-road-pai", str(DD / RIDX)]
    rec = dict(d3_equals_stored=d3 == b, sp_diff=[list(q) for q in d], appended=x[len(b):], want=want, sp_argv=" ".join(x), pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 1)


def split_cmp(d, r, upto, allow_new, strict=False):
    x, y = C.scalars(d, upto), C.scalars(r, upto)
    mx, my = {k: v for k, v in x.items() if MONI.match(k[0])}, {k: v for k, v in y.items() if MONI.match(k[0])}
    keys = sorted(set(x) & set(y))
    diff = {k: abs(x[k] - y[k]) for k in keys}
    core = [k for k in keys if strict or not MONI.match(k[0])]
    only_new = sorted({k[0] for k in set(x) - set(y)})
    only_ref = sorted(f"{k[0]}@{k[1]}" for k in set(y) - set(x))
    rec = dict(ref_run=str(r), upto=upto, scalars_compared=len(core), identical=int(sum(diff[k] == 0 for k in core)),
               worst=sorted(((f"{k[0]}@{k[1]}", diff[k]) for k in core), key=lambda kv: -kv[1])[:8], only_ref=only_ref, only_new_tags=only_new,
               monitor_slices=dict(compared=len(set(mx) & set(my)), identical=int(sum(mx[k] == my[k] for k in set(mx) & set(my))), gated=strict))
    rec["bit_identical"] = bool(core and rec["identical"] == len(core) and not only_ref and (allow_new or not only_new))
    return rec


def cmd_ident(a):
    import signal
    import subprocess
    assert not C.run_dirs(a.tag), f"{a.tag} has a run dir already"
    p = subprocess.Popen(a.cmd, start_new_session=True)
    d = None
    while p.poll() is None:
        time.sleep(5)
        d = d or (C.run_dirs(a.tag) or [None])[-1]
        if d and (d / "events.jsonl").exists() and any(e.get("kind") == "scalar" and e["step"] >= a.stop for e in C.events(d)):
            break
    rc = p.poll()
    if rc is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL), p.wait()
    d = d or (C.run_dirs(a.tag) or [None])[-1]
    assert d, "no run dir"
    rec = split_cmp(d, C.run_dirs(a.ref)[-1], a.upto, False) | dict(tag=a.tag, ref=a.ref, run=str(d), trainer_rc_before_stop=rc)
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["bit_identical"] else 2)


def cmd_pair(a):
    da, db = C.run_dirs(a.a)[-1], C.run_dirs(a.b)[-1]
    rec = split_cmp(db, da, a.upto, False, a.strict) | dict(a=a.a, b=a.b, run_a=str(da), run_b=str(db))
    rec["pool_rows_b"] = C.pool_rows(db)
    rec["pool_drawn"] = bool(any(v > 0 for v in rec["pool_rows_b"].values()))
    log = (db / "log.txt").read_text()
    m = re.search(r'"road_pai_rows": (\d+)', log)
    rec["road_pai_rows_b"] = int(m.group(1)) if m else None
    rec["pass_"] = bool(rec["bit_identical"] and rec["pool_drawn"])
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["pass_"] else 2)


def cmd_smoke(a):
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "dgc.py"), "smoke", "--var", "b", "--tag", a.tag, "--on-start", str(a.on_start),
                        "--want", str(O / "results/w_NTV1D2S.json"), "--out", a.out + ".dgc.json"])
    base = json.loads(Path(a.out + ".dgc.json").read_text())
    d = C.run_dirs(a.tag)[-1]
    log = (d / "log.txt").read_text()
    m = re.search(r'"road_pai_rows": (\d+)', log)
    ix = np.load(SP_ / "road_spai.npz")
    tb = np.load(CR / POOLS / "tab.npz")
    pn = set(tb["names"].astype(str).tolist())
    want = int(sum(o and t in pn for o, t in zip(ix["ok"], ix["tokens"].astype(str))))
    rp = int(m.group(1)) if m else None
    rd = {e["step"]: float(e["value"]) for e in C.events(d) if e.get("kind") == "scalar" and e["tag"] == "loss/road_ho"}
    post = [v for s, v in rd.items() if s > a.on_start]
    ck = dict(dgc=base["checks"], dgc_pass=base["pass_"], road_pai_rows=rp, road_pai_rows_want=want, road_ho_after_T=post)
    ok = base["pass_"] and rp == want and want > 0 and bool(post) and all(np.isfinite(post)) and any(v > 0 for v in post)
    rec = dict(tag=a.tag, checks=ck, pass_=bool(ok), dgc_rc=r.returncode)
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1, default=float))
    sys.exit(0 if ok else 2)


# ---------------------------------------------------------------- build summary
def cmd_build(a):
    import dg4
    sys.path.insert(0, str(REPO / "experiments/round6/scripts"))
    ld = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None  # noqa: E731
    res = O / "results"
    A, R = dg4.facts_ok(O / "tax/roll.csv"), dg4.facts_ok(O4 / "tax/roll.csv")
    from collections import Counter
    from jevdrive import stats
    roll = {}
    for nm, X in (("D3 (NT4D3-F-s0)", A), ("D1 (step 4 roll)", R)):
        z = X[X.score == 0]
        roll[nm] = dict(scenes=len(X), mean=float(X.score.mean()), zeros=len(z), zero_fclass=dict(Counter(z.fclass)), full=int((X.score >= 0.98).sum()))
    cm = A.index.intersection(R.index)
    roll["paired D3 - D1"] = stats.paired(A.score[cm].values, R.score[cm].values) | dict(scenes=len(cm))
    disk = {k: dg4.du(v) for k, v in dict(roll=O / "roll", rows=O / "rows", teach=O / "teach", cache=O / "cache", pool=CR / NP3, pool_warp=CR / f"{NP3}@warp",
                                             merged=CR / POOLS, merged_warp=CR / f"{POOLS}@warp", feat_merged=DD / f"runs/route_mem/feat/{POOLS}.npz",
                                             spai=SP_, fwd=O / "fwd", tax=O / "tax", w=O / "w").items()}
    disk["total"] = sum(disk.values())
    g = {k: ld(res / f"{k}.json") for k in ("g13", "g4", "gw_draw_files", "diff_SP", "gcode", "gw0", "gpre_SP", "smoke_SP")}
    rec = dict(student=D3, pool_iter3=ld(RES / f"rows_{NP3}.json"), roll=roll, gates={k: (v or {}).get("pass_", (v or {}).get("bit_identical")) for k, v in g.items()},
               g13={k: g["g13"][k] for k in ("G1", "G2", "G3", "fallback", "ok_share", "candidates", "by_cls", "plan_beyond_56m")} if g["g13"] else None,
               g4=g["g4"], disk_gib=disk, cost=dg4.cost(O), t=time.strftime("%F %T"))
    C.dump(res / "build.json", rec)
    print(json.dumps(rec, indent=1, default=float))


def cmd_bs(a):
    import dgt
    dgt.cmd_bs(a)


# ---------------------------------------------------------------- verdict
def cmd_verdict(a):
    import dg4
    import dgt as T_
    from jevdrive import stats
    o = Path(a.dir)
    P1L, NT, R8, IP = T_.P1L, T_.NT, T_.R8, T_.IP
    labs = [SP, D3, D3S1, D1, D1S1, P1L, NT, R8, IP]
    OLD = {t: T_.facts(f"{t}-281ag") for t in labs}
    NEW = T_.ref_new()
    NEW[D1] = T_.new_series(O3 / "sv2/new/rollouts.csv", f"{D1}-ag")
    NEW[D1S1] = T_.new_series(O3 / "sv2c/new/rollouts.csv", f"{D1S1}-ag")
    NEW[D3] = T_.new_series(O4 / "sv2/new/rollouts.csv", f"{D3}-ag")
    NEW[D3S1] = T_.new_series(O4 / "sv2s1/new/rollouts.csv", f"{D3S1}-ag")
    NEW[SP] = T_.new_series(o / "sv2/new/rollouts.csv", f"{SP}-ag")
    nm = {SP: "S-PAI", D3: "D3-s0", D3S1: "D3-s1", D1: "D1-s0", D1S1: "D1-s1", P1L: "P1L", NT: "NTV0-s0", R8: "R8K-s2", IP: "INTP75-NR (ref.)"}
    pairs = [(SP, y) for y in (D3, D3S1, D1, R8, IP)]
    res = T_.score_block(labs, OLD, NEW, pairs, nm)
    S = res["_S"]
    res["sp_vs_d3_mean"] = {sc: stats.paired(S[sc][SP], (S[sc][D3] + S[sc][D3S1]) / 2) for sc in ("new", "old")}
    lines = {}
    for sc in ("new", "old"):
        x = res["pairs"][f"{SP}:{D3}"][sc]
        lines[sc] = dict(sp_d3=x, line=("own PAI S helps" if x["mean"] >= LINE else "own PAI S hurts" if x["mean"] <= -LINE else "neutral"), registered=sc == "new")
    res["lines"] = lines
    hw = T_.highway([(SP, o / "bs", f"{SP}-281ag"), (D3, O4 / "bs", f"{D3}-281ag")]
                    + [(t, DD / "runs/pai_av/behavior_stats/out", f"{t}-281ag") for t in (NT, R8)], NEW)
    res["highway"] = hw
    # val160
    vs = {SP: (o / "v160/sv2/new/rollouts.csv", f"{SP}-v160"), D3: (O4 / "v160/sv2/new/rollouts.csv", f"{D3}-v160"),
          D3S1: (O4 / "v160/sv2/new/rollouts.csv", f"{D3S1}-v160"), R8: (OV / "sv2/new/rollouts.csv", f"{R8}-v160"), D1: (OV / "sv2/new/rollouts.csv", f"{D1}-v160")}
    VOLD = {t: T_.facts(f"{t}-v160ag") for t in vs}
    VNEW = {t: T_.new_series(f, r) for t, (f, r) in vs.items()}
    vres = T_.score_block(list(vs), VOLD, VNEW, [(SP, y) for y in (D3, D3S1, R8, D1)], nm)
    VS = vres["_S"]
    vres["sp_vs_d3_mean"] = {sc: stats.paired(VS[sc][SP], (VS[sc][D3] + VS[sc][D3S1]) / 2) for sc in ("new", "old")}
    vres["line"] = {sc: (lambda x: dict(sp_d3=x, line="helps" if x["mean"] >= LINE else "hurts" if x["mean"] <= -LINE else "neutral"))(
        vres["pairs"][f"{SP}:{D3}"][sc]) for sc in ("new", "old")}
    res["val160"] = T_.strip(vres)
    res["gates"] = T_.gates(o)
    rf = o / "results/ratio_SP.json"
    if rf.exists():
        r = json.loads(rf.read_text())
        res["ratio"] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=C.grad_ratio(C.run_dirs(SP)[-1], 10000, 19500)["ratio"],
                            d3_after_T=C.grad_ratio(C.run_dirs(D3)[-1], 10000, 19500)["ratio"])
    res["ade"] = {t: dict(at_10000=C.dev_ade(t, 10000), end=C.dev_ade(t)) for t in (SP, D3)}
    res["cost"] = dg4.cost(o)
    L = ["# NTV1 S-PAI verdict: D3's recipe with half the S slots on D3's own PAI states (PAI 281 + val160, ACC + GAP, Harmonizer OFF)", ""]
    L += T_.md_block(res, nm, "PAI 281")
    dm = res["sp_vs_d3_mean"]
    L += ["", f"S-PAI - D3 2-seed mean: new {C.f3(dm['new'])}, old {C.f3(dm['old'])}.", "", "## Registered line (new scorer registered, old reported)", ""]
    for sc, q in lines.items():
        L.append(f"- {sc} scorer: S-PAI - D3-s0 {C.f3(q['sp_d3'])} (helps >= +{LINE}, hurts <= -{LINE}) -> **{q['line']}**" + ("" if q["registered"] else " (reported)"))
    L += T_.hw_md(hw, nm)
    L += [""] + T_.md_block(vres, nm, "val160")
    L += ["", f"val160 S-PAI - D3-s0: new {C.f3(vres['line']['new']['sp_d3'])} ({vres['line']['new']['line']}), old {C.f3(vres['line']['old']['sp_d3'])}; "
          f"S-PAI - D3 2-seed mean: new {C.f3(vres['sp_vs_d3_mean']['new'])}, old {C.f3(vres['sp_vs_d3_mean']['old'])}."]
    L += ["", "## Gates", ""] + [f"- {k}: {v}" for k, v in res["gates"].items()]
    if "ratio" in res:
        r = res["ratio"]
        L.append(f"- S / imit gradient ratio: 500 steps after T {r['r500']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}, "
                 f"{'pass' if r['pass_'] else 'FAIL'}); whole post-T segment {r['after_T']:.3f} (D3 {r['d3_after_T']:.3f})")
    L += ["", "| navtrain dev ADE | step 10 000 | end |", "|:--|--:|--:|"] + [
        f"| {nm.get(t, t)} | {v['at_10000'] or float('nan'):.4f} | {v['end'] or float('nan'):.4f} |" for t, v in res["ade"].items()]
    cs = res["cost"]
    L += ["", "## Cost", "", "GPU job-hours: " + ", ".join(f"{k} {v:.2f}" for k, v in cs.items() if k not in ("gpu_job_h", "wall_h"))
          + f"; total {cs['gpu_job_h']:.1f}; wall {cs['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", T_.strip(res))
    print("\n".join(L))


# ---------------------------------------------------------------- final arm: S-PAI seed 1 (plans/2026-10-24-ntv1-final.md)
def cmd_fdiff(a):
    import hashlib
    b = C.stored_argv(SP)
    s0, x = build("SP", SP), build("SP", a.tag, seed=a.seed)
    d = [(b[i - 1], b[i], x[i]) for i in range(len(b)) if b[i] != x[i]] if len(x) == len(b) else None
    want = {"--tag": a.tag, "--seed": str(a.seed)}
    t0 = min(time.mktime(time.strptime(d.name, "%Y%m%d-%H%M%S")) for d in C.run_dirs(SP))    # run dir name = the run's start
    files = {}
    for f in (CR / POOLS / "tab.npz", DD / WREL / "NTV1D2S.npz", DD / RIDX):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        files[str(f)] = dict(sha256=h.hexdigest(), mtime=time.strftime("%F %T", time.localtime(f.stat().st_mtime)), before_s0=f.stat().st_mtime < t0)
    ok = s0 == b and d is not None and {q[0]: q[2] for q in d} == want and len(d) == len(want) and all(v["before_s0"] for v in files.values())
    rec = dict(s0_equals_stored=s0 == b, diff=[list(q) for q in d] if d is not None else "length differs", want=want, argv=" ".join(x),
               s0_train_start=time.strftime("%F %T", time.localtime(t0)), files=files, pass_=bool(ok))
    C.dump(a.out, rec)
    print(json.dumps(rec, indent=1))
    sys.exit(0 if ok else 1)


def cmd_fverdict(a):
    import dg4
    import dgt as T_
    from jevdrive import stats
    o, S1 = Path(a.dir), a.tag
    P1L, NT, R8, IP = T_.P1L, T_.NT, T_.R8, T_.IP
    labs = [S1, SP, D3, D3S1, NT, R8, IP]
    nm = {S1: "S-PAI-s1", SP: "S-PAI-s0", D3: "D3-s0", D3S1: "D3-s1", NT: "NTV0-s0", R8: "R8K-s2", IP: "INTP75-NR (ref.)"}
    OLD = {t: T_.facts(f"{t}-281ag") for t in labs}
    NEW = T_.ref_new()
    NEW = {NT: NEW[NT], R8: NEW[R8], IP: NEW[IP]}
    NEW[D3] = T_.new_series(O4 / "sv2/new/rollouts.csv", f"{D3}-ag")
    NEW[D3S1] = T_.new_series(O4 / "sv2s1/new/rollouts.csv", f"{D3S1}-ag")
    NEW[SP] = T_.new_series(O / "sv2/new/rollouts.csv", f"{SP}-ag")
    NEW[S1] = T_.new_series(o / "sv2/new/rollouts.csv", f"{S1}-ag")
    ref = (R8, IP, D3, D3S1)

    def block(labs, OLD, NEW):
        res = T_.score_block(labs, OLD, NEW, [(S1, SP)] + [(x, y) for x in (S1, SP) for y in ref], nm)
        S = res["_S"]
        mean = {sc: (S[sc][SP] + S[sc][S1]) / 2 for sc in ("new", "old")}
        d3m = {sc: (S[sc][D3] + S[sc][D3S1]) / 2 for sc in ("new", "old")}
        res["mean2"] = dict(new=float(mean["new"].mean()), old=float(mean["old"].mean()),
                            zeros_new=float(((S["new"][SP] == 0).sum() + (S["new"][S1] == 0).sum()) / 2),
                            zeros_old=float(((S["old"][SP] == 0).sum() + (S["old"][S1] == 0).sum()) / 2),
                            d3_new=float(d3m["new"].mean()), d3_old=float(d3m["old"].mean()),
                            vs={k: {sc: stats.paired(mean[sc], f(sc)) for sc in ("new", "old")}
                                for k, f in ((nm[R8], lambda sc: S[sc][R8]), (nm[IP], lambda sc: S[sc][IP]), ("D3 2-seed mean", lambda sc: d3m[sc]))})
        return res

    def verdict_words(q):
        return "higher (CI excludes 0)" if q["lo"] > 0 else "lower (CI excludes 0)" if q["hi"] < 0 else "no established difference (CI includes 0)"

    def mean_md(res, title):
        m = res["mean2"]
        L = ["", f"### {title}: S-PAI 2-seed mean (per scene (s0 + s1) / 2, {res['scenes']} common scenes)", "",
             f"S-PAI 2-seed mean: new {m['new']:.3f}, old {m['old']:.3f}; zeros (mean of the two seeds) new {m['zeros_new']:.1f}, old {m['zeros_old']:.1f}. "
             f"D3 2-seed mean on the same scenes: new {m['d3_new']:.3f}, old {m['d3_old']:.3f}.", "",
             "| S-PAI 2-seed mean - | new scorer | old scorer | reading (new) |", "|:--|--:|--:|:--|"]
        for k, q in m["vs"].items():
            L.append(f"| {k} | {C.f3(q['new'])} | {C.f3(q['old'])} | {verdict_words(q['new'])} |")
        return L

    res = block(labs, OLD, NEW)
    hw = T_.highway([(S1, o / "bs", f"{S1}-281ag"), (SP, O / "bs", f"{SP}-281ag"), (D3, O4 / "bs", f"{D3}-281ag")]
                    + [(t, DD / "runs/pai_av/behavior_stats/out", f"{t}-281ag") for t in (NT, R8)], NEW)
    res["highway"] = hw
    OVB = DD / "runs/round9/val160b"
    vs = {S1: (o / "v160/sv2/new/rollouts.csv", f"{S1}-v160"), SP: (O / "v160/sv2/new/rollouts.csv", f"{SP}-v160"),
          D3: (O4 / "v160/sv2/new/rollouts.csv", f"{D3}-v160"), D3S1: (O4 / "v160/sv2/new/rollouts.csv", f"{D3S1}-v160"),
          R8: (OV / "sv2/new/rollouts.csv", f"{R8}-v160"), IP: (OVB / "sv2/new/rollouts.csv", f"{IP}-v160")}
    VOLD = {t: T_.facts(f"{t}-v160ag") for t in vs}
    VNEW = {t: T_.new_series(f, r) for t, (f, r) in vs.items()}
    vres = block(list(vs), VOLD, VNEW)
    res["val160"] = T_.strip(vres)
    res["gates"] = T_.gates(o)
    rf = o / "results/ratio_SP.json"
    if rf.exists():
        r = json.loads(rf.read_text())
        res["ratio"] = dict(r500=r["ratio"], pass_=r["pass_"], after_T=C.grad_ratio(C.run_dirs(S1)[-1], 10000, 19500)["ratio"],
                            s0_after_T=C.grad_ratio(C.run_dirs(SP)[-1], 10000, 19500)["ratio"])
    res["ade"] = {t: dict(at_10000=C.dev_ade(t, 10000), end=C.dev_ade(t)) for t in (S1, SP, D3, D3S1)}
    res["cost"] = dg4.cost(o)
    L = ["# NTV1 final: S-PAI seed 1 and the S-PAI 2-seed mean (PAI 281 + val160, ACC + GAP, Harmonizer OFF)", "",
         "Seed spread on this exam is 0.03-0.04: no claim beyond the CIs (plans/2026-10-24-ntv1-final.md section 4)."]
    L += mean_md(res, "PAI 281") + mean_md(vres, "val160")
    L += [""] + T_.md_block(res, nm, "PAI 281 (per read)")
    L += T_.hw_md(hw, nm)
    L += [""] + T_.md_block(vres, nm, "val160 (per read)")
    L += ["", "## Gates", ""] + [f"- {k}: {v}" for k, v in res["gates"].items()]
    if "ratio" in res:
        r = res["ratio"]
        L.append(f"- S / imit gradient ratio: 500 steps after T {r['r500']:.3f} (band {C.RATIO_BAND[0]:.2f}-{C.RATIO_BAND[1]:.2f}, "
                 f"{'pass' if r['pass_'] else 'FAIL'}); whole post-T segment {r['after_T']:.3f} (S-PAI-s0 {r['s0_after_T']:.3f})")
    L += ["", "| navtrain dev ADE | step 10 000 | end |", "|:--|--:|--:|"] + [
        f"| {nm.get(t, t)} | {v['at_10000'] or float('nan'):.4f} | {v['end'] or float('nan'):.4f} |" for t, v in res["ade"].items()]
    cs = res["cost"]
    L += ["", "## Cost", "", "GPU job-hours: " + ", ".join(f"{k} {v:.2f}" for k, v in cs.items() if k not in ("gpu_job_h", "wall_h"))
          + f"; total {cs['gpu_job_h']:.1f}; wall {cs['wall_h']:.1f} h."]
    Path(f"{a.out}.md").write_text("\n".join(L) + "\n")
    C.dump(f"{a.out}.json", T_.strip(res))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = ap.add_subparsers(dest="sub", required=True)
    p = s.add_parser("raster")
    p.add_argument("--chunk", required=True), p.add_argument("--rows", required=True), p.add_argument("--cfg-glob", required=True)
    p.add_argument("--workers", type=int, default=0)
    for k in ("index", "weights", "build"):
        s.add_parser(k)
    for k in ("wcheck", "diff"):
        s.add_parser(k).add_argument("--out", required=True)
    p = s.add_parser("argv")
    p.add_argument("arm", choices=("D3", "SP", "SPZ")), p.add_argument("tag")
    p.add_argument("--steps", type=int), p.add_argument("--on-start", type=int), p.add_argument("--save-at"), p.add_argument("--mix", action="store_true")
    p.add_argument("--seed", type=int)
    p = s.add_parser("fdiff")
    p.add_argument("--tag", required=True), p.add_argument("--seed", type=int, required=True), p.add_argument("--out", required=True)
    p = s.add_parser("fverdict")
    p.add_argument("--dir", required=True), p.add_argument("--tag", required=True), p.add_argument("--out", required=True)
    p = s.add_parser("ident")
    p.add_argument("--tag", required=True), p.add_argument("--ref", required=True), p.add_argument("--upto", type=int, required=True)
    p.add_argument("--stop", type=int, required=True), p.add_argument("--out", required=True), p.add_argument("cmd", nargs=argparse.REMAINDER)
    p = s.add_parser("pair")
    p.add_argument("--a", required=True), p.add_argument("--b", required=True), p.add_argument("--upto", type=int, required=True)
    p.add_argument("--out", required=True), p.add_argument("--strict", action="store_true")
    p = s.add_parser("smoke")
    p.add_argument("--tag", required=True), p.add_argument("--on-start", type=int, required=True), p.add_argument("--out", required=True)
    p = s.add_parser("bs")
    p.add_argument("--out", required=True), p.add_argument("--tags", required=True), p.add_argument("--workers", type=int, default=16)
    p = s.add_parser("verdict")
    p.add_argument("--dir", required=True), p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.sub == "ident" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    dict(raster=cmd_raster, index=cmd_index, weights=cmd_weights, wcheck=cmd_wcheck, argv=cmd_argv, diff=cmd_diff, ident=cmd_ident, pair=cmd_pair,
         smoke=cmd_smoke, build=cmd_build, bs=cmd_bs, verdict=cmd_verdict, fdiff=cmd_fdiff, fverdict=cmd_fverdict)[a.sub](a)


if __name__ == "__main__":
    main()
