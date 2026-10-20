#!/usr/bin/env python3
"""ROUND5 PAI road raster (plans/2026-10-21-pai-road-raster.md): the nuPlan road.npz raster for the PAI human-state rows, so the existing
road hinge (on_train LossesOn `both` / `mixed`, --road-lam / --road-margin) can act on them. Lane script: scripts/road_pai.sh.

  build   (box, AlpaSim venv, CPU)  --pool on1_PAIH --rows <row dir> --cfg-glob '<roll>/*/runs/*/sim/generated-user-config-0.yaml'
          one raster per (scene, k) of the pool at the row's anchor (row file: rig pose = rear axle, scene local frame). Surface (plan
          amendment 1) = the evaluator's PAI offroad rule: the 4-connected component of the ego's box centre among the canvas cells that no
          road edge at the ego's level (|dz| <= 3 m, the scorer's test) crosses, plus the lane union (rf_tax.Map); canvas and SDF =
          opb_labels.canvas_fill / mask_sdf (the nuPlan road.npz recipe: 0.25 m, EDT both sides, 2 x 2 mean, clipped +-20 m, float16).
          -> <out>/parts/<scene>.npz (resumable) -> <out>/sdf.npy (U, 128, 96) float16 + <out>/keys.npz (scene, k, anchor, empty)
  index   (box, op-train env, CPU)  --pool on1_X   every PAI human row of the pool (names ..:paih|paih05|paih15:<scene>:<k>; stretch rows share
          the plain row's t0 pose and raster) -> <out>/road_on1_X.npz in the dedup layout lib/drivable_hinge.Hinge reads (tokens, ok, ix,
          sdf_file; + ok_primary, ok_strict, m_t0, m_lab, h_lab, why) for on_train --on-road-pai, and <out>/index_on1_X.json (fallback, G1).
          Margins are drivable_hinge.Hinge.margins on this very file (Pacifica footprint, 0.1 s interpolation): m_t0 = min SDF of the zero
          plan (t0 footprint), m_lab = min SDF along the row's own label `fut`, h_lab = its hinge value at --margin.
          Fallback (plan 4): (a) m_t0 < 0, (b) m_lab < -0.5, (c) no surface / raster -> ok_primary False. G1 = share of ok_primary rows whose
          label triggers the hinge (m_lab < margin); G1 > 1 %: the registered rule (d) (label triggers) is added: ok = ok_strict.
  g1ref   (box, op-train env, CPU)  the G1 read on nuPlan: navtrain rows' own fut on the NAVSIM drivable raster (P2H10's hinge on imitation
          rows) and on the road-and-lane raster (the pool rows' road hinge), plus on1_JTPH's `tv` rows -> <out>/g1ref.json
  g2      (box, AlpaSim venv, CPU)  R2J 281 ACC + GAP offroad zeros: the road hinge on R2J's model plan at the decision 2.5 s before the
          event (D1's event time and decision; anchor = controller pose at that decision), control = one decision every 2 s on R2J's
          solved scenes -> <out>/g2.json, g2.csv
  report  (box)  index / g1ref / g2 / g3 json -> experiments/round5/results/road_pai.{md,json}
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "experiments/route_follow/scripts"), str(REPO / "experiments/op_probe/scripts")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
CR, RES = D / "runs/op_parity/cache", REPO / "experiments/round5/results"
OUT = Path(os.environ.get("RP_OUT") or D / "runs/round5/road_pai")    # RP_OUT: a smoke elsewhere
PAI_PRE = ("paih", "paih05", "paih15")
LAYERS = np.array(["NUREC_ROAD_EDGE_SIDE", "NUREC_LANE", "NUREC_ROAD_AREA"])   # pai_raster: road-edge component + lane union
B_OUT, MARGIN = -0.5, 0.3                                 # fallback (b) depth; the recipe's --road-margin
EDGE_DZ, EDGE_R = 3.0, 80.0                               # the scorer's MAX_ROAD_EDGE_ELEVATION_DELTA_M; edges beyond 80 m miss the canvas
WHY = ("ok", "t0_off", "label_out", "no_surface", "label_trig")
SPEED = ((0, 0.3), (0.3, 5), (5, 10), (10, 20), (20, 99))
SPLIT = "physical_ai_av/pai-rows@v1"


def pai_key(name: str):
    """(scene, k) of a PAI human row name (plain, stretched or merged: on:[<prefix>:]paih[05|15]:<scene>:<k>), else None."""
    f = name.split(":")
    return (f[-2], int(f[-1])) if len(f) >= 4 and f[-3] in PAI_PRE else None


def scene_cfgs(pattern: str) -> dict:
    """scene id -> the first generated AlpaSim config that lists it (d1.py mapcov's rule)."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        for s in re.findall(r"scene_id: (clipgt-[0-9a-f-]+)", open(f).read()):
            out.setdefault(s, f)
    return out


def scene_geom(scene: str, cfg: str, xy: np.ndarray) -> dict:
    """One NuRec scene's data source (the runtime's scene loader, rf_tax.loader) -> lane union around xy (rf_tax.Map: lane polygons + road
    areas, or None), road-edge polylines [(n, 3) x y z] (onp_rows.load_map's), the rig trajectory (T, 3), the rig -> box-centre offsets."""
    import rf_tax as RT
    from trajdata.maps.vec_map_elements import MapElementType as MT
    t = time.time()
    ds = RT.loader(cfg).get_data_source(scene)
    vm, vc = ds.map, ds.rig.vehicle_config
    tr = np.asarray(ds.rig.trajectory.positions, float)
    g = dict(U=None, E=[], traj=np.c_[tr[:, :2], tr[:, 2] if tr.shape[1] > 2 else np.full(len(tr), np.nan)],
             xc=float(vc.aabb_x_offset_m + vc.aabb_x_m / 2), zc=float(vc.aabb_z_offset_m + vc.aabb_z_m / 2))   # onp_rows: rig_to_aabb (x 1.4675)
    if vm is not None:
        for e in vm.elements.get(MT.ROAD_EDGE, {}).values():
            q = np.asarray(e.polyline.points, float)
            if len(q) > 1:
                g["E"].append(np.c_[q[:, :2], q[:, 2] if q.shape[1] > 2 else np.full(len(q), np.nan)])
        if len(vm.lanes):
            U = RT.Map(vm, xy).U
            g["U"] = None if U.is_empty else U
    g["P"] = np.concatenate(g["E"]) if g["E"] else np.zeros((0, 3))
    g["pid"] = np.repeat(np.arange(len(g["E"])), [len(e) for e in g["E"]])
    g["t_load"] = time.time() - t
    return g


def pai_raster(g: dict, pose: np.ndarray):
    """The evaluator's PAI offroad rule (eval/scorers/offroad.py: off = no lane / lane union contains the footprint AND it touches a road
    edge within 3 m of the ego's elevation) on opb_labels' canvas: inside = the 4-connected component of the ego's box centre among the cells
    no same-level road edge crosses, plus the lane union; SDF by opb_labels.mask_sdf. -> (NH, NW) float16, share of the canvas inside, edges."""
    import cv2
    import opb_labels as OL
    from scipy.ndimage import label
    P, pid = g["P"], g["pid"]
    tr = g["traj"]
    z_ego = tr[np.argmin(((tr[:, :2] - pose[:2]) ** 2).sum(1)), 2] + g["zc"]          # log z beside the pose + rig -> box centre (the scorer's ego z)
    bar, keep = np.zeros((OL.NX, OL.NY), np.uint8), []
    d2 = ((P[:, :2] - pose[:2]) ** 2).sum(1)
    near = d2 < EDGE_R ** 2
    if near.any():
        q, dq, zq = pid[near], d2[near], P[near, 2]
        o = np.lexsort((dq, q))
        first = o[np.r_[True, np.diff(q[o]) != 0]]                                    # each edge's vertex nearest the ego in plan view
        keep = q[first][~(np.abs(zq[first] - z_ego) > EDGE_DZ)].tolist()               # the scorer's level test (no z: kept)
        if keep:
            cv2.polylines(bar, [OL._ring(g["E"][i], pose) for i in keep], False, 1, shift=4)
    lab, _ = label(bar == 0)
    i0, j0 = int((g["xc"] - OL.CX0) / OL.FINE), int(-OL.CY0 / OL.FINE)
    w = lab[i0 - 8:i0 + 9, j0 - 8:j0 + 9]                                               # an edge pixel under the box centre: nearest free cell
    ii, jj = np.nonzero(w)
    seed = w[ii, jj][np.argmin((ii - 8) ** 2 + (jj - 8) ** 2)] if len(ii) else 0
    ins = (lab == seed) if seed else np.zeros_like(bar, bool)
    if g["U"] is not None:
        ins |= OL.canvas_fill(g["U"], pose) > 0
    frac = float(ins.mean())
    sdf = np.full((OL.NH, OL.NW), 20.0, np.float32) if ins.all() else np.clip(OL.mask_sdf(ins), -20, 20)
    return sdf.astype(np.float16), frac, len(keep)


def rasters(g: dict, poses: np.ndarray):
    out = [pai_raster(g, p) for p in poses]
    return np.stack([o[0] for o in out]), np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out], np.int32)


# ---------------------------------------------------------------- build (AlpaSim venv)
def _build_one(job):
    scene, cfg, ks, rowf, part = job
    z = np.load(rowf)
    pos = {int(k): i for i, k in enumerate(z["k"])}
    miss = [k for k in ks if k not in pos]
    assert not miss, f"{scene}: pool rows k {miss[:5]} not in {rowf}"
    A = z["anchor"][[pos[k] for k in ks]].astype(np.float64)
    g = scene_geom(scene, cfg, A[:, :2])
    t = time.time()
    sdf, frac, ne = rasters(g, A)
    t_ras = (time.time() - t) / len(A)
    empty = g["U"] is None and not g["E"]
    tmp = part.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, k=np.asarray(ks, np.int32), anchor=A, sdf=sdf, frac=frac, n_edges=ne, empty=empty)
    os.replace(tmp, part)
    return dict(scene=scene, n=len(ks), empty=empty, edges=len(g["E"]), lanes=g["U"] is not None, frac_med=float(np.median(frac)),
                t_load=g["t_load"], t_ras=t_ras)


def _part_done(job) -> bool:
    p = job[4]
    if not p.exists():
        return False
    with np.load(p) as z:
        return np.array_equal(z["k"], np.asarray(job[2], np.int32))


def cmd_build(a):
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    names = np.load(CR / a.pool / "tab.npz")["names"].astype(str)
    keys = [pai_key(n) for n in names]
    assert all(keys), f"{a.pool}: a row name without a PAI human key"
    by = {}
    for s, k in keys:
        by.setdefault(s, []).append(k)
    cfg = scene_cfgs(a.cfg_glob)
    sc = sorted(by)[:a.limit or None]
    assert set(sc) <= set(cfg), f"no config for {sorted(set(sc) - set(cfg))[:3]}"
    parts = OUT / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    with Run("round5", "road-pai-build", config=vars(a)) as run:
        sp = splits.load(SPLIT)
        run.use_split(sp)
        assert sp.mask(np.array(sc)).all(), "a pool scene outside pai-rows@v1"
        jobs = [(s, cfg[s], sorted(by[s]), Path(a.rows) / f"{s}.npz", parts / f"{s}.npz") for s in sc]
        run.info(f"{a.pool}: {len(names)} rows, {len(sc)} scenes, {sum(map(len, by.values()))} distinct (scene, k); workers {a.workers or 'n_cpus'}")
        t0 = time.time()
        res = par.pmap(_build_one, jobs, workers=a.workers or None, run=run, desc="scenes", skip=_part_done)
        res.raise_if_failed()
        st = [v for v in res.values if v]
        n = sum(len(by[s]) for s in sc)
        sdf = np.lib.format.open_memmap(OUT / "sdf.tmp.npy", "w+", np.float16, (n, 128, 96))
        S, K, A, E, Fr, Ne, o = [], [], [], [], [], [], 0
        for s in sc:
            with np.load(parts / f"{s}.npz") as z:
                m = len(z["k"])
                sdf[o:o + m] = z["sdf"]
                S += [s] * m
                K.append(z["k"]), A.append(z["anchor"]), Fr.append(z["frac"]), Ne.append(z["n_edges"])
                E += [bool(z["empty"])] * m
                o += m
        sdf.flush()
        del sdf
        os.replace(OUT / "sdf.tmp.npy", OUT / "sdf.npy")
        Fr, Ne = np.concatenate(Fr), np.concatenate(Ne)
        np.savez(OUT / "keys.npz", scene=np.array(S), k=np.concatenate(K), anchor=np.concatenate(A), empty=np.array(E), frac=Fr, n_edges=Ne)
        rec = dict(pool=a.pool, rows=len(names), scenes=len(sc), rasters=n, built_now=len(st), empty_scenes=sum(v["empty"] for v in st),
                   scenes_without_edges=sum(v["edges"] == 0 for v in st), scenes_without_lanes=sum(not v["lanes"] for v in st),
                   rasters_no_edge=float((Ne == 0).mean()), rasters_inside_gt90=float((Fr > 0.9).mean()), inside_frac_med=float(np.median(Fr)),
                   sdf_gb=(OUT / "sdf.npy").stat().st_size / 1e9, parts_gb=sum(f.stat().st_size for f in parts.glob("*.npz")) / 1e9,
                   t_load_med=float(np.median([v["t_load"] for v in st])) if st else None,
                   t_ras_med=float(np.median([v["t_ras"] for v in st])) if st else None, build_s=time.time() - t0, split=sp.id)
        (OUT / "build.json").write_text(json.dumps(rec, indent=1))
        run.summary |= rec
        print(json.dumps(rec))


# ---------------------------------------------------------------- index (op-train env)
def _margins(H, rows, fut, bs=4096):
    """-> m_t0 (n,), m_lab (n,), h_lab (n,) on rows (Hinge rows), fut (n, 8, 3)."""
    import torch
    m0, ml, hl = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = torch.as_tensor(rows[i:i + bs])
            f = torch.as_tensor(fut[i:i + bs], dtype=torch.float32)
            z = torch.zeros(len(r), 8)
            m0.append(H.margins(z, z, z, r).amin(1))
            g = H.margins(f[..., 0], f[..., 1], f[..., 2], r)
            ml.append(g.amin(1)), hl.append(torch.relu(H.margin - g).mean(1))
    return (torch.cat(m0).numpy(), torch.cat(ml).numpy(), torch.cat(hl).numpy()) if len(rows) else (np.zeros(0),) * 3


def _speed_rows(v, trig, ok):
    return {f"{lo}-{hi}": dict(rows=int(m.sum()), hinged=int((m & ok).sum()), trig_of_hinged=float(trig[m & ok].mean()) if (m & ok).any() else None)
            for lo, hi in SPEED for m in [(v >= lo) & (v < hi)]}


def cmd_index(a):
    import torch
    from drivable_hinge import Hinge
    from jevdrive.common import n_cpus
    torch.set_num_threads(n_cpus())
    T = np.load(CR / a.pool / "tab.npz")
    names = T["names"].astype(str)
    keys = [pai_key(n) for n in names]
    sel = np.array([k is not None for k in keys])
    K = np.load(OUT / "keys.npz")
    at = {(s, int(k)): i for i, (s, k) in enumerate(zip(K["scene"].tolist(), K["k"].tolist()))}
    tok = names[sel]
    ix = np.array([at.get(keys[i], -1) for i in np.flatnonzero(sel)], np.int64)
    fut, v0 = T["fut"][sel].astype(np.float32), T["speed"][sel].astype(np.float32)
    part = T["pool"][sel].astype(str) if "pool" in T.files else np.full(len(tok), a.pool)
    no_s = (ix < 0) | K["empty"][np.maximum(ix, 0)]
    f = OUT / f"road_{a.pool}.npz"
    meta = dict(sdf_file="sdf.npy", x0=-8.0, y0=-24.0, res=0.5, layers=LAYERS, margin=MARGIN)
    np.savez(f, tokens=tok, ok=~no_s, ix=np.maximum(ix, 0), **meta)       # provisional: every row with a surface
    H = Hinge(f, tok, "cpu", MARGIN, bank=True)                             # the trainer's own loader on this file
    rows = np.flatnonzero(H.ok.numpy())
    m0, ml, hl = np.full(len(tok), np.nan, np.float32), np.full(len(tok), np.nan, np.float32), np.full(len(tok), np.nan, np.float32)
    m0[rows], ml[rows], hl[rows] = _margins(H, rows, fut[rows])
    ra, rb = ~no_s & (m0 < 0), ~no_s & (m0 >= 0) & (ml < B_OUT)
    okp = ~no_s & ~ra & ~rb
    trig = ~no_s & (ml < MARGIN)
    g1 = float(trig[okp].mean())
    strict = g1 > 0.01
    oks = okp & ~trig
    ok = oks if strict else okp
    why = np.select([no_s, ra, rb, okp & trig & strict], [3, 1, 2, 4], 0).astype(np.int8)       # index into WHY
    np.savez(f, tokens=tok, ok=ok, ok_primary=okp, ok_strict=oks, ix=np.maximum(ix, 0), m_t0=m0, m_lab=ml, h_lab=hl, why=why, **meta)
    Hc = Hinge(f, tok, "cpu", MARGIN, bank=True)                            # read back: the trainer sees `ok` rows only, rasters by ix
    rr = np.flatnonzero(Hc.ok.numpy())
    chk = _margins(Hc, rr[:20000], fut[rr[:20000]])[1]
    by_part = {p: dict(rows=int((part == p).sum()), ok_primary=float(okp[part == p].mean()), ok=float(ok[part == p].mean()),
                       g1_hinged=float(trig[okp & (part == p)].mean()) if (okp & (part == p)).any() else None) for p in sorted(set(part))}
    rec = dict(pool=a.pool, rows_pool=len(names), rows_pai=int(sel.sum()), rasters=int(len(np.unique(ix[ix >= 0]))),
               fallback=dict(no_surface=float(no_s.mean()), t0_off=float(ra.mean()), label_out=float(rb.mean()), ok_primary=float(okp.mean())),
               G1=dict(margin=MARGIN, trig_of_hinged=g1, trig_of_all=float((trig & okp).sum() / len(tok)), trig_all_rows_any=float(trig.mean()),
                       line=0.01, passed_primary=not strict, rule_d_added=strict, ok_shipped=float(ok.mean()), ok_strict=float(oks.mean()),
                       m_lab_q=np.nanpercentile(ml[okp], [1, 5, 25, 50]).round(3).tolist() if okp.any() else None),
               by_speed=_speed_rows(v0, trig, okp), by_part=by_part, readback_equal=bool(np.array_equal(chk, ml[rr[:20000]])),
               readback_rows=int(Hc.ok.sum()), file=str(f), file_mb=f.stat().st_size / 1e6)
    (OUT / f"index_{a.pool}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("pool", "rows_pai", "rasters", "fallback", "G1", "readback_equal")}))
    if not rec["readback_equal"]:
        raise SystemExit(2)


# ---------------------------------------------------------------- g1ref (op-train env): the same read on nuPlan rows
def cmd_g1ref(a):
    import torch
    from drivable_hinge import Hinge
    from jevdrive.common import n_cpus
    torch.set_num_threads(n_cpus())
    tabs = [np.load(CR / f"navtrain_full.s{i}of12" / "tab.npz") for i in range(12)]
    tok = np.concatenate([t["names"] for t in tabs]).astype(str)
    fut = np.concatenate([t["fut"] for t in tabs]).astype(np.float32)
    out = {}
    for nm, lf in (("navtrain_drivable", "runs/op_probe/labels/navtrain_all.npz"), ("navtrain_road", "runs/body1/labels_road/navtrain_s01234567891011.npz")):
        H = Hinge(D / lf, tok, "cpu", MARGIN, bank=True)
        r = np.flatnonzero(H.ok.numpy())
        m0, ml, _ = _margins(H, r, fut[r])
        okp = (m0 >= 0) & (ml >= B_OUT)
        out[nm] = dict(file=lf, rows=len(tok), labelled=len(r), trig_all_labelled=float((ml < MARGIN).mean()),
                       t0_off=float((m0 < 0).mean()), label_out=float(((m0 >= 0) & (ml < B_OUT)).mean()),
                       trig_of_hinged_pai_rule=float((ml[okp] < MARGIN).mean()), trig_below0_all=float((ml < 0).mean()),
                       m_lab_q=np.percentile(ml, [1, 5, 25, 50]).round(3).tolist())
        del H
    (OUT / "g1ref.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


# ---------------------------------------------------------------- g2 (AlpaSim venv): R2J's offroad zeros
def first_event(o, key):                                   # d1_mac.first_event
    s = o["ser"].get(key)
    if s is None:
        return np.nan
    s = np.asarray(s, float)
    m = np.nonzero(s[:, 1] > 0.5)[0]
    return float(s[m[0], 0]) if len(m) else np.nan


def ego_at(C, t):                                          # d1_mac.ego_at (controller state = rear axle, rollout local frame)
    return np.array([np.interp(t, C["t"], C["x"]), np.interp(t, C["t"], C["y"]), np.interp(t, C["t"], np.unwrap(C["yaw"]))])


def _g2_one(job):
    scene, cfg, poses = job
    g = scene_geom(scene, cfg, poses[:, :2])
    return rasters(g, poses)[0], g["U"] is None and not g["E"]


def cmd_g2(a):
    from jevdrive import par
    from jevdrive.common import n_cpus
    import torch
    from drivable_hinge import Hinge
    torch.set_num_threads(n_cpus())
    F = {r["scene"]: r for r in csv.DictReader(open(a.facts)) if r["rpc"] != "True"}
    R = {o["scene"]: o for o in pickle.load(open(a.tax, "rb"))["rolls"] if "ctrl" in o}
    Dc = pickle.load(open(a.dec, "rb"))
    cfg = scene_cfgs(a.cfg_glob)
    dec = []                                               # (kind, scene, k, t_k, te, pose, plan, v0)
    for s, r in sorted(F.items()):
        if s not in R or s not in Dc or not len(Dc[s]["t0"]):
            continue
        o, d = R[s], Dc[s]
        C = o["ctrl"]
        if r["fclass"] == "offroad":
            te = first_event(o, "offroad")
            if not np.isfinite(te):
                te = first_event(o, "left_corridor_laterally")
            if not np.isfinite(te):
                tev = float(r["t_ev_s"]) if r["t_ev_s"] not in ("", "nan") else np.nan
                te = float(C["t"][0]) + (tev * 1e6 if np.isfinite(tev) else float(C["t"][-1] - C["t"][0]))
            k = min(max(int(np.searchsorted(d["t0"], te - 2.5e6)), 0), len(d["t0"]) - 1)
            dec.append(("zero", s, k, float(d["t0"][k]), te, ego_at(C, float(d["t0"][k])), d["pm"][k], float(d["v0"][k])))
        elif float(r["score"]) >= 0.98:
            for k in range(0, len(d["t0"]), a.every):
                dec.append(("solved", s, k, float(d["t0"][k]), np.nan, ego_at(C, float(d["t0"][k])), d["pm"][k], float(d["v0"][k])))
    sc = sorted({x[1] for x in dec})
    jobs = [(s, cfg[s], np.array([x[5] for x in dec if x[1] == s])) for s in sc]
    res = par.pmap(_g2_one, jobs, desc="g2 scenes")
    res.raise_if_failed()
    sdf, empty = {}, {}
    for (s, _, _), (ras, e) in zip(jobs, res.values):
        sdf[s], empty[s] = iter(ras), e
    tok = np.array([f"{x[1]}:{x[2]}" for x in dec])
    f = OUT / "g2_rasters.npz"
    np.savez(f, tokens=tok, ok=np.array([not empty[x[1]] for x in dec]), sdf=np.stack([next(sdf[x[1]]) for x in dec]))
    H = Hinge(f, tok, "cpu", MARGIN)
    P = np.stack([x[6] for x in dec]).astype(np.float32)
    m0, ml, hl = _margins(H, np.arange(len(dec)), P)
    kind = np.array([x[0] for x in dec])
    rows = []
    for i, x in enumerate(dec):
        rows.append(dict(kind=x[0], scene=x[1], k=x[2], to_event_s=round((x[4] - x[3]) * 1e-6, 2) if x[0] == "zero" else "",
                         v0=round(x[7], 2), m_t0=round(float(m0[i]), 3), m_plan=round(float(ml[i]), 3), h_plan=round(float(hl[i]), 4),
                         trig=bool(ml[i] < MARGIN), off_t0=bool(m0[i] < 0), no_surface=bool(empty[x[1]])))
    with open(OUT / "g2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    z, c = kind == "zero", kind == "solved"
    trig, off = ml < MARGIN, m0 < 0
    rec = dict(read=Path(a.facts).stem, zeros=int(z.sum()), zeros_trig=int((trig & z).sum()), zeros_trig_share=float(trig[z].mean()),
               zeros_off_t0=int((off & z).sum()), zeros_trig_hinged=int((trig & ~off & z).sum()), zeros_plan_below0=int(((ml < 0) & z).sum()),
               zeros_h_med=float(np.median(hl[z])), line=0.5, passed=bool(trig[z].mean() > 0.5),
               solved_decisions=int(c.sum()), solved_scenes=len({x[1] for x in dec if x[0] == "solved"}), solved_trig_share=float(trig[c].mean()),
               solved_trig_hinged_share=float((trig & ~off)[c & ~off].mean()) if (c & ~off).any() else None, solved_off_t0=float(off[c].mean()),
               every=a.every, no_surface_scenes=sum(empty.values()))
    (OUT / "g2.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


# ---------------------------------------------------------------- report
def cmd_report(a):
    J = lambda n: json.loads((OUT / n).read_text()) if (OUT / n).exists() else None  # noqa: E731
    pools = [p for p in a.pools]
    ix = {p: J(f"index_{p}.json") for p in pools}
    b, g1r, g2 = J("build.json"), J("g1ref.json"), J("g2.json")
    g3 = {k: json.loads(Path(v).read_text()) for k, v in (x.split("=", 1) for x in a.g3) if Path(v).exists()}
    du = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9
    out = dict(build=b, index=ix, g1ref=g1r, g2=g2, g3=g3, disk_gb=du)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "road_pai.json").write_text(json.dumps(out, indent=1, default=float))
    pc = lambda v: "-" if v is None else f"{100 * v:.1f} %"  # noqa: E731
    L = ["# PAI road raster for the human-state rows (ROUND5)", "",
         "Plan: [plans/2026-10-21-pai-road-raster.md](../plans/2026-10-21-pai-road-raster.md). Code: `scripts/road_pai.py` / `road_pai.sh`. "
         "Trainer: `rm_train.py` / `on_train.py --on-road-pai <index> [--on-road-pai-w W]` (default off).", "",
         "## Format", "",
         "nuPlan road.npz layout: SDF (m, + inside) 128 x 96 cells of 0.5 m, x -8 .. 56 m, y -24 .. 24 m, rear axle (= PAI rig) frame at t0, float16, "
         "clipped +-20 m, opb_labels canvas (0.25 m fill, EDT, 2 x 2 mean); surface = NuRec lane polygons + road areas (rf_tax.Map). Stored once per "
         f"(scene, k): `$DATA_DIR/runs/round5/road_pai/sdf.npy` ({b['rasters'] if b else '?'} rasters, {b['sdf_gb'] if b else 0:.2f} GB); one small index per pool "
         "(`road_<pool>.npz`: tokens, ok, ix, sdf_file; drivable_hinge.Hinge reads it).", "",
         "## Fallback and G1 (human's own label path, margin 0.3 m, Pacifica footprint)", "",
         "| pool | PAI rows | no surface | (a) t0 off | (b) label > 0.5 m out | hinged (a-c) | G1: label triggers, of hinged | passed | shipped ok |",
         "|---|--:|--:|--:|--:|--:|--:|:-:|--:|"]
    for p, r in ix.items():
        if r:
            fb, g = r["fallback"], r["G1"]
            L.append(f"| {p} | {r['rows_pai']} | {pc(fb['no_surface'])} | {pc(fb['t0_off'])} | {pc(fb['label_out'])} | {pc(fb['ok_primary'])} | "
                     f"{pc(g['trig_of_hinged'])} | {'yes' if g['passed_primary'] else 'no (rule d added)'} | {pc(g['ok_shipped'])} |")
    if g1r:
        L += ["", "nuPlan reference (navtrain rows' own fut, same geometry):", "", "| raster | labelled | label triggers (all labelled) | t0 off | "
              "label triggers, of rows the PAI rule (a, b) keeps |", "|---|--:|--:|--:|--:|"]
        L += [f"| {k} | {v['labelled']} | {pc(v['trig_all_labelled'])} | {pc(v['t0_off'])} | {pc(v['trig_of_hinged_pai_rule'])} |" for k, v in g1r.items()]
    if g2:
        L += ["", "## G2: R2J 281 ACC + GAP offroad zeros, the hinge on R2J's own plan 2.5 s before the event", "",
              f"- zeros {g2['zeros']}: hinge triggers on **{g2['zeros_trig']} ({pc(g2['zeros_trig_share'])})** (line > 50 %: "
              f"{'pass' if g2['passed'] else 'FAIL'}); t0 footprint already off {g2['zeros_off_t0']} (rule a would leave those unhinged), "
              f"triggers on a hinged row {g2['zeros_trig_hinged']}; plan corner below 0 m {g2['zeros_plan_below0']}",
              f"- control: R2J's solved scenes ({g2['solved_scenes']}), one decision every {g2['every'] / 10:.0f} s ({g2['solved_decisions']}): triggers "
              f"{pc(g2['solved_trig_share'])} (hinged rows {pc(g2['solved_trig_hinged_share'])}), t0 off {pc(g2['solved_off_t0'])}"]
    if g3:
        L += ["", "## G3: identity (R2J's command, 60 steps eager)", ""]
        L += [f"- {k}: identical **{v.get('identical')}** (scalars {v.get('scalars')}, max diff {v.get('scalar_max_abs_diff')}, weights "
              f"{v.get('weight_max_abs_diff')}, on_hash_60 {v.get('on_hash_60')})" for k, v in g3.items()]
    L += ["", f"Disk under `$DATA_DIR/runs/round5/road_pai/`: {du:.2f} GB.", ""]
    (RES / "road_pai.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pool", default="on1_PAIH")
    p.add_argument("--rows", default=str(D / "runs/pai_rows/rows/H"))
    p.add_argument("--cfg-glob", default=str(D / "runs/pai_rows/roll/H/*/runs/*/sim/generated-user-config-0.yaml"))
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="first n scenes only (smoke, with RP_OUT)")
    p = sub.add_parser("index")
    p.add_argument("--pool", required=True)
    sub.add_parser("g1ref")
    p = sub.add_parser("g2")
    p.add_argument("--facts", default=str(D / "runs/pai_fix/facts/R2J-L18-s0-281ag.csv"))
    p.add_argument("--tax", default=str(D / "runs/pai_fix/tax/R2J-L18-s0-281ag.pkl"))
    p.add_argument("--dec", default=str(D / "runs/round5/d1/dec_R2J-281ag.pkl"))
    p.add_argument("--cfg-glob", default=str(D / "runs/round2/pai281/R2J-L18-s0-ag/runs/*/sim/generated-user-config-0.yaml"))
    p.add_argument("--every", type=int, default=20, help="control: every n-th decision (10 Hz) of a solved scene")
    p = sub.add_parser("report")
    p.add_argument("--pools", nargs="+", default=["on1_PAIH", "on1_PAIH05", "on1_PAIH15", "on1_JTPH"])
    p.add_argument("--g3", nargs="*", default=[], help="NAME=<bitcmp json>")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
