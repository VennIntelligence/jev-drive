#!/usr/bin/env python3
"""NR1 lateral constraint labels (prereg plans/2026-10-17-nr1-prereg.md section 3.2; the loss is lib/nr1_lat.py). Box.

  build --pool on1_L18NR1 --rows L18RB0:rb0,L18RTB1:rtb1 --out DIR [--workers N]      (AlpaSim venv, CPU)
      per pool row (rows meta runs/onpol_nuplan/rows/<rows>/meta.npz, merged by prefix in pool order), every cell centre of the (128, 96) 0.5 m
      row-frame raster (drivable_hinge's X0 / Y0) against the run's own scene map (the evaluator's lane polygon and centre-line yaw at the
      projection, zeros_mech zm_read.Lanes, unedited): per heading bin -90 .. +90 deg (30 deg steps, the row's t0 heading = 0)
        same_b   some containing lane within 60 deg of the bin heading      onc_b   some containing lane beyond 120 deg
        junction two containing lanes more than 30 deg apart (overlapping connectors; zm_read's junction flag), per cell, heading-free
        own      the cell lies in the ego's own lane chain: the lanes holding the t0 rear axle within 60 deg of the t0 heading, their successors
                 (up to 100 m of centre line) and predecessors (up to 20 m)
      onc mask_b = onc_b & ~same_b & ~junction; soft mask_b = same_b & ~own & ~junction (none when no own lane is found at t0); each -> signed
      distance (m, + outside, 20 where empty) -> DIR/onc.npy, DIR/soft.npy (n, 7, 128, 96) float16 (written by the workers into the memmaps),
      DIR/meta.npz (names, bins, ok, own_found, junction share, errors).
  gate --pool on1_L18NR1 --rows L18RB0:rb0,L18RTB1:rtb1 --lat DIR [--v2 FILE] --out NAME      (op-train, CPU torch)
      the registered label gate: activation (term > 0) of every lib/nr1_lat term and variant on the human's logged 4 s path from its own t0
      pose (rows meta logfut, moved to the row frame), on the rows' served plans and on V2's plans of the same rows (df.py v2plans); the
      recipe's existing road / agent hinges reported beside them -> results/<NAME>.{json,md}, DIR/<NAME>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R), str(R / "lib"), str(R / "experiments/defect_fix/lib"), str(R / "experiments/zeros_mech/scripts"),
                str(R / "experiments/anchor_offset/scripts"), str(R / "experiments/body1/lib")]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RES = R / "experiments/defect_fix/results"
X0, Y0, RES_M, NH, NW = -8.0, -24.0, 0.5, 128, 96
XC, YC = np.meshgrid(X0 + RES_M * (np.arange(NH) + 0.5), Y0 + RES_M * (np.arange(NW) + 0.5), indexing="ij")
BINS = np.arange(-90.0, 91.0, 30.0)
SAME, ONC, JUNC = np.radians(60.0), np.radians(120.0), np.radians(30.0)
OWN_FWD, OWN_BACK, EPS = 100.0, 20.0, 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def merged_meta(pool, rows):
    """Rows meta of the parts, in the merged pool's order (rf_pait merge: names on:<prefix>:<name>)."""
    tb = np.load(DATA / "runs/op_parity/cache" / pool / "tab.npz")
    pn = tb["names"].astype(str)
    parts = []
    for spec in rows.split(","):
        r, pre = spec.split(":")
        M = dict(np.load(DATA / "runs/onpol_nuplan/rows" / r / "meta.npz"))
        M["names"] = np.array([f"on:{pre}:{t[3:]}" for t in M["names"].astype(str).tolist()])
        parts.append(M)
    keys = ("names", "run", "scene", "anchor", "t_us", "logfut", "poses", "k")
    M = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    assert np.array_equal(M["names"], pn), "rows meta != the pool's rows"
    return M


def own_lanes(lanes, vm, x, y, z, psi):
    """Ids of the ego's own lane chain (module docstring)."""
    import shapely
    pt = shapely.Point(float(x), float(y))
    seed = []
    for ln in vm.get_lanes_within(np.array([x, y, z], float), 10.0):
        poly, cl = lanes.lane(ln)
        if poly.contains(pt) and abs(wrap(float(lanes.yaw(cl, pt)) - psi)) < SAME:
            seed.append(ln.id)
    own = set(seed)
    for attr, cap in (("next_lanes", OWN_FWD), ("prev_lanes", OWN_BACK)):
        front, seen = [(i, 0.0) for i in seed], set(seed)
        while front:
            i, dist = front.pop()
            if dist >= cap:
                continue
            try:
                ln = vm.get_road_lane(i)
            except Exception:  # noqa: BLE001
                continue
            L = float(np.hypot(*np.diff(np.asarray(ln.center.points)[:, :2], axis=0).T).sum())
            for j in getattr(ln, attr):
                if j not in seen:
                    seen.add(j)
                    own.add(j)
                    front.append((j, dist + L))
    return own, bool(seed)


def sdf_of(mask):
    from scipy.ndimage import distance_transform_edt as edt
    if not mask.any():
        return np.full(mask.shape, 20.0, np.float32)
    return (np.where(mask, -(edt(mask) - 0.5), edt(~mask) - 0.5) * RES_M).astype(np.float32)


def work(job):
    import shapely
    import zm_read as Z
    from ao_e1 import loader
    cfg, scene, idx, anchor, t_us, out, dil = job
    if dil > 0:
        from scipy.ndimage import binary_dilation
        disk = np.hypot(*np.mgrid[-int(dil / RES_M):int(dil / RES_M) + 1, -int(dil / RES_M):int(dil / RES_M) + 1]) * RES_M <= dil
    onc_mm = np.load(Path(out) / "onc.npy", mmap_mode="r+")
    soft_mm = np.load(Path(out) / "soft.npy", mmap_mode="r+")
    res = []
    try:
        ds = loader(cfg).get_data_source(scene)
        tr = ds.rig.trajectory
        Tg, Pg = np.asarray(tr.timestamps_us, float), np.asarray(tr.positions, float)
        lanes = Z.Lanes(ds.map)
        vm = lanes.vm
    except Exception as e:  # noqa: BLE001
        return [(int(i), False, False, 0.0, f"loader: {type(e).__name__}: {e}") for i in idx]
    psb = np.radians(BINS)
    for j, i in enumerate(idx):
        try:
            a = anchor[j]
            z = float(np.interp(t_us[j], Tg, Pg[:, 2]))
            P = Z.to_local(a, np.c_[XC.ravel(), YC.ravel()])
            n = len(P)
            same, onc = np.zeros((len(BINS), n), bool), np.zeros((len(BINS), n), bool)
            d1, junc, own_c = np.full(n, np.nan), np.zeros(n, bool), np.zeros(n, bool)
            own, found = own_lanes(lanes, vm, a[0], a[1], z, a[2])
            for ln in vm.get_lanes_within(np.array([P[:, 0].mean(), P[:, 1].mean(), z], float), 60.0):
                poly, cl = lanes.lane(ln)
                inn = shapely.contains_xy(poly, P[:, 0], P[:, 1])
                if not inn.any():
                    continue
                pts = shapely.points(P[inn])
                dd = shapely.line_locate_point(cl, pts)
                q1 = shapely.get_coordinates(shapely.line_interpolate_point(cl, np.minimum(dd + EPS, cl.length)))
                q0 = shapely.get_coordinates(shapely.line_interpolate_point(cl, np.maximum(dd - EPS, 0)))
                hd = np.arctan2(q1[:, 1] - q0[:, 1], q1[:, 0] - q0[:, 0]) - a[2]           # lane direction relative to the t0 heading
                ii = np.flatnonzero(inn)
                for b, pb in enumerate(psb):
                    dlt = np.abs(wrap(hd - pb))
                    same[b, ii] |= dlt < SAME
                    onc[b, ii] |= dlt > ONC
                prev = d1[ii]
                fresh = np.isnan(prev)
                junc[ii[~fresh]] |= np.abs(wrap(hd[~fresh] - prev[~fresh])) > JUNC
                d1[ii[fresh]] = hd[fresh]
                if ln.id in own:
                    own_c[ii] = True
            if dil > 0:                                               # amendment 1: the junction mask grown by `dil` m (the whole intersection)
                junc = binary_dilation(junc.reshape(NH, NW), structure=disk).ravel()
            for b in range(len(BINS)):
                m_onc = (onc[b] & ~same[b] & ~junc).reshape(NH, NW)
                m_soft = (same[b] & ~own_c & ~junc).reshape(NH, NW) if found else np.zeros((NH, NW), bool)
                onc_mm[i, b] = sdf_of(m_onc).astype(np.float16)
                soft_mm[i, b] = sdf_of(m_soft).astype(np.float16)
            res.append((int(i), True, found, float(junc.mean()), ""))
        except Exception as e:  # noqa: BLE001
            res.append((int(i), False, False, 0.0, f"{type(e).__name__}: {e}"))
    onc_mm.flush(), soft_mm.flush()
    return res


def cmd_build(a):
    M = merged_meta(a.pool, a.rows)
    n = len(M["names"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("onc", "soft"):
        mm = np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.float16, (n, len(BINS), NH, NW))
        mm[:] = 20.0
        mm.flush()
        del mm
    keys = sorted(set(zip(M["run"].astype(str).tolist(), M["scene"].astype(str).tolist())))
    jobs = []
    for run, sc in keys:
        ii = np.flatnonzero((M["run"].astype(str) == run) & (M["scene"].astype(str) == sc))
        jobs.append((str(Path(run) / "generated-user-config-0.yaml"), sc, ii, M["anchor"][ii], M["t_us"][ii], str(out), a.dilate))
    ok, found, jun, err = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.float32), []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(work, jobs, chunksize=1):
            for i, o, f, j, e in res:
                ok[i], found[i], jun[i] = o, f, j
                if e:
                    err.append((i, e))
    np.savez(out / "meta.npz", names=M["names"], bins=BINS, ok=ok, own_found=found, junction_share=jun, points=np.array(a.points), dilate=np.array(a.dilate))
    st = dict(rows=n, ok=int(ok.sum()), own_found=float(found[ok].mean()), errors=len(err), err_sample=err[:5], junction_cell_share=float(jun[ok].mean()))
    (out / "build.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1), flush=True)
    if err and len(err) > 0.01 * n:
        raise SystemExit(2)


# ---------------------------------------------------------------- gate (op-train)
TURN_DEG, NEAR_M = 45.0, 1.0
VARIANTS_ONC = (0.0, 0.3, 0.5)                   # allowed depth (m) into the opposite-lane region, in the registered order
VARIANTS_SOFT = (0.0, 0.3, 0.5)
VARIANTS_BUF = (0.4, 0.25)                       # side buffer (m)
LINE_HARD, LINE_SOFT = 0.01, 0.05


def cmd_gate(a):
    import torch
    import nr1_lat as NL
    from agent_hinge import AgentHinge
    from drivable_hinge import Hinge
    M = merged_meta(a.pool, a.rows)
    n = len(M["names"])
    cr = DATA / "runs/op_parity/cache" / a.pool
    dev = torch.device("cpu")
    pack = NL.load(cr, M["names"], dev, file=a.lat)
    lf = M["logfut"].astype(np.float64)                               # the human's 0 .. 4 s poses, already in the row's rear-axle frame (ol_zmonc /
    okh = np.isfinite(lf).all((1, 2))                                 # ol_onclab2 read them so: to_local(anchor, logfut), heading anchor + yaw)
    H = np.nan_to_num(lf)
    H[..., 2] = np.unwrap(H[..., 2], axis=1)
    near = okh & (np.hypot(H[:, 0, 0], H[:, 0, 1]) <= NEAR_M)
    turn = np.degrees(np.abs(H[:, -1, 2] - H[:, 0, 2])) >= TURN_DEG
    plans = dict(human=(H[:, 1:], H[:, 0]), served=(M["poses"].astype(np.float64), None))
    if a.v2:
        z = np.load(a.v2)
        assert np.array_equal(z["names"].astype(str), M["names"]), "--v2 rows"
        plans["V2"] = (z["poses"].astype(np.float64), None)
    road = Hinge(cr / "road.npz", M["names"], dev, 0.3, bank=True)
    ah3 = AgentHinge(cr / "agent.npz", M["names"], dev, 0.3, 0.0)
    rows_all = torch.arange(n)
    res = dict(rows=n, rows_near=int(near.sum()), human_ok=int(okh.sum()), turn_rows=int(turn.sum()), lat=str(a.lat), plans={})

    def dense0(pp, q0):
        P0 = torch.cat([torch.zeros_like(pp[:, :1]) if q0 is None else q0[:, None], pp], 1)
        return torch.einsum("kj,bjc->bkc", pack["M"], P0)

    def per_row(fn, P, p0):
        """fn(dense poses, rows, raw poses, start) -> (n_chunk,) or (n_chunk, k) numpy, over all rows in chunks."""
        out = []
        for i in range(0, n, 1024):
            r = rows_all[i:i + 1024]
            pp = torch.as_tensor(P[i:i + 1024], dtype=torch.float32)
            q0 = None if p0 is None else torch.as_tensor(p0[i:i + 1024], dtype=torch.float32)
            with torch.no_grad():
                out.append(fn(dense0(pp, q0), r, pp, q0).numpy())
        return np.concatenate(out)

    for name, (P, p0) in plans.items():
        mo = per_row(lambda d, r, pp, q0: NL.region_m(d, r, pack, "onc").amin(1) * pack["ok"][r].float() + 1e3 * (~pack["ok"][r]).float(), P, p0)
        ms = per_row(lambda d, r, pp, q0: NL.region_m(d, r, pack, "soft").amin(1) * pack["ok"][r].float() + 1e3 * (~pack["ok"][r]).float(), P, p0)
        sd = per_row(lambda d, r, pp, q0: NL.side_dist(d, r, pack["ah"]).amin(1) + 1e3 * (~pack["ah"].ok[r]).float(), P, p0)
        A = {f"onc_d{x}": mo < -x for x in VARIANTS_ONC} | {f"soft_d{x}": ms < -x for x in VARIANTS_SOFT} | {f"side_b{x}": sd < x for x in VARIANTS_BUF}
        A["road_existing"] = per_row(lambda d, r, pp, q0: road_from(road, pp, r, q0), P, p0) > 0
        A["agent_existing"] = per_row(lambda d, r, pp, q0: agent_from(ah3, pp, r, q0), P, p0) > 0
        base = near if name == "human" else np.ones(n, bool)
        res["plans"][name] = {k: dict(all=float(v[base].mean()), straight=float(v[base & ~turn].mean()), turn=float(v[base & turn].mean()),
                                      all_rows=float(v[okh if name == "human" else np.ones(n, bool)].mean())) for k, v in A.items()}
        res["plans"][name]["_rows"] = int(base.sum())
        if name == "human":
            Ahum = A
    pick, tried = None, []
    for b in VARIANTS_BUF:                                            # registered order: protection first (no depth, the wider buffer)
        for dpt in VARIANTS_ONC:
            u = float((Ahum[f"onc_d{dpt}"] | Ahum[f"side_b{b}"])[near].mean())
            tried.append(dict(onc_depth=dpt, buf=b, hard_union=u, onc=float(Ahum[f"onc_d{dpt}"][near].mean()), side=float(Ahum[f"side_b{b}"][near].mean())))
            if pick is None and u <= LINE_HARD:
                pick = dict(onc_depth=dpt, buf=b, hard_union=u)
    soft = next((dict(soft_depth=d, soft=float(Ahum[f"soft_d{d}"][near].mean())) for d in VARIANTS_SOFT if Ahum[f"soft_d{d}"][near].mean() <= LINE_SOFT), None)
    alone_onc = next((d for d in VARIANTS_ONC if Ahum[f"onc_d{d}"][near].mean() <= LINE_HARD), None)
    alone_side = next((b for b in VARIANTS_BUF if Ahum[f"side_b{b}"][near].mean() <= LINE_HARD), None)
    res |= dict(hard_tried=tried, hard_pick=pick, soft_pick=soft, onc_alone=alone_onc, side_alone=alone_side,
                verdict=("PASS" if pick and soft else "PARTIAL" if (pick or soft or alone_onc is not None or alone_side is not None) else "FAIL"))
    L = [f"# NR1 lateral label gate ({a.pool}, {n} rows; human = the logged path from its own t0 pose on the {int(near.sum())} rows with the ego within "
         f"{NEAR_M} m of it)", "", "activation share (term > 0) on all / straight / turn (>= 45 deg) rows:", "",
         "| term | " + " | ".join(plans) + " |", "|---|" + "--:|" * len(plans)]
    for k in res["plans"]["human"]:
        if k.startswith("_"):
            continue
        L.append(f"| {k} | " + " | ".join(f"{res['plans'][p][k]['all']:.4f} / {res['plans'][p][k]['straight']:.4f} / {res['plans'][p][k]['turn']:.4f}" for p in plans) + " |")
    L += ["", f"hard (new terms' union on human paths <= {LINE_HARD}): " + "; ".join(f"depth {t['onc_depth']} buf {t['buf']}: {t['hard_union']:.4f}" for t in tried),
          f"pick: {pick}; soft (<= {LINE_SOFT}): {soft}; alone: onc depth {alone_onc}, side buf {alone_side}", f"verdict: {res['verdict']}"]
    if a.alone and alone_onc is not None:                             # amendment 1 (exploratory arm): onc and side each on their own line
        env = dict(ONC=1, SIDE=int(alone_side is not None), DEPTH=alone_onc, BUF=alone_side if alone_side is not None else VARIANTS_BUF[0])
    elif pick:                                                        # the registered pick -> the trainings' flags (df.sh lat_flags)
        env = dict(ONC=1, SIDE=1, DEPTH=pick["onc_depth"], BUF=pick["buf"])
    elif alone_onc is not None:
        env = dict(ONC=1, SIDE=0, DEPTH=alone_onc, BUF=VARIANTS_BUF[0])
    elif alone_side is not None:
        env = dict(ONC=0, SIDE=1, DEPTH=0.0, BUF=alone_side)
    else:
        env = dict(ONC=0, SIDE=0, DEPTH=0.0, BUF=VARIANTS_BUF[0])
    env |= dict(SOFT=int(soft is not None), SDEPTH=soft["soft_depth"] if soft else 0.0)
    res["flags"] = env
    (DATA / "runs/defect_fix" / a.env_out).write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    L.append(f"trainings: {env}")
    RES.mkdir(parents=True, exist_ok=True)
    for p in (RES / f"{a.out}.json", Path(a.lat) / f"{a.out}.json"):
        p.write_text(json.dumps(res, indent=1))
    (RES / f"{a.out}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def road_from(road, pp, r, q0):
    """The recipe's road hinge (margin 0.3, footprint corners) on a path from q0 (None = the origin) -> (n,)."""
    import torch
    P0 = torch.cat([torch.zeros_like(pp[:, :1]) if q0 is None else q0[:, None], pp], 1)
    d = torch.einsum("kj,bjc->bkc", road.M, P0)
    C = road.C[road.src[r]]
    c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
    cx = d[..., :1] + c * C[:, None, :, 0] - s * C[:, None, :, 1]
    cy = d[..., 1:2] + s * C[:, None, :, 0] + c * C[:, None, :, 1]
    xy = torch.stack([cx, cy], -1).reshape(len(pp), -1, 2)
    g = torch.stack([(xy[..., 1] + 24.0) / 24.0 - 1.0, (xy[..., 0] + 8.0) / 32.0 - 1.0], -1)[:, :, None]
    m = torch.nn.functional.grid_sample(road.sdf[r].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
    return torch.relu(0.3 - m[:, 4:]).mean(1) * road.ok[r].float()


def agent_from(ah, pp, r, q0):
    """The recipe's agent hinge (margin 0.3 ahead, contact beside) per row, path from q0 (None = the origin)."""
    dist, counted, corridor = ah.distances(pp[..., 0], pp[..., 1], pp[..., 2], r, p0=q0)
    m = (0.3 - dist).clamp_min(0)
    m = m.where(corridor, (0.0 - dist).clamp_min(0))
    return m.masked_fill(~counted, 0.0).amax(-1)[:, 1:].mean(1) * ah.ok[r].float()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pool", default="on1_L18NR1"), p.add_argument("--rows", default="L18RB0:rb0,L18RTB1:rtb1"), p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--dilate", type=float, default=0.0, help="amendment 1: grow the junction mask by this many metres (0 = registered build)")
    p.add_argument("--points", default="both", choices=("both", "ra"), help="amendment 1: points the region terms read (meta key)")
    p = sub.add_parser("gate")
    p.add_argument("--pool", default="on1_L18NR1"), p.add_argument("--rows", default="L18RB0:rb0,L18RTB1:rtb1"), p.add_argument("--lat", required=True)
    p.add_argument("--v2", default=""), p.add_argument("--out", default="lat_gate")
    p.add_argument("--env-out", default="lat_pick.env", help="flag file of the trainings (runs/defect_fix/<name>)")
    p.add_argument("--alone", action="store_true", help="amendment 1: each new hard term on its own line (no union line)")
    a = ap.parse_args()
    {"build": cmd_build, "gate": cmd_gate}[a.cmd](a)
