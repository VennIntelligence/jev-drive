"""op-adapt r2, R0 (todos/2026-09-29-op-adapt-r2-prereg.md section 2.3): does a 2x tele view make small pedestrians readable
in openpilot's frozen stage 3? Zero training.

Tele frame: the front image re-projected at focal 1820 (2 x the road frame's 910) into 512 x 256, its centre on the road
frame's centre ray (K = [[1820, 0, 256], [0, 1820, -32.8]]), same camgeom path as the road / wide frames (rotation only,
nearest neighbour, chroma 2 x 2 mean). The trunk input is (tele, wide) in place of (road, wide): the tele frame takes the
road frame's slot, the wide frame is unchanged (the road frame is what the 2x magnifies).

  cosmos   every finished Cosmos G4 pair, E1's read-out rows (5 Hz slot >= 9, x+ mask px >= 100), 4 streams (C+ C- K+ K-):
           stage-3 (`permute_73`) pooled mean|max and GT-box-cell pooled (E1 S1, box + 8 px, road|tele cells united with
           the wide frame's) for the road and the tele input -> r0/cosmos.npz
  p5       the P5 v1 BA pedestrian-scope observation pairs (x+ / x- frames, 4 414), same features; the pedestrian box is the
           hazard walker's 3-D box (0.6 x 0.6 x 1.86 m around its actor location) projected into the front camera
           -> r0/p5.npz
  probe    E1's probes (op_adapt_readout.oof_probe, 5 folds by instance / route, trained on all rows of the set), AUC in
           the px_eq bins, tele - road paired delta (cluster bootstrap); the open-Z rule -> r0/verdict.json, r0/*.csv

  CUDA_VISIBLE_DEVICES=1 taskset -c 108,109 $DATA_DIR/envs/op-train/bin/python scripts/op_adapt_r0.py cosmos --workers 2
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive import camgeom as G  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

OUT = data_dir() / "runs" / "op_adapt_r2" / "r0"
PAIRS = data_dir() / "runs" / "cosmos_full" / "pairs"
K_TELE = np.array([[1820.0, 0, 256.0], [0, 1820.0, 128.0 - 2 * (128.0 - 47.6)], [0, 0, 1]])
VIEWS = {"road": ("road", "wide"), "tele": ("tele", "wide")}
STREAMS = (("carla", 0), ("carla", 1), ("cosmos", 0), ("cosmos", 1))       # C+ C- K+ K- (0 = plus)
STEP, NSLOT, MIN_SLOT, MIN_PX, EXPAND = 4, 24, 9, 100, 8
BINS = ((0, 500), (500, 1500), (1500, np.inf), (0, np.inf))
F_COSMOS, F_P5_SEG = 640.0 / np.tan(np.radians(32.0)), 1113.5 / 2
P5_VIS = 68                                  # P5's own visibility threshold in px_eq (M0)
WALKER_HALF = np.array([0.3, 0.3, 0.93])
_CTX = {}


# ---------------------------------------------------------------- rendering
def gather_index(cal: dict) -> dict:
    """{view: flat index into the cameras' concatenated planes} for road, wide and tele; cal = {cam: WOD-format calib}."""
    sizes = [(c["width"], c["height"]) for c in cal.values()]
    Ks = dict(G.OP_K, tele=K_TELE)
    out = {}
    for k, K in Ks.items():
        src, U, V = G.choose_sources(np, G.pinhole_rays(np, K, G.OP_W, G.OP_H), cal)
        assert (src >= 0).all(), f"{k}: model frame not covered"
        out[k] = G.nn_gather_index(src, U, V, sizes).ravel()
    return out


def pack(cat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    import wod_zeroshot_openpilot as WZ
    return WZ._pack(cat[idx].reshape(G.OP_H, G.OP_W, 3))


def model_frames(cats: list, idx: dict) -> dict:
    """cats: per frame the concatenated YCbCr planes (n_pix, 3) -> {view: (n, 2, 6, 128, 256)}."""
    per = {k: np.stack([pack(c, idx[k]) for c in cats]) for k in ("road", "wide", "tele")}
    return {v: np.stack([per[a], per[b]], 1) for v, (a, b) in VIEWS.items()}


def box_cells(box, idx: np.ndarray, W: int, H: int, idx_wide: np.ndarray) -> np.ndarray:
    """E1's S1 cells (op_cosmos_probe.box_cells) at stage 3 (8 x 16), for one view's first frame and the wide frame; box
    (x0, y0, x1, y1) on the first camera (W x H), only model pixels sampled from that camera count."""
    out = np.zeros((8, 16), bool)
    if box is None or not np.isfinite(box).all():
        return out
    x0, y0, x1, y1 = box[0] - EXPAND, box[1] - EXPAND, box[2] + EXPAND, box[3] + EXPAND
    for ix in (idx, idx_wide):
        own = ix < W * H
        sy, sx = np.divmod(np.where(own, ix, 0), W)
        inb = own & (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
        if not inb.any():
            d = np.where(own, np.hypot(sx - (box[0] + box[2]) / 2, sy - (box[1] + box[3]) / 2), np.inf)
            j = int(d.argmin())
            if d[j] > 40:
                continue
            inb[j] = True
        out |= inb.reshape(256, 512).reshape(8, 32, 16, 32).any((1, 3))
    return out


def _ycc(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).convert("YCbCr")).reshape(-1, 3)


# ---------------------------------------------------------------- Cosmos pairs
def cosmos_job(pair):
    import cosmos_openpilot as CO
    from jevdrive.cosmos_full import load_pair
    idx = _CTX["idx"]
    d = PAIRS / pair
    gt = np.load(d / "gt.npz")
    px = gt["px"][::STEP].astype(int)[:NSLOT]
    S = np.flatnonzero((px >= MIN_PX) & (np.arange(len(px)) >= MIN_SLOT))
    if not len(S):
        return pair, None
    fr = {}
    for kind in ("carla", "cosmos"):
        p, m = load_pair(pair, kind)
        fr[(kind, 0)], fr[(kind, 1)] = p[::STEP][:NSLOT], m[::STEP][:NSLOT]
    need = sorted(set(S) | set(S - 1))
    pos = {s: i for i, s in enumerate(need)}
    frames = {v: [] for v in VIEWS}
    for st in STREAMS:
        mf = model_frames([_ycc(fr[st][s]) for s in need], idx)
        for v in VIEWS:
            cur = mf[v][[pos[s] for s in S]]
            prev = mf[v][[pos[s - 1] for s in S]]
            frames[v].append(np.stack([prev, cur], 1))                       # (len(S), 2, 2, 6, 128, 256)
    box = gt["box"][::STEP][:NSLOT][S].astype(float)
    box[box[:, 0] < 0] = np.nan
    W, H = CO.W, CO.H
    cells = {v: np.stack([box_cells(b, idx[VIEWS[v][0]], W, H, idx["wide"]) for b in box]) for v in VIEWS}
    spec = json.load(open(d / "spec.json"))
    return pair, {"S": S, "px": px[S], "frames": {v: np.stack(frames[v]) for v in VIEWS}, "cells": cells,
                  "inst": spec["inst"], "family": spec["family"]}


def _init_cosmos():
    import cosmos_openpilot as CO
    _CTX["idx"] = gather_index({1: CO.calib()})


def pool(x):
    x = x.float()
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


def masked_pool(x, m):
    x, m = x.float(), m[:, None]
    n = m.sum((2, 3)).clamp(min=1)
    mx = x.masked_fill(~m, float("-inf")).amax((2, 3))
    return torch.cat([(x * m).sum((2, 3)) / n, torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))], 1)


@torch.no_grad()
def stage3(net, pairs: np.ndarray, cells: np.ndarray, batch: int = 128):
    """pairs (n, 2 [prev, cur], 2 [first, wide], 6, 128, 256) uint8, cells (n, 8, 16) -> pooled, box-pooled (n, 2048) fp16."""
    out_p, out_b = [], []
    for i in range(0, len(pairs), batch):
        p = torch.as_tensor(pairs[i:i + batch]).cuda()
        o = net.run_batched(A.vision_feeds(p[:, 0], p[:, 1]), [A.TRUNK_OUT])[A.TRUNK_OUT][:, 0]
        m = torch.as_tensor(cells[i:i + batch]).cuda()
        out_p.append(pool(o).half().cpu()), out_b.append(masked_pool(o, m).half().cpu())
    return torch.cat(out_p).numpy(), torch.cat(out_b).numpy()


def cosmos(a, rl):
    pairs = sorted(p.parent.name for p in PAIRS.glob("*/done.json"))[: a.limit or None]
    rl.info(f"{len(pairs)} Cosmos pairs, {a.workers} workers")
    net = None
    acc = {k: [] for k in ("pair", "slot", "px", "inst", "family")}
    feats = {f"{v}_{k}": [] for v in VIEWS for k in ("pool", "box")}
    has = {v: [] for v in VIEWS}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_cosmos) as ex:
        list(ex.map(int, range(a.workers)))
        net = A.load("cinque", torch.float16).cuda()
        for i, (pair, r) in enumerate(ex.map(cosmos_job, pairs)):
            if r is not None:
                n = len(r["S"])
                acc["pair"] += [pair] * n
                acc["slot"] += list(r["S"])
                acc["px"] += list(r["px"])
                acc["inst"] += [r["inst"]] * n
                acc["family"] += [r["family"]] * n
                for v in VIEWS:
                    fr = r["frames"][v].reshape(-1, *r["frames"][v].shape[2:])              # (4 * n, 2, 2, 6, 128, 256)
                    cl = np.tile(r["cells"][v], (len(STREAMS), 1, 1))
                    P, B = stage3(net, fr, cl)
                    feats[f"{v}_pool"].append(P.reshape(len(STREAMS), n, -1).transpose(1, 0, 2))
                    feats[f"{v}_box"].append(B.reshape(len(STREAMS), n, -1).transpose(1, 0, 2))
                    has[v].append(r["cells"][v].any((1, 2)))
            if (i + 1) % 50 == 0 or i + 1 == len(pairs):
                el = time.time() - t0
                rl.info(f"[{i + 1}/{len(pairs)}] {len(acc['pair'])} rows, {el / 60:.1f} min, ETA {(len(pairs) - i - 1) * el / (i + 1) / 60:.0f} min")
                rl.event("progress", n=i + 1, rows=len(acc["pair"]), wall_s=el)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "cosmos.npz", **{k: np.array(v) for k, v in acc.items()}, **{k: np.concatenate(v) for k, v in feats.items()},
             **{f"has_{v}": np.concatenate(x) for v, x in has.items()})
    rl.event("end", pairs=len(pairs), rows=len(acc["pair"]), wall_s=time.time() - t0)


# ---------------------------------------------------------------- P5
def _p5_box(W5: dict, k: int, cal: dict):
    """The hazard walker with the most visible pixels at tick k (nearest actor record within 2 ticks), its 3-D box projected
    into the front camera -> (x0, y0, x1, y1) or None."""
    from jevdrive.p4_carla import REAR_AXLE_X
    fpx = W5["frames"].loc[k].px if k in W5["frames"].index else None
    fpx = fpx if isinstance(fpx, dict) else {}
    hz = [h for h in W5["hazards"] if fpx.get(str(h), 0) > 0] or W5["hazards"]
    if not hz or k not in W5["pose"].index:
        return None
    wid = max(hz, key=lambda h: fpx.get(str(h), 0))
    act = W5["act"]
    sel = np.flatnonzero(act["id"] == wid)
    if not len(sel):
        return None
    j = sel[np.argmin(np.abs(act["k"][sel] - k))]
    if abs(act["k"][j] - k) > 2:
        return None
    e = W5["pose"].loc[k]
    yaw = np.radians(float(e.yaw))
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy, dz = act["xyz"][j] - np.array([e.x, e.y, e.z])
    ctr = np.array([c * dx + s * dy - REAR_AXLE_X, -(-s * dx + c * dy), dz])     # rear axle, x fwd, y left, z up
    corners = ctr + WALKER_HALF * np.array([[a, b, h] for a in (-1, 1) for b in (-1, 1) for h in (-1, 1)])
    E = np.asarray(cal["extrinsic"], np.float64).reshape(4, 4)
    rays = corners - E[:3, 3]
    if (rays @ E[:3, 0] <= 0.5).any():
        return None
    u, v, _ = G.waymo_project(np, rays, cal)
    b = np.array([u.min(), v.min(), u.max(), v.max()])
    b[0::2], b[1::2] = b[0::2].clip(0, cal["width"]), b[1::2].clip(0, cal["height"])
    return b if (b[2] - b[0]) >= 1 and (b[3] - b[1]) >= 1 else None


def box_job(job):
    """x+ route id, [(obs, k)] -> [(obs, box or NaN)] (the x- world has no walker; its rows reuse the pair's x+ box)."""
    from jevdrive import p5_pairs as P
    rid, items = job
    W5 = P.load_world(P.attempt(data_dir() / "runs" / "p5v1" / "gen-ba", rid))
    out = []
    for i, k in items:
        b = _p5_box(W5, k, _CTX["cal"][1])
        out.append((i, np.full(4, np.nan) if b is None else b))
    return out


def _jpeg_ycc(path: str) -> np.ndarray:
    """libjpeg's own YCbCr, as p5_openpilot.render_blobs (so the road view equals the round-1 P5 cache input)."""
    from PIL import Image
    im = Image.open(path)
    im.draft("YCbCr", im.size)
    return np.asarray(im.convert("YCbCr")).reshape(-1, 3)


def p5_job(items):
    """[(row, cur path, prev path or "", box)] -> [(row, {view: (2, 2, 6, 128, 256)}, {view: cells})]."""
    idx, cf = _CTX["idx"], _CTX["cal"][1]
    res = []
    for row, cur, prev, box in items:
        cats = [np.concatenate([_jpeg_ycc(f.replace("/front/", f"/{c}/")) for c in ("front", "front_left", "front_right")])
                if f else None for f in (prev, cur)]
        mc = model_frames([cats[1]], idx)
        mp = model_frames([cats[0]], idx) if cats[0] is not None else {v: np.zeros_like(x) for v, x in mc.items()}
        b = box if np.isfinite(box).all() else None
        res.append((row, {v: np.stack([mp[v][0], mc[v][0]]) for v in VIEWS},
                    {v: box_cells(b, idx[VIEWS[v][0]], cf["width"], cf["height"], idx["wide"]) for v in VIEWS}))
    return res


def _init_p5(cal):
    _CTX["cal"] = cal
    _CTX["idx"] = gather_index(cal)


def p5(a, rl):
    os.environ["P5_SET"] = "carla_p5v1_ba"
    from jevdrive import p5_openpilot as PP
    from jevdrive import p5_pairs as P
    plan = json.loads((PP.root() / "op_plan.json").read_text())
    cal = {int(c): d for c, d in plan["calib"].items()}
    cal = {c: cal[c] for c in (1, 2, 3)}
    fams = ("DynamicObjectCrossing", "ParkingCrossingPedestrian", "PedestrianCrossing", "VehicleTurningRoutePedestrian")
    o = pd.read_parquet(P.processed() / "obs.parquet")
    o = o[o.family.isin(fams)].reset_index(drop=True)
    pairs = pd.read_csv(P.processed() / "pairs.csv", dtype={"base_id": str, "plus": str, "minus": str})
    o = o.merge(pairs[["base_id", "seed", "plus"]], on=["base_id", "seed"], how="left")
    assert o.plus.notna().all()
    prev_of, path_of = {}, {}                           # frame name -> front path, previous front path in its stream
    for st in plan["streams"]:
        for j, (n, fl) in enumerate(zip(st["names"], st["files"])):
            path_of[n], prev_of[n] = fl[0], (st["files"][j - 1][0] if j else "")
    rt = pd.DataFrame([(i, sg, fn, path_of.get(fn, ""), prev_of.get(fn, ""))
                       for i, r in o.iterrows() for sg, fn in ((1, r.fn_plus), (0, r.fn_minus))],
                      columns=["obs", "sign", "fn", "cur", "prev"])
    ok = rt.groupby("obs").cur.transform(lambda c: (c != "").all())
    rl.info(f"P5: {len(o)} pedestrian observation pairs; {int((~ok).sum() // 2)} pairs without both stream frames dropped")
    rt = rt[ok].reset_index(drop=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_p5, initargs=(cal,)) as ex:
        jobs = [(rid, [(int(i), int(o.k[i])) for i in g.index]) for rid, g in o[o.index.isin(rt.obs)].groupby("plus")]
        box_of = dict(x for part in ex.map(box_job, jobs) for x in part)
        box = np.stack([box_of[i] for i in rt.obs])
        rl.info(f"P5 boxes: {int(np.isfinite(box[:, 0]).sum() // 2)} / {len(rt) // 2} pairs, {(time.time() - t0) / 60:.1f} min")
        net = A.load("cinque", torch.float16).cuda()
        feats = {f"{v}_{k}": np.zeros((len(rt), 2048), np.float16) for v in VIEWS for k in ("pool", "box")}
        has = {v: np.zeros(len(rt), bool) for v in VIEWS}
        items = [(i, r.cur, r.prev, box[i]) for i, r in enumerate(rt.itertuples())]
        chunks = [items[s:s + 64] for s in range(0, len(items), 64)]
        for c, res in enumerate(ex.map(p5_job, chunks)):
            ids = np.array([x[0] for x in res])
            for v in VIEWS:
                cl = np.stack([x[2][v] for x in res])
                feats[f"{v}_pool"][ids], feats[f"{v}_box"][ids] = stage3(net, np.stack([x[1][v] for x in res]), cl)
                has[v][ids] = cl.any((1, 2))
            if (c + 1) % 20 == 0:
                rl.info(f"P5 {(c + 1) * 64}/{len(rt)} frames, {(time.time() - t0) / 60:.1f} min")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "p5.npz", obs=rt.obs.to_numpy(), sign=rt.sign.to_numpy(), fn=rt.fn.to_numpy().astype(str),
             route=o.base_id.to_numpy()[rt.obs].astype(str), family=o.family.to_numpy()[rt.obs].astype(str),
             px_eq=o.factor_px.to_numpy()[rt.obs] * (F_COSMOS / F_P5_SEG) ** 2, box=box, **feats,
             **{f"has_{v}": x for v, x in has.items()})
    rl.event("end", frames=len(rt), wall_s=time.time() - t0, boxes=int(np.isfinite(box[:, 0]).sum()))


# ---------------------------------------------------------------- probes and the verdict
def _probe_set(rl, name, X: dict, y, g, px, keep_box, lo_vis=0):
    """X[view_probe] rows -> out-of-fold scores (probe trained on all rows of the set), per px bin AUC and tele - road delta."""
    import op_adapt_readout as R
    sc = {}
    for k, x in X.items():
        m = keep_box if k.endswith("box") else np.ones(len(y), bool)
        s = np.full(len(y), np.nan)
        s[m] = R.oof_probe(x[m].astype(np.float32), y[m], g[m])
        sc[k] = s
    rows = []
    for lo, hi in BINS:
        for pr in ("pool", "box"):
            m = (px >= max(lo, lo_vis)) & (px < hi) & np.isfinite(sc[f"road_{pr}"]) & np.isfinite(sc[f"tele_{pr}"])
            if len(np.unique(y[m])) < 2:
                continue
            r = R.paired_boot(y[m], sc[f"tele_{pr}"][m], sc[f"road_{pr}"][m], g[m])
            rows.append({"set": name, "probe": pr, "px_lo": max(lo, lo_vis), "px_hi": hi, "n": r["n"], "groups": r["groups"],
                         "auc_road": r["auc_ref"], "auc_road_ci": r["auc_ref_ci"], "auc_tele": r["auc"], "auc_tele_ci": r["auc_ci"],
                         "delta": r["delta"], "delta_ci": r["delta_ci"]})
            rl.info(f"{name:13s} {pr:4s} px [{max(lo, lo_vis)},{hi}): n {r['n']:5d} groups {r['groups']:4d}  road {r['auc_ref']:.3f}  "
                    f"tele {r['auc']:.3f}  delta {r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}]")
    return rows


def probe(a, rl):
    rows = []
    z = dict(np.load(OUT / "cosmos.npz", allow_pickle=True))
    two = lambda v: np.r_[v, v]  # noqa: E731
    kb = two(z["has_road"] & z["has_tele"])
    for cell, (ip, im) in {"carla": (0, 1), "cosmos": (2, 3)}.items():
        X = {f"{v}_{p}": np.concatenate([z[f"{v}_{p}"][:, ip], z[f"{v}_{p}"][:, im]]) for v in VIEWS for p in ("pool", "box")}
        y = np.r_[np.ones(len(z["pair"])), np.zeros(len(z["pair"]))].astype(int)
        rows += _probe_set(rl, f"cosmos-{cell}", X, y, two(z["inst"]).astype(str), two(z["px"]).astype(float), kb)
    p = dict(np.load(OUT / "p5.npz", allow_pickle=True))
    X = {f"{v}_{pr}": p[f"{v}_{pr}"] for v in VIEWS for pr in ("pool", "box")}
    kb = np.isfinite(p["box"][:, 0]) & p["has_road"] & p["has_tele"]
    kb = pd.Series(kb).groupby(p["obs"]).transform("all").to_numpy()
    rows += _probe_set(rl, "p5", X, p["sign"].astype(int), p["route"], p["px_eq"], kb, lo_vis=P5_VIS)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "r0.csv", index=False)
    small = df[(df.px_hi == 500) & (df.probe == "pool")]
    verdict = {"rule": "open Z if any set's < 500 px_eq bin has pooled stage-3 AUC (tele) >= 0.65 AND tele - road paired delta CI low > 0",
               "sets": {r.set: {"px": [r.px_lo, r.px_hi], "n": r.n, "groups": r.groups, "auc_road": r.auc_road, "auc_tele": r.auc_tele,
                                "auc_tele_ci": r.auc_tele_ci, "delta": r.delta, "delta_ci": r.delta_ci,
                                "pass": bool(r.auc_tele >= 0.65 and r.delta_ci[0] > 0)} for r in small.itertuples()}}
    verdict["open_Z"] = any(v["pass"] for v in verdict["sets"].values())
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
    rl.info(json.dumps(verdict, default=float))
    rl.event("verdict", open_Z=verdict["open_Z"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cosmos", "p5", "probe"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rl = RunLog("op_adapt_r2", "r0", a.cmd)
    rl.event("start", args=vars(a))
    {"cosmos": cosmos, "p5": p5, "probe": probe}[a.cmd](a, rl)


if __name__ == "__main__":
    main()
