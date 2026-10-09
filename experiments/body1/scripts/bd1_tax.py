"""BODY1 scene taxonomy on navtrain (plans/2026-10-10-body1-prereg.md, 2.1): two tags per token from logged quantities only, the three user
classes, city and the body1 log split. CPU.

  labels   the agent / SDF label npz files -> memory-mapped .npy under $DATA_DIR/runs/body1/labels/ (once)
  splits   --logs FILE   register navsim/body1-hold-logs (sha256(log) % 10 == 0) and navsim/body1-train-logs (run where the repo is committed)
  tax      [--shards 0] -> $DATA_DIR/runs/body1/taxonomy/tax[-s0].npz (one row per navtrain token, global row order) + count tables (csv)

Manoeuvre (first match): launch (v0 < 1 m/s, 4 s arc > 2 m), stop (end speed < 1 m/s from v0 > 2), turn 20-45 / > 45 deg (|logged 4 s heading
change|, bench TURN_BINS), go-around (< 20 deg and the logged sweep passes an object that the reference sweep hits), straight (the rest).
  reference sweep = the lane-following proxy: the constant-curvature continuation of the motion at t0 (curvature from the logged pose 0.5 s
  earlier, |kappa| <= 0.2 / m; 0 below 0.5 m of travel), driven with the logged arc-length profile, objects at matching times. An object is
  gone around when the reference sweep touches it, the logged sweep does not, it starts ahead of the ego front and ends abeam or behind.
Hazard flags (logged sweep, ego box 5.176 x 2.297 m, objects at matching times with their motion):
  in-path   an object, while its nearest point is ahead of the ego front, lies inside the part of the logged sweep (ego box widened by 0.3 m
            per side) that the ego reaches later (annotated object times 0, 0.5 .. 4 s against the 0.1 s sweep)
  side      an object not in-path overlaps the ego longitudinally with a lateral gap < 1.0 m at some step
  boundary  drivable margin of the logged sweep (min corner SDF over the steps inside the raster) < 0.5 m
Primary hazard = in-path > side > boundary > none. User classes (bits; primary class = the first that holds):
  1 obstacle ahead     in-path or go-around
  2 turn               turn 20-45 / > 45 with a side or boundary flag, plus every token with |heading change| > 45 deg
  3 leaving the road   straight / launch with the boundary flag, or straight with no flag
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402
import sweep as SW  # noqa: E402

CITY = {"us-nv-las-vegas-strip": "Las Vegas", "us-ma-boston": "Boston", "us-pa-pittsburgh-hazelwood": "Pittsburgh", "sg-one-north": "Singapore"}
WIDEN, SIDE_GAP, BND_MARGIN, KAPPA_MAX = 0.3, 1.0, 0.5, 0.2


def ref_sweep(hist, s):
    """Constant-curvature continuation of the motion at t0: hist (N, 3) logged pose 0.5 s before t0 (t0 frame), s (N, 41) arc -> (N, 41, 3)."""
    ds = np.hypot(hist[:, 0], hist[:, 1])
    k = np.clip(np.where(ds > 0.5, -hist[:, 2] / np.maximum(ds, 0.5), 0.0), -KAPPA_MAX, KAPPA_MAX)[:, None]
    th = k * s
    small = np.abs(th) < 1e-4
    ths = np.where(small, 1.0, th)
    return np.stack([np.where(small, s, s * np.sin(ths) / ths), np.where(small, 0.5 * s * th, s * (1 - np.cos(ths)) / ths), th], -1).astype(np.float32)


def tax_chunk(fut, hist, v0, box, valid, cls, sdf) -> dict:
    """One chunk of tokens -> per-token taxonomy columns."""
    N = len(fut)
    ok = ~np.isnan(fut).any((1, 2))
    f = np.nan_to_num(fut).astype(np.float32)
    d = SW.dense(f)
    s = SW.arc(d)
    arc4, v_end = s[:, -1], (s[:, -1] - s[:, -6]) / 0.5
    dyaw = np.degrees(np.unwrap(np.concatenate([np.zeros((N, 1)), f[..., 2]], 1), axis=1)[:, -1])
    b, v, _ = SW.dense_boxes(box, valid)
    has = (cls >= 0)[:, None]
    v &= has
    ex, ey = SW.ego_centre(d)
    hl, hw = b[..., 3] / 2, b[..., 4] / 2
    clr, hit = SW.clearance(ex[..., None], ey[..., None], d[..., 2:3], b[..., 0], b[..., 1], b[..., 2], hl, hw)          # (N, 41, K) matching times
    clr, hit = np.where(v, clr, SW.BIG), hit & v
    # in-path: object at an annotated time against the widened ego boxes of later steps
    v9 = valid & has
    ahead9 = SW.nearest_lon(d[:, ::5, None, :], box) > SW.HALF_L
    hit9 = SW.sat(ex[:, None, :, None], ey[:, None, :, None], d[:, None, :, None, 2], SW.HALF_L, SW.HALF_W + WIDEN,
                  box[:, :, None, :, 0], box[:, :, None, :, 1], box[:, :, None, :, 2], box[:, :, None, :, 3] / 2, box[:, :, None, :, 4] / 2)   # (N, 9, 41, K)
    later = (np.arange(SW.NS)[None] > 5 * np.arange(9)[:, None])[None, :, :, None]
    inpath_k = (v9 & ahead9 & (hit9 & later).any(2)).any(1)                                                               # (N, K)
    disp = np.where(v9.any(1), np.hypot(*(box[:, :, :, :2].max(1, where=v9[..., None], initial=-1e9) - box[:, :, :, :2].min(1, where=v9[..., None], initial=1e9)).transpose(2, 0, 1)), 0)
    # side: longitudinal overlap and a lateral gap < 1 m at some step
    px, py = SW.corners(b[..., 0], b[..., 1], b[..., 2], hl, hw)
    lon, lat = SW.ego_frame(d[:, :, None, None, :], px, py)
    gap = np.maximum(lat.min(-1) - SW.HALF_W, -lat.max(-1) - SW.HALF_W)
    side_k = (v & (lon.max(-1) > -SW.HALF_L) & (lon.min(-1) < SW.HALF_L) & (gap < SIDE_GAP)).any(1) & ~inpath_k
    # go-around: the reference sweep hits it, the logged sweep does not, it starts ahead and ends abeam or behind
    r = ref_sweep(hist, s)
    rx, ry = SW.ego_centre(r)
    hit_ref = SW.sat(rx[..., None], ry[..., None], r[..., 2:3], SW.HALF_L, SW.HALF_W, b[..., 0], b[..., 1], b[..., 2], hl, hw) & v
    ahead = SW.nearest_lon(d[:, :, None, :], b) > SW.HALF_L
    was_ahead = np.maximum.accumulate(v & ahead, 1)
    ga_k = hit_ref.any(1) & ~hit.any(1) & (v & ~ahead & was_ahead).any(1)
    m, ins = SW.corner_margins(d[:, None], sdf)
    cov = ins.all(-1)[:, 0]
    margin = np.where(cov, m.min(-1)[:, 0], SW.BIG).min(1)
    f_in, f_side, f_bnd = inpath_k.any(1), side_k.any(1), margin < BND_MARGIN
    ad = np.abs(dyaw)
    man = np.select([~ok, (v0 < 1) & (arc4 > 2), (v_end < 1) & (v0 > 2), (ad >= 20) & (ad < 45), ad >= 45, ga_k.any(1)], [6, 0, 1, 2, 3, 4], 5).astype(np.int8)
    haz = np.select([f_in, f_side, f_bnd], [0, 1, 2], 3).astype(np.int8)
    c1 = ok & (f_in | (man == 4))
    c2 = ok & ((np.isin(man, (2, 3)) & (f_side | f_bnd)) | (ad > 45))
    c3 = ok & ((np.isin(man, (5, 0)) & f_bnd) | ((man == 5) & (haz == 3)))
    return dict(ok=ok, v0=v0.astype(np.float32), arc=arc4.astype(np.float32), v_end=v_end.astype(np.float32), dyaw=dyaw.astype(np.float32), man=man, haz=haz,
                f_in=f_in & ok, f_side=f_side & ok, f_bnd=f_bnd & ok, f_ga=ga_k.any(1) & ok, in_static=(inpath_k & (disp < 0.5)).any(1) & ok,
                still=ok & (v0 < 1) & (arc4 <= 2), n_obj=v9[:, 0].sum(1).astype(np.int8), clr_log=clr.min((1, 2)).astype(np.float32),
                hit_log=hit.any((1, 2)) & ok, margin_log=margin.astype(np.float32), cov_log=cov.sum(1).astype(np.uint8),
                c1=c1, c2=c2, c3=c3, pc=np.select([c1, c2, c3], [1, 2, 3], 0).astype(np.int8))


def work(k):
    L, t, o = B.labels(), B.tab("log", k), B.shard_offsets()
    out = []
    for i in range(0, len(t["names"]), 128):
        j = slice(i, min(i + 128, len(t["names"])))
        g = slice(o[k] + j.start, o[k] + j.stop)
        out.append(tax_chunk(t["fut"][j], t["pose"][j, 2], t["speed"][j], np.asarray(L["box"][g]), np.asarray(L["valid"][g]), np.asarray(L["cls"][g]),
                             np.asarray(L["sdf"][g])))
    return {q: np.concatenate([c[q] for c in out]) for q in out[0]}


def log_city(log):
    fr = pickle.load(open(B.data_dir() / "datasets/navsim/navsim_logs/trainval" / f"{log}.pkl", "rb"))
    return CITY.get(fr[0]["map_location"], fr[0]["map_location"])


def counts(T, by, where=None) -> list:
    """Rows of a count table: one per combination of the integer / string columns `by` among tokens where `where`."""
    import pandas as pd
    d = pd.DataFrame({q: T[q] for q in by})
    d = d[where] if where is not None else d
    return d.value_counts(sort=False).rename("n").reset_index().to_dict("records")


def cmd_tax(a):
    from jevdrive import par, stats
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    shards = a.shards if a.shards is not None else list(range(B.NSH))
    tag = "tax" + ("" if a.shards is None else "-s" + "".join(map(str, shards)))
    with Run("body1", tag, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        for x in (B.HOLD, B.TRAIN):
            try:
                run.use_split(splits.load(x))
            except Exception as e:  # the split files arrive with the next pull; membership is the hash rule either way
                run.info(f"{x} not registered here yet ({type(e).__name__})")
        B.build_labels(run)
        res = par.pmap(work, shards, run=run, workers=min(a.workers or n_cpus(), 40, len(shards)))
        res.raise_if_failed()
        T = {q: np.concatenate([r[q] for r in res.values]) for q in res.values[0]}
        o = B.shard_offsets()
        gi = np.concatenate([np.arange(o[k], o[k + 1]) for k in shards])
        L = B.labels()
        logs = L["log"][gi]
        ul = np.unique(logs)
        rc = par.pmap(log_city, ul.tolist(), run=run, workers=min(a.workers or n_cpus(), 40), desc="city")
        rc.raise_if_failed()
        city = dict(zip(ul.tolist(), rc.values))
        T |= dict(gi=gi, token=L["tokens"][gi], log=logs, city=np.array([city[g] for g in logs]), hold=np.array([B.is_hold(g) for g in logs]),
                  shard=np.concatenate([np.full(o[k + 1] - o[k], k, np.int8) for k in shards]))
        out = B.root() / "taxonomy"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{tag}.tmp.npz", **T)
        (out / f"{tag}.tmp.npz").replace(out / f"{tag}.npz")
        T["man_s"], T["haz_s"], T["cls_s"] = (np.array(n)[T[q]] for n, q in ((B.MAN, "man"), (B.HAZ, "haz"), (B.CLS, "pc")))
        T["split"] = np.where(T["hold"], "hold", "train")
        T["gt45"] = np.abs(T["dyaw"]) > 45
        res_d = run.path("tables", "x").parent
        tabs = {"cells": ["man_s", "haz_s"], "cells_city": ["man_s", "haz_s", "city"], "cells_split": ["man_s", "haz_s", "split"],
                "flags": ["man_s", "f_in", "f_side", "f_bnd", "f_ga"], "classes": ["cls_s", "c1", "c2", "c3", "split"], "classes_city": ["cls_s", "city", "split"],
                "class_cells": ["cls_s", "man_s", "haz_s"], "gt45": ["gt45", "man_s", "haz_s", "f_in", "f_side", "f_bnd"], "still": ["cls_s", "still"]}
        for name, by in tabs.items():
            stats.write_table(counts(T, by), res_d / name)
        n = len(gi)
        run.summary.update(n=n, n_logs=len(ul), hold_tokens=int(T["hold"].sum()), hold_logs=int(sum(B.is_hold(g) for g in ul)), no_future=int((~T["ok"]).sum()),
                           classes={c: int((T["pc"] == i).sum()) for i, c in enumerate(B.CLS)}, c1=int(T["c1"].sum()), c2=int(T["c2"].sum()), c3=int(T["c3"].sum()),
                           gt45=int(T["gt45"].sum()), man={m: int((T["man"] == i).sum()) for i, m in enumerate(B.MAN)},
                           haz={h: int((T["haz"] == i).sum()) for i, h in enumerate(B.HAZ)}, logged_agent_contact=float(T["hit_log"].mean()),
                           logged_boundary_contact=float((T["margin_log"] < 0).mean()), out=str(out / f"{tag}.npz"), tables=str(res_d))
        run.info(str(run.summary))


def cmd_splits(a):
    from jevdrive.data import splits
    logs = sorted({x.strip() for x in open(a.logs) if x.strip()})
    origin = f"navsim navtrain logs ({len(logs)}, the logs of the op_parity navtrain_full caches); hold = int(sha256(log), 16) % 10 == 0"
    kw = dict(unit="log", origin=origin, used_by=["experiments/body1"], status="frozen")
    h = splits.define("navsim", "body1-hold-logs", [g for g in logs if B.is_hold(g)], notes="BODY1 held-out logs (G1 gate, thresholds); superset of op-parity-full-dev's logs", **kw)
    t = splits.define("navsim", "body1-train-logs", [g for g in logs if not B.is_hold(g)], notes="BODY1 predictor training logs", **kw)
    splits.check_disjoint(h, t)
    print(h.id, len(h.members), t.id, len(t.members))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("labels")
    p = sp.add_parser("splits")
    p.add_argument("--logs", required=True)
    p = sp.add_parser("tax")
    p.add_argument("--shards", type=int, nargs="+", default=None)
    p.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    {"labels": lambda a: B.build_labels(), "splits": cmd_splits, "tax": cmd_tax}[a.cmd](a)
