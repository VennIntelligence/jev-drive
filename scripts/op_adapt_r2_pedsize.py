"""op-adapt round 2 pre-registration, measurement M0 (todos/2026-09-29-op-adapt-r2-prereg.md): how large and how far
are the pedestrians in each labelled set, in one common unit. CPU only, no model.

Unit: visible pedestrian mask pixels in the Cosmos camera (1280 x 704, 64 deg HFOV, f = 1024.2 px), the unit of the
E1 size bins (decision 63: < 500 px unreadable in both cells, >= 1500 px readable). Every other camera is scaled by
(1024.2 / f_cam)^2 in area. The openpilot road model frame (f = 910) sees 0.79 x that area, the wide frame 0.20 x.

  P5 v1 BA        pedestrian-scope observation frames of the D0 exam (obs.parquet, 4 414 pairs): factor_px, the
                  visible walker pixels of the half-resolution segmentation view at the Waymo front intrinsics
                  (f = 1113.5 / 2); distance = ego rear axle -> that walker, from the x+ world's actors.npz
  Cosmos G4 full  every finished pair: gt.npz per-frame mask px and box; distance from the pass-2 world's actors.npz;
                  "e1" = E1 read-out slots (5 Hz, slot >= 9, px >= 100), "all" = 5 Hz window frames with the pedestrian
                  visible at P5's threshold (>= 68 equivalent px)
  nuScenes        corridor-pedestrian keyframes (nusc_labels.parquet), train and val: GT distance measured; px derived
                  with a pinhole in the Cosmos camera (height 1.75 m, CAM_FRONT 1.70 m ahead of the rear axle) and the
                  mask / box-height^2 ratio measured on the Cosmos pairs
  WOD train       corridor-pedestrian frames, not uncertain (wod_train_labels.parquet): lifted distance; px measured
                  from the matching YOLO26x front-camera detection's box (w x h scaled, times the Cosmos fill ratio)
  NAVSIM navtrain corridor-pedestrian tokens: GT distance only

  python scripts/op_adapt_r2_pedsize.py --out research/results/op-adapt-r2/pedsize
"""
import argparse, json, math, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import p5_pairs as P  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402

F_COSMOS = 640.0 / math.tan(math.radians(32.0))
F_P5_SEG = 1113.5 / 2
NUSC_CAM_X, PED_H = 1.70, 1.75
REAR_AXLE_X = -1.388633220                     # hero rear axle relative to the actor centre (p4_carla)
PX_BINS = (0, 100, 500, 1500, 6000, np.inf)
D_BINS = (0, 10, 20, 30, np.inf)
PED_FAMS = ("DynamicObjectCrossing", "ParkingCrossingPedestrian", "PedestrianCrossing", "VehicleTurningRoutePedestrian")


def _latest_attempt(gen: Path, rid: str) -> Path | None:
    best, n = None, -1
    for a in sorted((gen / "attempts" / rid).glob("*")):
        f = a / "pose.jsonl"
        if f.exists() and (a / "actors.npz").exists():
            k = sum(1 for _ in open(f))
            if k > n:
                best, n = a, k
    return best


def _walker_dist(pose: pd.DataFrame, act: dict, k: int, ids) -> dict:
    """{walker id: horizontal distance ego rear axle -> walker} at tick k."""
    if k not in pose.index:
        return {}
    e = pose.loc[k]
    yaw = math.radians(float(e.yaw))
    c, s = math.cos(yaw), math.sin(yaw)
    out = {}
    sel = np.flatnonzero((act["k"] == k) & np.isin(act["id"], list(ids)))
    for j in sel:
        dx, dy = act["xyz"][j][0] - e.x, act["xyz"][j][1] - e.y
        fwd, lat = c * dx + s * dy, -s * dx + c * dy
        out[int(act["id"][j])] = float(np.hypot(fwd - REAR_AXLE_X, lat))
    return out


def _p5_job(args):
    gen, rid, ks = args
    a = P.attempt(gen, rid)
    if a is None:
        return [(rid, k, np.nan) for k in ks]
    W = P.load_world(a)
    hz = set(W["hazards"])
    rows = []
    for k in ks:
        px = W["frames"].loc[k].px if k in W["frames"].index else {}
        px = px if isinstance(px, dict) else {}
        d = _walker_dist(W["pose"], W["act"], k, hz)
        vis = {i: px.get(str(i), 0) for i in d}
        i = max(vis, key=vis.get) if vis else None
        rows.append((rid, k, d[i] if i is not None else np.nan))
    return rows


def p5() -> pd.DataFrame:
    d = P.processed()
    assert d.name == "carla_p5v1_ba", d
    o = pd.read_parquet(d / "obs.parquet")
    o = o[o.family.isin(PED_FAMS)].copy()
    pairs = pd.read_csv(d / "pairs.csv", dtype={"base_id": str, "plus": str})
    o = o.merge(pairs[["base_id", "seed", "plus"]], on=["base_id", "seed"], how="left")
    gen = data_dir() / "runs/p5v1/gen-ba"
    jobs = [(gen, rid, sorted(g.k.astype(int).unique())) for rid, g in o.groupby("plus")]
    with Pool(min(48, n_cpus())) as p:
        r = [x for part in p.imap_unordered(_p5_job, jobs) for x in part]
    dist = pd.DataFrame(r, columns=["plus", "k", "dist"])
    o = o.merge(dist, on=["plus", "k"], how="left")
    return pd.DataFrame({"set": "P5 v1 BA (exam)", "family": o.family, "route": o.base_id,
                         "px": o.factor_px * (F_COSMOS / F_P5_SEG) ** 2, "dist": o.dist, "h_box": np.nan})


def _cosmos_job(pdir: Path):
    spec = json.loads((pdir / "spec.json").read_text())
    z = np.load(pdir / "gt.npz")
    px, box = z["px"], z["box"]
    gen = data_dir() / "runs/cosmos_full" / spec.get("gen", "gen")
    a = _latest_attempt(gen, spec["ids"]["plus"])
    dists = np.full(len(px), np.nan)
    if a is not None:
        W = P.load_world(a)
        hz = [h for h, t in zip(W["hazards"], W["hazard_types"]) if str(t).startswith("walker")]
        for j in range(len(px)):
            d = _walker_dist(W["pose"], W["act"], spec["k0"] + j, hz)
            dists[j] = min(d.values()) if d else np.nan
    rows = []
    for j in range(len(px)):
        if j % 4:
            continue
        rows.append({"pair": spec["pair"], "family": spec["family"], "route": str(spec["inst"]), "slot": j // 4,
                     "px": float(px[j]), "h_box": float(box[j][3] - box[j][1]) if px[j] > 0 else np.nan,
                     "w_box": float(box[j][2] - box[j][0]) if px[j] > 0 else np.nan, "dist": dists[j]})
    return rows


def cosmos() -> pd.DataFrame:
    root = data_dir() / "runs/cosmos_full/pairs"
    ds = sorted(p for p in root.iterdir() if (p / "gt.npz").exists() and (p / "spec.json").exists())
    with Pool(min(48, n_cpus())) as p:
        r = [x for part in p.imap_unordered(_cosmos_job, ds) for x in part]
    return pd.DataFrame(r)


def real(fill: float, k_h2: float) -> list[pd.DataFrame]:
    op = data_dir() / "processed/op_adapt"
    out = []
    nu = pd.read_parquet(op / "nusc_labels.parquet")
    for split in ("train", "val"):
        s = nu[(nu.split == split) & nu.ped_corr]
        h = F_COSMOS * PED_H / np.maximum(s.ped_dist.to_numpy() - NUSC_CAM_X, 1.0)
        out.append(pd.DataFrame({"set": f"nuScenes {split} (px derived)", "family": "real", "route": s.scene,
                                 "px": k_h2 * h ** 2, "dist": s.ped_dist.to_numpy(), "h_box": h}))
    w = pd.read_parquet(op / "wod_train_labels.parquet")
    w = w[w.ped_corr & ~w.uncertain]
    y = data_dir() / "processed/real_transfer/yolo/wod_train"
    d = pd.read_parquet(y / "dets.parquet", columns=["frame_id", "cam", "prompt", "score", "x0", "y0", "x1", "y1", "area",
                                                      "gx", "gy", "lift_ok", "h_feat"])
    d = d[(d.prompt == "pedestrian") & d.lift_ok & d.frame_id.isin(set(w.frame_id))]
    d["r"] = np.hypot(d.gx, d.gy)
    d = d.merge(w[["frame_id", "ped_dist", "sequence"]], on="frame_id")
    d = d[np.abs(d.r - d.ped_dist) < 1e-3].sort_values("score", ascending=False).drop_duplicates("frame_id")
    d = d[d.cam == "front"]                    # the corridor pedestrian as openpilot's front view sees it
    cal = json.loads((data_dir() / "processed/carla_p5v1_ba/op_plan.json").read_text())["calib"]["1"]
    fv = (d.y1 - d.y0) * float(cal["intrinsic"][1]) / float(cal["height"]) / d.h_feat
    sc = F_COSMOS / fv
    wd = pd.DataFrame({"set": "WOD train (YOLO box)", "family": "real", "route": d.sequence, "cam": d.cam,
                       "px": fill * (d.x1 - d.x0) * (d.y1 - d.y0) * sc ** 2, "dist": d.ped_dist,
                       "h_box": (d.y1 - d.y0) * sc, "mask_area_raw": d.area,
                       "mask_over_box_raw": d.area / ((d.x1 - d.x0) * (d.y1 - d.y0)), "f_cam": fv})
    out.append(wd)
    n = pd.read_parquet(op / "navtrain_labels.parquet")
    n = n[n.ped_corr]
    out.append(pd.DataFrame({"set": "NAVSIM navtrain (distance only)", "family": "real", "route": n["map"],
                             "px": np.nan, "dist": n.ped_dist, "h_box": np.nan}))
    return out


def summarise(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bins = [], []
    for s, g in t.groupby("set", sort=False):
        px, dd = g.px.dropna(), g.dist.dropna()
        r = {"set": s, "n": len(g), "routes": g.route.nunique()}
        if len(px):
            r.update({f"px_p{q}": float(np.percentile(px, q)) for q in (10, 25, 50, 75, 90)})
            for lo, hi in zip(PX_BINS[:-1], PX_BINS[1:]):
                r[f"px_{lo}-{hi}"] = float(((px >= lo) & (px < hi)).mean())
        if len(dd):
            r.update({f"dist_p{q}": float(np.percentile(dd, q)) for q in (10, 50, 90)})
            for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
                r[f"d_{lo}-{hi}"] = float(((dd >= lo) & (dd < hi)).mean())
        rows.append(r)
        if len(px) and len(dd):
            gg = g.dropna(subset=["px", "dist"])
            for lo, hi in zip(D_BINS[:-1], D_BINS[1:]):
                m = (gg.dist >= lo) & (gg.dist < hi)
                if m.sum():
                    bins.append({"set": s, "dist_bin": f"{lo}-{hi}", "n": int(m.sum()),
                                 "px_median": float(gg.px[m].median()), "px_p10": float(gg.px[m].quantile(.1)),
                                 "px_p90": float(gg.px[m].quantile(.9))})
    return pd.DataFrame(rows), pd.DataFrame(bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/results/op-adapt-r2/pedsize")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    c = cosmos()
    v = c[c.px > 0]
    fill = float((v.px / (v.h_box * v.w_box)).median())
    k_h2 = float((v.px / v.h_box ** 2).median())
    c_all = c[c.px >= 20 * (F_COSMOS / F_P5_SEG) ** 2].assign(set="Cosmos G4 full, all visible 5 Hz frames")
    c_e1 = c[(c.slot >= 9) & (c.px >= 100)].assign(set="Cosmos G4 full, E1 read-out slots")
    t5 = p5()
    fam = t5.groupby("family").agg(n=("px", "size"), px_median=("px", "median"), dist_median=("dist", "median"),
                                   px_lt500=("px", lambda x: float((x < 500).mean())))
    parts = [t5, c_e1, c_all] + real(fill, k_h2)
    t = pd.concat(parts, ignore_index=True)
    summ, bins = summarise(t)
    summ.to_csv(out / "summary.csv", index=False)
    bins.to_csv(out / "px_by_dist.csv", index=False)
    fam.to_csv(out / "p5_by_family.csv")
    wod = [p for p in parts if len(p) and p.set.iloc[0].startswith("WOD")][0]
    meta = {"F_COSMOS": F_COSMOS, "area_scale_p5": (F_COSMOS / F_P5_SEG) ** 2, "cosmos_fill_mask_over_box": fill,
            "cosmos_mask_over_h2": k_h2, "cosmos_pairs": int(c.pair.nunique()),
            "wod_mask_over_box_raw_median": float(wod.mask_over_box_raw.median()),
            "wod_f_cam_median": float(wod.f_cam.median()), "wod_cams": wod.cam.value_counts().to_dict(),
            "p5_rows_with_dist": int(t5.dist.notna().sum()), "p5_rows": len(t5)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 250)
    print(json.dumps(meta, indent=1))
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"))
    print(bins.to_string(float_format=lambda x: f"{x:.0f}"))
    print(fam.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
