#!/usr/bin/env python
"""Export nuScenes open-loop failure cases and error breakdowns for the visual review page. No model inference.

Reads (all existing, CPU only):
  $DATA_DIR/processed/nusc_zs/val_index.pkl, sets.json          the exam's index and evaluation sets (decision 39)
  $DATA_DIR/runs/nusc_zs/preds/{op_cinque_none,cv}.npz          cached predictions (LIDAR_TOP point, 6 x 0.5 s)
  $DATA_DIR/runs/nusc_zs/score/{per_sample.pkl,results.csv}     stored per-frame metrics, used as the check
  $DATA_DIR/runs/nusc_backbones/ladder/*/nusc_preds.npz         frozen-feature + ridge head predictions (decision 40)
  nuScenes CAM_FRONT keyframes, calibration and the map expansion
Writes $DATA_DIR/runs/openloop_visual_review_20261002/nusc/: stats.json (checks + tables), cases.json, jpg frames.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python experiments/zeroshot_openloop/scripts/nusc_review_export.py
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.common import data_dir, dataroot  # noqa: E402

OUT = data_dir() / "runs/openloop_visual_review_20261002/nusc"
OUT.mkdir(parents=True, exist_ok=True)
SET, MODEL, ROW = "main", "op_cinque_none", "openpilot Cinque"
PATCH = 60.0  # map half-size around the ego, metres


def metrics(pred, samples):
    h = Z.horizons(Z.per_sample(pred, samples))
    h["l2_avg"] = (h["l2_1s"] + h["l2_2s"] + h["l2_3s"]) / 3
    h["l2pt_avg"] = (h["l2pt_1s"] + h["l2pt_2s"] + h["l2pt_3s"]) / 3
    h["col_bevp_avg"] = (h["col_bevp_1s"] + h["col_bevp_2s"] + h["col_bevp_3s"]) / 3
    h["col_vad_avg"] = (h["col_vad_1s"] + h["col_vad_2s"] + h["col_vad_3s"]) / 3
    return h


def rear_path(xy, yaw, d):
    """Scored LIDAR_TOP-point path -> rear-axle path (inverse of Z.to_lidar_point at the same steps)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([xy[:, 0] - (c * d[0] - s * d[1]) + d[0], xy[:, 1] - (s * d[0] + c * d[1]) + d[1]], -1)


def hit_boxes(xy, boxes):
    """Per step: indices of the agent boxes the BEV-Planner ego box overlaps, and the ego box corners."""
    yaw = Z.traj_yaw(xy)
    hits, egos = [], []
    for k, b in enumerate(boxes):
        cx, cy = xy[k, 0] + Z.EGO_DX * np.cos(yaw[k]), xy[k, 1] + Z.EGO_DX * np.sin(yaw[k])
        ego = Z._corners(np.array(cx), np.array(cy), np.array(Z.EGO_L), np.array(Z.EGO_W), np.array(yaw[k]))
        egos.append(ego.round(3).tolist())
        hits.append(np.where(Z._overlap(ego, Z._corners(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4])))[0].tolist()
                    if len(b) else [])
    return hits, egos


def main():
    idx = Z.load_index()
    sets = json.load(open(Z.index_path().with_name("sets.json")))
    by = {e["token"]: e for e in idx["samples"]}
    toks = sets[SET]
    S = [by[t] for t in toks]
    load = lambda n: (lambda z: dict(zip(z["tokens"].tolist(), z["pred"])))(np.load(Z.root("preds") / f"{n}.npz"))  # noqa: E731
    P = {k: np.stack([load(n)[t] for t in toks]).astype(np.float64) for k, n in (("op", MODEL), ("cv", "cv"))}
    H = {k: metrics(p, S) for k, p in P.items()}
    gt = np.stack([e["gt"] for e in S]).astype(np.float64)

    # ---- check against the stored per-frame values and the stored table
    stored = pickle.load(open(Z.root("score") / "per_sample.pkl", "rb"))
    assert stored["sets"][SET] == toks
    check = {}
    for k, row in (("op", ROW), ("cv", "cv")):
        sv = stored["values"][(SET, row)]
        check[row] = {m: {"recomputed": float(H[k][m].mean()), "stored": float(sv[m].mean()),
                          "max_abs_frame_diff": float(np.abs(H[k][m] - sv[m]).max())}
                      for m in ("l2_1s", "l2_2s", "l2_3s", "l2_avg", "l2pt_avg", "col_bevp_avg", "col_vad_avg",
                                "col_bevp_3s")}

    # ---- per-frame descriptors
    v0 = np.array([Z.cv_speed(idx["scenes"][e["scene"]], e["t0"]) for e in S])
    cmd = np.array([e["cmd"] for e in S])
    v_end = np.linalg.norm(gt[:, 5] - gt[:, 4], axis=-1) / 0.5
    dist3 = np.linalg.norm(gt[:, 5], axis=-1)
    dv = v_end - v0
    lon = {k: P[k][:, 5, 0] - gt[:, 5, 0] for k in P}
    lat = {k: P[k][:, 5, 1] - gt[:, 5, 1] for k in P}
    pd3 = np.linalg.norm(P["op"][:, 5], axis=-1)
    col = {k: H[k]["col_bevp_3s"] > 0 for k in P}

    def row(name, m):
        r = {"group": name, "n": int(m.sum())}
        if not m.any():
            return r
        for k in P:
            r |= {f"{k}_l2_avg": float(H[k]["l2_avg"][m].mean()), f"{k}_l2pt_3s": float(H[k]["l2pt_3s"][m].mean()),
                  f"{k}_lon3": float(lon[k][m].mean()), f"{k}_abslat3": float(np.abs(lat[k][m]).mean()),
                  f"{k}_col_avg": float(H[k]["col_bevp_avg"][m].mean()), f"{k}_col3_n": int(col[k][m].sum())}
        r["share_of_excess_l2"] = float((H["op"]["l2_avg"] - H["cv"]["l2_avg"])[m].sum()
                                        / (H["op"]["l2_avg"] - H["cv"]["l2_avg"]).sum())
        return r

    allm = np.ones(len(S), bool)
    stop0 = v0 < 0.5
    groups = {
        "command": [("all", allm)] + [(c, cmd == c) for c in ("straight", "left", "right")],
        "speed": [(n, (v0 >= a) & (v0 < b)) for n, a, b in
                  (("< 0.5 m/s", 0, .5), ("0.5-3", .5, 3), ("3-6", 3, 6), ("6-10", 6, 10), (">= 10", 10, 99))],
        "scenario": [("stopped, stays stopped (GT moves < 0.5 m in 3 s)", stop0 & (dist3 < 0.5)),
                     ("stopped, GT pulls away (>= 0.5 m)", stop0 & (dist3 >= 0.5)),
                     ("moving, GT slows (dv <= -1.5 m/s)", ~stop0 & (dv <= -1.5)),
                     ("moving, GT steady (|dv| < 1.5)", ~stop0 & (np.abs(dv) < 1.5)),
                     ("moving, GT speeds up (dv >= 1.5)", ~stop0 & (dv >= 1.5))],
        "location": [(l, np.array([e["location"] == l for e in S])) for l in sorted({e["location"] for e in S})]}
    tables = {k: [row(n, m) for n, m in g] for k, g in groups.items()}
    slow = ~stop0 & (dv <= -1.5)
    turn = cmd != "straight"
    pairs = {"both": int((col["op"] & col["cv"]).sum()), "cv_only": int((~col["op"] & col["cv"]).sum()),
             "op_only": int((col["op"] & ~col["cv"]).sum()), "neither": int((~col["op"] & ~col["cv"]).sum())}
    where = lambda m: {"n": int(m.sum()), "turn_cmd": int((m & turn).sum()), "straight_gt_slows": int((m & ~turn & slow).sum()),  # noqa: E731
                       "straight_stopped_start": int((m & ~turn & stop0).sum()),
                       "straight_other": int((m & ~turn & ~slow & ~stop0).sum())}
    col_where = {"cv_only": where(~col["op"] & col["cv"]), "op_only": where(col["op"] & ~col["cv"]),
                 "both": where(col["op"] & col["cv"])}
    lon_q = {k: np.percentile(lon[k], [5, 25, 50, 75, 95]).tolist() for k in P}
    sign = {"op_overshoot_gt1m": int((lon["op"] > 1).sum()), "op_undershoot_lt-1m": int((lon["op"] < -1).sum()),
            "cv_overshoot_gt1m": int((lon["cv"] > 1).sum()), "cv_undershoot_lt-1m": int((lon["cv"] < -1).sum()),
            "op_better_l2_frames": int((H["op"]["l2_avg"] < H["cv"]["l2_avg"]).sum())}
    first_col = lambda k: np.where(H[k]["col_bevp_3s"] > 0, np.argmax(np.stack(  # noqa: E731
        [Z.collisions(P[k][n], S[n]["boxes"], True) & ~Z.collisions(gt[n], S[n]["boxes"], True) for n in range(len(S))]), 1), 9)
    # speed scaling: ratio of predicted to logged 3 s distance on moving, straight frames
    mv = ~stop0 & ~turn & (dist3 > 5)
    ratio = {"n": int(mv.sum()), "median_pred_over_gt_dist3": float(np.median(pd3[mv] / dist3[mv])),
             "median_cv_over_gt_dist3": float(np.median(np.linalg.norm(P["cv"][mv, 5], axis=-1) / dist3[mv]))}

    ratio |= {"n_pred_under_half_gt": int((pd3[mv] < 0.5 * dist3[mv]).sum()),
              "n_pred_over_1p5_gt": int((pd3[mv] > 1.5 * dist3[mv]).sum())}
    # where the ego box is relative to the logged ego at the first collision step (non-reactive logged agents)
    fcs = {k: first_col(k) for k in P}

    def col_kind(k):
        n_ = np.where(col[k])[0]
        dl = np.array([P[k][n, fcs[k][n], 0] - gt[n, fcs[k][n], 0] for n in n_])
        return {"n": len(n_), "behind_log_le_-1m": int((dl <= -1).sum()), "ahead_of_log_ge_1m": int((dl >= 1).sum()),
                "within_1m_lon": int((np.abs(dl) < 1).sum())}
    col_kind_ = {k: col_kind(k) for k in P}
    sg = np.array([e["location"].startswith("singapore") for e in S])
    tables["turn_by_side"] = [row(f"{c} turn, {n}", (cmd == c) & m) for c in ("left", "right")
                              for n, m in (("Boston (drive on the right)", ~sg), ("Singapore (drive on the left)", sg))]

    # ---- frozen-feature head (decision 40 follow-up): its own protocol, reproduced from its cache
    lz = np.load(sorted(glob.glob(str(data_dir() / "runs/nusc_backbones/ladder/*/nusc_preds.npz")))[-1])
    lt = lz["token"].astype(str)
    fut = lz["fut"]
    ade = lambda p: float(np.linalg.norm(p - fut, axis=-1).mean())  # noqa: E731
    lv0 = np.array([Z.cv_speed(idx["scenes"][by[t]["scene"]], by[t]["t0"]) for t in lt])
    tt = np.arange(1, 13) * 0.25
    lcv = np.stack([lv0[:, None] * tt[None], 0 * lv0[:, None] * tt[None]], -1)
    k6 = np.arange(1, 12, 2)      # 0.5 ... 3.0 s
    stp3 = lambda p: float(np.mean([np.linalg.norm(p[:, k6] - fut[:, k6], axis=-1)[:, :n].mean(1) for n in (2, 4, 6)]))  # noqa: E731
    ladder = {"n": len(lt), "in_main": int(np.isin(lt, toks).sum()),
              "ade": {k[5:]: ade(lz[k]) for k in lz.files if k.startswith("pred_")}, "ade_cv_rear_axle": ade(lcv),
              "stp3_style_l2_avg_rear_axle": {**{k[5:]: stp3(lz[k]) for k in
                                                 ("pred_op-cinque temporal", "pred_ridge ego", "pred_native op-cinque (no fit)",
                                                  "pred_op-cinque temporal (no cmd)")}, "cv": stp3(lcv)}}
    lpos = {t: i for i, t in enumerate(lt)}

    # ---- case selection: fixed rules, one frame per rule, distinct scenes
    l2 = H["op"]["l2_avg"]
    fc = fcs
    nocol = ~col["op"] & ~col["cv"]
    rules = [
        ("overshoot", "failure", "speed profile: keeps speed while the log brakes",
         "straight, v0 > 3 m/s, GT slows, no collision for either; largest positive 3 s longitudinal error",
         ~turn & (v0 > 3) & slow & nocol, lon["op"]),
        ("undershoot", "failure", "speed profile: slower than the log",
         "straight, v0 > 3 m/s, no collision for either; most negative 3 s longitudinal error",
         ~turn & (v0 > 3) & nocol, -lon["op"]),
        ("phantom_start", "failure", "stationary start: predicts pulling away, log stays stopped",
         "v0 < 0.5 m/s, GT moves < 0.5 m in 3 s; largest predicted 3 s displacement", stop0 & (dist3 < 0.5), pd3),
        ("missed_start", "failure", "stationary start: stays put while the log pulls away",
         "v0 < 0.5 m/s, GT moves >= 3 m in 3 s; smallest predicted / logged 3 s distance", stop0 & (dist3 >= 3), -pd3 / dist3),
        ("turn_miss", "failure", "turn: lateral miss",
         "turn command (|GT lateral offset at 3 s| >= 2 m), no collision for either; largest |lateral error| at 3 s",
         turn & nocol, np.abs(lat["op"])),
        ("col_straight", "failure", "collision on a straight frame (CV does not collide)",
         "straight, Cinque collides within 3 s (BEV-Planner rule), CV does not; earliest collision step, then larger v0",
         ~turn & col["op"] & ~col["cv"], -fc["op"] * 100 + v0),
        ("col_turn", "failure", "collision on a turn frame",
         "turn command, Cinque collides within 3 s; earliest collision step, then larger v0",
         turn & col["op"], -fc["op"] * 100 + v0),
        ("cv_col_lead", "control", "contrast: CV collides, Cinque does not (straight, log brakes)",
         "straight, GT slows, CV collides within 3 s, Cinque does not; largest v0", ~turn & slow & col["cv"] & ~col["op"], v0),
        ("cv_col_turn", "control", "contrast: CV collides, Cinque does not (turn)",
         "turn command, CV collides within 3 s, Cinque does not; largest v0", turn & col["cv"] & ~col["op"], v0),
        ("control_straight", "control", "normal control, straight",
         "straight, v0 > 5 m/s, no collision; L2 closest to the median L2 of that group",
         ~turn & (v0 > 5) & nocol, None),
        ("control_turn", "control", "normal control, turn",
         "turn command, v0 > 3 m/s, no collision; smallest L2", turn & (v0 > 3) & nocol, -l2)]
    from nuscenes.map_expansion.map_api import NuScenesMap
    from PIL import Image
    from shapely.geometry import box as sbox
    maps, seen, cases = {}, set(), []
    for cid, kind, ftype, rule, m, key in rules:
        if key is None:
            key = -np.abs(l2 - np.median(l2[m]))
        order = [n for n in np.argsort(-np.where(m, key, -np.inf), kind="stable") if m[n]]
        n = next((n for n in order if S[n]["scene"] not in seen), None)
        if n is None:
            print("no candidate", cid, flush=True)
            continue
        e = S[n]
        sc = idx["scenes"][e["scene"]]
        seen.add(e["scene"])
        cam = sc["cams"]["CAM_FRONT"]
        d = sc["lidar_xyz"]
        # camera frames: the keyframe at t0 and the six logged future keyframes (log replay, not a model rollout)
        frames = []
        for j, t in enumerate(np.r_[0.0, e["fut_t"]]):
            f = Z.frame_at(sc, "CAM_FRONT", e["t0"] + t * 1e6)
            im = Image.open(dataroot() / cam["path"][f]).convert("RGB").resize((960, 540), Image.LANCZOS)
            fn = f"nusc-{len(cases) + 1:02d}-f{j}.jpg"
            im.save(OUT / fn, quality=85)
            frames.append({"file": fn, "t": float(t), "src": cam["path"][f], "key": bool(cam["key"][f])})
        # map layers around the ego, in the t0 ego frame
        if sc["location"] not in maps:
            maps[sc["location"]] = NuScenesMap(str(dataroot()), sc["location"])
        nm = maps[sc["location"]]
        xyz, R = Z.ego_at(sc, e["t0"])
        o, R2 = xyz[0, :2], R[0][:2, :2]
        loc = lambda p: ((np.asarray(p)[:, :2] - o) @ R2).round(2).tolist()  # noqa: E731
        patch = sbox(o[0] - PATCH, o[1] - PATCH, o[0] + PATCH, o[1] + PATCH)
        recs = nm.get_records_in_patch(patch.bounds, ["drivable_area", "road_divider", "lane_divider", "ped_crossing"],
                                       mode="intersect")
        polys, lines = [], []
        for layer in ("drivable_area", "ped_crossing"):
            for tok in recs[layer]:
                r = nm.get(layer, tok)
                for pt in r.get("polygon_tokens", [r.get("polygon_token")]):
                    g = nm.extract_polygon(pt).intersection(patch)
                    for q in (list(g.geoms) if hasattr(g, "geoms") else [g]):
                        if q.geom_type == "Polygon" and not q.is_empty:
                            polys.append({"kind": layer, "exterior": loc(q.exterior.coords),
                                          "holes": [loc(h.coords) for h in q.interiors]})
        for layer in ("road_divider", "lane_divider"):
            for tok in recs[layer]:
                g = nm.extract_line(nm.get(layer, tok)["line_token"]).intersection(patch)
                for q in (list(g.geoms) if hasattr(g, "geoms") else [g]):
                    if q.geom_type == "LineString" and not q.is_empty:
                        lines.append({"kind": layer, "xy": loc(q.coords)})
        paths, hits, egos = {}, {}, {}
        for k, arr in (("gt", gt[n]), ("op", P["op"][n]), ("cv", P["cv"][n])):
            paths[k] = arr.round(3).tolist()
            hits[k], egos[k] = hit_boxes(arr, e["boxes"])
        rear = {"gt": e["gt_rear"].round(3).tolist(),
                "op": rear_path(P["op"][n], Z.traj_yaw(P["op"][n]), d).round(3).tolist(),
                "cv": rear_path(P["cv"][n], np.zeros(6), d).round(3).tolist()}
        head = None
        if e["token"] in lpos:
            i = lpos[e["token"]]
            head = {"pred_rear_12": lz["pred_op-cinque temporal"][i].round(3).tolist(),
                    "gt_rear_12": fut[i].round(3).tolist(),
                    "ade": float(np.linalg.norm(lz["pred_op-cinque temporal"][i] - fut[i], axis=-1).mean()),
                    "ade_native": float(np.linalg.norm(lz["pred_native op-cinque (no fit)"][i] - fut[i], axis=-1).mean())}
        g = lambda k, name: float(H[k][name][n])  # noqa: E731
        cases.append({
            "id": f"nusc-{len(cases) + 1:02d}", "rule_id": cid, "kind": kind, "failure_type": ftype, "rule": rule,
            "n_candidates": int(m.sum()), "token": e["token"], "scene": e["scene"], "i": e["i"], "location": e["location"],
            "cmd": e["cmd"], "v0": float(v0[n]), "v_end_gt": float(v_end[n]), "hist_s": float(e["hist_s"]),
            "fut_t": e["fut_t"].round(3).tolist(), "paths": paths, "rear": rear, "gt_yaw": e["gt_yaw"].round(4).tolist(),
            "ego_boxes": egos, "hits": hits, "gt_hits": hit_boxes(gt[n], e["boxes"])[0],
            "boxes": [b.round(3).tolist() for b in e["boxes"]], "head": head,
            "scores": {k: {"l2_1s": g(k, "l2_1s"), "l2_2s": g(k, "l2_2s"), "l2_3s": g(k, "l2_3s"), "l2_avg": g(k, "l2_avg"),
                           "l2pt_3s": g(k, "l2pt_3s"), "lon_err_3s": float(lon[k][n]), "lat_err_3s": float(lat[k][n]),
                           "col_bevp_3s": g(k, "col_bevp_3s"), "col_vad_3s": g(k, "col_vad_3s"),
                           "first_col_step": int(fc[k][n])} for k in P},
            "cam": {"K": cam["K"].tolist(), "R": cam["R"].tolist(), "t_ego": cam["t_ego"].tolist(), "scale": 0.6},
            "lidar_xyz": d.tolist(), "frames": frames, "polygons": polys, "lines": lines})
        print(cases[-1]["id"], cid, e["scene"], e["token"], e["cmd"], f"v0={v0[n]:.1f}", "cand", int(m.sum()),
              {k: round(float(H[k]["l2_avg"][n]), 2) for k in P}, flush=True)
    src = {"index": str(Z.index_path()), "preds": str(Z.root("preds")), "per_sample": str(Z.root("score") / "per_sample.pkl"),
           "ladder": sorted(glob.glob(str(data_dir() / "runs/nusc_backbones/ladder/*/nusc_preds.npz")))[-1]}
    json.dump({"set": SET, "n": len(S), "scenes": len({e["scene"] for e in S}), "check": check, "tables": tables,
               "collision_pairs_3s": pairs, "collision_where": col_where, "lon_err_3s_percentiles": lon_q, "sign": sign,
               "speed_ratio": ratio, "collision_kind": col_kind_, "ladder": ladder, "sources": src}, open(OUT / "stats.json", "w"), indent=1)
    json.dump({"selection": "one frame per fixed rule, first in rule order from a distinct scene; illustrative only",
               "cases": cases, "sources": src}, open(OUT / "cases.json", "w"))
    print("DONE", len(cases), OUT, flush=True)


if __name__ == "__main__":
    main()
