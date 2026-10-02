"""Export WOD-E2E val failure cases for the open-loop visual review. CPU only, NO model inference, no training.

Reads only cached files:
  $L/readout/{O,main-s*}/eval/rater.npz          openpilot Cinque native / op-adapt L plans on the 479 val rater frames
  runs/wod_zeroshot/score/<latest>/per_frame.npz zero-shot exam: stored per-frame predictions and RFS (Cinque, Alpamayo, cv, logged)
  processed/wod_zeroshot/{sets.npz,sets.json,op_calib.json} + datasets/waymo_e2e/front3 shards (camera JPEG spans)
Writes <out>/table.json (checks, per-frame scores, per-category tables) and <out>/cases.json + <out>/img/*.jpg.

  CUDA_VISIBLE_DEVICES= REPO=~/data/jev-drive $DATA_DIR/envs/op-train/bin/python wod_review_export.py <out>
"""
import io, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
sys.path[:0] = [str(REPO / "experiments/op_adapt_l/scripts"), str(REPO)]
import op_adapt_l_rfs_diagnosis as G  # noqa: E402
from jevdrive import waymo as W, wod_zeroshot as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

OUT = Path(sys.argv[1]); (OUT / "img").mkdir(parents=True, exist_ok=True)
XS = 1.06                                             # the longitudinal calibration of the submitted test entry
NAT, ADA, SEEDS = "native_x106", "main-s0_x106", G.SEEDS
GIF_T = [-2.0, -1.0] + [0.5 * k for k in range(11)]   # seconds relative to the rated frame; frames are 10 Hz


def miss(geo, i):
    """How frame i loses score: the horizon with the lower best value, the rater that gives that value there."""
    hv, nl, nt, sl, st = (geo[k][i] for k in ("hv", "nl", "nt", "sl", "st"))
    h = int(hv.max(0).argmin()); p = int(hv[:, h].argmax())
    if max(nl[p, h], nt[p, h]) <= 1.0:
        kind = "in_region"
    elif nl[p, h] >= nt[p, h]:
        kind = "lon_short" if sl[p, h] < 0 else "lon_long"
    else:
        kind = "lateral"
    return kind, h, p


def main():
    d = G.load()
    names, n, cat = d["names"], len(d["names"]), d["cat"]
    z = np.load(sorted((data_dir() / "runs/wod_zeroshot/score").iterdir())[-1] / "per_frame.npz")
    assert (z["names"] == names).all() and np.abs(z["traj"] - d["traj"]).max() == 0 and (z["scores"] == d["scores"]).all()
    chk = {"per_frame_npz": str(sorted((data_dir() / "runs/wod_zeroshot/score").iterdir())[-1] / "per_frame.npz")}

    # ---- check 1: every stored per-frame score of the zero-shot exam, recomputed with the project scorer
    stored = {}
    for k in [f[4:] for f in z.files if f.startswith("rfs/")]:
        p, s = z[f"pred/{k}"].astype(np.float64), z[f"rfs/{k}"]
        r = np.stack([W.rater_feedback_score(p[:, j], d["traj"], d["scores"], z["speed"].astype(np.float64)) for j in range(p.shape[1])], 1)
        stored[k] = {"max_abs_diff": float(np.abs(r - s).max()), "frames_differ_gt_1e-6": int((np.abs(r - s) > 1e-6).any(1).sum()),
                     "leaderboard_recomputed": W.rfs_by_cluster(r.mean(1), cat)[0]}
    chk["stored_per_frame_vs_recomputed"] = stored
    chk["speed_max_abs_diff_sets_vs_per_frame"] = float(np.abs(z["speed"] - d["speed"]).max())

    # ---- every trajectory row used on the page, (n, 20, 2), and its per-frame RFS + geometry
    med = np.array([Z.medoid(z["pred/alpamayo_nav"][i]) for i in range(n)])
    wp = {"native_x1": G.waypoints(d, "O", 1.0), NAT: G.waypoints(d, "O", XS)}
    for m in SEEDS:
        wp[f"{m}_x1"], wp[f"{m}_x106"] = G.waypoints(d, m, 1.0), G.waypoints(d, m, XS)
    wp |= {"zs_cinque": z["pred/op_cinque"][:, 0].astype(np.float64), "alpamayo_medoid": z["pred/alpamayo_nav"][np.arange(n), med].astype(np.float64),
           "cv": z["pred/cv"][:, 0].astype(np.float64), "logged": z["pred/logged_future"][:, 0].astype(np.float64),
           "rater_best": z["pred/rater_best"][:, 0].astype(np.float64)}
    sc, geo = {}, {}
    for k, p in wp.items():
        sl, st, lt, at = G.geometry(p, d["traj"], d["speed"])
        per, ins, nl, nt, hv = G.score_from(sl, st, lt, at, d["scores"])
        assert np.abs(per - W.rater_feedback_score(p, d["traj"], d["scores"], d["speed"])).max() < 1e-9
        sc[k], geo[k] = per, dict(sl=sl, st=st, lng_thr=lt, lat_thr=at, inside=ins, nl=nl, nt=nt, hv=hv)
    lb = {k: W.rfs_by_cluster(v, cat)[0] for k, v in sc.items()}
    chk["leaderboard_val"] = lb
    chk["native_vs_zeroshot_cinque_plan_max_abs_m"] = float(np.abs(wp["native_x1"] - wp["zs_cinque"]).max())

    # ---- check 2: decision 79 numbers (selfcheck.json, q1_coverage.csv, q1_category_headroom.csv) and decision 34 clusters.csv
    R = REPO / "experiments/op_adapt_l/results/rfs-diagnosis"
    sf = json.loads((R / "selfcheck.json").read_text())
    chk["selfcheck_max_abs_diff"] = float(max([abs(sf["rfs_x1.0"]["O"] - lb["native_x1"]), abs(sf["rfs_x1.06"]["O"] - lb[NAT])]
                                              + [abs(sf["rfs_x1.0"][m] - lb[f"{m}_x1"]) for m in SEEDS] + [abs(sf["rfs_x1.06"][m] - lb[f"{m}_x106"]) for m in SEEDS]))
    gap1 = d["scores"].max(1) - sc["native_x1"]
    q1 = pd.read_csv(R / "q1_coverage.csv").set_index("slice")
    chk["q1_coverage_max_abs_diff"] = float(max(max(abs(q1.loc[s, "rfs_orig_mean"] - sc["native_x1"][d["slice"][s]].mean()), abs(q1.loc[s, "frames"] - d["slice"][s].sum()))
                                                for s in q1.index if d["slice"][s].any()))
    qc = pd.read_csv(R / "q1_category_headroom.csv").set_index("category")
    chk["q1_category_max_abs_diff"] = float(max(max(abs(qc.loc[c, "rfs_orig"] - sc["native_x1"][cat == c].mean()), abs(qc.loc[c, "gap_mean"] - gap1[cat == c].mean())) for c in qc.index))
    cl = pd.read_csv(REPO / "experiments/zeroshot_openloop/results/wod-zeroshot/clusters.csv", index_col=0)
    chk["clusters_csv_max_abs_diff"] = float(max(abs(cl.loc[r, c] - sc[k][cat == c].mean()) for r, k in (("op_cinque", "zs_cinque"), ("cv", "cv"), ("logged_future", "logged"),
                                                 ("alpamayo_nav | medoid-of-6", "alpamayo_medoid"), ("rater_best", "rater_best")) for c in cl.columns))

    # ---- per-frame table
    mean_ada = np.mean([sc[f"{m}_x106"] for m in SEEDS], 0)
    best = d["scores"].max(1)
    gap = best - sc[NAT]
    mk = [miss(geo[NAT], i) for i in range(n)]
    kind = np.array([k if g > 1e-9 else "match_best" for (k, _, _), g in zip(mk, gap)])
    cats = sorted(np.unique(cat)); C = len(cats)
    w = np.array([1.0 / (C * (cat == c).sum()) for c in cat])
    frames = pd.DataFrame({"name": names, "cluster": cat, "intent": z["intent"], "label": d["label"].astype(str), "speed": d["speed"],
                           "best_rater": best, "gap": gap, "miss": kind, "ada_mean": mean_ada, **{k: v for k, v in sc.items()}})
    for s in ("start", "stay", "stop", "turn_onset", "control"):
        frames[f"s_{s}"] = d["slice"][s]
    cols = [("native_x1", "Cinque 原生"), (NAT, "Cinque ×1.06"), ("ada_mean", "适配 L ×1.06（3 seed 均值）"), ("alpamayo_medoid", "Alpamayo medoid-of-6"),
            ("cv", "cv"), ("logged", "logged future"), ("rater_best", "最高分 rater 轨迹")]
    tab_cat = [[c, int((cat == c).sum())] + [round(float(frames.loc[cat == c, k].mean()), 3) for k, _ in cols]
               + [round(float(gap[cat == c].mean()), 3), round(float((w * gap)[cat == c].sum()), 4), round(float(100 * (w * gap)[cat == c].sum() / (w * gap).sum()), 1),
                  round(float(100 * (~geo[NAT]["inside"][cat == c]).mean()), 1)] for c in cats]
    tab_cat.append(["榜单口径（cluster 均值）", n] + [round(float((w * frames[k]).sum()), 3) for k, _ in cols] + [round(float((gap.reshape(-1) * w).sum() * C / C), 3), round(float((w * gap).sum()), 4), 100.0,
                                                                                                         round(float(100 * (w * C * (~geo[NAT]["inside"]) / C).sum()), 1)])
    lab_order = ["start", "stay", "stop", "turn_onset", "in_turn", "nudge", "lane_change", "control", "other"]
    tab_lab = [[s, int((frames.label == s).sum())] + [round(float(frames.loc[frames.label == s, k].mean()), 3) for k in ("native_x1", NAT, "ada_mean", "logged")]
               + [round(float(gap[frames.label == s].mean()), 3), round(float(100 * (w * gap)[frames.label == s].sum() / (w * gap).sum()), 1)] for s in lab_order if (frames.label == s).any()]
    kinds = ["match_best", "in_region", "lon_short", "lon_long", "lateral"]
    tab_miss = [[k, int((kind == k).sum()), round(float(100 * (kind == k).mean()), 1), round(float(sc[NAT][kind == k].mean()), 3), round(float(gap[kind == k].mean()), 3),
                 round(float(100 * (w * gap)[kind == k].sum() / (w * gap).sum()), 1), int(((kind == k) & (d["speed"] < 0.5)).sum())] for k in kinds]
    tab_cat_miss = [[c] + [round(float(100 * (w * gap)[(cat == c) & (kind == k)].sum() / (w * gap)[cat == c].sum()), 1) for k in kinds[1:]] for c in cats]
    table = {"checks": chk, "columns": [k for k, _ in cols], "column_names": [v for _, v in cols], "by_cluster": tab_cat, "by_label": tab_lab, "by_miss": tab_miss,
             "cluster_by_miss": tab_cat_miss, "frames": json.loads(frames.to_json(orient="records")), "standstill_frames": int((d["speed"] < 0.5).sum())}
    (OUT / "table.json").write_text(json.dumps(table, indent=1))

    # ---- case selection. Everything is ranked on the submitted configuration (Cinque native x1.06) by gap = best rater label - RFS,
    #      ties broken by frame name; one frame per sequence; first match of each rule in this order.
    seeds_d = np.stack([sc[f"{m}_x106"] - sc[NAT] for m in SEEDS])
    stand = d["speed"] < 0.5
    rules = [("start_missed", "起步：rater 走，plan 留在原地", d["slice"]["start"] & (kind == "lon_short"), -gap),
             ("false_start", "静止帧：rater 等，plan 起步", stand & (kind == "lon_long"), -gap),
             ("stop_missed", "停车：rater 减速 / 停，plan 继续走", d["slice"]["stop"] & (kind == "lon_long"), -gap),
             ("turn_onset", "起转：plan 没有跟上转向", d["slice"]["turn_onset"] & (kind == "lateral"), -gap),
             ("lateral_cruise", "巡航帧横向偏差（绕行 / 换道）", (frames.label == "other").to_numpy() & (kind == "lateral") & (d["speed"] > 3), -gap),
             ("pedestrian", "行人类：纵向不符", (cat == "Pedestrian") & np.isin(kind, ["lon_long", "lon_short"]) & ~stand, -gap),
             ("too_slow_cruise", "巡航帧 plan 偏慢", (frames.label == "other").to_numpy() & (kind == "lon_short") & (d["speed"] > 5), -gap),
             ("adapt_regression", "适配后变差（3 seed 同号）", (seeds_d < -1e-9).all(0), seeds_d.mean(0)),
             ("adapt_recovery", "适配后变好（3 seed 同号）", (seeds_d > 1e-9).all(0), -seeds_d.mean(0)),
             ("control", "对照：巡航帧，原生与适配都在最高分 rater 区域内", d["slice"]["control"] & (sc[NAT] >= best - 1e-9) & (mean_ada >= best - 1e-9) & (best >= 9), np.zeros(n)),
             ("control", "对照：起步帧，原生捕获起步", d["slice"]["start"] & (sc[NAT] >= best - 1e-9) & (best >= 9), np.zeros(n))]
    spans, _ = Z.load_spans()
    calib = json.loads((Z.root() / "op_calib.json").read_text())
    front3 = data_dir() / "datasets/waymo_e2e/front3"

    def jpg(name, cam):                                   # cam: 0 front, 1 front_left, 2 front_right (W.CAMS order)
        sp = spans[name]
        with open(front3 / sp[0], "rb") as f:
            f.seek(sp[1 + 2 * cam])
            return Image.open(io.BytesIO(f.read(sp[2 + 2 * cam]))).convert("RGB")

    seen, cases = set(), []
    for group, title, mask, key in rules:
        cand = [i for i in np.lexsort((names, key)) if mask[i] and d["seg"][i] not in seen]
        if not cand:
            print("NO CANDIDATE", group, title, flush=True); continue
        i = cand[0]; seen.add(d["seg"][i]); name = str(names[i]); seq, f = name.rsplit("-", 1)
        cid = f"wod-{len(cases) + 1:02d}"
        files = {}
        for c, cn in enumerate(W.CAMS):
            fn = f"img/{cid}-t0-{cn}.jpg"; jpg(name, c).save(OUT / fn, quality=88); files[cn] = fn
        strip = []
        for t in GIF_T:
            hn = f"{seq}-{int(f) + int(round(t * 10)):03d}"
            if hn not in spans:
                continue
            im = jpg(hn, 0); im.thumbnail((486, 540)); fn = f"img/{cid}-front-{t:+.1f}.jpg"; im.save(OUT / fn, quality=80); strip.append({"t": t, "file": fn, "frame": hn})
        lng, _ = W._rater_frames(d["traj"][i:i + 1])
        g = {m: {k: geo[m][k][i].tolist() for k in ("sl", "st", "nl", "nt", "hv", "lng_thr", "lat_thr")} | {"inside": bool(geo[m]["inside"][i]), "miss": list(miss(geo[m], i))} for m in (NAT, ADA)}
        cases.append({"id": cid, "group": group, "rule_title": title, "candidates": len(cand), "name": name, "cluster": str(cat[i]), "intent": W.INTENTS[int(z["intent"][i])],
                      "label": str(d["label"][i]), "slices": [s for s in ("start", "stay", "stop", "turn_onset", "control", "nudge", "in_turn", "lane_change") if d["slice"][s][i]],
                      "speed": float(d["speed"][i]), "shard": spans[name][0], "rater_traj": d["traj"][i].round(3).tolist(), "rater_scores": d["scores"][i].tolist(),
                      "rater_lng_dir": lng[0][:, G.HOR].round(5).tolist(), "horizons_idx": G.HOR.tolist(),
                      "pred": {k: wp[k][i].round(3).tolist() for k in (NAT, "native_x1", ADA, "main-s1_x106", "main-s2_x106", "alpamayo_medoid", "cv", "logged")},
                      "rfs": {k: float(sc[k][i]) for k in sc}, "rfs_stored_zeroshot": {k: z[f"rfs/{k}"][i].tolist() for k in ("op_cinque", "cv", "logged_future", "alpamayo_nav")},
                      "alpamayo_medoid_index": int(med[i]), "capture": {m: {k: bool(v[i]) for k, v in d["cap"][m].items()} for m in ["O"] + SEEDS},
                      "geo": g, "gap": float(gap[i]), "miss": str(kind[i]), "calib": calib[seq], "images_t0": files, "front_strip": strip})
        print(cid, group, name, cat[i], f"v0={d['speed'][i]:.1f}", f"nat={sc[NAT][i]:.2f} ada={sc[ADA][i]:.2f} best={best[i]:.0f}", kind[i], len(cand), flush=True)
    (OUT / "cases.json").write_text(json.dumps({"xscale": XS, "native": NAT, "adapted": ADA, "gif_t": GIF_T, "cases": cases}))
    print(json.dumps(chk, indent=1)); print("DONE", len(cases), flush=True)


if __name__ == "__main__":
    main()
