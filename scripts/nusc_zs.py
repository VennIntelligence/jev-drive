#!/usr/bin/env python
"""nuScenes open-loop zero-shot exam, project venv (todos/2026-09-24-zeroshot-exam/nuscenes-physicalai.md).

  index   val keyframe index (jevdrive.nuscenes_zs.build_index) -> $DATA_DIR/processed/nusc_zs/val_index.pkl
  sets    the pre-registered evaluation sets (main / full-history / Alpamayo subsets) -> sets.json next to it
  cv      constant-velocity baseline predictions (straight ahead at the current speed) and the logged future
  score   every prediction file x every evaluation set: L2 / collision (VAD and BEV-Planner conventions), scene
          bootstrap CIs, paired differences vs CV, by command -> $DATA_DIR/runs/nusc_zs/score/

    .venv/bin/python scripts/nusc_zs.py index
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

MIN_HIST_S = 1.5    # main set: t0 at least 1.5 s after the scene's first keyframe (Alpamayo's egomotion window)
FULL_HIST_S = 5.0   # full-history subset: openpilot's ~5 s context is filled


def sets_path() -> Path:
    return Z.index_path().with_name("sets.json")


def build_sets(idx: dict, seed: int = 0) -> dict:
    s = idx["samples"]
    valid = [e["token"] for e in s if e["valid"]]
    main = [e["token"] for e in s if e["valid"] and e["hist_s"] >= MIN_HIST_S - 1e-3]
    full = [e["token"] for e in s if e["valid"] and e["hist_s"] >= FULL_HIST_S - 1e-3]
    rng = np.random.default_rng(seed)
    perm = [main[k] for k in rng.permutation(len(main))]
    return {"valid": valid, "main": main, "fullhist": full,
            "quarter": sorted(perm[:len(main) // 4], key=main.index),   # seed-0 1/4 of main, in index order
            "half": sorted(perm[:len(main) // 2], key=main.index)}


def cmd_index(a, log):
    idx = Z.build_index(log.info)
    p = Z.index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(idx, f, protocol=5)
    sets = build_sets(idx)
    json.dump(sets, open(sets_path(), "w"))
    s = {e["token"]: e for e in idx["samples"]}
    for k, v in sets.items():
        cmds = [s[t]["cmd"] for t in v]
        log.info(f"set {k}: n={len(v)} scenes={len({s[t]['scene'] for t in v})} "
                 f"cmd={ {c: cmds.count(c) for c in Z.CMD_NAMES} }")
    miss = {c: sum(not (Z.dataroot() / p).exists() for sc in idx["scenes"].values() for p in sc["cams"][c]["path"])
            for c in Z.CAMS}
    log.info(f"missing files per camera over the val scenes: {miss}")
    log.info(f"-> {p}")


def cmd_cv(a, log):
    """Baselines through our own pipeline: cv (straight at the current speed, BEV-Planner's GoStraight) and cvv
    (constant velocity vector, heading of the last 0.5 s displacement); both at the LIDAR_TOP point."""
    idx = Z.load_index()
    sc = idx["scenes"]
    rows = {"cv": {}, "cvv": {}}
    for e in idx["samples"]:
        if not e["valid"]:
            continue
        scene = sc[e["scene"]]
        v = Z.cv_speed(scene, e["t0"])
        t = e["fut_t"]
        rows["cv"][e["token"]] = Z.to_lidar_point(t, np.c_[v * t, 0 * t], 0 * t, scene["lidar_xyz"], t)[0]
        xyz, R = Z.ego_at(scene, [e["t0"] - 5e5, e["t0"]])
        d = (xyz[1] - xyz[0]) @ R[1]          # displacement in the t0 frame
        h = np.arctan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 0.1 else 0.0
        rows["cvv"][e["token"]] = Z.to_lidar_point(t, np.c_[v * t * np.cos(h), v * t * np.sin(h)], 0 * t + h,
                                                   scene["lidar_xyz"], t)[0]
    for k, r in rows.items():
        toks = sorted(r)
        np.savez(Z.root("preds") / f"{k}.npz", tokens=np.array(toks), pred=np.stack([r[t] for t in toks]).astype(np.float32))
    log.info(f"cv / cvv for {len(rows['cv'])} valid samples")


ALP_T = np.arange(1, 65) * 0.1
ROWS = {  # row name -> prediction source; Alpamayo rows come from the JSON lines of the run
    "logged future": "gt", "cv": "cv", "cvv": "cvv",
    "openpilot small": "op_small_none", "openpilot Cinque": "op_cinque_none", "openpilot Lebowski": "op_lebowski_none",
    "openpilot small + cmd": "op_small_cmd", "openpilot Cinque + cmd": "op_cinque_cmd",
    "openpilot Lebowski + cmd": "op_lebowski_cmd",
    "Alpamayo 1.5 nav": "alp:nav", "Alpamayo 1.5 no-nav": "alp:nonav"}


def load_preds(idx: dict) -> dict:
    """{source: {token: (6, 2) LIDAR_TOP points}} for every prediction file present."""
    import json as js
    by = {e["token"]: e for e in idx["samples"]}
    out = {"gt": {t: e["gt"] for t, e in by.items() if e["valid"]}}
    for f in Z.root("preds").glob("*.npz"):
        if f.stem.endswith("_tail"):
            continue
        z = np.load(f)
        out[f.stem] = dict(zip(z["tokens"].tolist(), z["pred"]))
    for f in Z.root("alpamayo").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                r = js.loads(line)
            except js.JSONDecodeError:
                continue
            e = by[r["token"]]
            sc = idx["scenes"][e["scene"]]
            xyz = np.asarray(r["xyz"])
            out.setdefault("alp:" + r["variant"], {})[r["token"]] = Z.to_lidar_point(
                ALP_T, xyz[:, :2], np.asarray(r["yaw"]), sc["lidar_xyz"], e["fut_t"])[0].astype(np.float32)
    return out


def cmd_score(a, log):
    import pandas as pd
    idx = Z.load_index()
    sets = json.load(open(sets_path()))
    by = {e["token"]: e for e in idx["samples"]}
    preds = load_preds(idx)
    log.info("prediction sources: " + ", ".join(f"{k} ({len(v)})" for k, v in preds.items()))
    out = Z.root("score")
    rng = np.random.default_rng(0)
    rows, cmd_rows, per_sample = [], [], {}
    for set_name in a.sets.split(","):
        toks = sets[set_name]
        samples = [by[t] for t in toks]
        scene_ids = np.unique([e["scene"] for e in samples], return_inverse=True)[1]
        n_sc = scene_ids.max() + 1
        Bidx = rng.integers(0, n_sc, (a.boot, n_sc))
        turn = np.array([e["cmd"] != "straight" for e in samples])
        vals = {}
        for row, src in ROWS.items():
            p = preds.get(src, {})
            if not all(t in p for t in toks):
                continue
            pred = np.stack([p[t] for t in toks]).astype(np.float64)
            h = Z.horizons(Z.per_sample(pred, samples))
            h["l2_avg"] = (h["l2_1s"] + h["l2_2s"] + h["l2_3s"]) / 3
            h["l2pt_avg"] = (h["l2pt_1s"] + h["l2pt_2s"] + h["l2pt_3s"]) / 3
            for c in ("col_vad", "col_bevp"):
                h[f"{c}_avg"] = (h[f"{c}_1s"] + h[f"{c}_2s"] + h[f"{c}_3s"]) / 3
            gt = np.stack([by[t]["gt"] for t in toks])
            h["lon_err_3s"] = pred[:, -1, 0] - gt[:, -1, 0]
            vals[row] = h
            per_sample[(set_name, row)] = h
        if "cv" not in vals:
            continue

        def boot(v):  # scene-cluster bootstrap of a per-sample mean
            s, c = np.bincount(scene_ids, v, n_sc), np.bincount(scene_ids, None, n_sc)
            return s[Bidx].sum(1) / c[Bidx].sum(1)
        for row, h in vals.items():
            r = {"set": set_name, "row": row, "n": len(toks)}
            for k, v in h.items():
                r[k] = float(v.mean())
            for k in ("l2_avg", "col_vad_avg", "col_bevp_avg"):
                r[k + "_lo"], r[k + "_hi"] = np.percentile(boot(h[k]), [2.5, 97.5])
                d = h[k] - vals["cv"][k]
                r["d_" + k] = float(d.mean())
                r["d_" + k + "_lo"], r["d_" + k + "_hi"] = np.percentile(boot(d), [2.5, 97.5])
            rows.append(r)
            for grp, m in (("straight", ~turn), ("turn", turn)):
                cr = {"set": set_name, "row": row, "cmd": grp, "n": int(m.sum())}
                for k in ("l2_avg", "l2_3s", "col_vad_avg", "col_bevp_avg", "col_bevp_3s"):
                    cr[k] = float(h[k][m].mean())
                cmd_rows.append(cr)
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(cmd_rows).to_csv(out / "by_command.csv", index=False)
    with open(out / "per_sample.pkl", "wb") as f:
        pickle.dump({"sets": {k: sets[k] for k in a.sets.split(",")}, "values": per_sample}, f)
    show = ["set", "row", "n", "l2_1s", "l2_2s", "l2_3s", "l2_avg", "d_l2_avg", "d_l2_avg_lo", "d_l2_avg_hi",
            "col_vad_avg", "col_bevp_1s", "col_bevp_2s", "col_bevp_3s", "col_bevp_avg", "d_col_bevp_avg_lo",
            "d_col_bevp_avg_hi"]
    pd.set_option("display.width", 250)
    log.info("\n" + df[show].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("index", "cv", "score"))
    ap.add_argument("--sets", default="main,valid,fullhist,quarter")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    log = RunLog("nusc_zs", a.cmd)
    {"index": cmd_index, "cv": cmd_cv, "score": cmd_score}[a.cmd](a, log)
    log.event("end")
