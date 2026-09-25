"""openpilot's open-loop standing on WOD-E2E and NAVSIM, side by side with our heads, Alpamayo, constant velocity and the
published leaderboards. Pre-registration: todos/2026-09-25-openpilot-openloop-comparison.md (G0-G3).

  wod     G0 + G1: every row on the 479 val rater frames -- RFS (leaderboard cluster mean, CI by within-cluster
          resampling), paired frame-mean deltas against cv / `cls ego K1024` / Cinque's native plan, the standstill and
          junction subsets, ADE@5s vs rater_best; and the NAVSIM-timeline variants of the openpilot native plan
  navsim  G2 / G3 readouts from the official devkit's per-token scores (written by `scripts/navsim_zs_score.sh`)

    python -m jevdrive.openloop_standing wod [--desire-run <heads_train run of p5route 2b>]
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import get_logger

log = get_logger(__name__)
B = 10_000
REPO = Path(__file__).resolve().parents[1]
MODELS = ("small", "cinque", "lebowski")
TIMELINE = ("ctx1.5", "nav2hz", "nav2hz-dilate")
# published WOD-E2E test-split entries (not pairable; arXiv 2510.26125 Table 8, RAP from the leaderboard page)
WOD_BOARD = {"RAP": (8.043, 2.65), "Poutine": (7.986, 2.741), "UniPlan": (7.779, 2.986), "HMVLM": (7.736, 3.071),
             "DiffusionLTF": (7.717, 2.977), "AutoVLA": (7.556, 2.958), "NaiveEMMA": (7.528, 3.018)}


def _cluster_ci(v: np.ndarray, codes: np.ndarray, rng) -> tuple[float, float]:
    """95% CI of the cluster mean, resampling frames within each cluster (the exam's stratified bootstrap), vectorised."""
    k = codes.max() + 1
    groups = [np.flatnonzero(codes == c) for c in range(k)]
    reps = np.zeros(B)
    for g in groups:
        reps += v[g][rng.integers(0, len(g), (B, len(g)))].mean(1)
    return tuple(np.percentile(reps / k, [2.5, 97.5]))


def _paired(d: np.ndarray, rng) -> tuple[float, float, float]:
    """Frame-mean paired delta with a frame bootstrap (one rater frame per sequence, so frame = sequence)."""
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), *np.percentile(bs, [2.5, 97.5])


def wod_preds(desire_run: str = "") -> tuple[dict, dict]:
    """{row name: (479, K, 20, 2) predictions} on op_route's rater context, K = samples (6 for Alpamayo, else 1)."""
    from . import op_route as R
    from . import wod_zeroshot as Z
    ctx = R.rater_context()
    names = ctx["name"]
    z = np.load(REPO / "research/results/wod-zeroshot/per_frame.npz", allow_pickle=False)
    at = pd.Series(np.arange(len(z["names"])), index=z["names"]).reindex(names).to_numpy()
    assert not np.isnan(at).any()
    at = at.astype(int)
    pf = lambda k: z[f"pred/{k}"][at]  # noqa: E731
    preds = {"logged future": pf("logged_future"), "cv": pf("cv"), "ours ridge ego": pf("ours ridge ego"),
             "ours cls ego K1024": pf("ours cls ego"), "ours cls_late qwen4b+ego": pf("ours cls_late vision+ego"),
             "alpamayo nav (expected of 6)": pf("alpamayo_nav"), "alpamayo no-nav (expected of 6)": pf("alpamayo_nonav")}
    alp = pf("alpamayo_nav")
    preds["alpamayo nav (medoid of 6)"] = alp[np.arange(len(alp)), [Z.medoid(a) for a in alp]][:, None]
    for m in MODELS:
        preds[f"op-{m} native"] = pf(f"op_{m}")
    heads = R.run_preds(R.NOD40_RUN, names)
    for m in ("cinque", "lebowski"):
        preds[f"op-{m} temporal + ridge_late"] = heads[f"A ridge_late op-{m} temporal"][:, None]
        preds[f"op-{m} temporal + cls_late"] = heads[f"cls_late op-{m} temporal"][:, None]
    for m in MODELS:
        for v in TIMELINE:
            d = Z.root("preds", f"op_{m}@{v}")
            if all((d / f"{n}.npz").exists() for n in names):
                preds[f"op-{m} native @{v}"] = np.stack([np.load(d / f"{n}.npz")["wod"] for n in names])[:, None]
    if desire_run:
        nat = R.native(names, "trainval_desire", " +desire")
        dh = R.run_preds(desire_run, names, suffix=" +desire")
        for m in ("cinque", "lebowski"):
            preds[f"op-{m} native +desire"] = nat[f"native op-{m} +desire"][:, None]
            preds[f"op-{m} native (stream, no desire)"] = R.native(names)[f"native op-{m}"][:, None]
            preds[f"op-{m} temporal +desire + ridge_late"] = dh[f"A ridge_late op-{m} temporal +desire"][:, None]
            preds[f"op-{m} temporal +desire + cls_late"] = dh[f"cls_late op-{m} temporal +desire"][:, None]
    return preds, ctx


def wod(desire_run: str = "") -> dict:
    from . import waymo as W
    preds, ctx = wod_preds(desire_run)
    rt, rs, sp, cl = ctx["rtraj"], ctx["rscore"], ctx["speed"], ctx["cluster"]
    best = rt[np.arange(len(rt)), rs.argmax(1)]
    codes = pd.factorize(cl)[0]
    rfs = {k: np.stack([W.rater_feedback_score(p[:, j], rt, rs, sp) for j in range(p.shape[1])]).mean(0)
           for k, p in preds.items()}
    ade = {k: np.linalg.norm(p - best[:, None], axis=-1).mean(-1).mean(1) for k, p in preds.items()}
    subs = {"standstill": sp < 0.5, "junction": np.isin(cl, ("Interections", "Multi-Lane Maneuvers"))}
    refs = {"cv": "cv", "cls ego": "ours cls ego K1024", "cinque native": "op-cinque native"}
    rows = []
    for k, v in rfs.items():
        rng = np.random.default_rng(0)
        lo, hi = _cluster_ci(v, codes, rng)
        r = {"row": k, "n": len(v), "rfs_cluster": W.rfs_by_cluster(v, cl)[0], "ci_lo": lo, "ci_hi": hi,
             "rfs_frame": v.mean(), "ade5_rater_best": ade[k].mean(),
             **{f"rfs_{s}": v[m].mean() for s, m in subs.items()}}
        for rn, ref in refs.items():
            d, dlo, dhi = _paired(v - rfs[ref], np.random.default_rng(1))
            r |= {f"d_{rn}": d, f"d_{rn}_lo": dlo, f"d_{rn}_hi": dhi}
        for s, m in subs.items():
            d, dlo, dhi = _paired((v - rfs["cv"])[m], np.random.default_rng(2))
            r |= {f"d_cv_{s}": d, f"d_cv_{s}_lo": dlo, f"d_cv_{s}_hi": dhi}
        rows.append(r)
    main = pd.DataFrame(rows)
    # G1: the NAVSIM timeline on the same frames, against each model's exam plan (base)
    g1 = []
    for m in MODELS:
        base = rfs[f"op-{m} native"]
        for v in TIMELINE:
            k = f"op-{m} native @{v}"
            if k not in rfs:
                continue
            d, lo, hi = _paired(rfs[k] - base, np.random.default_rng(3))
            lon = preds[k][:, 0, -1, 0] - preds["logged future"][:, 0, -1, 0]
            g1.append({"model": m, "variant": v, "rfs_cluster": W.rfs_by_cluster(rfs[k], cl)[0],
                       "base_rfs_cluster": W.rfs_by_cluster(base, cl)[0], "d_vs_base": d, "lo": lo, "hi": hi,
                       "S_share_of_margin_over_cv": (base.mean() - rfs[k].mean()) / (base.mean() - rfs["cv"].mean()),
                       "lon5_bias": lon.mean(), "lon5_bias_base": (preds[f"op-{m} native"][:, 0, -1, 0]
                                                                  - preds["logged future"][:, 0, -1, 0]).mean(),
                       "ade5_rater_best": ade[k].mean()})
        for a, b in (("nav2hz", "ctx1.5"), ("nav2hz-dilate", "nav2hz")):
            ka, kb = f"op-{m} native @{a}", f"op-{m} native @{b}"
            if ka in rfs and kb in rfs:
                d, lo, hi = _paired(rfs[ka] - rfs[kb], np.random.default_rng(4))
                g1.append({"model": m, "variant": f"{a} - {b}", "d_vs_base": d, "lo": lo, "hi": hi})
    board = pd.DataFrame([{"row": f"{k} (test split)", "rfs_cluster": v[0], "ade5_rater_best": v[1]} for k, v in WOD_BOARD.items()])
    np.savez_compressed(REPO / "research/results/openpilot-openloop/wod_rfs_per_frame.npz", names=ctx["name"],
                        cluster=cl, speed=sp, **{f"rfs/{k}": v for k, v in rfs.items()})
    return {"wod_main": main, "wod_timeline": pd.DataFrame(g1), "wod_board": board}


NAV_AGENTS = {  # devkit run name -> row label (exam rows from navsim.md, head rows from jevdrive.navsim_heads)
    "human": "human (log)", "cv": "constant velocity", "alpamayo_nav": "Alpamayo 1.5 nav",
    "small_none": "op-small native", "cinque_none": "op-cinque native", "lebowski_none": "op-lebowski native",
    "cinque_cmd": "op-cinque native +cmd desire", "lebowski_cmd": "op-lebowski native +cmd desire",
    "heads_ctrv": "ctrv", "heads_ridge_ego": "ours ridge ego", "heads_cls_ego_K1024": "ours cls ego K1024",
    "heads_ridge_late_cinque_temporal": "op-cinque temporal + ridge_late",
    "heads_cls_late_cinque_temporal": "op-cinque temporal + cls_late",
    "heads_ridge_late_lebowski_temporal": "op-lebowski temporal + ridge_late",
    "heads_cls_late_lebowski_temporal": "op-lebowski temporal + cls_late"}
NAV_PAIRS = [("heads_ridge_late_cinque_temporal", "heads_ridge_ego"), ("heads_cls_late_cinque_temporal", "heads_cls_ego_K1024"),
             ("heads_ridge_late_lebowski_temporal", "heads_ridge_ego"), ("heads_cls_late_lebowski_temporal", "heads_cls_ego_K1024"),
             ("heads_ridge_late_cinque_temporal", "cinque_none"), ("heads_cls_late_cinque_temporal", "cinque_none"),
             ("heads_ridge_late_lebowski_temporal", "lebowski_none"), ("heads_cls_late_lebowski_temporal", "lebowski_none"),
             ("heads_ridge_ego", "cv"), ("heads_cls_ego_K1024", "cv"), ("heads_ctrv", "cv"), ("heads_cls_late_cinque_temporal", "cv"),
             ("heads_cls_late_lebowski_temporal", "cv"), ("cinque_none", "cv"), ("lebowski_none", "cv"), ("small_none", "cv"),
             ("cinque_none", "alpamayo_nav"), ("lebowski_none", "alpamayo_nav"), ("heads_cls_ego_K1024", "alpamayo_nav"),
             ("heads_cls_late_cinque_temporal", "alpamayo_nav"), ("heads_cls_late_lebowski_temporal", "alpamayo_nav")]
# published navtest / navhard numbers (not pairable): NAVSIM v1 paper arXiv 2406.15349 Table 2 (PDMS), SimWAM arXiv
# 2608.07468 Tables 2 / 3 (EPDMS, navhard two-stage) as collected in research/openpilot-and-open-driving-models.md
NAV_BOARD = [("Ego Status MLP (blind, navtrain)", 65.6, None, None), ("TransFuser", 84.0, 76.7, 23.1),
             ("DiffusionDrive", 88.1, 84.5, 27.5), ("DiffusionDriveV2", 91.2, None, None), ("ReCogDrive (VLM)", 90.8, 83.6, 25.7),
             ("SimWAM", 91.5, 90.2, 37.6)]
V1_SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
V2_SUB = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
          "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def _latest(ver: str, split: str, name: str):
    import glob
    from .common import data_dir
    fs = sorted(glob.glob(str(data_dir() / "runs/navsim/eval" / f"{ver}_{split}_{name}" / "*" / "*.csv")))
    return pd.read_csv(fs[-1]) if fs else None


def navsim() -> dict:
    """G2 / G3: PDMS (v1.1) and EPDMS (v2) on navtest per row with token-bootstrap CIs, the pre-registered paired
    deltas on per-token scores, and navhard two-stage EPDMS (aggregate only: its two-stage weighting is not per token)."""
    rows, pairs, hard, tok = [], [], [], {}
    for ver, metric, subs in (("v1", "PDMS", V1_SUB), ("v2", "EPDMS", V2_SUB)):
        for name, label in NAV_AGENTS.items():
            df = _latest(ver, "navtest", name)
            if df is None:
                continue
            df = df[df["token"].str.fullmatch(r"[0-9a-f]{16,17}") & df["valid"].astype(bool)]   # the exam's convention
            sc = df.set_index("token")["score"].astype(float)
            tok[(metric, name)] = sc
            v = sc.to_numpy()
            bs = v[np.random.default_rng(0).integers(0, len(v), (2000, len(v)))].mean(1)
            rows.append({"metric": metric, "row": label, "agent": name, "n": len(v), "score": 100 * v.mean(),
                         "ci_lo": 100 * np.percentile(bs, 2.5), "ci_hi": 100 * np.percentile(bs, 97.5),
                         **{c: 100 * df[c].mean() for c in subs if c in df}})
        for a, b in NAV_PAIRS:
            if (metric, a) in tok and (metric, b) in tok:
                x, y = tok[(metric, a)].align(tok[(metric, b)], join="inner")
                d = (x - y).to_numpy()
                bs = d[np.random.default_rng(1).integers(0, len(d), (B, len(d)))].mean(1)
                pairs.append({"metric": metric, "a": NAV_AGENTS[a], "b": NAV_AGENTS[b], "n": len(d), "diff": 100 * d.mean(),
                              "ci_lo": 100 * np.percentile(bs, 2.5), "ci_hi": 100 * np.percentile(bs, 97.5)})
    for name, label in NAV_AGENTS.items():
        df = _latest("v2", "navhard_two_stage", name)
        if df is None:
            continue
        summ = df[df["token"].str.startswith("extended_pdm_score")].set_index("token")["score"]
        hard.append({"row": label, "agent": name, "stage1": 100 * summ.get("extended_pdm_score_stage_one", np.nan),
                     "stage2": 100 * summ.get("extended_pdm_score_stage_two", np.nan),
                     "EPDMS": 100 * summ.get("extended_pdm_score_combined", np.nan)})
    board = pd.DataFrame(NAV_BOARD, columns=["row", "PDMS", "EPDMS", "navhard EPDMS"])
    return {"navsim_navtest": pd.DataFrame(rows), "navsim_paired": pd.DataFrame(pairs),
            "navsim_navhard": pd.DataFrame(hard), "navsim_board": board}


def continuation_share(wod_main: pd.DataFrame, nav: pd.DataFrame | None) -> pd.DataFrame:
    """I2: how much of each benchmark's top score a planner that never looks at the road already gets.
    share = baseline / top for higher-is-better metrics, top / baseline for L2. Sources per row in `source`."""
    from . import waymo as W
    from . import op_route as R
    rows = []
    add = lambda bench, metric, base, v, top, top_name, ref, ref_name, src, hib=True: rows.append({  # noqa: E731
        "benchmark": bench, "metric": metric, "baseline": base, "value": v, "top": top, "top_entry": top_name,
        "share_of_top": (v / top if hib else top / v) if v is not None else np.nan,
        "reference": ref, "reference_name": ref_name,
        "share_of_reference": (v / ref if hib else (np.nan if v == 0 else ref / v)) if (v is not None and ref) else np.nan,
        "source": src})
    wm = wod_main.set_index("row")["rfs_cluster"]
    ctx = R.rater_context()
    past = W.load_ego()[0][W.load_rater(W.load_index())[0]]
    ctrv = W.rater_feedback_score(W.baselines(past)["ctrv"], ctx["rtraj"], ctx["rscore"], ctx["speed"])
    ctrv_c = W.rfs_by_cluster(ctrv, ctx["cluster"])[0]
    for b, v, src in (("cv", wm["cv"], "wod_main.csv"), ("ctrv", ctrv_c, "waymo.baselines, same 479 frames"),
                      ("ego head: ridge ego", wm["ours ridge ego"], "wod_main.csv"),
                      ("ego head: cls ego K1024", wm["ours cls ego K1024"], "wod_main.csv")):
        add("WOD-E2E val (479 rater frames)", "RFS", b, v, 8.043, "RAP (test)", wm["logged future"], "logged future", src)
    if nav is not None:
        for metric, top, top_name in (("PDMS", 91.5, "SimWAM"), ("EPDMS", 90.2, "SimWAM")):
            n = nav[nav.metric == metric].set_index("agent")["score"]
            for b, a in (("cv", "cv"), ("ctrv", "heads_ctrv"), ("ego head: ridge ego", "heads_ridge_ego"),
                         ("ego head: cls ego K1024", "heads_cls_ego_K1024")):
                if a in n:
                    add("NAVSIM navtest", metric, b, n[a], top, top_name, n.get("human"), "human (log, our devkit)",
                        "navsim_navtest.csv")
        add("NAVSIM navtest", "PDMS", "ego MLP (literature)", 65.6, 91.5, "SimWAM", 94.8, "human (paper)", "arXiv 2406.15349 Tab. 2")
    # nuScenes, BEV-Planner's unified implementation on the 5119 valid samples (nuscenes-physicalai.md), L2 mean 1-3 s
    for b, v, src in (("cv (GoStraight)", 0.83, "BEV-Planner Tab. 1"), ("cv (ours, valid)", (0.38 + 0.82 + 1.40) / 3, "nuscenes-physicalai.md"),
                      ("ego MLP (literature)", 0.35, "BEV-Planner Tab. 1")):
        add("nuScenes val", "L2 avg (m)", b, v, 0.37, "VAD-Base (with ego status)", None, "", src, hib=False)
    add("Bench2Drive open loop (50 clips)", "L2 avg 2 s (m)", "ego MLP: AD-MLP (literature)", 3.64, 0.73, "UniAD-Base", None, "",
        "arXiv 2406.03877 Tab. 3", hib=False)
    add("Bench2Drive closed loop (220 routes)", "DS", "ego MLP: AD-MLP (literature)", 18.05, 90.6, "BLUE (decision 38)", None, "",
        "arXiv 2406.03877 Tab. 3; decision 38")
    hp = REPO / "research/results/openpilot-openloop/hugsim_base.csv"
    if hp.exists():
        h = pd.read_csv(hp).set_index("tag")["hd"]
        for tag in ("cv-official", "cv-fixed"):
            add("HUGSIM (64 scenarios)", "HD-Score", tag, h[tag], 0.299, "UniAD (paper Tab. 13)", h[tag.replace("cv", "ltf")],
                "LTF, same controller", "hugsim-exam scored-base")
    return pd.DataFrame(rows)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("wod", "navsim", "share"))
    ap.add_argument("--desire-run", default="")
    a = ap.parse_args()
    (REPO / "research/results/openpilot-openloop").mkdir(parents=True, exist_ok=True)
    rl = RunLog("openloop_standing", a.step)
    if a.step == "wod":
        out = wod(a.desire_run)
    elif a.step == "navsim":
        out = navsim()
    else:
        res = REPO / "research/results/openpilot-openloop"
        nav = pd.read_csv(res / "navsim_navtest.csv") if (res / "navsim_navtest.csv").exists() else None
        out = {"continuation_share": continuation_share(pd.read_csv(res / "wod_main.csv"), nav)}
    for name, t in out.items():
        t.to_csv(rl.dir / f"{name}.csv", index=False)
        t.to_csv(REPO / "research/results/openpilot-openloop" / f"{name}.csv", index=False, float_format="%.4f")
        rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".3f"))
    rl.close()


if __name__ == "__main__":
    main()
