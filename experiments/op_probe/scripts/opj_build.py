"""op_probe joint diagnosis (P2 vs WA-JEPA), data build: one per-token table per benchmark from outputs already on the box. No model runs.

  attrs   (envs/navsim2, CPU)  scene attributes per navtest token from the OpenScene logs + nuPlan map: city, local sun elevation, junction
                               (t0 rear axle or any logged future pose inside an INTERSECTION polygon), agents within 30 m, red light listed
  table   (.venv, CPU)         navtest table: scene attributes, logged-motion classes, devkit sub-scores of P2-F-s0 / s1 and WA-JEPA, op_probe
                               sets, true footprint margins / excursion geometry of each plan, op_probe probe-predicted margins along both
                               plans; decoder per-token scores; navhard (Protocol G) per-token table
  replay  (.venv, CPU)         WA-JEPA plans of the op_probe eval tokens -> joint/wa_eval_poses.npz (+ tokens) and P2-F-s0's devkit replay rows
                               (score/ablate_P2.csv, key full) -> joint/p2_eval_score.csv; then score WA with the devkit replay:
                               envs/navsim2/bin/python experiments/op_probe/scripts/opb_score.py --poses joint/wa_eval_poses.npz
                                 --tokens joint/wa_eval_tokens.txt --out joint/wa_eval_score.csv --procs 96
  cases   (.venv, CPU)         representative paired cases per quadrant (rule in pick_cases) -> joint/cases_pick.json
  export  (envs/navsim2, CPU)  scene export of the picked cases (pp_gap_export.export_case) -> joint/cases.json
  frames  (.venv, CPU)         model inputs of the picked cases: P2's protocol-W road frame (as pp_gap_frames) and the WA-JEPA camera views
                               -> joint/frames.npz

Out: $DATA_DIR/runs/op_probe/joint/. The figure / page generator is opj_figs.py (Mac).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_R / "experiments/op_parity/scripts"), str(_pl.Path(__file__).parent)]
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ROOT = D / "runs/op_probe"
OUT = ROOT / "joint"
CACHE = D / "runs/op_parity/cache"
GAP = D / "runs/op_parity/gap/gap_navtest_shap.npz"
PRED = D / "runs/op_lb/lb_navtest/preds"
WA_TRAJ = D / "runs/top10_t2/navsim/wajepa/20260926-122804/trajectory_cache/done_union.pkl"
TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
CITY = {"us-nv-las-vegas-strip": ("Las Vegas", 36.11, -115.17), "us-ma-boston": ("Boston", 42.34, -71.05),
        "us-pa-pittsburgh-hazelwood": ("Pittsburgh", 40.41, -79.95), "sg-one-north": ("Singapore", 1.30, 103.79)}
X0, Y0, RES05 = -8.0, -24.0, 0.5
CORNERS = np.array([[4.049, 1.1485], [4.049, -1.1485], [-1.127, 1.1485], [-1.127, -1.1485]])   # front-left, front-right, rear-left, rear-right
PROBE_STAGES = ["E", "P2-V", "P2-M", "P2-T", "P2-H", "WA-Cf", "WA-Ca", "WA-T", "WA-H"]


# ---------------------------------------------------------------- attrs (navsim2)
def sun_elevation(ts_us, lat, lon):
    """Solar elevation (deg) from UTC epoch microseconds (NOAA low-precision formulas, ~0.5 deg)."""
    t = ts_us / 1e6 / 86400.0 + 2440587.5 - 2451545.0                     # days since J2000
    g = np.radians((357.529 + 0.98560028 * t) % 360)
    q = (280.459 + 0.98564736 * t) % 360
    lam = np.radians(q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.00000036 * t)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)))
    gmst = (18.697374558 + 24.06570982441908 * t) % 24
    ha = np.radians(((gmst * 15 + lon - ra + 180) % 360) - 180)
    la = np.radians(lat)
    return float(np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha))))


def _attrs_log(job):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    import opb_labels as OB
    log, toks, futs = job
    fr = pickle.load(open(D / "datasets/navsim/navsim_logs/test" / f"{log}.pkl", "rb"))
    at = {f["token"]: f for f in fr}
    loc = fr[0]["map_location"]
    m = OB._map(loc)
    out = []
    for t, fut in zip(toks, futs):
        f = at[t]
        x, y = f["ego2global_translation"][:2]
        h = OB._quat_yaw(f["ego2global_rotation"])
        pts = [(0.0, 0.0)] + ([] if np.isnan(fut[0, 0]) else [tuple(p[:2]) for p in fut])
        c, s = np.cos(h), np.sin(h)
        inter = [m.is_in_layer(Point2D(x + c * px - s * py, y + s * px + c * py), L.INTERSECTION) for px, py in pts]
        a = f["anns"]
        d = np.hypot(a["gt_boxes"][:, 0], a["gt_boxes"][:, 1]) if len(a["gt_boxes"]) else np.zeros(0)
        nm = np.asarray(a["gt_names"])
        near = d < 30
        tl = f.get("traffic_lights") or []
        out.append(dict(token=t, map=loc, ts=int(f["timestamp"]), junction_t0=bool(inter[0]), junction_path=bool(any(inter)),
                        n_veh=int((near & (nm == "vehicle")).sum()), n_ped=int((near & (nm == "pedestrian")).sum()),
                        n_bike=int((near & (nm == "bicycle")).sum()), red_light=bool(any(bool(v[1]) for v in tl)), n_tl=len(tl)))
    return out


def cmd_attrs(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_probe", "joint-attrs", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        tab = np.load(CACHE / "lb_navtest/tab.npz")
        jobs = {}
        for t, lg, fu in zip(tab["names"], tab["log"], tab["fut"]):
            jobs.setdefault(lg, ([], []))
            jobs[lg][0].append(t), jobs[lg][1].append(fu)
        res = par.pmap(_attrs_log, [(lg, v[0], v[1]) for lg, v in jobs.items()], run=run, workers=a.workers)
        res.raise_if_failed()
        df = pd.DataFrame([r for part in res.values for r in part])
        df["city"] = df["map"].map(lambda k: CITY[k][0])
        df["sun_el"] = [sun_elevation(ts, CITY[mp][1], CITY[mp][2]) for ts, mp in zip(df.ts, df["map"])]
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "attrs_navtest.csv", index=False)
        run.summary.update(n=len(df), junction=float(df.junction_path.mean()), night=float((df.sun_el < -6).mean()))
        run.info(json.dumps(run.summary))


# ---------------------------------------------------------------- table (.venv)
def motion(fut, v0):
    """Logged-motion descriptors of the 4 s future (t0 rear-axle frame)."""
    if np.isnan(fut[0, 0]):
        return dict(dyaw=np.nan, y4=np.nan, ymax=np.nan, path_len=np.nan, v_end=np.nan, maneuver="unknown")
    P = np.vstack([[0, 0, 0], fut])
    dyaw = float(np.degrees(np.unwrap(P[:, 2])[-1]))
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    v_end = float(seg[-1] / 0.5)
    ymax = float(P[np.argmax(np.abs(P[:, 1])), 1])
    L = float(seg.sum())
    if v0 < 0.5 and L < 1.0:
        man = "stationary"
    elif v0 < 1.5 and v_end > 2.5:
        man = "launch"
    elif v0 > 2.5 and v_end < 0.5:
        man = "stop"
    elif dyaw > 20:
        man = "left turn"
    elif dyaw < -20:
        man = "right turn"
    elif abs(dyaw) < 8 and abs(fut[-1, 1]) > 2.0 and L > 10:
        man = "lane change"
    elif abs(dyaw) >= 8:
        man = "curve"
    else:
        man = "straight"
    return dict(dyaw=dyaw, y4=float(fut[-1, 1]), ymax=ymax, path_len=L, v_end=v_end, maneuver=man)


def foot_geom(sdf, p8, fut):
    """True footprint margin of a plan on the 0.5 m SDF and the geometry of its worst corner.
    Returns margin (m, min over 41 x 4 corners), the worst corner's side (+1 left, -1 right) and time (s), and the plan's end error vs the
    logged future: heading (deg) and lateral offset (m, along the logged end pose's left normal)."""
    import opb_probe as OP
    fp = OP.footprint(p8)
    v = OP.sample(sdf, RES05, fp)
    k = int(np.argmin(v))
    side = 1 if (k % 4) in (0, 2) else -1
    out = dict(margin=float(v[k]), side=side, t_worst=0.1 * (k // 4), corner_front=(k % 4) < 2)
    if not np.isnan(fut[0, 0]):
        h = fut[-1, 2]
        n = np.array([-np.sin(h), np.cos(h)])
        out.update(e_yaw=float(np.degrees(np.angle(np.exp(1j * (p8[-1, 2] - h))))), e_lat=float((p8[-1, :2] - fut[-1, :2]) @ n),
                   e_lon=float((p8[-1, :2] - fut[-1, :2]) @ np.array([np.cos(h), np.sin(h)])),
                   ade=float(np.linalg.norm(p8[:, :2] - fut[:, :2], axis=1).mean()), len_ratio=float(
                       np.linalg.norm(np.diff(np.vstack([[0, 0, 0], p8])[:, :2], axis=0), axis=1).sum()
                       / max(np.linalg.norm(np.diff(np.vstack([[0, 0, 0], fut])[:, :2], axis=0), axis=1).sum(), 1e-3)))
    return out


def _geom_job(args):
    sdf, plans, fut, rasters = args
    import opb_probe as OP
    r = {}
    for k, p in plans.items():
        if p is None:
            continue
        for kk, vv in foot_geom(sdf, p, fut).items():
            r[f"{k}_{kk}"] = vv
    sdf1 = OP.raster1m(sdf[None])[0]
    if plans.get("WA") is not None:
        r["WA_margin_true1"] = OP.margin(sdf1, 1.0, plans["WA"])
        for st, g in rasters.items():
            r[f"pm_WA|{st}"] = OP.margin(g.astype(np.float32), 1.0, plans["WA"])
    return r


def cmd_table(a):
    import pandas as pd
    from jevdrive import par
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_probe", "joint-table", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        tab = np.load(CACHE / "lb_navtest/tab.npz")
        names = tab["names"]
        z = np.load(GAP)
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        ix = np.array([pos[t] for t in names])
        df = pd.DataFrame({"token": names, "log": tab["log"], "v0": tab["speed"], "lht": tab["lht"],
                           "cmd": np.array(["left", "straight", "right", "unknown"])[tab["cmd"][:, -1].argmax(1)]})
        mot = pd.DataFrame([motion(f, v) for f, v in zip(tab["fut"], tab["speed"])])
        df = pd.concat([df, mot], axis=1)
        at = pd.read_csv(OUT / "attrs_navtest.csv").set_index("token").loc[names].reset_index(drop=True)
        df = pd.concat([df, at.drop(columns=["map"])], axis=1)
        for m, X in (("P2s0", z["X2"][0][ix]), ("P2s1", z["X2"][1][ix]), ("WA", z["Xw"][ix])):
            for j, t in enumerate(TERMS):
                df[f"{m}_{t}"] = X[:, j]
            ec = X[:, 8]
            df[f"{m}_score"] = np.prod(X[:, :4], -1) * (5 * X[:, 4] + 5 * X[:, 5] + 2 * X[:, 6] + 2 * X[:, 7] + 2 * np.nan_to_num(ec)) / (14 + 2 * np.isfinite(ec))
        S = np.load(ROOT / "sets/navtest_sets.npz")
        assert (S["tokens"] == names).all()
        for k in ("F", "R", "FF", "PP"):
            df[f"set_{k}"] = S[k]
        df["pp1500"] = False
        df.loc[np.load(ROOT / "sets/pp1500_rows.npy"), "pp1500"] = True
        # plans
        plans = {}
        for k, f in (("P2s0", "warp-cinque_PPP2-F-s0__base.npz"), ("P2s1", "warp-cinque_PPP2-F-s1__base.npz")):
            q = np.load(PRED / f)
            plans[k] = dict(zip(q["tokens"].tolist(), q["poses"]))
        plans["WA"] = pickle.load(open(WA_TRAJ, "rb"))["trajectories"]
        lab = dict(np.load(ROOT / "labels/navtest.npz"))
        assert (lab["tokens"] == names).all()
        rasters = {st: np.load(ROOT / f"probe/raster_{st}.npy", mmap_mode="r") for st in PROBE_STAGES}
        jobs = [(lab["sdf"][i].astype(np.float32), {k: (None if names[i] not in v else np.asarray(v[names[i]], np.float64)) for k, v in plans.items()},
                 tab["fut"][i], {st: np.asarray(r[i]) for st, r in rasters.items()}) for i in range(len(names))]
        res = par.pmap(_geom_job, jobs, run=run, workers=a.workers, desc="geometry")
        res.raise_if_failed()
        df = pd.concat([df, pd.DataFrame(res.values)], axis=1)
        po = np.load(ROOT / "probe/probe_outputs.npz")
        assert (po["tokens"] == names).all()
        df["P2s0_margin_true1"] = po["margin_true1"]
        df["P2s0_margin_true_probe"] = po["margin_true"]
        for st in PROBE_STAGES:
            df[f"pm_P2|{st}"] = po[f"{st}/margin"]
            df[f"corr_err|{st}"] = np.nanmean(np.abs(po[f"{st}/corr_direct"] - po["corr_true"]), 1)
        df["ok_label"] = lab["ok"]
        df.to_parquet(OUT / "navtest_tokens.parquet")
        # decoders: per token, per stage x objective
        dec = []
        for f, tag in (("decode.csv", ""), ("decode_h10.csv", "10")):
            q = pd.read_csv(ROOT / "score" / f)
            q[["stage", "obj"]] = q.key.str.split("|", expand=True)
            q["obj"] = q.obj + tag
            dec.append(q[["stage", "obj", "token", "drivable_area_compliance", "score", "raw_out", "out_depth"]])
        pd.concat(dec).rename(columns={"drivable_area_compliance": "DAC"}).to_parquet(OUT / "decoders.parquet")
        ab = pd.read_csv(ROOT / "score/ablate_P2.csv")
        ab[["token", "key", "drivable_area_compliance", "score"]].rename(columns={"drivable_area_compliance": "DAC"}).to_parquet(OUT / "ablate.parquet")
        # navhard, Protocol G (decision 145): P2-F seeds on GIMM frames, WA-JEPA harness run
        ar = D / "runs/op_parity"
        wa = pd.read_csv(ar / "navhard/harness/wajepa/harness_tokens.csv").set_index("token")
        nh = pd.DataFrame({"token": wa.index, "stage": wa.stage.values, "group": wa.group.values, "WA_weight": wa.weight.values})
        for m, f in (("WA", wa), ("P2s0", pd.read_csv(ar / "navhard_gimm/harness/P2-F-s0/harness_tokens.csv").set_index("token")),
                     ("P2s1", pd.read_csv(ar / "navhard_gimm/harness/P2-F-s1/harness_tokens.csv").set_index("token"))):
            f = f.loc[nh.token]
            for t, c in zip(TERMS, COLS):
                nh[f"{m}_{t}"] = f[c].values
            nh[f"{m}_score"], nh[f"{m}_weight"] = f.score.values, f.weight.values
        ht = np.load(CACHE / "lb_navhard/tab.npz")
        hp = {t: i for i, t in enumerate(ht["names"].tolist())}
        hi = np.array([hp[t] for t in nh.token])
        nh["v0"] = ht["speed"][hi]
        nh["log"] = ht["log"][hi]
        nh = pd.concat([nh, pd.DataFrame([motion(ht["fut"][i], ht["speed"][i]) for i in hi])], axis=1)
        nh.to_parquet(OUT / "navhard_tokens.parquet")
        run.summary.update(n=len(df), n_navhard=len(nh), wa_missing=int(df.WA_margin.isna().sum()))
        run.info(json.dumps(run.summary))


def cmd_replay(a):
    import pandas as pd
    w = pickle.load(open(WA_TRAJ, "rb"))["trajectories"]
    ev = [t for t in (x.strip() for x in open(ROOT / "sets/eval_tokens.txt")) if t and t in w]
    np.savez(OUT / "wa_eval_poses.npz", tokens=np.array(ev), WA=np.stack([np.asarray(w[t], np.float32) for t in ev]))
    (OUT / "wa_eval_tokens.txt").write_text("\n".join(ev) + "\n")
    q = pd.read_csv(ROOT / "score/ablate_P2.csv")
    q[q.key == "full"].to_csv(OUT / "p2_eval_score.csv", index=False)
    print(len(ev), "WA eval tokens")


# ---------------------------------------------------------------- cases
QUAD = {"P2 fails, WA passes": ("P2", "WA"), "WA fails, P2 passes": ("WA", "P2"), "both fail": ("both", None)}
GATES = ["NC", "DAC", "DDC", "TLC", "TTC"]


def gate_fail(df, m):
    return np.logical_or.reduce([df[f"{m}_{t}"] < 1 for t in GATES])


def pick_cases(df, per=3, seed=0):
    """Selection rule (shown on the page): a token 'fails' for a model when any of NC, DAC, DDC, TLC, TTC is below 1 (P2 = seed 0, the plan
    that is drawn; both P2 seeds must agree). Quadrants: P2 fails & WA passes, WA fails & P2 passes, both fail. Eligible: logged speed > 2 m/s,
    label raster ok, WA plan present. Within a quadrant, the failing sub-metric of the token is its first failing gate in the order above; each
    quadrant takes `per` tokens drawn uniformly at random (numpy seed 0) from the eligible set, stratified so that the drawn sub-metrics follow
    the quadrant's own sub-metric frequencies (largest-remainder rounding), distinct logs. No look at the plans before drawing."""
    rng = np.random.default_rng(seed)
    f2 = gate_fail(df, "P2s0") & gate_fail(df, "P2s1")
    p2 = ~gate_fail(df, "P2s0") & ~gate_fail(df, "P2s1")
    fw, pw = gate_fail(df, "WA"), ~gate_fail(df, "WA")
    elig = (df.v0 > 2) & df.ok_label & df.WA_margin.notna()
    masks = {"P2 fails, WA passes": f2 & pw, "WA fails, P2 passes": fw & p2, "both fail": f2 & fw}
    picks = []
    for q, m in masks.items():
        sub = df[m & elig].copy()
        who = "WA" if q.startswith("WA") else "P2s0"
        sub["gate"] = [next(t for t in GATES if r[f"{who}_{t}"] < 1) for _, r in sub.iterrows()]
        freq = sub.gate.value_counts()
        quota = freq / freq.sum() * per
        n = np.floor(quota).astype(int)
        for g in (quota - n).sort_values(ascending=False).index[: per - n.sum()]:
            n[g] += 1
        used = set()
        for g, k in n.items():
            pool = sub[sub.gate == g].index.to_numpy()
            pool = pool[rng.permutation(len(pool))]
            got = 0
            for i in pool:
                if got >= k or df.at[i, "log"] in used:
                    continue
                used.add(df.at[i, "log"])
                picks.append(dict(quadrant=q, gate=g, token=df.at[i, "token"], row=int(i), n_quadrant=int(m.sum()), n_eligible=len(sub),
                                  n_gate=int(freq[g])))
                got += 1
    return picks


def cmd_cases(a):
    import pandas as pd
    df = pd.read_parquet(OUT / "navtest_tokens.parquet")
    picks = pick_cases(df, a.per)
    json.dump({"rule": " ".join(pick_cases.__doc__.split()), "picks": picks}, open(OUT / "cases_pick.json", "w"), indent=1)
    for p in picks:
        print(p)


def cmd_export(a):
    import glob
    import lzma
    import pp_gap_export as GE
    picks = json.load(open(OUT / "cases_pick.json"))["picks"]
    idx = {e["token"]: e for e in pickle.load(open(D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
    mcache = {Path(p).parent.name: p for p in glob.glob(str(D / "runs/navsim/metric_cache/v2_navtest/*/*/*/metric_cache.pkl"))}
    plans = GE.load_plans()
    fz = np.load(D / "runs/navsim_zs/index/navtest_future.npz")
    fut = dict(zip(fz["tokens"].tolist(), fz["poses"]))
    z = np.load(GAP)
    zpos = {t: i for i, t in enumerate(z["tokens"].tolist())}
    cases = []
    for p in picks:
        with lzma.open(mcache[p["token"]], "rb") as f:
            mc = pickle.load(f)
        c = GE.export_case(p["token"], mc, idx[p["token"]], plans, fut, zpos[p["token"]], z)
        c.update(p)
        cases.append(c)
        print("exported", p["quadrant"], p["gate"], p["token"], flush=True)
    json.dump({"cases": cases}, open(OUT / "cases.json", "w"))


def cmd_frames(a):
    """P2's current road frame under protocol W (slot t0, as pp_gap_frames) and WA-JEPA's four views (L0 F0 R0 B0, resized to its
    512 x 256 input) for the picked tokens."""
    import op_lb as OL
    import pp_prep as PP
    from PIL import Image
    from pp_gap_frames import rgb
    from jevdrive import op_interp as I  # noqa: F401
    picks = json.load(open(OUT / "cases_pick.json"))["picks"]
    mt = OL.meta("lb_navtest")
    names = {t: i for i, t in enumerate(mt["names"])}
    keys = OL.Keys("lb_navtest")
    ts, src = OL._steps(0.0, False)
    idx = {e["token"]: e for e in pickle.load(open(D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
    out = {}
    for p in picks:
        t = p["token"]
        i = names[t]
        kf = np.asarray(keys[i])
        sf = PP._warp_job((kf, mt["pose"][i], mt["vel"][i], mt["cam"][i], np.asarray(mt["syn_t"])))
        s = PP.STEPS[-1]
        img = kf[src[s][1]] if src[s][0] == "k" else sf[src[s][1]]
        out[f"{t}/p2_road"] = rgb(img[0])
        out[f"{t}/p2_wide"] = rgb(img[1])
        log = idx[t]["log_name"]
        fr = {f["token"]: f for f in pickle.load(open(D / "datasets/navsim/navsim_logs/test" / f"{log}.pkl", "rb"))}[t]
        for cam in ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_B0"):
            im = Image.open(D / "datasets/navsim/sensor_blobs/test" / fr["cams"][cam]["data_path"]).convert("RGB").resize((512, 256), Image.BILINEAR)
            out[f"{t}/wa_{cam}"] = np.asarray(im)
        c0 = fr["cams"]["CAM_F0"]
        out[f"{t}/f0_K"] = np.asarray(c0["cam_intrinsic"], np.float64)
        out[f"{t}/f0_R"] = np.asarray(c0["sensor2lidar_rotation"], np.float64)
        out[f"{t}/f0_T"] = np.asarray(c0["sensor2lidar_translation"], np.float64)
        out[f"{t}/lidar2ego"] = np.asarray(fr["lidar2ego"], np.float64)
        out[f"{t}/f0_full"] = np.asarray(Image.open(D / "datasets/navsim/sensor_blobs/test" / c0["data_path"]).convert("RGB").resize((960, 540), Image.BILINEAR))
        print("frames", t, flush=True)
    np.savez_compressed(OUT / "frames.npz", **out)
    np.savez_compressed(OUT / "frames_small.npz", **{k: np.asarray(Image.fromarray(v).resize((384, 192), Image.LANCZOS))
                                                     for k, v in out.items() if k.endswith(("/p2_road", "/wa_CAM_F0"))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("attrs", "table"):
        p = sp.add_parser(n)
        p.add_argument("--workers", type=int, default=96)
    p = sp.add_parser("cases")
    p.add_argument("--per", type=int, default=3)
    sp.add_parser("replay")
    sp.add_parser("export")
    sp.add_parser("frames")
    a = ap.parse_args()
    {"attrs": cmd_attrs, "table": cmd_table, "replay": cmd_replay, "cases": cmd_cases, "export": cmd_export, "frames": cmd_frames}[a.cmd](a)
