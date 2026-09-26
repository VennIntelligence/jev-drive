"""Top-10 intersection exams, executor T1: SparseDriveV2 and ZTRS on our paired and zero-shot exams
(todos/2026-09-26-top10-intersection.md, sections 5.1-5.5 and the [T1] entries, each written before the numbers).

  plan   frames an exam needs -> processed/top10_exam/<set>/plan.json: the three current source JPEGs, the source rig
         (camgeom records) and the virtual NAVSIM cameras rendered from it, NAVSIM ego statuses, output offset
  judge  the runner's trajectories -> the exam's own judge, unchanged: P5 v1 BA = p5_exam.exam (+ elicit_e4's per-pair
         window), I3 = elicit_i3's judge (p5_exam.exam without the TFv6 columns)

The runner is scripts/top10_exam_infer.py (one per model env). Sets: p5 (P5 v1 BehaviorAgent), i3 (HUGSIM pairs),
wod, nusc, p6 (P6 v0 exam frames of night queue 3, lane C: jevdrive.nq3_p6 frame list, P5 v1's recorder and rig).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import navsim_rig as R
from .common import data_dir, get_logger

log = get_logger(__name__)
SETS = {"p5": "carla_p5v1_ba", "i3": "hugsim_pairs", "wod": None, "nusc": None, "p6": "carla_p6"}
MODELS = {"sparsedrivev2": "SparseDriveV2", "ztrs": "ZTRS"}
# I3's origin is the front camera; place it where nuPlan's CAM_F0 sits over the rear axle (the [T1] 10:05 entry)
I3_FRONT = np.array([1.67, 0.0, 1.52])
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "top10-exams"      # shared with T2: every T1 file is t1_*


def root(*p) -> Path:
    d = data_dir() / "processed" / "top10_exam"
    d.joinpath(*p[:-1]).mkdir(parents=True, exist_ok=True) if p else d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*p)


def _yaw(rec: dict) -> float:
    E = np.asarray(rec["extrinsic"]).reshape(4, 4)
    return float(np.degrees(np.arctan2(E[1, 0], E[0, 0])))


def _rig(src: list, shift=np.zeros(3)) -> dict:
    """src: camgeom records (front, front_left, front_right). Virtual l0 / f0 / r0 = nuPlan cameras at the mapped
    source camera's yaw and position (shifted into the rear-axle ego frame)."""
    src = [dict(r) for r in src]
    for r in src:
        E = np.asarray(r["extrinsic"], np.float64).reshape(4, 4).copy()
        E[:3, 3] += shift
        r["extrinsic"] = E.ravel().tolist()
    pos = [np.asarray(r["extrinsic"]).reshape(4, 4)[:3, 3] for r in src]
    order = (1, 0, 2)                                           # l0 <- front_left, f0 <- front, r0 <- front_right
    return {"src": src, "virt": [R.virtual(n, _yaw(src[i]), pos[i]) for n, i in zip(R.NAMES, order)],
            "primary": list(order)}


def _frames(t: pd.DataFrame, past: np.ndarray, rows: np.ndarray) -> dict:
    from .night2_n3 import nav_ego
    f = t.iloc[rows]
    files = np.stack([f.files.map(lambda x: x[k]).to_numpy() for k in (3, 7, 11)], 1)   # current front, left, right
    assert all(p.endswith(f"/{c}/{n:07d}.jpg") for row, n in zip(files, f.frame)
               for p, c in zip(row, ("front", "front_left", "front_right")))
    return {"frame_name": f.frame_name.to_numpy(), "files": files,
            "ego": nav_ego(past[rows], f.intent.to_numpy())}


def plan_wod() -> dict:
    """WOD-E2E val rater (479) + ADE-extra (958) frames: FRONT / FRONT_LEFT / FRONT_RIGHT of frame f from the
    zero-shot exam's packages, one rig per sequence (the calibration of its first planned frame)."""
    from . import wod_zeroshot as Z
    from .night2_n3 import nav_ego
    S = Z.load_sets()
    names = np.r_[S["rater"]["name"], S["extra"]["name"]].astype(str)
    seq = np.r_[S["rater"]["sequence"], S["extra"]["sequence"]].astype(str)
    past = np.r_[S["rater"]["past"], S["extra"]["past"]]
    intent = np.r_[S["rater"]["intent"], S["extra"]["intent"]]
    pk = [str(Z.root("packages") / f"{n}.npz") for n in names]
    files = np.array([[f"{p}::jpg_3_{c}" for c in (1, 2, 3)] for p in pk], object)   # k = 3 is frame f itself
    rigs = {}
    for s_, p in zip(seq, pk):
        if s_ not in rigs:
            z = np.load(p)
            c = Z.read_calib(z, (1, 2, 3))
            rigs[s_] = _rig([{k: (np.asarray(v).tolist() if not isinstance(v, int) else v) for k, v in c[i].items()}
                             for i in (1, 2, 3)])
    return {"frame_name": names, "files": files, "ego": nav_ego(past, intent), "rig": seq}, rigs, np.zeros(2)


NUSC_CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def nusc_past(scene: dict, t0: int) -> np.ndarray:
    """(16, 6) WOD-shaped past from the 20 Hz ego poses (no CAN bus on the box, [T1] nuScenes entry): rear-axle
    position at t0 - 3.75 ... t0 in the t0 frame, velocity by central difference over +-50 ms, and the velocity change
    per 0.25 s step in the accel columns (the WOD / P4 convention)."""
    from . import nuscenes_zs as N
    t = t0 + (np.arange(16) - 15) * 250_000
    x0, R0 = N.ego_at(scene, [t0])
    to0 = lambda xyz: (xyz - x0[0]) @ R0[0]                        # noqa: E731  world -> t0 frame (rows: R^T x)
    pos = to0(N.ego_at(scene, t)[0])
    v = (to0(N.ego_at(scene, t + 50_000)[0]) - to0(N.ego_at(scene, t - 50_000)[0])) / 0.1
    dv = np.r_[np.zeros((1, 3)), np.diff(v, axis=0)]
    return np.c_[pos[:, :2], v[:, :2], dv[:, :2]].astype(np.float32)


def plan_nusc() -> tuple:
    """nuScenes val main (4636): CAM_FRONT / FRONT_LEFT / FRONT_RIGHT keyframe images, one rig per scene; command from
    the VAD converter rule (as the zero-shot exam) -> NAVSIM one-hot through the WOD intent codes."""
    from . import nuscenes_zs as N
    from .common import dataroot
    from .night2_n3 import nav_ego
    idx = N.load_index()
    main = set(json.loads(N.index_path().with_name("sets.json").read_text())["main"])
    rows = [e for e in idx["samples"] if e["token"] in main]
    files, rig, past, intent, rigs = [], [], [], [], {}
    code = {"straight": 1, "left": 2, "right": 3}                  # waymo.INTENTS
    for e in rows:
        sc = idx["scenes"][e["scene"]]
        files.append([str(dataroot() / sc["cams"][c]["path"][N.frame_at(sc, c, e["t0"])]) for c in NUSC_CAMS])
        rig.append(e["scene"])
        past.append(nusc_past(sc, e["t0"]))
        intent.append(code[e["cmd"]])
        if e["scene"] not in rigs:
            rigs[e["scene"]] = _rig([{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in N.cam_calib(sc, c).items()} for c in NUSC_CAMS])
    fr = {"frame_name": np.array([e["token"] for e in rows]), "files": np.array(files, object), "rig": np.array(rig),
          "ego": nav_ego(np.stack(past), np.array(intent))}
    return fr, rigs, np.zeros(2)


def p6_frames(priorities=(0, 1, 2)) -> set:
    """Frame names of the P6 v0 exam frame list (jevdrive.nq3_p6) with these priorities."""
    f = pd.read_parquet(data_dir() / "processed" / SETS["p6"] / "nq3_exam_frames.parquet", columns=["frame_name", "priority"])
    return set(f.frame_name[f.priority.isin(priorities)])


def plan_p6(priorities=(0, 1, 2)) -> tuple:
    """The frames of the P6 v0 exam frame list with these priorities. P6 was recorded by P5 v1's recorder: same index
    layout, same rig, so the p5 branch's frames, ego statuses and rig apply unchanged."""
    from .p5_openpilot import carla_calib
    d = data_dir() / "processed" / SETS["p6"]
    t = pd.read_parquet(d / "index.parquet")
    need = p6_frames(priorities)
    rows = np.flatnonzero(t.frame_name.isin(need).to_numpy())
    fr = _frames(t, np.load(d / "past.npy", mmap_mode="r"), rows)
    c = carla_calib()
    fr["rig"] = np.full(len(rows), "p5", object)
    return fr, {"p5": _rig([c["1"], c["2"], c["3"]])}, np.zeros(2)


def plan(set_: str, priorities=(0, 1, 2)) -> dict:
    from . import elicit_i3 as I, p5_exam as E
    if set_ in ("wod", "nusc", "p6"):
        fr, rigs, offset = {"wod": plan_wod, "nusc": plan_nusc, "p6": lambda: plan_p6(priorities)}[set_]()
        return _write_plan(set_, fr, rigs, offset)
    with I.p5_set(SETS[set_]):
        t, past, _, obs, null, _ = E.load()
    need = set(obs.fn_plus) | set(obs.fn_minus) | set(null.fn_plus) | set(null.fn_null)
    rows = np.flatnonzero(t.frame_name.isin(need).to_numpy())
    fr = _frames(t, past, rows)
    if set_ == "p5":
        from .p5_openpilot import carla_calib
        c = carla_calib()
        rigs = {"p5": _rig([c["1"], c["2"], c["3"]])}
        fr["rig"] = np.full(len(rows), "p5", object)
        offset = np.zeros(2)
    else:
        from PIL import Image
        keys = t.base_id.to_numpy()[rows]
        rigs = {}
        for k in np.unique(keys):
            wh = Image.open(fr["files"][np.flatnonzero(keys == k)[0], 0]).size
            c = I.scene_calib(json.loads((I._root() / "scenes" / k / "meta.json").read_text())["cams"], wh)
            rigs[k] = _rig([c["1"], c["2"], c["3"]], I3_FRONT)
        fr["rig"] = keys
        offset = I3_FRONT[:2]
    return _write_plan(set_, fr, rigs, offset)


def _write_plan(set_, fr, rigs, offset) -> dict:
    order = np.argsort(fr["rig"], kind="stable")               # workers see one rig at a time
    fr = {k: v[order].tolist() for k, v in fr.items()}
    rigs = {k: {"src": r["src"], "primary": r["primary"],
                "virt": [{n: np.asarray(a).tolist() for n, a in v.items()} for v in r["virt"]]} for k, r in rigs.items()}
    p = {"set": set_, "frames": fr, "rigs": rigs, "offset": offset.tolist()}
    root(set_, "plan.json").write_text(json.dumps(p))             # JSON: the model envs run numpy 1.23
    info = {"set": set_, "rigs": len(rigs),
            "yaws": {k: [round(_yaw(s), 1) for s in r["src"]] for k, r in list(rigs.items())[:3]},
            "frames": len(fr["frame_name"])}
    (root(set_, "plan_info.json")).write_text(json.dumps(info, indent=1))
    log.info("plan %s", info)
    return info


def load_preds(set_: str, model: str, t: pd.DataFrame) -> np.ndarray:
    """(len(t), 20, 2) on the 0.25 s grid in the exam's frame; NaN on frames the exam does not reference."""
    z = np.load(root(set_, f"{model}.npz"), allow_pickle=True)
    pos = pd.Series(np.arange(len(t)), index=t.frame_name)
    out = np.full((len(t), 20, 2), np.nan, np.float32)
    out[pos[z["frame_name"]].to_numpy()] = z["grid"]
    return out


def judge(rl, set_: str, models=tuple(MODELS)):
    from . import elicit_e4 as E4, elicit_i3 as I, p5_exam as E
    with I.p5_set(SETS[set_]):
        t, _, _, obs, null, pairs = E.load()
    preds = {MODELS[m]: load_preds(set_, m, t) for m in models}
    if set_ == "i3":
        E.TFV6 = {}
    o, n = E.deltas(obs, null, t, preds)
    ex = list(E.TFV6) + list(preds)
    res = E.exam(o, n, pairs, ex)
    d = rl.dir
    res["flips"].to_csv(d / "flip_rates.csv", index=False)
    res["validity"].to_csv(d / "label_validity.csv", index=False)
    res["obs"].to_parquet(d / "obs_scored.parquet", index=False)
    n.to_parquet(d / "null_scored.parquet", index=False)
    summ = {"tau_exp": res["tau_exp"], "pooled_families": res["pooled_families"],
            **{f"tau {k}": v for k, v in res["taus"].items()}}
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    fl = res["flips"]
    rl.log.info("%s: tau_exp %.3f, pooled %s\n%s", set_, res["tau_exp"], res["pooled_families"],
                fl[fl.scope == "pooled"].to_markdown(index=False, floatfmt=".3f"))
    out = [fl.assign(set=set_, window="per frame")]
    if set_ == "p5":                                            # E4 (b): per pair, null per case
        ob = res["obs"].copy()
        ob["t_vis"] = pairs.set_index(["base_id", "seed"]).t_vis.reindex(
            pd.MultiIndex.from_frame(ob[["base_id", "seed"]])).to_numpy()
        s0 = pairs[pairs.seed == 0].set_index("base_id")
        nn = n.copy()
        nn["t_vis"] = s0.t_vis.fillna(s0.t_trig).reindex(nn.base_id).to_numpy()
        lat = E4.latency
        E4.latency = lambda e: 0.0 if e in preds else lat(e)
        pp = E4.score(ob, nn, res["taus"], res["pooled_families"], list(preds), nn)
        E4.latency = lat
        pp = pp[pp.window != "(a) gated"]
        pp.to_csv(d / "per_pair.csv", index=False)
        rl.log.info("per frame / per pair\n%s", pp.pivot_table(index="examinee", columns=["scope", "window"],
                                                              values="flip", sort=False).to_markdown(floatfmt=".3f"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    fl.to_csv(RESULTS / f"t1_{set_}_flip_rates.csv", index=False)
    if set_ == "p5":
        pp.to_csv(RESULTS / "t1_p5_per_pair.csv", index=False)
    return res


# ---------------------------------------------------------------- WOD-E2E val (zeroshot-exam/wod-e2e.md + decision 22)

WOD_REFS = ("cv", "logged future", "ours cls ego", "Alpamayo 1.5 nav", "openpilot Cinque")


def _wzs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wod_zeroshot_script", REPO / "scripts" / "wod_zeroshot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _wod_rows(names: np.ndarray, past: np.ndarray, fut: np.ndarray, models) -> dict:
    """Every row's (n, K, 20, 2) prediction on these frames: ours, the stored zero-shot runs, the baselines."""
    from . import waymo as W
    wz = _wzs()
    rows = {"cv": W.baselines(past)["cv"][:, None], "logged future": fut[:, None, :, :2]}
    rows |= {k: v[:, None] for k, v in wz._ours(list(names)).items() if k == "ours cls ego"}
    alp, _ = wz._alp(list(names), "nav")
    op = wz._op(list(names), "cinque")
    rows |= {"Alpamayo 1.5 nav": alp, "openpilot Cinque": op[:, None]}
    for m in models:
        z = np.load(root("wod", f"{m}.npz"), allow_pickle=True)
        at = pd.Series(np.arange(len(z["frame_name"])), index=z["frame_name"].astype(str))
        rows[MODELS[m]] = z["grid"][at[names].to_numpy()][:, None]
    return rows


def judge_wod(rl, models=tuple(MODELS), B: int = 10000):
    """Rater frames: RFS cluster mean (leaderboard) + frame mean, paired deltas under one stratified bootstrap.
    Rater + ADE-extra frames: ADE@5 s vs the logged future on s_ego deciles 1-9 (the main judge, decision 22) and on
    the top decile, paired deltas under a sequence bootstrap. K > 1 rows (Alpamayo, 6 samples) score E[1 sample]."""
    from . import traj, waymo as W, wod_zeroshot as Z
    wz = _wzs()
    S = Z.load_sets()
    r = S["rater"]
    rn = r["name"].astype(str)
    rows = _wod_rows(rn, r["past"], r["future"], models)
    speed, tr, sc = W.init_speed(r["past"]), r["traj"], r["scores"]
    rfs = {k: np.mean([W.rater_feedback_score(p[:, j].astype(np.float64), tr.astype(np.float64), sc, speed)
                       for j in range(p.shape[1])], 0) for k, p in rows.items()}
    cl = r["cluster"].astype(str)
    rng = np.random.default_rng(0)
    sidx = wz._strat_idx(cl, B, rng)
    cboot = lambda x: np.mean([x[g].mean(1) for g in sidx], 0)            # noqa: E731
    cmean = lambda x: float(pd.Series(x).groupby(cl).mean().mean())       # noqa: E731
    out = []
    for k, x in rfs.items():
        b = cboot(x)
        row = {"row": k, "n": len(x), "rfs": cmean(x), "rfs_lo": np.percentile(b, 2.5), "rfs_hi": np.percentile(b, 97.5),
               "rfs_frame": float(x.mean()), "floored": float((x <= W.RFS_FLOOR + 1e-9).mean())}
        for ref in WOD_REFS:
            if ref != k:
                d = b - cboot(rfs[ref])
                row |= {f"d_{ref}": cmean(x) - cmean(rfs[ref]), f"d_{ref}_lo": np.percentile(d, 2.5),
                        f"d_{ref}_hi": np.percentile(d, 97.5)}
        out.append(row)
    res_rfs = pd.DataFrame(out)
    # ADE on the s_ego deciles: rater + ADE-extra frames, decile edges on the whole of val (P0's s_ego)
    e = S["extra"]
    names = np.r_[rn, e["name"].astype(str)]
    seq = np.r_[r["sequence"], e["sequence"]].astype(str)
    fut = np.r_[r["future"], e["future"]][..., :2]
    allrows = _wod_rows(names, np.r_[r["past"], e["past"]], np.r_[r["future"], e["future"]], models)
    z = np.load(data_dir() / wz.P0_PREDS, allow_pickle=True)
    s_all = z["s_ego"]
    edges = np.quantile(s_all, np.linspace(0, 1, 11))[1:-1]
    s_at = pd.Series(s_all, index=z["frame_name"].astype(str))[names].to_numpy()
    dec = np.clip(np.searchsorted(edges, s_at, "right"), 0, 9)
    ade = {k: np.linalg.norm(p - fut[:, None], axis=-1).mean(-1).mean(1) for k, p in allrows.items()}
    out = []
    for scope, m in (("s_ego dec 1-9", dec < 9), ("s_ego top decile", dec == 9), ("all", np.ones(len(dec), bool))):
        for k, a in ade.items():
            if k == "logged future":
                continue
            lo, hi = traj.boot_ci(a[m], seq[m], b=B)
            row = {"scope": scope, "row": k, "n": int(m.sum()), "ade5": float(a[m].mean()), "lo": lo, "hi": hi}
            for ref in ("cv", "ours cls ego", "Alpamayo 1.5 nav", "openpilot Cinque"):
                if ref != k:
                    v = a[m] - ade[ref][m]
                    dl, dh = traj.boot_ci(v, seq[m], b=B)
                    row |= {f"d_{ref}": float(v.mean()), f"d_{ref}_lo": dl, f"d_{ref}_hi": dh}
            out.append(row)
    res_ade = pd.DataFrame(out)
    for df, name in ((res_rfs, "t1_wod_rfs.csv"), (res_ade, "t1_wod_ade_sego.csv")):
        df.to_csv(rl.dir / name, index=False)
        df.to_csv(RESULTS / name, index=False)
    np.savez_compressed(rl.dir / "per_frame.npz", names=names, dec=dec, **{f"ade/{k}": v for k, v in ade.items()},
                        **{f"rfs/{k}": v for k, v in rfs.items()})
    rl.log.info("RFS (rater, n=%d)\n%s", len(rn), res_rfs[["row", "rfs", "rfs_lo", "rfs_hi", "rfs_frame", "d_cv", "d_cv_lo",
                                                          "d_cv_hi", "floored"]].to_markdown(index=False, floatfmt=".3f"))
    rl.log.info("ADE@5 s vs log by s_ego decile\n%s", res_ade[["scope", "row", "n", "ade5", "lo", "hi", "d_cv", "d_cv_lo",
                                                              "d_cv_hi"]].to_markdown(index=False, floatfmt=".3f"))


# ---------------------------------------------------------------- nuScenes main (zeroshot-exam/nuscenes-physicalai.md)

def judge_nusc(rl, models=tuple(MODELS)):
    """The zero-shot exam's own scoring (scripts/nusc_zs.py cmd_score, main + valid sets) with our rows added: raw
    rear-axle trajectories + headings -> LIDAR_TOP points at the GT times (nuscenes_zs.to_lidar_point, as Alpamayo's)."""
    import importlib.util
    from types import SimpleNamespace
    from . import nuscenes_zs as N
    spec = importlib.util.spec_from_file_location("nusc_zs_script", REPO / "scripts" / "nusc_zs.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    idx = N.load_index()
    by = {e["token"]: e for e in idx["samples"]}
    extra = {}
    for m in models:
        z = np.load(root("nusc", f"{m}.npz"), allow_pickle=True)
        dt = 0.5 if z["raw"].shape[1] == 8 else 0.1
        ts = np.arange(1, z["raw"].shape[1] + 1) * dt
        extra[f"t1:{m}"] = {t: N.to_lidar_point(ts, r[:, :2], r[:, 2], idx["scenes"][by[t]["scene"]]["lidar_xyz"],
                                                by[t]["fut_t"])[0].astype(np.float32)
                            for t, r in zip(z["frame_name"].astype(str), z["raw"])}
        S.ROWS[MODELS[m]] = f"t1:{m}"
    lp = S.load_preds
    S.load_preds = lambda i: lp(i) | extra
    zr = N.root
    N.root = lambda *p: rl.dir if p == ("score",) else zr(*p)      # never overwrite the zero-shot exam's own score dir
    try:
        S.cmd_score(SimpleNamespace(sets="main", boot=10000), rl)
    finally:
        N.root, S.load_preds = zr, lp
    for f in ("results.csv", "by_command.csv"):
        (RESULTS / f"t1_nusc_{f}").write_text((rl.dir / f).read_text())


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "judge"))
    ap.add_argument("--set", choices=tuple(SETS), required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--priorities", default="0,1,2", help="p6: frame-list priorities to plan")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(plan(a.set, tuple(int(x) for x in a.priorities.split(","))))
    else:
        rl = RunLog("top10_exam", f"judge-{a.set}")
        RESULTS.mkdir(parents=True, exist_ok=True)
        ms = tuple(a.models.split(","))
        {"wod": lambda: judge_wod(rl, ms), "nusc": lambda: judge_nusc(rl, ms)}.get(a.set, lambda: judge(rl, a.set, ms))()
        rl.close()


if __name__ == "__main__":
    main()
