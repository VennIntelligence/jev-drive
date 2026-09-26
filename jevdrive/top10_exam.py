"""Top-10 intersection exams, executor T1: SparseDriveV2 and ZTRS on our paired and zero-shot exams
(todos/2026-09-26-top10-intersection.md, sections 5.1-5.5 and the [T1] entries, each written before the numbers).

  plan   frames an exam needs -> processed/top10_exam/<set>/plan.json: the three current source JPEGs, the source rig
         (camgeom records) and the virtual NAVSIM cameras rendered from it, NAVSIM ego statuses, output offset
  judge  the runner's trajectories -> the exam's own judge, unchanged: P5 v1 BA = p5_exam.exam (+ elicit_e4's per-pair
         window), I3 = elicit_i3's judge (p5_exam.exam without the TFv6 columns)

The runner is scripts/top10_exam_infer.py (one per model env). Sets: p5 (P5 v1 BehaviorAgent), i3 (HUGSIM pairs).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import navsim_rig as R
from .common import data_dir, get_logger

log = get_logger(__name__)
SETS = {"p5": "carla_p5v1_ba", "i3": "hugsim_pairs", "wod": None}
MODELS = {"sparsedrivev2": "SparseDriveV2", "ztrs": "ZTRS"}
# I3's origin is the front camera; place it where nuPlan's CAM_F0 sits over the rear axle (the [T1] 10:05 entry)
I3_FRONT = np.array([1.67, 0.0, 1.52])
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "research" / "results" / "top10-exams"


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
    return {"src": src, "virt": [R.virtual(_yaw(src[i]), pos[i]) for i in order], "primary": list(order)}


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


def plan(set_: str) -> dict:
    from . import elicit_i3 as I, p5_exam as E
    if set_ == "wod":
        fr, rigs, offset = plan_wod()
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
    fl.to_csv(RESULTS / f"{set_}_flip_rates.csv", index=False)
    if set_ == "p5":
        pp.to_csv(RESULTS / "p5_per_pair.csv", index=False)
    return res


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "judge"))
    ap.add_argument("--set", choices=tuple(SETS), required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    if a.cmd == "plan":
        print(plan(a.set))
    else:
        rl = RunLog("top10_exam", f"judge-{a.set}")
        judge(rl, a.set, tuple(a.models.split(",")))
        rl.close()


if __name__ == "__main__":
    main()
