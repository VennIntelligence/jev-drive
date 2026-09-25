"""Route into the head / into the backbone: does it buy back standstill and junction frames?
Pre-registration: todos/2026-09-25-openpilot-temporal-p5-and-route.md (experiments 2a and 2b).

  2a   zero GPU: the decision-40 (iii) heads_train predictions re-sliced onto the rater frames that start below
       0.5 m/s and the Interections / Multi-Lane Maneuvers clusters; paired RFS deltas of `cls_late` against cv,
       `cls ego` and the model's native plan
  2b   the same heads refitted on `temporal` extracted with a desire pulse from the WOD routing intent, and the
       native plan of that same extraction; paired deltas against the desire-free run on the full rater set,
       the two 2a subsets and decision 22's ADE readouts

Statistics are decision 40's unchanged: frame-mean RFS paired by sequence bootstrap (waymo_p1.paired).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import get_logger

log = get_logger(__name__)
MODELS = ("cinque", "lebowski")
STANDSTILL = 0.5                                    # m/s, decision 34's "starts at rest" rater frames
JUNCTION = ("Interections", "Multi-Lane Maneuvers")  # the clusters where the route decides (sic, WOD's spelling)
NOD40_RUN = "/root/autodl-tmp/ujs/runs/drive_backbones/heads_train/20260925-110819"
TAG = "p3drive_heads"


def rater_context() -> dict:
    """The 479 val rater frames: names, sequences, clusters, init speed, cv, rater arrays and the subset masks."""
    from . import waymo as W
    df = W.load_index()
    past, _ = W.load_ego()
    r, rtraj, rscore = W.load_rater(df)
    names = W.frame_names(df)[r]
    speed = W.init_speed(past[r])
    cl = df.cluster.astype(str).to_numpy()[r]
    subsets = {"all rater frames": np.ones(len(r), bool), f"standstill (v0 < {STANDSTILL} m/s)": speed < STANDSTILL,
               "Interections + Multi-Lane": np.isin(cl, JUNCTION),
               "Interections": cl == JUNCTION[0], "Multi-Lane Maneuvers": cl == JUNCTION[1]}
    return {"name": names, "seq": df.sequence.astype(str).to_numpy()[r], "cluster": cl, "speed": speed,
            "rtraj": rtraj, "rscore": rscore, "cv": W.baselines(past[r])["cv"], "sub": subsets}


def run_preds(run_dir, names: np.ndarray, tag: str = TAG, suffix: str = "") -> dict:
    """Every arm of a heads_train run on the given frames, arm names suffixed (e.g. ' +desire')."""
    z = np.load(Path(run_dir) / f"{tag}_preds_dir0.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["frame_name"])), index=z["frame_name"].astype(str)).reindex(names).to_numpy()
    assert not np.isnan(pos).any(), "rater frames missing from the run"
    pos = pos.astype(int)
    return {k[5:] + suffix: z[k][pos] for k in z.files if k.startswith("pred_")}


def native(names: np.ndarray, split: str = "trainval", suffix: str = "") -> dict:
    """openpilot's own plans (no fit) from a streamed extraction, converted to WOD rear-axle waypoints."""
    from .drive_backbones import root
    out = {}
    for m in MODELS:
        z = np.load(root() / f"op_{m}_{split}_native.npz")
        pos = pd.Series(np.arange(len(z["name"])), index=z["name"].astype(str)).reindex(names).to_numpy()
        assert not np.isnan(pos).any(), f"{split}: rater frames missing from the {m} extraction"
        out[f"native op-{m}{suffix}"] = z["wod"][pos.astype(int)]
    return out


def score(ctx: dict, preds: dict) -> dict:
    from . import waymo as W
    return {k: W.rater_feedback_score(p, ctx["rtraj"], ctx["rscore"], ctx["speed"]) for k, p in preds.items()}


def arm_table(ctx: dict, rfs: dict) -> pd.DataFrame:
    from . import waymo as W
    rows = []
    for sname, m in ctx["sub"].items():
        for k, v in rfs.items():
            rows.append({"subset": sname, "arm": k, "n": int(m.sum()), "rfs_frame_mean": float(v[m].mean()),
                         "rfs_cluster_mean": W.rfs_by_cluster(v[m], ctx["cluster"][m])[0]})
    return pd.DataFrame(rows)


def paired_table(ctx: dict, rfs: dict, pairs: list) -> pd.DataFrame:
    from . import waymo_p1 as P1
    return pd.DataFrame([{"subset": sname, "arm": a, "vs": b, **P1.paired(rfs[a], rfs[b], ctx["seq"], m)}
                         for sname, m in ctx["sub"].items() for a, b in pairs if a in rfs and b in rfs])


def exp2a(run_dir: str = NOD40_RUN) -> dict:
    ctx = rater_context()
    preds = run_preds(run_dir, ctx["name"]) | native(ctx["name"]) | {"cv": ctx["cv"]}
    rfs = score(ctx, preds)
    pairs = []
    for m in MODELS:
        c, rl = f"cls_late op-{m} temporal", f"A ridge_late op-{m} temporal"
        pairs += [(c, "cv"), (c, "cls ego K1024"), (c, f"native op-{m}"), (rl, "cv"), (f"native op-{m}", "cv")]
    pairs += [("cls ego K1024", "cv"), ("ridge ego", "cv")]
    log.info("2a: %s", {k: int(v.sum()) for k, v in ctx["sub"].items()})
    return {"route2a_arms": arm_table(ctx, rfs), "route2a_paired": paired_table(ctx, rfs, pairs)}


def exp2b(desire_run: str, run_dir: str = NOD40_RUN) -> dict:
    """Desire arms against their desire-free twins: native plans, ridge_late, cls_late (and ridge / cls ego,
    which do not read the feature and must come out unchanged)."""
    ctx = rater_context()
    preds = (run_preds(run_dir, ctx["name"]) | native(ctx["name"]) | {"cv": ctx["cv"]}
             | run_preds(desire_run, ctx["name"], suffix=" +desire") | native(ctx["name"], "trainval_desire", " +desire"))
    rfs = score(ctx, preds)
    pairs = []
    for m in MODELS:
        for a in (f"native op-{m}", f"A ridge_late op-{m} temporal", f"cls_late op-{m} temporal"):
            pairs += [(a + " +desire", a), (a + " +desire", "cv")]
    pairs += [("cls ego K1024 +desire", "cls ego K1024"), ("ridge ego +desire", "ridge ego")]
    return {"route2b_arms": arm_table(ctx, rfs), "route2b_paired": paired_table(ctx, rfs, pairs),
            "route2b_ade": ade_paired(desire_run, run_dir)}


def ade_paired(desire_run: str, run_dir: str = NOD40_RUN) -> pd.DataFrame:
    """Decision 22's ADE readouts (pre-onset / all frames / straight, s_ego deciles 1-9) for every desire arm
    against its desire-free twin, frame by frame on the same val rows (both runs share rows, split and s_ego)."""
    from . import waymo as W
    from . import waymo_l0 as l0
    from . import waymo_p1 as P1
    a = np.load(Path(run_dir) / f"{TAG}_preds_dir0.npz", allow_pickle=True)
    b = np.load(Path(desire_run) / f"{TAG}_preds_dir0.npz", allow_pickle=True)
    assert (a["frame_name"] == b["frame_name"]).all() and np.allclose(a["s_ego"], b["s_ego"]), "runs are not paired"
    df = W.load_index()
    past, future = W.load_ego()
    sub_all = W.subsets(df, past, future)
    idx = pd.Series(np.arange(len(df)), index=W.frame_names(df)).reindex(a["frame_name"].astype(str)).to_numpy().astype(int)
    s, gt, seq = a["s_ego"], a["fut"], df.sequence.astype(str).to_numpy()[idx]
    keep19 = np.clip(np.searchsorted(np.quantile(s, np.linspace(0, 1, 11))[1:-1], s, "right"), 0, 9) <= 8
    masks = {"pre-onset, deciles 1-9": sub_all["pre_onset"][idx] & keep19, "all frames, deciles 1-9": keep19,
             "straight, deciles 1-9": sub_all["straight_yaw"][idx] & keep19}
    rows = []
    for k in (x for x in b.files if x.startswith("pred_")):
        e1, e0 = l0.ade(b[k], gt), l0.ade(a[k], gt)
        rows += [{"arm": k[5:] + " +desire", "vs": k[5:], "readout": n, **P1.paired(e1, e0, seq, m)} for n, m in masks.items()]
    return pd.DataFrame(rows)


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("2a", "2b"))
    ap.add_argument("--desire-run", default="")
    a = ap.parse_args()
    rl = RunLog("op_route", a.step)
    out = exp2a() if a.step == "2a" else exp2b(a.desire_run)
    for name, t in out.items():
        t.to_csv(rl.dir / f"{name}.csv", index=False)
        rl.log.info("%s\n%s", name, t.to_markdown(index=False, floatfmt=".3f"))
    rl.close()


if __name__ == "__main__":
    main()
