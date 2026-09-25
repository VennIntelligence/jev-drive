"""Fusion diagnostics Q6-SAM: the three rule gates of Q6 on SAM 3.1's perceived state instead of the simulator's
(the perception-limited floor). Pre-registration: todos/2026-09-25-fusion-diagnostics.md (Q6, and the [Q6-SAM] lines of
the deviation log). Gates, ego table, magnitudes and the exam are `fusion_diag.q6gt` unchanged; only the object states
on the observation and null frames come from here.

States per (run, tick k): every SAM detection of the P5 batch (score > 0.5; prompts pedestrian -> pedestrian,
vehicle / emergency vehicle / cyclist -> vehicle) from the three cameras, ground-contact pixel lifted onto the flat
ground (fusion_q4.lift), moved from the rear-axle frame to Q6's ego frame (origin at the vehicle location, the
actor origin: x += REAR_AXLE_X). Velocity: per class, Hungarian association in world coordinates (ego pose from
pose.jsonl, yaw only, as Q6 does) with the previous camera frame (k - 4, 0.2 s); gate 1.5 m for pedestrians, 4 m for
vehicles; velocity = displacement / 0.2 s, rotated into the current ego axes; unassociated detections get zero
velocity.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fusion_diag as FD
from . import fusion_q4 as Q
from .common import data_dir, get_logger

log = get_logger(__name__)
CLS = {"pedestrian": "pedestrian", "vehicle": "vehicle", "emergency vehicle": "vehicle", "cyclist": "vehicle"}
ASSOC = {"pedestrian": 1.5, "vehicle": 4.0}
DT = 0.2


def _run_states(run: str, ks: list, d: pd.DataFrame, gen: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    from scipy.optimize import linear_sum_assignment
    from . import p5_pairs as P
    adir = P.attempt(gen, run)
    if adir is None:
        return None
    _, ego = FD.gt_run(adir, run, sorted(set(ks) | {k - 4 for k in ks}))
    pose = pd.read_json(adir / "pose.jsonl", lines=True).drop_duplicates("frame")
    pose["k"] = (pose.t / 0.05).round().astype(int)
    f2k = dict(zip(pose.frame, pose.k))
    pk = pose.set_index("k")
    d = d.assign(k=d.frame.map(f2k)).dropna(subset=["k"])
    d["k"] = d.k.astype(int)
    # ego-frame (actor origin) positions and world positions
    d["x"], d["y"] = d.gx + Q.REAR_AXLE_X, d.gy
    th = -np.radians(pk.yaw.reindex(d.k).to_numpy())
    ex, ey = pk.x.reindex(d.k).to_numpy(), -pk.y.reindex(d.k).to_numpy()
    c, s = np.cos(th), np.sin(th)
    d["wx"], d["wy"] = ex + c * d.x - s * d.y, ey + s * d.x + c * d.y        # ego -> world (right-handed)
    d["vx"], d["vy"] = 0.0, 0.0
    by = {k: g for k, g in d.groupby("k")}
    for k in ks:
        cur, prev = by.get(k), by.get(k - 4)
        if cur is None or prev is None:
            continue
        for cls, lim in ASSOC.items():
            a, b = cur[cur.cls == cls], prev[prev.cls == cls]
            if not len(a) or not len(b):
                continue
            C = np.hypot(a.wx.to_numpy()[:, None] - b.wx.to_numpy()[None], a.wy.to_numpy()[:, None] - b.wy.to_numpy()[None])
            r, cc = linear_sum_assignment(np.where(C <= lim, C, 1e6))
            ok = C[r, cc] <= lim
            r, cc = r[ok], cc[ok]
            vw = np.stack([a.wx.to_numpy()[r] - b.wx.to_numpy()[cc], a.wy.to_numpy()[r] - b.wy.to_numpy()[cc]], 1) / DT
            t_ = -np.radians(pk.loc[k, "yaw"])
            R = np.array([[np.cos(t_), np.sin(t_)], [-np.sin(t_), np.cos(t_)]])  # world -> ego
            v = vw @ R.T
            d.loc[a.index[r], "vx"], d.loc[a.index[r], "vy"] = v[:, 0], v[:, 1]
    st = d[d.k.isin(ks)].assign(run=run, obj=np.arange(int(d.k.isin(ks).sum())))
    return st[["run", "k", "obj", "cls", "x", "y", "vx", "vy"]], ego[ego.k.isin(ks)]


def sam_gates(need: dict, det_dir: Path) -> pd.DataFrame:
    from joblib import Parallel, delayed
    L = Q.root("lists")
    p5 = pd.read_parquet(L / "p5.parquet")
    d = Q.load_dets(det_dir)
    d = d[d.prompt.isin(list(CLS))]
    d = Q.lift_dets(d, d.key.str.split("|").str[1].to_numpy(), Q.p5_calib())
    d = d[d.lift_ok].assign(cls=lambda x: x.prompt.map(CLS))
    fn = d.key.str.split("|").str[0]
    d["run"], d["frame"] = fn.str.rsplit("-", n=1).str[0], fn.str.rsplit("-", n=1).str[1].astype(int)
    gen = data_dir() / "runs" / "p5_pairs" / "gen"
    groups = {r: g for r, g in d.groupby("run")}
    empty = d.iloc[:0]
    res = Parallel(16, verbose=2)(delayed(_run_states)(r, sorted(ks), groups.get(r, empty), gen) for r, ks in sorted(need.items()))
    res = [x for x in res if x is not None]
    st = pd.concat([x[0] for x in res], ignore_index=True)
    eg = pd.concat([x[1] for x in res], ignore_index=True)
    log.info("SAM states: %d objects on %d frames (%.1f per frame); %d ego frames", len(st),
             st.groupby(["run", "k"]).ngroups, len(st) / max(1, len(eg)), len(eg))
    G = FD.gates(st, eg).set_index(["run", "k"])
    return G


def main():
    import argparse
    from .runlog import RunLog
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", default=str(data_dir() / "processed" / "fusion_diag" / "sam" / "p5"))
    a = ap.parse_args()
    rl = RunLog("fusion_diag", "q6sam")
    rl.event("start", args=vars(a))
    out = FD.q6gt(rl, obs_gates=lambda need: sam_gates(need, Path(a.dets)))
    for name, v in out.items():
        if isinstance(v, pd.DataFrame):
            if name.startswith("_"):
                v.to_parquet(rl.dir / f"{name[1:]}.parquet", index=False)
                continue
            name = name.replace("_gt", "_sam")
            v.to_csv(rl.dir / f"{name}.csv", index=False)
            rl.log.info("%s\n%s", name, v.to_markdown(index=False, floatfmt=".3f"))
        else:
            (rl.dir / f"{name.strip('_')}.json").write_text(json.dumps(v, indent=2, default=str))
    rl.event("end")
    rl.close()


if __name__ == "__main__":
    main()
