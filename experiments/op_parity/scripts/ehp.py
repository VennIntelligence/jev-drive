"""Ego-history probe (results/ego_history_probe.md): does P2H's / WA-JEPA's planned speed before a sharp turn ride on the logged ego history?

Inference only. The ego inputs (velocity, acceleration, 4-pose history) of a navtest token are replaced by a constant-velocity history at the
token's current speed, the plan is recomputed, planned distance / speed are compared with the unmodified input.

  select   token sets T (turn approach), P (pre-flip), C (matched straight controls)  -> $DATA_DIR/runs/op_parity/ehp/sets.csv (+ results/ehp_sets.csv)
           The rule is fixed in the result file before any plan is read; this step reads logs / inputs only.
  p2h      P2H10-F-s0 / s1 plans on ALL navtest tokens x variants (orig, cv, cvlong, acc0, pose)  -> ehp/p2h_<arm>_<variant>.npy (n, 8, 3)
  wareq    WA-JEPA request file for the selected tokens x the same variants                    -> ehp/wa_req_<variant>.npz
  report   tables with cluster bootstrap over logs                                             -> results/ehp_*.csv, printed markdown

Variants (all edit only ego inputs; command, images unchanged): v = token speed (|vel| at t0), dt = 0.5 s.
  orig    unmodified
  cv      full constant velocity: vx = v, vy = 0, ax = ay = 0, poses x = -v * dt * (3, 2, 1, 0), y = yaw = 0
  cvlong  longitudinal only: vx = v, ax = 0, past-pose x = -v * dt * k (y, yaw, vy, ay untouched)
  acc0    ax = ay = 0 only (poses and velocity untouched)
  pose    past poses x = -v * dt * k only (velocity / acceleration untouched)
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_openloop/lib"), str(_pl.Path(__file__).parent)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

OUT = data_dir() / "runs/op_parity/ehp"
RES = _R / "experiments/op_parity/results"
VARIANTS = ["orig", "cv", "cvlong", "acc0", "pose"]
ARMS = ["P2H10-F-s0", "P2H10-F-s1"]
DT = 0.5


def tab():
    return dict(np.load(data_dir() / "runs/op_parity/cache/lb_navtest/tab.npz"))


def edit(pose, vel, acc, v, variant):
    """pose (n, 4, 3), vel / acc (n, 4, 2) body frame (row 3 = t0), v (n,) -> edited copies."""
    pose, vel, acc = pose.copy(), vel.copy(), acc.copy()
    k = np.array([3, 2, 1, 0], np.float32)
    px = -v[:, None] * DT * k[None]
    if variant in ("cv", "cvlong"):
        vel[:, 3, 0] = v
        acc[:, 3, 0] = 0
    if variant == "cv":
        vel[:, 3, 1] = 0
        acc[:, 3, 1] = 0
        pose[:, :, 1] = 0
        pose[:, :, 2] = 0
    if variant == "acc0":
        acc[:, 3, :] = 0
    if variant in ("cv", "cvlong", "pose"):
        pose[:, :, 0] = px
    return pose, vel, acc


def features(t, variant):
    import parity_adapter as PA
    v = np.hypot(t["vel"][:, 3, 0], t["vel"][:, 3, 1])
    pose, vel, acc = edit(t["pose"], t["vel"], t["acc"], v, variant)
    return PA.ego_features(pose, vel, acc, t["cmd"][:, -1])        # cmd of t0 (n, 4); `present` = 1


def cmd_select(a):
    import pandas as pd
    t = tab()
    names = t["names"].tolist()
    pq = pd.read_parquet(data_dir() / "runs/op_probe/joint/navtest_tokens.parquet").set_index("token").loc[names]
    d = pd.DataFrame({"token": names, "log": t["log"], "ts": pq.ts.values.astype(np.int64), "v0": np.hypot(t["vel"][:, 3, 0], t["vel"][:, 3, 1]),
                      "yaw4": np.degrees(t["fut"][:, 7, 2]), "hist_yaw": np.degrees(t["pose"][:, 0, 2]),
                      "dv_hist": np.hypot(*t["vel"][:, 0].T) - np.hypot(*t["vel"][:, 3].T),
                      "cmd": pq.cmd.values})
    d["row"] = np.arange(len(d))
    d = d.sort_values(["log", "ts"]).reset_index(drop=True)
    d["set"] = ""
    base = (d.v0 >= 3) & (d.hist_yaw.abs() < 10)
    turn = d.cmd.isin(["left", "right"])
    d.loc[base & turn & (d.yaw4.abs() > 45), "set"] = "T"
    # P: straight-command tokens 0.5-3.0 s before a straight -> turn switch whose command run contains a token with |yaw4| > 45
    prev = d.groupby("log").cmd.shift(1)
    dtp = d.groupby("log").ts.diff() / 1e6
    sw = np.flatnonzero(turn.values & (prev.values == "straight") & (dtp.values < 0.75))
    for i in sw:
        j = i
        while j + 1 < len(d) and d.log[j + 1] == d.log[i] and d.cmd[j + 1] == d.cmd[i] and (d.ts[j + 1] - d.ts[j]) / 1e6 < 0.75:
            j += 1
        if (d.yaw4[i:j + 1].abs() > 45).any():
            for m in range(max(0, i - 6), i - 0):
                if d.log[m] == d.log[i] and d.cmd[m] == "straight" and 0.45 < (d.ts[i] - d.ts[m]) / 1e6 < 3.05 and base[m] and d["set"][m] == "":
                    d.loc[m, "set"] = "P"
    # controls: straight command, |yaw4| < 5, matched without replacement on (v0, dv_hist), other log, T first, then P
    pool = d[base & (d.cmd == "straight") & (d.yaw4.abs() < 5) & (d["set"] == "")]
    used = set()
    pairs = []
    for S in ["T", "P"]:
        for i in d.index[d["set"] == S]:
            cand = pool[(pool.log != d.log[i]) & ~pool.index.isin(used)]
            dist = np.hypot((cand.v0 - d.v0[i]) / 1.0, (cand.dv_hist - d.dv_hist[i]) / 0.5)
            if len(cand) and dist.min() < 2.0:
                j = dist.idxmin()
                used.add(j)
                pairs.append((i, j))
    d["match"] = ""
    for i, j in pairs:
        d.loc[j, "set"] = "C" + d["set"][i]
        d.loc[j, "match"] = d.token[i]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "sets.csv", index=False)
    d[d["set"] != ""].to_csv(RES / "ehp_sets.csv", index=False)
    print(d["set"].value_counts().to_string())
    for s in ["T", "P", "CT", "CP"]:
        x = d[d["set"] == s]
        print(s, len(x), "logs", x.log.nunique(), "v0 med %.2f" % x.v0.median(), "dv_hist med %.2f" % x.dv_hist.median(), "yaw4 med %.1f" % x.yaw4.abs().median())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in [("select", cmd_select)]:
        sp.add_parser(n).set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
