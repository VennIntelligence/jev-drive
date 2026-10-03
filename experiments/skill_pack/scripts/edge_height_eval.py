"""Lane EDGE addendum 2 readout: metric scale of openpilot's outputs under the virtual camera heights of edge_height.py,
against the shipped GIMM run on the same tokens. navsim2 env, CPU. Map columns come from sections_navtest.pkl
(edge_sections.py); model columns are recomputed from each plan file the same way.
Output: experiments/skill_pack/results/roadedge/height.json and tables/height_by_x.csv.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
RUN = L.D / "runs/skill_pack/edge_diag"
OUT = REPO / "experiments/skill_pack/results/roadedge"


def model_cols(f, keep=None):
    z = np.load(f)
    sl = json.loads(str(z["info"]))["heads_slices"]
    names = z["names"].tolist()
    rows = [i for i, t in enumerate(names) if keep is None or t in keep]
    H = z["heads"][rows]
    ll = H[:, sl["lane_lines"]:sl["lane_lines"] + 264].reshape(-1, 4, 33, 2)
    lp = 1 / (1 + np.exp(-H[:, sl["lane_lines_prob"]:sl["lane_lines_prob"] + 8][:, 1::2]))
    re = H[:, sl["road_edges"]:sl["road_edges"] + 132].reshape(-1, 2, 33, 2)
    pv = z["plan_vel"][rows][:, 0, 0]
    out = []
    for j, i in enumerate(rows):
        for x in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
            xm = x - 1.65                                 # ego x -> calib x (CAM_F0 ~1.65 m ahead of the rear axle)
            g = lambda a: float(np.interp(xm, X_IDXS, a))  # noqa: E731
            out.append(dict(token=names[i], x=x, ow=g(ll[j, 2, :, 0]) - g(ll[j, 1, :, 0]), p1=lp[j, 1], p2=lp[j, 2],
                            rw=g(re[j, 1, :, 0]) - g(re[j, 0, :, 0]), hz=(g(ll[j, 1, :, 1]) + g(ll[j, 2, :, 1])) / 2, pv0=pv[j]))
    return pd.DataFrame(out)


def main():
    S = pd.read_pickle(RUN / "sections_navtest.pkl")
    idx = {e["token"]: e for e in pickle.load(open(L.D / "runs/navsim_zs/index/navtest_slim.pkl", "rb"))}
    arms = {f.stem: f for f in sorted((RUN / "height").glob("hgt_r*.npz"))}
    toks = set(np.load(next(iter(arms.values())))["names"].tolist())
    arms = {"shipped_gimm": L.D / "runs/op_lb/lb_navtest/plans/gimm@cinque.npz", **arms}
    res, tabs = {}, []
    for name, f in arms.items():
        M = model_cols(f, toks).merge(S[["token", "x", "inter", "ego0L", "ego0R", "hum_y", "scL", "scR"]], on=["token", "x"])
        M["v"] = M.token.map(lambda t: float(np.linalg.norm(idx[t]["vel"][-1])))
        lv = (~M.inter) & M.ego0L.notna() & (M.p1 > .5) & (M.p2 > .5) & ~(M.hum_y.abs() > 1)
        ev = (~M.inter) & M.scL.notna()
        M["lane_ratio"] = M.ow / (M.ego0L - M.ego0R)
        M["road_ratio"] = M.rw / (M.scL - M.scR)
        sp = M.drop_duplicates("token")
        sp = sp[sp.v > 3]
        res[name] = dict(n_tokens=int(M.token.nunique()), n_lane_sections=int(lv.sum()), lane_ratio=float(M[lv].lane_ratio.median()),
                         road_ratio=float(M[ev].road_ratio.median()), speed_ratio=float((sp.pv0 / sp.v).median()),
                         h_model=float(M[lv].hz.median()), lane_prob_both=float(((M.p1 > .5) & (M.p2 > .5)).mean()))
        t = M[lv].groupby("x").lane_ratio.median().rename("lane_ratio").to_frame().join(M[ev].groupby("x").road_ratio.median())
        t["arm"] = name
        tabs.append(t.reset_index())
    # paired lane ratio on sections valid in both height arms
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pd.concat(tabs).round(4).to_csv(OUT / "tables/height_by_x.csv", index=False)
    (OUT / "height.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(pd.concat(tabs).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
