"""Lane EDGE tables (plan: experiments/skill_pack/plans/2026-10-04-roadedge-diagnosis-plan.md) from edge_sections.py output.

navsim2 env, CPU. Boards: navtest, navhard stage one (real), navhard stage two (synthetic). Definitions of the plan:
  lane-valid section: no INTERSECTION within |y| < 15, an ego lane (lane polygon containing y = 0), both ego lines prob > 0.5,
                      logged future |y| <= 1 m at x (or not reached);  edge-valid: no INTERSECTION, centre inside the scorer polygon.
  outward edge error: + = the model edge lies beyond the scorer boundary (left: reL - scL, right: scR - reR).
  pixel-faithful ("corrected"): model lateral times k = (CAM_F0 z + ZG) / (model lane-line z), i.e. the model's own image
                      reading put on the real ground plane (ZG = 0.35 m, the rear-axle height of the NAVSIM ego origin).
  ground offset: bottom z of vehicle boxes 4-25 m away in 40 random navtest logs (OpenScene annotations).
Outputs: experiments/skill_pack/results/roadedge/{scale.json, tables/scale_by_x.csv, tables/selection.csv}.
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

ZG = 0.35
RUN = L.D / "runs/skill_pack/edge_diag"
OUT = REPO / "experiments/skill_pack/results/roadedge"


def ground_offset():
    ps = sorted(glob.glob(str(L.D / "datasets/navsim/navsim_logs/test/*.pkl")))
    z = []
    for p in np.random.default_rng(0).choice(ps, 40, replace=False):
        for f in pickle.load(open(p, "rb"))[::10]:
            b, n = f["anns"]["gt_boxes"], np.asarray(f["anns"]["gt_names"])
            if b is None or not len(b):
                continue
            r = np.hypot(b[:, 0], b[:, 1])
            m = (n == "vehicle") & (r > 4) & (r < 25)
            z += list(b[m, 2] - b[m, 5] / 2)
    return dict(n_boxes=len(z), bottom_z_q10_25_50_75_90=np.percentile(z, [10, 25, 50, 75, 90]).round(3).tolist())


def main():
    D = []
    for b in ("navtest", "navhard"):
        d = pd.read_pickle(RUN / f"sections_{b}.pkl")
        d["board"] = "navtest" if b == "navtest" else "navhard_" + d.stage
        D.append(d)
    d = pd.concat(D, ignore_index=True)
    cz = {}
    for f in ("navtest_slim", "navhard_two_stage_slim"):
        for e in pickle.load(open(L.D / f"runs/navsim_zs/index/{f}.pkl", "rb")):
            cz[e["token"]] = e["cams"][-1]["CAM_F0"]["t"][2]
    d["k"] = (d.token.map(cz) + ZG) / (d.token.map(cz) - (d.z1 + d.z2) / 2)
    lv = (~d.inter) & d.ego0L.notna() & (d.p1 > .5) & (d.p2 > .5) & ~(d.hum_y.abs() > 1)
    ev = (~d.inter) & d.scL.notna()
    d["lane_ratio"] = (d.ll1 - d.ll2) / (d.ego0L - d.ego0R)
    d["road_ratio"] = (d.reL - d.reR) / (d.scL - d.scR)
    d["oL"], d["oR"] = d.reL - d.scL, d.scR - d.reR
    d["oLc"], d["oRc"] = d.reL * d.k - d.scL, d.scR - d.reR * d.k
    d["beyond"] = (d.oL > 1) | (d.oR > 1)
    d["beyond_c"] = (d.oLc > 1) | (d.oRc > 1)
    d["gda_wider"] = ((d.extL - d.scL) > .5) | ((d.scR - d.extR) > .5)
    d["sc_is_lanes"] = ((d.scL - d.laneL).abs() < .3) & ((d.scR - d.laneR).abs() < .3)
    rows = []
    for (b, x), g in d.groupby(["board", "x"]):
        gl, ge = g[lv.loc[g.index]], g[ev.loc[g.index]]
        rows.append(dict(board=b, x=x, n_lane=len(gl), map_lane_w=gl.ego0L.sub(gl.ego0R).median(), model_lane_w=gl.ll1.sub(gl.ll2).median(),
                         lane_ratio=gl.lane_ratio.median(), lane_ratio_c=(gl.lane_ratio * gl.k).median(),
                         lane_err_L=(gl.ll1 - gl.ego0L).median(), lane_err_R=(gl.ego0R - gl.ll2).median(),
                         centre_off=((gl.ll1 + gl.ll2) / 2 - (gl.ego0L + gl.ego0R) / 2).median(),
                         h_model=(g.token.map(cz) - (g.z1 + g.z2) / 2)[lv.loc[g.index]].median(),
                         n_edge=len(ge), road_ratio=ge.road_ratio.median(), road_ratio_c=(ge.road_ratio * ge.k).median(),
                         out_L=ge.oL.median(), out_R=ge.oR.median(), out_L_c=ge.oLc.median(), out_R_c=ge.oRc.median(),
                         beyond=ge.beyond.mean(), beyond_c=ge.beyond_c.mean(), gda_wider=ge.gda_wider.mean(), scorer_is_lane_set=ge.sc_is_lanes.mean()))
    T = pd.DataFrame(rows)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    T.round(4).to_csv(OUT / "tables/scale_by_x.csv", index=False)

    # DAC outcome (native, official CSVs) vs "edge > 1 m beyond" (H5) and spatial over-turn
    ld = lambda n: pd.read_csv(sorted(glob.glob(str(L.D / "runs/navsim/eval" / n / "*/*.csv")))[-1]).set_index("token")  # noqa: E731
    nt, nh = ld("v1_navtest_opi_lb_navtest_gimm-cinque__base"), ld(L.NATIVE_CSV)
    dac = pd.concat([nt.drivable_area_compliance, nh.drivable_area_compliance_stage_one.fillna(nh.drivable_area_compliance_stage_two)])
    d["dac"] = d.token.map(dac[~dac.index.duplicated()])
    sel = d[ev].groupby(["board", "dac"]).agg(section_beyond=("beyond", "mean"), n_sections=("beyond", "size")).reset_index()
    tk = d[ev].groupby(["board", "token"]).agg(beyond=("beyond", "any"), dac=("dac", "first")).groupby(["board", "dac"]).beyond.agg(["mean", "size"])
    sel = sel.merge(tk.rename(columns={"mean": "token_any_beyond", "size": "n_tokens"}).reset_index(), on=["board", "dac"])
    sel.round(4).to_csv(OUT / "tables/selection.csv", index=False)

    m = d.hum_y.abs().gt(1) & d.plan_y.notna() & (d.board == "navtest")
    spatial = {f"x{x:g}": dict(n=int((m & (d.x == x)).sum()), slope=float((d.plan_y * d.hum_y)[m & (d.x == x)].sum() / (d.hum_y ** 2)[m & (d.x == x)].sum()))
               for x in (5.0, 10.0, 15.0, 20.0)}
    sp = {}
    for b, pf, ix in (("navtest", "lb_navtest", "navtest_slim"), ("navhard", "lb_navhard", "navhard_two_stage_slim")):
        z = np.load(L.D / f"runs/op_lb/{pf}/plans/gimm@cinque.npz")
        v = {e["token"]: (float(np.linalg.norm(e["vel"][-1])), e["stage"]) for e in pickle.load(open(L.D / f"runs/navsim_zs/index/{ix}.pkl", "rb"))}
        vv = np.array([v[t][0] for t in z["names"]])
        st = np.array([v[t][1] for t in z["names"]])
        for s in sorted(set(st)):
            k = (vv > 3) & (st == s)
            sp[f"{b}_{s}"] = dict(n=int(k.sum()), plan_v0_over_logged=float(np.median(z["plan_vel"][k, 0, 0] / vv[k])))
    res = dict(ground=ground_offset(), plan_speed=sp, spatial_lateral_slope_navtest=spatial, zg=ZG)
    (OUT / "scale.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(T.round(3).to_string(index=False))
    print(sel.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
