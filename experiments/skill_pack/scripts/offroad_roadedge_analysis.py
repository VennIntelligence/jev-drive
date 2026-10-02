"""Q3 tables: openpilot's predicted road edges vs the map's drivable-area boundary, and the plan vs its own edges.

navsim2 env, CPU. Inputs: roadedge.pkl (offroad_roadedge.py), dep_corner.pkl (offroad_dep_corner.py), features.pkl and
stage2_table.pkl (offroad_analysis.py). Definitions are those of the plan file (Q3), with one refinement written down in the
report: the edge error is measured at the departing corner (where the real boundary is known exactly), not on 5 m cross-sections.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402

HW = 1.15           # half width of the ego car used by the footprint test (nuPlan Pacifica 2.297 m)
TOL = 1.0           # edge counts as right if within TOL of the real boundary at the departing corner


def edge_at(R, x):
    """Model edges [left, right] at ego-frame x (None outside the grid)."""
    ex = R["ex"]
    if x < ex[0] or x > ex[-1]:
        return None
    return np.array([np.interp(x, ex, R["ey"][0]), np.interp(x, ex, R["ey"][1])])


def cross_flags(R, path, hw):
    """path (n, 2) ego-frame points; which sides the car crosses with half width hw at 2 < x < 30."""
    left = right = False
    for x, y in path:
        if not (2.0 <= x <= 30.0):
            continue
        e = edge_at(R, x)
        if e is None:
            continue
        left |= bool(y + hw > e[0])
        right |= bool(y - hw < e[1])
    return left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/skill_pack/results/navhard-offroad/tables"))
    a = ap.parse_args()
    out = Path(a.out)
    RE = pickle.load(open(L.OUT / "roadedge.pkl", "rb"))
    F = pickle.load(open(L.OUT / "features.pkl", "rb"))
    DC = pickle.load(open(L.OUT / "dep_corner.pkl", "rb"))
    st2 = pd.read_pickle(L.OUT / "stage2_table.pkl").set_index("token")
    rows = []
    for t, R in RE.items():
        r = st2.loc[t]
        xs, mp = R["xs"], R["map"]
        row = dict(token=t, native_dac=int(r.native_dac), n4_dac=int(r.n4_dac), cmd=r.cmd, map=r["map"])
        # grid-level edge error where the centre line is inside the drivable area (context: error on passes and failures alike)
        eL = np.array([np.interp(x, R["ex"], R["ey"][0]) for x in xs]) - np.where(mp[:, 2] > 0, mp[:, 0], np.nan)
        eR = np.array([np.interp(x, R["ex"], R["ey"][1]) for x in xs]) - np.where(mp[:, 2] > 0, mp[:, 1], np.nan)
        row.update(eL_med_abs=float(np.nanmedian(np.abs(eL))) if np.isfinite(eL).any() else np.nan,
                   eR_med_abs=float(np.nanmedian(np.abs(eR))) if np.isfinite(eR).any() else np.nan,
                   eL_med=float(np.nanmedian(eL)) if np.isfinite(eL).any() else np.nan,
                   eR_med=float(np.nanmedian(eR)) if np.isfinite(eR).any() else np.nan)
        paths = {"native": R["plan"][:21, :2], "n4": F[t]["feat"]["n4"]["raw_dense"][:, :2]}
        for nm, path in paths.items():
            for hw, tag in ((0.0, "c"), (HW, "f")):
                row[f"{nm}_cross_{tag}_L"], row[f"{nm}_cross_{tag}_R"] = cross_flags(R, path, hw)
        for m in ("native", "n4"):
            dc = DC.get(t, {}).get(m)
            row[f"{m}_dep"] = dc is not None
            if dc is None:
                continue
            xc, yc = dc["corner"]
            side = "L" if dc["side"] > 0 else "R"
            k = 0 if side == "L" else 1
            e = edge_at(R, float(max(xc, R["ex"][0] + 1e-3)))
            row[f"{m}_xd"], row[f"{m}_yd"], row[f"{m}_side"] = float(xc), float(yc), side
            # signed: > 0 means the model's edge lies beyond the real boundary (it believes there is more road than there is)
            row[f"{m}_edge_err"] = float((1 if side == "L" else -1) * (e[k] - yc)) if e is not None else np.nan
            row[f"{m}_dep_side_cross_f"] = bool(row[f"{m}_cross_f_{side}"])
            row[f"{m}_dep_side_cross_c"] = bool(row[f"{m}_cross_c_{side}"])
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_pickle(L.OUT / "roadedge_table.pkl")
    res = []
    for m in ("native", "n4"):
        fail = D[(D[f"{m}_dac"] == 0) & D[f"{m}_dep"].fillna(False).astype(bool)].copy()

        def cat(r):
            e = r[f"{m}_edge_err"]
            if not np.isfinite(e):
                return "undefined"
            kind = "right" if abs(e) <= TOL else "too_far" if e > 0 else "too_near"
            return f"{kind}_{'plan_crosses' if r[f'{m}_dep_side_cross_f'] else 'plan_inside'}"
        fail["cat"] = fail.apply(cat, axis=1)
        n = len(fail)
        c = fail.cat.value_counts()
        for k in ("right_plan_crosses", "right_plan_inside", "too_far_plan_crosses", "too_far_plan_inside", "too_near_plan_crosses",
                  "too_near_plan_inside", "undefined"):
            res.append(dict(model=m, category=k, n=int(c.get(k, 0)), share=float(c.get(k, 0) / max(1, n))))
        right = int(c.get("right_plan_crosses", 0) + c.get("right_plan_inside", 0))
        wrong = int(sum(v for k, v in c.items() if k.startswith("too_")))
        res.append(dict(model=m, category="TOTAL edge right (|err| <= 1 m)", n=right, share=right / max(1, n)))
        res.append(dict(model=m, category="TOTAL edge wrong (|err| > 1 m)", n=wrong, share=wrong / max(1, n)))
        res.append(dict(model=m, category="all failures with a departure", n=n, share=1.0))
        res.append(dict(model=m, category="median signed edge error at departure (m)", n=n, share=float(fail[f"{m}_edge_err"].median())))
        res.append(dict(model=m, category="median |edge error| at departure (m)", n=n, share=float(fail[f"{m}_edge_err"].abs().median())))
        fail.to_pickle(L.OUT / f"roadedge_fail_{m}.pkl")
        # controls: the plan-vs-own-edge test on passes; edge error on passes / failures on the 5 m grid
        nm = m
        ok = D[D[f"{m}_dac"] == 1]
        cross_any = lambda g: g[[f"{nm}_cross_f_L", f"{nm}_cross_f_R"]].any(axis=1)  # noqa: E731
        res.append(dict(model=m, category="plan crosses own edge, footprint test, any side: failures (share)", n=n, share=float(cross_any(fail).mean())))
        res.append(dict(model=m, category="plan crosses own edge, footprint test, any side: passes (share)", n=len(ok), share=float(cross_any(ok).mean())))
        res.append(dict(model=m, category="plan crosses own edge, centre test, departure side: failures (share)", n=n, share=float(fail[f"{m}_dep_side_cross_c"].mean())))
    R_ = pd.DataFrame(res)
    R_.to_csv(out / "q3_road_edge_split.csv", index=False)
    print(R_.round(3).to_string(index=False))
    q = []
    for nm_, g in (("pass", D[D.native_dac == 1]), ("fail", D[D.native_dac == 0])):
        for side in ("eL_med_abs", "eR_med_abs", "eL_med", "eR_med"):
            v = g[side].dropna()
            q.append(dict(set=nm_, stat=side, n=len(v), q25=float(v.quantile(.25)), median=float(v.median()), q75=float(v.quantile(.75)), q90=float(v.quantile(.9)),
                          within_1m=float((v.abs() <= 1).mean())))
    pd.DataFrame(q).to_csv(out / "q3_edge_error_grid.csv", index=False)
    print(pd.DataFrame(q).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
