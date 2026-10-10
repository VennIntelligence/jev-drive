"""FOV check report (experiments/corridor): tables of fov_build.py's output -> experiments/corridor/results/fov.md.
  .venv/bin/python experiments/corridor/scripts/fov_report.py
Unit = token x member (SH30-F-s0, SH30-F-s1, WA-JEPA); failure = DAC < 1 on navtest tokens with a logged 4 s heading change > 45 deg.
95% CIs are log-cluster bootstraps (B 10 000; decision 207's CB). Privileged inputs (map, logged future, departure points): analysis only.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts"), str(REPO / "research")]
import pt_swap as PS  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
FOV = D / "runs/corridor/fov"
PT = D / "runs/op_parity/pt_swap"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
LOGS = D / "datasets/navsim/navsim_logs/test"
OUTMD = Path(os.environ.get("FOV_MD", REPO / "experiments/corridor/results/fov.md"))
HALF = np.degrees(np.arctan(256 / 455))        # wide frame half angle, deg
F = lambda t: PS.pc(t, "{:.1f}", 100)          # noqa: E731  percent with CI


def rig_lines():
    fr = pickle.load(open(sorted(LOGS.glob("*.pkl"))[0], "rb"))[5]
    L = []
    for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
        v = fr["cams"][c]
        R, K, t = np.asarray(v["sensor2lidar_rotation"]), np.asarray(v["cam_intrinsic"]), np.asarray(v["sensor2lidar_translation"])
        yaw = np.degrees(np.arctan2(R[1, 2], R[0, 2]))
        hf = np.degrees(2 * np.arctan(960 / K[0, 0]))
        L.append(f"| {c} | {yaw:+.1f} | {hf:.1f} | {np.degrees(2 * np.arctan(540 / K[1, 1])):.1f} | {t[0]:.2f}, {t[1]:+.2f}, {t[2]:.2f} |")
    return L


def main():
    tok = pd.read_parquet(FOV / "tok.parquet")
    un = pd.read_parquet(FOV / "unit.parquet")
    sc = pd.read_csv(PT / "score_all.csv", usecols=["key", "token", "drivable_area_compliance"])
    sc = sc[sc.key.isin(["sh0_pp", "sh1_pp", "wa_pp"]) & sc.token.isin(set(tok.token))]
    sc = sc[np.isfinite(sc.drivable_area_compliance)]
    un = un.assign(unit=un.unit.str[:-3])
    M = sc.rename(columns={"key": "unit"}).assign(unit=lambda d: d.unit.str[:-3])
    M["fail"] = M.drivable_area_compliance < 1
    M = M.merge(tok, on="token").merge(un.drop(columns=["log", "sgn", "dpsi"]), on=["token", "unit"], how="left")
    # plan under-turn flag as decision 240 / corr_report: gain < 0.9
    Z = np.load(PT / "poses.npz")
    pos = {t: i for i, t in enumerate(Z["tokens"].astype(str))}
    tab = np.load(TAB)
    fut = tab["fut"].astype(float)
    tpos = {t: i for i, t in enumerate(tab["names"].astype(str))}
    gain = []
    for u, tk in zip(M.unit, M.token):
        P = np.r_[0.0, Z[f"{u}_pp"][pos[tk]][:, 2]]
        f = fut[tpos[tk], -1, 2]
        gain.append(np.unwrap(P)[-1] / f if abs(f) > 0.05 else np.nan)
    M["gain"] = gain
    inside = M.fail & (M.lqr_side == M.sgn)
    M["cls"] = np.where(~M.fail, "pass", np.where(inside, "inside", np.where(M.gain < 0.9, "cannot", "other")))
    sh = M.unit.isin(["sh0", "sh1"])
    wa_t = M[M.unit == "wa"].set_index("token").fail
    M["wa_fail"] = M.token.map(wa_t)
    cb = PS.CB(M.log.to_numpy())
    L = []
    P = L.append
    logs_n = M.log.nunique()
    sets = {
        "SH30 pass": sh & ~M.fail, "SH30 DAC fail": sh & M.fail, "- cut-inside": sh & (M.cls == "inside"),
        "- cannot-make-turn": sh & (M.cls == "cannot"), "- other fail": sh & (M.cls == "other"),
        "SH30 fail, WA-JEPA pass": sh & M.fail & (M.wa_fail == False), "SH30 fail, WA-JEPA fail": sh & M.fail & (M.wa_fail == True),  # noqa: E712
        "WA-JEPA pass": (M.unit == "wa") & ~M.fail, "WA-JEPA DAC fail": (M.unit == "wa") & M.fail}

    def mean(col, m, scale=True):
        x = M[col].to_numpy(float)
        t = cb.mean(x, m.to_numpy() & np.isfinite(x))
        return F(t) if scale else PS.pc(t)

    def mean_c(col, m, inv=False):
        x = 1 - M[col].to_numpy(float) if inv else M[col].to_numpy(float)
        return F(cb.mean(x, m.to_numpy() & np.isfinite(x)))

    def nfin(col, m):
        return int((m & M[col].notna()).sum())

    P("# FOV check: was the road the driver needed inside the model's view at decision time?\n")
    P("Written 2026-10-10 (corridor lane). Measurement only: stored SH30 / WA-JEPA plans and DAC scores (decision 207's `score_all.csv`), the logged "
      "future, the nuPlan drivable area of the v2 metric cache; CPU. Code `scripts/fov_build.py` (geometry, departure replay) and `scripts/fov_report.py`. "
      "Raw rows: `$DATA_DIR/runs/corridor/fov/{tok,unit}.parquet`. 95% CIs are log-cluster bootstraps (B 10 000). No decision entry.\n")
    P("The map, the logged future and the replayed departure points are privileged: this is an analysis of where the road lies relative to the "
      "camera, not a method input. **Occlusion by buildings, vehicles or hedges is not measurable from the map**; every 'visible' below means "
      "'inside the frame geometrically', an upper bound on what the model can actually see.\n")
    P("## Views used (from code)\n")
    P("SH30 is a front-camera model (P2 family: Cinque + ego / pose / command; the side / rear camera arm P3 was tested and added nothing, "
      "navtest EPDMS 86.10 vs 86.57, `experiments/op_parity` pilot). Its image input is openpilot's two frames rendered from NAVSIM `CAM_F0` "
      "(`jevdrive/navsim_zs.py` `OpenpilotMaps`, `jevdrive/openpilot/frames.py`), optical axis along the ego x axis, 512 x 256:\n")
    P("| frame | focal px | horizontal FOV | vertical FOV | horizon row | ground nearer than this is below the frame (camera height 1.87 m) |\n|:--|--:|--:|--:|--:|--:|")
    P(f"| wide | 455 | {2 * np.degrees(np.arctan(256 / 455)):.1f} deg (+-{HALF:.1f}) | {np.degrees(np.arctan(151.8 / 455) + np.arctan(104.2 / 455)):.1f} deg "
      f"({np.degrees(np.arctan(151.8 / 455)):.1f} up / {np.degrees(np.arctan(104.2 / 455)):.1f} down) | 151.8 | {1.87 / (104.2 / 455):.1f} m ahead |")
    P(f"| road (narrow) | 910 | {2 * np.degrees(np.arctan(256 / 910)):.1f} deg | {np.degrees(np.arctan(47.6 / 910) + np.arctan(208.4 / 910)):.1f} deg | 47.6 | {1.87 / (208.4 / 910):.1f} m ahead |")
    P("\nThe wide frame contains the road frame horizontally, so the model's horizontal field is **58.7 deg, +-29.4 deg about the ego heading**. "
      "The source camera bounds it: the native CAM_F0 image is 1920 x 1080; the sampling coverage of both frames is 100% on the checked calibration. "
      "Native cameras of the log (heading in the ego frame, left +; WA-JEPA reads L0 / F0 / R0 / B0):\n")
    P("| camera | mount yaw deg | native HFOV deg | native VFOV deg | position x, y, z (m, ego) |\n|:--|--:|--:|--:|:--|")
    L.extend(rig_lines())
    P("\nSo WA-JEPA's horizontal union is about 4 x 64 deg minus overlaps (nearly all-round); ours is 58.7 deg. The frame is built from one "
      "camera; the three-front-camera wide input the rig note asked for was never used by SH30. Whether the model uses its wide frame at all "
      "is a separate question (`experiments/op_fov`: widening it did not help).\n")
    P("Definitions. Positions are in the t0 rear-axle frame; the camera sits at the CAM_F0 mount (x +1.67 m); bearing = angle of the point seen "
      "from the camera against the ego heading, left +. 'in FOV' = |bearing| <= 29.4 deg and ahead; 'in image' also needs the pinhole row inside "
      "the 256-row frame for a ground point (z = -0.35 m). 'History' = the 4 keyframes the model reads (t0 - 1.5, -1.0, -0.5, 0 s), the point "
      "being in the frame of at least one of them. Logged path = the 4 s logged future, sampled every 0.25 m. Edges = nearest drivable-area "
      "boundary on each side of the logged path (lateral search 12 m, every 1 m; inside = the turn side); a hit is absent where the road is "
      "wider than 12 m (intersections). 'Departure' = first footprint corner outside the scorer's drivable polygons along the LQR replay "
      "(decision 153's replay, the same replay that scores DAC). 'Needed road point' = the logged-path point at the plan's own arc length up to "
      "the departure. 'Relevant edge' = the drivable-area boundary within 10 m of the departure point.\n")
    P(f"Population: navtest tokens with |logged heading change at 4 s| > 45 deg: {len(tok)} tokens in {tok.log.nunique()} logs; "
      f"{int(sets['SH30 DAC fail'].sum())} SH30 failing token-seeds (of {int(sh.sum())}), {int(sets['WA-JEPA DAC fail'].sum())} WA-JEPA failing tokens "
      f"(of {int((M.unit == 'wa').sum())}; 119 navtest tokens have no WA-JEPA plan). Failure classes are decision 240's: cut-inside = the first corner "
      "outside is on the turn side; cannot-make-turn = otherwise, plan heading gain < 0.9.\n")

    # ---------------- 1 bearings
    P("## 1. Bearing of the road from the t0 camera\n")
    P("Median bearing (deg, signed so that the turn direction is positive) of the logged path at 1 - 4 s and at fixed arc lengths, and the share "
      "of path points beyond the wide frame's +-29.4 deg. Points nearer than 8 m ahead of the camera are below the frame and are listed in the "
      "1 s column only as a bearing.\n")
    rows = []
    for nm, m in sets.items():
        r = {"set": nm, "n": int(m.sum())}
        for k in ("1s", "2s", "3s", "4s", "s10", "s20", "s30"):
            col = f"brg_path_{k}"
            x = M[col] * M.sgn
            r[f"med {k}"] = float(np.nanmedian(x[m])) if nfin(col, m) else np.nan
        for k in ("4s", "s10", "s20", "s30"):
            col = f"brg_path_{k}"
            M["_o"] = (M[col].abs() > HALF).astype(float).where(M[col].notna())
            r[f"outside FOV {k} %"] = mean("_o", m)
        rows.append(r)
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("`med`: median of sgn * bearing, deg. `outside FOV`: share of members whose path point (at that time / arc length) has |bearing| > 29.4 deg, "
      "among members whose path reaches it (arc-length columns drop slow tokens; 30 m is reached by few).\n")
    rows = []
    for nm, m in sets.items():
        r = {"set": nm}
        for k in ("s10", "s20", "s30", "s40"):
            col = f"brg_cl_{k}"
            x = M[col] * M.sgn
            r[f"centreline med {k}"] = float(np.nanmedian(x[m])) if nfin(col, m) else np.nan
        for side in ("in", "out"):
            for k in ("s10", "s20"):
                col = f"brg_edge_{side}_{k}"
                x = M[col] * M.sgn
                r[f"{side}side edge med {k}"] = float(np.nanmedian(x[m])) if nfin(col, m) else np.nan
                r[f"n {side} {k}"] = nfin(col, m)
        rows.append(r)
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("Exit-lane centreline arc length counts from abeam of the ego along the driven lane sequence (privileged map; matched tokens only); edge "
      "columns are the hit nearest to the path point at that arc length, `n` = members with a hit.\n")

    # ---------------- 2 fractions
    P("## 2. Share of the road inside the model's view at t0, and over the history keyframes\n")
    P("Mean over members of the share of the object that is inside the wide frame (SH30) at t0 / in at least one of the 4 keyframes. "
      "`FOV` = horizontal test, `image` = also inside the 256 rows.\n")
    rows = []
    for nm, m in sets.items():
        r = {"set": nm, "n": int(m.sum())}
        for obj, pre in (("logged path 4 s", "path"), ("centreline 40 m", "cl"), ("inside edge", "edge_in"), ("outside edge", "edge_out")):
            for kind in ("fov", "img"):
                for when in ("t0", "any"):
                    r[f"{obj} {kind} {when}"] = mean(f"{pre}_w_{kind}_{when}", m)
        rows.append(r)
    df = pd.DataFrame(rows)
    for obj in ("logged path 4 s", "centreline 40 m", "inside edge", "outside edge"):
        P(f"**{obj}** (% of arc length / edge samples)\n")
        P(PS.md(df[["set", "n"] + [c for c in df.columns if c.startswith(obj + " ")]].rename(columns=lambda c: c.replace(obj + " ", "")), 1) + "\n")
    P("All fractions in this section are over points at least 8.2 m ahead of the camera: ground nearer than that falls below the bottom row of the "
      "wide frame at any heading (near-field blind spot, not a side-view question). The share of the 4 s path that is nearer than that:\n")
    P(PS.md(pd.DataFrame([{"set": nm, "path arc nearer than 8.2 m (mean share) %": mean_c("path_far_frac", m, True)} for nm, m in sets.items()]), 1) + "\n")
    # share of members with the whole path in FOV at t0
    rows = []
    for nm, m in sets.items():
        M["_all"] = (M.path_w_fov_t0 >= 0.999).astype(float)
        M["_half"] = (M.path_w_fov_t0 >= 0.5).astype(float)
        M["_allh"] = (M.path_w_fov_any >= 0.999).astype(float)
        rows.append({"set": nm, "whole 4 s path in FOV at t0 %": mean("_all", m), "at least half %": mean("_half", m),
                     "whole path in FOV over history %": mean("_allh", m)})
    P(PS.md(pd.DataFrame(rows), 1) + "\n")

    # ---------------- 3 departures
    P("## 3. Where the plan leaves the drivable area, against the view\n")
    P("Failures only; the replayed departure point (first footprint corner outside along the LQR replay). Members whose replay shows no departure "
      "(DAC < 1 from another cause) are excluded: " + ", ".join(f"{k} {int(((M.unit == k) & M.fail & ~(M.lqr_out == True)).sum())}" for k in ("sh0", "sh1", "wa")) + ".\n")  # noqa: E712
    rows = []
    fsets = {k: v & (M.lqr_out == True) for k, v in sets.items() if "fail" in k.lower() or k.startswith("-")}  # noqa: E712
    for nm, m in fsets.items():
        r = {"set": nm, "n": int(m.sum())}
        for col, lab in (("lqr_pt_w_fov_t0", "departure in FOV t0"), ("lqr_pt_w_img_t0", "departure in image t0"), ("lqr_pt_w_fov_any", "departure in FOV any history"),
                         ("lqr_pt_w_img_any", "departure in image any history"), ("lqr_need_w_fov_t0", "needed-road point in FOV t0"),
                         ("lqr_need_w_img_t0", "needed-road point in image t0"), ("lqr_need_w_fov_any", "needed-road point in FOV any history")):
            M["_v"] = M[col].astype(float)
            r[lab + " %"] = mean("_v", m)
        r["relevant edge in FOV t0 (mean share) %"] = mean("lqr_edge_w_fov_t0", m)
        r["relevant edge in FOV any history %"] = mean("lqr_edge_w_fov_any", m)
        r["median departure bearing (turn side +) deg"] = float(np.nanmedian((M.lqr_brg * M.sgn)[m])) if m.any() else np.nan
        r["median departure range m"] = float(np.nanmedian(np.hypot(M.lqr_x[m] - 1.67, M.lqr_y[m] + 0.026))) if m.any() else np.nan
        rows.append(r)
    d3 = pd.DataFrame(rows)
    P(PS.md(d3[["set", "n"] + [c for c in d3.columns if c.startswith("departure")]], 1) + "\n")
    P(PS.md(d3[["set", "n"] + [c for c in d3.columns if c.startswith(("needed", "relevant", "median"))]], 1) + "\n")
    P("The 'in FOV' columns are the headline: the share of failures whose departure point (or needed-road point) lies inside the horizontal field at "
      "t0 / in some keyframe. Its complement is the share out of view. WA-JEPA's row evaluates WA-JEPA's own departure points against SH30's "
      "geometry (our frame), not against its own cameras; see section 4.\n")
    # departure corner side
    P("Where the departure lies relative to the turn (share on the inside of the turn / forward range):\n")
    rows = []
    for nm, m in fsets.items():
        rows.append({"set": nm, "n": int(m.sum()), "departure on the turn side %": F(cb.mean((M.lqr_y * M.sgn > 0).astype(float).to_numpy(), m.to_numpy())),
                     "departure beyond 29.4 deg bearing %": F(cb.mean((M.lqr_brg.abs() > HALF).astype(float).to_numpy(), m.to_numpy()))})
    P(PS.md(pd.DataFrame(rows), 1) + "\n")

    # ---------------- 4 WA-JEPA and concentration
    P("## 4. Does SH30 fail where the road leaves its view and WA-JEPA does not?\n")
    P("Difference in the share of the 4 s logged path inside the wide frame at t0 (fail minus pass), SH30 and WA-JEPA, and in WA-JEPA's own four "
      "cameras. The path geometry is the same logged path for both models; a view-limited failure would show a more negative difference for "
      "SH30 than for WA-JEPA in the SH30 view and a flat one in WA-JEPA's view.\n")

    def diff(col, m1, m0):
        x = M[col].to_numpy(float)
        a1, a0 = m1.to_numpy() & np.isfinite(x), m0.to_numpy() & np.isfinite(x)
        s1, p1 = cb._s(x, a1)
        s0, p0 = cb._s(x, a0)
        n1, q1 = cb._s(np.ones_like(x), a1)
        n0, q0 = cb._s(np.ones_like(x), a0)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = s1 / n1 - s0 / n0
        lo, hi = np.nanquantile(b, [0.025, 0.975])
        return 100 * (p1 / q1 - p0 / q0), 100 * lo, 100 * hi

    rows = []
    for lab, mf, mp_ in (("SH30", sets["SH30 DAC fail"], sets["SH30 pass"]), ("WA-JEPA", sets["WA-JEPA DAC fail"], sets["WA-JEPA pass"])):
        for view, pre in (("SH30 wide frame", "w"), ("WA-JEPA 4 cameras", "a")):
            r = {"model": lab, "view": view, "n fail": int(mf.sum()), "n pass": int(mp_.sum())}
            for col, nm in (("path_%s_fov_t0", "path FOV t0"), ("path_%s_img_t0", "path image t0"), ("path_%s_fov_any", "path FOV any history")):
                r[nm + " fail - pass (pp)"] = PS.pc(diff(col % pre, mf, mp_), "{:+.1f}")
            rows.append(r)
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("Failure rate (DAC) by the largest path bearing within 4 s (turn side +), per model; columns are bearing bins in the SH30 view:\n")
    M["_maxb"] = (M[["brg_path_1s", "brg_path_2s", "brg_path_3s", "brg_path_4s"]].mul(M.sgn, axis=0)).max(axis=1)
    bins = [(-999, 20, "< 20"), (20, 29.4, "20 - 29.4"), (29.4, 45, "29.4 - 45"), (45, 999, "> 45")]
    rows = []
    for lab, mk in (("SH30", sh), ("WA-JEPA", M.unit == "wa")):
        r = {"model": lab}
        for lo, hi, nm in bins:
            m = mk & (M._maxb > lo) & (M._maxb <= hi)
            r[f"{nm} deg: n"] = int(m.sum())
            r[f"{nm} deg: DAC fail %"] = F(cb.mean(M.fail.astype(float).to_numpy(), m.to_numpy()))
        rows.append(r)
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("Same by the share of the logged path inside the SH30 frame at t0:\n")
    bins = [(-1, 0.5, "< 50%"), (0.5, 0.8, "50 - 80%"), (0.8, 0.999, "80 - 99.9%"), (0.999, 2, "100%")]
    rows = []
    for lab, mk in (("SH30", sh), ("WA-JEPA", M.unit == "wa")):
        r = {"model": lab}
        for lo, hi, nm in bins:
            m = mk & (M.path_w_fov_t0 > lo) & (M.path_w_fov_t0 <= hi)
            r[f"{nm}: n"] = int(m.sum())
            r[f"{nm}: DAC fail %"] = F(cb.mean(M.fail.astype(float).to_numpy(), m.to_numpy()))
        rows.append(r)
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("Failure rate of the part of the population whose logged path is mostly out of the SH30 frame (< 80% of its far path inside at t0) against the "
      "rest, per model, and the difference of the two differences (positive = SH30's failures lean more on the out-of-view part than WA-JEPA's, "
      "the view-limit signature). A joint log-cluster bootstrap.\n")
    x = M.fail.to_numpy(float)
    fin = np.isfinite(M.path_w_fov_t0.to_numpy(float))
    low = (M.path_w_fov_t0 < 0.8).to_numpy() & fin
    high = (M.path_w_fov_t0 >= 0.8).to_numpy() & fin

    def rate(mk, g):
        m = mk.to_numpy() & g
        f, pf = cb._s(x, m)
        n, pn = cb._s(np.ones_like(x), m)
        return f / n, pf / pn

    rows, dd = [], {}
    for lab, mk in (("SH30", sh), ("WA-JEPA", M.unit == "wa")):
        bl, pl = rate(mk, low)
        bh, ph = rate(mk, high)
        dd[lab] = (bl - bh, pl - ph)
        lo, hi = np.nanquantile(bl - bh, [0.025, 0.975])
        rows.append({"model": lab, "fail % < 80% in frame": f"{100 * pl:.1f}", "fail % >= 80% in frame": f"{100 * ph:.1f}",
                     "difference (pp)": PS.pc((100 * (pl - ph), 100 * lo, 100 * hi), "{:+.1f}")})
    b = dd["SH30"][0] - dd["WA-JEPA"][0]
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    rows.append({"model": "SH30 - WA-JEPA", "fail % < 80% in frame": "", "fail % >= 80% in frame": "",
                 "difference (pp)": PS.pc((100 * (dd["SH30"][1] - dd["WA-JEPA"][1]), 100 * lo, 100 * hi), "{:+.1f}")})
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("The in-frame share is not independent of the turn: tighter turns put more of the path beyond the frame and are harder for any model. The same "
      "comparison stratified by the logged 4 s heading change (45 - 70, 70 - 110, > 110 deg; strata weighted by their size), so the contrast is "
      "within similar turn angles:\n")
    strata = [(45, 70), (70, 110), (110, 400)]
    w = np.array([((M.dpsi > lo_) & (M.dpsi <= hi_)).to_numpy()[sh.to_numpy() & (low | high)].sum() for lo_, hi_ in strata], float)
    w /= w.sum()
    dd = {}
    rows = []
    for lab, mk in (("SH30", sh), ("WA-JEPA", M.unit == "wa")):
        tot = 0
        for k, (lo_, hi_) in enumerate(strata):
            sg = ((M.dpsi > lo_) & (M.dpsi <= hi_)).to_numpy()
            bl, pl = rate(mk, low & sg)
            bh, ph = rate(mk, high & sg)
            tot = tot + w[k] * np.array([bl - bh, pl - ph], dtype=object)
        dd[lab] = (tot[0].astype(float), float(tot[1]))
        lo, hi = np.nanquantile(dd[lab][0], [0.025, 0.975])
        rows.append({"model": lab, "stratified difference (pp)": PS.pc((100 * dd[lab][1], 100 * lo, 100 * hi), "{:+.1f}")})
    b = dd["SH30"][0] - dd["WA-JEPA"][0]
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    rows.append({"model": "SH30 - WA-JEPA", "stratified difference (pp)": PS.pc((100 * (dd["SH30"][1] - dd["WA-JEPA"][1]), 100 * lo, 100 * hi), "{:+.1f}")})
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    P("WA-JEPA failure departure points in WA-JEPA's own cameras (share inside the union of its four native cameras, pinhole test):\n")
    rows = []
    for nm in ("WA-JEPA DAC fail", "SH30 DAC fail"):
        m = sets[nm] & (M.lqr_out == True)  # noqa: E712
        r = {"set": nm, "n": int(m.sum())}
        for col, lab in (("lqr_pt_a_fov_t0", "departure in its 4 cameras (FOV) t0 %"), ("lqr_pt_a_img_t0", "departure in its 4 cameras (image) t0 %"),
                         ("lqr_need_a_img_t0", "needed-road point in its 4 cameras (image) t0 %")):
            M["_v"] = M[col].astype(float)
            r[lab] = mean("_v", m)
        rows.append(r)
    P(PS.md(pd.DataFrame(rows), 1) + "\n")
    OUTMD.parent.mkdir(parents=True, exist_ok=True)
    OUTMD.write_text("\n".join(L) + "\n")
    M.drop(columns=[c for c in M.columns if c.startswith("_")]).to_csv(FOV / "members.csv.gz", index=False)
    print(OUTMD, "members", len(M), "logs", logs_n)


if __name__ == "__main__":
    main()
