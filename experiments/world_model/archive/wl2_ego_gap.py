"""WL-2 ego lateral channel: CARLA route-centerline state vs what openpilot's own outputs say (existing logs only).

Inputs are op-arb / op-drive attempt dirs (plans.jsonl, ticks.jsonl, route.json; pulled to a local dir):
  truth  e_y, e_psi of the simulator rear-axle pose w.r.t. the route polyline (the WL-2 privileged source)
  op     plan-extrapolation estimate: the plan (t = 2 s and 5 s points, rear-axle frame) is taken as the lane-centre
         line ahead and extended back to x = 0 -> (e_y_hat, e_psi_hat). Lane-line / road-edge geometry is NOT in these
         logs (only the four lane probabilities), so this is the best openpilot-only estimate the logs allow.
Conventions: ego frame x forward, y left; e_y > 0 = ego left of the centre line; e_psi > 0 = ego yawed left of it.
Usage: wl2_ego_gap.py steps VAL_XML OUT.csv ARMS_DIR... | wl2_ego_gap.py report STEPS.csv OUTDIR
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

V_MIN = 3.0        # below this the plan is shorter than the lookahead and the estimator is undefined
NEAR_M = 30.0      # near a command: next non-lane-follow maneuver within this distance, or inside a junction
CENTRED = 0.30     # |e_y| below this counts as on the lane centre for the bias probes
LANE_LOW = 0.5     # min(inner-left, inner-right) lane probability below this = low confidence


def route_state(route_xy, pose):
    """Ego-frame route polyline, signed e_y, e_psi and the index of the nearest route point."""
    x, y, yaw = pose
    d = route_xy - np.array([x, y])
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.stack([c * d[:, 0] + s * d[:, 1], s * d[:, 0] - c * d[:, 1]], -1)   # CARLA world is left-handed
    i = int(np.argmin(np.hypot(*loc.T)))
    i = min(i, len(loc) - 2)
    a, b = loc[i], loc[i + 1]
    t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    e_y = float(-a @ np.array([-t[1], t[0]]))                                     # origin relative to a, left normal
    return loc, i, e_y, -math.atan2(t[1], t[0])


def y_at(pts, x):
    o = np.argsort(pts[:, 0])
    return float(np.interp(x, pts[o, 0], pts[o, 1], left=np.nan, right=np.nan))


def town_weather(val_xml):
    out = {}
    for r in ET.parse(val_xml).getroot().iter("route"):
        w = r.find("weathers/weather")
        g = lambda k: float(w.get(k)) if w is not None else -1.0  # noqa: E731
        out[r.get("id")] = dict(town=r.get("town"), night=g("sun_altitude_angle") <= -45, rain=g("precipitation") >= 30,
                                fog=g("fog_density") >= 50)
    return out


def run_steps(d, arm, meta):
    ticks = {}
    for line in open(d / "ticks.jsonl"):
        r = json.loads(line)
        if r.get("truth") and len(r["truth"]) == 3:
            ticks[r["frame"]] = r
    xy = np.array(json.load(open(d / "route.json"))["xy"], float)
    rows, prev = [], None
    for line in open(d / "plans.jsonl"):
        p = json.loads(line)
        tk = ticks.get(p["frame"])
        if tk is None or "op_xy" not in p:
            continue
        loc, i, e_y, e_psi = route_state(xy, tk["truth"])
        ctx = p.get("ctx") or {}
        cmd = p.get("cmd") or [-1, 1e9]
        v = p["v"]
        rec = dict(arm=arm, route=d.parent.name, frame=p["frame"], t=p["t"], warm=bool(p["warm"]), v=v, e_y=e_y,
                   e_psi=e_psi, junc=int(ctx.get("junc", 0)), cmd=cmd[0], cmd_d=cmd[1], lane_l=p["lane"][1],
                   lane_r=p["lane"][2], pose_v=p["pose_v"], acc_op=p["aplan"][0], lat=str(p.get("lat", "route")),
                   a_true=(v - prev[1]) / 0.05 if prev and p["frame"] - prev[0] == 1 else np.nan,
                   yr15=y_at(loc[i:i + 60], 15.0), yr30=y_at(loc[i:i + 90], 30.0))
        prev = (p["frame"], v)
        op = np.array([[0.0, 0.0]] + p["op_xy"], float)                         # x, y at t = 0, 1, 2, 3, 5 s
        if v >= V_MIN and op[-1, 0] > 20.0:
            (x2, y2), (x5, y5) = op[2], op[4]
            slope = (y5 - y2) / max(x5 - x2, 1e-6)
            rec.update(ehat_y=-(y2 - slope * x2), ehat_psi=-math.atan(slope), yp15=y_at(op, 15.0),
                       x1=op[1, 0], y1=op[1, 1], x2=x2, y2=y2, x3=op[3, 0], y3=op[3, 1], x5=x5, y5=y5, act_k=p["act_k"])
        rec.update(meta.get(d.parent.name, {}))
        rows.append(rec)
    return pd.DataFrame(rows)


def cmd_steps(val_xml, out, *roots):
    """roots: dirs that contain <arm>/attempts/<route>/<n>/{plans,ticks}.jsonl (a run dir's arms/); out '-' = stdout."""
    meta, frames = town_weather(val_xml), []
    for root in roots:
        label = Path(root).parent.name
        best = {}                                            # one attempt per (arm, route): the longest log
        for pl in Path(root).glob("*/attempts/*/*/plans.jsonl"):
            key = (pl.parents[3].name, pl.parents[1].name)
            if (pl.parent / "ticks.jsonl").exists() and pl.stat().st_size > best.get(key, (0, None))[0]:
                best[key] = (pl.stat().st_size, pl)
        for (arm, _), (_, pl) in sorted(best.items()):
            frames.append(run_steps(pl.parent, f"{label}/{arm}", meta))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if out == "report":                     # build and report in one go (run where the logs are); prints to stdout
        return report(df, None)
    df.to_csv(out, index=False)
    print(len(df), "steps,", df.route.nunique(), "routes,", df.arm.nunique(), "arms", file=sys.stderr)


def stats(x):
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return dict(n=len(x), med=np.median(x) if len(x) else np.nan, p95=np.quantile(x, .95) if len(x) else np.nan)


def learned_readout(d, folds=5):
    """Ridge readout of (e_y, e_psi) from openpilot's logged outputs only (plan points, desired curvature, speed, lane probs),
    route-grouped CV. What a learned deployable estimator can reach without lane-line geometry."""
    cols = ["x2", "y2", "y1", "y3", "y5", "x5", "act_k", "v", "lane_l", "lane_r"]
    g = d.dropna(subset=cols + ["e_y", "e_psi"]).copy()
    if len(g) < 200:
        return {}
    X = g[cols].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.c_[X, np.ones(len(X))]
    keys = (g.arm + "/" + g.route.astype(str)).to_numpy()
    uk = np.unique(keys)
    fold = {k: i % folds for i, k in enumerate(np.random.RandomState(0).permutation(uk))}
    f = np.array([fold[k] for k in keys])
    out = {}
    for tgt in ("e_y", "e_psi"):
        y = g[tgt].to_numpy(float)
        pred = np.full(len(y), np.nan)
        for k in range(folds):
            tr, te = f != k, f == k
            w = np.linalg.solve(X[tr].T @ X[tr] + 1.0 * np.eye(X.shape[1]), X[tr].T @ y[tr])
            pred[te] = X[te] @ w
        out[tgt] = {"ridge_err": stats(pred - y), "zero_err": stats(y), "n": int(len(y))}
        near = g.near.to_numpy(bool)
        out[tgt]["ridge_err_lanefollow"] = stats((pred - y)[~near])
        out[tgt]["zero_err_lanefollow"] = stats(y[~near])
    return out


def cmd_report(steps, outdir):
    report(pd.read_csv(steps), outdir)


def report(df, outdir):
    out = Path(outdir) if outdir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    df["lane_min"] = df[["lane_l", "lane_r"]].min(axis=1)
    df["near"] = (df.junc == 1) | (df.cmd_d < NEAR_M)
    df["route_lat"] = df.lat.isin(["route", "base", "nan"])
    df["age"] = df.groupby(["arm", "route"]).t.transform(lambda t: t - t.min())
    d = df[~df.warm & (df.v >= V_MIN)].copy()
    d["d_ey"], d["d_epsi"] = d.ehat_y - d.e_y, d.ehat_psi - d.e_psi
    d["d_v"], d["d_a"] = d.pose_v - d.v, d.acc_op - d.a_true
    d["d15"] = d.yp15 - d.yr15
    n_all, n_rows = len(df[~df.warm]), len(d)
    lines = []

    def add(group, sub):
        for name, col in (("e_y (m)", "d_ey"), ("e_psi (rad)", "d_epsi"), ("plan-route @15 m (m)", "d15")):
            s = stats(sub[col])
            lines.append(dict(group=group, quantity=name, **s))

    add("all", d)
    d0 = d[d.route_lat]
    add("route-steered arms, all", d0)
    add("  lane-follow", d0[~d0.near])
    lf = d0[~d0.near]
    add("    lane-follow, straight route (|route y@30 m| < 0.6 m)", lf[lf.yr30.abs() < 0.6])
    add("    lane-follow, curved route", lf[lf.yr30.abs() >= 0.6])
    add("  near junction / command", d0[d0.near])
    for k, sub in d0.groupby("town"):
        add(f"  {k}", sub)
    for k, sub in d0.groupby("night"):
        add("  night" if k else "  day", sub)
    for k, sub in d0.groupby("rain"):
        add("  rain" if k else "  no rain", sub)
    for k, sub in d0.groupby("fog"):
        add("  fog (night)" if k else "  no fog", sub)
    bins = pd.cut(d0.lane_min, [-1, .3, .5, .7, 1.01], labels=["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"])
    for k, sub in d0.groupby(bins, observed=True):
        add(f"  lane prob (min of inner pair) {k}", sub)
    d0 = d0.assign(lane_max=d0[["lane_l", "lane_r"]].max(axis=1))
    for k, sub in d0.groupby(pd.cut(d0.lane_max, [-1, .3, .5, .7, 1.01], labels=["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]), observed=True):
        add(f"  lane prob (max of inner pair) {k}", sub)
    add("openpilot-steered arms", d[~d.route_lat])
    res = pd.DataFrame(lines)
    if out:
        res.to_csv(out / "gap_by_group.csv", index=False)

    # v / a from openpilot vs truth (lane-follow, route arms)
    v_ = d0[~d0.near]
    allw = df[~df.warm]
    avail = {"v<0.5 (stationary)": float((allw.v < 0.5).mean()), "0.5<=v<3": float(((allw.v >= 0.5) & (allw.v < V_MIN)).mean()),
             "v>=3": float((allw.v >= V_MIN).mean())}
    mv = allw[allw.v >= V_MIN]
    lmax = mv[["lane_l", "lane_r"]].max(axis=1)
    lmin = mv[["lane_l", "lane_r"]].min(axis=1)
    avail |= {"moving: both inner lines < 0.3 (no ego-lane line)": float((lmax < .3).mean()), "moving: max < 0.5": float((lmax < .5).mean()),
              "moving: min < 0.5 (either line weak)": float((lmin < .5).mean()),
              "moving, lane-follow: max < 0.5": float((lmax[~mv.near] < .5).mean()), "moving, near junction/command: max < 0.5": float((lmax[mv.near] < .5).mean()),
              "moving: plan shorter than 20 m": float(((mv.v >= V_MIN) & mv.x5.lt(20)).mean()) if "x5" in mv else float("nan")}
    misc = {"availability": avail, "steps_after_warmup": n_all, "steps_with_estimate": n_rows, "frac_v_below_min": float((df[~df.warm].v < V_MIN).mean()),
            "frac_lane_low_of_moving": float((d.lane_min < LANE_LOW).mean()),
            "frac_lane_low_lanefollow_route": float((v_.lane_min < LANE_LOW).mean()),
            "frac_lane_low_near": float((d0[d0.near].lane_min < LANE_LOW).mean()),
            "v_gap": stats(v_.d_v), "a_gap": stats(v_.d_a),
            "e_y_true_lanefollow": stats(v_.e_y), "e_psi_true_lanefollow": stats(v_.e_psi)}
    # calibration probes on straight, centred, lane-follow, route-steered steps
    st = v_[(v_.e_y.abs() < CENTRED) & (v_.yr30.abs() < 0.6) & (v_.yr15.abs() < 0.3)]
    sg = lambda x: dict(n=int(np.isfinite(x).sum()), median=float(np.nanmedian(x)), p25=float(np.nanquantile(x, .25)),  # noqa: E731
                        p75=float(np.nanquantile(x, .75)))
    misc["straight"] = {"plan_minus_route_15m_signed_m": sg(st.d15), "ehat_y_minus_e_y_signed_m": sg(st.d_ey),
                        "ehat_psi_minus_e_psi_signed_rad": sg(st.d_epsi),
                        "lane_prob_left_minus_right": sg(st.lane_l - st.lane_r), "lane_prob_inner_left": sg(st.lane_l),
                        "lane_prob_inner_right": sg(st.lane_r)}
    per_route = st.groupby(["arm", "route"]).d15.median()
    misc["straight"]["per_route_median_d15_range"] = [float(per_route.min()), float(per_route.max()), int(len(per_route))]
    misc["straight"]["per_route_median_d15_quantiles_10_50_90"] = [float(x) for x in per_route.quantile([.1, .5, .9])]
    misc["straight"]["share_routes_within_0.15m"] = float((per_route.abs() < 0.15).mean())
    # gain on curves: plan y at 15 m vs route y at 15 m (regression through the origin), lane-follow
    cv = v_[(v_.yr15.abs() > 0.5) & (v_.yr15.abs() < 5) & v_.yp15.notna()]
    if len(cv):
        misc["curve_gain_plan_over_route_15m"] = {"n": len(cv), "gain": float((cv.yp15 * cv.yr15).sum() / (cv.yr15 ** 2).sum()),
                                                  "median_ratio": float(np.median(cv.yp15 / cv.yr15))}
    # warm-up trend: |plan - route| at 15 m by seconds since the route started (all moving lane-follow steps)
    b = pd.cut(d0[~d0.near].age, [0, 3, 6, 10, 20, 40, 1e9])
    misc["d15_abs_by_age_s"] = {str(k): stats(sub.d15) for k, sub in d0[~d0.near].groupby(b, observed=True)}
    misc["learned_readout_cv"] = learned_readout(d)
    if out:
        json.dump(misc, open(out / "misc.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(json.dumps(misc, indent=1, default=float))


if __name__ == "__main__":
    {"steps": cmd_steps, "report": cmd_report}[sys.argv[1]](*sys.argv[2:])
