"""SPIN-FIX P0 (turn under-steer split), offline geometry on stored RSG-L18-s0 PAI rollouts (Mac): in the 6 s before each 'turned too little' /
'straight at turn' event, at every inference whose model plan points >= 30 deg into the turn at 4 s, the heading the MPC's 1-2 s window asks
for under (a) the served reference (VCONT + LEAD re-timing, as run), (b) the model plan at its own timing, (c) track_ref full built from the
served plan; against the plan's own 2 s heading, and the car's actual heading change over the next 2 s.
  sf_turn.py <turn_cases.csv> <turn_drive.jsonl.gz> <rows.csv.gz> <out.csv>"""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/alpasim/lib")
import track_ref as TR

cases, drive, rows, out = sys.argv[1:5]
C = pd.read_csv(cases)
R = pd.read_csv(rows); R = R[(R.label == "rsg") & R.session.isin(C.rid)].sort_values(["session", "t"])
D = {}
for line in gzip.open(drive, "rt"):
    r = json.loads(line); D.setdefault(r["session"], []).append(r)
g = np.arange(41) * 0.1


def head(poses, t):
    P = np.vstack([[0, 0, 0], np.asarray(poses)])
    return float(np.interp(t, np.arange(9) * 0.5, P[:, 2]))


res = []
for c in C.itertuples():
    rec = sorted(D.get(c.rid, []), key=lambda r: r["now"])
    x = R[R.session == c.rid]
    if not rec or x.empty or not np.isfinite(c.t_ev_s):
        continue
    n0, yaw, tt = rec[0]["now"], np.unwrap(x.yaw.values), x.t.values
    for r in rec:
        t = (r["now"] - n0) / 1e6 + 0.1
        pm = r.get("poses_model") or r["poses"]
        if not (c.t_ev_s - 6 <= t <= c.t_ev_s) or abs(pm[-1][2]) < np.radians(30):
            continue
        sg = np.sign(pm[-1][2])
        (tr, xr, yr, hr), _ = TR.build(r["poses"], r["fix"]["v0"] if r.get("fix") else 0.0, 0.0, "full")
        i = int(np.abs(tt - t).argmin()); j = int(np.abs(tt - (t + 2.0)).argmin())
        res.append(dict(scene=c.scene[7:15], grp=c.grp, t=round(t, 1), v=r["fix"]["v0"] if r.get("fix") else np.nan,
                        plan2=sg * head(pm, 2.0), plan4=sg * pm[-1][2], served15=sg * head(r["poses"], 1.5), served2=sg * head(r["poses"], 2.0),
                        model15=sg * head(pm, 1.5), track15=sg * float(np.interp(1.5, tr, hr)), track2=sg * float(np.interp(2.0, tr, hr)),
                        car2=sg * (yaw[j] - yaw[i])))
T = pd.DataFrame(res)
T.to_csv(out, index=False)
deg = lambda s: np.degrees(s).round(1)  # noqa: E731
A = T.groupby(["grp", "scene"]).agg(n=("t", "size"), v=("v", "median"), plan2=("plan2", "median"), plan4=("plan4", "median"),
                                     served2=("served2", "median"), track2=("track2", "median"), car2=("car2", "median"))
for k in ("plan2", "plan4", "served2", "track2", "car2"):
    A[k] = deg(A[k])
A["v"] = A.v.round(2)
A["car/plan2"] = (A.car2 / A.plan2).round(2); A["served/plan2"] = (A.served2 / A.plan2).round(2); A["track/plan2"] = (A.track2 / A.plan2).round(2)
pd.set_option("display.width", 200)
print(A.to_string())
print("\nmedian over cases:", A[["v", "plan2", "served2", "track2", "car2", "car/plan2", "served/plan2", "track/plan2"]].median().round(2).to_dict())
