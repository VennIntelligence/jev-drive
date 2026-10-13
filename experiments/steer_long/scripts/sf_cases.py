"""SPIN-FIX case table (Mac): per spin-set scene, the spin rollout (stored RSG-L18-s0 read at gains 20, else the gains-5 YS / ybase rollout with a
handoff spin) and what the control path did around the spin onset: speed, the served plan's 2 s arc length against the ego speed (deficit), the
plan's 1-2 s curvature x v^2, steering command / state peaks, acceleration minimum.
  sf_cases.py <rows.csv.gz> <rollouts.json> <drive.jsonl.gz> <spin_set.txt> <out.csv>"""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sf_detect import detect

rows, meta, drive, sset, out = sys.argv[1:6]
R = pd.read_csv(rows).sort_values(["session", "t"])
M = pd.DataFrame(json.load(open(meta))).set_index("session")
S = open(sset).read().split()
R = R[R.scene.isin(S)]
D = {}
for line in gzip.open(drive, "rt"):
    r = json.loads(line)
    if r["scene"] in S:
        D.setdefault(r["session"], []).append(r)
pref = ["rsg", "ys0", "ys1", "ybase"]
res = []
for sc in S:
    cand = []
    for s, x in R[R.scene == sc].groupby("session"):
        d = detect(x.t.values, x.vx.values, x.vy.values, x.wz.values, x.delta.values)
        if d["spin"] and x.label.iloc[0] in pref:
            cand.append((pref.index(x.label.iloc[0]), s, x, d))
    if not cand:
        res.append(dict(scene=sc)); continue
    _, s, x, d = min(cand, key=lambda c: c[0])
    t, vx, wz, u, de, acc = (x[k].values for k in ("t", "vx", "wz", "u", "delta", "acc"))
    t1 = d["t_first"]; i1 = int(np.abs(t - t1).argmin())
    w = (t >= t1 - 1.0) & (t <= t1 + 0.5)
    rec = sorted(D.get(s, []), key=lambda r: r["now"])
    info = dict(scene=sc, label=x.label.iloc[0], session=s, score=M.score[s], kind=d["kind"], t_on=t1, where="handoff" if t1 <= 3.2 else "mid",
                v_on=vx[i1], u_pk=np.abs(u[w]).max(), d_pk=np.abs(de[w]).max(), acc_min=acc[w].min(), wz_pk=np.abs(wz[w]).max())
    if rec:
        n0 = rec[0]["now"]
        tr = np.array([(r["now"] - n0) / 1e6 + 0.1 for r in rec])
        defs, apath, vpl = [], [], []
        for r, tt in zip(rec, tr):
            if t1 - 1.0 <= tt <= t1 + 0.05:
                v = vx[int(np.abs(t - tt).argmin())]
                p = np.vstack([[0, 0, 0], np.asarray(r["poses"])])
                seg = np.hypot(*np.diff(p[:, :2], axis=0).T)
                defs.append(2 * v - seg[:4].sum()); vpl.append(seg[2:4].sum() / 1.0)
                k = (p[4, 2] - p[2, 2]) / max(seg[2:4].sum(), 1e-3)
                apath.append(abs(k) * (seg[2:4].sum()) ** 2)
        if defs:
            info.update(deficit2=max(defs), v_plan12_min=min(vpl), A_path_max=max(apath))
    res.append(info)
T = pd.DataFrame(res)
T["mech"] = np.select([T.deficit2 >= 3, T.A_path_max >= 3], ["decel", "plan-curv"], "other")
T.to_csv(out, index=False)
pd.set_option("display.width", 250)
print(T.drop(columns=["session"]).assign(scene=T.scene.str[7:15]).round(2).to_string())
